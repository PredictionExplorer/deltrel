"""Immutable single-game revisions with durable, exactly-once position credit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import PurePosixPath
import re
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Sequence
import uuid

import numpy as np

from .contracts import (
    ALL_TARGETS,
    FEATURE_SCHEMA_HASH,
    RULES_HASH,
    TARGET_OUTCOME,
    TARGET_POLICY,
    TARGET_SOFT_POLICY,
    TARGET_TEACHER,
)
from .replay import ReplaySample, write_replay_shard
from .runtime import validate_identifier
from .topology import SUPPORTED_RINGS, get_topology

if TYPE_CHECKING:
    from .replay_store import GameRevisionReceipt, ReplayStore, ShardRecord

_FINAL = re.compile(r"(?<=:)final=([^:]+)(?=:|$)")
_FINALIZED_TAGS = {"board-full", "clinch-loser-fill", "exact-endgame"}
_MUTABLE_FINAL_FIELDS = {
    "target_mask",
    "outcome",
    "final_scores",
    "final_capes",
    "final_ownership",
    "final_alive",
    "weight",
    "teacher_policy",
    "teacher_outcome",
    "teacher_score_margin",
    "opponent_reply",
    "opponent_reply_ply",
    "second_stone",
    "second_stone_ply",
    "final_shores",
    "final_networks",
}


def schema_version(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute(
            "SELECT value FROM store_metadata WHERE key='manifest_schema_version'"
        ).fetchone()
    except sqlite3.OperationalError as error:
        if "no such table" in str(error):
            return 5
        raise
    return int(row[0]) if row is not None else 5


def initialize_publication_tables(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        counter_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(run_counters)")
        }
        for column in ("replay_revision", "enriched_rows"):
            if column not in counter_columns:
                connection.execute(
                    f"ALTER TABLE run_counters ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0 CHECK({column} >= 0)"
                )
        shard_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(shards)")
        }
        if "fresh_sample_count" not in shard_columns:
            connection.execute(
                "ALTER TABLE shards ADD COLUMN fresh_sample_count INTEGER CHECK(fresh_sample_count >= 0 AND fresh_sample_count <= sample_count)"
            )
        connection.execute("""CREATE TABLE IF NOT EXISTS game_publications (
            game_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, generation_family TEXT NOT NULL,
            latest_shard_id INTEGER NOT NULL, revision INTEGER NOT NULL CHECK(revision > 0),
            sample_count INTEGER NOT NULL CHECK(sample_count > 0), policy_digest TEXT NOT NULL,
            payload_digest TEXT NOT NULL, context_digest TEXT NOT NULL,
            finalized INTEGER NOT NULL CHECK(finalized IN (0, 1)),
            enriched_samples INTEGER NOT NULL CHECK(enriched_samples >= 0),
            record_json TEXT NOT NULL, updated_ns INTEGER NOT NULL
        )""")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS game_publications_run ON game_publications(run_id, generation_family)"
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def revision_counts(
    connection: sqlite3.Connection, run_id: str, family: str
) -> tuple[int, int]:
    try:
        row = connection.execute(
            "SELECT replay_revision, enriched_rows FROM run_counters WHERE run_id=? AND generation_family=?",
            (run_id, family),
        ).fetchone()
    except sqlite3.OperationalError as error:
        if schema_version(connection) == 5 and any(
            term in str(error) for term in ("no such column", "no such table")
        ):
            return 0, 0
        raise
    return (int(row[0]), int(row[1])) if row is not None else (0, 0)


def _digest_rows(
    samples: Sequence[ReplaySample],
    *,
    immutable: bool,
    future_limit: int | None = None,
    include_auxiliary: bool = True,
) -> tuple[str, ...]:
    digest = hashlib.sha256()
    prefixes = [digest.hexdigest()]
    for sample in samples:
        for field in fields(sample):
            name = field.name
            # These counts are a deterministic view of the already-hashed
            # final ownership/alive maps, including for historical schema-v5.
            if name in {"final_shores", "final_networks"}:
                continue
            policy_name = name.removesuffix("_ply")
            if policy_name in {"opponent_reply", "second_stone"} and (
                not include_auxiliary
                or getattr(sample, policy_name) is None
                or (
                    future_limit is not None
                    and getattr(sample, f"{policy_name}_ply") >= future_limit
                )
            ):
                continue
            if immutable and name in _MUTABLE_FINAL_FIELDS:
                continue
            value = getattr(sample, name)
            if immutable and name == "search_provenance":
                value = _FINAL.sub("final=pending-policy", value)
            digest.update(name.encode() + b"\0")
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
                header = json.dumps(
                    [array.dtype.str, array.shape], separators=(",", ":")
                ).encode()
                data = header + b"\0" + array.tobytes()
            else:
                data = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
        prefixes.append(digest.hexdigest())
    return tuple(prefixes)


def _validate_future_policies(samples: Sequence[ReplaySample]) -> None:
    """Future labels are views of immutable decisions, never new guesses."""
    for source in samples:
        for name in ("opponent_reply", "second_stone"):
            target = getattr(source, name)
            if target is None:
                continue
            destination = getattr(source, f"{name}_ply")
            if destination >= len(samples):
                raise ValueError(
                    "future policy destination lies outside published game"
                )
            future = samples[destination]
            if name == "second_stone":
                if (
                    destination != source.ply + 1
                    or future.to_move != source.to_move
                    or future.moves_left != 1
                    or future.opening
                ):
                    raise ValueError("second-stone target crosses a turn boundary")
                expected = future.policy
            else:
                expected_destination = next(
                    (
                        s.ply
                        for s in samples[source.ply + 1 :]
                        if s.to_move != source.to_move
                    ),
                    None,
                )
                if destination != expected_destination:
                    raise ValueError(
                        "opponent-reply target is not the next opponent decision"
                    )
                expected = np.zeros(source.stones.size + 1, dtype=np.float32)
                if "swap=taken" in future.search_provenance.split(":"):
                    expected[-1] = 1.0
                else:
                    expected[:-1] = future.policy
            if not np.array_equal(target, expected):
                raise ValueError("future policy changed the recorded later decision")


def _record_json(record: ShardRecord, store: ReplayStore) -> str:
    values = asdict(record)
    values["path"] = record.path.relative_to(store.root).as_posix()
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _record_values(serialized: str) -> dict[str, Any]:
    values = json.loads(serialized)
    if not isinstance(values, dict) or not isinstance(values.get("path"), str):
        raise ValueError("publication record metadata is invalid")
    path = PurePosixPath(values["path"])
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("shards",):
        raise ValueError("publication record path escapes replay shards")
    return values


def _restore_record(serialized: str, store: ReplayStore) -> ShardRecord:
    from .replay_store import ShardRecord

    values = _record_values(serialized)
    values["path"] = store.root / values["path"]
    return ShardRecord(**values)


def _lease(connection: sqlite3.Connection, context: dict[str, Any]) -> None:
    row = connection.execute(
        "SELECT generation FROM actor_generations WHERE run_id=? AND generation_family=? AND actor_id=?",
        (context["run_id"], context["generation_family"], context["actor_id"]),
    ).fetchone()
    if row is None or int(row[0]) != context["generation"]:
        raise ValueError("replay generation is not the actor's active lease")


def append_revision(
    store: ReplayStore,
    samples: Sequence[ReplaySample],
    *,
    finalized: bool,
    phase_min: int,
    phase_max: int,
    model_version: str,
    model_step: int,
    model_identity: str,
    run_id: str,
    generation_family: str,
    actor_id: str,
    generation: int,
) -> GameRevisionReceipt:
    from .replay_store import (
        DuplicateGameError,
        GameRevisionReceipt,
        ShardRecord,
        _sha256,
        REPLAY_PUBLICATION_MANIFEST_SCHEMA_VERSION,
    )

    if type(finalized) is not bool or not samples:
        raise ValueError(
            "revision requires nonempty samples and a boolean finalized flag"
        )
    if store.connection.in_transaction:
        raise ValueError("game publication cannot run inside an external transaction")
    # Snapshot and revalidate mutable ReplaySample arrays before hashing/writing.
    batch = [replace(deepcopy(sample)) for sample in samples]
    first = batch[0]
    if len(batch) > get_topology(first.rings).n + 1:
        raise ValueError("game prefix exceeds the board's placement/swap bound")
    if any(
        sample.game_id != first.game_id or sample.ply != index
        for index, sample in enumerate(batch)
    ):
        raise ValueError(
            "game revision requires one game with contiguous original plies starting at zero"
        )
    if any(
        sample.rings != first.rings or sample.variant_label != first.variant_label
        for sample in batch
    ):
        raise ValueError("game revision must be ring- and variant-homogeneous")
    if (
        type(phase_min) is not int
        or type(phase_max) is not int
        or phase_min < 0
        or phase_max < phase_min
    ):
        raise ValueError("invalid phase range")
    if (
        type(model_step) is not int
        or model_step < 0
        or type(generation) is not int
        or generation < 0
    ):
        raise ValueError("model step and generation must be non-negative integers")
    for name, value in (
        ("run_id", run_id),
        ("generation_family", generation_family),
        ("actor_id", actor_id),
        ("model_identity", model_identity),
    ):
        validate_identifier(name, value)
    if (
        not re.fullmatch(r"sha256-[0-9a-f]{64}", model_identity)
        or model_version != model_identity
    ):
        raise ValueError("model version must equal its content-addressed identity")
    context = dict(
        run_id=run_id,
        generation_family=generation_family,
        actor_id=actor_id,
        generation=generation,
        model_identity=model_identity,
        model_version=model_version,
        model_step=model_step,
        ring=first.rings,
        variant=first.variant_label,
        phase_min=phase_min,
    )
    for sample in batch:
        if any(
            getattr(sample, key) != context[key]
            for key in (
                "run_id",
                "generation_family",
                "actor_id",
                "generation",
                "model_identity",
            )
        ):
            raise ValueError(
                "sample provenance disagrees with game publication context"
            )
        if (
            sample.rules_hash != RULES_HASH
            or sample.feature_schema_hash != FEATURE_SCHEMA_HASH
        ):
            raise ValueError("sample contract hashes are incompatible")
        tags = _FINAL.findall(sample.search_provenance)
        expected_tags = _FINALIZED_TAGS if finalized else {"pending-policy"}
        if len(tags) != 1 or tags[0] not in expected_tags:
            raise ValueError("revision search provenance has an invalid final tag")
        policy_mask = TARGET_POLICY | TARGET_SOFT_POLICY
        if (
            sample.target_mask & policy_mask != policy_mask
            or sample.target_mask & TARGET_TEACHER
        ):
            raise ValueError(
                "every revision row requires policy and soft-policy targets without a teacher"
            )
        if not finalized and sample.target_mask != policy_mask:
            raise ValueError(
                "intermediate game revisions must contain policy-only targets"
            )
        if finalized:
            required = (
                TARGET_OUTCOME | policy_mask
                if tags[0] == "clinch-loser-fill"
                else ALL_TARGETS
            )
            if sample.target_mask & required != required:
                raise ValueError(
                    "finalized game revision is missing completed-game targets"
                )
            if (
                sample.target_mask != first.target_mask
                or any(
                    not np.array_equal(getattr(sample, name), getattr(first, name))
                    for name in (
                        "final_scores",
                        "final_capes",
                        "final_ownership",
                        "final_alive",
                    )
                )
                or tags != _FINAL.findall(first.search_provenance)
            ):
                raise ValueError("completed-game targets disagree across the prefix")
    row = store.connection.execute(
        "SELECT generation_family FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if row is None or row[0] != generation_family:
        raise ValueError("replay run is not registered to this generation family")
    _lease(store.connection, context)
    policy_digests = _digest_rows(batch, immutable=True)
    _validate_future_policies(batch)
    payload_digests = _digest_rows(batch, immutable=False)
    context_digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    created_ns = time.time_ns()
    path = store.shard_directory / f"shard-{created_ns}-{uuid.uuid4().hex}.npz"
    committed = False
    commit_issued = False
    try:
        write_replay_shard(path, batch, compressed=True)
        checksum = _sha256(path)
        store.connection.execute("BEGIN IMMEDIATE")
        _lease(store.connection, context)
        store._assert_manifest_identity()
        if (
            store.connection.execute(
                "SELECT 1 FROM run_counters WHERE history_complete != 1 LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise ValueError(
                "game publication requires complete durable credit history"
            )
        version = schema_version(store.connection)
        if version not in (5, 6):
            raise ValueError("game publication manifest schema is incompatible")
        old = store.connection.execute(
            "SELECT * FROM game_publications WHERE game_id=?", (first.game_id,)
        ).fetchone()
        game = store.connection.execute(
            "SELECT * FROM games WHERE game_id=?", (first.game_id,)
        ).fetchone()
        if old is None and game is not None:
            raise DuplicateGameError(
                "replay manifest already contains a completed game ID"
            )
        if old is not None:
            if game is None or old["context_digest"] != context_digest:
                raise ValueError("game publication context changed")
            old_count = int(old["sample_count"])
            if (
                bool(old["finalized"]) == finalized
                and old_count == len(batch)
                and old["payload_digest"] == payload_digests[-1]
            ):
                record = _restore_record(str(old["record_json"]), store)
                store.connection.execute("COMMIT")
                return GameRevisionReceipt(record, 0, 0, 0, int(old["revision"]), True)
            if old["finalized"]:
                raise ValueError("cannot reopen or change a finalized game publication")
            if len(batch) < old_count:
                raise ValueError("game publication prefix cannot shrink")
            if policy_digests[old_count] != old["policy_digest"]:
                raise ValueError(
                    "previously published semantic state or policy changed"
                )
            if not finalized:
                previous_view = _digest_rows(
                    batch[:old_count], immutable=False, future_limit=old_count
                )[-1]
                legacy_view = _digest_rows(
                    batch[:old_count], immutable=False, include_auxiliary=False
                )[-1]
                if old["payload_digest"] not in (previous_view, legacy_view):
                    raise ValueError(
                        "pending publication payload changed before finalization"
                    )
            if not finalized and len(batch) == old_count:
                raise ValueError("pending game publication made no progress")
            previous_record = _restore_record(str(old["record_json"]), store)
            if phase_max < previous_record.phase_max:
                raise ValueError("game publication phase cannot move backwards")
            revision = int(old["revision"]) + 1
        else:
            old_count, revision = 0, 1
        new_samples = len(batch) - old_count
        enriched = old_count if finalized else 0
        store.connection.execute(
            "UPDATE store_metadata SET value=? WHERE key='manifest_schema_version'",
            (str(REPLAY_PUBLICATION_MANIFEST_SCHEMA_VERSION),),
        )
        cursor = store.connection.execute(
            """INSERT INTO shards(
            relative_path, created_ns, sample_count, ring, phase_min, phase_max,
            model_version, model_step, model_identity, run_id, generation_family,
            actor_id, generation, game_count, rules_hash, feature_schema_hash,
            checksum_sha256, variant, segment, fresh_sample_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                path.relative_to(store.root).as_posix(),
                created_ns,
                len(batch),
                first.rings,
                phase_min,
                phase_max,
                model_version,
                model_step,
                model_identity,
                run_id,
                generation_family,
                actor_id,
                generation,
                1,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                checksum,
                first.variant_label,
                first.segment,
                new_samples,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("revision shard insert did not return an identifier")
        shard_id = int(cursor.lastrowid)
        record = ShardRecord(
            shard_id=shard_id,
            path=path,
            created_ns=created_ns,
            sample_count=len(batch),
            ring=first.rings,
            phase_min=phase_min,
            phase_max=phase_max,
            model_version=model_version,
            model_step=model_step,
            model_identity=model_identity,
            run_id=run_id,
            generation_family=generation_family,
            actor_id=actor_id,
            generation=generation,
            game_count=1,
            checksum_sha256=checksum,
            state="ready",
            quarantine_reason=None,
            variant=first.variant_label,
            segment=first.segment,
        )
        if old is not None:
            store.connection.execute(
                "UPDATE shards SET state='superseded' WHERE id=? AND state='ready'",
                (int(old["latest_shard_id"]),),
            )
            store.connection.execute(
                "UPDATE games SET shard_id=? WHERE game_id=?", (shard_id, first.game_id)
            )
        else:
            store.connection.execute(
                "INSERT INTO games(game_id,shard_id,run_id,generation_family,actor_id,generation,ring,model_identity) VALUES (?,?,?,?,?,?,?,?)",
                (
                    first.game_id,
                    shard_id,
                    run_id,
                    generation_family,
                    actor_id,
                    generation,
                    first.rings,
                    model_identity,
                ),
            )
        store.connection.execute(
            """INSERT INTO game_publications(
            game_id,run_id,generation_family,latest_shard_id,revision,sample_count,
            policy_digest,payload_digest,context_digest,finalized,enriched_samples,record_json,updated_ns
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
            latest_shard_id=excluded.latest_shard_id, revision=excluded.revision,
            sample_count=excluded.sample_count, policy_digest=excluded.policy_digest,
            payload_digest=excluded.payload_digest, finalized=excluded.finalized,
            enriched_samples=excluded.enriched_samples, record_json=excluded.record_json,
            updated_ns=excluded.updated_ns""",
            (
                first.game_id,
                run_id,
                generation_family,
                shard_id,
                revision,
                len(batch),
                policy_digests[-1],
                payload_digests[-1],
                context_digest,
                int(finalized),
                enriched,
                _record_json(record, store),
                time.time_ns(),
            ),
        )
        updated = store.connection.execute(
            """UPDATE run_counters SET
            committed_samples=committed_samples+?, replay_revision=replay_revision+1,
            enriched_rows=enriched_rows+?, updated_ns=? WHERE run_id=? AND generation_family=?""",
            (new_samples, enriched, time.time_ns(), run_id, generation_family),
        )
        if updated.rowcount != 1:
            raise ValueError("game publication run counter is missing")
        commit_issued = True
        store.connection.execute("COMMIT")
        committed = True
        return GameRevisionReceipt(
            record, new_samples, enriched, int(finalized), revision, False
        )
    except BaseException:
        if store.connection.in_transaction:
            store.connection.execute("ROLLBACK")
            commit_issued = False
        raise
    finally:
        # COMMIT may reach durable storage before its acknowledgement fails.
        # Keep the immutable payload whenever the outcome is ambiguous; orphan
        # reconciliation can later remove an unreferenced file safely.
        if not committed and not commit_issued:
            path.unlink(missing_ok=True)


def validate_publications(
    connection: sqlite3.Connection,
    *,
    run_id: str | None = None,
    generation_family: str | None = None,
) -> None:
    """Validate durable heads/counters without requiring GC-retired payloads."""

    owns = not connection.in_transaction
    if owns:
        connection.execute("BEGIN")
    try:
        version = schema_version(connection)
        if version == 5:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if (
                "game_publications" in tables
                and connection.execute(
                    "SELECT 1 FROM game_publications LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise ValueError("schema-five metadata cannot contain game revisions")
            if "run_counters" in tables:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(run_counters)")
                }
                for column in ("replay_revision", "enriched_rows"):
                    if (
                        column in columns
                        and connection.execute(
                            f"SELECT 1 FROM run_counters WHERE {column} != 0 LIMIT 1"
                        ).fetchone()
                        is not None
                    ):
                        raise ValueError(
                            "schema-five metadata cannot contain publication counters"
                        )
            if "shards" in tables:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(shards)")
                }
                if (
                    "fresh_sample_count" in columns
                    and connection.execute(
                        "SELECT 1 FROM shards WHERE fresh_sample_count IS NOT NULL LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    raise ValueError(
                        "schema-five metadata cannot contain revision credit rows"
                    )
            return
        if version != 6:
            raise ValueError("replay manifest schema is incompatible")
        required = {"game_publications", "games", "shards", "run_counters"}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not required <= tables:
            raise ValueError("schema-six replay publication tables are missing")
        counter_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(run_counters)")
        }
        shard_info = tuple(connection.execute("PRAGMA table_info(shards)"))
        shard_columns = {row[1] for row in shard_info}
        if (
            not {"replay_revision", "enriched_rows", "history_complete"}
            <= counter_columns
            or "fresh_sample_count" not in shard_columns
        ):
            raise ValueError("schema-six logical credit columns are missing")
        # Joining must not multiply a publication when a malformed replacement
        # table has lost its unique key. Require the canonical key contracts;
        # merely observing no duplicates today would not prove join cardinality.
        for table, key, kind, columns in (
            ("shards", "id", "INTEGER", shard_info),
            (
                "games",
                "game_id",
                "TEXT",
                tuple(connection.execute("PRAGMA table_info(games)")),
            ),
            (
                "game_publications",
                "game_id",
                "TEXT",
                tuple(connection.execute("PRAGMA table_info(game_publications)")),
            ),
        ):
            primary = [
                (row[1], str(row[2]).upper(), row[5]) for row in columns if row[5]
            ]
            if primary != [(key, kind, 1)]:
                raise ValueError(f"schema-six {table} primary key is incompatible")
        publication_columns = (
            "game_id",
            "run_id",
            "generation_family",
            "latest_shard_id",
            "revision",
            "sample_count",
            "policy_digest",
            "payload_digest",
            "context_digest",
            "finalized",
            "enriched_samples",
            "record_json",
        )
        game_columns = (
            "shard_id",
            "run_id",
            "generation_family",
            "actor_id",
            "generation",
            "ring",
            "model_identity",
        )
        metadata_keys = (
            "created_ns",
            "sample_count",
            "ring",
            "phase_min",
            "phase_max",
            "model_version",
            "model_step",
            "model_identity",
            "run_id",
            "generation_family",
            "actor_id",
            "generation",
            "game_count",
            "checksum_sha256",
            "variant",
            "segment",
        )
        shard_columns_for_validation = (
            "relative_path",
            "state",
            "fresh_sample_count",
            "rules_hash",
            "feature_schema_hash",
            *metadata_keys,
        )
        # The primary-key joins preserve one row per durable publication. Keep
        # explicit presence columns: a missing game is corruption, whereas a
        # GC-retired shard is allowed and its durable descriptor is still checked.
        query = (
            "SELECT "
            + ",".join(
                (
                    *(f"p.{column}" for column in publication_columns),
                    "g.game_id",
                    *(f"g.{column}" for column in game_columns),
                    "s.id",
                    *(f"s.{column}" for column in shard_columns_for_validation),
                )
            )
            + (
                " FROM game_publications AS p"
                " LEFT JOIN games AS g ON g.game_id=p.game_id"
                " LEFT JOIN shards AS s ON s.id=p.latest_shard_id"
            )
        )
        parameters: list[str] = []
        clauses = []
        for column, value in (
            ("run_id", run_id),
            ("generation_family", generation_family),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        if clauses:
            query += " WHERE " + " AND ".join(f"p.{clause}" for clause in clauses)
        totals: dict[tuple[str, str], list[int]] = {}
        game_offset = len(publication_columns) + 1
        shard_offset = game_offset + len(game_columns)
        for joined in connection.execute(query, parameters):
            row = tuple(joined)
            (
                game_id,
                run,
                family,
                shard_id,
                revision,
                count,
                policy,
                payload,
                context,
                finalized,
                enriched,
                serialized,
            ) = row[: len(publication_columns)]
            if not (
                isinstance(revision, int)
                and 1 <= revision <= count + 1
                and count > 0
                and shard_id > 0
                and finalized in (0, 1)
                and 0 <= enriched <= count
                and (finalized or enriched == 0)
            ):
                raise ValueError("invalid game publication counters")
            if any(
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in (policy, payload, context)
            ):
                raise ValueError("invalid game publication digest")
            values = _record_values(serialized)
            keys = (
                "run_id",
                "generation_family",
                "actor_id",
                "generation",
                "model_identity",
                "model_version",
                "model_step",
                "ring",
                "variant",
                "phase_min",
            )
            restored_context = {key: values.get(key) for key in keys}
            expected_context = hashlib.sha256(
                json.dumps(
                    restored_context, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            ring = values.get("ring")
            if (
                type(ring) is not int
                or ring not in SUPPORTED_RINGS
                or context != expected_context
                or count > get_topology(ring).n + 1
            ):
                raise ValueError("publication immutable context is invalid")
            if any(
                values.get(key) != value
                for key, value in (
                    ("shard_id", shard_id),
                    ("sample_count", count),
                    ("run_id", run),
                    ("generation_family", family),
                    ("game_count", 1),
                )
            ):
                raise ValueError("publication head and record metadata disagree")
            game = (
                None
                if row[len(publication_columns)] is None
                else row[game_offset:shard_offset]
            )
            expected_game = (
                shard_id,
                run,
                family,
                values.get("actor_id"),
                values.get("generation"),
                values.get("ring"),
                values.get("model_identity"),
            )
            if game is None or tuple(game) != expected_game:
                raise ValueError("publication head and game identity disagree")
            shard = None if row[shard_offset] is None else row[shard_offset + 1 :]
            if shard is not None and (
                shard[0] != values["path"]
                or shard[1] not in ("ready", "quarantined")
                or not isinstance(shard[2], int)
                or not 0 <= shard[2] <= count
                or (finalized and shard[2] != count - enriched)
                or shard[3:5] != (f"{RULES_HASH:016x}", f"{FEATURE_SCHEMA_HASH:016x}")
                or any(
                    shard[5 + index] != values.get(key)
                    for index, key in enumerate(metadata_keys)
                )
            ):
                raise ValueError("publication head references an invalid live shard")
            total = totals.setdefault((run, family), [0, 0, 0])
            total[0] += revision
            total[1] += enriched
            total[2] += count
        counter_query = "SELECT run_id,generation_family,replay_revision,enriched_rows,committed_samples,history_complete FROM run_counters"
        if clauses:
            counter_query += " WHERE " + " AND ".join(clauses)
        seen = set()
        for (
            run,
            family,
            revision,
            enriched,
            credit,
            history_complete,
        ) in connection.execute(counter_query, parameters):
            expected = totals.get((run, family), [0, 0, 0])
            if (
                history_complete != 1
                or (revision, enriched) != tuple(expected[:2])
                or credit < expected[2]
            ):
                raise ValueError("publication ledger and logical run counters disagree")
            seen.add((run, family))
        if set(totals) - seen:
            raise ValueError("publication ledger run counter is missing")
    except sqlite3.Error as error:
        raise ValueError(f"invalid replay publication schema: {error}") from error
    finally:
        if owns:
            connection.execute("ROLLBACK")
