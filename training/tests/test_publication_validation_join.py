"""Differential proof against the frozen validator, including GC tombstones."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from hypothesis import given, settings, strategies as st
import pytest

from deltreltrain import replay_publication as publication
from deltreltrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH


REFERENCE = Path(__file__).parent / "fixtures/validate_publications_before_join.py.txt"
REFERENCE_FUNCTION_SHA256 = (
    "9cbc894c881d64f5387894e4484c96c3983e20a10fdb4534e15a13dbf74e9880"
)
namespace = vars(publication).copy()
exec(compile(REFERENCE.read_text(), str(REFERENCE), "exec"), namespace)
reference_validate = namespace["validate_publications"]


def test_reference_function_is_pinned_to_the_prejoin_validator():
    text = REFERENCE.read_text()
    source = text[text.index("def validate_publications(") :]
    assert hashlib.sha256(source.encode()).hexdigest() == REFERENCE_FUNCTION_SHA256
    assert f"# Function SHA-256: {REFERENCE_FUNCTION_SHA256}" in text


def database(heads=4):
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.executescript("""
        CREATE TABLE store_metadata(key TEXT PRIMARY KEY,value TEXT);
        INSERT INTO store_metadata VALUES('manifest_schema_version','6');
        CREATE TABLE game_publications(
            game_id TEXT PRIMARY KEY,run_id TEXT,generation_family TEXT,
            latest_shard_id INTEGER,revision INTEGER,sample_count INTEGER,
            policy_digest TEXT,payload_digest TEXT,context_digest TEXT,
            finalized INTEGER,enriched_samples INTEGER,record_json TEXT);
        CREATE TABLE games(game_id TEXT PRIMARY KEY,shard_id INTEGER,run_id TEXT,
            generation_family TEXT,actor_id TEXT,generation INTEGER,ring INTEGER,
            model_identity TEXT);
        CREATE TABLE shards(id INTEGER PRIMARY KEY,relative_path TEXT,state TEXT,
            fresh_sample_count INTEGER,rules_hash TEXT,feature_schema_hash TEXT,
            created_ns INTEGER,sample_count INTEGER,ring INTEGER,phase_min INTEGER,
            phase_max INTEGER,model_version TEXT,model_step INTEGER,
            model_identity TEXT,run_id TEXT,generation_family TEXT,actor_id TEXT,
            generation INTEGER,game_count INTEGER,checksum_sha256 TEXT,variant TEXT,
            segment TEXT);
        CREATE TABLE run_counters(run_id TEXT,generation_family TEXT,
            replay_revision INTEGER,enriched_rows INTEGER,committed_samples INTEGER,
            history_complete INTEGER,PRIMARY KEY(run_id,generation_family));
    """)
    for index in range(heads):
        side = "a" if index % 2 == 0 else "b"
        values = {
            "shard_id": index + 1,
            "path": f"shards/{index}.npz",
            "created_ns": index + 1,
            "sample_count": 2,
            "ring": 4,
            "phase_min": 0,
            "phase_max": 1,
            "model_version": "model",
            "model_step": 10,
            "model_identity": "model",
            "run_id": f"run-{side}",
            "generation_family": f"family-{side}",
            "actor_id": "actor",
            "generation": 0,
            "game_count": 1,
            "checksum_sha256": "a" * 64,
            "variant": "double",
            "segment": "standard",
        }
        context = {
            key: values[key]
            for key in (
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
        }
        context_digest = hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        connection.execute(
            "INSERT INTO game_publications VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"game-{index}",
                values["run_id"],
                values["generation_family"],
                index + 1,
                1,
                2,
                "b" * 64,
                "c" * 64,
                context_digest,
                0,
                0,
                json.dumps(values),
            ),
        )
        connection.execute(
            "INSERT INTO games VALUES(?,?,?,?,?,?,?,?)",
            (
                f"game-{index}",
                index + 1,
                values["run_id"],
                values["generation_family"],
                "actor",
                0,
                4,
                "model",
            ),
        )
        connection.execute(
            "INSERT INTO shards VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                index + 1,
                values["path"],
                "ready",
                2,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                *(
                    values[key]
                    for key in (
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
                ),
            ),
        )
        connection.execute(
            "INSERT INTO run_counters VALUES(?,?,1,0,2,1) ON CONFLICT(run_id,generation_family)"
            " DO UPDATE SET replay_revision=replay_revision+1,committed_samples=committed_samples+2",
            (values["run_id"], values["generation_family"]),
        )
    return connection


def outcome(validator, connection, **filters):
    try:
        validator(connection, **filters)
    except Exception as error:
        return type(error).__name__, str(error)
    return None


MUTATIONS = (
    "DELETE FROM shards WHERE id=1",
    "DELETE FROM games WHERE game_id='game-0'",
    "UPDATE game_publications SET revision=0 WHERE game_id='game-0'",
    "UPDATE game_publications SET revision=4 WHERE game_id='game-0'",
    "UPDATE game_publications SET sample_count=0 WHERE game_id='game-0'",
    "UPDATE game_publications SET finalized=2 WHERE game_id='game-0'",
    "UPDATE game_publications SET enriched_samples=1 WHERE game_id='game-0'",
    "UPDATE game_publications SET policy_digest='INVALID' WHERE game_id='game-0'",
    "UPDATE game_publications SET payload_digest='INVALID' WHERE game_id='game-0'",
    "UPDATE game_publications SET context_digest='INVALID' WHERE game_id='game-0'",
    "UPDATE game_publications SET record_json='[]' WHERE game_id='game-0'",
    "UPDATE game_publications SET record_json='broken' WHERE game_id='game-0'",
    "UPDATE shards SET state='quarantined' WHERE id=1",
    "UPDATE shards SET state='pending' WHERE id=1",
    "UPDATE shards SET fresh_sample_count=NULL WHERE id=1",
    "UPDATE shards SET fresh_sample_count=3 WHERE id=1",
    "UPDATE shards SET rules_hash='wrong' WHERE id=1",
    "UPDATE shards SET feature_schema_hash='wrong' WHERE id=1",
    "UPDATE shards SET relative_path='shards/wrong.npz' WHERE id=1",
    "UPDATE run_counters SET history_complete=0 WHERE run_id='run-a'",
    "UPDATE run_counters SET replay_revision=replay_revision+1 WHERE run_id='run-a'",
    "UPDATE run_counters SET enriched_rows=enriched_rows+1 WHERE run_id='run-a'",
    "UPDATE run_counters SET committed_samples=0 WHERE run_id='run-a'",
    "DELETE FROM run_counters WHERE run_id='run-a'",
    *(
        f"UPDATE games SET {key}='wrong' WHERE game_id='game-0'"
        for key in (
            "shard_id",
            "run_id",
            "generation_family",
            "actor_id",
            "generation",
            "ring",
            "model_identity",
        )
    ),
    *(
        f"UPDATE shards SET {key}='wrong' WHERE id=1"
        for key in (
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
    ),
)


@pytest.mark.parametrize("mutation", MUTATIONS)
@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_join_matches_reference_for_each_validation_branch(mutation, row_factory):
    with database() as connection:
        connection.row_factory = row_factory
        connection.execute(mutation)
        assert outcome(publication.validate_publications, connection) == outcome(
            reference_validate, connection
        )


@given(
    changes=st.lists(st.sampled_from(MUTATIONS), min_size=0, max_size=3),
    run=st.sampled_from((None, "run-a", "run-b", "missing")),
    family=st.sampled_from((None, "family-a", "family-b", "missing")),
)
@settings(max_examples=60, deadline=None)
def test_join_and_reference_agree_for_combined_corruption_and_filters(
    changes, run, family
):
    with database() as connection:
        for change in changes:
            connection.execute(change)
        filters = dict(run_id=run, generation_family=family)
        assert outcome(
            publication.validate_publications, connection, **filters
        ) == outcome(reference_validate, connection, **filters)


@pytest.mark.parametrize(
    "key,value",
    [
        ("path", "../escape"),
        ("path", "/absolute"),
        ("shard_id", 999),
        ("sample_count", 3),
        ("game_count", 2),
        ("ring", 99),
        ("actor_id", "other"),
    ],
)
def test_descriptor_validation_is_retained(key, value):
    with database() as connection:
        row = connection.execute(
            "SELECT record_json FROM game_publications WHERE game_id='game-0'"
        ).fetchone()
        record = json.loads(row[0])
        record[key] = value
        connection.execute(
            "UPDATE game_publications SET record_json=? WHERE game_id='game-0'",
            (json.dumps(record),),
        )
        expected = outcome(reference_validate, connection)
        assert expected is not None
        assert outcome(publication.validate_publications, connection) == expected


def test_retired_shard_allowed_but_missing_game_remains_corruption():
    with database() as connection:
        connection.execute("DELETE FROM shards WHERE id=1")
        assert outcome(publication.validate_publications, connection) is None
        connection.execute("DELETE FROM games WHERE game_id='game-0'")
        assert outcome(publication.validate_publications, connection) == (
            "ValueError",
            "publication head and game identity disagree",
        )


@pytest.mark.parametrize("heads", [0, 1, 100])
def test_validation_statement_count_is_constant_and_caller_transaction_is_preserved(
    heads,
):
    with database(heads) as connection:
        statements = []
        connection.set_trace_callback(statements.append)
        publication.validate_publications(connection)
        assert len(statements) == 10
        assert sum("LEFT JOIN" in statement for statement in statements) == 1
        statements.clear()
        reference_validate(connection)
        assert len(statements) == 8 + 2 * heads
        connection.set_trace_callback(None)
        connection.execute("BEGIN")
        publication.validate_publications(connection)
        assert connection.in_transaction
        connection.execute("ROLLBACK")


@pytest.mark.parametrize("table,key", [("games", "game_id"), ("shards", "id")])
def test_nonunique_join_rows_cannot_make_inflated_counters_valid(table, key):
    with database(1) as connection:
        connection.execute(f"CREATE TABLE malformed AS SELECT * FROM {table}")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"ALTER TABLE malformed RENAME TO {table}")
        connection.execute(f"INSERT INTO {table} SELECT * FROM {table}")
        connection.execute(
            "UPDATE run_counters SET replay_revision=2,committed_samples=4"
        )
        assert outcome(reference_validate, connection) == (
            "ValueError",
            "publication ledger and logical run counters disagree",
        )
        assert outcome(publication.validate_publications, connection) == (
            "ValueError",
            f"schema-six {table} primary key is incompatible",
        )


@pytest.mark.parametrize(
    "table,key",
    [
        ("games", "game_id"),
        ("shards", "id"),
        ("game_publications", "game_id"),
    ],
)
@pytest.mark.parametrize(
    "key_contract", ["missing", "composite", "unique_index_only", "wrong_affinity"]
)
def test_malformed_key_contracts_fail_closed_even_without_duplicate_data(
    table, key, key_contract
):
    with database(1) as connection:
        columns = list(connection.execute(f"PRAGMA table_info({table})"))
        definitions = [
            f"{row[1]} {('BLOB' if row[1] == key and key_contract == 'wrong_affinity' else row[2])}"
            for row in columns
        ]
        if key_contract == "composite":
            definitions.append(f"PRIMARY KEY({key},run_id)")
        elif key_contract == "wrong_affinity":
            definitions.append(f"PRIMARY KEY({key})")
        connection.execute("CREATE TABLE malformed(" + ",".join(definitions) + ")")
        connection.execute(f"INSERT INTO malformed SELECT * FROM {table}")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"ALTER TABLE malformed RENAME TO {table}")
        if key_contract == "unique_index_only":
            connection.execute(
                f"CREATE UNIQUE INDEX recreated_unique_key ON {table}({key})"
            )
        # These layouts were not checked by the old implementation. Explicit
        # schema rejection is intentional integrity tightening, not parity.
        assert outcome(publication.validate_publications, connection) == (
            "ValueError",
            f"schema-six {table} primary key is incompatible",
        )


def test_final_enrichment_fresh_credit_equation_remains_checked():
    with database() as connection:
        connection.execute(
            "UPDATE game_publications SET finalized=1,enriched_samples=1 WHERE game_id='game-0'"
        )
        connection.execute(
            "UPDATE run_counters SET enriched_rows=1 WHERE run_id='run-a'"
        )
        connection.execute("UPDATE shards SET fresh_sample_count=1 WHERE id=1")
        assert outcome(publication.validate_publications, connection) is None
        assert outcome(reference_validate, connection) is None
        connection.execute("UPDATE shards SET fresh_sample_count=2 WHERE id=1")
        assert (
            outcome(publication.validate_publications, connection)
            == outcome(reference_validate, connection)
            == ("ValueError", "publication head references an invalid live shard")
        )


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "heads",
        "revision_counter",
        "enriched_counter",
        "credit_row",
        "legacy_rows",
    ],
)
def test_schema_five_validation_behavior_is_unchanged(case):
    with database(0 if case == "empty" else 1) as connection:
        connection.execute(
            "UPDATE store_metadata SET value='5' WHERE key='manifest_schema_version'"
        )
        if case != "heads":
            connection.execute("DELETE FROM game_publications")
            connection.execute(
                "UPDATE run_counters SET replay_revision=0,enriched_rows=0"
            )
            connection.execute("UPDATE shards SET fresh_sample_count=NULL")
        if case == "revision_counter":
            connection.execute("UPDATE run_counters SET replay_revision=1")
        elif case == "enriched_counter":
            connection.execute("UPDATE run_counters SET enriched_rows=1")
        elif case == "credit_row":
            connection.execute("UPDATE shards SET fresh_sample_count=1")
        expected = outcome(reference_validate, connection)
        assert outcome(publication.validate_publications, connection) == expected
        assert (expected is None) == (case in ("empty", "legacy_rows"))


def test_validation_failure_preserves_callers_transaction_and_uncommitted_rows():
    with database(1) as connection:
        connection.execute("BEGIN")
        connection.execute("UPDATE game_publications SET revision=0")
        expected = outcome(reference_validate, connection)
        assert expected is not None and connection.in_transaction
        assert outcome(publication.validate_publications, connection) == expected
        assert connection.in_transaction
        assert (
            connection.execute("SELECT revision FROM game_publications").fetchone()[0]
            == 0
        )
        connection.execute("ROLLBACK")
        assert (
            connection.execute("SELECT revision FROM game_publications").fetchone()[0]
            == 1
        )
