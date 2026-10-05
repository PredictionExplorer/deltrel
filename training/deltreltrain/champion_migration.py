"""Publish a losslessly re-identified, proof-linked legacy champion on CPU.

The legacy result remains a result about the legacy identity. A separate receipt
binds that verified publication to the new one; ordinary loaders stay strict.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, cast

import torch
import yaml

from .checkpoint import (
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_VERSION,
    inspect_checkpoint,
    load_ema_checkpoint,
    load_model_manifest,
    sha256_file,
    verify_file,
    write_model_pointer,
)
from .config import ArenaConfig, load_config
from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from .model import MODEL_SCHEMA_VERSION, GraphResTNet
from .rebrand import SOURCE_FEATURE_HASH, SOURCE_RULES_HASH, migrate_checkpoint
from .rebrand import migrate_payload
from .runtime import validate_identifier

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LEGACY_RULES = f"fnv1a64:{SOURCE_RULES_HASH:016x}"
_LEGACY_FEATURES = f"{SOURCE_FEATURE_HASH:016x}"
_MAX_PROOF_BYTES = 64 * 1024 * 1024
_REQUIRED_PROOF = {
    "assessment.json",
    "verification.json",
    "capture.json",
    "result.json",
    "allocation.json",
    "resume.json",
    "run.json",
    "candidate-manifest.json",
    "champion-manifest.json",
    "published-champion-pointer.json",
}


def _summary_equal(actual: Any, expected: Any, *, path: tuple[str, ...] = ()) -> bool:
    """Only derived display Elo tolerates platform libm rounding (four ULPs).

    Score, confidence bounds, E-values, decisions and contracts remain exact.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _summary_equal(value, expected[key], path=(*path, key))
            for key, value in actual.items()
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _summary_equal(a, b, path=path)
            for a, b in zip(actual, expected, strict=True)
        )
    display_elo = (
        bool(path)
        and path[-1] in {"elo_difference", "anytime_elo_interval"}
        and (
            (len(path) == 2 and path[0] in {"aggregate", "balanced_aggregate"})
            or (len(path) == 3 and path[0] == "per_cell")
            or (len(path) == 5 and path[0] == "per_ring" and path[2] == "cells")
        )
    )
    if display_elo and type(actual) is float and type(expected) is float:
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and abs(actual - expected) <= 4 * max(math.ulp(actual), math.ulp(expected))
        )
    return _canonical(actual) == _canonical(expected)


def _validate_recorded_evaluation(documents: Mapping[str, dict[str, Any]]) -> None:
    """Recompute the unchanged statistic; never relabel the original result."""
    from .adaptive_promotion import next_allocation, plan_complete, plan_targets
    from .arena import ArenaGame, ArenaPair, summarize_completed_arena_pairs
    from .balanced_evaluation import evaluation_contract, pair_key
    from .search_options import parse_search_execution

    result = documents["result.json"]
    state = documents["resume.json"]["arena_state"]
    values = dict(state["config"])
    for name in (
        "rings",
        "segment_handicaps",
        "segment_handicap_pda",
        "handicap_severity_cycle",
        "required_regression_rings",
    ):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    for name in ("per_ring_regression_floor_elo", "promotion_pair_ratios"):
        if name in values:
            values[name] = {int(key): value for key, value in values[name].items()}
    if "search_execution" in values:
        values["search_execution"] = parse_search_execution(values["search_execution"])
    config = ArenaConfig(**values)
    expected_contract = evaluation_contract(config)
    expected_contract.update(
        rules_hash=_LEGACY_RULES, rules_schema="edgeconnect.star.rules.v3"
    )
    expected_contract.pop("identity")
    expected_contract["identity"] = (
        "sha256-" + hashlib.sha256(_canonical(expected_contract)).hexdigest()
    )
    if expected_contract != result["evaluation_contract"]:
        raise ValueError(
            "recorded arena configuration differs from its legacy contract"
        )

    def parse_pairs(rows: Any) -> list[ArenaPair]:
        if not isinstance(rows, list):
            raise ValueError("recorded pairs must be a list")
        return [
            ArenaPair(**{**row, "outcomes": tuple(row["outcomes"])}) for row in rows
        ]

    pairs = parse_pairs(result["pairs"])
    by_pair = {pair_key(pair): pair for pair in pairs}
    if len(by_pair) != len(pairs):
        raise ValueError("duplicate recorded pair")
    games = [ArenaGame(**row) for row in result["games"]]
    by_game = {
        (game.ring, game.variant, game.pair, game.candidate_player): game
        for game in games
    }
    if len(by_game) != len(games):
        raise ValueError("duplicate recorded game")
    completed_states = {}
    for row in state["game_states"]:
        game = ArenaGame(**row["result"])
        key = (game.ring, game.variant, game.pair, game.candidate_player)
        if (
            key in completed_states
            or by_game.get(key) != game
            or game.searched_moves != len(row["actions"])
        ):
            raise ValueError("completed native-history evidence disagrees with result")
        completed_states[key] = game
    if completed_states != by_game:
        raise ValueError("native-history evidence misses recorded games")
    for pair in pairs:
        seats = [by_game[(pair.ring, pair.variant, pair.pair, seat)] for seat in (0, 1)]
        if pair.outcomes != tuple(game.outcome for game in seats) or any(
            getattr(game, key) != getattr(pair, key)
            for game in seats
            for key in ("opening_seed", "opening_action", "forced_opening", "segment")
        ):
            raise ValueError("pair outcomes disagree with the recorded games")
    resumed = parse_pairs(state["pairs"])
    if (
        len(resumed) != len(pairs)
        or {pair_key(pair): pair for pair in resumed} != by_pair
    ):
        raise ValueError("resume pairs disagree with the result")
    if sorted(state["games"], key=_canonical) != sorted(
        result["games"], key=_canonical
    ):
        raise ValueError("resume games disagree with the result")
    previous = None
    for index, record in enumerate(documents["allocation.json"]["plan_history"]):
        observed = parse_pairs(record["decision_pairs"])
        if (
            record["plan_index"] != index
            or record["decision_pairs_sha256"]
            != hashlib.sha256(
                _canonical([asdict(pair) for pair in observed])
            ).hexdigest()
        ):
            raise ValueError("allocation decision hash/index is invalid")
        if any(by_pair.get(pair_key(pair)) != pair for pair in observed):
            raise ValueError("allocation decision evidence differs from result")
        if (previous is None and observed) or (
            previous is not None
            and (
                not plan_complete(previous, observed, config)
                or len(observed) != sum(plan_targets(previous, config).values())
            )
        ):
            raise ValueError("allocation predates a completed boundary")
        summary = cast(
            dict[str, Any],
            summarize_completed_arena_pairs(
                observed,
                config,
                completed_allocation_targets=plan_targets(previous, config)
                if previous
                else None,
            ),
        )
        summary["evaluation_contract"] = expected_contract
        if not _summary_equal(summary, record["decision_summary"]):
            raise ValueError("allocation summary differs from recorded evidence")
        sticky = previous is not None and previous.get("phase") == "handicap_review"
        review = record.get("phase") == "handicap_review"
        if review and not sticky and summary["promotion"]["decision"] != "promote":
            raise ValueError("allocation review lacks its promotion trigger")
        expected = next_allocation(
            config,
            previous_plan=previous,
            summary=summary,
            review_only=review or sticky,
        )
        if expected is None and sticky and not review:
            expected = next_allocation(config, previous_plan=previous, summary=summary)
        actual = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "plan_index",
                "decision_pairs",
                "decision_pairs_sha256",
                "decision_summary",
            }
        }
        if expected != actual:
            raise ValueError("allocation plan differs from its policy")
        previous = record
    if (
        previous is None
        or not plan_complete(previous, pairs, config)
        or sum(plan_targets(previous, config).values()) != len(pairs)
    ):
        raise ValueError("promotion is not at the complete last allocation")
    summary = cast(
        dict[str, Any],
        summarize_completed_arena_pairs(
            pairs, config, completed_allocation_targets=plan_targets(previous, config)
        ),
    )
    summary["evaluation_contract"] = expected_contract
    summary["promotion"]["allocation_boundary_complete"] = True
    if not _summary_equal(summary, {key: result.get(key) for key in summary}):
        raise ValueError("promotion statistics differ from the recorded pairs")


def _sha(value: object) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError("expected an explicit lowercase SHA-256")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _canonical(value: object) -> bytes:
    return _json_bytes(value).rstrip(b"\n")


def _object(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant in {label}: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is missing")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("artifact path must be a normalized relative path")
    return path


def _file(root: Path, relative: Path) -> Path:
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("artifact must be a regular file without symlinks")
    return path


def _entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_new(source: Path, target: Path, sha: str, size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    verify_file(target, expected_sha256=sha, expected_bytes=size)


def _load_proof(
    root: Path, expected_sha256: str
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    manifest_path = _file(root, Path("manifest.json"))
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != _sha(expected_sha256):
        raise ValueError("proof manifest hash changed")
    manifest = _object(manifest_bytes, "proof manifest")
    entries = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen"
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= 256
    ):
        raise ValueError("unsupported frozen proof inventory")
    raw = {"manifest.json": manifest_bytes}
    total = len(manifest_bytes)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid proof inventory entry")
        relative = _relative(entry.get("path"))
        name = relative.as_posix()
        size = entry.get("bytes")
        if name in raw or type(size) is not int or size < 0:
            raise ValueError("duplicate or invalid proof inventory entry")
        total += size
        if total > _MAX_PROOF_BYTES:
            raise ValueError("proof bundle exceeds the bounded metadata size")
        path = _file(root, relative)
        verify_file(
            path, expected_sha256=_sha(entry.get("sha256")), expected_bytes=size
        )
        raw[name] = path.read_bytes()
        if hashlib.sha256(raw[name]).hexdigest() != entry["sha256"]:
            raise ValueError("proof file changed during read")
    if not _REQUIRED_PROOF <= raw.keys():
        raise ValueError("proof bundle is missing required verification evidence")
    documents = {name: _object(raw[name], name) for name in _REQUIRED_PROOF}
    return raw, documents


def _validate_proof(
    raw: Mapping[str, bytes],
    documents: Mapping[str, dict[str, Any]],
    pointer_bytes: bytes,
    expected_contract: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = _object(pointer_bytes, "source champion pointer")
    manifest = documents["candidate-manifest.json"]
    baseline = documents["champion-manifest.json"]
    result = documents["result.json"]
    assessment = documents["assessment.json"]
    verification = documents["verification.json"]
    capture = documents["capture.json"]
    allocation = documents["allocation.json"]
    resume = documents["resume.json"]
    identity = documents["run.json"]
    if pointer_bytes != raw["published-champion-pointer.json"]:
        raise ValueError("source pointer is not the verified published champion")
    if (
        pointer.get("format") != "startrain.model-pointer"
        or pointer.get("schema_version") != 2
        or pointer.get("role") != "champion"
    ):
        raise ValueError("source must be an immediately preceding champion pointer")
    for published in (manifest, baseline):
        if (
            published.get("format") != "startrain.model-manifest"
            or published.get("schema_version") != 3
            or published.get("rules_hash") != _LEGACY_RULES
            or published.get("feature_schema_hash") != _LEGACY_FEATURES
            or published.get("model_schema_version") != MODEL_SCHEMA_VERSION
            or published.get("weights") != "ema"
        ):
            raise ValueError("legacy manifest contracts are incompatible")
        if (
            published.get("model_identity")
            != "sha256-" + _sha(published.get("checkpoint_sha256"))
            or published.get("model_version") != published["model_identity"]
        ):
            raise ValueError("legacy manifest identity is incompatible")
        for field in ("run_id", "generation_family"):
            if validate_identifier(field, published.get(field)) != identity.get(field):
                raise ValueError("proof lineage is incompatible")
        if (
            type(published.get("model_step")) is not int
            or published["model_step"] < 0
            or type(published.get("checkpoint_bytes")) is not int
            or published["checkpoint_bytes"] <= 0
        ):
            raise ValueError("legacy publication counters are invalid")
    for field in ("model_identity", "model_step", "run_id", "generation_family"):
        if pointer.get(field) != manifest.get(field):
            raise ValueError("champion pointer and winning manifest disagree")
    if pointer.get("manifest_sha256") != hashlib.sha256(
        raw["candidate-manifest.json"]
    ).hexdigest() or pointer.get("manifest_bytes") != len(
        raw["candidate-manifest.json"]
    ):
        raise ValueError("champion pointer manifest hash is incompatible")
    if (
        result.get("candidate") != manifest["model_identity"]
        or result.get("baseline") != baseline["model_identity"]
        or result["candidate"] == result["baseline"]
        or result.get("candidate_step") != manifest["model_step"]
        or result.get("champion_step") != baseline["model_step"]
        or result.get("result_kind") != "promotion"
        or result.get("terminal") is not True
        or result.get("conclusive") is not True
    ):
        raise ValueError(
            "proof does not conclusively promote this candidate against this baseline"
        )
    contract = result.get("evaluation_contract")
    if not isinstance(contract, dict) or contract.get("identity") != expected_contract:
        raise ValueError("unexpected evaluation contract")
    body = {key: value for key, value in contract.items() if key != "identity"}
    if expected_contract != "sha256-" + hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("evaluation contract checksum is invalid")
    if (
        contract.get("rules_hash") != _LEGACY_RULES
        or contract.get("search_algorithm") != "gumbel-completed-q-v3-conditional-keep"
        or contract.get("schema_version") != 3
    ):
        raise ValueError("unsupported legacy evaluation semantics")
    promotion = result.get("promotion")
    if not isinstance(promotion, dict) or any(
        promotion.get(key) != value
        for key, value in {
            "decision": "promote",
            "sequential_state": "accept_alternative",
            "cell_vetoes": [],
            "allocation_boundary_complete": True,
            "allocation_boundary_ready": True,
            "minimum_ready": True,
        }.items()
    ):
        raise ValueError("proof is not a complete, veto-free promotion")
    test = promotion["statistical_test"]
    alpha = contract["statistical_parameters"]["alpha"]
    evidence = test["promotion"]
    if (
        test.get("name") != contract.get("statistical_test")
        or test.get("observation_unit") != contract.get("observation_unit")
        or not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not 0 < alpha < 1
        or evidence.get("threshold") != 1 / alpha
        or evidence.get("error_probability") != alpha
        or not math.isfinite(evidence["e_value"])
        or evidence["e_value"] < evidence["threshold"]
    ):
        raise ValueError("promotion evidence does not meet its pinned error control")
    capture_body = {key: value for key, value in capture.items() if key != "sha256"}
    if (
        capture.get("sha256") != hashlib.sha256(_canonical(capture_body)).hexdigest()
        or verification.get("capture_sha256") != capture["sha256"]
    ):
        raise ValueError("verification is not bound to the original capture")
    captured = {entry["path"]: entry for entry in capture["files"]}
    for name in (
        "result.json",
        "allocation.json",
        "resume.json",
        "run.json",
        "candidate-manifest.json",
        "champion-manifest.json",
        "published-champion-pointer.json",
    ):
        entry = captured[name]
        if entry["sha256"] != hashlib.sha256(raw[name]).hexdigest() or entry[
            "bytes"
        ] != len(raw[name]):
            raise ValueError("verification capture hash disagrees with the proof")
    if (
        assessment.get("status") != "VERIFIED_CONCLUSIVE_PROMOTION"
        or assessment.get("verification_sha256")
        != hashlib.sha256(raw["verification.json"]).hexdigest()
        or verification.get("source_commit") != capture.get("source_commit")
        or verification.get("published_champion_pointer") != pointer
        or verification.get("promotion") != promotion
    ):
        raise ValueError(
            "conclusive verification receipt is not bound to this publication"
        )
    for record in (allocation, resume):
        if (
            record.get("candidate_identity") != result["candidate"]
            or record.get("baseline_identity") != result["baseline"]
            or record.get("run_id") != identity["run_id"]
            or record.get("generation_family") != identity["generation_family"]
        ):
            raise ValueError("companion evidence identity is incompatible")
    if allocation.get("evaluation_contract") != contract:
        raise ValueError("allocation contract disagrees with the result")
    pairs, games = result["pairs"], result["games"]
    state = resume["arena_state"]
    if (
        not isinstance(pairs, list)
        or not pairs
        or len(games) != 2 * len(pairs)
        or verification.get("pairs") != len(pairs)
        or verification.get("games") != len(games)
        or verification.get("resume_scope")
        != {
            "boundary_finished_games": len(games),
            "finished_games_checked": len(games),
            "game_states": len(games),
            "unfinished_games": 0,
        }
        or len(state["game_states"]) != len(games)
        or any(row.get("result") is None for row in state["game_states"])
    ):
        raise ValueError("verification does not cover every completed game")
    if (
        state.get("candidate") != result["candidate"]
        or state.get("baseline") != result["baseline"]
    ):
        raise ValueError("resumed game identities disagree")
    _validate_recorded_evaluation(documents)
    return pointer, manifest


def _equal_payload(expected: Any, actual: Any, path: tuple[str, ...] = ()) -> int:
    """Compare every leaf, including tensor bits, after the approved name mapping."""
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or expected.layout != actual.layout
        ):
            raise ValueError(f"migration changed tensor schema at {path}")
        before = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        after = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(before, after):
            raise ValueError(f"migration changed tensor values at {path}")
        return 1
    if type(expected) is not type(actual):
        raise ValueError(f"migration changed value type at {path}")
    if isinstance(expected, dict):
        if list(expected) != list(actual):
            raise ValueError(f"migration changed mapping/order at {path}")
        return sum(
            _equal_payload(value, actual[key], (*path, str(key)))
            for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise ValueError(f"migration changed sequence at {path}")
        return sum(
            _equal_payload(a, b, (*path, str(index)))
            for index, (a, b) in enumerate(zip(expected, actual, strict=True))
        )
    if expected != actual:
        raise ValueError(f"migration changed metadata at {path}")
    return 0


class PublicationDurabilityError(RuntimeError):
    """A complete pointer became visible; retain it for verification/recovery."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit(stage: Path, destination: Path) -> None:
    """Reserve a new root; expose its champion pointer only after every dependency."""
    destination.mkdir(mode=0o700)  # Atomic no-clobber reservation, including races.
    try:
        for source in sorted(stage.iterdir()):
            if source.name != "champion.json":
                os.rename(source, destination / source.name)
        # Persist every renamed dependency and the newly reserved root before
        # making a champion visible, including nested provenance directories.
        for directory in sorted(
            (p for p in destination.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        os.link(stage / "champion.json", destination / "champion.json")
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except BaseException as exc:
        if os.path.lexists(destination / "champion.json"):
            raise PublicationDurabilityError(
                f"complete publication retained at {destination}; final durability "
                "confirmation failed. Verify publication.json before use; do not "
                "delete or overwrite this publication automatically."
            ) from exc
        shutil.rmtree(destination)
        raise


def migrate_verified_champion(
    champion_pointer: str | Path,
    proof_bundle: str | Path,
    destination: str | Path,
    *,
    expected_pointer_sha256: str,
    expected_proof_manifest_sha256: str,
    expected_contract_identity: str,
) -> dict[str, Any]:
    """Create a new CPU-validated serving publication with its full proof closure.

    Supports the immediately preceding StarTrain publication layout only. The
    caller must supply independently approved hashes, not discover and bless them
    from the same untrusted inputs. Existing output roots are always refused.
    """
    _sha(expected_pointer_sha256)
    _sha(expected_proof_manifest_sha256)
    if not expected_contract_identity.startswith("sha256-"):
        raise ValueError("expected an explicit evaluation contract identity")
    _sha(expected_contract_identity.removeprefix("sha256-"))
    source_pointer = Path(champion_pointer).expanduser().absolute()
    if source_pointer.is_symlink():
        raise ValueError("source pointer cannot be a symlink")
    source_pointer = source_pointer.resolve(strict=True)
    bundle = Path(proof_bundle).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().absolute()
    if os.path.lexists(target):
        raise FileExistsError("migration requires a new publication directory")
    target = target.resolve(strict=False)
    if (
        target == source_pointer.parent
        or target.is_relative_to(source_pointer.parent)
        or target.is_relative_to(bundle)
    ):
        raise ValueError("migration output cannot be inside its source evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    pointer_bytes = source_pointer.read_bytes()
    if hashlib.sha256(pointer_bytes).hexdigest() != expected_pointer_sha256:
        raise ValueError("source champion pointer hash changed")
    raw, documents = _load_proof(bundle, expected_proof_manifest_sha256)
    pointer, old_manifest = _validate_proof(
        raw, documents, pointer_bytes, expected_contract_identity
    )
    manifest_relative = (
        Path("manifests") / f"manifest-{pointer['manifest_sha256']}.json"
    )
    if pointer.get("manifest") != manifest_relative.as_posix():
        raise ValueError("unsupported legacy publication manifest layout")
    checkpoint_relative = (
        Path("checkpoints") / f"sha256-{old_manifest['checkpoint_sha256']}.pt"
    )
    if old_manifest.get("checkpoint") != "../" + checkpoint_relative.as_posix():
        raise ValueError("unsupported legacy publication checkpoint layout")
    # Resolve the result by its captured name, not by an arbitrary JSON path.
    result_entries = [
        entry
        for entry in documents["capture.json"]["files"]
        if entry["path"] == "result.json"
    ]
    result_name = Path(result_entries[0]["source"]).name
    if pointer.get("promotion_result") != "../arena/" + result_name:
        raise ValueError("legacy champion does not link this exact promotion result")
    source_manifest = _file(source_pointer.parent, manifest_relative)
    source_checkpoint = _file(source_pointer.parent, checkpoint_relative)
    if source_manifest.read_bytes() != raw["candidate-manifest.json"]:
        raise ValueError("source manifest differs from verified publication")
    verify_file(
        source_checkpoint,
        expected_sha256=old_manifest["checkpoint_sha256"],
        expected_bytes=old_manifest["checkpoint_bytes"],
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.migration-", dir=target.parent
    ) as temporary:
        stage = Path(temporary)
        original = stage / "provenance/original/learner"
        _write_new(original / "champion.json", pointer_bytes)
        _write_new(original / manifest_relative, raw["candidate-manifest.json"])
        _copy_new(
            source_checkpoint,
            original / checkpoint_relative,
            old_manifest["checkpoint_sha256"],
            old_manifest["checkpoint_bytes"],
        )
        _write_new(original.parent / "arena" / result_name, raw["result.json"])
        for name, data in raw.items():
            _write_new(stage / "provenance/proof" / _relative(name), data)

        converted_path = stage / "checkpoints/migrated.pt"
        migrate_checkpoint(original / checkpoint_relative, converted_path)
        source_payload = torch.load(
            original / checkpoint_relative, map_location="cpu", weights_only=True
        )
        if source_payload.get("format") != "startrain.checkpoint":
            raise ValueError(
                "source checkpoint package is not the supported predecessor"
            )
        converted = torch.load(converted_path, map_location="cpu", weights_only=True)
        tensor_count = _equal_payload(migrate_payload(source_payload), converted)
        metadata = inspect_checkpoint(
            converted_path,
            expected_run_id=old_manifest["run_id"],
            expected_generation_family=old_manifest["generation_family"],
        )
        if metadata["step"] != old_manifest["model_step"] or not metadata["has_ema"]:
            raise ValueError(
                "converted checkpoint step/EMA disagrees with its champion"
            )
        profile = dict(metadata["config"])
        profile["train"] = {**profile["train"], "precision": "fp32", "compile": False}
        profile_path = stage / "profile-serving.yaml"
        _write_new(profile_path, yaml.safe_dump(profile, sort_keys=False).encode())
        experiment = load_config(profile_path)
        checkpoint_hash = sha256_file(converted_path)
        checkpoint = converted_path.with_name(f"sha256-{checkpoint_hash}.pt")
        converted_path.rename(checkpoint)
        manifest_payload = {
            "format": MODEL_MANIFEST_FORMAT,
            "schema_version": MODEL_MANIFEST_VERSION,
            "model_version": f"sha256-{checkpoint_hash}",
            "model_identity": f"sha256-{checkpoint_hash}",
            "model_step": metadata["step"],
            "checkpoint": f"../checkpoints/{checkpoint.name}",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "weights": "ema",
            "run_id": old_manifest["run_id"],
            "generation_family": old_manifest["generation_family"],
            "rules_hash": RULES_HASH_WIRE,
            "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "created_ns": time.time_ns(),
        }
        serialized = _json_bytes(manifest_payload)
        manifest_hash = hashlib.sha256(serialized).hexdigest()
        new_manifest_path = stage / "manifests" / f"manifest-{manifest_hash}.json"
        _write_new(new_manifest_path, serialized)
        manifest = load_model_manifest(new_manifest_path)
        model = GraphResTNet(experiment.model)
        load_ema_checkpoint(
            checkpoint,
            model=model,
            map_location="cpu",
            expected_model_config=asdict(experiment.model),
            expected_game_config=asdict(experiment.game),
            expected_run_id=manifest.run_id,
            expected_generation_family=manifest.generation_family,
            expected_sha256=checkpoint_hash,
            expected_bytes=manifest.checkpoint_bytes,
        )
        del model, converted, source_payload
        receipt = {
            "format": "deltreltrain.champion-identity-migration",
            "schema_version": 1,
            "claim": "lossless-identity-translation-of-verified-legacy-champion",
            "source_model_identity": old_manifest["model_identity"],
            "destination_model_identity": manifest.model_identity,
            "model_step": manifest.model_step,
            "run_id": manifest.run_id,
            "generation_family": manifest.generation_family,
            "source_pointer": _entry(original / "champion.json", stage),
            "source_checkpoint": _entry(original / checkpoint_relative, stage),
            "source_promotion_result": _entry(
                stage / "provenance/proof/result.json", stage
            ),
            "source_proof_inventory": _entry(
                stage / "provenance/proof/manifest.json", stage
            ),
            "source_verification": _entry(
                stage / "provenance/proof/verification.json", stage
            ),
            "evaluation_contract_identity": expected_contract_identity,
            "destination_manifest": _entry(new_manifest_path, stage),
            "destination_checkpoint": _entry(checkpoint, stage),
            "serving_profile": _entry(profile_path, stage),
            "preservation": {
                "all_payload_leaves_equal_after_approved_mapping": True,
                "tensor_leaves_compared_bitwise": tensor_count,
                "ema_loaded_on_cpu": True,
                "step": metadata["step"],
                "epoch": metadata["epoch"],
            },
            "migration_code": {
                "bridge_sha256": sha256_file(Path(__file__)),
                "rebrand_sha256": sha256_file(Path(__file__).with_name("rebrand.py")),
            },
            "profile_changes": {"train.precision": "fp32", "train.compile": False},
            "inference_configuration": {
                "precision": experiment.train.precision,
                "compile": experiment.train.compile,
                "score_utility_weight": experiment.selfplay.score_utility_weight,
            },
            "proof_scope": "Original proof still names only the legacy model. This receipt links a lossless representation; no new game outcome or absolute Elo claim is created.",
        }
        receipt_bytes = _json_bytes(receipt)
        receipt_path = (
            stage
            / "receipts"
            / f"identity-migration-{hashlib.sha256(receipt_bytes).hexdigest()}.json"
        )
        _write_new(receipt_path, receipt_bytes)
        # Deliberately no promotion_result/bootstrap: the old proof did not test
        # this new content hash. The publication inventory binds the mapping.
        write_model_pointer(stage / "champion.json", manifest, role="champion")
        inventory = {
            "format": "deltreltrain.verified-champion-publication",
            "schema_version": 1,
            "migration_receipt": _entry(receipt_path, stage),
            "files": [
                _entry(path, stage)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            ],
        }
        _write_new(stage / "publication.json", _json_bytes(inventory))
        # Revalidate external mutable inputs immediately before publication.
        if (
            source_pointer.read_bytes() != pointer_bytes
            or source_manifest.read_bytes() != raw["candidate-manifest.json"]
        ):
            raise ValueError("source publication changed during migration")
        verify_file(
            source_checkpoint,
            expected_sha256=old_manifest["checkpoint_sha256"],
            expected_bytes=old_manifest["checkpoint_bytes"],
        )
        current_raw, _ = _load_proof(bundle, expected_proof_manifest_sha256)
        if current_raw != raw:
            raise ValueError("proof bundle changed during migration")
        _commit(stage, target)
    return {
        "destination": str(target),
        "champion": str(target / "champion.json"),
        "profile": str(target / "profile-serving.yaml"),
        "publication": str(target / "publication.json"),
        "publication_sha256": sha256_file(target / "publication.json"),
        "model_identity": manifest.model_identity,
        "source_model_identity": old_manifest["model_identity"],
        "model_step": manifest.model_step,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--proof-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-pointer-sha256", required=True)
    parser.add_argument("--expected-proof-manifest-sha256", required=True)
    parser.add_argument("--expected-contract-identity", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    print(
        json.dumps(
            migrate_verified_champion(
                args.champion,
                args.proof_bundle,
                args.output,
                expected_pointer_sha256=args.expected_pointer_sha256,
                expected_proof_manifest_sha256=args.expected_proof_manifest_sha256,
                expected_contract_identity=args.expected_contract_identity,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
