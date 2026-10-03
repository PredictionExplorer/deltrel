"""Exact post-screen admission for the already qualified champion-only mode.

Control-plane registration only: this module is not needed by coordinator or
learner execution. Original recovery/continuation authority remains immutable.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config import ExperimentConfig, load_config
from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE, SEARCH_ALGORITHM_ID
from .checkpoint import sha256_file, verify_file
from .strength_recovery import completed_screen, digest, load_plan as recovery_plan

PLAN = "strength-freshness-plan.json"
INPUT = "strength-freshness-input.yaml"
INSTALLED = "profile-strength-freshness.yaml"
SOURCE = "profile-strength-continuation.yaml"
SOURCE_MIGRATIONS = "strength-freshness-source-migrations.jsonl"
PROVENANCE = "strength-freshness-provenance"
FORMAT = "deltreltrain.strength-freshness-transition"
CONTINUATION_PLAN = "strength-continuation-plan.json"
CONTINUATION_STATE = "strength-continuation-state.json"
FIELDS = ("champion_only_replay_freshness", "protected_champion_after_ns")
BOUNDARY = "strength-freshness-boundary.json"
ATTEMPTS = PROVENANCE + "/attempts"
REPREPARE = "reprepare.json"


def attempt_directory(generation: int) -> str:
    return PROVENANCE if generation == 0 else f"{ATTEMPTS}/{generation:06d}"


def boundary_relative_path(generation: int) -> str:
    return (
        BOUNDARY if generation == 0 else attempt_directory(generation) + "/" + BOUNDARY
    )


def validate_repreparation(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    *,
    generation: int,
    previous_receipt_sha256: str | None,
    previous_boundary_sha256: str,
) -> None:
    """Validate the append-only attempt chain without following mutable pointers."""
    prior = receipt.get("abandoned_boundary", {})
    controls = receipt.get("source_authority", {})
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("format") != "deltreltrain.strength-freshness-repreparation"
        or receipt.get("status") != "abandoned-before-intent"
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or type(receipt.get("generation")) is not int
        or receipt["generation"] != generation
        or not 1 <= generation <= 999999
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", receipt.get("attempt_id", "")
        )
        or receipt.get("previous_receipt_sha256") != previous_receipt_sha256
        or prior.get("path")
        != str(Path(plan["run_root"]) / boundary_relative_path(generation - 1))
        or prior.get("sha256") != previous_boundary_sha256
        or receipt.get("continuation_started_ns") != plan["continuation_started_ns"]
        or receipt.get("activation_after_ns") != plan["activation_after_ns"]
        or type(receipt.get("created_ns")) is not int
        or receipt["created_ns"] < plan["created_ns"]
        or receipt.get("durable_intent_absent") is not True
        or receipt.get("sha256")
        != digest({k: v for k, v in receipt.items() if k != "sha256"})
        or set(controls)
        != {
            "continuous-migrations.jsonl",
            "profile.sha256",
            "source-commit.txt",
            "learner/recovery.json",
        }
    ):
        raise ValueError("freshness re-preparation chain or authority differs")
    for entry in controls.values():
        data = entry["text"].encode()
        if entry["sha256"] != hashlib.sha256(data).hexdigest() or entry["bytes"] != len(
            data
        ):
            raise ValueError("freshness re-preparation control snapshot differs")
    ledger = controls["continuous-migrations.jsonl"]
    profile = controls["profile.sha256"]["text"].split()
    if (
        any(ledger[k] != plan["source_migrations"][k] for k in ("sha256", "bytes"))
        or controls["source-commit.txt"]["text"].strip()
        != plan["original_source_commit"]
        or profile
        != [plan["source_profile"]["sha256"], str(Path(plan["run_root"]) / SOURCE)]
        or receipt["observed_migration_record"]["recovery_pointer_sha256"]
        != controls["learner/recovery.json"]["sha256"]
    ):
        raise ValueError(
            "freshness re-preparation did not preserve original R3 authority"
        )


def repreparations(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    folder = root / ATTEMPTS
    if folder.resolve() != folder:
        raise ValueError("freshness attempts may not traverse symbolic links")
    if not folder.exists():
        return []
    directories = sorted(folder.iterdir())
    result = []
    previous = None
    seen_ids = set()
    for directory in directories:
        if (
            directory.resolve() != directory
            or not directory.is_dir()
            or not re.fullmatch(r"[0-9]{6}", directory.name)
        ):
            raise ValueError("freshness attempt directory is invalid")
        # A crash before the one atomic receipt publication may leave an empty
        # next-generation directory; it carries no authority and is retryable.
        if not any(directory.iterdir()):
            continue
        generation = len(result) + 1
        if directory.name != f"{generation:06d}":
            raise ValueError("freshness attempt generations are not contiguous")
        receipt_path = directory / REPREPARE
        receipt = document(receipt_path)
        prior = root / boundary_relative_path(generation - 1)
        prior_data = read(prior)
        if pinned(receipt["abandoned_boundary"]) != prior_data:
            raise ValueError("abandoned freshness boundary differs")
        boundary = json.loads(prior_data)
        if boundary.get("plan_sha256") != plan["plan_sha256"] or boundary.get(
            "sha256"
        ) != digest({k: v for k, v in boundary.items() if k != "sha256"}):
            raise ValueError("abandoned freshness boundary identity differs")
        validate_repreparation(
            plan,
            receipt,
            generation=generation,
            previous_receipt_sha256=previous,
            previous_boundary_sha256=hashlib.sha256(prior_data).hexdigest(),
        )
        if generation > 1 and (
            boundary.get("generation") != generation - 1
            or boundary.get("repreparation_sha256") != previous
        ):
            raise ValueError("abandoned freshness boundary belongs to another attempt")
        if receipt["attempt_id"] in seen_ids:
            raise ValueError("freshness attempt identity was reused")
        seen_ids.add(receipt["attempt_id"])
        result.append(receipt)
        previous = sha256_file(receipt_path)
    return result


def current_boundary_path(root: Path, plan: dict[str, Any] | None = None) -> Path:
    registered = document(root / PLAN) if plan is None else plan
    generation = len(repreparations(root, registered))
    path = root / boundary_relative_path(generation)
    if generation and path.exists():
        boundary = document(path)
        if boundary.get("generation") != generation or boundary.get(
            "repreparation_sha256"
        ) != sha256_file(root / attempt_directory(generation) / REPREPARE):
            raise ValueError("freshness boundary belongs to another attempt")
    return path


def read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("freshness control metadata is missing or unsafe")
    return path.read_bytes()


def document(path: Path) -> dict[str, Any]:
    value = json.loads(read(path))
    if not isinstance(value, dict):
        raise ValueError("freshness control metadata must be an object")
    return value


def pinned(pin: dict[str, Any]) -> bytes:
    data = read(Path(pin["path"]))
    if len(data) != pin.get("bytes") or hashlib.sha256(data).hexdigest() != pin.get(
        "sha256"
    ):
        raise ValueError("freshness control artifact differs from its pin")
    return data


def target_config(source: ExperimentConfig, after_ns: int) -> ExperimentConfig:
    if (
        source.learner.champion_only_replay_freshness
        or source.learner.protected_champion_after_ns is not None
        or source.learner.protected_champion_fraction != 0
        or source.orchestration.model_refresh.selfplay_source != "champion"
        or type(after_ns) is not int
        or after_ns <= 0
    ):
        raise ValueError(
            "freshness requires the unmodified champion-only source profile"
        )
    return replace(
        source,
        learner=replace(
            source.learner,
            champion_only_replay_freshness=True,
            protected_champion_after_ns=after_ns,
        ),
    )


def changes(source: ExperimentConfig, target: ExperimentConfig) -> list[dict[str, Any]]:
    before, after = source.as_dict()["learner"], target.as_dict()["learner"]
    return [
        {
            "path": "learner." + name,
            "from": before.get(name, "<missing>"),
            "to": after[name],
        }
        for name in FIELDS
    ]


def validate_registration(
    root: Path,
) -> tuple[dict[str, Any], ExperimentConfig, ExperimentConfig]:
    root = root.resolve()
    plan = document(root / PLAN)
    if (
        plan.get("format") != FORMAT
        or plan.get("schema_version") != 1
        or plan.get("plan_sha256")
        != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        or Path(plan.get("run_root", "")).resolve() != root
        or plan.get("target_profile_name") != INSTALLED
    ):
        raise ValueError("freshness plan identity or checksum differs")
    recovery = recovery_plan(root)
    continuation = document(root / CONTINUATION_PLAN)
    completed = completed_screen(root)
    seal = document(root / "strength-screen-seal.json")
    if (
        recovery is None
        or plan.get("recovery_plan_sha256") != recovery["plan_sha256"]
        or continuation.get("format") != "deltreltrain.strength-continuation"
        or continuation.get("plan_sha256")
        != digest({k: v for k, v in continuation.items() if k != "plan_sha256"})
        or plan.get("continuation_plan_sha256") != continuation["plan_sha256"]
        or continuation.get("recovery_plan_sha256") != recovery["plan_sha256"]
        or seal.get("sha256")
        != digest({k: v for k, v in seal.items() if k != "sha256"})
        or seal.get("continuation_plan_sha256") != continuation["plan_sha256"]
        or seal.get("recovery_plan_sha256") != recovery["plan_sha256"]
        or seal.get("ablation") != completed
        or seal.get("ablation_sha256")
        != hashlib.sha256(read(root / "ablation.json")).hexdigest()
        or plan.get("continuation_started_ns") != completed["resource_released_ns"]
    ):
        raise ValueError("freshness activation requires the exact sealed continuation")
    for name, filename in (
        ("source_profile", SOURCE),
        ("target_profile", INPUT),
        ("source_migrations", SOURCE_MIGRATIONS),
    ):
        if Path(plan[name]["path"]).resolve() != root / filename:
            raise ValueError("freshness input escaped its registered path")
        pinned(plan[name])
    for pin in plan["artifacts"]:
        path = Path(pin["path"])
        if path.resolve() != path or not path.is_relative_to(root / PROVENANCE):
            raise ValueError("freshness provenance escaped the run")
        pinned(pin)
    for name, filename in (
        ("recovery_plan", "strength-recovery-plan.json"),
        ("continuation_plan", CONTINUATION_PLAN),
        ("screen_seal", "strength-screen-seal.json"),
        ("ablation", "ablation.json"),
    ):
        if (
            plan["original_controls"][name]["sha256"]
            != hashlib.sha256(read(root / filename)).hexdigest()
        ):
            raise ValueError("original screen/continuation control changed")
    if plan["source_profile"]["sha256"] != continuation["target_profile"]["sha256"]:
        raise ValueError("freshness source is not the registered continuation profile")
    if pinned(
        {
            **continuation["target_profile"],
            "path": str(root / "strength-continuation-input.yaml"),
        }
    ) != read(root / SOURCE):
        raise ValueError("original continuation input differs from installed profile")
    original = load_config(root / "profile-elo-ablation.yaml")
    if sha256_file(root / "profile-elo-ablation.yaml") != recovery["profile_sha256"]:
        raise ValueError("original screen profile changed")
    source, target = load_config(root / SOURCE), load_config(root / INPUT)
    if digest(source.as_dict()) != digest(
        replace(
            original,
            learner=replace(original.learner, candidate_interval_examples=3_000_000),
        ).as_dict()
    ):
        raise ValueError(
            "source continuation is not the registered cadence-only profile"
        )
    after = plan.get("activation_after_ns")
    created = plan.get("created_ns")
    if (
        type(after) is not int
        or type(created) is not int
        or not plan["continuation_started_ns"] <= after <= created
    ):
        raise ValueError("freshness activation timestamp must be fixed and nonfuture")
    if digest(target.as_dict()) != digest(target_config(source, after).as_dict()):
        raise ValueError("freshness changes more than the exact two learner fields")
    old_chain = read(root / SOURCE_MIGRATIONS)
    rows = [json.loads(line) for line in old_chain.splitlines()]
    if (
        not rows
        or rows[-1].get("to_profile") != SOURCE
        or rows[-1].get("to_profile_sha256") != plan["source_profile"]["sha256"]
        or rows[-1].get("to_source_commit") != continuation["source_commit"]
        or plan.get("original_source_commit") != continuation["source_commit"]
    ):
        raise ValueError("freshness source migration authority differs")
    state = document(root / CONTINUATION_STATE)
    if (
        state.get("plan_sha256") != continuation["plan_sha256"]
        or state.get("continuation_started_ns") != plan["continuation_started_ns"]
    ):
        raise ValueError("original continuation clock or identity changed")
    return plan, source, target


def validate_transition(
    source: ExperimentConfig, target: ExperimentConfig, root: Path
) -> bool:
    try:
        plan, registered_source, registered_target = validate_registration(root)
        return (
            digest(source.as_dict()) == digest(registered_source.as_dict())
            and digest(target.as_dict()) == digest(registered_target.as_dict())
            and read(root / "continuous-migrations.jsonl")
            == read(root / SOURCE_MIGRATIONS)
            and (root / "source-commit.txt").read_text().strip()
            == plan["original_source_commit"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_record(
    plan: dict[str, Any],
    source: ExperimentConfig,
    target: ExperimentConfig,
    row: dict[str, Any],
) -> None:
    previous = json.loads(pinned(plan["source_migrations"]).splitlines()[-1])
    expected = {
        "kind": "profile",
        "from_profile": SOURCE,
        "to_profile": INSTALLED,
        "from_profile_sha256": plan["source_profile"]["sha256"],
        "to_profile_sha256": plan["target_profile"]["sha256"],
        "from_source_commit": plan["original_source_commit"],
        "to_source_commit": plan["runtime"]["source_commit"],
        "from_config_sha256": previous["to_config_sha256"],
        "to_config_sha256": digest(target.as_dict()),
        "run_id": target.orchestration.run_id,
        "changes": changes(source, target),
    }
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError(
            "second migration differs from the registered freshness transition"
        )
    if row.get("utd_segment") is not None or row.get("strength_epoch") is not None:
        raise ValueError("freshness cannot restart UTD or evaluation epochs")


def validate_installed(root: Path) -> tuple[dict[str, Any], ExperimentConfig]:
    plan, source, target = validate_registration(root)
    if read(root / INSTALLED) != pinned(plan["target_profile"]):
        raise ValueError("installed freshness profile differs")
    before = read(root / SOURCE_MIGRATIONS)
    current = read(root / "continuous-migrations.jsonl")
    if not current.startswith(before):
        raise ValueError("freshness migration rewrote prior history")
    tail = current[len(before) :].splitlines()
    if len(tail) != 1:
        raise ValueError("freshness requires exactly one additional migration")
    row = json.loads(tail[0])
    boundary = document(current_boundary_path(root, plan))
    if (
        boundary.get("plan_sha256") != plan["plan_sha256"]
        or boundary.get("sha256")
        != digest({k: v for k, v in boundary.items() if k != "sha256"})
        or boundary.get("migration_record") != row
        or boundary.get("continuation_started_ns") != plan["continuation_started_ns"]
    ):
        raise ValueError(
            "freshness migration differs from its retained stopped boundary"
        )
    pinned(boundary["cuda_qualification"])
    validate_record(plan, source, target, row)
    if (root / "source-commit.txt").read_text().strip() != plan["runtime"][
        "source_commit"
    ]:
        raise ValueError("active source authority is not the qualified runtime")
    fields = (root / "profile.sha256").read_text().split()
    if (
        len(fields) != 2
        or fields[0] != plan["target_profile"]["sha256"]
        or (root / fields[1]).resolve() != root / INSTALLED
    ):
        raise ValueError("active profile authority is not registered freshness")
    return plan, target


def verify_runtime(plan: dict[str, Any]) -> None:
    runtime = plan["runtime"]
    root = Path(runtime["training_root"])
    qualification = json.loads(pinned(runtime["qualification"]))
    sums = pinned(runtime["source_checksums"])
    if (
        qualification.get("status") != "qualified-cpu-native-only"
        or qualification.get("source_commit") != runtime["source_commit"]
        or Path(qualification.get("release", "")).resolve() != root.parent
        or qualification.get("source_manifest_sha256")
        != hashlib.sha256(sums).hexdigest()
        or qualification["environment"]["native_rules"] != RULES_HASH_WIRE
        or qualification["environment"]["native_features"] != FEATURE_SCHEMA_HASH
        or qualification["environment"]["native_search"] != SEARCH_ALGORITHM_ID
        or (root.parent / "SOURCE_COMMIT").read_text().strip()
        != runtime["source_commit"]
    ):
        raise ValueError("freshness runtime differs from its independent qualification")
    if runtime["python"] != str(root / ".venv/bin/python") or runtime[
        "orchestrator"
    ] != str(root / ".venv/bin/deltreltrain-orchestrate"):
        raise ValueError(
            "freshness requires explicit qualified interpreter and entrypoint"
        )
    for line in sums.decode().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root.parent / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.resolve() != path
            or sha256_file(path) != expected
        ):
            raise ValueError("qualified runtime source changed")
    for pin in qualification["environment"]["native_binaries"]:
        if not Path(pin["path"]).is_relative_to(root / ".venv"):
            raise ValueError("qualified native escaped its runtime")
        verify_file(
            Path(pin["path"]),
            expected_sha256=pin["sha256"],
            expected_bytes=pin["bytes"],
        )
    if {pin["path"] for pin in runtime["execution_pins"]} != {
        runtime["python"],
        runtime["orchestrator"],
    }:
        raise ValueError("runtime execution pins must bind both exact entrypoints")
    for pin in runtime["execution_pins"]:
        path = Path(pin["path"])
        if str(path.resolve()) != pin["resolved_path"]:
            raise ValueError("runtime entrypoint resolved identity changed")
        verify_file(
            path.resolve(),
            expected_sha256=pin["sha256"],
            expected_bytes=pin["bytes"],
        )
    if Path(runtime["pyvenv"]["path"]) != root / ".venv/pyvenv.cfg":
        raise ValueError("runtime venv configuration escaped its qualified root")
    pinned(runtime["pyvenv"])
    for name, expected_root in (
        ("training_module", root / "deltreltrain"),
        ("native_file", root / ".venv"),
    ):
        origin = Path(qualification["environment"][name])
        if not origin.resolve().is_relative_to(expected_root) or origin.is_symlink():
            raise ValueError("qualified import origin escaped its runtime")
    if {pin["path"] for pin in runtime["import_pins"]} != {
        qualification["environment"]["training_module"],
        qualification["environment"]["native_file"],
    }:
        raise ValueError("runtime import pins must match qualified origins")
    for pin in runtime["import_pins"]:
        pinned(pin)
