"""Prepare/apply one sealed continuation's exact champion freshness transition.

This support operator never edits either qualified runtime. Committed migration
history is never reversed: recovery after commitment proceeds on qualified R4.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from deltreltrain import strength_freshness as freshness
from deltreltrain.checkpoint import sha256_file, verify_file
from deltreltrain.config import load_config
from deltreltrain.strength_recovery import completed_screen, digest, load_plan
from scripts.active_profile import resolve_active_profile, validate_profile_for_monitor
from scripts import deployment_metadata as transaction
from scripts import migrate_continuous_profile as migration
from scripts import run_strength_recovery_continuation as continuation
from scripts.prepare_strength_recovery import artifact, verify_artifact

JOURNAL = "strength-freshness-apply.json"
BOUNDARY = "strength-freshness-boundary.json"
RECEIPT = "strength-freshness-receipt.json"


def _new(path: Path, data: bytes) -> None:
    if path.resolve() != path.absolute():
        raise ValueError("freshness output may not traverse symbolic links")
    migration._atomic_replace_bytes(path, data, mode=0o444, overwrite=False)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _support_pins() -> list[dict[str, Any]]:
    base = Path(__file__).resolve().parents[1]
    return [
        {**artifact(base / relative), "relative_path": relative}
        for relative in (
            "scripts/activate_strength_freshness.py",
            "scripts/migrate_continuous_profile.py",
            "scripts/deployment_metadata.py",
            "scripts/run_strength_recovery_continuation.py",
            "scripts/active_profile.py",
            "deltreltrain/strength_freshness.py",
            "deltreltrain/strength_recovery_archive.py",
        )
    ]


def _verify_support(plan: dict[str, Any]) -> None:
    base = Path(__file__).resolve().parents[1]
    for pin in plan["support_implementation"]:
        relative = Path(pin["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("freshness support implementation path escaped")
        verify_artifact({**pin, "path": str(base / relative)})


def execution_pin(path: Path) -> dict[str, Any]:
    """A venv interpreter may link to the explicitly pinned host interpreter."""
    return {
        **artifact(path.resolve()),
        "path": str(path),
        "resolved_path": str(path.resolve()),
    }


def prepare(
    root: Path, *, runtime_root: Path, qualification: Path, after_ns: int
) -> dict[str, Any]:
    root = root.resolve()
    runtime_root = runtime_root.resolve()
    if (root / freshness.PLAN).exists():
        existing, _, _ = freshness.validate_registration(root)
        if (
            existing["activation_after_ns"] != after_ns
            or existing["runtime"]["training_root"] != str(runtime_root)
            or existing["runtime"]["qualification"]["sha256"]
            != sha256_file(qualification)
        ):
            raise ValueError(
                "existing freshness registration differs; never recompute its timestamp"
            )
        _verify_support(existing)
        freshness.verify_runtime(existing)
        return existing
    selected = resolve_active_profile(root)
    source, classification = validate_profile_for_monitor(root, selected)
    if (
        classification != "registered_recovery_continuation"
        or selected.path != root / freshness.SOURCE
    ):
        raise ValueError("freshness may follow only the sealed installed continuation")
    original = load_plan(root)
    assert original is not None
    old_plan = freshness.document(root / freshness.CONTINUATION_PLAN)
    metadata = completed_screen(root)
    state = freshness.document(root / freshness.CONTINUATION_STATE)
    if (
        state.get("continuation_started_ns") != metadata["resource_released_ns"]
        or state.get("plan_sha256") != old_plan["plan_sha256"]
    ):
        raise ValueError("freshness requires the original continuation clock")
    created = time.time_ns()
    if (
        type(after_ns) is not int
        or not metadata["resource_released_ns"] <= after_ns <= created
    ):
        raise ValueError("freshness timestamp must be explicit, fixed and nonfuture")
    target = freshness.target_config(source, after_ns)
    qualified = freshness.document(qualification)
    runtime = {
        "training_root": str(runtime_root),
        "source_commit": qualified["source_commit"],
        "python": str(runtime_root / ".venv/bin/python"),
        "orchestrator": str(runtime_root / ".venv/bin/deltreltrain-orchestrate"),
        "qualification": artifact(qualification),
        "source_checksums": artifact(runtime_root.parent / "SOURCE_SHA256SUMS"),
        "pyvenv": artifact(runtime_root / ".venv/pyvenv.cfg"),
        "import_pins": [
            artifact(Path(qualified["environment"][name]))
            for name in ("training_module", "native_file")
        ],
        "execution_pins": [
            execution_pin(runtime_root / ".venv/bin/python"),
            execution_pin(runtime_root / ".venv/bin/deltreltrain-orchestrate"),
        ],
    }
    freshness.verify_runtime({"runtime": runtime})
    # Validate historic implementation at its original location before preserving
    # it; new support code never masquerades as that original pinned controller.
    verify_artifact(old_plan["implementation"])
    for pin in original["implementation_pins"]:
        verify_artifact(pin)
    support = _support_pins()
    controls = {
        name: root / file
        for name, file in (
            ("recovery_plan", "strength-recovery-plan.json"),
            ("continuation_plan", freshness.CONTINUATION_PLAN),
            ("screen_seal", "strength-screen-seal.json"),
            ("ablation", "ablation.json"),
        )
    }
    destinations = [
        root / freshness.INPUT,
        root / freshness.SOURCE_MIGRATIONS,
        root / freshness.PLAN,
        root / freshness.PROVENANCE,
    ]
    if any(p.exists() or p.is_symlink() for p in destinations):
        raise ValueError("freshness preparation will not overwrite existing evidence")
    copied = []
    original_controls = {}

    def preserve(path: Path, name: str) -> dict[str, Any]:
        target = root / freshness.PROVENANCE / name
        _new(target, freshness.read(path))
        pin = artifact(target)
        copied.append(pin)
        return pin

    try:
        for name, path in controls.items():
            original_controls[name] = preserve(path, "original-" + path.name)
        runtime["qualification"] = preserve(qualification, "r4-qualification.json")
        runtime["source_checksums"] = preserve(
            runtime_root.parent / "SOURCE_SHA256SUMS", "r4-SOURCE_SHA256SUMS"
        )
        preserve(runtime_root / ".venv/pyvenv.cfg", "r4-pyvenv.cfg")
        preserve(
            Path(old_plan["implementation"]["path"]),
            "original-continuation-controller.py",
        )
        for index, pin in enumerate(support):
            preserve(Path(pin["path"]), f"support/{index:02d}-{Path(pin['path']).name}")
        _new(
            root / freshness.INPUT,
            yaml.safe_dump(
                json.loads(json.dumps(target.as_dict())), sort_keys=False
            ).encode(),
        )
        chain = freshness.read(root / "continuous-migrations.jsonl")
        if not chain.endswith(b"\n"):
            raise ValueError(
                "source migration ledger must have a complete newline-terminated record"
            )
        _new(root / freshness.SOURCE_MIGRATIONS, chain)
        plan = {
            "format": freshness.FORMAT,
            "schema_version": 1,
            "created_ns": created,
            "run_root": str(root),
            "recovery_plan_sha256": original["plan_sha256"],
            "continuation_plan_sha256": old_plan["plan_sha256"],
            "original_source_commit": old_plan["source_commit"],
            "continuation_started_ns": metadata["resource_released_ns"],
            "activation_after_ns": after_ns,
            "source_profile": artifact(root / freshness.SOURCE),
            "target_profile": artifact(root / freshness.INPUT),
            "target_profile_name": freshness.INSTALLED,
            "source_migrations": artifact(root / freshness.SOURCE_MIGRATIONS),
            "original_controls": original_controls,
            "artifacts": copied,
            "runtime": runtime,
            "support_implementation": support,
            "recovery_policy": "After the durable migration intent, complete only its original after-bytes at the unchanged stopped boundary and recover forward on the qualified runtime; never erase history or automatically restart R3.",
        }
        plan["plan_sha256"] = digest(plan)
        _new(root / freshness.PLAN, _json_bytes(plan))
        freshness.validate_registration(root)
        return plan
    except Exception:
        # All writes above are new preparation artifacts, never active authority.
        # Preserve them for diagnosis instead of silently overwriting a retry.
        raise


def runtime_environment(plan: dict[str, Any]) -> dict[str, str]:
    runtime = Path(plan["runtime"]["training_root"])
    # No inherited Python path/startup, dynamic-loader injection or stale CUDA
    # assignment may override the explicit qualified child environment.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "LD_", "DYLD_", "CUDA_"))
    }
    environment.update(
        PYTHONPATH=str(runtime),
        VIRTUAL_ENV=str(runtime / ".venv"),
        PYTHONNOUSERSITE="1",
        PATH=str(runtime / ".venv/bin") + os.pathsep + environment.get("PATH", ""),
    )
    return environment


def cuda_qualification(
    root: Path, plan: dict[str, Any], path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Admit a separately reviewed real CUDA receipt, never fabricate one here."""
    data = freshness.read(path)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("CUDA qualification differs from its explicit reviewed hash")
    receipt = json.loads(data)
    runtime = plan["runtime"]
    cpu = json.loads(freshness.pinned(runtime["qualification"]))
    expected = {
        "schema_version": 1,
        "format": "deltreltrain.strength-freshness-cuda-qualification",
        "status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "source_commit": runtime["source_commit"],
        "source_manifest_sha256": runtime["source_checksums"]["sha256"],
        "profile_sha256": plan["target_profile"]["sha256"],
        "recovery_pointer_sha256": sha256_file(root / "learner/recovery.json"),
        "execution_pins": runtime["execution_pins"],
        "pyvenv": runtime["pyvenv"],
        "training_module": cpu["environment"]["training_module"],
        "native_file": cpu["environment"]["native_file"],
        "native_binaries": cpu["environment"]["native_binaries"],
        "checks": {
            name: True
            for name in (
                "cuda_available",
                "learner_resume_step",
                "ema_preserved",
                "native_cuda_search",
                "all_workers_released",
            )
        },
    }
    if any(
        digest(receipt.get(key)) != digest(value) for key, value in expected.items()
    ):
        raise ValueError(
            "CUDA qualification does not bind this exact stopped transition"
        )
    return artifact(path)


def _unchanged_files(root: Path) -> list[Path]:
    paths = [
        root / name
        for name in (
            "run.json",
            "learner/recovery.json",
            "learner/champion.json",
            "learner/resume-cutover.json",
            "learner/utd-segment.json",
            "learner/cadence.json",
            freshness.CONTINUATION_STATE,
        )
    ]
    config = load_config(root / freshness.SOURCE)
    paths.append(root / config.orchestration.directories.status / "work-schedule.json")
    arena = root / "arena"
    if arena.exists():
        paths.extend(p for p in arena.rglob("*") if p.is_file())
    return sorted(p for p in paths if p.is_file())


def _boundary(
    root: Path,
    plan: dict[str, Any],
    operation: migration.MigrationPlan,
    cuda: dict[str, Any],
) -> dict[str, Any]:
    cuda_bytes = freshness.pinned(cuda)
    cuda_path = root / freshness.PROVENANCE / "r4-cuda-qualification.json"
    if cuda_path.exists():
        if freshness.read(cuda_path) != cuda_bytes:
            raise ValueError("preserved CUDA qualification changed")
    else:
        _new(cuda_path, cuda_bytes)
    cuda = artifact(cuda_path)
    info = operation.backup_evidence["recovery"]
    assert isinstance(info, dict)
    checkpoint = root / str(info["checkpoint"])
    retained = root / freshness.PROVENANCE / "checkpoints" / checkpoint.name
    if retained.resolve() != retained:
        raise ValueError("stopped checkpoint retention may not traverse symbolic links")
    retained.parent.mkdir(parents=True, exist_ok=True)
    if not retained.exists():
        os.link(checkpoint, retained)
        migration._fsync_directory(retained.parent)
    verify_file(
        retained,
        expected_sha256=str(info["checkpoint_sha256"]),
        expected_bytes=int(info["checkpoint_bytes"]),
    )
    controls = []
    authority_before = {}
    for name in ("continuous-migrations.jsonl", "profile.sha256", "source-commit.txt"):
        original = root / name
        target = root / freshness.PROVENANCE / "boundary" / name
        if not target.exists():
            _new(target, freshness.read(original))
        if freshness.read(target) != freshness.read(original):
            raise ValueError("stopped profile authority changed before intent")
        pin = artifact(target)
        controls.append(pin)
        authority_before[name] = pin
    for path in _unchanged_files(root):
        if (
            path.suffix == ".json"
            and path.stat().st_size <= 1024**2
            and path.parent != root / "arena"
        ):
            target = root / freshness.PROVENANCE / "boundary" / path.relative_to(root)
            if not target.exists():
                _new(target, path.read_bytes())
            if target.read_bytes() != path.read_bytes():
                raise ValueError(
                    "stopped boundary control differs from its preserved snapshot"
                )
            controls.append(artifact(target))
    body = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "continuation_started_ns": plan["continuation_started_ns"],
        "checkpoint": artifact(retained),
        "controls": controls,
        "authority_before": authority_before,
        "cuda_qualification": cuda,
        "migration_record": dict(operation.migration_record),
        "unchanged": [artifact(path) for path in _unchanged_files(root)],
    }
    body["sha256"] = digest(body)
    path = root / BOUNDARY
    if path.exists():
        if freshness.document(path) != body:
            raise ValueError(
                "a different stopped boundary requires new explicit preparation"
            )
    else:
        _new(path, _json_bytes(body))
    return body


def _receipt(root, plan) -> dict[str, Any]:
    freshness.validate_installed(root)
    result = {
        "schema_version": 1,
        "status": "committed-recover-forward-only",
        "plan_sha256": plan["plan_sha256"],
        "boundary_sha256": sha256_file(root / BOUNDARY),
        "original_source_commit": plan["original_source_commit"],
        "active_source_commit": plan["runtime"]["source_commit"],
        "continuation_started_ns": plan["continuation_started_ns"],
    }
    result["sha256"] = digest(result)
    path = root / RECEIPT
    if path.exists():
        if freshness.document(path) != result:
            raise ValueError("committed freshness receipt changed")
    else:
        _new(path, _json_bytes(result))
    return result


def apply(
    root: Path, *, cuda_receipt: Path | None = None, cuda_sha256: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    plan, _, _ = freshness.validate_registration(root)
    _verify_support(plan)
    freshness.verify_runtime(plan)
    continuation.require_idle_gpus()
    journal = root / JOURNAL
    if journal.exists():
        return repair(root)
    if cuda_receipt is None or cuda_sha256 is None:
        raise ValueError(
            "reviewed actual R4 CUDA qualification is required before migration intent"
        )
    cuda = cuda_qualification(root, plan, cuda_receipt, cuda_sha256)
    operation = migration.plan_migration(
        migration.MigrationRequest(
            old_profile=root / freshness.SOURCE,
            new_profile=root / freshness.INPUT,
            target_profile_name=freshness.INSTALLED,
            run_root=root,
            reason="Activate qualified champion-only freshness after sealed recovery continuation; preserve all learned state and original clocks",
            to_source_commit=plan["runtime"]["source_commit"],
        )
    )
    if (
        operation.utd_segment_payload is not None
        or operation.strength_epoch_payload is not None
    ):
        raise ValueError("freshness must preserve UTD and evaluation epochs")
    if (
        json.loads(freshness.pinned(cuda))["recovery_pointer_sha256"]
        != operation.migration_record["recovery_pointer_sha256"]
    ):
        raise ValueError("stopped recovery boundary changed after CUDA qualification")
    if (root / BOUNDARY).exists():
        retained = freshness.document(root / BOUNDARY)
        old_record = retained["migration_record"]

        def without_timestamp(record):
            return {k: v for k, v in record.items() if k != "timestamp_ns"}

        if without_timestamp(old_record) != without_timestamp(
            operation.migration_record
        ):
            raise ValueError("prepared stopped boundary changed before durable intent")
        operation = replace(operation, migration_record=old_record)
    boundary = _boundary(root, plan, operation, cuda)
    transaction.record_intent(operation, journal)
    migration.apply_migration(operation, rollback_on_failure=False)
    for pin in boundary["unchanged"]:
        verify_artifact(pin)
    transaction.mark_committed(journal)
    return _receipt(root, plan)


def repair(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan, source, target = freshness.validate_registration(root)
    _verify_support(plan)
    freshness.verify_runtime(plan)
    continuation.require_idle_gpus()
    journal = freshness.document(root / JOURNAL)
    boundary = freshness.document(root / BOUNDARY)
    if (
        boundary.get("sha256")
        != digest({k: v for k, v in boundary.items() if k != "sha256"})
        or boundary.get("plan_sha256") != plan["plan_sha256"]
        or journal.get("status") not in ("pending", "committed")
        or journal.get("target_profile") != freshness.INSTALLED
        or journal.get("recovery_pointer_sha256")
        != boundary["migration_record"]["recovery_pointer_sha256"]
    ):
        raise ValueError("freshness repair journal or stopped boundary differs")
    freshness.pinned(boundary["cuda_qualification"])
    freshness.validate_record(plan, source, target, boundary["migration_record"])
    checksum = (
        f"{plan['target_profile']['sha256']}  {root / freshness.INSTALLED}\n".encode()
    )
    expected = {
        freshness.INSTALLED: (freshness.pinned(plan["target_profile"]), 0o444),
        str(Path(freshness.INSTALLED).with_suffix(".sha256")): (checksum, 0o444),
        "continuous-migrations.jsonl": (
            freshness.pinned(plan["source_migrations"])
            + migration._json_bytes(boundary["migration_record"]),
            0o644,
        ),
        "profile.sha256": (checksum, 0o644),
        "source-commit.txt": (
            (plan["runtime"]["source_commit"] + "\n").encode(),
            0o644,
        ),
    }
    before: dict[str, bytes | None] = {
        name: freshness.pinned(pin)
        for name, pin in boundary["authority_before"].items()
    }
    if before.get("continuous-migrations.jsonl") != freshness.pinned(
        plan["source_migrations"]
    ):
        raise ValueError("stopped authority differs from original migration history")
    before.update(
        {
            freshness.INSTALLED: None,
            str(Path(freshness.INSTALLED).with_suffix(".sha256")): None,
        }
    )
    if set(before) != set(expected):
        raise ValueError("freshness stopped authority set differs")
    writes = journal.get("writes", [])
    if len(writes) != len(expected) or {row["path"] for row in writes} != set(expected):
        raise ValueError("freshness repair may write only the registered metadata")
    for row in writes:
        if (
            transaction._decode(row["before"]) != before[row["path"]]
            or (transaction._decode(row["after"]), row["after_mode"])
            != expected[row["path"]]
        ):
            raise ValueError(
                "freshness repair after-bytes differ from its registration"
            )
    # Never bypass the live coordinator exclusion, even when every write happened
    # before a crash. Once a committed R4 run advances, normal launch (not repair)
    # is the restart path.
    with migration._coordinator_write_guard(root):
        if journal["status"] == "pending":
            for pin in boundary["unchanged"]:
                verify_artifact(pin)
            record = boundary["migration_record"]
            replay = migration._read_replay_boundary(
                root,
                run_id=record["run_id"],
                generation_family=record["generation_family"],
                created_ns=freshness.document(root / "run.json")["created_ns"],
            )
            if (replay.committed_samples, replay.updated_ns) != (
                record["committed_replay_samples"],
                record["replay_counter_updated_ns"],
            ):
                raise ValueError("replay credit boundary changed before forward repair")
        status = transaction.repair_interrupted_intent(
            root / JOURNAL, root, forward=True
        )
    if status != "committed":
        raise ValueError("freshness repair did not complete forward")
    return _receipt(root, plan)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    with (root / ".strength-continuation.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _ = freshness.validate_installed(root)
        _verify_support(plan)
        freshness.verify_runtime(plan)
        if freshness.document(root / JOURNAL).get("status") != "committed":
            raise ValueError("repair metadata transaction before runtime launch")
        _receipt(root, plan)
        old = freshness.document(root / freshness.CONTINUATION_PLAN)
        return continuation.supervise_continuation(
            root,
            root / freshness.INSTALLED,
            old,
            plan["runtime"]["orchestrator"],
            interpreter=plan["runtime"]["python"],
            environment=runtime_environment(plan),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "apply", "repair", "run"):
        command = sub.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--runtime-root", type=Path, required=True)
            command.add_argument("--qualification", type=Path, required=True)
            command.add_argument("--after-ns", type=int, required=True)
        elif name == "apply":
            command.add_argument("--cuda-qualification", type=Path, required=True)
            command.add_argument("--cuda-sha256", required=True)
    args = parser.parse_args()
    try:
        result = (
            prepare(
                args.run_root,
                runtime_root=args.runtime_root,
                qualification=args.qualification,
                after_ns=args.after_ns,
            )
            if args.command == "prepare"
            else apply(
                args.run_root,
                cuda_receipt=args.cuda_qualification,
                cuda_sha256=args.cuda_sha256,
            )
            if args.command == "apply"
            else {"repair": repair, "run": run}[args.command](args.run_root)
        )
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(78) from None


if __name__ == "__main__":
    main()
