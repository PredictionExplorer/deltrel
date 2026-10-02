#!/usr/bin/env python3
"""Run a bounded recovery screen, seal it, and continue the same qualified run."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

import yaml

from scripts.migrate_continuous_profile import (
    MigrationRequest,
    migrate_continuous_profile,
)
from scripts.prepare_strength_recovery import artifact, verify_artifact
from scripts.run_elo_ablation import (
    _coordinator_owner_is_live,
    _resolve_executable,
    _terminate,
)
from scripts.run_strength_recovery import run as run_screen
from deltreltrain.config import load_config
from deltreltrain.runtime import SignalLatch, atomic_json
from deltreltrain.search_allocation_gate import validate_production_ring_allocations
from deltreltrain.strength_recovery import completed_screen, digest, load_plan
from deltreltrain.strength_recovery_archive import backup_files

PLAN = "strength-continuation-plan.json"
STATE = "strength-continuation-state.json"
SEAL = "strength-screen-seal.json"
FORMAT = "deltreltrain.strength-continuation"
INSTALLED = "profile-strength-continuation.yaml"


def qualified_environment() -> dict[str, str]:
    environment = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    inherited = [
        part
        for part in environment.get("PYTHONPATH", "").split(os.pathsep)
        if part and part != root
    ]
    environment["PYTHONPATH"] = os.pathsep.join([root, *inherited])
    return environment


@contextmanager
def qualified_sources():
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = qualified_environment()["PYTHONPATH"]
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


def prepare(root: Path, *, source_commit: str) -> dict[str, Any]:
    root = root.resolve()
    recovery = load_plan(root)
    if recovery is None or re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is None:
        raise ValueError(
            "continuation requires a pinned recovery plan and source commit"
        )
    backup_files(root)
    original = root / "profile-elo-ablation.yaml"
    verify_artifact({**recovery["profile"], "path": str(original)})
    source = load_config(original)
    target = replace(
        source, learner=replace(source.learner, candidate_interval_examples=3_000_000)
    )
    validate_production_ring_allocations(target, _fresh=True)
    target_path = root / "strength-continuation-input.yaml"
    contents = yaml.safe_dump(json.loads(json.dumps(target.as_dict())), sort_keys=False)
    if target_path.exists():
        if target_path.is_symlink() or target_path.read_text() != contents:
            raise ValueError("existing continuation profile differs")
    else:
        target_path.write_text(contents)
        target_path.chmod(0o444)
    plan = {
        "format": FORMAT,
        "schema_version": 1,
        "recovery_plan_sha256": recovery["plan_sha256"],
        "target_profile": artifact(target_path),
        "target_profile_name": INSTALLED,
        "implementation": {
            **artifact(Path(__file__)),
            "relative_path": "scripts/run_strength_recovery_continuation.py",
        },
        "source_commit": source_commit,
        "provisioned_gpus": 8,
        "candidate_interval_examples": 3_000_000,
        "continuation_budget": "continuous-until-operator-stop-or-failure",
    }
    plan["plan_sha256"] = digest(plan)
    path = root / PLAN
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != plan:
            raise ValueError("existing continuation plan differs")
    else:
        atomic_json(path, plan)
        path.chmod(0o444)
    return plan


def verify_plan(root: Path) -> dict[str, Any]:
    plan = json.loads((root / PLAN).read_text())
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    recovery = load_plan(root)
    if (
        plan.get("format") != FORMAT
        or plan.get("schema_version") != 1
        or plan.get("plan_sha256") != digest(body)
        or recovery is None
        or plan.get("recovery_plan_sha256") != recovery["plan_sha256"]
        or plan.get("target_profile_name") != INSTALLED
        or plan.get("candidate_interval_examples") != 3_000_000
        or plan.get("provisioned_gpus") != 8
    ):
        raise ValueError("continuation plan identity changed")
    verify_artifact(plan["target_profile"])
    verify_artifact({**plan["implementation"], "path": str(Path(__file__))})
    implementation_root = Path(__file__).resolve().parents[1]
    for pin in recovery["implementation_pins"]:
        relative = Path(pin["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("recovery implementation path is unsafe")
        verify_artifact({**pin, "path": str(implementation_root / relative)})
    source = load_config(root / "profile-elo-ablation.yaml")
    target = load_config(Path(plan["target_profile"]["path"]))
    if target != replace(
        source, learner=replace(source.learner, candidate_interval_examples=3_000_000)
    ):
        raise ValueError(
            "continuation changes more than the declared candidate cadence"
        )
    return plan


def seal_screen(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    metadata = completed_screen(root)
    backup_files(root)
    receipt = {
        "schema_version": 1,
        "continuation_plan_sha256": plan["plan_sha256"],
        "recovery_plan_sha256": plan["recovery_plan_sha256"],
        "ablation": metadata,
        "ablation_sha256": artifact(root / "ablation.json")["sha256"],
        "endpoint": json.loads(
            (root / "strength-recovery-snapshots/43200/snapshot.json").read_text()
        ),
        "provisioned_gpus": 8,
        "screen_provisioned_gpu_hours": 8
        * (metadata["resource_released_ns"] - metadata["measurement_started_ns"])
        / 3.6e12,
    }
    receipt["sha256"] = digest(receipt)
    path = root / SEAL
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError("completed screen changed after sealing")
    else:
        atomic_json(path, receipt)
        path.chmod(0o444)
        (root / "ablation.json").chmod(0o444)
    return receipt


def handoff(root: Path, plan: dict[str, Any]) -> Path:
    seal_screen(root, plan)
    target = root / INSTALLED
    if target.exists():
        verify_artifact({**plan["target_profile"], "path": str(target)})
        checksum = (root / "profile.sha256").read_text().split()[0]
        lines = (root / "continuous-migrations.jsonl").read_text().splitlines()
        latest = json.loads(lines[-1]) if lines else {}
        if checksum != plan["target_profile"]["sha256"] or (
            latest.get("to_profile") != INSTALLED
            or latest.get("to_profile_sha256") != checksum
            or latest.get("to_source_commit") != plan["source_commit"]
        ):
            raise ValueError(
                "existing continuation does not have a completed migration"
            )
        return target
    migrate_continuous_profile(
        MigrationRequest(
            old_profile=root / "profile-elo-ablation.yaml",
            new_profile=Path(plan["target_profile"]["path"]),
            target_profile_name=INSTALLED,
            run_root=root,
            reason="Completed twelve-hour recovery screen; separately accounted continuous training at unchanged calibrated rates and search",
            to_source_commit=plan["source_commit"],
        ),
        apply=True,
    )
    verify_artifact({**plan["target_profile"], "path": str(target)})
    return target


def require_idle_gpus() -> None:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    if query.stdout.strip():
        raise RuntimeError(
            "release all experimental GPU workers before launching the eight-GPU lineage"
        )


def supervise_continuation(
    root: Path, profile: Path, plan: dict[str, Any], orchestrator: str
) -> dict[str, Any]:
    executable = _resolve_executable(orchestrator)
    config = load_config(profile)
    validate_production_ring_allocations(config, _fresh=True)
    state_path = root / STATE
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("continuation process state belongs to another plan")
    state.setdefault("continuation_started_ns", time.time_ns())
    state.setdefault("attempts", [])
    state.update(
        schema_version=1,
        plan_sha256=plan["plan_sha256"],
        phase="continuation",
        profile=str(profile),
        provisioned_gpus=8,
    )
    latch = SignalLatch()
    previous = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    latch.install()
    (root / "logs").mkdir(exist_ok=True)
    try:
        for retry in range(4):
            require_idle_gpus()
            lock = root / "coordinator.lock"
            if lock.exists():
                owner = json.loads(lock.read_text())
                if type(owner.get("pid")) is not int or owner["pid"] <= 0:
                    raise RuntimeError("coordinator lock is malformed")
                if _coordinator_owner_is_live(lock):
                    raise RuntimeError("a live coordinator owns the continuation root")
                # The existing coordinator performs its normal dead-owner
                # cleanup when it acquires this lock; do not race that protocol.
            attempt = {"started_ns": time.time_ns(), "pid": None, "status": "starting"}
            state["attempts"].append(attempt)
            atomic_json(state_path, state)
            with (root / "logs/strength-continuation.log").open("ab") as output:
                process = subprocess.Popen(
                    [executable, "--config", str(profile)],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=qualified_environment(),
                )
                attempt.update(pid=process.pid, status="running")
                atomic_json(state_path, state)
                try:
                    while process.poll() is None and not latch.is_set():
                        time.sleep(1)
                finally:
                    teardown = _terminate(
                        process,
                        terminate_grace_seconds=config.orchestration.shutdown.terminate_grace_seconds
                        + 32,
                        kill_grace_seconds=config.orchestration.shutdown.kill_grace_seconds,
                    )
                attempt.update(
                    status="stopped", teardown=teardown, stopped_ns=time.time_ns()
                )
            state["continuation_provisioned_gpu_hours"] = (
                8 * (time.time_ns() - state["continuation_started_ns"]) / 3.6e12
            )
            if not teardown.get("clean") or not teardown.get("process_group_released"):
                state.update(
                    phase="failed", error="continuation workers did not release cleanly"
                )
                atomic_json(state_path, state)
                raise RuntimeError(state["error"])
            if latch.is_set():
                state.update(phase="stopped", stopped_ns=time.time_ns())
                atomic_json(state_path, state)
                return state
            code = process.returncode
            if code != 75 and (code is None or code >= 0):
                state.update(
                    phase="failed", error=f"continuation exited with code {code}"
                )
                atomic_json(state_path, state)
                raise RuntimeError(state["error"])
            state.update(phase="retrying", last_exit_code=code)
            atomic_json(state_path, state)
            if retry < 3:
                time.sleep(5)
        state.update(
            phase="failed",
            error="continuation exhausted its bounded crash restart allowance",
            stopped_ns=time.time_ns(),
        )
        atomic_json(state_path, state)
        raise RuntimeError(state["error"])
    finally:
        signal.signal(signal.SIGINT, previous[0])
        signal.signal(signal.SIGTERM, previous[1])


def run(root: Path, *, orchestrator: str) -> dict[str, Any]:
    root = root.resolve()
    with (root / ".strength-continuation.lock").open("a") as lock, qualified_sources():
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(root)
        metadata = json.loads((root / "ablation.json").read_text())
        if metadata.get("measurement_status") != "complete":
            require_idle_gpus()
            result = run_screen(root=root, orchestrator=orchestrator)
            if result.get("status") != "complete":
                # The server service may retry a transient screen crash. Its
                # original budget survives; signals/fatal exits never hand off.
                return {"phase": "screen", **result}
        profile = handoff(root, plan)
        return supervise_continuation(root, profile, plan, orchestrator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--orchestrator", default="deltreltrain-orchestrate")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.run_root, source_commit=args.source_commit)
        else:
            result = run(args.run_root, orchestrator=args.orchestrator)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 78
    print(json.dumps(result, indent=2))
    if result.get("status") == "retryable":
        return 75
    return 78 if result.get("status") in ("failed", "error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
