#!/usr/bin/env python3
"""Checkpointed profile deployment with a compatible-reader recovery path.

The plan pins an already staged release and candidate profile. Run preparation
and qualification while training remains active. An external systemd oneshot
should invoke --execute, with --recover in ExecStopPost for interruption safety.
No command runs merely by importing this module or without an explicit action.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any

from startrain.runtime import atomic_json
from scripts.migrate_continuous_profile import (
    MigrationRequest,
    apply_migration,
    plan_migration,
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def active_authority(root: Path) -> tuple[Path, str]:
    checksum, filename = (root / "profile.sha256").read_text().strip().split(maxsplit=1)
    path = Path(filename)
    require(digest(path) == checksum, "active profile checksum mismatch")
    return path, (root / "source-commit.txt").read_text().strip()


def validate_boundary(
    root: Path, *, strict: bool, expected_workers: list[str] | None = None
) -> dict[str, Any]:
    coordinator = read(root / "status/coordinator.json")
    learner = read(root / "status/learner.heartbeat.json")
    checkpoint = read(root / "learner/recovery.json")
    workers = coordinator.get("workers", {})
    if expected_workers is not None:
        require(
            set(workers) == set(expected_workers),
            "stopped worker inventory differs from plan",
        )
    for pid in [
        coordinator.get("coordinator_pid"),
        *(w.get("pid") for w in workers.values()),
    ]:
        require(
            not pid or not Path(f"/proc/{pid}").exists(), f"worker still running: {pid}"
        )
    step = learner.get("step")
    examples = learner.get("examples_consumed")
    if type(step) is int:
        require(checkpoint["step"] <= step, "checkpoint is ahead of learner")
    if type(examples) is int:
        require(
            checkpoint["examples_consumed"] <= examples,
            "checkpoint examples are ahead of learner",
        )
    if strict:
        require(
            coordinator.get("state") == "stopped" and not coordinator.get("failure"),
            "coordinator did not stop cleanly",
        )
        require(
            workers
            and all(
                w.get("state") == "stopped" and w.get("last_exit_code") == 0
                for w in workers.values()
            ),
            "a worker did not exit cleanly",
        )
        require(
            checkpoint["step"] == step and checkpoint["examples_consumed"] == examples,
            "final learner progress was not checkpointed",
        )
    file = root / "learner" / checkpoint["checkpoint"]
    require(
        file.resolve().parent == (root / "learner/recovery").resolve(),
        "recovery checkpoint escaped its immutable directory",
    )
    require(
        file.stat().st_size == checkpoint["checkpoint_bytes"]
        and digest(file) == checkpoint["checkpoint_sha256"],
        "recovery checkpoint verification failed",
    )
    return {
        "checkpoint": checkpoint,
        "learner": learner,
        "coordinator": coordinator,
        "discarded_uncheckpointed_steps": step - checkpoint["step"]
        if type(step) is int
        else None,
        "discarded_uncheckpointed_examples": examples - checkpoint["examples_consumed"]
        if type(examples) is int
        else None,
        "progress_available": type(step) is int and type(examples) is int,
        "observed_ns": time.time_ns(),
    }


class Deployment:
    def __init__(self, plan_path: Path):
        self.plan = read(plan_path)
        p = self.plan
        self.root = Path(p["run_root"])
        self.base = Path(p["evidence_root"])
        self.release = Path(p["target_release"])
        self.old_release = Path(p["source_release"])
        self.source = Path(p["source_profile"])
        self.candidate = Path(p["candidate_profile"])
        self.target = self.root / p["target_profile_name"]
        self.main = p["main_unit"]
        require(
            re.fullmatch(r"[a-zA-Z0-9_.@-]+\.service", self.main), "invalid main unit"
        )
        self.prefix = self.main.removesuffix(".service")
        self.support = [
            self.prefix + "-monitor.service",
            *[
                self.prefix + f"-{name}.timer"
                for name in ("report", "backup", "disaster-backup")
            ],
        ]
        self.oneshots = [
            self.prefix + f"-{name}.service"
            for name in ("report", "backup", "disaster-backup")
        ]
        self.units = [self.main, self.support[0], *self.oneshots]
        overrides = p.get("unit_source_releases", {})
        require(
            isinstance(overrides, dict) and set(overrides) <= set(self.units),
            "unit source releases must name only managed services",
        )
        self.unit_source_releases: dict[str, Path] = {}
        for unit, value in overrides.items():
            require(
                isinstance(value, str)
                and Path(value).is_absolute()
                and str(Path(value)) == value
                and ".." not in Path(value).parts
                and Path(value) != self.release,
                f"invalid source release for {unit}",
            )
            self.unit_source_releases[unit] = Path(value)
        reason = p.get(
            "migration_reason",
            "Pie-standard training policy; preserve learned state and replay",
        )
        require(
            isinstance(reason, str)
            and reason == reason.strip()
            and 1 <= len(reason) <= 256
            and "\n" not in reason
            and "\r" not in reason,
            "migration reason must be a single descriptive line",
        )
        self.migration_reason = reason
        self.python = str(self.release / "training/.venv/bin/python")
        self.env = os.environ | {
            "PYTHONPATH": str(self.release / "training"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        for name in ("source_commit", "target_commit"):
            require(re.fullmatch(r"[0-9a-f]{40}", p[name]), f"invalid {name}")
        require(
            self.root.is_absolute()
            and self.base.is_absolute()
            and self.release.is_absolute(),
            "deployment paths must be absolute",
        )
        require(
            self.target.parent == self.root and self.target.suffix == ".yaml",
            "unsafe target profile name",
        )
        require(
            self.release != self.old_release, "deployment needs an isolated release"
        )
        require(
            (self.release / "SOURCE_COMMIT").read_text().strip() == p["target_commit"],
            "staged source commit differs from plan",
        )
        require(
            digest(self.source) == p["source_profile_sha256"], "source profile changed"
        )
        require(
            digest(self.candidate) == p["candidate_profile_sha256"],
            "candidate profile changed",
        )
        require(
            digest(Path(p["smoke_script"])) == p["smoke_script_sha256"],
            "smoke script changed",
        )
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, value: Mapping[str, object]) -> None:
        atomic_json(self.base / name, value)

    def run(
        self,
        command: list[str],
        name: str,
        *,
        timeout: int = 180,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> str:
        print(json.dumps({"event": name, "started_ns": time.time_ns()}), flush=True)
        with (self.base / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                cwd=cwd or self.release / "training",
                env=environment or self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        require(result.returncode == 0, f"{name} failed; see its log")
        return (self.base / (name + ".log")).read_text()

    def show(self, unit: str, field: str) -> str:
        return subprocess.check_output(
            ["systemctl", "show", unit, "-p", field, "--value"], text=True
        ).strip()

    def script(self, name: str, args: list[str], label: str, timeout: int = 180) -> str:
        return self.run(
            [self.python, str(self.release / "training/scripts" / name), *args],
            label,
            timeout=timeout,
        )

    def unit_source_release(self, unit: str) -> Path:
        return self.unit_source_releases.get(unit, self.old_release)

    def validate_source_unit(self, unit: str, contents: str) -> None:
        expected = self.unit_source_release(unit)
        training = str(expected / "training")
        require(
            str(expected) in contents, f"saved unit lacks its source release: {unit}"
        )
        require(
            self.show(unit, "WorkingDirectory") == training,
            f"source unit working directory differs from plan: {unit}",
        )
        require(
            training + "/" in self.show(unit, "ExecStart"),
            f"source unit command differs from plan: {unit}",
        )

    def validate_prepared(self) -> None:
        require(
            active_authority(self.root) == (self.source, self.plan["source_commit"]),
            "live source authority changed",
        )
        require(
            self.show(self.main, "ActiveState") == "active",
            "source workload is not active",
        )
        require(
            self.show(self.main, "WorkingDirectory")
            == str(self.unit_source_release(self.main) / "training"),
            "source unit points to another release",
        )
        self.run(
            ["sha256sum", "-c", str(self.release / "SOURCE_SHA256SUMS")],
            "verify-source",
            timeout=120,
            cwd=self.release,
        )
        self.script(
            "validate_continuous_profile.py",
            ["--config", str(self.candidate)],
            "validate-candidate",
        )
        self.script(
            "hardware_health_preflight.py",
            [
                "--config",
                str(self.candidate),
                "--output",
                str(self.base / "hardware-before.json"),
            ],
            "hardware-before",
        )
        require(not self.target.exists(), "target already exists before deployment")
        require(
            set(read(self.root / "status/coordinator.json")["workers"])
            == set(self.plan["expected_workers"]),
            "source worker inventory differs from plan",
        )
        snapshot = Path(self.plan["preserved_snapshot"])
        require(
            not snapshot.exists() and snapshot.parent.is_dir(),
            "snapshot destination is not prepared",
        )

    def prepare(self) -> None:
        require(
            not (self.base / "prepared.json").exists(),
            "deployment already prepared; use recovery",
        )
        self.validate_prepared()
        folder = self.base / "units"
        folder.mkdir()
        for unit in self.units:
            path = Path(self.show(unit, "FragmentPath"))
            require(
                path == Path("/etc/systemd/system") / unit and not path.is_symlink(),
                "unit is not the expected regular local fragment",
            )
            require(not self.show(unit, "DropInPaths"), "unit has unplanned drop-ins")
            self.validate_source_unit(unit, path.read_text())
            shutil.copy2(path, folder / unit)
        self.save(
            "prepared.json",
            {
                "plan": self.plan,
                "unit_sha256": {unit: digest(folder / unit) for unit in self.units},
                "support_states": {
                    unit: {
                        "active": self.show(unit, "ActiveState"),
                        "enabled": self.show(unit, "UnitFileState"),
                    }
                    for unit in self.support
                },
                "prepared_ns": time.time_ns(),
            },
        )

    def drain_support(self) -> None:
        self.run(["sudo", "systemctl", "stop", *self.support], "drain-timers")
        deadline = time.monotonic() + 1800
        while any(
            self.show(unit, "ActiveState") in {"active", "activating", "deactivating"}
            for unit in self.oneshots
        ):
            require(
                time.monotonic() < deadline, "support jobs did not drain in 30 minutes"
            )
            time.sleep(10)

    def restore_support(self) -> None:
        states = read(self.base / "prepared.json")["support_states"]
        active = [unit for unit in self.support if states[unit]["active"] == "active"]
        if active:
            self.run(["sudo", "systemctl", "start", *active], "restore-support")
        for unit in self.support:
            require(
                self.show(unit, "UnitFileState") == states[unit]["enabled"],
                "support enablement changed",
            )
            require(
                self.show(unit, "ActiveState") == states[unit]["active"],
                "support activation was not restored",
            )

    def stop(self, *, strict: bool) -> dict[str, Any]:
        self.save("stop-requested.json", {"requested_ns": time.time_ns()})
        self.run(
            ["sudo", "systemctl", "stop", self.main], "stop-workload", timeout=1100
        )
        require(self.show(self.main, "MainPID") == "0", "main process survived stop")
        boundary = validate_boundary(
            self.root, strict=strict, expected_workers=self.plan["expected_workers"]
        )
        self.save("stopped.json", boundary)
        owners = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
        require(not owners.strip(), "GPU workloads remain after shutdown")
        return boundary

    def install_units(self, profile: Path, *, original: bool = False) -> None:
        prepared = read(self.base / "prepared.json")
        for unit in self.units:
            source = self.base / "units" / unit
            require(
                digest(source) == prepared["unit_sha256"][unit], "saved unit changed"
            )
            contents = source.read_text()
            if not original:
                old_release = self.unit_source_release(unit)
                require(
                    str(old_release) in contents,
                    f"saved unit lacks its source release: {unit}",
                )
                contents = contents.replace(
                    str(old_release), str(self.release)
                ).replace(str(self.source), str(profile))
            destination = self.base / (unit + ".next")
            destination.write_text(contents)
            self.run(
                [
                    "sudo",
                    "install",
                    "-m",
                    "644",
                    str(destination),
                    "/etc/systemd/system/" + unit,
                ],
                "install-" + unit,
            )
        self.run(["sudo", "systemctl", "daemon-reload"], "reload-units")

    def migrate(
        self,
        source: Path,
        candidate: Path,
        target: str,
        source_commit: str,
        *,
        recovery: bool = False,
    ) -> dict[str, object]:
        request = MigrationRequest(
            old_profile=source,
            new_profile=candidate,
            target_profile_name=target,
            reason=self.migration_reason,
            run_root=self.root,
            from_source_commit=source_commit,
            to_source_commit=self.plan["target_commit"],
        )
        plan = plan_migration(request)
        require(
            recovery or plan.heartbeat_step == plan.learner_step,
            "migration would discard learner progress",
        )
        self.save("migration-plan-" + target + ".json", plan.output(mode="dry-run"))
        from scripts.deployment_metadata import record_intent, mark_committed

        intent = self.base / ("metadata-intent-" + target + ".json")
        record_intent(plan, intent)
        result = apply_migration(plan)
        mark_committed(intent)
        self.save("migration-result-" + target + ".json", result)
        return result

    def wait_ready(
        self, checkpoint: dict[str, Any], launched_ns: int
    ) -> dict[str, Any]:
        expected = "recovery:" + str(
            (self.root / "learner" / checkpoint["checkpoint"]).resolve()
        )
        deadline, stable_since = time.monotonic() + 1200, None
        while time.monotonic() < deadline:
            now = time.time_ns()
            coordinator = read(self.root / "status/coordinator.json")
            learner = read(self.root / "status/learner.heartbeat.json")
            workers = coordinator.get("workers", {})
            current = coordinator.get("timestamp_ns", 0) > launched_ns
            okay = (
                current
                and coordinator.get("state") == "running"
                and set(workers) == set(self.plan["expected_workers"])
            )
            okay = okay and self.show(self.main, "MainPID") == str(
                coordinator.get("coordinator_pid")
            )
            okay = okay and self.show(self.main, "ActiveState") == "active"
            okay = (
                okay and 0 <= now - coordinator.get("timestamp_ns", 0) < 30_000_000_000
            )
            require(
                not current or not coordinator.get("failure"),
                "coordinator reported failure",
            )
            for name, worker in workers.items():
                path = Path(worker["heartbeat"])
                heartbeat = read(path) if path.exists() else {}
                if current:
                    require(
                        not worker.get("restart_count")
                        and not worker.get("failure_reason"),
                        "worker failed or restarted",
                    )
                okay = okay and worker.get("state") in ("running", "paused")
                okay = (
                    okay
                    and heartbeat.get("pid") == worker.get("pid")
                    and launched_ns < heartbeat.get("heartbeat_ns", 0) <= now
                )
                if worker.get("state") != "paused":
                    okay = (
                        okay
                        and 0 <= now - heartbeat.get("heartbeat_ns", 0) < 30_000_000_000
                    )
                inference = heartbeat.get("inference", {})
                if current and heartbeat.get("heartbeat_ns", 0) > launched_ns:
                    require(
                        not inference.get("failed_requests", 0),
                        "inference request failed",
                    )
                if name.startswith("actor-gpu-") and worker.get("state") != "paused":
                    okay = okay and inference.get("completed_requests", 0) > 0
            progressed = learner.get("step", 0) > checkpoint["step"]
            if current and learner.get("heartbeat_ns", 0) > launched_ns and progressed:
                require(
                    learner.get("resumed_from") == expected,
                    "learner resumed a different checkpoint",
                )
            okay = okay and progressed and learner.get("resumed_from") == expected
            observation = {
                "ready": bool(okay),
                "step": learner.get("step"),
                "expected_resume": expected,
                "actual_resume": learner.get("resumed_from"),
                "observed_ns": now,
            }
            self.save("readiness.json", observation)
            print(json.dumps(observation), flush=True)
            if okay:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 60:
                    return observation
            else:
                stable_since = None
            time.sleep(10)
        raise RuntimeError("new workload failed sustained readiness")

    def start(self, profile: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
        self.script(
            "preflight_run_state.py",
            ["--run-root", str(self.root), "--profile", str(profile), "--dry-run"],
            "state-preflight",
            timeout=240,
        )
        launched = time.time_ns()
        self.save(
            "launch.json",
            {
                "launched_ns": launched,
                "checkpoint": checkpoint,
                "profile": str(profile),
            },
        )
        self.run(
            ["sudo", "systemctl", "start", self.main], "start-workload", timeout=1400
        )
        return self.wait_ready(checkpoint, launched)

    def execute(self) -> None:
        self.prepare()
        self.drain_support()
        self.validate_prepared()
        boundary = self.stop(strict=True)
        from scripts.preserve_replay_snapshot import preserve_stopped_snapshot

        preserved = preserve_stopped_snapshot(
            self.root, Path(self.plan["preserved_snapshot"])
        )
        self.save("preserved-snapshot.json", preserved)
        self.run(
            [
                self.python,
                self.plan["smoke_script"],
                "--profile",
                str(self.candidate),
                "--checkpoint",
                str(self.root / "learner" / boundary["checkpoint"]["checkpoint"]),
                "--output",
                str(self.base / "cuda-smoke.json"),
            ],
            "cuda-smoke",
            timeout=900,
            environment=self.env | {"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.migrate(
            self.source, self.candidate, self.target.name, self.plan["source_commit"]
        )
        self.install_units(self.target)
        observation = self.start(self.target, boundary["checkpoint"])
        self.restore_support()
        self.save(
            "complete.json",
            {
                "status": "complete",
                "source_commit": self.plan["target_commit"],
                "profile": str(self.target),
                "readiness": observation,
                "completed_ns": time.time_ns(),
            },
        )

    def recover(self) -> None:
        if (self.base / "complete.json").exists() or (
            self.base / "recovered.json"
        ).exists():
            return
        if not (self.base / "prepared.json").exists():
            return
        require(
            read(self.base / "prepared.json").get("plan") == self.plan,
            "recovery plan differs from prepared authority",
        )
        if not (self.base / "stop-requested.json").exists():
            self.restore_support()
            return
        boundary = self.stop(strict=False)
        if not boundary["progress_available"]:
            self.save("early-startup-heartbeat.json", boundary)
            heartbeat = boundary["learner"]
            pid = heartbeat.get("pid")
            require(
                type(pid) is int and pid > 0 and not Path(f"/proc/{pid}").exists(),
                "early-start recovery requires the stopped learner identity",
            )
            # Preserve the raw evidence: this stopped status is derived from
            # the checkpoint, and makes no claim about unreported updates.
            atomic_json(
                self.root / "status/learner.heartbeat.json",
                {
                    "schema_version": 1,
                    "worker": "learner",
                    "pid": pid,
                    "heartbeat_ns": time.time_ns(),
                    "phase": "stopped-checkpoint-recovery",
                    "step": boundary["checkpoint"]["step"],
                    "examples_consumed": boundary["checkpoint"]["examples_consumed"],
                    "observed_progress_available": False,
                    "derived_from_checkpoint_sha256": boundary["checkpoint"][
                        "checkpoint_sha256"
                    ],
                },
            )
        from scripts.deployment_metadata import repair_interrupted_intent

        for intent in sorted(self.base.glob("metadata-intent-*.json")):
            repair_interrupted_intent(intent, self.root)
        profile, commit = active_authority(self.root)
        recovery_path = self.base / "recovery-intent.json"
        if profile != self.source:
            if (
                recovery_path.exists()
                and profile == self.root / read(recovery_path)["profile_name"]
            ):
                require(
                    commit == self.plan["target_commit"]
                    and digest(profile) == digest(self.source),
                    "rollback authority differs from intent",
                )
            else:
                require(
                    profile == self.target and commit == self.plan["target_commit"],
                    "unexpected recovery authority",
                )
                name = (
                    "profile-pie-compatible-rollback-" + str(time.time_ns()) + ".yaml"
                )
                self.save("recovery-intent.json", {"profile_name": name})
                self.migrate(profile, self.source, name, commit, recovery=True)
                profile = self.root / name
            self.install_units(profile)
        else:
            require(
                commit == self.plan["source_commit"],
                "source commit changed during recovery",
            )
            self.install_units(self.source, original=True)
        observation = self.start(profile, boundary["checkpoint"])
        self.restore_support()
        self.save(
            "recovered.json",
            {
                "status": "recovered",
                "profile": str(profile),
                "readiness": observation,
                "stopped_boundary": boundary,
                "recovered_ns": time.time_ns(),
            },
        )


class DeploymentInterrupted(Exception):
    """Catchable by the metadata migrator's transactional rollback."""


def interrupt(signum: int, _frame: object) -> None:
    raise DeploymentInterrupted(f"deployment interrupted by signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deployment = Deployment(args.plan)
    if args.recover:
        deployment.recover()
    else:
        try:
            deployment.execute()
        except BaseException as error:
            deployment.save(
                "failure.json", {"error": str(error), "failed_ns": time.time_ns()}
            )
            raise


if __name__ == "__main__":
    main()
