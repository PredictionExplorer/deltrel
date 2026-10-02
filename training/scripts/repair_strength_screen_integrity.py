#!/usr/bin/env python3
"""Repair one clean R3 screen's missing final integrity receipt, then resume it.

Deploy this file outside an immutable release and invoke it by absolute filename
with PYTHONPATH pointing to that release. Imported validators and controller
remain the exact implementation registered in the recovery plan.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from scripts.preflight_run_state import state_apply_guard
from scripts.preserve_replay_snapshot import _stopped_boundary
from scripts.prepare_strength_recovery import verify_artifact
from scripts.run_elo_ablation import _post_cutoff_integrity
from scripts import run_strength_recovery_continuation as controller
from startrain.strength_recovery import completed_screen, digest, load_plan
from startrain.strength_recovery_archive import backup_files

FORMAT = "startrain.strength-screen-integrity-repair"
RECEIPT = "strength-screen-integrity-repair.json"
PENDING = "strength-screen-integrity-repair.pending.json"


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError(f"unsafe or oversized control metadata: {path}")
    data = path.read_bytes()
    if len(data) > 8 * 1024**2:
        raise ValueError("control metadata grew beyond its limit")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("control metadata must be an object")
    return value


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _write(path: Path, data: bytes, *, replace: bool) -> None:
    fd, name = tempfile.mkstemp(prefix=".screen-integrity-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            if os.geteuid() == 0:
                owner = path.stat() if path.exists() else path.parent.stat()
                os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def service_state(unit: str) -> dict[str, str]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service", unit) is None:
        raise ValueError("an explicit safe systemd service name is required")
    fields = (
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "InvocationID",
        "ExecStart",
    )
    result = subprocess.run(
        ["systemctl", "show", unit, "--no-pager", "--property=" + ",".join(fields)],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if any(field not in values for field in fields):
        raise ValueError("systemd service evidence is incomplete")
    return values


def _failed_service(state: dict[str, str], root: Path) -> None:
    if (
        state.get("ActiveState") != "failed"
        or state.get("Result") != "exit-code"
        or state.get("MainPID") != "0"
        or state.get("ExecMainCode") != "1"
        or state.get("ExecMainStatus") != "78"
        or re.fullmatch(r"[0-9a-f]{32}", state.get("InvocationID", "")) is None
        or "scripts.run_strength_recovery_continuation run"
        not in state.get("ExecStart", "")
        or str(root) not in state.get("ExecStart", "")
    ):
        raise ValueError(
            "only the stopped recovery controller's exact exit-78 invocation may be repaired"
        )


def _original_screen(
    root: Path, metadata: dict[str, Any], plan: dict[str, Any]
) -> None:
    started, cutoff, released = (
        metadata.get(name)
        for name in (
            "measurement_started_ns",
            "measurement_cutoff_ns",
            "resource_released_ns",
        )
    )
    teardown = metadata.get("measurement_teardown")
    attempts = metadata.get("measurement_attempts")
    if (
        plan.get("wall_budget_seconds") != 43200
        or metadata.get("wall_budget_seconds") != 43200
        or metadata.get("profile_sha256") != plan["profile_sha256"]
        or metadata.get("measurement_status") != "complete"
        or metadata.get("measurement_outcome") != "budget_completion"
        or metadata.get("measurement_completion_status") != "complete"
        or metadata.get("measurement_stop_reason") != "wall_budget"
        or metadata.get("integrity") is not None
        or metadata.get("integrity_status") is not None
        or metadata.get("measurement_failure") is not None
        or metadata.get("measurement_warnings") not in (None, [])
        or not isinstance(teardown, dict)
        or teardown.get("clean") is not True
        or teardown.get("process_group_released") is not True
        or type(started) is not int
        or type(cutoff) is not int
        or type(released) is not int
        or min(started, cutoff, released) <= 0
        or not started < cutoff <= released
        or cutoff - started < plan["wall_budget_seconds"] * 10**9
        or not isinstance(attempts, list)
        or not attempts
    ):
        raise ValueError(
            "screen is not an otherwise clean completed twelve-hour screen with missing integrity"
        )
    attempt = attempts[-1]
    if (
        not isinstance(attempt, dict)
        or metadata.get("measurement_attempt_count") != len(attempts)
        or any(
            attempt.get(key) != value
            for key, value in {
                "status": "complete",
                "completion_status": "complete",
                "outcome": "budget_completion",
                "stop_reason": "wall_budget",
                "measurement_cutoff_ns": cutoff,
                "resource_released_ns": released,
                "teardown": teardown,
                "integrity": None,
                "integrity_status": None,
                "failure": None,
            }.items()
        )
    ):
        raise ValueError("screen's final attempt does not match its clean completion")
    endpoint = _json(_read(root / "strength-recovery-snapshots/43200/snapshot.json"))
    if (
        endpoint.get("plan_sha256") != plan["plan_sha256"]
        or endpoint.get("scheduled_seconds") != 43200
    ):
        raise ValueError("screen lacks its exact retained twelve-hour endpoint")


def validate_receipt(value: dict[str, Any]) -> tuple[bytes, bytes]:
    if (
        value.get("format") != FORMAT
        or value.get("schema_version") != 1
        or value.get("sha256")
        != digest({k: v for k, v in value.items() if k != "sha256"})
    ):
        raise ValueError("integrity repair receipt schema or hash differs")
    decoded = []
    for name in ("original", "repaired"):
        pin = value[name]
        data = base64.b64decode(pin["payload_base64"], validate=True)
        if len(data) != pin["bytes"] or _sha(data) != pin["sha256"]:
            raise ValueError("embedded integrity repair metadata differs from its pin")
        decoded.append(data)
    before, after = map(_json, decoded)
    integrity = value["integrity"]
    expected = json.loads(json.dumps(before))
    expected.update(integrity=integrity, integrity_status="valid")
    expected["measurement_attempts"][-1].update(
        integrity=integrity, integrity_status="valid"
    )
    if (
        integrity.get("valid") is not True
        or integrity.get("status") != "valid"
        or after != expected
    ):
        raise ValueError("repair changes more than the measured integrity fields")
    return decoded[0], decoded[1]


def _unmigrated(
    root: Path, recovery: dict[str, Any], continuation: dict[str, Any]
) -> None:
    for name in (controller.SEAL, controller.INSTALLED, "continuous-migrations.jsonl"):
        if (root / name).exists():
            raise ValueError(
                "continuation already advanced; missing-integrity repair is inapplicable"
            )
    profile = root / "profile-elo-ablation.yaml"
    verify_artifact({**recovery["profile"], "path": str(profile)})
    if _read(root / "profile.sha256").decode().split() != [
        recovery["profile_sha256"],
        profile.name,
    ]:
        raise ValueError("active profile authority changed")
    if (
        _read(root / "source-commit.txt").decode().strip()
        != continuation["source_commit"]
    ):
        raise ValueError("qualified source authority changed")
    if (root / "coordinator.lock").exists():
        raise ValueError("a coordinator lock still exists")


def repair(
    root: Path, *, unit: str, apply: bool = False, restart: bool = False
) -> dict[str, Any]:
    if restart and not apply:
        raise ValueError("restart requires explicit apply")
    root = root.expanduser().resolve()
    state = service_state(unit)
    _failed_service(state, root)
    with (root / ".strength-continuation.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        continuation = controller.verify_plan(root)
        recovery = load_plan(root)
        assert recovery is not None
        _unmigrated(root, recovery, continuation)
        boundary = _stopped_boundary(root)
        controller.require_idle_gpus()
        path = root / "ablation.json"
        current = _read(path)
        saved = root / RECEIPT
        pending = root / PENDING
        receipt = (
            _json(_read(saved if saved.exists() else pending))
            if saved.exists() or pending.exists()
            else None
        )
        if receipt is not None:
            original, repaired = validate_receipt(receipt)
            if (
                receipt.get("unit") != unit
                or receipt.get("service") != state
                or receipt.get("recovery_plan_sha256") != recovery["plan_sha256"]
                or receipt.get("continuation_plan_sha256")
                != continuation["plan_sha256"]
                or receipt.get("stopped_boundary_sha256") != digest(boundary)
                or current not in (original, repaired)
            ):
                raise ValueError(
                    "repair state, plan, boundary, or failed service invocation changed"
                )
            _original_screen(root, _json(original), recovery)
        else:
            original = current
            metadata = _json(original)
            _original_screen(root, metadata, recovery)
            # Retain exact endpoint and provenance closure before admitting it.
            backup_files(root)
            integrity = _post_cutoff_integrity(
                root,
                root / "profile-elo-ablation.yaml",
                cutoff_ns=metadata["measurement_cutoff_ns"],
            )
            if integrity.get("valid") is not True:
                raise ValueError(
                    f"final state integrity failed: {integrity.get('failure')}"
                )
            metadata.update(integrity=integrity, integrity_status="valid")
            metadata["measurement_attempts"][-1].update(
                integrity=integrity, integrity_status="valid"
            )
            repaired = _bytes(metadata)
            receipt = {
                "format": FORMAT,
                "schema_version": 1,
                "unit": unit,
                "service": state,
                "recovery_plan_sha256": recovery["plan_sha256"],
                "continuation_plan_sha256": continuation["plan_sha256"],
                "stopped_boundary_sha256": digest(boundary),
                "integrity": integrity,
                "created_ns": time.time_ns(),
                **{
                    name: {
                        "sha256": _sha(data),
                        "bytes": len(data),
                        "payload_base64": base64.b64encode(data).decode(),
                    }
                    for name, data in (("original", original), ("repaired", repaired))
                },
            }
            receipt["sha256"] = digest(receipt)
            validate_receipt(receipt)
        if not apply:
            return {"status": "eligible", "receipt_sha256": receipt["sha256"]}
        # The controller lock and a final coordinator exclusion protect the
        # measured stop boundary. The preflight itself must run outside the
        # coordinator exclusion because it correctly rejects live owners.
        with state_apply_guard(root):
            if (
                _stopped_boundary(root) != boundary
                or service_state(unit) != state
                or _read(path) != current
            ):
                raise ValueError("stopped state changed during integrity verification")
            if not saved.exists() and not pending.exists():
                _write(pending, _bytes(receipt), replace=False)
            if current != repaired:
                _write(path, repaired, replace=True)
            completed_screen(root)
            if not saved.exists():
                _write(saved, _bytes(receipt), replace=False)
            pending.unlink(missing_ok=True)
    if restart:
        # Repeating this request for the SAME failed invocation is safe. A
        # subsequent controller failure has a new InvocationID and is rejected.
        if service_state(unit) != state:
            raise ValueError("service invocation changed before restart")
        subprocess.run(
            ["systemctl", "start", "--no-block", unit], check=True, timeout=15
        )
    return {
        "status": "repaired",
        "receipt_sha256": receipt["sha256"],
        "restart_requested": restart,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restart-unit", action="store_true")
    args = parser.parse_args()
    try:
        result = repair(
            args.run_root, unit=args.unit, apply=args.apply, restart=args.restart_unit
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"status": "ineligible", "error": str(error)}))
        return 78
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
