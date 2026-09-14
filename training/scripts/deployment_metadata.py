"""Durable repair journal for an interrupted, stopped-run metadata migration.

Only the exact before/after bytes of the owned migration may be repaired.
Checkpoint and replay data are never restored or overwritten by this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from startrain.runtime import atomic_json
from scripts.migrate_continuous_profile import (
    MigrationPlan,
    _atomic_write_bytes,
    _fsync_directory,
    _json_bytes,
)


def _bytes(path: Path) -> bytes | None:
    if path.is_symlink():
        raise ValueError("metadata repair does not follow symbolic links")
    return path.read_bytes() if path.exists() else None


def _encode(value: bytes | None) -> str | None:
    return base64.b64encode(value).decode() if value is not None else None


def _decode(value: str | None) -> bytes | None:
    return base64.b64decode(value, validate=True) if value is not None else None


def record_intent(plan: MigrationPlan, path: Path) -> None:
    if path.exists():
        raise ValueError("migration intent already exists")
    root = plan.run_root
    log = root / "continuous-migrations.jsonl"
    previous_log = _bytes(log) or b""
    separator = b"\n" if previous_log and not previous_log.endswith(b"\n") else b""
    writes = {
        plan.target_profile: (plan.target_profile_bytes, 0o444),
        plan.target_profile_checksum: (plan.profile_sha256_bytes, 0o444),
        log: (previous_log + separator + _json_bytes(plan.migration_record), 0o644),
        root / "profile.sha256": (plan.profile_sha256_bytes, 0o644),
        root / "source-commit.txt": (plan.source_commit_bytes, 0o644),
    }
    if plan.utd_segment_payload is not None:
        writes[plan.utd_segment_path] = (_json_bytes(plan.utd_segment_payload), 0o644)
    if plan.strength_epoch_payload is not None:
        writes[plan.strength_epoch_path] = (
            _json_bytes(plan.strength_epoch_payload),
            0o644,
        )
    atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "pending",
            "run_root": str(root),
            "target_profile": plan.target_profile.name,
            "recovery_pointer_sha256": plan.migration_record["recovery_pointer_sha256"],
            "writes": [
                {
                    "path": str(file.relative_to(root)),
                    "before": _encode(_bytes(file)),
                    "before_mode": file.stat().st_mode & 0o777
                    if file.exists()
                    else None,
                    "after": _encode(after),
                    "after_mode": mode,
                }
                for file, (after, mode) in writes.items()
            ],
        },
    )


def mark_committed(path: Path) -> None:
    value = json.loads(path.read_text())
    value["status"] = "committed"
    atomic_json(path, value)


def repair_interrupted_intent(path: Path, root: Path) -> str:
    value: dict[str, Any] = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("run_root") != str(root):
        raise ValueError("migration repair journal belongs to another root")
    if value.get("status") in ("committed", "repaired"):
        return value["status"]
    if value.get("status") != "pending":
        raise ValueError("unknown migration journal state")
    target = value.get("target_profile")
    if (
        not isinstance(target, str)
        or Path(target).name != target
        or not target.endswith(".yaml")
    ):
        raise ValueError("repair journal target profile is invalid")
    allowed = {
        target,
        str(Path(target).with_suffix(".sha256")),
        "continuous-migrations.jsonl",
        "profile.sha256",
        "source-commit.txt",
        "learner/utd-segment.json",
        "strength-epoch.json",
    }
    names = [row.get("path") for row in value.get("writes", [])]
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError("repair journal may name only unique owned metadata paths")
    rows = []
    for row in value["writes"]:
        file = root / row["path"]
        if file.resolve().is_relative_to(root.resolve()) is False or file.is_symlink():
            raise ValueError("migration repair path escaped its run")
        before, after, actual = (
            _decode(row["before"]),
            _decode(row["after"]),
            _bytes(file),
        )
        # A killed append may leave a strict prefix of the owned new log line.
        partial_log = (
            file.name == "continuous-migrations.jsonl"
            and actual is not None
            and (
                after is not None
                and after.startswith(actual)
                and actual.startswith(before or b"")
            )
        )
        if actual not in (before, after) and not partial_log:
            raise ValueError("metadata differs from both authorized transaction states")
        rows.append((file, row, before, after, actual))
    if all(actual == after for _, _, _, after, actual in rows):
        mark_committed(path)
        return "committed"
    pointer = _bytes(root / "learner/recovery.json")
    if (
        pointer is None
        or hashlib.sha256(pointer).hexdigest() != value["recovery_pointer_sha256"]
    ):
        raise ValueError("cannot repair metadata after learned state advanced")
    for file, row, before, _, _ in rows:
        if before is None:
            file.unlink(missing_ok=True)
            _fsync_directory(file.parent)
        else:
            _atomic_write_bytes(file, before, mode=row["before_mode"], overwrite=True)
    value["status"] = "repaired"
    atomic_json(path, value)
    return "repaired"
