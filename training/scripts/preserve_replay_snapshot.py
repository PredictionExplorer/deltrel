#!/usr/bin/env python3
"""Preserve a stopped run locally without copying immutable training payloads.

The manifest and mutable controls are independent copies. Published replay
shards and content-addressed model artifacts are hardlinked, so subsequent GC
cannot remove their archived contents. Full payload hashing is a separate,
read-only operation that can run after training resumes. This local archive
does not replace off-host disaster recovery.
"""

from __future__ import annotations

import argparse
import ctypes
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from typing import Any


FORMAT = "startrain.preserved-stopped-run"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"(?:sha256-([0-9a-f]{64})\.pt|manifest-([0-9a-f]{64})\.json)\Z")
_CONTROL_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".sha256", ".txt"}


class PreservationError(ValueError):
    """The stopped boundary or archive cannot be established safely."""


def _absolute_directory(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    for part in reversed((path, *path.parents)):
        if not stat.S_ISDIR(part.lstat().st_mode):
            raise PreservationError(
                f"directory is missing, non-directory, or symlink: {part}"
            )
    return path


def _path(root: Path, logical: str) -> Path:
    value = PurePosixPath(logical)
    if (
        not isinstance(logical, str)
        or not logical
        or value.is_absolute()
        or value.as_posix() != logical
        or ".." in value.parts
        or "\\" in logical
    ):
        raise PreservationError(f"unsafe relative artifact path: {logical!r}")
    result = root
    for part in value.parts:
        result /= part
        if result.is_symlink():
            raise PreservationError(f"artifact path contains a symlink: {result}")
    return result


def _regular(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise PreservationError(f"artifact is not a regular file: {path}")
    return value


def _signature(path: Path) -> tuple[int, int, int, int]:
    value = _regular(path)
    # Adding a hardlink changes ctime, but never file content or mtime.
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _wal_signature(path: Path) -> tuple[int, int, int, int] | None:
    # SQLite may create an empty WAL sidecar when opening a WAL-mode database
    # read-only. It carries no committed pages and is not a changed boundary.
    if not path.exists():
        return None
    value = _signature(path)
    return value if value[2] else None


def _hash(path: Path) -> str:
    before = _signature(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if _signature(path) != before:
        raise PreservationError(f"artifact changed while hashing: {path}")
    return digest


def _json(path: Path) -> dict[str, Any]:
    _regular(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PreservationError(f"artifact is not a JSON object: {path}")
    return value


def _fsync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def _alive(pid: object) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _stopped_boundary(root: Path) -> dict[str, Any]:
    run = _json(_path(root, "run.json"))
    coordinator = _json(_path(root, "status/coordinator.json"))
    heartbeat = _json(_path(root, "status/learner.heartbeat.json"))
    recovery = _json(_path(root, "learner/recovery.json"))
    workers = coordinator.get("workers")
    if (
        coordinator.get("state") != "stopped"
        or coordinator.get("failure")
        or not isinstance(workers, dict)
        or not workers
        or _alive(coordinator.get("coordinator_pid"))
        or _alive(heartbeat.get("pid"))
        or any(
            not isinstance(row, dict)
            or row.get("state") != "stopped"
            or row.get("last_exit_code") != 0
            or row.get("failure_reason")
            or row.get("pid") is not None
            for row in workers.values()
        )
        or heartbeat.get("phase") != "stopped"
    ):
        raise PreservationError("coordinator and every worker must be cleanly stopped")
    for name in ("step", "examples_consumed"):
        if (
            type(recovery.get(name)) is not int
            or recovery[name] < 0
            or heartbeat.get(name) != recovery[name]
        ):
            raise PreservationError(
                "learner progress differs from the final recovery checkpoint"
            )
    for name in ("run_id", "generation_family"):
        if (
            not isinstance(run.get(name), str)
            or not run[name]
            or recovery.get(name) != run[name]
        ):
            raise PreservationError("recovery and run identities disagree")
    checksum = recovery.get("checkpoint_sha256")
    if not isinstance(checksum, str) or not _SHA.fullmatch(checksum):
        raise PreservationError("recovery checksum is invalid")
    logical = "learner/" + str(recovery.get("checkpoint", ""))
    if logical != f"learner/recovery/sha256-{checksum}.pt":
        raise PreservationError(
            "recovery checkpoint is not content-addressed under learner/recovery"
        )
    checkpoint = _path(root, logical)
    if _regular(checkpoint).st_size != recovery.get("checkpoint_bytes"):
        raise PreservationError("recovery checkpoint size disagrees with its pointer")
    return {
        "run": run,
        "coordinator": coordinator,
        "heartbeat": heartbeat,
        "recovery": recovery,
    }


def _ledger_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT id,relative_path,checksum_sha256,sample_count,ring,state,variant,segment "
            "FROM shards WHERE state IN ('ready','superseded') ORDER BY id"
        )
    ]
    for row in rows:
        if (
            not isinstance(row["checksum_sha256"], str)
            or not _SHA.fullmatch(row["checksum_sha256"])
            or type(row["sample_count"]) is not int
            or row["sample_count"] <= 0
        ):
            raise PreservationError("manifest contains invalid replay metadata")
        path = PurePosixPath(row["relative_path"])
        if len(path.parts) != 2 or path.parts[0] != "shards" or path.suffix != ".npz":
            raise PreservationError(
                "manifest replay path is outside the immutable shard directory"
            )
    return rows


def _publish_directory(stage: Path, destination: Path) -> None:
    """Atomic directory publication that also refuses an existing empty directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        arguments = (-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(stage), os.fsencode(destination), 4)
    else:
        raise PreservationError(
            "atomic exclusive directory publication requires Linux or macOS"
        )
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    _fsync(destination.parent)


def preserve_stopped_snapshot(run_root: Path, destination: Path) -> dict[str, Any]:
    root = _absolute_directory(Path(run_root))
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    parent = _absolute_directory(destination.parent)
    if (
        destination == root
        or root in destination.parents
        or destination in root.parents
    ):
        raise PreservationError("archive must be outside the source run")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if root.stat().st_dev != parent.stat().st_dev:
        raise PreservationError("zero-copy preservation requires the same filesystem")
    boundary = _stopped_boundary(root)
    checkpoint = boundary["recovery"]
    if (
        _hash(_path(root, "learner/" + checkpoint["checkpoint"]))
        != checkpoint["checkpoint_sha256"]
    ):
        raise PreservationError("final recovery checkpoint checksum failed")
    source_database = _path(root, "replay/manifest.sqlite3")
    database_fence = _signature(source_database)
    wal_path = _path(root, "replay/manifest.sqlite3-wal")
    wal_fence = _wal_signature(wal_path)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    files: dict[str, dict[str, Any]] = {}
    fences: dict[str, tuple[int, int, int, int]] = {}
    directories: set[Path] = {stage}

    def capture(
        logical: str,
        *,
        checksum: str | None = None,
        link: bool = False,
        kind: str = "control",
    ) -> None:
        if logical in files:
            if checksum is not None and files[logical]["sha256"] != checksum:
                raise PreservationError("conflicting artifact checksum")
            return
        source = _path(root, logical)
        before = _signature(source)
        if before[0] != root.stat().st_dev:
            raise PreservationError(f"artifact crosses filesystems: {logical}")
        target = _path(stage, logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        directories.update(
            (target.parent, *[p for p in target.parents if stage in p.parents])
        )
        if link:
            if checksum is None:
                raise PreservationError(
                    "hardlinked payload must have an authoritative checksum"
                )
            os.link(source, target, follow_symlinks=False)
        else:
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fchmod(outgoing.fileno(), 0o444)
                os.fsync(outgoing.fileno())
            actual = _hash(target)
            if checksum is not None and actual != checksum:
                raise PreservationError(
                    f"control checksum disagrees with reference: {logical}"
                )
            checksum = actual
        if _signature(source) != before or (link and _signature(target) != before):
            raise PreservationError(f"artifact changed during preservation: {logical}")
        fences[logical] = before
        files[logical] = {
            "sha256": checksum,
            "bytes": before[2],
            "kind": kind,
            "capture": "hardlink" if link else "copy",
        }

    try:
        database = stage / "replay/manifest.sqlite3"
        database.parent.mkdir()
        directories.add(database.parent)
        with closing(
            sqlite3.connect(f"{source_database.as_uri()}?mode=ro", uri=True)
        ) as source_db:
            with closing(sqlite3.connect(database)) as copied_db:
                source_db.backup(copied_db)
                copied_db.execute("PRAGMA journal_mode=DELETE")
                if copied_db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise PreservationError("copied replay manifest failed quick_check")
                rows = _ledger_rows(copied_db)
                counters = [
                    dict(row) for row in copied_db.execute("SELECT * FROM run_counters")
                ]
        with database.open("rb") as stream:
            os.fsync(stream.fileno())
        files["replay/manifest.sqlite3"] = {
            "sha256": _hash(database),
            "bytes": database.stat().st_size,
            "kind": "replay-manifest",
            "capture": "sqlite-backup",
        }
        for row in rows:
            capture(
                "replay/" + row["relative_path"],
                checksum=row["checksum_sha256"],
                link=True,
                kind="replay-shard",
            )
        for namespace in ("learner", "learner/selfplay"):
            for directory in ("checkpoints", "manifests", "recovery"):
                folder = _path(root, f"{namespace}/{directory}")
                if not folder.exists():
                    continue
                for source in sorted(folder.iterdir()):
                    match = _MODEL.fullmatch(source.name)
                    if match is None:
                        raise PreservationError(
                            f"unexpected file in immutable model directory: {source}"
                        )
                    capture(
                        str(source.relative_to(root)),
                        checksum=next(value for value in match.groups() if value),
                        link=True,
                        kind="model",
                    )
        for folder in (
            root,
            _path(root, "learner"),
            _path(root, "learner/selfplay"),
            _path(root, "arena"),
        ):
            if folder.exists():
                for source in sorted(folder.iterdir()):
                    if (
                        source.suffix in _CONTROL_SUFFIXES
                        and source.name != "metrics.jsonl"
                    ):
                        capture(str(source.relative_to(root)))
        for logical in (
            "status/coordinator.json",
            "status/learner.heartbeat.json",
            "replay/initialized.json",
        ):
            if _path(root, logical).exists():
                capture(logical)
        authority = _path(root, "profile.sha256")
        if authority.exists():
            parts = authority.read_text().strip().split(maxsplit=1)
            if len(parts) != 2 or not _SHA.fullmatch(parts[0]):
                raise PreservationError("active profile checksum authority is invalid")
            profile = Path(parts[1])
            logical = (
                str(profile.relative_to(root)) if profile.is_absolute() else parts[1]
            )
            capture(logical, checksum=parts[0])
            from scripts.training_disaster_recovery import (
                _active_allocation_gate,
                _allocation_gate_dependency_closure,
            )

            gate = _active_allocation_gate(_path(root, logical))
            if gate is not None:
                capture(gate)

                def read_evidence(logical: str, digest: str) -> dict[str, Any]:
                    capture(logical, checksum=digest)
                    return _json(_path(stage, logical))

                for name, digest, _ in _allocation_gate_dependency_closure(
                    _json(_path(stage, gate)), read_evidence
                ):
                    capture(name, checksum=digest)
        if _stopped_boundary(root) != boundary:
            raise PreservationError(
                "stopped learner boundary changed during preservation"
            )
        if (
            _signature(source_database) != database_fence
            or _wal_signature(wal_path) != wal_fence
        ):
            raise PreservationError(
                "source replay manifest changed during preservation"
            )
        if any(
            _signature(_path(root, name)) != signature
            for name, signature in fences.items()
        ):
            raise PreservationError(
                "source controls or payloads changed during preservation"
            )
        inventory = {
            "schema_version": 1,
            "files": files,
            "replay_rows": rows,
            "run_counters": counters,
        }
        _write_json(stage / "inventory.json", inventory)
        result = {
            "format": FORMAT,
            "schema_version": 1,
            "status": "preserved",
            "source_run_root": str(root),
            "destination": str(destination),
            "created_ns": time.time_ns(),
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint["step"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "inventory_sha256": _hash(stage / "inventory.json"),
            "files": len(files),
            "replay_shards": len(rows),
            "payload_checksum_verification": "deferred-to-offline-verify",
        }
        _write_json(stage / "complete.json", result)
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            _fsync(directory)
        _publish_directory(stage, destination)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_snapshot(destination: Path) -> dict[str, Any]:
    root = _absolute_directory(Path(destination))
    marker = _json(root / "complete.json")
    if (
        marker.get("format") != FORMAT
        or marker.get("schema_version") != 1
        or marker.get("status") != "preserved"
    ):
        raise PreservationError("snapshot completion marker is invalid")
    if _hash(root / "inventory.json") != marker.get("inventory_sha256"):
        raise PreservationError("snapshot inventory checksum failed")
    inventory = _json(root / "inventory.json")
    files = inventory["files"]
    if not isinstance(files, dict) or len(files) != marker.get("files"):
        raise PreservationError("snapshot inventory file count is invalid")
    for logical, entry in files.items():
        path = _path(root, logical)
        if _regular(path).st_size != entry["bytes"] or _hash(path) != entry["sha256"]:
            raise PreservationError(f"archived artifact checksum failed: {logical}")
    database = _path(root, "replay/manifest.sqlite3")
    with closing(
        sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)
    ) as db:
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise PreservationError("archived replay manifest failed quick_check")
        rows = _ledger_rows(db)
        if [dict(row) for row in db.execute("SELECT * FROM run_counters")] != inventory[
            "run_counters"
        ]:
            raise PreservationError(
                "archived replay counters disagree with the inventory"
            )
        if rows != inventory["replay_rows"] or len(rows) != marker["replay_shards"]:
            raise PreservationError(
                "archive replay inventory disagrees with its manifest"
            )
        for row in rows:
            entry = files.get("replay/" + row["relative_path"])
            if (
                entry is None
                or entry["kind"] != "replay-shard"
                or entry["sha256"] != row["checksum_sha256"]
            ):
                raise PreservationError(
                    "archived replay manifest has an unpreserved shard"
                )
    if _json(_path(root, "learner/recovery.json")) != marker["checkpoint"]:
        raise PreservationError(
            "archived checkpoint pointer differs from the stopped boundary"
        )
    return {
        "status": "verified",
        "destination": str(root),
        "files": len(files),
        "replay_shards": len(rows),
        "checkpoint_step": marker["checkpoint_step"],
        "checkpoint_sha256": marker["checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preserve = commands.add_parser("preserve")
    preserve.add_argument("--run-root", type=Path, required=True)
    preserve.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = (
        preserve_stopped_snapshot(args.run_root, args.destination)
        if args.command == "preserve"
        else verify_snapshot(args.destination)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
