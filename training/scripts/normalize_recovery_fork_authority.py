#!/usr/bin/env python3
"""Archive inherited parent control history without inventing child migrations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from scripts.preflight_run_state import state_apply_guard
from scripts.preserve_replay_snapshot import _stopped_boundary
from deltreltrain.checkpoint import verify_file
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import digest, load_plan

RECEIPT = "strength-recovery-fork-authority.json"
PENDING = "strength-recovery-fork-authority.pending.json"
FORMAT = "deltreltrain.recovery-fork-authority-normalization"
ARCHIVES = {
    "continuous-migrations.jsonl": "ablation-parent/inherited-continuous-migrations.jsonl",
    "source-commit.txt": "ablation-parent/inherited-source-commit.txt",
}
PROTECTED = (
    "run.json",
    "profile-elo-ablation.yaml",
    "profile.sha256",
    "ablation.json",
    "strength-recovery-plan.json",
    "strength-recovery-provenance.json",
    "learner/champion.json",
    "learner/candidate.json",
    "learner/recovery.json",
    "learner/resume-cutover.json",
    "learner/cadence.json",
    "learner/utd-segment.json",
    "learner/champion-warm-start.json",
)


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"unsafe or oversized control metadata: {path}")
    data = path.read_bytes()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("control metadata grew beyond its limit")
    return data


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError("control metadata must be a JSON object")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".fork-authority-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
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


def _fingerprints(root: Path) -> dict[str, str]:
    return {
        name: _sha(_read(root / name)) for name in PROTECTED if (root / name).exists()
    }


def validate_receipt(receipt: dict[str, Any]) -> dict[str, bytes]:
    if (
        receipt.get("format") != FORMAT
        or receipt.get("schema_version") != 1
        or receipt.get("sha256")
        != digest({k: v for k, v in receipt.items() if k != "sha256"})
    ):
        raise ValueError("fork authority receipt hash or schema is invalid")
    recovered = {}
    for entry in receipt["archives"]:
        if (
            entry.get("original") not in ARCHIVES
            or entry.get("path") != ARCHIVES[entry["original"]]
        ):
            raise ValueError("fork authority archive path is invalid")
        data = base64.b64decode(entry["payload_base64"], validate=True)
        if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
            raise ValueError("embedded parent history does not match its pin")
        if entry["path"] in recovered:
            raise ValueError("duplicate inherited authority archive")
        recovered[entry["path"]] = data
    return recovered


def plan_normalization(root: Path, source_commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is None:
        raise ValueError("qualified source commit must be an explicit full Git ID")
    plan = load_plan(root)
    if plan is None:
        raise ValueError("normalization requires a registered recovery fork")
    continuation = root / "strength-continuation-plan.json"
    declared = plan.get("prepared_source_commit")
    if continuation.exists():
        future = _json(continuation)
        if (
            future.get("plan_sha256")
            != digest({k: v for k, v in future.items() if k != "plan_sha256"})
            or future.get("recovery_plan_sha256") != plan["plan_sha256"]
        ):
            raise ValueError("continuation source authority is not pinned to this fork")
        if declared is not None and declared != future.get("source_commit"):
            raise ValueError("prepared and continuation source authorities disagree")
        declared = future.get("source_commit")
    if declared != source_commit:
        raise ValueError(
            "requested source commit differs from the already qualified plan"
        )
    for pin in plan["implementation_pins"]:
        verify_file(
            pin["path"], expected_sha256=pin["sha256"], expected_bytes=pin["bytes"]
        )
    verify_file(
        root / "profile-elo-ablation.yaml",
        expected_sha256=plan["profile_sha256"],
        expected_bytes=plan["profile"]["bytes"],
    )
    if (root / RECEIPT).exists():
        receipt = _json(root / RECEIPT)
        embedded = validate_receipt(receipt)
        if (
            receipt.get("recovery_plan_sha256") != plan["plan_sha256"]
            or receipt.get("qualified_source_commit") != source_commit
        ):
            raise ValueError("existing normalization belongs to another fork or source")
        for name, data in embedded.items():
            if _read(root / name) != data:
                raise ValueError("inherited archive differs from its durable receipt")
        if (root / "continuous-migrations.jsonl").exists() or _read(
            root / "source-commit.txt"
        ).decode().strip() != source_commit:
            raise ValueError(
                "child authority has advanced since normalization; do not reapply"
            )
        return {"status": "already_normalized", "receipt": receipt}
    parent = Path(plan["source_run_root"]).resolve()
    if root == parent or root in parent.parents or parent in root.parents:
        raise ValueError("normalization must never operate on the source root")
    metadata = _json(root / "ablation.json")
    if (
        metadata.get("report") != "deltreltrain-elo-ablation-branch"
        or metadata.get("treatment") != "strength-recovery"
        or metadata.get("source_run_root") != str(parent)
        or metadata.get("profile_sha256") != plan["profile_sha256"]
    ):
        raise ValueError("child is not the declared isolated recovery fork")
    prepared = metadata.get("prepared_ns")
    if type(prepared) is not int or prepared <= 0:
        raise ValueError("fork preparation boundary is invalid")
    pending = _json(root / PENDING) if (root / PENDING).exists() else None
    pending_archives = validate_receipt(pending) if pending is not None else {}
    if pending is not None and (
        pending.get("recovery_plan_sha256") != plan["plan_sha256"]
        or pending.get("qualified_source_commit") != source_commit
    ):
        raise ValueError("pending normalization belongs to another plan")
    archives = []
    for original, archive in ARCHIVES.items():
        source = parent / original
        if not source.exists():
            if (root / original).exists() and not (
                original == "source-commit.txt" and pending
            ):
                raise ValueError("child has control history absent from its parent")
            continue
        expected = _read(source)
        current = root / original
        actual = _read(current) if current.exists() else None
        allowed_after_interruption = (
            pending is not None
            and pending_archives.get(archive) == expected
            and (
                actual is None
                if original.endswith(".jsonl")
                else actual == (source_commit + "\n").encode()
            )
        )
        if actual != expected and not allowed_after_interruption:
            raise ValueError(
                "child control history differs from parent; a child migration may exist"
            )
        if original.endswith(".jsonl"):
            rows = [json.loads(line) for line in expected.splitlines()]
            if not rows or any(
                row.get("run_id") != metadata["source_run_id"]
                or row.get("generation_family") != metadata["source_generation_family"]
                or type(row.get("timestamp_ns")) is not int
                or not 0 < row["timestamp_ns"] < prepared
                for row in rows
            ):
                raise ValueError(
                    "journal includes history outside the inherited parent boundary"
                )
            source_profile = plan["source_pins"][0]
            if (
                rows[-1].get("to_profile_sha256") != source_profile["sha256"]
                or rows[-1].get("to_profile") != Path(source_profile["path"]).name
                or rows[-1].get("to_source_commit")
                != _read(parent / "source-commit.txt").decode().strip()
            ):
                raise ValueError(
                    "inherited journal head differs from the pinned parent profile/source"
                )
        archives.append(
            {
                "original": original,
                "path": archive,
                "sha256": _sha(expected),
                "bytes": len(expected),
                "payload_base64": base64.b64encode(expected).decode(),
            }
        )
    receipt = pending or {
        "format": FORMAT,
        "schema_version": 1,
        "recovery_plan_sha256": plan["plan_sha256"],
        "run_root": str(root),
        "source_run_root": str(parent),
        "qualified_source_commit": source_commit,
        "archives": archives,
        "protected_control_fingerprints": _fingerprints(root),
        "created_ns": time.time_ns(),
        "scope": "Only archive unchanged inherited control history and identify the already qualified source. No historical migration is synthesized; profile, model, optimizer, replay and budget clock are unchanged.",
    }
    if pending is not None and pending["archives"] != archives:
        raise ValueError("parent history changed during an interrupted normalization")
    receipt.setdefault("sha256", digest(receipt))
    return {"status": "prepared", "receipt": receipt}


def normalize(root: Path, *, source_commit: str, apply: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    planned = plan_normalization(root, source_commit)
    if not apply or planned["status"] == "already_normalized":
        return planned
    metadata = _json(root / "ablation.json")
    if (root / "coordinator.lock").exists():
        raise ValueError("stop the child coordinator cleanly before normalization")
    if (
        metadata.get("measurement_started_ns") is not None
        or (root / "status/coordinator.json").exists()
    ):
        _stopped_boundary(root)
    receipt = planned["receipt"]
    protected = receipt["protected_control_fingerprints"]
    if _fingerprints(root) != protected:
        raise ValueError(
            "protected child controls changed since normalization planning"
        )
    originals = {
        name: _read(root / name) if (root / name).exists() else None
        for name in ARCHIVES
    }
    embedded = validate_receipt(receipt)
    with state_apply_guard(root):
        if _fingerprints(root) != protected:
            raise ValueError("protected controls changed before authority transaction")
        for name, data in embedded.items():
            path = root / name
            if path.exists():
                if _read(path) != data:
                    raise ValueError("existing inherited archive has different bytes")
            else:
                _write(path, data, replace=False)
        atomic_json(root / PENDING, receipt)
        try:
            (root / "continuous-migrations.jsonl").unlink(missing_ok=True)
            _write(
                root / "source-commit.txt",
                (source_commit + "\n").encode(),
                replace=True,
            )
            if _fingerprints(root) != protected:
                raise ValueError("normalization changed protected child state")
            atomic_json(root / RECEIPT, receipt)
            (root / RECEIPT).chmod(0o444)
            (root / PENDING).unlink()
        except BaseException:
            (root / RECEIPT).unlink(missing_ok=True)
            for name, data in originals.items():
                if data is None:
                    (root / name).unlink(missing_ok=True)
                else:
                    _write(root / name, data, replace=True)
            raise
    return {"status": "normalized", "receipt": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = normalize(
        args.run_root, source_commit=args.source_commit, apply=args.apply
    )
    receipt = result["receipt"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "qualified_source_commit": receipt["qualified_source_commit"],
                "receipt": str(args.run_root.resolve() / RECEIPT),
                "receipt_sha256": receipt["sha256"],
                "archives": [
                    {k: item[k] for k in ("original", "path", "sha256", "bytes")}
                    for item in receipt["archives"]
                ],
                "protected_control_files": len(
                    receipt["protected_control_fingerprints"]
                ),
                "training_state_modified_by_command": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
