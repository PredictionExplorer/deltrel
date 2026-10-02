"""Byte-bound recovery experiment inputs and endpoint backup dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

from .checkpoint import sha256_file, verify_file
from .runtime import atomic_json
from .strength_recovery import FORMAT, PLAN_NAME, digest

PROVENANCE = "strength-recovery-provenance.json"
SNAPSHOTS = "strength-recovery-snapshots"
CONTINUATION_PLAN = "strength-continuation-plan.json"
CONTINUATION_FORMAT = "deltreltrain.strength-continuation"
CONTINUATION_INPUT = "strength-continuation-input.yaml"
INSTALLED_PROFILE = "profile-elo-ablation.yaml"


def _entries(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    groups = {
        "source": plan["source_pins"],
        "implementation": plan["implementation_pins"],
        "profile": [plan["profile"]],
    }
    for group, pins in groups.items():
        for index, pin in enumerate(pins):
            output.append(
                {
                    "source_path": pin["path"],
                    "path": f"strength-recovery-provenance/{group}/{index:03d}-{Path(pin['path']).name}",
                    "sha256": pin["sha256"],
                    "bytes": pin["bytes"],
                }
            )
    return output


def preserve_provenance(root: Path, plan: Mapping[str, Any]) -> None:
    """Copy mutable and immutable source inputs; never share a mutable inode."""
    entries = _entries(plan)
    for entry in entries:
        source, target = Path(entry["source_path"]), root / entry["path"]
        verify_file(
            source, expected_sha256=entry["sha256"], expected_bytes=entry["bytes"]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        verify_file(
            target, expected_sha256=entry["sha256"], expected_bytes=entry["bytes"]
        )
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        target.chmod(0o444)
    payload = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "artifacts": entries,
    }
    payload["sha256"] = digest(payload)
    atomic_json(root / PROVENANCE, payload)
    (root / PROVENANCE).chmod(0o444)


def validate_archive(
    *,
    read: Callable[[str], bytes],
    available: set[str],
    fingerprint: Callable[[str], tuple[str, int]] | None = None,
) -> dict[str, str]:
    """Validate closure with streaming hashes or independently checked catalog pins.

    With a fingerprint callback, ``read`` loads only small JSON metadata.
    Catalog callers apply their declared payload verification policy first.
    Stat identities alone are never acceptable fingerprints for live files.
    """
    if PLAN_NAME not in available:
        if (
            PROVENANCE in available
            or CONTINUATION_PLAN in available
            or any(path.startswith(SNAPSHOTS + "/") for path in available)
        ):
            raise ValueError("recovery artifacts lack their immutable plan")
        return {}
    checksums: dict[str, str] = {}

    def safe(logical: str) -> None:
        pure = PurePosixPath(logical)
        if pure.is_absolute() or ".." in pure.parts or logical not in available:
            raise ValueError(f"missing or unsafe recovery dependency: {logical}")

    def contents(logical: str) -> bytes:
        safe(logical)
        if fingerprint is not None and fingerprint(logical)[1] > 1024 * 1024:
            raise ValueError("recovery JSON metadata exceeds one MiB")
        data = read(logical)
        if len(data) > 1024 * 1024:
            raise ValueError("recovery JSON metadata exceeds one MiB")
        checksums[logical] = hashlib.sha256(data).hexdigest()
        return data

    def document(logical: str) -> dict[str, Any]:
        payload = json.loads(contents(logical))
        if not isinstance(payload, dict):
            raise ValueError("recovery metadata must be an object")
        return payload

    def pinned(entry: Mapping[str, Any]) -> None:
        logical = str(entry["path"])
        safe(logical)
        if fingerprint is None:
            data = read(logical)
            actual, size = hashlib.sha256(data).hexdigest(), len(data)
        else:
            actual, size = fingerprint(logical)
        checksums[logical] = actual
        if actual != entry["sha256"] or size != entry["bytes"]:
            raise ValueError("recovery dependency checksum or size differs")

    plan = document(PLAN_NAME)
    if (
        plan.get("format") != FORMAT
        or plan.get("schema_version") != 1
        or plan.get("plan_sha256")
        != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    ):
        raise ValueError("recovery plan checksum or identity differs")
    provenance = document(PROVENANCE)
    if (
        provenance.get("schema_version") != 1
        or provenance.get("plan_sha256") != plan["plan_sha256"]
        or provenance.get("sha256")
        != digest({k: v for k, v in provenance.items() if k != "sha256"})
        or provenance.get("artifacts") != _entries(plan)
    ):
        raise ValueError("recovery provenance differs from the pinned plan")
    for entry in provenance["artifacts"]:
        pinned(entry)
    source_root = Path(plan["run_root"])
    installed_profile = plan.get("installed_profile_name")
    if installed_profile is None and CONTINUATION_PLAN in available:
        # Older recovery plans predate the explicit field. Their continuation
        # still depends on the original installed profile after migration.
        installed_profile = INSTALLED_PROFILE
    if installed_profile is not None:
        if installed_profile != INSTALLED_PROFILE:
            raise ValueError("recovery installed profile name is invalid")
        pinned({**plan["profile"], "path": installed_profile})
    if CONTINUATION_PLAN in available:
        continuation = document(CONTINUATION_PLAN)
        if (
            continuation.get("format") != CONTINUATION_FORMAT
            or continuation.get("schema_version") != 1
            or continuation.get("recovery_plan_sha256") != plan["plan_sha256"]
            or continuation.get("plan_sha256")
            != digest({k: v for k, v in continuation.items() if k != "plan_sha256"})
        ):
            raise ValueError("continuation plan checksum or recovery identity differs")
        target = continuation.get("target_profile")
        if not isinstance(target, dict) or not isinstance(target.get("path"), str):
            raise ValueError("continuation target profile pin is invalid")
        relative = Path(target["path"]).relative_to(source_root).as_posix()
        if relative != CONTINUATION_INPUT:
            raise ValueError("continuation target escaped its immutable input path")
        pinned({**target, "path": relative})
    for seconds in plan["schedule_seconds"]:
        logical = f"{SNAPSHOTS}/{seconds}/snapshot.json"
        if logical not in available:
            continue
        receipt = document(logical)
        if (
            receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("scheduled_seconds") != seconds
        ):
            raise ValueError("recovery endpoint belongs to another plan or time")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "manifest",
            "checkpoint",
        }:
            raise ValueError("recovery endpoint artifact set is invalid")
        relocated = {}
        for name, pin in artifacts.items():
            relative = Path(pin["path"]).relative_to(source_root).as_posix()
            expected_parent = f"{SNAPSHOTS}/{seconds}/{'manifests' if name == 'manifest' else 'checkpoints'}"
            if str(PurePosixPath(relative).parent) != expected_parent:
                raise ValueError(
                    "recovery endpoint artifact escaped its immutable directory"
                )
            relocated[name] = {**pin, "path": relative}
            pinned(relocated[name])
        manifest = document(relocated["manifest"]["path"])
        checkpoint = relocated["checkpoint"]
        if (
            manifest.get("checkpoint")
            != "../checkpoints/" + PurePosixPath(checkpoint["path"]).name
            or manifest.get("checkpoint_sha256") != checkpoint["sha256"]
            or manifest.get("checkpoint_bytes") != checkpoint["bytes"]
            or manifest.get("model_identity") != receipt.get("model_identity")
            or manifest.get("model_step") != receipt.get("model_step")
        ):
            raise ValueError("recovery endpoint manifest/checkpoint linkage differs")
    return checksums


def backup_files(root: Path) -> dict[str, str]:
    """List committed endpoints and continuation inputs with verified closure."""
    available = {
        name
        for name in (
            PLAN_NAME,
            PROVENANCE,
            CONTINUATION_PLAN,
            CONTINUATION_INPUT,
            INSTALLED_PROFILE,
        )
        if (root / name).exists()
    }
    for name in ("strength-recovery-provenance", SNAPSHOTS):
        folder = root / name
        if folder.is_symlink():
            raise ValueError("recovery backup cannot follow symbolic links")
        if folder.exists():
            for path in folder.rglob("*"):
                if path.is_symlink():
                    raise ValueError("recovery backup cannot follow symbolic links")
                if path.is_file():
                    available.add(path.relative_to(root).as_posix())

    def read(logical: str) -> bytes:
        path = root / logical
        if path.is_symlink() or not path.is_file():
            raise ValueError("recovery backup dependency is unsafe")
        return path.read_bytes()

    def fingerprint(logical: str) -> tuple[str, int]:
        path = root / logical
        if path.is_symlink() or not path.is_file():
            raise ValueError("recovery backup dependency is unsafe")
        return sha256_file(path), path.stat().st_size

    return validate_archive(read=read, available=available, fingerprint=fingerprint)
