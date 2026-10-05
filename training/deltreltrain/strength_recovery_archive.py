"""Byte-bound recovery experiment inputs and endpoint backup dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

from . import strength_freshness as freshness_attempts
from .checkpoint import sha256_file, verify_file
from .runtime import atomic_json
from .strength_recovery import FORMAT, PLAN_NAME, digest

PROVENANCE = "strength-recovery-provenance.json"
SNAPSHOTS = "strength-recovery-snapshots"
CONTINUATION_PLAN = "strength-continuation-plan.json"
CONTINUATION_FORMAT = "deltreltrain.strength-continuation"
CONTINUATION_INPUT = "strength-continuation-input.yaml"
INSTALLED_PROFILE = "profile-elo-ablation.yaml"
FRESHNESS_PLAN = "strength-freshness-plan.json"
FRESHNESS_FORMAT = "deltreltrain.strength-freshness-transition"
FRESHNESS_INPUT = "strength-freshness-input.yaml"
FRESHNESS_INSTALLED = "profile-strength-freshness.yaml"
FRESHNESS_SOURCE = "profile-strength-continuation.yaml"
FRESHNESS_MIGRATIONS = "strength-freshness-source-migrations.jsonl"
FRESHNESS_PROVENANCE = "strength-freshness-provenance"
FRESHNESS_BOUNDARY = "strength-freshness-boundary.json"
FRESHNESS_RECEIPT = "strength-freshness-receipt.json"
FRESHNESS_FILES = {
    FRESHNESS_PLAN,
    FRESHNESS_INPUT,
    FRESHNESS_INSTALLED,
    FRESHNESS_MIGRATIONS,
    FRESHNESS_BOUNDARY,
    FRESHNESS_RECEIPT,
}


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
    freshness_present = bool(available & FRESHNESS_FILES) or any(
        name.startswith(FRESHNESS_PROVENANCE + "/") for name in available
    )
    if freshness_present and FRESHNESS_PLAN not in available:
        raise ValueError("freshness artifacts lack their immutable plan")
    if PLAN_NAME not in available:
        if (
            PROVENANCE in available
            or CONTINUATION_PLAN in available
            or freshness_present
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
    continuation: dict[str, Any] | None = None
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
    if freshness_present:
        if continuation is None:
            raise ValueError("freshness artifacts lack their continuation plan")
        freshness = document(FRESHNESS_PLAN)
        if (
            freshness.get("format") != FRESHNESS_FORMAT
            or type(freshness.get("schema_version")) is not int
            or freshness["schema_version"] != 1
            or freshness.get("plan_sha256")
            != digest({k: v for k, v in freshness.items() if k != "plan_sha256"})
            or freshness.get("run_root") != plan["run_root"]
            or not source_root.is_absolute()
            or source_root.as_posix() != plan["run_root"]
            or ".." in source_root.parts
            or freshness.get("recovery_plan_sha256") != plan["plan_sha256"]
            or freshness.get("continuation_plan_sha256") != continuation["plan_sha256"]
            or freshness.get("target_profile_name") != FRESHNESS_INSTALLED
            or type(freshness.get("activation_after_ns")) is not int
            or type(freshness.get("continuation_started_ns")) is not int
            or type(freshness.get("created_ns")) is not int
            or not 0
            < freshness["continuation_started_ns"]
            <= freshness["activation_after_ns"]
            <= freshness["created_ns"]
        ):
            raise ValueError("freshness plan checksum, root or identity differs")

        def relocated_pin(entry: Mapping[str, Any], *, exact=None, parent=None):
            raw = entry.get("path")
            if not isinstance(raw, str):
                raise ValueError("freshness artifact path is invalid")
            original = Path(raw)
            if (
                not original.is_absolute()
                or original.as_posix() != raw
                or ".." in original.parts
            ):
                raise ValueError("freshness artifact path is not canonical")
            relative = original.relative_to(source_root).as_posix()
            if (exact is not None and relative != exact) or (
                parent is not None
                and not PurePosixPath(relative).is_relative_to(parent)
            ):
                raise ValueError("freshness artifact escaped its registered path")
            if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
                raise ValueError("freshness artifact byte count is invalid")
            pinned({**entry, "path": relative})
            return relative

        for field, filename in (
            ("source_profile", FRESHNESS_SOURCE),
            ("target_profile", FRESHNESS_INPUT),
            ("source_migrations", FRESHNESS_MIGRATIONS),
        ):
            relocated_pin(freshness[field], exact=filename)
        if any(
            freshness["source_profile"][field] != continuation["target_profile"][field]
            for field in ("sha256", "bytes")
        ):
            raise ValueError("freshness source is not the registered continuation")
        if FRESHNESS_INSTALLED in available:
            pinned({**freshness["target_profile"], "path": FRESHNESS_INSTALLED})
        artifacts = freshness.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("freshness provenance artifact set is invalid")
        declared = set()
        for entry in artifacts:
            relative = relocated_pin(entry, parent=FRESHNESS_PROVENANCE)
            if relative in declared:
                raise ValueError("duplicate freshness provenance artifact")
            declared.add(relative)
        # Dependencies referenced by the admission chain must be preserved,
        # rather than silently retaining only their former external filenames.
        for entry in [
            freshness["runtime"]["qualification"],
            freshness["runtime"]["source_checksums"],
            *freshness["original_controls"].values(),
        ]:
            if entry not in artifacts:
                raise ValueError("freshness proof dependency is not preserved")
        attempt_receipts = sorted(
            name
            for name in available
            if name.startswith(freshness_attempts.ATTEMPTS + "/")
            and name.endswith("/" + freshness_attempts.REPREPARE)
        )
        previous_receipt = None
        seen_ids = set()
        for generation, logical in enumerate(attempt_receipts, 1):
            expected = (
                freshness_attempts.attempt_directory(generation)
                + "/"
                + freshness_attempts.REPREPARE
            )
            if logical != expected:
                raise ValueError(
                    "freshness re-preparation generations are not contiguous"
                )
            attempt = document(logical)
            prior = freshness_attempts.boundary_relative_path(generation - 1)
            relocated_pin(attempt["abandoned_boundary"], exact=prior)
            freshness_attempts.validate_repreparation(
                freshness,
                attempt,
                generation=generation,
                previous_receipt_sha256=previous_receipt,
                previous_boundary_sha256=checksums[prior],
            )
            if attempt["attempt_id"] in seen_ids:
                raise ValueError("freshness attempt identity was reused")
            seen_ids.add(attempt["attempt_id"])
            previous_receipt = checksums[logical]
            declared.add(logical)
        active_boundary = freshness_attempts.boundary_relative_path(
            len(attempt_receipts)
        )
        if FRESHNESS_INSTALLED in available and active_boundary not in available:
            raise ValueError("installed freshness lacks its stopped boundary")
        for generation in range(len(attempt_receipts) + 1):
            logical = freshness_attempts.boundary_relative_path(generation)
            if logical not in available:
                continue
            prefix = freshness_attempts.attempt_directory(generation)
            boundary = document(logical)
            if (
                type(boundary.get("schema_version")) is not int
                or boundary["schema_version"] != 1
                or boundary.get("plan_sha256") != freshness["plan_sha256"]
                or boundary.get("continuation_started_ns")
                != freshness["continuation_started_ns"]
                or boundary.get("sha256")
                != digest({k: v for k, v in boundary.items() if k != "sha256"})
            ):
                raise ValueError("freshness boundary checksum or identity differs")
            if generation:
                if (
                    boundary.get("generation") != generation
                    or boundary.get("repreparation_sha256")
                    != checksums[prefix + "/" + freshness_attempts.REPREPARE]
                ):
                    raise ValueError("freshness boundary belongs to another attempt")
                declared.add(logical)
            checkpoint = boundary["checkpoint"]
            relative = relocated_pin(
                checkpoint, parent=FRESHNESS_PROVENANCE + "/checkpoints"
            )
            if str(
                PurePosixPath(relative).parent
            ) != FRESHNESS_PROVENANCE + "/checkpoints" or PurePosixPath(
                relative
            ).name not in {
                checkpoint["sha256"] + ".pt",
                "sha256-" + checkpoint["sha256"] + ".pt",
            }:
                raise ValueError("freshness checkpoint is not content addressed")
            declared.add(relative)
            cuda = boundary.get("cuda_qualification")
            if not isinstance(cuda, dict):
                raise ValueError("freshness stopped boundary lacks CUDA qualification")
            declared.add(
                relocated_pin(
                    cuda,
                    exact=prefix + "/r4-cuda-qualification.json",
                )
            )
            controls = boundary.get("controls")
            if not isinstance(controls, list) or not controls:
                raise ValueError("freshness stopped controls are missing")
            for entry in controls:
                relative = relocated_pin(entry, parent=prefix + "/boundary")
                if relative in declared:
                    raise ValueError("duplicate freshness stopped artifact")
                declared.add(relative)
        if FRESHNESS_RECEIPT in available:
            receipt = document(FRESHNESS_RECEIPT)
            if (
                active_boundary not in checksums
                or receipt.get("schema_version") != 1
                or receipt.get("status") != "committed-recover-forward-only"
                or receipt.get("plan_sha256") != freshness["plan_sha256"]
                or receipt.get("boundary_sha256") != checksums[active_boundary]
                or receipt.get("boundary_path", FRESHNESS_BOUNDARY) != active_boundary
                or receipt.get("continuation_started_ns")
                != freshness["continuation_started_ns"]
                or receipt.get("original_source_commit")
                != freshness["original_source_commit"]
                or receipt.get("active_source_commit")
                != freshness["runtime"]["source_commit"]
                or receipt.get("sha256")
                != digest({k: v for k, v in receipt.items() if k != "sha256"})
            ):
                raise ValueError("freshness commit receipt differs from its boundary")
        if any(
            name.startswith(FRESHNESS_PROVENANCE + "/") and name not in declared
            for name in available
        ):
            raise ValueError("orphan freshness provenance artifact")
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
            FRESHNESS_SOURCE,
            *FRESHNESS_FILES,
        )
        if (root / name).exists()
    }
    for name in ("strength-recovery-provenance", SNAPSHOTS, FRESHNESS_PROVENANCE):
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
