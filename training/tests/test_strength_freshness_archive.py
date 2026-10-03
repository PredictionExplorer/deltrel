"""Exact freshness proof and stopped checkpoint survive backup relocation."""

import json
import os
from pathlib import Path
import shutil

import pytest

from deltreltrain.checkpoint import sha256_file
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import FORMAT, PLAN_NAME, digest
from deltreltrain.strength_recovery_archive import (
    CONTINUATION_FORMAT,
    CONTINUATION_INPUT,
    CONTINUATION_PLAN,
    FRESHNESS_BOUNDARY,
    FRESHNESS_FILES,
    FRESHNESS_FORMAT,
    FRESHNESS_INPUT,
    FRESHNESS_INSTALLED,
    FRESHNESS_MIGRATIONS,
    FRESHNESS_PLAN,
    FRESHNESS_PROVENANCE,
    FRESHNESS_RECEIPT,
    FRESHNESS_SOURCE,
    INSTALLED_PROFILE,
    backup_files,
    preserve_provenance,
    validate_archive,
)


def pin(path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def sealed(path, body):
    value = {**body, "plan_sha256": digest(body)}
    atomic_json(path, value)
    return value


def make_run(root):
    root.mkdir()
    source = root.parent / "source.yaml"
    source.write_text("kind: initial-recovery\n")
    binary = root.parent / "checkpoint.pt"
    binary.write_bytes(b"binary" * 200000)
    recovery = sealed(
        root / PLAN_NAME,
        {
            "format": FORMAT,
            "schema_version": 1,
            "run_root": str(root),
            "source_pins": [pin(binary)],
            "implementation_pins": [],
            "profile": pin(source),
            "installed_profile_name": INSTALLED_PROFILE,
            "schedule_seconds": [7200, 21600, 43200],
        },
    )
    preserve_provenance(root, recovery)
    shutil.copyfile(source, root / INSTALLED_PROFILE)
    target = root / CONTINUATION_INPUT
    target.write_text("kind: continuous\n")
    shutil.copyfile(target, root / FRESHNESS_SOURCE)
    sealed(
        root / CONTINUATION_PLAN,
        {
            "format": CONTINUATION_FORMAT,
            "schema_version": 1,
            "recovery_plan_sha256": recovery["plan_sha256"],
            "target_profile": pin(target),
            "target_profile_name": FRESHNESS_SOURCE,
            "source_commit": "prior",
        },
    )
    return recovery


def make_freshness(root: Path, *, applied: bool = False):
    recovery = make_run(root)
    continuation = json.loads((root / CONTINUATION_PLAN).read_text())
    (root / FRESHNESS_INPUT).write_text("kind: exact-freshness\n")
    (root / FRESHNESS_MIGRATIONS).write_text('{"old":"migration"}\n')
    provenance = root / FRESHNESS_PROVENANCE
    provenance.mkdir()
    qualification = provenance / "qualification.json"
    qualification.write_text('{"qualification":"pinned"}\n')
    sums = provenance / "SOURCE_SHA256SUMS"
    sums.write_text("pinned source metadata\n")
    original = provenance / "original-recovery-plan.json"
    shutil.copyfile(root / "strength-recovery-plan.json", original)
    plan = sealed(
        root / FRESHNESS_PLAN,
        {
            "format": FRESHNESS_FORMAT,
            "schema_version": 1,
            "run_root": str(root),
            "created_ns": 30,
            "activation_after_ns": 20,
            "continuation_started_ns": 10,
            "recovery_plan_sha256": recovery["plan_sha256"],
            "continuation_plan_sha256": continuation["plan_sha256"],
            "target_profile_name": FRESHNESS_INSTALLED,
            "source_profile": pin(root / FRESHNESS_SOURCE),
            "target_profile": pin(root / FRESHNESS_INPUT),
            "source_migrations": pin(root / FRESHNESS_MIGRATIONS),
            "artifacts": [pin(qualification), pin(sums), pin(original)],
            "runtime": {
                "qualification": pin(qualification),
                "source_checksums": pin(sums),
                "source_commit": "qualified",
            },
            "original_controls": {"recovery_plan": pin(original)},
            "original_source_commit": "prior",
        },
    )
    if applied:
        shutil.copyfile(root / FRESHNESS_INPUT, root / FRESHNESS_INSTALLED)
        checkpoint = root.parent / "checkpoint.pt"
        checksum = sha256_file(checkpoint)
        retained = provenance / "checkpoints" / f"sha256-{checksum}.pt"
        retained.parent.mkdir()
        os.link(checkpoint, retained)
        control = provenance / "boundary/learner/recovery.json"
        control.parent.mkdir(parents=True)
        control.write_text('{"step":572377}\n')
        cuda = provenance / "r4-cuda-qualification.json"
        cuda.write_text('{"qualification":"explicit-pinned-CUDA"}\n')
        boundary = {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "continuation_started_ns": 10,
            "checkpoint": pin(retained),
            "cuda_qualification": pin(cuda),
            "controls": [pin(control)],
        }
        atomic_json(root / FRESHNESS_BOUNDARY, {**boundary, "sha256": digest(boundary)})
        receipt = {
            "schema_version": 1,
            "status": "committed-recover-forward-only",
            "plan_sha256": plan["plan_sha256"],
            "boundary_sha256": sha256_file(root / FRESHNESS_BOUNDARY),
            "continuation_started_ns": 10,
            "original_source_commit": "prior",
            "active_source_commit": "qualified",
        }
        atomic_json(root / FRESHNESS_RECEIPT, {**receipt, "sha256": digest(receipt)})
    return plan


@pytest.mark.parametrize("applied", [False, True])
def test_closure_relocates_repeatedly_and_streams_payloads(
    tmp_path, applied, monkeypatch
):
    root = tmp_path / "run"
    make_freshness(root, applied=applied)
    expected = backup_files(root)
    assert {
        FRESHNESS_PLAN,
        FRESHNESS_INPUT,
        FRESHNESS_SOURCE,
        FRESHNESS_MIGRATIONS,
    } <= expected.keys()
    if applied:
        assert {
            FRESHNESS_BOUNDARY,
            FRESHNESS_RECEIPT,
            FRESHNESS_INSTALLED,
        } <= expected.keys()
        retained = next((root / FRESHNESS_PROVENANCE / "checkpoints").iterdir())
        assert retained.stat().st_ino == (root.parent / "checkpoint.pt").stat().st_ino
    original = Path.read_bytes

    def guarded(path):
        if path.suffix == ".pt":
            pytest.fail("binary checkpoint must use streaming fingerprint")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    for number in range(2):
        restored = tmp_path / f"restore-{number}"
        shutil.copytree(root, restored)
        assert backup_files(restored) == expected

        def metadata(logical):
            assert logical.endswith(".json")
            return (restored / logical).read_bytes()

        assert (
            validate_archive(
                read=metadata,
                available=set(expected),
                fingerprint=lambda logical: (
                    sha256_file(restored / logical),
                    (restored / logical).stat().st_size,
                ),
            )
            == expected
        )


@pytest.mark.parametrize("name", sorted(FRESHNESS_FILES))
def test_orphan_freshness_control_rejected(tmp_path, name):
    (tmp_path / name).write_text("{}\n")
    with pytest.raises(ValueError, match="lack"):
        backup_files(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        FRESHNESS_INPUT,
        FRESHNESS_SOURCE,
        FRESHNESS_MIGRATIONS,
        FRESHNESS_BOUNDARY,
        FRESHNESS_RECEIPT,
    ],
)
@pytest.mark.parametrize("mutation", ["tamper", "symlink"])
def test_tampered_control_or_symlink_fails(tmp_path, name, mutation):
    root = tmp_path / "run"
    make_freshness(root, applied=True)
    target = root / name
    target.unlink()
    if mutation == "tamper":
        target.write_text("{}\n")
    else:
        target.symlink_to(root / FRESHNESS_PLAN)
    with pytest.raises(ValueError):
        backup_files(root)


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "recovery",
        "continuation",
        "escape",
        "qualification",
        "activation",
        "installed-name",
    ],
)
def test_resealed_wrong_plan_is_not_admitted(tmp_path, change):
    root = tmp_path / "run"
    plan = make_freshness(root)
    plan.pop("plan_sha256")
    if change == "root":
        plan["run_root"] = str(tmp_path / "other")
    elif change in {"recovery", "continuation"}:
        plan[change + "_plan_sha256"] = "wrong"
    elif change == "escape":
        plan["target_profile"]["path"] = str(root / "../outside.yaml")
    elif change == "qualification":
        plan["artifacts"] = plan["artifacts"][1:]
    elif change == "activation":
        plan["activation_after_ns"] = True
    else:
        plan["target_profile_name"] = "profile.yaml"
    sealed(root / FRESHNESS_PLAN, plan)
    with pytest.raises(ValueError):
        backup_files(root)


def test_retained_checkpoint_tamper_and_extra_provenance_fail_closed(tmp_path):
    root = tmp_path / "run"
    make_freshness(root, applied=True)
    expected = backup_files(root)
    checkpoint = next((root / FRESHNESS_PROVENANCE / "checkpoints").iterdir())
    with checkpoint.open("r+b") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        backup_files(root)
    assert expected


def test_orphan_provenance_and_symlink_directory_fail_closed(tmp_path):
    root = tmp_path / "run"
    make_freshness(root)
    extra = root / FRESHNESS_PROVENANCE / "not-in-plan.json"
    extra.write_text("{}\n")
    with pytest.raises(ValueError, match="orphan"):
        backup_files(root)
    extra.unlink()
    (root / FRESHNESS_PROVENANCE / "escape").symlink_to(
        tmp_path, target_is_directory=True
    )
    with pytest.raises(ValueError, match="symbolic"):
        backup_files(root)


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "escape"])
def test_boundary_requires_exact_cuda_qualification(tmp_path, mutation):
    root = tmp_path / "run"
    make_freshness(root, applied=True)
    boundary_path = root / FRESHNESS_BOUNDARY
    body = json.loads(boundary_path.read_text())
    body.pop("sha256")
    if mutation == "missing":
        body.pop("cuda_qualification")
    elif mutation == "corrupt":
        Path(body["cuda_qualification"]["path"]).write_text('{"other":true}\n')
    else:
        body["cuda_qualification"]["path"] = str(root / "qualification.json")
    atomic_json(boundary_path, {**body, "sha256": digest(body)})
    with pytest.raises(ValueError):
        backup_files(root)


def test_installed_target_requires_stopped_checkpoint_and_cuda_boundary(tmp_path):
    root = tmp_path / "run"
    make_freshness(root)
    shutil.copyfile(root / FRESHNESS_INPUT, root / FRESHNESS_INSTALLED)
    with pytest.raises(ValueError, match="lacks its stopped boundary"):
        backup_files(root)
