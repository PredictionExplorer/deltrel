import base64
import hashlib
import json
import shutil
import time

import pytest

from test_strength_recovery import source as source
from scripts import normalize_recovery_fork_authority as repair
from scripts import prepare_strength_recovery as preparation
from scripts import run_strength_recovery_continuation as controller
from scripts.training_disaster_recovery import _SnapshotBuilder
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import PLAN_NAME


@pytest.fixture
def fork(source, tmp_path):
    root, output = tmp_path / "fork", tmp_path / "plan"
    preparation.prepare(
        source_profile=source.profile,
        destination=root,
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    preparation.apply(output / PLAN_NAME)
    controller.prepare(root, source_commit="b" * 40)
    metadata = json.loads((root / "ablation.json").read_text())
    row = {
        "schema_version": 1,
        "run_id": source.config.orchestration.run_id,
        "generation_family": "family",
        "timestamp_ns": metadata["prepared_ns"] - 1,
        "to_profile": source.profile.name,
        "to_profile_sha256": hashlib.sha256(source.profile.read_bytes()).hexdigest(),
        "to_source_commit": "a" * 40,
    }
    for destination in (source.root, root):
        (destination / "continuous-migrations.jsonl").write_text(json.dumps(row) + "\n")
        (destination / "source-commit.txt").write_text("a" * 40 + "\n")
    return root, source.root


def test_normalization_archives_parent_bytes_without_touching_training_state(fork):
    root, parent = fork
    before = repair._fingerprints(root)
    parent_bytes = {name: (parent / name).read_bytes() for name in repair.ARCHIVES}
    report = repair.normalize(root, source_commit="b" * 40, apply=True)
    assert report["status"] == "normalized"
    assert repair._fingerprints(root) == before
    assert not (root / "continuous-migrations.jsonl").exists()
    assert (root / "source-commit.txt").read_text().strip() == "b" * 40
    embedded = repair.validate_receipt(report["receipt"])
    for original, archive in repair.ARCHIVES.items():
        assert (
            (root / archive).read_bytes() == parent_bytes[original] == embedded[archive]
        )
        assert (parent / original).read_bytes() == parent_bytes[original]
    assert (
        repair.normalize(root, source_commit="b" * 40, apply=True)["status"]
        == "already_normalized"
    )


def test_child_migration_or_unqualified_source_cannot_be_rewritten(fork):
    root, _ = fork
    with pytest.raises(ValueError, match="qualified plan"):
        repair.normalize(root, source_commit="c" * 40, apply=True)
    journal = root / "continuous-migrations.jsonl"
    journal.write_text(journal.read_text() + '{"child":"new migration"}\n')
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="child migration"):
        repair.normalize(root, source_commit="b" * 40, apply=True)
    assert journal.read_bytes() == before
    assert not (root / repair.RECEIPT).exists()


def test_active_coordinator_blocks_apply_but_allows_read_only_plan(fork):
    root, _ = fork
    assert repair.normalize(root, source_commit="b" * 40)["status"] == "prepared"
    atomic_json(root / "coordinator.lock", {"pid": 123, "created_ns": 1})
    with pytest.raises(ValueError, match="stop the child"):
        repair.normalize(root, source_commit="b" * 40, apply=True)


def test_started_fork_requires_matching_clean_stop_boundary(fork):
    root, _ = fork
    metadata = json.loads((root / "ablation.json").read_text())
    metadata["measurement_started_ns"] = time.time_ns()
    atomic_json(root / "ablation.json", metadata)
    with pytest.raises((ValueError, OSError)):
        repair.normalize(root, source_commit="b" * 40, apply=True)
    recovery = json.loads((root / "learner/recovery.json").read_text())
    atomic_json(
        root / "status/coordinator.json",
        {
            "state": "stopped",
            "failure": None,
            "coordinator_pid": None,
            "workers": {
                "learner": {"state": "stopped", "last_exit_code": 0, "pid": None}
            },
        },
    )
    atomic_json(
        root / "status/learner.heartbeat.json",
        {
            "phase": "stopped",
            "pid": None,
            "step": recovery["step"],
            "examples_consumed": recovery["examples_consumed"],
        },
    )
    assert (
        repair.normalize(root, source_commit="b" * 40, apply=True)["status"]
        == "normalized"
    )


def test_failed_authority_write_rolls_back_and_retry_is_safe(fork, monkeypatch):
    root, _ = fork
    before = {name: (root / name).read_bytes() for name in repair.ARCHIVES}
    original = repair._write
    failed = False

    def write(path, data, *, replace):
        nonlocal failed
        if path == root / "source-commit.txt" and data.startswith(b"b") and not failed:
            failed = True
            raise OSError("injected authority write failure")
        return original(path, data, replace=replace)

    monkeypatch.setattr(repair, "_write", write)
    with pytest.raises(OSError, match="injected"):
        repair.normalize(root, source_commit="b" * 40, apply=True)
    assert before == {name: (root / name).read_bytes() for name in repair.ARCHIVES}
    assert not (root / "coordinator.lock").exists()
    assert not (root / repair.RECEIPT).exists()
    assert (
        repair.normalize(root, source_commit="b" * 40, apply=True)["status"]
        == "normalized"
    )


def test_existing_r3_backup_catalog_recovers_exact_archives_from_receipt(
    fork, tmp_path
):
    root, _ = fork
    receipt = repair.normalize(root, source_commit="b" * 40, apply=True)["receipt"]
    backup = tmp_path / "backup"
    backup.mkdir()
    builder = _SnapshotBuilder(root, backup)
    logical, entry = builder.add_run_file(root / repair.RECEIPT, "run-metadata")
    assert logical == repair.RECEIPT
    from scripts.training_disaster_recovery import _object_path

    archived_receipt = json.loads(_object_path(backup, entry.sha256).read_text())
    shutil.rmtree(root / "ablation-parent")
    # No new backup schema is necessary: the root JSON catalog already binds
    # every exact historical byte, independently of the original parent.
    assert repair.validate_receipt(archived_receipt) == repair.validate_receipt(receipt)
    corrupted = json.loads(json.dumps(archived_receipt))
    corrupted["archives"][0]["payload_base64"] = base64.b64encode(b"tampered").decode()
    with pytest.raises(ValueError):
        repair.validate_receipt(corrupted)


def test_new_preparation_requires_and_installs_qualified_source_authority(
    source, tmp_path
):
    source_commit = "a" * 40
    (source.root / "source-commit.txt").write_text(source_commit + "\n")
    row = {
        "schema_version": 1,
        "run_id": source.config.orchestration.run_id,
        "generation_family": "family",
        "timestamp_ns": time.time_ns() - 1,
        "to_profile": source.profile.name,
        "to_profile_sha256": hashlib.sha256(source.profile.read_bytes()).hexdigest(),
        "to_source_commit": source_commit,
    }
    journal = json.dumps(row) + "\n"
    (source.root / "continuous-migrations.jsonl").write_text(journal)
    root, output = tmp_path / "next-fork", tmp_path / "next-plan"
    arguments = dict(
        source_profile=source.profile,
        destination=root,
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    with pytest.raises(ValueError, match="explicit qualified source"):
        preparation.prepare(**arguments)
    assert not output.exists() and not root.exists()
    preparation.prepare(**arguments, source_commit="b" * 40)
    preparation.apply(output / PLAN_NAME)
    assert (root / "source-commit.txt").read_text().strip() == "b" * 40
    assert not (root / "continuous-migrations.jsonl").exists()
    assert (
        root / repair.ARCHIVES["continuous-migrations.jsonl"]
    ).read_text() == journal
    assert (source.root / "continuous-migrations.jsonl").read_text() == journal
    assert (source.root / "source-commit.txt").read_text().strip() == source_commit
    receipt = repair._json(root / repair.RECEIPT)
    assert receipt["qualified_source_commit"] == "b" * 40
