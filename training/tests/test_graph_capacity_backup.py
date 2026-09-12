from dataclasses import replace
import hashlib

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from scripts.validate_continuous_profile import validate_continuous_config
from startrain import search_allocation_gate as gate
from startrain.config import load_config
from test_graph_capacity_admission import graph_admission_fixture
from test_search_allocation_gate import fixture as allocation_fixture, write_json
from test_training_disaster_recovery import _fixture, _snapshot, _snapshot_payload


def prepared(tmp_path, monkeypatch, *, graph_capacity=True):
    run = _fixture(tmp_path)
    factory = graph_admission_fixture if graph_capacity else allocation_fixture
    _, target, root, envelope = factory(
        tmp_path,
        monkeypatch,
        root=run.root,
        run_id="run-disaster-test",
        family="family-disaster-test",
    )
    if graph_capacity:
        # Keep these outside the generic status JSON tree. Only explicit gate
        # dependency capture can preserve them, so this catches omitted graph
        # reference enumeration even when normal status files are backed up.
        directory = root / "measurement-evidence"
        directory.mkdir()
        for reference in envelope["graph_cache_reports"]:
            old_path = root / reference["path"]
            new_path = directory / old_path.name
            old_path.rename(new_path)
            reference["path"] = str(new_path.relative_to(root))
        write_json(gate.allocation_gate_path(target), envelope)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = hashlib.sha256(run.profile.read_bytes()).hexdigest()
    (root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    validate_continuous_config(load_config(run.profile))
    return run, target, envelope


def test_graph_reports_are_captured_by_hash_and_restore_original_root_admission(
    tmp_path, monkeypatch
):
    run, target, envelope = prepared(tmp_path, monkeypatch)
    reports = envelope["graph_cache_reports"]
    assert len(reports) == 2
    original_bytes = {
        reference["path"]: (run.root / reference["path"]).read_bytes()
        for reference in reports
    }
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for reference in reports:
        entry = catalog[reference["path"]]
        assert entry["sha256"] == reference["sha256"]
        assert entry["bytes"] == len(original_bytes[reference["path"]])
        assert entry["kind"] == "status-json"
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    original_root = run.root
    original_root.rename(tmp_path / "lost-source")
    recovery.restore_snapshot(snapshot, original_root)
    for logical, content in original_bytes.items():
        assert (original_root / logical).read_bytes() == content
    assert gate.allocation_gate_path(target).is_file()
    restored = load_config(original_root / run.profile.name)
    assert restored.orchestration.model_refresh.inference.cuda_graph_max_entries == 32
    validate_continuous_config(restored)


@pytest.mark.parametrize("report_index", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_backup_rechecks_both_graph_reports_after_cached_admission(
    tmp_path, monkeypatch, report_index, damage
):
    run, _, envelope = prepared(tmp_path, monkeypatch)
    path = run.root / envelope["graph_cache_reports"][report_index]["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    backup = tmp_path / "backup"
    with pytest.raises(
        recovery.DisasterRecoveryError, match="allocation evidence is invalid"
    ):
        _snapshot(run, backup)
    assert not (backup / "latest.json").exists()


@pytest.mark.parametrize("report_index", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "mismatched_hash"])
def test_snapshot_verifier_requires_each_pinned_graph_report_in_catalog(
    tmp_path, monkeypatch, report_index, damage
):
    run, _, envelope = prepared(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    references = envelope["graph_cache_reports"]
    logical = references[report_index]["path"]
    original = recovery._snapshot_envelope

    def damaged_catalog(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        if damage == "missing":
            del catalog[logical]
        else:
            other = catalog[references[1 - report_index]["path"]]
            catalog[logical] = replace(
                catalog[logical], sha256=other.sha256, bytes=other.bytes
            )
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", damaged_catalog)
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="allocation evidence dependency is missing or mismatched",
    ):
        recovery.verify_snapshot(snapshot)
    destination = tmp_path / "must-not-restore"
    with pytest.raises(recovery.DisasterRecoveryError):
        recovery.restore_snapshot(snapshot, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".must-not-restore.restore-*"))


@pytest.mark.parametrize("report_index", [0, 1])
def test_tampered_graph_report_object_prevents_verification_and_restore(
    tmp_path, monkeypatch, report_index
):
    run, _, envelope = prepared(tmp_path, monkeypatch)
    backup = tmp_path / "backup"
    snapshot = _snapshot(run, backup)
    reference = envelope["graph_cache_reports"][report_index]
    stored = recovery._object_path(backup, reference["sha256"])
    stored.chmod(0o644)
    stored.write_bytes(b"corrupted graph measurement")
    with pytest.raises(recovery.DisasterRecoveryError, match="byte length|SHA-256"):
        recovery.verify_snapshot(snapshot)
    destination = tmp_path / "must-not-restore"
    with pytest.raises(recovery.DisasterRecoveryError, match="byte length|SHA-256"):
        recovery.restore_snapshot(snapshot, destination)
    assert not destination.exists()


def test_legacy_allocation_gate_requires_no_graph_reports(tmp_path, monkeypatch):
    run, target, envelope = prepared(tmp_path, monkeypatch, graph_capacity=False)
    assert "graph_cache_reports" not in envelope
    original_references = recovery._allocation_gate_references(envelope)
    assert len(original_references) == 9
    snapshot = _snapshot(run, tmp_path / "backup")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    run.root.rename(tmp_path / "lost-source")
    recovery.restore_snapshot(snapshot, run.root)
    validate_continuous_config(load_config(run.profile))
    assert gate.allocation_gate_path(target).is_file()


def test_profile_without_allocation_gate_keeps_legacy_backup_path(
    tmp_path, monkeypatch
):
    run = _fixture(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary backup must not enumerate allocation dependencies")

    monkeypatch.setattr(recovery, "_allocation_gate_references", unexpected)
    snapshot = _snapshot(run, tmp_path / "backup")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
