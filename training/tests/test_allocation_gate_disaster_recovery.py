import hashlib

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from scripts.validate_continuous_profile import validate_continuous_config
from startrain.config import load_config
from startrain import search_allocation_gate as gate
from test_search_allocation_gate import fixture as gate_fixture, install_real_reports
from test_training_disaster_recovery import _fixture, _snapshot, _snapshot_payload


def prepared(tmp_path, monkeypatch):
    run = _fixture(tmp_path)
    _, target, root, envelope = gate_fixture(
        tmp_path,
        monkeypatch,
        root=run.root,
        run_id="run-disaster-test",
        family="family-disaster-test",
        stub_analysis=False,
    )
    install_real_reports(root, target, envelope)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = hashlib.sha256(run.profile.read_bytes()).hexdigest()
    (root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    validate_continuous_config(load_config(run.profile))
    return run, target, envelope


def test_snapshot_preserves_all_gate_dependencies_and_original_root_admission(
    tmp_path, monkeypatch
):
    run, target, envelope = prepared(tmp_path, monkeypatch)
    baseline = run.root / envelope["baseline_profile"]["path"]
    baseline_bytes = baseline.read_bytes()
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    assert (
        catalog[str(baseline.relative_to(run.root))]["sha256"]
        == envelope["baseline_profile"]["sha256"]
    )
    assert catalog[str(baseline.relative_to(run.root))]["kind"] == "profile"
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    original_root = run.root
    original_root.rename(tmp_path / "lost-source")
    recovery.restore_snapshot(snapshot, original_root)
    assert baseline.read_bytes() == baseline_bytes
    assert gate.allocation_gate_path(target).is_file()
    validate_continuous_config(load_config(original_root / run.profile.name))


def test_gate_profile_relocation_fails_before_creating_destination(
    tmp_path, monkeypatch
):
    run, _, _ = prepared(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    destination = tmp_path / "relocated"
    with pytest.raises(
        recovery.DisasterRecoveryError, match="cannot relocate.*search-allocation gate"
    ):
        recovery.restore_snapshot(snapshot, destination, relocate_profile=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".relocated.restore-*"))


def test_snapshot_verifier_rejects_a_missing_gate_baseline_dependency(
    tmp_path, monkeypatch
):
    run, _, envelope = prepared(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    original = recovery._snapshot_envelope

    def missing(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        del catalog[envelope["baseline_profile"]["path"]]
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", missing)
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="allocation evidence dependency is missing",
    ):
        recovery.verify_snapshot(snapshot)


def test_snapshot_refuses_tampered_baseline_even_after_cached_admission(
    tmp_path, monkeypatch
):
    run, _, envelope = prepared(tmp_path, monkeypatch)
    baseline = run.root / envelope["baseline_profile"]["path"]
    baseline.write_text(baseline.read_text() + "\n")
    with pytest.raises(
        recovery.DisasterRecoveryError, match="allocation evidence is invalid"
    ):
        _snapshot(run, tmp_path / "backup")
