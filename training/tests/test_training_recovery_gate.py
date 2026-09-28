"""Recovery trials inherit measured search admission and its backup closure."""

from dataclasses import replace
import hashlib
import json

import pytest
import yaml

from startrain import search_allocation_gate as gate
from startrain.training_recovery_policy import (
    protected_champion_config,
    reuse_config,
)
from scripts.prepare_efficiency_scheduling_gate import prepare_scheduling_gate
from scripts.prepare_training_recovery_profile import prepare_recovery_gate
from scripts import training_disaster_recovery as recovery
from test_allocation_gate_disaster_recovery import prepared
from test_efficiency_scheduling_gate import scheduling_fixture
from test_search_allocation_gate import reference, write_json
from test_training_disaster_recovery import _snapshot, _snapshot_payload


def append_transition(source, source_path, target, path):
    path.write_text(yaml.safe_dump(target.as_dict()))
    original = gate.allocation_gate_path(source).read_bytes()
    result = prepare_recovery_gate(source_path, path)
    assert gate.allocation_gate_path(source).read_bytes() == original
    assert result["new_objective_performance_qualified"] is False
    assert result["measurement_scope"] == gate.RECOVERY_TRANSITION_SCOPE
    return result


def chain(tmp_path, monkeypatch):
    _, source, root, original, previous_path, source_path = scheduling_fixture(
        tmp_path, monkeypatch
    )
    prepare_scheduling_gate(previous_path, source_path)
    protected = protected_champion_config(source, after_ns=100)
    protection_path = root / "protected.yaml"
    first = append_transition(source, source_path, protected, protection_path)
    target = reuse_config(protected)
    target_path = root / "reuse.yaml"
    second = append_transition(protected, protection_path, target, target_path)
    return root, source, protected, target, original, first, second


def test_two_recovery_treatments_preserve_every_prior_admission(tmp_path, monkeypatch):
    root, source, protected, target, _, first, second = chain(tmp_path, monkeypatch)
    assert first["treatment"] == "protected_champion"
    assert second["treatment"] == "reuse_clock"
    envelope = json.loads(gate.allocation_gate_path(target).read_text())
    assert {
        "training_policy_transition",
        "promotion_allocation_transition",
        "auxiliary_prediction_transition",
        "efficiency_scheduling_transition",
    } <= envelope.keys()
    assert second["source_gate"] == reference(
        root, gate.allocation_gate_path(protected)
    )
    assert envelope["training_recovery_transition"] == second["receipt"]
    gate.validate_production_ring_allocations(target, _fresh=True)
    assert set(gate._VERIFIED_GATES[str(gate.allocation_gate_path(protected))]) <= set(
        gate._VERIFIED_GATES[str(gate.allocation_gate_path(target))]
    )


@pytest.mark.parametrize(
    "mutation", ["claim", "treatment", "scope", "unknown", "alias", "dropped"]
)
def test_recovery_receipt_rejects_forgery(tmp_path, monkeypatch, mutation):
    root, _, _, target, _, _, second = chain(tmp_path, monkeypatch)
    path = gate.allocation_gate_path(target)
    envelope = json.loads(path.read_text())
    receipt_path = root / second["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    if mutation == "claim":
        receipt["new_objective_performance_qualified"] = True
    elif mutation == "treatment":
        receipt["treatment"] = "protected_champion"
    elif mutation == "scope":
        receipt["measurement_scope"] = "current-training-workload"
    elif mutation == "unknown":
        receipt["approved"] = True
    elif mutation == "alias":
        alias = root / "same-byte-gate.json"
        alias.write_bytes((root / receipt["source_gate"]["path"]).read_bytes())
        receipt["source_gate"] = reference(root, alias)
    else:
        del envelope["efficiency_scheduling_transition"]
    receipt_path.chmod(0o644)
    write_json(receipt_path, receipt)
    envelope["training_recovery_transition"] = reference(root, receipt_path)
    path.chmod(0o644)
    write_json(path, envelope)
    with pytest.raises(ValueError, match="training recovery transition"):
        gate.validate_production_ring_allocations(target, _fresh=True)


def test_live_recovery_chain_depth_is_bounded(tmp_path, monkeypatch):
    _, _, _, target, _, _, _ = chain(tmp_path, monkeypatch)
    monkeypatch.setattr(gate, "MAX_RECOVERY_TRANSITIONS", 1)
    with pytest.raises(ValueError, match="bounded depth"):
        gate.validate_production_ring_allocations(target, _fresh=True)


def test_inherited_search_reports_are_rehashed_for_recovery(tmp_path, monkeypatch):
    root, _, _, target, original, _, _ = chain(tmp_path, monkeypatch)
    report = root / original["groups"][0]["report"]["path"]
    signature = gate._signature
    fixed = signature(report)
    monkeypatch.setattr(
        gate, "_signature", lambda path: fixed if path == report else signature(path)
    )
    data = report.read_bytes()
    report.write_bytes(b" " + data[1:])
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_production_ring_allocations(target, _fresh=True)


def recovery_snapshot_fixture(tmp_path, monkeypatch):
    run, source, _ = prepared(tmp_path, monkeypatch)
    source_path = run.root / "frozen-source.yaml"
    source_path.write_text(yaml.safe_dump(source.as_dict()))
    protected = protected_champion_config(source, after_ns=100)
    protected_path = run.root / "protected.yaml"
    first = append_transition(source, source_path, protected, protected_path)
    target = reuse_config(protected)
    target_path = run.root / "reuse.yaml"
    second = append_transition(protected, protected_path, target, target_path)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = hashlib.sha256(run.profile.read_bytes()).hexdigest()
    (run.root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    return (
        run,
        target,
        [
            receipt[key]
            for receipt in (first, second)
            for key in ("receipt", "source_gate", "source_profile")
        ],
    )


def test_backup_restores_both_recovery_wrappers_and_original_measurements(
    tmp_path, monkeypatch
):
    from startrain.plateau_evidence import PlateauVerdict, record_verdict

    run, target, refs = recovery_snapshot_fixture(tmp_path, monkeypatch)
    verdict = PlateauVerdict("candidate", 20, "champion", 10, "contract", "reject", 100)
    record_verdict(run.root / "arena", verdict)
    verdict_path = f"arena/plateau-verdicts/{verdict.identity}.json"
    verdict_bytes = (run.root / verdict_path).read_bytes()
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    assert catalog[verdict_path]["sha256"] == hashlib.sha256(verdict_bytes).hexdigest()
    for ref in refs:
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
    original = run.root
    original.rename(tmp_path / "lost-source")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    recovery.restore_snapshot(snapshot, original)
    assert (original / verdict_path).read_bytes() == verdict_bytes
    gate.validate_production_ring_allocations(target, _fresh=True)


@pytest.mark.parametrize("missing_index", range(6))
def test_offline_recovery_requires_every_receipt_profile_and_gate(
    tmp_path, monkeypatch, missing_index
):
    run, _, refs = recovery_snapshot_fixture(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    original = recovery._snapshot_envelope

    def missing(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        catalog = dict(catalog)
        del catalog[refs[missing_index]["path"]]
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", missing)
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="allocation evidence dependency is missing",
    ):
        recovery.verify_snapshot(snapshot)


def test_offline_closure_rejects_depth_scope_and_evidence_changes(
    tmp_path, monkeypatch
):
    root, _, _, target, _, _, _ = chain(tmp_path, monkeypatch)
    envelope = json.loads(gate.allocation_gate_path(target).read_text())

    def reader(path, digest):
        data = (root / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        return json.loads(data)

    dependencies = recovery._allocation_gate_dependency_closure(envelope, reader)
    assert len(dependencies) > 6
    modified = dict(envelope, groups=[])
    with pytest.raises(
        recovery.DisasterRecoveryError, match="changed its source evidence"
    ):
        recovery._allocation_gate_dependency_closure(modified, reader)
    with pytest.raises(recovery.DisasterRecoveryError, match="bounded chain"):
        recovery._allocation_gate_dependency_closure(
            envelope, reader, allow_recovery_transition=False
        )
    monkeypatch.setattr(gate, "MAX_RECOVERY_TRANSITIONS", 1)
    with pytest.raises(recovery.DisasterRecoveryError, match="bounded chain"):
        recovery._allocation_gate_dependency_closure(envelope, reader)


def test_changed_search_settings_cannot_borrow_recovery_authority(
    tmp_path, monkeypatch
):
    root, _, source, _, _, _, _ = chain(tmp_path, monkeypatch)
    changed = replace(
        source,
        selfplay=replace(
            source.selfplay, full_simulations=source.selfplay.full_simulations + 1
        ),
    )
    path = root / "bad.yaml"
    path.write_text(yaml.safe_dump(changed.as_dict()))
    with pytest.raises(ValueError, match="exactly one declared treatment"):
        prepare_recovery_gate(root / "protected.yaml", path)
