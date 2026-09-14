import copy
import hashlib
import json

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from startrain import search_allocation_gate as gate
from startrain.pie_promotion import pie_promotion_config
from test_policy_gate_disaster_recovery import prepared_transition
from test_search_allocation_gate import reference, write_json
from test_training_disaster_recovery import _snapshot, _snapshot_payload


def promotion_transition(tmp_path, monkeypatch):
    run, _, source, training_receipt, source_gate = prepared_transition(
        tmp_path, monkeypatch
    )
    source_profile = run.root / "promotion-source.yaml"
    source_profile.write_text(yaml.safe_dump(source.as_dict()))
    target = pie_promotion_config(source)
    receipt = {
        "format": gate.PROMOTION_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": gate.PROMOTION_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": reference(run.root, source_profile),
        "source_gate": reference(run.root, gate.allocation_gate_path(source)),
        "source_config_sha256": gate.canonical_config_sha256(source),
        "target_config_sha256": gate.canonical_config_sha256(target),
        "measurement_scope": gate.PROMOTION_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    receipt_path = run.root / "status/promotion-transition.json"
    write_json(receipt_path, receipt)
    target_gate = {
        **source_gate,
        "target_config_sha256": gate.canonical_config_sha256(target),
        "promotion_allocation_transition": reference(run.root, receipt_path),
    }
    write_json(gate.allocation_gate_path(target), target_gate)
    gate.validate_production_ring_allocations(target)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = reference(run.root, run.profile)["sha256"]
    (run.root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    refs = [
        receipt["source_profile"],
        receipt["source_gate"],
        target_gate["promotion_allocation_transition"],
        training_receipt["source_profile"],
        training_receipt["source_gate"],
        target_gate["training_policy_transition"],
    ]
    return run, target, target_gate, refs


def test_snapshot_restores_both_transitions_and_original_measurement_closure(
    tmp_path, monkeypatch
):
    run, target, envelope, refs = promotion_transition(tmp_path, monkeypatch)
    training_receipt = json.loads(
        (run.root / envelope["training_policy_transition"]["path"]).read_text()
    )
    promotion_receipt = json.loads(
        (run.root / envelope["promotion_allocation_transition"]["path"]).read_text()
    )
    assert (
        training_receipt["target_config_sha256"]
        == promotion_receipt["source_config_sha256"]
    )
    assert training_receipt["target_config_sha256"] != envelope["target_config_sha256"]
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for ref in refs:
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
    original_root = run.root
    original_root.rename(tmp_path / "lost-original")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    recovery.restore_snapshot(snapshot, original_root)
    gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize("missing_index", range(6))
def test_offline_verifier_requires_every_nested_transition_dependency(
    tmp_path, monkeypatch, missing_index
):
    run, _, _, refs = promotion_transition(tmp_path, monkeypatch)
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


def test_closure_rejects_chained_promotion_and_changed_inherited_evidence(
    tmp_path, monkeypatch
):
    run, _, envelope, _ = promotion_transition(tmp_path, monkeypatch)

    def reader(path, _):
        import json

        return json.loads((run.root / path).read_text())

    receipt = reader(envelope["promotion_allocation_transition"]["path"], None)
    source_path = receipt["source_gate"]["path"]

    def chained(path, digest):
        data = reader(path, digest)
        if path == source_path:
            data["promotion_allocation_transition"] = envelope[
                "promotion_allocation_transition"
            ]
        return data

    with pytest.raises(recovery.DisasterRecoveryError, match="cannot be chained"):
        recovery._allocation_gate_dependency_closure(envelope, chained)
    changed = copy.deepcopy(envelope)
    changed["groups"] = []
    with pytest.raises(recovery.DisasterRecoveryError, match="changed its source"):
        recovery._allocation_gate_dependency_closure(changed, reader)


@pytest.mark.parametrize(
    "transition_kind", ["training_policy_transition", "promotion_allocation_transition"]
)
def test_offline_closure_rejects_same_byte_source_gate_alias_like_live_validator(
    tmp_path, monkeypatch, transition_kind
):
    if transition_kind == "promotion_allocation_transition":
        run, target, envelope, _ = promotion_transition(tmp_path, monkeypatch)
    else:
        run, _, target, _, envelope = prepared_transition(tmp_path, monkeypatch)
    receipt_path = run.root / envelope[transition_kind]["path"]
    receipt = json.loads(receipt_path.read_text())
    original_reference = receipt["source_gate"]
    alias = run.root / "status/same-byte-source-gate-alias.json"
    alias.write_bytes((run.root / original_reference["path"]).read_bytes())
    receipt["source_gate"] = reference(run.root, alias)
    assert receipt["source_gate"]["sha256"] == original_reference["sha256"]
    write_json(receipt_path, receipt)
    envelope[transition_kind] = reference(run.root, receipt_path)
    write_json(gate.allocation_gate_path(target), envelope)

    def reader(path, checksum):
        contents = (run.root / path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == checksum
        return json.loads(contents)

    with pytest.raises(ValueError, match="source configuration's original gate"):
        gate.validate_production_ring_allocations(target, _fresh=True)
    with pytest.raises(
        recovery.DisasterRecoveryError, match="source configuration's original gate"
    ):
        recovery._allocation_gate_dependency_closure(envelope, reader)


def test_offline_closure_rejects_a_training_transition_after_promotion(
    tmp_path, monkeypatch
):
    run, _, _, receipt, envelope = prepared_transition(tmp_path, monkeypatch)

    def reader(path, checksum):
        contents = (run.root / path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == checksum
        payload = json.loads(contents)
        if path == receipt["source_gate"]["path"]:
            payload["promotion_allocation_transition"] = envelope[
                "training_policy_transition"
            ]
        return payload

    with pytest.raises(recovery.DisasterRecoveryError, match="cannot be chained"):
        recovery._allocation_gate_dependency_closure(envelope, reader)
