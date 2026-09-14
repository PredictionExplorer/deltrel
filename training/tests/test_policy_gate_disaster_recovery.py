import json

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from startrain import search_allocation_gate as gate
from startrain.pie_policy import pie_training_config
from test_allocation_gate_disaster_recovery import prepared
from test_search_allocation_gate import reference, write_json
from test_training_disaster_recovery import _snapshot, _snapshot_payload


def prepared_transition(tmp_path, monkeypatch):
    run, source, original = prepared(tmp_path, monkeypatch)
    source_profile = run.root / "policy-source.yaml"
    source_profile.write_text(yaml.safe_dump(source.as_dict()))
    target = pie_training_config(source)
    receipt = {
        "format": gate.POLICY_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": gate.POLICY_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": reference(run.root, source_profile),
        "source_gate": reference(run.root, gate.allocation_gate_path(source)),
        "source_config_sha256": gate.canonical_config_sha256(source),
        "target_config_sha256": gate.canonical_config_sha256(target),
        "measurement_scope": gate.POLICY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    receipt_path = run.root / "status/policy-transition.json"
    write_json(receipt_path, receipt)
    target_gate = {
        **original,
        "target_config_sha256": gate.canonical_config_sha256(target),
        "training_policy_transition": reference(run.root, receipt_path),
    }
    write_json(gate.allocation_gate_path(target), target_gate)
    gate.validate_production_ring_allocations(target)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = reference(run.root, run.profile)["sha256"]
    (run.root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    utd_path = run.root / "learner/utd-segment.json"
    utd = json.loads(utd_path.read_text())
    utd.update(training_objective="ring10_pie", baseline_committed_replay_samples=0)
    write_json(utd_path, utd)
    return run, source, target, receipt, target_gate


def test_policy_gate_snapshot_restores_receipt_source_and_original_evidence(
    tmp_path, monkeypatch
):
    run, source, target, receipt, target_gate = prepared_transition(
        tmp_path, monkeypatch
    )
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for ref in (
        receipt["source_profile"],
        receipt["source_gate"],
        target_gate["training_policy_transition"],
    ):
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
    assert catalog[receipt["source_profile"]["path"]]["kind"] == "profile"
    original_root = run.root
    original_root.rename(tmp_path / "lost-original")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    recovery.restore_snapshot(snapshot, original_root)
    assert gate.allocation_gate_path(source).is_file()
    gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize(
    "missing", ["source_profile", "source_gate", "training_policy_transition"]
)
def test_snapshot_verifier_rejects_missing_transition_dependency(
    tmp_path, monkeypatch, missing
):
    run, _, _, receipt, target_gate = prepared_transition(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    original = recovery._snapshot_envelope
    ref = (
        target_gate[missing]
        if missing == "training_policy_transition"
        else receipt[missing]
    )

    def altered(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        del catalog[ref["path"]]
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", altered)
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="allocation evidence dependency is missing",
    ):
        recovery.verify_snapshot(snapshot)


def test_dependency_closure_retains_controlled_graph_live_evidence_and_rejects_chains():
    def ref(name):
        return {"path": f"status/{name}.json", "sha256": "a" * 64}

    source = {
        "format": gate.GATE_FORMAT,
        "schema_version": 1,
        "run_id": "run",
        "target_config_sha256": "b" * 64,
        "baseline_profile": ref("baseline"),
        "groups": [],
        "graph_cache_controlled_activation": ref("activation"),
    }
    target = {
        **source,
        "target_config_sha256": "c" * 64,
        "training_policy_transition": ref("transition"),
    }
    receipt = {
        "format": gate.POLICY_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": gate.POLICY_TRANSITION_CLASS,
        "run_id": "run",
        "source_profile": ref("source-profile"),
        "source_gate": ref("search-allocation-gates/" + "b" * 64),
        "source_config_sha256": "b" * 64,
        "target_config_sha256": "c" * 64,
        "measurement_scope": gate.POLICY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    values = {
        "status/transition.json": receipt,
        "status/search-allocation-gates/" + "b" * 64 + ".json": source,
        "status/activation.json": {
            "format": "startrain.graph-cache-controlled-activation",
            "schema_version": 1,
            "live_workload_evidence": ref("live-workload"),
        },
    }
    dependencies = recovery._allocation_gate_dependency_closure(
        target, lambda path, _: values[path]
    )
    assert ("status/live-workload.json", "a" * 64, "status-json") in dependencies
    assert ("status/source-profile.json", "a" * 64, "profile") in dependencies
    source["training_policy_transition"] = ref("transition")
    with pytest.raises(recovery.DisasterRecoveryError, match="cannot be chained"):
        recovery._allocation_gate_dependency_closure(
            target, lambda path, _: values[path]
        )
