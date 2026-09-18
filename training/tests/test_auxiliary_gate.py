"""Auxiliary admission inherits all evidence without making new speed claims."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from scripts import prepare_auxiliary_training_profile as preparation
from scripts.prepare_pie_promotion_gate import prepare_promotion_gate
from scripts import training_disaster_recovery as recovery
from startrain import search_allocation_gate as gate
from startrain.auxiliary_policy import (
    AUXILIARY_DEFAULT_WEIGHTS,
    auxiliary_training_config,
)
from startrain.config import load_config
from test_pie_promotion_gate import promotion_fixture
from test_promotion_gate_disaster_recovery import promotion_transition
from test_search_allocation_gate import reference, write_json
from test_training_disaster_recovery import _snapshot, _snapshot_payload


def fixture(tmp_path, monkeypatch, *, controlled=False, rollback=False):
    _, source, root, original, older_path, source_path = promotion_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    prepare_promotion_gate(older_path, source_path)
    target = auxiliary_training_config(source, recovery=rollback)
    target_path = root / "auxiliary-profile.yaml"
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    return source, target, root, original, source_path, target_path


def read_transition(target):
    root = Path(target.orchestration.directories.root)
    path = gate.allocation_gate_path(target)
    envelope = json.loads(path.read_text())
    receipt_path = root / envelope["auxiliary_prediction_transition"]["path"]
    return path, envelope, receipt_path, json.loads(receipt_path.read_text())


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
def test_auxiliary_gate_preserves_entire_measured_closure(
    tmp_path, monkeypatch, controlled, rollback
):
    source, target, root, _, source_path, target_path = fixture(
        tmp_path, monkeypatch, controlled=controlled, rollback=rollback
    )
    old = gate.allocation_gate_path(source).read_bytes()
    result = preparation.prepare_auxiliary_gate(source_path, target_path)
    assert (
        result["deployment_performed"]
        is result["new_objective_performance_qualified"]
        is False
    )
    path, envelope, receipt_path, receipt = read_transition(target)
    inherited = dict(envelope)
    inherited.pop("auxiliary_prediction_transition")
    inherited["target_config_sha256"] = gate.canonical_config_sha256(source)
    assert inherited == json.loads(old)
    assert {
        "training_policy_transition",
        "promotion_allocation_transition",
    } <= envelope.keys()
    assert receipt["source_gate"] == reference(root, gate.allocation_gate_path(source))
    assert path.stat().st_mode & 0o222 == receipt_path.stat().st_mode & 0o222 == 0
    assert set(gate._VERIFIED_GATES[str(gate.allocation_gate_path(source))]) <= set(
        gate._VERIFIED_GATES[str(path)]
    )
    preparation.ensure_auxiliary_gate(source_path, target_path)
    with pytest.raises(FileExistsError, match="never overwritten"):
        preparation.prepare_auxiliary_gate(source_path, target_path)
    assert gate.allocation_gate_path(source).read_bytes() == old


@pytest.mark.parametrize(
    "mutation", ["trunk", "loss", "optimizer", "learner", "search", "arena"]
)
def test_auxiliary_gate_rejects_every_nonauxiliary_setting(
    tmp_path, monkeypatch, mutation
):
    _, target, _, _, source_path, target_path = fixture(tmp_path, monkeypatch)
    if mutation == "trunk":
        target = replace(target, model=replace(target.model, dropout=0.1))
    elif mutation == "loss":
        target = replace(target, loss=replace(target.loss, ownership=0.1))
    elif mutation == "optimizer":
        target = replace(target, optimizer=replace(target.optimizer, adamw_lr=0.0001))
    elif mutation == "learner":
        target = replace(
            target,
            learner=replace(
                target.learner,
                max_replay_lag_steps=target.learner.max_replay_lag_steps + 1,
            ),
        )
    elif mutation == "search":
        target = replace(
            target,
            selfplay=replace(
                target.selfplay, full_simulations=target.selfplay.full_simulations + 1
            ),
        )
    else:
        target = replace(target, arena=replace(target.arena, alpha=0.04))
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    with pytest.raises(ValueError, match="must preserve every"):
        preparation.prepare_auxiliary_gate(source_path, target_path)
    assert not gate.allocation_gate_path(target).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "claim",
        "scope",
        "unknown",
        "source_hash",
        "source_alias",
        "inheritance",
        "chain",
    ],
)
def test_auxiliary_gate_fails_closed_for_forged_or_chained_receipts(
    tmp_path, monkeypatch, mutation
):
    source, target, root, _, source_path, target_path = fixture(tmp_path, monkeypatch)
    preparation.prepare_auxiliary_gate(source_path, target_path)
    path, envelope, receipt_path, receipt = read_transition(target)
    path.chmod(0o644)
    receipt_path.chmod(0o644)
    if mutation == "claim":
        receipt["new_objective_performance_qualified"] = True
    elif mutation == "scope":
        receipt["measurement_scope"] = "new-training-performance"
    elif mutation == "unknown":
        receipt["skip_reports"] = True
    elif mutation == "source_hash":
        receipt["source_config_sha256"] = "0" * 64
    elif mutation == "source_alias":
        alias = root / "alias.json"
        alias.write_bytes(gate.allocation_gate_path(source).read_bytes())
        receipt["source_gate"] = reference(root, alias)
    elif mutation == "inheritance":
        envelope.pop("training_policy_transition")
    else:
        source_gate = gate.allocation_gate_path(source)
        original = json.loads(source_gate.read_text())
        original["auxiliary_prediction_transition"] = envelope[
            "auxiliary_prediction_transition"
        ]
        source_gate.chmod(0o644)
        write_json(source_gate, original)
        receipt["source_gate"] = reference(root, source_gate)
    write_json(receipt_path, receipt)
    envelope["auxiliary_prediction_transition"] = reference(root, receipt_path)
    write_json(path, envelope)
    with pytest.raises(ValueError, match="auxiliary transition"):
        gate.validate_production_ring_allocations(target)


def test_auxiliary_gate_rehashes_all_source_evidence_even_with_unchanged_signatures(
    tmp_path, monkeypatch
):
    _, target, root, original, source_path, target_path = fixture(tmp_path, monkeypatch)
    preparation.prepare_auxiliary_gate(source_path, target_path)
    report = root / original["groups"][0]["report"]["path"]
    original_signature = gate._signature
    fixed = original_signature(report)
    monkeypatch.setattr(
        gate,
        "_signature",
        lambda path: fixed if path == report else original_signature(path),
    )
    data = report.read_bytes()
    report.write_bytes(b" " + data[1:])
    with pytest.raises(ValueError, match="hash mismatch"):
        preparation.ensure_auxiliary_gate(source_path, target_path)


@pytest.mark.parametrize("rollback", [False, True])
def test_prepared_profile_has_exact_weights_and_never_overwrites(
    tmp_path, monkeypatch, rollback
):
    source, _, root, _, source_path, _ = fixture(tmp_path, monkeypatch)
    output = root / "prepared.yaml"
    source_bytes = source_path.read_bytes()
    result = preparation.prepare_profile(source_path, output, recovery=rollback)
    config = load_config(output)
    assert config == auxiliary_training_config(source, recovery=rollback)
    assert {name: getattr(config.loss, name) for name in AUXILIARY_DEFAULT_WEIGHTS} == {
        name: 0.0 if rollback else weight
        for name, weight in AUXILIARY_DEFAULT_WEIGHTS.items()
    }
    assert result["deployment_performed"] is False
    assert source_path.read_bytes() == source_bytes
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        preparation.prepare_profile(source_path, output, recovery=rollback)


def test_interrupted_receipt_publication_can_resume_without_rewriting_it(
    tmp_path, monkeypatch
):
    _, target, _, _, source_path, target_path = fixture(tmp_path, monkeypatch)
    publish = preparation._publish

    def interrupted(root, path, payload):
        if path == gate.allocation_gate_path(target):
            raise RuntimeError("interrupted before gate")
        return publish(root, path, payload)

    monkeypatch.setattr(preparation, "_publish", interrupted)
    with pytest.raises(RuntimeError, match="interrupted before gate"):
        preparation.prepare_auxiliary_gate(source_path, target_path)
    receipt = gate.allocation_gate_path(target).with_suffix(
        ".auxiliary-transition.json"
    )
    original = receipt.read_bytes(), receipt.stat().st_ino
    monkeypatch.setattr(preparation, "_publish", publish)
    preparation.ensure_auxiliary_gate(source_path, target_path)
    assert (receipt.read_bytes(), receipt.stat().st_ino) == original
    gate.validate_production_ring_allocations(target)


def disaster_fixture(tmp_path, monkeypatch):
    run, source, _, old_refs = promotion_transition(tmp_path, monkeypatch)
    source_path = run.root / "auxiliary-source.yaml"
    source_path.write_bytes(run.profile.read_bytes())
    target = auxiliary_training_config(source)
    target_path = run.root / "auxiliary-target.yaml"
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    preparation.prepare_auxiliary_gate(source_path, target_path)
    _, envelope, _, receipt = read_transition(target)
    run.profile.write_bytes(target_path.read_bytes())
    (run.root / "profile.sha256").write_text(
        f"{reference(run.root, run.profile)['sha256']}  {run.profile.name}\n"
    )
    refs = [
        *old_refs,
        receipt["source_profile"],
        receipt["source_gate"],
        envelope["auxiliary_prediction_transition"],
    ]
    return run, target, envelope, receipt, refs


def test_backup_restores_policy_promotion_auxiliary_and_all_original_dependencies(
    tmp_path, monkeypatch
):
    run, target, _, _, refs = disaster_fixture(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for ref in refs:
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
    original_root = run.root
    original_root.rename(tmp_path / "lost-original")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    recovery.restore_snapshot(snapshot, original_root)
    gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize("missing_index", [6, 7, 8])
def test_offline_verifier_requires_auxiliary_source_profile_gate_and_receipt(
    tmp_path, monkeypatch, missing_index
):
    run, _, _, _, refs = disaster_fixture(tmp_path, monkeypatch)
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


def test_offline_closure_rejects_auxiliary_nesting_or_dropped_evidence(
    tmp_path, monkeypatch
):
    run, _, envelope, receipt, _ = disaster_fixture(tmp_path, monkeypatch)

    def reader(path, digest):
        return gate._json(gate._read_ref(run.root, {"path": path, "sha256": digest})[1])

    def chained(path, digest):
        result = reader(path, digest)
        if path == receipt["source_gate"]["path"]:
            result["auxiliary_prediction_transition"] = envelope[
                "auxiliary_prediction_transition"
            ]
        return result

    with pytest.raises(recovery.DisasterRecoveryError, match="cannot be chained"):
        recovery._allocation_gate_dependency_closure(envelope, chained)
    changed = deepcopy(envelope)
    changed["groups"] = []
    with pytest.raises(recovery.DisasterRecoveryError, match="changed its source"):
        recovery._allocation_gate_dependency_closure(changed, reader)
