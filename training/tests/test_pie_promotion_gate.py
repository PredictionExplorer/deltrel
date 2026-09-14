"""Promotion-only admission retains the complete original qualification chain."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

import scripts.prepare_pie_promotion_gate as preparation
from scripts.prepare_pie_policy_gate import prepare_policy_gate
from scripts.validate_continuous_profile import validate_continuous_config
from startrain import search_allocation_gate as gate
from startrain.pie_promotion import pie_promotion_config
from test_pie_policy_gate import policy_fixture
from test_search_allocation_gate import reference, write_json


def promotion_fixture(tmp_path, monkeypatch, *, controlled=False):
    _, source, root, original, training_source_path, source_path = policy_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    prepare_policy_gate(training_source_path, source_path)
    target = pie_promotion_config(source)
    target_path = root / "adaptive-promotion-profile.yaml"
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    return source, target, root, original, source_path, target_path


def read_transition(target):
    root = Path(target.orchestration.directories.root)
    path = gate.allocation_gate_path(target)
    envelope = json.loads(path.read_text())
    receipt_path = root / envelope["promotion_allocation_transition"]["path"]
    return path, envelope, receipt_path, json.loads(receipt_path.read_text())


def rewrite_transition(target, mutation):
    path, envelope, receipt_path, receipt = read_transition(target)
    mutation(receipt)
    receipt_path.chmod(0o644)
    path.chmod(0o644)
    write_json(receipt_path, receipt)
    envelope["promotion_allocation_transition"] = reference(
        Path(target.orchestration.directories.root), receipt_path
    )
    write_json(path, envelope)


@pytest.mark.parametrize("controlled", [False, True])
def test_promotion_wrapper_preserves_training_transition_and_original_closure(
    tmp_path, monkeypatch, controlled
):
    source, target, root, _, source_path, target_path = promotion_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    original_bytes = gate.allocation_gate_path(source).read_bytes()
    source_bytes = source_path.read_bytes()
    result = preparation.prepare_promotion_gate(source_path, target_path)
    assert result["deployment_performed"] is False
    assert result["new_objective_performance_qualified"] is False
    assert result["measurement_scope"] == gate.POLICY_TRANSITION_SCOPE
    validate_continuous_config(target)
    validate_continuous_config(source)
    path, envelope, receipt_path, receipt = read_transition(target)
    assert "training_policy_transition" in envelope
    inherited = deepcopy(envelope)
    inherited.pop("promotion_allocation_transition")
    inherited["target_config_sha256"] = gate.canonical_config_sha256(source)
    assert inherited == json.loads(original_bytes)
    assert gate.allocation_gate_path(source).read_bytes() == original_bytes
    assert source_path.read_bytes() == source_bytes
    assert receipt["source_gate"] == reference(root, gate.allocation_gate_path(source))
    assert receipt["classification"] == gate.PROMOTION_TRANSITION_CLASS
    assert path.stat().st_mode & 0o222 == receipt_path.stat().st_mode & 0o222 == 0
    source_proof = gate._VERIFIED_GATES[str(gate.allocation_gate_path(source))]
    assert set(source_proof) <= set(gate._VERIFIED_GATES[str(path)])
    if controlled:
        assert "graph_cache_controlled_activation" in envelope
    with pytest.raises(FileExistsError, match="never overwritten"):
        preparation.prepare_promotion_gate(source_path, target_path)


@pytest.mark.parametrize(
    "mutation", ["simulations", "alpha", "model", "learner", "pda", "cache"]
)
def test_promotion_admission_allows_no_other_config_change(
    tmp_path, monkeypatch, mutation
):
    _, target, _, _, source_path, target_path = promotion_fixture(tmp_path, monkeypatch)
    if mutation == "simulations":
        target = replace(
            target,
            arena=replace(target.arena, simulations=target.arena.simulations + 1),
        )
    elif mutation == "alpha":
        target = replace(target, arena=replace(target.arena, alpha=0.04))
    elif mutation == "model":
        target = replace(target, model=replace(target.model, dropout=0.1))
    elif mutation == "learner":
        target = replace(
            target,
            learner=replace(
                target.learner,
                max_replay_lag_steps=target.learner.max_replay_lag_steps + 1,
            ),
        )
    elif mutation == "pda":
        pdas = (0, *target.selfplay.variants.handicap_pda[1:])
        target = replace(
            target,
            arena=replace(target.arena, segment_handicap_pda=pdas),
            selfplay=replace(
                target.selfplay,
                variants=replace(target.selfplay.variants, handicap_pda=pdas),
            ),
        )
    else:
        refresh = target.orchestration.model_refresh
        target = replace(
            target,
            orchestration=replace(
                target.orchestration,
                model_refresh=replace(
                    refresh,
                    inference=replace(refresh.inference, cuda_graph_max_entries=33),
                ),
            ),
        )
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    with pytest.raises(ValueError, match="preserve every non-allocation"):
        preparation.prepare_promotion_gate(source_path, target_path)
    assert not gate.allocation_gate_path(target).exists()


@pytest.mark.parametrize(
    "mutation", ["claim", "scope", "unknown", "source_hash", "source_alias"]
)
def test_receipt_cannot_widen_qualification_or_change_source_authority(
    tmp_path, monkeypatch, mutation
):
    source, target, root, _, source_path, target_path = promotion_fixture(
        tmp_path, monkeypatch
    )
    preparation.prepare_promotion_gate(source_path, target_path)

    def mutate(receipt):
        if mutation == "claim":
            receipt["new_objective_performance_qualified"] = True
        elif mutation == "scope":
            receipt["measurement_scope"] = "new-objective-elo-per-hour"
        elif mutation == "unknown":
            receipt["skip_original_reports"] = True
        elif mutation == "source_hash":
            receipt["source_config_sha256"] = "0" * 64
        else:
            alias = root / "source-gate-alias.json"
            alias.write_bytes(gate.allocation_gate_path(source).read_bytes())
            receipt["source_gate"] = reference(root, alias)

    rewrite_transition(target, mutate)
    with pytest.raises(ValueError, match="promotion transition"):
        gate.validate_production_ring_allocations(target)


def test_promotion_transitions_cannot_chain_or_cycle(tmp_path, monkeypatch):
    source, target, root, _, source_path, target_path = promotion_fixture(
        tmp_path, monkeypatch
    )
    preparation.prepare_promotion_gate(source_path, target_path)
    source_gate = gate.allocation_gate_path(source)
    inherited = json.loads(source_gate.read_text())
    inherited["promotion_allocation_transition"] = reference(
        root, gate.allocation_gate_path(target)
    )
    source_gate.chmod(0o644)
    write_json(source_gate, inherited)
    rewrite_transition(
        target, lambda receipt: receipt.update(source_gate=reference(root, source_gate))
    )
    with pytest.raises(ValueError, match="cannot be chained"):
        gate.validate_production_ring_allocations(target)


def test_copied_gate_cannot_drop_original_reports_or_training_receipt(
    tmp_path, monkeypatch
):
    _, target, _, _, source_path, target_path = promotion_fixture(tmp_path, monkeypatch)
    preparation.prepare_promotion_gate(source_path, target_path)
    path, envelope, _, _ = read_transition(target)
    path.chmod(0o644)
    envelope.pop("training_policy_transition")
    write_json(path, envelope)
    with pytest.raises(ValueError, match="preserve the complete source gate"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize("controlled", [False, True])
def test_adaptive_admission_rehashes_source_even_when_stat_signatures_collide(
    tmp_path, monkeypatch, controlled
):
    _, target, root, original, source_path, target_path = promotion_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    preparation.prepare_promotion_gate(source_path, target_path)
    report = root / original["groups"][0]["report"]["path"]
    old_signature = gate._signature
    fixed = old_signature(report)
    monkeypatch.setattr(
        gate,
        "_signature",
        lambda path: fixed if path == report else old_signature(path),
    )
    contents = report.read_bytes()
    report.write_bytes(b" " + contents[1:])
    assert report.stat().st_size == len(contents)
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_production_ring_allocations(target)


def test_source_change_after_receipt_leaves_no_target_admission(tmp_path, monkeypatch):
    _, target, root, original, source_path, target_path = promotion_fixture(
        tmp_path, monkeypatch
    )
    publish = preparation._publish

    def changed(root_arg, path, payload):
        result = publish(root_arg, path, payload)
        if path.name.endswith(".promotion-transition.json"):
            report = root / original["groups"][0]["report"]["path"]
            report.write_text(report.read_text() + "\n")
        return result

    monkeypatch.setattr(preparation, "_publish", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        preparation.prepare_promotion_gate(source_path, target_path)
    assert not gate.allocation_gate_path(target).exists()
