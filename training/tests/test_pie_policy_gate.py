"""Policy-only continuation retains, but cannot widen, measured admission."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from scripts.prepare_pie_policy_gate import prepare_policy_gate
from scripts.validate_continuous_profile import validate_continuous_config
from deltreltrain import search_allocation_gate as gate
from deltreltrain.pie_policy import pie_training_config
from test_graph_cache_controlled_activation import controlled_admission_fixture
from test_search_allocation_gate import fixture, reference, write_json


def policy_fixture(tmp_path, monkeypatch, *, controlled=False):
    factory = controlled_admission_fixture if controlled else fixture
    _, source, root, original = factory(tmp_path, monkeypatch)
    target = pie_training_config(source)
    source_path, target_path = root / "source-profile.yaml", root / "pie-profile.yaml"
    source_path.write_text(yaml.safe_dump(source.as_dict()))
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    return source, target, root, original, source_path, target_path


def read_transition(target):
    root = Path(target.orchestration.directories.root)
    path = gate.allocation_gate_path(target)
    envelope = json.loads(path.read_text())
    receipt_path = root / envelope["training_policy_transition"]["path"]
    return path, envelope, receipt_path, json.loads(receipt_path.read_text())


def rewrite_transition(target, mutate):
    path, envelope, receipt_path, receipt = read_transition(target)
    receipt_path.chmod(0o644)
    path.chmod(0o644)
    mutate(receipt)
    write_json(receipt_path, receipt)
    envelope["training_policy_transition"] = reference(
        Path(target.orchestration.directories.root), receipt_path
    )
    write_json(path, envelope)


@pytest.mark.parametrize("controlled", [False, True])
def test_policy_receipt_preserves_exact_original_evidence_and_legacy_admission(
    tmp_path, monkeypatch, controlled
):
    source, target, root, original, source_path, target_path = policy_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    original_bytes = gate.allocation_gate_path(source).read_bytes()
    source_bytes = source_path.read_bytes()
    result = prepare_policy_gate(source_path, target_path)
    assert result["classification"] == gate.POLICY_TRANSITION_CLASS
    assert result["new_objective_performance_qualified"] is False
    assert result["deployment_performed"] is False
    validate_continuous_config(target)
    validate_continuous_config(source)
    assert gate.allocation_gate_path(source).read_bytes() == original_bytes
    assert source_path.read_bytes() == source_bytes
    path, new_gate, receipt_path, receipt = read_transition(target)
    inherited = deepcopy(new_gate)
    inherited.pop("training_policy_transition")
    inherited["target_config_sha256"] = gate.canonical_config_sha256(source)
    assert inherited == original
    assert receipt["measurement_scope"] == gate.POLICY_TRANSITION_SCOPE
    assert receipt["source_gate"] == reference(root, gate.allocation_gate_path(source))
    verified = gate._VERIFIED_GATES[str(path)]
    assert set(gate._VERIFIED_GATES[str(gate.allocation_gate_path(source))]) <= set(
        verified
    )
    assert str(source_path.relative_to(root)) in verified
    assert str(receipt_path.relative_to(root)) in verified
    assert path.stat().st_mode & 0o222 == receipt_path.stat().st_mode & 0o222 == 0
    if controlled:
        assert target.orchestration.model_refresh.inference.cuda_graph_max_entries == 32
        assert "graph_cache_controlled_activation" in inherited
    with pytest.raises(FileExistsError, match="never overwritten"):
        prepare_policy_gate(source_path, target_path)


@pytest.mark.parametrize(
    "mutation",
    ["search_budget", "pda", "swap", "cache", "model", "learner", "optimizer"],
)
def test_continuation_cannot_change_execution_or_learning_settings(
    tmp_path, monkeypatch, mutation
):
    _, target, _, _, source_path, target_path = policy_fixture(tmp_path, monkeypatch)
    if mutation == "search_budget":
        target = replace(
            target, selfplay=replace(target.selfplay, full_simulations=385)
        )
    elif mutation == "pda":
        pdas = (0, *target.selfplay.variants.handicap_pda[1:])
        target = replace(
            target,
            selfplay=replace(
                target.selfplay,
                variants=replace(target.selfplay.variants, handicap_pda=pdas),
            ),
            arena=replace(target.arena, segment_handicap_pda=pdas),
        )
    elif mutation == "swap":
        target = replace(
            target,
            selfplay=replace(
                target.selfplay,
                variants=replace(target.selfplay.variants, swap_dead_zone=0.01),
            ),
        )
    elif mutation == "cache":
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
    else:
        target = replace(target, optimizer=replace(target.optimizer, adamw_lr=0.001))
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    with pytest.raises(ValueError, match="preserve every non-policy"):
        prepare_policy_gate(source_path, target_path)
    assert not gate.allocation_gate_path(target).exists()


@pytest.mark.parametrize(
    "mutation", ["claim", "scope", "source_hash", "unknown_field", "source_gate_alias"]
)
def test_receipt_cannot_claim_new_qualification_or_rebind_original_authority(
    tmp_path, monkeypatch, mutation
):
    source, target, root, _, source_path, target_path = policy_fixture(
        tmp_path, monkeypatch
    )
    prepare_policy_gate(source_path, target_path)

    def change(receipt):
        if mutation == "claim":
            receipt["new_objective_performance_qualified"] = True
        elif mutation == "scope":
            receipt["measurement_scope"] = "new-objective-elo-per-hour"
        elif mutation == "source_hash":
            receipt["source_config_sha256"] = "0" * 64
        elif mutation == "unknown_field":
            receipt["skip_original_gate"] = True
        else:
            alias = root / "gate-copy.json"
            alias.write_bytes(gate.allocation_gate_path(source).read_bytes())
            receipt["source_gate"] = reference(root, alias)

    rewrite_transition(target, change)
    with pytest.raises(ValueError, match="policy transition"):
        gate.validate_production_ring_allocations(target)


def test_cached_policy_admission_rechecks_original_reports(tmp_path, monkeypatch):
    _, target, root, original, source_path, target_path = policy_fixture(
        tmp_path, monkeypatch
    )
    prepare_policy_gate(source_path, target_path)
    report = root / original["groups"][0]["report"]["path"]
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_production_ring_allocations(target)


def test_policy_gate_cannot_replace_original_evidence_groups(tmp_path, monkeypatch):
    _, target, _, _, source_path, target_path = policy_fixture(tmp_path, monkeypatch)
    prepare_policy_gate(source_path, target_path)
    path, envelope, _, _ = read_transition(target)
    path.chmod(0o644)
    envelope["groups"] = envelope["groups"][:1]
    write_json(path, envelope)
    with pytest.raises(ValueError, match="preserve the complete original gate"):
        gate.validate_production_ring_allocations(target)


def test_policy_receipts_cannot_chain(tmp_path, monkeypatch):
    source, target, root, _, source_path, target_path = policy_fixture(
        tmp_path, monkeypatch
    )
    prepare_policy_gate(source_path, target_path)
    original_path = gate.allocation_gate_path(source)
    original = json.loads(original_path.read_text())
    original["training_policy_transition"] = {
        "path": "missing.json",
        "sha256": "0" * 64,
    }
    write_json(original_path, original)
    rewrite_transition(
        target,
        lambda receipt: receipt.update(source_gate=reference(root, original_path)),
    )
    with pytest.raises(ValueError, match="cannot be chained"):
        gate.validate_production_ring_allocations(target)


def test_controlled_cache_evidence_remains_required_after_policy_change(
    tmp_path, monkeypatch
):
    _, target, root, original, source_path, target_path = policy_fixture(
        tmp_path, monkeypatch, controlled=True
    )
    prepare_policy_gate(source_path, target_path)
    activation_path = root / original["graph_cache_controlled_activation"]["path"]
    activation_path.write_text(activation_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_production_ring_allocations(target)
