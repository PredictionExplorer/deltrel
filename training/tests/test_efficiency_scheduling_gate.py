"""Scheduling inherits measured search authority without widening its scope."""

from dataclasses import replace
import hashlib
import json

import pytest
import yaml

from deltreltrain import search_allocation_gate as gate
from deltreltrain.config_compatibility import without_arena_clinch_default
from deltreltrain.efficiency_scheduling import efficiency_scheduling_config
from scripts.prepare_auxiliary_training_profile import prepare_auxiliary_gate
import scripts.prepare_efficiency_scheduling_gate as preparation
from scripts.prepare_efficiency_scheduling_gate import prepare_scheduling_gate
from test_auxiliary_gate import fixture as auxiliary_fixture
from test_search_allocation_gate import reference, write_json


def scheduling_fixture(tmp_path, monkeypatch, *, controlled=False):
    _, source, root, original, previous_path, source_path = auxiliary_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    prepare_auxiliary_gate(previous_path, source_path)
    target = efficiency_scheduling_config(source)
    target_path = root / "efficiency-scheduling.yaml"
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    return source, target, root, original, source_path, target_path


@pytest.mark.parametrize("controlled", [False, True])
def test_outer_scheduling_receipt_preserves_full_admission_chain(
    tmp_path, monkeypatch, controlled
):
    source, target, root, _, source_path, target_path = scheduling_fixture(
        tmp_path, monkeypatch, controlled=controlled
    )
    previous_bytes = gate.allocation_gate_path(source).read_bytes()
    result = prepare_scheduling_gate(source_path, target_path)
    assert result["classification"] == gate.SCHEDULING_TRANSITION_CLASS
    assert result["new_objective_performance_qualified"] is False
    assert result["deployment_performed"] is False
    assert gate.allocation_gate_path(source).read_bytes() == previous_bytes
    envelope = json.loads(gate.allocation_gate_path(target).read_text())
    assert {
        "auxiliary_prediction_transition",
        "promotion_allocation_transition",
        "training_policy_transition",
    } <= envelope.keys()
    inherited = dict(envelope)
    inherited.pop("efficiency_scheduling_transition")
    inherited["target_config_sha256"] = gate.canonical_config_sha256(source)
    assert inherited == json.loads(previous_bytes)
    assert set(gate._VERIFIED_GATES[str(gate.allocation_gate_path(source))]) <= set(
        gate._VERIFIED_GATES[str(gate.allocation_gate_path(target))]
    )
    for ref in (result["gate"], result["receipt"]):
        assert (root / ref["path"]).stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="never overwrite"):
        prepare_scheduling_gate(source_path, target_path)


def test_disabled_defaults_keep_legacy_gate_hash_and_enabled_values_are_distinct(
    tmp_path, monkeypatch
):
    source, target, _, _, _, _ = scheduling_fixture(tmp_path, monkeypatch)
    payload = without_arena_clinch_default(source.as_dict())
    for section, names in (
        (
            "historical_evaluation",
            ("measurement_service_fraction", "measurement_max_wait_seconds"),
        ),
        (
            "model_refresh",
            ("history_horizon_enabled", "history_horizon_initial_seconds"),
        ),
    ):
        for name in names:
            payload["orchestration"][section].pop(name, None)
    old = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert gate.canonical_config_sha256(source) == old
    assert gate.canonical_config_sha256(target) != old
    changed = replace(
        source,
        orchestration=replace(
            source.orchestration,
            model_refresh=replace(
                source.orchestration.model_refresh,
                history_horizon_initial_seconds=3601.0,
            ),
        ),
    )
    assert gate.canonical_config_sha256(changed) != old


@pytest.mark.parametrize(
    "mutation", ["search", "model", "learner", "alpha", "share", "horizon", "cooldown"]
)
def test_scheduling_transition_rejects_every_undeclared_change(
    tmp_path, monkeypatch, mutation
):
    _, target, _, _, source_path, target_path = scheduling_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "search":
        target = replace(
            target,
            arena=replace(target.arena, simulations=target.arena.simulations + 1),
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
    elif mutation == "alpha":
        target = replace(target, arena=replace(target.arena, alpha=0.04))
    elif mutation == "horizon":
        target = replace(
            target,
            orchestration=replace(
                target.orchestration,
                model_refresh=replace(
                    target.orchestration.model_refresh,
                    history_horizon_initial_seconds=7200.0,
                ),
            ),
        )
    else:
        changes = (
            {"measurement_service_fraction": 0.3}
            if mutation == "share"
            else {"cooldown_seconds": 600.0}
        )
        target = replace(
            target,
            orchestration=replace(
                target.orchestration,
                historical_evaluation=replace(
                    target.orchestration.historical_evaluation, **changes
                ),
            ),
        )
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    with pytest.raises(ValueError, match="preserve every non-scheduling"):
        prepare_scheduling_gate(source_path, target_path)
    assert not gate.allocation_gate_path(target).exists()


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "claim", "source_hash", "source_alias", "inheritance", "chain"],
)
def test_scheduling_receipt_rejects_forgery_and_recursive_reinterpretation(
    tmp_path, monkeypatch, mutation
):
    source, target, root, _, source_path, target_path = scheduling_fixture(
        tmp_path, monkeypatch
    )
    result = prepare_scheduling_gate(source_path, target_path)
    path = gate.allocation_gate_path(target)
    envelope = json.loads(path.read_text())
    receipt_path = root / result["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    path.chmod(0o644)
    receipt_path.chmod(0o644)
    if mutation == "unknown":
        receipt["skip_reports"] = True
    elif mutation == "claim":
        receipt["new_objective_performance_qualified"] = True
    elif mutation == "source_hash":
        receipt["source_config_sha256"] = "0" * 64
    elif mutation == "source_alias":
        alias = root / "same-byte-gate.json"
        alias.write_bytes(gate.allocation_gate_path(source).read_bytes())
        receipt["source_gate"] = reference(root, alias)
    elif mutation == "inheritance":
        envelope.pop("auxiliary_prediction_transition")
    else:
        source_gate = gate.allocation_gate_path(source)
        original = json.loads(source_gate.read_text())
        original["efficiency_scheduling_transition"] = result["receipt"]
        source_gate.chmod(0o644)
        write_json(source_gate, original)
        receipt["source_gate"] = reference(root, source_gate)
    write_json(receipt_path, receipt)
    envelope["efficiency_scheduling_transition"] = reference(root, receipt_path)
    write_json(path, envelope)
    with pytest.raises(ValueError, match="scheduling transition"):
        gate.validate_production_ring_allocations(target, _fresh=True)


def test_scheduling_receipt_rehashes_inherited_reports_despite_stat_collision(
    tmp_path, monkeypatch
):
    _, target, root, original, source_path, target_path = scheduling_fixture(
        tmp_path, monkeypatch
    )
    prepare_scheduling_gate(source_path, target_path)
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


def test_interrupted_gate_publication_reuses_only_identical_immutable_receipt(
    tmp_path, monkeypatch
):
    _, target, _, _, source_path, target_path = scheduling_fixture(
        tmp_path, monkeypatch
    )
    publish = preparation._publish

    def interrupted(root, path, value):
        if path == gate.allocation_gate_path(target):
            raise OSError("publication interrupted")
        return publish(root, path, value)

    monkeypatch.setattr(preparation, "_publish", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        prepare_scheduling_gate(source_path, target_path)
    receipt = gate.allocation_gate_path(target).with_suffix(
        ".scheduling-transition.json"
    )
    before = (receipt.read_bytes(), receipt.stat().st_ino)
    monkeypatch.setattr(preparation, "_publish", publish)
    preparation.ensure_scheduling_gate(source_path, target_path)
    preparation.ensure_scheduling_gate(source_path, target_path)
    assert (receipt.read_bytes(), receipt.stat().st_ino) == before
    gate.validate_production_ring_allocations(target, _fresh=True)
