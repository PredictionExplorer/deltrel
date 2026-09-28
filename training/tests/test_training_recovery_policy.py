from dataclasses import replace
from pathlib import Path

import pytest

from startrain.config import load_config
from startrain.training_recovery_policy import (
    protected_champion_config,
    reuse_config,
    validate_training_recovery_transition,
)


def source_config():
    config = load_config(Path(__file__).parents[1] / "configs/h100-8gpu-pie-even.yaml")
    return replace(
        config,
        learner=replace(
            config.learner,
            target_updates_per_new_sample=1.5,
            candidate_interval_examples=7_500_000,
            selfplay_snapshot_interval_examples=1_500_000,
            selfplay_snapshot_warmup_interval_examples=1_500_000,
            max_replay_lag_steps=120_000,
        ),
    )


def test_reuse_profile_preserves_model_search_and_fresh_data_clocks():
    source = source_config()
    target = reuse_config(source)
    assert validate_training_recovery_transition(source, target) == "reuse_clock"
    assert target.learner.target_updates_per_new_sample == 2
    assert target.learner.reuse_clock_reference_target == 1.5
    assert target.learner.max_replay_lag_steps == 160_000
    assert target.learner.candidate_interval_examples == 10_000_000
    assert target.learner.selfplay_snapshot_interval_examples == 2_000_000
    assert target.train == source.train
    assert target.model == source.model
    assert target.optimizer == source.optimizer
    assert target.selfplay == source.selfplay
    assert target.arena == source.arena
    restored = reuse_config(target, target=1.5)
    assert restored.learner.reuse_clock_reference_target == 1.5
    assert restored.learner.max_replay_lag_steps == 120_000
    assert restored.learner.candidate_interval_examples == 7_500_000


def test_each_treatment_is_separately_admitted():
    source = source_config()
    protected = protected_champion_config(source, after_ns=100)
    assert (
        validate_training_recovery_transition(source, protected) == "protected_champion"
    )
    coupled = reuse_config(protected)
    assert validate_training_recovery_transition(protected, coupled) == "reuse_clock"
    with pytest.raises(ValueError, match="exactly one"):
        validate_training_recovery_transition(source, coupled)
    with pytest.raises(ValueError, match="exactly one"):
        validate_training_recovery_transition(
            source,
            replace(protected, train=replace(protected.train, gradient_clip_norm=2)),
        )


def test_recovery_can_preserve_reference_without_increasing_reuse():
    source = source_config()
    compatible = reuse_config(source, target=1.5)
    assert compatible.learner.reuse_clock_reference_target == 1.5
    assert validate_training_recovery_transition(source, compatible) == "reuse_clock"
    with pytest.raises(ValueError):
        reuse_config(compatible, target=1.5)


@pytest.mark.parametrize("target", [True, 0, -1, 2.1, float("nan"), float("inf")])
def test_reuse_targets_are_bounded(target):
    with pytest.raises(ValueError):
        reuse_config(source_config(), target=target)
