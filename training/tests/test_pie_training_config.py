from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.prepare_pie_training_profile import pie_training_config, prepare_profile
from scripts.validate_continuous_profile import validate_continuous_config
from startrain.balanced_evaluation import balanced_cells, evaluation_contract
from startrain.checkpoint import game_configs_compatible
from startrain.config import ConfigError, RingWeightStage, load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def legacy_profile():
    return load_config(CONFIGS / "h100-8gpu-largest-board-priority.yaml")


def pie_profile():
    return load_config(CONFIGS / "h100-8gpu-pie-even.yaml")


def test_pie_profile_preserves_model_and_execution_and_changes_the_objective():
    before, after = legacy_profile(), pie_profile()
    assert after == pie_training_config(before)
    assert pie_training_config(after) == after
    validate_continuous_config(after)
    for section in ("model", "loss", "optimizer", "train", "data"):
        assert getattr(before, section) == getattr(after, section)
    assert before.orchestration.gpus == after.orchestration.gpus
    assert before.orchestration.model_refresh == after.orchestration.model_refresh
    assert before.learner.target_updates_per_new_sample == (
        after.learner.target_updates_per_new_sample
    )
    assert after.selfplay.pie and after.game.pie_rule
    assert after.selfplay.pie_even_training
    assert after.selfplay.variants.segment_fractions == {
        "standard": 0.0,
        "classic": 0.0,
        "handicap": 0.1,
        "pie": 0.9,
    }
    assert set(balanced_cells(after.arena)) == {
        "r10/classic-pie",
        "r10/double-pie",
        "r10/classic-handicap",
        "r10/double-handicap",
    }
    assert evaluation_contract(before.arena) != evaluation_contract(after.arena)


@pytest.mark.parametrize("share", [0.0, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("field", ["pie_classic_share", "handicap_classic_share"])
def test_pie_objective_requires_both_move_modes_equally(share, field):
    config = pie_profile()
    with pytest.raises(ConfigError, match="equally split"):
        replace(
            config,
            selfplay=replace(
                config.selfplay,
                variants=replace(config.selfplay.variants, **{field: share}),
            ),
        )


def test_pie_objective_rejects_wrong_global_allocation_or_replay_quotas():
    config = pie_profile()
    with pytest.raises(ConfigError, match="90% pie"):
        replace(
            config,
            selfplay=replace(
                config.selfplay,
                variants=replace(config.selfplay.variants, handicap=0.2, pie=0.8),
            ),
        )
    with pytest.raises(ConfigError, match="90% pie"):
        replace(config, learner=replace(config.learner, segment_quotas={"pie": 1.0}))


def test_pie_objective_rejects_legacy_evaluation_and_missing_board_schedule():
    config = pie_profile()
    with pytest.raises(ConfigError, match="four-cell"):
        replace(config, arena=replace(config.arena, variant_policy="legacy_six"))
    with pytest.raises(ConfigError, match="step 0"):
        replace(
            config,
            orchestration=replace(
                config.orchestration,
                ring_mixture=replace(
                    config.orchestration.ring_mixture, step_weights=()
                ),
            ),
        )
    with pytest.raises(ConfigError, match="ring-10 majority"):
        replace(
            config,
            orchestration=replace(
                config.orchestration,
                ring_mixture=replace(
                    config.orchestration.ring_mixture,
                    step_weights=(RingWeightStage(0, (0.25, 0.25, 0.25, 0.25)),),
                ),
            ),
        )


def test_legacy_profile_hash_payload_does_not_gain_disabled_policy_fields():
    config = legacy_profile()
    assert "pie_even_training" not in config.as_dict()["selfplay"]
    assert "variant_policy" not in config.as_dict()["arena"]
    assert pie_profile().as_dict()["selfplay"]["pie_even_training"] is True
    assert pie_profile().as_dict()["arena"]["variant_policy"] == "pie_even"


def test_profile_preparation_is_local_roundtrippable_and_never_overwrites(tmp_path):
    source = CONFIGS / "h100-8gpu-largest-board-priority.yaml"
    before = source.read_bytes()
    target = tmp_path / "new-profile.yaml"
    report = prepare_profile(source, target)
    assert report["deployment_performed"] is False
    assert load_config(target) == pie_training_config(load_config(source))
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_profile(source, target)
    assert target.read_bytes() == original
    assert source.read_bytes() == before


def test_checkpoint_default_pie_change_preserves_identical_rule_family_only():
    before, after = asdict(legacy_profile().game), asdict(pie_profile().game)
    assert game_configs_compatible(before, after)
    assert game_configs_compatible(after, before)
    assert not game_configs_compatible(before, {**after, "mode": "classic"})
    assert not game_configs_compatible(
        before,
        {
            **after,
            "variants": {**after["variants"], "handicap_max": 8},
        },
    )
    assert not game_configs_compatible(
        before,
        {
            **before,
            "variants": {**before["variants"], "pie_allowed": False},
        },
    )
