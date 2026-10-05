from dataclasses import asdict, replace

import numpy as np
import pytest

from deltreltrain.policy_targets import constrain_policy_target
from deltreltrain.selfplay import SelfPlayActor, SelfPlayConfig


def test_actor_changes_only_recorded_targets_and_records_treatment():
    from types import SimpleNamespace
    import torch

    actor = SelfPlayActor(
        object(),
        SimpleNamespace(model_version="m", model_step=0, model_identity="m"),
        object(),
        SelfPlayConfig(policy_target_scale=0.25),
    )
    result = SimpleNamespace(
        action_offsets=[0, 2],
        actions=[2, 7],
        policy_target=[0.1, 0.9],
        priors=[0.8, 0.2],
        selected_actions=[7],
        terminal=[False],
    )
    trajectories = [[]]
    actor._record_decisions(
        trajectories,
        [SimpleNamespace(stones=torch.zeros(50))],
        SimpleNamespace(stones_placed=[3]),
        result,
        full_search=True,
        simulations=64,
        search_seed=17,
    )
    decision = trajectories[0][0]
    np.testing.assert_allclose(decision.policy[[2, 7]], [0.625, 0.375])
    assert np.count_nonzero(decision.policy) == 2
    assert "target_mix_v1=0.25" in decision.search_evidence
    assert result.selected_actions == [7]
    assert result.policy_target == [0.1, 0.9]


def test_profile_default_authority_omits_disabled_target_fields():
    from deltreltrain.config import load_config

    config = load_config("configs/small.yaml")
    assert "policy_target_scale" not in config.as_dict()["selfplay"]
    treatment = replace(
        config, selfplay=replace(config.selfplay, policy_target_scale=0.5)
    )
    assert treatment.as_dict()["selfplay"]["policy_target_scale"] == 0.5


def test_mixture_is_target_only_and_has_exact_endpoints():
    prior = np.array([0.8, 0.2, 0], dtype=np.float32)
    target = np.array([0.1, 0.9, 0], dtype=np.float32)
    before = target.copy()
    result = constrain_policy_target(target, prior, scale=0.25)
    np.testing.assert_allclose(result.probabilities, [0.625, 0.375, 0])
    np.testing.assert_array_equal(target, before)
    np.testing.assert_array_equal(
        constrain_policy_target(target, prior).probabilities, target
    )
    np.testing.assert_array_equal(
        constrain_policy_target(target, prior, scale=0).probabilities, prior
    )


@pytest.mark.parametrize("limit", [0, 1e-5, 0.01, 0.1, 1.0])
def test_bound_applies_to_returned_distribution_and_is_maximal(limit):
    prior = np.array([0.99, 0.009, 0.001])
    target = np.array([0.01, 0.09, 0.9])
    result = constrain_policy_target(target, prior, max_kl=limit)
    assert result.kl_nats <= limit + 1e-12
    assert np.isclose(result.probabilities.sum(), 1)
    assert (result.probabilities >= 0).all()
    if 0 < result.applied_scale < 0.999:
        larger = constrain_policy_target(
            target, prior, scale=result.applied_scale + 1e-5
        )
        assert larger.kl_nats > limit


def test_finite_kl_does_not_invent_prior_support():
    result = constrain_policy_target(
        np.array([0.0, 1.0]), np.array([1.0, 0.0]), max_kl=10
    )
    np.testing.assert_array_equal(result.probabilities, [1, 0])
    assert result.applied_scale == result.kl_nats == 0


@pytest.mark.parametrize(
    "values", [[-1, 2], [float("nan"), 1], [float("inf"), 1], [0, 0], []]
)
def test_bad_probabilities_fail(values):
    with pytest.raises(ValueError):
        constrain_policy_target(np.array(values), np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_target_scale": True},
        {"policy_target_scale": -1},
        {"policy_target_scale": 1.1},
        {"policy_target_scale": float("nan")},
        {"policy_target_max_kl": True},
        {"policy_target_max_kl": -1},
        {"policy_target_max_kl": float("inf")},
    ],
)
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        replace(SelfPlayConfig(), **kwargs)


def test_config_serializes_treatment_explicitly():
    assert asdict(SelfPlayConfig())["policy_target_scale"] == 1.0
    assert (
        asdict(replace(SelfPlayConfig(), policy_target_max_kl=0.2))[
            "policy_target_max_kl"
        ]
        == 0.2
    )
