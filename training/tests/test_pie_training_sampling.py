"""Distribution and admission contracts shared by actors and replay."""

from collections import Counter
from dataclasses import replace

import pytest

from deltreltrain.selfplay import GameVariant, SelfPlayConfig, VariantMixtureConfig
from deltreltrain.variant_training import (
    is_pie_training,
    training_mode_weights,
    training_segment_quotas,
    training_variant_allowed,
)


RING_WEIGHTS = {4: 0.05, 6: 0.05, 8: 0.05, 10: 0.85}


def pie_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        rings=10,
        pie=True,
        pie_even_training=True,
        variants=VariantMixtureConfig(
            enabled=True,
            standard=0,
            classic=0,
            handicap=0.1,
            pie=0.9,
            pie_classic_share=0.5,
            handicap_classic_share=0.5,
        ),
    )


def test_board_conditional_allocation_preserves_global_90_10_and_both_move_rules():
    aggregate = Counter()
    for ring, board_share in RING_WEIGHTS.items():
        modes = training_mode_weights(ring, RING_WEIGHTS)
        assert sum(modes.values()) == pytest.approx(1)
        assert modes["classic-standard"] == modes["double-standard"] == 0
        for mode, weight in modes.items():
            aggregate[mode] += board_share * weight
        if ring != 10:
            assert modes["classic-pie"] == modes["double-pie"] == 0.5
            assert modes["classic-handicap"] == modes["double-handicap"] == 0
    assert aggregate == pytest.approx(
        {
            "classic-standard": 0,
            "double-standard": 0,
            "classic-pie": 0.45,
            "double-pie": 0.45,
            "classic-handicap": 0.05,
            "double-handicap": 0.05,
        }
    )
    assert training_segment_quotas(10, RING_WEIGHTS)["handicap"] == pytest.approx(
        2 / 17
    )
    assert training_segment_quotas(10, {4: 5, 6: 5, 8: 5, 10: 85}) == pytest.approx(
        training_segment_quotas(10, RING_WEIGHTS)
    )


@pytest.mark.parametrize(
    ("weights", "fraction"),
    [
        ({}, 0.1),
        ({4: 0, 10: 0}, 0.1),
        ({4: 1}, 0.1),
        ({4: 0.95, 10: 0.05}, 0.1),
        ({10: -1}, 0),
        ({10: float("inf")}, 0),
        ({10: float("nan")}, 0),
        ({10: True}, 0.1),
        ({9: 1}, 0.1),
        (RING_WEIGHTS, float("nan")),
        (RING_WEIGHTS, -0.1),
        (RING_WEIGHTS, 1.1),
        (RING_WEIGHTS, True),
    ],
)
def test_unrealizable_or_invalid_allocations_fail(weights, fraction):
    with pytest.raises(ValueError):
        training_segment_quotas(10, weights, handicap_fraction=fraction)


def test_zero_handicap_needs_no_largest_board_allocation():
    assert training_segment_quotas(4, {4: 1}, handicap_fraction=0) == {
        "standard": 0,
        "classic": 0,
        "handicap": 0,
        "pie": 1,
    }


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_admission_and_direct_selfplay_enforce_pie_and_largest_board_handicap(
    ring, mode
):
    for handicap in range(1, 10):
        for pie in (False, True):
            allowed = (handicap == 1 and pie) or (
                handicap >= 2 and ring == 10 and not pie
            )
            assert training_variant_allowed(ring, mode, handicap, pie) is allowed
            if allowed:
                direct = SelfPlayConfig(
                    rings=ring,
                    mode=mode,
                    handicap=handicap,
                    pie=pie,
                    pie_even_training=True,
                )
                assert direct.variant == GameVariant(mode, handicap, pie)
            else:
                with pytest.raises(ValueError, match="pie"):
                    SelfPlayConfig(
                        rings=ring,
                        mode=mode,
                        handicap=handicap,
                        pie=pie,
                        pie_even_training=True,
                    )


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_random_actor_draws_conditioned_mixture_and_retain_every_handicap_mode(ring):
    master = pie_config()
    mixture = master.variant_mixture_for_ring(ring, RING_WEIGHTS)
    counts = Counter()
    handicap_modes = set()
    for seed in range(40_000):
        roll = (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        variant = mixture.draw(roll)
        assert training_variant_allowed(
            ring, variant.mode, variant.handicap, variant.pie
        )
        counts[variant.segment] += 1
        if variant.handicap > 1:
            handicap_modes.add((variant.handicap, variant.mode))
    assert counts["handicap"] / 40_000 == pytest.approx(
        2 / 17 if ring == 10 else 0, abs=0.001
    )
    if ring == 10:
        assert handicap_modes == {
            (handicap, mode)
            for handicap in range(2, 10)
            for mode in ("classic", "double")
        }
    assert master.variants.handicap == 0.1
    assert master.variants.pie == 0.9


def test_opt_in_guard_preserves_legacy_game_rules_and_mixtures():
    assert is_pie_training("ring10_pie")
    assert not is_pie_training("ring10_priority")
    legacy = SelfPlayConfig(variants=VariantMixtureConfig(enabled=True))
    assert legacy.variant_mixture_for_ring(4, RING_WEIGHTS) is legacy.variants
    assert legacy.with_variant(GameVariant(handicap=9)).handicap == 9
    assert not legacy.with_variant(GameVariant(mode="classic")).pie
    with pytest.raises(ValueError, match="cannot draw"):
        replace(legacy, pie=True, pie_even_training=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(pie_config(), pie_even_training=1)
