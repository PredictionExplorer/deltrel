from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from deltreltrain.arena import ArenaPair, summarize_completed_arena_pairs
from deltreltrain.balanced_evaluation import (
    ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT,
    balanced_categories,
    balanced_opening_seed,
    cell_variant,
    cycle_log_e_value,
    evaluation_contract,
    maximum_weighted_null_expectation,
    stratified_pair_log_e_value,
    summarize_balanced_pairs,
    weighted_stratum_log_e_value,
)
from deltreltrain.config import ArenaConfig


def config():
    return ArenaConfig(
        rings=(10,),
        balanced_cells=True,
        variant_policy="pie_even",
        allocation_policy="adaptive_pie",
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=40,
    )


def pairs(counts, outcome=lambda cell, index: (1, -1)):
    cfg = config()
    result = []
    for category, count in zip(balanced_categories(cfg, 10), counts, strict=True):
        for index in range(count):
            variant = cell_variant(category, index, cfg)
            result.append(
                ArenaPair(
                    ring=10,
                    pair=index,
                    opening_seed=balanced_opening_seed(cfg.seed, 10, variant, index),
                    opening_action=None,
                    forced_opening=False,
                    outcomes=outcome(category, index),
                    variant=variant.label,
                    segment=variant.segment,
                )
            )
    return result


def targets(counts):
    return {
        f"r10/{cell}": count
        for cell, count in zip(balanced_categories(config(), 10), counts, strict=True)
    }


def test_knapsack_matches_independent_exhaustive_vertex_solution():
    rng = np.random.default_rng(18022)
    weights = [0.45, 0.45, 0.05, 0.05]
    for _ in range(20):
        masses = rng.uniform(0, 10, 4).tolist()
        capacity = float(rng.uniform())
        values = [0.0]
        for fractional in range(4):
            others = [index for index in range(4) if index != fractional]
            for binary in itertools.product((0.0, 1.0), repeat=3):
                point = [0.0] * 4
                for index, value in zip(others, binary, strict=True):
                    point[index] = value
                remainder = capacity - sum(
                    w * p for w, p in zip(weights, point, strict=True)
                )
                if remainder < 0:
                    continue
                point[fractional] = min(1.0, remainder / weights[fractional])
                values.append(sum(a * p for a, p in zip(masses, point, strict=True)))
        assert maximum_weighted_null_expectation(
            weights, masses, capacity
        ) == pytest.approx(max(values))


def test_partial_severity_bound_is_dominated_by_fixed_null_mixture():
    means = [0.9, 0.1, 0.8, 0.2]
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5]
    for count in range(1, len(scores) + 1):
        true_expected = sum(means[index % 4] for index in range(count))
        logs = [
            lam * (sum(scores[:count]) - true_expected) - lam * lam * count / 8
            for lam in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        ]
        direct = (
            max(logs)
            + math.log(sum(math.exp(value - max(logs)) for value in logs))
            - math.log(8)
        )
        assert (
            stratified_pair_log_e_value(
                scores[:count], cycle_length=4, null_mean=0.5, direction="greater"
            )
            <= direct + 1e-12
        )
    for direction in ("greater", "less"):
        assert stratified_pair_log_e_value(
            scores[:4], cycle_length=4, null_mean=0.45, direction=direction
        ) == pytest.approx(
            cycle_log_e_value(
                [sum(scores[:4]) / 4],
                pairs_per_cycle=4,
                null_mean=0.45,
                direction=direction,
            )
        )


def test_global_reflection_and_initial_information_match_paired_design():
    coefficient = ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT
    weights = [0.45, 0.45] + [0.0125] * 8
    masses = [4.0, 4.0] + [coefficient] * 8
    squared = 8 + 8 * coefficient * coefficient
    assert sum(masses) ** 2 / squared == pytest.approx(4 / 0.41)
    kwargs = dict(
        weighted_score=5.0,
        squared_coefficients=squared,
        stratum_weights=weights,
        coefficient_masses=masses,
    )
    assert weighted_stratum_log_e_value(
        **kwargs, null_mean=0.6, direction="less"
    ) == pytest.approx(
        weighted_stratum_log_e_value(
            **{**kwargs, "weighted_score": sum(masses) - 5},
            null_mean=0.4,
            direction="greater",
        )
    )


def test_no_global_boundary_means_no_global_promotion_even_with_many_wins():
    cfg = config()
    evidence = pairs((40, 40, 4, 4), lambda *_: (1, 1))
    result = summarize_balanced_pairs(evidence, cfg)
    assert result["promotion"]["decision"] == "continue"
    assert result["aggregate"]["pairs"] == 0
    assert result["aggregate"]["anytime_confidence_sequence"] == [0.0, 1.0]
    assert result["per_cell"]["r10/classic-pie"]["included_pairs"] == 40
    complete = summarize_completed_arena_pairs(
        evidence, cfg, completed_allocation_targets=targets((40, 40, 4, 4))
    )
    assert complete["promotion"]["decision"] == "promote"


def test_partial_new_plan_preserves_last_completed_global_boundary():
    cfg = config()
    original = pairs((4, 4, 4, 4))
    prior = summarize_balanced_pairs(
        original, cfg, completed_allocation_targets=targets((4, 4, 4, 4))
    )
    later = pairs((13, 13, 5, 5), lambda _, index: (1, -1) if index < 4 else (1, 1))
    current = summarize_balanced_pairs(
        later, cfg, completed_allocation_targets=targets((4, 4, 4, 4))
    )
    for key in ("score_rate", "anytime_confidence_sequence", "pairs"):
        assert current["aggregate"][key] == prior["aggregate"][key]
    finished = summarize_balanced_pairs(
        later, cfg, completed_allocation_targets=targets((13, 13, 5, 5))
    )
    assert finished["aggregate"]["pairs"] == 36
    assert finished["aggregate"]["score_rate"] > prior["aggregate"]["score_rate"]
    assert finished["per_cell"]["r10/classic-handicap"]["included_pairs"] == 5


def test_partial_severity_point_estimate_is_uniform_across_severities():
    evidence = pairs(
        (4, 4, 5, 5),
        lambda cell, index: (
            (1, 1) if cell.endswith("handicap") and index % 4 == 0 else (-1, -1)
        ),
    )
    result = summarize_balanced_pairs(
        evidence, config(), completed_allocation_targets=targets((4, 4, 5, 5))
    )
    assert result["per_cell"]["r10/classic-handicap"]["score_rate"] == 0.25
    assert result["aggregate"]["score_rate"] == pytest.approx(0.025)


@pytest.mark.parametrize("requested", [(4, 4, 4, 3), (4, 4, 5, 4), (100, 100, 4, 4)])
def test_invalid_unfinished_or_overbudget_boundaries_are_rejected(requested):
    with pytest.raises(ValueError, match="completed allocation targets"):
        summarize_balanced_pairs(
            pairs((4, 4, 4, 4)),
            config(),
            completed_allocation_targets=targets(requested),
        )


def test_sparse_pair_hole_cannot_be_hidden_by_a_claimed_completed_boundary():
    evidence = pairs((4, 4, 4, 4))
    missing = [
        pair
        for pair in evidence
        if not (pair.variant == "pie-classic" and pair.pair == 2)
    ]
    with pytest.raises(ValueError, match="completed allocation targets"):
        summarize_balanced_pairs(
            missing, config(), completed_allocation_targets=targets((4, 4, 4, 4))
        )


def test_cell_regression_guard_remains_actionable_without_global_future_data():
    evidence = pairs(
        (4, 4, 40, 4),
        lambda cell, _: (-1, -1) if cell == "classic-handicap" else (1, -1),
    )
    result = summarize_balanced_pairs(evidence, config())
    assert result["promotion"]["decision"] == "reject_ring_regression"
    assert result["promotion"]["cell_vetoes"] == ["r10/classic-handicap"]


def test_equal_contract_is_preserved_and_adaptive_contract_pins_statistics():
    adaptive = config()
    old = evaluation_contract(replace(adaptive, allocation_policy="equal_cells"))
    new = evaluation_contract(adaptive)
    assert old["schema_version"] == 2 and "allocation_policy" not in old
    assert new["schema_version"] == 3 and new["identity"] != old["identity"]
    assert new["pair_coefficients"]["handicap_initial"] == 1 / 9
    assert "precommitted" in new["observation_unit"]
