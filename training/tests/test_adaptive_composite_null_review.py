"""Independent checks of the composite-null bound and completed-plan boundary."""

from dataclasses import replace
from itertools import product
import math
import random

import pytest

from startrain.arena import summarize_completed_arena_pairs
from startrain.balanced_evaluation import (
    HOEFFDING_LAMBDAS,
    balanced_cells,
    maximum_weighted_null_expectation,
    weighted_stratum_log_e_value,
)
from test_adaptive_promotion import config, pairs_for


def vertex_maximum(weights, masses, threshold):
    """Enumerate LP vertices: all but at most one coordinate must be 0 or 1."""
    candidates = []
    for fixed in product((0.0, 1.0), repeat=len(weights)):
        if math.fsum(w * x for w, x in zip(weights, fixed)) <= threshold + 1e-14:
            candidates.append(math.fsum(a * x for a, x in zip(masses, fixed)))
        for free in range(len(weights)):
            remaining = threshold - math.fsum(
                weights[index] * fixed[index]
                for index in range(len(weights))
                if index != free
            )
            fraction = remaining / weights[free]
            if 0 <= fraction <= 1:
                point = list(fixed)
                point[free] = fraction
                candidates.append(math.fsum(a * x for a, x in zip(masses, point)))
    return max(candidates)


def test_fractional_knapsack_matches_independent_vertex_enumeration():
    rng = random.Random(17012026)
    for _ in range(100):
        raw = [rng.uniform(0.1, 1) for _ in range(4)]
        weights = [value / sum(raw) for value in raw]
        masses = [rng.uniform(0, 100) for _ in weights]
        for threshold in (0.0, rng.random(), 1.0):
            assert maximum_weighted_null_expectation(
                weights, masses, threshold
            ) == pytest.approx(vertex_maximum(weights, masses, threshold), abs=1e-10)


@pytest.mark.parametrize("direction", ["greater", "less"])
def test_composite_evidence_is_dominated_by_every_fixed_true_null_process(direction):
    rng = random.Random(17012026)
    for _ in range(100):
        weights = [0.45, 0.45, 0.05, 0.05]
        masses = [rng.uniform(0, 100) for _ in weights]
        means = [rng.random() for _ in weights]
        threshold = math.fsum(w * mu for w, mu in zip(weights, means))
        score = rng.random() * sum(masses)
        squared = rng.uniform(0.1, sum(masses))
        observed = score if direction == "greater" else sum(masses) - score
        true_expectation = math.fsum(
            a * (mu if direction == "greater" else 1 - mu)
            for a, mu in zip(masses, means)
        )
        logs = [
            lam * (observed - true_expectation) - lam * lam * squared / 8
            for lam in HOEFFDING_LAMBDAS
        ]
        top = max(logs)
        fixed_null_log_e = top + math.log(
            math.fsum(math.exp(value - top) for value in logs) / len(logs)
        )
        tested = weighted_stratum_log_e_value(
            weighted_score=score,
            squared_coefficients=squared,
            stratum_weights=weights,
            coefficient_masses=masses,
            null_mean=threshold,
            direction=direction,
        )
        assert tested <= fixed_null_log_e + 1e-10


def test_uncommitted_completion_outcomes_do_not_enter_the_global_statistic():
    cfg = config()
    boundary = dict.fromkeys(balanced_cells(cfg), 4)
    latest = {
        "r10/classic-pie": 13,
        "r10/double-pie": 13,
        "r10/classic-handicap": 5,
        "r10/double-handicap": 5,
    }
    original = pairs_for(cfg, latest)
    favorable = [
        replace(pair, outcomes=(1, 1)) if pair.pair >= 4 else pair for pair in original
    ]
    unfavorable = [
        replace(pair, outcomes=(-1, -1)) if pair.pair >= 4 else pair
        for pair in original
    ]
    results = [
        summarize_completed_arena_pairs(
            pairs, cfg, completed_allocation_targets=boundary
        )
        for pairs in (favorable, list(reversed(unfavorable)))
    ]
    for field in (
        "weighted_score_sum",
        "squared_coefficient_sum",
        "score_rate",
        "anytime_confidence_sequence",
        "completed_allocation_targets",
        "global_strata",
    ):
        assert (
            results[0]["balanced_aggregate"][field]
            == results[1]["balanced_aggregate"][field]
        )
    assert (
        results[0]["per_cell"]["r10/classic-pie"]["score_rate"]
        != results[1]["per_cell"]["r10/classic-pie"]["score_rate"]
    )
