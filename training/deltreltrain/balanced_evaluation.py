"""Fixed board/rule objectives for paired evaluation.

Equal-allocation contracts evaluate complete handicap-severity cycles across
every cell. Adaptive promotion evaluates completed precommitted allocations
and maintains separate per-cell prefix guards. Seat-reversed games remain one
paired observation. Immutable objective weights give legacy cells equal weight
and pie-even contracts 90% even games and 10% handicap games.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Mapping, Sequence

from .selfplay import GameVariant
from .contracts import RULES_SCHEMA_ID, RULES_HASH_WIRE, SEARCH_ALGORITHM_ID
from .topology import SUPPORTED_RINGS

if TYPE_CHECKING:
    from .arena import ArenaPair
    from .config import ArenaConfig

BALANCED_OBSERVATION_MODEL = "equal-24-cells-complete-severity-cycle-v1"
BALANCED_CATEGORIES = (
    "classic-standard",
    "double-standard",
    "classic-pie",
    "double-pie",
    "classic-handicap",
    "double-handicap",
)
PIE_EVEN_CATEGORIES = ("classic-pie", "double-pie")
HANDICAP_CATEGORIES = ("classic-handicap", "double-handicap")
HOEFFDING_LAMBDAS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ADAPTIVE_OBSERVATION_MODEL = "adaptive-allocation-stratified-paired-hoeffding-v3"
ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT = 1 / 9


def balanced_opening_seed(
    seed: int, ring: int, variant: GameVariant, pair_index: int
) -> int:
    material = f"balanced-cell-pair-opening-v1:{seed}:{ring}:{variant.label}:{pair_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def balanced_search_seed(opening_seed: int, candidate_player: int, move: int) -> int:
    """A cell/pair/seat/move stream, independent of cohort membership or order."""
    material = (
        f"balanced-root-search-v2:{opening_seed}:{candidate_player}:{move}".encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def cycle_log_e_value(
    scores: Sequence[float],
    *,
    pairs_per_cycle: float,
    null_mean: float,
    direction: str,
) -> float:
    """Hoeffding mixtures checked only at complete fixed-allocation cycles.

    Pair scores are independent across hashed seed streams, bounded in [0,1],
    and may have different means by cell/severity. A full cycle averages those
    means with fixed weights. For pair weights a_i summing to one per cycle,
    the effective pair count is 1/sum(a_i²). Scaling a cycle's weighted score
    by that count makes its squared coefficient sum equal the count, so
    Hoeffding's lemma gives exp(lambda*S-lambda²*N/8) at cycle boundaries.
    Equal allocation reduces to the actual pair count. The two games of each
    pair stay together and are never treated as independent observations.
    """
    if (
        not math.isfinite(pairs_per_cycle)
        or pairs_per_cycle <= 0
        or not 0 <= null_mean <= 1
        or direction not in ("greater", "less")
    ):
        raise ValueError("invalid paired cycle e-process parameters")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("cycle scores must be bounded in [0, 1]")
    count = len(scores) * pairs_per_cycle
    centered = pairs_per_cycle * math.fsum(scores) - count * null_mean
    if direction == "less":
        centered = -centered
    logs = [value * centered - value * value * count / 8 for value in HOEFFDING_LAMBDAS]
    maximum = max(logs)
    return (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in logs))
        - math.log(len(logs))
    )


def cycle_confidence_sequence(
    scores: Sequence[float],
    *,
    pairs_per_cycle: float,
    error_probability: float,
) -> tuple[float, float]:
    if not 0 < error_probability < 1:
        raise ValueError("cycle error probability must be in (0, 1)")
    threshold = math.log(1 / error_probability)
    bounds = []
    for direction, endpoint in (("greater", 0.0), ("less", 1.0)):
        if (
            not scores
            or cycle_log_e_value(
                scores,
                pairs_per_cycle=pairs_per_cycle,
                null_mean=endpoint,
                direction=direction,
            )
            < threshold
        ):
            bounds.append(endpoint)
            continue
        low, high = 0.0, 1.0
        for _ in range(52):
            middle = (low + high) / 2
            exceeds = (
                cycle_log_e_value(
                    scores,
                    pairs_per_cycle=pairs_per_cycle,
                    null_mean=middle,
                    direction=direction,
                )
                >= threshold
            )
            if exceeds == (direction == "greater"):
                low = middle
            else:
                high = middle
        bounds.append(low if direction == "greater" else high)
    return bounds[0], bounds[1]


def stratified_pair_log_e_value(
    scores: Sequence[float],
    *,
    cycle_length: int,
    null_mean: float,
    direction: str,
) -> float:
    """Anytime evidence for an equally weighted deterministic severity cycle.

    Let each severity's stationary mean be mu_s and their average be at most
    mu. In a contiguous prefix n=qL+r, each severity is sampled q or q+1 times.
    Its cumulative expected score is at most qL*mu+min(r,L*mu). Subtracting that
    upper bound gives evidence dominated by the mixture centered at the true severity
    means. Ville's inequality therefore applies at every prefix, including
    unfinished cycles. Inverting X handles the opposite direction. Different
    pairs use independent seed streams; the two reversed games remain one
    bounded observation and may be arbitrarily dependent.

    A fixed candidate/baseline/search contract and absolute contiguous pair
    indices are essential. These per-cell sequences may be sampled adaptively
    and interleaved without imposing a global cycle-completion barrier.
    """
    if (
        type(cycle_length) is not int
        or cycle_length <= 0
        or not 0 <= null_mean <= 1
        or direction not in ("greater", "less")
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)
    ):
        raise ValueError("invalid stratified paired e-process inputs")
    observations = (
        scores if direction == "greater" else tuple(1 - value for value in scores)
    )
    mean = null_mean if direction == "greater" else 1 - null_mean
    quotient, remainder = divmod(len(scores), cycle_length)
    maximum_expected = quotient * cycle_length * mean + min(
        remainder, cycle_length * mean
    )
    centered = math.fsum(observations) - maximum_expected
    logs = [
        value * centered - value * value * len(scores) / 8
        for value in HOEFFDING_LAMBDAS
    ]
    maximum = max(logs)
    return (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in logs))
        - math.log(len(logs))
    )


def stratified_pair_confidence_sequence(
    scores: Sequence[float], *, cycle_length: int, error_probability: float
) -> tuple[float, float]:
    """One-sided-error bounds valid at every contiguous paired prefix."""
    if not 0 < error_probability < 1:
        raise ValueError("stratified pair error probability must be in (0, 1)")
    threshold = math.log(1 / error_probability)
    bounds = []
    for direction, endpoint in (("greater", 0.0), ("less", 1.0)):
        if (
            not scores
            or stratified_pair_log_e_value(
                scores,
                cycle_length=cycle_length,
                null_mean=endpoint,
                direction=direction,
            )
            < threshold
        ):
            bounds.append(endpoint)
            continue
        low, high = 0.0, 1.0
        for _ in range(52):
            middle = (low + high) / 2
            exceeds = (
                stratified_pair_log_e_value(
                    scores,
                    cycle_length=cycle_length,
                    null_mean=middle,
                    direction=direction,
                )
                >= threshold
            )
            if exceeds == (direction == "greater"):
                low = middle
            else:
                high = middle
        bounds.append(low if direction == "greater" else high)
    return bounds[0], bounds[1]


def maximum_weighted_null_expectation(
    weights: Sequence[float], masses: Sequence[float], null_mean: float
) -> float:
    """Maximize cumulative expectation subject to the fixed objective's null.

    Each unknown stratum mean lies in [0,1]. Its contribution consumes objective
    weight w_s and earns coefficient mass A_s, so this bounded linear program
    is exactly fractional knapsack, ordered by A_s/w_s.
    """
    if (
        len(weights) != len(masses)
        or not weights
        or not 0 <= null_mean <= 1
        or any(not math.isfinite(w) or w <= 0 for w in weights)
        or any(not math.isfinite(a) or a < 0 for a in masses)
        or not math.isclose(math.fsum(weights), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("invalid weighted null stratum inputs")
    remaining = null_mean
    contributions = []
    for weight, mass in sorted(
        zip(weights, masses, strict=True),
        key=lambda item: item[1] / item[0],
        reverse=True,
    ):
        amount = min(weight, remaining)
        contributions.append(amount * mass / weight)
        remaining = max(0.0, remaining - amount)
    return math.fsum(contributions)


def weighted_stratum_log_e_value(
    *,
    weighted_score: float,
    squared_coefficients: float,
    stratum_weights: Sequence[float],
    coefficient_masses: Sequence[float],
    null_mean: float,
    direction: str,
) -> float:
    """Composite-null evidence at complete precommitted allocation boundaries.

    Coefficients are fixed by cell and absolute pair index before outcomes are
    observed. For each fixed vector of stationary stratum means, the mixture
    exp(lambda*(S-sum A_s*mu_s)-lambda^2*V/8) is a nonnegative supermartingale
    across predictable, fully observed allocations. Maximizing sum A_s*mu_s
    over the weighted null makes our evidence no larger than that martingale
    under every null vector. Ville's bound remains valid for adaptive plans.
    Async incomplete allocations must never enter this global test: outcome-
    dependent completion order is not a predictable sampling policy.
    """
    total = math.fsum(coefficient_masses)
    if (
        direction not in ("greater", "less")
        or not math.isfinite(weighted_score)
        or not -1e-12 <= weighted_score <= total + 1e-12
        or not math.isfinite(squared_coefficients)
        or squared_coefficients < 0
    ):
        raise ValueError("invalid weighted paired evidence inputs")
    observed = weighted_score if direction == "greater" else total - weighted_score
    mean = null_mean if direction == "greater" else 1 - null_mean
    expected = maximum_weighted_null_expectation(
        stratum_weights, coefficient_masses, mean
    )
    logs = [
        value * (observed - expected) - value * value * squared_coefficients / 8
        for value in HOEFFDING_LAMBDAS
    ]
    maximum = max(logs)
    return (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in logs))
        - math.log(len(logs))
    )


def weighted_stratum_confidence_sequence(
    *,
    weighted_score: float,
    squared_coefficients: float,
    stratum_weights: Sequence[float],
    coefficient_masses: Sequence[float],
    error_probability: float,
) -> tuple[float, float]:
    if not 0 < error_probability < 1:
        raise ValueError("weighted stratum error probability must be in (0, 1)")
    threshold = math.log(1 / error_probability)
    bounds = []
    for direction, endpoint in (("greater", 0.0), ("less", 1.0)):

        def evidence(mean: float) -> float:
            return weighted_stratum_log_e_value(
                weighted_score=weighted_score,
                squared_coefficients=squared_coefficients,
                stratum_weights=stratum_weights,
                coefficient_masses=coefficient_masses,
                null_mean=mean,
                direction=direction,
            )

        if not squared_coefficients or evidence(endpoint) < threshold:
            bounds.append(endpoint)
            continue
        low, high = 0.0, 1.0
        for _ in range(52):
            middle = (low + high) / 2
            if (evidence(middle) >= threshold) == (direction == "greater"):
                low = middle
            else:
                high = middle
        bounds.append(low if direction == "greater" else high)
    return bounds[0], bounds[1]


def category(variant: GameVariant) -> str:
    rule = "pie" if variant.pie else "handicap" if variant.handicap > 1 else "standard"
    return f"{variant.mode}-{rule}"


def cell_key(ring: int, variant: GameVariant) -> str:
    return f"r{ring}/{category(variant)}"


def balanced_cells(config: ArenaConfig) -> tuple[str, ...]:
    return tuple(
        f"r{ring}/{name}"
        for ring in config.rings
        for name in balanced_categories(config, ring)
    )


def balanced_categories(config: ArenaConfig, ring: int) -> tuple[str, ...]:
    """The playable categories on one configured board under this contract."""
    if ring not in config.rings:
        raise ValueError("balanced arena ring is not configured")
    if config.variant_policy == "pie_even":
        return PIE_EVEN_CATEGORIES + (
            HANDICAP_CATEGORIES if ring == max(SUPPORTED_RINGS) else ()
        )
    return BALANCED_CATEGORIES


def balanced_cell_weights(config: ArenaConfig) -> dict[str, float]:
    cells = balanced_cells(config)
    if config.variant_policy == "legacy_six":
        return {key: 1 / len(cells) for key in cells}
    handicap = [key for key in cells if key.endswith("-handicap")]
    even_count = len(cells) - len(handicap)
    return {
        key: 0.1 / len(handicap)
        if key in handicap
        else (0.9 if handicap else 1.0) / even_count
        for key in cells
    }


def effective_pairs_per_cycle(config: ArenaConfig) -> float:
    """Hoeffding information per cycle, accounting for unequal cell weights."""
    length = len(config.handicap_severity_cycle)
    if config.variant_policy == "legacy_six":
        return length * len(balanced_cells(config))
    return length / math.fsum(w * w for w in balanced_cell_weights(config).values())


def balanced_observation_model(config: ArenaConfig) -> str:
    """Keep the original all-board contract while naming subsets truthfully."""
    if config.allocation_policy == "adaptive_pie":
        return ADAPTIVE_OBSERVATION_MODEL
    if config.variant_policy == "pie_even":
        rings = "-".join(str(ring) for ring in config.rings)
        mix = "90-even-10-handicap" if max(SUPPORTED_RINGS) in config.rings else "even"
        return f"pie-{mix}-rings-{rings}-complete-severity-cycle-v2"
    if config.rings == SUPPORTED_RINGS:
        return BALANCED_OBSERVATION_MODEL
    rings = "-".join(str(ring) for ring in config.rings)
    return (
        f"equal-{len(balanced_cells(config))}-cells-rings-{rings}"
        "-complete-severity-cycle-v1"
    )


def cell_variant(name: str, pair_index: int, config: ArenaConfig) -> GameVariant:
    mode, rule = name.split("-", 1)
    severity = config.handicap_severity_cycle
    return GameVariant(
        mode=mode,
        pie=rule == "pie",
        handicap=severity[pair_index % len(severity)] if rule == "handicap" else 1,
    )


def pair_key(pair: ArenaPair) -> tuple[int, str, int]:
    return pair.ring, category(GameVariant.parse(pair.variant)), pair.pair


def evaluation_contract(config: ArenaConfig) -> dict[str, object]:
    """A budget-specific immutable contract; candidate-dependent seeds are separate."""
    contract: dict[str, object] = {
        "schema_version": 1,
        "objective": balanced_observation_model(config),
        "rules_schema": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH_WIRE,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "cells": list(balanced_cells(config)),
        "cell_weight": 1 / len(balanced_cells(config)),
        "handicap_severity_cycle": list(config.handicap_severity_cycle),
        "handicap_pda": list(config.segment_handicap_pda),
        "simulations": config.simulations,
        "max_considered": config.max_considered,
        "c_visit": config.c_visit,
        "c_scale": config.c_scale,
        "unforced_opening_fraction": config.unforced_opening_fraction,
        "swap_dead_zone": config.swap_dead_zone,
        "seed_schedule": "sha256-balanced-cell-pair-opening-v1-root-seat-move-v2",
        "native_root_seed": "explicit-seed-statehash-fixed-tree-index-zero",
        "handicap_schedule": "absolute-pair-index-mod-severity-cycle-v1",
        "observation_unit": "all-cells-complete-role-reversed-severity-cycle",
        "statistical_test": "complete-cycle-paired-hoeffding-mixture-v1",
        "hoeffding_lambdas": list(HOEFFDING_LAMBDAS),
        "pair_independence": "independent seed streams across pairs; arbitrary dependence within each seat reversal",
    }
    if config.variant_policy == "pie_even":
        contract.pop("cell_weight")
        contract.update(
            schema_version=2,
            variant_policy=config.variant_policy,
            cell_weights=balanced_cell_weights(config),
            statistical_test="complete-cycle-weighted-paired-hoeffding-mixture-v2",
            evaluation_allocation="equal-pairs-per-cell-per-severity-cycle",
            effective_pairs_per_cycle=effective_pairs_per_cycle(config),
        )
    if config.allocation_policy == "adaptive_pie":
        from .adaptive_promotion import allocation_contract

        contract.pop("effective_pairs_per_cycle")
        contract.update(
            schema_version=3,
            allocation_policy=config.allocation_policy,
            allocation=allocation_contract(config),
            evaluation_allocation="initial-coverage-then-predictable-adaptive-cell-targets",
            observation_unit="completed-precommitted-allocation-of-role-reversed-pairs",
            statistical_test="adaptive-allocation-composite-null-hoeffding-mixture-v3",
            global_null_bound="weighted-stratum-fractional-knapsack-v1",
            pair_coefficients={
                "pie": 1.0,
                "handicap_initial": ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT,
                "handicap_after_initial": 1.0,
                "initial_pairs_per_cell": max(
                    config.minimum_pairs_per_ring, len(config.handicap_severity_cycle)
                ),
            },
            cell_diagnostic_tail_error_allocation="configured-tail-error-times-cell-weight",
            partial_severity_bound="qLmu-plus-min-r-Lmu-v1",
            statistical_parameters={
                "alpha": config.alpha,
                "beta": config.beta,
                "confidence": config.confidence,
                "null_elo": config.null_elo,
                "alternative_elo": config.alternative_elo,
                "cell_regression_floor_elo": config.cell_regression_floor_elo,
            },
        )
    execution = config.search_execution.contract()
    if execution is not None:
        contract["search_execution"] = execution
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "identity": "sha256-" + hashlib.sha256(encoded).hexdigest()}


def grouped_cell_pairs(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, list[ArenaPair]]:
    grouped: dict[str, list[ArenaPair]] = {key: [] for key in balanced_cells(config)}
    for pair in pairs:
        variant = GameVariant.parse(pair.variant)
        key = cell_key(pair.ring, variant)
        if key not in grouped:
            raise ValueError("balanced arena pair has an unconfigured cell")
        if variant != cell_variant(category(variant), pair.pair, config):
            raise ValueError("balanced arena pair violates the fixed handicap schedule")
        grouped[key].append(pair)
    for values in grouped.values():
        values.sort(key=lambda pair: pair.pair)
        indices = [pair.pair for pair in values]
        if len(indices) != len(set(indices)):
            raise ValueError("balanced cell pair indices must be unique")
    if len({pair.opening_seed for pair in pairs}) != len(pairs):
        raise ValueError("balanced pairs must use distinct seed streams across cells")
    return grouped


def _contiguous_prefix(values: Sequence[ArenaPair]) -> int:
    for expected, pair in enumerate(values):
        if pair.pair != expected:
            return expected
    return len(values)


def completed_counts_by_ring(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[int, int]:
    grouped = grouped_cell_pairs(pairs, config)
    return {
        ring: min(
            _contiguous_prefix(grouped[f"r{ring}/{name}"])
            for name in balanced_categories(config, ring)
        )
        for ring in config.rings
    }


def _elo(probability: float) -> float | None:
    return (
        400 * math.log10(probability / (1 - probability))
        if 0 < probability < 1
        else None
    )


def summarize_balanced_pairs(
    pairs: Sequence[ArenaPair],
    config: ArenaConfig,
    *,
    completed_allocation_targets: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if config.allocation_policy == "adaptive_pie":
        return summarize_adaptive_pairs(
            pairs, config, completed_allocation_targets=completed_allocation_targets
        )
    from .arena import (
        _expected_score,
        _reported_e_value,
    )

    grouped = grouped_cell_pairs(pairs, config)
    cycle_length = len(config.handicap_severity_cycle)
    complete_cycles = min(
        _contiguous_prefix(values) // cycle_length for values in grouped.values()
    )
    included = complete_cycles * cycle_length
    per_cell_scores = {
        key: tuple(
            math.fsum(pair.score_rate for pair in values[start : start + cycle_length])
            / cycle_length
            for start in range(0, included, cycle_length)
        )
        for key, values in grouped.items()
    }
    weights = balanced_cell_weights(config)
    cycle_scores = tuple(
        math.fsum(scores[index] for scores in per_cell_scores.values()) / len(grouped)
        if config.variant_policy == "legacy_six"
        else math.fsum(
            weights[key] * scores[index] for key, scores in per_cell_scores.items()
        )
        for index in range(complete_cycles)
    )
    cell_error = (1 - config.confidence) / (2 * len(grouped))
    per_cell: dict[str, object] = {}
    vetoes = []
    floor_score = _expected_score(config.cell_regression_floor_elo)
    for key, scores in per_cell_scores.items():
        mean = math.fsum(scores) / len(scores) if scores else None
        interval = (
            cycle_confidence_sequence(
                scores, pairs_per_cycle=cycle_length, error_probability=cell_error
            )
            if scores
            else (0.0, 1.0)
        )
        regression_log_e = (
            cycle_log_e_value(
                scores,
                pairs_per_cycle=cycle_length,
                null_mean=floor_score,
                direction="less",
            )
            if scores
            else 0.0
        )
        regressed = regression_log_e >= math.log(1 / cell_error)
        if regressed:
            vetoes.append(key)
        per_cell[key] = {
            "pairs": len(grouped[key]),
            "included_pairs": included,
            "complete_cycles": complete_cycles,
            "score_rate": mean,
            "elo_difference": _elo(mean) if mean is not None else None,
            "status": "missing"
            if not grouped[key]
            else "incomplete"
            if not scores
            else "saturated"
            if mean in (0, 1)
            else "measured",
            "anytime_confidence_sequence": list(interval),
            "anytime_elo_interval": [_elo(bound) for bound in interval],
            "error_probability_per_side": cell_error,
            "floor_elo": config.cell_regression_floor_elo,
            "regression_log_e_value": regression_log_e,
            "regression_status": "regress" if regressed else "not_established",
        }
    if cycle_scores:
        mean = math.fsum(cycle_scores) / complete_cycles
        pairs_per_cycle = effective_pairs_per_cycle(config)
        promotion_e = cycle_log_e_value(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            null_mean=_expected_score(config.null_elo),
            direction="greater",
        )
        rejection_e = cycle_log_e_value(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            null_mean=_expected_score(config.alternative_elo),
            direction="less",
        )
        state = (
            "accept_alternative"
            if promotion_e >= math.log(1 / config.alpha)
            else "accept_null"
            if rejection_e >= math.log(1 / config.beta)
            else "continue"
        )
        lower, _ = cycle_confidence_sequence(
            cycle_scores,
            pairs_per_cycle=pairs_per_cycle,
            error_probability=config.alpha,
        )
        _, upper = cycle_confidence_sequence(
            cycle_scores, pairs_per_cycle=pairs_per_cycle, error_probability=config.beta
        )
    else:
        mean = None
        state, promotion_e, rejection_e = "continue", 0.0, 0.0
        lower, upper = 0.0, 1.0
    minimum_ready = included >= config.minimum_pairs_per_ring
    decision = "continue"
    if minimum_ready:
        if vetoes:
            decision = "reject_ring_regression"
        elif state == "accept_alternative":
            decision = "promote"
        elif state == "accept_null":
            decision = "reject"
    aggregate = {
        "observation_model": balanced_observation_model(config),
        "status": "missing"
        if not pairs
        else "incomplete"
        if not cycle_scores
        else "saturated"
        if mean in (0, 1)
        else "measured",
        "cells": len(grouped),
        "complete_cycles": complete_cycles,
        "pairs_per_cycle": cycle_length * len(grouped),
        "pairs": included * len(grouped),
        "available_pairs": len(pairs),
        "games": included * len(grouped) * 2,
        "score_rate": mean,
        "elo_difference": _elo(mean) if mean is not None else None,
        "anytime_confidence_sequence": [lower, upper],
        "anytime_elo_interval": [_elo(lower), _elo(upper)],
        "cycle_scores": list(cycle_scores),
        "cell_weights": weights,
        "missing_cells": [key for key, values in grouped.items() if not values],
    }
    if config.variant_policy == "pie_even":
        aggregate["effective_pairs_per_cycle"] = effective_pairs_per_cycle(config)
    return {
        "evaluation_contract": evaluation_contract(config),
        "balanced_aggregate": aggregate,
        "aggregate": aggregate,
        "per_cell": per_cell,
        "per_ring": {
            str(ring): {
                "cells": {
                    key: summary
                    for key, summary in per_cell.items()
                    if key.startswith(f"r{ring}/")
                }
            }
            for ring in config.rings
        },
        "promotion": {
            "decision": decision,
            "sequential_state": state,
            "pair_model": balanced_observation_model(config),
            "minimum_ready": minimum_ready,
            "cell_vetoes": vetoes,
            "ring_floors": {},
            "regression_source": "cell" if vetoes else None,
            "confidence_sequence": [lower, upper],
            "statistical_test": {
                "name": "complete-cycle-weighted-paired-hoeffding-mixture-e-process-v2"
                if config.variant_policy == "pie_even"
                else "complete-cycle-paired-hoeffding-mixture-e-process",
                "observation_unit": balanced_observation_model(config),
                "promotion": {
                    "log_e_value": promotion_e,
                    "e_value": _reported_e_value(promotion_e),
                    "threshold": 1 / config.alpha,
                },
                "rejection": {
                    "log_e_value": rejection_e,
                    "e_value": _reported_e_value(rejection_e),
                    "threshold": 1 / config.beta,
                },
                "cell_guard_familywise_error": 1 - config.confidence,
            },
        },
    }


def summarize_adaptive_pairs(
    pairs: Sequence[ArenaPair],
    config: ArenaConfig,
    *,
    completed_allocation_targets: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Test only completed allocation prefixes; keep all cell diagnostics current."""
    from .arena import _expected_score, _reported_e_value
    from .adaptive_promotion import allocation_contract

    grouped = grouped_cell_pairs(pairs, config)
    weights = balanced_cell_weights(config)
    guard_error = (1 - config.confidence) / (2 * len(grouped))
    floor_score = _expected_score(config.cell_regression_floor_elo)
    per_cell: dict[str, dict[str, object]] = {}
    vetoes = []
    lower = upper = 0.0
    point_low = point_high = 0.0
    included_total = 0
    included_counts: list[int] = []
    for key, available in grouped.items():
        included = _contiguous_prefix(available)
        scores = tuple(pair.score_rate for pair in available[:included])
        handicap = key.endswith("-handicap")
        severities = config.handicap_severity_cycle if handicap else (1,)
        length = len(severities)
        strata = {
            str(severity): scores[offset::length]
            for offset, severity in enumerate(severities)
        }
        means = [
            math.fsum(values) / len(values) for values in strata.values() if values
        ]
        mean = math.fsum(means) / length if len(means) == length else None
        cell_point_low = math.fsum(means) / length
        cell_point_high = (math.fsum(means) + length - len(means)) / length
        promotion_error = config.alpha * weights[key]
        rejection_error = config.beta * weights[key]
        intervals = {
            error: stratified_pair_confidence_sequence(
                scores, cycle_length=length, error_probability=error
            )
            for error in {guard_error, promotion_error, rejection_error}
        }
        guard = intervals[guard_error]
        cell_lower, cell_upper = (
            intervals[promotion_error][0],
            intervals[rejection_error][1],
        )
        regression_log_e = stratified_pair_log_e_value(
            scores, cycle_length=length, null_mean=floor_score, direction="less"
        )
        regressed = regression_log_e >= math.log(1 / guard_error)
        if regressed:
            vetoes.append(key)
        lower += weights[key] * cell_lower
        upper += weights[key] * cell_upper
        point_low += weights[key] * cell_point_low
        point_high += weights[key] * cell_point_high
        included_total += included
        included_counts.append(included)
        per_cell[key] = {
            "pairs": len(available),
            "included_pairs": included,
            "score_rate": mean,
            "score_rate_bounds": [cell_point_low, cell_point_high],
            "elo_difference": _elo(mean) if mean is not None else None,
            "status": "missing"
            if not available
            else "incomplete"
            if mean is None
            else "saturated"
            if mean in (0, 1)
            else "measured",
            "anytime_confidence_sequence": list(guard),
            "anytime_elo_interval": [_elo(bound) for bound in guard],
            "aggregate_confidence_sequence": [cell_lower, cell_upper],
            "aggregate_error_probability": {
                "lower": promotion_error,
                "upper": rejection_error,
            },
            "error_probability_per_side": guard_error,
            "floor_elo": config.cell_regression_floor_elo,
            "regression_log_e_value": regression_log_e,
            "regression_status": "regress" if regressed else "not_established",
            "strata": {
                severity: {
                    "pairs": len(values),
                    "score_rate": math.fsum(values) / len(values) if values else None,
                }
                for severity, values in strata.items()
            },
        }
    initial = max(config.minimum_pairs_per_ring, len(config.handicap_severity_cycle))
    minimum_ready = all(count >= initial for count in included_counts)
    diagnostic_interval = [lower, upper]
    targets = (
        {key: 0 for key in grouped}
        if completed_allocation_targets is None
        else dict(completed_allocation_targets)
    )
    if (
        set(targets) != set(grouped)
        or any(type(value) is not int or value < 0 for value in targets.values())
        or any(
            targets[key] > _contiguous_prefix(values) for key, values in grouped.items()
        )
        or sum(targets.values()) > len(grouped) * config.max_pairs_per_ring
        or (any(targets.values()) and min(targets.values()) < initial)
    ):
        raise ValueError(
            "completed allocation targets must prove a complete covered prefix within budget"
        )
    stratum_weights = []
    coefficient_masses = []
    weighted_scores = []
    squared = []
    point_terms = []
    global_strata = {}
    for key, available in grouped.items():
        handicap = key.endswith("-handicap")
        severities = config.handicap_severity_cycle if handicap else (1,)
        for offset, severity in enumerate(severities):
            observed = available[offset : targets[key] : len(severities)]
            coefficients = [
                ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT
                if handicap and pair.pair < initial
                else 1.0
                for pair in observed
            ]
            objective_weight = weights[key] / len(severities)
            mass = math.fsum(coefficients)
            stratum_weights.append(objective_weight)
            coefficient_masses.append(mass)
            weighted_scores.extend(
                a * pair.score_rate
                for a, pair in zip(coefficients, observed, strict=True)
            )
            squared.extend(a * a for a in coefficients)
            stratum_mean = (
                math.fsum(pair.score_rate for pair in observed) / len(observed)
                if observed
                else None
            )
            if stratum_mean is not None:
                point_terms.append(objective_weight * stratum_mean)
            global_strata[f"{key}/{severity}"] = {
                "pairs": len(observed),
                "coefficient_mass": mass,
                "objective_weight": objective_weight,
                "score_rate": stratum_mean,
            }
    boundary_ready = bool(sum(targets.values()))
    mean = math.fsum(point_terms) if boundary_ready else None
    global_inputs = {
        "weighted_score": math.fsum(weighted_scores),
        "squared_coefficients": math.fsum(squared),
        "stratum_weights": stratum_weights,
        "coefficient_masses": coefficient_masses,
    }
    promotion_e = weighted_stratum_log_e_value(
        **global_inputs, null_mean=_expected_score(config.null_elo), direction="greater"
    )
    rejection_e = weighted_stratum_log_e_value(
        **global_inputs,
        null_mean=_expected_score(config.alternative_elo),
        direction="less",
    )
    lower, _ = weighted_stratum_confidence_sequence(
        **global_inputs, error_probability=config.alpha
    )
    _, upper = weighted_stratum_confidence_sequence(
        **global_inputs, error_probability=config.beta
    )
    state = (
        "accept_alternative"
        if boundary_ready and promotion_e >= math.log(1 / config.alpha)
        else "accept_null"
        if boundary_ready and rejection_e >= math.log(1 / config.beta)
        else "continue"
    )
    decision = "continue"
    if minimum_ready:
        if vetoes:
            decision = "reject_ring_regression"
        elif state == "accept_alternative":
            decision = "promote"
        elif state == "accept_null":
            decision = "reject"
    aggregate = {
        "observation_model": ADAPTIVE_OBSERVATION_MODEL,
        "status": "missing"
        if not pairs
        else "incomplete"
        if mean is None
        else "saturated"
        if mean in (0, 1)
        else "measured",
        "cells": len(grouped),
        "pairs": sum(targets.values()),
        "available_pairs": len(pairs),
        "available_contiguous_pairs": included_total,
        "games": 2 * sum(targets.values()),
        "score_rate": mean,
        "score_rate_bounds": [mean, mean] if mean is not None else [0.0, 1.0],
        "elo_difference": _elo(mean) if mean is not None else None,
        "anytime_confidence_sequence": [lower, upper],
        "anytime_elo_interval": [_elo(lower), _elo(upper)],
        "two_sided_error_probability_bound": config.alpha + config.beta,
        "cell_weights": weights,
        "completed_allocation_targets": targets if boundary_ready else None,
        "allocation_boundary_ready": boundary_ready,
        "global_strata": global_strata,
        "weighted_score_sum": global_inputs["weighted_score"],
        "squared_coefficient_sum": global_inputs["squared_coefficients"],
        "diagnostic_cell_union_bounds": diagnostic_interval,
        "confidence_scope": "complete-precommitted-allocation-boundaries-only",
        "missing_cells": [key for key, values in grouped.items() if not values],
    }
    return {
        "evaluation_contract": evaluation_contract(config),
        "balanced_aggregate": aggregate,
        "aggregate": aggregate,
        "per_cell": per_cell,
        "per_ring": {
            str(ring): {
                "cells": {
                    key: value
                    for key, value in per_cell.items()
                    if key.startswith(f"r{ring}/")
                }
            }
            for ring in config.rings
        },
        "allocation": allocation_contract(config),
        "promotion": {
            "decision": decision,
            "sequential_state": state,
            "pair_model": ADAPTIVE_OBSERVATION_MODEL,
            "minimum_ready": minimum_ready,
            "allocation_boundary_ready": boundary_ready,
            "cell_vetoes": vetoes,
            "ring_floors": {},
            "regression_source": "cell" if vetoes else None,
            "confidence_sequence": [lower, upper],
            "statistical_test": {
                "name": "adaptive-allocation-composite-null-hoeffding-mixture-v3",
                "observation_unit": "completed-precommitted-allocation-of-role-reversed-pairs",
                "promotion": {
                    "log_e_value": promotion_e,
                    "e_value": _reported_e_value(promotion_e),
                    "threshold": 1 / config.alpha,
                    "lower_score_bound": lower,
                    "null_score": _expected_score(config.null_elo),
                    "error_probability": config.alpha,
                },
                "rejection": {
                    "log_e_value": rejection_e,
                    "e_value": _reported_e_value(rejection_e),
                    "threshold": 1 / config.beta,
                    "upper_score_bound": upper,
                    "alternative_score": _expected_score(config.alternative_elo),
                    "error_probability": config.beta,
                },
                "cell_guard_familywise_error": 1 - config.confidence,
            },
        },
    }
