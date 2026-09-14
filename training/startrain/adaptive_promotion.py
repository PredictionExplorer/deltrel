"""Bounded, resumable pair allocation for the pie-heavy promotion screen.

Allocation changes which independent streams receive work, never their objective
weights. A persisted plan is finished before another ordinary allocation starts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arena import ArenaPair
    from .config import ArenaConfig


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("adaptive allocation count must be an integer")
    return value


def allocation_contract(config: ArenaConfig) -> dict[str, object]:
    from .balanced_evaluation import balanced_cells

    budget = len(balanced_cells(config)) * config.max_pairs_per_ring
    return {
        "schema_version": 1,
        "policy": "initial-coverage-then-pie-heavy-cost-aware-v1",
        "initial_pairs_per_cell": max(
            config.minimum_pairs_per_ring, len(config.handicap_severity_cycle)
        ),
        "continuation_pairs_per_pie_cell": 9,
        "continuation_pairs_per_handicap_cell": 1,
        "maximum_total_pairs": budget,
        "maximum_extra_handicap_pairs": budget // 2,
        "maximum_extra_cycles_per_allocation": 1,
        "handicap_cost_proxy": "two-seats-times-simulations-times-mean-(1+2^pda)/2",
        "uncertainty_priority": "decision-relevant-weighted-width-reduction-per-cost",
        "suspected_regression": "point-below-floor-and-interval-straddles-floor",
        "promotion_review": "finish-reserved-suspected-regression-cycles-within-extra-cap",
        "review_phase_persistence": "continue-suspected-checks-after-global-evidence-dips-until-resolved-or-bounded",
        "global_decision_boundary": "all-pairs-in-committed-allocation-complete",
        "tail_allocation": "largest-remainder-9-9-1-1-stable-cell-order",
    }


def cell_pair_prefixes(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, int]:
    from .balanced_evaluation import grouped_cell_pairs

    prefixes = {}
    for cell, values in grouped_cell_pairs(pairs, config).items():
        indices = {pair.pair for pair in values}
        prefix = 0
        while prefix in indices:
            prefix += 1
        prefixes[cell] = prefix
    return prefixes


def plan_complete(
    plan: Mapping[str, object], pairs: Sequence[ArenaPair], config: ArenaConfig
) -> bool:
    targets = plan_targets(plan, config)
    prefixes = cell_pair_prefixes(pairs, config)
    return all(prefixes[cell] >= target for cell, target in targets.items())


def plan_targets(plan: Mapping[str, object], config: ArenaConfig) -> dict[str, int]:
    from .balanced_evaluation import balanced_cells

    targets = plan.get("cell_pair_targets")
    if (
        not isinstance(targets, dict)
        or set(targets) != set(balanced_cells(config))
        or any(type(value) is not int or value < 0 for value in targets.values())
        or sum(targets.values())
        > _integer(allocation_contract(config)["maximum_total_pairs"])
    ):
        raise ValueError("adaptive allocation has invalid cell targets")
    return dict(targets)


def _expected_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def _cell_search_cost(cell: str, config: ArenaConfig) -> float:
    if cell.endswith("-pie"):
        return 2.0 * config.simulations
    factors = [
        (1.0 + 2.0 ** config.segment_handicap_pda[severity - 2]) / 2.0
        for severity in config.handicap_severity_cycle
    ]
    return 2.0 * config.simulations * math.fsum(factors) / len(factors)


def select_handicap_check(
    config: ArenaConfig,
    summary: Mapping[str, object],
    *,
    extra_pairs_used: int,
    remaining_pairs: int,
    suspected_only: bool = False,
) -> dict[str, object] | None:
    """Choose at most one affordable, decision-relevant full severity cycle."""
    from .balanced_evaluation import balanced_cell_weights

    cycle = len(config.handicap_severity_cycle)
    cap = _integer(allocation_contract(config)["maximum_extra_handicap_pairs"])
    if remaining_pairs < cycle or extra_pairs_used + cycle > cap:
        return None
    per_cell = summary.get("per_cell")
    aggregate = summary.get("balanced_aggregate")
    if not isinstance(per_cell, Mapping) or not isinstance(aggregate, Mapping):
        return None
    bounds = aggregate.get("anytime_confidence_sequence", [0.0, 1.0])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    lower, upper = map(float, bounds)
    weights = balanced_cell_weights(config)
    null, alternative = (
        _expected_score(config.null_elo),
        _expected_score(config.alternative_elo),
    )
    candidates = []
    pie_utilities = []
    for cell, weight in weights.items():
        details = per_cell.get(cell)
        if not isinstance(details, Mapping):
            continue
        guard_interval = details.get("anytime_confidence_sequence")
        interval = details.get("aggregate_confidence_sequence", guard_interval)
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or not isinstance(guard_interval, (list, tuple))
            or len(guard_interval) != 2
        ):
            continue
        lo, hi = map(float, interval)
        width = max(0.0, hi - lo)
        count = int(details.get("included_pairs", details.get("pairs", 0)))
        increment = 1 if cell.endswith("-pie") else cycle
        reduction = weight * width * (1.0 - math.sqrt(count / (count + increment)))
        cost = increment * _cell_search_cost(cell, config)
        utility = reduction / cost
        if cell.endswith("-pie"):
            pie_utilities.append(utility)
            continue
        mean = details.get("score_rate")
        floor = _expected_score(config.cell_regression_floor_elo)
        guard_lower, guard_upper = map(float, guard_interval)
        suspected = (
            mean is not None
            and float(mean) < floor
            and guard_lower <= floor <= guard_upper
        )
        relevant = (
            lower <= null < lower + weight * width
            or upper - weight * width < alternative <= upper
        )
        candidates.append((suspected, utility, cell, relevant, cost, reduction))
    best_pie = max(pie_utilities, default=math.inf)
    eligible = [
        item
        for item in candidates
        if item[0] or (not suspected_only and item[3] and item[1] >= best_pie)
    ]
    if not eligible:
        return None
    suspected, utility, cell, _, cost, reduction = sorted(
        eligible, key=lambda item: (-int(item[0]), -item[1], item[2])
    )[0]
    return {
        "cell": cell,
        "pairs": cycle,
        "reason": "suspected_regression"
        if suspected
        else "decision_relevant_uncertainty",
        "expected_weighted_width_reduction": reduction,
        "estimated_search_cost": cost,
        "width_reduction_per_cost": utility,
        "best_pie_width_reduction_per_cost": best_pie,
    }


def _tail_counts(cells: Sequence[str], budget: int) -> dict[str, int]:
    weights = {cell: 9 if cell.endswith("-pie") else 1 for cell in cells}
    total = sum(weights.values())
    requested = min(budget, total)
    exact = {cell: requested * weights[cell] / total for cell in cells}
    counts = {cell: math.floor(value) for cell, value in exact.items()}
    order = sorted(cells, key=lambda cell: (-(exact[cell] - counts[cell]), cell))
    for cell in order[: requested - sum(counts.values())]:
        counts[cell] += 1
    return counts


def next_allocation(
    config: ArenaConfig,
    *,
    previous_plan: Mapping[str, object] | None,
    summary: Mapping[str, object],
    review_only: bool = False,
) -> dict[str, object] | None:
    """Plan after coverage, or extend existing targets for a bounded review.

    The caller must persist this result before dispatching any game. Review
    extensions retain every existing target, including all unfinished seats.
    """
    from .balanced_evaluation import balanced_cells

    contract = allocation_contract(config)
    cells = balanced_cells(config)
    budget = _integer(contract["maximum_total_pairs"])
    if previous_plan is None:
        if review_only:
            return None
        initial = _integer(contract["initial_pairs_per_cell"])
        if initial * len(cells) > budget:
            raise ValueError("adaptive allocation budget cannot cover initial cells")
        return {
            "phase": "initial",
            "cell_pair_targets": dict.fromkeys(cells, initial),
            "extra_handicap_pairs_total": 0,
            "extra_handicap_check": None,
        }
    targets = plan_targets(previous_plan, config)
    extra_used = _integer(previous_plan["extra_handicap_pairs_total"])
    remaining = budget - sum(targets.values())
    if remaining <= 0:
        return None
    increments = (
        dict.fromkeys(cells, 0) if review_only else _tail_counts(cells, remaining)
    )
    check = select_handicap_check(
        config,
        summary,
        extra_pairs_used=extra_used,
        remaining_pairs=remaining - sum(increments.values()),
        suspected_only=review_only,
    )
    if check is not None:
        increments[str(check["cell"])] += _integer(check["pairs"])
        extra_used += _integer(check["pairs"])
    if not any(increments.values()):
        return None
    return {
        "phase": "handicap_review" if review_only else "continuation",
        "cell_pair_targets": {cell: targets[cell] + increments[cell] for cell in cells},
        "extra_handicap_pairs_total": extra_used,
        "extra_handicap_check": check,
    }


def pending_suspected_review(
    plan: Mapping[str, object], pairs: Sequence[ArenaPair], config: ArenaConfig
) -> bool:
    check = plan.get("extra_handicap_check")
    if not isinstance(check, Mapping) or check.get("reason") != "suspected_regression":
        return False
    cell = str(check["cell"])
    return cell_pair_prefixes(pairs, config)[cell] < plan_targets(plan, config)[cell]


def allocation_metrics(
    pairs: Sequence[ArenaPair], config: ArenaConfig
) -> dict[str, object]:
    from .balanced_evaluation import grouped_cell_pairs

    counts = {
        cell: len(values) for cell, values in grouped_cell_pairs(pairs, config).items()
    }
    total = sum(counts.values())
    pie = sum(count for cell, count in counts.items() if cell.endswith("-pie"))
    return {
        "allocation_policy": "adaptive_pie",
        "completed_pairs_by_cell": counts,
        "completed_pairs": total,
        "completed_games": 2 * total,
        "actual_pie_pair_share": pie / total if total else None,
        "actual_handicap_pair_share": (total - pie) / total if total else None,
        "maximum_total_pairs": allocation_contract(config)["maximum_total_pairs"],
        "objective_weights_are_sampling_shares": False,
    }
