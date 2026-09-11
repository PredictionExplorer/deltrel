"""Pure, bounded analysis of warmed frozen-position search-allocation reports.

The reference is finite-search evidence, not ground-truth Q or playing strength.
Repeats and aliases measure timing only. File/model/config provenance and held-out
selection authority must be verified by the caller before admitting a profile.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

TIMING_SETUP_COUNTERS = (
    "graph_captures",
    "graph_warmup_calls",
    "graph_evictions",
    "graph_fallbacks",
    "graph_validation_failures",
    "graph_validation_replays",
)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _game_key(identifier: Any) -> str:
    if not isinstance(identifier, str) or len(identifier) > 512:
        raise ValueError("invalid position identity")
    pieces = identifier.rsplit("/", 1)
    if len(pieces) != 2 or not pieces[0] or not pieces[1].isdigit():
        raise ValueError("position identity must end in /ply")
    return pieces[0]


def _position_rows(
    record: Mapping[str, Any], ring: int, dead_zone: float
) -> dict[str, dict[str, Any]]:
    positions = record.get("positions")
    if not isinstance(positions, list) or not 1 <= len(positions) <= 512:
        raise ValueError("report requires 1..512 positions per dataset")
    result = {}
    for row in positions:
        identifier = row["id"]
        _game_key(identifier)
        if identifier in result:
            raise ValueError("duplicate position identity")
        cell = row["cell"]
        if (
            not isinstance(cell, (list, tuple))
            or len(cell) != 3
            or cell[0] != ring
            or cell[1]
            not in (
                "standard-classic",
                "standard-double",
                "handicap-classic",
                "handicap-double",
                "pie-classic",
                "pie-double",
            )
            or cell[2] not in ("early", "middle", "late")
        ):
            raise ValueError("invalid ring/mode/phase cell")
        if type(row["pda"]) is not int or not -3 <= row["pda"] <= 3:
            raise ValueError("invalid PDA")
        actions, visits, q, policy = (
            row[name] for name in ("actions", "visits", "q_values", "policy_target")
        )
        nodes = 5 * ring * (ring + 1) // 2
        if (
            not all(isinstance(values, list) for values in (actions, visits, q, policy))
            or not 1 <= len(actions) <= nodes
            or not len(actions) == len(visits) == len(q) == len(policy)
            or len(set(actions)) != len(actions)
            or any(
                type(action) is not int or not 0 <= action < nodes for action in actions
            )
        ):
            raise ValueError("invalid ordered legal action support")
        if any(
            type(count) is not int or not 0 <= count <= 2**32 - 1 for count in visits
        ):
            raise ValueError("invalid visit counts")
        amount = _integer(row["simulations"], "simulations", minimum=1)
        if sum(visits) != amount:
            raise ValueError("fresh visit counts do not match the simulation budget")
        if any(not -1.00001 <= _number(value, "Q value") <= 1.00001 for value in q):
            raise ValueError("Q value outside utility range")
        if any(
            not 0 <= _number(value, "policy probability") <= 1.00001 for value in policy
        ) or not math.isclose(sum(policy), 1, abs_tol=1e-4):
            raise ValueError("invalid policy distribution")
        selected = row["selected_action"]
        if (
            type(selected) is not int
            or selected not in actions
            or visits[actions.index(selected)] == 0
        ):
            raise ValueError("selected action lacks actual search visits")
        value = _number(row["selected_value"], "selected Q")
        if not math.isclose(value, q[actions.index(selected)], rel_tol=0, abs_tol=1e-6):
            raise ValueError("selected value disagrees with selected action Q")
        if type(row["swap"]) is not bool or type(row["swap_available"]) is not bool:
            raise ValueError("swap fields must be boolean")
        if row["swap"] != (row["swap_available"] and value < -dead_zone):
            raise ValueError("swap decision disagrees with the configured dead zone")
        result[identifier] = row
    return result


def _records(
    report: Mapping[str, Any],
    dataset: str,
    ring: int,
    pair: tuple[int, int],
    width: int,
    full: bool,
    repeats: int,
    dead_zone: float,
) -> list[dict[str, Any]]:
    found = [
        row
        for row in report["results"]
        if row.get("stage") == "measured"
        and row.get("dataset") == dataset
        and row.get("ring") == ring
        and tuple(row.get("raw_pair", ())) == pair
        and row.get("first_visit_batch_size") == width
        and row.get("full") is full
    ]
    if len(found) != repeats or {row.get("repeat") for row in found} != set(
        range(repeats)
    ):
        raise ValueError("missing, duplicate or mismatched timing repeats")
    for record in found:
        _integer(record["repeat"], "repeat index")
    found.sort(key=lambda row: row["repeat"])
    first = _position_rows(found[0], ring, dead_zone)
    for record in found:
        if record.get("timing_admissible") is not True:
            raise ValueError("report marked timing inadmissible")
        for name in TIMING_SETUP_COUNTERS:
            if (
                _integer(record["inference"].get(name), name) != 0
                or _integer(record["timing_setup_counters"].get(name), name) != 0
            ):
                raise ValueError(
                    "measured timing contains graph setup, eviction or fallback"
                )
        if _number(record["seconds"], "measured seconds") <= 0:
            raise ValueError("timing duration must be positive")
        _integer(record["requested_rows"], "requested neural rows", minimum=1)
        if _position_rows(record, ring, dead_zone) != first:
            raise ValueError("timing repeats changed the frozen search trace")
    return found


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    known = [row for row in rows if row["loss"] is not None]
    groups: dict[str, list[float]] = defaultdict(list)
    for row in known:
        groups[_game_key(row["id"])].append(row["loss"])
    if not groups:
        return {
            "positions": len(rows),
            "games": 0,
            "assessed_positions": 0,
            "coverage": 0.0,
            "mean_loss": None,
            "upper95": None,
            "loss_gt_0_1": 0,
            "max_loss": None,
        }
    keys = sorted(groups)
    sums = np.array([math.fsum(groups[key]) for key in keys], dtype=np.float64)
    counts = np.array([len(groups[key]) for key in keys], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty(replicates, dtype=np.float64)
    # Bounded transient memory, even at the maximum 512 independent games.
    for start in range(0, replicates, 2048):
        count = min(2048, replicates - start)
        chosen = rng.integers(len(keys), size=(count, len(keys)))
        bootstrap[start : start + count] = sums[chosen].sum(axis=1) / counts[
            chosen
        ].sum(axis=1)
    values = [row["loss"] for row in known]
    return {
        "positions": len(rows),
        "games": len(groups),
        "assessed_positions": len(known),
        "coverage": len(known) / len(rows),
        "mean_loss": math.fsum(values) / len(values),
        "upper95": float(np.quantile(bootstrap, 0.95, method="linear")),
        "loss_gt_0_1": sum(value > 0.1 for value in values),
        "max_loss": max(values),
    }


def _group_analysis(
    reference: Mapping[str, dict[str, Any]],
    baseline: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, dict[str, Any]],
    *,
    seed: int,
    replicates: int,
    require_swaps: bool,
) -> dict[str, Any]:
    if set(reference) != set(baseline) or set(reference) != set(candidate):
        raise ValueError("paired searches do not contain identical positions")
    if require_swaps:
        if any(not row["swap_available"] for row in candidate.values()) or {
            row["cell"][1] for row in candidate.values()
        } != {"pie-classic", "pie-double"}:
            raise ValueError(
                "swap holdout requires actual swap opportunities in both modes"
            )
    else:
        counts = Counter(tuple(row["cell"]) for row in candidate.values())
        if len(counts) != 18 or len(set(counts.values())) != 1:
            raise ValueError(
                "balanced holdout must cover all six modes and three phases equally"
            )
    rows, decisions = [], []
    for identifier in sorted(reference):
        ref, old, new = (
            reference[identifier],
            baseline[identifier],
            candidate[identifier],
        )
        if (
            ref["actions"] != old["actions"]
            or ref["actions"] != new["actions"]
            or any(
                old[key] != new[key] or old[key] != ref[key]
                for key in ("cell", "pda", "swap_available")
            )
        ):
            raise ValueError(
                "paired searches differ in legal actions or position context"
            )
        i, j = (
            ref["actions"].index(old["selected_action"]),
            ref["actions"].index(new["selected_action"]),
        )
        assessed = ref["visits"][i] > 0 and ref["visits"][j] > 0
        rows.append(
            {
                "id": identifier,
                "cell": list(ref["cell"]),
                "pda": ref["pda"],
                "loss": ref["q_values"][i] - ref["q_values"][j] if assessed else None,
                "baseline_action": old["selected_action"],
                "candidate_action": new["selected_action"],
                "reference_visits_at_baseline": ref["visits"][i],
                "reference_visits_at_candidate": ref["visits"][j],
            }
        )
        if new["swap_available"]:
            decisions.append(
                {
                    "id": identifier,
                    "mode": ref["cell"][1],
                    "baseline_swap": old["swap"],
                    "candidate_swap": new["swap"],
                    "reference_swap": ref["swap"],
                    "baseline_keep_value": old["selected_value"],
                    "candidate_keep_value": new["selected_value"],
                    "reference_keep_value": ref["selected_value"],
                }
            )
    failures = sum(
        row["baseline_swap"] == row["reference_swap"]
        and row["candidate_swap"] != row["reference_swap"]
        for row in decisions
    )
    return {
        **_bootstrap(rows, seed=seed, replicates=replicates),
        "rows": rows,
        "swap_available_positions": len(decisions),
        "new_swap_failures": failures,
        "new_reference_swap_disagreements": failures,
        "decision_changes": sum(
            row["baseline_swap"] != row["candidate_swap"] for row in decisions
        ),
        "baseline_reference_disagreements": sum(
            row["baseline_swap"] != row["reference_swap"] for row in decisions
        ),
        "candidate_reference_disagreements": sum(
            row["candidate_swap"] != row["reference_swap"] for row in decisions
        ),
        "swap_decisions": decisions,
    }


def _costs(
    old_full: Sequence[Mapping[str, Any]],
    new_full: Sequence[Mapping[str, Any]],
    old_fast: Sequence[Mapping[str, Any]],
    new_fast: Sequence[Mapping[str, Any]],
    p0: float,
    p1: float,
    same_width: bool,
) -> dict[str, Any]:
    def seconds(records):
        return [row["seconds"] / len(row["positions"]) for row in records]

    full0, full1, fast0, fast1 = map(seconds, (old_full, new_full, old_fast, new_fast))
    same_full = same_width and all(
        left["positions"] == right["positions"]
        for left, right in zip(old_full, new_full, strict=True)
    )
    old_low, new_high = min(fast0), max(fast1)
    if same_full:
        a, b = ((1 - p0) / p0) * old_low, ((1 - p1) / p1) * new_high
        full_bound = max(full0 + full1) if a >= b else min(full0 + full1)
        ratio = (full_bound + a) / (full_bound + b)
        minimum_p = p0 * new_high / ((1 - p0) * old_low + p0 * new_high)
        bound_kind = "shared-full-observed-extrema"
    else:
        old_cost_low = p0 * min(full0) + (1 - p0) * old_low
        new_cost_high = p1 * max(full1) + (1 - p1) * new_high
        ratio = (p1 / new_cost_high) / (p0 / old_cost_low)
        denominator = old_cost_low - p0 * max(full1) + p0 * new_high
        minimum_p = p0 * new_high / denominator if denominator > 0 else None
        if minimum_p is not None and minimum_p > 1:
            minimum_p = None
        bound_kind = "separate-full-observed-extrema"
    baseline_cost = p0 * float(np.median(full0)) + (1 - p0) * float(np.median(fast0))
    candidate_cost = p1 * float(np.median(full1)) + (1 - p1) * float(np.median(fast1))
    return {
        "method": bound_kind,
        "conservative_full_target_rate_ratio": ratio,
        "minimum_full_probability_for_rate_parity": minimum_p,
        "baseline_full_probability": p0,
        "candidate_full_probability": p1,
        "baseline_full_seconds_per_root": full0,
        "candidate_full_seconds_per_root": full1,
        "baseline_fast_seconds_per_root": fast0,
        "candidate_fast_seconds_per_root": fast1,
        "median_baseline_seconds_per_root": baseline_cost,
        "median_candidate_seconds_per_root": candidate_cost,
        "median_position_throughput_ratio": baseline_cost / candidate_cost,
        "median_full_target_rate_ratio": (p1 / candidate_cost) / (p0 / baseline_cost),
        "scope": "conservative over observed timing-repeat ranges; not a statistical throughput confidence bound",
    }


def analyze_search_allocation_report(
    report: Mapping[str, Any],
    *,
    ring: int,
    candidate_fast_simulations: int,
    full_probability: float,
    fast_policy_weight: float,
    first_visit_batch_size: int,
    baseline_full_probability: float,
    baseline_first_visit_batch_size: int = 1,
    swap_dead_zone: float = 0.02,
    bootstrap_seed: int = 17012026,
    bootstrap_replicates: int = 50_000,
    excluded_game_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Recompute quality and warmed-cost evidence without trusting pass summaries.

    Hashes, profile/model binding, and selection provenance are caller concerns.
    A ValueError means structurally invalid evidence; an unmet quality/rate gate
    remains a finite reported measurement for the caller's admission decision.
    """
    plan = report.get("plan", {})
    if report.get("status") != "passed" or plan.get("schema_version") != 2:
        raise ValueError("requires a successful warmed schema-v2 report")
    if type(ring) is not int or ring not in (4, 6, 8, 10):
        raise ValueError("unsupported ring")
    repeats = _integer(plan.get("repeats"), "repeats", minimum=2)
    if (
        repeats > 5
        or not isinstance(report.get("results"), list)
        or not 1 <= len(report["results"]) <= 1000
    ):
        raise ValueError("unbounded report or timing repeat count")
    p0, p1 = (
        _number(baseline_full_probability, "baseline probability"),
        _number(full_probability, "candidate probability"),
    )
    if (
        not 0 < p0 < 1
        or not 0 < p1 < 1
        or not 0 <= _number(fast_policy_weight, "fast policy weight") <= 1
    ):
        raise ValueError("invalid allocation probabilities or weight")
    if not 0 <= _number(swap_dead_zone, "swap dead zone") < 1:
        raise ValueError("invalid swap dead zone")
    _integer(bootstrap_seed, "bootstrap seed")
    if not 1_000 <= _integer(bootstrap_replicates, "bootstrap replicates") <= 100_000:
        raise ValueError("bootstrap replicates outside bounded range")
    _integer(candidate_fast_simulations, "candidate fast cap", minimum=1)
    for width in (first_visit_batch_size, baseline_first_visit_batch_size):
        if type(width) is not int or not 1 <= width <= 64:
            raise ValueError("invalid search width")
    pairs = plan.get("pairs")
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs)
    ):
        raise ValueError("missing raw budget pairs")
    for pair in pairs:
        for cap in pair:
            _integer(cap, "raw cap", minimum=1)
    baseline_pair = tuple(pairs[0])
    candidate_pair = (candidate_fast_simulations, baseline_pair[1])
    if candidate_pair not in [tuple(pair) for pair in pairs]:
        raise ValueError("candidate with unchanged full cap was not measured")
    if candidate_fast_simulations > baseline_pair[0]:
        raise ValueError("allocation evidence requires a non-increasing fast cap")
    groups, selected_records, references = {}, {}, {}
    all_ids: set[str] = set()
    excluded = set(excluded_game_ids)
    for dataset in ("balanced", "swap-probes"):
        refs = [
            row
            for row in report["results"]
            if row.get("stage") == "cold-reference"
            and row.get("dataset") == dataset
            and row.get("ring") == ring
        ]
        if (
            len(refs) != 1
            or tuple(refs[0].get("raw_pair", ())) != baseline_pair
            or refs[0].get("full") is not True
            or refs[0].get("first_visit_batch_size") != 1
        ):
            raise ValueError("missing or ambiguous deep-search reference")
        reference = _position_rows(refs[0], ring, swap_dead_zone)
        references[dataset] = reference
        old = _records(
            report,
            dataset,
            ring,
            baseline_pair,
            baseline_first_visit_batch_size,
            False,
            repeats,
            swap_dead_zone,
        )
        new = _records(
            report,
            dataset,
            ring,
            candidate_pair,
            first_visit_batch_size,
            False,
            repeats,
            swap_dead_zone,
        )
        baseline = _position_rows(old[0], ring, swap_dead_zone)
        candidate = _position_rows(new[0], ring, swap_dead_zone)
        if all_ids & set(reference) or any(
            _game_key(identifier) in excluded for identifier in reference
        ):
            raise ValueError(
                "holdout overlaps prior games or repeats positions across datasets"
            )
        all_ids.update(reference)
        groups[dataset] = _group_analysis(
            reference,
            baseline,
            candidate,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
            require_swaps=dataset == "swap-probes",
        )
        selected_records[dataset] = (old, new)
    old_full = _records(
        report,
        "balanced",
        ring,
        baseline_pair,
        baseline_first_visit_batch_size,
        True,
        repeats,
        swap_dead_zone,
    )
    new_full = _records(
        report,
        "balanced",
        ring,
        candidate_pair,
        first_visit_batch_size,
        True,
        repeats,
        swap_dead_zone,
    )
    old_rows, new_rows = (
        _position_rows(old_full[0], ring, swap_dead_zone),
        _position_rows(new_full[0], ring, swap_dead_zone),
    )
    if set(old_rows) != set(new_rows) or any(
        old_rows[key]["simulations"] != new_rows[key]["simulations"] for key in old_rows
    ):
        raise ValueError("candidate changed actual full-search budgets")
    for identifier in old_rows:
        ref = references["balanced"].get(identifier)
        if ref is None or any(
            row[field] != ref[field]
            for row in (old_rows[identifier], new_rows[identifier])
            for field in ("cell", "pda", "swap_available", "actions", "simulations")
        ):
            raise ValueError("full-search timing context differs from its reference")
    old_fast, new_fast = selected_records["balanced"]
    if set(old_rows) != {row["id"] for row in old_fast[0]["positions"]}:
        raise ValueError("full/fast timing populations differ")
    cost = _costs(
        old_full,
        new_full,
        old_fast,
        new_fast,
        p0,
        p1,
        first_visit_batch_size == baseline_first_visit_batch_size,
    )
    return {
        "schema_version": 1,
        "method": "paired-reference-Q-game-cluster-percentile-v1",
        "bootstrap": {
            "unit": "whole game",
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "generator": "PCG64",
            "upper_quantile": 0.95,
            "quantile_method": "linear",
            "repeated_searches_count_as_new_quality_samples": False,
        },
        "baseline": {
            "raw_fast_simulations": baseline_pair[0],
            "raw_full_simulations": baseline_pair[1],
            "full_probability": p0,
            "first_visit_batch_size": baseline_first_visit_batch_size,
        },
        "candidate": {
            "rings": ring,
            "raw_fast_simulations": candidate_fast_simulations,
            "raw_full_simulations": baseline_pair[1],
            "full_probability": p1,
            "fast_policy_weight": fast_policy_weight,
            "first_visit_batch_size": first_visit_batch_size,
        },
        "full_budget_preserved": True,
        "excluded_games_checked": len(excluded),
        "balanced": groups["balanced"],
        "swaps": groups["swap-probes"],
        "cost": cost,
        "new_swap_failures": sum(
            group["new_swap_failures"] for group in groups.values()
        ),
        "swap_failure_definition": "new candidate/deep-reference swap disagreement where baseline fast agreed; not a proven outcome error",
        "limitations": [
            "Q is a finite-search estimate, not ground truth or an Elo guarantee.",
            "Placement regret and explicit swap decisions are checked separately.",
            "Input provenance and independent selection must be bound by the caller.",
        ],
    }
