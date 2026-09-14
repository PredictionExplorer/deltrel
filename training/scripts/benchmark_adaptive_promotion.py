#!/usr/bin/env python3
"""Bounded synthetic paired-evaluation power/cost check; never plays real games."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from startrain.adaptive_promotion import _tail_counts, allocation_contract
from startrain.config import ArenaConfig
from startrain.balanced_evaluation import (
    ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT,
    cycle_log_e_value,
    stratified_pair_log_e_value,
    weighted_stratum_log_e_value,
)


CELLS = (
    "r10/classic-pie",
    "r10/double-pie",
    "r10/classic-handicap",
    "r10/double-handicap",
)
WEIGHTS = (0.45, 0.45, 0.05, 0.05)
THRESHOLD = math.log(20)
GUARD_THRESHOLD = math.log(160)
ALTERNATIVE = 1 / (1 + 10 ** (-35 / 400))
FLOOR = 1 / (1 + 10 ** (100 / 400))
HANDICAP_COST = np.array([1.5, 2.5, 2.5, 4.5])


def observations(rng, pie_probability, handicap_probabilities, *, neutral_handicap):
    values = np.zeros((4, 160), dtype=float)
    for cell in range(4):
        if cell < 2:
            probabilities = pie_probability[cell]
        else:
            probabilities = np.take(
                handicap_probabilities[cell - 2], np.arange(160) % 4
            )
        values[cell] = rng.random(160) < probabilities
    if neutral_handicap:
        values[2:] = 0.5
    return values


def evidence(values, counts, adaptive):
    cells = [values[cell, :count] for cell, count in enumerate(counts)]
    means = [
        float(cell.mean())
        if index < 2
        else float(np.mean([cell[offset::4].mean() for offset in range(4)]))
        for index, cell in enumerate(cells)
    ]
    guards = [
        stratified_pair_log_e_value(
            cell, cycle_length=1 if index < 2 else 4, null_mean=FLOOR, direction="less"
        )
        for index, cell in enumerate(cells)
    ]
    if adaptive:
        weights, masses = [], []
        score, squared = 0.0, 0.0
        for index, cell in enumerate(cells):
            coefficient = np.ones(len(cell))
            if index >= 2:
                coefficient[:4] = ADAPTIVE_INITIAL_HANDICAP_COEFFICIENT
            score += float(np.dot(coefficient, cell))
            squared += float(np.dot(coefficient, coefficient))
            length = 1 if index < 2 else 4
            for offset in range(length):
                weights.append(WEIGHTS[index] / length)
                masses.append(float(coefficient[offset::length].sum()))
        promotion = weighted_stratum_log_e_value(
            weighted_score=score,
            squared_coefficients=squared,
            stratum_weights=weights,
            coefficient_masses=masses,
            null_mean=0.5,
            direction="greater",
        )
        rejection = weighted_stratum_log_e_value(
            weighted_score=score,
            squared_coefficients=squared,
            stratum_weights=weights,
            coefficient_masses=masses,
            null_mean=ALTERNATIVE,
            direction="less",
        )
    else:
        cycles = [
            sum(
                WEIGHTS[cell] * float(values[cell, start : start + 4].mean())
                for cell in range(4)
            )
            for start in range(0, counts[0], 4)
        ]
        promotion = cycle_log_e_value(
            cycles, pairs_per_cycle=4 / 0.41, null_mean=0.5, direction="greater"
        )
        rejection = cycle_log_e_value(
            cycles, pairs_per_cycle=4 / 0.41, null_mean=ALTERNATIVE, direction="less"
        )
    return promotion, rejection, guards, means


def trial(values, *, adaptive, extra_cap, stress=False):
    counts = [4, 4, 4, 4]
    extra = 0
    for _ in range(41):
        promotion, rejection, guards, means = evidence(values, counts, adaptive)
        remaining = 160 - sum(counts)
        suspects = [
            cell
            for cell in (2, 3)
            if means[cell] < FLOOR and guards[cell] < GUARD_THRESHOLD
        ]
        review = (
            min(suspects, key=lambda cell: (means[cell], cell)) if suspects else None
        )
        if any(value >= GUARD_THRESHOLD for value in guards):
            decision = "cell_regression"
            break
        if promotion >= THRESHOLD:
            if (
                adaptive
                and review is not None
                and extra + 4 <= extra_cap
                and remaining >= 4
            ):
                counts[review] += 4
                extra += 4
                continue
            decision = "promote"
            break
        if rejection >= THRESHOLD:
            decision = "reject"
            break
        if remaining <= 0:
            decision = "inconclusive"
            break
        if not adaptive:
            counts = [count + 4 for count in counts]
            continue
        additions = _tail_counts(CELLS, remaining)
        remaining -= sum(additions.values())
        if stress:
            # Deliberately outcome-dependent null stress, not the production
            # cost heuristic: favor the currently best observed handicap mode.
            review = max((2, 3), key=lambda cell: (means[cell], -cell))
        if review is not None and extra + 4 <= extra_cap and remaining >= 4:
            additions[CELLS[review]] += 4
            extra += 4
        counts = [
            count + additions[cell] for cell, count in zip(CELLS, counts, strict=True)
        ]
    else:
        raise RuntimeError("bounded simulation exhausted its allocation limit")
    cost = (
        counts[0]
        + counts[1]
        + sum(
            float(HANDICAP_COST[np.arange(counts[cell]) % 4].sum()) for cell in (2, 3)
        )
    )
    return {
        "decision": decision,
        "pairs": sum(counts),
        "games": 2 * sum(counts),
        "search_cost_proxy": cost,
        "extra_handicap_pairs": extra,
    }


def run_benchmark(*, trials=256, null_trials=1000, seed=71341):
    if not 1 <= trials <= 5000 or not 1 <= null_trials <= 10000:
        raise ValueError("benchmark trial counts exceed their bounded range")
    rng = np.random.default_rng(seed)
    cfg = ArenaConfig(
        rings=(10,),
        balanced_cells=True,
        variant_policy="pie_even",
        allocation_policy="adaptive_pie",
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=40,
    )
    extra_cap = allocation_contract(cfg)["maximum_extra_handicap_pairs"]
    if type(extra_cap) is not int:
        raise ValueError("allocation contract has an invalid extra-pair cap")
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "startrain/balanced_evaluation.py",
        root / "startrain/adaptive_promotion.py",
    ]
    source_sha256 = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    scenarios = [
        (
            "heterogeneous_null",
            null_trials,
            (0.55, 0.45),
            ((0.95, 0.05, 0.8, 0.2), (0.05, 0.95, 0.2, 0.8)),
            False,
            True,
        ),
        (
            "modest_pie_gain",
            trials,
            (0.65, 0.65),
            ((0.5,) * 4, (0.5,) * 4),
            True,
            False,
        ),
        (
            "decisive_pie_gain",
            trials,
            (0.75, 0.75),
            ((0.5,) * 4, (0.5,) * 4),
            True,
            False,
        ),
    ]
    output = []
    started = time.perf_counter()
    for name, count, pie, handicaps, neutral, stress in scenarios:
        results = {False: [], True: []}
        for _ in range(count):
            values = observations(rng, pie, handicaps, neutral_handicap=neutral)
            for adaptive in results:
                results[adaptive].append(
                    trial(
                        values,
                        adaptive=adaptive,
                        extra_cap=extra_cap,
                        stress=stress and adaptive,
                    )
                )
        summaries = {}
        for adaptive, rows in results.items():
            summaries["adaptive" if adaptive else "equal_cells"] = {
                "trials": count,
                "decision_rates": {
                    decision: sum(row["decision"] == decision for row in rows) / count
                    for decision in (
                        "promote",
                        "reject",
                        "cell_regression",
                        "inconclusive",
                    )
                },
                **{
                    f"mean_{key}": float(np.mean([row[key] for row in rows]))
                    for key in (
                        "pairs",
                        "games",
                        "search_cost_proxy",
                        "extra_handicap_pairs",
                    )
                },
            }
        output.append(
            {
                "scenario": name,
                "pie_pair_means": pie,
                "handicap_stratum_means": handicaps,
                "neutral_handicap_pairs_constant_half": neutral,
                "outcome_dependent_null_stress": stress,
                **summaries,
            }
        )
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest()
        != source_sha256[str(path.relative_to(root))]
        for path in sources
    ):
        raise RuntimeError("benchmark sources changed during execution")
    return {
        "report": "adaptive-paired-promotion-synthetic-power-cost",
        "schema_version": 1,
        "seed": seed,
        "runtime": platform.python_version(),
        "simulation_wall_seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "revision_authority": "source-file-digests; no claim that current HEAD contains these edits",
        "budget": {
            "maximum_pairs": 160,
            "initial_pairs_per_cell": 4,
            "base_continuation": [9, 9, 1, 1],
            "extra_handicap_cap": extra_cap,
        },
        "scope": "controlled synthetic allocation comparison; reviews use suspected-regression points; full production sticky-review and cost-relevance selector not simulated",
        "cost_model": "pair-equivalent search proxy: pie1; handicap severities2/4/6/9 cost1.5/2.5/2.5/4.5 from frozen PDA budgets; excludes game lengths and inference batching",
        "limitations": [
            "No measured GPU time, total game simulations, live promotion rate, or Elo/hour is claimed.",
            "Each synthetic Bernoulli represents a maximally correlated role-reversed pair; neutral handicap scenarios use deterministic paired score0.5.",
            "Null stress allocates extra handicap work from completed prior outcomes; all decisions use complete committed allocation boundaries.",
        ],
        "results": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=71341)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        trials=args.trials, null_trials=args.null_trials, seed=args.seed
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
