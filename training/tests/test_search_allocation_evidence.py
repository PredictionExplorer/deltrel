"""Fail-closed paired search evidence without repeat/alias pseudoreplication."""

from copy import deepcopy

import pytest

from startrain.contracts import SEARCH_ALGORITHM_ID
from startrain.search_allocation_evidence import (
    analyze_search_allocation_report,
    TIMING_SETUP_COUNTERS,
)


def valid_report(*, candidate_fast_simulations=8, loss=0.01, width8_full_scale=1.0):
    """Reusable internally valid raw schema-v2 fixture; callers bind plan hashes."""
    baseline = [32, 384]
    candidate = [candidate_fast_simulations, 384]
    records = []
    for dataset in ("balanced", "swap-probes"):
        cells = [
            (f"{segment}-{mode}", phase)
            for segment in ("standard", "handicap", "pie")
            for mode in ("classic", "double")
            for phase in ("early", "middle", "late")
        ]
        if dataset == "swap-probes":
            cells = [
                (f"pie-{mode}", "early")
                for mode in ("classic", "double")
                for _ in range(4)
            ]

        def positions(full, low=False):
            rows = []
            for index, (mode, phase) in enumerate(cells):
                game = f"{dataset}-{mode}" if dataset == "balanced" else f"swap-{index}"
                ply = ("early", "middle", "late").index(phase)
                chosen = 2 if low else 1
                simulations = 640 if full else 13 if low else 53
                visits = (
                    [simulations - 2, 1, 1] if chosen == 1 else [1, simulations - 2, 1]
                )
                q = [0.5, 0.5 - loss, 0.4]
                rows.append(
                    {
                        "id": f"run/{game}/{ply}",
                        "cell": [10, mode, phase],
                        "pda": 0,
                        "actions": [1, 2, 3],
                        "visits": visits,
                        "q_values": q,
                        "policy_target": [0.5, 0.3, 0.2],
                        "selected_action": chosen,
                        "selected_value": q[chosen - 1],
                        "simulations": simulations,
                        "swap_available": dataset == "swap-probes",
                        "swap": False,
                    }
                )
            return rows

        records.append(
            {
                "stage": "cold-reference",
                "dataset": dataset,
                "ring": 10,
                "raw_pair": baseline,
                "first_visit_batch_size": 1,
                "full": True,
                "positions": positions(True),
            }
        )
        for pair in (baseline, candidate):
            for width in (1, 8):
                for full in (True, False):
                    if dataset == "swap-probes" and full:
                        continue
                    for repeat in range(3):
                        low = not full and pair == candidate
                        rows = positions(full, low)
                        seconds = 0.02 if full else 0.001 if low else 0.008
                        if full and width == 8:
                            seconds *= width8_full_scale
                        seconds *= 1 + (repeat - 1) * 0.01
                        records.append(
                            {
                                "stage": "measured",
                                "dataset": dataset,
                                "ring": 10,
                                "raw_pair": pair,
                                "first_visit_batch_size": width,
                                "full": full,
                                "repeat": repeat,
                                "positions": deepcopy(rows),
                                "seconds": seconds * len(rows),
                                "requested_rows": sum(
                                    row["simulations"] + 1 for row in rows
                                ),
                                "timing_admissible": True,
                                "inference": {key: 0 for key in TIMING_SETUP_COUNTERS},
                                "timing_setup_counters": {
                                    key: 0 for key in TIMING_SETUP_COUNTERS
                                },
                                "canonical_arm": f"{dataset}-{full}-{width}"
                                if full
                                else f"{dataset}-{pair[0]}-{width}",
                                "shared_measurement": full,
                            }
                        )
    return {
        "status": "passed",
        "plan": {
            "schema_version": 2,
            "pairs": [baseline, candidate],
            "widths": [1, 8],
            "repeats": 3,
            "seed": 1701,
            "waves": ["full", "fast"],
            "search_algorithm": SEARCH_ALGORITHM_ID,
            "runtime": "profile",
            "precision": "bf16",
        },
        "results": records,
    }


def analyze(report, **changes):
    return analyze_search_allocation_report(
        report,
        ring=10,
        candidate_fast_simulations=8,
        full_probability=0.125,
        fast_policy_weight=0.05306122448979592,
        first_visit_batch_size=changes.pop("first_visit_batch_size", 1),
        baseline_full_probability=0.35,
        bootstrap_replicates=1000,
        **changes,
    )


def test_game_cluster_bootstrap_uses_each_position_once_and_is_deterministic():
    report = valid_report()
    left, right = analyze(report), analyze(deepcopy(report))
    assert left == right
    assert left["balanced"]["positions"] == 18
    assert left["balanced"]["games"] == 6
    assert left["swaps"]["positions"] == 8
    assert left["swaps"]["games"] == 8
    assert left["balanced"]["coverage"] == left["swaps"]["coverage"] == 1
    assert left["balanced"]["mean_loss"] == pytest.approx(0.01)
    assert left["balanced"]["upper95"] == pytest.approx(0.01)
    assert left["new_swap_failures"] == 0
    assert left["full_budget_preserved"]
    assert left["cost"]["method"] == "shared-full-observed-extrema"
    assert left["cost"]["conservative_full_target_rate_ratio"] > 1


def test_changed_width_uses_separate_full_cost_bounds_and_can_fail_teacher_rate():
    result = analyze(valid_report(width8_full_scale=3), first_visit_batch_size=8)
    assert result["cost"]["method"] == "separate-full-observed-extrema"
    assert result["cost"]["conservative_full_target_rate_ratio"] < 1
    assert result["cost"]["minimum_full_probability_for_rate_parity"] is None


def test_cost_ratio_uses_observed_extrema_not_advertised_weighted_costs():
    report = valid_report()
    report["weighted_costs"] = [{"conservative_full_target_rate_ratio": 999999}]
    for row in report["results"]:
        if (
            row.get("stage") == "measured"
            and row["dataset"] == "balanced"
            and row["raw_pair"][0] == 8
            and not row["full"]
            and row["first_visit_batch_size"] == 1
            and row["repeat"] == 2
        ):
            row["seconds"] *= 10
    result = analyze(report)
    assert result["cost"]["conservative_full_target_rate_ratio"] < 1
    assert result["cost"]["minimum_full_probability_for_rate_parity"] > 0.125


@pytest.mark.parametrize("metric", TIMING_SETUP_COUNTERS)
def test_measured_graph_setup_or_failure_cannot_be_advertised_as_clean(metric):
    report = valid_report()
    row = next(row for row in report["results"] if row.get("stage") == "measured")
    row["inference"][metric] = 1
    with pytest.raises(ValueError, match="setup"):
        analyze(report)


def test_repeated_trace_mutation_and_duplicate_repeats_fail_closed():
    report = valid_report()
    row = next(row for row in report["results"] if row.get("stage") == "measured")
    report["results"].append(deepcopy(row))
    with pytest.raises(ValueError, match="repeats"):
        analyze(report)
    report = valid_report()
    row = next(
        row
        for row in report["results"]
        if row.get("stage") == "measured" and row["repeat"] == 1 and not row["full"]
    )
    row["positions"][0]["policy_target"] = [0.4, 0.4, 0.2]
    with pytest.raises(ValueError, match="trace"):
        analyze(report)


def test_unvisited_reference_actions_are_unassessed_never_imputed_regret():
    report = valid_report()
    ref = report["results"][0]["positions"][0]
    ref["visits"] = [639, 0, 1]
    result = analyze(report)
    assert result["balanced"]["coverage"] == pytest.approx(17 / 18)
    assert (
        next(row for row in result["balanced"]["rows"] if row["id"] == ref["id"])[
            "loss"
        ]
        is None
    )


def test_new_swap_disagreement_is_separate_from_placement_loss():
    report = valid_report()
    for row in report["results"]:
        if (
            row.get("stage") == "measured"
            and row["dataset"] == "swap-probes"
            and row["raw_pair"][0] == 8
        ):
            first = row["positions"][0]
            first["q_values"][1] = first["selected_value"] = -0.05
            first["swap"] = True
    result = analyze(report)
    assert result["swaps"]["new_swap_failures"] == result["new_swap_failures"] == 1
    assert result["swaps"]["decision_changes"] == 1
    assert result["swaps"]["mean_loss"] == pytest.approx(0.01)


def test_missing_swaps_and_prior_game_overlap_fail_closed():
    report = valid_report()
    report["results"] = [
        row for row in report["results"] if row["dataset"] != "swap-probes"
    ]
    with pytest.raises(ValueError, match="reference"):
        analyze(report)
    with pytest.raises(ValueError, match="overlaps"):
        analyze(valid_report(), excluded_game_ids=["run/balanced-standard-classic"])


def test_actual_full_cap_cannot_change_even_if_nominal_pair_is_unchanged():
    report = valid_report()
    for row in report["results"]:
        if (
            row.get("stage") == "measured"
            and row["dataset"] == "balanced"
            and row["full"]
            and row["raw_pair"][0] == 8
        ):
            row["positions"][0]["simulations"] -= 1
            row["positions"][0]["visits"][0] -= 1
    with pytest.raises(ValueError, match="full-search"):
        analyze(report)
