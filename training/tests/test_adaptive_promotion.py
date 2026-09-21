from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import pytest
import torch

import deltreltrain.promotion as promotion_module
from deltreltrain.adaptive_promotion import (
    allocation_metrics,
    cell_pair_prefixes,
    next_allocation,
    pending_suspected_review,
    plan_complete,
    select_handicap_check,
)
from deltreltrain.arena import ArenaPair, summarize_completed_arena_pairs
from deltreltrain.balanced_evaluation import (
    balanced_cells,
    balanced_opening_seed,
    cell_variant,
    pair_key,
)
from deltreltrain.config import ArenaConfig

from test_promotion import _promotion_wave_case


def config(**changes):
    return replace(
        ArenaConfig(
            balanced_cells=True,
            variant_policy="pie_even",
            allocation_policy="adaptive_pie",
            rings=(10,),
            pairs_per_ring=4,
            minimum_pairs_per_ring=4,
            continuation_pairs_per_ring=4,
            max_pairs_per_ring=40,
        ),
        **changes,
    )


def pairs_for(cfg, targets, outcome=(1, -1)):
    pairs = []
    for cell, target in targets.items():
        ring_text, name = cell.split("/")
        ring = int(ring_text[1:])
        for index in range(target):
            variant = cell_variant(name, index, cfg)
            pairs.append(
                ArenaPair(
                    ring=ring,
                    pair=index,
                    opening_seed=balanced_opening_seed(cfg.seed, ring, variant, index),
                    opening_action=None,
                    forced_opening=False,
                    outcomes=outcome,
                    variant=variant.label,
                    segment=variant.segment,
                )
            )
    return pairs


def uncertainty_summary(cfg, *, suspicious=False):
    return {
        "balanced_aggregate": {"anytime_confidence_sequence": [0.6, 0.9]},
        "per_cell": {
            cell: {
                "score_rate": 0.2
                if suspicious and cell == "r10/classic-handicap"
                else 0.6,
                "included_pairs": 4,
                "anytime_confidence_sequence": [0.1, 0.9],
                "aggregate_confidence_sequence": [0.1, 0.9],
            }
            for cell in balanced_cells(cfg)
        },
    }


def test_initial_32_games_then_ninety_percent_pie_without_160_game_cycle():
    cfg = config()
    initial = next_allocation(cfg, previous_plan=None, summary={})
    assert initial is not None
    assert initial["cell_pair_targets"] == dict.fromkeys(balanced_cells(cfg), 4)
    assert sum(initial["cell_pair_targets"].values()) * 2 == 32
    continued = next_allocation(cfg, previous_plan=initial, summary={})
    assert continued is not None
    targets = continued["cell_pair_targets"]
    assert targets == {
        "r10/classic-pie": 13,
        "r10/double-pie": 13,
        "r10/classic-handicap": 5,
        "r10/double-handicap": 5,
    }
    assert sum(targets.values()) - 16 == 20
    assert (targets["r10/classic-pie"] + targets["r10/double-pie"] - 8) / 20 == 0.9


def test_fixed_whole_evaluation_budget_allows_more_than_old_cap_in_pie_cells():
    cfg = config()
    plan = None
    history = []
    while following := next_allocation(cfg, previous_plan=plan, summary={}):
        history.append(following)
        plan = following
        assert sum(plan["cell_pair_targets"].values()) <= 160
    assert len(history) == 9
    assert plan["cell_pair_targets"] == {
        "r10/classic-pie": 69,
        "r10/double-pie": 69,
        "r10/classic-handicap": 11,
        "r10/double-handicap": 11,
    }
    metrics = allocation_metrics(pairs_for(cfg, plan["cell_pair_targets"]), cfg)
    assert metrics["completed_pairs"] == 160 and metrics["completed_games"] == 320
    assert metrics["actual_pie_pair_share"] == pytest.approx(138 / 160)


def test_sparse_later_completions_do_not_finish_current_plan():
    cfg = config()
    plan = next_allocation(cfg, previous_plan=None, summary={})
    pairs = pairs_for(cfg, plan["cell_pair_targets"])
    missing = next(
        pair for pair in pairs if pair.variant == "pie-classic" and pair.pair == 1
    )
    partial = [pair for pair in pairs if pair != missing]
    assert cell_pair_prefixes(partial, cfg)["r10/classic-pie"] == 1
    assert not plan_complete(plan, partial, cfg)
    assert plan_complete(plan, partial + [missing], cfg)


def test_extra_handicap_checks_are_decision_relevant_cost_aware_and_bounded():
    cfg = config()
    summary = uncertainty_summary(cfg)
    assert (
        select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=100)
        is None
    )
    # Mere uncertainty away from either decision threshold gets no extra work.
    summary["per_cell"]["r10/classic-handicap"]["aggregate_confidence_sequence"] = [
        0,
        1,
    ]
    assert (
        select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=100)
        is None
    )
    # Make the handicap uncertainty relevant and more useful than the nearly
    # resolved pie streams despite the deliberately larger PDA cost.
    summary["balanced_aggregate"]["anytime_confidence_sequence"] = [0.49, 0.6]
    for cell in ("r10/classic-pie", "r10/double-pie"):
        summary["per_cell"][cell]["included_pairs"] = 64
        summary["per_cell"][cell]["aggregate_confidence_sequence"] = [0.599, 0.601]
    check = select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=100)
    assert check is not None and check["reason"] == "decision_relevant_uncertainty"
    assert check["pairs"] == 4
    assert check["estimated_search_cost"] > 4 * 2 * cfg.simulations
    assert (
        select_handicap_check(cfg, summary, extra_pairs_used=80, remaining_pairs=100)
        is None
    )
    assert (
        select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=3)
        is None
    )
    # Suspected regression can independently justify one cycle, but not a
    # proven regression whose interval is already wholly below the floor.
    summary = uncertainty_summary(cfg, suspicious=True)
    check = select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=100)
    assert check is not None and check["reason"] == "suspected_regression"
    summary["per_cell"]["r10/classic-handicap"]["anytime_confidence_sequence"] = [
        0,
        0.2,
    ]
    assert (
        select_handicap_check(cfg, summary, extra_pairs_used=0, remaining_pairs=100)
        is None
    )


def _case(tmp_path, monkeypatch, **changes):
    case = _promotion_wave_case(tmp_path, monkeypatch)
    case.supervisor.experiment = replace(case.experiment, arena=config(**changes))
    return case


def test_durable_targets_survive_partial_calls_and_contract_tampering_fails(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    completed = pairs_for(cfg, first["cell_pair_targets"])
    # Fast cells and sparse later indices never extend the plan on restart.
    for partial in (completed[:5], completed[:-1], list(reversed(completed[1:]))):
        resumed = subject._adaptive_allocation(case.candidate, case.champion, partial)
        assert resumed == first
    second = subject._adaptive_allocation(case.candidate, case.champion, completed)
    assert second["plan_index"] == 1
    assert second["cell_pair_targets"]["r10/classic-pie"] == 13
    path = subject._result_path(case.candidate, case.champion)
    allocation_path = path.with_name(f"{path.stem}.allocation.json")
    payload = json.loads(allocation_path.read_text())
    assert len(payload["plan_history"]) == 2
    payload["plan_history"][1]["cell_pair_targets"]["r10/classic-pie"] += 1
    allocation_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees with its policy"):
        subject._adaptive_allocation(case.candidate, case.champion, completed)


def test_review_reserves_one_cycle_preserves_pending_work_and_stops_at_extra_cap(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)

    def passing_screen(pairs, config, **options):
        result = summarize_completed_arena_pairs(pairs, config, **options)
        if pairs:
            result["promotion"]["decision"] = "promote"
        return result

    monkeypatch.setattr(
        promotion_module, "summarize_completed_arena_pairs", passing_screen
    )
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    accumulated = [
        replace(pair, outcomes=(-1, -1))
        if "classic" in pair.variant and "handicap" in pair.variant
        else pair
        for pair in pairs_for(cfg, first["cell_pair_targets"])
    ]
    review = subject._adaptive_allocation(
        case.candidate, case.champion, accumulated, review_only=True
    )
    assert review["cell_pair_targets"]["r10/classic-handicap"] == 8
    assert pending_suspected_review(review, accumulated, cfg)
    assert (
        subject._adaptive_allocation(
            case.candidate, case.champion, accumulated, review_only=True
        )
        == review
    )
    # The pure planner's hard limit applies even if every subsequent check
    # remains suspicious; production recomputes suspicion from immutable pairs.
    summary = uncertainty_summary(cfg, suspicious=True)
    for _ in range(19):
        review = next_allocation(
            cfg, previous_plan=review, summary=summary, review_only=True
        )
        assert review is not None
    assert review["extra_handicap_pairs_total"] == 80
    assert (
        next_allocation(cfg, previous_plan=review, summary=summary, review_only=True)
        is None
    )


def test_supervisor_resumes_small_slices_counts_real_pairs_and_exhausts_global_budget(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, max_pairs_per_ring=10)
    subject = case.supervisor
    stopped = False
    targets_seen = []

    class PartialRunner:
        def __init__(self, **options):
            self.config = options["config"]

        def run(self, *, cell_pair_targets, previous_pairs, **_options):
            nonlocal stopped
            targets_seen.append(dict(cell_pair_targets))
            finished = {pair_key(pair) for pair in previous_pairs}
            pending = [
                pair
                for pair in pairs_for(self.config, cell_pair_targets)
                if pair_key(pair) not in finished
            ]
            completed = pending[:3]
            assert completed
            stopped = True
            return {
                "candidate": case.candidate.model_identity,
                "baseline": case.champion.model_identity,
                "pairs": [asdict(pair) for pair in completed],
                "games": [],
                **summarize_completed_arena_pairs(
                    previous_pairs + completed, self.config
                ),
            }

    monkeypatch.setattr(promotion_module, "ArenaRunner", PartialRunner)
    previous = None
    for _ in range(30):
        stopped = False
        subject._evaluate_candidate_session(
            candidate=case.candidate,
            champion=case.champion,
            previous=previous,
            stop_requested=lambda: stopped,
            progress=None,
            once=True,
        )
        result_path = subject._result_path(case.candidate, case.champion)
        previous = json.loads(result_path.read_text())
        assert previous["sampling_allocation"]["completed_pairs"] == len(
            previous["pairs"]
        )
        if previous["terminal"]:
            break
    else:
        pytest.fail("adaptive evaluation did not stop at its budget")
    assert len(previous["pairs"]) == 40
    assert previous["promotion"]["decision"] == "reject_max_pairs"
    assert previous["sampling_allocation"]["maximum_total_pairs"] == 40
    assert targets_seen[:6] == [dict.fromkeys(balanced_cells(config()), 4)] * 6
    assert len(set(tuple(sorted(target.items())) for target in targets_seen)) == 3
    assert previous["evaluation_metrics"]["requested_pairs"] == 1
    assert subject._historical_arena_config().allocation_policy == "equal_cells"
    assert (
        subject._historical_arena_config().simulations == config().strength_simulations
    )


def test_promotion_reviews_suspicious_handicap_then_waits_for_reserved_cycle(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)

    def passing_screen(pairs, config, **options):
        result = summarize_completed_arena_pairs(pairs, config, **options)
        if all(value >= 4 for value in cell_pair_prefixes(pairs, config).values()):
            # Isolate coordination from the statistical test's power: the
            # statistics suite separately verifies when this boundary is met.
            result["promotion"]["decision"] = "promote"
        return result

    monkeypatch.setattr(
        promotion_module, "summarize_completed_arena_pairs", passing_screen
    )
    initial = subject._adaptive_allocation(case.candidate, case.champion, [])
    pairs = [
        replace(pair, outcomes=(1, 1))
        if pair.segment == "pie"
        else replace(pair, outcomes=(-1, -1))
        if "classic" in pair.variant
        else pair
        for pair in pairs_for(cfg, initial["cell_pair_targets"])
    ]
    accumulated = []

    def persist(completed, previous, plan):
        result = {
            "candidate": case.candidate.model_identity,
            "baseline": case.champion.model_identity,
            "pairs": [asdict(pair) for pair in completed],
            "games": [],
            **passing_screen(accumulated + completed, cfg),
        }
        decision, terminal = subject._persist_wave(
            candidate=case.candidate,
            champion=case.champion,
            previous=previous,
            accumulated=accumulated,
            result=result,
            arena_config=cfg,
            round_started=time.perf_counter(),
            metric_device=torch.device("cpu"),
            collect_cuda_metrics=False,
            progress=None,
            wave_index=0,
            pair_starts={10: 0},
            pair_counts={10: 4},
            cell_pair_targets=plan["cell_pair_targets"],
            allocation_plan=plan,
        )
        return decision, terminal, result

    decision, terminal, first_result = persist(pairs, None, initial)
    assert decision == "continue" and not terminal
    assert (
        first_result["promotion"]["deferred_for_handicap_review"][
            "extra_handicap_check"
        ]["reason"]
        == "suspected_regression"
    )
    review = subject._adaptive_allocation(case.candidate, case.champion, accumulated)
    assert review["phase"] == "handicap_review"
    finished = {pair_key(pair) for pair in accumulated}
    pending = [
        replace(pair, outcomes=(-1, -1))
        for pair in pairs_for(cfg, review["cell_pair_targets"])
        if pair_key(pair) not in finished
    ]
    assert len(pending) == 4
    decision, terminal, partial_result = persist(pending[:3], first_result, review)
    assert decision == "continue" and not terminal
    assert partial_result["promotion"]["provisional_global_decision"] == "promote"
    assert not partial_result["promotion"]["allocation_boundary_complete"]
    assert (
        subject._adaptive_allocation(case.candidate, case.champion, accumulated)
        == review
    )


@pytest.mark.parametrize("handicap_loses", [False, True])
def test_real_statistics_promote_strong_pie_but_review_catastrophic_handicap(
    tmp_path, monkeypatch, handicap_loses
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor

    class FullRunner:
        def __init__(self, **options):
            self.config = options["config"]

        def run(self, *, cell_pair_targets, previous_pairs, **_options):
            finished = {pair_key(pair) for pair in previous_pairs}
            completed = [
                replace(
                    pair,
                    outcomes=(1, 1)
                    if pair.segment == "pie"
                    else (-1, -1)
                    if handicap_loses
                    else (1, -1),
                )
                for pair in pairs_for(self.config, cell_pair_targets)
                if pair_key(pair) not in finished
            ]
            return {
                "candidate": case.candidate.model_identity,
                "baseline": case.champion.model_identity,
                "pairs": [asdict(pair) for pair in completed],
                "games": [],
                **summarize_completed_arena_pairs(
                    previous_pairs + completed, self.config
                ),
            }

    monkeypatch.setattr(promotion_module, "ArenaRunner", FullRunner)
    previous = None
    decisions = []
    for _ in range(40):
        subject._evaluate_candidate_session(
            candidate=case.candidate,
            champion=case.champion,
            previous=previous,
            stop_requested=lambda: False,
            progress=None,
            once=True,
        )
        previous = json.loads(
            subject._result_path(case.candidate, case.champion).read_text()
        )
        decisions.append(previous["promotion"]["decision"])
        if previous["terminal"]:
            break
    else:
        pytest.fail("real-statistics adaptive evaluation failed to terminate")
    assert len(previous["pairs"]) <= 160
    if handicap_loses:
        assert "promote" not in decisions
        assert previous["promotion"]["decision"] == "reject_ring_regression"
        assert any("handicap" in cell for cell in previous["promotion"]["cell_vetoes"])
        assert previous["wave_plan"]["extra_handicap_pairs_total"] > 40
    else:
        assert previous["promotion"]["decision"] == "promote"
        assert previous["wave_plan"]["extra_handicap_pairs_total"] == 0
