"""Absolute per-cell work bounds preserve paired seeds, evidence and resumes."""

import json

import pytest

from deltreltrain.arena import ArenaPair, ArenaRunner
from deltreltrain.balanced_evaluation import (
    balanced_cells,
    cell_variant,
    pair_key,
    summarize_balanced_pairs,
)
from deltreltrain.config import ArenaConfig
from test_arena_resume import Clock, Evaluator


def config(**changes):
    settings = {
        "rings": (4, 10),
        "balanced_cells": True,
        "variant_policy": "pie_even",
        "pairs_per_ring": 4,
        "minimum_pairs_per_ring": 4,
        "max_pairs_per_ring": 12,
        "simulations": 1,
        "max_considered": 1,
        "bootstrap_samples": 200,
    }
    return ArenaConfig(**(settings | changes))


def runner(cfg=None, *, native=None, clock=None):
    return ArenaRunner(
        native_module=object() if native is None else native,
        candidate=Evaluator("candidate", clock),
        baseline=Evaluator("baseline", clock),
        config=cfg or config(),
    )


def make_pair(subject, ring, name, index):
    variant = cell_variant(name, index, subject.config)
    _, _, seed, opening = subject._pair_specifications(ring, [index], variant)[0]
    return ArenaPair(
        ring=ring,
        pair=index,
        opening_seed=seed,
        opening_action=opening,
        forced_opening=opening is not None,
        outcomes=(1, -1),
        variant=variant.label,
        segment=variant.segment,
    )


def record_work(subject, monkeypatch):
    observed = []

    def play(ring, by_variant, games, pairs, **_kwargs):
        for variant, indices in by_variant.items():
            for index in indices:
                name = f"{variant.mode}-{'pie' if variant.pie else 'handicap'}"
                if not variant.pie and variant.handicap == 1:
                    name = f"{variant.mode}-standard"
                pair = make_pair(subject, ring, name, index)
                observed.append(pair)
                pairs.append(pair)
        return True

    monkeypatch.setattr(subject, "_play_balanced_groups", play)
    return observed


def test_uneven_targets_fill_only_missing_indices_and_keep_all_prior_statistics(
    monkeypatch,
):
    subject = runner()
    prior = [
        make_pair(subject, 4, "classic-pie", 0),
        make_pair(subject, 4, "classic-pie", 2),
        make_pair(subject, 10, "double-handicap", 0),
        # Completed evidence remains included even after stopping work on a cell.
        make_pair(subject, 10, "double-pie", 4),
    ]
    targets = dict.fromkeys(balanced_cells(subject.config), 0)
    targets.update(
        {"r4/classic-pie": 3, "r10/classic-handicap": 5, "r10/double-handicap": 2}
    )
    observed = record_work(subject, monkeypatch)
    result = subject.run(cell_pair_targets=targets, previous_pairs=prior)
    assert {pair_key(pair) for pair in observed} == {
        (4, "classic-pie", 1),
        *((10, "classic-handicap", index) for index in range(5)),
        (10, "double-handicap", 1),
    }
    assert result["evaluation_metrics"]["requested_pairs"] == 7
    assert result["evaluation_metrics"]["completed_pairs"] == 7
    assert result["search"]["cell_pair_targets"] == targets
    expected = summarize_balanced_pairs(prior + observed, subject.config)
    assert result["balanced_aggregate"] == expected["balanced_aggregate"]
    assert result["per_cell"] == expected["per_cell"]
    assert {
        pair.pair: pair.variant
        for pair in observed
        if pair_key(pair)[:2] == (10, "classic-handicap")
    } == {
        0: "handicap-2-classic",
        1: "handicap-4-classic",
        2: "handicap-6-classic",
        3: "handicap-9-classic",
        4: "handicap-2-classic",
    }


def test_zero_targets_request_no_games_and_snapshot_the_supplied_mapping(monkeypatch):
    subject = runner()
    observed = record_work(subject, monkeypatch)
    targets = dict.fromkeys(balanced_cells(subject.config), 0)
    result = subject.run(cell_pair_targets=targets)
    assert observed == []
    assert result["evaluation_metrics"]["requested_pairs"] == 0
    targets["r4/classic-pie"] = 100
    assert result["search"]["cell_pair_targets"]["r4/classic-pie"] == 0


def test_cell_target_can_exceed_the_legacy_per_ring_cap(monkeypatch):
    subject = runner(config(max_pairs_per_ring=4))
    observed = record_work(subject, monkeypatch)
    targets = dict.fromkeys(balanced_cells(subject.config), 0)
    targets["r4/classic-pie"] = 8
    result = subject.run(cell_pair_targets=targets)
    assert [pair.pair for pair in observed] == list(range(8))
    assert result["evaluation_metrics"]["requested_pairs"] == 8


@pytest.mark.parametrize("bad", [-1, True, False, 1.0, "1", None])
def test_targets_reject_noninteger_or_negative_bounds_before_resuming(monkeypatch, bad):
    subject = runner()
    targets = dict.fromkeys(balanced_cells(subject.config), 0)
    targets["r4/classic-pie"] = bad
    monkeypatch.setattr(
        subject,
        "_initialize_resume",
        lambda *_: pytest.fail("invalid targets reached resume initialization"),
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        subject.run(cell_pair_targets=targets)


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_ring", "non_mapping"])
def test_targets_require_exact_configured_cell_keys(fault):
    subject = runner()
    targets = dict.fromkeys(balanced_cells(subject.config), 0)
    if fault == "missing":
        targets.pop("r4/classic-pie")
    elif fault == "extra":
        targets["r4/classic-handicap"] = 1
    elif fault == "wrong_ring":
        targets["r8/classic-pie"] = targets.pop("r4/classic-pie")
    else:
        targets = [(cell, 0) for cell in targets]
    with pytest.raises(ValueError, match="cell_pair_targets"):
        subject.run(cell_pair_targets=targets)


@pytest.mark.parametrize("legacy_option", ["pair_starts", "pair_counts"])
def test_targets_cannot_mix_even_empty_legacy_wave_arguments(legacy_option):
    subject = runner()
    with pytest.raises(ValueError, match="cannot be combined"):
        subject.run(
            cell_pair_targets=dict.fromkeys(balanced_cells(subject.config), 0),
            **{legacy_option: {}},
        )


def test_targets_require_balanced_mode():
    subject = runner(config(balanced_cells=False, variant_policy="legacy_six"))
    with pytest.raises(ValueError, match="balanced evaluation"):
        subject.run(cell_pair_targets={})


def test_invalid_previous_pairs_are_rejected_before_resume_initialization(monkeypatch):
    subject = runner()
    prior = make_pair(subject, 4, "classic-pie", 0)
    monkeypatch.setattr(
        subject,
        "_initialize_resume",
        lambda *_: pytest.fail("invalid prior reached resume initialization"),
    )
    with pytest.raises(ValueError, match="unique"):
        subject.run(
            cell_pair_targets=dict.fromkeys(balanced_cells(subject.config), 1),
            previous_pairs=[prior, prior],
        )


def test_legacy_ring_ranges_remain_unchanged_when_targets_are_absent(monkeypatch):
    subject = runner(config(rings=(4,), variant_policy="legacy_six"))
    prior = [
        make_pair(subject, 4, name, 0)
        for name in (
            "classic-standard",
            "double-standard",
            "classic-pie",
            "double-pie",
            "classic-handicap",
            "double-handicap",
        )
    ]
    observed = record_work(subject, monkeypatch)
    result = subject.run(pair_starts={4: 1}, pair_counts={4: 2}, previous_pairs=prior)
    assert len(observed) == 12
    assert {pair.pair for pair in observed} == {1, 2}
    assert result["evaluation_metrics"]["requested_pairs"] == 12
    assert "cell_pair_targets" not in result["search"]


@pytest.mark.native
def test_native_targets_resume_committed_games_with_the_same_seeds_and_final_results():
    native = pytest.importorskip("deltrel_native")
    cfg = config(rings=(4,))
    targets = {"r4/classic-pie": 2, "r4/double-pie": 1}
    complete = runner(cfg, native=native).run(
        cell_pair_targets=targets, checkpoint=lambda _state: None
    )
    clock = Clock()
    partial = runner(cfg, native=native, clock=clock).run(
        cell_pair_targets=targets,
        checkpoint=lambda _state: None,
        stop_requested=lambda: clock.now >= 5,
    )
    assert partial["interrupted"]
    saved = json.loads(json.dumps(partial["resume_state"]))
    assert any(entry["actions"] for entry in saved["game_states"])
    assert any(entry["result"] is None for entry in saved["game_states"])
    snapshots = []
    with pytest.raises(ValueError, match="already-started resume pair"):
        runner(cfg, native=native).run(
            cell_pair_targets=dict.fromkeys(targets, 0),
            resume_state=saved,
            checkpoint=snapshots.append,
        )
    assert snapshots == []
    resumed = runner(cfg, native=native).run(
        cell_pair_targets=targets,
        previous_pairs=[ArenaPair(**pair) for pair in partial["pairs"]],
        resume_state=saved,
    )

    def order(pair):
        return pair["ring"], pair["variant"], pair["pair"]

    assert sorted(partial["pairs"] + resumed["pairs"], key=order) == sorted(
        complete["pairs"], key=order
    )
    assert resumed["resume_state"] == complete["resume_state"]
    assert resumed["balanced_aggregate"] == complete["balanced_aggregate"]
    assert resumed["evaluation_metrics"]["requested_pairs"] == sum(
        targets.values()
    ) - len(partial["pairs"])


@pytest.mark.native
def test_targets_may_skip_proven_pairs_already_present_in_previous_results():
    native = pytest.importorskip("deltrel_native")
    cfg = config(rings=(4,))
    targets = {"r4/classic-pie": 1, "r4/double-pie": 0}
    full = runner(cfg, native=native).run(
        cell_pair_targets=targets, checkpoint=lambda _state: None
    )
    prior = [ArenaPair(**pair) for pair in full["pairs"]]
    result = runner(cfg, native=native).run(
        cell_pair_targets=dict.fromkeys(targets, 0),
        previous_pairs=prior,
        resume_state=full["resume_state"],
    )
    assert result["pairs"] == []
    assert result["evaluation_metrics"]["requested_pairs"] == 0
    assert result["per_cell"] == full["per_cell"]
    assert result["resume_state"]["pairs"] == full["resume_state"]["pairs"]
