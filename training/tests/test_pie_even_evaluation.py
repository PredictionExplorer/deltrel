from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from startrain.arena import ArenaPair
from startrain.balanced_evaluation import (
    balanced_categories,
    balanced_cell_weights,
    balanced_cells,
    balanced_opening_seed,
    cell_variant,
    completed_counts_by_ring,
    cycle_confidence_sequence,
    cycle_log_e_value,
    effective_pairs_per_cycle,
    evaluation_contract,
    summarize_balanced_pairs,
)
from startrain.balanced_strength import balanced_strength_summary
from startrain.config import ArenaConfig
from startrain.promotion import _balanced_round_plan
from startrain.selfplay import GameVariant

from test_arena_resume import Clock, runner
from test_balanced_strength import frontier, measurement


def config(**changes):
    values = dict(
        balanced_cells=True,
        variant_policy="pie_even",
        rings=(10,),
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        continuation_pairs_per_ring=4,
        max_pairs_per_ring=400,
    )
    values.update(changes)
    return ArenaConfig(**values)


def pairs_for(cfg, count, outcomes=lambda name, index: (1, -1)):
    pairs = []
    for ring in cfg.rings:
        for name in balanced_categories(cfg, ring):
            for index in range(count):
                variant = cell_variant(name, index, cfg)
                pairs.append(
                    ArenaPair(
                        ring=ring,
                        pair=index,
                        opening_seed=balanced_opening_seed(
                            cfg.seed, ring, variant, index
                        ),
                        opening_action=None,
                        forced_opening=False,
                        outcomes=outcomes(name, index),
                        variant=variant.label,
                        segment=variant.segment,
                    )
                )
    return pairs


@pytest.mark.parametrize("rings", [(10,), (4, 6, 8, 10), (4, 6)])
def test_pie_even_cells_and_weights_exclude_small_board_handicap(rings):
    cfg = config(rings=rings)
    weights = balanced_cell_weights(cfg)
    assert set(weights) == set(balanced_cells(cfg))
    assert math.fsum(weights.values()) == pytest.approx(1)
    assert not any("standard" in key for key in weights)
    assert all(key.startswith("r10/") for key in weights if "handicap" in key)
    handicap_weight = math.fsum(
        value for key, value in weights.items() if "handicap" in key
    )
    assert handicap_weight == pytest.approx(0.1 if 10 in rings else 0)
    even = [value for key, value in weights.items() if key.endswith("-pie")]
    assert len(set(even)) == 1 and len(even) == 2 * len(rings)
    if rings == (10,):
        assert list(weights.values()) == [0.45, 0.45, 0.05, 0.05]


def test_weighted_summary_retains_complete_pairs_and_correct_information_count():
    cfg = config()
    pairs = pairs_for(
        cfg, 8, lambda name, _: (1, 1) if name.endswith("-pie") else (-1, -1)
    )
    result = summarize_balanced_pairs(pairs, cfg)
    aggregate = result["balanced_aggregate"]
    assert aggregate["score_rate"] == pytest.approx(0.9)
    assert aggregate["cycle_scores"] == pytest.approx([0.9, 0.9])
    assert aggregate["pairs_per_cycle"] == 16
    assert aggregate["pairs"] == 32 and aggregate["games"] == 64
    information = 4 / (0.45**2 * 2 + 0.05**2 * 2)
    assert aggregate["effective_pairs_per_cycle"] == pytest.approx(information)
    lower, upper = cycle_confidence_sequence(
        [0.9, 0.9], pairs_per_cycle=information, error_probability=cfg.alpha
    )
    assert aggregate["anytime_confidence_sequence"] == pytest.approx([lower, upper])
    # Reordering or finishing additional work in one cell cannot move the gate.
    extra = [
        pair
        for pair in pairs_for(cfg, 12)
        if pair.pair >= 8 and pair.variant == "pie-classic"
    ]
    assert (
        summarize_balanced_pairs(list(reversed(pairs)) + extra, cfg)[
            "balanced_aggregate"
        ]["cycle_scores"]
        == aggregate["cycle_scores"]
    )
    partial = [
        pair for pair in pairs if not (pair.variant == "pie-classic" and pair.pair == 2)
    ]
    assert completed_counts_by_ring(partial, cfg) == {10: 2}
    assert (
        summarize_balanced_pairs(partial, cfg)["balanced_aggregate"]["complete_cycles"]
        == 0
    )
    assert _balanced_round_plan(partial, cfg) == ({10: 2}, {10: 2})


def test_weighted_hoeffding_uses_squared_pair_coefficients_and_controls_null_error():
    cfg = config()
    count = effective_pairs_per_cycle(cfg)
    weights = np.array(list(balanced_cell_weights(cfg).values()))
    # Four severities per cell: weights/4 are the individual pair coefficients.
    pair_coefficients = np.repeat(weights / 4, 4)
    assert np.sum((count * pair_coefficients) ** 2) == pytest.approx(count)
    scores = [0.6, 0.7, 0.5]
    lambdas = evaluation_contract(cfg)["hoeffding_lambdas"]
    explicit = math.log(
        math.fsum(
            math.exp(lam * count * (sum(scores) - 1.5) - lam**2 * 3 * count / 8)
            for lam in lambdas
        )
        / len(lambdas)
    )
    assert cycle_log_e_value(
        scores, pairs_per_cycle=count, null_mean=0.5, direction="greater"
    ) == pytest.approx(explicit)
    narrow = cycle_confidence_sequence(
        [0.65] * 20, pairs_per_cycle=16, error_probability=0.05
    )
    honest = cycle_confidence_sequence(
        [0.65] * 20, pairs_per_cycle=count, error_probability=0.05
    )
    assert honest[0] < narrow[0] < narrow[1] < honest[1]
    # Heterogeneous cells have weighted null mean 0.5. Each Bernoulli models
    # perfectly correlated reversed seats, so this never assumes independent games.
    rng = np.random.default_rng(8186)
    probabilities = np.array([0.4, 0.6, 1.0, 0.0])
    trajectories = (rng.binomial(4, probabilities, size=(1500, 30, 4)) / 4) @ weights
    crossings = sum(
        any(
            cycle_log_e_value(
                scores[:stop], pairs_per_cycle=count, null_mean=0.5, direction="greater"
            )
            >= math.log(20)
            for stop in range(1, 31)
        )
        for scores in trajectories
    )
    assert crossings / len(trajectories) <= 0.065


def test_new_contract_preserves_legacy_identity_and_rejects_forbidden_evidence():
    cfg = config()
    legacy = replace(cfg, variant_policy="legacy_six")
    old = evaluation_contract(legacy)
    new = evaluation_contract(cfg)
    assert old["schema_version"] == 1 and "variant_policy" not in old
    assert old["cell_weight"] == 1 / 6
    assert new["schema_version"] == 2 and new["variant_policy"] == "pie_even"
    assert new["identity"] != old["identity"] and "cell_weight" not in new
    assert new["cell_weights"] == balanced_cell_weights(cfg)
    for pair in (
        ArenaPair(
            ring=10,
            pair=0,
            opening_seed=5,
            opening_action=None,
            forced_opening=False,
            outcomes=(1, -1),
        ),
        replace(pairs_for(cfg, 4)[8], ring=4),
    ):
        with pytest.raises(ValueError, match="unconfigured cell"):
            summarize_balanced_pairs([pair], config(rings=(4, 10)))
    handicap_pairs = [pair for pair in pairs_for(cfg, 4) if "handicap" in pair.variant]
    for mode in ("classic", "double"):
        assert [
            GameVariant.parse(pair.variant).handicap
            for pair in handicap_pairs
            if GameVariant.parse(pair.variant).mode == mode
        ] == list(cfg.handicap_severity_cycle)


def test_handicap_regression_guard_remains_active_under_ten_percent_weight():
    cfg = config()
    evidence = pairs_for(
        cfg, 80, lambda name, _: (-1, -1) if name == "classic-handicap" else (1, 1)
    )
    summary = summarize_balanced_pairs(evidence, cfg)
    assert summary["balanced_aggregate"]["score_rate"] == pytest.approx(0.95)
    assert summary["promotion"]["decision"] == "reject_ring_regression"
    assert summary["promotion"]["cell_vetoes"] == ["r10/classic-handicap"]


def test_strength_recomputes_weighted_metrics_and_isolates_legacy_epoch(tmp_path):
    cfg = config(simulations=1024)
    frontier(tmp_path, "champion")
    legacy = measurement("previous", "old-anchor", 100, rings=(10,))
    evidence = pairs_for(
        cfg,
        80,
        lambda name, index: (
            (1, 1) if name.endswith("pie") and index % 4 < 3 else (-1, -1)
        ),
    )
    current = {
        **measurement("champion", "previous", 200, rings=(10,)),
        "evaluation_contract": evaluation_contract(cfg),
        "pairs": [asdict(pair) for pair in evidence],
        "balanced_aggregate": {"score_rate": 1.0, "effective_pairs_per_cycle": 99999},
    }
    report = balanced_strength_summary(
        tmp_path,
        [legacy, current],
        wall_seconds=3600,
        provisioned_gpus=8,
        evaluation_config=cfg,
    )
    assert report["expected_cells"] == 4 and report["anchor_identity"] == "previous"
    assert len(report["path"]) == 1 and "weighted-cell" in report["method"]
    assert report["rating"] == pytest.approx(400 * math.log10(0.675 / 0.325))
    expected = cycle_confidence_sequence(
        [0.675] * 20,
        pairs_per_cycle=effective_pairs_per_cycle(cfg),
        error_probability=0.05 / 4,
    )
    assert report["path"][0]["score_confidence_interval"] == pytest.approx(expected)
    assert "another evaluation contract" in report["excluded_results"][0]["reason"]


@pytest.mark.native
def test_runner_uses_new_cells_reverses_roles_and_resumes_exactly():
    native = pytest.importorskip("star_native")
    cfg = config(rings=(4, 10), simulations=2, max_considered=2)
    counts = {4: 1, 10: 1}
    complete = runner(native, cfg).run(pair_counts=counts, checkpoint=lambda _: None)
    assert len(complete["pairs"]) == 6 and len(complete["games"]) == 12
    assert complete["evaluation_metrics"]["requested_pairs"] == 6
    assert set(complete["search"]["segments"]) == {"pie", "handicap"}
    for pair in complete["pairs"]:
        variant = GameVariant.parse(pair["variant"])
        assert variant.pie if variant.handicap == 1 else pair["ring"] == 10
        seats = [
            game["candidate_player"]
            for game in complete["games"]
            if (game["ring"], game["variant"], game["pair"])
            == (pair["ring"], pair["variant"], pair["pair"])
        ]
        assert sorted(seats) == [0, 1]
    clock = Clock()
    partial = runner(native, cfg, clock).run(
        pair_counts=counts,
        checkpoint=lambda _: None,
        stop_requested=lambda: clock.now >= 15,
    )
    assert partial["interrupted"]
    restored = runner(native, cfg).run(
        pair_counts=counts, resume_state=json.loads(json.dumps(partial["resume_state"]))
    )
    assert restored["pairs"] == complete["pairs"]
    assert restored["resume_state"] == complete["resume_state"]
    with pytest.raises(ValueError, match="evaluation contract"):
        runner(native, replace(cfg, variant_policy="legacy_six")).run(
            resume_state=partial["resume_state"]
        )
    for forbidden in ("double", "handicap-2-classic"):
        corrupted = json.loads(json.dumps(complete["resume_state"]))
        entry = next(game for game in corrupted["game_states"] if game["ring"] == 4)
        entry.update(variant=forbidden, actions=[], result=None)
        with pytest.raises(ValueError, match="balanced cell schedule"):
            runner(native, cfg)._initialize_resume(corrupted, None)
    subject = runner(native, cfg)
    for ring, variant in ((4, GameVariant(handicap=4)), (10, GameVariant())):
        with pytest.raises(ValueError, match="pie-even policy"):
            subject._pair_specifications(ring, [0], variant)
