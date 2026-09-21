from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from deltreltrain.arena import ArenaPair, ArenaRunner
from deltreltrain.balanced_evaluation import evaluation_contract
from deltreltrain.config import ArenaConfig
from deltreltrain.inference import InferenceResponse
from deltreltrain.selfplay import GameVariant
from deltreltrain.search_options import SearchExecutionConfig


class Evaluator:
    def __init__(self, identity: str) -> None:
        self.model_version = identity
        self.evaluator_calls = 0
        self.evaluator_rows = 0

    def evaluate(self, requests):
        self.evaluator_calls += 1
        self.evaluator_rows += len(requests)
        return InferenceResponse(
            list(requests.tokens),
            [-1.0] * len(requests),
            list(requests.legal_offsets),
            [0.0] * len(requests.legal_actions),
        )


def config(enabled: bool = True) -> ArenaConfig:
    return ArenaConfig(
        rings=(4,),
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=4,
        simulations=2,
        max_considered=2,
        bootstrap_samples=200,
        exact_clinch_termination=enabled,
    )


def runner(native, enabled=True, *, sequential=False):
    if sequential:
        native_type = native.StateBatch
        native = SimpleNamespace(
            StateBatch=lambda *args, **kwargs: native_type(*args, **kwargs),
            SearchBatch=native.SearchBatch,
        )
    return ArenaRunner(
        native_module=native,
        candidate=Evaluator("candidate"),
        baseline=Evaluator("baseline"),
        config=config(enabled),
        stable_pair_seeds=True,
    )


def play(subject, variant, saved=None, *, interrupt=False):
    snapshots = []
    subject._initialize_resume(saved, snapshots.append)
    specifications = subject._pair_specifications(4, range(4), variant)
    with subject._inference_owner() as executor:
        games = subject._play_ring_batch(
            4,
            specifications,
            variant=variant,
            progress=None,
            inference_executor=executor,
            stop_requested=lambda: (
                interrupt and bool(snapshots and snapshots[-1]["games"])
            ),
        )
    return games, snapshots[-1]


VARIANTS = [
    GameVariant(mode=mode, pie=pie, handicap=handicap)
    for mode in ("classic", "double")
    for pie, handicap in (
        (False, 1),
        (True, 1),
        (False, 2),
        (False, 4),
        (False, 6),
        (False, 9),
    )
]


@pytest.mark.native
@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda variant: variant.label)
def test_exact_clinch_preserves_outcomes_and_actual_history(variant, sequential):
    native = pytest.importorskip("deltrel_native")
    full_runner = runner(native, False, sequential=sequential)
    full, full_state = play(full_runner, variant)
    early_runner = runner(native, sequential=sequential)
    early, early_state = play(early_runner, variant)
    assert len(early) == len(full) == 8
    assert [replace(game, searched_moves=0) for game in early] == [
        replace(game, searched_moves=0) for game in full
    ]
    assert early_state["pairs"] == full_state["pairs"]
    assert sum(game.searched_moves for game in early) < sum(
        game.searched_moves for game in full
    )
    assert (
        early_runner.candidate.evaluator_rows + early_runner.baseline.evaluator_rows
        < (full_runner.candidate.evaluator_rows + full_runner.baseline.evaluator_rows)
    )
    for shortened, complete in zip(
        early_state["game_states"], full_state["game_states"], strict=True
    ):
        assert shortened["actions"] == complete["actions"][: len(shortened["actions"])]
        assert shortened["result"]["searched_moves"] == len(shortened["actions"])
        # A successful proof must not have replaced the actual played prefix
        # with synthetic loser-filled placements in the persisted history.
        early_runner._verify_resume_winner(
            shortened,
            variant,
            next(
                game
                for game in early
                if game.pair == shortened["pair"]
                and game.candidate_player == shortened["candidate_player"]
            ),
        )


@pytest.mark.native
@pytest.mark.parametrize("sequential", [False, True])
def test_interrupted_clinches_survive_resume_and_disabling_new_clinches(sequential):
    native = pytest.importorskip("deltrel_native")
    variant = GameVariant(mode="double", handicap=9)
    subject = runner(native, sequential=sequential)
    _, partial = play(subject, variant, interrupt=True)
    assert 0 < len(partial["games"]) < 8
    assert (
        any(entry["result"] is None for entry in partial["game_states"]) or sequential
    )
    resumed, state = play(runner(native, sequential=sequential), variant, partial)
    expected, expected_state = play(runner(native, sequential=sequential), variant)
    assert resumed == expected
    assert state == expected_state
    rollback_games, rollback_state = play(
        runner(native, False, sequential=sequential), variant, partial
    )
    prior = {
        (game["pair"], game["candidate_player"]): game for game in partial["games"]
    }
    for game in rollback_games:
        key = game.pair, game.candidate_player
        if key in prior:
            assert asdict(game) == prior[key]
    assert rollback_state["pairs"] == expected_state["pairs"]


@pytest.mark.native
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("corruption", ["winner", "unproved_history"])
def test_resume_rejects_forged_clinch_completion(enabled, corruption):
    native = pytest.importorskip("deltrel_native")
    variant = GameVariant(mode="classic", pie=True)
    _, saved = play(runner(native), variant)
    forged = json.loads(json.dumps(saved))
    entry = forged["game_states"][0]
    game = entry["result"]
    if corruption == "winner":
        game["winner"] = 1 - game["winner"]
        game["outcome"] = -game["outcome"]
    else:
        entry["actions"] = []
        game["searched_moves"] = 0
        game["swapped"] = False
    with pytest.raises(ValueError, match="winner.*proof"):
        runner(native, enabled)._initialize_resume(forged, None)


@pytest.mark.native
def test_pending_pie_swap_is_excluded_from_native_proof(monkeypatch):
    native = pytest.importorskip("deltrel_native")
    subject = runner(native)
    states = native.StateBatch(4, 1, mode="classic", pie=True)
    states.apply_many([0], [0])
    assert states.data().swap_available == [True]

    def unexpected(*args, **kwargs):
        pytest.fail("pending pie swap must not enter the completion proof")

    monkeypatch.setattr(subject, "_semantic_subset", unexpected)
    assert subject._proven_clinch_winners(states) == {}


@pytest.mark.native
def test_old_snapshot_without_option_resumes_and_other_contract_changes_fail():
    native = pytest.importorskip("deltrel_native")
    variant = GameVariant()
    _, old = play(runner(native, False), variant, interrupt=True)
    old["config"].pop("exact_clinch_termination", None)
    games, _ = play(runner(native), variant, old)
    assert len(games) == 8
    invalid = json.loads(json.dumps(old))
    invalid["config"]["simulations"] += 1
    with pytest.raises(ValueError, match="evaluation contract"):
        runner(native)._initialize_resume(invalid, None)
    invalid = json.loads(json.dumps(old))
    invalid["config"]["exact_clinch_termination"] = "yes"
    with pytest.raises(ValueError, match="exact clinch option"):
        runner(native)._initialize_resume(invalid, None)


def test_clinch_option_keeps_statistical_contract_identical():
    assert evaluation_contract(config(True)) == evaluation_contract(config(False))


def test_clinch_option_requires_independent_game_execution():
    with pytest.raises(ValueError, match="stable pair seeds"):
        ArenaRunner(
            native_module=object(),
            candidate=Evaluator("candidate"),
            baseline=Evaluator("baseline"),
            config=config(),
        )


@pytest.mark.native
def test_clinch_option_rejects_group_dependent_subtree_reuse():
    native = pytest.importorskip("deltrel_native")
    with pytest.raises(ValueError, match="no subtree reuse"):
        ArenaRunner(
            native_module=native,
            candidate=Evaluator("candidate"),
            baseline=Evaluator("baseline"),
            config=replace(
                config(), search_execution=SearchExecutionConfig(subtree_reuse=True)
            ),
            stable_pair_seeds=True,
        )


@pytest.mark.native
def test_balanced_cycle_statistics_and_resumed_pair_accounting_are_unchanged():
    native = pytest.importorskip("deltrel_native")

    def balanced(enabled):
        return ArenaRunner(
            native_module=native,
            candidate=Evaluator("candidate"),
            baseline=Evaluator("baseline"),
            config=replace(config(enabled), balanced_cells=True),
        )

    complete = balanced(True).run(checkpoint=lambda _snapshot: None)
    full = balanced(False).run(checkpoint=lambda _snapshot: None)
    assert complete["pairs"] == full["pairs"]
    for name in ("balanced_aggregate", "promotion", "evaluation_contract"):
        assert complete[name] == full[name]
    snapshots = []
    partial = balanced(True).run(
        checkpoint=snapshots.append,
        stop_requested=lambda: bool(snapshots and snapshots[-1]["pairs"]),
    )
    assert partial["interrupted"]
    assert 0 < len(partial["pairs"]) < len(complete["pairs"])
    resumed = balanced(True).run(
        resume_state=partial["resume_state"],
        previous_pairs=[ArenaPair(**pair) for pair in partial["pairs"]],
    )
    assert len(resumed["pairs"]) + len(partial["pairs"]) == len(complete["pairs"])
    for name in ("balanced_aggregate", "promotion", "evaluation_contract"):
        assert resumed[name] == complete[name]
