from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from scripts.benchmark_arena_bookkeeping import legacy_snapshot, synthetic_fixture
from startrain.arena import ArenaGame, ArenaPair, ArenaRunner
from startrain.config import ArenaConfig
from startrain.selfplay import GameVariant


def test_large_snapshot_matches_previous_implementation_byte_for_byte():
    subject = synthetic_fixture()
    assert json.dumps(subject._resume_snapshot()) == json.dumps(
        legacy_snapshot(subject)
    )


@pytest.mark.native
@pytest.mark.parametrize("clinches", [False, True])
def test_resumed_native_snapshots_match_legacy_and_normalize_extensions(clinches):
    from test_arena_resume import Clock, config, runner as native_runner

    native = pytest.importorskip("star_native")
    cfg = config(exact_clinch_termination=clinches)
    clock = Clock()
    interrupted = native_runner(native, cfg, clock).run(
        checkpoint=lambda _snapshot: None,
        stop_requested=lambda: clock.now >= 15,
    )
    assert interrupted["interrupted"]
    saved = interrupted["resume_state"]
    # JSON-compatible caller values are normalized by the existing resume
    # reader, including extension fields unknown to the writer.
    saved["game_states"][0]["extension"] = {7: (["nested"], {11: (1, 2)})}
    subject = native_runner(native, cfg)
    snapshots = []

    def checkpoint(snapshot):
        assert json.dumps(snapshot) == json.dumps(legacy_snapshot(subject))
        snapshots.append(snapshot)

    complete = subject.run(resume_state=saved, checkpoint=checkpoint)
    assert not complete["interrupted"]
    # Active rows are rewritten after a move; snapshot compatibility also
    # holds at the load boundary, where the extension is still retained.
    loaded = native_runner(native, cfg)
    loaded._initialize_resume(saved, None)
    normalized = loaded._resume_snapshot()
    assert normalized["game_states"][0]["extension"] == {
        "7": [["nested"], {"11": [1, 2]}]
    }
    assert json.dumps(normalized) == json.dumps(legacy_snapshot(loaded))
    assert len(snapshots) > 1


def runner() -> ArenaRunner:
    return ArenaRunner(
        native_module=object(),
        candidate=SimpleNamespace(model_version="candidate"),
        baseline=SimpleNamespace(model_version="baseline"),
        config=ArenaConfig(rings=(4,)),
        stable_pair_seeds=True,
    )


def test_snapshot_preserves_json_schema_and_detaches_every_mutable_container():
    subject = runner()
    subject._initialize_resume(None, lambda snapshot: None)
    variant = GameVariant()
    specifications = subject._pair_specifications(4, range(3), variant)
    histories = [[0, 1, 2] for _ in specifications]
    results = [
        ArenaGame(
            ring=4,
            pair=pair,
            candidate_player=seat,
            opening_seed=seed,
            opening_action=opening,
            forced_opening=opening is not None,
            winner=seat,
            outcome=1,
            searched_moves=3,
        )
        for pair, seat, seed, opening in specifications[:3]
    ]
    subject._save_resume_games(4, variant, specifications, histories, results)
    # Readers preserve extension metadata; nested values must also be detached.
    subject._resume_games[(4, variant.label, 0, 0)]["extension"] = {"nested": [[1, 2]]}
    snapshot = subject._resume_snapshot()
    reference = json.loads(json.dumps(snapshot))
    first = results[0]
    expected_pair = ArenaPair(
        ring=first.ring,
        pair=first.pair,
        opening_seed=first.opening_seed,
        opening_action=first.opening_action,
        forced_opening=first.forced_opening,
        outcomes=(1, 1),
    )
    assert snapshot == reference
    assert snapshot["pairs"] == json.loads(json.dumps([asdict(expected_pair)]))
    assert snapshot["games"] == [asdict(game) for game in results]
    assert snapshot["progress"] == {
        "completed_games": 3,
        "completed_pairs": 1,
        "completed_moves": 18,
    }
    assert len(snapshot["game_states"]) == 6

    # Callback edits cannot alter live progress or alias the separate games
    # list. Later snapshots must be independent in the opposite direction too.
    snapshot["game_states"][0]["actions"].append(3)
    snapshot["game_states"][0]["result"]["winner"] = 1
    snapshot["game_states"][0]["extension"]["nested"][0].append(3)
    snapshot["games"][1]["winner"] = 0
    snapshot["pairs"][0]["outcomes"][0] = -1
    snapshot["config"]["rings"].append(6)
    assert snapshot["games"][0]["winner"] == 0
    assert snapshot["game_states"][1]["result"]["winner"] == 1
    assert subject._resume_snapshot() == reference
    subject._resume_games[(4, variant.label, 2, 1)]["actions"].append(4)
    assert len(reference["game_states"][-1]["actions"]) == 3


@pytest.mark.parametrize(
    ("terminal", "clinches", "scored", "expected"),
    [
        ([False, False], {}, False, []),
        ([False, False], {0: 1}, False, [(0, 1)]),
        ([False, False], {0: 1, 1: 0}, False, [(0, 1), (1, 0)]),
        ([True, False], {0: 1}, False, [(0, 1)]),
        ([True, False], {}, True, [(0, 0)]),
        ([True, False], {1: 0}, True, [(0, 0), (1, 0)]),
        ([True, True], {}, True, [(0, 0), (1, 1)]),
    ],
)
def test_checkpoint_scores_only_when_a_terminal_winner_is_needed(
    terminal, clinches, scored, expected
):
    subject = runner()
    calls = []

    def score_data():
        calls.append(True)
        assert scored, "unneeded static scoring reached the native boundary"
        return SimpleNamespace(winner=[0, 1])

    states = SimpleNamespace(
        data=lambda: SimpleNamespace(terminal=terminal), score_data=score_data
    )
    variant = GameVariant()
    games = subject._terminal_batch_games(
        4,
        subject._pair_specifications(4, [0], variant),
        variant,
        states,
        [2, 3],
        [False, False],
        clinches,
    )
    assert bool(calls) == scored
    assert [(game.candidate_player, game.winner) for game in games] == expected


def test_checkpoint_keeps_terminal_winner_validation():
    subject = runner()
    variant = GameVariant()
    states = SimpleNamespace(
        data=lambda: SimpleNamespace(terminal=[True, False]),
        score_data=lambda: SimpleNamespace(winner=[-1, -1]),
    )
    with pytest.raises(RuntimeError, match="cannot be tied"):
        subject._terminal_batch_games(
            4,
            subject._pair_specifications(4, [0], variant),
            variant,
            states,
            [2, 3],
            [False, False],
        )
