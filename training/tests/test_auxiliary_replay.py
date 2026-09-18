"""Observed future decisions and final components survive immutable replay."""

from dataclasses import replace
import json

import numpy as np
import pytest

from startrain.native import positions_from_native, score_results_from_native
from startrain.replay import (
    ReplaySample,
    ReplaySchemaError,
    _AUXILIARY_SAMPLE_ARRAY_NAMES,
    _final_components,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
    read_replay_shard,
    write_replay_shard,
)
from startrain.replay_publication import _digest_rows, _validate_future_policies
from startrain.selfplay import _Decision, _future_policy_targets
from startrain.symmetry import D5Transform
from startrain.topology import get_topology


def trajectory(
    *, mode="double", pie=False, handicap=1, actions=(0, 1, 2, 3, 4), rings=4
):
    native = pytest.importorskip("star_native")
    state = native.StateBatch(rings, 1, mode=mode, pie=pie, handicap=handicap)
    decisions = []
    for ply, action in enumerate(actions):
        position = positions_from_native(state.data())[0]
        policy = (position.stones.numpy() == -1).astype(np.float32)
        policy /= policy.sum()
        decisions.append(
            _Decision(
                position,
                policy,
                True,
                4,
                ply,
                7,
                ply,
                1.0,
                0.0,
                swapped=action == state.node_count,
            )
        )
        state.apply_many([0], [action])
    return decisions, state


def samples_from(decisions, final=None):
    return [
        ReplaySample.from_position(
            d.position,
            policy=d.policy,
            final_score=final,
            search_provenance=f"test:final={'board-full' if final else 'pending-policy'}:swap={'taken' if d.swapped else 'no'}:end",
            policy_provenance="test",
            game_id="aux-game",
            ply=d.ply,
            **future,
        )
        for d, future in zip(decisions, _future_policy_targets(decisions), strict=True)
    ]


@pytest.mark.native
def test_double_targets_skip_own_second_placement_and_stop_at_prefix_end():
    decisions, _ = trajectory()
    labels = _future_policy_targets(decisions)
    assert labels[0]["opponent_reply_ply"] == 1
    assert "second_stone" not in labels[0]  # One-stone opening.
    assert labels[1]["second_stone_ply"] == 2
    assert labels[1]["opponent_reply_ply"] == 3
    assert labels[2]["opponent_reply_ply"] == 3
    assert "second_stone" not in labels[2]
    assert labels[3]["second_stone_ply"] == 4
    assert "opponent_reply" not in labels[3]
    assert labels[4] == {}
    rows = samples_from(decisions)
    _validate_future_policies(rows)
    saved = rows[1].second_stone.copy()
    decisions[2].policy[:] = 0
    np.testing.assert_array_equal(rows[1].second_stone, saved)


@pytest.mark.native
@pytest.mark.parametrize(
    "mode,handicap", [("classic", 1), ("classic", 3), ("double", 3)]
)
def test_classic_and_handicap_do_not_create_false_second_stone(mode, handicap):
    decisions, _ = trajectory(mode=mode, handicap=handicap, actions=tuple(range(7)))
    labels = _future_policy_targets(decisions)
    for d, label in zip(decisions, labels, strict=True):
        if mode == "classic" or d.position.opening:
            assert "second_stone" not in label
    if handicap == 3:
        assert all(labels[index]["opponent_reply_ply"] == 3 for index in range(3))


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_actual_pie_swap_is_explicit_and_recoloring_keeps_player_perspective(mode):
    nodes = get_topology(4).n
    decisions, _ = trajectory(mode=mode, pie=True, actions=(0, nodes, 1, 2, 3))
    rows = samples_from(decisions)
    assert rows[0].opponent_reply[-1] == 1
    assert rows[0].opponent_reply[:-1].sum() == 0
    assert rows[1].second_stone is None
    assert rows[1].opponent_reply_ply == 2
    assert rows[1].to_move == 1 and rows[2].to_move == 0
    assert rows[2].stones[0] == 1 and rows[2].swapped
    _validate_future_policies(rows)


@pytest.mark.native
def test_missing_next_search_policy_does_not_skip_to_later_opponent_turn():
    decisions, _ = trajectory(actions=tuple(range(8)))
    decisions[3].policy = None
    labels = _future_policy_targets(decisions)
    assert "opponent_reply" not in labels[1]
    assert "opponent_reply" not in labels[2]
    decisions[2].ply = 99
    with pytest.raises(ValueError, match="contiguous"):
        _future_policy_targets(decisions)


@pytest.mark.native
def test_replay_capability_roundtrip_d5_padding_and_cached_components(tmp_path):
    nodes = get_topology(4).n
    decisions, state = trajectory(pie=True, actions=(0, nodes, 1, 2, 3))
    state.apply_many([0] * (nodes - 4), list(range(4, nodes)))
    final = score_results_from_native(state.score_data())[0]
    rows = samples_from(decisions, final)
    expected_peries = [p.peries for p in final.players]
    expected_stars = [p.stars for p in final.players]
    assert rows[0].final_peries.tolist() == expected_peries
    assert rows[0].final_stars.tolist() == expected_stars
    path = write_replay_shard(tmp_path / "aux.npz", rows)
    restored = read_replay_shard(path)
    assert decode_replay_shard(path).metadata["auxiliary_targets_version"] == 1
    for a, b in zip(rows, restored, strict=True):
        for name in ("opponent_reply", "second_stone", "final_peries", "final_stars"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    misses = _final_components.cache_info().misses
    augmented = augment_sample(restored[0], D5Transform(1, True))
    assert _final_components.cache_info().misses == misses
    assert augmented.opponent_reply[-1] == 1
    assert augmented.final_stars.tolist() == expected_stars
    permutation = get_topology(4).d5_permutation(1, True).numpy()
    np.testing.assert_array_equal(
        augmented.final_ownership[permutation], restored[0].final_ownership
    )
    larger, _ = trajectory(rings=6, pie=True, actions=(0, get_topology(6).n, 1))
    batch = collate_replay_samples(
        [restored[0], restored[1], samples_from(larger)[0]], prefer_native=False
    )
    assert batch.targets.opponent_reply.shape == (3, get_topology(6).n + 1)
    assert batch.targets.opponent_reply[0, -1] == 1
    assert batch.targets.opponent_reply[0, nodes] == 0
    assert batch.targets.final_peries.tolist()[:2] == [
        expected_peries,
        expected_peries[::-1],
    ]
    assert batch.targets.final_quarks.tolist()[:2] == [
        [p.quarks for p in final.players],
        [p.quarks for p in reversed(final.players)],
    ]
    assert batch.targets.final_peries_mask.tolist() == [True, True, False]
    assert _final_components.cache_info().misses == misses


@pytest.mark.native
def test_historical_v5_derives_counts_but_keeps_future_targets_unavailable(tmp_path):
    decisions, state = trajectory()
    nodes = state.node_count
    state.apply_many([0] * (nodes - 5), list(range(5, nodes)))
    final = score_results_from_native(state.score_data())[0]
    path = write_replay_shard(tmp_path / "old.npz", samples_from(decisions, final))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            k: archive[k]
            for k in archive.files
            if k not in _AUXILIARY_SAMPLE_ARRAY_NAMES
        }
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata.pop("auxiliary_targets_version")
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    rows = read_replay_shard(path)
    batch = collate_replay_samples(rows)
    assert rows[0].schema_version == 5
    assert all(row.opponent_reply is None and row.second_stone is None for row in rows)
    assert not batch.targets.opponent_reply_mask.any()
    assert not batch.targets.second_stone_mask.any()
    assert batch.targets.final_stars_mask.all()
    assert rows[0].final_stars.tolist() == [p.stars for p in final.players]


@pytest.mark.native
def test_auxiliary_capability_corruption_fails_closed(tmp_path):
    decisions, _ = trajectory()
    rows = samples_from(decisions)
    path = write_replay_shard(tmp_path / "bad.npz", rows)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays.pop("opponent_reply_swap")
    np.savez(path, **arrays)
    with pytest.raises(ReplaySchemaError, match="missing arrays"):
        decode_replay_shard(path)
    wrong = rows[1].second_stone.copy()
    wrong[:] = 0
    wrong[0] = 1  # Already occupied at the source.
    with pytest.raises(ReplaySchemaError, match="occupied"):
        replace(rows[1], second_stone=wrong)
    wrong = np.roll(rows[0].opponent_reply[:-1], 1)
    wrong[0] = 0
    wrong /= wrong.sum()
    tampered = replace(rows[0], opponent_reply=np.append(wrong, np.float32(0)))
    with pytest.raises(ValueError, match="recorded later"):
        _validate_future_policies([tampered, *rows[1:]])


@pytest.mark.native
def test_stored_component_counts_require_available_spatial_labels(tmp_path):
    decisions, _ = trajectory()
    path = write_replay_shard(tmp_path / "masked-counts.npz", samples_from(decisions))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["final_peries"][0] = [1, 1]
    np.savez(path, **arrays)
    with pytest.raises(ReplaySchemaError, match="requires available spatial"):
        read_replay_shard(path)


@pytest.mark.native
def test_growing_prefix_digests_allow_only_newly_observed_future_labels():
    decisions, _ = trajectory()
    first = samples_from(decisions[:2])
    later = samples_from(decisions[:4])
    assert first[1].second_stone is None and later[1].second_stone is not None
    assert (
        _digest_rows(first, immutable=True)[-1]
        == _digest_rows(later, immutable=True)[2]
    )
    assert (
        _digest_rows(first, immutable=False)[-1]
        == _digest_rows(later[:2], immutable=False, future_limit=2)[-1]
    )
    no_aux = [
        replace(
            s,
            opponent_reply=None,
            opponent_reply_ply=-1,
            second_stone=None,
            second_stone_ply=-1,
        )
        for s in first
    ]
    assert (
        _digest_rows(no_aux, immutable=False)[-1]
        == _digest_rows(first, immutable=False, include_auxiliary=False)[-1]
    )


@pytest.mark.native
def test_live_prefix_enrichment_rolling_stop_is_immutable_and_never_recredits(tmp_path):
    from test_live_policy_publication import credit, setup
    from startrain.replay_publication import validate_publications

    actor, store, identity = setup(tmp_path, games=7, rolling=True)
    snapshots = []
    original = store.append_game_revision

    def capture(samples, **kwargs):
        receipt = original(samples, **kwargs)
        snapshots.append(read_replay_shard(receipt.record.path))
        return receipt

    store.append_game_revision = capture
    try:
        actor.run(
            stop_requested=lambda: (
                actor.refilled_games > 0
                and actor.full_decisions + actor.fast_decisions
                > actor.completed_decisions + 5
            )
        )
        previous = {}
        enrichments = 0
        for rows in snapshots:
            _validate_future_policies(rows)
            for row in rows:
                key = (row.game_id, row.ply)
                for name in ("opponent_reply", "second_stone"):
                    before = getattr(previous[key], name) if key in previous else None
                    after = getattr(row, name)
                    if before is not None:
                        np.testing.assert_array_equal(before, after)
                    elif key in previous and after is not None:
                        enrichments += 1
                previous[key] = row
        assert enrichments > 0 and actor.refilled_games > 0
        assert credit(store, identity) == len(previous) == actor.persisted_decisions
        validate_publications(store.connection)
        assert any(
            rows[-1].opponent_reply is None and rows[-1].second_stone is None
            for rows in snapshots
        )
    finally:
        store.close()


@pytest.mark.native
def test_clinch_official_completion_counts_and_no_unobserved_future():
    from test_selfplay_streaming import Sink, actor, config

    native = pytest.importorskip("star_native")
    sink = Sink()
    worker = actor(
        native, config(games=2, batch_size=2, clinch_finalization="loser-fill"), sink
    )
    worker.run()
    clinched = [
        s for s in sink.samples if "final=clinch-loser-fill" in s.search_provenance
    ]
    assert clinched
    for row in clinched:
        assert row.final_peries is not None and row.final_stars is not None
        for player in (0, 1):
            assert row.final_scores[player] == row.final_peries[player] + int(
                row.final_quarks[player] >= 3
            ) + 2 * (int(row.final_stars[1 - player]) - int(row.final_stars[player]))
    for game in {s.game_id for s in sink.samples}:
        last = max((s for s in sink.samples if s.game_id == game), key=lambda s: s.ply)
        assert last.opponent_reply is None and last.second_stone is None


@pytest.mark.native
@pytest.mark.parametrize("omit_swap", [False, True])
def test_interrupted_filtered_policy_rows_remap_future_destinations(omit_swap):
    from startrain.selfplay import GameVariant
    from test_selfplay_streaming import Sink, actor, config

    native = pytest.importorskip("star_native")
    actions = (0, get_topology(4).n, 1, 2, 3, 4) if omit_swap else tuple(range(6))
    decisions, _ = trajectory(pie=omit_swap, actions=actions)
    decisions[1].policy = None
    sink = Sink()
    worker = actor(native, config(games=1, batch_size=1), sink)
    trajectories = [decisions]
    count = worker._salvage_interrupted_policy(
        trajectories,
        [0],
        [("streaming-test", 7, "streaming-test")],
        ["interrupted-game"],
        GameVariant(mode="double", pie=omit_swap),
    )
    assert count == 5
    assert [s.ply for s in sink.samples] == list(range(5))
    for sample in sink.samples:
        for name in ("opponent_reply", "second_stone"):
            if getattr(sample, name) is not None:
                destination = getattr(sample, f"{name}_ply")
                assert sample.ply < destination < count
    _validate_future_policies(sink.samples)
    if omit_swap:
        assert sink.samples[0].opponent_reply is None
    else:
        assert sink.samples[1].opponent_reply_ply == 2
        assert sink.samples[2].second_stone_ply == 3
