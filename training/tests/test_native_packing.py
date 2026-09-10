"""Lazy native exports retain exact snapshots and isolate writable buffers."""

from concurrent.futures import ThreadPoolExecutor
import gc

import pytest

from test_native_inference_keys import adapter


@pytest.fixture
def native():
    module = pytest.importorskip("star_native")
    request = module.SearchBatch(module.StateBatch(4, 1), simulations=1).root_requests()
    assert hasattr(request, "state_data_materialized"), "rebuild native packing"
    return module


STATE_FIELDS = (
    "rings",
    "node_count",
    "batch_size",
    "zero_bits",
    "one_bits",
    "legal_bits",
    "current_turn_bits",
    "previous_turn_bits",
    "own_previous_turn_bits",
    "handicap_bits",
    "hashes",
    "stones_placed",
    "to_move",
    "moves_left",
    "opening",
    "mid_turn",
    "terminal",
    "mode",
    "handicap",
    "pie",
    "pie_pending",
    "swap_available",
    "swapped",
    "turn_size",
    "current_turn_total",
    "turn_count",
)
FEATURE_FIELDS = (
    "rings",
    "node_features",
    "global_features",
    "node_mask",
    "legal_action_mask",
    "score_components",
    "node_owner",
    "alive_stones",
)


def snapshot(data):
    return {name: getattr(data, name) for name in STATE_FIELDS}


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
@pytest.mark.parametrize(
    "mode,handicap,pie",
    [
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 3, False),
        ("double", 4, False),
        ("classic", 1, True),
        ("double", 1, True),
    ],
)
def test_legacy_export_is_lazy_exact_and_detached_from_later_game_mutations(
    native, ring, mode, handicap, pie
):
    states = native.StateBatch(ring, 2, mode=mode, handicap=handicap, pie=pie)
    states.apply_many([0, 1], [0, 1])
    expected = snapshot(states.data())
    search = native.SearchBatch(states, simulations=1)
    request = search.root_requests()
    keys = request.inference_keys()
    assert (request.rings, request.node_count) == (ring, states.node_count)
    request.prefetch_features([1])
    request.selected_features([1, 0, 1])
    assert request.encoded_feature_rows == 2
    assert not request.state_data_materialized
    states.apply_many([0, 1], [2, 3])
    del search, states
    gc.collect()
    exported = request.states
    assert request.state_data_materialized
    assert snapshot(exported) == expected
    for name in STATE_FIELDS:
        value = getattr(exported, name)
        if isinstance(value, list) and value:
            value[:] = [0] * len(value)
    assert snapshot(request.states) == expected
    assert request.inference_keys() == keys


def test_trusted_cache_misses_and_hits_never_materialize_legacy_state_data(native):
    states = native.StateBatch(10, 3)
    inference = adapter()
    for _ in range(2):
        request = native.SearchBatch(
            states, simulations=1, pda_by_seat=[(0, 0), (0, 0), (2, -2)]
        ).root_requests()
        response = inference.evaluate(request)
        assert len(response.values) == 3
        assert not request.state_data_materialized
    assert request.encoded_feature_rows == 0
    assert inference.model.rows == [2]


def test_borrowed_feature_rows_preserve_selection_and_output_ownership(native):
    states = native.StateBatch(10, 3)
    states.apply_many([0, 1, 2], [0, 1, 2])
    request = native.SearchBatch(
        states, simulations=1, pda_by_seat=[(0, 0), (2, -2), (-1, 1)]
    ).root_requests()
    selected = request.selected_features([2, 0, 2])
    assert request.encoded_feature_rows == 2
    expected = {name: bytes(getattr(selected, name)) for name in FEATURE_FIELDS}
    assert not request.state_data_materialized
    full = request.features
    assert request.encoded_feature_rows == 3
    for name in FEATURE_FIELDS:
        original = bytes(getattr(full, name))
        # Every feature buffer has a uniform row stride in this homogeneous batch.
        stride = len(original) // 3
        assert expected[name] == b"".join(
            original[row * stride : (row + 1) * stride] for row in [2, 0, 2]
        )
        exposed = getattr(selected, name)
        exposed[:] = bytes(len(exposed))
    del selected, full, states
    gc.collect()
    again = request.selected_features([2, 0, 2])
    assert {name: bytes(getattr(again, name)) for name in FEATURE_FIELDS} == expected
    assert request.encoded_feature_rows == 3
    assert not request.state_data_materialized


def test_concurrent_state_exports_and_feature_reads_keep_one_snapshot(native):
    states = native.StateBatch(10, 4)
    states.apply_many([0, 1, 2, 3], [0, 1, 2, 3])
    expected = snapshot(states.data())
    request = native.SearchBatch(states, simulations=1).root_requests()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: (
                    snapshot(request.states),
                    bytes(request.selected_features([3, 1, 3]).node_features),
                ),
                range(12),
            )
        )
    assert all(state == expected for state, _ in results)
    assert len({features for _, features in results}) == 1
    assert request.encoded_feature_rows == 2
    assert request.state_data_materialized


def test_invalid_and_empty_selections_remain_nonmutating(native):
    request = native.SearchBatch(native.StateBatch(4, 2), simulations=1).root_requests()
    with pytest.raises(ValueError, match="out of range"):
        request.selected_features([0, 2])
    assert request.encoded_feature_rows == 0
    assert not request.state_data_materialized
    empty = request.selected_features([])
    assert empty.batch_size == empty.max_nodes == 0
    assert all(not bytes(getattr(empty, field)) for field in FEATURE_FIELDS)
    assert request.encoded_feature_rows == 0
    assert not request.state_data_materialized
    # Empty root batches arise naturally when every game row is terminal.
    states = native.StateBatch(4, 1)
    for node in range(states.node_count):
        states.apply_many([0], [node])
    terminal = native.SearchBatch(states, simulations=1).root_requests()
    assert len(terminal) == 0
    assert (terminal.rings, terminal.node_count) == (0, 0)
    assert terminal.states.batch_size == 0
    assert terminal.state_data_materialized
