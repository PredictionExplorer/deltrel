"""Prediction-only clearing must not invalidate safe warmed execution state."""

from types import SimpleNamespace

import pytest

from test_native_inference_keys import adapter


def test_prediction_only_clear_forces_recomputation_preserves_namespace_and_graphs():
    native = pytest.importorskip("deltrel_native")
    inference = adapter()
    request = native.SearchBatch(native.StateBatch(4, 2), simulations=1).root_requests()
    first = inference.evaluate(request)
    assert inference.model.rows == [1]
    assert inference.evaluate(request).values == first.values
    assert inference.model.rows == [1]
    namespace = inference._cache_namespace
    topology = dict(inference._topology_cache)
    graph_state = SimpleNamespace(
        clear=lambda: pytest.fail("prediction reset cleared graphs")
    )
    weight_stamp = object()
    inference._graphs = graph_state
    inference._graph_weight_stamps = weight_stamp
    inference.model.clear_inference_caches = lambda: pytest.fail(
        "prediction reset cleared model caches"
    )
    inference.clear_prediction_cache()
    assert inference._cache_namespace == namespace
    assert inference._graphs is graph_state
    assert inference._graph_weight_stamps is weight_stamp
    assert inference._topology_cache == topology
    inference._graphs = None
    second = inference.evaluate(request)
    assert second == first
    assert inference.model.rows == [1, 1]
    assert inference._cache_namespace == namespace
    del inference.model.clear_inference_caches
    inference.close()


def test_full_reset_still_clears_all_execution_state():
    inference = adapter()
    cleared = []
    inference._graphs = SimpleNamespace(clear=lambda: cleared.append("graphs"))
    inference._graph_weight_stamps = object()
    inference._cache_namespace = ("old-model",)
    inference.model.clear_inference_caches = lambda: cleared.append("model")
    inference.clear_inference_cache()
    assert cleared == ["graphs", "model"]
    assert inference._cache_namespace is None
    assert inference._graph_weight_stamps is None
    inference._graphs = None
    inference.close()
