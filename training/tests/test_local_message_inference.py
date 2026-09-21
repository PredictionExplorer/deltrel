from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from deltreltrain.features import encode_batch
from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig
from deltreltrain.local_message_inference import _reference_mean, _source_class_mean
from deltreltrain.model import GraphResTNet, ModelConfig
from test_model import position, randomize_v3_parameters
from test_inference_efficiency import encoded_requests


@pytest.fixture(autouse=True)
def isolate_experimental_compilation():
    # These cases compile several modes, dtypes and shapes of the same forward
    # code object. Keep their variants out of subsequent training tests, and
    # start each experiment independently without changing Dynamo's limits.
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


def model():
    torch.manual_seed(127)
    result = GraphResTNet(
        ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1)
    ).eval()
    randomize_v3_parameters(result)
    return result


@pytest.mark.parametrize("rings", [(4, 4), (6, 6), (8, 8), (10, 10), (4, 6), (6, 10)])
@pytest.mark.parametrize("mode", ["project-first", "source-class"])
@pytest.mark.parametrize("bf16", [False, True])
def test_all_heads_match_without_changing_checkpoint(rings, mode, bf16):
    baseline = model()
    optimized = deepcopy(baseline)
    state = deepcopy(optimized.state_dict())
    optimized.set_local_message_execution(mode)
    batch = encode_batch([position(r) for r in rings])
    with (
        torch.inference_mode(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16),
    ):
        expected = baseline(*batch.model_args())
        actual = optimized(*batch.model_args())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    assert state.keys() == optimized.state_dict().keys()
    for key, value in optimized.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert optimized.parameter_count() == baseline.parameter_count()
    assert baseline.inference_execution_signature == ("graph-inference-v1", False)
    assert (
        optimized.inference_execution_signature
        != baseline.inference_execution_signature
    )


def test_boundary_rejects_training_gradients_and_unsupported_operator():
    network = model()
    batch = encode_batch([position(4)])
    network.set_local_message_execution("source-class")
    with pytest.raises(ValueError, match="requires inference"):
        network(*batch.model_args())
    network.train()
    with torch.no_grad(), pytest.raises(ValueError, match="requires inference"):
        network(*batch.model_args())
    network.set_local_message_execution("baseline")
    network(*batch.model_args()).outcome_logits.sum().backward()
    assert any(p.grad is not None for p in network.parameters())
    with pytest.raises(ValueError):
        network.set_local_message_execution("unsupported")
    gated = GraphResTNet(replace(network.config, local_operator="source_gated")).eval()
    with pytest.raises(ValueError, match="mean"):
        gated.set_local_message_execution("source-class")


@pytest.mark.parametrize("mode", ["project-first", "source-class"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_compact_gather_preserves_projected_messages(mode, bf16, compiled):
    network = model()
    network.set_local_message_execution(mode)
    batch = encode_batch([position(6), position(10)])
    with (
        torch.inference_mode(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16),
    ):
        expected = network(*batch.model_args())
        network.set_compact_inference_gather(True)
        runner = (
            torch.compile(network, backend="aot_eager", fullgraph=True)
            if compiled
            else network
        )
        actual = runner(*batch.model_args())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_source_class_operator_handles_broadcast_and_empty_neighbor_rows():
    table = torch.arange(2 * 5 * 3 * 8, dtype=torch.float64).reshape(2, 5, 3, 8)
    indices = torch.tensor(
        [[0, 2, 4], [1, 3, 0], [2, 4, 0], [3, 0, 0], [4, 0, 0]]
    ).expand(2, -1, -1)
    kinds = torch.tensor([[0, 1, 2]]).expand(2, 5, -1)
    mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
            [True, False, False],
            [True, False, True],
            [True, True, True],
        ]
    ).expand(2, -1, -1)
    with torch.inference_mode():
        actual = _source_class_mean(table, indices, kinds, mask)
    expected = torch.zeros(2, 5, 8, dtype=torch.float64)
    for b in range(2):
        for n in range(5):
            selected = [
                table[b, indices[b, n, k], kinds[b, n, k]]
                for k in range(3)
                if mask[b, n, k]
            ]
            if selected:
                expected[b, n] = torch.stack(selected).mean(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_source_class_path_compiles_and_invalidates_adapter_prediction_cache(
    monkeypatch,
):
    network = model()
    batch = encode_batch([position(4), position(4)])
    network.set_local_message_execution("source-class")
    compiled = torch.compile(network, backend="aot_eager", fullgraph=True)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = network(*batch.model_args())
        actual = compiled(*batch.model_args())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    adapter = GraphInferenceAdapter(
        network,
        config=InferenceConfig(cache_max_entries=8, cache_max_bytes=100000),
        model_version="sha256-local-message-test",
    )
    monkeypatch.setattr(
        "deltreltrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    requests = encoded_requests(batch)
    adapter.evaluate(requests)
    before = adapter.metrics_snapshot()
    adapter.evaluate(requests)
    assert adapter.metrics_snapshot().cache_hits > before.cache_hits
    namespace = adapter.namespace
    network.set_local_message_execution("baseline")
    assert adapter.namespace != namespace
    calls = adapter.metrics_snapshot().neural_calls
    adapter.evaluate(requests)
    assert adapter.metrics_snapshot().neural_calls == calls + 1
    adapter.close()


@pytest.mark.cuda
@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_cuda_fused_reduction_and_compiled_model_match_reference(ring):
    baseline = model().cuda()
    optimized = deepcopy(baseline)
    optimized.set_local_message_execution("source-class")
    inputs = encode_batch([position(ring), position(ring)]).to("cuda")
    compiled = torch.compile(optimized, fullgraph=True, dynamic=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = baseline(*inputs.model_args())
        actual = compiled(*inputs.model_args())
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)
    table = torch.randn(2, inputs.max_nodes, 3, 192, device="cuda")
    expected_mean = _reference_mean(
        table, inputs.neighbor_index, inputs.neighbor_edge_type, inputs.neighbor_mask
    )
    actual_mean = _source_class_mean(
        table, inputs.neighbor_index, inputs.neighbor_edge_type, inputs.neighbor_mask
    )
    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-5, atol=1e-6)
