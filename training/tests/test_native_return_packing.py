"""Native CSR return packing agrees with the independently masked tensor path."""

import pytest

from test_native_inference_keys import GeneralRequest, adapter


@pytest.mark.native
@pytest.mark.parametrize("rings", [4, 6, 8, 10])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("native_groups", [(True, True), (True, False), (False, True)])
def test_batched_uneven_legal_rows_preserve_order_details_and_cache(
    rings, warm, native_groups
):
    native = pytest.importorskip("deltrel_native")
    requests = []
    for lengths in ((0, 1, 7), (4, 0)):
        states = native.StateBatch(rings, len(lengths))
        for action in range(max(lengths)):
            rows = [row for row, length in enumerate(lengths) if length > action]
            states.apply_many(rows, [action] * len(rows))
        requests.append(native.SearchBatch(states, simulations=1).root_requests())
    fast, reference = adapter(), adapter()
    weights = (0.1, 0.7)
    if warm:
        # The empty board repeats across requests; the second request also
        # introduces a miss after warming only the first group.
        fast.evaluate(requests[0])
        reference.evaluate(GeneralRequest(requests[0]))
    native_prepared = [
        fast.prepare_requests(
            request if is_native else GeneralRequest(request),
            score_utility_weight=weight,
        )
        for request, weight, is_native in zip(
            requests, weights, native_groups, strict=True
        )
    ]
    tensor_prepared = [
        reference.prepare_requests(GeneralRequest(request), score_utility_weight=weight)
        for request, weight in zip(requests, weights, strict=True)
    ]
    expected = reference.evaluate_prepared(
        tensor_prepared, include_details=(False, True)
    )
    actual = fast.evaluate_prepared(native_prepared, include_details=(False, True))
    assert actual == expected
    # A caller owns its lists; modifying a returned response cannot poison the
    # immutable raw-prediction cache or a different request in the same batch.
    actual[0][0].policy_logits[0] = 123456.0
    repeated = fast.evaluate_prepared(native_prepared, include_details=(False, True))
    assert repeated == expected
