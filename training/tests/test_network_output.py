from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from deltrelserve.app import create_app
from deltrelserve.network_output import network_output_payload
from deltrelserve.runtime import NativeAnalysisService
from deltrelserve.schemas import AnalyzeRequest, NetworkOutput
from deltreltrain.features import encode_batch
from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig
from deltreltrain.inference_batching import BoundedInferenceBroker
from deltreltrain.model import GraphResTNet, ModelConfig
from deltreltrain.native import load_deltrel_native
from test_deltrelserve import FakeService, request_payload, server_config


def adapter(auxiliary: bool = True) -> GraphInferenceAdapter:
    torch.manual_seed(62)
    return GraphInferenceAdapter(
        GraphResTNet(
            ModelConfig(
                width=16,
                rrt_groups=1,
                attention_heads=4,
                kv_heads=1,
                auxiliary_predictions=auxiliary,
            )
        ).eval(),
        model_identity="root-diagnostic-test",
        model_version="root-diagnostic-test",
        model_step=17,
        config=InferenceConfig(cache_max_entries=20, cache_max_bytes=1_000_000),
    )


@pytest.mark.parametrize("rings", [4, 6, 8, 10])
def test_all_heads_are_complete_masked_normalized_and_bounded(rings: int) -> None:
    payload = request_payload()
    payload.update(
        rings=rings,
        stones=[-1] * (5 * rings * (rings + 1) // 2),
        include_network_output=True,
    )
    request = AnalyzeRequest.model_validate(payload)
    evaluator = adapter()
    before = {
        name: value.clone() for name, value in evaluator.model.state_dict().items()
    }
    raw = evaluator.evaluate_network_output(request.position())
    result = network_output_payload(
        raw, request, auxiliary_ready=True, swap_recommended=False
    )
    parsed = NetworkOutput.model_validate(result)
    assert parsed.node_count == len(request.stones)
    assert parsed.auxiliary_status == "ready"
    assert len(result["heads"]) == 11
    assert len(json.dumps(result)) < 256 * 1024
    assert parsed.heads.second_stone is not None
    assert not parsed.heads.second_stone.applicable
    assert parsed.heads.opponent_reply is not None
    # The raw swap slot is preserved even where swapping is not applicable.
    assert parsed.heads.opponent_reply.mask[-1]
    assert parsed.heads.opponent_reply.logits[-1] is not None
    assert parsed.heads.final_shores is not None
    for row in range(2):
        tail = slice(row * 51 + 5 * rings + 1, (row + 1) * 51)
        assert not any(parsed.heads.final_shores.mask[tail])
        assert all(value is None for value in parsed.heads.final_shores.logits[tail])
        assert all(
            value == 0 for value in parsed.heads.final_shores.probabilities[tail]
        )
    # Inspection is read-only and returns exactly the model's root logits.
    with torch.inference_mode():
        expected = evaluator.model(*encode_batch([request.position()]).model_args())
    assert torch.equal(raw["ownership_logits"], expected.ownership_logits)
    assert torch.equal(raw["alive_logits"], expected.alive_logits)
    for name, value in evaluator.model.state_dict().items():
        assert torch.equal(before[name], value)


def test_absent_and_untrained_heads_are_explicit() -> None:
    request = AnalyzeRequest.model_validate(request_payload())
    absent = network_output_payload(
        adapter(False).evaluate_network_output(request.position()),
        request,
        auxiliary_ready=False,
        swap_recommended=False,
    )
    assert absent["auxiliary_status"] == "absent"
    assert all(
        absent["heads"][name] is None
        for name in (
            "opponent_reply",
            "second_stone",
            "final_shores",
            "final_networks",
            "final_capes",
        )
    )
    untrained = network_output_payload(
        adapter().evaluate_network_output(request.position()),
        request,
        auxiliary_ready=False,
        swap_recommended=False,
    )
    assert untrained["auxiliary_status"] == "untrained"
    assert untrained["heads"]["final_shores"] is not None


def test_masks_conditional_forecasts_and_invalid_outputs() -> None:
    payload = request_payload()
    payload.update(
        opening=False,
        to_move=1,
        moves_left=2,
        history={
            "current_turn": [],
            "previous_turn": [0],
            "own_previous_turn": [],
            "handicap_stones": [0],
        },
    )
    payload["stones"][0] = 0
    request = AnalyzeRequest.model_validate(payload)
    raw = adapter().evaluate_network_output(request.position())
    result = network_output_payload(
        raw, request, auxiliary_ready=True, swap_recommended=False
    )
    for name in ("policy", "soft_policy", "opponent_reply", "second_stone"):
        assert result["heads"][name]["mask"][0] is False
        assert result["heads"][name]["logits"][0] is None
        assert result["heads"][name]["probabilities"][0] == 0
    assert result["heads"]["second_stone"]["applicable"]
    swapped = network_output_payload(
        raw, request, auxiliary_ready=True, swap_recommended=True
    )
    assert not swapped["heads"]["second_stone"]["applicable"]
    malformed = json.loads(json.dumps(result))
    malformed["heads"]["ownership"]["probabilities"][0] = 2.0
    with pytest.raises(ValidationError):
        NetworkOutput.model_validate(malformed)
    malformed = json.loads(json.dumps(result))
    malformed["heads"]["outcome"]["logits"][0] += 10
    with pytest.raises(ValidationError):
        NetworkOutput.model_validate(malformed)
    raw["alive_logits"] = raw["alive_logits"].clone()
    raw["alive_logits"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        network_output_payload(
            raw, request, auxiliary_ready=True, swap_recommended=False
        )


def test_shared_broker_diagnostics_and_legacy_response_compatibility(tmp_path) -> None:
    request = AnalyzeRequest.model_validate(request_payload())
    evaluator = adapter()
    broker = BoundedInferenceBroker(
        max_batch_rows=2, max_pending_requests=2, max_wait_seconds=0
    )
    try:
        cohort = broker.cohort_adapter(evaluator)
        output = cohort.evaluate_network_output(request.position())
        assert output["ownership_logits"].shape == (1, 50, 3)
    finally:
        broker.shutdown()
    config = server_config(tmp_path)
    with TestClient(create_app(config, service=FakeService())) as client:
        response = client.post("/v2/move", json=request_payload())
        assert response.status_code == 200
        assert "network_output" not in response.json()
        assert client.get("/v2/health").json()["network_output_schema_version"] == 1


@pytest.mark.native
def test_diagnostics_do_not_change_native_search_or_selection(tmp_path) -> None:
    evaluator = adapter()

    @contextmanager
    def lease():
        yield SimpleNamespace(
            model=SimpleNamespace(
                evaluator=evaluator, search_cache=None, auxiliary_predictions_ready=True
            ),
            reload_ms=0.0,
        )

    service = NativeAnalysisService(
        server_config(tmp_path),
        native_module=load_deltrel_native(required=True),
        model_manager=SimpleNamespace(lease=lease),
    )
    payload = request_payload()
    regular = service.analyze(AnalyzeRequest.model_validate(payload), threading.Event())
    payload.update(include_network_output=True, include_predictions=True)
    inspected = service.analyze(
        AnalyzeRequest.model_validate(payload), threading.Event()
    )
    for name in (
        "action",
        "root_actions",
        "root_policy",
        "root_q",
        "root_visits",
        "outcome",
        "root_value",
    ):
        assert inspected[name] == regular[name]
    assert "network_output" not in regular
    NetworkOutput.model_validate(inspected["network_output"])
    assert inspected["network_output"]["heads"]["outcome"]["probabilities"][
        1
    ] == pytest.approx(inspected["outcome"]["win"], abs=1e-6)
