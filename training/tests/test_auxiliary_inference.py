from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from deltrelserve.app import create_app
from deltrelserve.runtime import (
    AnalysisError,
    NativeAnalysisService,
    _auxiliary_heads_ready,
    _auxiliary_payload,
)
from deltrelserve.schemas import AnalyzeRequest, AuxiliaryPredictions
from deltreltrain.auxiliary_inference import AuxiliaryPrediction
from deltreltrain.auxiliary_upgrade import AUXILIARY_LOSSES
from deltreltrain.features import encode_batch
from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig
from deltreltrain.model import GraphResTNet, ModelConfig
from test_inference_efficiency import encoded_requests, position
from test_deltrelserve import (
    FakeEvaluator,
    FakeSearchBatch,
    FakeService,
    request_payload,
    server_config,
)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        "deltreltrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    model = GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=1,
            attention_heads=4,
            kv_heads=1,
            auxiliary_predictions=True,
        )
    )
    return GraphInferenceAdapter(
        model,
        model_identity="immutable-test",
        model_version="test",
        config=InferenceConfig(
            cache_max_entries=20, cache_max_bytes=1_000_000, deduplicate=True
        ),
    )


def test_root_heads_are_cached_and_leaves_skip_them(adapter):
    calls = []
    adapter.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["include_auxiliary"]),
        with_kwargs=True,
    )
    request = encoded_requests(encode_batch([position()]))
    leaf = adapter.evaluate(request)
    assert calls == [False]
    detailed = adapter.evaluate_detailed(request)
    assert calls == [False, True]
    assert detailed.response == leaf
    assert detailed.auxiliary_predictions is not None
    assert len(detailed.auxiliary_predictions) == 1
    prediction = detailed.auxiliary_predictions[0]
    assert all(0 <= value <= 20 for value in prediction.final_shores)
    assert all(0 <= value <= 10 for value in prediction.final_networks)
    assert sum(prediction.opponent_reply_probabilities) == pytest.approx(1)
    assert sum(prediction.second_stone_probabilities) == pytest.approx(1)
    assert adapter.evaluate_detailed(request) == detailed
    assert adapter.evaluate(request) == leaf
    assert calls == [False, True]
    # Returned mutable summaries cannot contaminate immutable cache records.
    prediction.final_shores[0] = 999
    assert (
        adapter.evaluate_detailed(request).auxiliary_predictions[0].final_shores[0]
        <= 20
    )


def test_mixed_batched_details_and_cache_routing(adapter):
    a = adapter.prepare_requests(
        encoded_requests(encode_batch([position(pda=0)]), token_start=10)
    )
    b = adapter.prepare_requests(
        encoded_requests(encode_batch([position(pda=2)]), token_start=20)
    )
    first = adapter.evaluate_prepared((a, b), include_details=(False, True))
    assert first[0][1] is None
    assert first[1][1].auxiliary_predictions is not None
    assert first[0][0].tokens == [10]
    assert first[1][0].tokens == [20]
    assert adapter.evaluate_prepared((b,), include_details=(True,))[0] == first[1]
    assert adapter.evaluate_prepared((a,), include_details=(False,))[0] == first[0]
    assert adapter.metrics_snapshot().neural_calls == 1


def test_direct_details_agree_with_cached_details(adapter):
    request = encoded_requests(encode_batch([position()]))
    direct = GraphInferenceAdapter(adapter.model)
    expected = direct.evaluate_detailed(request)
    actual = adapter.evaluate_detailed(request)
    assert actual.response == expected.response
    assert actual.auxiliary_predictions == expected.auxiliary_predictions
    assert actual.score_expectations == pytest.approx(
        expected.score_expectations, abs=1e-5
    )


def test_invalid_auxiliary_output_is_rejected(adapter):
    request = encoded_requests(encode_batch([position()]))
    with torch.no_grad():
        adapter.model.final_networks_head.bias[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite auxiliary"):
        adapter.evaluate_detailed(request)


def prediction() -> AuxiliaryPrediction:
    return AuxiliaryPrediction(
        final_shores=[12.0, 8.0],
        final_networks=[2.0, 3.0],
        final_capes=[3.5, 1.5],
        cape_bonus_probability=[0.8, 0.2],
        opponent_reply_probabilities=[0.0] * 50 + [1.0],
        second_stone_probabilities=[0.0, 1.0] + [0.0] * 48,
    )


def test_final_player_mapping_and_future_move_applicability():
    opening = AnalyzeRequest.model_validate({**request_payload(), "pie": True})
    result = _auxiliary_payload(prediction(), opening)
    AuxiliaryPredictions.model_validate(result)
    assert result["opponent_reply"] == {
        "player": 1,
        "kind": "swap",
        "node": None,
        "probability": 1.0,
    }
    assert result["second_stone"] is None
    payload = request_payload()
    payload.update(
        stones=[0] + [-1] * 49, to_move=1, opening=False, moves_left=2, history=None
    )
    request = AnalyzeRequest.model_validate(payload)
    result = _auxiliary_payload(prediction(), request)
    AuxiliaryPredictions.model_validate(result)
    assert result["final_counts"][0]["shores"] == 8
    assert result["final_counts"][1]["shores"] == 12
    assert result["second_stone"] == {
        "player": 1,
        "kind": "place",
        "node": 1,
        "probability": 1.0,
    }
    assert result["opponent_reply"]["kind"] == "place"


@pytest.mark.parametrize("values", ([float("nan"), 0], [21, 0], [1]))
def test_invalid_final_counts_are_rejected(values):
    with pytest.raises(AnalysisError, match="invalid final component"):
        _auxiliary_payload(
            replace(prediction(), final_shores=values),
            AnalyzeRequest.model_validate(request_payload()),
        )


def test_pie_swap_never_forecasts_a_second_stone_for_the_ended_turn():
    payload = request_payload()
    payload.update(
        stones=[0] + [-1] * 49,
        to_move=1,
        opening=False,
        moves_left=2,
        pie=True,
        swap_available=True,
        history=None,
    )
    request = AnalyzeRequest.model_validate(payload)
    assert _auxiliary_payload(prediction(), request)["second_stone"] is not None
    swapped = _auxiliary_payload(prediction(), request, swap_recommended=True)
    assert swapped["second_stone"] is None
    assert swapped["opponent_reply"]["player"] == 0


def test_schema_rejects_incomplete_or_misidentified_predictions():
    result = _auxiliary_payload(
        prediction(), AnalyzeRequest.model_validate(request_payload())
    )
    with pytest.raises(ValidationError):
        AuxiliaryPredictions.model_validate(
            {**result, "final_counts": result["final_counts"][:1]}
        )
    with pytest.raises(ValidationError):
        AuxiliaryPredictions.model_validate(
            {
                **result,
                "opponent_reply": {
                    "player": 0,
                    "kind": "place",
                    "node": 2,
                    "probability": 0.5,
                },
            }
        )


def test_legacy_model_reports_no_predictions_only_when_requested():
    roots = SimpleNamespace(tokens=[1], legal_offsets=[0, 2], legal_actions=[0, 1])
    evaluator = FakeEvaluator()
    arguments = dict(
        evaluator=evaluator,
        reload_ms=0,
        search_ms=0,
        total_ms=0,
        node_count=50,
    )
    request = AnalyzeRequest.model_validate(request_payload())
    result = NativeAnalysisService._response_payload(
        FakeSearchBatch(None).results(),
        evaluator.evaluate_detailed(roots),
        request=request,
        **arguments,
    )
    assert "predictions" not in result
    result = NativeAnalysisService._response_payload(
        FakeSearchBatch(None).results(),
        evaluator.evaluate_detailed(roots),
        request=request.model_copy(update={"include_predictions": True}),
        **arguments,
    )
    assert result["predictions"] is None


@pytest.mark.parametrize(
    "step, extra, ready",
    [
        (0, {}, False),
        (1, {}, True),
        (100, {"auxiliary_upgrade": {"source_step": 100}}, False),
        (101, {"auxiliary_upgrade": {"source_step": 100}}, False),
        (101, {"auxiliary_upgrade": {}}, False),
    ],
)
def test_new_heads_are_hidden_until_they_have_training_updates(step, extra, ready):
    config = ModelConfig(auxiliary_predictions=True)
    assert _auxiliary_heads_ready(config, {"step": step, "extra": extra}) is ready
    assert not _auxiliary_heads_ready(
        replace(config, auxiliary_predictions=False), {"step": step, "extra": extra}
    )


def test_untrained_heads_are_never_exposed_as_learned_counts():
    roots = SimpleNamespace(tokens=[1], legal_offsets=[0, 2], legal_actions=[0, 1])
    evaluator = FakeEvaluator()
    detailed = replace(
        evaluator.evaluate_detailed(roots), auxiliary_predictions=[prediction()]
    )
    request = AnalyzeRequest.model_validate(
        {**request_payload(), "include_predictions": True}
    )
    result = NativeAnalysisService._response_payload(
        FakeSearchBatch(None).results(),
        detailed,
        evaluator=evaluator,
        reload_ms=0,
        search_ms=0,
        total_ms=0,
        node_count=50,
        request=request,
        auxiliary_ready=False,
    )
    assert result["predictions"] is None


def test_migrated_head_readiness_requires_actual_supervision_for_every_head():
    config = ModelConfig(auxiliary_predictions=True)
    supervision = {name: 101 for name in AUXILIARY_LOSSES}
    metadata = {
        "step": 105,
        "extra": {
            "auxiliary_upgrade": {"source_step": 100},
            "auxiliary_supervision": supervision,
        },
    }
    assert _auxiliary_heads_ready(config, metadata)
    for missing in AUXILIARY_LOSSES:
        partial = {name: step for name, step in supervision.items() if name != missing}
        assert not _auxiliary_heads_ready(
            config,
            {
                **metadata,
                "extra": {**metadata["extra"], "auxiliary_supervision": partial},
            },
        )


@pytest.mark.parametrize("invalid_step", [None, True, "101", 101.0, -1, 100, 106])
def test_migrated_readiness_rejects_malformed_or_unsupervised_steps(invalid_step):
    supervision = {name: 101 for name in AUXILIARY_LOSSES}
    supervision["opponent_reply"] = invalid_step
    assert not _auxiliary_heads_ready(
        ModelConfig(auxiliary_predictions=True),
        {
            "step": 105,
            "extra": {
                "auxiliary_upgrade": {"source_step": 100},
                "auxiliary_supervision": supervision,
            },
        },
    )


@pytest.mark.parametrize(
    "upgrade", [None, False, 100, {"source_step": True}, {"source_step": "100"}]
)
def test_malformed_upgrade_marker_cannot_use_born_auxiliary_fallback(upgrade):
    assert not _auxiliary_heads_ready(
        ModelConfig(auxiliary_predictions=True),
        {
            "step": 105,
            "extra": {
                "auxiliary_upgrade": upgrade,
                "auxiliary_supervision": {name: 101 for name in AUXILIARY_LOSSES},
            },
        },
    )


def test_legacy_http_clients_keep_their_exact_response_shape(tmp_path):
    class OptionalPredictionsService(FakeService):
        def analyze(self, request, cancellation):
            result = super().analyze(request, cancellation)
            if request.include_predictions:
                result["predictions"] = None
            return result

    with TestClient(
        create_app(server_config(tmp_path), service=OptionalPredictionsService())
    ) as client:
        legacy = client.post("/v2/analyze", json=request_payload())
        assert legacy.status_code == 200
        assert "predictions" not in legacy.json()
        modern = client.post(
            "/v2/analyze", json={**request_payload(), "include_predictions": True}
        )
        assert modern.status_code == 200
        assert modern.json()["predictions"] is None
