"""Serve valid FP32 beliefs without relaxing the external probability contract."""

from dataclasses import replace
from types import SimpleNamespace
import math

import pytest

from deltrelserve.runtime import AnalysisError, NativeAnalysisService
from deltrelserve.schemas import AnalyzeResponse
from deltreltrain.contracts import SCORE_MARGIN_MIN
from test_deltrelserve import FakeEvaluator, FakeSearchBatch


def fixture():
    evaluator = FakeEvaluator()
    roots = SimpleNamespace(tokens=[1], legal_offsets=[0, 2], legal_actions=[0, 1])
    return (
        evaluator,
        FakeSearchBatch(None).results(),
        evaluator.evaluate_detailed(roots),
    )


def payload(evaluator, results, detailed):
    return NativeAnalysisService._response_payload(
        results,
        detailed,
        evaluator=evaluator,
        reload_ms=0,
        search_ms=0,
        total_ms=0,
        node_count=50,
    )


@pytest.mark.parametrize("drift", [-8e-6, 8e-6])
def test_near_unit_fp32_beliefs_are_normalized_for_wire_and_browser(drift):
    evaluator, results, original = fixture()
    scores = [0.0] * 303
    scores[-1] = 0.75 * (1 + drift)
    scores[-2] = 0.25 * (1 + drift)
    outcome = [0.2 * (1 + drift), 0.8 * (1 + drift)]
    detailed = replace(
        original,
        outcome_probabilities=[outcome],
        outcome_values=[outcome[1] - outcome[0]],
        score_probabilities=[scores],
        score_expectations=[
            math.fsum(p * (i + SCORE_MARGIN_MIN) for i, p in enumerate(scores))
        ],
    )
    results.policy_target = [0.25 * (1 + drift), 0.75 * (1 + drift)]
    result = payload(evaluator, results, detailed)
    wire = AnalyzeResponse.model_validate({**result, "request_id": "rounding"})
    assert math.fsum(wire.score_belief.probabilities) == pytest.approx(1.0, abs=1e-15)
    assert wire.score_belief.expected_margin == pytest.approx(150.75, abs=1e-12)
    assert wire.outcome.loss + wire.outcome.win == pytest.approx(1.0, abs=1e-15)
    assert wire.value == wire.outcome.win - wire.outcome.loss
    assert sum(wire.root_policy) == pytest.approx(1.0, abs=1e-15)
    assert wire.search_value == original.response.values[0]
    assert wire.root_value == results.root_values[0]
    assert wire.root_visits == results.visits
    assert wire.action.code == results.selected_actions[0]
    # Response normalization never mutates the cached/raw prediction lists.
    assert detailed.score_probabilities[0] == scores
    assert math.fsum(scores) == pytest.approx(1 + drift)


@pytest.mark.parametrize("field", ["outcome", "scores", "policy"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -0.01, 0.0, 1.1])
def test_normalization_never_repairs_invalid_distributions(field, invalid):
    evaluator, results, detailed = fixture()
    values = {
        "outcome": detailed.outcome_probabilities[0],
        "scores": detailed.score_probabilities[0],
        "policy": results.policy_target,
    }[field]
    values[:] = [0.0] * len(values)
    values[0] = invalid
    with pytest.raises(AnalysisError):
        payload(evaluator, results, detailed)


@pytest.mark.parametrize("field", ["score_expectations", "outcome_values"])
def test_normalization_rejects_inconsistent_derived_belief_values(field):
    evaluator, results, detailed = fixture()
    getattr(detailed, field)[0] = 0.4
    with pytest.raises(AnalysisError, match="disagree"):
        payload(evaluator, results, detailed)
