"""Exact extracted predicates remain data checks, never admission shortcuts."""

from copy import deepcopy

import pytest

from scripts import strength_freshness_cpu_preservation as p
from tests.test_strength_freshness_cpu_preservation import observations, verify


@pytest.fixture
def sample():
    return getattr(observations, "__wrapped__")()


def reason(call):
    with pytest.raises(p.PreservationViolation) as error:
        call()
    return str(error.value)


def test_valid_helpers_preserve_actual_results_without_mutating_inputs(sample):
    policy, before, _, _ = sample
    snapshot = deepcopy(sample)
    assert p._check_recipe_values(policy["recipe"]) is None
    assert p._check_support(policy, before["support"]) == policy["expected_monitors"]
    assert [
        p._check_metric_payload(row, policy["recipe"])
        for row in before["progress"]["metrics"]
    ] == [500, 500]
    assert verify(sample)["status"] == "passed"
    assert sample == snapshot


@pytest.mark.parametrize(
    "path,value",
    [
        (("learning_rates",), [0.1, 0.2]),
        (("learning_rates",), [0.1, 0, 0.2]),
        (("learning_rates",), [0.1, float("nan"), 0.2]),
        (("ema_decay",), 1.0),
        (("ema_decay",), True),
        (("bootstrap_step",), -1),
        (("replay_minimum_shard_id_exclusive",), True),
        (("utd_segment_baseline_committed_replay_samples",), -1),
        (("utd_segment_baseline_examples_consumed",), -1),
        (("utd_segment_target_updates_per_new_sample",), 0),
    ],
)
def test_recipe_mutants_have_same_old_refusal_order(sample, path, value):
    policy = sample[0]
    policy["recipe"][path[0]] = value
    assert reason(lambda: p._check_recipe_values(policy["recipe"])) == reason(
        lambda: p._recipe(policy)
    )


@pytest.mark.parametrize("birth", [None, 0, True, -1])
def test_bare_recipe_does_not_satisfy_old_lifetime_admission(sample, birth):
    policy = sample[0]
    assert p._check_recipe_values(policy["recipe"]) is None
    policy["learner_birth_upper_ns"] = birth
    assert reason(lambda: p._recipe(policy)) == "learner-lifetime-policy"
    assert reason(lambda: verify(sample)) == "learner-lifetime-policy"


@pytest.mark.parametrize(
    "path,value",
    [
        (("losses",), {}),
        (("losses", "policy"), float("nan")),
        (("gradient_norm",), -1),
        (("gradient_pre_clip_norm",), float("inf")),
        (("gradient_post_clip_norm",), True),
        (("nonfinite_loss_count",), 1),
        (("nonfinite_gradient_count",), 1),
        (("gradient_diagnostics", "global_norm_finite"), False),
        (("gradient_diagnostics", "nonfinite_gradient_tensors"), 1),
        (("learning_rates",), [0.1, 0.2, 0.3]),
        (("ema", "decay"), 0.9),
        (("ema", "num_updates"), 999),
        (("replay_minimum_shard_id_exclusive",), 1),
        (("utd_segment_baseline_committed_replay_samples",), 1),
        (("utd_segment_baseline_examples_consumed",), 1),
        (("utd_segment_target_updates_per_new_sample",), 2),
        (("segment_updates_per_new_sample",), 1.500000002),
        (("segment_updates_per_new_sample",), -1),
        (("gradient_diagnostics", "batch", "six_mode_unknown"), {"private": 1}),
        (("policy_batch_metrics", "unknown_provenance_rows"), 1),
        (("gradient_diagnostics", "batch", "rows"), 0),
        (("gradient_diagnostics", "batch", "label_availability", "policy"), 0),
        (("gradient_diagnostics", "batch", "label_availability", "outcome"), 513),
    ],
)
def test_metric_mutants_keep_same_payload_and_old_snapshot_refusal(sample, path, value):
    policy, before, _, verified = sample
    metric = before["progress"]["metrics"][0]
    parent = metric
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    assert reason(lambda: p._check_metric_payload(metric, policy["recipe"])) == reason(
        lambda: p._snapshot(policy, before, verified)
    )


def test_zero_outcome_payload_is_allowed_but_aggregate_requirement_remains(sample):
    policy, before, _, verified = sample
    for metric in before["progress"]["metrics"]:
        metric["gradient_diagnostics"]["batch"]["label_availability"]["outcome"] = 0
        assert p._check_metric_payload(metric, policy["recipe"]) == 0
    assert (
        reason(lambda: p._snapshot(policy, before, verified))
        == "missing-outcome-labels"
    )


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("worker", "metric-worker-lifetime-or-order"),
        ("future", "stale-or-future-evidence"),
        ("before-birth", "metric-worker-lifetime-or-order"),
        ("step-order", "metric-step-order"),
        ("stamp-order", "metric-worker-lifetime-or-order"),
        ("heartbeat-birth", "learner-heartbeat-lifetime-join"),
    ],
)
def test_old_temporal_and_owner_checks_still_precede_payload(
    sample, monkeypatch, fault, expected
):
    policy, before, _, verified = sample
    metrics = before["progress"]["metrics"]
    if fault == "worker":
        metrics[0]["worker"] = "private-wrong-worker"
    elif fault == "future":
        metrics[0]["timestamp_ns"] = before["clock"]["wall_ns"] + 1
    elif fault == "before-birth":
        policy["learner_birth_upper_ns"] = metrics[0]["timestamp_ns"]
    elif fault == "step-order":
        metrics[0]["step"] = before["progress"]["step"] + 1
    elif fault == "stamp-order":
        metrics[1]["timestamp_ns"] = metrics[0]["timestamp_ns"]
    else:
        policy["learner_birth_upper_ns"] = before["progress"]["heartbeat_ns"]
    calls = []
    actual = p._check_metric_payload

    def measured(metric, recipe):
        calls.append(metric["step"])
        return actual(metric, recipe)

    monkeypatch.setattr(p, "_check_metric_payload", measured)
    assert reason(lambda: p._snapshot(policy, before, verified)) == expected
    assert len(calls) == (1 if fault == "stamp-order" else 0)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "static",
        "job-type",
        "job-age",
        "monitor-owner",
        "monitor-state",
        "timer",
        "overdue",
        "inactive-failed",
    ],
)
def test_support_mutants_preserve_old_snapshot_checks(sample, fault):
    policy, before, _, verified = sample
    rows = before["support"]
    if fault == "missing":
        rows.pop("backup.timer")
    elif fault == "static":
        rows["backup.service"]["definition_sha256"] = "0" * 64
    elif fault == "job-type":
        rows["backup.service"]["job"] = {"kind": "stop", "age_seconds": 1}
    elif fault == "job-age":
        rows["backup.service"]["job"] = {"kind": "start", "age_seconds": 1801}
    elif fault == "monitor-owner":
        rows["monitor.service"]["process"]["start_ticks"] += 1
    elif fault == "monitor-state":
        rows["monitor.service"]["active"] = "inactive"
    elif fault == "timer":
        rows["backup.timer"]["active"] = "inactive"
    elif fault == "overdue":
        rows["backup.service"].update(active="active", running_seconds=1801)
    else:
        rows["backup.service"]["result"] = "exit-code"
    assert reason(lambda: p._check_support(policy, rows)) == reason(
        lambda: p._snapshot(policy, before, verified)
    )


@pytest.mark.parametrize("state", ["active", "activating", "deactivating"])
def test_support_live_previous_success_is_not_reinterpreted_as_completion(
    sample, state
):
    policy, before, _, verified = sample
    rows = before["support"]
    rows["backup.service"].update(
        active=state,
        result="old-failure-result",
        running_seconds=1800,
        job={"kind": "start", "age_seconds": 1800},
    )
    assert p._check_support(policy, rows) == policy["expected_monitors"]
    assert (
        p._snapshot(policy, before, verified)["stable_support"]
        == policy["expected_monitors"]
    )
    assert rows["backup.service"]["active"] == state


def test_support_still_precedes_champion_and_metric_checks(sample):
    policy, before, _, verified = sample
    before["support"]["backup.timer"]["active"] = "inactive"
    before["champion_identity"] = "unverified"
    before["progress"]["metrics"][0]["losses"] = {}
    assert reason(lambda: p._snapshot(policy, before, verified)) == "timer-inactive"
