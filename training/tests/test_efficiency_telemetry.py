from __future__ import annotations

import pytest

from deltreltrain.efficiency_telemetry import learner_efficiency


def row(second, step, *, examples, replay):
    return {
        "timestamp_ns": second * 1_000_000_000,
        "step": step,
        "losses": {"total": 1.0},
        "examples_consumed": examples,
        "total_replay_samples": replay,
        "metrics_interval_steps": 10,
        "metrics_interval_wall_seconds": 20,
        "metrics_interval_utd_sleep_seconds": 12,
        "device_step_seconds": 0.4,
    }


def test_efficiency_reports_actual_window_and_component_fractions():
    records = [
        row(10, 100, examples=51200, replay=10000),
        row(30, 110, examples=56320, replay=13000),
    ]
    summary = learner_efficiency(records, now_ns=35_000_000_000)
    assert summary["available"]
    assert summary["wall_seconds"] == 20
    assert summary["updates_per_hour"] == 1800
    assert summary["fresh_positions_per_second"] == 150
    assert summary["examples_per_second"] == 256
    assert summary["utd_sleep_fraction"] == 0.6
    assert summary["device_update_fraction"] == 0.2
    assert summary["observation_age_seconds"] == 5


def test_efficiency_does_not_cross_a_learner_rewind_or_use_future_metrics():
    records = [
        row(10, 100, examples=51200, replay=10000),
        row(30, 110, examples=56320, replay=13000),
        row(40, 90, examples=46080, replay=13000),
        row(60, 100, examples=51200, replay=16000),
        row(90, 1000, examples=512000, replay=17000),
    ]
    summary = learner_efficiency(records, now_ns=65_000_000_000)
    assert summary["observation_start_ns"] == 40_000_000_000
    assert summary["updates"] == 10
    assert summary["wall_seconds"] == 20


@pytest.mark.parametrize("invalid", [None, True, -1, float("nan")])
def test_missing_or_invalid_timing_is_unknown_not_idle(invalid):
    records = [
        row(10, 100, examples=51200, replay=10000),
        row(30, 110, examples=56320, replay=13000),
    ]
    records[1]["device_step_seconds"] = invalid
    summary = learner_efficiency(records, now_ns=35_000_000_000)
    assert summary["device_update_fraction"] is None
    assert summary["updates_per_hour"] == 1800
