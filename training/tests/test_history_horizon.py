from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from deltreltrain.config import ConfigError, ModelRefreshConfig
from deltreltrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_history_horizon_defaults,
)
from deltreltrain.history_horizon import HistoryHorizonTracker
from deltreltrain.selfplay import GameVariant
from test_cohort_work import fake_actor


def forecast(tracker, **kwargs):
    return tracker.forecast(
        learner_step=583587,
        max_lag_steps=120000,
        ring=10,
        modes=("pie-classic",),
        initial_seconds=3600.0,
        **kwargs,
    )


def measured_tracker():
    tracker = HistoryHorizonTracker()
    tracker.observe_learner(step=583575, heartbeat_ns=1_000_000_000_000, pid=42)
    tracker.observe_learner(step=583587, heartbeat_ns=1_030_000_000_000, pid=42)
    return tracker


def enable_horizon(actor):
    actor.experiment = replace(
        actor.experiment,
        orchestration=replace(
            actor.experiment.orchestration,
            model_refresh=replace(
                actor.experiment.orchestration.model_refresh,
                history_horizon_enabled=True,
            ),
        ),
    )


def test_history_admission_requires_measured_rate_and_rejects_expiring_work():
    assert forecast(HistoryHorizonTracker()).minimum_model_step == 583588
    horizon = forecast(measured_tracker())
    assert horizon.steps_per_second == pytest.approx(0.4)
    assert horizon.seconds == 4500
    assert horizon.minimum_model_step == 465388
    # The observed production task started with only 298 updates of headroom.
    assert 463885 < horizon.minimum_model_step
    assert horizon.minimum_model_step < 500000  # Younger history remains useful.


def test_task_duration_uses_recent_matching_ring_and_mode_including_pauses():
    tracker = measured_tracker()
    tracker.observe_task(ring=10, mode="pie-classic", seconds=5400)
    tracker.observe_task(ring=4, mode="pie-classic", seconds=9000)
    tracker.observe_task(ring=10, mode="pie-double", seconds=12000)
    assert forecast(tracker).seconds == 6750
    for _ in range(8):
        tracker.observe_task(ring=10, mode="pie-classic", seconds=1800)
    assert forecast(tracker).seconds == 4500


def test_repeated_and_out_of_order_heartbeat_reads_do_not_invent_rate():
    tracker = measured_tracker()
    expected = forecast(tracker)
    for _ in range(4):
        tracker.observe_learner(step=583587, heartbeat_ns=1_030_000_000_000, pid=42)
        tracker.observe_learner(step=583575, heartbeat_ns=1_000_000_000_000, pid=42)
    assert forecast(tracker) == expected


@pytest.mark.parametrize("step,pid", [(10, 42), (583600, 43)])
def test_rewind_or_learner_restart_requires_a_new_rate(step, pid):
    tracker = measured_tracker()
    tracker.observe_learner(step=step, heartbeat_ns=1_060_000_000_000, pid=pid)
    assert forecast(tracker).steps_per_second is None


def test_old_rates_expire_and_invalid_observations_do_not_create_rates():
    tracker = measured_tracker()
    tracker.observe_learner(step=584000, heartbeat_ns=2_000_000_000_000, pid=42)
    assert forecast(tracker).steps_per_second is None
    tracker.observe_learner(step=584000, heartbeat_ns=2_030_000_000_000, pid=42)
    assert forecast(tracker).steps_per_second == 0.0
    tracker.invalidate_rate()
    tracker.observe_learner(step=True, heartbeat_ns=-1, pid=0)
    tracker.observe_task(ring=10, mode="pie-classic", seconds=float("nan"))
    assert forecast(tracker).steps_per_second is None


def test_actor_applies_admission_margin_without_changing_history_exclusions(
    tmp_path, monkeypatch
):
    actor, _, _, _ = fake_actor(tmp_path, monkeypatch)
    enable_horizon(actor)
    actor._history_horizon_tracker = measured_tracker()
    calls = []
    provider = object()
    actor.history_pool = SimpleNamespace(
        last_selection_metrics={},
        select=lambda **kwargs: (calls.append(kwargs), provider)[1],
    )
    result = actor._select_history_provider(
        random_source=random.Random(17),
        learner_step=583587,
        exclude={"candidate", "champion"},
        ring=10,
        modes=("pie-classic",),
    )
    assert result is provider
    assert (
        calls[0]["minimum_model_step"]
        > 583587 - actor.experiment.learner.max_replay_lag_steps
    )
    assert calls[0]["maximum_model_step"] == 583587
    assert calls[0]["exclude"] == {"candidate", "champion"}
    actor._history_horizon_tracker.invalidate_rate()
    assert (
        actor._select_history_provider(
            random_source=random.Random(17), learner_step=583587, exclude=set()
        )
        is None
    )
    assert len(calls) == 1
    actor.work_coordinator.close()
    actor.registry.close()


def test_expiry_stops_only_refill_once_and_preserves_quota_metadata(
    tmp_path, monkeypatch
):
    actor, _, _, _ = fake_actor(tmp_path, monkeypatch)
    enable_horizon(actor)
    actor._history_horizon_tracker = measured_tracker()
    actor.experiment = replace(
        actor.experiment,
        learner=replace(actor.experiment.learner, max_replay_lag_steps=120000),
    )
    learner_step = [583587]
    monkeypatch.setattr(
        actor,
        "_read_learner_scheduling_step",
        lambda **kwargs: (learner_step[0], "test"),
    )
    clock = [0.0]
    monkeypatch.setattr("deltreltrain.actor.time.monotonic", lambda: clock[0])
    metadata = {
        "model_role": "history",
        "model_step": 465400,
        "ring": 10,
        "variant": GameVariant(mode="classic", pie=True),
        "candidate_identity_at_start": "candidate",
        "champion_identity_at_start": "champion",
        "games": 128,
    }
    original = dict(metadata)
    gate = actor._stop_refill_gate(metadata)
    assert gate() is False
    learner_step[0] += 20
    clock[0] = 3
    assert gate() is True
    learner_step[0] = 0
    clock[0] = 6
    assert gate() is True
    assert metadata == original
    records = [json.loads(line) for line in actor.metrics_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "history_horizon_refill_stopped"
    assert records[0]["pid"] > 0
    actor.work_coordinator.close()
    actor.registry.close()


def test_unknown_rate_preserves_requested_history_and_variant_on_fallback(
    tmp_path, monkeypatch
):
    actor, store, _, _ = fake_actor(tmp_path, monkeypatch)
    enable_horizon(actor)
    actor.history_pool = SimpleNamespace(last_selection_metrics={})
    variant = GameVariant(mode="classic", pie=True)
    bundle = actor._new_work_bundle(
        None,
        store,
        selection=("history", 10, "classic-pie"),
        preserved_variant=variant,
    )
    assert bundle.metadata["requested_model_role"] == "history"
    assert bundle.metadata["model_role"] == "champion"
    assert bundle.metadata["variant"] == variant
    assert bundle.metadata["history_metrics"]["history_horizon_rate_available"] is False
    bundle.release_reservation()
    actor.work_coordinator.close()
    actor.registry.close()


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "3600"])
def test_horizon_config_rejects_invalid_duration(value):
    with pytest.raises(ConfigError, match="history_horizon_initial_seconds"):
        ModelRefreshConfig(history_horizon_initial_seconds=value)


def test_horizon_config_defaults_keep_legacy_authority_but_enabled_does_not():
    config = ModelRefreshConfig()
    payload = {"orchestration": {"model_refresh": asdict(config)}}
    legacy = without_history_horizon_defaults(payload)
    assert "history_horizon_enabled" not in legacy["orchestration"]["model_refresh"]
    assert legacy in compatible_config_epoch_payloads(payload)
    enabled = {
        "orchestration": {
            "model_refresh": asdict(replace(config, history_horizon_enabled=True))
        }
    }
    assert all(
        row["orchestration"]["model_refresh"]["history_horizon_enabled"] is True
        for row in compatible_config_epoch_payloads(enabled)
    )
    assert "history_horizon_enabled" in payload["orchestration"]["model_refresh"]
