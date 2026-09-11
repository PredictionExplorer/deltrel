from fractions import Fraction
import json
import math
import random
from types import SimpleNamespace

import pytest

from startrain import learner as module
from startrain.config import LearnerConfig, TrainConfig
from startrain.learner import LearnerLoop, UTDSegmentState
from startrain.utd_wait import (
    AdaptiveCreditWait,
    new_samples_for_batch,
    sleep_interruptibly,
)


class Clock:
    def __init__(self, now=0.0):
        self.now = now
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class CounterStore:
    def __init__(self, count):
        self.count = count
        self.queries = 0

    def total_committed_sample_count(self, **_scope):
        self.queries += 1
        return self.count()

    def committed_sample_history_is_complete(self, **_scope):
        return True


def fake_learner(count=lambda: 0, *, poll=2.0, target=1.5, batch=512):
    learner = object.__new__(LearnerLoop)
    learner.rank = 0
    learner.world_size = 1
    learner.step = learner.examples_consumed = 0
    learner.learner_config = LearnerConfig(
        target_updates_per_new_sample=target, replay_poll_seconds=poll
    )
    learner.train_config = TrainConfig(per_rank_batch_size=batch)
    learner.run_identity = SimpleNamespace(run_id="run", generation_family="family")
    learner.store = CounterStore(count)
    learner._utd_segment_state = UTDSegmentState("run", "family", target, 0, 0)
    return learner


def install_clock(monkeypatch, clock):
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)


@pytest.mark.parametrize(
    "target", [Fraction(1, 4), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(7, 3)]
)
def test_exact_next_batch_boundary_and_existing_allowance_are_unchanged(target):
    rng = random.Random(982)
    for _ in range(500):
        samples, examples, batch = (
            rng.randrange(100_000),
            rng.randrange(150_000),
            rng.choice([1, 32, 512, 1024]),
        )
        missing = new_samples_for_batch(
            target=target,
            segment_samples=samples,
            segment_examples=examples,
            batch_size=batch,
        )
        # Independent exact rational oracle, including already-banked credit.
        assert (samples + missing) * target >= examples + batch
        if missing:
            assert (samples + missing - 1) * target < examples + batch
        assert (missing == 0) == (
            max(0, math.floor(samples * target) - examples) // batch > 0
        )


@pytest.mark.parametrize("target", [0.25, 1.0, 1.5, 2.0, 2.333])
def test_real_budget_keeps_prospective_segment_baselines(target):
    counter = [1_000]
    learner = fake_learner(lambda: counter[0], target=target)
    state = UTDSegmentState("run", "family", target, 123_456, 1_000)
    learner._utd_segment_state = state
    for samples in (0, 1, 341, 342, 1000, 2000, 100_000):
        counter[0] = 1000 + samples
        for consumed in (0, 512, 1024, 1_000_000):
            learner.examples_consumed = 123_456 + consumed
            expected = (
                max(0, math.floor(Fraction(str(target)) * samples) - consumed) // 512
            )
            assert learner._utd_step_budget() == expected
            assert learner._utd_segment_state == state
            assert learner.examples_consumed == 123_456 + consumed


def test_rate_requires_measurement_window_and_stale_estimate_falls_back():
    policy = AdaptiveCreditWait(2)
    policy.observe(0, 0)
    policy.observe(350, 0.0001)
    assert policy.decide(175, 0.0001).seconds == 2
    assert policy.rows_per_second is None
    policy.observe(350, 1)
    decision = policy.decide(175, 1)
    assert decision.seconds == pytest.approx(0.5)
    assert decision.rows_per_second == pytest.approx(350)
    assert policy.decide(1, 7).reason == "stale_rate"
    assert policy.decide(1, 7).seconds == 2


@pytest.mark.parametrize("reset", [(300, 2), (351, 0.5)])
def test_counter_or_clock_regression_discards_old_rate(reset):
    policy = AdaptiveCreditWait(2)
    policy.observe(0, 0)
    policy.observe(350, 1)
    policy.observe(*reset)
    assert policy.decide(1, reset[1]).reason == "no_rate"
    assert policy.decide(1, reset[1]).seconds == 2


@pytest.mark.parametrize("maximum", [0.001, 0.05, 2.0])
def test_idle_backoff_and_sleep_bounds(maximum):
    policy = AdaptiveCreditWait(maximum)
    policy.observe(0, 0)
    policy.observe(1_000_000, 1)
    now = 1.0
    waits = []
    for _ in range(10):
        decision = policy.decide(1, now)
        waits.append(decision.seconds)
        assert min(0.05, maximum) <= decision.seconds <= maximum
        now += decision.seconds + 1e-9
        policy.observe(1_000_000, now)
    assert waits[-1] == maximum


def test_sleep_interrupts_and_reports_measured_time_including_oversleep():
    clock = Clock()
    elapsed = sleep_interruptibly(
        2,
        stop_requested=lambda: clock.now >= 0.15,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    assert elapsed == pytest.approx(0.2)
    assert max(clock.sleeps) <= 0.1
    clock = Clock()

    def oversleep(seconds):
        clock.now += seconds + 0.08

    actual = sleep_interruptibly(
        0.05, stop_requested=lambda: False, clock=clock.monotonic, sleep=oversleep
    )
    assert actual == pytest.approx(0.13)


def test_no_rate_waits_configured_period_and_only_probes_credit(monkeypatch):
    clock = Clock()
    install_clock(monkeypatch, clock)
    learner = fake_learner()

    def forbidden(*_args, **_kwargs):
        pytest.fail("a fast credit probe must not reopen or scan replay")

    for name in (
        "_window_refresh_reason",
        "_select_replay_spans",
        "_eligible_replay_counts",
        "_open_replay_window",
    ):
        monkeypatch.setattr(learner, name, forbidden)
    state = learner._utd_segment_state
    summary = learner._wait_for_utd_credit(stop_requested=lambda: False)
    assert summary["actual_sleep_seconds"] == pytest.approx(2)
    assert summary["requested_sleep_seconds"] == pytest.approx(2)
    assert summary["credit_polls"] == 2
    assert summary["wake_reason"] == "poll_deadline"
    assert not summary["credit_ready"]
    assert learner.examples_consumed == 0
    assert learner._utd_segment_state == state


def test_burst_then_idle_cannot_busy_poll_or_extend_validation_deadline(monkeypatch):
    clock = Clock(1)
    install_clock(monkeypatch, clock)
    learner = fake_learner(lambda: 341)
    policy = learner._credit_wait_policy()
    policy.observe(0, 0)
    policy.observe(341, 1)
    summary = learner._wait_for_utd_credit(stop_requested=lambda: False)
    assert 2 <= summary["credit_polls"] <= 8
    assert summary["actual_sleep_seconds"] == pytest.approx(2)
    assert summary["wait_wall_seconds"] == pytest.approx(2)
    assert summary["new_rows_for_next_batch"] == 1
    assert not summary["credit_ready"]
    assert learner.examples_consumed == 0


def test_stop_during_wait_is_bounded_without_consuming_credit(monkeypatch):
    clock = Clock()
    install_clock(monkeypatch, clock)
    learner = fake_learner()
    summary = learner._wait_for_utd_credit(stop_requested=lambda: clock.now >= 0.15)
    assert summary["actual_sleep_seconds"] == pytest.approx(0.2)
    assert summary["wake_reason"] == "stopped"
    assert learner.examples_consumed == learner.step == 0


def test_nonzero_rank_uses_broadcast_wakeup_and_never_queries_store(monkeypatch):
    clock = Clock()
    install_clock(monkeypatch, clock)
    learner = fake_learner()
    learner.rank, learner.world_size = 1, 2
    controls = iter(
        [
            {
                "credit_ready": False,
                "done": False,
                "seconds": 0.1,
                "reason": "credit_forecast",
                "rows_per_second": 100.0,
                "missing_rows": 10,
            },
            {
                "credit_ready": True,
                "done": True,
                "seconds": 0,
                "reason": "credit_ready",
                "rows_per_second": 100.0,
                "missing_rows": 0,
            },
        ]
    )

    def broadcast(value):
        assert value is None
        return next(controls)

    monkeypatch.setattr(learner, "_collective_stop", lambda stop: stop)
    monkeypatch.setattr(learner, "_broadcast_object", broadcast)
    summary = learner._wait_for_utd_credit(stop_requested=lambda: False)
    assert learner.store.queries == 0
    assert summary["credit_ready"]
    assert summary["actual_sleep_seconds"] == pytest.approx(0.1)
    assert learner.examples_consumed == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_probe_error_is_broadcast_to_every_rank(monkeypatch, rank):
    learner = fake_learner()
    learner.rank, learner.world_size = rank, 2
    seen = []

    def failure():
        raise OSError("counter unavailable")

    def broadcast(value):
        seen.append(value)
        return value if rank == 0 else {"error": "OSError: counter unavailable"}

    monkeypatch.setattr(learner, "_collective_stop", lambda stop: stop)
    monkeypatch.setattr(learner, "_utd_step_budget", failure)
    monkeypatch.setattr(learner, "_broadcast_object", broadcast)
    with pytest.raises(
        RuntimeError, match="UTD credit probe failed: OSError: counter unavailable"
    ):
        learner._wait_for_utd_credit(stop_requested=lambda: False)
    assert seen == (
        [{"error": "OSError: counter unavailable"}] if rank == 0 else [None]
    )
    assert learner.examples_consumed == 0


def test_faster_arrivals_reduce_detection_latency_without_extra_updates(monkeypatch):
    rate = 312_000 * 4 / 3600
    all_latencies = []
    final_counts = []
    for adaptive in (False, True):
        clock = Clock()
        install_clock(monkeypatch, clock)
        learner = fake_learner(lambda: math.floor(rate * clock.now))
        latencies = []
        # Same 160 authorized optimizer updates, with unchanged 0.407s work.
        for step in range(160):
            while learner._utd_step_budget() == 0:
                if adaptive:
                    learner._wait_for_utd_credit(stop_requested=lambda: False)
                else:
                    clock.sleep(2)
            first_legal_sample = math.ceil(
                Fraction((step + 1) * 512, 1) / Fraction(3, 2)
            )
            latencies.append(clock.now - first_legal_sample / rate)
            learner.examples_consumed += 512
            learner.step += 1
            clock.sleep(0.407)
        assert learner.examples_consumed == 160 * 512
        assert learner.examples_consumed <= math.floor(
            Fraction(3, 2) * learner.store.count()
        )
        # Compare the same final replay boundary: all banked allowance survives.
        learner.store.count = lambda: 100_000
        final_counts.append(learner.step + learner._utd_step_budget())
        all_latencies.append(sorted(latencies[10:])[75])
    assert final_counts == [292, 292]
    assert all_latencies[1] < 0.15
    assert all_latencies[0] > 0.7


def test_real_replay_arrival_preserves_window_allowance_and_logs_actual_sleep(
    tmp_path, monkeypatch
):
    from test_replay_window_freshness import append_rows, fixture

    with fixture(tmp_path, target=0.125) as (learner, store, identity):
        original_sleep = module.time.sleep
        appended = False

        def arrival(seconds):
            nonlocal appended
            if not appended:
                appended = True
                append_rows(store, identity, 16)
            original_sleep(seconds)

        monkeypatch.setattr(module.time, "sleep", arrival)
        assert learner.run(steps=2) == 2
        assert appended
        assert learner.examples_consumed == 4
        assert learner._utd_step_budget() == 0
        state = json.loads(learner.utd_segment_path.read_text())
        assert state["target_updates_per_new_sample"] == 0.125
        assert state["baseline_examples_consumed"] == 0
        assert state["baseline_committed_replay_samples"] == 0
        events = [
            json.loads(line)
            for line in (tmp_path / "learner" / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        waits = [event for event in events if event.get("event") == "utd_wait"]
        assert len(waits) == 1
        assert (
            waits[0]["actual_sleep_seconds"] >= waits[0]["requested_sleep_seconds"] > 0
        )
        assert waits[0]["credit_ready"]
        assert (
            len(
                [
                    event
                    for event in events
                    if event.get("event") == "replay_window_allocated"
                ]
            )
            == 1
        )
        metrics = [
            event for event in events if "metrics_interval_utd_sleep_seconds" in event
        ]
        assert sum(
            event["metrics_interval_utd_sleep_seconds"] for event in metrics
        ) == pytest.approx(waits[0]["actual_sleep_seconds"])
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 0
        )
