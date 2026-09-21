"""Bounded wakeup forecasts; these helpers never grant training credit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import TypedDict


def new_samples_for_batch(
    *, target: Fraction, segment_samples: int, segment_examples: int, batch_size: int
) -> int:
    """Exact additional positions needed to authorize the next complete batch."""
    if target <= 0 or min(segment_samples, segment_examples) < 0 or batch_size <= 0:
        raise ValueError("invalid UTD credit boundary")
    needed = segment_examples + batch_size
    minimum_samples = (
        needed * target.denominator + target.numerator - 1
    ) // target.numerator
    return max(0, minimum_samples - segment_samples)


@dataclass(frozen=True)
class WaitDecision:
    seconds: float
    reason: str
    rows_per_second: float | None
    missing_rows: int | None


class WaitSummary(TypedDict):
    actual_sleep_seconds: float
    requested_sleep_seconds: float
    wait_wall_seconds: float
    credit_polls: int
    credit_ready: bool
    wake_reason: str
    estimated_committed_rows_per_second: float | None
    new_rows_for_next_batch: int | None
    poll_ceiling_seconds: float


class AdaptiveCreditWait:
    """Estimate recent arrivals, with bounded retries when arrivals stop.

    Rate measurements accumulate for up to one second instead of interpreting
    a commit between two adjacent SQL reads as an enormous sustained rate.
    Zero-arrival observations back off; an old estimate is never trusted
    indefinitely. The configured polling period remains the hard wait ceiling.
    """

    def __init__(self, maximum_seconds: float):
        if (
            isinstance(maximum_seconds, bool)
            or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0
        ):
            raise ValueError("UTD polling ceiling must be finite and positive")
        self.maximum_seconds = float(maximum_seconds)
        self.minimum_seconds = min(0.05, self.maximum_seconds)
        self.measurement_seconds = min(1.0, self.maximum_seconds)
        self.anchor_time: float | None = None
        self.anchor_count: int | None = None
        self.last_time: float | None = None
        self.last_count: int | None = None
        self.last_progress: float | None = None
        self.rows_per_second: float | None = None
        self.empty_observations = 0

    def observe(self, count: int, now: float) -> None:
        if type(count) is not int or count < 0 or not math.isfinite(now):
            raise ValueError("invalid committed-counter observation")
        if (
            self.last_time is None
            or self.last_count is None
            or now < self.last_time
            or count < self.last_count
        ):
            self.anchor_time = self.last_time = now
            self.anchor_count = self.last_count = count
            self.last_progress = None
            self.rows_per_second = None
            self.empty_observations = 0
            return
        elapsed = now - self.last_time
        if count > self.last_count:
            self.last_progress = now
            self.empty_observations = 0
        elif elapsed >= self.minimum_seconds:
            self.empty_observations = min(20, self.empty_observations + 1)
        self.last_count, self.last_time = count, now
        assert self.anchor_time is not None and self.anchor_count is not None
        window = now - self.anchor_time
        if window >= self.measurement_seconds:
            measured = (count - self.anchor_count) / window
            if self.rows_per_second is None:
                if measured > 0:
                    self.rows_per_second = measured
            else:
                alpha = -math.expm1(-window / 5.0)
                self.rows_per_second += alpha * (measured - self.rows_per_second)
            self.anchor_count, self.anchor_time = count, now

    def decide(self, missing_rows: int | None, now: float) -> WaitDecision:
        if missing_rows is not None and (
            type(missing_rows) is not int or missing_rows < 0
        ):
            raise ValueError("missing replay rows must be nonnegative")
        rate = self.rows_per_second
        if missing_rows is None or rate is None or rate <= 0:
            return WaitDecision(self.maximum_seconds, "no_rate", rate, missing_rows)
        if self.last_progress is None or now - self.last_progress > max(
            3.0, 3 * self.maximum_seconds
        ):
            return WaitDecision(self.maximum_seconds, "stale_rate", rate, missing_rows)
        estimate = max(
            self.minimum_seconds, min(self.maximum_seconds, missing_rows / rate)
        )
        backoff = min(
            self.maximum_seconds,
            self.minimum_seconds * 2**self.empty_observations,
        )
        return WaitDecision(
            max(estimate, backoff),
            "idle_backoff" if backoff > estimate else "credit_forecast",
            rate,
            missing_rows,
        )


def sleep_interruptibly(
    seconds: float,
    *,
    stop_requested: Callable[[], bool],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> float:
    """Sleep in bounded chunks and return actual measured sleeping time."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("sleep duration must be finite and nonnegative")
    started = clock()
    # A bounded iteration count also makes a no-op test sleeper safe.
    for _ in range(math.ceil(seconds / 0.1)):
        if stop_requested():
            break
        remaining = seconds - (clock() - started)
        if remaining <= 0:
            break
        sleep(min(0.1, remaining))
    return max(0.0, clock() - started)
