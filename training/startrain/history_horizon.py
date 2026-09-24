"""Bounded forecasts for admitting history work before replay expiry.

The forecast only changes new-game admission. It never interrupts a game,
changes its pinned model, or removes already published supervision.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryHorizon:
    seconds: float
    steps_per_second: float | None
    minimum_model_step: int

    def metrics(self) -> dict[str, object]:
        return {
            "history_horizon_seconds": self.seconds,
            "history_horizon_steps_per_second": self.steps_per_second,
            "history_horizon_minimum_model_step": self.minimum_model_step,
            "history_horizon_rate_available": self.steps_per_second is not None,
        }


class HistoryHorizonTracker:
    """Share recent learner rates and completed-task durations across cohorts.

    Rate observations use the learner heartbeat's time and process identity,
    so repeated reads and actor scheduling delays cannot invent progress. A
    rewind or process change clears the rate. Until two observations span at
    least thirty seconds, no historical model is admitted. Completed tasks
    provide conservative duration estimates, including shared-GPU pauses.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._progress: deque[tuple[int, int]] = deque(maxlen=256)
        self._learner_pid: int | None = None
        self._durations: dict[tuple[int, str], deque[float]] = {}

    def observe_learner(self, *, step: int, heartbeat_ns: int, pid: int) -> None:
        if any(type(value) is not int for value in (step, heartbeat_ns, pid)):
            return
        if step < 0 or heartbeat_ns <= 0 or pid <= 0:
            return
        with self._lock:
            if self._progress and heartbeat_ns <= self._progress[-1][0]:
                return
            if self._learner_pid != pid or (
                self._progress and step < self._progress[-1][1]
            ):
                self._progress.clear()
                self._learner_pid = pid
            self._progress.append((heartbeat_ns, step))
            cutoff = heartbeat_ns - 600_000_000_000
            while self._progress and self._progress[0][0] < cutoff:
                self._progress.popleft()

    def observe_task(self, *, ring: int, mode: str, seconds: float) -> None:
        if ring not in (4, 6, 8, 10) or not mode or not math.isfinite(seconds):
            return
        if seconds <= 0:
            return
        with self._lock:
            # Callers use the finite set of authoritative variant labels.
            key = (ring, mode)
            if key not in self._durations and len(self._durations) >= 64:
                return
            self._durations.setdefault(key, deque(maxlen=8)).append(seconds)

    def invalidate_rate(self) -> None:
        with self._lock:
            self._progress.clear()
            self._learner_pid = None

    def forecast(
        self,
        *,
        learner_step: int,
        max_lag_steps: int,
        ring: int | None,
        modes: tuple[str, ...],
        initial_seconds: float,
    ) -> HistoryHorizon:
        with self._lock:
            durations = [
                duration
                for (measured_ring, mode), values in self._durations.items()
                if (ring is None or measured_ring == ring)
                and (not modes or mode in modes)
                for duration in values
            ]
            seconds = 1.25 * max([initial_seconds, *durations])
            rate = None
            if self._progress:
                latest_ns, latest_step = self._progress[-1]
                rates = [
                    (latest_step - step) * 1e9 / (latest_ns - observed_ns)
                    for observed_ns, step in self._progress
                    if latest_ns - observed_ns >= 30_000_000_000
                ]
                if rates:
                    rate = max(rates)
            minimum = (
                learner_step + 1
                if rate is None
                else max(
                    0, learner_step - max_lag_steps + math.ceil(rate * seconds) + 1
                )
            )
            return HistoryHorizon(seconds, rate, minimum)
