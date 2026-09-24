"""Observed learner rates with explicit time windows and timing coverage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Observation:
    timestamp_ns: int
    step: int
    record: Mapping[str, object]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def learner_efficiency(
    records: Sequence[Mapping[str, object]], *, now_ns: int
) -> dict[str, object]:
    """Summarize the latest contiguous monotonic segment of bounded telemetry.

    Timings are measured components, not GPU utilization or achieved FLOPs.
    Missing component observations remain unknown rather than becoming zero.
    """
    segment: list[_Observation] = []
    for row in records:
        if not isinstance(row.get("losses"), Mapping):
            continue
        timestamp, step = row.get("timestamp_ns"), row.get("step")
        if (
            type(timestamp) is not int
            or timestamp <= 0
            or timestamp > now_ns
            or type(step) is not int
            or step < 0
        ):
            continue
        if segment and (
            timestamp <= segment[-1].timestamp_ns or step <= segment[-1].step
        ):
            segment = []
        segment.append(_Observation(timestamp, step, row))
    output: dict[str, object] = {
        "schema_version": 1,
        "scope": "bounded-recent-monotonic-observations",
        "available": False,
        "observations": len(segment),
        "timing_kind": "measured-components-not-gpu-utilization",
    }
    if len(segment) < 2:
        output["reason"] = "insufficient_monotonic_observations"
        return output
    first, last = segment[0], segment[-1]
    started, ended = first.timestamp_ns, last.timestamp_ns
    elapsed = (ended - started) / 1e9
    updates = last.step - first.step

    def rate(key: str) -> float | None:
        before, after = _number(first.record.get(key)), _number(last.record.get(key))
        return (
            (after - before) / elapsed
            if before is not None and after is not None and after >= before
            else None
        )

    def timing_fraction(key: str, *, per_step: bool = False) -> float | None:
        total = 0.0
        for observation in segment[1:]:
            row = observation.record
            value = _number(row.get(key))
            count = _number(row.get("metrics_interval_steps")) if per_step else 1.0
            if value is None or value < 0 or count is None or count <= 0:
                return None
            total += value * count
        return total / elapsed

    output.update(
        available=True,
        observation_start_ns=started,
        observation_end_ns=ended,
        observation_age_seconds=(now_ns - ended) / 1e9,
        wall_seconds=elapsed,
        updates=updates,
        updates_per_hour=updates * 3600 / elapsed,
        examples_per_second=rate("examples_consumed"),
        fresh_positions_per_second=rate("total_replay_samples"),
        measured_interval_fraction=timing_fraction("metrics_interval_wall_seconds"),
        utd_sleep_fraction=timing_fraction("metrics_interval_utd_sleep_seconds"),
        device_update_fraction=timing_fraction("device_step_seconds", per_step=True),
    )
    return output
