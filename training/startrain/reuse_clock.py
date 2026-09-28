"""Prospective LR/EMA clocks for full-state changes in replay reuse.

Optimizer updates stay on their original integer clock. The learning-rate
schedule and EMA instead advance in equivalent updates at a pinned reference
reuse ratio. All anchors are checkpointed; no historical replay earns credit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .config import SchedulerConfig

REUSE_CLOCK_KEY = "reuse_clock"


def _positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"reuse clock {name} must be finite and positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class ReuseClock:
    reference_target: float
    target: float
    anchor_step: int
    anchor_scheduler_step: int
    anchor_reference_age: float
    reference_ema_decay: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported reuse clock schema")
        _positive(self.reference_target, "reference target")
        _positive(self.target, "target")
        for value in (self.anchor_step, self.anchor_scheduler_step):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "reuse clock anchor steps must be nonnegative integers"
                )
        if (
            isinstance(self.anchor_reference_age, bool)
            or not isinstance(self.anchor_reference_age, (int, float))
            or not math.isfinite(self.anchor_reference_age)
            or self.anchor_reference_age < 0
        ):
            raise ValueError("reuse clock reference age is invalid")
        if (
            isinstance(self.reference_ema_decay, bool)
            or not isinstance(self.reference_ema_decay, (int, float))
            or not math.isfinite(self.reference_ema_decay)
            or not 0 <= self.reference_ema_decay < 1
        ):
            raise ValueError("reuse clock EMA decay is invalid")

    @property
    def rate(self) -> float:
        return self.reference_target / self.target

    @property
    def ema_decay(self) -> float:
        return self.reference_ema_decay**self.rate

    def reference_age(self, scheduler_step: int) -> float:
        if scheduler_step < self.anchor_scheduler_step:
            raise ValueError("scheduler precedes its reuse clock anchor")
        return (
            self.anchor_reference_age
            + (scheduler_step - self.anchor_scheduler_step) * self.rate
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ReuseClock:
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("invalid reuse clock fields") from exc


def checkpoint_clock(metadata: Mapping[str, object]) -> ReuseClock | None:
    extra = metadata.get("extra")
    value = extra.get(REUSE_CLOCK_KEY) if isinstance(extra, Mapping) else None
    config = metadata.get("config")
    learner = config.get("learner") if isinstance(config, Mapping) else None
    if value is None:
        if (
            isinstance(learner, Mapping)
            and learner.get("reuse_clock_reference_target") is not None
        ):
            raise ValueError("checkpoint reuse clock metadata is missing")
        return None
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint reuse clock must be a mapping")
    clock = ReuseClock.from_mapping(value)
    step = metadata.get("step")
    scheduler_step = metadata.get("scheduler_step")
    if (
        type(step) is not int
        or clock.anchor_step > step
        or not isinstance(learner, Mapping)
        or learner.get("reuse_clock_reference_target") != clock.reference_target
        or learner.get("target_updates_per_new_sample") != clock.target
    ):
        raise ValueError("checkpoint reuse clock differs from its step/profile")
    if type(scheduler_step) is not int or scheduler_step < clock.anchor_scheduler_step:
        raise ValueError("checkpoint scheduler precedes its reuse clock anchor")
    decay = metadata.get("ema_decay")
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(decay)
        or not math.isclose(decay, clock.ema_decay, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("checkpoint EMA decay differs from its reuse clock")
    return clock


def continuation_clock(
    *,
    previous: ReuseClock | None,
    previous_target: float | None,
    reference_target: float,
    target: float,
    step: int,
    scheduler_step: int,
    reference_ema_decay: float,
) -> ReuseClock:
    """Re-anchor only at a real target transition, preserving schedule age."""
    _positive(reference_target, "reference target")
    _positive(target, "target")
    if (
        type(step) is not int
        or step < 0
        or type(scheduler_step) is not int
        or scheduler_step < 0
    ):
        raise ValueError("reuse clock continuation steps must be nonnegative integers")
    if previous is not None:
        if (
            previous.reference_target != reference_target
            or previous.reference_ema_decay != reference_ema_decay
            or previous.anchor_step > step
        ):
            raise ValueError("reuse continuation changed its reference clock")
        age = previous.reference_age(scheduler_step)
        if previous.target == target:
            return previous
    else:
        if previous_target is not None and previous_target != reference_target:
            raise ValueError("first reuse clock must reference the saved reuse target")
        age = float(scheduler_step)
    return ReuseClock(
        reference_target=reference_target,
        target=target,
        anchor_step=step,
        anchor_scheduler_step=scheduler_step,
        anchor_reference_age=age,
        reference_ema_decay=reference_ema_decay,
    )


def schedule_multiplier(age: float, config: SchedulerConfig) -> float:
    if config.warmup_steps and age < config.warmup_steps:
        return (age + 1) / config.warmup_steps
    progress = min(
        1.0,
        max(
            0.0,
            (age - config.warmup_steps) / (config.total_steps - config.warmup_steps),
        ),
    )
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * (
        0.5 * (1.0 + math.cos(math.pi * progress))
    )


def clocked_multiplier(clock: ReuseClock, config: SchedulerConfig):
    # A function, rather than a callable class, keeps LambdaLR's state portable
    # under torch.load(weights_only=True). Authority lives in checkpoint extra.
    def multiplier(step: int) -> float:
        return schedule_multiplier(clock.reference_age(step), config)

    return multiplier
