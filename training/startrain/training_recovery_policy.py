"""Narrow, independently reviewable learning-recovery profile treatments."""

from __future__ import annotations

from dataclasses import replace
import math

from .config import ExperimentConfig


def protected_champion_config(
    source: ExperimentConfig, *, fraction: float = 0.25, after_ns: int | None
) -> ExperimentConfig:
    """Protect fresh champion games without changing model or variant quotas."""
    return replace(
        source,
        learner=replace(
            source.learner,
            protected_champion_fraction=fraction,
            protected_champion_max_age_seconds=21600.0,
            protected_champion_after_ns=after_ns,
        ),
    )


def reuse_config(source: ExperimentConfig, *, target: float = 2.0) -> ExperimentConfig:
    """Prepare a prospective full-state continuation with unchanged data clocks."""
    previous = source.learner.target_updates_per_new_sample
    if previous is None or previous <= 0:
        raise ValueError("reuse continuation requires an existing positive target")
    if (
        isinstance(target, bool)
        or not isinstance(target, (float, int))
        or not math.isfinite(target)
        or not 0 < target <= 2.0
        or (
            target == previous
            and source.learner.reuse_clock_reference_target is not None
        )
    ):
        raise ValueError("reuse treatment requires a new clock or target at most 2.0")
    scale = target / previous
    updates = {}
    for name in (
        "candidate_interval_examples",
        "selfplay_snapshot_interval_examples",
        "selfplay_snapshot_warmup_interval_examples",
    ):
        value = getattr(source.learner, name)
        if value is None:
            updates[name] = None
        else:
            scaled = value * scale
            if abs(scaled - round(scaled)) > 1e-6:
                raise ValueError(f"{name} cannot preserve its exact fresh-data cadence")
            updates[name] = round(scaled)
    return replace(
        source,
        learner=replace(
            source.learner,
            target_updates_per_new_sample=float(target),
            reuse_clock_reference_target=(
                source.learner.reuse_clock_reference_target or previous
            ),
            max_replay_lag_steps=round(source.learner.max_replay_lag_steps * scale),
            **updates,
        ),
        orchestration=replace(
            source.orchestration,
            plateau=replace(
                source.orchestration.plateau,
                max_learner_champion_lag_steps=round(
                    source.orchestration.plateau.max_learner_champion_lag_steps * scale
                ),
            ),
        ),
    )


def validate_training_recovery_transition(
    source: ExperimentConfig, target: ExperimentConfig
) -> str:
    """Reject coupled treatments and any unreviewed search/model changes."""
    protected = protected_champion_config(
        source,
        fraction=target.learner.protected_champion_fraction,
        after_ns=target.learner.protected_champion_after_ns,
    )
    if source != target and target == protected:
        return "protected_champion"
    reuse_target = target.learner.target_updates_per_new_sample
    if reuse_target is not None:
        try:
            expected = reuse_config(source, target=reuse_target)
        except ValueError:
            pass
        else:
            if target == expected:
                return "reuse_clock"
    raise ValueError("training recovery must change exactly one declared treatment")
