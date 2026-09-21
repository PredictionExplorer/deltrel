"""Opt-in promotion allocation; existing training and strength evidence are retained."""

from dataclasses import replace

from .config import ExperimentConfig


def pie_promotion_config(source: ExperimentConfig) -> ExperimentConfig:
    """Change only promotion allocation on an existing pie-even training profile.

    Historical crossplay derives an equal-allocation arena from this profile.
    Model, replay, search budgets, inference, and learner settings are unchanged.
    """
    if (
        source.orchestration.training_objective != "ring10_pie"
        or source.arena.variant_policy != "pie_even"
        or source.arena.rings != (10,)
        or not source.arena.balanced_cells
    ):
        raise ValueError("adaptive promotion requires an existing ring10_pie profile")
    return replace(
        source, arena=replace(source.arena, allocation_policy="adaptive_pie")
    )


def validate_pie_promotion_transition(
    source: ExperimentConfig, target: ExperimentConfig
) -> None:
    """Admit an allocation-only transition, never an implicit execution change."""
    if source.arena.allocation_policy != "equal_cells":
        raise ValueError("promotion transition requires equal_cells source allocation")
    if target != pie_promotion_config(source):
        raise ValueError(
            "promotion transition must preserve every non-allocation setting"
        )
