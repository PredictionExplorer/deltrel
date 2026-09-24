"""Exact scheduling-only transition; search and learning contracts stay pinned."""

from dataclasses import replace

from .config import ExperimentConfig


def efficiency_scheduling_config(source: ExperimentConfig) -> ExperimentConfig:
    """Enable the protected measurement share and history eligibility horizon."""
    historical = source.orchestration.historical_evaluation
    refresh = source.orchestration.model_refresh
    if (
        historical.measurement_service_fraction != 0.0
        or historical.measurement_max_wait_seconds != 3600.0
        or refresh.history_horizon_enabled is not False
        or refresh.history_horizon_initial_seconds != 3600.0
    ):
        raise ValueError("scheduling transition requires the exact disabled defaults")
    return replace(
        source,
        orchestration=replace(
            source.orchestration,
            historical_evaluation=replace(
                historical,
                measurement_service_fraction=0.2,
                measurement_max_wait_seconds=3600.0,
            ),
            model_refresh=replace(
                refresh,
                history_horizon_enabled=True,
                history_horizon_initial_seconds=3600.0,
            ),
        ),
    )


def validate_efficiency_scheduling_transition(
    source: ExperimentConfig, target: ExperimentConfig
) -> None:
    if target != efficiency_scheduling_config(source):
        raise ValueError(
            "scheduling transition must preserve every non-scheduling setting "
            "and use the declared service share and history horizon"
        )
