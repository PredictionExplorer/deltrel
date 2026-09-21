"""Exact additive prediction-profile transformation, without search changes."""

from dataclasses import replace

from .auxiliary_upgrade import AUXILIARY_LOSSES
from .config import ExperimentConfig

AUXILIARY_DEFAULT_WEIGHTS = {
    "opponent_reply": 0.1,
    "second_stone": 0.1,
    "final_shores": 0.05,
    "final_networks": 0.05,
    "final_capes": 0.025,
}


def auxiliary_training_config(
    source: ExperimentConfig, *, recovery: bool = False
) -> ExperimentConfig:
    """Append prediction heads; recovery retains them with zero loss weights."""
    if type(recovery) is not bool:
        raise TypeError("auxiliary recovery must be boolean")
    if source.model.auxiliary_predictions:
        raise ValueError("auxiliary transition requires an original headless source")
    return replace(
        source,
        model=replace(source.model, auxiliary_predictions=True),
        loss=replace(
            source.loss,
            **(
                {name: 0.0 for name in AUXILIARY_LOSSES}
                if recovery
                else AUXILIARY_DEFAULT_WEIGHTS
            ),
        ),
    )


def validate_auxiliary_prediction_transition(
    source: ExperimentConfig, target: ExperimentConfig
) -> None:
    """Allow only false-to-true heads plus their five finite loss coefficients."""
    if source.model.auxiliary_predictions or not target.model.auxiliary_predictions:
        raise ValueError(
            "auxiliary transition requires an original headless source and enabled target"
        )
    if replace(target.model, auxiliary_predictions=False) != source.model:
        raise ValueError(
            "auxiliary transition must preserve every original model setting"
        )
    restored_loss = replace(
        target.loss, **{name: getattr(source.loss, name) for name in AUXILIARY_LOSSES}
    )
    if (
        restored_loss != source.loss
        or replace(target, model=source.model, loss=source.loss) != source
    ):
        raise ValueError(
            "auxiliary transition must preserve every non-auxiliary setting"
        )
