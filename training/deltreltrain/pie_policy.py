"""The exact policy-only transition from six variants to pie-standard play."""

from dataclasses import replace

from .config import ExperimentConfig


def pie_training_config(source: ExperimentConfig) -> ExperimentConfig:
    """Change game allocation while preserving the source's execution settings.

    The source must already declare its board mixture. In particular, this
    transformation does not replace measured actor/search/optimizer settings
    with those from a historical example profile.
    """
    fractions = {"standard": 0.0, "classic": 0.0, "handicap": 0.1, "pie": 0.9}
    variants = replace(
        source.selfplay.variants,
        enabled=True,
        **fractions,
        pie_classic_share=0.5,
        handicap_classic_share=0.5,
    )
    return replace(
        source,
        game=replace(source.game, handicap=1, pie_rule=True),
        selfplay=replace(
            source.selfplay,
            rings=10,
            handicap=1,
            pie=True,
            pie_even_training=True,
            variants=variants,
        ),
        learner=replace(
            source.learner,
            use_ring_mixture_curriculum=True,
            segment_quotas=fractions,
        ),
        orchestration=replace(source.orchestration, training_objective="ring10_pie"),
        arena=replace(
            source.arena,
            rings=(10,),
            balanced_cells=True,
            variant_policy="pie_even",
            required_regression_rings=(),
            per_ring_regression_floor_elo={},
            promotion_pair_ratios={},
            segment_pairs_per_ring={},
            segment_regression_floor_elo={},
            segment_handicap_classic_share=0.5,
            segment_handicap_pda=variants.handicap_pda,
        ),
    )


def validate_pie_policy_transition(
    source: ExperimentConfig, target: ExperimentConfig
) -> None:
    """Admit exactly one policy transition, with no execution changes."""
    if source.orchestration.training_objective != "ring10_priority":
        raise ValueError("pie policy transition requires a ring10_priority source")
    if target != pie_training_config(source):
        raise ValueError(
            "pie policy transition must preserve every non-policy setting, "
            "including model, search, inference, cache, and producer configuration"
        )
