"""Bounded replay from a verified champion whose model step has aged out."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re


@dataclass(frozen=True, slots=True)
class ProtectedChampionReplay:
    model_identity: str
    model_step: int
    minimum_first_published_ns: int
    maximum_first_published_ns: int
    max_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.model_identity, str) or not re.fullmatch(
            r"sha256-[0-9a-f]{64}", self.model_identity
        ):
            raise ValueError("protected champion requires a content-addressed identity")
        if type(self.model_step) is not int or self.model_step < 0:
            raise ValueError("protected champion step must be nonnegative")
        if (
            type(self.minimum_first_published_ns) is not int
            or type(self.maximum_first_published_ns) is not int
            or self.minimum_first_published_ns <= 0
            or self.maximum_first_published_ns < self.minimum_first_published_ns
        ):
            raise ValueError("protected replay requires an ordered publication window")
        if (
            isinstance(self.max_fraction, bool)
            or not isinstance(self.max_fraction, int | float)
            or not math.isfinite(self.max_fraction)
            or not 0 < self.max_fraction <= 0.5
        ):
            raise ValueError("protected replay fraction must be in (0, 0.5]")

    def capacity(self, ordinary: int, protected: int) -> int:
        """Never let protected rows displace the required ordinary majority."""
        fraction = Fraction(str(self.max_fraction))
        limit = (
            ordinary * fraction.numerator // (fraction.denominator - fraction.numerator)
        )
        return ordinary + min(protected, limit)

    def quota(self, total: int) -> int:
        fraction = Fraction(str(self.max_fraction))
        return total * fraction.numerator // fraction.denominator


def replay_eligibility_clause(
    current_model_step: int,
    max_model_lag_steps: int,
    protection: ProtectedChampionReplay | None,
    *,
    protected_only: bool = False,
) -> tuple[str, tuple[object, ...]]:
    """The exception preserves the ordinary future/stale-model exclusion."""
    lower = max(0, current_model_step - max_model_lag_steps)
    ordinary = "model_step BETWEEN ? AND ?"
    ordinary_values: tuple[object, ...] = (lower, current_model_step)
    if protection is None:
        if protected_only:
            raise ValueError("protected-only selection requires a champion")
        return ordinary, ordinary_values
    protected = (
        "(model_step < ? AND model_identity = ? AND model_step = ? "
        "AND typeof(first_published_ns) = 'integer' "
        "AND first_published_ns <= created_ns "
        "AND first_published_ns BETWEEN ? AND ?)"
    )
    protected_values: tuple[object, ...] = (
        lower,
        protection.model_identity,
        protection.model_step,
        protection.minimum_first_published_ns,
        protection.maximum_first_published_ns,
    )
    if protected_only:
        return protected, protected_values
    return f"({ordinary} OR {protected})", ordinary_values + protected_values


def replay_commit_eligibility_metrics(
    *, ordinary_eligible: bool, protected_candidate: bool, samples: int
) -> dict[str, object]:
    """Generation eligibility is never a claim that the learner consumed rows.

    A protected source still requires original-publication age checks and the
    per-cell cap. Leave its legacy boolean/counts unknown until selection.
    """
    pending = protected_candidate and not ordinary_eligible
    return {
        "replay_eligibility_scope": "commit_filter_not_learner_consumption",
        "ordinary_replay_eligible_at_commit": ordinary_eligible,
        "protected_champion_candidate_at_commit": protected_candidate,
        "replay_eligibility_status": (
            "ordinary"
            if ordinary_eligible
            else "protected_pending_selection"
            if pending
            else "ineligible"
        ),
        "replay_eligible_at_commit": None if pending else ordinary_eligible,
        "eligible_samples_at_commit": None
        if pending
        else samples
        if ordinary_eligible
        else 0,
        "ineligible_samples_at_commit": None
        if pending
        else 0
        if ordinary_eligible
        else samples,
        "protected_pending_selection_samples": samples if pending else 0,
    }
