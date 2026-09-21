"""Shared allocation and eligibility rules for pie-standard training.

The handicap fraction is a share of the complete training mix, not a share
of each board. Conditioning it on the largest board preserves both the
configured board allocation and the overall even/handicap allocation.
Legacy objectives intentionally keep their historical rule support.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .contracts import MAX_HANDICAP, MODES
from .topology import SUPPORTED_RINGS


def is_pie_training(objective: str) -> bool:
    return objective == "ring10_pie"


def training_segment_quotas(
    ring: int,
    ring_weights: Mapping[int, float],
    *,
    handicap_fraction: float = 0.1,
) -> dict[str, float]:
    """Return segment probabilities conditional on one supported board size."""
    if type(ring) is not int or ring not in SUPPORTED_RINGS:
        raise ValueError("training ring must be one of (4, 6, 8, 10)")
    if (
        isinstance(handicap_fraction, bool)
        or not isinstance(handicap_fraction, int | float)
        or not math.isfinite(handicap_fraction)
        or not 0 <= handicap_fraction <= 1
    ):
        raise ValueError("handicap_fraction must be finite and in [0, 1]")
    if not ring_weights or any(
        type(key) is not int
        or key not in SUPPORTED_RINGS
        or isinstance(weight, bool)
        or not isinstance(weight, int | float)
        or not math.isfinite(weight)
        or weight < 0
        for key, weight in ring_weights.items()
    ):
        raise ValueError("ring weights must be finite non-negative supported rings")
    total = sum(ring_weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("ring weights must have a positive finite total")
    largest_share = ring_weights.get(10, 0.0) / total
    if handicap_fraction > largest_share:
        raise ValueError("handicap fraction cannot exceed the ring-10 allocation")
    handicap = (
        handicap_fraction / largest_share
        if ring == 10 and handicap_fraction > 0
        else 0.0
    )
    return {
        "standard": 0.0,
        "classic": 0.0,
        "handicap": handicap,
        "pie": 1.0 - handicap,
    }


def training_mode_weights(
    ring: int,
    ring_weights: Mapping[int, float],
    *,
    handicap_fraction: float = 0.1,
) -> dict[str, float]:
    """Split each eligible segment equally between the two move rules."""
    segments = training_segment_quotas(
        ring, ring_weights, handicap_fraction=handicap_fraction
    )
    return {
        f"{mode}-{segment}": segments[segment] / 2
        for segment in ("standard", "pie", "handicap")
        for mode in ("classic", "double")
    }


def training_variant_allowed(ring: int, mode: str, handicap: int, pie: bool) -> bool:
    """Accept pie even games everywhere and handicap only on the largest board."""
    if (
        type(ring) is not int
        or ring not in SUPPORTED_RINGS
        or type(mode) is not str
        or mode not in MODES
        or type(handicap) is not int
        or type(pie) is not bool
    ):
        return False
    if handicap == 1:
        return pie
    return ring == 10 and 2 <= handicap <= MAX_HANDICAP and not pie
