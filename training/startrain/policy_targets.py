"""Optional, bounded policy improvement targets with unchanged search actions.

The strength is a convex mixture weight, not a search Q scale. KL is measured
from the resulting target to the root prior, in nats. No probability floor is
invented: a zero-prior action cannot gain mass under a finite KL constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PolicyTarget:
    probabilities: NDArray[np.float32]
    applied_scale: float
    kl_nats: float


def validate_policy_target_settings(scale: float, max_kl: float | None) -> None:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("policy_target_scale must be a finite number in [0, 1]")
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("policy_target_scale must be a finite number in [0, 1]")
    if max_kl is not None and (
        isinstance(max_kl, bool)
        or not isinstance(max_kl, (int, float))
        or not math.isfinite(max_kl)
        or max_kl < 0
    ):
        raise ValueError("policy_target_max_kl must be finite and nonnegative")


def _distribution(values: NDArray, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    mass = float(result.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"{name} must have positive finite mass")
    return result / mass


def _kl(target: NDArray[np.float64], prior: NDArray[np.float64]) -> float:
    positive = target > 0
    if (prior[positive] == 0).any():
        return math.inf
    return max(
        0.0,
        float(
            np.sum(
                target[positive] * (np.log(target[positive]) - np.log(prior[positive]))
            )
        ),
    )


def constrain_policy_target(
    target: NDArray,
    prior: NDArray,
    *,
    scale: float = 1.0,
    max_kl: float | None = None,
) -> PolicyTarget:
    """Blend toward search, optionally bounded by KL(target || root prior).

    KL is convex and nondecreasing along this line from the prior. Bisection
    therefore retains the largest admissible mixture weight. Evaluate the
    rounded FP32 distribution itself so the returned target honors the bound
    (within floating-point normalization error), not merely an FP64 precursor.
    """
    validate_policy_target_settings(scale, max_kl)
    search = _distribution(target, "target")
    base = _distribution(prior, "prior")
    if search.shape != base.shape:
        raise ValueError("target and prior must have matching shapes")

    def candidate(weight: float) -> tuple[NDArray[np.float32], float]:
        # Avoid 0 * inf and ensure an exact endpoint for scale zero.
        mixed = base if weight == 0 else (1 - weight) * base + weight * search
        rounded = np.asarray(mixed, dtype=np.float32)
        return rounded, _kl(_distribution(rounded, "rounded target"), base)

    applied = float(scale)
    if max_kl == 0 or (max_kl is not None and ((base == 0) & (search > 0)).any()):
        applied = 0.0
    result, divergence = candidate(applied)
    if max_kl is not None and divergence > max_kl and applied > 0:
        low, high = 0.0, applied
        for _ in range(48):
            middle = (low + high) / 2
            _, value = candidate(middle)
            if value <= max_kl:
                low = middle
            else:
                high = middle
        applied = low
        result, divergence = candidate(applied)
    return PolicyTarget(result, applied, divergence)
