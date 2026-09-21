"""CPU-side diagnostics for the policy targets actually consumed by the learner."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Protocol, Sequence

from .contracts import TARGET_POLICY


class PolicySample(Protocol):
    target_mask: int
    weight: float
    policy_weight: float
    policy_provenance: str


@dataclass(frozen=True, slots=True)
class PolicyBatchMetrics:
    rows: int
    policy_rows: int
    full_rows: int
    fast_rows: int
    weight_sum: float
    full_weight_sum: float
    effective_rows: float

    @classmethod
    def from_samples(cls, samples: Sequence[PolicySample]) -> PolicyBatchMetrics:
        policy = [sample for sample in samples if sample.target_mask & TARGET_POLICY]
        weights = [sample.weight * sample.policy_weight for sample in policy]
        full = [
            sample.policy_provenance.split("-")[:3] == ["completed", "q", "full"]
            for sample in policy
        ]
        fast = [
            sample.policy_provenance.split("-")[:3] == ["completed", "q", "fast"]
            for sample in policy
        ]
        total = math.fsum(weights)
        squared = math.fsum(weight * weight for weight in weights)
        return cls(
            rows=len(samples),
            policy_rows=len(policy),
            full_rows=sum(full),
            fast_rows=sum(fast),
            weight_sum=total,
            full_weight_sum=math.fsum(
                weight for weight, is_full in zip(weights, full, strict=True) if is_full
            ),
            effective_rows=total * total / squared if squared else 0.0,
        )


@dataclass(slots=True)
class PolicyBatchAccumulator:
    batches: int = 0
    missing_batches: int = 0
    rows: int = 0
    policy_rows: int = 0
    full_rows: int = 0
    fast_rows: int = 0
    weighted_batches: int = 0
    all_fast_batches: int = 0
    weight_sum: float = 0.0
    full_weight_sum: float = 0.0
    full_share_sum: float = 0.0
    effective_rows_sum: float = 0.0

    def update(self, batch: PolicyBatchMetrics | None) -> None:
        if batch is None:
            self.missing_batches += 1
            return
        self.batches += 1
        self.rows += batch.rows
        self.policy_rows += batch.policy_rows
        self.full_rows += batch.full_rows
        self.fast_rows += batch.fast_rows
        self.weight_sum += batch.weight_sum
        self.full_weight_sum += batch.full_weight_sum
        self.effective_rows_sum += batch.effective_rows
        if batch.weight_sum > 0:
            self.weighted_batches += 1
            self.full_share_sum += batch.full_weight_sum / batch.weight_sum
            self.all_fast_batches += batch.fast_rows == batch.policy_rows

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "batches": self.batches,
            "missing_batches": self.missing_batches,
            "rows": self.rows,
            "policy_rows": self.policy_rows,
            "full_rows": self.full_rows,
            "fast_rows": self.fast_rows,
            "unknown_provenance_rows": self.policy_rows
            - self.full_rows
            - self.fast_rows,
            "full_weight_fraction": self.full_weight_sum / self.weight_sum
            if self.weight_sum
            else None,
            "mean_batch_full_weight_fraction": self.full_share_sum
            / self.weighted_batches
            if self.weighted_batches
            else None,
            "mean_batch_effective_policy_rows": self.effective_rows_sum / self.batches
            if self.batches
            else None,
            "all_fast_batch_fraction": self.all_fast_batches / self.weighted_batches
            if self.weighted_batches
            else None,
        }

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, field.default)
