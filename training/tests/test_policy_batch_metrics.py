from dataclasses import dataclass, replace
import pickle

import pytest

from startrain.contracts import TARGET_OUTCOME, TARGET_POLICY
from startrain.policy_batch_metrics import PolicyBatchAccumulator, PolicyBatchMetrics
from startrain.replay import collate_replay_samples
from test_replay import sample_for


@dataclass
class Sample:
    policy_provenance: str
    weight: float = 1.0
    policy_weight: float = 1.0
    target_mask: int = TARGET_POLICY


def test_weighted_policy_rows_exclude_nonpolicy_and_keep_unknown_provenance():
    metrics = PolicyBatchMetrics.from_samples(
        [
            Sample("completed-q-full-live-policy-only", 2.0),
            Sample("completed-q-fast", 4.0, 0.05),
            Sample("imported", 0.5),
            Sample("completed-q-full", 100.0, target_mask=TARGET_OUTCOME),
        ]
    )
    assert (
        metrics.rows,
        metrics.policy_rows,
        metrics.full_rows,
        metrics.fast_rows,
    ) == (4, 3, 1, 1)
    assert metrics.weight_sum == pytest.approx(2.7)
    assert metrics.full_weight_sum == 2.0
    assert metrics.effective_rows == pytest.approx(2.7**2 / (4 + 0.04 + 0.25))


def test_normalized_batch_share_exposes_all_fast_batches():
    accumulator = PolicyBatchAccumulator()
    accumulator.update(PolicyBatchMetrics.from_samples([Sample("completed-q-full")]))
    accumulator.update(
        PolicyBatchMetrics.from_samples(
            [Sample("completed-q-fast", policy_weight=0.05)] * 10
        )
    )
    accumulator.update(None)
    result = accumulator.as_dict()
    assert result["full_weight_fraction"] == pytest.approx(2 / 3)
    assert result["mean_batch_full_weight_fraction"] == 0.5
    assert result["all_fast_batch_fraction"] == 0.5
    assert result["mean_batch_effective_policy_rows"] == pytest.approx(5.5)
    assert result["missing_batches"] == 1
    assert result["unknown_provenance_rows"] == 0
    accumulator.reset()
    assert accumulator.as_dict() == PolicyBatchAccumulator().as_dict()


def test_empty_zero_weight_and_unknown_batches_are_not_all_fast():
    accumulator = PolicyBatchAccumulator()
    accumulator.update(PolicyBatchMetrics.from_samples([]))
    accumulator.update(
        PolicyBatchMetrics.from_samples([Sample("completed-q-fast", policy_weight=0)])
    )
    assert accumulator.as_dict()["all_fast_batch_fraction"] is None
    accumulator.update(PolicyBatchMetrics.from_samples([Sample("completed-q")]))
    assert accumulator.as_dict()["all_fast_batch_fraction"] == 0
    assert accumulator.as_dict()["unknown_provenance_rows"] == 1


@pytest.mark.parametrize("native", [False, True])
def test_collation_transfer_and_spawn_preserve_cpu_diagnostics(native):
    samples = [
        replace(sample_for(), policy_provenance="completed-q-full"),
        replace(sample_for(), policy_provenance="completed-q-fast", policy_weight=0.2),
    ]
    batch = collate_replay_samples(samples, prefer_native=native)
    expected = PolicyBatchMetrics.from_samples(samples)
    assert batch.policy_metrics == expected
    assert batch.to("cpu").policy_metrics is batch.policy_metrics
    assert pickle.loads(pickle.dumps(batch)).policy_metrics == expected
