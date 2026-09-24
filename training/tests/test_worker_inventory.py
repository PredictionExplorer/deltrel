from __future__ import annotations

import pytest

from deltreltrain.worker_inventory import actor_metric_exclusion


def classify(metric, *, heartbeat=None, state="running"):
    return actor_metric_exclusion(
        metric,
        workers={"actor-gpu-1": {"role": "actor", "state": state, "pid": 42}},
        heartbeats={"actor-gpu-1": heartbeat or {"pid": 42, "phase": "selfplay"}},
        run_identity={"run_id": "run", "generation_family": "family"},
    )


def test_inventory_excludes_retired_lane_and_parent_but_keeps_current_cohort():
    shared = {"pid": 42, "phase": "shared_cohorts", "cohorts": 4}
    assert classify({"worker": "actor-gpu-1-lane-0"}, heartbeat=shared) == (
        "not_in_coordinator_inventory"
    )
    assert classify({"worker": "actor-gpu-1"}, heartbeat=shared) == (
        "retired_standalone_actor"
    )
    assert (
        classify({"worker": "actor-gpu-1-cohort-3", "pid": 42}, heartbeat=shared)
        is None
    )
    assert classify({"worker": "actor-gpu-1-cohort-4"}, heartbeat=shared) == (
        "cohort_not_in_inventory"
    )


@pytest.mark.parametrize("pid", [41, True, "42"])
def test_restarted_process_metrics_never_describe_current_worker(pid):
    assert classify({"worker": "actor-gpu-1", "pid": pid}) == (
        "process_identity_mismatch"
    )


def test_explicit_run_mismatch_is_rejected_and_paused_actor_remains_in_inventory():
    assert classify({"worker": "actor-gpu-1", "run_id": "old-run"}) == (
        "run_identity_mismatch"
    )
    assert classify({"worker": "actor-gpu-1", "pid": 42}, state="paused") is None
    assert classify({"worker": "actor-gpu-1"}, state="failed") == "owner_not_live"
