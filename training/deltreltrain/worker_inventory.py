"""Authoritative actor identities for current-state monitoring.

Historical throughput remains a time-window calculation. A metric's existence
does not make its producer a member of the current coordinator inventory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_SHARED_PHASES = frozenset(
    {
        "shared_cohorts",
        "arena_gpu_quiescing",
        "arena_gpu_pause",
        "arena_gpu_resume",
        "arena_gpu_resuming",
        "cohort_draining",
    }
)
_LIVE_STATES = frozenset({"running", "paused", "pausing"})


def actor_metric_exclusion(
    metric: Mapping[str, object],
    *,
    workers: Mapping[str, object],
    heartbeats: Mapping[str, Mapping[str, object]],
    run_identity: Mapping[str, object],
) -> str | None:
    """Explain why a row cannot describe a current actor, or accept it.

    Older rows omit PID/run fields. Accept those only for an inventoried owner;
    callers expose that weaker identity assurance separately. Explicit identity
    mismatches are never hidden by the compatibility path.
    """
    name = metric.get("worker")
    if not isinstance(name, str):
        return "missing_worker"
    parent = name if name in workers else re.sub(r"-cohort-\d+$", "", name)
    owner = workers.get(parent)
    if not isinstance(owner, Mapping) or owner.get("role") != "actor":
        return "not_in_coordinator_inventory"
    if owner.get("state") not in _LIVE_STATES:
        return "owner_not_live"
    heartbeat = heartbeats.get(parent, {})
    for source in (metric, heartbeat):
        if "pid" in source and (
            type(source["pid"]) is not int or source["pid"] != owner.get("pid")
        ):
            return "process_identity_mismatch"
    for key in ("run_id", "generation_family"):
        if key in metric and metric[key] != run_identity.get(key):
            return "run_identity_mismatch"
    cohorts = heartbeat.get("cohorts")
    if parent == name and (
        heartbeat.get("phase") in _SHARED_PHASES
        or (type(cohorts) is int and cohorts > 1)
    ):
        return "retired_standalone_actor"
    if parent != name and type(cohorts) is int:
        index = int(name.rsplit("-cohort-", 1)[1])
        if not 0 <= index < cohorts:
            return "cohort_not_in_inventory"
    return None
