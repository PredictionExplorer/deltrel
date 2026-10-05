"""Join bounded telemetry to current Linux process ownership before a canary.

This module has no filesystem, database, process or GPU side effects. The Linux
adapter owns bounded reads, actual runtime/cgroup verification, and a repeated
process-identity check around capture. A heartbeat alone is never ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import re
from typing import Any, TypeGuard, cast

from deltreltrain.strength_freshness_guard import Clock, Process, Unit


class ProgressViolation(ValueError):
    """A coherent current capture contradicts the registered training contract."""


class _Pending(Exception):
    """Fresh evidence is incomplete; callers may wait within the same deadline."""


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise ProgressViolation(reason)


def _int(value: object) -> TypeGuard[int]:
    return type(value) is int and value >= 0


def _finite(value: object) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(cast(float, value))


def _object(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgressViolation(reason)
    return value


@dataclass(frozen=True)
class ProgressPolicy:
    run_id: str
    generation_family: str
    expected_workers: tuple[str, ...]
    cohort_parents: Mapping[str, str]
    learning_rates: tuple[float, ...]
    ema_decay: float
    ema_bootstrap_step: int
    replay_watermark: int
    utd_target: float
    utd_baseline_samples: int
    utd_baseline_examples: int
    continuation_started_ns: int
    profile_sha256: str
    source_commit: str
    max_age_seconds: int = 120
    min_metric_intervals: int = 2

    def __post_init__(self) -> None:
        _require(
            all(
                isinstance(x, str) and x for x in (self.run_id, self.generation_family)
            ),
            "run-identity",
        )
        _require(
            isinstance(self.expected_workers, tuple)
            and len(self.expected_workers) == len(set(self.expected_workers))
            and "learner" in self.expected_workers
            and all(
                isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x)
                for x in self.expected_workers
            ),
            "worker-policy",
        )
        _require(
            isinstance(self.cohort_parents, Mapping)
            and len(self.cohort_parents) == 24
            and all(
                isinstance(k, str)
                and k
                and v in self.expected_workers
                and v != "learner"
                for k, v in self.cohort_parents.items()
            ),
            "cohort-policy",
        )
        _require(
            bool(self.learning_rates)
            and all(_finite(x) and x > 0 for x in self.learning_rates)
            and _finite(self.ema_decay)
            and 0 < self.ema_decay < 1
            and _finite(self.utd_target)
            and self.utd_target > 0,
            "learning-policy",
        )
        _require(
            all(
                _int(x)
                for x in (
                    self.ema_bootstrap_step,
                    self.replay_watermark,
                    self.utd_baseline_samples,
                    self.utd_baseline_examples,
                )
            )
            and _int(self.continuation_started_ns)
            and self.continuation_started_ns > 0
            and type(self.max_age_seconds) is int
            and 1 <= self.max_age_seconds <= 120
            and type(self.min_metric_intervals) is int
            and 2 <= self.min_metric_intervals <= 20,
            "counter-or-freshness-policy",
        )
        _require(
            isinstance(self.profile_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", self.profile_sha256)
            and isinstance(self.source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", self.source_commit),
            "runtime-policy-pins",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProgressPolicy:
        names = {f.name for f in fields(cls)}
        required = names - {"max_age_seconds", "min_metric_intervals"}
        _require(required <= set(value) <= names, "progress-policy-keys")
        copy = dict(value)
        for name in ("expected_workers", "learning_rates"):
            _require(isinstance(copy[name], list), "progress-policy-array")
            copy[name] = tuple(copy[name])
        return cls(**copy)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["expected_workers"] = list(self.expected_workers)
        value["learning_rates"] = list(self.learning_rates)
        return value


def verify_progress(
    policy: ProgressPolicy,
    capture: Mapping[str, Any],
    *,
    role: str,
    unit: Unit,
    clock: Clock,
    births: Mapping[Process, int],
    verified_champions: Mapping[str, str],
    arena_prefix_preserved: bool,
) -> dict[str, Any] | None:
    """Return a canary observation, or None while fresh startup data is pending.

    ``births`` maps currently rechecked unit members to conservative *upper*
    realtime nanoseconds of their kernel start time. The adapter must derive it
    from the same boot and account for clock/tick uncertainty. It must sandwich
    bounded capture with PID/start/cgroup/runtime checks. ``verified_champions``
    binds independently validated current/previous champion identities to proof
    hashes; this function never treats a candidate or an asserted role as proof.

    Metrics lack PID fields in the pinned R3/R4 runtime, so only fresh intervals
    strictly after the current learner's birth and no later than its heartbeat
    are admitted. Historical intervals cannot certify a restarted process.
    """
    _require(role in ("r3", "r4"), "runtime-role")
    _require(type(clock.wall_ns) is int and clock.wall_ns > 0, "capture-clock")
    if unit.active != "active" or unit.job is not None or unit.main is None:
        return None
    _require(bool(unit.invocation_id), "runtime-invocation")
    _require(len(unit.members) == len(set(unit.members)), "duplicate-process-identity")
    _require(
        all(
            type(p.pid) is int
            and p.pid > 0
            and type(p.start_ticks) is int
            and p.start_ticks > 0
            for p in unit.members
        ),
        "invalid-process-identity",
    )
    members = {p.pid: p for p in unit.members}
    _require(len(members) == len(unit.members), "reused-pid-in-capture")
    _require(unit.main in unit.members, "controller-outside-runtime-cgroup")
    _require(bool(verified_champions), "verified-champion-proof-required")
    _require(
        all(
            isinstance(k, str)
            and re.fullmatch(r"sha256-[0-9a-f]{64}", k)
            and isinstance(v, str)
            and re.fullmatch(r"[0-9a-f]{64}", v)
            for k, v in verified_champions.items()
        ),
        "champion-proof-pins",
    )
    maximum_age = policy.max_age_seconds * 1_000_000_000

    def owned(pid: object) -> Process:
        if type(pid) is not int or pid not in members:
            raise _Pending("process-not-current")
        process = members[pid]
        birth = births.get(process)
        if not _int(birth) or birth >= clock.wall_ns:
            raise ProgressViolation("process-birth-proof")
        return process

    def fresh(stamp: object, process: Process) -> None:
        if type(stamp) is not int:
            raise ProgressViolation("telemetry-timestamp")
        _require(stamp <= clock.wall_ns, "future-telemetry")
        if stamp <= births[process] or clock.wall_ns - stamp > maximum_age:
            raise _Pending("telemetry-not-fresh-for-process")

    def failure_record(row: Mapping[str, Any]) -> bool:
        return (
            row.get("phase") in {"failed", "nonfinite_abort"}
            or bool(row.get("failure"))
            or any(
                k in row and row[k] != 0
                for k in ("nonfinite_loss_count", "nonfinite_gradient_count")
            )
        )

    try:
        owned(unit.main.pid)
        coordinator = _object(capture.get("coordinator"), "coordinator-object")
        coordinator_process = owned(coordinator.get("coordinator_pid"))
        if coordinator_process.start_ticks < unit.main.start_ticks:
            return None
        fresh(coordinator.get("timestamp_ns"), coordinator_process)
        _require(
            not coordinator.get("failure")
            and not coordinator.get("hardware_failure_class")
            and coordinator.get("state") != "failed",
            "coordinator-failure",
        )
        if coordinator.get("state") != "running" or coordinator.get("draining"):
            return None
        workers = _object(coordinator.get("workers"), "worker-inventory")
        heartbeats = _object(capture.get("heartbeats"), "heartbeat-inventory")
        cohorts = _object(capture.get("cohorts"), "cohort-inventory")
        # A pending startup peer must not hide an attributable failure elsewhere
        # in this same capture. Inspect negative evidence before readiness exits.
        for name in policy.expected_workers:
            worker = workers.get(name)
            if not isinstance(worker, Mapping):
                continue
            _require(
                not worker.get("failure_class") and worker.get("state") != "failed",
                "worker-failure",
            )
            try:
                process = owned(worker.get("pid"))
            except _Pending:
                continue
            if process.start_ticks < coordinator_process.start_ticks:
                continue
            records = [(name, heartbeats.get(name))]
            records.extend(
                (cohort, cohorts.get(cohort))
                for cohort, parent in policy.cohort_parents.items()
                if parent == name
            )
            for record_name, row in records:
                if not isinstance(row, Mapping) or row.get("pid") != process.pid:
                    continue
                try:
                    fresh(row.get("heartbeat_ns"), process)
                except _Pending:
                    continue
                _require(row.get("worker") == record_name, "heartbeat-worker-binding")
                _require(not failure_record(row), "current-worker-failed")
            if name == "learner":
                pending_metrics = capture.get("metrics")
                _require(
                    isinstance(pending_metrics, list) and len(pending_metrics) <= 1000,
                    "bounded-metric-capture",
                )
                for row in cast(list[Any], pending_metrics):
                    if not isinstance(row, Mapping) or not failure_record(row):
                        continue
                    stamp = row.get("timestamp_ns")
                    _require(type(stamp) is int, "metric-clock")
                    if cast(int, stamp) > births[process]:
                        _require(cast(int, stamp) <= clock.wall_ns, "future-metric")
                        raise ProgressViolation("current-learner-failure-record")
        if set(workers) != set(policy.expected_workers):
            return None
        current: dict[str, Process] = {}
        for name in policy.expected_workers:
            worker = _object(workers[name], "worker-object")
            _require(
                not worker.get("failure_class") and worker.get("state") != "failed",
                "worker-failure",
            )
            if worker.get("state") != "running" or name not in heartbeats:
                return None
            _require(
                worker.get("restart_count") == 0
                and type(worker.get("restart_count")) is int,
                "worker-restarted-during-handoff",
            )
            _require(not worker.get("failure_class"), "worker-failure")
            process = owned(worker.get("pid"))
            if process.start_ticks < coordinator_process.start_ticks:
                return None
            heartbeat = _object(heartbeats[name], "heartbeat-object")
            if heartbeat.get("pid") != process.pid:
                return None
            _require(heartbeat.get("worker") == name, "heartbeat-worker-binding")
            fresh(heartbeat.get("heartbeat_ns"), process)
            _require(
                heartbeat.get("phase") not in {"failed", "nonfinite_abort"},
                "current-worker-failed",
            )
            current[name] = process
        _require(len(set(current.values())) == len(current), "workers-share-process")
        if set(cohorts) != set(policy.cohort_parents):
            return None
        for name, parent in policy.cohort_parents.items():
            heartbeat = _object(cohorts[name], "cohort-object")
            if heartbeat.get("pid") != current[parent].pid:
                return None
            _require(heartbeat.get("worker") == name, "cohort-worker-binding")
            fresh(heartbeat.get("heartbeat_ns"), current[parent])
            _require(
                heartbeat.get("phase") not in {"failed", "nonfinite_abort"},
                "current-cohort-failed",
            )
            requested, actual = (
                heartbeat.get("requested_model_role"),
                heartbeat.get("model_role"),
            )
            identity = heartbeat.get("model_version")
            _require(
                requested in (None, "champion") and actual in (None, "champion"),
                "candidate-teacher",
            )
            _require(
                identity is None or identity in verified_champions,
                "unverified-teacher-identity",
            )
            if requested is None or actual is None or identity is None:
                if heartbeat.get("phase") in {
                    "starting",
                    "waiting_for_champion",
                    "replay_reconciliation",
                }:
                    return None
                raise ProgressViolation("missing-assigned-teacher-evidence")

        _require(
            capture.get("profile_sha256") == policy.profile_sha256
            and capture.get("source_commit") == policy.source_commit,
            "active-runtime-drift",
        )
        continuity = _object(capture.get("continuation_state"), "continuation-state")
        _require(
            continuity.get("continuation_started_ns") == policy.continuation_started_ns,
            "original-continuation-clock-changed",
        )
        counter = _object(capture.get("replay_counter"), "replay-counter")
        _require(
            counter.get("run_id") == policy.run_id
            and counter.get("generation_family") == policy.generation_family,
            "replay-counter-identity",
        )
        _require(
            _int(counter.get("committed_samples"))
            and _int(counter.get("updated_ns"))
            and counter["updated_ns"] <= clock.wall_ns,
            "replay-counter-fields",
        )
        learner = heartbeats["learner"]
        _require(_int(learner.get("step")), "learner-step")
        raw_metrics = capture.get("metrics")
        if not isinstance(raw_metrics, list) or len(raw_metrics) > 1000:
            raise ProgressViolation("bounded-metric-capture")
        metrics = []
        for row in raw_metrics:
            row = _object(row, "metric-object")
            failure = failure_record(row)
            if not row.get("losses") and not failure:
                continue
            stamp = row.get("timestamp_ns")
            if type(stamp) is not int:
                raise ProgressViolation("metric-clock")
            _require(stamp <= clock.wall_ns, "future-metric")
            if stamp <= births[current["learner"]]:
                continue
            # A fresh explicit abort can precede the next heartbeat/coordinator
            # update. Do not hide it behind a stale step or missing loss values.
            _require(not failure, "current-learner-failure-record")
            if clock.wall_ns - stamp > maximum_age:
                continue
            _require(_int(row.get("step")), "metric-step")
            if stamp > learner["heartbeat_ns"] or row["step"] > learner["step"]:
                continue
            metrics.append(row)
        if len(metrics) < policy.min_metric_intervals:
            return None
        metrics = metrics[-20:]
        prior_step, prior_stamp = -1, -1
        labels = {"rows": 0, "outcome": 0, "policy": 0}
        for row in metrics:
            _require(row.get("worker") == "learner", "metric-worker")
            _require(
                row["step"] > prior_step and row["timestamp_ns"] > prior_stamp,
                "metric-regression-or-replay",
            )
            prior_step, prior_stamp = row["step"], row["timestamp_ns"]
            losses = _object(row["losses"], "losses-object")
            _require(
                bool(losses) and all(_finite(v) for v in losses.values()),
                "nonfinite-loss",
            )
            _require(
                _finite(row.get("gradient_norm")) and row["gradient_norm"] >= 0,
                "nonfinite-gradient",
            )
            _require(
                all(
                    type(row.get(k)) is int and row[k] == 0
                    for k in ("nonfinite_loss_count", "nonfinite_gradient_count")
                ),
                "nonfinite-update-counter",
            )
            _require(
                row.get("learning_rates") == list(policy.learning_rates),
                "learning-rate-drift",
            )
            ema = _object(row.get("ema"), "ema-object")
            _require(
                ema.get("decay") == policy.ema_decay
                and type(ema.get("num_updates")) is int
                and ema["num_updates"] == row["step"] - policy.ema_bootstrap_step
                and ema["num_updates"] >= 0,
                "ema-clock-or-decay",
            )
            _require(
                row.get("replay_minimum_shard_id_exclusive") == policy.replay_watermark
                and row.get("utd_segment_target_updates_per_new_sample")
                == policy.utd_target
                and row.get("utd_segment_baseline_committed_replay_samples")
                == policy.utd_baseline_samples
                and row.get("utd_segment_baseline_examples_consumed")
                == policy.utd_baseline_examples,
                "replay-credit-contract",
            )
            ratio = row.get("segment_updates_per_new_sample")
            _require(
                _finite(ratio) and 0 <= ratio <= policy.utd_target + 1e-9,
                "replay-credit-overrun",
            )
            batch = _object(
                _object(row.get("gradient_diagnostics"), "gradient-diagnostics").get(
                    "batch"
                ),
                "diagnostic-batch",
            )
            available = _object(batch.get("label_availability"), "diagnostic-labels")
            _require(
                _int(batch.get("rows"))
                and batch["rows"] > 0
                and all(
                    _int(available.get(k)) and available[k] <= batch["rows"]
                    for k in ("outcome", "policy")
                ),
                "label-counts",
            )
            policy_metrics = _object(
                row.get("policy_batch_metrics"), "policy-provenance"
            )
            _require(
                not batch.get("six_mode_unknown")
                and type(policy_metrics.get("unknown_provenance_rows")) is int
                and policy_metrics["unknown_provenance_rows"] == 0,
                "unknown-replay-provenance",
            )
            labels["rows"] += batch["rows"]
            for name in ("outcome", "policy"):
                labels[name] += available[name]
        if not labels["outcome"] or not labels["policy"] or not arena_prefix_preserved:
            return None
        neural_rows = 0
        for parent in set(policy.cohort_parents.values()):
            physical = _object(
                _object(heartbeats[parent].get("inference"), "actor-inference").get(
                    "physical_inference"
                ),
                "physical-neural-work",
            )
            _require(
                _int(physical.get("neural_rows"))
                and _int(physical.get("neural_calls")),
                "physical-neural-counters",
            )
            if physical["neural_rows"] == 0 or physical["neural_calls"] == 0:
                return None
            neural_rows += physical["neural_rows"]
        proof = {
            "policy": policy.as_dict(),
            "role": role,
            "unit": asdict(unit),
            "clock": asdict(clock),
            "births": [
                {"process": asdict(p), "upper_wall_ns": births[p]}
                for p in sorted(
                    {unit.main, coordinator_process, *current.values()},
                    key=lambda p: p.pid,
                )
            ],
            "verified_champions": dict(verified_champions),
            "coordinator": coordinator,
            "heartbeats": heartbeats,
            "cohorts": cohorts,
            "metrics": metrics,
            "continuation_state": continuity,
            "replay_counter": counter,
            "arena_prefix_preserved": True,
        }
        proof_sha = hashlib.sha256(
            json.dumps(
                proof, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        return {
            "role": role,
            "invocation_id": unit.invocation_id,
            "continuation_started_ns": policy.continuation_started_ns,
            "profile_sha256": policy.profile_sha256,
            "source_commit": policy.source_commit,
            "learning_rates": list(policy.learning_rates),
            "ema_decay": policy.ema_decay,
            "workers": sorted(current),
            "worker_processes": {
                name: asdict(process) for name, process in current.items()
            },
            "coordinator_process": asdict(coordinator_process),
            "cohorts": len(cohorts),
            "actor_sources": ["champion"],
            "finite_metrics": True,
            "arena_prefix_preserved": True,
            "replay_committed_samples": counter["committed_samples"],
            "step": metrics[-1]["step"],
            "neural_work": neural_rows,
            "metric_intervals": len(metrics),
            "labels": labels,
            "captured_ns": clock.wall_ns,
            "evidence_sha256": proof_sha,
        }
    except _Pending:
        return None
