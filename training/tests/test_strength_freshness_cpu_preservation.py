from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.strength_freshness_cpu_preservation import (
    PROCESS_FIELDS,
    PHYSICAL_WORK_CONTRACT,
    R3_SOURCE_COMMIT,
    STATIC_FIELDS,
    PreservationViolation,
    verify_preservation,
)


@pytest.fixture
def observations():
    now = 1_800_000_000_000_000_000
    workers = [
        "learner",
        *[f"actor-gpu-{i}" for i in range(1, 7)],
        "actor-cpu-ring4",
        "arena-promotion",
    ]
    cohorts = [f"cohort-{i}" for i in range(24)]
    static: dict[str, Any] = {k: "a" * 64 for k in STATIC_FIELDS}
    static.update(
        run_id="run",
        generation_family="family",
        source_commit=R3_SOURCE_COMMIT,
        continuation_started_ns=now - 10**12,
        runtime_name="training.service",
        runtime_cgroup="/system.slice/training.service",
    )
    processes = {
        name: {
            "pid": i + 100,
            "start_ticks": i + 1000,
            "cgroup": static["runtime_cgroup"],
            "invocation_id": "1" * 32,
            "restarts": 0,
            "origin_sha256": "c" * 64,
            "heartbeat_ns": now - 10**9,
        }
        for i, name in enumerate(["controller", "coordinator", *workers])
    }
    roles = ["learner", *[f"actor-gpu-{i}" for i in range(1, 7)], "arena-promotion"]
    gpu_roles = {f"GPU-{i}": role for i, role in enumerate(roles)}
    owners = [
        {
            "uuid": uuid,
            "role": role,
            "unit": static["runtime_name"],
            **{
                k: processes[role][k]
                for k in ("pid", "start_ticks", "cgroup", "invocation_id")
            },
        }
        for uuid, role in gpu_roles.items()
    ]
    support = {
        "monitor.service": {
            "kind": "long_running",
            "definition_sha256": "d" * 64,
            "environment_sha256": "e" * 64,
            "boot_links_sha256": "f" * 64,
            "enabled": True,
        },
        "backup.service": {
            "kind": "oneshot",
            "definition_sha256": "d" * 64,
            "environment_sha256": "e" * 64,
            "boot_links_sha256": "f" * 64,
            "enabled": False,
        },
        "backup.timer": {
            "kind": "timer",
            "definition_sha256": "d" * 64,
            "environment_sha256": "e" * 64,
            "boot_links_sha256": "f" * 64,
            "enabled": True,
        },
    }
    monitor = {
        "pid": 200,
        "start_ticks": 500,
        "cgroup": "/system.slice/monitor.service",
        "invocation_id": "2" * 32,
        "restarts": 0,
        "origin_sha256": "d" * 64,
    }
    recipe = {
        "learning_rates": [0.000533667, 8.005e-6, 8.005e-6],
        "ema_decay": 0.9999,
        "bootstrap_step": 1000,
        "replay_minimum_shard_id_exclusive": 900,
        "utd_segment_target_updates_per_new_sample": 1.5,
        "utd_segment_baseline_committed_replay_samples": 2000,
        "utd_segment_baseline_examples_consumed": 3000,
    }
    policy = {
        "static": static,
        "physical_work_contract_sha256": PHYSICAL_WORK_CONTRACT,
        "workers": workers,
        "cohorts": cohorts,
        "gpu_uuids": list(gpu_roles),
        "gpu_roles": gpu_roles,
        "support": support,
        "expected_monitors": {"monitor.service": monitor},
        "expected_processes": {
            name: {k: row[k] for k in PROCESS_FIELDS} for name, row in processes.items()
        },
        "recipe": recipe,
        "maximum_age_ns": 120 * 10**9,
        "maximum_seconds": 600,
        "maximum_support_seconds": 1800,
        "learner_birth_upper_ns": now - 1000 * 10**9,
    }
    metrics = [
        {
            **{
                k: v
                for k, v in recipe.items()
                if k not in {"ema_decay", "bootstrap_step"}
            },
            "timestamp_ns": now - (3 - i) * 10**9,
            "worker": "learner",
            "step": step,
            "ema": {"decay": 0.9999, "num_updates": step - 1000},
            "losses": {"policy": 1.0, "outcome": 0.5},
            "gradient_norm": 1.0,
            "gradient_pre_clip_norm": 1.0,
            "gradient_post_clip_norm": 0.5,
            "nonfinite_loss_count": 0,
            "nonfinite_gradient_count": 0,
            "segment_updates_per_new_sample": 1.49,
            "gradient_diagnostics": {
                "global_norm_finite": True,
                "nonfinite_gradient_tensors": 0,
                "batch": {
                    "rows": 512,
                    "six_mode_unknown": {},
                    "label_availability": {"policy": 512, "outcome": 500},
                },
            },
            "policy_batch_metrics": {"unknown_provenance_rows": 0},
        }
        for i, step in enumerate([1010, 1020])
    ]
    champion = "sha256-" + "e" * 64
    before = {
        "clock": {"boot_id": "boot", "monotonic": 100.0, "wall_ns": now},
        "static": copy.deepcopy(static),
        "runtime_active": True,
        "processes": processes,
        "gpu_owners": owners,
        "champion_identity": champion,
        "cohorts": {
            name: {
                "model_identity": champion,
                "heartbeat_ns": now - 10**9,
                "progress_ns": now - 10**9,
                "games": 100,
            }
            for name in cohorts
        },
        "support": {
            name: {
                **row,
                "active": "inactive" if row["kind"] == "oneshot" else "active",
                "result": "success",
                "exit_code": 0,
                "running_seconds": 0,
                "job": None,
            }
            for name, row in support.items()
        },
        "progress": {
            "step": 1020,
            "phase": "update_to_data_wait",
            "heartbeat_ns": now - 10**9,
            "metrics": metrics,
        },
    }
    before["support"]["monitor.service"]["process"] = copy.deepcopy(monitor)
    after = copy.deepcopy(before)
    after["clock"].update(monotonic=110.0, wall_ns=now + 10 * 10**9)
    actors = {}
    for role in [f"actor-gpu-{i}" for i in range(1, 7)]:
        pair = {}
        for label, seconds, calls, rows in [
            ("first", 7, 100, 10000),
            ("second", 9, 102, 11024),
        ]:
            pair[label] = {
                "clock": {
                    "boot_id": "boot",
                    "monotonic": 100 + seconds + 0.5,
                    "wall_ns": now + int((seconds + 0.5) * 10**9),
                },
                "heartbeat": {
                    "worker": role,
                    "pid": processes[role]["pid"],
                    "phase": "shared_cohorts",
                    "heartbeat_ns": now + seconds * 10**9,
                    "inference": {
                        "worker_phase": "idle",
                        "worker_phase_since_ns": now + (seconds - 1) * 10**9,
                        "failed_requests": 0,
                        "worker_failures": 0,
                        "physical_inference": {
                            "neural_calls": calls,
                            "neural_rows": rows,
                        },
                    },
                },
            }
        actors[role] = pair
        after["processes"][role]["heartbeat_ns"] = now + 9 * 10**9
    after["physical_work"] = {"kind": "actor_broker", "actors": actors}
    for row in after["cohorts"].values():
        row["games"] += 1
    return policy, before, after, {champion: "f" * 64}


def window(observations):
    before = observations[1]["clock"]
    return {
        "attempt": {
            "attempt_id": "cpu-preservation-01",
            "nonce": "a" * 32,
            "plan_sha256": "b" * 64,
            "boot_id": before["boot_id"],
            "started_monotonic": before["monotonic"] + 3,
            "started_wall_ns": before["wall_ns"] + 3 * 10**9,
        },
        "cleanup_clock": {
            "boot_id": before["boot_id"],
            "monotonic": before["monotonic"] + 5,
            "wall_ns": before["wall_ns"] + 5 * 10**9,
        },
        "audit_clock": {
            "boot_id": before["boot_id"],
            "monotonic": before["monotonic"] + 11,
            "wall_ns": before["wall_ns"] + 11 * 10**9,
        },
    }


def verify(observations, **kwargs):
    policy, before, after, champions = observations
    return verify_preservation(
        policy,
        before,
        after,
        verified_champions=champions,
        **(kwargs or window(observations)),
    )


def test_credit_wait_is_not_a_stall_and_does_not_certify_execution(observations):
    result = verify(observations)
    assert (
        result["status"] == "passed" and result["step_before"] == result["step_after"]
    )
    assert not result["execution_qualified"] and not result["cuda_qualified"]


@pytest.mark.parametrize(
    "field",
    [
        "profile_sha256",
        "source_manifest_sha256",
        "native_sha256",
        "runtime_definition_sha256",
        "runtime_environment_sha256",
        "runtime_boot_links_sha256",
    ],
)
def test_static_authority_changes_refuse(observations, field):
    observations[2]["static"][field] = "0" * 64
    with pytest.raises(PreservationViolation):
        verify(observations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("start_ticks", 9999),
        ("invocation_id", "3" * 32),
        ("restarts", 1),
        ("cgroup", "/other"),
        ("origin_sha256", "0" * 64),
    ],
)
def test_pid_reuse_restarts_import_and_cgroup_changes_refuse(
    observations, field, value
):
    observations[2]["processes"]["learner"][field] = value
    with pytest.raises(PreservationViolation):
        verify(observations)


def test_unplanned_before_owner_is_not_a_valid_baseline(observations):
    observations[1]["processes"]["controller"]["pid"] = 999
    observations[2]["processes"]["controller"]["pid"] = 999
    with pytest.raises(PreservationViolation):
        verify(observations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("uuid", "GPU-unknown"),
        ("role", "actor-gpu-1"),
        ("invocation_id", "3" * 32),
        ("unit", "foreign.service"),
    ],
)
def test_gpu_ownership_is_exact(observations, field, value):
    observations[2]["gpu_owners"][0][field] = value
    with pytest.raises(PreservationViolation):
        verify(observations)


@pytest.mark.parametrize("state", ["activating", "active", "deactivating"])
def test_scheduled_oneshot_may_advance_without_being_declared_complete(
    observations, state
):
    row = observations[2]["support"]["backup.service"]
    row.update(
        active=state,
        result="success",
        running_seconds=30,
        job={"kind": "start", "age_seconds": 2},
        pid=500,
        invocation_id="new",
    )
    assert verify(observations)["status"] == "passed"


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "overdue",
        "queued-overdue",
        "monitor-restart",
        "static-change",
        "timer-off",
    ],
)
def test_support_faults_are_not_normal_progress(observations, fault):
    support = observations[2]["support"]
    if fault == "failed":
        support["backup.service"].update(result="exit-code", exit_code=1)
    elif fault == "overdue":
        support["backup.service"].update(active="active", running_seconds=1801)
    elif fault == "queued-overdue":
        support["backup.service"]["job"] = {"kind": "start", "age_seconds": 1801}
    elif fault == "monitor-restart":
        support["monitor.service"]["process"]["start_ticks"] += 1
    elif fault == "static-change":
        support["backup.service"]["environment_sha256"] = "0" * 64
    else:
        support["backup.timer"]["active"] = "inactive"
    with pytest.raises(PreservationViolation):
        verify(observations)


def test_verified_promotion_and_old_champion_cohorts_are_allowed(observations):
    _, _, after, verified = observations
    new = "sha256-" + "b" * 64
    verified[new] = "a" * 64
    after["champion_identity"] = new
    after["cohorts"]["cohort-0"]["model_identity"] = new
    assert verify(observations)["champion_changed"]


def test_unknown_teacher_never_becomes_champion_by_label(observations):
    observations[2]["cohorts"]["cohort-0"]["model_identity"] = "candidate"
    with pytest.raises(PreservationViolation):
        verify(observations)


@pytest.mark.parametrize(
    "fault",
    [
        "nan-loss",
        "infinite-norm",
        "diagnostic",
        "lr",
        "ema-origin",
        "watermark",
        "utd",
        "provenance",
        "no-labels",
    ],
)
def test_training_contract_faults_refuse(observations, fault):
    metrics = observations[2]["progress"]["metrics"]
    m = metrics[-1]
    if fault == "nan-loss":
        m["losses"]["policy"] = float("nan")
    elif fault == "infinite-norm":
        m["gradient_post_clip_norm"] = float("inf")
    elif fault == "diagnostic":
        m["gradient_diagnostics"]["global_norm_finite"] = False
    elif fault == "lr":
        m["learning_rates"][0] *= 2
    elif fault == "ema-origin":
        m["ema"]["num_updates"] -= 1
    elif fault == "watermark":
        m["replay_minimum_shard_id_exclusive"] -= 1
    elif fault == "utd":
        m["segment_updates_per_new_sample"] = 1.51
    elif fault == "provenance":
        m["policy_batch_metrics"]["unknown_provenance_rows"] = 1
    else:
        for m in metrics:
            m["gradient_diagnostics"]["batch"]["label_availability"]["outcome"] = 0
    with pytest.raises(PreservationViolation):
        verify(observations)


@pytest.mark.parametrize(
    "fault",
    [
        "boot",
        "window",
        "future",
        "stale",
        "worker-heartbeat",
        "step",
        "neural-reset",
        "games",
    ],
)
def test_clock_liveness_and_progress_cannot_be_replayed(observations, fault):
    after = observations[2]
    if fault == "boot":
        after["clock"]["boot_id"] = "new"
    elif fault == "window":
        after["clock"]["monotonic"] += 600
    elif fault == "future":
        after["progress"]["heartbeat_ns"] = after["clock"]["wall_ns"] + 1
    elif fault == "stale":
        after["cohorts"]["cohort-0"]["progress_ns"] -= 121 * 10**9
    elif fault == "worker-heartbeat":
        after["processes"]["actor-gpu-1"]["heartbeat_ns"] -= 121 * 10**9
    elif fault == "step":
        after["progress"]["step"] -= 1
    elif fault == "neural-reset":
        actor_pair(observations)["second"]["heartbeat"]["inference"][
            "physical_inference"
        ]["neural_rows"] = 1
    else:
        after["cohorts"]["cohort-0"]["games"] = 0
    with pytest.raises(PreservationViolation):
        verify(observations)


def test_no_mutation_of_observation_or_policy(observations):
    original = copy.deepcopy(observations)
    verify(observations)
    assert observations == original


def test_identical_snapshots_do_not_prove_post_cleanup_work(observations):
    bounds = window(observations)
    observations[2].clear()
    observations[2].update(copy.deepcopy(observations[1]))
    with pytest.raises(PreservationViolation):
        verify(observations, **bounds)


def actor_pair(observations):
    return observations[2]["physical_work"]["actors"]["actor-gpu-1"]


@pytest.mark.parametrize("field", ["neural_rows", "neural_calls"])
def test_positive_old_physical_counters_must_actually_advance(observations, field):
    pair = actor_pair(observations)
    pair["second"]["heartbeat"]["inference"]["physical_inference"][field] = pair[
        "first"
    ]["heartbeat"]["inference"]["physical_inference"][field]
    with pytest.raises(PreservationViolation, match="physical-work-not-proved"):
        verify(observations)


def test_verified_promotion_does_not_invent_a_physical_counter_epoch(observations):
    _, _, after, verified = observations
    new = "sha256-" + "b" * 64
    verified[new] = "a" * 64
    after["champion_identity"] = new
    actor_pair(observations)["second"]["heartbeat"]["inference"]["physical_inference"][
        "neural_rows"
    ] = 10
    with pytest.raises(PreservationViolation, match="counter-reset"):
        verify(observations)


def test_fresh_periodic_heartbeat_cannot_replace_precleanup_phase(observations):
    pair = actor_pair(observations)
    pair["first"]["heartbeat"]["inference"]["worker_phase_since_ns"] = (
        window(observations)["cleanup_clock"]["wall_ns"] - 1
    )
    with pytest.raises(PreservationViolation, match="physical-producer-phase-time"):
        verify(observations)


@pytest.mark.parametrize(
    "fault",
    [
        "before-after-start",
        "before-old",
        "after-before-cleanup",
        "after-future",
        "after-old",
        "audit-over-600",
        "cleanup-over-540",
        "boot",
    ],
)
def test_attempt_cleanup_and_current_audit_are_bound(observations, fault):
    bounds = window(observations)
    if fault == "before-after-start":
        bounds["attempt"]["started_monotonic"] = 99
    elif fault == "before-old":
        bounds["attempt"]["started_wall_ns"] += 121 * 10**9
    elif fault == "after-before-cleanup":
        bounds["cleanup_clock"]["monotonic"] = 111
    elif fault == "after-future":
        bounds["audit_clock"]["wall_ns"] = observations[2]["clock"]["wall_ns"] - 1
    elif fault == "after-old":
        bounds["audit_clock"]["wall_ns"] += 121 * 10**9
    elif fault == "audit-over-600":
        bounds["audit_clock"]["monotonic"] = 704
    elif fault == "cleanup-over-540":
        bounds["cleanup_clock"]["monotonic"] = 644
    else:
        bounds["attempt"]["boot_id"] = "other-boot"
    with pytest.raises(PreservationViolation):
        verify(observations, **bounds)


@pytest.mark.parametrize(
    "fault", ["worker", "prebirth", "equal-time", "reversed-time", "after-heartbeat"]
)
def test_metrics_join_actual_learner_lifetime_and_heartbeat(observations, fault):
    p, _, after, _ = observations
    metrics = after["progress"]["metrics"]
    if fault == "worker":
        metrics[-1]["worker"] = "actor-gpu-1"
    elif fault == "prebirth":
        p["learner_birth_upper_ns"] = metrics[0]["timestamp_ns"]
    elif fault == "equal-time":
        metrics[-1]["timestamp_ns"] = metrics[0]["timestamp_ns"]
    elif fault == "reversed-time":
        metrics[-1]["timestamp_ns"] = metrics[0]["timestamp_ns"] - 1
    else:
        metrics[-1]["timestamp_ns"] = after["progress"]["heartbeat_ns"] + 1
    with pytest.raises(PreservationViolation, match="metric-worker-lifetime-or-order"):
        verify(observations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rates", [True, 8e-6, 8e-6]),
        ("learning_rates", [float("nan"), 8e-6, 8e-6]),
        ("learning_rates", [0.1]),
        ("ema_decay", True),
        ("ema_decay", 1.0),
        ("bootstrap_step", 1000.0),
        ("replay_minimum_shard_id_exclusive", True),
        ("utd_segment_baseline_committed_replay_samples", 2000.0),
        ("utd_segment_baseline_examples_consumed", -1),
        ("utd_segment_target_updates_per_new_sample", True),
        ("utd_segment_target_updates_per_new_sample", float("inf")),
    ],
)
def test_numeric_recipe_policy_is_strict(observations, field, value):
    observations[0]["recipe"][field] = value
    with pytest.raises(PreservationViolation, match="numeric-recipe"):
        verify(observations)


def test_verified_preservation_is_consumable_by_observer_without_invented_clock(
    observations, tmp_path
):
    from scripts import strength_freshness_cpu_lifecycle as life
    from test_strength_freshness_cpu_lifecycle import owner, terminal

    timestamps = window(observations)
    anchor = life.Anchor.from_dict(timestamps["attempt"])
    log = life.RawLog(tmp_path / "observer", anchor)

    def clock(elapsed):
        return life.Clock(
            anchor.boot_id,
            anchor.started_monotonic + elapsed,
            anchor.started_wall_ns + int(elapsed * 10**9),
        )

    begun = life.role_started(
        log, clock(1.05), "dispatcher", owner(anchor, "dispatcher")
    )
    cases = {
        "fixture": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    produced = life.dispatcher_result(log, clock(1.2), begun, cases, set(cases))
    dispatched = life.dispatcher_complete(
        log,
        clock(1.4),
        begun,
        produced,
        terminal(anchor, "dispatcher", 1.4, 1.3),
        set(cases),
    )
    facts = {
        "dispatcher_barrier": {
            "started": begun,
            "unit": terminal(anchor, "dispatcher", 2, 1.3),
            "clock": life.asdict(clock(2)),
        },
        "units": {
            "dummy.service": {
                "Id": "dummy.service",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": 0,
                "Job": "0",
                "members": [],
            }
        },
        "jobs": [],
        "members": [],
        "links": [],
        "leftovers": [],
    }
    cleaned = life.cleanup_complete(
        log, life.Clock(**timestamps["cleanup_clock"]), facts, {"dummy.service"}
    )
    result = verify(observations)
    candidate = life.observer_candidate(
        log,
        life.Clock(**timestamps["audit_clock"]),
        cleaned,
        dispatched,
        set(cases),
        result,
    )
    proof = log.load(candidate, "observer-candidate")
    observed = log.load(proof["raw"], "raw-observer")["r3_gate"]
    assert observed["owners_unchanged"] is observed["productive"] is True
    assert observed["observed_monotonic"] == observations[2]["clock"]["monotonic"]
    assert observed["observed_monotonic"] != timestamps["audit_clock"]["monotonic"]
    assert observed["step_before"] == observed["step_after"]
    assert all(value is False for value in proof["claims"].values())


def test_old_cohort_search_progress_is_valid_with_fresh_heartbeats_and_all_actor_work(
    observations,
):
    for snap in observations[1:3]:
        snap["cohorts"]["cohort-0"]["progress_ns"] = (
            observations[1]["clock"]["wall_ns"] - 500 * 10**9
        )
    result = verify(observations)
    assert result["physical_work"]["kind"] == "actor_broker"
    assert len(result["physical_work"]["actors"]) == 6
    assert result["step_after"] == result["step_before"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing-actor",
        "role",
        "pid",
        "owner-generation",
        "parent-draining",
        "missing-phase",
        "old-phase",
        "phase-regression",
        "future-phase",
        "failed-request",
        "worker-failure",
        "rows-reset",
        "calls-reset",
        "warmup-only",
        "periodic-rewrite",
        "read-order",
        "read-boot",
        "pre-cleanup-read",
        "future-read",
        "future-heartbeat",
        "unjoined-heartbeat",
        "legacy-field",
        "contract",
        "source",
        "cohort-future",
    ],
)
def test_physical_proof_faults_do_not_become_preservation(observations, fault):
    policy, before, after, _ = observations
    pair = actor_pair(observations)
    first, second = pair["first"], pair["second"]
    raw = second["heartbeat"]
    inference = raw["inference"]
    if fault == "missing-actor":
        after["physical_work"]["actors"].pop("actor-gpu-6")
    elif fault == "role":
        raw["worker"] = "actor-gpu-2"
    elif fault == "pid":
        raw["pid"] += 1
    elif fault == "owner-generation":
        after["processes"]["actor-gpu-1"]["start_ticks"] += 1
    elif fault == "parent-draining":
        raw["phase"] = "cohort_draining"
    elif fault == "missing-phase":
        inference.pop("worker_phase_since_ns")
    elif fault == "old-phase":
        first["heartbeat"]["inference"]["worker_phase_since_ns"] = before["clock"][
            "wall_ns"
        ]
    elif fault == "phase-regression":
        inference["worker_phase_since_ns"] = (
            first["heartbeat"]["inference"]["worker_phase_since_ns"] - 1
        )
    elif fault == "future-phase":
        inference["worker_phase_since_ns"] = raw["heartbeat_ns"] + 1
    elif fault == "failed-request":
        inference["failed_requests"] = 1
    elif fault == "worker-failure":
        inference["worker_failures"] = 1
    elif fault == "rows-reset":
        inference["physical_inference"]["neural_rows"] = 0
    elif fault == "calls-reset":
        inference["physical_inference"]["neural_calls"] = 0
    elif fault == "warmup-only":
        inference["physical_inference"] = {
            **first["heartbeat"]["inference"]["physical_inference"],
            "total_neural_rows": 1000000,
            "graph_warmup_rows": 1000000,
        }
    elif fault == "periodic-rewrite":
        inference["physical_inference"] = copy.deepcopy(
            first["heartbeat"]["inference"]["physical_inference"]
        )
    elif fault == "read-order":
        second["clock"] = copy.deepcopy(first["clock"])
    elif fault == "read-boot":
        second["clock"]["boot_id"] = "other"
    elif fault == "pre-cleanup-read":
        first["clock"] = copy.deepcopy(before["clock"])
    elif fault == "future-read":
        second["clock"]["wall_ns"] = after["clock"]["wall_ns"] + 1
    elif fault == "future-heartbeat":
        raw["heartbeat_ns"] = second["clock"]["wall_ns"] + 1
    elif fault == "unjoined-heartbeat":
        after["processes"]["actor-gpu-1"]["heartbeat_ns"] = raw["heartbeat_ns"] - 1
    elif fault == "legacy-field":
        after["progress"]["neural_heartbeat_ns"] = after["clock"]["wall_ns"]
    elif fault == "contract":
        policy["physical_work_contract_sha256"] = "e" * 64
    elif fault == "source":
        for value in (policy["static"], before["static"], after["static"]):
            value["source_commit"] = "e" * 40
    else:
        after["cohorts"]["cohort-0"]["progress_ns"] = after["clock"]["wall_ns"] + 1
    with pytest.raises(PreservationViolation):
        verify(observations)


def learner_route(observations):
    _, before, after, _ = observations
    now = before["clock"]["wall_ns"]
    after["physical_work"] = {"kind": "learner_metrics"}
    after["progress"]["step"] = 1040
    after["progress"]["heartbeat_ns"] = now + 9 * 10**9
    after["processes"]["learner"]["heartbeat_ns"] = now + 9 * 10**9
    for row, step, elapsed in zip(after["progress"]["metrics"], [1030, 1040], [6, 8]):
        row["step"] = step
        row["ema"]["num_updates"] = step - 1000
        row["timestamp_ns"] = now + elapsed * 10**9
    return after


def test_two_producer_learner_events_prove_intervening_post_cleanup_work(observations):
    learner_route(observations)
    result = verify(observations)
    assert result["physical_work"]["kind"] == "learner_metrics"
    assert result["step_after"] > result["step_before"]
    assert result["format"].endswith("-v2") and not result["cuda_qualified"]


@pytest.mark.parametrize(
    "fault",
    [
        "one-after",
        "none-after",
        "same-step",
        "nan",
        "wrong-worker",
        "extra-fields",
        "fake-source",
    ],
)
def test_learner_route_cannot_relabel_old_or_invalid_work(observations, fault):
    after = learner_route(observations)
    metrics = after["progress"]["metrics"]
    cleanup = window(observations)["cleanup_clock"]["wall_ns"]
    if fault == "one-after":
        metrics[0]["timestamp_ns"] = cleanup - 1
    elif fault == "none-after":
        metrics[0]["timestamp_ns"] = cleanup - 2
        metrics[1]["timestamp_ns"] = cleanup - 1
    elif fault == "same-step":
        metrics[1]["step"] = metrics[0]["step"]
    elif fault == "nan":
        metrics[1]["losses"]["policy"] = float("nan")
    elif fault == "wrong-worker":
        metrics[1]["worker"] = "actor-gpu-1"
    elif fault == "extra-fields":
        after["physical_work"]["invented_completion_ns"] = cleanup + 1
    else:
        for value in (
            observations[0]["static"],
            observations[1]["static"],
            after["static"],
        ):
            value["source_commit"] = "d" * 40
    with pytest.raises(PreservationViolation):
        verify(observations)


def test_later_batching_phase_and_same_inference_phase_remain_source_ordered(
    observations,
):
    pair = actor_pair(observations)
    pair["second"]["heartbeat"]["inference"]["worker_phase"] = "batching"
    assert verify(observations)["productive"]
    for sample in pair.values():
        sample["heartbeat"]["inference"]["worker_phase"] = "inference"
    pair["second"]["heartbeat"]["inference"]["worker_phase_since_ns"] = pair["first"][
        "heartbeat"
    ]["inference"]["worker_phase_since_ns"]
    assert verify(observations)["productive"]


@pytest.mark.parametrize(
    "field,value",
    [("neural_calls", True), ("neural_rows", float("nan")), ("neural_rows", -1)],
)
def test_physical_counters_have_integer_source_semantics(observations, field, value):
    actor_pair(observations)["second"]["heartbeat"]["inference"]["physical_inference"][
        field
    ] = value
    with pytest.raises(PreservationViolation, match="physical-neural-counters"):
        verify(observations)
