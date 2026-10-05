from copy import deepcopy
from dataclasses import replace

import pytest

from deltreltrain.strength_freshness_guard import Clock, Process, Unit
from scripts.strength_freshness_progress import (
    ProgressPolicy,
    ProgressViolation,
    verify_progress,
)


@pytest.fixture
def capture():
    now = 1_800_000_000_000_000_000
    actors = [f"actor-gpu-{i}" for i in range(1, 7)]
    workers = ("learner", *actors, "actor-cpu-ring4", "arena-promotion")
    processes = {name: Process(100 + i, 500 + i) for i, name in enumerate(workers)}
    coordinator, controller = Process(80, 480), Process(70, 470)
    members = (controller, coordinator, *processes.values())
    unit = Unit(
        "training.service",
        "a" * 64,
        "/system.slice/training.service",
        invocation_id="1" * 32,
        main=controller,
        members=members,
        active="active",
        substate="running",
    )
    policy = ProgressPolicy(
        run_id="run",
        generation_family="family",
        expected_workers=workers,
        cohort_parents={
            f"{name}-cohort-{i}": name for name in actors for i in range(4)
        },
        learning_rates=(0.000533667, 0.000008005, 0.000008005),
        ema_decay=0.9999,
        ema_bootstrap_step=1000,
        replay_watermark=900,
        utd_target=1.5,
        utd_baseline_samples=2000,
        utd_baseline_examples=3000,
        continuation_started_ns=now - 100_000_000_000,
        profile_sha256="b" * 64,
        source_commit="c" * 40,
    )
    heartbeats = {
        name: {
            "worker": name,
            "pid": process.pid,
            "heartbeat_ns": now - 100_000_000,
            "step": 1020,
            "inference": {
                "neural_batches": 999999,
                "physical_inference": {"neural_rows": 1000, "neural_calls": 50},
            },
        }
        for name, process in processes.items()
    }
    champion = "sha256-" + "d" * 64
    metrics = []
    for i, step in enumerate((1010, 1020)):
        metrics.append(
            {
                "worker": "learner",
                "timestamp_ns": now - (2 - i) * 1_000_000_000,
                "step": step,
                "losses": {"policy": 1.0, "outcome": 0.5},
                "gradient_norm": 2.0,
                "nonfinite_loss_count": 0,
                "nonfinite_gradient_count": 0,
                "learning_rates": list(policy.learning_rates),
                "ema": {"decay": 0.9999, "num_updates": step - 1000},
                "replay_minimum_shard_id_exclusive": 900,
                "utd_segment_target_updates_per_new_sample": 1.5,
                "utd_segment_baseline_committed_replay_samples": 2000,
                "utd_segment_baseline_examples_consumed": 3000,
                "segment_updates_per_new_sample": 1.49,
                "gradient_diagnostics": {
                    "batch": {
                        "rows": 512,
                        "six_mode_unknown": {},
                        "label_availability": {"policy": 512, "outcome": 500},
                    }
                },
                "policy_batch_metrics": {"unknown_provenance_rows": 0},
            }
        )
    data = {
        "coordinator": {
            "coordinator_pid": coordinator.pid,
            "timestamp_ns": now - 100_000_000,
            "state": "running",
            "draining": False,
            "failure": None,
            "workers": {
                name: {"pid": process.pid, "state": "running", "restart_count": 0}
                for name, process in processes.items()
            },
        },
        "heartbeats": heartbeats,
        "cohorts": {
            name: {
                "worker": name,
                "pid": processes[parent].pid,
                "heartbeat_ns": now - 100_000_000,
                "model_version": champion,
                "requested_model_role": "champion",
                "model_role": "champion",
            }
            for name, parent in policy.cohort_parents.items()
        },
        "metrics": metrics,
        "continuation_state": {
            "continuation_started_ns": policy.continuation_started_ns
        },
        "profile_sha256": policy.profile_sha256,
        "source_commit": policy.source_commit,
        "replay_counter": {
            "run_id": "run",
            "generation_family": "family",
            "committed_samples": 2500,
            "updated_ns": now - 100_000_000,
        },
    }
    arguments = dict(
        role="r4",
        unit=unit,
        clock=Clock("boot", 20.0, now),
        births={p: now - 10_000_000_000 for p in members},
        verified_champions={champion: "e" * 64},
        arena_prefix_preserved=True,
    )
    return policy, data, arguments


def test_fresh_current_processes_produce_bound_evidence_without_mutation(capture):
    policy, data, args = capture
    before = deepcopy(data)
    result = verify_progress(policy, data, **args)
    assert result is not None
    assert result["step"] == 1020
    assert result["neural_work"] == 6000  # Never the broker's 999999 dispatches.
    assert result["labels"] == {"rows": 1024, "policy": 1024, "outcome": 1000}
    assert result["finite_metrics"] and len(result["evidence_sha256"]) == 64
    assert result["worker_processes"]["learner"] == {"pid": 100, "start_ticks": 500}
    assert result["coordinator_process"] == {"pid": 80, "start_ticks": 480}
    assert data == before
    assert verify_progress(policy, data, **args) == result
    assert ProgressPolicy.from_dict(policy.as_dict()) == policy


@pytest.mark.parametrize(
    "change",
    [
        "old-learner",
        "old-coordinator",
        "old-cohort",
        "missing-cohort",
        "old-metrics",
        "pid-reuse",
        "missing-worker",
        "queued-unit",
        "no-neural-work",
    ],
)
def test_stale_or_pending_startup_never_certifies_progress(capture, change):
    policy, data, args = capture
    old = args["clock"].wall_ns - 121_000_000_000
    if change == "old-learner":
        data["heartbeats"]["learner"]["heartbeat_ns"] = old
    elif change == "old-coordinator":
        data["coordinator"]["timestamp_ns"] = old
    elif change == "old-cohort":
        next(iter(data["cohorts"].values()))["heartbeat_ns"] = old
    elif change == "missing-cohort":
        data["cohorts"].pop(next(iter(data["cohorts"])))
    elif change == "old-metrics":
        for row in data["metrics"]:
            row["timestamp_ns"] = old
    elif change == "pid-reuse":
        prior = args["unit"].members[2]
        new = Process(prior.pid, prior.start_ticks + 100)
        args["unit"] = replace(
            args["unit"],
            members=tuple(new if p == prior else p for p in args["unit"].members),
        )
        args["births"][new] = args["clock"].wall_ns - 50_000_000
    elif change == "missing-worker":
        data["coordinator"]["workers"].pop("learner")
    elif change == "queued-unit":
        args["unit"] = replace(args["unit"], job={"id": 1, "kind": "start"})
    else:
        data["heartbeats"]["actor-gpu-1"]["inference"]["physical_inference"][
            "neural_rows"
        ] = 0
    assert verify_progress(policy, data, **args) is None


@pytest.mark.parametrize(
    "change",
    [
        "rate",
        "ema",
        "watermark",
        "utd",
        "utd-baseline",
        "nan-loss",
        "nan-gradient",
        "nonfinite-counter",
        "unknown-provenance",
        "clock",
        "source",
        "profile",
        "future-heartbeat",
        "future-metric",
        "label-overflow",
        "worker-restart",
        "duplicate-metric",
    ],
)
def test_current_contract_drift_refuses(capture, change):
    policy, data, args = capture
    row = data["metrics"][-1]
    if change == "rate":
        row["learning_rates"][0] *= 2
    elif change == "ema":
        row["ema"]["num_updates"] = row["step"] - 572377
    elif change == "watermark":
        row["replay_minimum_shard_id_exclusive"] = 0
    elif change == "utd":
        row["segment_updates_per_new_sample"] = 2.0
    elif change == "utd-baseline":
        row["utd_segment_baseline_examples_consumed"] = 0
    elif change == "nan-loss":
        row["losses"]["policy"] = float("nan")
    elif change == "nan-gradient":
        row["gradient_norm"] = float("nan")
    elif change == "nonfinite-counter":
        row["nonfinite_gradient_count"] = 1
    elif change == "unknown-provenance":
        row["policy_batch_metrics"]["unknown_provenance_rows"] = 1
    elif change == "clock":
        data["continuation_state"]["continuation_started_ns"] += 1
    elif change == "source":
        data["source_commit"] = "a" * 40
    elif change == "profile":
        data["profile_sha256"] = "a" * 64
    elif change == "future-heartbeat":
        data["heartbeats"]["learner"]["heartbeat_ns"] = args["clock"].wall_ns + 1
    elif change == "future-metric":
        row["timestamp_ns"] = args["clock"].wall_ns + 1
    elif change == "label-overflow":
        row["gradient_diagnostics"]["batch"]["label_availability"]["outcome"] = 513
    elif change == "worker-restart":
        data["coordinator"]["workers"]["learner"]["restart_count"] = 1
    else:
        data["metrics"].append(deepcopy(row))
    with pytest.raises(ProgressViolation):
        verify_progress(policy, data, **args)


def test_only_verified_champion_identities_may_overlap(capture):
    policy, data, args = capture
    old = "sha256-" + "f" * 64
    first = next(iter(data["cohorts"].values()))
    first["model_version"] = old
    with pytest.raises(ProgressViolation, match="unverified-teacher"):
        verify_progress(policy, data, **args)
    args["verified_champions"][old] = "1" * 64
    assert verify_progress(policy, data, **args) is not None
    first["requested_model_role"] = "candidate"
    with pytest.raises(ProgressViolation, match="candidate-teacher"):
        verify_progress(policy, data, **args)


@pytest.mark.parametrize(
    "phase", ["starting", "waiting_for_champion", "replay_reconciliation"]
)
def test_fresh_unassigned_startup_cohorts_are_pending(capture, phase):
    policy, data, args = capture
    for heartbeat in data["cohorts"].values():
        for key in ("requested_model_role", "model_role", "model_version"):
            heartbeat.pop(key)
        heartbeat["phase"] = phase
    assert verify_progress(policy, data, **args) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_model_role", "candidate"),
        ("model_role", "history"),
        ("model_version", "sha256-" + "a" * 64),
    ],
)
def test_starting_phase_does_not_hide_an_explicit_bad_teacher(capture, field, value):
    policy, data, args = capture
    heartbeat = next(iter(data["cohorts"].values()))
    for key in ("requested_model_role", "model_role", "model_version"):
        heartbeat.pop(key)
    heartbeat["phase"] = "starting"
    heartbeat[field] = value
    with pytest.raises(ProgressViolation):
        verify_progress(policy, data, **args)


def test_waits_for_value_labels_and_preserved_arena_prefix(capture):
    policy, data, args = capture
    args["arena_prefix_preserved"] = False
    assert verify_progress(policy, data, **args) is None
    args["arena_prefix_preserved"] = True
    for row in data["metrics"]:
        row["gradient_diagnostics"]["batch"]["label_availability"]["outcome"] = 0
    assert verify_progress(policy, data, **args) is None


def test_new_metrics_ahead_of_heartbeat_wait_without_using_old_process_rows(capture):
    policy, data, args = capture
    data["heartbeats"]["learner"]["step"] = 1010
    assert verify_progress(policy, data, **args) is None


@pytest.mark.parametrize("parent", ["controller", "coordinator"])
def test_surviving_old_children_cannot_certify_a_new_runtime(capture, parent):
    policy, data, args = capture
    unit = args["unit"]
    old = unit.main if parent == "controller" else unit.members[1]
    replacement = Process(old.pid, 9000)
    args["unit"] = replace(
        unit,
        main=replacement if parent == "controller" else unit.main,
        members=tuple(replacement if p == old else p for p in unit.members),
    )
    args["births"][replacement] = args["clock"].wall_ns - 200_000_000
    assert verify_progress(policy, data, **args) is None


def test_normal_replay_credit_wait_is_not_itself_a_stall(capture):
    policy, data, args = capture
    data["heartbeats"]["learner"]["phase"] = "update_to_data_wait"
    assert verify_progress(policy, data, **args) is not None


@pytest.mark.parametrize("target", ["coordinator", "worker"])
def test_recorded_current_failure_is_not_pending_telemetry(capture, target):
    policy, data, args = capture
    if target == "coordinator":
        data["coordinator"].update(state="failed", failure="worker exited")
    else:
        data["coordinator"]["workers"]["learner"].update(
            state="failed", failure_class="process_exit"
        )
    with pytest.raises(ProgressViolation, match="failure"):
        verify_progress(policy, data, **args)


@pytest.mark.parametrize("target", ["cohort", "actor", "learner"])
def test_fresh_failed_heartbeat_cannot_reuse_earlier_finite_progress(capture, target):
    policy, data, args = capture
    if target == "cohort":
        heartbeat = next(iter(data["cohorts"].values()))
    else:
        heartbeat = data["heartbeats"][
            "learner" if target == "learner" else "actor-gpu-1"
        ]
    heartbeat["phase"] = "failed"
    with pytest.raises(ProgressViolation, match="failed"):
        verify_progress(policy, data, **args)


@pytest.mark.parametrize("current", [True, False])
def test_abort_without_losses_is_checked_before_positive_metric_filter(
    capture, current
):
    policy, data, args = capture
    data["metrics"].append(
        {
            "worker": "learner",
            "phase": "nonfinite_abort",
            "step": 1021,
            "timestamp_ns": args["clock"].wall_ns - (1 if current else 20_000_000_000),
            "nonfinite_loss_count": 1,
            "nonfinite_gradient_count": 0,
        }
    )
    if current:
        with pytest.raises(ProgressViolation, match="failure-record"):
            verify_progress(policy, data, **args)
    else:
        assert verify_progress(policy, data, **args) is not None


def test_a_fatal_record_from_the_current_lifetime_does_not_age_into_success(capture):
    policy, data, args = capture
    for process in args["births"]:
        args["births"][process] = args["clock"].wall_ns - 300_000_000_000
    data["metrics"].insert(
        0,
        {
            "worker": "learner",
            "phase": "nonfinite_abort",
            "step": 1001,
            "timestamp_ns": args["clock"].wall_ns - 200_000_000_000,
            "nonfinite_loss_count": 1,
            "nonfinite_gradient_count": 0,
        },
    )
    with pytest.raises(ProgressViolation, match="failure-record"):
        verify_progress(policy, data, **args)


@pytest.mark.parametrize("failure", ["cohort", "metric"])
def test_pending_peer_does_not_hide_a_current_failure(capture, failure):
    policy, data, args = capture
    first = next(iter(data["cohorts"].values()))
    for key in ("requested_model_role", "model_role", "model_version"):
        first.pop(key)
    first["phase"] = "starting"
    if failure == "cohort":
        list(data["cohorts"].values())[-1]["phase"] = "failed"
    else:
        data["metrics"].append(
            {
                "worker": "learner",
                "phase": "nonfinite_abort",
                "timestamp_ns": args["clock"].wall_ns - 1,
                "nonfinite_loss_count": 1,
            }
        )
    with pytest.raises(ProgressViolation):
        verify_progress(policy, data, **args)


def test_wrong_process_birth_proof_or_policy_not_accepted(capture):
    policy, data, args = capture
    args["births"].pop(args["unit"].members[2])
    with pytest.raises(ProgressViolation, match="birth"):
        verify_progress(policy, data, **args)
    raw = policy.as_dict()
    raw["extra_ignored_setting"] = True
    with pytest.raises(ProgressViolation, match="keys"):
        ProgressPolicy.from_dict(raw)
    with pytest.raises(ProgressViolation, match="freshness"):
        replace(policy, max_age_seconds=121)
