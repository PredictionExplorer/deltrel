from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from scripts import strength_freshness_cpu_collect_records as r

SHA = "a" * 64
MODEL = "sha256-" + SHA


def heartbeat(role, pid):
    return {
        "schema_version": 1,
        "worker": role,
        "pid": pid,
        "heartbeat_ns": 200,
        "progress_ns": 150,
        "phase": "training" if role == "learner" else "shared_cohorts",
        "step": 20,
    }


def records():
    kernels = {
        role: {
            "pid": 100 + i,
            "start_ticks": 20 + i,
            "cgroup": "/system.slice/run.service",
            "invocation_id": "a" * 32,
            "origin_sha256": SHA,
        }
        for i, role in enumerate(("controller", "coordinator", *r.WORKERS))
    }
    workers = {
        role: {
            "pid": kernels[role]["pid"],
            "role": "learner"
            if role == "learner"
            else "arena"
            if role == "arena-promotion"
            else "actor",
            "state": "running",
            "restart_count": 0,
            "failure_reason": None,
            "failure_class": None,
            "failure_exit_code": None,
        }
        for role in r.WORKERS
    }
    coord = {
        "schema_version": 1,
        "timestamp_ns": 200,
        "coordinator_pid": kernels["coordinator"]["pid"],
        "state": "running",
        "draining": False,
        "failure": None,
        "hardware_failure_class": None,
        "hardware_failure_reason": None,
        "workers": workers,
    }
    hearts = {role: heartbeat(role, kernels[role]["pid"]) for role in r.WORKERS}
    return coord, hearts, kernels


def processes():
    coord, hearts, kernels = records()
    return r.process_records(coord, hearts, kernel_processes=kernels, unit_restarts=2)


def metric(stamp=150, step=10):
    result: dict[str, Any] = dict.fromkeys(r.METRIC_FIELDS, 0)
    result.update(
        schema_version=1,
        timestamp_ns=stamp,
        step=step,
        worker="learner",
        losses={"total": 1.5},
        gradient_diagnostics={
            "global_norm_finite": True,
            "nonfinite_gradient_tensors": 0,
            "batch": {"label_availability": {"policy": 512, "outcome": 225}},
        },
        learning_rates=[0.000533667, 0.000008005, 0.000008005],
        ema={"decay": 0.9999, "num_updates": step},
        policy_batch_metrics={"unknown_provenance_rows": 0},
        full_original_extra={"keep": [1, 2]},
    )
    return result


def progress(rows=None, heart=None):
    owner = processes()["learner"]
    h = heart if heart is not None else heartbeat("learner", owner["pid"])
    return r.learner_progress(
        h,
        rows if rows is not None else [metric(), metric(160, 20)],
        process=owner,
        max_age_ns=100,
        capture_wall_ns=210,
        birth_upper_ns=100,
    )


def authority():
    path = "/run/real/profile-strength-continuation.yaml"
    data = b"learner: fixture\n"
    run = {
        "schema_version": 1,
        "run_id": "variant-network",
        "generation_family": "variant-network-a",
        "created_ns": 10,
    }
    cont = {
        "schema_version": 1,
        "phase": "continuation",
        "continuation_started_ns": 100,
        "plan_sha256": SHA,
        "profile": path,
        "provisioned_gpus": 8,
        "attempts": [{"pid": 101, "started_ns": 101, "status": "running"}],
    }
    args = {
        "profile_sha256_bytes": (
            hashlib.sha256(data).hexdigest() + "  " + path + "\n"
        ).encode(),
        "profile_bytes": data,
        "profile_path": path,
        "source_commit_bytes": (r.R3_SOURCE_COMMIT + "\n").encode(),
        "expected_source_commit": r.R3_SOURCE_COMMIT,
    }
    return run, cont, args


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b"[]",
        b"{",
        b"\xff",
        b'{"x":1}\n{"y":2}',
        b'{"x":' + b"[" * 34 + b"0" + b"]" * 34 + b"}",
    ],
)
def test_json_refuses_malformed_duplicate_nonfinite_or_complex(raw):
    with pytest.raises(r.RecordRefusal):
        r.parse_json(raw)


def test_json_preserves_large_ints_and_raw_fields():
    row = {"wall_ns": 1791033122275362000, "raw": {"legacy_field": [1, None, "x"]}}
    assert r.parse_json(json.dumps(row).encode()) == row
    with pytest.raises(r.RecordRefusal):
        r.parse_json(b"{}", maximum_bytes=True)


def test_tail_only_discards_boundary_fragments_with_counts():
    raw = b'partial\n{"x":1}\n{"x":2}\nunterminated'
    got = r.parse_metric_tail(raw, offset=100)
    assert got == {
        "rows": [{"x": 1}, {"x": 2}],
        "discarded_first_bytes": 8,
        "discarded_last_bytes": 12,
    }
    assert r.parse_metric_tail(b'{"x":1}\n', offset=1)["rows"] == []
    assert r.parse_metric_tail(b'{"x":1}', offset=0)["discarded_last_bytes"] == 7


@pytest.mark.parametrize(
    "raw", [b'{"x":1}\nBAD\n', b'{"x":1}\n\n', b'{"x":1}\n{"x":NaN}\n', b"[]\n"]
)
def test_complete_tail_line_cannot_be_skipped(raw):
    with pytest.raises(r.RecordRefusal):
        r.parse_metric_tail(raw, offset=0)


def test_tail_overflow_does_not_hide_old_failure():
    with pytest.raises(r.RecordRefusal, match="tail-row-bound"):
        r.parse_metric_tail(b"{}\n{}\n", offset=0, max_rows=1)


def test_process_roster_kernel_join_and_counter_scopes():
    coord, hearts, kernels = records()
    original = copy.deepcopy((coord, hearts, kernels))
    coord["workers"]["learner"]["restart_count"] = 3
    got = r.process_records(coord, hearts, kernel_processes=kernels, unit_restarts=2)
    assert (
        len(got) == 11
        and got["controller"]["restarts"] == 2
        and got["coordinator"]["restarts"] == 2
        and got["learner"]["restarts"] == 3
    )
    assert "heartbeat_ns" not in got["controller"]
    got["learner"]["pid"] = 1
    assert kernels == original[2]


@pytest.mark.parametrize(
    "fault",
    [
        "coordinator_pid",
        "worker_pid",
        "heartbeat_pid",
        "role",
        "stopped",
        "draining",
        "failure",
        "missing",
        "foreign_kernel",
        "invocation",
        "restarts",
        "bool",
    ],
)
def test_process_mismatches_refuse(fault):
    coord, hearts, kernels = records()
    if fault == "coordinator_pid":
        coord["coordinator_pid"] += 1
    if fault == "worker_pid":
        coord["workers"]["learner"]["pid"] += 1
    if fault == "heartbeat_pid":
        hearts["learner"]["pid"] += 1
    if fault == "role":
        coord["workers"]["learner"]["role"] = "actor"
    if fault == "stopped":
        coord["workers"]["learner"]["state"] = "stopped"
    if fault == "draining":
        coord["draining"] = True
    if fault == "failure":
        coord["failure"] = "secret-error"
    if fault == "missing":
        del hearts["learner"]
    if fault == "foreign_kernel":
        kernels["learner"]["cgroup"] = "/system.slice/other.service"
    if fault == "invocation":
        kernels["learner"]["invocation_id"] = "b" * 32
    if fault == "restarts":
        kernels["learner"]["restarts"] = 3
    if fault == "bool":
        coord["workers"]["learner"]["restart_count"] = False
    with pytest.raises(r.RecordRefusal) as error:
        r.process_records(coord, hearts, kernel_processes=kernels, unit_restarts=0)
    assert "secret-error" not in str(error.value)


def cohorts():
    owners = processes()
    for actor in r.ACTORS:
        owners[actor]["birth_upper_ns"] = 100
    rows = {}
    for name in r.COHORTS:
        parent = name.rsplit("-cohort-", 1)[0]
        row = heartbeat(name, owners[parent]["pid"])
        row.update(
            model_role="champion",
            requested_model_role="champion",
            model_version=MODEL,
            cumulative_games=40,
            cohort=0,
            games=128,
        )
        rows[name] = row
    return rows, owners


def test_cohort_maps_actual_cumulative_games_and_retains_times():
    rows, owners = cohorts()
    got = r.cohort_records(rows, processes=owners, verified_champions={MODEL: SHA})
    assert len(got) == 24 and got[r.COHORTS[0]] == {
        "model_identity": MODEL,
        "games": 40,
        "heartbeat_ns": 200,
        "progress_ns": 150,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("worker", "other"),
        ("model_role", "candidate"),
        ("requested_model_role", "history"),
        ("model_version", "sha256-" + "b" * 64),
        ("cohort", True),
        ("progress_ns", 201),
        ("cumulative_games", -1),
    ],
)
def test_cohort_unknown_teacher_or_wrong_join_refuses(field, value):
    rows, owners = cohorts()
    rows[r.COHORTS[0]][field] = value
    with pytest.raises(r.RecordRefusal):
        r.cohort_records(rows, processes=owners, verified_champions={MODEL: SHA})


def test_metric_loss_records_remain_unmodified_and_wait_is_valid():
    rows = [
        metric(),
        {
            "schema_version": 1,
            "timestamp_ns": 155,
            "worker": "learner",
            "event": "utd_wait",
        },
        metric(160, 20),
    ]
    original = copy.deepcopy(rows)
    got = progress(rows)
    assert got["metrics"] == [rows[0], rows[2]]
    got["metrics"][0]["full_original_extra"]["keep"].append(3)
    assert rows == original
    h = heartbeat("learner", processes()["learner"]["pid"])
    h["phase"] = "update_to_data_wait"
    assert progress(heart=h)["phase"] == "update_to_data_wait"


@pytest.mark.parametrize(
    "fault",
    [
        "timestamp",
        "after_heartbeat",
        "wrong_worker",
        "abort",
        "error",
        "counter",
        "unknown_event",
        "missing_field",
        "reversed_step",
        "missing_labels",
    ],
)
def test_metric_selection_never_hides_failure(fault):
    rows = [metric(), metric(160, 20)]
    if fault == "timestamp":
        rows[1]["timestamp_ns"] = 149
    if fault == "after_heartbeat":
        rows[1]["timestamp_ns"] = 201
    if fault == "wrong_worker":
        rows[0]["worker"] = "actor-gpu-1"
    if fault == "abort":
        rows.insert(
            0,
            {
                "schema_version": 1,
                "timestamp_ns": 105,
                "worker": "learner",
                "phase": "nonfinite_abort",
            },
        )
    if fault == "error":
        rows[0]["error"] = "secret-error"
    if fault == "counter":
        rows[0]["nonfinite_gradient_count"] = 1
    if fault == "unknown_event":
        rows.insert(
            1,
            {
                "schema_version": 1,
                "timestamp_ns": 155,
                "worker": "learner",
                "event": "new_unsupported",
            },
        )
    if fault == "missing_field":
        del rows[0]["gradient_diagnostics"]
    if fault == "reversed_step":
        rows[1]["step"] = rows[0]["step"]
    if fault == "missing_labels":
        del rows[0]["losses"]
    with pytest.raises(r.RecordRefusal) as error:
        progress(rows)
    assert "secret-error" not in str(error.value)


def test_old_lifetime_rows_excluded_but_current_old_failure_refuses():
    old = {"schema_version": 1, "timestamp_ns": 80, "phase": "nonfinite_abort"}
    assert len(progress([old, metric(), metric(160, 20)])["metrics"]) == 2
    old["timestamp_ns"] = 105
    old["worker"] = "learner"
    with pytest.raises(r.RecordRefusal):
        progress([old, metric(), metric(160, 20)])


def test_authority_exact_bytes_not_yaml_reinterpretation():
    run, cont, args = authority()
    got = r.authority_records(run, cont, **args)
    assert (
        got["continuation_started_ns"] == 100
        and got["source_commit"] == r.R3_SOURCE_COMMIT
    )
    assert got["profile_sha256"] == hashlib.sha256(args["profile_bytes"]).hexdigest()


@pytest.mark.parametrize(
    "fault",
    [
        "profile_bytes",
        "profile_path",
        "checksum",
        "source",
        "phase",
        "clock",
        "attempt",
        "run",
        "schema",
    ],
)
def test_authority_refuses_wrong_or_absent_facts(fault):
    run, cont, args = authority()
    if fault == "profile_bytes":
        args["profile_bytes"] += b" "
    if fault == "profile_path":
        cont["profile"] = "/other/profile.yaml"
    if fault == "checksum":
        args["profile_sha256_bytes"] += b"other\n"
    if fault == "source":
        args["source_commit_bytes"] = ("b" * 40).encode()
    if fault == "phase":
        cont["phase"] = "screen"
    if fault == "clock":
        cont["continuation_started_ns"] = 9
    if fault == "attempt":
        cont["attempts"][-1]["status"] = "stopped"
    if fault == "run":
        run["run_id"] = "../other"
    if fault == "schema":
        run["schema_version"] = True
    with pytest.raises(r.RecordRefusal):
        r.authority_records(run, cont, **args)


def pointer():
    return {
        "schema_version": 2,
        "format": "startrain.model-pointer",
        "role": "champion",
        "model_identity": MODEL,
        "model_step": 572377,
        "run_id": "variant-network",
        "generation_family": "variant-network-a",
        "manifest": "manifests/manifest-" + SHA + ".json",
        "manifest_sha256": SHA,
        "manifest_bytes": 701,
        "updated_ns": 120,
    }


def test_champion_requires_independently_verified_identity():
    assert r.champion_identity(pointer(), verified_champions={MODEL: SHA}) == MODEL
    with pytest.raises(r.RecordRefusal):
        r.champion_identity(pointer(), verified_champions={})


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "other"),
        ("role", "candidate"),
        ("schema_version", True),
        ("manifest", "../other"),
        ("manifest_sha256", "b" * 64),
        ("model_step", True),
    ],
)
def test_champion_pointer_shape(field, value):
    row = pointer()
    row[field] = value
    with pytest.raises(r.RecordRefusal):
        r.champion_identity(row, verified_champions={MODEL: SHA})


def test_cohort_uses_parent_kernel_bound_not_self_report():
    rows, owners = cohorts()
    parent = r.ACTORS[0]
    owners[parent]["birth_upper_ns"] = 120
    rows[r.COHORTS[0]]["process_started_ns"] = 1
    assert r.cohort_records(rows, processes=owners, verified_champions={MODEL: SHA})
    rows[r.COHORTS[0]]["progress_ns"] = 119
    with pytest.raises(r.RecordRefusal, match="cohort-before-birth"):
        r.cohort_records(rows, processes=owners, verified_champions={MODEL: SHA})


def test_process_keeps_explicit_kernel_birth_and_refuses_stale_heartbeat():
    coord, hearts, kernels = records()
    kernels["learner"]["birth_upper_ns"] = 190
    got = r.process_records(coord, hearts, kernel_processes=kernels, unit_restarts=0)
    assert got["learner"]["birth_upper_ns"] == 190
    kernels["learner"]["birth_upper_ns"] = 201
    with pytest.raises(r.RecordRefusal, match="heartbeat-before-birth"):
        r.process_records(coord, hearts, kernel_processes=kernels, unit_restarts=0)


def test_cohort_requires_caller_kernel_birth_bound():
    rows, owners = cohorts()
    del owners[r.ACTORS[0]]["birth_upper_ns"]
    with pytest.raises(r.RecordRefusal, match="record-fields"):
        r.cohort_records(rows, processes=owners, verified_champions={MODEL: SHA})


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_norm_finite", False),
        ("nonfinite_gradient_tensors", 1),
        ("nonfinite_gradient_tensors", False),
    ],
)
def test_stale_current_lifetime_diagnostic_failure_cannot_be_filtered(field, value):
    stale = metric(105, 1)
    stale["gradient_diagnostics"][field] = value
    with pytest.raises(r.RecordRefusal, match="metric-recorded-failure"):
        progress([stale, metric(), metric(160, 20)])


@pytest.mark.parametrize("stamp", [105, 150])
def test_unknown_event_on_loss_row_is_not_hidden(stamp):
    row = metric(stamp, 1)
    row["event"] = "new_unsupported"
    with pytest.raises(r.RecordRefusal, match="metric-event-unsupported"):
        progress([row, metric(155, 10), metric(160, 20)])
