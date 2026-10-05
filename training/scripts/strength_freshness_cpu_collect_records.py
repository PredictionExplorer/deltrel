"""Pure source-backed R3 telemetry normalization; no collection or qualification.

Inputs are privately captured JSON values/bytes plus caller-qualified kernel
owners. Preserve producer timestamps and complete loss records. Collection-time
freshness, provenance, active-plan admission and final v2 verdict remain caller
responsibilities; no filesystem, YAML, training package or runtime imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence, Set
from pathlib import PurePosixPath
from typing import Any, TypeGuard

R3_SOURCE_COMMIT = "7e77c135bb101f08d1534f0ca5406cb309e2e3d2"
ACTORS = tuple(f"actor-gpu-{i}" for i in range(1, 7))
WORKERS = ("learner", *ACTORS, "actor-cpu-ring4", "arena-promotion")
COHORTS = tuple(f"{actor}-cohort-{i}" for actor in ACTORS for i in range(4))
PROCESS_FIELDS = (
    "pid",
    "start_ticks",
    "cgroup",
    "invocation_id",
    "restarts",
    "origin_sha256",
)
MAX_JSON_BYTES = 1024 * 1024
MAX_TAIL_BYTES = 4 * MAX_JSON_BYTES
MAX_TAIL_ROWS = 4096
SAFE_EVENTS = {
    "replay_window_consumed",
    "replay_window_allocated",
    "replay_window_refreshed",
    "replay_loader_pool_started",
    "replay_loader_pool_rebound",
    "replay_loader_pool_shutdown",
    "recovery_checkpoint",
    "recovery_gc",
    "replay_gc",
    "replay_reconciliation",
    "promotion_candidate",
    "selfplay_snapshot",
    "selfplay_model_gc",
    "utd_wait",
}
METRIC_FIELDS = {
    "schema_version",
    "timestamp_ns",
    "worker",
    "step",
    "losses",
    "gradient_norm",
    "gradient_pre_clip_norm",
    "gradient_post_clip_norm",
    "nonfinite_loss_count",
    "nonfinite_gradient_count",
    "gradient_diagnostics",
    "learning_rates",
    "ema",
    "replay_minimum_shard_id_exclusive",
    "utd_segment_baseline_committed_replay_samples",
    "utd_segment_baseline_examples_consumed",
    "utd_segment_target_updates_per_new_sample",
    "segment_updates_per_new_sample",
    "policy_batch_metrics",
}


class RecordRefusal(ValueError):
    """Safe fixed reason; never includes arbitrary captured JSON content."""


def require(ok: object, code: str) -> None:
    if not ok:
        raise RecordRefusal(code)


def integer(value: object, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"sha256-[0-9a-f]{64}", value) is not None
    )


def _mapping(value: object) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), "record-object")
    assert isinstance(value, Mapping)
    return value


def _fields(value: object, required: Set[str]) -> Mapping[str, Any]:
    row = _mapping(value)
    require(required <= set(row), "record-fields")
    return row


def _schema(row: Mapping[str, Any], version: int = 1) -> None:
    require(
        type(row.get("schema_version")) is int and row["schema_version"] == version,
        "record-schema",
    )


def _bounded_value(
    value: Any, *, depth: int = 0, budget: list[int] | None = None
) -> None:
    if budget is None:
        budget = [200_000]
    budget[0] -= 1
    require(depth <= 32 and budget[0] >= 0, "record-complexity")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        require(math.isfinite(value), "record-nonfinite")
        return
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "record-key")
        for item in value.values():
            _bounded_value(item, depth=depth + 1, budget=budget)
    elif isinstance(value, list):
        for item in value:
            _bounded_value(item, depth=depth + 1, budget=budget)
    else:
        raise RecordRefusal("record-type")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        require(key not in result, "duplicate-json-key")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise RecordRefusal("record-nonfinite")


def parse_json(raw: bytes, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    require(
        type(raw) is bytes
        and integer(maximum_bytes, 1)
        and maximum_bytes <= MAX_JSON_BYTES
        and 0 < len(raw) <= maximum_bytes,
        "json-byte-bound",
    )
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
    except RecordRefusal:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise RecordRefusal("malformed-json") from None
    require(isinstance(value, dict), "record-object")
    _bounded_value(value)
    return value


def parse_metric_tail(
    raw: bytes, *, offset: int, max_rows: int = MAX_TAIL_ROWS
) -> dict[str, Any]:
    """Validate every complete line; only uncertain first/last fragments are dropped.

    With offset>0 the first line is conservatively discarded even when the read
    happened to start on a boundary. No malformed complete line is skipped, and
    exceeding max_rows refuses instead of hiding a failure behind truncation.
    """
    require(
        type(raw) is bytes
        and len(raw) <= MAX_TAIL_BYTES
        and integer(offset)
        and integer(max_rows, 1)
        and max_rows <= MAX_TAIL_ROWS,
        "tail-bound",
    )
    first = 0
    if offset and raw:
        index = raw.find(b"\n")
        first = len(raw) if index < 0 else index + 1
    remaining = raw[first:]
    last = 0
    if remaining and not remaining.endswith(b"\n"):
        index = remaining.rfind(b"\n")
        last = len(remaining) if index < 0 else len(remaining) - index - 1
        remaining = remaining[: len(remaining) - last]
    lines = remaining.split(b"\n")[:-1] if remaining else []
    require(len(lines) <= max_rows, "tail-row-bound")
    rows = [parse_json(line) for line in lines]
    return {"rows": rows, "discarded_first_bytes": first, "discarded_last_bytes": last}


def _heartbeat(row: Mapping[str, Any], *, worker: str, pid: int) -> None:
    row = _fields(
        row, {"schema_version", "worker", "pid", "heartbeat_ns", "phase", "progress_ns"}
    )
    _schema(row)
    require(
        integer(pid, 1)
        and integer(row["pid"], 1)
        and row["pid"] == pid
        and row["worker"] == worker,
        "heartbeat-owner",
    )
    require(
        integer(row["heartbeat_ns"], 1)
        and integer(row["progress_ns"], 1)
        and row["progress_ns"] <= row["heartbeat_ns"],
        "heartbeat-time",
    )
    require(
        isinstance(row["phase"], str)
        and row["phase"]
        not in {
            "starting",
            "stopped",
            "failed",
            "nonfinite_abort",
            "paused",
            "pause_ready",
        },
        "heartbeat-phase",
    )
    for key in ("failure", "failure_reason", "failure_class", "error", "error_type"):
        require(row.get(key) is None, "heartbeat-failure")


def process_records(
    coordinator: Mapping[str, Any],
    heartbeats: Mapping[str, Mapping[str, Any]],
    *,
    kernel_processes: Mapping[str, Mapping[str, Any]],
    unit_restarts: int,
) -> dict[str, Any]:
    row = _fields(
        coordinator,
        {
            "schema_version",
            "timestamp_ns",
            "coordinator_pid",
            "state",
            "draining",
            "failure",
            "hardware_failure_reason",
            "hardware_failure_class",
            "workers",
        },
    )
    _schema(row)
    require(
        row["state"] == "running"
        and row["draining"] is False
        and all(
            row[k] is None
            for k in ("failure", "hardware_failure_reason", "hardware_failure_class")
        ),
        "coordinator-state",
    )
    require(
        integer(row["timestamp_ns"], 1) and integer(unit_restarts),
        "coordinator-counter",
    )
    require(
        set(kernel_processes) == {"controller", "coordinator", *WORKERS}
        and set(heartbeats) == set(WORKERS)
        and isinstance(row["workers"], dict)
        and set(row["workers"]) == set(WORKERS),
        "process-roster",
    )
    output = {}
    for role, owner in kernel_processes.items():
        owner = _fields(owner, set(PROCESS_FIELDS) - {"restarts"})
        require(
            integer(owner["pid"], 1)
            and integer(owner["start_ticks"], 1)
            and isinstance(owner["cgroup"], str)
            and owner["cgroup"].startswith("/system.slice/")
            and isinstance(owner["invocation_id"], str)
            and re.fullmatch(r"[0-9a-f]{32}", owner["invocation_id"])
            and _sha(owner["origin_sha256"]),
            "kernel-owner",
        )
        normalized = {k: owner[k] for k in PROCESS_FIELDS if k != "restarts"}
        if "birth_upper_ns" in owner:
            require(integer(owner["birth_upper_ns"], 1), "kernel-birth-bound")
            normalized["birth_upper_ns"] = owner["birth_upper_ns"]
        if role in {"controller", "coordinator"}:
            normalized["restarts"] = unit_restarts
            if role == "coordinator":
                require(
                    row["coordinator_pid"] == owner["pid"]
                    and integer(row["coordinator_pid"], 1),
                    "coordinator-owner",
                )
                normalized["heartbeat_ns"] = row["timestamp_ns"]
        else:
            worker = _fields(
                row["workers"][role],
                {
                    "pid",
                    "role",
                    "state",
                    "restart_count",
                    "failure_reason",
                    "failure_class",
                    "failure_exit_code",
                },
            )
            expected_role = (
                "learner"
                if role == "learner"
                else "arena"
                if role == "arena-promotion"
                else "actor"
            )
            require(
                worker["role"] == expected_role
                and worker["state"] == "running"
                and integer(worker["pid"], 1)
                and worker["pid"] == owner["pid"]
                and integer(worker["restart_count"]),
                "worker-owner-or-state",
            )
            require(
                all(
                    worker[k] is None
                    for k in ("failure_reason", "failure_class", "failure_exit_code")
                ),
                "worker-failure",
            )
            _heartbeat(heartbeats[role], worker=role, pid=owner["pid"])
            normalized["restarts"] = worker["restart_count"]
            normalized["heartbeat_ns"] = heartbeats[role]["heartbeat_ns"]
        if "restarts" in owner:
            require(
                type(owner["restarts"]) is int
                and owner["restarts"] == normalized["restarts"],
                "kernel-restarts-conflict",
            )
        if "birth_upper_ns" in normalized and "heartbeat_ns" in normalized:
            require(
                normalized["birth_upper_ns"] < normalized["heartbeat_ns"],
                "heartbeat-before-birth",
            )
        output[role] = normalized
    require(
        len({r["pid"] for r in output.values()}) == len(output)
        and len({r["invocation_id"] for r in output.values()}) == 1
        and len({r["cgroup"] for r in output.values()}) == 1,
        "process-unit-join",
    )
    return copy.deepcopy(output)


def cohort_records(
    raw_by_name: Mapping[str, Mapping[str, Any]],
    *,
    processes: Mapping[str, Mapping[str, Any]],
    verified_champions: Mapping[str, str],
) -> dict[str, Any]:
    require(set(raw_by_name) == set(COHORTS), "cohort-roster")
    output = {}
    for name, raw in raw_by_name.items():
        parent = name.rsplit("-cohort-", 1)[0]
        owner = _fields(
            processes.get(parent), {"pid", "heartbeat_ns", "birth_upper_ns"}
        )
        _heartbeat(raw, worker=name, pid=owner["pid"])
        # Caller-qualified kernel lifetime, never heartbeat process_started_ns.
        require(
            integer(owner["birth_upper_ns"], 1)
            and owner["birth_upper_ns"] < raw["progress_ns"] <= raw["heartbeat_ns"],
            "cohort-before-birth",
        )
        row = _fields(
            raw,
            {
                "model_role",
                "requested_model_role",
                "model_version",
                "cumulative_games",
                "cohort",
            },
        )
        require(
            row["model_role"] == row["requested_model_role"] == "champion"
            and _identity(row["model_version"])
            and row["model_version"] in verified_champions
            and _sha(verified_champions[row["model_version"]]),
            "cohort-teacher",
        )
        require(
            integer(row["cumulative_games"]) and integer(row["cohort"]),
            "cohort-counter",
        )
        output[name] = {
            "model_identity": row["model_version"],
            "games": row["cumulative_games"],
            "heartbeat_ns": row["heartbeat_ns"],
            "progress_ns": row["progress_ns"],
        }
    return output


def learner_progress(
    heartbeat: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    process: Mapping[str, Any],
    max_age_ns: int,
    capture_wall_ns: int,
    birth_upper_ns: int,
) -> dict[str, Any]:
    owner = _fields(process, {"pid", "heartbeat_ns"})
    _heartbeat(heartbeat, worker="learner", pid=owner["pid"])
    require(
        integer(max_age_ns, 1)
        and max_age_ns <= 120 * 10**9
        and integer(capture_wall_ns, 1)
        and integer(birth_upper_ns, 1),
        "learner-window",
    )
    require(
        heartbeat["phase"] in {"training", "update_to_data_wait"}
        and integer(heartbeat.get("step"))
        and heartbeat["heartbeat_ns"] == owner["heartbeat_ns"]
        and birth_upper_ns < heartbeat["heartbeat_ns"] <= capture_wall_ns
        and capture_wall_ns - heartbeat["heartbeat_ns"] <= max_age_ns,
        "learner-heartbeat-join",
    )
    require(
        isinstance(metrics, (list, tuple)) and len(metrics) <= MAX_TAIL_ROWS,
        "metric-window-bound",
    )
    selected = []
    previous = -1
    for row in metrics:
        row = _fields(row, {"schema_version", "timestamp_ns"})
        _schema(row)
        stamp = row["timestamp_ns"]
        require(integer(stamp, 1) and previous < stamp, "metric-time-order")
        previous = stamp
        if stamp <= birth_upper_ns:
            continue
        require(stamp <= heartbeat["heartbeat_ns"], "metric-after-heartbeat")
        require(row.get("worker") == "learner", "metric-worker")
        require(
            row.get("phase") != "nonfinite_abort"
            and all(
                row.get(k) is None
                for k in ("error", "error_type", "failure", "failure_reason")
            ),
            "metric-recorded-failure",
        )
        for key in ("nonfinite_loss_count", "nonfinite_gradient_count"):
            if key in row:
                require(integer(row[key]) and row[key] == 0, "metric-recorded-failure")
        if "event" in row:
            require(row["event"] in SAFE_EVENTS, "metric-event-unsupported")
        if "losses" not in row:
            require(row.get("event") in SAFE_EVENTS, "metric-event-unsupported")
            continue
        _fields(row, METRIC_FIELDS)
        diagnostics = _fields(
            row["gradient_diagnostics"],
            {"global_norm_finite", "nonfinite_gradient_tensors"},
        )
        require(
            diagnostics["global_norm_finite"] is True
            and integer(diagnostics["nonfinite_gradient_tensors"])
            and diagnostics["nonfinite_gradient_tensors"] == 0,
            "metric-recorded-failure",
        )
        require(
            integer(row["step"]) and row["step"] <= heartbeat["step"], "metric-step"
        )
        if capture_wall_ns - stamp <= max_age_ns:
            selected.append(copy.deepcopy(dict(row)))
    require(2 <= len(selected) <= 100, "fresh-metric-count")
    require(
        all(a["step"] < b["step"] for a, b in zip(selected, selected[1:])),
        "metric-step-order",
    )
    return {
        "step": heartbeat["step"],
        "phase": heartbeat["phase"],
        "heartbeat_ns": heartbeat["heartbeat_ns"],
        "metrics": selected,
    }


def authority_records(
    run: Mapping[str, Any],
    continuation: Mapping[str, Any],
    *,
    profile_sha256_bytes: bytes,
    profile_bytes: bytes,
    profile_path: str,
    source_commit_bytes: bytes,
    expected_source_commit: str,
) -> dict[str, Any]:
    run = _fields(run, {"schema_version", "run_id", "generation_family", "created_ns"})
    _schema(run)
    for key in ("run_id", "generation_family"):
        require(
            isinstance(run[key], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run[key]),
            "run-identity",
        )
    row = _fields(
        continuation,
        {
            "schema_version",
            "phase",
            "continuation_started_ns",
            "plan_sha256",
            "profile",
            "provisioned_gpus",
            "attempts",
        },
    )
    _schema(row)
    require(
        row["phase"] == "continuation"
        and integer(row["continuation_started_ns"], 1)
        and integer(run["created_ns"], 1)
        and run["created_ns"] <= row["continuation_started_ns"]
        and _sha(row["plan_sha256"])
        and type(row["provisioned_gpus"]) is int
        and row["provisioned_gpus"] == 8,
        "continuation-authority",
    )
    require(
        isinstance(profile_path, str)
        and profile_path.startswith("/")
        and str(PurePosixPath(profile_path)) == profile_path
        and ".." not in PurePosixPath(profile_path).parts
        and row["profile"] == profile_path,
        "profile-path",
    )
    require(
        type(profile_bytes) is bytes
        and 0 < len(profile_bytes) <= MAX_JSON_BYTES
        and type(profile_sha256_bytes) is bytes
        and len(profile_sha256_bytes) <= 4096,
        "profile-bound",
    )
    checksum = hashlib.sha256(profile_bytes).hexdigest()
    require(
        profile_sha256_bytes
        in {
            (checksum + "  " + profile_path).encode(),
            (checksum + "  " + profile_path + "\n").encode(),
        },
        "profile-checksum-authority",
    )
    require(
        isinstance(expected_source_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", expected_source_commit)
        and type(source_commit_bytes) is bytes
        and source_commit_bytes
        in {expected_source_commit.encode(), (expected_source_commit + "\n").encode()},
        "source-authority",
    )
    require(
        isinstance(row["attempts"], list)
        and row["attempts"]
        and len(row["attempts"]) <= 1000,
        "continuation-attempts",
    )
    for attempt in row["attempts"]:
        attempt = _fields(attempt, {"pid", "started_ns", "status"})
        require(
            integer(attempt["pid"], 1)
            and integer(attempt["started_ns"], 1)
            and attempt["started_ns"] >= row["continuation_started_ns"]
            and isinstance(attempt["status"], str),
            "continuation-attempt",
        )
    require(row["attempts"][-1]["status"] == "running", "continuation-not-running")
    return {
        "run_id": run["run_id"],
        "generation_family": run["generation_family"],
        "continuation_started_ns": row["continuation_started_ns"],
        "profile_sha256": checksum,
        "source_commit": expected_source_commit,
    }


def champion_identity(
    pointer: Mapping[str, Any], *, verified_champions: Mapping[str, str]
) -> str:
    row = _fields(
        pointer,
        {
            "schema_version",
            "format",
            "role",
            "model_identity",
            "model_step",
            "run_id",
            "generation_family",
            "manifest",
            "manifest_sha256",
            "manifest_bytes",
            "updated_ns",
        },
    )
    _schema(row, 2)
    require(
        row["format"] == "startrain.model-pointer"
        and row["role"] == "champion"
        and _identity(row["model_identity"])
        and integer(row["model_step"])
        and integer(row["updated_ns"], 1)
        and _sha(row["manifest_sha256"])
        and integer(row["manifest_bytes"], 1),
        "champion-pointer",
    )
    for key in ("run_id", "generation_family"):
        require(
            isinstance(row[key], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", row[key]),
            "champion-run",
        )
    require(
        isinstance(row["manifest"], str)
        and re.fullmatch(r"manifests/manifest-[0-9a-f]{64}\.json", row["manifest"])
        and row["manifest"] == "manifests/manifest-" + row["manifest_sha256"] + ".json",
        "champion-manifest",
    )
    identity = row["model_identity"]
    require(
        identity in verified_champions and _sha(verified_champions[identity]),
        "unverified-champion",
    )
    return identity
