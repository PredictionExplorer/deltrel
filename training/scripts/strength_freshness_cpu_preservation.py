"""Pure R3 before/after checks for the isolated CPU qualification experiment.

The collector must supply actual kernel/file/metric observations, not simulated
authority. Promotion proof validation is a separate prerequisite for the
verified_champions argument. This gate does not qualify a handoff or promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, TypeGuard


class PreservationViolation(ValueError):
    pass


STATIC_FIELDS = {
    "run_id",
    "generation_family",
    "continuation_started_ns",
    "runtime_name",
    "runtime_cgroup",
    "profile_sha256",
    "source_commit",
    "source_manifest_sha256",
    "native_sha256",
    "runtime_definition_sha256",
    "runtime_environment_sha256",
    "runtime_boot_links_sha256",
}
PROCESS_FIELDS = {
    "pid",
    "start_ticks",
    "cgroup",
    "invocation_id",
    "restarts",
    "origin_sha256",
}
SUPPORT_STATIC = {
    "kind",
    "definition_sha256",
    "environment_sha256",
    "boot_links_sha256",
    "enabled",
}


def require(condition: object, reason: str) -> None:
    if not condition:
        raise PreservationViolation(reason)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def integer(value: object, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fresh(stamp: object, now: int, age: int) -> None:
    require(integer(stamp, 1) and 0 <= now - stamp <= age, "stale-or-future-evidence")


def _clock(clock: Mapping[str, Any]) -> None:
    require(
        isinstance(clock["boot_id"], str)
        and clock["boot_id"]
        and finite(clock["monotonic"])
        and clock["monotonic"] >= 0
        and integer(clock["wall_ns"], 1),
        "capture-clock",
    )


def _attempt_window(
    policy: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    attempt: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    require(
        set(attempt)
        == {
            "attempt_id",
            "nonce",
            "plan_sha256",
            "boot_id",
            "started_monotonic",
            "started_wall_ns",
        }
        and isinstance(attempt["attempt_id"], str)
        and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt["attempt_id"])
        and isinstance(attempt["nonce"], str)
        and re.fullmatch(r"[0-9a-f]{32}", attempt["nonce"])
        and sha(attempt["plan_sha256"]),
        "attempt-binding",
    )
    start = {
        "boot_id": attempt["boot_id"],
        "monotonic": attempt["started_monotonic"],
        "wall_ns": attempt["started_wall_ns"],
    }
    a, b = before["clock"], after["clock"]
    for value in (start, a, b, cleanup, audit):
        _clock(value)
        require(value["boot_id"] == start["boot_id"], "production-boot-changed")
    age = policy["maximum_age_ns"]
    for key, scale in (("monotonic", 1), ("wall_ns", 10**9)):
        require(
            0 <= start[key] - a[key] <= age / 10**9 * scale,
            "before-not-fresh-pre-work",
        )
        require(
            start[key] <= cleanup[key] <= b[key] <= audit[key]
            and cleanup[key] - start[key] <= 540 * scale
            and audit[key] - start[key] <= policy["maximum_seconds"] * scale
            and 0 < b[key] - a[key] <= policy["maximum_seconds"] * scale
            and audit[key] - b[key] <= age / 10**9 * scale,
            "attempt-cleanup-audit-window",
        )


def _recipe(policy: Mapping[str, Any]) -> None:
    recipe = policy["recipe"]
    rates = recipe["learning_rates"]
    require(
        isinstance(rates, list)
        and len(rates) == 3
        and all(finite(v) and v > 0 for v in rates)
        and finite(recipe["ema_decay"])
        and 0 <= recipe["ema_decay"] < 1,
        "numeric-recipe-rates-or-ema",
    )
    require(
        all(
            integer(recipe[k])
            for k in (
                "bootstrap_step",
                "replay_minimum_shard_id_exclusive",
                "utd_segment_baseline_committed_replay_samples",
                "utd_segment_baseline_examples_consumed",
            )
        )
        and finite(recipe["utd_segment_target_updates_per_new_sample"])
        and recipe["utd_segment_target_updates_per_new_sample"] > 0,
        "numeric-recipe-credit",
    )
    require(integer(policy["learner_birth_upper_ns"], 1), "learner-lifetime-policy")


def _process(row: Mapping[str, Any], cgroup: str) -> dict[str, Any]:
    require(PROCESS_FIELDS <= set(row), "process-fields")
    require(integer(row["pid"], 1) and integer(row["start_ticks"], 1), "process-birth")
    require(
        row["cgroup"] == cgroup and integer(row["restarts"]),
        "process-cgroup-or-restarts",
    )
    require(
        isinstance(row["invocation_id"], str)
        and re.fullmatch(r"[0-9a-f]{32}", row["invocation_id"]),
        "process-invocation",
    )
    require(sha(row["origin_sha256"]), "process-origin")
    return {k: row[k] for k in PROCESS_FIELDS}


def _snapshot(
    policy: Mapping[str, Any], snap: Mapping[str, Any], verified: Mapping[str, str]
) -> dict[str, Any]:
    require(snap["static"] == policy["static"], "production-static-authority-drift")
    clock = snap["clock"]
    _clock(clock)
    now, age = clock["wall_ns"], policy["maximum_age_ns"]
    require(snap["runtime_active"] is True, "runtime-inactive")
    roles = {"controller", "coordinator", *policy["workers"]}
    require(set(snap["processes"]) == roles, "required-process-inventory")
    processes = {
        name: _process(row, policy["static"]["runtime_cgroup"])
        for name, row in snap["processes"].items()
    }
    require(
        processes == policy["expected_processes"], "production-owner-not-predeclared"
    )
    for name, row in snap["processes"].items():
        if name != "controller":
            _fresh(row["heartbeat_ns"], now, age)
    require(
        len({p["pid"] for p in processes.values()}) == len(processes),
        "duplicate-process",
    )
    require(
        len({p["invocation_id"] for p in processes.values()}) == 1,
        "process-unit-invocation",
    )
    owners = snap["gpu_owners"]
    require(
        len(owners) == len(policy["gpu_uuids"]) == 8
        and {r["uuid"] for r in owners} == set(policy["gpu_uuids"]),
        "gpu-inventory",
    )
    gpu = []
    for row in owners:
        require(
            row["role"] in policy["workers"]
            and row["unit"] == policy["static"]["runtime_name"],
            "foreign-gpu-role",
        )
        require(row["role"] == policy["gpu_roles"][row["uuid"]], "gpu-role-placement")
        process = processes[row["role"]]
        require(
            all(
                row[k] == process[k]
                for k in ("pid", "start_ticks", "cgroup", "invocation_id")
            ),
            "gpu-process-identity",
        )
        gpu.append(dict(row))
    require(set(snap["support"]) == set(policy["support"]), "support-inventory")
    stable_support = {}
    for name, row in snap["support"].items():
        expected = policy["support"][name]
        require({k: row[k] for k in SUPPORT_STATIC} == expected, "support-static-drift")
        kind = row["kind"]
        require(kind in {"long_running", "timer", "oneshot"}, "support-kind")
        job = row["job"]
        if job is not None:
            require(
                kind == "oneshot"
                and job["kind"] == "start"
                and finite(job["age_seconds"])
                and 0 <= job["age_seconds"] <= policy["maximum_support_seconds"],
                "support-job-unhealthy",
            )
        if kind == "long_running":
            require(
                row["active"] == "active" and row["result"] in {"", "success"},
                "monitor-unhealthy",
            )
            stable_support[name] = _process(row["process"], "/system.slice/" + name)
            require(
                stable_support[name] == policy["expected_monitors"][name],
                "monitor-owner-not-predeclared",
            )
        elif kind == "timer":
            require(row["active"] == "active", "timer-inactive")
        elif row["active"] in {"active", "activating", "deactivating"}:
            # The displayed Result may belong to the previous invocation.
            require(
                finite(row["running_seconds"])
                and 0 <= row["running_seconds"] <= policy["maximum_support_seconds"],
                "support-overdue",
            )
        else:
            require(
                row["active"] == "inactive"
                and row["result"] in {"", "success"}
                and row["exit_code"] == 0,
                "support-failed",
            )
    champion = snap["champion_identity"]
    require(champion in verified and sha(verified[champion]), "unverified-champion")
    require(set(snap["cohorts"]) == set(policy["cohorts"]), "cohort-inventory")
    for row in snap["cohorts"].values():
        require(
            row["model_identity"] in verified and sha(verified[row["model_identity"]]),
            "unverified-teacher",
        )
        _fresh(row["heartbeat_ns"], now, age)
        _fresh(row["progress_ns"], now, age)
        require(integer(row["games"]), "cohort-games")
    progress = snap["progress"]
    require(
        integer(progress["step"])
        and progress["phase"] in {"training", "update_to_data_wait"},
        "learner-not-productive",
    )
    _fresh(progress["heartbeat_ns"], now, age)
    require(
        policy["learner_birth_upper_ns"]
        < progress["heartbeat_ns"]
        <= snap["processes"]["learner"]["heartbeat_ns"],
        "learner-heartbeat-lifetime-join",
    )
    _fresh(progress["neural_heartbeat_ns"], now, age)
    require(integer(progress["neural_rows"], 1), "no-physical-neural-work")
    metrics = progress["metrics"]
    require(
        isinstance(metrics, list) and 2 <= len(metrics) <= 100, "finite-metric-window"
    )
    previous, previous_stamp = -1, -1
    recipe = policy["recipe"]
    labels = 0
    for metric in metrics:
        _fresh(metric["timestamp_ns"], now, age)
        require(
            metric["worker"] == "learner"
            and max(previous_stamp, policy["learner_birth_upper_ns"])
            < metric["timestamp_ns"]
            <= progress["heartbeat_ns"],
            "metric-worker-lifetime-or-order",
        )
        require(
            integer(metric["step"]) and previous < metric["step"] <= progress["step"],
            "metric-step-order",
        )
        previous = metric["step"]
        previous_stamp = metric["timestamp_ns"]
        require(
            metric["losses"]
            and all(finite(v) for v in metric["losses"].values())
            and all(
                finite(metric[k]) and metric[k] >= 0
                for k in (
                    "gradient_norm",
                    "gradient_pre_clip_norm",
                    "gradient_post_clip_norm",
                )
            ),
            "nonfinite-update",
        )
        require(
            all(
                integer(metric[k]) and metric[k] == 0
                for k in ("nonfinite_loss_count", "nonfinite_gradient_count")
            ),
            "nonfinite-counter",
        )
        require(
            metric["gradient_diagnostics"]["global_norm_finite"] is True
            and metric["gradient_diagnostics"]["nonfinite_gradient_tensors"] == 0,
            "nonfinite-gradient-diagnostic",
        )
        require(
            metric["learning_rates"] == recipe["learning_rates"]
            and isinstance(metric["learning_rates"], list)
            and all(finite(v) for v in metric["learning_rates"])
            and finite(metric["ema"]["decay"])
            and metric["ema"]["decay"] == recipe["ema_decay"]
            and integer(metric["ema"]["num_updates"])
            and metric["ema"]["num_updates"]
            == metric["step"] - recipe["bootstrap_step"],
            "calibrated-rates-or-ema",
        )
        require(
            all(
                integer(metric[k]) and metric[k] == recipe[k]
                for k in (
                    "replay_minimum_shard_id_exclusive",
                    "utd_segment_baseline_committed_replay_samples",
                    "utd_segment_baseline_examples_consumed",
                )
            )
            and finite(metric["utd_segment_target_updates_per_new_sample"])
            and metric["utd_segment_target_updates_per_new_sample"]
            == recipe["utd_segment_target_updates_per_new_sample"],
            "replay-credit-origin",
        )
        require(
            finite(metric["segment_updates_per_new_sample"])
            and 0
            <= metric["segment_updates_per_new_sample"]
            <= recipe["utd_segment_target_updates_per_new_sample"] + 1e-9,
            "replay-credit-overrun",
        )
        batch = metric["gradient_diagnostics"]["batch"]
        require(
            not batch["six_mode_unknown"]
            and metric["policy_batch_metrics"]["unknown_provenance_rows"] == 0,
            "unknown-policy-provenance",
        )
        require(integer(batch["rows"], 1), "batch-rows")
        for key in ("policy", "outcome"):
            n = batch["label_availability"][key]
            require(integer(n) and n <= batch["rows"], "batch-labels")
        require(batch["label_availability"]["policy"] > 0, "missing-policy-labels")
        labels += batch["label_availability"]["outcome"]
    require(labels > 0, "missing-outcome-labels")
    return {
        "processes": processes,
        "gpu_owners": sorted(gpu, key=lambda r: r["uuid"]),
        "stable_support": stable_support,
    }


def verify_preservation(
    policy: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    verified_champions: Mapping[str, str],
    attempt: Mapping[str, Any],
    cleanup_clock: Mapping[str, Any],
    audit_clock: Mapping[str, Any],
) -> dict[str, Any]:
    """Check caller-qualified captures inside the original attempt window.

    ``attempt`` is the separately sealed lifecycle Anchor, avoiding a circular
    plan/policy hash. Cleanup and audit clocks must be actual qualified facts.
    Collectors stamp each capture clock after its bounded observations; future
    telemetry is refused without a timing tolerance. The learner birth upper
    bound must derive from the predeclared kernel process identity, not from
    a self-reported metric or heartbeat field.
    A legitimate physical-counter reset conservatively refuses this CPU gate
    until its real counter-generation mapping is qualified. Such refusal is
    not a declaration that production training failed and authorizes no action.
    """
    require(set(policy["static"]) == STATIC_FIELDS, "static-policy-scope")
    static = policy["static"]
    require(
        isinstance(static["source_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", static["source_commit"]),
        "source-commit-policy",
    )
    require(
        integer(static["continuation_started_ns"], 1)
        and all(
            isinstance(static[k], str) and static[k]
            for k in ("run_id", "generation_family", "runtime_name")
        ),
        "run-identity-policy",
    )
    require(
        static["runtime_cgroup"] == "/system.slice/" + static["runtime_name"],
        "runtime-cgroup-policy",
    )
    require(
        integer(policy["maximum_age_ns"], 1)
        and policy["maximum_age_ns"] <= 120 * 10**9,
        "freshness-policy",
    )
    require(
        finite(policy["maximum_seconds"]) and 0 < policy["maximum_seconds"] <= 600,
        "lifecycle-policy",
    )
    require(
        finite(policy["maximum_support_seconds"])
        and 0 < policy["maximum_support_seconds"] <= 1800,
        "support-deadline-policy",
    )
    _recipe(policy)
    _attempt_window(policy, before, after, attempt, cleanup_clock, audit_clock)
    for key in STATIC_FIELDS:
        if key.endswith("sha256"):
            require(sha(policy["static"][key]), "static-policy-sha")
    require(
        len(set(policy["workers"])) == len(policy["workers"]) == 9
        and len(set(policy["cohorts"])) == len(policy["cohorts"]) == 24,
        "production-population-policy",
    )
    require(
        set(policy["gpu_roles"]) == set(policy["gpu_uuids"])
        and len(set(policy["gpu_roles"].values())) == 8,
        "gpu-placement-policy",
    )
    require(
        set(policy["expected_processes"])
        == {"controller", "coordinator", *policy["workers"]},
        "process-policy-inventory",
    )
    require(
        set(policy["expected_monitors"])
        == {
            name for name, p in policy["support"].items() if p["kind"] == "long_running"
        },
        "monitor-policy-inventory",
    )
    first = _snapshot(policy, before, verified_champions)
    last = _snapshot(policy, after, verified_champions)
    a, b = before["clock"], after["clock"]
    require(a["boot_id"] == b["boot_id"], "production-boot-changed")
    require(
        0 <= b["monotonic"] - a["monotonic"] <= policy["maximum_seconds"]
        and 0 <= b["wall_ns"] - a["wall_ns"] <= policy["maximum_seconds"] * 10**9,
        "observation-window",
    )
    require(first == last, "production-ownership-changed")
    old, new = before["progress"], after["progress"]
    require(
        new["step"] >= old["step"],
        "production-progress-reversed",
    )
    teacher_changed = before["champion_identity"] != after["champion_identity"]
    require(
        new["neural_rows"] > old["neural_rows"]
        and new["neural_heartbeat_ns"] > old["neural_heartbeat_ns"]
        and new["neural_heartbeat_ns"] > cleanup_clock["wall_ns"],
        "physical-work-not-proved-or-counter-reset",
    )
    require(
        all(
            after["cohorts"][k]["games"] >= before["cohorts"][k]["games"]
            for k in policy["cohorts"]
        ),
        "cohort-progress-reversed",
    )
    return {
        "format": "strength-freshness-cpu-r3-preservation-v1",
        "status": "passed",
        "owners_unchanged": True,
        "productive": True,
        "observed_monotonic": b["monotonic"],
        "policy_sha256": digest(policy),
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "verified_champions_sha256": digest(verified_champions),
        "attempt": dict(attempt),
        "cleanup_clock": dict(cleanup_clock),
        "audit_clock": dict(audit_clock),
        "elapsed_seconds": b["monotonic"] - a["monotonic"],
        "step_before": old["step"],
        "step_after": new["step"],
        "champion_changed": teacher_changed,
        "execution_qualified": False,
        "cuda_qualified": False,
        "scope": "CPU-experiment before/after preservation and finite progress only; no handoff, promotion, original native-history or playing-strength qualification.",
    }
