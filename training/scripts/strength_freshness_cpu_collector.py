"""Bounded read-only capture composition, without request authority or a CLI.

The caller authenticates registration, champion proofs, cleanup and any prior
support witness receipt. Reuse one ReadOnlyIO instance and its original deadline
and byte budget. Returned evidence is a measurement, never a qualification grant.
"""

from __future__ import annotations

import copy
import math
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from scripts import strength_freshness_cpu_collect_identity as identities
from scripts import strength_freshness_cpu_collect_processes as processes
from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_collect_support as supports
from scripts import strength_freshness_cpu_preservation as preservation
from scripts import strength_freshness_cpu_readonly as readonly


class CaptureRefusal(ValueError):
    """Fixed code plus redacted partial observation inventory for the caller."""

    def __init__(self, reason: str, provenance: Mapping[str, Any]):
        super().__init__(reason)
        self.provenance = copy.deepcopy(dict(provenance))


def require(ok: object, reason: str) -> None:
    if not ok:
        raise identities.CollectionRefusal(reason)


def _same_raw_clock(clock: Mapping[str, Any]) -> dict[str, Any]:
    identities.capture_clock(clock)
    return dict(clock)


_MISSING_COORDINATOR_PID: Any = object()


def _read_static_authority(
    identity, *, coordinator_pid: int, verified_champions: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """Read registered static facts; this operation does not admit its caller."""
    io, reg = identity.io, identity.reg
    policy = reg["policy"]

    def read(key: str) -> bytes:
        raw = identity.take(io.read(reg["keys"][key]))
        require(type(raw) is bytes, "metadata-bytes")
        return raw

    def read_json(key: str) -> dict[str, Any]:
        return records.parse_json(read(key))

    source = identity.verify_source_pins()
    cached = identity.verify_cached_references()
    actual_default = identity.take(io.query("default-target"))
    expected_default = reg["boot"]["default_target"]
    require(
        actual_default in {expected_default, expected_default + "\n"},
        "actual-default-target-drift",
    )
    targets = {
        name: identities.properties(identity.take(io.query("registered-target", name)))
        for name in io.scope.targets
    }
    for name, props in targets.items():
        require(
            props.get("Id") == name and props.get("LoadState") == "loaded",
            "boot-target-not-loaded",
        )
    units = {name: identity.unit(name) for name in reg["units"]}
    unit_static = {
        name: identity.unit_static(name, props, targets)
        for name, props in units.items()
    }
    runtime = policy["static"]["runtime_name"]
    for name, static in unit_static.items():
        if name == runtime:
            continue
        require(
            {"kind": reg["units"][name]["kind"], **static} == policy["support"][name],
            "support-static-policy",
        )
    require(
        set(units) == {runtime, *policy["support"]},
        "static-unit-policy-inventory",
    )
    run, continuation = read_json("run"), read_json("continuation")
    authority = records.authority_records(
        run,
        continuation,
        profile_sha256_bytes=read("profile_authority"),
        profile_bytes=read("profile"),
        profile_path=io.scope.files[reg["keys"]["profile"]].path,
        source_commit_bytes=read("run_source"),
        expected_source_commit=policy["static"]["source_commit"],
    )
    expected_source = policy["static"]["source_commit"]
    require(
        read("release_source")
        in {expected_source.encode(), (expected_source + "\n").encode()},
        "release-source-drift",
    )
    manifest = read("source_manifest")
    require(
        identities.sha(manifest) == policy["static"]["source_manifest_sha256"],
        "source-manifest-drift",
    )
    native_keys = {
        key for value in reg["origins"].values() for key in value["native_keys"]
    }
    require(
        native_keys
        and all(
            cached[key]["sha256"] == policy["static"]["native_sha256"]
            for key in native_keys
        ),
        "native-policy-binding",
    )
    # The legacy parsed-dict caller looked up this field only at this point.
    if coordinator_pid is _MISSING_COORDINATOR_PID:
        raise KeyError("coordinator_pid")
    require(
        continuation["attempts"][-1]["pid"] == coordinator_pid,
        "continuation-coordinator-owner",
    )
    pointer = read_json("champion")
    require(
        all(pointer[k] == authority[k] for k in ("run_id", "generation_family")),
        "champion-run-binding",
    )
    champion = records.champion_identity(pointer, verified_champions=verified_champions)
    static = {
        **authority,
        "source_manifest_sha256": identities.sha(manifest),
        "native_sha256": policy["static"]["native_sha256"],
        "runtime_name": runtime,
        "runtime_cgroup": units[runtime]["ControlGroup"],
        "runtime_definition_sha256": unit_static[runtime]["definition_sha256"],
        "runtime_environment_sha256": unit_static[runtime]["environment_sha256"],
        "runtime_boot_links_sha256": unit_static[runtime]["boot_links_sha256"],
    }
    require(static == policy["static"], "predeclared-static-drift")
    source_check = {
        "source_pins": source,
        "cached_references": cached,
        "static_sha256": preservation.digest(static),
        "default_target": expected_default,
    }
    return static, units, champion, source_check


def _check_final_support_states(
    policy_support, unit_specs, *, provenance, final_units
) -> None:
    """Refuse later unit facts contradicting the retained support observation."""
    support_states = provenance["unit_states"]
    require(
        set(support_states) == set(policy_support),
        "final-support-state-inventory",
    )
    for name, expected in support_states.items():
        fields = supports.DYNAMIC + (
            supports.SERVICE_DYNAMIC if unit_specs[name]["kind"] != "timer" else ()
        )
        require(
            set(expected) == set(fields)
            and all(
                key in final_units[name] and final_units[name][key] == expected[key]
                for key in fields
            ),
            "final-support-state-drift",
        )


def _project_support_at_clock(rows, observed_clock, final_clock):
    """Extend caller-owned rows in place, preserving the legacy alias behavior.

    This is the existing monotonic upper-bound projection, not an additional
    wall-clock policy or proof that a live invocation has completed.
    """
    require(
        observed_clock["boot_id"] == final_clock["boot_id"]
        and observed_clock["monotonic_ns"] <= final_clock["monotonic_ns"],
        "support-final-clock",
    )
    elapsed_ns = final_clock["monotonic_ns"] - observed_clock["monotonic_ns"]
    age_extension = (elapsed_ns + 999_999_999) // 1_000_000_000
    for row in rows.values():
        if row["running_seconds"] is not None:
            row["running_seconds"] += age_extension
        if row["job"] is not None:
            row["job"]["age_seconds"] += age_extension
    return rows, {
        "observed_clock": copy.deepcopy(observed_clock),
        "capture_clock": dict(final_clock),
        "added_upper_bound_seconds": age_extension,
    }


class _Capture:
    def __init__(
        self,
        identity,
        *,
        verified,
        phase,
        cleanup,
        physical_kind,
        interval,
        prior_witnesses,
        sleep,
    ):
        self.identity, self.io = identity, identity.io
        self.reg, self.policy = identity.reg, identity.reg["policy"]
        self.verified = copy.deepcopy(dict(verified))
        self.phase, self.cleanup, self.physical_kind = phase, cleanup, physical_kind
        self.interval, self.sleep = interval, sleep
        self.witnesses = {}
        self._witness_input = copy.deepcopy(prior_witnesses)
        self.process_collector = processes.ProcessCollector(identity)
        self.support_collector = supports.SupportCollector(identity)
        self.start = _same_raw_clock(identity.clock())
        require(
            0 < self.io.deadline - self.start["monotonic_ns"] / 1e9 <= 600,
            "capture-window-bound",
        )
        preservation._recipe(self.policy)
        require(
            type(self.policy["maximum_age_ns"]) is int
            and 0 < self.policy["maximum_age_ns"] <= 120 * 10**9,
            "capture-freshness-bound",
        )
        self.support_provenance = []
        self.source_checks = []
        self.tail_slices = []
        self.polls = 0
        self.producer_samples: dict[str, Any] = {}

    def provenance(self) -> dict[str, Any]:
        return {
            "format": "strength-preservation-capture-provenance-v1",
            "schema_version": 1,
            "phase": self.phase,
            "registration_sha256": preservation.digest(self.reg),
            "policy_sha256": preservation.digest(self.policy),
            "verified_champions_sha256": preservation.digest(self.verified),
            "encoding_contract_sha256": identities.CONTRACT,
            "read_start": dict(self.start),
            "raw_inventory": copy.deepcopy(self.identity.audit),
            "raw_inventory_sha256": preservation.digest(self.identity.audit),
            "identity_derivations": copy.deepcopy(self.identity.derivations),
            "support_witnesses": copy.deepcopy(self.witnesses),
            "support_provenance": copy.deepcopy(self.support_provenance),
            "source_checks": copy.deepcopy(self.source_checks),
            "tail_slices": copy.deepcopy(self.tail_slices),
            "polls": self.polls,
        }

    def _pause(self) -> None:
        now = self.identity.clock()
        remaining = self.io.deadline - now["monotonic_ns"] / 1e9
        require(remaining > self.interval, "capture-poll-deadline")
        require(self.polls < 256, "capture-poll-limit")
        self.polls += 1
        self.sleep(self.interval)

    def _read(self, key: str) -> bytes:
        raw = self.identity.take(self.io.read(self.reg["keys"][key]))
        require(type(raw) is bytes, "metadata-bytes")
        return raw

    def _json(self, key: str) -> dict[str, Any]:
        return records.parse_json(self._read(key))

    def _beats(self) -> dict[str, Any]:
        return {
            role: records.parse_json(self.identity.take(self.io.read(key)))
            for role, key in self.reg["heartbeats"].items()
        }

    def _cohorts(self) -> dict[str, Any]:
        rows = {}
        require(
            set(self.reg["cohorts"]) == set(records.COHORTS), "registered-cohort-names"
        )
        for name, item in self.reg["cohorts"].items():
            require(
                item["worker"] == name
                and item["parent_role"] == name.rsplit("-cohort-", 1)[0],
                "registered-cohort-parent",
            )
            rows[name] = records.parse_json(
                self.identity.take(self.io.read(item["key"]))
            )
        return rows

    def _tail(self) -> list[dict[str, Any]]:
        observed = self.io.tail(self.reg["keys"]["metrics"])
        raw = self.identity.take(observed)
        require(type(raw) is bytes, "metric-tail-bytes")
        parsed = records.parse_metric_tail(raw, offset=observed.audit["offset"])
        self.tail_slices.append(
            {
                "raw": copy.deepcopy(observed.audit["raw"]),
                "offset": observed.audit["offset"],
                "discarded_first_bytes": parsed["discarded_first_bytes"],
                "discarded_last_bytes": parsed["discarded_last_bytes"],
                "complete_rows": len(parsed["rows"]),
            }
        )
        return parsed["rows"]

    def _static(self, coordinator) -> tuple[dict[str, Any], dict[str, Any], str]:
        static, units, champion, source_check = _read_static_authority(
            self.identity,
            coordinator_pid=coordinator.get(
                "coordinator_pid", _MISSING_COORDINATOR_PID
            ),
            verified_champions=self.verified,
        )
        self.source_checks.append(source_check)
        return static, units, champion

    def _owners(self, units, coordinator, beats):
        got = self.process_collector.capture(
            unit_properties=units, coordinator=coordinator, heartbeats=beats
        )
        got["processes"] = records.process_records(
            coordinator,
            beats,
            kernel_processes=got["processes"],
            unit_restarts=processes.natural(
                units[self.policy["static"]["runtime_name"]]["NRestarts"]
            ),
        )
        return got

    def _support(self, owners):
        result = self.support_collector.capture(
            monitors=owners["monitors"], previous_witnesses=self._witness_input
        )
        self.witnesses = copy.deepcopy(result.witnesses)
        self._witness_input = copy.deepcopy(result.witnesses)
        self.support_provenance.append(copy.deepcopy(result.provenance))
        return copy.deepcopy(result.support)

    def _progress(self, tail, beats, owners, clock):
        learner = owners["processes"]["learner"]
        # The latest heartbeat is sampled after the metric tail. Kernel owner
        # fields stay fixed; only this joined producer heartbeat can advance.
        learner = {**learner, "heartbeat_ns": beats["learner"]["heartbeat_ns"]}
        return records.learner_progress(
            beats["learner"],
            tail,
            process=learner,
            max_age_ns=self.policy["maximum_age_ns"],
            capture_wall_ns=clock["wall_ns"],
            birth_upper_ns=owners["processes"]["learner"]["birth_upper_ns"],
        )

    def _learner_samples(self, owners):
        while True:
            tail = self._tail()
            beats = self._beats()
            now = identities.capture_clock(self.identity.clock())
            try:
                progress = self._progress(tail, beats, owners, now)
            except records.RecordRefusal as error:
                require(
                    str(error) in {"metric-after-heartbeat", "fresh-metric-count"},
                    str(error),
                )
                if str(error) == "fresh-metric-count":
                    eligible = [
                        r
                        for r in tail
                        if "losses" in r
                        and type(r.get("timestamp_ns")) is int
                        and now["wall_ns"] - self.policy["maximum_age_ns"]
                        <= r["timestamp_ns"]
                        <= beats["learner"]["heartbeat_ns"]
                    ]
                    require(len(eligible) < 2, "metric-window-overfull")
                self._pause()
                continue
            if self.phase == "before" or self.physical_kind == "actor_broker":
                return tail, beats
            post = [
                r
                for r in progress["metrics"]
                if r["timestamp_ns"] > self.cleanup["wall_ns"]
            ]
            if len(post) >= 2:
                return tail, beats
            self._pause()

    def _broker_samples(self, owners):
        actors = {}
        for role in sorted(preservation.ACTOR_ROLES):
            pair = {}
            for label in ("first", "second"):
                while True:
                    raw = records.parse_json(
                        self.identity.take(self.io.read(self.reg["heartbeats"][role]))
                    )
                    clock = identities.capture_clock(self.identity.clock())
                    owner = owners["processes"][role]
                    require(
                        raw.get("worker") == role
                        and type(raw.get("pid")) is int
                        and raw["pid"] == owner["pid"],
                        "broker-owner",
                    )
                    require(
                        type(raw.get("heartbeat_ns")) is int
                        and owner["birth_upper_ns"]
                        < raw["heartbeat_ns"]
                        <= clock["wall_ns"],
                        "broker-heartbeat-lifetime",
                    )
                    inference = raw.get("inference")
                    require(isinstance(inference, dict), "broker-inference-fields")
                    assert isinstance(inference, dict)
                    require(
                        all(
                            type(inference.get(k)) is int and inference[k] == 0
                            for k in ("failed_requests", "worker_failures")
                        ),
                        "broker-recorded-failure",
                    )
                    counters = inference.get("physical_inference")
                    require(
                        isinstance(counters, dict)
                        and all(
                            type(counters.get(k)) is int and counters[k] >= 0
                            for k in ("neural_calls", "neural_rows")
                        ),
                        "broker-neural-counters",
                    )
                    assert isinstance(counters, dict)
                    stamp = inference.get("worker_phase_since_ns")
                    allowed = {"idle", "waiting_for_device", "inference"} | (
                        {"batching"} if label == "second" else set()
                    )
                    ready = (
                        raw.get("phase") == "shared_cohorts"
                        and type(stamp) is int
                        and self.cleanup["wall_ns"] < stamp <= raw["heartbeat_ns"]
                        and inference.get("worker_phase") in allowed
                    )
                    if label == "second":
                        first = pair["first"]["heartbeat"]
                        previous = first["inference"]["physical_inference"]
                        require(
                            all(
                                counters[k] >= previous[k]
                                for k in ("neural_calls", "neural_rows")
                            ),
                            "broker-counter-reset",
                        )
                        ready = (
                            ready
                            and raw["heartbeat_ns"] > first["heartbeat_ns"]
                            and stamp >= first["inference"]["worker_phase_since_ns"]
                            and all(
                                counters[k] > previous[k]
                                for k in ("neural_calls", "neural_rows")
                            )
                        )
                    if ready:
                        pair[label] = {"clock": clock, "heartbeat": raw}
                        break
                    self._pause()
            actors[role] = pair
        self.producer_samples = {"kind": "actor_broker", "actors": actors}

    def run(self) -> dict[str, Any]:
        coordinator = self._json("coordinator")
        static, units, _champion = self._static(coordinator)
        beats = self._beats()
        first_owners = self._owners(units, coordinator, beats)
        self._support(first_owners)
        if self.phase == "after" and self.physical_kind == "actor_broker":
            self._broker_samples(first_owners)
        tail, _sampled_beats = self._learner_samples(first_owners)
        cohorts = self._cohorts()
        support = self._support(first_owners)
        # Recheck source/authority and manager facts, then refresh producer
        # heartbeat files after all tail/cohort reads before final owner join.
        final_coordinator = self._json("coordinator")
        final_static, final_units, champion = self._static(final_coordinator)
        require(static == final_static, "static-changed-during-capture")
        _check_final_support_states(
            self.policy["support"],
            self.reg["units"],
            provenance=self.support_provenance[-1],
            final_units=final_units,
        )
        final_beats = self._beats()
        final_owners = self._owners(final_units, final_coordinator, final_beats)
        processes.same_owners(first_owners, final_owners)
        clock_raw = self.identity.clock()
        clock = identities.capture_clock(clock_raw)
        support, support_projection = _project_support_at_clock(
            support, self.support_provenance[-1]["clock"], clock_raw
        )
        progress = self._progress(tail, final_beats, final_owners, clock)
        capture = {
            "clock": clock,
            "static": final_static,
            "runtime_active": True,
            "processes": final_owners["processes"],
            "gpu_owners": final_owners["gpu_owners"],
            "support": support,
            "champion_identity": champion,
            "cohorts": records.cohort_records(
                cohorts,
                processes=final_owners["processes"],
                verified_champions=self.verified,
            ),
            "progress": progress,
        }
        if self.phase == "after":
            capture["physical_work"] = (
                self.producer_samples
                if self.physical_kind == "actor_broker"
                else {"kind": "learner_metrics"}
            )
        # Reuse the exact v2 pure checks, without producing its cross-window
        # passed verdict or asserting any permission to execute a qualification.
        preservation._snapshot(self.policy, capture, self.verified)
        if self.phase == "after":
            preservation._physical_work(self.policy, capture, self.cleanup)
        provenance = self.provenance()
        provenance["support_age_projection"] = support_projection
        provenance.update(
            read_end=dict(clock_raw),
            capture_sha256=preservation.digest(capture),
            owner_rechecks=2,
            source_rechecks=2,
        )
        return {
            "format": "strength-preservation-capture-measurement-v1",
            "schema_version": 1,
            "phase": self.phase,
            "capture": capture,
            "provenance": provenance,
            "execution_qualified": False,
            "cuda_qualified": False,
        }


def collect_capture(
    registration: Mapping[str, Any],
    io: readonly.ReadOnlyIO,
    *,
    verified_champions: Mapping[str, str],
    phase: str,
    cleanup_clock: Mapping[str, Any] | None = None,
    physical_kind: str = "learner_metrics",
    poll_interval_seconds: float = 0.25,
    previous_support_witnesses: Mapping[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Return measured fields/provenance only; caller owns all request authority."""
    identity = None
    capture = None
    try:
        require(
            phase in {"before", "after"}
            and physical_kind in {"learner_metrics", "actor_broker"},
            "capture-mode",
        )
        require(
            type(poll_interval_seconds) in {int, float}
            and math.isfinite(poll_interval_seconds)
            and 0.05 <= poll_interval_seconds <= 5,
            "poll-interval",
        )
        require(
            type(io.deadline) in {int, float} and math.isfinite(io.deadline),
            "capture-deadline",
        )
        require(
            isinstance(verified_champions, Mapping)
            and verified_champions
            and all(
                re.fullmatch(r"sha256-[0-9a-f]{64}", k) and preservation.sha(v)
                for k, v in verified_champions.items()
            ),
            "verified-champion-map",
        )
        if phase == "after":
            require(
                isinstance(cleanup_clock, Mapping)
                and set(cleanup_clock) == {"boot_id", "monotonic", "wall_ns"},
                "cleanup-clock-required",
            )
            assert isinstance(cleanup_clock, Mapping)
            preservation._clock(cleanup_clock)
        else:
            require(
                cleanup_clock is None and previous_support_witnesses is None,
                "before-no-cleanup-authority",
            )
        identity = identities.IdentityCollector(registration, io, sleep=sleep)
        capture = _Capture(
            identity,
            verified=verified_champions,
            phase=phase,
            cleanup=cleanup_clock,
            physical_kind=physical_kind,
            interval=float(poll_interval_seconds),
            prior_witnesses=previous_support_witnesses,
            sleep=sleep,
        )
        if phase == "after":
            assert cleanup_clock is not None
            start = identities.capture_clock(capture.start)
            require(
                start["boot_id"] == cleanup_clock["boot_id"]
                and start["monotonic"] > cleanup_clock["monotonic"]
                and start["wall_ns"] > cleanup_clock["wall_ns"],
                "capture-before-cleanup",
            )
        return capture.run()
    except Exception as error:
        if isinstance(error, CaptureRefusal):
            raise
        safe_errors = (
            identities.CollectionRefusal,
            identities.facts.FactViolation,
            records.RecordRefusal,
            processes.ProcessRefusal,
            supports.SupportRefusal,
            preservation.PreservationViolation,
            readonly.ReadRefusal,
        )
        reason = (
            str(error)
            if isinstance(error, safe_errors)
            else "capture-component-refusal"
        )
        if re.fullmatch(r"[a-z][a-z0-9-]{0,127}", reason) is None:
            reason = "capture-component-refusal"
        provenance = (
            capture.provenance()
            if capture is not None
            else {
                "format": "strength-preservation-capture-provenance-v1",
                "phase": phase if phase in {"before", "after"} else None,
                "raw_inventory": []
                if identity is None
                else copy.deepcopy(identity.audit),
            }
        )
        if isinstance(error, readonly.ReadRefusal) and error.audit:
            provenance["failed_operation"] = copy.deepcopy(error.audit)
        raise CaptureRefusal(reason, provenance) from None
