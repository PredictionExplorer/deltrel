"""Local BEFORE measurements conditional on independently admitted premises.

This module never launches a process, writes a file, requests learner work, or
grants execution authority. One session owns its observer, publication tracker,
private append inputs and original IO window. Raw records never leave the session.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any

from scripts import strength_freshness_cpu_collector as common
from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_collect_support as support
from scripts import strength_freshness_cpu_learner_window as append
from scripts import strength_freshness_cpu_observed_kernel as kernel
from scripts import strength_freshness_cpu_preservation as preservation
from scripts import strength_freshness_cpu_publication_renewal as renewal
from scripts import strength_freshness_cpu_readonly as readonly
from scripts import strength_freshness_cpu_window_contract as contract


class BeforeRefusal(ValueError):
    """Fixed code only; never private source, record or exception content."""


class BeforeIncomplete(BeforeRefusal):
    """The single bounded measurement lacks evidence; no production-fault claim."""


def require(value: object, reason: str) -> None:
    if not value:
        raise BeforeRefusal(reason)


def digest(value: Any) -> str:
    return append.sha(append.encoded(value))


def _path_hash(value: str) -> str:
    return append.sha(value.encode("utf-8"))


@dataclass(frozen=True)
class BeforeMeasurement:
    _capture: bytes = field(repr=False)
    _provenance: bytes = field(repr=False)

    def private_copy(self) -> dict[str, Any]:
        return {
            "capture_value": json.loads(self._capture),
            "provenance_value_without_capture_pin": json.loads(self._provenance),
        }

    def safe_summary(self) -> dict[str, Any]:
        value = json.loads(self._capture)
        return {
            "phase": "before",
            "status": "conditional-before-measurement",
            "capture_sha256": append.sha(self._capture),
            "read_end": value["read_end"],
            "qualifying_records": len(value["common"]["progress"]["metrics"]),
            "execution_qualified": False,
            "historical_birth_qualified": False,
            "preservation_authority_granted": False,
        }


class BeforeMeasurementSession:
    def __init__(
        self,
        registration,
        io,
        *,
        expected_window,
        external_premises,
        verified_champions,
        bindings,
    ):
        self._observer = kernel.ObservedKernelObserver(
            registration,
            io,
            expected_window=expected_window,
            external_premises=external_premises,
        )
        self._registration, self._io = registration, io
        self._reg = registration.private_copy()
        self._window = deepcopy(self._observer.window)
        require(self._window["phase"] == "before", "before-only-window")
        require(io.maximum_bytes == append.MAX_RAW, "before-original-runtime-budget")
        self._premises = deepcopy(external_premises)
        self._champions = deepcopy(verified_champions)
        self._bindings = deepcopy(bindings)
        self._writer = {
            "binding": deepcopy(self._reg["learner_writer"]["binding"]),
            "evidence_sha256": self._reg["learner_writer"]["evidence_pin"]["sha256"],
            "evidence_bytes": self._reg["learner_writer"]["evidence_pin"]["bytes"],
        }
        self._expected = {
            "contract": "learner-append-observed-window-v1",
            "bindings": self._bindings,
            "expected_window": self._window,
            "writer_binding": self._writer["binding"],
            "expected_policy": self._reg["policy"],
            "verified_champions": self._champions,
            "expected_publication_writers": self._reg["publication_writers"],
        }
        contract.checked_expected(self._expected, registration)
        require(
            self._bindings["external_premises_sha256"] == digest(self._premises)
            and self._bindings["source_qualification_sha256"]
            == self._premises["source_context"]["qualification_sha256"],
            "before-external-premise-binding",
        )
        preservation._check_recipe_values(self._reg["policy"]["recipe"])
        self._tracker = renewal.PublicationRenewalTracker(
            registration,
            io,
            expected_window=self._window,
            external_premises=self._premises,
            verified_champions=self._champions,
        )
        self._support = support.SupportCollector(self._observer.identity)
        self._inventory: list[dict[str, Any]] = []
        self._kernel: list[tuple[kernel.KernelSnapshot, dict[str, Any]]] = []
        self._source_checks: list[dict[str, Any]] = []
        self._support_observations: list[dict[str, Any]] = []
        self._witnesses: dict[str, Any] = {}
        self._tracker_audit_count = 0
        self._last_clock = None
        self._failure: str | None = None
        self._begun = False
        self._finished = False
        self._b: dict[str, Any] | None = None
        self._e: dict[str, Any] | None = None

    def _live(self):
        require(self._failure is None, "before-chain-refused")
        require(not self._finished, "before-chain-finished")
        require(self._observer.io is self._io, "before-original-io")
        self._observer._check_context()

    def _failed(self, error: Exception):
        known = (
            BeforeRefusal,
            kernel.KernelRefusal,
            renewal.RenewalRefusal,
            append.AppendRefusal,
            append.AppendIncomplete,
            readonly.ReadRefusal,
            identity.CollectionRefusal,
            records.RecordRefusal,
            support.SupportRefusal,
            preservation.PreservationViolation,
        )
        reason = str(error) if isinstance(error, known) else "before-component-refused"
        if re.fullmatch(r"[a-z][a-z0-9-]{0,127}", reason) is None:
            reason = "before-component-refused"
        self._failure = reason
        if isinstance(
            error,
            (BeforeIncomplete, append.AppendIncomplete, renewal.RenewalIncomplete),
        ):
            raise BeforeIncomplete(reason) from None
        raise BeforeRefusal(reason) from None

    def _audits(self, rows):
        for row in rows:
            first, last = row["read_start"], row["read_end"]
            append.order(first, last)
            for axis in ("monotonic_ns", "wall_ns"):
                require(
                    self._window["phase_start"][axis]
                    <= first[axis]
                    <= last[axis]
                    < self._window["deadline"][axis]
                    and (
                        self._last_clock is None
                        or self._last_clock[axis] <= first[axis]
                    ),
                    "before-observation-clock",
                )
            require(
                first["boot_id"] == self._window["phase_start"]["boot_id"],
                "before-observation-boot",
            )
            self._last_clock = deepcopy(last)
            self._inventory.append(deepcopy(row))
        require(
            len(append.encoded(self._inventory)) <= 4 * 2**20, "before-inventory-bound"
        )

    def _identity_call(self, fn):
        start = len(self._observer.identity.audit)
        result = fn()
        self._audits(self._observer.identity.audit[start:])
        return result

    def _tracker_call(self, fn):
        result = fn()
        audits = self._tracker.private_audits()
        self._audits(audits[self._tracker_audit_count :])
        self._tracker_audit_count = len(audits)
        return result

    def _append(self, observation):
        self._audits([observation.audit])
        return {
            "value": deepcopy(observation.value),
            "audit": deepcopy(dict(observation.audit)),
        }

    def _source_facts(self, checks, cached, owners):
        sources = {}
        for key, row in checks["source_pins"].items():
            sources[key] = {
                "literal_path_sha256": _path_hash(row["literal_path"]),
                "resolved_path_sha256": _path_hash(row["resolved_path"]),
                **{k: row[k] for k in ("sha256", "bytes", "mode", "uid", "gid")},
            }
        cache = {}
        for key, row in cached.items():
            reference = self._reg["cached_references"][key]
            cache[key] = {
                "literal_path_sha256": _path_hash(row["literal"]),
                "resolved_path_sha256": _path_hash(row["resolved"]),
                "sha256": checks["cached_references"][key]["sha256"],
                "qualification_sha256": reference["qualification_sha256"],
                "literal_stat": {
                    k: row["literal_stat"][k] for k in identity.FINGERPRINT
                },
                "resolved_stat": {
                    k: row["resolved_stat"][k] for k in identity.FINGERPRINT
                },
                "content_hashed": False,
            }
        return {
            "source_pins": sources,
            "cached_references": cache,
            "unit_static": deepcopy(checks["unit_static"]),
            "default_target": checks["default_target"],
            "origins": {role: row["origin_sha256"] for role, row in owners.items()},
        }

    def _unit_facts(self, units, statics):
        result = {}
        for name, row in units.items():
            kind = self._reg["units"][name]["kind"]
            fields = set(support.DYNAMIC) | {
                "LoadState",
                "NeedDaemonReload",
                "UnitFileState",
                "Result",
            }
            if kind != "timer":
                fields.update(support.SERVICE_DYNAMIC)
            result[name] = {
                "kind": kind,
                "properties": {k: row[k] for k in sorted(fields)},
                "static": deepcopy(statics[name]),
            }
        return result

    def _capture_kernel(self):
        snapshot = self._identity_call(self._observer.capture_current)
        private = snapshot.private_copy()
        cached = deepcopy(self._observer.identity.cached_measurements)
        if self._kernel:
            kernel.same_kernel_observations(self._kernel[0][0], snapshot)
        projection = {
            "format": contract.KERNEL,
            "measurement_sha256": snapshot.safe_summary()["measurement_sha256"],
            **{
                k: deepcopy(private[k])
                for k in (
                    "registration_sha256",
                    "external_premises_sha256",
                    "window_sha256",
                    "read_start",
                    "read_end",
                    "owners",
                    "monitor",
                    "gpu_owners",
                    "auxiliaries",
                    "audit_inventory",
                )
            },
            "unit_facts": self._unit_facts(
                private["units"], private["source_checks"]["unit_static"]
            ),
            "source_facts": self._source_facts(
                private["source_checks"], cached, private["owners"]
            ),
        }
        self._kernel.append((snapshot, projection))
        return snapshot

    def _static(self, snapshot):
        private = snapshot.private_copy()
        index = len(self._observer.identity.audit)
        static, units, champion, checks = self._identity_call(
            lambda: common._read_static_authority(
                self._observer.identity,
                coordinator_pid=private["owners"]["coordinator"]["pid"],
                verified_champions=self._champions,
            )
        )
        cached = deepcopy(self._observer.identity.cached_measurements)
        # These digests come from the preceding actual issued snapshot. Both it
        # and this helper independently require exact equality to the same policy.
        # Retain both audit brackets; do not relabel cached or raw stat facts.
        statics = private["source_checks"]["unit_static"]
        audits = self._observer.identity.audit[index:]
        self._source_checks.append(
            {
                "static": deepcopy(static),
                "static_sha256": checks["static_sha256"],
                "read_start": deepcopy(audits[0]["read_start"]),
                "read_end": deepcopy(audits[-1]["read_end"]),
                "source_facts": self._source_facts(
                    {**checks, "unit_static": statics}, cached, private["owners"]
                ),
                "unit_facts": self._unit_facts(units, statics),
                "audit_inventory": deepcopy(audits),
            }
        )
        return static, units, champion

    def _capture_support(self, snapshot):
        got = self._identity_call(
            lambda: self._support.capture(
                monitors=snapshot.private_copy()["monitor"],
                previous_witnesses=self._witnesses,
            )
        )
        self._witnesses = deepcopy(got.witnesses)
        self._support_observations.append(deepcopy(got.provenance))
        preservation._check_support(self._reg["policy"], got.support)
        return got

    def begin(self):
        try:
            self._live()
            require(not self._begun, "before-begin-once")
            first = self._capture_kernel()
            self._initial_static, _, _ = self._static(first)
            self._capture_support(first)
            self._tracker_call(lambda: self._tracker.fence(first))
            self._b = self._append(self._io.append_fence("metrics"))
            require(
                self._b["value"]["file_identity"]
                == self._writer["binding"]["file_identity"],
                "before-writer-file",
            )
            self._begun = True
            return {
                "phase": "before",
                "status": "pending",
                "renewed": 0,
                "renewal_complete": False,
                "execution_authorized": False,
                "read_start": deepcopy(self._inventory[0]["read_start"]),
                "read_end": deepcopy(self._last_clock),
            }
        except Exception as error:
            self._failed(error)

    def observe(self):
        try:
            self._live()
            require(self._begun, "before-begin-required")
            require(
                self._tracker._round < renewal.MAX_ROUNDS - 1,
                "before-observe-round-bound",
            )
            audit_start = len(self._inventory)
            current = self._capture_kernel()
            poll = self._tracker_call(lambda: self._tracker.observe(current))
            return {
                "phase": "before",
                "status": "pending",
                "renewed": poll.renewed,
                "renewal_complete": poll.renewed == 34,
                "execution_authorized": False,
                "read_start": deepcopy(self._inventory[audit_start]["read_start"]),
                "read_end": deepcopy(self._last_clock),
            }
        except Exception as error:
            self._failed(error)

    def _ranges(self, end):
        assert self._b is not None
        begin = self._b["value"]["prefix"]["start"]
        maximum = self._io.scope.tails["metrics"].maximum_bytes
        endpoints = [(i, min(i + maximum, end)) for i in range(begin, end, maximum)]
        if not endpoints:
            endpoints = [(begin, end)]
        require(len(endpoints) <= append.MAX_SPANS, "before-range-count")
        return [
            self._append(
                self._io.read_append_range(
                    "metrics",
                    file_identity=self._b["value"]["file_identity"],
                    start=a,
                    end=b,
                )
            )
            for a, b in endpoints
        ]

    def _owner(self, snapshot):
        private = snapshot.private_copy()
        row = private["owners"]["learner"]
        owner = {
            **{
                k: row[k]
                for k in (
                    "pid",
                    "start_ticks",
                    "ppid",
                    "cgroup",
                    "invocation_id",
                    "origin_sha256",
                    "clock_ticks_per_second",
                )
            },
            "boot_id": private["read_end"]["boot_id"],
            "uid": row["uids"]["real"],
            "pid_namespace_inode": row["namespaces"]["pid"],
            "time_namespace_inode": row["namespaces"]["time"],
        }
        require(
            append.encoded(owner)
            == append.encoded(self._writer["binding"]["owner_key"]),
            "before-writer-owner",
        )
        return {
            "owner_key": owner,
            "writer_evidence_sha256": self._writer["evidence_sha256"],
            "read_start": private["read_start"],
            "read_end": private["read_end"],
        }

    def finish(self) -> BeforeMeasurement:
        try:
            self._live()
            require(self._begun and self._e is None, "before-finish-once")
            assert self._b is not None
            self._e = self._append(self._io.append_fence("metrics"))
            endpoint = self._e["value"]["size_at_fstat"]
            originals, rereads = self._ranges(endpoint), self._ranges(endpoint)
            reporter_kernel = self._capture_kernel()
            self._tracker_call(lambda: self._tracker.observe(reporter_kernel))
            static, _, champion = self._static(reporter_kernel)
            require(static == self._initial_static, "before-static-changed")
            got = self._capture_support(reporter_kernel)
            guard = self._append(
                self._io.read_append_range(
                    "metrics",
                    file_identity=self._b["value"]["file_identity"],
                    start=endpoint,
                    end=endpoint,
                )
            )
            if any(
                guard["audit"][k]["bytes"] != endpoint
                for k in ("named_before", "stat_before", "stat_after", "named_after")
            ):
                raise BeforeIncomplete("before-growth-after-fixed-end")
            final_kernel = self._capture_kernel()
            final_private = final_kernel.private_copy()
            common._check_final_support_states(
                self._reg["policy"]["support"],
                self._reg["units"],
                provenance=got.provenance,
                final_units=final_private["units"],
            )
            renewed = self._tracker_call(lambda: self._tracker.finish(final_kernel))
            final_clock = renewed.safe_proof()["read_end"]
            private_records = self._tracker.private_records()
            first, after_b = self._kernel[0], self._kernel[1]
            owner_rows = [
                self._owner(first[0]),
                self._owner(after_b[0]),
                self._owner(final_kernel),
            ]
            # Avoid a duplicate final owner when no intermediate observe was used.
            if after_b[0] is final_kernel:
                owner_rows = [owner_rows[0], owner_rows[-1]]
            window = {
                "format": append.WINDOW,
                "schema_version": 1,
                **self._window,
                "final_clock": final_clock,
                "before_fence": self._b,
                "causal_fence": self._b,
                "end_fence": self._e,
                "segments": originals,
                "rereads": rereads,
                "owner_observations": owner_rows,
                "recipe": self._reg["policy"]["recipe"],
            }
            proof = append.verify_append(
                window,
                expected_writer=self._writer,
                writer_evidence=self._registration.private_writer_evidence(),
                expected_window=self._window,
            ).public()
            support_rows, projection = common._project_support_at_clock(
                deepcopy(got.support), got.provenance["clock"], final_clock
            )
            preservation._check_support(self._reg["policy"], support_rows)
            common_value = self._common(
                final_private,
                private_records,
                originals,
                proof,
                static=static,
                champion=champion,
                support_rows=support_rows,
                final_clock=final_clock,
            )
            owners = [
                contract.project_owner(self._owner(s), p)
                for s, p in (first, after_b, self._kernel[-1])
            ]
            safe_b = {
                "format": "strength-observed-before-fence-commitment-v1",
                "schema_version": 1,
                "contract": self._expected["contract"],
                "phase": "before",
                "bindings": deepcopy(self._bindings),
                **contract.project_append(self._b, self._writer["binding"]),
                "owner_before": owners[0],
                "owner_after": owners[1],
            }
            capture = {
                "format": "strength-preservation-observed-window-capture-v1",
                "schema_version": 1,
                "contract": self._expected["contract"],
                "phase": "before",
                "bindings": deepcopy(self._bindings),
                "read_start": deepcopy(self._inventory[0]["read_start"]),
                "read_end": final_clock,
                "common": common_value,
                "metric_window": {
                    "before_fence": safe_b,
                    "causal_fence_sha256": digest(safe_b),
                    "end_fence": contract.project_append(
                        self._e, self._writer["binding"]
                    ),
                    "append_proof": proof,
                },
            }
            provenance = {
                "format": "strength-preservation-observed-window-provenance-bundle-v1",
                "schema_version": 1,
                "contract": self._expected["contract"],
                "phase": "before",
                "bindings": deepcopy(self._bindings),
                "read_start": capture["read_start"],
                "read_end": final_clock,
                "audit_inventory": deepcopy(self._inventory),
                "kernel_projections": [p for _, p in self._kernel],
                "source_checks": self._source_checks,
                "renewal": renewed.safe_proof(),
                "owner_observations": owners,
                "metric_spans": {
                    "originals": [
                        contract.project_append(x, self._writer["binding"])
                        for x in originals
                    ],
                    "rereads": [
                        contract.project_append(x, self._writer["binding"])
                        for x in rereads
                    ],
                    "end_guard": contract.project_append(
                        guard, self._writer["binding"]
                    ),
                },
                "support_witnesses": self._witnesses,
                "support_provenance": self._support_observations,
                "support_age_projection": projection,
                "privacy_scope": contract.PRIVACY,
            }
            capture_raw, provenance_raw = (
                append.encoded(capture),
                append.encoded(provenance),
            )
            require(
                len(capture_raw) <= 2**20 and len(provenance_raw) <= 4 * 2**20,
                "before-safe-output-bound",
            )
            self._finished = True
            return BeforeMeasurement(capture_raw, provenance_raw)
        except Exception as error:
            self._failed(error)
            raise AssertionError("unreachable")

    def _common(
        self,
        final,
        private,
        spans,
        proof,
        *,
        static,
        champion,
        support_rows,
        final_clock,
    ):
        rows = {name: x["row"] for name, x in private.items()}
        runtime = self._reg["policy"]["static"]["runtime_name"]
        processes = records.process_records(
            rows["coordinator"],
            {r: rows[r] for r in records.WORKERS},
            kernel_processes={
                r: v for r, v in final["owners"].items() if r != "monitor"
            },
            unit_restarts=int(final["units"][runtime]["NRestarts"]),
        )
        for role, row in processes.items():
            require(
                {k: row[k] for k in preservation.PROCESS_FIELDS}
                == self._reg["policy"]["expected_processes"][role],
                "before-process-policy",
            )
            if role != "controller":
                preservation._fresh(
                    row["heartbeat_ns"],
                    final_clock["wall_ns"],
                    self._reg["policy"]["maximum_age_ns"],
                )
        raw = b"".join(x["value"]["raw"] for x in spans)
        start = spans[0]["value"]["start"]
        selected = proof["qualifying_records"]
        require(2 <= len(selected) <= 100, "before-metric-count")
        metrics = []
        learner = rows["learner"]
        require(
            learner["phase"] in {"training", "update_to_data_wait"},
            "before-learner-phase",
        )
        for ref in selected:
            line = raw[ref["start"] - start : ref["end"] - start]
            metric = records.parse_json(line)
            require(
                metric["timestamp_ns"] <= learner["heartbeat_ns"]
                and metric["step"] <= learner["step"],
                "before-metric-heartbeat-join",
            )
            preservation._fresh(
                metric["timestamp_ns"],
                final_clock["wall_ns"],
                self._reg["policy"]["maximum_age_ns"],
            )
            require(
                preservation._check_metric_payload(
                    metric, self._reg["policy"]["recipe"]
                )
                == ref["outcome_labels"],
                "before-metric-label-join",
            )
            metrics.append(
                contract.project_metric(
                    line,
                    record_ref={k: ref[k] for k in ("start", "end", "sha256")},
                    recipe=self._reg["policy"]["recipe"],
                )
            )
        cohorts = {}
        for name in records.COHORTS:
            row = rows[name]
            preservation._fresh(
                row["heartbeat_ns"],
                final_clock["wall_ns"],
                self._reg["policy"]["maximum_age_ns"],
            )
            cohorts[name] = {
                "model_identity": row["model_version"],
                "games": row["cumulative_games"],
                "heartbeat_ns": row["heartbeat_ns"],
                "progress_ns": row["progress_ns"],
            }
        return {
            "static": static,
            "owners": final["owners"],
            "processes": processes,
            "gpu_owners": final["gpu_owners"],
            "support": support_rows,
            "champion_identity": champion,
            "cohorts": cohorts,
            "progress": {
                "step": learner["step"],
                "phase": learner["phase"],
                "heartbeat_ns": learner["heartbeat_ns"],
                "metrics": metrics,
            },
        }
