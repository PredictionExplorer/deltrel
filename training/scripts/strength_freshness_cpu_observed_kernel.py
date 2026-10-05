"""Conditional current-kernel measurements, never preservation/admission success.

External source/access/role commitments remain caller-qualified premises. This
module reads no reporter JSON and creates no historical birth estimate. Issued
snapshots establish in-process provenance only, not hostile-Python/root security.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_collect_processes as processes
from scripts import strength_freshness_cpu_learner_window as append
from scripts import strength_freshness_cpu_observed_registration as registration_module
from scripts import strength_freshness_cpu_readonly as readonly

FORMAT = "strength-observed-kernel-measurement-v1"
PREMISES = "strength-observed-external-premises-v1"
MAX_ISSUED = 258  # baseline + at most256 shared rounds + final snapshot
MAX_PRIVATE_BYTES = 4 * 2**20
CONTEXT_KEYS = ("source_pins", "cached_references", "origins", "kernel_context")
OWNER_FIELDS = {
    "pid",
    "start_ticks",
    "ppid",
    "cgroup",
    "invocation_id",
    "origin_sha256",
    "uids",
    "namespaces",
    "clock_ticks_per_second",
}


class KernelRefusal(ValueError):
    """Fixed non-private code; failed observations never become empty success."""


def require(ok: object, code: str) -> None:
    if not ok:
        raise KernelRefusal(code)


def encoded(value: Any) -> bytes:
    return registration_module.encoded(value)


def digest(value: Any) -> str:
    return append.sha(encoded(value))


def _copy(value):
    return json.loads(encoded(value))


def _fields(value, keys, reason):
    require(isinstance(value, dict) and set(value) == set(keys), reason)


def validate_expected_window(value, *, boot_id: str) -> dict[str, Any]:
    """Validate exact external clocks; a bounded shape does not authorize them."""
    try:
        result = _copy(value)
        _fields(
            result,
            {"phase", "phase_start", "deadline", "cleanup_clock"},
            "kernel-window-fields",
        )
        require(result["phase"] in {"before", "after"}, "kernel-window-phase")
        start, end = result["phase_start"], result["deadline"]
        append.clock(start)
        append.clock(end)
        require(start["boot_id"] == end["boot_id"] == boot_id, "kernel-window-boot")
        require(
            all(
                0 < end[k] - start[k] <= 105 * 10**9
                for k in ("monotonic_ns", "wall_ns")
            ),
            "kernel-window-bound",
        )
        cleanup = result["cleanup_clock"]
        if result["phase"] == "before":
            require(cleanup is None, "kernel-before-cleanup")
        else:
            append.clock(cleanup)
            require(
                cleanup["boot_id"] == boot_id
                and all(cleanup[k] <= start[k] for k in ("monotonic_ns", "wall_ns")),
                "kernel-cleanup-order",
            )
        return result
    except KernelRefusal:
        raise
    except Exception:
        raise KernelRefusal("kernel-window-invalid") from None


def validate_external_premises(registration, value) -> dict[str, Any]:
    """Reparse approved raw bytes and match commitments, never issue qualification."""
    try:
        require(
            type(registration) is registration_module.ObservedRegistration,
            "kernel-registration-type",
        )
        result = _copy(value)
        _fields(
            result,
            {
                "format",
                "schema_version",
                "registration_sha256",
                "source_context",
                "learner_writer_evidence_sha256",
                "publication_sources",
                "publication_access",
            },
            "kernel-premise-fields",
        )
        require(
            result["format"] == PREMISES
            and type(result["schema_version"]) is int
            and result["schema_version"] == 1,
            "kernel-premise-format",
        )
        # Public dataclass construction is not provenance: inspect original bytes
        # under the independently supplied approval, and refuse canonical aliases.
        raw = registration._raw
        header = identity.strict_json(raw)
        checked = registration_module.parse_observed_registration(
            raw,
            approved_sha256=result["registration_sha256"],
            scope=identity.scope_from(header["scope"]),
            writer_evidence=registration.private_writer_evidence(),
        )
        reg = checked.private_copy()
        require(
            encoded(reg) == encoded(registration.private_copy()),
            "kernel-registration-canonical-drift",
        )
        context = result["source_context"]
        _fields(
            context,
            {*(k + "_sha256" for k in CONTEXT_KEYS), "qualification_sha256"},
            "kernel-source-context-fields",
        )
        require(
            all(context[k + "_sha256"] == digest(reg[k]) for k in CONTEXT_KEYS)
            and append.digest(context["qualification_sha256"]),
            "kernel-source-context-binding",
        )
        require(
            result["learner_writer_evidence_sha256"]
            == reg["learner_writer"]["evidence_pin"]["sha256"]
            == append.sha(registration.private_writer_evidence()),
            "kernel-writer-premise",
        )
        publications = reg["publication_writers"]
        require(
            encoded(result["publication_sources"])
            == encoded(
                {
                    name: row["source_contract_sha256"]
                    for name, row in publications.items()
                }
            ),
            "kernel-publication-source-premise",
        )
        require(
            encoded(result["publication_access"])
            == encoded(
                {
                    name: row["access_qualification_pin"]
                    for name, row in publications.items()
                }
            ),
            "kernel-publication-access-premise",
        )
        return result
    except KernelRefusal:
        raise
    except Exception:
        raise KernelRefusal("kernel-premise-invalid") from None


@dataclass(frozen=True, init=False)
class KernelSnapshot:
    _canonical: bytes = field(repr=False)
    _issuer: Any = field(repr=False)
    _seal: object = field(repr=False)

    def __init__(self):
        raise TypeError("kernel-snapshot-issued-only")

    def private_copy(self) -> dict[str, Any]:
        _issued(self)
        return json.loads(self._canonical)

    def safe_summary(self) -> dict[str, Any]:
        raw = self.private_copy()
        return {
            "format": "strength-observed-kernel-summary-v1",
            "measurement_sha256": append.sha(self._canonical),
            "registration_sha256": raw["registration_sha256"],
            "external_premises_sha256": raw["external_premises_sha256"],
            "window_sha256": raw["window_sha256"],
            "read_start": raw["read_start"],
            "read_end": raw["read_end"],
            "owners": len(raw["owners"]),
            "gpu_owners": len(raw["gpu_owners"]),
            "owner_sha256": digest(raw["owners"]),
            "source_checks_sha256": digest(raw["source_checks"]),
            "audit_inventory_sha256": digest(raw["audit_inventory"]),
            "worker_reporters_read": False,
            "historical_birth_qualified": False,
            "writer_qualified": False,
            "runtime_qualified": False,
            "preservation_passed": False,
            "execution_authorized": False,
        }


def _issued(snapshot):
    try:
        require(type(snapshot) is KernelSnapshot, "kernel-snapshot-type")
        issuer = snapshot._issuer
        require(
            type(issuer) is ObservedKernelObserver
            and snapshot._seal is issuer._seal
            and any(item is snapshot for item in issuer._issued),
            "kernel-snapshot-not-issued",
        )
        require(
            issuer._issued_hashes[id(snapshot)] == append.sha(snapshot._canonical),
            "kernel-snapshot-bytes-drift",
        )
        return issuer
    except KernelRefusal:
        raise
    except Exception:
        raise KernelRefusal("kernel-snapshot-not-issued") from None


def require_snapshot_context(
    snapshot, registration, io, expected_window, external_premises
) -> None:
    try:
        issuer = _issued(snapshot)
        issuer._check_context()
        require(
            io is issuer.io
            and type(registration) is registration_module.ObservedRegistration,
            "kernel-snapshot-context",
        )
        require(
            registration.sha256 == issuer._registration_sha
            and append.sha(registration._canonical)
            == issuer._registration_canonical_sha
            and append.sha(registration.private_writer_evidence())
            == issuer._writer_sha,
            "kernel-snapshot-registration",
        )
        require(
            encoded(expected_window) == issuer._window_bytes
            and encoded(external_premises) == issuer._premise_bytes,
            "kernel-snapshot-authority",
        )
    except KernelRefusal:
        raise
    except Exception:
        raise KernelRefusal("kernel-snapshot-context-invalid") from None


def same_kernel_observations(first: KernelSnapshot, second: KernelSnapshot) -> None:
    try:
        issuer = _issued(first)
        require(_issued(second) is issuer, "kernel-snapshot-issuer-changed")
        issuer._check_context()
        a, b = first.private_copy(), second.private_copy()
        require(
            all(
                a[k] == b[k]
                for k in (
                    "registration_sha256",
                    "external_premises_sha256",
                    "window_sha256",
                    "owners",
                    "monitor",
                    "gpu_owners",
                    "auxiliaries",
                    "source_checks",
                )
            ),
            "kernel-observation-drift",
        )
        require(
            all(
                a["read_end"][k] <= b["read_start"][k]
                for k in ("monotonic_ns", "wall_ns")
            ),
            "kernel-snapshot-order",
        )
    except KernelRefusal:
        raise
    except Exception:
        raise KernelRefusal("kernel-snapshot-comparison-invalid") from None


class _WindowIdentity(identity._IdentityMeasurements):
    def __init__(self, observer):
        self.observer = observer
        super().__init__(observer.reg, observer.io)

    def take(self, observation):
        # Preserve even a later-refused audit privately; never serialize raw values.
        value = super().take(observation)
        self.observer._check_observation(observation.audit)
        return value


class ObservedKernelObserver:
    """One retained IO/window and issued snapshots; no reporter or success route."""

    def __init__(self, registration, io, *, expected_window, external_premises):
        self.premises = validate_external_premises(registration, external_premises)
        self.reg = registration.private_copy()
        self._reg_bytes = encoded(self.reg)
        require(
            isinstance(io, readonly.ReadOnlyIO)
            and io.scope == identity.scope_from(self.reg["scope"]),
            "kernel-io-scope",
        )
        self.window = validate_expected_window(
            expected_window, boot_id=self.reg["kernel_context"]["boot_id"]
        )
        self.io, self._registration = io, registration
        self._original_io = io
        self._scope, self._backend = io.scope, io._backend
        self._deadline, self._maximum = io.deadline, io.maximum_bytes
        require(
            type(self._maximum) is int and 0 < self._maximum <= 24 * 2**20,
            "kernel-io-budget",
        )
        require(
            self.window["phase_start"]["monotonic_ns"] / 1e9
            < self._deadline
            <= self.window["deadline"]["monotonic_ns"] / 1e9,
            "kernel-io-deadline",
        )
        self._registration_sha = self.premises["registration_sha256"]
        self._registration_canonical_sha = append.sha(registration._canonical)
        self._writer_sha = append.sha(registration.private_writer_evidence())
        self._window_bytes, self._premise_bytes = (
            encoded(self.window),
            encoded(self.premises),
        )
        self._consumed, self._returned = io._consumed, io._returned
        self._seal = object()
        self._issued: list[KernelSnapshot] = []
        self._issued_hashes: dict[int, str] = {}
        self._issued_bytes = 0
        self._failed = False
        self._last_end: dict[str, Any] | None = None
        self.identity = _WindowIdentity(self)

    def _check_context(self):
        require(not self._failed, "kernel-observer-closed-after-refusal")
        require(
            self.io is self._original_io and self.identity.io is self._original_io,
            "kernel-io-instance-changed",
        )
        require(
            self.io.scope is self._scope
            and self.io._backend is self._backend
            and type(self.io.deadline) is type(self._deadline)
            and self.io.deadline == self._deadline
            and type(self.io.maximum_bytes) is int
            and self.io.maximum_bytes == self._maximum,
            "kernel-io-context-changed",
        )
        require(
            self._registration.sha256 == self._registration_sha
            and append.sha(self._registration._canonical)
            == self._registration_canonical_sha
            and append.sha(self._registration.private_writer_evidence())
            == self._writer_sha,
            "kernel-registration-changed",
        )
        require(
            encoded(self.window) == self._window_bytes
            and encoded(self.premises) == self._premise_bytes
            and encoded(self.reg) == self._reg_bytes
            and encoded(self.identity.reg) == self._reg_bytes,
            "kernel-authority-changed",
        )
        require(
            type(self.io._consumed) is int
            and self._consumed <= self.io._consumed <= self._maximum
            and type(self.io._returned) is int
            and self._returned <= self.io._returned <= self._maximum,
            "kernel-io-counter-reset",
        )
        self._consumed, self._returned = self.io._consumed, self.io._returned

    def _check_observation(self, audit):
        self._check_context()
        a, b = audit["read_start"], audit["read_end"]
        append.clock(a)
        append.clock(b)
        require(
            a["boot_id"] == b["boot_id"] == self.reg["kernel_context"]["boot_id"],
            "kernel-observation-boot",
        )
        require(
            all(
                self.window["phase_start"][k]
                <= a[k]
                <= b[k]
                < self.window["deadline"][k]
                for k in ("monotonic_ns", "wall_ns")
            ),
            "kernel-observation-window",
        )
        require(
            self._last_end is None
            or all(self._last_end[k] <= a[k] for k in ("monotonic_ns", "wall_ns")),
            "kernel-observation-regression",
        )
        self._last_end = dict(b)

    def _namespace(self, subject):
        value = self.identity.take(self.io.namespaces(subject))
        require(set(value) == {"pid", "time"}, "kernel-namespace-fields")
        result = {}
        for key in ("pid", "time"):
            inode = value[key]["stat"]["inode"]
            require(
                type(inode) is int
                and inode > 0
                and value[key]["literal"] == f"{key}:[{inode}]",
                "kernel-namespace-identity",
            )
            result[key] = inode
        return result

    def _source_units(self):
        i, reg, policy = self.identity, self.reg, self.reg["policy"]
        sources, cached = i.verify_source_pins(), i.verify_cached_references()
        expected_default = reg["boot"]["default_target"]
        require(
            i.take(self.io.query("default-target"))
            in {expected_default, expected_default + "\n"},
            "kernel-default-target",
        )
        targets = {
            name: identity.properties(i.take(self.io.query("registered-target", name)))
            for name in self.io.scope.targets
        }
        require(
            all(
                props.get("Id") == name and props.get("LoadState") == "loaded"
                for name, props in targets.items()
            ),
            "kernel-target-state",
        )
        units = {name: i.unit(name) for name in reg["units"]}
        statics = {
            name: i.unit_static(name, props, targets) for name, props in units.items()
        }
        runtime = policy["static"]["runtime_name"]
        for name, actual in statics.items():
            if name == runtime:
                require(
                    all(
                        actual[k] == policy["static"]["runtime_" + k]
                        for k in (
                            "definition_sha256",
                            "environment_sha256",
                            "boot_links_sha256",
                        )
                    ),
                    "kernel-runtime-static-drift",
                )
            else:
                require(
                    {"kind": reg["units"][name]["kind"], **actual}
                    == policy["support"][name],
                    "kernel-support-static-drift",
                )
        return units, {
            "source_pins": sources,
            "cached_references": cached,
            "unit_static": statics,
            "default_target": expected_default,
            "coverage": [
                "selected-source-bytes",
                "qualified-cache-stat-references",
                "unit-definition-environment-boot",
                "actual-process-origins",
            ],
            "absent": [
                "profile-run-continuation-semantics",
                "champion-teacher-semantic-proof",
                "support-job-health-and-age",
                "reporter-renewal",
                "recipe-credit-and-metric-work",
                "historical-birth",
            ],
        }

    def _kernel_facts(self, units):
        i, policy = self.identity, self.reg["policy"]
        runtime = policy["static"]["runtime_name"]
        props = units[runtime]
        require(
            props["ActiveState"] == "active"
            and props["SubState"] == "running"
            and props["Result"] == "success"
            and props["ControlGroup"] == policy["static"]["runtime_cgroup"],
            "kernel-runtime-state",
        )
        members = i.take(self.io.members(runtime))
        admissions = {a.pid: a for a in members}
        roles = {role: row["pid"] for role, row in policy["expected_processes"].items()}
        require(
            len(admissions) == len(members)
            and len(set(roles.values())) == len(roles)
            and set(roles.values()) <= set(admissions),
            "kernel-runtime-members",
        )
        require(
            processes.natural(props["MainPID"], minimum=1) == roles["controller"],
            "kernel-controller-pid",
        )
        private = {
            pid: i.take(self.io.process(a, maps=pid in roles.values()))
            for pid, a in admissions.items()
        }
        cores = {}
        for role, pid in roles.items():
            raw = private[pid]
            expected = policy["expected_processes"][role]
            require(
                raw["pid"] == pid and raw["start_ticks"] == expected["start_ticks"],
                "kernel-role-generation",
            )
            if role != "controller":
                require(
                    raw["ppid"]
                    == roles["controller" if role == "coordinator" else "coordinator"],
                    "kernel-role-parent",
                )
            core = processes._current_process_fields(i, role, pid, raw, props)
            require(
                all(core[k] == expected[k] for k in core), "kernel-predeclared-role"
            )
            if role in {"controller", "coordinator"}:
                require(
                    processes.natural(props["NRestarts"]) == expected["restarts"],
                    "kernel-manager-restarts",
                )
            cores[role] = core
        auxiliary = processes._current_auxiliaries(self.reg, private, roles)
        monitor = {}
        monitor_admissions = {}
        for name, expected in policy["expected_monitors"].items():
            row, admitted, raw = processes._current_monitor(
                i, name, units[name], expected
            )
            require(units[name]["Result"] == "success", "kernel-monitor-result")
            monitor[name] = row
            cores["monitor"] = {k: v for k, v in row.items() if k != "restarts"}
            private[admitted.pid] = raw
            admissions[admitted.pid] = admitted
            monitor_admissions[name] = admitted
        require(set(cores) == set(registration_module.ROLES), "kernel-owner-roster")
        context = self.reg["kernel_context"]
        # Current HZ is observed; BOOTTIME brackets are never converted to birth.
        bracket = i.take(self.io.birth_bracket())
        require(
            bracket["clock_ticks_per_second"] == context["clock_ticks_per_second"],
            "kernel-clock-ticks",
        )
        require(
            self._namespace("self") == context["namespace_expectations"]["self"]
            and self._namespace(1) == context["namespace_expectations"]["pid1"],
            "kernel-reader-namespace",
        )
        owners = {}
        for role, core in cores.items():
            admitted = admissions[core["pid"]]
            raw = private[core["pid"]]
            credential = i.take(self.io.process_credentials(admitted))
            require(
                all(
                    credential[k] == raw[k]
                    for k in ("pid", "start_ticks", "ppid", "cgroup")
                )
                and encoded(credential["uids"])
                == encoded(context["credentials"][role]),
                "kernel-credentials",
            )
            ns = self._namespace(admitted)
            require(
                ns == context["namespace_expectations"]["owners"][role],
                "kernel-owner-namespace",
            )
            owners[role] = {
                **core,
                "ppid": raw["ppid"],
                "uids": credential["uids"],
                "namespaces": ns,
                "clock_ticks_per_second": bracket["clock_ticks_per_second"],
            }
            require(set(owners[role]) == OWNER_FIELDS, "kernel-owner-fields")
        gpu = processes.gpu_records(
            i.take(self.io.query("gpu-inventory")),
            i.take(self.io.query("gpu-owners")),
            policy=policy,
            processes=cores,
        )
        require(
            set(i.take(self.io.members(runtime))) == set(members),
            "kernel-final-runtime-members",
        )
        for name, admitted in monitor_admissions.items():
            require(
                tuple(i.take(self.io.members(name))) == (admitted,),
                "kernel-final-monitor-members",
            )
        return {
            "owners": owners,
            "monitor": monitor,
            "gpu_owners": gpu,
            "auxiliaries": auxiliary,
        }

    def capture_current(self) -> KernelSnapshot:
        try:
            self._check_context()
            require(len(self._issued) < MAX_ISSUED, "kernel-snapshot-count")
            i, policy = self.identity, self.reg["policy"]
            audit_start = len(i.audit)
            read_start = i.clock()
            units, source_checks = self._source_units()
            first = self._kernel_facts(units)
            runtime = policy["static"]["runtime_name"]
            # Repeat actual source/static and controlling manager facts after owners.
            later_units, later_sources = self._source_units()
            require(source_checks == later_sources, "kernel-final-source-drift")
            for name in {runtime, *first["monitor"]}:
                require(
                    all(
                        units[name][k] == later_units[name][k]
                        for k in (
                            "MainPID",
                            "ControlGroup",
                            "InvocationID",
                            "NRestarts",
                            "ActiveState",
                            "SubState",
                            "Result",
                        )
                    ),
                    "kernel-final-manager-drift",
                )
            # The closing owner/GPU/membership bracket follows the source scan;
            # a worker lost during that scan must not leave stale issued facts.
            second = self._kernel_facts(later_units)
            require(first == second, "kernel-within-capture-owner-drift")
            owners, monitor = second["owners"], second["monitor"]
            gpu, auxiliary = second["gpu_owners"], second["auxiliaries"]
            read_end = i.clock()
            payload = {
                "format": FORMAT,
                "registration_sha256": self._registration_sha,
                "external_premises_sha256": append.sha(self._premise_bytes),
                "window_sha256": append.sha(self._window_bytes),
                "read_start": read_start,
                "read_end": read_end,
                "units": later_units,
                "owners": owners,
                "monitor": monitor,
                "gpu_owners": gpu,
                "auxiliaries": auxiliary,
                "source_checks": source_checks,
                "audit_inventory": i.audit[audit_start:],
                "provisional_reporter_fields": {
                    "coordinator": "not-read",
                    "worker_health": "not-read",
                    "worker_restart_counts": "not-read",
                    "publication_renewal": "deferred",
                },
            }
            raw_payload = encoded(payload)
            require(
                self._issued_bytes + len(raw_payload) <= MAX_PRIVATE_BYTES,
                "kernel-snapshot-storage-budget",
            )
            snapshot = object.__new__(KernelSnapshot)
            object.__setattr__(snapshot, "_canonical", raw_payload)
            object.__setattr__(snapshot, "_issuer", self)
            object.__setattr__(snapshot, "_seal", self._seal)
            self._issued.append(snapshot)
            self._issued_hashes[id(snapshot)] = append.sha(raw_payload)
            self._issued_bytes += len(raw_payload)
            self._check_context()
            return snapshot
        except Exception as error:
            self._failed = True
            if isinstance(error, KernelRefusal):
                raise
            if isinstance(
                error,
                (
                    identity.CollectionRefusal,
                    processes.ProcessRefusal,
                    readonly.ReadRefusal,
                ),
            ):
                raise KernelRefusal(str(error)) from None
            raise KernelRefusal("kernel-observation-unavailable") from None
