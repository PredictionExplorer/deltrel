"""Real measurement helpers over a closed synthetic backend, no target facts."""

from copy import deepcopy
import json
from typing import Any, cast

import pytest
from scripts import strength_freshness_cpu_observed_kernel as k
from scripts import strength_freshness_cpu_observed_registration as registration
from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_readonly as ro
from tests.test_strength_freshness_cpu_observed_registration import (
    registration_fixture,
    SECRET,
)
from tests.test_strength_freshness_cpu_collect_identity import (
    identity_fixture,
    fingerprint,
)
from tests.test_strength_freshness_cpu_readonly import proc_stat

BOOT = "11111111-1111-1111-1111-111111111111"
BASE = 1_800_000_000_000_000_000


class KernelBackend:
    """Explicit stat/manager/kernel facts; unknown operations never reach host."""

    def __init__(self, reg, old_io, private):
        self.reg = reg
        self.ns = 100 * 10**9
        self.wall_jump = 0
        self.calls = []
        self.data = deepcopy(old_io.data)
        self.props = deepcopy(old_io.props)
        self.raw = {}
        self.credentials = {}
        self.namespace_overrides = {}
        self.hz = 100
        self.raw_template = deepcopy(private)
        self.fault = None
        self.process_reads = 0

    def advance(self, seconds):
        self.ns += int(seconds * 10**9)

    def monotonic_ns(self):
        return self.ns

    def wall_ns(self):
        return BASE + self.ns + self.wall_jump

    def boottime_ns(self):
        return self.ns

    def hertz(self):
        return self.hz

    def touch(self, kind, subject, deadline):
        self.calls.append((kind, subject))
        self.ns += 10_000
        ro.require(self.ns / 1e9 < deadline, "absolute-read-deadline")

    @staticmethod
    def debit(raw, maximum, allowance, charge):
        ro.require(len(raw) <= maximum and len(raw) <= allowance, "fixture-byte-budget")
        charge(len(raw))
        return raw

    def read_file(self, path, maximum, deadline, *, tail=False, charge, allowance):
        self.touch("read", path, deadline)
        if path == "/proc/sys/kernel/random/boot_id":
            raw = (BOOT + "\n").encode()
        elif path.startswith("/sys/fs/cgroup/") and path.endswith("/cgroup.procs"):
            group = path[len("/sys/fs/cgroup") : -len("/cgroup.procs")]
            raw = (
                "\n".join(
                    str(pid) for pid, row in self.raw.items() if row["cgroup"] == group
                )
                + "\n"
            ).encode()
        else:
            key = next(
                key
                for key, row in self.reg["scope"]["files"].items()
                if row["path"] == path
            )
            raw = self.data[key]
        raw = self.debit(raw, maximum, allowance, charge)
        st = {**fingerprint(len(raw)), "links": 1}
        return raw, {
            "stat_before": st,
            "stat_after": st,
            "offset": 0,
            "end_offset": len(raw),
        }

    def publication_file(self, path, maximum, deadline, *, charge, allowance):
        raw, meta = self.read_file(
            path, maximum, deadline, charge=charge, allowance=allowance
        )
        return raw, {
            **meta,
            "named_before": meta["stat_before"],
            "named_after": meta["stat_after"],
        }

    def directory(self, path, deadline):
        self.touch("directory", path, deadline)
        assert path in {
            "/etc/systemd/system",
            "/run/systemd/system",
        } or path.startswith("/sys/fs/cgroup/")
        return []

    def stat_path(self, path, deadline, *, follow=False):
        self.touch("stat", path, deadline)
        for name, row in self.reg["scope"]["cached"].items():
            for part in ("literal", "resolved"):
                if row[part] == path:
                    return deepcopy(self.reg["cached_references"][name][part + "_stat"])
        raise AssertionError("unknown cached path")

    def link(self, path, deadline):
        self.touch("link", path, deadline)
        row = next(
            row
            for row in self.reg["scope"]["cached"].values()
            if row["literal"] == path
        )
        return {"literal": row["resolved"], "resolved": row["resolved"]}

    def proc(
        self,
        pid,
        deadline,
        *,
        details=False,
        expected=None,
        maps=True,
        charge,
        allowance,
    ):
        self.touch("proc", pid, deadline)
        row = deepcopy(self.raw[pid])
        if details:
            self.process_reads += 1
            if self.fault == "later-origin" and self.process_reads > 12:
                row["cmdline"] = b"unapproved\0"
        value = {
            "stat_before": proc_stat(pid, row["start_ticks"], row["ppid"]),
            "stat_after": proc_stat(pid, row["start_ticks"], row["ppid"]),
            "cgroup_before": ("0::" + row["cgroup"] + "\n").encode(),
            "cgroup_after": ("0::" + row["cgroup"] + "\n").encode(),
        }
        if details:
            assert expected is not None
            ro.require(
                (row["start_ticks"], row["cgroup"])
                == (expected.start_ticks, expected.cgroup),
                "fixture-admission-drift",
            )
            value.update(
                {name: row[name] for name in ("exe", "cwd", "cmdline", "environ")}
            )
            if maps:
                value["maps"] = row["maps"]
        for raw in value.values():
            body = raw if isinstance(raw, bytes) else raw.encode()
            self.debit(body, ro.MAX_MAP_BYTES, allowance, charge)
            allowance -= len(body)
        return value

    def proc_credentials(self, admission, deadline, *, charge, allowance):
        value = self.proc(
            admission.pid,
            deadline,
            expected=admission,
            charge=charge,
            allowance=allowance,
        )
        remaining = allowance - sum(len(x) for x in value.values())
        uids = self.credentials[admission.pid]
        value["status"] = self.debit(
            (
                "Name:\tprivate-kernel-name\nUid:\t"
                + "\t".join(
                    str(uids[x]) for x in ("real", "effective", "saved", "filesystem")
                )
                + "\n"
            ).encode(),
            ro.MAX_PROC_BYTES,
            remaining,
            charge,
        )
        return value

    def namespace(self, pid, deadline):
        self.touch("namespace", pid, deadline)
        ns = self.namespace_overrides.get(pid, {"pid": 42, "time": 43})
        return {
            key: {"literal": f"{key}:[{value}]", "stat": {"device": 0, "inode": value}}
            for key, value in ns.items()
        }

    def command(self, argv, deadline, *, charge, allowance):
        self.touch("command", argv, deadline)
        assert ro._readonly_vector(argv)
        if argv[:2] == ("systemctl", "get-default"):
            raw = (self.reg["boot"]["default_target"] + "\n").encode()
        elif argv[:2] == ("systemctl", "show"):
            name = argv[2]
            props = (
                self.props[name]
                if name in self.props
                else {"Id": name, "LoadState": "loaded", "Wants": "", "Requires": ""}
            )
            raw = (
                "\n".join(key + "=" + val for key, val in props.items()) + "\n"
            ).encode()
        elif argv[1].startswith("--query-gpu="):
            raw = "\n".join(
                f"{i}, {gpu}" for i, gpu in enumerate(self.reg["policy"]["gpu_uuids"])
            ).encode()
        elif argv[1].startswith("--query-compute-apps="):
            p = self.reg["policy"]
            raw = "\n".join(
                f"{p['expected_processes'][role]['pid']}, {gpu}"
                for gpu, role in p["gpu_roles"].items()
            ).encode()
            if self.fault == "foreign-gpu":
                raw = b"999999, " + raw.split(b", ", 1)[1]
        else:
            raise AssertionError("unexpected kernel command")
        self.debit(raw, ro.MAX_FILE_BYTES, allowance, charge)
        return {"stdout": raw, "stderr": b"", "returncode": 0}


def observed_kernel_fixture(monkeypatch, transform_registration=None):
    """Synthetic qualified-premise fixture, real IO/identity/observer methods.

    Source constants are explicitly synthetic in this unit test, not claimed R3
    source qualification. The optional hook is test-only and runs before parsing.
    """
    reg, _, _, old_io, _ = registration_fixture()
    _, _, private = identity_fixture()
    backend = KernelBackend(reg, old_io, private)
    reg["kernel_context"]["boot_id"] = BOOT
    for name, owner, attr in [
        ("learner", registration.append, "LEARNER_SOURCE"),
        ("training", registration.append, "TRAINING_SOURCE"),
        ("runtime", registration, "RUNTIME_SOURCE"),
        ("actor", registration, "ACTOR_SOURCE"),
        ("coordinator", registration, "COORDINATOR_SOURCE"),
    ]:
        raw = ("synthetic immutable " + name + " source\n").encode()
        checksum = identity.sha(raw)
        monkeypatch.setattr(owner, attr, checksum)
        backend.data["source-" + name] = raw
        reg["source_pins"]["source-" + name] = {"sha256": checksum, "bytes": len(raw)}
    if transform_registration is not None:
        transform_registration(reg, backend)
    expected = {
        **reg["policy"]["expected_processes"],
        "monitor": next(iter(reg["policy"]["expected_monitors"].values())),
    }
    for role, row in expected.items():
        raw = deepcopy(private)
        raw.update(
            pid=row["pid"],
            start_ticks=row["start_ticks"],
            cgroup=row["cgroup"],
            ppid=1
            if role in {"controller", "monitor"}
            else expected["controller" if role == "coordinator" else "coordinator"][
                "pid"
            ],
        )
        if role == "learner":
            raw["environ"] += b"PRIVATE_TEST=" + SECRET.encode() + b"\0"
        backend.raw[row["pid"]] = raw
        backend.credentials[row["pid"]] = deepcopy(
            reg["kernel_context"]["credentials"][role]
        )
    for name, props in backend.props.items():
        role = "controller" if name == identity.RUNTIME else "monitor"
        row = expected[role]
        props.update(
            ActiveState="active",
            SubState="waiting" if name.endswith(".timer") else "running",
            Result="success",
            MainPID=str(row["pid"]),
            ExecMainPID=str(row["pid"]),
            ControlGroup="/system.slice/" + name,
            InvocationID=row["invocation_id"],
            NRestarts=str(row["restarts"]),
        )
    scope = identity.scope_from(reg["scope"])
    builder_io = ro.ReadOnlyIO(
        scope, backend=backend, deadline=1000, maximum_bytes=24 * 2**20
    )
    measured = identity._IdentityMeasurements(reg, builder_io)
    for role, row in expected.items():
        row["origin_sha256"] = measured.origin(role, backend.raw[row["pid"]])
    for name, props in backend.props.items():
        static = measured.unit_static(name, props, {})
        if name == identity.RUNTIME:
            for key in ("definition_sha256", "environment_sha256", "boot_links_sha256"):
                reg["policy"]["static"]["runtime_" + key] = static[key]
        else:
            reg["policy"]["support"][name] = {
                "kind": reg["units"][name]["kind"],
                **static,
            }
    binding = reg["learner_writer"]["binding"]
    binding.update(
        learner_source_sha256=registration.append.LEARNER_SOURCE,
        training_source_sha256=registration.append.TRAINING_SOURCE,
    )
    binding["owner_key"].update(
        boot_id=BOOT,
        origin_sha256=expected["learner"]["origin_sha256"],
        uid=reg["kernel_context"]["credentials"]["learner"]["real"],
    )
    evidence = registration.encoded(binding)
    reg["learner_writer"]["evidence_pin"].update(
        sha256=identity.sha(evidence), bytes=len(evidence)
    )
    raw = registration.encoded(reg)
    parsed = registration.parse_observed_registration(
        raw, approved_sha256=identity.sha(raw), scope=scope, writer_evidence=evidence
    )
    io = ro.ReadOnlyIO(
        scope, backend=backend, deadline=backend.ns / 1e9 + 20, maximum_bytes=24 * 2**20
    )
    start = {"boot_id": BOOT, "monotonic_ns": backend.ns, "wall_ns": backend.wall_ns()}
    window = {
        "phase": "before",
        "phase_start": start,
        "deadline": {
            **start,
            "monotonic_ns": start["monotonic_ns"] + 20 * 10**9,
            "wall_ns": start["wall_ns"] + 20 * 10**9,
        },
        "cleanup_clock": None,
    }
    premises = {
        "format": k.PREMISES,
        "schema_version": 1,
        "registration_sha256": parsed.sha256,
        "source_context": {
            **{name + "_sha256": k.digest(reg[name]) for name in k.CONTEXT_KEYS},
            "qualification_sha256": "d" * 64,
        },
        "learner_writer_evidence_sha256": identity.sha(evidence),
        "publication_sources": {
            name: row["source_contract_sha256"]
            for name, row in reg["publication_writers"].items()
        },
        "publication_access": {
            name: row["access_qualification_pin"]
            for name, row in reg["publication_writers"].items()
        },
    }
    backend.calls.clear()
    observer = k.ObservedKernelObserver(
        parsed, io, expected_window=window, external_premises=premises
    )
    return parsed, cast(Any, io), window, premises, observer


def test_actual_helpers_issue_private_kernel_snapshot_without_reporters(monkeypatch):
    reg, io, window, premises, observer = observed_kernel_fixture(monkeypatch)
    first = observer.capture_current()
    second = observer.capture_current()
    k.require_snapshot_context(first, reg, io, window, premises)
    k.same_kernel_observations(first, second)
    raw = first.private_copy()
    assert set(raw["owners"]) == registration.ROLES
    assert len(raw["gpu_owners"]) == 8
    assert all(set(owner) == k.OWNER_FIELDS for owner in raw["owners"].values())
    assert all(
        "restarts" not in owner
        and "heartbeat_ns" not in owner
        and "birth_upper_ns" not in owner
        for owner in raw["owners"].values()
    )
    assert raw["provisional_reporter_fields"]["worker_restart_counts"] == "not-read"
    assert "profile-run-continuation-semantics" in raw["source_checks"]["absent"]
    pub_paths = {
        reg.private_copy()["scope"]["files"][p["key"]]["path"]
        for p in reg.private_copy()["publication_writers"].values()
    }
    assert not any(
        kind == "read" and value in pub_paths for kind, value in io._backend.calls
    )
    summary = first.safe_summary()
    assert summary["owners"] == 12 and not summary["preservation_passed"]
    assert SECRET not in repr(first) and SECRET not in json.dumps(summary)
    detached = first.private_copy()
    detached["owners"].clear()
    assert len(first.private_copy()["owners"]) == 12


@pytest.mark.parametrize(
    "kind",
    [
        "dict",
        "canonical",
        "evidence",
        "raw",
        "premise",
        "window",
        "boolean-clock",
        "scope",
        "quota",
    ],
)
def test_constructor_rejects_unqualified_context_before_io(monkeypatch, kind):
    parsed, io, window, premises, _ = observed_kernel_fixture(monkeypatch)
    reg = parsed
    if kind == "dict":
        reg = parsed.private_copy()
    elif kind == "canonical":
        altered = parsed.private_copy()
        altered["policy"]["expected_processes"]["learner"]["pid"] += 1
        reg = registration.ObservedRegistration(
            parsed._raw, registration.encoded(altered), parsed._writer_evidence
        )
    elif kind == "evidence":
        reg = registration.ObservedRegistration(parsed._raw, parsed._canonical, b"{}")
    elif kind == "raw":
        reg = registration.ObservedRegistration(
            b"{}", parsed._canonical, parsed._writer_evidence
        )
    elif kind == "premise":
        premises["source_context"]["qualification_sha256"] = None
    elif kind == "window":
        window["deadline"]["monotonic_ns"] += 106 * 10**9
    elif kind == "boolean-clock":
        window["phase_start"]["monotonic_ns"] = True
    elif kind == "scope":
        io.scope = ro.ReadScope({"foreign.service": "service"}, (), {}, {}, {})
    else:
        io.maximum_bytes = 32 * 2**20
    calls = len(io._backend.calls)
    with pytest.raises(k.KernelRefusal):
        k.ObservedKernelObserver(
            reg, io, expected_window=window, external_premises=premises
        )
    assert len(io._backend.calls) == calls


@pytest.mark.parametrize(
    "field",
    [
        "registration_sha256",
        "learner_writer_evidence_sha256",
        "publication_sources",
        "publication_access",
    ],
)
def test_external_premise_mismatch_is_not_self_approved(monkeypatch, field):
    reg, io, window, premises, _ = observed_kernel_fixture(monkeypatch)
    premises[field] = "f" * 64 if field.endswith("sha256") else {}
    with pytest.raises(k.KernelRefusal):
        k.ObservedKernelObserver(
            reg, io, expected_window=window, external_premises=premises
        )
    assert io._backend.calls == []


@pytest.mark.parametrize("kind", ["copy", "empty", "dict", "bytes", "issuer"])
def test_unissued_or_wrong_issuer_snapshot_refuses_without_io(monkeypatch, kind):
    import copy

    reg, io, window, premises, observer = observed_kernel_fixture(monkeypatch)
    original = observer.capture_current()
    if kind == "copy":
        bad = copy.copy(original)
    elif kind == "empty":
        bad = object.__new__(k.KernelSnapshot)
    elif kind == "dict":
        bad = original.private_copy()
    elif kind == "bytes":
        bad = original
        object.__setattr__(bad, "_canonical", b"{}")
    else:
        other = k.ObservedKernelObserver(
            reg, io, expected_window=window, external_premises=premises
        )
        bad = other.capture_current()
    calls = len(io._backend.calls)
    with pytest.raises(k.KernelRefusal):
        if kind == "issuer":
            k.same_kernel_observations(original, cast(k.KernelSnapshot, bad))
        else:
            k.require_snapshot_context(bad, reg, io, window, premises)
    assert len(io._backend.calls) == calls


def test_snapshot_has_no_public_constructor():
    with pytest.raises(TypeError, match="issued-only"):
        k.KernelSnapshot()


@pytest.mark.parametrize(
    "kind", ["io", "window", "premises", "counter", "maximum", "registration-state"]
)
def test_retained_snapshot_context_cannot_be_replaced(monkeypatch, kind):
    reg, io, window, premises, observer = observed_kernel_fixture(monkeypatch)
    snapshot = observer.capture_current()
    supplied_io = io
    if kind == "io":
        supplied_io = ro.ReadOnlyIO(
            io.scope,
            backend=io._backend,
            deadline=io.deadline,
            maximum_bytes=io.maximum_bytes,
        )
    elif kind == "window":
        window["deadline"]["wall_ns"] += 1
    elif kind == "premises":
        premises["source_context"]["qualification_sha256"] = "e" * 64
    elif kind == "counter":
        io._consumed = 0
    elif kind == "maximum":
        io.maximum_bytes += 1
    else:
        observer.reg["policy"]["expected_processes"]["learner"]["pid"] += 1
    calls = len(io._backend.calls)
    with pytest.raises(k.KernelRefusal):
        k.require_snapshot_context(snapshot, reg, supplied_io, window, premises)
    assert len(io._backend.calls) == calls


@pytest.mark.parametrize(
    "fault",
    [
        "generation",
        "parent",
        "group",
        "uid",
        "namespace",
        "hz",
        "manager",
        "restart",
        "monitor",
        "foreign-gpu",
        "unowned-child",
        "source",
        "cache",
        "later-origin",
    ],
)
def test_current_owner_and_source_faults_refuse_stickily(monkeypatch, fault):
    reg, io, _, _, observer = observed_kernel_fixture(monkeypatch)
    b = io._backend
    learner = reg.private_copy()["policy"]["expected_processes"]["learner"]["pid"]
    if fault == "generation":
        b.raw[learner]["start_ticks"] += 1
    elif fault == "parent":
        b.raw[learner]["ppid"] += 1
    elif fault == "group":
        b.raw[learner]["cgroup"] = "/foreign"
    elif fault == "uid":
        b.credentials[learner]["effective"] += 1
    elif fault == "namespace":
        b.namespace_overrides[learner] = {"pid": 99, "time": 43}
    elif fault == "hz":
        b.hz += 1
    elif fault == "manager":
        b.props[identity.RUNTIME]["InvocationID"] = "f" * 32
    elif fault == "restart":
        b.props[identity.RUNTIME]["NRestarts"] = "1"
    elif fault == "monitor":
        b.props["monitor.service"]["MainPID"] = "999"
    elif fault == "unowned-child":
        b.raw[999] = {**deepcopy(b.raw[learner]), "pid": 999, "ppid": 9999}
    elif fault == "source":
        b.data["source"] = b"private-secret source drift"
    elif fault == "cache":
        next(iter(b.reg["cached_references"].values()))["resolved_stat"]["inode"] += 1
    else:
        b.fault = fault
    with pytest.raises(k.KernelRefusal) as result:
        observer.capture_current()
    assert "private-secret" not in str(result.value)
    calls = len(b.calls)
    with pytest.raises(k.KernelRefusal, match="closed-after-refusal"):
        observer.capture_current()
    assert len(b.calls) == calls and not observer._issued


@pytest.mark.parametrize("fault", ["wall", "monotonic", "read-failure", "storage"])
def test_original_window_and_incomplete_observation_never_issue_snapshot(
    monkeypatch, fault
):
    _, io, _, _, observer = observed_kernel_fixture(monkeypatch)
    b = io._backend
    if fault == "wall":
        b.wall_jump = 21 * 10**9
    elif fault == "monotonic":
        b.advance(21)
    elif fault == "storage":
        monkeypatch.setattr(k, "MAX_PRIVATE_BYTES", 1)
    else:

        def refused(*args, **kwargs):
            raise ro.ReadRefusal("synthetic-read-refusal")

        monkeypatch.setattr(b, "read_file", refused)
    with pytest.raises(k.KernelRefusal):
        observer.capture_current()
    assert not observer._issued


def test_old_process_birth_failure_still_precedes_gpu_reads(monkeypatch):
    from tests.test_strength_freshness_cpu_collect_processes import sample, capture

    case = getattr(sample, "__wrapped__")()
    seen = []
    original_birth = case.collector._births

    def birth(*args):
        seen.append("birth")
        return original_birth(*args)

    monkeypatch.setattr(case.collector, "_births", birth)
    monkeypatch.setattr(
        case.io,
        "query",
        lambda *args: seen.append("gpu") or pytest.fail("GPU read before failed birth"),
    )
    case.identity.reg["birth_reference"]["qualification_sha256"] = None
    with pytest.raises(k.processes.ProcessRefusal, match="birth-unqualified"):
        capture(case)
    assert seen == ["birth"]


def test_old_identity_new_format_and_null_birth_still_refuse_zero_io():
    reg, io, _ = identity_fixture()
    for kind in ("format", "birth"):
        bad = deepcopy(reg)
        if kind == "format":
            bad["format"] = registration.FORMAT
        else:
            bad["birth_reference"]["qualification_sha256"] = None
        with pytest.raises(identity.CollectionRefusal):
            identity.IdentityCollector(bad, cast(ro.ReadOnlyIO, io))
        assert io.calls == []


def test_wall_regression_between_coherent_reads_poisoned(monkeypatch):
    reg, io, window, premises, _ = observed_kernel_fixture(monkeypatch)
    for key in ("monotonic_ns", "wall_ns"):
        window["phase_start"][key] -= 10**9
    observer = k.ObservedKernelObserver(
        reg, io, expected_window=window, external_premises=premises
    )
    backend = io._backend
    original = backend.read_file
    moved = False

    def clock_moves_before_next_observation(*args, **kwargs):
        nonlocal moved
        if observer._last_end is not None and not moved:
            backend.wall_jump -= 1_000_000
            moved = True
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "read_file", clock_moves_before_next_observation)
    with pytest.raises(k.KernelRefusal, match="kernel-observation-regression"):
        observer.capture_current()
    assert moved and len(observer.identity.audit) == 2
    last = observer.identity.audit[-1]
    assert (
        window["phase_start"]["wall_ns"]
        <= last["read_start"]["wall_ns"]
        <= last["read_end"]["wall_ns"]
        < window["deadline"]["wall_ns"]
    )
    calls = len(backend.calls)
    with pytest.raises(k.KernelRefusal, match="closed-after-refusal"):
        observer.capture_current()
    assert len(backend.calls) == calls


def test_malformed_context_refusal_does_not_echo_private_input(monkeypatch):
    reg, io, window, premises, observer = observed_kernel_fixture(monkeypatch)
    snapshot = observer.capture_current()
    bad = deepcopy(window)
    bad["private-secret"] = object()
    with pytest.raises(
        k.KernelRefusal, match="kernel-snapshot-context-invalid"
    ) as caught:
        k.require_snapshot_context(snapshot, reg, io, bad, premises)
    assert "private-secret" not in str(caught.value)


def test_worker_loss_during_final_source_scan_cannot_issue_stale_snapshot(monkeypatch):
    reg, io, _, _, observer = observed_kernel_fixture(monkeypatch)
    backend = io._backend
    learner = reg.private_copy()["policy"]["expected_processes"]["learner"]["pid"]
    source_path = io.scope.files["source"].path
    original = backend.read_file
    source_reads = 0

    def disappears(path, *args, **kwargs):
        nonlocal source_reads
        result = original(path, *args, **kwargs)
        if path == source_path:
            source_reads += 1
            if source_reads == 2:
                del backend.raw[learner]
        return result

    monkeypatch.setattr(backend, "read_file", disappears)
    with pytest.raises(k.KernelRefusal, match="kernel-runtime-members"):
        observer.capture_current()
    assert source_reads == 2 and not observer._issued


@pytest.mark.parametrize("replace_target", ["observer", "identity", "both"])
def test_original_io_object_cannot_be_replaced_by_matching_copy(
    monkeypatch, replace_target
):
    _, io, _, _, observer = observed_kernel_fixture(monkeypatch)
    observer.capture_current()
    replacement = ro.ReadOnlyIO(
        io.scope,
        backend=io._backend,
        deadline=io.deadline,
        maximum_bytes=io.maximum_bytes,
    )
    replacement._consumed, replacement._returned = io._consumed, io._returned
    replacement._admitted = set(io._admitted)
    assert replacement is not io and replacement._consumed > 0
    if replace_target in {"observer", "both"}:
        observer.io = replacement
    if replace_target in {"identity", "both"}:
        observer.identity.io = replacement
    calls = len(io._backend.calls)
    with pytest.raises(k.KernelRefusal, match="kernel-io-instance-changed"):
        observer.capture_current()
    assert len(io._backend.calls) == calls
