"""Outside caller CPU/fake-kernel tests: never run a Linux child or target."""

from dataclasses import asdict, replace
from pathlib import Path
import signal
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import strength_freshness_cpu_outer_audit as m
from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r
from tests.test_strength_freshness_cpu_outer_runtime import admission, pin


def isolate_fake_kernel_signal(monkeypatch):
    # FakeKernel cases also need an explicit signal observation. Replacing this
    # module's reference avoids changing the shared signal module or the worker's
    # real process disposition, which other tests/libraries may legitimately own.
    facade = SimpleNamespace(
        SIGCHLD=signal.SIGCHLD,
        SIG_DFL=signal.SIG_DFL,
        getsignal=lambda which: signal.SIG_DFL,
    )
    monkeypatch.setattr(m, "signal", facade)
    return facade


@pytest.fixture(autouse=True)
def fake_kernel_signal(monkeypatch):
    return isolate_fake_kernel_signal(monkeypatch)


def caller():
    a = admission("caller")
    a.outer["caller"] = asdict(a.identity)
    a.kernel.stdout = [b"{}", b""]
    return a


def no_io(monkeypatch):
    monkeypatch.setattr(m, "recheck_sources", lambda a: None)
    monkeypatch.setattr(
        m, "verify_candidate", lambda *args: {"checked": "synthetic-terminal-only"}
    )


def successful_handoff(a, family, window):
    assert family.root is not None
    return {"start": {"clock": asdict(a.preflight_start)}}


def test_actual_owned_family_path_reaps_supervisor_without_self_exit_claim(monkeypatch):
    no_io(monkeypatch)
    monkeypatch.setattr(m, "observe_handoff", successful_handoff)
    a = caller()
    report = m.run_caller(a, authorization_path="/input/outer.json")
    assert a.kernel.forked == 1 and a.kernel.released
    assert a.kernel.reaped == [200] and not a.kernel.children()
    assert report["supervisor_family"]["child_terminal"]["waitpid"] == {
        "pid": 200,
        "status": 0,
    }
    assert report["supervisor_family"]["family_closed"]["direct_children"] == []
    assert report["caller_future_exit_claimed"] is False
    assert report["target_execution_qualified"] is False
    assert report["caller_admission"]["role"] == "caller"
    assert a.reader.published == ["/input/outside-caller-audit.json"]


def test_fixed_spec_initial_deadline_and_separate_aggregate_alarm():
    start = o.Clock("boot", o.SECOND, 10**15)
    w = m.CallerWindow(start)
    s = m.SupervisorSpec(
        "session",
        "supervisor",
        "/qualified/python",
        "/qualified/control",
        "/input/outer.json",
        "a" * 64,
        w.work_ns,
        b"{}",
        ("b" * 64, "c" * 64),
    )
    assert s.deadline_monotonic_ns == start.monotonic_ns + 119 * o.SECOND
    assert s.process_deadline_ns == start.monotonic_ns + 720 * o.SECOND
    assert s.argv()[6:8] == ("--role", "supervisor")
    assert s.contract()["resources"]["address_space_hard_bytes"] == 16 * 2**30
    assert s.contract()["argv_template"][-1] == "@approved-outer-intent-sha256"
    with pytest.raises(o.OuterRefusal):
        replace(s, role="shell").argv()


def test_caller_window_does_not_reuse_early_operator_cutoff():
    start = o.Clock("boot", o.SECOND, 10**15)
    w = m.CallerWindow(start)
    assert w.compact()["work"]["monotonic_ns"] == 120 * o.SECOND
    dummy = replace(
        start, monotonic_ns=101 * o.SECOND, wall_ns=start.wall_ns + 100 * o.SECOND
    )
    active = w.handoff(dummy, dummy)
    assert active.work_ns == dummy.monotonic_ns + 599 * o.SECOND
    assert (
        active.compact()["gate"]["monotonic_ns"] == dummy.monotonic_ns + 600 * o.SECOND
    )
    assert (
        active.compact()["gate"]["monotonic_ns"] <= start.monotonic_ns + 720 * o.SECOND
    )
    late = replace(
        start,
        monotonic_ns=start.monotonic_ns + 118 * o.SECOND,
        wall_ns=start.wall_ns + 118 * o.SECOND,
    )
    with pytest.raises(o.OuterRefusal):
        w.handoff(dummy, late)


@pytest.mark.parametrize(
    "kind",
    [
        "nonzero",
        "signal",
        "orphan",
        "parent",
        "nohandoff",
        "readiness",
        "output",
        "waitpid",
        "pidfd",
    ],
)
def test_failure_matrix_cannot_publish_complete_and_never_signals_foreign(
    monkeypatch, kind
):
    no_io(monkeypatch)
    monkeypatch.setattr(
        m,
        "observe_handoff",
        successful_handoff if kind != "nohandoff" else lambda *args: None,
    )
    a = caller()
    if kind == "nonzero":
        a.kernel.exit_code = 1
    elif kind == "signal":
        a.kernel.exit_kind = "CLD_KILLED"
        a.kernel.exit_code = 9
    elif kind == "orphan":
        a.kernel.orphans = 1
    elif kind == "parent":
        a.kernel.parent_dead_at = a.kernel.t
    elif kind == "readiness":
        a.kernel.ready = False
    elif kind == "output":
        a.kernel.stdout = [b"x" * (2**20 + 1)]
    elif kind == "waitpid":
        a.kernel.wait_mismatch = True
    elif kind == "pidfd":
        a.kernel.pidfd_wrong = True
    with pytest.raises((r.RuntimeRefusal, o.OuterRefusal)):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert "/input/outside-caller-audit.json" not in a.reader.files
    assert all(pid in (200, 300) for pid, _ in a.kernel.signalled)
    assert 50 not in [pid for pid, _ in a.kernel.signalled]


@pytest.mark.parametrize("handler", [signal.SIG_IGN, lambda signum, frame: None])
def test_fake_kernel_success_is_independent_of_preceding_signal_observer(
    monkeypatch, handler
):
    actual_getsignal = signal.getsignal
    actual_handler = signal.getsignal(signal.SIGCHLD)
    preceding = SimpleNamespace(
        SIGCHLD=signal.SIGCHLD,
        SIG_DFL=signal.SIG_DFL,
        getsignal=lambda which: handler,
    )
    monkeypatch.setattr(m, "signal", preceding)
    assert m.signal.getsignal(signal.SIGCHLD) is handler
    isolate_fake_kernel_signal(monkeypatch)
    test_actual_owned_family_path_reaps_supervisor_without_self_exit_claim(monkeypatch)
    assert preceding.getsignal(signal.SIGCHLD) is handler
    assert signal.getsignal is actual_getsignal
    assert signal.getsignal(signal.SIGCHLD) is actual_handler


@pytest.mark.parametrize("handler", [signal.SIG_IGN, lambda signum, frame: None])
def test_bad_inherited_sigchld_refuses_before_fork(monkeypatch, handler):
    no_io(monkeypatch)
    monkeypatch.setattr(m.signal, "getsignal", lambda which: handler)
    a = caller()
    with pytest.raises(r.RuntimeRefusal, match="sigchld"):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert a.kernel.forked == 0


def test_preexisting_child_is_not_claimed_or_killed(monkeypatch):
    no_io(monkeypatch)
    a = caller()
    a.kernel.processes[300] = replace(
        a.identity, pid=300, ppid=a.identity.pid, start_ticks=30
    )
    with pytest.raises(o.OuterRefusal, match="family-not-fresh"):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert a.kernel.forked == 0 and a.kernel.signalled == []


@pytest.mark.parametrize(
    "field,value",
    [("pid", 999), ("start_ticks", 999), ("cgroup", "/other"), ("boot_id", "other")],
)
def test_caller_lifetime_must_already_be_prebound(field, value):
    a = caller()
    a.outer["caller"][field] = value
    with pytest.raises(r.RuntimeRefusal, match="self-not-prebound"):
        r.caller_parent(a.kernel, a.identity, a.outer["caller"])
    assert a.kernel.forked == 0


def test_source_alias_cannot_satisfy_actual_caller_origin(monkeypatch):
    a = caller()
    a.outer["source_pins"].append(
        pin("/qualified/control/other/strength_freshness_cpu_outer_audit.py")
    )
    monkeypatch.setattr(
        m,
        "__file__",
        "/qualified/control/scripts/strength_freshness_cpu_outer_audit.py",
    )
    with pytest.raises(r.RuntimeRefusal, match="source-missing"):
        m.recheck_sources(a)


def test_proof_recheck_cost_cannot_renew_final_gate(monkeypatch):
    no_io(monkeypatch)
    monkeypatch.setattr(m, "observe_handoff", successful_handoff)
    a = caller()

    def slow(*args):
        a.kernel.wall_jump += 600 * o.SECOND
        return {}

    monkeypatch.setattr(m, "verify_candidate", slow)
    with pytest.raises(r.RuntimeRefusal, match="final-owner-or-deadline"):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert (
        a.kernel.reaped == [200]
        and "/input/outside-caller-audit.json" not in a.reader.files
    )


def test_fixed_cli_missing_admission_reports_only_generic_failure(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        assert kwargs == {"role": "caller"}
        raise r.RuntimeRefusal("private-secret-not-for-output")

    monkeypatch.setattr(r, "load_admission", refuse)
    assert m.main(["--authorization", "/input/outer.json", "--sha256", "a" * 64]) == 1
    captured = capsys.readouterr()
    assert "private-secret" not in captured.err and captured.out == ""
    assert "caller-audit-refused" in captured.err


def test_cli_does_not_accept_arbitrary_command():
    with pytest.raises(SystemExit):
        m.main(
            [
                "--authorization",
                "/input/a",
                "--sha256",
                "a" * 64,
                "--command",
                "anything",
            ]
        )


def test_platform_refusal_precedes_any_target_fork(monkeypatch):
    monkeypatch.setattr(r.sys, "platform", "darwin")
    with pytest.raises(r.RuntimeRefusal, match="root-linux"):
        r.load_admission("/input/a", "a" * 64, role="caller")


def test_handoff_refuses_foreign_actual_operator_birth(monkeypatch):
    a = caller()
    sup = replace(a.identity, pid=200, start_ticks=20, ppid=100)
    op = replace(a.identity, pid=300, start_ticks=30, ppid=200)
    a.kernel.processes[200] = sup
    a.kernel.processes[300] = replace(op, start_ticks=31)
    before = {"enclosing": {"supervisor": asdict(sup), "operator": asdict(op)}}
    b = r.encode(before)
    before_pin = pin("/input/before-execution.json", b)
    start = {
        "format": "strength-freshness-dummy-start-v1",
        "schema_version": 1,
        "outer_intent_sha256": a.approved_intent_sha256,
        "before_execution_pin": before_pin,
        "clock": asdict(a.preflight_start),
    }
    a.reader.files.update(
        {
            "/input/dummy-start.ack.json": b"{}",
            "/input/dummy-start.json": r.encode(start),
            before_pin["path"]: b,
        }
    )
    with pytest.raises(r.RuntimeRefusal, match="live-operator-parentage"):
        m.observe_handoff(
            a, SimpleNamespace(root=sup), m.CallerWindow(a.preflight_start)
        )
    assert a.reader.published == []


def composed_fixture(monkeypatch):
    """Actual existing pure completion/family schema with synthetic kernel facts."""
    import copy
    import hashlib
    from tests.test_strength_freshness_cpu_completion import valid_completion

    a = caller()
    a.approved_intent_sha256 = r.sha(a.raw_intent)
    start = o.Clock("boot", 10000000000000001, 1800000000100000001)
    # This start deliberately does not survive float→nanoseconds round trip.
    assert int((start.monotonic_ns / o.SECOND) * o.SECOND) != start.monotonic_ns
    a.outer["preflight_start"] = asdict(start)
    now = [
        replace(
            start,
            monotonic_ns=start.monotonic_ns + 21 * o.SECOND,
            wall_ns=start.wall_ns + 21 * o.SECOND,
        )
    ]
    a.kernel.clock = lambda: now[0]
    sup = replace(a.identity, pid=200, start_ticks=20, ppid=100, pgid=200, sid=200)
    op = replace(a.identity, pid=300, start_ticks=30, ppid=200, pgid=300, sid=300)
    guard = replace(a.identity, pid=400, start_ticks=40, ppid=300, pgid=400, sid=400)
    child = replace(a.identity, pid=500, start_ticks=50, ppid=400, pgid=500, sid=500)
    a.kernel.processes.update({200: sup, 300: op})
    identities = {10: asdict(sup), 20: asdict(op), 30: asdict(guard), 40: asdict(child)}

    def store(path, doc):
        raw = doc if isinstance(doc, bytes) else r.encode(doc)
        p = pin(path, raw)
        a.reader.files[str(path)] = raw
        return p

    def at(base, sec):
        return asdict(
            replace(
                base,
                monotonic_ns=base.monotonic_ns + sec * o.SECOND,
                wall_ns=base.wall_ns + sec * o.SECOND,
            )
        )

    def transform(x, old_start, new_start) -> Any:
        if isinstance(x, dict):
            if set(x) == r.completion.IDENTITY:
                return identities[x["pid"]]
            if set(x) == {"boot_id", "monotonic_ns", "wall_ns"}:
                return {
                    "boot_id": "boot",
                    "monotonic_ns": x["monotonic_ns"]
                    - old_start["monotonic_ns"]
                    + new_start.monotonic_ns,
                    "wall_ns": x["wall_ns"] - old_start["wall_ns"] + new_start.wall_ns,
                }
            return {k: transform(v, old_start, new_start) for k, v in x.items()}
        if isinstance(x, list):
            return [transform(v, old_start, new_start) for v in x]
        if type(x) is int and x in identities:
            return identities[x]["pid"]
        return x

    sup_admit = {
        "role": "supervisor",
        "self": asdict(sup),
        "clock": at(start, 0),
        "subreaper_before": 0,
        "subreaper_after": 1,
        "task_ids": [sup.pid],
    }
    enclosing = {"operator": asdict(op), "supervisor": asdict(sup)}

    def phase(name, phase_start, anchor=None, plan_pin=None, before_pin=None):
        _, exp, _, evidence, _ = valid_completion(name)
        docs = {
            n: transform(r.completion.parse(b), exp["phase_start"], phase_start)
            for n, b in evidence.items()
        }
        parent = (
            Path("/input")
            if name == "before"
            else Path("/run/edgeconnect-cpuqual-" + a.outer["nonce"] + "/external")
        )
        inputs = parent if name == "before" else parent / "capture-inputs"
        request = store(inputs / "request.json", {"fixture_request": name})
        registration = store(
            inputs / "registration.json", {"fixture_registration": True}
        )
        launch_body: dict[str, Any] = {"request": request, "registration": registration}
        if name == "after":
            launch_body.update(plan=plan_pin, anchor=anchor_pin)
        launch = store(inputs / "launch.json", launch_body)
        capture = store(parent / ("r3-" + name + ".json"), {"fixture_capture": name})
        provenance_raw = r.encode({"fixture_provenance": name})
        provenance = store(
            parent / ("r3-" + name + ".provenance-" + r.sha(provenance_raw) + ".json"),
            provenance_raw,
        )
        receipt = store(
            parent / ("r3-" + name + ".receipt.json"),
            {
                "status": "complete",
                "capture_pin": capture,
                "request_sha256": request["sha256"],
                "registration_sha256": registration["sha256"],
                "derivations": {
                    "launch_sha256": launch["sha256"],
                    "provenance_pin": provenance,
                },
            },
        )
        artifacts = dict(
            launch=launch,
            request=request,
            registration=registration,
            capture=capture,
            receipt=receipt,
            provenance=provenance,
        )
        budget = (
            r.completion.before_budget(asdict(phase_start))
            if name == "before"
            else r.completion.after_budget(asdict(phase_start), anchor)
        )
        guardian_hash = r.guardian_contract_sha256(
            a.outer["python"]["path"],
            a.outer["control_root"],
            a.outer["authorization_path"],
            a.role_source_hashes,
        )
        collector_hash = r.completion.collector_contract_sha256(
            a.outer["python"]["path"],
            a.outer["control_root"],
            launch,
            budget["work"]["monotonic_ns"],
        )
        binding = {
            "phase": name,
            "nonce": a.outer["nonce"],
            "boot_id": "boot",
            "outer_intent_sha256": a.approved_intent_sha256,
            "qualified_outer_source_sha256": a.source_sha256,
            "phase_start": asdict(phase_start),
        }
        for key, exe in [
            ("collector_family", collector_hash),
            ("guardian_family", guardian_hash),
        ]:
            docs[key]["binding"] = binding
            docs[key]["budget"] = budget
            for event in ("child_started", "child_released"):
                docs[key][event]["exec_contract_sha256"] = exe
        docs["supervisor_admission"] = sup_admit
        epins = {
            n: store(
                parent
                / "execution-evidence"
                / (name + "-" + n.replace("_", "-") + ".json"),
                d,
            )
            for n, d in docs.items()
        }
        value = {
            "format": r.completion.FORMAT,
            "schema_version": 1,
            "status": "natural-exit-complete",
            **binding,
            "budget": budget,
            "enclosing": enclosing,
            "exec_contracts": {"guardian": guardian_hash, "collector": collector_hash},
            "artifacts": artifacts,
            "plan_sha256": None if plan_pin is None else plan_pin["sha256"],
            "anchor_sha256": None if anchor is None else r.sha(r.encode(anchor)),
            "before_execution_pin": before_pin,
            "terminal_clock": at(phase_start, 13),
            "published_clock": at(phase_start, 14),
            "evidence_pins": epins,
        }
        cp = store(
            parent
            / (
                "before-execution.json"
                if name == "before"
                else "r3-after.execution.json"
            ),
            value,
        )
        a.reader.roots = tuple(
            set(a.reader.roots) | {parent, inputs, parent / "execution-evidence"}
        )
        if name == "before":
            a.outer.update(
                before_launch=launch, before_request=request, registration=registration
            )
        return {
            "phase": name,
            "completion_pin": cp,
            "artifacts": artifacts,
            "evidence_pins": epins,
        }, docs

    before, docs = phase("before", start)
    dummy = replace(
        start,
        monotonic_ns=start.monotonic_ns + 20 * o.SECOND,
        wall_ns=start.wall_ns + 20 * o.SECOND,
    )
    start_body = {
        "format": "strength-freshness-dummy-start-v1",
        "schema_version": 1,
        "outer_intent_sha256": a.approved_intent_sha256,
        "before_execution_pin": before["completion_pin"],
        "clock": asdict(dummy),
    }
    start_pin = store("/input/dummy-start.json", start_body)
    ack_pin = store(
        "/input/dummy-start.ack.json",
        r.dummy_start_ack(
            outer_intent_sha256=a.approved_intent_sha256,
            nonce=a.outer["nonce"],
            start_pin=start_pin,
            before_execution_pin=before["completion_pin"],
            enclosing=enclosing,
        ),
    )
    handoff = m.observe_handoff(a, SimpleNamespace(root=sup), m.CallerWindow(start))
    assert handoff is not None and handoff["ack_pin"] == ack_pin
    plan_pin = store("/input/plan.json", {"nonce": a.outer["nonce"]})
    anchor = {
        "attempt_id": "fixture",
        "nonce": a.outer["nonce"],
        "plan_sha256": plan_pin["sha256"],
        "boot_id": "boot",
        "started_monotonic": dummy.monotonic_ns / o.SECOND,
        "started_wall_ns": dummy.wall_ns,
    }
    anchor_pin = store("/input/anchor.json", anchor)
    phase_start = replace(
        dummy,
        monotonic_ns=dummy.monotonic_ns + 540 * o.SECOND,
        wall_ns=dummy.wall_ns + 540 * o.SECOND,
    )
    after, _ = phase("after", phase_start, anchor, plan_pin, before["completion_pin"])
    external = Path("/run/edgeconnect-cpuqual-" + a.outer["nonce"])
    audit_pin = store(
        external / "evidence/audit.json",
        {"status": "passed_cpu_scope", "fixture": True},
    )
    operator = {
        "format": "strength-freshness-operator-result-v1",
        "status": "passed_cpu_scope",
        "before_bundle": before,
        "after_bundle": after,
        "start_pin": start_pin,
        "audit": {
            "status": "passed_cpu_scope",
            "plan_sha256": plan_pin["sha256"],
            "anchor_sha256": anchor_pin["sha256"],
            "evidence_pins": {"external_audit": audit_pin},
        },
    }
    family = copy.deepcopy(docs["guardian_family"])
    # Reuse the real pure family schema, but this is the supervisor→operator pair.
    family["role_admission"] = sup_admit
    for key in ("child_started", "child_terminal"):
        family[key]["owner"] = asdict(sup)
    for key in ("child_started", "child_released", "child_terminal"):
        family[key]["child"] = asdict(op)
    family["child_started"]["pidfd_target_pid"] = op.pid
    family["child_terminal"]["waitid"]["pid"] = op.pid
    family["child_terminal"]["waitpid"]["pid"] = op.pid
    family["family_closed"].update(owner=asdict(sup), task_ids=[sup.pid])
    family["binding"] = {**family["binding"], "phase": "session"}
    family["budget"] = o.BudgetWindow(start, dummy).compact()
    op_spec = o.RoleSpec(
        "session",
        "operator",
        "/qualified/python",
        "/qualified/control",
        "/input/outer.json",
        a.approved_intent_sha256,
        start.monotonic_ns + 118 * o.SECOND,
        b"{}",
        a.role_source_hashes,
    )
    for event in ("child_started", "child_released"):
        family[event]["exec_contract_sha256"] = o.digest(op_spec.contract())
    family["output"].update(
        stdout_sha256=hashlib.sha256(r.encode(operator)).hexdigest(),
        stdout_bytes=len(r.encode(operator)),
    )
    candidate = {
        "format": "strength-freshness-supervisor-result-v1",
        "status": "candidate_cpu_scope",
        "operator_result": operator,
        "operator_family": family,
        "supervisor_admission": sup_admit,
        "outside_caller_terminal_audit_required": True,
        "execution_qualified": False,
    }
    raw = r.encode(candidate)
    store("/input/outer-result.json", raw)
    outer_family = {
        "child_released": {"clock": at(start, 0)},
        "child_terminal": {"clock": at(dummy, 598)},
    }
    now[0] = r.clock(at(dummy, 599))
    a.reader.files["/input/outer.json"] = a.raw_intent
    monkeypatch.setattr(m, "recheck_sources", lambda a: None)
    return a, raw, outer_family, handoff, m.CallerWindow(start, dummy), candidate, now


def test_real_before_after_completion_and_candidate_composition(monkeypatch):
    args = composed_fixture(monkeypatch)
    pins = m.verify_candidate(*args[:5])
    assert pins["before_execution"] == args[3]["start"]["before_execution_pin"]
    assert pins["anchor"]["path"] == "/input/anchor.json"


@pytest.mark.parametrize("mutation", ["output", "plan", "ack", "source", "late"])
def test_composed_candidate_mismatches_refuse(monkeypatch, mutation):
    a, raw, family, handoff, window, _, now = composed_fixture(monkeypatch)
    if mutation == "output":
        a.reader.files["/input/outer-result.json"] = b"{}"
    elif mutation == "plan":
        a.reader.files["/input/plan.json"] = b'{"nonce":"other"}'
    elif mutation == "ack":
        a.reader.files["/input/dummy-start.ack.json"] = b"{}"
    elif mutation == "source":

        def refuse(a):
            raise r.RuntimeRefusal("source-drift")

        monkeypatch.setattr(m, "recheck_sources", refuse)
    else:
        now[0] = replace(now[0], wall_ns=now[0].wall_ns + o.SECOND)
    with pytest.raises(
        (r.RuntimeRefusal, r.completion.CompletionRefusal, AssertionError)
    ):
        m.verify_candidate(a, raw, family, handoff, window)


def test_lost_kernel_wait_authority_cannot_be_replaced_by_reported_exit(monkeypatch):
    no_io(monkeypatch)
    monkeypatch.setattr(m, "observe_handoff", successful_handoff)
    a = caller()

    def no_child(_fd):
        raise ChildProcessError("no actual wait authority")

    a.kernel.reap = no_child
    with pytest.raises(ChildProcessError):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert "/input/outside-caller-audit.json" not in a.reader.files
    assert a.kernel.signalled == []


def test_reused_supervisor_pid_does_not_grant_signal_or_completion(monkeypatch):
    no_io(monkeypatch)
    monkeypatch.setattr(m, "observe_handoff", successful_handoff)
    a = caller()
    release = a.kernel.release

    def reuse(fd):
        release(fd)
        a.kernel.processes[200] = replace(a.kernel.processes[200], start_ticks=999)

    a.kernel.release = reuse
    with pytest.raises(o.OuterRefusal):
        m.run_caller(a, authorization_path="/input/outer.json")
    assert "/input/outside-caller-audit.json" not in a.reader.files
    assert a.kernel.signalled == []


def admission_fixture(tmp_path, monkeypatch, variation):
    """Shared loader with real small-file hashing and explicit fake kernel/syscalls."""
    import os
    from tests.test_strength_freshness_cpu_outer_runtime import MemoryStore, FakeKernel

    k = FakeKernel()
    k.t = o.SECOND
    store = MemoryStore()
    inputs = tmp_path / "input"
    inputs.mkdir()
    inputs.chmod(0o700)
    control = Path(r.__file__).resolve().parent.parent
    files = [
        control / "scripts" / name
        for name in sorted(
            r.REQUIRED_SOURCE_NAMES | {"strength_freshness_cpu_outer_audit.py"}
        )
    ]
    source_pins = [pin(p, p.read_bytes()) for p in files]
    for p, expected in zip(files, source_pins):
        store.files[expected["path"]] = p.read_bytes()
    python = tmp_path / "python"
    python.write_bytes(b"synthetic executable image")
    python.chmod(0o775)
    python_pin = {
        **pin(python, python.read_bytes()),
        "resolved_path": str(python),
        "metadata": r.file_identity(python.stat()),
    }
    store.files[str(python)] = python.read_bytes()

    def keep(name, doc):
        raw = r.encode(doc)
        p = pin(inputs / name, raw)
        store.files[p["path"]] = raw
        return p

    site_root = tmp_path / "site"
    site_root.mkdir()
    startup = site_root / "bootstrap.pth"
    startup.write_bytes(b"import fixture\n")
    lib = site_root / "library.so"
    lib.write_bytes(b"cached fixture library")
    startup_meta = r.file_identity(startup.stat())
    lib_meta = r.file_identity(lib.stat())
    store.files[str(startup)] = startup.read_bytes()
    cache = keep(
        "file-cache.json",
        {
            "format": "strength-freshness-qualified-file-cache-v1",
            "status": "independently-qualified",
            "evidence_kind": "real-host-content-hash",
            "synthetic": False,
            "path": str(lib),
            "sha256": r.sha(lib.read_bytes()),
            "metadata": lib_meta,
        },
    )
    inventory = keep(
        "site-inventory.json",
        {
            "format": "strength-freshness-qualified-site-files-v1",
            "schema_version": 1,
            "root": str(site_root),
            "startup_directories": [
                {"path": str(site_root), "entries": sorted([startup.name, lib.name])}
            ],
            "startup_files": [
                {
                    "path": str(startup),
                    "sha256": r.sha(startup.read_bytes()),
                    "metadata": startup_meta,
                }
            ],
            "cached_files": [
                {
                    "path": str(lib),
                    "sha256": r.sha(lib.read_bytes()),
                    "metadata": lib_meta,
                    "cache_receipt": cache,
                }
            ],
        },
    )
    raw_proof = keep(
        "raw-fake-proof.json",
        {"scope": "synthetic test only; not a real qualification"},
    )
    common = {
        "schema_version": 1,
        "status": "passed-real-linux",
        "evidence_kind": "real-linux-process",
        "synthetic": False,
        "boot_id": "boot",
        "source_closure_sha256": o.digest(source_pins),
        "python_sha256": python_pin["sha256"],
        "resource_policy_sha256": o.digest(r.resource_policy()),
        "raw_evidence_pins": [raw_proof],
    }
    session = keep(
        "session.json",
        {
            **common,
            "format": r.SESSION,
            "caller_cgroup": k.owner.cgroup,
            "pid_namespace_inode": k.owner.pid_namespace_inode,
            "checks": {key: True for key in r.SESSION_CHECKS},
        },
    )
    site = keep(
        "site.json",
        {
            **common,
            "format": "strength-freshness-real-helper-site-qualification-v1",
            "helper_environment_sha256": o.digest(
                {**o.ENV, "PYTHONPATH": str(control)}
            ),
            "checks": {
                key: True
                for key in (
                    "site_enabled_import_closure",
                    "no_gpu_initialization",
                    "resource_limits",
                    "actual_helper_origin",
                    "no_user_site",
                    "loader_environment",
                )
            },
            "site_inventory_pin": inventory,
        },
    )
    cfg = {
        "format": r.ADMISSION,
        "schema_version": 1,
        "nonce": "b" * 32,
        "preflight_start": asdict(k.clock()),
        "input_root": str(inputs),
        "control_root": str(control),
        "python": python_pin,
        "source_pins": source_pins,
        "session_qualification": None if variation == "missing-receipt" else session,
        "helper_site_qualification": site,
        "caller": asdict(k.owner),
        "before_launch": pin(inputs / "launch.json"),
        "before_request": pin(inputs / "request.json"),
        "registration": pin(inputs / "registration.json"),
        "verified_champions": {},
        "resource_policy_sha256": o.digest(r.resource_policy()),
    }
    if variation == "wrong-self":
        cfg["caller"]["start_ticks"] += 1
    raw = r.encode({"nonce": cfg["nonce"], "boot_id": "boot", "outer": cfg})
    auth = inputs / "outer.json"
    auth.write_bytes(raw)
    auth.chmod(0o444)
    store.files[str(auth)] = raw
    # Resource and process surfaces are fake, but the unchanged loader, actual
    # interpreter hash/stat routine and current site inventory validation execute.
    monkeypatch.setattr(r.sys, "platform", "linux")
    monkeypatch.setattr(
        r.sys,
        "flags",
        SimpleNamespace(
            dont_write_bytecode=1, no_site=1, ignore_environment=1, no_user_site=0
        ),
    )
    monkeypatch.setattr(r.os, "geteuid", lambda: 0)
    monkeypatch.setattr(r.sys, "executable", str(python))
    real_open = os.open
    monkeypatch.setattr(r.os, "readlink", lambda path: str(python))
    monkeypatch.setattr(
        r.os,
        "open",
        lambda path, flags, *args, **kwargs: real_open(
            python if path == "/proc/self/exe" else path, flags, *args, **kwargs
        ),
    )
    monkeypatch.setattr(r.os, "environ", dict(o.ENV))
    # Model the fresh -S process inventory; pytest itself has unrelated training imports.
    monkeypatch.setattr(
        r.sys,
        "modules",
        {
            name: module
            for name, module in r.sys.modules.items()
            if not (name == "torch" or name.startswith(("torch.", "deltreltrain")))
        },
    )
    limits = {
        r.resource.RLIMIT_AS: (r.resource.RLIM_INFINITY, r.resource.RLIM_INFINITY),
        r.resource.RLIMIT_NOFILE: (256, 256),
        r.resource.RLIMIT_FSIZE: (r.resource.RLIM_INFINITY, r.resource.RLIM_INFINITY),
    }
    calls = []

    def limit(which, value):
        limits[which] = value
        calls.append(which)

    monkeypatch.setattr(r.resource, "getrlimit", lambda which: limits[which])
    monkeypatch.setattr(r.resource, "setrlimit", limit)
    monkeypatch.setattr(r.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    monkeypatch.setattr(
        r.os, "sched_setaffinity", lambda pid, cpus: None, raising=False
    )
    monkeypatch.setattr(r.os, "getpriority", lambda *args: 19)
    monkeypatch.setattr(r.os, "nice", lambda adjustment: 19)
    return k, store, str(auth), r.sha(raw), calls


@pytest.mark.parametrize("variation", ["positive", "wrong-self", "missing-receipt"])
def test_actual_shared_caller_loader_branch(tmp_path, monkeypatch, variation):
    k, store, path, digest, calls = admission_fixture(tmp_path, monkeypatch, variation)
    if variation == "positive":
        admitted = r.load_admission(path, digest, role="caller", kernel=k, store=store)
        assert (
            admitted.identity == k.owner and admitted.parent.identity == k.processes[50]
        )
        assert admitted.role == "caller" and len(calls) == 3
        assert admitted.session_pin["path"].endswith("/session.json")
    else:
        reason = (
            "caller-self-not-prebound"
            if variation == "wrong-self"
            else "real-session-qualification-missing"
        )
        with pytest.raises(r.RuntimeRefusal, match=reason):
            r.load_admission(path, digest, role="caller", kernel=k, store=store)
        assert calls == []
    assert k.forked == 0 and k.signalled == []
