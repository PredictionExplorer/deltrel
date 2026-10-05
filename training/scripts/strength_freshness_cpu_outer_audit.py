"""Fixed terminal caller: attest a reaped supervisor, never our future exit.

The external preparer must bind this actual caller lifetime and independently
qualify the original session/bootstrap before this CLI can release a child.
No target qualification receipt is manufactured here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import signal
import sys

from scripts import strength_freshness_cpu_completion as c
from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r

FORMAT = "strength-freshness-outside-caller-audit-v1"


@dataclass(frozen=True)
class CallerWindow(o.BudgetWindow):
    """Supervisor work119/599; this caller's cleanup/audit120/600, no renewal."""

    def compact(self) -> dict:
        base = super().compact()
        return {
            "work": dict(base["cleanup"]),
            "cleanup": dict(base["gate"]),
            "gate": dict(base["gate"]),
        }

    def handoff(self, dummy_start: o.Clock, observed: o.Clock) -> CallerWindow:
        # Admission retains118, although terminal supervisor work may use119.
        o.BudgetWindow(self.start).handoff(dummy_start, observed)
        o.require(self.dummy_start is None, "late-or-repeated-handoff")
        return CallerWindow(self.start, dummy_start)


@dataclass(frozen=True)
class SupervisorSpec(o.RoleSpec):
    """One exact existing supervisor entrypoint; no caller-selected program."""

    def argv(self) -> tuple[str, ...]:
        o.require(
            self.role == "supervisor" and self.phase == "session",
            "fixed-supervisor-role",
        )
        # Reuse all existing fixed path/hash/frame validation, then replace only
        # the closed role token. Runtime's supervisor ignores the unused frame.
        template = o.RoleSpec(
            "session",
            "operator",
            self.python,
            self.control_root,
            self.authorization_path,
            self.authorization_sha256,
            self.deadline_monotonic_ns,
            self.frame,
            self.qualified_sources,
        )
        args = list(template.argv())
        args[7] = "supervisor"
        return tuple(args)

    @property
    def process_deadline_ns(self) -> int:
        return self.deadline_monotonic_ns + (720 - 119) * o.SECOND

    def contract(self) -> dict:
        args = list(self.argv())
        args[-1] = "@approved-outer-intent-sha256"
        return {
            "format": "strength-freshness-fixed-supervisor-contract-v1",
            "argv_template": args,
            "cwd": self.control_root,
            "env": dict(o.ENV),
            "qualified_sources": {
                "primitive": self.qualified_sources[0],
                "runtime": self.qualified_sources[1],
            },
            "resources": {
                **o.CAPTURE_LIMITS,
                "address_space_hard_bytes": 16 * 2**30,
                "inherited_frame_fd": 127,
                "inherited_frame_max_bytes": 65536,
            },
        }


class CallerKernel(o.LinuxKernel):
    def __init__(self, admitted: r.Admission):
        super().__init__()
        self.admitted = admitted

    def admit_execution(self, binding, spec) -> None:
        a = self.admitted
        r.require(
            type(spec) is SupervisorSpec and a.role == "caller",
            "caller-fixed-supervisor-only",
        )
        assert isinstance(spec, SupervisorSpec)
        r.require(
            self.self_identity() == a.identity and self.task_ids() == (a.identity.pid,),
            "caller-current-owner",
        )
        r.caller_parent(self, a.identity, a.outer["caller"])
        r.require(not self.exited(a.parent.pidfd), "caller-parent-lost")
        r.require(
            spec.authorization_path == a.outer["authorization_path"]
            and spec.authorization_sha256 == a.approved_intent_sha256
            and spec.python == a.outer["python"]["path"]
            and spec.control_root == a.outer["control_root"]
            and spec.qualified_sources == a.role_source_hashes,
            "caller-exec-authority",
        )
        r.require(
            binding.nonce == a.outer["nonce"]
            and binding.intent_sha256 == a.approved_intent_sha256
            and binding.source_sha256 == a.source_sha256
            and binding.session_admission_sha256 == a.session_pin["sha256"],
            "caller-child-binding",
        )
        r.current_interpreter(a.reader, a.outer["python"], hash_image=False)
        recheck_sources(a)
        self._admitted_exec = o.digest(spec.contract())


def recheck_sources(a: r.Admission) -> None:
    expected = str(
        Path(a.outer["control_root"]) / "scripts/strength_freshness_cpu_outer_audit.py"
    )
    r.require(str(Path(__file__).resolve()) == expected, "caller-audit-module-origin")
    pins = {p["path"]: p for p in a.outer["source_pins"]}
    r.require(expected in pins, "caller-audit-source-missing")
    for p in pins.values():
        a.reader.read(p, source=True)


def _fixed(a, path, maximum=2**20):
    a.reader.roots = tuple(set(a.reader.roots) | {path.parent})
    return a.reader.fixed(path, maximum=maximum)


def _evidence(a, raw, phase, parent):
    pins = c.evidence_pins(raw)
    expected = {
        name: parent
        / "execution-evidence"
        / (phase + "-" + name.replace("_", "-") + ".json")
        for name in pins
    }
    r.require(
        all(Path(pins[n]["path"]) == expected[n] for n in pins), "caller-evidence-path"
    )
    a.reader.roots = tuple(set(a.reader.roots) | {parent / "execution-evidence"})
    return {name: a.reader.read(p, maximum=c.MAX_EVIDENCE) for name, p in pins.items()}


def verify_capture(
    a,
    phase,
    completion_pin,
    *,
    enclosing,
    at,
    anchor=None,
    plan_pin=None,
    before_pin=None,
):
    parent = (
        Path(a.outer["input_root"])
        if phase == "before"
        else Path("/run") / ("edgeconnect-cpuqual-" + a.outer["nonce"]) / "external"
    )
    r.require(
        Path(completion_pin["path"])
        == parent
        / ("before-execution.json" if phase == "before" else "r3-after.execution.json"),
        "caller-completion-path",
    )
    raw = a.reader.read(completion_pin)
    value = c.parse(raw)
    launch = (
        a.outer["before_launch"]
        if phase == "before"
        else _fixed(a, parent / "capture-inputs/launch.json")[0]
    )
    artifacts = r.bind_outputs(a, phase, launch)
    phase_start = (
        a.preflight_start if phase == "before" else r.clock(value["phase_start"])
    )
    budget = (
        c.before_budget(asdict(phase_start))
        if phase == "before"
        else c.after_budget(asdict(phase_start), anchor)
    )
    expected = {
        "nonce": a.outer["nonce"],
        "boot_id": a.preflight_start.boot_id,
        "outer_intent_sha256": a.approved_intent_sha256,
        "qualified_outer_source_sha256": a.source_sha256,
        "phase_start": asdict(phase_start),
        "budget": budget,
        "enclosing": enclosing,
        "exec_contracts": {
            "guardian": r.guardian_contract_sha256(
                a.outer["python"]["path"],
                a.outer["control_root"],
                a.outer["authorization_path"],
                a.role_source_hashes,
            ),
            "collector": c.collector_contract_sha256(
                a.outer["python"]["path"],
                a.outer["control_root"],
                launch,
                budget["work"]["monotonic_ns"],
            ),
        },
        "artifacts": artifacts,
        "plan_sha256": None if plan_pin is None else plan_pin["sha256"],
        "anchor_sha256": None if anchor is None else r.sha(r.encode(anchor)),
        "before_execution_pin": before_pin,
    }
    if phase == "before":
        r.require(
            artifacts["request"] == a.outer["before_request"]
            and artifacts["registration"] == a.outer["registration"],
            "caller-before-inputs",
        )
    evidence = _evidence(a, raw, phase, parent)
    if phase == "before":
        c.validate_before(raw, expected, asdict(at), evidence=evidence)
    else:
        # Historical producer verification, not a new claim of fresh telemetry.
        assert anchor is not None
        c.validate_after(
            raw, expected, value["published_clock"], evidence=evidence, anchor=anchor
        )
        c.order(value["published_clock"], asdict(at))
    return value


def observe_handoff(a, family, window) -> dict | None:
    root = Path(a.outer["input_root"])
    if not a.reader.exists(root / "dummy-start.ack.json"):
        return None
    start_pin, raw = _fixed(a, root / "dummy-start.json")
    start = r.shape(
        c.parse(raw),
        {
            "format",
            "schema_version",
            "outer_intent_sha256",
            "before_execution_pin",
            "clock",
        },
        "caller-start-fields",
    )
    r.require(
        start["format"] == "strength-freshness-dummy-start-v1"
        and type(start["schema_version"]) is int
        and start["schema_version"] == 1
        and start["outer_intent_sha256"] == a.approved_intent_sha256,
        "caller-start-binding",
    )
    recorded = r.clock(start["clock"])
    r.require(
        Path(c.pin(start["before_execution_pin"])["path"])
        == root / "before-execution.json",
        "caller-before-fixed-path",
    )
    before_raw = a.reader.read(start["before_execution_pin"])
    before = c.parse(before_raw)
    enclosing = before["enclosing"]
    r.require(
        family.root is not None and enclosing["supervisor"] == asdict(family.root),
        "caller-supervisor-identity",
    )
    assert family.root is not None
    op = c.identity(enclosing["operator"])
    r.require(
        op["ppid"] == family.root.pid
        and asdict(a.kernel.identity(op["pid"])) == op
        and a.kernel.identity(family.root.pid) == family.root,
        "caller-live-operator-parentage",
    )
    verify_capture(
        a, "before", start["before_execution_pin"], enclosing=enclosing, at=recorded
    )
    ack_pin, ack_raw = _fixed(a, root / "dummy-start.ack.json")
    r.validate_dummy_start_ack(
        c.parse(ack_raw),
        outer_intent_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
        start_pin=start_pin,
        before_execution_pin=start["before_execution_pin"],
        enclosing=enclosing,
    )
    now = a.kernel.clock()
    r.require(not a.kernel.exited(a.parent.pidfd), "caller-parent-lost")
    new_window = window.handoff(recorded, now)
    r.store_budget(
        a,
        new_window.compact()["gate"]["monotonic_ns"],
        new_window.compact()["gate"]["wall_ns"],
    )
    return {
        "start": start,
        "start_pin": start_pin,
        "ack_pin": ack_pin,
        "enclosing": enclosing,
    }


def verify_candidate(a, raw, family, handoff, window):
    root = Path(a.outer["input_root"])
    candidate_pin, stored = _fixed(a, root / "outer-result.json")
    r.require(stored == raw, "caller-supervisor-output-drift")
    value = r.shape(
        c.parse(raw),
        {
            "format",
            "status",
            "operator_result",
            "operator_family",
            "supervisor_admission",
            "outside_caller_terminal_audit_required",
            "execution_qualified",
        },
        "caller-candidate-fields",
    )
    r.require(
        value["format"] == "strength-freshness-supervisor-result-v1"
        and value["status"] == "candidate_cpu_scope"
        and value["outside_caller_terminal_audit_required"] is True
        and value["execution_qualified"] is False,
        "caller-candidate-scope",
    )
    sup, op = handoff["enclosing"]["supervisor"], handoff["enclosing"]["operator"]
    c._admission(
        value["supervisor_admission"], "supervisor", sup, asdict(a.preflight_start)
    )
    op_spec = o.RoleSpec(
        "session",
        "operator",
        a.outer["python"]["path"],
        a.outer["control_root"],
        a.outer["authorization_path"],
        a.approved_intent_sha256,
        a.preflight_start.monotonic_ns + 118 * o.SECOND,
        b"{}",
        a.role_source_hashes,
    )
    binding = {
        "phase": "session",
        "nonce": a.outer["nonce"],
        "boot_id": a.preflight_start.boot_id,
        "outer_intent_sha256": a.approved_intent_sha256,
        "qualified_outer_source_sha256": a.source_sha256,
        "phase_start": asdict(a.preflight_start),
    }
    c._family(
        value["operator_family"],
        role="supervisor",
        owner=sup,
        child=op,
        exec_hash=o.digest(op_spec.contract()),
        binding=binding,
        budget=o.BudgetWindow(window.start, window.dummy_start).compact(),
    )
    c.order(
        value["operator_family"]["family_closed"]["clock"],
        family["child_terminal"]["clock"],
    )
    result = r.shape(
        value["operator_result"],
        {"format", "status", "before_bundle", "after_bundle", "start_pin", "audit"},
        "caller-operator-result",
    )
    r.require(
        result["format"] == "strength-freshness-operator-result-v1"
        and result["status"] == "passed_cpu_scope"
        and result["start_pin"] == handoff["start_pin"],
        "caller-operator-result-binding",
    )
    encoded_result = r.encode(result)
    r.require(
        value["operator_family"]["output"]["stdout_sha256"] == r.sha(encoded_result)
        and value["operator_family"]["output"]["stdout_bytes"] == len(encoded_result),
        "caller-operator-output-drift",
    )
    r.require(
        result["before_bundle"]["completion_pin"]
        == handoff["start"]["before_execution_pin"],
        "caller-before-result",
    )
    after = result["after_bundle"]
    after_pin = c.pin(after["completion_pin"])
    external = Path("/run") / ("edgeconnect-cpuqual-" + a.outer["nonce"]) / "external"
    a.reader.roots = tuple(set(a.reader.roots) | {external})
    launch_pin, launch_raw = _fixed(a, external / "capture-inputs/launch.json")
    del launch_pin
    launch = c.parse(launch_raw)
    plan_pin, anchor_pin = c.pin(launch["plan"]), c.pin(launch["anchor"])
    r.require(
        Path(plan_pin["path"]) == root / "plan.json"
        and Path(anchor_pin["path"]) == root / "anchor.json",
        "caller-plan-paths",
    )
    plan = c.parse(a.reader.read(plan_pin))
    anchor = c.parse(a.reader.read(anchor_pin))
    r.require(
        plan["nonce"] == a.outer["nonce"]
        and anchor["nonce"] == a.outer["nonce"]
        and anchor["plan_sha256"] == plan_pin["sha256"]
        and anchor["boot_id"] == a.preflight_start.boot_id
        and anchor["started_wall_ns"] == handoff["start"]["clock"]["wall_ns"]
        and anchor["started_monotonic"]
        == handoff["start"]["clock"]["monotonic_ns"] / o.SECOND,
        "caller-anchor-original-start",
    )
    now = a.kernel.clock()
    verify_capture(
        a,
        "after",
        after_pin,
        enclosing=handoff["enclosing"],
        at=now,
        anchor=anchor,
        plan_pin=plan_pin,
        before_pin=handoff["start"]["before_execution_pin"],
    )
    audit = r.final_audit(r.encode(result["audit"]))
    r.require(
        audit["plan_sha256"] == plan_pin["sha256"]
        and audit["anchor_sha256"] == anchor_pin["sha256"],
        "caller-final-audit-pins",
    )
    for p in audit["evidence_pins"].values():
        r.require(
            Path(p["path"]).parent == external.parent / "evidence",
            "caller-audit-proof-scope",
        )
        a.reader.roots = tuple(set(a.reader.roots) | {external.parent / "evidence"})
        a.reader.read(p)
    for p in (handoff["start_pin"], handoff["ack_pin"], a.intent_pin):
        a.reader.read(p)
    recheck_sources(a)
    r.require(window.remaining_ns("gate", a.kernel.clock()) > 0, "caller-audit-late")
    return {
        "candidate": candidate_pin,
        "plan": plan_pin,
        "anchor": anchor_pin,
        "before_execution": handoff["start"]["before_execution_pin"],
        "after_execution": after_pin,
        "start": handoff["start_pin"],
        "ack": handoff["ack_pin"],
        "cpu_audit_evidence": audit["evidence_pins"],
    }


def run_caller(a: r.Admission, *, authorization_path: str) -> dict:
    r.require(
        a.role == "caller" and a.identity == a.kernel.self_identity(),
        "caller-admission-role",
    )
    r.caller_parent(a.kernel, a.identity, a.outer["caller"])
    r.require(
        signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "caller-sigchld-default"
    )
    recheck_sources(a)
    if type(a.kernel) is o.LinuxKernel:
        a.kernel = CallerKernel(a)
    proof = o.initialize_role(a.kernel, "caller", a.identity)
    window = CallerWindow(a.preflight_start)
    r.store_budget(
        a, window.compact()["gate"]["monotonic_ns"], window.compact()["gate"]["wall_ns"]
    )
    # Child alarm is a nonrenewed720s aggregate backstop; the caller enforces
    # the stricter observed preflight or original dummy deadline continuously.
    spec = SupervisorSpec(
        "session",
        "supervisor",
        a.outer["python"]["path"],
        a.outer["control_root"],
        authorization_path,
        a.approved_intent_sha256,
        window.work_ns,
        b"{}",
        a.role_source_hashes,
    )
    family = o.OwnedFamily(a.kernel, proof, window)
    handoff = None

    def update(_old):
        nonlocal handoff
        handoff = observe_handoff(a, family, window)
        return None if handoff is None else r.clock(handoff["start"]["clock"])

    try:
        family.start_capture(spec, r.family_binding(a, window, spec), a.parent)
        result = family.finish(a.parent, handoff_check=update)
    except BaseException:
        if family.spawn is not None:
            family.start_failure = "caller-observation-refused"
            try:
                family.finish(a.parent)
            except BaseException:
                pass
        raise
    r.require(
        result["natural_complete"] is True
        and handoff is not None
        and isinstance(family.budget, CallerWindow),
        "caller-supervisor-incomplete",
    )
    pins = verify_candidate(
        a, bytes(family.output["stdout"]), result, handoff, family.budget
    )
    now = a.kernel.clock()
    r.require(
        family.budget.remaining_ns("gate", now) > 0
        and not a.kernel.children()
        and a.kernel.task_ids() == (a.identity.pid,)
        and not a.kernel.exited(a.parent.pidfd),
        "caller-final-owner-or-deadline",
    )
    report = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "observed-supervisor-family-complete-cpu-scope",
        "outer_intent_sha256": a.approved_intent_sha256,
        "nonce": a.outer["nonce"],
        "caller": asdict(a.identity),
        "caller_admission": proof.raw(),
        "supervisor_family": result,
        "proof_pins": pins,
        "original_budget": family.budget.compact(),
        "audit_clock": asdict(now),
        "caller_future_exit_claimed": False,
        "target_execution_qualified": False,
    }
    a.reader.publish(
        Path(a.outer["input_root"]) / "outside-caller-audit.json", r.encode(report)
    )
    r.require(
        family.budget.remaining_ns("gate", a.kernel.clock()) > 0,
        "caller-publication-late",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    admitted = None
    try:
        admitted = r.load_admission(args.authorization, args.sha256, role="caller")
        result = run_caller(admitted, authorization_path=args.authorization)
        sys.stdout.buffer.write(r.encode(result))
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        r.publish_failure(admitted, "caller")
        sys.stderr.write('{"status":"incomplete","reason":"caller-audit-refused"}\n')
        return 1
    finally:
        if admitted is not None:
            admitted.kernel.close(admitted.parent.pidfd)


if __name__ == "__main__":
    raise SystemExit(main())
