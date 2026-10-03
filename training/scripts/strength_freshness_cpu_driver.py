"""Prospective dummy observer/cleanup/publisher; never touches training authority."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import re
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping

from scripts import strength_freshness_cpu_lifecycle as life
from scripts import strength_freshness_cpu_qualification as q


class EvidenceLog(life.RawLog):
    def record(self, event: str, data: Any) -> dict[str, Any]:
        return super().record(
            event.replace(".", "-"),
            data if isinstance(data, Mapping) else {"value": data},
        )


class TargetIO(q.linux.LinuxIO):
    """Only new primitive is conditional dummy-file removal; caller fences paths."""

    def wall_ns(self) -> int:
        return time.time_ns()

    def unlink_file(
        self, path: Path, expected: Mapping[str, Any], deadline: float
    ) -> None:
        before = path.lstat()
        q.require(
            self.now() < deadline
            and path.resolve() == path
            and stat.S_ISREG(before.st_mode),
            "unlink-file-kind",
        )
        data = self.read(path, 2**20, deadline)
        metadata = self.file_metadata(path, deadline)
        q.require(
            q.sha(data) == expected["sha256"]
            and len(data) == expected["bytes"]
            and metadata == expected["metadata"],
            "unlink-file-changed",
        )
        after = path.lstat()
        q.require(
            (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
            and self.now() < deadline,
            "unlink-file-raced",
        )
        path.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def entries(
    log: life.RawLog, event: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    paths = sorted(log.root.glob(event + "-*.json"))
    q.require(len(paths) <= 4096, "receipt-count")
    result = []
    for path in paths:
        info = path.lstat()
        q.require(
            stat.S_ISREG(info.st_mode) and info.st_size <= 262144, "receipt-file-bound"
        )
        data = path.read_bytes()
        pin = {"path": str(path), "sha256": q.sha(data), "bytes": len(data)}
        result.append((pin, log.load(pin, event)))
    return result


def find_one(log: life.RawLog, event: str) -> dict[str, Any]:
    found = entries(log, event)
    q.require(len(found) == 1, "unique-receipt-" + event)
    return found[0][0]


def role_name(context: q.AuthorizedContext, role: str) -> str:
    names = [n for n, v in context.plan.value["units"].items() if v["role"] == role]
    q.require(len(names) == 1, "unique-role")
    return names[0]


def owner(context: q.AuthorizedContext) -> dict[str, Any]:
    unit = context.unit
    q.require(unit.main is not None, "own-main-required")
    assert unit.main is not None
    return {
        "unit": unit.name,
        "pid": unit.main.pid,
        "invocation_id": unit.invocation_id,
        "start_monotonic_us": int(
            context.io.raw_units[unit.name]["ExecMainStartTimestampMonotonic"]
        ),
        "cgroup": unit.cgroup,
    }


def terminal(
    context: q.AuthorizedContext, name: str, deadline: float
) -> dict[str, Any]:
    unit, _ = context.host.snapshot(name, deadline, partial=True)
    raw: dict[str, Any] = dict(context.io.raw_units[name])
    detail = context.io.command(
        [
            "systemctl",
            "show",
            name,
            "--all",
            "--no-pager",
            "--property=ExecMainCode,ExecMainExitTimestampMonotonic",
        ],
        deadline,
    )
    rows = dict(line.split("=", 1) for line in detail.splitlines() if "=" in line)
    q.require(
        set(rows) == {"ExecMainCode", "ExecMainExitTimestampMonotonic"},
        "actual-exit-code-required",
    )
    raw.update(rows)
    raw.update(
        members=[asdict(x) for x in unit.members],
        observed_boot_id=context.clock().boot_id,
        observed_monotonic=context.clock().monotonic,
    )
    # The parser's empty-cgroup fallback is a path convention, not observed fact.
    # Preserve the actual ControlGroup string; missing required facts fail gates.
    context.log.record("raw-terminal-unit", raw)
    return raw


def poll(context: q.AuthorizedContext, phase: str, predicate):
    end = context.anchor.effective_deadline(phase, context.clock())
    while True:
        q.require(context.clock().monotonic < end, "poll-deadline-" + phase)
        value = predicate(end)
        if value is not None:
            return value
        remaining = end - context.clock().monotonic
        q.require(remaining > 0.2, "poll-reserve-" + phase)
        context.io._backend.sleep(min(0.2, remaining))


def _drain(
    context: q.AuthorizedContext, name: str, deadline: float
) -> q.linux.core.Unit:
    while context.io.now() < deadline:
        unit, _ = context.host.snapshot(name, deadline, partial=True)
        if unit.dead:
            return unit
        context.io._backend.sleep(min(0.2, max(0, deadline - context.io.now())))
    raise q.Refusal("cleanup-cgroup-not-drained")


def cleanup_resources(
    context: q.AuthorizedContext, purpose: str, deadline: float
) -> dict[str, Any]:
    p, io = context.plan.value, context.io
    q.require(
        (purpose == "cleanup-workloads" and io._purpose == "cleanup")
        or (purpose == "retire-cleanup-watchdog" and io._purpose == "finalizer"),
        "cleanup-capability-purpose",
    )
    roles = {"workload"} if purpose == "cleanup-workloads" else {"cleanup", "watchdog"}
    names = [n for n, v in p["units"].items() if v["role"] in roles]
    actual, stop_errors = {}, []
    barrier = None
    dispatcher = (
        role_name(context, "dispatcher") if purpose == "cleanup-workloads" else None
    )

    def request_stop(name: str, *, disable: bool, action_deadline: float) -> None:
        try:
            context.host.snapshot(name, action_deadline, partial=True)
        except Exception as error:
            stop_errors.append(
                {
                    "unit": name,
                    "operation": "observe",
                    "error": type(error).__name__,
                    "reason": str(error)[:256],
                }
            )
            return  # No stop authority is inferred for an unproved resource.
        for verb in ["disable", "stop"] if disable else ["stop"]:
            try:
                # Each action independently rechecks the registered bytes/links.
                # Failed disablement must not hide a still-useful owned stop.
                io.action(verb, name, action_deadline)
            except Exception as error:
                stop_errors.append(
                    {
                        "unit": name,
                        "operation": verb,
                        "error": type(error).__name__,
                        "reason": str(error)[:256],
                    }
                )

    barrier_error = None
    if dispatcher is not None:
        try:
            barrier_end = min(
                deadline,
                context.anchor.effective_deadline(
                    "dispatcher_terminal", context.clock()
                ),
            )
            request_stop(dispatcher, disable=False, action_deadline=barrier_end)
            _drain(context, dispatcher, barrier_end)
            barrier = {
                "started": find_one(context.log, "dispatcher-started"),
                "unit": terminal(context, dispatcher, barrier_end),
                "clock": asdict(context.clock()),
            }
            context.log.record("raw-dispatcher-cleanup-barrier", barrier)
            life.checked_dispatcher_barrier(context.log, barrier)
        except Exception as error:
            barrier_error = error
            stop_errors.append(
                {
                    "unit": dispatcher,
                    "operation": "drain-proof",
                    "error": type(error).__name__,
                    "reason": str(error)[:256],
                }
            )
    # The dispatcher proof is attempted first within405. Even if it fails,
    # independently known workloads still receive bounded stop requests within
    # the unchanged cleanup reserve. No deletion occurs on an unproved barrier.
    for name in names:
        request_stop(name, disable=True, action_deadline=deadline)
    if barrier_error is not None:
        context.log.record(
            "cleanup-stop-errors", {"errors": stop_errors, "deletion_admitted": False}
        )
        raise q.Refusal("dispatcher-drain-unproved-no-deletion") from barrier_error
    context.log.record(
        "cleanup-stop-errors",
        {"errors": stop_errors, "deletion_admitted": not stop_errors},
    )
    q.require(not stop_errors, "owned-stop-incomplete-no-deletion")
    for name in names:
        unit = _drain(context, name, deadline)
        raw: dict[str, Any] = dict(io.raw_units[name])
        raw["members"] = []
        actual[name] = raw
        q.require(io.boot_links(name, deadline) == {}, "cleanup-link-remains")
        # Retain truthful failure terminal state, then reset only this fully dead
        # owned dummy unit so systemd can forget its removed definition.
        if unit.active == "failed":
            io.reset_failed(name, deadline)
            unit = _drain(context, name, deadline)
            actual[name] = {**io.raw_units[name], "members": []}
        q.require(
            unit.active == "inactive" and unit.substate == "dead",
            "cleanup-terminal-state",
        )
    file_names = set()
    for name in names:
        spec = p["units"][name]
        file_names.add(spec["installed_path"])
        for side in ("before", "after"):
            file_names.update(x["path"] for x in spec[side]["environment_files"])
    for path in sorted(file_names):
        io.remove_owned_file(Path(path), names, deadline)
    io.command(["systemctl", "daemon-reload"], deadline)
    jobs = context.host._jobs(deadline)
    leftovers, members, links = [], [], []
    for name in names:
        members.extend(asdict(x) for x in io.members("/system.slice/" + name, deadline))
        links.extend(io.boot_links(name, deadline))
    for path in file_names:
        if io.exists(Path(path)):
            leftovers.append(path)
    relevant_jobs = [v for v in jobs.values() if v["unit"] in names]
    facts = {
        "units": actual,
        "jobs": relevant_jobs,
        "members": members,
        "links": links,
        "leftovers": leftovers,
    }
    if barrier is not None:
        facts["dispatcher_barrier"] = barrier
    context.log.record("cleanup-inventory", facts)
    return facts


def monotonic_microseconds(value: str) -> int:
    """systemd255 format_timespan units, exact integral microseconds only.

    Constants: https://raw.githubusercontent.com/systemd/systemd/v255/src/basic/time-util.h
    Nonzero bare numbers are not interpreted ambiguously as seconds/microseconds.
    """
    if value == "0":
        return 0
    factors = {
        "y": 31557600,
        "month": 2629800,
        "w": 604800,
        "d": 86400,
        "h": 3600,
        "min": 60,
        "s": 1,
        "ms": Decimal(".001"),
        "us": Decimal(".000001"),
    }
    tokens = re.findall(r"([0-9]+(?:\.[0-9]+)?)(month|min|ms|us|y|w|d|h|s)", value)
    q.require(
        tokens and " ".join(n + u for n, u in tokens) == value,
        "systemd-monotonic-duration",
    )
    units = list(factors)
    ranks = [units.index(u) for _, u in tokens]
    q.require(ranks == sorted(set(ranks)), "systemd-duration-unit-order")
    microseconds = sum(
        (Decimal(n) * Decimal(factors[u]) * 1000000 for n, u in tokens), Decimal(0)
    )
    q.require(
        microseconds == microseconds.to_integral_value() and 0 < microseconds < 2**63,
        "systemd-microsecond-precision",
    )
    return int(microseconds)


def arm_receipt(
    plan: q.Plan, anchor: life.Anchor, backend: TargetIO, log: EvidenceLog
) -> dict[str, Any]:
    """Read-only admission after external reviewed setup armed the fixed timer."""
    io = q.ClosedIO(plan, anchor, backend, log, purpose="read")
    host = q.DummyReadHost(io)
    p = plan.value
    watchdog = next(n for n, v in p["units"].items() if v["role"] == "watchdog")
    cleanup = next(n for n, v in p["units"].items() if v["role"] == "cleanup")
    end = min(anchor.effective_deadline("work", io.clock()), io.now() + 10)
    q.verify_sources(plan, backend, log, end)
    timer, _ = host.snapshot(watchdog, end)
    cleanup_unit, _ = host.snapshot(cleanup, end)
    detail = io.command(
        [
            "systemctl",
            "show",
            watchdog,
            "--all",
            "--no-pager",
            "--property=NextElapseUSecMonotonic,AccuracyUSec,RandomizedDelayUSec,Unit,LastTriggerUSecMonotonic",
        ],
        end,
    )
    rows = dict(line.split("=", 1) for line in detail.splitlines() if "=" in line)
    timeout = io.command(
        [
            "systemctl",
            "show",
            cleanup,
            "--all",
            "--no-pager",
            "--property=TimeoutStartUSec,TimeoutStopUSec",
        ],
        end,
    )
    limits = dict(line.split("=", 1) for line in timeout.splitlines() if "=" in line)
    q.require(
        set(rows)
        == {
            "NextElapseUSecMonotonic",
            "AccuracyUSec",
            "RandomizedDelayUSec",
            "Unit",
            "LastTriggerUSecMonotonic",
        }
        and set(limits) == {"TimeoutStartUSec", "TimeoutStopUSec"},
        "timer-detail-required",
    )
    # Real manager/slack facts are separately pinned in the reviewed setup input.
    facts_path = Path(p["input_root"]) / "timer-manager-facts.json"
    pin = next((x for x in p["source_pins"] if x["path"] == str(facts_path)), None)
    q.require(pin is not None, "timer-manager-facts-unpinned")
    assert pin is not None
    backend.pin(pin, end)
    manager = io.json(facts_path, end)
    q.require(
        manager["boot_id"] == p["boot_id"]
        and 0 <= io.now() - manager["observed_monotonic"] <= 120,
        "fresh-manager-facts",
    )
    next_us = monotonic_microseconds(rows["NextElapseUSecMonotonic"])
    q.require(next_us > 0, "monotonic-timer-deadline-required")
    facts = {
        "active_waiting": timer.active == "active"
        and timer.substate == "waiting"
        and timer.job is None,
        "independent_cleanup": cleanup_unit.dead and rows["Unit"] == cleanup,
        "ownership_validated": True,
        "next_trigger_monotonic": next_us / 1e6,
        "accuracy_seconds": q.linux.systemd_seconds(rows["AccuracyUSec"]),
        "randomized_delay_seconds": 0
        if rows["RandomizedDelayUSec"] == "0"
        else q.linux.systemd_seconds(rows["RandomizedDelayUSec"]),
        "manager_slack_seconds": manager["timer_slack_ns"] / 1e9,
        "observed_dispatch_slack_seconds": manager["observed_dispatch_slack_seconds"],
        "cleanup_timeout_start_seconds": q.linux.systemd_seconds(
            limits["TimeoutStartUSec"]
        ),
        "cleanup_timeout_stop_seconds": q.linux.systemd_seconds(
            limits["TimeoutStopUSec"]
        ),
    }
    return life.acknowledge_arm(log, io.clock(), facts)


def verify_arm_health(io: q.ClosedIO) -> None:
    """The historical arm receipt alone never authorizes later workload work."""
    q.require(
        isinstance(io.facts, life.RawLog) and io._arm is not None, "bound-arm-log"
    )
    assert isinstance(io.facts, life.RawLog)
    assert io._arm is not None
    life.require_work(io.facts, io.clock(), io._arm)
    p = io.plan.value
    name = next(n for n, v in p["units"].items() if v["role"] == "watchdog")
    cleanup = next(n for n, v in p["units"].items() if v["role"] == "cleanup")
    end = min(io.anchor.deadline("work"), io.now() + 5)
    host = q.DummyReadHost(io)
    unit, _ = host.snapshot(name, end)
    standby, _ = host.snapshot(cleanup, end)
    q.require(
        unit.active == "active"
        and unit.substate == "waiting"
        and unit.job is None
        and standby.dead,
        "watchdog-no-longer-waiting",
    )
    raw = io.command(
        [
            "systemctl",
            "show",
            name,
            "--all",
            "--no-pager",
            "--property=NextElapseUSecMonotonic,AccuracyUSec,RandomizedDelayUSec,Unit,LastTriggerUSecMonotonic",
        ],
        end,
    )
    values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
    actual = monotonic_microseconds(values["NextElapseUSecMonotonic"]) / 1e6
    q.require(
        values["Unit"] == cleanup and abs(actual - io.anchor.deadline("work")) <= 0.001,
        "watchdog-original-deadline-drift",
    )


def _payload_log(context: q.AuthorizedContext, name: str) -> EvidenceLog:
    return EvidenceLog(
        Path(context.plan.value["units"][name]["payload"]["output_dir"]), context.anchor
    )


def _start(context: q.AuthorizedContext, name: str):
    end = context.anchor.effective_deadline("work", context.clock())
    before, _ = context.host.snapshot(name, end)
    q.require(before.dead, "payload-not-initially-drained")
    context.io.action("start", name, end)
    return end


def _payload_result(
    context: q.AuthorizedContext, name: str, event: str, prior: set[str] | None = None
):
    log = _payload_log(context, name)
    prior = prior or set()

    def ready(deadline):
        found = [
            (pin, data)
            for pin, data in entries(log, event)
            if pin["sha256"] not in prior
        ]
        q.require(len(found) <= 1, "ambiguous-payload-event")
        return found[0] if found else None

    return poll(context, "work", ready)


def case_observation(
    context: q.AuthorizedContext, case: str, details: Mapping[str, Any], *, scope: str
) -> dict[str, Any]:
    pin = context.log.record(
        "case-" + case,
        {
            "case": case,
            "details": dict(details),
            "scope": scope,
            "clock": asdict(context.clock()),
        },
    )
    return {"status": "passed", "evidence": pin, "scope": scope}


def run_cases(context: q.AuthorizedContext) -> dict[str, Any]:
    """Fixed case sequence; actual systemd facts cannot be supplied by a callback."""
    q.require(
        context.authorization["role"] == "dispatcher"
        and context.io._purpose == "dispatcher",
        "case-dispatcher-only",
    )
    p, io, cases = context.plan.value, context.io, {}
    b = p["bindings"]
    runner = q.CaseRunner(io)
    end = context.anchor.effective_deadline("work", context.clock())
    details = runner.typed_properties([b["holder"], b["contender"], b["timer"]], end)
    cases["typed-properties"] = case_observation(
        context,
        "typed-properties",
        details,
        scope="actual typed LinuxIO/systemd properties",
    )
    _start(context, b["holder"])
    holder_lock = _payload_result(context, b["holder"], "payload-lock")
    q.require(holder_lock[1]["acquired"] is True, "holder-did-not-lock")
    ownership = runner.pid_cgroup(b["holder"], end)
    _start(context, b["contender"])
    contender_lock = _payload_result(context, b["contender"], "payload-lock")
    q.require(
        contender_lock[1]["acquired"] is False
        and all(
            contender_lock[1][key] == holder_lock[1][key] for key in ("device", "inode")
        ),
        "real-lock-conflict",
    )
    cases["pid-cgroup-lease"] = case_observation(
        context,
        "pid-cgroup-lease",
        {
            "ownership": ownership,
            "holder": holder_lock[0],
            "contender": contender_lock[0],
        },
        scope="actual PIDs/cgroup/births and real flock conflict; no forced PID reuse",
    )
    # No unit poll occurred between dispatch and the child's own finished receipt.
    result_pin, result = _payload_result(context, b["contender"], "payload-result")
    unit, _ = context.host.snapshot(b["contender"], end)
    q.require(unit.dead, "fast-case-not-terminal-at-first-parent-poll")
    started_pin, started = entries(
        _payload_log(context, b["contender"]), "payload-started"
    )[-1]
    raw = terminal(context, b["contender"], end)
    birth = started["owner"]["start_ticks"] / os.sysconf("SC_CLK_TCK")
    actual_start = int(raw["ExecMainStartTimestampMonotonic"]) / 1e6
    q.require(
        unit.dead
        and raw["Result"] == "success"
        and str(raw["ExecMainStatus"]) == "0"
        and str(raw["ExecMainCode"]) == "1"
        and int(raw["ExecMainPID"]) == started["owner"]["pid"]
        and raw["InvocationID"] in ("", started["owner"]["invocation_id"])
        and birth <= actual_start <= started["clock"]["monotonic"]
        and actual_start - birth <= 1.0,
        "fast-natural-terminal-join",
    )
    cases["fast-natural-child-exit"] = case_observation(
        context,
        "fast-natural-child-exit",
        {"started": started_pin, "result": result_pin, "terminal": raw},
        scope="actual naturally finished child before first unit poll; retained systemd evidence required",
    )
    graceful = runner.bounded_drain(b["holder"], end, expected_failure=False)
    _start(context, b["stubborn"])
    _payload_result(context, b["stubborn"], "payload-descendant")
    forced = runner.bounded_drain(b["stubborn"], end, expected_failure=True)
    cases["bounded-drain"] = case_observation(
        context,
        "bounded-drain",
        {"graceful": graceful, "forced": forced},
        scope="actual graceful stop and separately intentional timeout/escalation",
    )
    _start(context, b["barrier"])
    _start(context, b["dependent"])
    pending = runner.cancel_pending(
        b["dependent"],
        b["barrier"],
        Path(p["units"][b["dependent"]]["payload"]["output_dir"]),
        end,
    )
    cases["pending-start-job"] = case_observation(
        context,
        "pending-start-job",
        pending,
        scope="actual queued dependency start cancelled before barrier release",
    )
    links = runner.persistent_boot_edges(b["timer"], end)
    _start(context, b["timer"])
    _payload_result(context, b["dependent"], "payload-result")
    io.action("stop", b["timer"], end)
    cases["persistent-boot-edges"] = case_observation(
        context,
        "persistent-boot-edges",
        links,
        scope="actual persistent target graph and timer; no real reboot",
    )
    from scripts.strength_freshness_cpu_support_case import verify_support_case

    shared = runner.wait_drained(b["barrier"], end)
    context.log.record("shared-barrier-support-drained", shared)
    support_result = verify_support_case(context)
    cases["support-partial-transaction"] = case_observation(
        context,
        "support-partial-transaction",
        support_result,
        scope="actual dummy writes/reload/new Invocation/lease; synthetic authority only",
    )
    # Committed isolated fixture checks are intentionally not called a real DR
    # transaction. Their child runner has its own fixed test-node allowlist.
    fixture_results = run_fixture_cases(context)
    for key in ("proof-freeze-and-retention", "execution-gate-refusal"):
        cases[key] = case_observation(
            context,
            key,
            fixture_results[key],
            scope="pinned target-interpreter isolated fixtures; no live backup/production execution",
        )
    # Sacrificial main only; observer/publisher/watchdog remain inaccessible.
    holder = restart_holder_for_sacrifice(context)
    io.kill_sacrificial(b["holder"], holder, end)
    cases["cleanup-and-retirement"] = {
        "status": "pending-independent-cleanup",
        "killed": asdict(holder),
    }
    return cases


def restart_holder_for_sacrifice(context: q.AuthorizedContext) -> q.linux.core.Unit:
    """Wait for a new receipt pair and join it to a real post-start unit owner."""
    name = context.plan.value["bindings"]["holder"]
    log = _payload_log(context, name)
    prior_started = entries(log, "payload-started")
    prior_locks = entries(log, "payload-lock")
    old_invocations = {data["owner"]["invocation_id"] for _, data in prior_started}
    began = context.clock().monotonic
    _start(context, name)
    started_pin, started = _payload_result(
        context, name, "payload-started", {pin["sha256"] for pin, _ in prior_started}
    )
    lock_pin, locked = _payload_result(
        context, name, "payload-lock", {pin["sha256"] for pin, _ in prior_locks}
    )
    expected = started["owner"]
    context.log.record(
        "raw-fresh-sacrificial-owner",
        {
            "started": started_pin,
            "lock": lock_pin,
            "prior_started": [pin for pin, _ in prior_started],
            "prior_locks": [pin for pin, _ in prior_locks],
        },
    )
    q.require(
        expected["unit"] == name
        and expected["invocation_id"] not in old_invocations
        and started["clock"]["monotonic"] >= began
        and locked["acquired"] is True
        and locked["pid"] == expected["pid"],
        "fresh-sacrificial-receipt-binding",
    )

    def ready(deadline):
        unit, _ = context.host.snapshot(name, deadline)
        if unit.main is None and (
            unit.active == "activating"
            or (unit.job and unit.job.get("kind") == "start")
        ):
            return None
        q.require(
            unit.main == q.linux.core.Process(expected["pid"], expected["start_ticks"])
            and unit.main in unit.members
            and unit.invocation_id == expected["invocation_id"]
            and unit.cgroup == expected["cgroup"],
            "fresh-sacrificial-actual-owner",
        )
        if unit.active == "activating" or (
            unit.job and unit.job.get("kind") == "start"
        ):
            return None
        q.require(
            unit.active == "active" and unit.job is None, "fresh-sacrificial-active"
        )
        return unit

    return poll(context, "work", ready)


def run_fixture_cases(context: q.AuthorizedContext) -> dict[str, Any]:
    from scripts.strength_freshness_cpu_fixture import run_bound_fixtures

    return run_bound_fixtures(context)


def run_role(context: q.AuthorizedContext) -> None:
    role = context.authorization["role"]
    q.require(role in {"dispatcher", "cleanup", "observer", "publisher"}, "driver-role")
    started = life.role_started(context.log, context.clock(), role, owner(context))
    try:
        if role == "dispatcher":
            cases = run_cases(context)
            life.dispatcher_result(
                context.log, context.clock(), started, cases, set(q.CASES)
            )
        elif role == "cleanup":
            expected = {
                n
                for n, v in context.plan.value["units"].items()
                if v["role"] == "workload"
            }
            life.run_cleanup(
                context.log,
                context.clock,
                expected,
                lambda purpose, deadline: cleanup_resources(context, purpose, deadline),
            )
        elif role == "observer":

            def dispatched(deadline):
                starts = entries(context.log, "dispatcher-started")
                results = entries(context.log, "dispatcher-result")
                q.require(
                    len(starts) <= 1 and len(results) <= 1, "ambiguous-dispatcher-proof"
                )
                if not starts or not results:
                    return None
                raw = terminal(context, role_name(context, "dispatcher"), deadline)
                if (
                    raw["members"]
                    or raw["MainPID"] != "0"
                    or raw["Job"] not in ("", "0", 0)
                ):
                    return None
                return life.dispatcher_complete(
                    context.log,
                    context.clock(),
                    starts[0][0],
                    results[0][0],
                    raw,
                    set(q.CASES),
                )

            completed = poll(context, "dispatcher_terminal", dispatched)
            cleanup = poll(
                context,
                "observer",
                lambda _: (
                    find_one(context.log, "cleanup-complete")
                    if entries(context.log, "cleanup-complete")
                    else None
                ),
            )
            after = load_preservation(context)
            life.observer_candidate(
                context.log, context.clock(), cleanup, completed, set(q.CASES), after
            )
        else:
            candidate = poll(
                context,
                "publisher",
                lambda _: (
                    find_one(context.log, "observer-candidate")
                    if entries(context.log, "observer-candidate")
                    else None
                ),
            )
            observer_started, cleanup_started = (
                find_one(context.log, "observer-started"),
                find_one(context.log, "cleanup-started"),
            )

            def exited(deadline):
                obs = terminal(context, role_name(context, "observer"), deadline)
                cleanup = terminal(context, role_name(context, "cleanup"), deadline)
                if (
                    obs["members"]
                    or cleanup["members"]
                    or obs["MainPID"] != "0"
                    or cleanup["MainPID"] != "0"
                ):
                    return None
                return obs, cleanup

            observer_terminal, cleanup_terminal = poll(context, "publisher", exited)
            context.io.acknowledge_retirement(
                candidate,
                observer_started,
                observer_terminal,
                cleanup_started,
                cleanup_terminal,
            )
            life.publish(
                context.log,
                context.clock,
                candidate,
                observer_started,
                observer_terminal,
                cleanup_started,
                cleanup_terminal,
                lambda purpose, deadline: cleanup_resources(context, purpose, deadline),
            )
    except BaseException as error:
        context.log.record(
            "role-failed",
            {"role": role, "type": type(error).__name__, "reason": str(error)[:512]},
        )
        raise


def load_preservation(context: q.AuthorizedContext) -> dict[str, Any]:
    from scripts.strength_freshness_cpu_preservation import verify_preservation

    p = context.plan.value["preservation"]

    def ready(deadline):
        path = Path(p["after_path"])
        return path if context.io.exists(path) else None

    path = poll(context, "observer", ready)
    end = context.anchor.effective_deadline("observer", context.clock())
    for key in ("policy", "before"):
        context.io.pin(p[key], end)
    policy = context.io.json(Path(p["policy"]["path"]), end)
    before = context.io.json(Path(p["before"]["path"]), end)
    metadata = context.io.file_metadata(path, end)
    q.require(
        metadata == {"uid": 0, "gid": 0, "mode": 0o444}, "external-r3-facts-protection"
    )
    envelope = context.io.json(path, end)
    q.exact(
        envelope,
        {"format", "plan_sha256", "anchor_sha256", "capture"},
        "external-capture-envelope",
    )
    q.require(
        envelope["format"] == "strength-freshness-cpu-r3-after-v1"
        and envelope["plan_sha256"] == context.plan.checksum
        and envelope["anchor_sha256"] == life.digest(context.anchor.as_dict()),
        "external-capture-attempt",
    )
    after = envelope["capture"]
    context.log.record(
        "raw-r3-preservation-inputs",
        {"policy": policy, "before": before, "after": after},
    )
    result = verify_preservation(
        policy,
        before,
        after,
        verified_champions=p["verified_champions"],
        attempt=context.anchor.as_dict(),
        cleanup_clock=context.log.load(
            find_one(context.log, "cleanup-complete"), "cleanup-complete"
        )["clock"],
        audit_clock=asdict(context.clock()),
    )
    context.log.record("r3-preservation", result)
    return result


def external_audit(context: q.AuthorizedContext) -> dict[str, Any]:
    """Final read-only closure, including the three explicitly retained definitions."""
    q.require(
        context.io._purpose == "read"
        and context.authorization["role"] == "external-read-only-auditor",
        "external-audit-read-only",
    )
    end = context.anchor.effective_deadline("audit", context.clock())
    p = context.plan.value
    retained = {}
    for role in ("dispatcher", "observer", "publisher"):
        name = role_name(context, role)
        unit, _ = context.host.snapshot(name, end)
        q.require(
            unit.dead
            and not unit.enabled
            and not p["units"][name]["boot_links"]
            and context.io.boot_links(name, end) == {},
            "inert-control-not-retired",
        )
        path = Path(p["units"][name]["installed_path"])
        context.io._known_file(path, end)
        data = context.io.read(path, 2**20, end)
        retained[name] = {
            "path": str(path),
            "sha256": q.sha(data),
            "bytes": len(data),
            "unit": asdict(unit),
        }
    retired = [
        n
        for n, v in p["units"].items()
        if v["role"] in {"workload", "cleanup", "watchdog"}
    ]
    jobs = context.host._jobs(end)
    q.require(
        not any(v["unit"] in p["units"] for v in jobs.values()), "final-dummy-jobs"
    )
    for name in retired:
        q.require(
            not context.io.members("/system.slice/" + name, end)
            and context.io.boot_links(name, end) == {},
            "final-dummy-owner-or-link",
        )
        spec = p["units"][name]
        paths = {spec["installed_path"]} | {
            e["path"]
            for side in ("before", "after")
            for e in spec[side]["environment_files"]
        }
        q.require(
            not any(context.io.exists(Path(path)) for path in paths),
            "final-installed-leftover",
        )
    context.log.record(
        "inert-control-input-inventory",
        {"retained": retained, "remaining_jobs": [], "remaining_retired_members": []},
    )
    published = find_one(context.log, "published-candidate")
    started = find_one(context.log, "publisher-started")
    observed = terminal(context, role_name(context, "publisher"), end)
    return life.audit(context.log, context.clock(), published, started, observed)
