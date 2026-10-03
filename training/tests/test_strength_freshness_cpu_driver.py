"""Prospective driver tests: no actual systemd, /proc, kernel lease, or GPU work."""

from dataclasses import asdict
import os
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import strength_freshness_cpu_driver as d
from scripts import strength_freshness_cpu_lifecycle as life
from scripts import strength_freshness_cpu_qualification as q
from test_strength_freshness_cpu_qualification import (
    Backend,
    BOOT,
    NONCE,
    PREFIX,
    make_plan,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("390s", 390000000),
        ("6min 30s", 390000000),
        ("1month 2w 2d 21h 59min 6.137464s", 4091346137464),
        ("2ms 3us", 2003),
    ],
)
def test_exact_systemd255_duration_parser(text, expected):
    assert d.monotonic_microseconds(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "390000000",
        "infinity",
        "1min 1h",
        "1s 1s",
        "-1s",
        "NaNs",
        "1.0000001s",
        "1month2w",
        "0.0000001us",
        "1s extra",
    ],
)
def test_ambiguous_nonfinite_or_unsupported_duration_refuses(text):
    with pytest.raises(q.Refusal):
        d.monotonic_microseconds(text)


def context(tmp_path, purpose="observer"):
    p, data = make_plan()
    plan = q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    anchor = life.Anchor("case-one", NONCE, plan.checksum, BOOT, 100, 10**18)
    backend = Backend(data)
    log = d.EvidenceLog(tmp_path / "evidence", anchor)
    io = q.ClosedIO(plan, anchor, backend, log, purpose=purpose)
    return SimpleNamespace(
        plan=plan,
        anchor=anchor,
        io=io,
        log=log,
        clock=io.clock,
        host=q.DummyReadHost(io),
        authorization={"role": purpose},
        unit=None,
    )


def test_real_rawlog_keeps_scalar_outputs_and_normalizes_only_event_name(tmp_path):
    c = context(tmp_path)
    pin = c.log.record("command.raw", "stdout\n")
    assert c.log.load(pin, "command-raw") == {"value": "stdout\n"}
    assert d.find_one(c.log, "command-raw") == pin
    with pytest.raises(life.Refusal):
        c.log.load(pin, "cleanup-complete")


def test_duplicate_role_receipt_is_not_chosen_arbitrarily(tmp_path):
    c = context(tmp_path)
    c.log.record("observer-candidate", {})
    c.log.record("observer-candidate", {})
    with pytest.raises(q.Refusal, match="unique-receipt"):
        d.find_one(c.log, "observer-candidate")


def test_observer_can_read_after_work_cutoff_but_never_mutate(tmp_path):
    c = context(tmp_path)
    c.io._backend.time = 491
    c.io._backend.response = "multi-user.target\n"
    assert c.io.command(["systemctl", "get-default"], 600) == "multi-user.target\n"
    with pytest.raises(q.Refusal, match="read-only-capability"):
        c.io.action("stop", PREFIX + "-worker.service", 600)
    assert not c.io._backend.actions


def test_missing_or_forged_arm_receipt_cannot_admit_work(tmp_path):
    c = context(tmp_path)
    with pytest.raises((life.Refusal, FileNotFoundError, KeyError)):
        c.io.acknowledge_arm(
            {
                "purpose": "workload-armed",
                "plan_sha256": c.plan.checksum,
                "anchor": c.anchor.as_dict(),
            }
        )
    assert c.io._arm is None


def test_arm_receipt_from_other_attempt_is_rejected(tmp_path):
    c = context(tmp_path)
    other = life.Anchor("other", NONCE, "2" * 64, BOOT, 100, 10**18)
    log = d.EvidenceLog(tmp_path / "other", other)
    pin = log.record("cleanup-armed", {"clock": asdict(c.clock())})
    with pytest.raises(life.Refusal, match="receipt-location"):
        c.io.acknowledge_arm(pin)


def test_old_arm_does_not_bypass_current_watchdog_health(tmp_path, monkeypatch):
    c = context(tmp_path, purpose="dispatcher")
    from test_strength_freshness_cpu_lifecycle import arm_facts

    pin = life.acknowledge_arm(c.log, c.clock(), arm_facts(c.anchor))
    c.io.acknowledge_arm(pin)
    monkeypatch.setattr(
        q.DummyReadHost,
        "snapshot",
        lambda self, name, end: (
            q.linux.core.Unit(name, "a" * 64, "/system.slice/" + name),
            {},
        ),
    )
    with pytest.raises(q.Refusal, match="watchdog-no-longer"):
        c.io.action("start", PREFIX + "-worker.service", 120)
    assert not c.io._backend.actions


def test_shared_directory_is_for_fixed_control_roles_only():
    p, data = make_plan()
    plan = q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    name = p["bindings"]["holder"]
    raw = data[p["units"][name]["installed_path"]]
    unsafe = raw.replace(
        ("ReadWritePaths=" + p["scratch_root"]).encode(),
        ("ReadWritePaths=" + p["scratch_root"] + " /etc/systemd/system").encode(),
    )
    with pytest.raises(q.Refusal, match="write-sandbox"):
        q.validate_unit_text(plan, name, unsafe)
    dispatcher = next(n for n, u in p["units"].items() if u["role"] == "dispatcher")
    raw = data[p["units"][dispatcher]["installed_path"]]
    with pytest.raises(q.Refusal, match="write-sandbox"):
        q.validate_unit_text(
            plan, dispatcher, raw.replace(b" /etc/systemd/system", b"")
        )
    observer = next(n for n, u in p["units"].items() if u["role"] == "observer")
    raw = data[p["units"][observer]["installed_path"]]
    with pytest.raises(q.Refusal, match="write-sandbox"):
        q.validate_unit_text(
            plan,
            observer,
            raw.replace(
                ("ReadWritePaths=" + p["scratch_root"]).encode(),
                (
                    "ReadWritePaths=" + p["scratch_root"] + " /etc/systemd/system"
                ).encode(),
            ),
        )


def test_workload_cannot_indirectly_start_observer_via_requires():
    p, data = make_plan()
    plan = q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    name = p["bindings"]["holder"]
    observer = next(n for n, u in p["units"].items() if u["role"] == "observer")
    raw = data[p["units"][name]["installed_path"]].replace(
        b"[Unit]\n", b"[Unit]\nRequires=" + observer.encode() + b"\n"
    )
    with pytest.raises(q.Refusal, match="dependency-escape"):
        q.validate_unit_text(plan, name, raw)


def test_cleanup_log_failure_is_incomplete_not_silent_success(tmp_path):
    c = context(tmp_path, purpose="cleanup")

    def broken(*args, **kwargs):
        raise OSError("evidence-full")

    c.log.record = broken
    with pytest.raises(OSError, match="evidence-full"):
        c.io.action("stop", PREFIX + "-worker.service", 120)
    assert not c.io._backend.actions
    # Fail-closed evidence is not a claim that arbitrary storage failure cleans
    # resources. Independent service bounds are the remaining process backstop.
    raw = c.io._backend.data[
        c.plan.value["units"][PREFIX + "-worker.service"]["installed_path"]
    ]
    assert b"RuntimeMaxSec=30s" in raw and b"TimeoutStopSec=5s" in raw


def test_natural_owner_uses_exact_observed_systemd_start(tmp_path):
    c = context(tmp_path)
    name = next(n for n, u in c.plan.value["units"].items() if u["role"] == "observer")
    c.unit = q.linux.core.Unit(
        name,
        "a" * 64,
        "/system.slice/" + name,
        "3" * 32,
        q.linux.core.Process(42, 9),
        entered_monotonic=100.123456,
    )
    c.io.raw_units[name] = {"ExecMainStartTimestampMonotonic": "100123456"}
    assert d.owner(cast(q.AuthorizedContext, c))["start_monotonic_us"] == 100123456


def test_prior_pid_registration_never_confers_signal_authority(tmp_path):
    c = context(tmp_path, purpose="dispatcher")
    name = c.plan.value["bindings"]["support_guard"]
    c.io.remember_prior_process(name, q.linux.core.Process(42, 100))
    c.io._backend.processes[42] = {
        "pid": 42,
        "start_ticks": 200,
        "cgroup": "0::/foreign",
        "ppid": 1,
    }
    assert c.io.process(42, 120)["start_ticks"] == 200
    with pytest.raises(q.Refusal, match="sacrificial-fixed-unit"):
        c.io.kill_sacrificial(name, q.linux.core.Unit(name, "a" * 64, "/foreign"), 120)


def test_retained_control_units_have_no_boot_edges():
    p, _ = make_plan()
    observer = next(n for n, u in p["units"].items() if u["role"] == "observer")
    p["units"][observer]["boot_links"] = {
        "/etc/systemd/system/multi-user.target.wants/" + observer: p["units"][observer][
            "installed_path"
        ]
    }
    with pytest.raises(q.Refusal, match="inert-control"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


@pytest.mark.parametrize(
    "purpose,operation",
    [
        ("observer", "cleanup-workloads"),
        ("workload", "retire-cleanup-watchdog"),
        ("cleanup", "retire-cleanup-watchdog"),
        ("finalizer", "cleanup-workloads"),
    ],
)
def test_cleanup_role_confusion_refuses_before_any_host_call(
    tmp_path, purpose, operation
):
    c = context(tmp_path, purpose=purpose)
    with pytest.raises(q.Refusal, match="cleanup-capability-purpose"):
        d.cleanup_resources(cast(q.AuthorizedContext, c), operation, 120)
    assert c.io._backend.actions == []
    assert list(c.log.root.glob("*.json")) == []


def cleanup_fixture(
    tmp_path,
    *,
    remaining_member=False,
    remaining_link=False,
    dispatcher_alive=False,
    dispatcher_foreign=False,
):
    c = context(tmp_path, purpose="cleanup")
    p = c.plan.value
    names = [n for n, spec in p["units"].items() if spec["role"] == "workload"]
    dispatcher = next(n for n, v in p["units"].items() if v["role"] == "dispatcher")
    life.role_started(
        c.log,
        c.clock(),
        "dispatcher",
        {
            "unit": dispatcher,
            "pid": os.getpid(),
            "invocation_id": "4" * 32,
            "start_monotonic_us": 100000000,
            "cgroup": "/system.slice/" + dispatcher,
        },
    )
    actions, removed = [], []
    files = {spec["installed_path"] for spec in p["units"].values()}
    files.update(
        e["path"]
        for spec in p["units"].values()
        for side in ("before", "after")
        for e in spec[side]["environment_files"]
    )
    backend = SimpleNamespace(time=101.0)
    backend.sleep = lambda seconds: setattr(backend, "time", backend.time + seconds)
    units = {
        n: q.linux.core.Unit(
            n,
            "a" * 64,
            "/system.slice/" + n,
            members=(q.linux.core.Process(77, 9),)
            if remaining_member and n == names[0]
            else (),
        )
        for n in names + [dispatcher]
    }
    if dispatcher_alive:
        units[dispatcher] = q.linux.core.Unit(
            dispatcher,
            "a" * 64,
            "/system.slice/" + dispatcher,
            "4" * 32,
            q.linux.core.Process(os.getpid(), 9),
            (q.linux.core.Process(os.getpid(), 9),),
            active="active",
            substate="running",
        )
    io = SimpleNamespace(
        _purpose="cleanup",
        _backend=backend,
        now=lambda: backend.time,
        raw_units={
            n: {
                "Id": n,
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": "0",
                "Job": "0",
                "ControlGroup": "/system.slice/" + n,
                "InvocationID": "5" * 32
                if dispatcher_foreign and n == dispatcher
                else "4" * 32,
                "ExecMainPID": str(os.getpid()),
                "ExecMainStartTimestampMonotonic": "100000000",
                "Result": "success",
                "ExecMainStatus": "0",
            }
            for n in names + [dispatcher]
        },
        action=lambda verb, name, deadline: actions.append((verb, name)),
        boot_links=lambda name, deadline: (
            {"unexpected": "link"} if remaining_link else {}
        ),
        members=lambda group, deadline: (),
        exists=lambda path: str(path) in files,
        command=lambda argv, deadline: (
            "ExecMainCode=1\nExecMainExitTimestampMonotonic=101000000\n"
            if argv[:2] == ["systemctl", "show"]
            else actions.append(tuple(argv))
        ),
    )

    def remove(path, owners, deadline):
        removed.append((str(path), tuple(owners)))
        files.discard(str(path))

    io.remove_owned_file = remove
    c.io = io
    c.clock = lambda: life.Clock(
        BOOT, backend.time, 10**18 + int((backend.time - 100) * 1e9)
    )
    c.host = SimpleNamespace(
        snapshot=lambda name, deadline, partial=False: (units[name], {}),
        _jobs=lambda deadline: {},
    )
    return c, names, actions, removed, files


def test_cleanup_retires_only_registered_workload_files_and_keeps_inert_control(
    tmp_path,
):
    c, names, actions, removed, files = cleanup_fixture(tmp_path)
    result = d.cleanup_resources(cast(q.AuthorizedContext, c), "cleanup-workloads", 120)
    dispatcher = next(
        n for n, u in c.plan.value["units"].items() if u["role"] == "dispatcher"
    )
    assert actions[0] == ("stop", dispatcher)
    assert {name for verb, name in actions if verb == "stop"} == set(names) | {
        dispatcher
    }
    assert {name for verb, name in actions if verb == "disable"} == set(names)
    assert result["dispatcher_barrier"]["unit"]["Id"] == dispatcher
    assert {n for n in result["units"]} == set(names)
    assert result["members"] == result["links"] == result["leftovers"] == []
    assert removed and all(set(owners) == set(names) for _, owners in removed)
    for name, spec in c.plan.value["units"].items():
        if spec["role"] != "workload":
            assert spec["installed_path"] in files


def test_residual_child_prevents_file_retirement_and_has_finite_drain(tmp_path):
    c, _, _, removed, _ = cleanup_fixture(tmp_path, remaining_member=True)
    with pytest.raises(q.Refusal, match="cleanup-cgroup-not-drained"):
        d.cleanup_resources(cast(q.AuthorizedContext, c), "cleanup-workloads", 102)
    assert c.io.now() <= 102.001
    assert removed == []


def test_unremoved_boot_link_prevents_definition_removal(tmp_path):
    c, _, _, removed, _ = cleanup_fixture(tmp_path, remaining_link=True)
    with pytest.raises(q.Refusal, match="cleanup-link-remains"):
        d.cleanup_resources(cast(q.AuthorizedContext, c), "cleanup-workloads", 120)
    assert removed == []


def test_dispatcher_failure_records_raw_failure_without_candidate_or_publication(
    tmp_path, monkeypatch
):
    c = context(tmp_path, purpose="dispatcher")
    name = next(
        n for n, u in c.plan.value["units"].items() if u["role"] == "dispatcher"
    )
    c.unit = q.linux.core.Unit(
        name,
        "a" * 64,
        "/system.slice/" + name,
        "3" * 32,
        q.linux.core.Process(os.getpid(), 9),
        entered_monotonic=100,
    )
    c.io.raw_units[name] = {"ExecMainStartTimestampMonotonic": "100000000"}

    def fault(_):
        raise q.Refusal("planned-case-failure")

    monkeypatch.setattr(d, "run_cases", fault)
    with pytest.raises(q.Refusal, match="planned-case-failure"):
        d.run_role(cast(q.AuthorizedContext, c))
    assert d.entries(c.log, "role-failed")[0][1]["reason"] == "planned-case-failure"
    assert not d.entries(c.log, "observer-candidate")
    assert not d.entries(c.log, "published")
    assert not c.io._backend.actions


@pytest.mark.parametrize(
    "next_deadline,valid", [("8min 10s", True), ("0", False), ("8min 11s", False)]
)
def test_arm_receipt_consumes_actual_duration_fields_and_keeps_original_cutoff(
    tmp_path, monkeypatch, next_deadline, valid
):
    p, data = make_plan()
    manager_path = p["input_root"] + "/timer-manager-facts.json"
    manager = q.encode(
        {
            "boot_id": BOOT,
            "observed_monotonic": 100.0,
            "timer_slack_ns": 50000,
            "observed_dispatch_slack_seconds": 0.2,
        }
    )
    manager_pin = {
        "path": manager_path,
        "sha256": q.sha(manager),
        "bytes": len(manager),
    }
    p["source_pins"].append(manager_pin)
    data[manager_path] = manager
    plan = q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    anchor = life.Anchor("case-one", NONCE, plan.checksum, BOOT, 100, 10**18)
    backend = Backend(data)

    def pin(expected, deadline):
        raw = backend.read(expected["path"], 2**20, deadline)
        assert len(raw) == expected["bytes"] and q.sha(raw) == expected["sha256"]

    monkeypatch.setattr(backend, "pin", pin, raising=False)
    cleanup = next(n for n, row in p["units"].items() if row["role"] == "cleanup")
    watchdog = next(n for n, row in p["units"].items() if row["role"] == "watchdog")

    def command(argv, deadline):
        backend.commands.append(argv)
        if argv[-1] == "--property=TimeoutStartUSec,TimeoutStopUSec":
            return "TimeoutStartUSec=2min\nTimeoutStopUSec=5s\n"
        return f"NextElapseUSecMonotonic={next_deadline}\nAccuracyUSec=1s\nRandomizedDelayUSec=0\nUnit={cleanup}\nLastTriggerUSecMonotonic=0\n"

    backend.command = command
    # Source/ownership observation is a fake-host precondition here. The actual
    # closed command parser, manager-file hash, clock and lifecycle arm gate run.
    monkeypatch.setattr(q, "verify_sources", lambda *args: None)
    monkeypatch.setattr(
        q.DummyReadHost,
        "snapshot",
        lambda self, name, end: (
            SimpleNamespace(
                active="active" if name == watchdog else "inactive",
                substate="waiting" if name == watchdog else "dead",
                job=None,
                dead=name == cleanup,
            ),
            {},
        ),
    )
    log = d.EvidenceLog(tmp_path / "arm", anchor)
    if valid:
        receipt = d.arm_receipt(plan, anchor, cast(d.TargetIO, backend), log)
        life.require_work(log, life.Clock(BOOT, 100, 10**18), receipt)
        raw = d.entries(log, "raw-arm")[0][1]
        assert raw["facts"]["next_trigger_monotonic"] == anchor.deadline("work")
    else:
        with pytest.raises((q.Refusal, life.Refusal)):
            d.arm_receipt(plan, anchor, cast(d.TargetIO, backend), log)
        assert not d.entries(log, "cleanup-armed")
    assert len(backend.commands) == 2
    assert backend.actions == []


@pytest.mark.parametrize(
    "drift", [None, "invocation", "pid", "start_ticks", "old_receipt"]
)
@pytest.mark.parametrize("pending_has_owner", [False, True])
def test_sacrificial_restart_waits_for_and_joins_only_new_owner(
    tmp_path, monkeypatch, drift, pending_has_owner
):
    c = context(tmp_path, purpose="dispatcher")
    name = c.plan.value["bindings"]["holder"]
    group = "/system.slice/" + name
    old_start = ({"sha256": "a" * 64}, {"owner": {"invocation_id": "1" * 32}})
    old_lock = ({"sha256": "b" * 64}, {"pid": 111})
    new_owner = {
        "unit": name,
        "pid": 222,
        "start_ticks": 90,
        "invocation_id": "2" * 32,
        "cgroup": group,
    }
    old_seen = []
    monkeypatch.setattr(d, "_payload_log", lambda *args: c.log)
    monkeypatch.setattr(
        d,
        "entries",
        lambda log, event: [old_start] if event == "payload-started" else [old_lock],
    )

    def start(ctx, unit):
        assert unit == name
        old_seen.append("start-after-old-capture")

    monkeypatch.setattr(d, "_start", start)

    def fresh(ctx, unit, event, prior=None):
        assert unit == name and old_seen
        assert prior == {"a" * 64 if event == "payload-started" else "b" * 64}
        if event == "payload-started":
            value = {"owner": dict(new_owner), "clock": {"monotonic": 100}}
            if drift == "old_receipt":
                value["owner"]["invocation_id"] = "1" * 32
            return {"sha256": "c" * 64}, value
        return {"sha256": "d" * 64}, {"acquired": True, "pid": 222}

    monkeypatch.setattr(d, "_payload_result", fresh)
    actual = dict(new_owner)
    if drift in {"pid", "start_ticks"}:
        actual[drift] += 1
    if drift == "invocation":
        actual["invocation_id"] = "3" * 32
    process = q.linux.core.Process(actual["pid"], actual["start_ticks"])
    pending = q.linux.core.Unit(
        name, "a" * 64, group, active="activating", job={"kind": "start"}
    )
    if pending_has_owner:
        pending = q.linux.core.Unit(
            name,
            "a" * 64,
            group,
            new_owner["invocation_id"],
            q.linux.core.Process(new_owner["pid"], new_owner["start_ticks"]),
            (q.linux.core.Process(new_owner["pid"], new_owner["start_ticks"]),),
            active="activating",
            job={"kind": "start"},
        )
    active = q.linux.core.Unit(
        name,
        "a" * 64,
        group,
        actual["invocation_id"],
        process,
        (process,),
        active="active",
        substate="running",
    )
    observed = []

    def snapshot(unit, deadline):
        observed.append((unit, deadline))
        return (pending if len(observed) == 1 else active), {}

    c.host = SimpleNamespace(snapshot=snapshot)
    if drift is None:
        assert d.restart_holder_for_sacrifice(cast(q.AuthorizedContext, c)) == active
        assert len(observed) == 2 and all(
            end == c.anchor.deadline("work") for _, end in observed
        )
        assert c.io._backend.time > 100
    else:
        with pytest.raises(q.Refusal, match="fresh-sacrificial"):
            d.restart_holder_for_sacrifice(cast(q.AuthorizedContext, c))
    assert not c.io._backend.actions and not c.io._backend.commands
    assert old_start[1]["owner"]["invocation_id"] == "1" * 32


@pytest.mark.parametrize("fault", ["alive", "foreign", "late", "stop-error"])
def test_dispatcher_barrier_failure_still_requests_owned_stops_without_deleting(
    tmp_path, fault
):
    c, names, actions, removed, files = cleanup_fixture(
        tmp_path,
        dispatcher_alive=fault == "alive",
        dispatcher_foreign=fault == "foreign",
    )
    dispatcher = next(
        n for n, u in c.plan.value["units"].items() if u["role"] == "dispatcher"
    )
    if fault == "late":
        c.io._backend.time = c.anchor.deadline("dispatcher_terminal") + 1
    if fault == "stop-error":
        original = c.io.action

        def fail_dispatcher(verb, name, deadline):
            if name == dispatcher:
                raise OSError("dispatcher-observation-unavailable")
            original(verb, name, deadline)

        c.io.action = fail_dispatcher
    with pytest.raises(q.Refusal):
        d.cleanup_resources(
            cast(q.AuthorizedContext, c),
            "cleanup-workloads",
            c.anchor.deadline("cleanup"),
        )
    assert set(names) <= {name for verb, name in actions if verb == "stop"}
    assert removed == []
    assert all(c.plan.value["units"][name]["installed_path"] in files for name in names)
    assert not d.entries(c.log, "cleanup-complete")


def test_observer_cannot_dispatch_case_or_fixture_children(tmp_path):
    from scripts import strength_freshness_cpu_fixture as fixtures

    c = context(tmp_path)
    with pytest.raises(q.Refusal, match="case-dispatcher-only"):
        d.run_cases(cast(q.AuthorizedContext, c))
    with pytest.raises(q.Refusal, match="fixture-dispatcher-only"):
        fixtures.run_bound_fixtures(cast(q.AuthorizedContext, c))
    with pytest.raises(q.Refusal, match="fixture-owner-role"):
        fixtures.run_worker(cast(q.AuthorizedContext, c))
    assert c.io._backend.actions == [] and c.io._backend.commands == []


def test_real_dispatcher_observer_publisher_receipt_chain_composes(
    tmp_path, monkeypatch
):
    """Production receipt APIs + role driver, with explicitly simulated Linux facts."""
    c = context(tmp_path, purpose="dispatcher")
    p = c.plan.value
    names = {
        role: next(n for n, v in p["units"].items() if v["role"] == role)
        for role in ("dispatcher", "observer", "publisher", "cleanup")
    }
    owner_by_role = {}
    fake_pid = [110]
    monkeypatch.setattr(life.os, "getpid", lambda: fake_pid[0])

    def at(t):
        c.io._backend.time = t
        return life.Clock(BOOT, t, 10**18 + int((t - 100) * 1e9))

    def configure(role, t):
        fake_pid[0] = {
            "dispatcher": 110,
            "observer": 220,
            "cleanup": 330,
            "publisher": 440,
        }[role]
        at(t)
        io = q.ClosedIO(
            c.plan,
            c.anchor,
            c.io._backend,
            c.log,
            purpose="finalizer" if role == "publisher" else role,
        )
        process = q.linux.core.Process(os.getpid(), round(t * 100))
        invocation = {
            "dispatcher": "1",
            "observer": "2",
            "publisher": "3",
            "cleanup": "4",
        }[role] * 32
        unit = q.linux.core.Unit(
            names[role],
            "a" * 64,
            "/system.slice/" + names[role],
            invocation,
            process,
            (process,),
            active="active",
            substate="running",
            entered_monotonic=t,
        )
        io.raw_units[names[role]] = {
            "ExecMainStartTimestampMonotonic": str(round(t * 1e6))
        }
        child = SimpleNamespace(
            plan=c.plan,
            anchor=c.anchor,
            log=c.log,
            io=io,
            host=q.DummyReadHost(io),
            unit=unit,
            authorization={"role": role},
            clock=io.clock,
        )
        owner_by_role[role] = d.owner(cast(q.AuthorizedContext, child))
        return child

    def observed(role, end, seen):
        o = owner_by_role[role]
        return {
            "Id": names[role],
            "InvocationID": o["invocation_id"],
            "MainPID": "0",
            "ExecMainPID": str(o["pid"]),
            "ExecMainStartTimestampMonotonic": str(o["start_monotonic_us"]),
            "ExecMainExitTimestampMonotonic": str(round(end * 1e6)),
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
            "ActiveState": "inactive",
            "SubState": "dead",
            "Job": "0",
            "ControlGroup": o["cgroup"],
            "members": [],
            "observed_boot_id": BOOT,
            "observed_monotonic": seen,
        }

    cases = {
        name: {
            "status": "pending-independent-cleanup"
            if name == "cleanup-and-retirement"
            else "passed"
        }
        for name in q.CASES
    }

    def simulated_cases(_):
        at(110)
        return cases

    monkeypatch.setattr(d, "run_cases", simulated_cases)
    dispatcher = configure("dispatcher", 100)
    d.run_role(cast(q.AuthorizedContext, dispatcher))
    result = d.find_one(c.log, "dispatcher-result")
    assert (
        c.log.load(result, "dispatcher-result")["cases"]["cleanup-and-retirement"][
            "status"
        ]
        == "pending-independent-cleanup"
    )
    real_poll = d.poll

    def poll_with_cleanup(ctx, phase, predicate):
        if phase == "observer" and not d.entries(c.log, "cleanup-complete"):
            cleanup = configure("cleanup", 490)
            life.role_started(
                c.log, cleanup.clock(), "cleanup", owner_by_role["cleanup"]
            )
            at(500)
            facts = {
                "units": {
                    n: {
                        "Id": n,
                        "ActiveState": "inactive",
                        "SubState": "dead",
                        "MainPID": 0,
                        "Job": 0,
                        "members": [],
                    }
                    for n, v in p["units"].items()
                    if v["role"] == "workload"
                },
                "jobs": [],
                "members": [],
                "links": [],
                "leftovers": [],
                "dispatcher_barrier": {
                    "started": d.find_one(c.log, "dispatcher-started"),
                    "unit": observed("dispatcher", 112, 500),
                    "clock": asdict(cleanup.clock()),
                },
            }
            life.cleanup_complete(c.log, cleanup.clock(), facts, set(facts["units"]))
            fake_pid[0] = 220
        return real_poll(ctx, phase, predicate)

    monkeypatch.setattr(d, "poll", poll_with_cleanup)
    ends = {"dispatcher": 112, "observer": 525, "cleanup": 501}
    monkeypatch.setattr(
        d,
        "terminal",
        lambda ctx, name, deadline: observed(
            next(r for r, n in names.items() if n == name),
            ends[next(r for r, n in names.items() if n == name)],
            ctx.clock().monotonic,
        ),
    )

    def preservation(_):
        at(520)
        return {
            "status": "passed",
            "owners_unchanged": True,
            "productive": True,
            "observed_monotonic": 520,
        }

    monkeypatch.setattr(d, "load_preservation", preservation)
    observer = configure("observer", 115)
    d.run_role(cast(q.AuthorizedContext, observer))
    completion = d.find_one(c.log, "dispatcher-complete")
    assert c.log.load(completion, "dispatcher-complete")["clock"]["monotonic"] == 115
    candidate = d.find_one(c.log, "observer-candidate")
    assert (
        life.checked_observer_candidate(c.log, candidate)["cases"][
            "cleanup-and-retirement"
        ]["status"]
        == "passed"
    )
    retired = []

    def retire(ctx, purpose, deadline):
        retired.append((ctx.authorization["role"], purpose, deadline))
        return {"jobs": [], "members": [], "links": [], "leftovers": []}

    monkeypatch.setattr(d, "cleanup_resources", retire)
    publisher = configure("publisher", 540)
    d.run_role(cast(q.AuthorizedContext, publisher))
    assert retired == [
        ("publisher", "retire-cleanup-watchdog", c.anchor.deadline("publisher"))
    ]
    at(699)
    receipt = life.audit(
        c.log,
        publisher.clock(),
        d.find_one(c.log, "published-candidate"),
        d.find_one(c.log, "publisher-started"),
        observed("publisher", 565, 699),
    )
    final = c.log.load(receipt, "cpu-scope-result")
    assert all(value is False for value in final["claims"].values())
    assert (
        life.checked_dispatcher_complete(c.log, completion)["clock"]["monotonic"] == 115
    )


def test_owned_stop_is_still_attempted_when_disablement_fails(tmp_path):
    c, names, actions, removed, _ = cleanup_fixture(tmp_path)
    action = c.io.action

    def fail_disable(verb, name, deadline):
        if name == names[0] and verb == "disable":
            raise OSError("owned-disable-failed")
        action(verb, name, deadline)

    c.io.action = fail_disable
    with pytest.raises(q.Refusal, match="owned-stop-incomplete"):
        d.cleanup_resources(cast(q.AuthorizedContext, c), "cleanup-workloads", 120)
    assert ("stop", names[0]) in actions
    assert removed == []
