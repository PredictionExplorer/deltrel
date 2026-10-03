"""Local CPU/fault evidence only; no systemd, GPU or production operations."""

from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import pytest

from scripts import strength_freshness_cpu_lifecycle as life


@pytest.fixture
def anchor():
    return life.Anchor.from_dict(
        {
            "attempt_id": "cpu-local-01",
            "nonce": "a" * 32,
            "plan_sha256": "b" * 64,
            "boot_id": "boot-local",
            "started_monotonic": 1000.0,
            "started_wall_ns": 10**18,
        }
    )


def clock(anchor, elapsed):
    return life.Clock(
        anchor.boot_id,
        anchor.started_monotonic + elapsed,
        anchor.started_wall_ns + int(elapsed * 1e9),
    )


@pytest.fixture
def log(tmp_path, anchor):
    return life.RawLog(tmp_path / "raw", anchor)


def arm_facts(anchor):
    return {
        "next_trigger_monotonic": anchor.deadline("work"),
        "accuracy_seconds": 1.0,
        "randomized_delay_seconds": 0.0,
        "manager_slack_seconds": 0.00005,
        "observed_dispatch_slack_seconds": 0.2,
        "cleanup_timeout_start_seconds": 120,
        "cleanup_timeout_stop_seconds": 5,
        "active_waiting": True,
        "independent_cleanup": True,
        "ownership_validated": True,
    }


def events(log):
    return [json.loads(p.read_bytes())["event"] for p in log.root.glob("*.json")]


def owner(anchor, role):
    return {
        "unit": "edgeconnect-cpuqual-" + anchor.nonce + "-" + role + ".service",
        "invocation_id": {"cleanup": "c", "observer": "d", "publisher": "e"}[role] * 32,
        "pid": {"cleanup": 101, "observer": 102, "publisher": 103}[role],
        "start_monotonic_us": int((anchor.started_monotonic + 1) * 1e6),
        "cgroup": "/system.slice/dummy-" + role + ".service",
    }


def started(log, anchor, role):
    # These controller paths use synthetic systemd facts. Actual role_started
    # self-PID admission is exercised separately below.
    return log.record(
        role + "-started",
        {"owner": owner(anchor, role), "clock": life.asdict(clock(anchor, 2))},
    )


def terminal(anchor, role, observed, ended):
    o = owner(anchor, role)
    return {
        "Id": o["unit"],
        "InvocationID": o["invocation_id"],
        "MainPID": "0",
        "ExecMainPID": str(o["pid"]),
        "ExecMainStartTimestampMonotonic": str(o["start_monotonic_us"]),
        "ExecMainExitTimestampMonotonic": str(
            int((anchor.started_monotonic + ended) * 1e6)
        ),
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
        "ActiveState": "inactive",
        "SubState": "dead",
        "Job": "0",
        "ControlGroup": o["cgroup"],
        "members": [],
        "observed_boot_id": anchor.boot_id,
        "observed_monotonic": anchor.started_monotonic + observed,
    }


def cleanup_facts():
    return {
        "units": {
            "dummy.service": {
                "Id": "dummy.service",
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": 0,
                "Job": "0",
                "members": [],
            }
        },
        "jobs": [],
        "members": [],
        "links": [],
        "leftovers": [],
    }


def candidate(log, anchor):
    cleaned = life.cleanup_complete(
        log, clock(anchor, 539), cleanup_facts(), {"dummy.service"}
    )
    return life.observer_candidate(
        log,
        clock(anchor, 545),
        cleaned,
        {"case": {"status": "passed", "scope": "synthetic-local"}},
        {"case"},
        {
            "status": "passed",
            "owners_unchanged": True,
            "productive": True,
            "observed_monotonic": anchor.started_monotonic + 544,
        },
    )


def retired():
    return {"jobs": [], "members": [], "links": [], "leftovers": []}


def test_original_clock_roundtrip_and_late_arming_do_not_reset_window(anchor):
    same = life.Anchor.from_dict(anchor.as_dict())
    assert same.deadline("work") == 1390
    assert same.remaining("work", clock(anchor, 380)) == 10
    assert same.remaining("observer", clock(anchor, 539), 5) == 26
    assert same.remaining("audit", clock(anchor, 599)) == 1


@pytest.mark.parametrize("phase,end", list(life.ENDS.items()))
def test_every_absolute_cutoff_refuses_equality(anchor, phase, end):
    with pytest.raises(life.Refusal, match="deadline"):
        anchor.remaining(phase, clock(anchor, end))


@pytest.mark.parametrize(
    "change", [{"boot_id": "next-boot"}, {"monotonic": 999}, {"wall_ns": 0}]
)
def test_boot_or_clock_regression_refuses(anchor, change):
    now = replace(clock(anchor, 1), **change)
    with pytest.raises(life.Refusal, match="clock"):
        anchor.remaining("audit", now)


def test_wall_forward_charges_remaining_budget(anchor):
    now = replace(clock(anchor, 10), wall_ns=anchor.started_wall_ns + 590 * 10**9)
    assert anchor.remaining("audit", now) == 10
    with pytest.raises(life.Refusal, match="deadline-work"):
        anchor.remaining("work", now)


def test_work_needs_existing_armed_receipt_and_no_work_after_cutoff(log, anchor):
    with pytest.raises((KeyError, life.Refusal)):
        life.require_work(log, clock(anchor, 1), {})
    armed = life.acknowledge_arm(log, clock(anchor, 30), arm_facts(anchor))
    assert life.require_work(log, clock(anchor, 389), armed) == 1390
    with pytest.raises(life.Refusal, match="deadline-work"):
        life.require_work(log, clock(anchor, 390), armed)


@pytest.mark.parametrize(
    "change",
    [
        {"active_waiting": False},
        {"independent_cleanup": False},
        {"ownership_validated": False},
        {"next_trigger_monotonic": 1420},
        {"randomized_delay_seconds": 1},
        {"accuracy_seconds": 60},
        {"manager_slack_seconds": 30},
        {"observed_dispatch_slack_seconds": 30},
        {"cleanup_timeout_start_seconds": 121},
        {"cleanup_timeout_stop_seconds": 6},
    ],
)
def test_arm_failure_raw_facts_are_retained_before_refusal(log, anchor, change):
    facts = arm_facts(anchor) | change
    with pytest.raises(life.Refusal):
        life.acknowledge_arm(log, clock(anchor, 1), facts)
    assert events(log) == ["raw-arm"]
    raw = json.loads(next(log.root.glob("*.json")).read_bytes())
    assert raw["data"]["facts"] == facts


def test_real_host_nonzero_timer_slack_is_retained(log, anchor):
    facts = arm_facts(anchor)
    armed = life.acknowledge_arm(log, clock(anchor, 1), facts)
    value = log.load(armed, "cleanup-armed")
    raw = log.load(value["raw"], "raw-arm")
    assert raw["facts"]["manager_slack_seconds"] == 0.00005


@pytest.mark.parametrize("field", ["jobs", "members", "links", "leftovers"])
def test_cleanup_leftovers_never_become_success(log, anchor, field):
    facts = cleanup_facts()
    facts[field] = ["still-present"]
    with pytest.raises(life.Refusal, match="cleanup-incomplete"):
        life.cleanup_complete(log, clock(anchor, 530), facts, {"dummy.service"})
    assert "raw-cleanup" in events(log)
    assert "cleanup-complete" not in events(log)


@pytest.mark.parametrize(
    "change",
    [{"Job": "7"}, {"members": [88]}, {"MainPID": 88}, {"Id": "foreign.service"}],
)
def test_cleanup_joins_exact_units_and_actual_death(log, anchor, change):
    facts = cleanup_facts()
    facts["units"]["dummy.service"].update(change)
    with pytest.raises(life.Refusal):
        life.cleanup_complete(log, clock(anchor, 530), facts, {"dummy.service"})


def test_independent_cleanup_survives_absent_aggregate_observer(log, anchor):
    calls = []

    def action(purpose, deadline):
        calls.append((purpose, deadline))
        return cleanup_facts()

    result = life.run_cleanup(
        log, lambda: clock(anchor, 520), {"dummy.service"}, action
    )
    assert calls == [("cleanup-workloads", 1540)]
    assert log.load(result, "cleanup-complete")
    assert "observer-candidate" not in events(log)
    assert "cpu-scope-result" not in events(log)


def test_cleanup_callback_death_records_failure_and_never_completion(log, anchor):
    def dies(_purpose, _deadline):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        life.run_cleanup(log, lambda: clock(anchor, 500), {"dummy.service"}, dies)
    assert set(events(log)) == {"cleanup-intent", "cleanup-action-failed"}


def test_full_lifecycle_requires_separate_natural_exits(log, anchor):
    item = candidate(log, anchor)
    calls = []

    def retire(purpose, deadline):
        calls.append((purpose, deadline))
        return retired()

    published = life.publish(
        log,
        lambda: clock(anchor, 580),
        item,
        started(log, anchor, "observer"),
        terminal(anchor, "observer", 580, 560),
        started(log, anchor, "cleanup"),
        terminal(anchor, "cleanup", 580, 530),
        retire,
    )
    assert calls == [("retire-cleanup-watchdog", 1590)]
    assert (
        log.load(published, "published-candidate")["status"]
        == "pending_external_terminal_audit"
    )
    assert "cpu-scope-result" not in events(log)
    result = life.audit(
        log,
        clock(anchor, 599),
        published,
        started(log, anchor, "publisher"),
        terminal(anchor, "publisher", 599, 588),
    )
    final = log.load(result, "cpu-scope-result")
    assert final["status"] == "passed_cpu_scope"
    assert all(v is False for v in final["claims"].values())


@pytest.mark.parametrize(
    "change",
    [
        {"Result": "timeout"},
        {"ExecMainCode": "2"},
        {"ExecMainStatus": "1"},
        {"Job": "18"},
        {"MainPID": "102"},
        {"members": [102]},
        {"InvocationID": "f" * 32},
        {"ExecMainPID": "999"},
        {"ExecMainStartTimestampMonotonic": "1002000000"},
        {"observed_boot_id": "other-boot"},
        {"observed_monotonic": 1500},
    ],
)
def test_observer_death_or_stale_identity_never_authorizes_retirement(
    log, anchor, change
):
    item = candidate(log, anchor)
    unit = terminal(anchor, "observer", 555, 550) | change
    calls = []
    with pytest.raises(life.Refusal):
        life.publish(
            log,
            lambda: clock(anchor, 555),
            item,
            started(log, anchor, "observer"),
            unit,
            started(log, anchor, "cleanup"),
            terminal(anchor, "cleanup", 555, 530),
            lambda p, d: calls.append((p, d)) or retired(),
        )
    assert not calls and "raw-publisher" in events(log)
    assert "published-candidate" not in events(log)


def test_cleanup_receipt_is_insufficient_while_cleanup_actor_lives(log, anchor):
    item = candidate(log, anchor)
    calls = []
    with pytest.raises(life.Refusal, match="terminal-not-natural"):
        life.publish(
            log,
            lambda: clock(anchor, 555),
            item,
            started(log, anchor, "observer"),
            terminal(anchor, "observer", 555, 550),
            started(log, anchor, "cleanup"),
            terminal(anchor, "cleanup", 555, 530) | {"MainPID": "101"},
            lambda p, d: calls.append((p, d)) or retired(),
        )
    assert not calls


def test_cleared_invocation_requires_exact_retained_exec_identity(anchor):
    unit = terminal(anchor, "publisher", 598, 589) | {"InvocationID": ""}
    life.natural_exit(
        anchor, clock(anchor, 598), owner(anchor, "publisher"), unit, "publisher"
    )
    with pytest.raises(life.Refusal, match="terminal-owner"):
        life.natural_exit(
            anchor,
            clock(anchor, 598),
            owner(anchor, "publisher"),
            unit | {"ExecMainPID": "0"},
            "publisher",
        )


def test_publisher_timeout_is_incomplete_even_with_published_receipt(log, anchor):
    published = life.publish(
        log,
        lambda: clock(anchor, 555),
        candidate(log, anchor),
        started(log, anchor, "observer"),
        terminal(anchor, "observer", 555, 550),
        started(log, anchor, "cleanup"),
        terminal(anchor, "cleanup", 555, 530),
        lambda _p, _d: retired(),
    )
    with pytest.raises(life.Refusal, match="terminal-not-natural"):
        life.audit(
            log,
            clock(anchor, 598),
            published,
            started(log, anchor, "publisher"),
            terminal(anchor, "publisher", 598, 589) | {"Result": "timeout"},
        )
    assert "cpu-scope-result" not in events(log)


def test_raw_receipt_no_clobber_tamper_and_wrong_anchor_refused(log, anchor, tmp_path):
    first = log.record("raw-sample", {"value": 1})
    second = log.record("raw-sample", {"value": 2})
    assert first["path"] != second["path"]
    other = life.RawLog(tmp_path / "other", replace(anchor, nonce="c" * 32))
    with pytest.raises(life.Refusal, match="location"):
        other.load(first, "raw-sample")
    path = Path(first["path"])
    path.chmod(0o600)
    path.write_text("{}")
    path.chmod(0o444)
    with pytest.raises(life.Refusal):
        log.load(first, "raw-sample")


def test_interrupted_atomic_record_cannot_publish_success(log, monkeypatch):
    def die(*_args, **_kwargs):
        raise OSError("injected link failure")

    monkeypatch.setattr(life.os, "link", die)
    with pytest.raises(OSError):
        log.record("cpu-scope-result", {"status": "passed_cpu_scope"})
    assert list(log.root.glob("*.json")) == []
    assert len(list(log.root.glob("*.partial"))) == 1


def authorization(tmp_path, monkeypatch, anchor):
    directory = tmp_path / "inputs"
    directory.mkdir(mode=0o700)
    source = directory / "lifecycle.py"
    source.write_bytes(b"pinned synthetic source")
    source.chmod(0o444)
    monkeypatch.setattr(life, "__file__", str(source))
    name = "edgeconnect-cpuqual-" + anchor.nonce + "-tiny.service"
    payload = {
        "seconds": 0.01,
        "output_dir": "/run/edgeconnect-cpuqual-" + anchor.nonce + "/payloads/" + name,
        "lock_path": None,
    }
    source_pins = [
        {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "bytes": source.stat().st_size,
        }
    ]
    plan = {
        "format": "strength-freshness-cpu-plan-v1",
        "schema_version": 1,
        "attempt_id": anchor.attempt_id,
        "nonce": anchor.nonce,
        "boot_id": anchor.boot_id,
        "source_pins": source_pins,
        "units": {name: {"role": "workload", "mode": "sleep", "payload": payload}},
    }
    plan_path = directory / "plan.json"
    plan_path.write_bytes(life.encoded(plan))
    plan_path.chmod(0o444)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    anchor = replace(anchor, plan_sha256=plan_sha)
    anchor_path = directory / "anchor.json"
    anchor_path.write_bytes(life.encoded(anchor.as_dict()))
    anchor_path.chmod(0o444)
    auth = {
        "format": "strength-freshness-cpu-authorization-v1",
        "schema_version": 1,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "anchor_path": str(anchor_path),
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "source_pins": source_pins,
        "role": "workload",
        "unit": name,
        "mode": "sleep",
        "payload": payload,
    }
    path = directory / "authorization.json"
    path.write_bytes(life.encoded(auth))
    path.chmod(0o600)
    return path, auth, anchor


def test_protected_companion_avoids_cycle_and_pins_actual_plan_source(
    tmp_path, monkeypatch, anchor
):
    path, auth, expected = authorization(tmp_path, monkeypatch, anchor)
    actual, loaded = life.load_payload_authorization(path, expected_uid=os.geteuid())
    assert actual == auth and loaded == expected
    assert "--sha256" not in path.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "cleanup"),
        ("mode", "shell"),
        ("plan_sha256", "0" * 64),
        ("anchor_sha256", "0" * 64),
    ],
)
def test_payload_authorization_drift_refused_before_any_payload(
    tmp_path, monkeypatch, anchor, field, value
):
    path, auth, _ = authorization(tmp_path, monkeypatch, anchor)
    auth[field] = value
    path.write_bytes(life.encoded(auth))
    with pytest.raises(life.Refusal):
        life.load_payload_authorization(path, expected_uid=os.geteuid())


def test_authorization_symlink_or_unprotected_file_refused(
    tmp_path, monkeypatch, anchor
):
    path, _, _ = authorization(tmp_path, monkeypatch, anchor)
    alias = path.with_name("alias.json")
    alias.symlink_to(path)
    with pytest.raises(life.Refusal, match="canonical"):
        life.load_payload_authorization(alias, expected_uid=os.geteuid())
    path.chmod(0o644)
    with pytest.raises(life.Refusal, match="protection"):
        life.load_payload_authorization(path, expected_uid=os.geteuid())


def test_fast_payload_has_real_current_pid_and_no_cleanup_capability(
    log, anchor, monkeypatch
):
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    monkeypatch.setattr(
        life,
        "_self_identity",
        lambda unit, invocation: {
            "unit": unit,
            "invocation_id": invocation,
            "pid": os.getpid(),
        },
    )
    result = life.run_payload(
        {"unit": "dummy.service", "mode": "fast_receipt", "payload": {"seconds": 0.0}},
        anchor,
        log,
        lambda: clock(anchor, 1),
    )
    assert log.load(result, "payload-result")["status"] == "completed"
    assert set(events(log)) == {"payload-started", "payload-result"}


def test_real_flock_contender_records_conflict_without_waiting(
    log, anchor, tmp_path, monkeypatch
):
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    monkeypatch.setattr(
        life,
        "_self_identity",
        lambda unit, invocation: {
            "unit": unit,
            "invocation_id": invocation,
            "pid": os.getpid(),
        },
    )
    lock = tmp_path / "qualification.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        life.run_payload(
            {
                "unit": "dummy.service",
                "mode": "lock_contender",
                "payload": {"seconds": 0.0, "lock_path": str(lock)},
            },
            anchor,
            log,
            lambda: clock(anchor, 1),
        )
    finally:
        os.close(fd)
    receipt = next(p for p in log.root.glob("payload-lock-*.json"))
    assert json.loads(receipt.read_bytes())["data"]["acquired"] is False


def test_role_started_binds_own_process_and_original_clock(log, anchor):
    o = owner(anchor, "observer") | {"pid": os.getpid()}
    pin = life.role_started(log, clock(anchor, 2), "observer", o)
    assert log.load(pin, "observer-started")["owner"] == o
    with pytest.raises(life.Refusal, match="start-owner"):
        life.role_started(
            log, clock(anchor, 2), "observer", o | {"pid": os.getpid() + 1}
        )


def test_real_tiny_descendant_is_reaped_and_sigterm_handler_restored(
    tmp_path, anchor, monkeypatch
):
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    monkeypatch.setattr(
        life,
        "_self_identity",
        lambda unit, invocation: {
            "unit": unit,
            "invocation_id": invocation,
            "pid": os.getpid(),
        },
    )
    anchor = replace(
        anchor, started_monotonic=time.monotonic(), started_wall_ns=time.time_ns()
    )
    log = life.RawLog(tmp_path / "descendant-raw", anchor)
    old = signal.getsignal(signal.SIGTERM)

    def now():
        return life.Clock(anchor.boot_id, time.monotonic(), time.time_ns())

    life.run_payload(
        {"unit": "dummy.service", "mode": "term_tree", "payload": {"seconds": 0.03}},
        anchor,
        log,
        now,
    )
    assert signal.getsignal(signal.SIGTERM) == old
    receipt = json.loads(next(log.root.glob("payload-descendant-*.json")).read_bytes())
    with pytest.raises(ChildProcessError):
        os.waitpid(receipt["data"]["pid"], os.WNOHANG)
    assert "payload-result" in events(log)


def test_actual_flock_holder_releases_owned_fd(log, anchor, tmp_path, monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    monkeypatch.setattr(
        life,
        "_self_identity",
        lambda unit, invocation: {
            "unit": unit,
            "invocation_id": invocation,
            "pid": os.getpid(),
        },
    )
    path = tmp_path / "lock"
    life.run_payload(
        {
            "unit": "dummy.service",
            "mode": "lock_holder",
            "payload": {"seconds": 0, "lock_path": str(path)},
        },
        anchor,
        log,
        lambda: clock(anchor, 1),
    )
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_duplicate_or_nonfinite_metadata_is_never_normalized(raw):
    with pytest.raises(life.Refusal):
        life.strict_json(raw)


def test_raw_write_failure_prevents_retirement_callback(log, anchor, monkeypatch):
    item = candidate(log, anchor)
    observer = started(log, anchor, "observer")
    cleanup = started(log, anchor, "cleanup")
    calls = []

    def fail(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(life.os, "fsync", fail)
    with pytest.raises(OSError):
        life.publish(
            log,
            lambda: clock(anchor, 555),
            item,
            observer,
            terminal(anchor, "observer", 555, 550),
            cleanup,
            terminal(anchor, "cleanup", 555, 530),
            lambda p, d: calls.append((p, d)) or retired(),
        )
    assert not calls


def test_incomplete_published_receipt_cannot_invent_cleanup_or_r3_proof(log, anchor):
    fake = log.record(
        "published-candidate",
        {"status": "pending_external_terminal_audit", "claims": life.FALSE_CLAIMS},
    )
    with pytest.raises((KeyError, life.Refusal)):
        life.audit(
            log,
            clock(anchor, 599),
            fake,
            started(log, anchor, "publisher"),
            terminal(anchor, "publisher", 599, 588),
        )
    assert "cpu-scope-result" not in events(log)


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_concurrent_raw_writers_cannot_exceed_shared_cap(log, monkeypatch, limit):
    if limit == "files":
        monkeypatch.setattr(life, "MAX_LOG_FILES", 1)
    else:
        sample = log.record("raw-concurrent", {"value": 1})
        size = sample["bytes"]
        Path(sample["path"]).unlink()
        monkeypatch.setattr(life, "MAX_LOG_BYTES", size)
    gate_r, gate_w = os.pipe()
    result_r, result_w = os.pipe()
    children = []
    try:
        for _ in range(2):
            pid = os.fork()
            if pid == 0:
                os.close(gate_w)
                os.close(result_r)
                try:
                    os.read(gate_r, 1)
                    log.record("raw-concurrent", {"value": 1})
                    outcome = b"P"
                except life.Refusal as exc:
                    outcome = b"F" if str(exc) == "raw-evidence-cap" else b"E"
                except BaseException:
                    outcome = b"E"
                os.write(result_w, outcome)
                os._exit(0)
            children.append(pid)
        os.close(gate_r)
        gate_r = -1
        os.close(result_w)
        result_w = -1
        os.write(gate_w, b"xx")
        result = b""
        while len(result) < 2:
            result += os.read(result_r, 2 - len(result))
        assert sorted(result) == sorted(b"PF")
        for pid in children:
            assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
        children.clear()
        assert len(list(log.root.glob("*.json"))) == 1
    finally:
        for fd in (gate_r, gate_w, result_r, result_w):
            if fd >= 0:
                os.close(fd)
        for pid in children:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


def test_raw_log_busy_is_bounded_and_cannot_publish(log):
    fd = os.open(log.root / ".append.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        began = time.monotonic()
        with pytest.raises(life.Refusal, match="raw-log-lock-busy"):
            log.record("raw-blocked", {})
        assert 0.15 <= time.monotonic() - began < 1
    finally:
        os.close(fd)
    assert list(log.root.glob("*.json")) == []


@pytest.mark.parametrize("mode", ["lock_holder", "term_tree"])
def test_real_sigterm_holder_exits_zero_and_tree_still_ignores_it(
    mode, tmp_path, anchor, monkeypatch
):
    """Signals only our unreaped fork child; all identity metadata is local-fixture scope."""
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    monkeypatch.setattr(
        life,
        "_self_identity",
        lambda unit, invocation: {
            "unit": unit,
            "invocation_id": invocation,
            "pid": os.getpid(),
        },
    )
    anchor = replace(
        anchor, started_monotonic=time.monotonic(), started_wall_ns=time.time_ns()
    )
    log = life.RawLog(tmp_path / mode, anchor)
    lock = tmp_path / "signal.lock"
    read_fd, write_fd = os.pipe()
    release_read, release_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        os.close(release_write)

        def original(signum, frame):
            pass

        signal.signal(signal.SIGTERM, original)
        try:
            payload: dict[str, object] = {
                "seconds": 10 if mode == "lock_holder" else 0.4
            }
            if mode == "lock_holder":
                payload["lock_path"] = str(lock)
            result = life.run_payload(
                {"unit": "dummy.service", "mode": mode, "payload": payload},
                anchor,
                log,
                lambda: life.Clock(anchor.boot_id, time.monotonic(), time.time_ns()),
            )
            assert signal.getsignal(signal.SIGTERM) is original
            assert log.load(result, "payload-result")["status"] == (
                "stopped" if mode == "lock_holder" else "completed"
            )
            os.write(write_fd, b"handler-restored")
            os.read(release_read, 1)
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(write_fd)
    os.close(release_read)
    try:
        ready = "payload-lock" if mode == "lock_holder" else "payload-descendant"
        deadline = time.monotonic() + 3
        while not list(log.root.glob(ready + "-*.json")):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        if mode == "lock_holder":
            probe = os.open(lock, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
        os.kill(child, signal.SIGTERM)
        if mode == "term_tree":
            time.sleep(0.03)
            assert os.waitpid(child, os.WNOHANG) == (0, 0)
        if mode == "lock_holder":
            while not list(log.root.glob("payload-stopped-*.json")):
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert os.waitpid(child, os.WNOHANG) == (0, 0)
            released = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(released)
        os.write(release_write, b"exit")
        while True:
            waited, status = os.waitpid(child, os.WNOHANG)
            if waited:
                child = None
                assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
                break
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert os.read(read_fd, 64) == b"handler-restored"
        if mode == "lock_holder":
            stopped = json.loads(
                next(log.root.glob("payload-stopped-*.json")).read_bytes()
            )["data"]
            assert stopped["signal"] == "SIGTERM" and stopped["lock_released"] is True
            probe = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
        else:
            assert not list(log.root.glob("payload-stopped-*.json"))
    finally:
        os.close(read_fd)
        os.close(release_write)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child, 0)
