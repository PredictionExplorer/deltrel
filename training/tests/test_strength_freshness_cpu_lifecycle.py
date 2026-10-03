"""Local CPU/fault evidence only; no systemd, GPU or production operations."""

from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
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
        "invocation_id": {
            "dispatcher": "b",
            "cleanup": "c",
            "observer": "d",
            "publisher": "e",
        }[role]
        * 32,
        "pid": {
            "dispatcher": os.getpid(),
            "cleanup": 101,
            "observer": 102,
            "publisher": 103,
        }[role],
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


def start_dispatcher(log, anchor):
    paths = list(log.root.glob("dispatcher-started-*.json"))
    if paths:
        assert len(paths) == 1
        data = paths[0].read_bytes()
        return {
            "path": str(paths[0]),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    return life.role_started(
        log, clock(anchor, 2), "dispatcher", owner(anchor, "dispatcher")
    )


def dispatcher_proof(log, anchor):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    result = life.dispatcher_result(log, clock(anchor, 100), begun, cases, set(cases))
    return life.dispatcher_complete(
        log,
        clock(anchor, 101),
        begun,
        result,
        terminal(anchor, "dispatcher", 101, 100.5),
        set(cases),
    )


def cleanup_facts(log, anchor):
    return {
        "dispatcher_barrier": {
            "started": start_dispatcher(log, anchor),
            "unit": terminal(anchor, "dispatcher", 400, 100.5),
            "clock": life.asdict(clock(anchor, 400)),
        },
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
    dispatched = dispatcher_proof(log, anchor)
    cleaned = life.cleanup_complete(
        log, clock(anchor, 539), cleanup_facts(log, anchor), {"dummy.service"}
    )
    return life.observer_candidate(
        log,
        clock(anchor, 545),
        cleaned,
        dispatched,
        {"case", "cleanup-and-retirement"},
        {
            "status": "passed",
            "owners_unchanged": True,
            "productive": True,
            "observed_monotonic": anchor.started_monotonic + 544,
        },
    )


def retired():
    return {"jobs": [], "members": [], "links": [], "leftovers": []}


def dispatched_result(log, anchor):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    result = life.dispatcher_result(log, clock(anchor, 100), begun, cases, set(cases))
    return begun, result, set(cases)


@pytest.mark.parametrize(
    "change,observed,ended",
    [
        ({"Result": "timeout"}, 101, 100.5),
        ({"ExecMainCode": "2", "ExecMainStatus": "15"}, 101, 100.5),
        ({"Job": "1"}, 101, 100.5),
        ({"members": [1]}, 101, 100.5),
        ({"MainPID": "1"}, 101, 100.5),
        ({"InvocationID": "f" * 32}, 101, 100.5),
        ({"ExecMainPID": "1"}, 101, 100.5),
        ({"ExecMainStartTimestampMonotonic": "1002000000"}, 101, 100.5),
        ({"observed_boot_id": "foreign"}, 101, 100.5),
        ({"observed_monotonic": 1090}, 101, 100.5),
        ({}, 405, 100.5),
        ({}, 407, 406),
        ({}, 101, 99),
    ],
)
def test_dispatcher_completion_needs_real_natural_owner_and_original_window(
    log, anchor, change, observed, ended
):
    begun, result, expected = dispatched_result(log, anchor)
    facts = terminal(anchor, "dispatcher", observed, ended) | change
    with pytest.raises(life.Refusal):
        life.dispatcher_complete(
            log, clock(anchor, observed), begun, result, facts, expected
        )
    assert "raw-dispatcher-terminal" in events(log)
    assert "dispatcher-complete" not in events(log)


@pytest.mark.parametrize("fault", ["missing", "extra", "premature-cleanup", "failed"])
def test_dispatcher_producer_cannot_change_required_cases_or_claim_cleanup(
    log, anchor, fault
):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    expected = set(cases)
    if fault == "missing":
        del cases["case"]
    elif fault == "extra":
        cases["unrequested"] = {"status": "passed"}
    elif fault == "premature-cleanup":
        cases["cleanup-and-retirement"]["status"] = "passed"
    else:
        cases["case"]["status"] = "failed"
    with pytest.raises(life.Refusal, match="case-set-or-status"):
        life.dispatcher_result(log, clock(anchor, 100), begun, cases, expected)
    assert "dispatcher-result" not in events(log)


def test_dispatcher_completion_is_once_only_and_can_be_read_later_without_backdating(
    log, anchor, monkeypatch
):
    proof = dispatcher_proof(log, anchor)
    saved = Path(proof["path"]).read_bytes()
    value = life.checked_dispatcher_complete(log, proof)
    with pytest.raises(life.Refusal, match="duplicate-dispatcher-complete"):
        life.dispatcher_complete(
            log,
            clock(anchor, 102),
            value["started"],
            value["result"],
            terminal(anchor, "dispatcher", 102, 100.5),
            set(value["expected_cases"]),
        )
    assert Path(proof["path"]).read_bytes() == saved

    def fresh_validator_must_not_be_called(*_args, **_kwargs):
        raise AssertionError("old check clock was presented as current")

    monkeypatch.setattr(life, "natural_exit", fresh_validator_must_not_be_called)
    historical = life.checked_dispatcher_complete(log, proof)
    assert historical["clock"]["monotonic"] == anchor.started_monotonic + 101
    clean = life.cleanup_complete(
        log, clock(anchor, 539), cleanup_facts(log, anchor), {"dummy.service"}
    )
    observed = life.observer_candidate(
        log,
        clock(anchor, 545),
        clean,
        proof,
        set(historical["expected_cases"]),
        {
            "status": "passed",
            "owners_unchanged": True,
            "productive": True,
            "observed_monotonic": anchor.started_monotonic + 544,
        },
    )
    checked = life.checked_observer_candidate(log, observed)
    assert checked["cases"]["cleanup-and-retirement"] == {
        "status": "passed",
        "cleanup": clean,
    }
    assert (
        historical["cases"]["cleanup-and-retirement"]["status"]
        == "pending-independent-cleanup"
    )


def test_work_result_389_natural_exit_391_retains_terminal_grace_and_history(
    log, anchor
):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    result = life.dispatcher_result(log, clock(anchor, 389), begun, cases, set(cases))
    proof = life.dispatcher_complete(
        log,
        clock(anchor, 392),
        begun,
        result,
        terminal(anchor, "dispatcher", 392, 391),
        set(cases),
    )
    historical = life.checked_dispatcher_complete(log, proof)
    assert historical["clock"] == life.asdict(clock(anchor, 392))
    assert (
        life._integer(historical["terminal"]["ExecMainExitTimestampMonotonic"]) / 1e6
        == anchor.started_monotonic + 391
    )
    facts = cleanup_facts(log, anchor)
    facts["dispatcher_barrier"]["unit"] = terminal(anchor, "dispatcher", 400, 391)
    cleaned = life.cleanup_complete(log, clock(anchor, 539), facts, {"dummy.service"})
    candidate_pin = life.observer_candidate(
        log,
        clock(anchor, 545),
        cleaned,
        proof,
        set(cases),
        {
            "status": "passed",
            "owners_unchanged": True,
            "productive": True,
            "observed_monotonic": anchor.started_monotonic + 544,
        },
    )
    assert life.checked_observer_candidate(log, candidate_pin)["dispatcher"] == proof
    assert life.checked_dispatcher_complete(log, proof)["clock"] == historical["clock"]


def test_dispatcher_late_work_result_does_not_use_terminal_grace(log, anchor):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    with pytest.raises(life.Refusal, match="deadline-work"):
        life.dispatcher_result(log, clock(anchor, 391), begun, cases, set(cases))
    assert "dispatcher-result" not in events(log)


def test_forced_dispatcher_death_allows_safe_cleanup_barrier_but_not_aggregate_success(
    log, anchor
):
    begun, result, expected = dispatched_result(log, anchor)
    dead = terminal(anchor, "dispatcher", 400, 399) | {
        "ActiveState": "failed",
        "SubState": "failed",
        "Result": "timeout",
        "ExecMainCode": "2",
        "ExecMainStatus": "9",
    }
    facts = cleanup_facts(log, anchor)
    facts["dispatcher_barrier"]["unit"] = dead
    life.checked_dispatcher_barrier(log, facts["dispatcher_barrier"])
    clean = life.cleanup_complete(log, clock(anchor, 539), facts, {"dummy.service"})
    assert log.load(clean, "cleanup-complete")
    with pytest.raises(life.Refusal):
        life.dispatcher_complete(log, clock(anchor, 400), begun, result, dead, expected)
    assert "dispatcher-complete" not in events(log)
    assert "observer-candidate" not in events(log)


@pytest.mark.parametrize("fault", ["late", "job", "member", "foreign", "source"])
def test_cleanup_dispatcher_barrier_refuses_late_or_unknown_authority(
    log, anchor, fault
):
    facts = cleanup_facts(log, anchor)
    barrier = facts["dispatcher_barrier"]
    if fault == "late":
        barrier["clock"] = life.asdict(clock(anchor, 406))
    elif fault == "job":
        barrier["unit"]["Job"] = "9"
    elif fault == "member":
        barrier["unit"]["members"] = [1]
    elif fault == "foreign":
        barrier["unit"]["InvocationID"] = "e" * 32
    else:
        path = Path(barrier["started"]["path"])
        value = json.loads(path.read_bytes())
        value["data"]["source_sha256"] = "0" * 64
        path.chmod(0o600)
        raw = life.encoded(value)
        path.write_bytes(raw)
        path.chmod(0o444)
        barrier["started"] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    with pytest.raises(life.Refusal):
        life.cleanup_complete(log, clock(anchor, 539), facts, {"dummy.service"})
    assert "cleanup-complete" not in events(log)


def test_missing_or_forged_dispatcher_completion_cannot_feed_observer(log, anchor):
    clean = life.cleanup_complete(
        log, clock(anchor, 539), cleanup_facts(log, anchor), {"dummy.service"}
    )
    fake = log.record_once("dispatcher-complete", {"claims": life.FALSE_CLAIMS})
    with pytest.raises((KeyError, life.Refusal)):
        life.observer_candidate(
            log,
            clock(anchor, 545),
            clean,
            fake,
            {"case", "cleanup-and-retirement"},
            {"status": "passed"},
        )
    assert "observer-candidate" not in events(log)


def test_dispatcher_proof_from_another_attempt_is_not_adopted(log, anchor):
    proof = dispatcher_proof(log, anchor)
    other = life.RawLog(log.root, replace(anchor, nonce="c" * 32))
    with pytest.raises(life.Refusal, match="receipt-binding"):
        life.checked_dispatcher_complete(other, proof)


def test_dispatcher_producer_is_its_own_started_process_and_cannot_renew_work(
    log, anchor, monkeypatch
):
    begun = start_dispatcher(log, anchor)
    cases = {
        "case": {"status": "passed"},
        "cleanup-and-retirement": {"status": "pending-independent-cleanup"},
    }
    with pytest.raises(life.Refusal, match="deadline-work"):
        life.dispatcher_result(log, clock(anchor, 390), begun, cases, set(cases))
    actual = os.getpid()
    monkeypatch.setattr(life.os, "getpid", lambda: actual + 1)
    with pytest.raises(life.Refusal, match="producer-owner"):
        life.dispatcher_result(log, clock(anchor, 100), begun, cases, set(cases))
    assert "dispatcher-result" not in events(log)


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
    facts = cleanup_facts(log, anchor)
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
    facts = cleanup_facts(log, anchor)
    facts["units"]["dummy.service"].update(change)
    with pytest.raises(life.Refusal):
        life.cleanup_complete(log, clock(anchor, 530), facts, {"dummy.service"})


def test_independent_cleanup_survives_absent_aggregate_observer(log, anchor):
    calls = []

    def action(purpose, deadline):
        calls.append((purpose, deadline))
        return cleanup_facts(log, anchor)

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


def test_real_tiny_descendant_is_reaped_and_sigterm_handler_restored(tmp_path):
    # The pytest/xdist worker may legitimately own SIGCHLD. Never reset that
    # handler. An isolated parent proves refusal, then exec gives its child the
    # default disposition without changing either pytest or production guards.
    child_script = r"""
import json, os, signal, sys, time
from pathlib import Path
from scripts import strength_freshness_cpu_lifecycle as life
assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
root = Path(sys.argv[1])
anchor = life.Anchor('isolated-child', 'a'*32, 'b'*64, 'local-test', time.monotonic(), time.time_ns())
log = life.RawLog(root, anchor)
os.environ['INVOCATION_ID'] = 'f'*32
life._self_identity = lambda unit, invocation: {'unit':unit, 'invocation_id':invocation, 'pid':os.getpid()}
old = signal.getsignal(signal.SIGTERM)
now = lambda: life.Clock(anchor.boot_id, time.monotonic(), time.time_ns())
life.run_payload({'unit':'dummy.service','mode':'term_tree','payload':{'seconds':0.03}}, anchor, log, now)
assert signal.getsignal(signal.SIGTERM) == old
row = json.loads(next(root.glob('payload-descendant-*.json')).read_bytes())
try:
    os.waitpid(row['data']['pid'], os.WNOHANG)
except ChildProcessError:
    pass
else:
    raise AssertionError('owned descendant was not reaped')
assert len(list(root.glob('payload-result-*.json'))) == 1
print('isolated-fork-reaped')
"""
    parent_script = r"""
import os, signal, subprocess, sys, time
from pathlib import Path
from scripts import strength_freshness_cpu_lifecycle as life
root = Path(sys.argv[1])
anchor = life.Anchor('foreign-handler', 'a'*32, 'b'*64, 'local-test', time.monotonic(), time.time_ns())
log = life.RawLog(root/'refused', anchor)
os.environ['INVOCATION_ID'] = 'f'*32
life._self_identity = lambda unit, invocation: {'unit':unit, 'invocation_id':invocation, 'pid':os.getpid()}
def foreign_handler(signum, frame): pass
signal.signal(signal.SIGCHLD, foreign_handler)
try:
    life.run_payload({'unit':'dummy.service','mode':'term_tree','payload':{'seconds':0.03}}, anchor, log, lambda:life.Clock(anchor.boot_id,time.monotonic(),time.time_ns()))
except life.Refusal as error:
    assert str(error) == 'payload-child-handler'
else:
    raise AssertionError('foreign SIGCHLD handler was silently accepted')
assert signal.getsignal(signal.SIGCHLD) is foreign_handler
assert not list((root/'refused').glob('payload-descendant-*.json'))
result = subprocess.run([sys.executable,'-c',sys.argv[2],str(root/'actual')],capture_output=True,text=True,check=True,timeout=5)
assert result.stdout.strip() == 'isolated-fork-reaped'
assert signal.getsignal(signal.SIGCHLD) is foreign_handler
print('foreign-refused-and-fresh-child-passed')
"""
    existing_handler = signal.getsignal(signal.SIGCHLD)
    completed = subprocess.run(
        [sys.executable, "-c", parent_script, str(tmp_path), child_script],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(life.__file__).resolve().parents[1]),
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert completed.stdout.strip() == "foreign-refused-and-fresh-child-passed"
    assert signal.getsignal(signal.SIGCHLD) == existing_handler


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
def test_real_sigterm_holder_exits_zero_and_tree_still_ignores_it(mode, tmp_path):
    # A caught SIGCHLD handler in the isolated launcher is reset by exec, while
    # the actual pytest worker's handler is untouched throughout this test.
    child_script = r"""
import importlib.util, os, signal, sys
from pathlib import Path
import pytest
assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
spec = importlib.util.spec_from_file_location('isolated_lifecycle_case', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
anchor = module.life.Anchor('isolated-signals', 'a'*32, 'b'*64, 'local-test', 0.0, 1)
with pytest.MonkeyPatch.context() as patch:
    module._real_sigterm_case(sys.argv[3], Path(sys.argv[2]), anchor, patch)
print('isolated-signal-case-passed')
"""
    launcher = r"""
import signal, subprocess, sys
def foreign_handler(signum, frame): pass
signal.signal(signal.SIGCHLD, foreign_handler)
result = subprocess.run([sys.executable, '-c', *sys.argv[1:]], capture_output=True, text=True, check=True, timeout=6)
assert result.stdout.strip() == 'isolated-signal-case-passed'
assert signal.getsignal(signal.SIGCHLD) is foreign_handler
print(result.stdout.strip())
"""
    old = signal.getsignal(signal.SIGCHLD)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            launcher,
            child_script,
            str(Path(__file__).resolve()),
            str(tmp_path),
            mode,
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(life.__file__).resolve().parents[1]),
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == "isolated-signal-case-passed"
    assert signal.getsignal(signal.SIGCHLD) == old


def _real_sigterm_case(mode, tmp_path, anchor, monkeypatch):
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
