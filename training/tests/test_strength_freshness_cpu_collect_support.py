from __future__ import annotations

import copy
import json
from typing import cast

import pytest

from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_collect_support as s
from scripts import strength_freshness_cpu_readonly as ro
from tests.test_strength_freshness_cpu_collect_identity import (
    FakeIO,
    MONITOR,
    TIMER,
    identity_fixture,
)

BACKUP = "backup.service"


class SupportIO(FakeIO):
    """Explicit private fixture values. No fallback OS/filesystem/process reads."""

    def __init__(self, reg, props, process):
        super().__init__(reg)
        self.props = copy.deepcopy(props)
        self.ns = 100_000_000_000
        self.boot = "boot"
        self.manager = dict(
            boot_id=self.boot,
            pid=1,
            start_ticks=1,
            ppid=0,
            cgroup="/init.scope",
            namespaces={
                kind: dict(literal=kind + ":[123]", stat=dict(device=1, inode=123))
                for kind in ("pid", "time")
            },
        )
        self.process_value = copy.deepcopy(process)
        self.process_value.update(
            pid=70, start_ticks=7, cgroup="/system.slice/" + MONITOR
        )
        self.owned = {
            MONITOR: [ro.ProcessAdmission(70, 7, "/system.slice/" + MONITOR)],
            BACKUP: [],
        }
        self.job_text = ""
        self.hook = lambda kind, target: None
        self.query_calls = []
        self.manager_calls = 0

    def observation(self, operation, subject, value, raw=None, **metadata):
        begin = dict(boot_id=self.boot, monotonic_ns=self.ns, wall_ns=10**18 + self.ns)
        self.ns += 1_000_000
        end = dict(boot_id=self.boot, monotonic_ns=self.ns, wall_ns=10**18 + self.ns)
        if raw is None:
            raw = identity.encoded(value)
        return ro.Observation(
            value,
            dict(
                operation=operation,
                subject=subject,
                read_start=begin,
                read_end=end,
                raw=dict(sha256=identity.sha(raw), bytes=len(raw)),
                **metadata,
            ),
        )

    def manager_identity(self):
        self.manager_calls += 1
        self.hook("manager", str(self.manager_calls))
        return self.observation("manager-identity", "pid1", copy.deepcopy(self.manager))

    def query(self, kind, name=None):
        target = name
        self.query_calls.append((kind, target))
        self.hook(kind, target)
        if kind == "jobs":
            raw = self.job_text
        elif kind == "default-target":
            raw = "multi-user.target\n"
        elif kind == "registered-target":
            assert target is not None
            raw = "Id=" + target + "\nLoadState=loaded\nWants=\nRequires=\n"
        elif kind == "unit":
            assert target is not None
            raw = "\n".join(f"{k}={v}" for k, v in self.props[target].items())
        else:
            raise AssertionError("unregistered fixture query")
        return self.observation(
            "query", kind + (":" + target if target else ""), raw, raw.encode()
        )

    def members(self, name):
        self.hook("members", name)
        return self.observation(
            "cgroup-members", name, tuple(self.owned[name]), b"members"
        )

    def process(self, admission):
        assert admission.pid == self.process_value["pid"]
        return self.observation(
            "process",
            str(admission.pid),
            copy.deepcopy(self.process_value),
            b"private-component-pins",
        )

    def clock(self):
        self.ns += 1_000_000
        return self.observation(
            "clock",
            "collector",
            dict(boot_id=self.boot, monotonic_ns=self.ns, wall_ns=10**18 + self.ns),
        )


def support_fixture():
    reg, oldio, process = identity_fixture()
    reg["scope"]["units"][BACKUP] = "service"
    reg["scope"]["files"]["backup-fragment"] = {"path": "/etc/systemd/system/" + BACKUP}
    reg["units"][BACKUP] = dict(
        kind="oneshot",
        fragment="backup-fragment",
        dropins=[],
        environment_files=[],
        owned_links=[],
    )
    props = copy.deepcopy(oldio.props)
    props[BACKUP] = {
        **props[MONITOR],
        "Id": BACKUP,
        "Names": BACKUP,
        "FragmentPath": "/etc/systemd/system/" + BACKUP,
    }
    for name, p in props.items():
        p.update(dict.fromkeys(s.DYNAMIC, ""))
        p.update(
            Id=name,
            ActiveState="active" if name in {MONITOR, TIMER} else "inactive",
            SubState="waiting"
            if name == TIMER
            else "running"
            if name == MONITOR
            else "dead",
            InactiveExitTimestampMonotonic="80000000",
            InvocationID="a" * 32 if name == MONITOR else "",
        )
        if name != TIMER:
            p.update(dict.fromkeys(s.SERVICE_DYNAMIC, "0"))
            p.update(
                MainPID="70" if name == MONITOR else "0",
                ControlGroup="/system.slice/" + name,
                Result="success",
            )
    io = SupportIO(reg, props, process)
    host = identity.IdentityCollector(reg, cast(ro.ReadOnlyIO, io))
    monitor = dict(
        pid=70,
        start_ticks=7,
        cgroup="/system.slice/" + MONITOR,
        invocation_id="a" * 32,
        restarts=0,
        origin_sha256=host.origin("monitor", io.process_value),
    )
    host.reg["policy"]["expected_monitors"] = {MONITOR: monitor}
    host.reg["policy"]["maximum_support_seconds"] = 1800
    host.reg["policy"]["support"] = {
        name: {
            "kind": reg["units"][name]["kind"],
            **host.unit_static(name, io.props[name], {}),
        }
        for name in (MONITOR, TIMER, BACKUP)
    }
    return s.SupportCollector(host), io, {MONITOR: monitor}


def activate(io, *, job="42", kind="start", state="running"):
    io.props[BACKUP].update(
        ActiveState="activating",
        SubState="start",
        InvocationID="b" * 32,
        Job=job,
        InactiveExitTimestampMonotonic=str(io.ns // 1000 - 1_000_000),
    )
    io.job_text = f"{job} {BACKUP} {kind} {state}\n" if job else ""


def test_real_identity_facts_support_composition_returns_safe_exact_map():
    collector, io, monitors = support_fixture()
    capture = collector.capture(monitors=monitors)
    assert set(capture.support) == {BACKUP, MONITOR, TIMER}
    keys = {
        "kind",
        "definition_sha256",
        "environment_sha256",
        "boot_links_sha256",
        "enabled",
        "active",
        "result",
        "exit_code",
        "process",
        "running_seconds",
        "job",
    }
    assert all(set(row) == keys for row in capture.support.values())
    assert capture.support[TIMER]["exit_code"] is None
    assert capture.support[MONITOR]["process"] == monitors[MONITOR]
    assert capture.support[BACKUP]["job"] is None
    assert len(capture.witnesses) == 3
    public = json.dumps(
        {
            "support": capture.support,
            "witnesses": capture.witnesses,
            "provenance": capture.provenance,
        }
    )
    assert "private-sentinel-value" not in public and "TOKEN=" not in public
    assert io.manager_calls == 2
    assert [k for k, _ in io.query_calls].count("jobs") == 2


def test_current_job_age_uses_prior_absence_read_start_not_first_seen():
    collector, io, monitors = support_fixture()
    old = collector.capture(monitors=monitors)
    absence = old.witnesses[BACKUP]["absence"]
    io.ns += 10_000_000_000
    activate(io)
    new = collector.capture(monitors=monitors, previous_witnesses=old.witnesses)
    expected = (
        new.provenance["clock"]["monotonic_ns"]
        - absence["read_start"]["monotonic_ns"]
        + 999_999_999
    ) // 1_000_000_000
    assert new.support[BACKUP]["job"] == {"kind": "start", "age_seconds": expected}
    assert expected > 10 and new.witnesses[BACKUP] == old.witnesses[BACKUP]
    activation = io.props[BACKUP]["InactiveExitTimestampMonotonic"]
    assert (
        new.support[BACKUP]["running_seconds"]
        == (
            new.provenance["clock"]["monotonic_ns"]
            - int(activation) * 1000
            + 999_999_999
        )
        // 1_000_000_000
    )


def test_unknown_preexisting_job_does_not_get_first_seen_zero_age():
    collector, io, monitors = support_fixture()
    activate(io)
    with pytest.raises(s.SupportRefusal, match="unknown-job-age"):
        collector.capture(monitors=monitors)


@pytest.mark.parametrize(
    "change",
    [
        lambda w: w["absence"].update(unit=MONITOR),
        lambda w: w["absence"].update(manager_id="f" * 64),
        lambda w: w["absence"]["read_start"].update(monotonic_ns=0),
        lambda w: w["proof"]["unit_reads"][0].update(job=12),
        lambda w: w.update(sha256="f" * 64),
    ],
)
def test_modified_witness_refuses(change):
    collector, io, monitors = support_fixture()
    prior = collector.capture(monitors=monitors).witnesses
    change(prior[BACKUP])
    activate(io)
    with pytest.raises(s.SupportRefusal):
        collector.capture(monitors=monitors, previous_witnesses=prior)


@pytest.mark.parametrize(
    "change",
    [
        lambda io: io.manager.update(start_ticks=9),
        lambda io: io.manager.update(boot_id="different"),
    ],
)
def test_genuine_previous_witness_from_another_manager_refuses(change):
    collector, io, monitors = support_fixture()
    prior = collector.capture(monitors=monitors).witnesses
    change(io)
    activate(io)
    with pytest.raises(s.SupportRefusal):
        collector.capture(monitors=monitors, previous_witnesses=prior)


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda io: io.manager.update(start_ticks=99), "manager-raced"),
        (
            lambda io: io.props[BACKUP].update(InvocationID="c" * 32),
            "support-unit-raced",
        ),
        (
            lambda io: io.props[BACKUP].update(ActiveState="activating"),
            "support-unit-raced",
        ),
        (lambda io: io.props[BACKUP].update(ExecMainStatus="1"), "support-unit-raced"),
    ],
)
def test_observation_races_refuse(change, code):
    collector, io, monitors = support_fixture()

    def hook(kind, target):
        if kind == "manager" and target == "2" and code == "manager-raced":
            change(io)
        if (
            kind == "jobs"
            and sum(k == "jobs" for k, _ in io.query_calls) == 2
            and code != "manager-raced"
        ):
            change(io)

    io.hook = hook
    with pytest.raises(s.SupportRefusal, match=code):
        collector.capture(monitors=monitors)


@pytest.mark.parametrize(
    "job_text,unit_job",
    [
        ("42 backup.service start running\n", ""),
        ("", "42"),
        ("43 backup.service start running\n", "42"),
    ],
)
def test_unit_job_and_manager_job_list_must_join(job_text, unit_job):
    collector, io, monitors = support_fixture()
    io.job_text = job_text
    io.props[BACKUP]["Job"] = unit_job
    with pytest.raises(s.SupportRefusal, match="job-unit-list-mismatch"):
        collector.capture(monitors=monitors)


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("InactiveExitTimestampMonotonic", "0", "support-integer-range"),
        ("InactiveExitTimestampMonotonic", "999999999999999", "activation-future"),
        ("InvocationID", "", "activation-invocation"),
    ],
)
def test_current_activation_age_rejects_missing_zero_future_or_unbound(
    key, value, code
):
    collector, io, monitors = support_fixture()
    activate(io, job="")
    io.props[BACKUP][key] = value
    with pytest.raises((s.SupportRefusal, s.facts.FactViolation), match=code):
        collector.capture(monitors=monitors)


def test_activation_includes_prestart_time_and_old_result_is_not_current_failure():
    collector, io, monitors = support_fixture()
    activate(io, job="")
    io.props[BACKUP].update(
        InactiveExitTimestampMonotonic="80000000",
        ExecMainStartTimestampMonotonic="99000000",
        Result="timeout",
        ExecMainStatus="9",
    )
    result = collector.capture(monitors=monitors)
    assert result.support[BACKUP]["running_seconds"] >= 20
    assert result.support[BACKUP]["result"] == "timeout"


def test_current_finished_failure_and_overdue_work_refuse():
    collector, io, monitors = support_fixture()
    io.props[BACKUP]["Result"] = "exit-code"
    with pytest.raises(s.SupportRefusal, match="support-inactive-failure"):
        collector.capture(monitors=monitors)
    io.props[BACKUP]["Result"] = "success"
    activate(io, job="")
    collector.policy["maximum_support_seconds"] = 0.1
    with pytest.raises(s.SupportRefusal, match="support-activation-overdue"):
        collector.capture(monitors=monitors)


def test_monitor_member_identity_and_origin_rechecked():
    collector, io, monitors = support_fixture()
    io.owned[MONITOR] = [ro.ProcessAdmission(70, 8, "/system.slice/" + MONITOR)]
    with pytest.raises(s.SupportRefusal, match="monitor-lifetime-drift"):
        collector.capture(monitors=monitors)


@pytest.mark.parametrize(
    "text",
    [
        "No jobs running.\n",
        "[]",
        "JOB UNIT TYPE STATE\n",
        "42 backup.service start running extra\n",
        "0 backup.service start running\n",
        "42 backup.service start unknown\n",
        "42 backup.service start running\n42 other.service start running\n",
        "42 backup.service start running\n43 backup.service start running\n",
    ],
)
def test_job_parser_refuses_incompatible_or_ambiguous_table(text):
    with pytest.raises(s.SupportRefusal):
        s.parse_jobs(text)


def test_saved_four_column_shape_with_synthetic_unit_names():
    text = "11315644 fixture-backup.service start running\n11316207 unrelated-backup.service                  start running\n"
    assert len(s.parse_jobs(text)) == 2
    collector, io, monitors = support_fixture()
    io.job_text = "99 unrelated.service stop running\n"
    assert collector.capture(monitors=monitors).support[BACKUP]["job"] is None


def test_old_absence_age_can_expire_without_a_new_first_seen_epoch():
    collector, io, monitors = support_fixture()
    prior = collector.capture(monitors=monitors).witnesses
    io.ns += 1801_000_000_000
    activate(io)
    with pytest.raises(s.SupportRefusal, match="support-job-overdue"):
        collector.capture(monitors=monitors, previous_witnesses=prior)


def test_completed_job_can_create_a_fresh_coherent_absence():
    collector, io, monitors = support_fixture()
    prior = collector.capture(monitors=monitors)
    io.ns += 2_000_000_000
    current = collector.capture(monitors=monitors, previous_witnesses=prior.witnesses)
    assert (
        current.witnesses[BACKUP]["absence"]["read_start"]["monotonic_ns"]
        > prior.witnesses[BACKUP]["absence"]["read_end"]["monotonic_ns"]
    )


def test_same_job_state_transition_during_scan_is_incomplete():
    collector, io, monitors = support_fixture()
    prior = collector.capture(monitors=monitors).witnesses
    io.query_calls.clear()
    activate(io, state="waiting")

    def hook(kind, target):
        if kind == "jobs" and sum(k == "jobs" for k, _ in io.query_calls) == 2:
            io.job_text = "42 backup.service start running\n"

    io.hook = hook
    with pytest.raises(s.SupportRefusal, match="support-job-raced"):
        collector.capture(monitors=monitors, previous_witnesses=prior)


def test_dispatcher_default_target_drift_is_not_hidden_by_policy():
    collector, io, monitors = support_fixture()
    collector.reg["boot"]["default_target"] = "graphical.target"
    with pytest.raises(s.SupportRefusal, match="default-target-drift"):
        collector.capture(monitors=monitors)


def test_support_definition_change_cannot_reuse_old_policy_digest():
    collector, io, monitors = support_fixture()
    io.data["backup-fragment"] = b"changed unit"
    with pytest.raises(s.SupportRefusal, match="support-static-drift"):
        collector.capture(monitors=monitors)


def test_original_io_deadline_refusal_is_not_retried_or_renewed():
    collector, io, monitors = support_fixture()

    def hook(kind, target):
        if kind == "jobs":
            raise ro.ReadRefusal("absolute-read-deadline")

    io.hook = hook
    with pytest.raises(ro.ReadRefusal, match="absolute-read-deadline"):
        collector.capture(monitors=monitors)
    assert sum(k == "jobs" for k, _ in io.query_calls) == 1


def test_safe_final_unit_states_bind_outer_rechecks_without_private_fields():
    collector, io, monitors = support_fixture()
    capture = collector.capture(monitors=monitors)
    states = capture.provenance["unit_states"]
    assert set(states) == {MONITOR, TIMER, BACKUP}
    for name, row in states.items():
        fields = s.DYNAMIC + (s.SERVICE_DYNAMIC if name != TIMER else ())
        assert row == {key: io.props[name][key] for key in fields}
        assert not {
            "Environment",
            "EnvironmentFiles",
            "ExecStart",
            "ExecStartPre",
            "ExecStop",
        }.intersection(row)
    assert states[BACKUP]["Job"] == ""
    io.props[BACKUP]["Job"] = "73"
    assert states[BACKUP]["Job"] == ""
    assert "private-sentinel-value" not in json.dumps(states)
