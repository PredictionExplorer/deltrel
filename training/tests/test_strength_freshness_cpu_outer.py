"""Synthetic kernel/fault tests: no fork, prctl, target or GPU operations."""

from dataclasses import asdict, replace
import hashlib
import signal

import pytest

from scripts import strength_freshness_cpu_outer as m


class FakeKernel:
    def __init__(self):
        self.t = 0
        self.wall_jump = 0
        self.owner = m.ProcessIdentity(100, 10, 50, 100, 100, 0, "boot", "/session", 99)
        self.processes = {
            100: self.owner,
            50: replace(self.owner, pid=50, start_ticks=5, ppid=1, pgid=50, sid=50),
        }
        self.fds = {99: 50}
        self.nextfd = 1000
        self.enabled = 0
        self.thread_extra = False
        self.fail_subreaper = False
        self.admitted = True
        self.forked = 0
        self.released = False
        self.ready = True
        self.exit_at = 0
        self.exit_code = 0
        self.exit_kind = "CLD_EXITED"
        self.parent_dead_at: int | None = None
        self.ignore_term = False
        self.ignore_kill = False
        self.signalled = []
        self.reaped = []
        self.closed = []
        self.stdout = [b"{}", b""]
        self.stderr = [b""]
        self.orphans = 0
        self.wait_mismatch = False
        self.pidfd_wrong = False

    def clock(self):
        return m.Clock("boot", self.t, 10**15 + self.t + self.wall_jump)

    def self_identity(self):
        return (
            replace(self.owner, ppid=1)
            if self.parent_dead_at is not None and self.t >= self.parent_dead_at
            else self.owner
        )

    def task_ids(self):
        return (100, 101) if self.thread_extra else (100,)

    def subreaper(self, enabled=None):
        if enabled is not None and not self.fail_subreaper:
            self.enabled = int(enabled)
        return self.enabled

    def admit_execution(self, binding, spec):
        assert binding.exec_sha256 == m.digest(spec.contract())
        m.require(self.admitted, "fake-admission-refused")

    def children(self):
        return tuple(p for p, v in self.processes.items() if p != 100 and v.ppid == 100)

    def identity(self, pid):
        return self.processes[pid]

    def open_pidfd(self, pid):
        self.nextfd += 1
        self.fds[self.nextfd] = pid
        return self.nextfd

    def pidfd_pid(self, fd):
        return self.fds[fd] + int(self.pidfd_wrong and fd != 99)

    def exited(self, fd):
        if fd == 99:
            return self.parent_dead_at is not None and self.t >= self.parent_dead_at
        return self.peek(fd) is not None

    def peek(self, fd):
        pid = self.fds[fd]
        if pid == 200 and self.t >= self.exit_at:
            return m.Terminal(pid, self.exit_kind, self.exit_code)
        if pid != 200 and (pid, signal.SIGKILL) in self.signalled:
            return m.Terminal(pid, "CLD_KILLED", signal.SIGKILL)
        return None

    def reap(self, fd):
        terminal = self.peek(fd)
        assert terminal is not None
        pid = terminal.pid
        self.reaped.append(pid)
        del self.processes[pid]
        if pid == 200:
            for index in range(self.orphans):
                # Simulates kernel subreaper adoption, not a PGID descendant.
                child = 300 + index
                self.processes[child] = replace(
                    self.owner,
                    pid=child,
                    start_ticks=30 + index,
                    ppid=100,
                    pgid=child,
                    sid=child,
                )
        status = (
            terminal.status << 8 if terminal.code == "CLD_EXITED" else terminal.status
        )
        return m.Reaped(pid, status + int(self.wait_mismatch))

    def signal(self, fd, sig):
        pid = self.fds[fd]
        self.signalled.append((pid, sig))
        if pid == 200 and not (
            (sig == signal.SIGTERM and self.ignore_term)
            or (sig == signal.SIGKILL and self.ignore_kill)
        ):
            self.exit_at = self.t
            self.exit_kind, self.exit_code = "CLD_KILLED", sig

    def close(self, fd):
        self.closed.append(fd)
        if fd == 900 and not self.released:
            self.exit_at = self.t
            self.exit_code = 125

    def fork_capture(self, spec):
        assert not self.forked
        self.forked += 1
        self.processes[200] = replace(
            self.owner, pid=200, start_ticks=20, ppid=100, pgid=200, sid=200
        )
        if not self.ready:
            self.exit_code = 125
        return m.Spawn(200, 900 if self.ready else -1, 901, 902)

    def release(self, fd):
        assert fd == 900 and self.fds and self.forked
        self.released = True
        self.closed.append(fd)

    def read_output(self, fd, maximum):
        data = self.stdout if fd == 901 else self.stderr
        return data.pop(0) if data else b""

    def sleep(self, seconds):
        assert 0 <= seconds <= 0.01
        self.t += max(1, round(seconds * m.SECOND))


def setup(*, role="guardian", budget=None, maximum_output=2**20):
    kernel = FakeKernel()
    budget = m.PhaseBudget.before(kernel.clock()) if budget is None else budget
    kernel.t = budget.phase_start.monotonic_ns
    proof = m.initialize_role(kernel, role, kernel.owner)
    family = m.OwnedFamily(kernel, proof, budget, maximum_output=maximum_output)
    spec = m.CaptureSpec(
        budget.phase,
        "/qualified/python",
        "/qualified/control",
        "/input/launch.json",
        "a" * 64,
        budget.work_ns,
    )
    binding = m.Binding(
        "b" * 32,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        m.digest(asdict(budget)),
        m.digest(spec.contract()),
    )
    parent = m.Handle(kernel.processes[50], 99)
    return kernel, family, spec, binding, parent


@pytest.mark.parametrize("role", sorted(m.ROLES))
def test_all_three_roles_set_and_read_back_subreaper(role):
    k, f, *_ = setup(role=role)
    assert k.enabled == 1 and f.proof.subreaper_before == 0
    assert f.proof.raw()["self"]["pid"] == 100


@pytest.mark.parametrize("fault", ["threads", "identity", "subreaper", "unknown-role"])
def test_role_admission_faults_never_fork(fault):
    k = FakeKernel()
    expected = k.owner
    if fault == "threads":
        k.thread_extra = True
    if fault == "identity":
        expected = replace(expected, start_ticks=999)
    if fault == "subreaper":
        k.fail_subreaper = True
    with pytest.raises(m.OuterRefusal):
        m.initialize_role(
            k, "unknown" if fault == "unknown-role" else "operator", expected
        )
    assert k.forked == 0


def test_linux_adapter_refuses_macos_without_kernel_effects(monkeypatch):
    monkeypatch.setattr(m.sys, "platform", "darwin")
    with pytest.raises(m.OuterRefusal, match="linux-pidfd-unavailable"):
        m.LinuxKernel()


def test_actual_linux_execution_admission_remains_unqualified():
    k, _, spec, binding, _ = setup()
    real = object.__new__(m.LinuxKernel)
    real._admitted_exec = None
    with pytest.raises(m.OuterRefusal, match="unqualified"):
        real.admit_execution(binding, spec)
    with pytest.raises(m.OuterRefusal, match="unqualified"):
        real.fork_capture(spec)
    assert k.forked == 0


def test_before_whole_phase_includes_cleanup_within120():
    _, f, *_ = setup()
    assert f.budget.compact() == {
        k: {"monotonic_ns": n * m.SECOND, "wall_ns": 10**15 + n * m.SECOND}
        for k, n in (("work", 105), ("cleanup", 115), ("gate", 120))
    }


@pytest.mark.parametrize(
    "launch,work,cleanup,gate",
    [(400, 505, 510, 515), (500, 555, 560, 565), (554, 555, 560, 565)],
)
def test_after_reserves_numeric_observer_tail(launch, work, cleanup, gate):
    anchor = m.Clock("boot", 0, 10**15)
    b = m.PhaseBudget.after(
        anchor, m.Clock("boot", launch * m.SECOND, 10**15 + launch * m.SECOND)
    )
    assert (b.work_ns, b.cleanup_ns, b.gate_ns) == tuple(
        n * m.SECOND for n in (work, cleanup, gate)
    )


@pytest.mark.parametrize("axis", ["boot", "monotonic", "wall", "expired"])
def test_after_wrong_original_clock_refuses(axis):
    a = m.Clock("boot", 10 * m.SECOND, 10**15)
    b = (
        replace(a, boot_id="wrong")
        if axis == "boot"
        else replace(a, monotonic_ns=0)
        if axis == "monotonic"
        else replace(a, wall_ns=a.wall_ns - 1)
        if axis == "wall"
        else replace(
            a,
            monotonic_ns=a.monotonic_ns + 555 * m.SECOND,
            wall_ns=a.wall_ns + 555 * m.SECOND,
        )
    )
    with pytest.raises(m.OuterRefusal):
        m.PhaseBudget.after(a, b)


def test_budget_cannot_be_renewed_by_constructing_looser_instance():
    k, f, *_ = setup()
    with pytest.raises(m.OuterRefusal, match="altered-phase-budget"):
        m.OwnedFamily(k, f.proof, replace(f.budget, cleanup_ns=999 * m.SECOND))


@pytest.mark.parametrize(
    "field,value",
    [
        ("python", "/bin/sh"),
        ("control_root", "/x/../bad"),
        ("launch_path", "relative"),
        ("launch_sha256", "x"),
        ("phase", "arbitrary-command"),
    ],
)
def test_fixed_command_contract_never_accepts_shell_or_arbitrary_role(field, value):
    _, _, spec, _, _ = setup()
    with pytest.raises(m.OuterRefusal):
        replace(spec, **{field: value}).argv()


def test_exec_gate_follows_bound_identity_and_fixed_contract():
    k, f, spec, binding, parent = setup()
    f.start_capture(spec, binding, parent)
    assert k.released and f.events[0]["gate_closed"] is True
    assert f.events[0]["pidfd_target_pid"] == 200
    assert f.events[0]["child"]["pgid"] == f.events[0]["child"]["sid"] == 200
    result = f.finish(parent)
    assert result["natural_complete"] and result["reason"] is None
    assert result["child_terminal"]["waitid"] == {
        "pid": 200,
        "code": "CLD_EXITED",
        "status": 0,
    }
    assert result["child_terminal"]["waitpid"] == {"pid": 200, "status": 0}
    assert result["family_closed"]["direct_children"] == []
    assert (
        result["signals"]
        == result["adopted_children"]
        == result["other_terminals"]
        == []
    )
    assert result["output"]["stdout_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert k.reaped == [200] and not k.children()


@pytest.mark.parametrize(
    "field",
    [
        "intent_sha256",
        "source_sha256",
        "session_admission_sha256",
        "budget_sha256",
        "exec_sha256",
    ],
)
def test_absent_or_wrong_binding_prevents_child_release(field):
    k, f, spec, binding, parent = setup()
    with pytest.raises(m.OuterRefusal):
        f.start_capture(spec, replace(binding, **{field: "bad"}), parent)
    assert not k.forked


def test_unqualified_session_is_not_made_runnable_by_hashes():
    k, f, spec, binding, parent = setup()
    k.admitted = False
    with pytest.raises(m.OuterRefusal, match="admission-refused"):
        f.start_capture(spec, binding, parent)
    assert not k.forked


def test_missing_ready_gate_child_is_reaped_and_never_complete():
    k, f, spec, binding, parent = setup()
    k.ready = False
    with pytest.raises(m.OuterRefusal, match="child-not-ready"):
        f.start_capture(spec, binding, parent)
    assert not k.released
    result = f.finish(parent)
    assert not result["natural_complete"] and result["reason"] == "start-not-complete"
    assert k.reaped == [200]


def test_late_work_cannot_become_natural_success_just_because_child_exited():
    k, f, spec, binding, parent = setup()
    f.start_capture(spec, binding, parent)
    k.t = 105 * m.SECOND
    result = f.finish(parent)
    assert not result["natural_complete"] and result["reason"] == "work-deadline"


@pytest.mark.parametrize("role", ["guardian", "operator", "supervisor"])
def test_setsided_orphan_is_adopted_and_cleanup_is_not_success(role):
    k, f, spec, binding, parent = setup(role=role)
    k.orphans = 1
    f.start_capture(spec, binding, parent)
    result = f.finish(parent)
    assert result["reason"] == "orphan-adopted" and not result["natural_complete"]
    assert result["adopted_children"][0]["child"]["sid"] == 300
    assert set(k.reaped) == {200, 300} and not k.children()
    assert (300, signal.SIGKILL) in k.signalled


@pytest.mark.parametrize("role", sorted(m.ROLES))
def test_parent_death_after_release_reparents_self_but_owned_cleanup_continues(role):
    k, f, spec, binding, parent = setup(role=role)
    k.exit_at = 999 * m.SECOND
    f.start_capture(spec, binding, parent)
    k.parent_dead_at = 0
    result = f.finish(parent)
    assert result["reason"] == "parent-died" and not result["natural_complete"]
    assert k.reaped == [200] and k.signalled == [(200, signal.SIGTERM)]


def test_already_dead_parent_prevents_new_work():
    k, f, spec, binding, parent = setup()
    k.parent_dead_at = 0
    with pytest.raises(m.OuterRefusal, match="parent-handle-binding"):
        f.start_capture(spec, binding, parent)
    assert k.forked == 0


def test_stubborn_child_gets_bounded_term_then_kill_once_each():
    k, f, spec, binding, parent = setup()
    k.exit_at = 999 * m.SECOND
    k.ignore_term = True
    f.start_capture(spec, binding, parent)
    k.t = 105 * m.SECOND
    result = f.finish(parent)
    assert result["reason"] == "work-deadline" and k.t < 115 * m.SECOND
    assert k.signalled == [(200, signal.SIGTERM), (200, signal.SIGKILL)]
    assert k.reaped == [200]


def test_uninterruptible_child_never_reports_empty_family_or_success():
    k, f, spec, binding, parent = setup()
    k.exit_at = 999 * m.SECOND
    k.ignore_term = k.ignore_kill = True
    f.start_capture(spec, binding, parent)
    k.t = 114 * m.SECOND
    result = f.finish(parent)
    assert not result["natural_complete"] and result["family_closed"] is None
    assert k.t == 115 * m.SECOND and k.children() == (200,)


def test_pid_reuse_or_owner_drift_never_signals_new_process():
    k, f, spec, binding, parent = setup()
    k.exit_at = 999 * m.SECOND
    f.start_capture(spec, binding, parent)
    k.processes[200] = replace(k.processes[200], start_ticks=999)
    k.t = 105 * m.SECOND
    with pytest.raises(m.OuterRefusal, match="signal-owner-drift"):
        f.finish(parent)
    assert not k.signalled


def test_output_overflow_is_bounded_failure_not_secret_receipt():
    k, f, spec, binding, parent = setup(maximum_output=2)
    k.stdout = [b"private-secret"]
    k.exit_at = 999 * m.SECOND
    f.start_capture(spec, binding, parent)
    result = f.finish(parent)
    assert result["reason"] == "output-limit" and not result["natural_complete"]
    assert "private-secret" not in str(result) and not k.children()


def test_waitid_alone_does_not_prove_matching_actual_reap():
    k, f, spec, binding, parent = setup()
    k.wait_mismatch = True
    f.start_capture(spec, binding, parent)
    with pytest.raises(m.OuterRefusal, match="wait-reap-drift"):
        f.finish(parent)


@pytest.mark.parametrize("fault", ["thread", "clock", "boot"])
def test_midphase_observation_drift_never_manufactures_completion(fault):
    k, f, spec, binding, parent = setup()
    f.start_capture(spec, binding, parent)
    if fault == "thread":
        k.thread_extra = True
    elif fault == "clock":
        k.wall_jump = -1
    else:
        k.owner = replace(k.owner, boot_id="reboot")
    with pytest.raises(m.OuterRefusal):
        f.finish(parent)


def test_foreign_existing_child_prevents_fresh_family():
    k, f, spec, binding, parent = setup()
    k.processes[999] = replace(k.owner, pid=999, ppid=100, start_ticks=12)
    with pytest.raises(m.OuterRefusal, match="family-not-fresh"):
        f.start_capture(spec, binding, parent)
    assert not k.forked and not k.signalled


def test_failed_pidfd_bind_closes_gate_then_owned_child_can_be_adopted_for_cleanup():
    k, f, spec, binding, parent = setup()
    k.pidfd_wrong = True
    with pytest.raises(m.OuterRefusal, match="pidfd-child-raced"):
        f.start_capture(spec, binding, parent)
    assert not k.released and 900 in k.closed
    k.pidfd_wrong = False
    result = f.finish(parent)
    assert not result["natural_complete"] and result["reason"] == "start-not-complete"
    assert k.reaped == [200] and not k.children()


def test_capture_limits_are_fixed_contract_material():
    _, _, spec, binding, _ = setup()
    assert spec.contract()["resources"] == m.CAPTURE_LIMITS
    assert binding.exec_sha256 == m.digest(spec.contract())
    assert m.CAPTURE_LIMITS["nice"] == 19
    assert m.CAPTURE_LIMITS["cpu_affinity_count"] == 1
    assert m.CAPTURE_LIMITS["address_space_bytes"] == 512 * 2**20


def test_parent_death_during_child_readiness_never_releases_exec(monkeypatch):
    k, f, spec, binding, parent = setup()
    original = k.fork_capture

    def fork_with_parent_death(spec):
        result = original(spec)
        k.t = m.SECOND
        k.parent_dead_at = k.t
        return result

    monkeypatch.setattr(k, "fork_capture", fork_with_parent_death)
    with pytest.raises(m.OuterRefusal, match="parent-died-before-release"):
        f.start_capture(spec, binding, parent)
    assert not k.released and 900 in k.closed
    result = f.finish(parent)
    assert not result["natural_complete"] and k.reaped == [200]
    assert not k.children() and f.budget.cleanup_ns == 115 * m.SECOND


def test_actual_family_producer_matches_pure_completion_consumer():
    from scripts import strength_freshness_cpu_completion as completion

    budget = m.PhaseBudget.before(m.Clock("boot", m.SECOND, 10**15 + m.SECOND))
    _, family, spec, binding, parent = setup(budget=budget)
    family.start_capture(spec, binding, parent)
    result = family.finish(parent)
    completion._family(
        result,
        role="guardian",
        owner=result["role_admission"]["self"],
        child=result["child_started"]["child"],
        exec_hash=binding.exec_sha256,
        binding=result["binding"],
        budget=family.budget.compact(),
    )
