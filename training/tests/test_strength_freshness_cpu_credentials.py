"""Closed credential reads against private proc-shaped files, no live /proc."""

from dataclasses import replace
import os

import pytest
from scripts import strength_freshness_cpu_readonly as r
from tests.test_strength_freshness_cpu_readonly import BOOT, GROUP, UNIT, proc_stat

PID = 31
ADMISSION = r.ProcessAdmission(PID, 90, GROUP)


class PrivateProc(r.System):
    def __init__(self):
        self.ns = 100_000_000_000

    def monotonic_ns(self):
        return self.ns

    def wall_ns(self):
        return 10**18 + self.ns

    def read_file(self, path, maximum, deadline, *, tail=False):
        assert path == "/proc/sys/kernel/random/boot_id", "no live host read"
        self._time(deadline)
        self.ns += 1
        return (BOOT + "\n").encode(), {}


def setup(
    tmp_path,
    monkeypatch,
    status=b"Name:\tprivate-process-name\nUid:\t1000\t1001\t1002\t1003\n",
):
    proc = tmp_path.resolve() / "proc-fixture"
    proc.mkdir()
    (proc / "stat").write_bytes(proc_stat())
    (proc / "cgroup").write_bytes(("0::" + GROUP + "\n").encode())
    (proc / "status").write_bytes(status)
    actual_open = os.open
    opened = []

    def fixed_open(path, flags, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            assert path in {"stat", "cgroup", "status"}
        else:
            assert path == "/proc/31", "only one registered proc directory"
            path = str(proc)
        fd = actual_open(path, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", fixed_open)
    backend = PrivateProc()
    scope = r.ReadScope({UNIT: "service"}, (), {}, {}, {})
    io = r.ReadOnlyIO(scope, backend=backend, deadline=200)
    # Explicit already admitted cgroup token; members() itself has separate tests.
    io._admitted.add(ADMISSION)
    return proc, backend, io, opened


def test_real_fixed_descriptor_read_returns_uid_vector_with_safe_audit(
    tmp_path, monkeypatch
):
    proc, backend, io, opened = setup(tmp_path, monkeypatch)
    observation = io.process_credentials(ADMISSION)
    assert observation.value == {
        "pid": 31,
        "start_ticks": 90,
        "ppid": 12,
        "cgroup": GROUP,
        "uids": {"real": 1000, "effective": 1001, "saved": 1002, "filesystem": 1003},
    }
    audit = observation.audit
    assert audit["operation"] == "process-credentials" and audit["subject"] == "31"
    assert audit["read_end"]["monotonic_ns"] == backend.ns
    assert audit["read_start"]["monotonic_ns"] < audit["read_end"]["monotonic_ns"]
    assert audit["raw_encoding"] == "component-pins-v1"
    assert audit["components"]["status"] == {
        "sha256": r.sha((proc / "status").read_bytes()),
        "bytes": (proc / "status").stat().st_size,
    }
    assert audit["raw"]["sha256"] == r.sha(r.encoded(audit["components"]))
    assert "private-process-name" not in repr(observation)
    assert "private-process-name" not in str(observation.value)
    assert b"private-process-name" not in r.encoded(audit)
    assert io._consumed == sum(x["bytes"] for x in audit["components"].values()) + 2 * (
        len(BOOT) + 1
    )
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize(
    "token",
    [
        31,
        None,
        replace(ADMISSION, start_ticks=91),
        replace(ADMISSION, cgroup="/foreign"),
    ],
)
def test_unadmitted_token_cannot_read_or_probe_a_pid(tmp_path, monkeypatch, token):
    _, backend, io, opened = setup(tmp_path, monkeypatch)
    with pytest.raises(r.ReadRefusal, match="credential-process-not-admitted"):
        io.process_credentials(token)
    assert opened == [] and io._consumed == 0 and backend.ns == 100_000_000_000


@pytest.mark.parametrize(
    "raw",
    [
        b"Name: x\n",
        b"Uid: 0 0 0\n",
        b"Uid: 0 0 0 0 0\n",
        b"Uid: -1 0 0 0\n",
        b"Uid: 0 0 0 4294967296\n",
        b"Uid: 0 0 0 0\nUid: 0 0 0 0\n",
        b"Uid: 0 0 0 0",
        b"Uid: 0 0 0 1.0\n",
        b"Uid: 0 0 0 \xff\n",
    ],
)
def test_malformed_missing_duplicate_or_truncated_uid_refuses(
    tmp_path, monkeypatch, raw
):
    _, _, io, _ = setup(tmp_path, monkeypatch, raw)
    with pytest.raises(r.ReadRefusal, match="credential-uid"):
        io.process_credentials(ADMISSION)


@pytest.mark.parametrize("kind", ["birth", "cgroup", "parent", "dead"])
def test_changed_owner_across_status_read_refuses(tmp_path, monkeypatch, kind):
    proc, _, io, _ = setup(tmp_path, monkeypatch)
    actual_read = os.read
    triggered = False

    def change(fd, n):
        nonlocal triggered
        raw = actual_read(fd, n)
        if b"Uid:" in raw and not triggered:
            triggered = True
            if kind == "birth":
                (proc / "stat").write_bytes(proc_stat(start=91))
            elif kind == "parent":
                (proc / "stat").write_bytes(proc_stat(ppid=14))
            elif kind == "dead":
                (proc / "stat").write_bytes(proc_stat(state="Z"))
            else:
                (proc / "cgroup").write_bytes(b"0::/foreign\n")
        return raw

    monkeypatch.setattr(os, "read", change)
    with pytest.raises(r.ReadRefusal):
        io.process_credentials(ADMISSION)
    assert triggered


def test_wrong_initial_birth_refuses_before_reading_status(tmp_path, monkeypatch):
    proc, _, io, _ = setup(tmp_path, monkeypatch)
    (proc / "stat").write_bytes(proc_stat(start=99))
    (proc / "status").unlink()
    with pytest.raises(r.ReadRefusal, match="credential-admission-drift"):
        io.process_credentials(ADMISSION)


@pytest.mark.parametrize("fault", ["symlink", "fifo", "limit", "deadline", "budget"])
def test_file_budget_and_original_deadline_fail_closed(tmp_path, monkeypatch, fault):
    proc, backend, io, opened = setup(tmp_path, monkeypatch)
    status = proc / "status"
    if fault == "symlink":
        status.unlink()
        status.symlink_to(proc / "stat")
    elif fault == "fifo":
        status.unlink()
        os.mkfifo(status)
    elif fault == "limit":
        status.write_bytes(b"Name: private\n" + b"x" * r.MAX_PROC_BYTES)
    elif fault == "budget":
        io.maximum_bytes = 50
    else:
        backend.ns = 200_000_000_000
    with pytest.raises((r.ReadRefusal, OSError)):
        io.process_credentials(ADMISSION)
    if fault == "deadline":
        assert not opened and io._consumed == 0
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_deadline_during_status_read_still_charges_returned_bytes(
    tmp_path, monkeypatch
):
    _, backend, io, _ = setup(tmp_path, monkeypatch)
    actual_read = os.read
    total = 0

    def late(fd, n):
        nonlocal total
        raw = actual_read(fd, n)
        total += len(raw)
        if b"Uid:" in raw:
            backend.ns = 201_000_000_000
        return raw

    monkeypatch.setattr(os, "read", late)
    with pytest.raises(r.ReadRefusal, match="absolute-read-deadline"):
        io.process_credentials(ADMISSION)
    assert io._consumed == total + len(BOOT) + 1
