"""Actual private file/proc reads and fake child syscalls; no target IO."""

from pathlib import Path
import os
import signal
from types import SimpleNamespace

import pytest

from scripts import strength_freshness_cpu_readonly as r
from tests.test_strength_freshness_cpu_readonly import BOOT, GROUP, proc_stat


class Files(r.System):
    def __init__(self, root):
        self.root = root.resolve()
        self.ns = 100 * 10**9
        self.boot = self.root / "boot-id"
        self.boot.write_bytes((BOOT + "\n").encode())

    def monotonic_ns(self):
        return self.ns

    def wall_ns(self):
        return 10**18 + self.ns

    def read_file(self, path, maximum, deadline, *, tail=False, charge, allowance):
        actual = str(self.boot) if path == "/proc/sys/kernel/random/boot_id" else path
        assert Path(actual).is_relative_to(self.root), "no host fallback"
        return super().read_file(
            actual, maximum, deadline, tail=tail, charge=charge, allowance=allowance
        )


def setup(tmp_path, budget=2**20):
    backend = Files(tmp_path)
    path = backend.root / "publication.json"
    path.write_bytes(b'{"private":"sentinel"}\n')
    path.chmod(0o600)
    scope = r.ReadScope(
        {"fixture.service": "service"},
        (),
        {"publication": r.FileKey(str(path))},
        {},
        {},
    )
    return (
        r.ReadOnlyIO(scope, backend=backend, deadline=110, maximum_bytes=budget),
        backend,
        path,
    )


def facade(**updates):
    values = {name: getattr(os, name) for name in dir(os)}
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("cap_kind", ["capture", "file"])
def test_actual_generic_syscall_bytes_never_exceed_remaining_or_field_cap(
    tmp_path, monkeypatch, cap_kind
):
    io, backend, path = setup(tmp_path)
    path.write_bytes(b"x" * 100)
    original = os.read
    calls = []

    def read(fd, n):
        data = original(fd, n)
        calls.append((n, len(data)))
        return data

    monkeypatch.setattr(r, "os", facade(read=read))
    charges = []
    maximum, allowance = (200, 7) if cap_kind == "capture" else (7, 200)
    if cap_kind == "file":
        # Simulate a proc/sys pseudo-file size0; actual file bytes are real.
        real = os.fstat

        def fstat(fd):
            st = real(fd)
            values = {k: getattr(st, k) for k in dir(st) if k.startswith("st_")}
            values["st_size"] = 0
            return SimpleNamespace(**values)

        monkeypatch.setattr(r.os, "fstat", fstat)
    with pytest.raises(
        r.ReadRefusal,
        match="capture-byte-budget" if cap_kind == "capture" else "file-output-limit",
    ):
        backend.read_file(
            str(path), maximum, 110, charge=charges.append, allowance=allowance
        )
    assert sum(charges) == sum(count for _, count in calls) == 7
    assert all(size <= 7 for size, _ in calls) and len(calls) == 1


def test_zero_stat_size_still_reads_content_and_eof(tmp_path, monkeypatch):
    _, backend, path = setup(tmp_path)
    content = path.read_bytes()
    charges = []
    real = os.fstat

    def fstat(fd):
        st = real(fd)
        values = {k: getattr(st, k) for k in dir(st) if k.startswith("st_")}
        values["st_size"] = 0
        return SimpleNamespace(**values)

    monkeypatch.setattr(r, "os", facade(fstat=fstat))
    raw, _ = backend.read_file(
        str(path), 1000, 110, charge=charges.append, allowance=1000
    )
    assert raw == content and sum(charges) == len(content)
    assert charges[-1] == 0


def test_boot_bytes_share_counter_and_success_has_no_double_debit(tmp_path):
    io, _, path = setup(tmp_path)
    data = path.read_bytes()
    obs = io.read("publication")
    assert obs.value == data
    assert io._consumed == 2 * len((BOOT + "\n").encode()) + len(data)
    assert "sentinel" not in repr(obs) and "sentinel" not in str(obs.audit)


@pytest.mark.parametrize("failure", ["late", "replaced", "rewritten"])
def test_refused_generic_reads_keep_all_actual_payload_debits(
    tmp_path, monkeypatch, failure
):
    io, backend, path = setup(tmp_path)
    content = path.read_bytes()
    real = os.read
    changed = False
    target_inode = path.stat().st_ino

    def read(fd, n):
        nonlocal changed
        target = os.fstat(fd).st_ino == target_inode
        raw = real(fd, n)
        if target and raw and not changed:
            changed = True
            if failure == "late":
                backend.ns = 110 * 10**9
            elif failure == "replaced":
                new = path.with_name("replacement")
                new.write_bytes(content)
                new.replace(path)
            else:
                path.write_bytes(b"z" * len(content))
        return raw

    monkeypatch.setattr(r, "os", facade(read=read))
    with pytest.raises(r.ReadRefusal):
        io.read("publication")
    assert io._consumed == len((BOOT + "\n").encode()) + len(content)


def test_refused_tail_read_keeps_partial_pread_bytes(tmp_path, monkeypatch):
    io, backend, path = setup(tmp_path)
    content = path.read_bytes()
    real = os.pread

    def pread(fd, n, offset):
        return real(fd, min(n, 3), offset) if offset == 0 else b""

    monkeypatch.setattr(r, "os", facade(pread=pread))
    with pytest.raises(r.ReadRefusal, match="tail-truncated"):
        backend.read_file(
            str(path), 1000, 110, tail=True, charge=io._charge, allowance=1000
        )
    assert io._consumed == 3 and content


def test_publication_has_actual_four_stats_and_fixed_known_size(tmp_path):
    io, backend, path = setup(tmp_path)
    raw = path.read_bytes()
    charges = []
    data, meta = backend.publication_file(
        str(path), len(raw), 110, charge=charges.append, allowance=len(raw)
    )
    assert data == raw and sum(charges) == len(raw)
    assert set(meta) == {
        "named_before",
        "stat_before",
        "stat_after",
        "named_after",
        "offset",
        "end_offset",
    }
    assert all(
        meta[k] == r._stat(path.stat())
        for k in ("named_before", "stat_before", "stat_after", "named_after")
    )
    obs = io.read_publication("publication")
    assert obs.value == raw and obs.audit["operation"] == "read-publication"
    assert io._consumed == 2 * len((BOOT + "\n").encode()) + len(raw)
    assert "sentinel" not in repr(obs) + str(obs.audit)


@pytest.mark.parametrize("key", ["/anything", "missing", "metrics", None])
def test_publication_key_scope_refuses_before_io(tmp_path, key):
    io, _, _ = setup(tmp_path)
    with pytest.raises(r.ReadRefusal, match="unregistered-publication"):
        io.read_publication(key)
    assert io._consumed == 0


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_publication_nonregular_or_alias_does_not_read_payload(tmp_path, kind):
    io, backend, path = setup(tmp_path)
    path.unlink()
    if kind == "symlink":
        path.symlink_to(backend.boot)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    charges = []
    with pytest.raises(r.ReadRefusal):
        backend.publication_file(
            str(path), 1000, 110, charge=charges.append, allowance=1000
        )
    assert charges == []


@pytest.mark.parametrize(
    "failure", ["replace", "grow", "rewrite", "late", "parent-replace"]
)
def test_publication_races_debit_before_refusal(tmp_path, monkeypatch, failure):
    io, backend, path = setup(tmp_path)
    content = path.read_bytes()
    real = os.pread
    done = False

    def pread(fd, n, offset):
        nonlocal done
        raw = real(fd, n, offset)
        if raw and not done:
            done = True
            if failure == "late":
                backend.ns = 110 * 10**9
            elif failure == "parent-replace":
                old = path.parent.with_name(path.parent.name + "-moved")
                path.parent.rename(old)
                path.parent.mkdir()
                path.write_bytes(content)
            elif failure == "replace":
                q = path.with_name("new")
                q.write_bytes(content)
                q.replace(path)
            elif failure == "grow":
                with path.open("ab") as stream:
                    stream.write(b"x")
            else:
                path.write_bytes(b"x" * len(content))
        return raw

    monkeypatch.setattr(r, "os", facade(pread=pread))
    charges = []
    with pytest.raises(r.ReadRefusal):
        backend.publication_file(
            str(path), 1000, 110, charge=charges.append, allowance=1000
        )
    assert sum(charges) == len(content)


def test_atomic_replacement_between_complete_publications_is_allowed(tmp_path):
    io, _, path = setup(tmp_path)
    first = io.read_publication("publication")
    new = path.with_name("new")
    new.write_bytes(b'{"revision":2}\n')
    new.chmod(0o600)
    new.replace(path)
    second = io.read_publication("publication")
    assert first.audit["raw"]["sha256"] != second.audit["raw"]["sha256"]
    assert first.audit["stat_before"]["inode"] != second.audit["stat_before"]["inode"]


def proc_setup(tmp_path, monkeypatch):
    io, backend, _ = setup(tmp_path)
    directory = tmp_path / "proc"
    directory.mkdir()
    values = {
        "stat": proc_stat(),
        "cgroup": ("0::" + GROUP + "\n").encode(),
        "cmdline": b"python\0private\0",
        "environ": b"SECRET=sentinel\0",
        "maps": b"private-maps\n",
    }
    for name, data in values.items():
        (directory / name).write_bytes(data)
    (directory / "exe").symlink_to("/qualified/python")
    (directory / "cwd").symlink_to("/qualified/work")
    real = os.open

    def open_(path, flags, *args, **kwargs):
        if path == "/proc/31":
            path = directory
        if kwargs.get("dir_fd") is None:
            assert Path(path).is_relative_to(tmp_path)
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(r, "os", facade(open=open_))
    return io, backend, directory, values


@pytest.mark.parametrize("kind", ["budget", "field", "late", "drift"])
def test_actual_proc_blocks_are_charged_and_strictly_limited(
    tmp_path, monkeypatch, kind
):
    io, backend, directory, values = proc_setup(tmp_path, monkeypatch)
    read = r.os.read
    calls = []
    done = False

    def observed(fd, n):
        nonlocal done
        raw = read(fd, n)
        calls.append((n, len(raw)))
        if raw and not done:
            done = True
            if kind == "late":
                backend.ns = 110 * 10**9
        return raw

    monkeypatch.setattr(r.os, "read", observed)
    if kind == "field":
        monkeypatch.setattr(r, "MAX_PROC_BYTES", 7)
    expected = r.ProcessAdmission(31, 91 if kind == "drift" else 90, GROUP)
    with pytest.raises(r.ReadRefusal):
        backend.proc(
            31,
            110,
            details=True,
            expected=expected,
            charge=io._charge,
            allowance=7 if kind == "budget" else 10000,
        )
    assert io._consumed == sum(count for _, count in calls) > 0
    if kind in ("budget", "field"):
        assert io._consumed == 7 and max(n for n, _ in calls) <= 7
    if kind == "drift":
        assert io._consumed == len(values["stat"]) + len(values["cgroup"])


def test_proc_success_is_not_double_charged_and_retains_old_shape(
    tmp_path, monkeypatch
):
    io, _, _, values = proc_setup(tmp_path, monkeypatch)
    raw, before = io._proc(
        31, details=True, expected=r.ProcessAdmission(31, 90, GROUP), maps=True
    )
    assert before["pid"] == 31 and raw["maps"] == values["maps"]
    assert io._consumed == 2 * (len(values["stat"]) + len(values["cgroup"])) + sum(
        len(values[k]) for k in ("cmdline", "environ", "maps")
    )


def query_kernel(monkeypatch, backend, stdout=b"hello", stderr=b"", code=0, late=False):
    pipes = {-101: bytearray(stdout), -102: bytearray(stderr)}
    reads = []
    signals = []
    waited = []

    class Pipe:
        def __init__(self, fd):
            self.fd = fd

        def fileno(self):
            return self.fd

        def close(self):
            pass

    child = SimpleNamespace(
        pid=999, stdout=Pipe(-101), stderr=Pipe(-102), returncode=None
    )

    def wait(timeout):
        child.returncode = code
        waited.append(timeout)
        return code

    child.wait = wait

    def popen(argv, **kwargs):
        assert (
            argv == ("systemctl", "get-default") and kwargs["start_new_session"] is True
        )
        assert kwargs["stdin"] is r.subprocess.DEVNULL
        return child

    class Selector:
        def __init__(self):
            self.entries = {}

        def register(self, fileobj, event, data):
            self.entries[fileobj.fd] = SimpleNamespace(
                fd=fileobj.fd, fileobj=fileobj, data=data
            )

        def unregister(self, fileobj):
            del self.entries[fileobj.fd]

        def get_map(self):
            return self.entries

        def select(self, timeout):
            backend.ns += 1_000_000
            return [(x, 1) for x in list(self.entries.values())]

        def close(self):
            pass

    real_read = os.read

    def read(fd, n):
        if fd not in pipes:
            return real_read(fd, n)
        raw = bytes(pipes[fd][:n])
        del pipes[fd][:n]
        reads.append((fd, n, len(raw)))
        if late and raw:
            backend.ns = 110 * 10**9
        return raw

    def killpg(pid, sig):
        assert pid == 999
        if sig == 0:
            raise ProcessLookupError
        signals.append(sig)

    os_proxy = facade(
        read=read,
        set_blocking=lambda *args: None,
        killpg=killpg,
        waitid=lambda *args: SimpleNamespace(si_pid=999),
        P_PID=1,
        WEXITED=4,
        WNOWAIT=0x1000000,
        WNOHANG=1,
    )
    monkeypatch.setattr(r, "os", os_proxy)
    monkeypatch.setattr(r.subprocess, "Popen", popen)
    monkeypatch.setattr(r.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(
        r,
        "signal",
        SimpleNamespace(
            SIGCHLD=signal.SIGCHLD,
            SIG_DFL=signal.SIG_DFL,
            SIGKILL=signal.SIGKILL,
            getsignal=lambda _: signal.SIG_DFL,
        ),
    )
    return reads, signals, waited


@pytest.mark.parametrize(
    "kind", ["capture", "stdout-cap", "stderr-cap", "late", "nonzero"]
)
def test_query_actual_pipe_bytes_and_owned_cleanup_on_all_refusals(
    tmp_path, monkeypatch, kind
):
    io, backend, _ = setup(tmp_path)
    if kind == "stdout-cap":
        monkeypatch.setattr(r, "MAX_FILE_BYTES", 7)
    if kind == "stderr-cap":
        monkeypatch.setattr(r, "MAX_PROC_BYTES", 7)
    stdout = b"abcdefghij"
    stderr = b"abcdefghij"
    reads, signals, waited = query_kernel(
        monkeypatch,
        backend,
        stdout,
        stderr,
        code=7 if kind == "nonzero" else 0,
        late=kind == "late",
    )
    if kind == "nonzero":
        with pytest.raises(r.ReadRefusal, match="query-returncode"):
            io.query("default-target")
        expected = 2 * len((BOOT + "\n").encode()) + len(stdout) + len(stderr)
        assert io._consumed == expected
    else:
        with pytest.raises(r.ReadRefusal):
            backend.command(
                ("systemctl", "get-default"),
                110,
                charge=io._charge,
                allowance=7 if kind == "capture" else 1000,
            )
        assert io._consumed == sum(x[2] for x in reads)
    assert signals == [signal.SIGKILL] and len(waited) == 1
    if kind == "late":
        assert [fd for fd, _, count in reads if count] == [-101]
        assert io._consumed == len(stdout)
    if kind == "capture":
        assert sum(x[2] for x in reads) == 7 and max(x[1] for x in reads) <= 7
    if kind == "stdout-cap":
        assert sum(x[2] for x in reads if x[0] == -101) == 7
    if kind == "stderr-cap":
        assert sum(x[2] for x in reads if x[0] == -102) == 7


def test_query_success_charges_each_stream_and_boot_once(tmp_path, monkeypatch):
    io, backend, _ = setup(tmp_path)
    query_kernel(monkeypatch, backend, stdout=b"answer", stderr=b"note")
    value = io.query("default-target")
    assert value.value == "answer"
    assert io._consumed == 2 * len((BOOT + "\n").encode()) + 10
    assert value.audit["stderr_bytes"] == 4


def test_old_backend_signature_has_no_silent_compatibility_fallback(
    tmp_path, monkeypatch
):
    io, backend, _ = setup(tmp_path)

    def old(path, maximum, deadline, *, tail=False):
        return b"", {}

    monkeypatch.setattr(backend, "read_file", old)
    with pytest.raises(TypeError):
        io.clock()
    assert io._consumed == 0


def test_existing_credential_stream_has_no_one_byte_overflow_probe(
    tmp_path, monkeypatch
):
    io, backend, _, _ = proc_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(r, "MAX_PROC_BYTES", 7)
    actual = r.os.read
    seen = []

    def read(fd, n):
        data = actual(fd, n)
        seen.append((n, len(data)))
        return data

    monkeypatch.setattr(r.os, "read", read)
    with pytest.raises(r.ReadRefusal, match="credential-proc-limit"):
        backend.proc_credentials(
            r.ProcessAdmission(31, 90, GROUP), 110, charge=io._charge, allowance=1000
        )
    assert io._consumed == sum(n for _, n in seen) == 7
    assert max(n for n, _ in seen) == 7


def test_generic_fifo_is_opened_nonblocking_then_refused(tmp_path, monkeypatch):
    _, backend, path = setup(tmp_path)
    path.unlink()
    os.mkfifo(path)
    actual = os.open

    def open_(target, flags, *args, **kwargs):
        if str(target) == str(path):
            assert flags & os.O_NONBLOCK
        return actual(target, flags, *args, **kwargs)

    monkeypatch.setattr(r, "os", facade(open=open_))
    charged = []
    with pytest.raises(r.ReadRefusal, match="regular-file"):
        backend.read_file(str(path), 1000, 110, charge=charged.append, allowance=1000)
    assert charged == []


def test_publication_regular_to_fifo_open_race_never_blocks(tmp_path, monkeypatch):
    _, backend, path = setup(tmp_path)
    actual = os.open
    opened = []

    def open_(target, flags, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            assert target == path.name and flags & os.O_NONBLOCK
            path.unlink()
            os.mkfifo(path)
        fd = actual(target, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(r, "os", facade(open=open_))
    charged = []
    with pytest.raises(r.ReadRefusal, match="file-raced"):
        backend.publication_file(
            str(path), 1000, 110, charge=charged.append, allowance=1000
        )
    assert charged == []
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_exhausted_generic_cap_does_not_probe_an_uncharged_eof_byte(
    tmp_path, monkeypatch
):
    _, backend, path = setup(tmp_path)
    path.write_bytes(b"1234567")
    actual = os.read
    reads = []

    def read(fd, n):
        data = actual(fd, n)
        reads.append((n, len(data)))
        return data

    monkeypatch.setattr(r, "os", facade(read=read))
    debits = []
    with pytest.raises(r.ReadRefusal, match="file-output-limit"):
        backend.read_file(str(path), 7, 110, charge=debits.append, allowance=1000)
    assert reads == [(7, 7)] and sum(debits) == 7


def test_exhausted_query_allowance_never_reads_even_a_second_eof_pipe(
    tmp_path, monkeypatch
):
    io, backend, _ = setup(tmp_path)
    reads, signals, waited = query_kernel(
        monkeypatch, backend, stdout=b"1234567", stderr=b""
    )
    with pytest.raises(r.ReadRefusal, match="capture-byte-budget"):
        backend.command(
            ("systemctl", "get-default"), 110, charge=io._charge, allowance=7
        )
    assert reads == [(-101, 7, 7)] and io._consumed == 7
    assert signals == [signal.SIGKILL] and len(waited) == 1


def test_failed_reads_do_not_refund_or_reset_capture_budget(tmp_path):
    io, backend, path = setup(tmp_path)
    path.write_bytes(b"x" * 100)
    io.maximum_bytes = 50
    for expected in (7, 14):
        with pytest.raises(r.ReadRefusal, match="capture-byte-budget"):
            backend.read_file(str(path), 1000, 110, charge=io._charge, allowance=7)
        assert io._consumed == expected
