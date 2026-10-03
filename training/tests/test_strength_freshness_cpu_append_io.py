"""Real private-file reads with simulated clocks; no host/runtime qualification."""

from dataclasses import replace
import os

import pytest
from scripts import strength_freshness_cpu_readonly as r

BOOT = "11111111-1111-1111-1111-111111111111"
STAT_KEYS = {
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "bytes",
    "mtime_ns",
    "ctime_ns",
    "links",
}


class LocalFiles(r.System):
    """Only temporary metrics files are real; clock/boot are explicit fixtures."""

    def __init__(self):
        self.ns = 1_000_000_000

    def monotonic_ns(self):
        return self.ns

    def wall_ns(self):
        return 10**18 + self.ns

    def read_file(self, path, maximum, deadline, *, tail=False):
        assert path == "/proc/sys/kernel/random/boot_id", "no real host metadata"
        self._time(deadline)
        self.ns += 1
        return (BOOT + "\n").encode(), {}


def make_io(
    tmp_path,
    data=b'{"private":"metric-value"}\n',
    *,
    cap=512 * 1024,
    budget=1024 * 1024,
):
    path = tmp_path.resolve() / "metrics.jsonl"
    path.write_bytes(data)
    path.chmod(0o600)
    scope = r.ReadScope(
        {"fixture.service": "service"},
        (),
        {},
        {"metrics": r.FileKey(str(path), cap)},
        {},
    )
    backend = LocalFiles()
    io = r.ReadOnlyIO(scope, deadline=100, backend=backend, maximum_bytes=budget)
    return path, backend, io


def read(io, fence, start=0, end=None):
    return io.read_append_range(
        "metrics",
        file_identity=fence.value["file_identity"],
        start=start,
        end=fence.value["size_at_fstat"] if end is None else end,
    )


def test_exact_private_public_schema_and_actual_end_clock(tmp_path):
    path, backend, io = make_io(tmp_path)
    fence = io.append_fence("metrics")
    part = read(io, fence)
    assert set(fence.value) == {
        "registered_key",
        "file_identity",
        "size_at_fstat",
        "prefix",
        "line_boundary",
    }
    assert set(part.value) == {
        "registered_key",
        "file_identity",
        "start",
        "end",
        "sha256",
        "raw",
        "size_before",
        "size_after",
    }
    assert fence.value["prefix"]["raw"] == part.value["raw"] == path.read_bytes()
    assert fence.value["line_boundary"] is True
    assert part.value["file_identity"] == {
        "path": str(path),
        "device": path.stat().st_dev,
        "inode": path.stat().st_ino,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "mode": 0o600,
    }
    for item, operation in ((fence, "append-fence"), (part, "append-range")):
        audit = item.audit
        assert set(audit) == {
            "operation",
            "subject",
            "read_start",
            "read_end",
            "raw",
            "named_before",
            "stat_before",
            "stat_after",
            "named_after",
            "offset",
            "end_offset",
        }
        assert audit["operation"] == operation and audit["subject"] == "metrics"
        assert audit["read_end"]["monotonic_ns"] > audit["read_start"]["monotonic_ns"]
        assert audit["raw"] == {
            "sha256": r.sha(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for field in ("named_before", "stat_before", "stat_after", "named_after"):
            assert set(audit[field]) == STAT_KEYS
        public = r.encoded(audit)
        assert b"metric-value" not in public and str(path).encode() not in public
        assert b"metric-value" not in repr(item).encode()
    assert part.audit["read_end"]["monotonic_ns"] == backend.ns


@pytest.mark.parametrize(
    "data,boundary", [(b"", True), (b"one\n", True), (b"one\npartial", False)]
)
def test_empty_complete_and_partial_fences(tmp_path, data, boundary):
    _, _, io = make_io(tmp_path, data)
    f = io.append_fence("metrics")
    assert f.value["prefix"] == {
        "start": 0,
        "end": len(data),
        "sha256": r.sha(data),
        "raw": data,
    }
    assert f.value["line_boundary"] is boundary
    assert read(io, f).value["raw"] == data


@pytest.mark.parametrize("cap,size", [(17, 30), (100_000, 100_001)])
def test_prefix_keeps_exact_bounded_bytes_without_line_skipping(tmp_path, cap, size):
    data = b"x" * size
    _, _, io = make_io(tmp_path, data, cap=cap)
    f = io.append_fence("metrics")
    wanted = min(cap, r.MAX_APPEND_PREFIX)
    assert f.value["prefix"]["start"] == size - wanted
    assert f.value["prefix"]["end"] == size
    assert f.value["prefix"]["raw"] == data[-wanted:]
    assert f.value["line_boundary"] is False


@pytest.mark.parametrize(
    "key,path_name", [("other", "metrics.jsonl"), ("metrics", "other.jsonl")]
)
def test_only_exact_metrics_slot_is_admitted(tmp_path, key, path_name):
    path, backend, io = make_io(tmp_path)
    alternate = path.with_name(path_name)
    io.scope = replace(io.scope, tails={key: r.FileKey(str(alternate))})
    for call in (
        lambda: io.append_fence(key),
        lambda: io.read_append_range(key, file_identity={}, start=0, end=1),
    ):
        with pytest.raises(r.ReadRefusal, match="registered-learner-metrics"):
            call()
    assert backend.ns == 1_000_000_000


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (2, 1), (True, 2), (0, False), (0.0, 2), (0, 524289)]
)
def test_range_rejects_noninteger_and_out_of_bounds_before_read(tmp_path, start, end):
    _, _, io = make_io(tmp_path)
    f = io.append_fence("metrics")
    used = io._consumed
    with pytest.raises(r.ReadRefusal, match="append-range-bounds"):
        read(io, f, start, end)
    assert io._consumed == used


def test_range_cannot_extend_past_initial_eof(tmp_path):
    _, _, io = make_io(tmp_path)
    f = io.append_fence("metrics")
    with pytest.raises(r.ReadRefusal, match="append-range-bounds"):
        read(io, f, end=f.value["size_at_fstat"] + 1)


@pytest.mark.parametrize(
    "field,new",
    [
        ("path", "/other/metrics.jsonl"),
        ("inode", True),
        ("mode", 0o10000),
        ("uid", -1),
        ("extra", 1),
    ],
)
def test_identity_shape_is_strict(tmp_path, field, new):
    _, _, io = make_io(tmp_path)
    f = io.append_fence("metrics")
    f.value["file_identity"][field] = new
    with pytest.raises(r.ReadRefusal, match="append-expected-identity"):
        read(io, f)


@pytest.mark.parametrize(
    "change", ["replace", "mode", "symlink", "fifo", "directory", "parent_alias"]
)
def test_named_identity_and_nonregular_aliases_refuse(tmp_path, change):
    path, _, io = make_io(tmp_path)
    f = io.append_fence("metrics")
    if change == "mode":
        path.chmod(0o640)
    elif change == "parent_alias":
        alias = path.parent / "alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        io.scope = replace(
            io.scope, tails={"metrics": r.FileKey(str(alias / path.name))}
        )
    else:
        path.rename(path.with_suffix(".old"))
        if change == "replace":
            path.write_bytes(b"replacement\n")
        elif change == "symlink":
            other = path.with_name("other")
            other.write_bytes(b"data\n")
            path.symlink_to(other)
        elif change == "fifo":
            os.mkfifo(path)
        else:
            path.mkdir()
    with pytest.raises(r.ReadRefusal):
        io.append_fence("metrics") if change == "parent_alias" else read(io, f)


def test_fixed_fence_and_range_do_not_follow_concurrent_append(tmp_path, monkeypatch):
    data = b"a" * 70000 + b"\n"
    path, backend, io = make_io(tmp_path, data)
    f = io.append_fence("metrics")
    pread = os.pread
    offsets = []

    def growing(fd, size, offset):
        block = pread(fd, min(size, 5000), offset)
        offsets.append((offset, len(block)))
        if len(offsets) == 1:
            with path.open("ab") as writer:
                writer.write(b"later\n")
        backend.ns += 10
        return block

    monkeypatch.setattr(os, "pread", growing)
    result = read(io, f)
    assert result.value["raw"] == data
    assert result.value["end"] == len(data)
    assert result.value["size_after"] == len(data) + 6
    assert offsets[0][0] == 0 and all(
        a + n == b for (a, n), (b, _) in zip(offsets, offsets[1:])
    )
    assert result.audit["read_end"]["monotonic_ns"] == backend.ns


@pytest.mark.parametrize(
    "fault", ["truncate", "replace", "mode", "rewrite", "deadline", "short"]
)
def test_postread_fault_refuses_closes_fd_and_charges_read_bytes(
    tmp_path, monkeypatch, fault
):
    path, backend, io = make_io(tmp_path, b"private-row\n")
    f = io.append_fence("metrics")
    pread = os.pread
    calls = []
    used = io._consumed

    def racing(fd, size, offset):
        data = pread(fd, size, offset)
        calls.append((fd, len(data)))
        if fault == "truncate":
            path.write_bytes(b"")
        elif fault == "replace":
            path.rename(path.with_suffix(".old"))
            path.write_bytes(b"private-row\n")
        elif fault == "mode":
            path.chmod(0o640)
        elif fault == "rewrite":
            path.write_bytes(b"changed-row\n")
            os.utime(path, ns=(1, 1))
        elif fault == "deadline":
            backend.ns = 101_000_000_000
        elif fault == "short":
            return b""
        return data

    monkeypatch.setattr(os, "pread", racing)
    with pytest.raises(r.ReadRefusal):
        read(io, f)
    expected_data = 0 if fault == "short" else len(b"private-row\n")
    assert io._consumed == used + len(BOOT) + 1 + expected_data
    with pytest.raises(OSError):
        os.fstat(calls[0][0])


def test_before_open_name_fd_replacement_is_rejected(tmp_path, monkeypatch):
    path, _, io = make_io(tmp_path)
    actual_open = os.open

    def swap(name, flags, *args, **kwargs):
        if name == str(path):
            path.rename(path.with_suffix(".old"))
            path.write_bytes(b"replacement\n")
        return actual_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(r.ReadRefusal, match="append-file-identity"):
        io.append_fence("metrics")


def test_rereads_share_input_budget_and_refuse_before_excess_read(
    tmp_path, monkeypatch
):
    data = b"x" * 70000 + b"\n"
    _, _, io = make_io(tmp_path, data)
    f = io.append_fence("metrics")
    first = read(io, f)
    consumed = io._consumed
    second = read(io, f)
    assert second.value["raw"] == first.value["raw"]
    assert io._consumed - consumed == len(data) + 2 * (len(BOOT) + 1)
    io.maximum_bytes = io._consumed + len(BOOT) + 1 + len(data) - 1
    calls = []
    monkeypatch.setattr(os, "pread", lambda *args: calls.append(args) or b"")
    with pytest.raises(r.ReadRefusal, match="capture-byte-budget"):
        read(io, f)
    assert calls == []


def test_rewrite_between_reads_is_exposed_as_changed_bytes_not_history_proof(tmp_path):
    path, _, io = make_io(tmp_path, b"original\n")
    f = io.append_fence("metrics")
    original = read(io, f)
    path.write_bytes(b"modified\n")
    reread = read(io, f)
    assert reread.value["file_identity"] == original.value["file_identity"]
    assert reread.value["sha256"] != original.value["sha256"]
    # IO reports evidence; the pure consumer rejects this mismatch. A hidden
    # identical truncate/regrow between reads is explicitly not observable.


def test_fence_keeps_initial_eof_when_append_occurs_during_read(tmp_path, monkeypatch):
    path, _, io = make_io(tmp_path, b"initial-partial")
    pread = os.pread

    def append(fd, size, offset):
        data = pread(fd, size, offset)
        with path.open("ab") as writer:
            writer.write(b"-completed\n")
        return data

    monkeypatch.setattr(os, "pread", append)
    fence = io.append_fence("metrics")
    assert fence.value["size_at_fstat"] == len(b"initial-partial")
    assert fence.value["prefix"]["raw"] == b"initial-partial"
    assert fence.value["line_boundary"] is False
    assert fence.audit["stat_after"]["bytes"] == path.stat().st_size


def test_partial_then_io_exception_retains_consumption_and_closes_fd(
    tmp_path, monkeypatch
):
    _, _, io = make_io(tmp_path, b"x" * 70000)
    fence = io.append_fence("metrics")
    used = io._consumed
    pread = os.pread
    fds = []

    def failing(fd, size, offset):
        fds.append(fd)
        if len(fds) == 2:
            raise OSError("simulated read failure")
        return pread(fd, size, offset)

    monkeypatch.setattr(os, "pread", failing)
    with pytest.raises(OSError, match="simulated read failure"):
        read(io, fence)
    assert io._consumed == used + len(BOOT) + 1 + 65536
    with pytest.raises(OSError):
        os.fstat(fds[0])


def test_expired_original_deadline_does_not_begin_new_read(tmp_path, monkeypatch):
    _, backend, io = make_io(tmp_path)
    backend.ns = 100_000_000_000
    monkeypatch.setattr(os, "open", lambda *args: pytest.fail("read after deadline"))
    with pytest.raises(r.ReadRefusal, match="absolute-read-deadline"):
        io.append_fence("metrics")
    assert io._consumed == 0
