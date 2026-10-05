"""Bounded pinned-file transport, using only real private temporary files."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from scripts import strength_freshness_cpu_capture_request as m


def setup(tmp_path, monkeypatch, raw=b"private-payload" * 10000):
    path = tmp_path.resolve() / "input.json"
    path.write_bytes(raw)
    path.chmod(0o444)
    pin = {"path": str(path), "sha256": m.digest(raw), "bytes": len(raw)}
    clock = [100.0]
    reader = m.PinnedReader(
        deadline=101, owner_uid=os.geteuid(), monotonic=lambda: clock[0]
    )
    calls = []
    closed = []
    private_os = SimpleNamespace(**vars(os))

    def read(fd, count):
        calls.append(count)
        return os.read(fd, count)

    def close(fd):
        closed.append(fd)
        return os.close(fd)

    private_os.read = read
    private_os.close = close
    # Instrument this module only, never the shared process-wide os module.
    monkeypatch.setattr(m, "os", private_os)
    return path, pin, reader, clock, calls, closed, private_os


@pytest.mark.parametrize("size", [1, 65536, 65537, m.FILE_LIMIT, m.PROVENANCE_LIMIT])
def test_exact_pinned_endpoint_has_no_growth_or_eof_probe(tmp_path, monkeypatch, size):
    raw = b"p" * size
    path, pin, reader, _, calls, closed, _ = setup(tmp_path, monkeypatch, raw)
    reader.charge(m.METADATA_BUDGET - size)
    limit = m.PROVENANCE_LIMIT if size > m.FILE_LIMIT else m.FILE_LIMIT
    assert reader.read(pin, root=path.parent, limit=limit) == raw
    assert sum(calls) == size and all(0 < n <= 65536 for n in calls)
    assert reader.consumed == m.METADATA_BUDGET
    assert len(closed) == 1
    assert set(reader.audit[0]) == {"pin", "device", "inode", "mode", "uid"}
    assert reader.audit[0]["pin"] == pin


@pytest.mark.parametrize("remaining", [0, 1, 65535, 65536, 70000])
def test_partial_budget_cannot_obtain_extra_file_bytes(
    tmp_path, monkeypatch, remaining
):
    _, pin, reader, _, calls, closed, _ = setup(tmp_path, monkeypatch)
    reader.charge(m.METADATA_BUDGET - remaining)
    with pytest.raises(m.RequestRefusal, match="^metadata-budget$"):
        reader.read(pin)
    assert sum(calls) == remaining
    assert reader.consumed == m.METADATA_BUDGET
    assert not reader.audit and len(closed) == 1
    count = len(calls)
    with pytest.raises(m.RequestRefusal, match="^metadata-budget$"):
        reader.read(pin)
    assert len(calls) == count and reader.consumed == m.METADATA_BUDGET


def test_short_os_reads_continue_with_same_original_ledger(tmp_path, monkeypatch):
    raw = b"private-body" * 10
    _, pin, reader, _, calls, closed, private_os = setup(tmp_path, monkeypatch, raw)
    actual = []

    def read(fd, count):
        calls.append(count)
        block = os.read(fd, min(count, 7))
        actual.append(len(block))
        return block

    private_os.read = read
    reader.charge(m.METADATA_BUDGET - len(raw))
    assert reader.read(pin) == raw
    assert sum(actual) == len(raw) and all(actual)
    assert calls[-1] == actual[-1]
    assert reader.consumed == m.METADATA_BUDGET and len(closed) == 1


@pytest.mark.parametrize("size", [65536, 65537])
def test_returned_block_charged_before_immediate_deadline_refusal(
    tmp_path, monkeypatch, size
):
    _, pin, reader, clock, calls, closed, private_os = setup(
        tmp_path, monkeypatch, b"p" * size
    )

    def read(fd, count):
        calls.append(count)
        block = os.read(fd, count)
        clock[0] = 101
        return block

    private_os.read = read
    with pytest.raises(m.RequestRefusal, match="^metadata-deadline$"):
        reader.read(pin)
    assert calls == [65536] and reader.consumed == 65536
    assert reader.deadline == 101 and reader.audit == [] and len(closed) == 1


@pytest.mark.parametrize(
    "fault", ["hash", "replace", "grow", "truncate", "eof", "stat"]
)
def test_bytes_remain_charged_on_later_file_refusal(tmp_path, monkeypatch, fault):
    raw = b"private-payload"
    path, pin, reader, _, calls, closed, private_os = setup(tmp_path, monkeypatch, raw)
    captured = []
    if fault == "hash":
        pin["sha256"] = "0" * 64

    def read(fd, count):
        calls.append(count)
        if fault == "eof":
            return b""
        block = os.read(fd, count)
        captured.append(len(block))
        if fault == "replace":
            replacement = path.with_name("replacement")
            replacement.write_bytes(raw)
            replacement.chmod(0o444)
            replacement.replace(path)
        if fault in {"grow", "truncate"}:
            path.chmod(0o644)
            path.write_bytes(raw + b"extra" if fault == "grow" else raw[:3])
            path.chmod(0o444)
        if fault == "stat":

            def fail(_fd):
                raise OSError("simulated-stat-error")

            private_os.fstat = fail
        return block

    private_os.read = read
    with pytest.raises((m.RequestRefusal, OSError)):
        reader.read(pin)
    assert reader.consumed == sum(captured)
    assert sum(calls) == len(raw)  # No read beyond the original fixed endpoint.
    assert not reader.audit and len(closed) == 1


@pytest.mark.parametrize(
    "fault", ["mode", "owner", "symlink", "size", "file_cap", "root", "deadline"]
)
def test_existing_protection_refuses_before_payload_reads(tmp_path, monkeypatch, fault):
    path, pin, reader, clock, calls, _, _ = setup(tmp_path, monkeypatch, b"safe")
    root = path.parent
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "owner":
        reader.owner_uid += 1
    elif fault == "symlink":
        alias = path.with_name("alias")
        alias.symlink_to(path)
        pin["path"] = str(alias)
    elif fault == "size":
        pin["bytes"] += 1
    elif fault == "file_cap":
        pin["bytes"] = m.FILE_LIMIT + 1
    elif fault == "root":
        root = root / "different"
    else:
        clock[0] = 101
    with pytest.raises(m.RequestRefusal):
        reader.read(pin, root=root)
    assert not calls and not reader.audit and reader.consumed == 0
