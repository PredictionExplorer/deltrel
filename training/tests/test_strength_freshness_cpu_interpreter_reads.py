"""Fixed image accounting using private temp files and module-local syscall faults.

The proc/executable mapping, clocks and any pre-existing usage are synthetic;
these tests do not execute or qualify an interpreter, Linux session or target.
"""

from __future__ import annotations

from copy import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r

CAP = 64 * 2**20


def setup(tmp_path, monkeypatch, raw=b"synthetic-image" * 5000, *, used=0):
    root = tmp_path.resolve()
    root.chmod(0o700)
    image = root / "image"
    image.write_bytes(raw)
    image.chmod(0o775)
    literal = root / "python"
    literal.symlink_to(image.name)
    value = {
        "path": str(literal),
        "resolved_path": str(image),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "metadata": r.file_identity(image.stat()),
    }
    clock = [o.Clock("boot", 10, 10)]
    store = r.ProtectedStore(
        roots=(root,),
        deadline_ns=1000,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: clock[0].monotonic_ns,
    )
    store.bind_deadline(o.Clock("boot", 1, 1), 1000, 1000, lambda: clock[0])
    # Explicit simulated prior usage, never a production seeding/reset API.
    store.interpreter_consumed = store._interpreter_accounted = used
    calls = []
    opens = []
    closed = []
    labels = {}
    io = SimpleNamespace(**vars(os))
    system = SimpleNamespace(**vars(sys))
    system.executable = str(literal)

    def opened(path, flags):
        kind = "proc" if path == "/proc/self/exe" else "named"
        fd = os.open(image if kind == "proc" else path, flags)
        labels[fd] = kind
        opens.append((kind, flags))
        return fd

    def read(fd, n):
        data = os.read(fd, n)
        calls.append((labels[fd], n, len(data)))
        return data

    def close(fd):
        closed.append(labels[fd])
        return os.close(fd)

    io.open = opened
    io.read = read
    io.close = close
    io.readlink = lambda path: (
        str(image) if path == "/proc/self/exe" else os.readlink(path)
    )
    io.stat = lambda path, *args, **kwargs: os.stat(
        image if path == "/proc/self/exe" else path, *args, **kwargs
    )
    monkeypatch.setattr(r, "os", io)
    monkeypatch.setattr(r, "sys", system)
    return store, value, image, literal, clock, calls, opens, closed, io, system


def named(store, value):
    return store.read(
        {
            "path": value["resolved_path"],
            "sha256": value["sha256"],
            "bytes": value["bytes"],
        },
        source=True,
        maximum=CAP,
        interpreter=True,
        expected_metadata=value["metadata"],
    )


def run(kind, store, value):
    return (
        r.current_interpreter(store, value) if kind == "full" else named(store, value)
    )


@pytest.mark.parametrize("size", [1, 65536, 65537])
def test_two_streams_one_exact_cap_no_eof_probe(tmp_path, monkeypatch, size):
    s, v, _, _, _, calls, opens, closed, _, _ = setup(
        tmp_path, monkeypatch, b"x" * size, used=CAP - 2 * size
    )
    r.current_interpreter(s, v)
    assert s.interpreter_consumed == s._interpreter_accounted == CAP and s.consumed == 0
    assert [k for k, _ in opens] == closed == ["proc", "named"]
    assert sum(n for _, n, _ in calls) == sum(n for _, _, n in calls) == 2 * size
    assert all(0 < n <= 65536 for _, n, _ in calls)
    before = len(calls)
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert len(calls) == before and s.interpreter_consumed == CAP


@pytest.mark.parametrize("remaining", [0, 1, 65536, 65537, 131073])
def test_second_stream_exhaustion_is_accounted_not_new_allowance(
    tmp_path, monkeypatch, remaining
):
    s, v, _, _, _, calls, _, _, _, _ = setup(
        tmp_path, monkeypatch, b"x" * 65537, used=CAP - remaining
    )
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert sum(n for _, _, n in calls) == remaining and s.interpreter_consumed == CAP
    assert s._interpreter_failure and s.consumed == 0


@pytest.mark.parametrize("kind", ["full", "named"])
def test_short_positive_reads_keep_fixed_endpoint(tmp_path, monkeypatch, kind):
    s, v, _, _, _, calls, _, _, io, _ = setup(tmp_path, monkeypatch, b"abcdefghij")

    def short(fd, n):
        data = os.read(fd, min(n, 3))
        calls.append(("mapped", n, len(data)))
        return data

    io.read = short
    run(kind, s, v)
    expected = 20 if kind == "full" else 10
    assert s.interpreter_consumed == sum(n for _, _, n in calls) == expected


@pytest.mark.parametrize("size", [0, -1, True, 1.0, CAP + 1])
def test_invalid_size_refuses_before_payload(tmp_path, monkeypatch, size):
    s, v, _, _, _, calls, opens, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    v["bytes"] = size
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert (
        calls == opens == [] and s.interpreter_consumed == 0 and s._interpreter_failure
    )


@pytest.mark.parametrize("field,value", [("sha256", "not-a-hash"), ("bytes", False)])
def test_invalid_admitted_pin_has_no_payload(tmp_path, monkeypatch, field, value):
    s, v, _, _, _, calls, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    v[field] = value
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert not calls and s.interpreter_consumed == 0


@pytest.mark.parametrize("kind", ["full", "named"])
@pytest.mark.parametrize("fault", ["eof", "error"])
def test_prior_debit_survives_short_read_or_private_error(
    tmp_path, monkeypatch, kind, fault
):
    s, v, _, _, _, calls, opens, closed, io, _ = setup(tmp_path, monkeypatch)
    old = io.read

    def failing(fd, n):
        if calls:
            if fault == "error":
                raise OSError("private-interpreter-content")
            calls.append(("mapped", n, 0))
            return b""
        return old(fd, n)

    io.read = failing
    with pytest.raises(r.RuntimeRefusal) as exc:
        run(kind, s, v)
    assert "private-interpreter-content" not in str(exc.value)
    assert s.interpreter_consumed == sum(n for _, _, n in calls) == 65536
    assert len(opens) == len(closed) and s._interpreter_failure
    count = len(calls)
    with pytest.raises(r.RuntimeRefusal):
        run(kind, s, v)
    assert len(calls) == count


@pytest.mark.parametrize("kind", ["full", "named"])
@pytest.mark.parametrize("fault", ["wall", "monotonic", "boot", "regression"])
def test_block_is_charged_before_clock_failure(tmp_path, monkeypatch, kind, fault):
    s, v, _, _, clock, calls, opens, closed, io, _ = setup(tmp_path, monkeypatch)
    old = io.read

    def late(fd, n):
        raw = old(fd, n)
        clock[0] = {
            "wall": replace(clock[0], wall_ns=1000),
            "monotonic": replace(clock[0], monotonic_ns=1000),
            "boot": replace(clock[0], boot_id="other"),
            "regression": replace(clock[0], wall_ns=9),
        }[fault]
        return raw

    io.read = late
    with pytest.raises(r.RuntimeRefusal):
        run(kind, s, v)
    assert (
        s.interpreter_consumed == 65536
        and len(calls) == 1
        and len(opens) == len(closed)
    )


@pytest.mark.parametrize("kind", ["full", "named"])
def test_post_hash_deadline_is_sticky(tmp_path, monkeypatch, kind):
    s, v, _, _, clock, calls, opens, closed, _, _ = setup(
        tmp_path, monkeypatch, b"body"
    )

    class LateHash:
        def __init__(self, *args):
            self.inner = hashlib.sha256(*args)

        def update(self, b):
            self.inner.update(b)

        def hexdigest(self):
            result = self.inner.hexdigest()
            clock[0] = replace(clock[0], wall_ns=1000)
            return result

    monkeypatch.setattr(r, "hashlib", SimpleNamespace(sha256=LateHash))
    with pytest.raises(r.RuntimeRefusal):
        run(kind, s, v)
    assert s.interpreter_consumed == sum(n for _, _, n in calls) == 4
    assert len(opens) == len(closed) and s._interpreter_failure


@pytest.mark.parametrize("kind", ["full", "named"])
def test_hash_mismatch_failure_and_metadata_only_cannot_reset(
    tmp_path, monkeypatch, kind
):
    s, v, _, _, _, calls, _, _, _, _ = setup(tmp_path, monkeypatch, b"body")
    v["sha256"] = "0" * 64
    with pytest.raises(r.RuntimeRefusal):
        run(kind, s, v)
    before = s.interpreter_consumed
    reason = s._interpreter_failure
    seen = len(calls)
    r.current_interpreter(s, v, hash_image=False)
    assert (
        len(calls) == seen
        and s.interpreter_consumed == before
        and s._interpreter_failure == reason
    )
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert len(calls) == seen


@pytest.mark.parametrize(
    "fault", ["mode", "mtime", "replace", "grow", "proc-deleted", "literal"]
)
def test_post_read_origin_and_full_metadata_changes_refuse(
    tmp_path, monkeypatch, fault
):
    s, v, image, literal, _, calls, opens, closed, io, _ = setup(
        tmp_path, monkeypatch, b"body"
    )
    old = io.read

    def changed(fd, n):
        data = old(fd, n)
        if fault == "mode":
            image.chmod(0o755)
        elif fault == "mtime":
            st = image.stat()
            os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        elif fault == "replace":
            other = image.with_name("other")
            other.write_bytes(b"body")
            other.chmod(0o775)
            os.replace(other, image)
        elif fault == "grow":
            image.write_bytes(b"bodyextra")
        elif fault == "proc-deleted":
            io.readlink = lambda p: str(image) + " (deleted)"
        else:
            other = image.with_name("other")
            other.write_bytes(b"body")
            literal.unlink()
            literal.symlink_to(other)
        return data

    io.read = changed
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert (
        s.interpreter_consumed == 4
        and len(calls) == 1
        and len(opens) == len(closed) == 1
    )
    assert s._interpreter_failure


@pytest.mark.parametrize("field", ["st_uid", "st_gid", "st_ctime_ns"])
def test_named_post_fd_full_identity_is_checked(tmp_path, monkeypatch, field):
    s, v, _, _, _, calls, _, _, io, _ = setup(tmp_path, monkeypatch, b"body")
    count = 0

    def fake(fd):
        nonlocal count
        st = os.fstat(fd)
        count += 1
        if count == 1:
            return st
        fields = {n: getattr(st, n) for n in dir(st) if n.startswith("st_")}
        fields[field] += 1
        return SimpleNamespace(**fields)

    io.fstat = fake
    with pytest.raises(r.RuntimeRefusal):
        named(s, v)
    assert (
        s.interpreter_consumed == sum(n for _, _, n in calls) == 4
        and s._interpreter_failure
    )


def test_named_replacement_before_open_has_zero_debit(tmp_path, monkeypatch):
    s, v, image, _, _, calls, _, closed, io, _ = setup(tmp_path, monkeypatch, b"body")
    old = io.open

    def replaced(path, flags):
        other = image.with_name("other")
        other.write_bytes(b"body")
        other.chmod(0o775)
        os.replace(other, image)
        return old(path, flags)

    io.open = replaced
    with pytest.raises(r.RuntimeRefusal):
        named(s, v)
    assert not calls and s.interpreter_consumed == 0 and len(closed) == 1


def test_permit_is_immutable_and_identity_only(tmp_path, monkeypatch):
    s, _, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    permit = s._interpreter_allowance(1)
    with pytest.raises(FrozenInstanceError):
        setattr(permit, "amount", 2)
    with pytest.raises(r.RuntimeRefusal):
        s._charge_interpreter(copy(permit), b"x")
    assert s.interpreter_consumed == 0 and s._interpreter_pending is permit
    with pytest.raises(r.RuntimeRefusal):
        s._charge_interpreter(permit, b"x")
    assert s.interpreter_consumed == 1 and s._interpreter_pending is None


@pytest.mark.parametrize("operation", ["charge", "abort"])
def test_foreign_permit_never_mutates_foreign_owner(tmp_path, monkeypatch, operation):
    a, v, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    b = r.ProtectedStore(
        roots=a.roots,
        deadline_ns=1000,
        owner_uid=a.owner_uid,
        monotonic_ns=a.monotonic_ns,
    )
    b.bind_deadline(o.Clock("boot", 1, 1), 1000, 1000, a.clock_source)
    pa = a._interpreter_allowance(1)
    pb = b._interpreter_allowance(1)
    with pytest.raises(r.RuntimeRefusal):
        a._charge_interpreter(
            pb, b"x"
        ) if operation == "charge" else a._abort_interpreter_read(pb)
    assert (
        b._interpreter_failure is None
        and b._interpreter_pending is pb
        and a._interpreter_pending is pa
    )
    b._charge_interpreter(pb, b"x")
    assert b.interpreter_consumed == 1
    with pytest.raises(r.RuntimeRefusal):
        a._charge_interpreter(pa, b"x")
    assert a.interpreter_consumed == 1


@pytest.mark.parametrize("operation", ["charge", "abort"])
def test_duplicate_or_stale_settlement_cannot_clear_new_allowance(
    tmp_path, monkeypatch, operation
):
    s, _, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    old = s._interpreter_allowance(1)
    s._charge_interpreter(old, b"x")
    current = s._interpreter_allowance(1)
    with pytest.raises(r.RuntimeRefusal):
        s._charge_interpreter(
            old, b"x"
        ) if operation == "charge" else s._abort_interpreter_read(old)
    assert s._interpreter_pending is current and s.interpreter_consumed == 1
    with pytest.raises(r.RuntimeRefusal):
        s._charge_interpreter(current, b"y")
    assert s.interpreter_consumed == 2


@pytest.mark.parametrize("counter", [0, CAP - 1])
def test_known_oversized_return_is_fully_debited_then_sticky(
    tmp_path, monkeypatch, counter
):
    s, _, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x", used=counter)
    token = s._interpreter_allowance(1)
    with pytest.raises(r.RuntimeRefusal, match="overrun"):
        s._charge_interpreter(token, b"XYZ")
    assert s.interpreter_consumed == counter + 3 and s._interpreter_pending is None
    with pytest.raises(r.RuntimeRefusal):
        s._interpreter_allowance(1)


@pytest.mark.parametrize("fault", ["counter", "pending", "amount", "nonbytes"])
def test_unknown_settlement_does_not_invent_a_charge(tmp_path, monkeypatch, fault):
    s, _, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    token = s._interpreter_allowance(1)
    if fault == "counter":
        s.interpreter_consumed = 1
    elif fault == "pending":
        s._interpreter_pending = None
    elif fault == "amount":
        object.__setattr__(token, "amount", 2)
    with pytest.raises(r.RuntimeRefusal):
        malformed: Any = None if fault == "nonbytes" else b"x"
        s._charge_interpreter(token, malformed)
    assert s._interpreter_accounted == 0 and s._interpreter_failure
    with pytest.raises(r.RuntimeRefusal):
        s._interpreter_allowance(1)


def test_overlap_charges_original_known_return_before_failure(tmp_path, monkeypatch):
    s, _, _, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"x")
    token = s._interpreter_allowance(1)
    with pytest.raises(r.RuntimeRefusal):
        s._interpreter_allowance(1)
    with pytest.raises(r.RuntimeRefusal):
        s._charge_interpreter(token, b"x")
    assert s.interpreter_consumed == 1 and s._interpreter_pending is None


def test_interpreter_failure_does_not_spend_ordinary_or_reservation(
    tmp_path, monkeypatch
):
    s, v, image, _, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"image")
    s._reserve_observed_metadata()
    capacity = s._metadata_capacity
    v["sha256"] = "0" * 64
    with pytest.raises(r.RuntimeRefusal):
        named(s, v)
    assert (
        s.interpreter_consumed == 5
        and s.consumed == s._metadata_used == 0
        and not s._ordinary_poisoned
    )
    assert s._metadata_capacity == capacity
    p = image.with_name("ordinary.json")
    p.write_bytes(b"{}")
    p.chmod(0o444)
    assert s.read({"path": str(p), "sha256": r.sha(b"{}"), "bytes": 2}) == b"{}"
    assert s.consumed == 2 and s.interpreter_consumed == 5


@pytest.mark.parametrize("fault", ["literal", "proc", "executable"])
def test_second_stream_origin_drift_cannot_return_success(tmp_path, monkeypatch, fault):
    s, v, image, literal, _, calls, opens, closed, io, system = setup(
        tmp_path, monkeypatch, b"body"
    )
    old = io.read

    def changed(fd, n):
        data = old(fd, n)
        if opens[-1][0] == "named":
            other = image.with_name("other")
            other.write_bytes(b"body")
            other.chmod(0o775)
            if fault == "literal":
                literal.unlink()
                literal.symlink_to(other)
            elif fault == "proc":
                io.readlink = lambda p: str(other)
            else:
                system.executable = str(other)
        return data

    io.read = changed
    with pytest.raises(r.RuntimeRefusal):
        r.current_interpreter(s, v)
    assert s.interpreter_consumed == 8 and len(opens) == len(closed) == 2
    assert sum(n for _, _, n in calls) == 8 and s._interpreter_failure
