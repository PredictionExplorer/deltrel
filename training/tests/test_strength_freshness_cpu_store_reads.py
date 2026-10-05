"""Ordinary store payload bounds; real private files, no target qualification."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r

CAP = 32 * 2**20


def setup(tmp_path, monkeypatch, raw=b"private-body" * 10000, *, bound=True):
    root = tmp_path.resolve()
    root.chmod(0o700)
    path = root / "proof.json"
    path.write_bytes(raw)
    path.chmod(0o444)
    pin = {"path": str(path), "sha256": r.sha(raw), "bytes": len(raw)}
    now = [o.Clock("boot", 10, 10)]
    store = r.ProtectedStore(
        roots=(root,),
        deadline_ns=100,
        owner_uid=os.geteuid(),
        monotonic_ns=lambda: now[0].monotonic_ns,
    )
    if bound:
        store.bind_deadline(o.Clock("boot", 1, 1), 100, 100, lambda: now[0])
    calls, returned, closed, opens = [], [], [], []
    local_os = SimpleNamespace(**vars(os))

    def read(fd, count):
        calls.append(count)
        data = os.read(fd, count)
        returned.append(len(data))
        return data

    def close(fd):
        closed.append(fd)
        return os.close(fd)

    def opened(path, flags):
        opens.append(flags)
        return os.open(path, flags)

    local_os.read, local_os.close, local_os.open = read, close, opened
    monkeypatch.setattr(r, "os", local_os)
    return path, pin, store, now, calls, returned, closed, opens, local_os


def invoke(kind, path, pin, store, *, maximum=2**20):
    if kind == "fixed":
        observed, raw = store.fixed(path, maximum=maximum)
        assert observed == {"path": str(path), "sha256": r.sha(raw), "bytes": len(raw)}
        return raw
    return store.read(pin, maximum=maximum)


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("size", [1, 65536, 65537, 2**20, 4 * 2**20])
def test_literal_endpoint_and_exact_budget_use_one_charged_pass(
    tmp_path, monkeypatch, kind, size
):
    raw = b"x" * size
    path, pin, store, _, calls, returned, closed, opens, _ = setup(
        tmp_path, monkeypatch, raw
    )
    store.consumed = CAP - size
    assert invoke(kind, path, pin, store, maximum=max(2**20, size)) == raw
    assert sum(calls) == sum(returned) == size
    assert all(0 < count <= 65536 for count in calls)
    assert store.consumed == CAP and store.interpreter_consumed == 0
    assert len(opens) == len(closed) == 1
    assert opens[0] & os.O_NONBLOCK and opens[0] & os.O_NOFOLLOW


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("remaining", [0, 1, 65535, 65536, 70000])
def test_remaining_budget_clamps_before_read_and_never_refunds(
    tmp_path, monkeypatch, kind, remaining
):
    path, pin, store, _, calls, returned, closed, _, _ = setup(tmp_path, monkeypatch)
    store.consumed = CAP - remaining
    with pytest.raises(r.RuntimeRefusal, match="^store-byte-budget$"):
        invoke(kind, path, pin, store)
    assert sum(calls) == sum(returned) == remaining
    assert store.consumed == CAP and len(closed) == 1
    count = len(calls)
    with pytest.raises(r.RuntimeRefusal, match="^store-byte-budget$"):
        invoke(kind, path, pin, store)
    assert len(calls) == count and store.consumed == CAP


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
def test_short_os_reads_charge_only_actual_returned_bytes(tmp_path, monkeypatch, kind):
    raw = b"sensitive-private-row" * 9
    path, pin, store, _, calls, returned, closed, _, io = setup(
        tmp_path, monkeypatch, raw
    )

    def short(fd, count):
        calls.append(count)
        data = os.read(fd, min(count, 7))
        returned.append(len(data))
        return data

    io.read = short
    assert invoke(kind, path, pin, store) == raw
    assert sum(returned) == len(raw) == store.consumed
    assert len(calls) > 1 and len(closed) == 1


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("fault", ["early-eof", "read-error"])
def test_prior_block_stays_charged_on_later_io_failure(
    tmp_path, monkeypatch, kind, fault
):
    path, pin, store, _, calls, returned, closed, _, io = setup(tmp_path, monkeypatch)
    old_read = io.read

    def fail_second(fd, count):
        if calls:
            calls.append(count)
            if fault == "read-error":
                raise OSError("injected-private-IO-failure")
            returned.append(0)
            return b""
        return old_read(fd, count)

    io.read = fail_second
    error = OSError if fault == "read-error" else r.RuntimeRefusal
    with pytest.raises(error):
        invoke(kind, path, pin, store)
    assert store.consumed == sum(returned) == 65536 and len(closed) == 1


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("fault", ["wall-expired", "mono-expired", "boot", "regressed"])
def test_returned_block_charged_before_actual_bound_clock_refusal(
    tmp_path, monkeypatch, kind, fault
):
    path, pin, store, now, calls, returned, closed, _, io = setup(tmp_path, monkeypatch)
    old_read = io.read

    def late(fd, count):
        data = old_read(fd, count)
        now[0] = {
            "wall-expired": replace(now[0], wall_ns=100),
            "mono-expired": replace(now[0], monotonic_ns=100),
            "boot": replace(now[0], boot_id="different"),
            "regressed": replace(now[0], wall_ns=9),
        }[fault]
        return data

    io.read = late
    with pytest.raises(r.RuntimeRefusal, match="store-(deadline|clock-drift)"):
        invoke(kind, path, pin, store)
    assert calls == [65536] and store.consumed == sum(returned) == 65536
    assert len(closed) == 1


def test_prebind_monotonic_behavior_remains_separate(tmp_path, monkeypatch):
    path, pin, store, now, _, _, _, _, io = setup(
        tmp_path, monkeypatch, b"raw", bound=False
    )
    now[0] = replace(now[0], wall_ns=1000)
    assert store.read(pin) == b"raw"
    old = io.read

    def late(fd, count):
        data = old(fd, count)
        now[0] = replace(now[0], monotonic_ns=100)
        return data

    io.read = late
    with pytest.raises(r.RuntimeRefusal, match="store-deadline"):
        store.fixed(path)
    assert store.consumed == 6


def test_wrong_hash_is_fully_charged_and_no_private_body_leaks(tmp_path, monkeypatch):
    raw = b"never-return-this-secret"
    _, pin, store, _, _, returned, closed, _, _ = setup(tmp_path, monkeypatch, raw)
    pin["sha256"] = "0" * 64
    with pytest.raises(r.RuntimeRefusal, match="^store-input-hash$") as error:
        store.read(pin)
    assert raw.decode() not in str(error.value)
    assert sum(returned) == store.consumed == len(raw) and len(closed) == 1


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("replacement", ["regular", "symlink", "fifo"])
def test_named_before_must_match_actual_opened_fd_before_payload(
    tmp_path, monkeypatch, kind, replacement
):
    path, pin, store, _, calls, _, closed, opens, io = setup(
        tmp_path, monkeypatch, b"body"
    )
    second = path.with_name("replacement.json")
    second.write_bytes(b"body")
    second.chmod(0o444)
    old_open = io.open

    def replace_before_open(value, flags):
        path.unlink()
        if replacement == "regular":
            second.rename(path)
        elif replacement == "symlink":
            path.symlink_to(second)
        else:
            os.mkfifo(path, 0o444)
        return old_open(value, flags)

    io.open = replace_before_open
    with pytest.raises((r.RuntimeRefusal, OSError)):
        invoke(kind, path, pin, store)
    assert not calls and store.consumed == 0
    assert len(opens) == 1 and len(closed) == (0 if replacement == "symlink" else 1)


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize("mutation", ["replace", "grow", "shrink", "mode", "mtime"])
def test_postread_named_and_fd_mutations_refuse_after_debit(
    tmp_path, monkeypatch, kind, mutation
):
    raw = b"sensitive"
    path, pin, store, _, _, returned, closed, _, io = setup(tmp_path, monkeypatch, raw)
    old_read = io.read

    def changed(fd, count):
        data = old_read(fd, count)
        if mutation == "replace":
            other = path.with_name("other.json")
            other.write_bytes(raw)
            other.chmod(0o444)
            os.replace(other, path)
        elif mutation == "mode":
            path.chmod(0o555)
        elif mutation == "mtime":
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        else:
            path.chmod(0o644)
            path.write_bytes(raw + b"extra" if mutation == "grow" else b"x")
            path.chmod(0o444)
        return data

    io.read = changed
    with pytest.raises(r.RuntimeRefusal, match="store-input-raced"):
        invoke(kind, path, pin, store)
    assert store.consumed == sum(returned) == len(raw) and len(closed) == 1


@pytest.mark.parametrize("field", ["st_uid", "st_gid", "st_ctime_ns"])
def test_full_post_fd_identity_includes_uid_gid_and_ctime(tmp_path, monkeypatch, field):
    _, pin, store, _, _, returned, closed, _, io = setup(tmp_path, monkeypatch, b"data")
    count = 0

    def changed(fd):
        nonlocal count
        st = os.fstat(fd)
        count += 1
        if count == 1:
            return st
        values = {name: getattr(st, name) for name in dir(st) if name.startswith("st_")}
        values[field] += 1
        return SimpleNamespace(**values)

    io.fstat = changed
    with pytest.raises(r.RuntimeRefusal, match="store-input-raced"):
        store.read(pin)
    assert store.consumed == sum(returned) == 4 and len(closed) == 1


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
@pytest.mark.parametrize(
    "fault", ["empty", "oversize", "wrong-mode", "wrong-owner", "wrong-root"]
)
def test_preconditions_refuse_without_payload(tmp_path, monkeypatch, kind, fault):
    raw = b"" if fault == "empty" else b"abcd"
    path, pin, store, _, calls, _, _, _, _ = setup(tmp_path, monkeypatch, raw)
    maximum = 3 if fault == "oversize" else 2**20
    if fault == "wrong-mode":
        path.chmod(0o600)
    elif fault == "wrong-owner":
        store.owner_uid += 1
    elif fault == "wrong-root":
        store.roots = ()
    with pytest.raises((r.RuntimeRefusal, ValueError)):
        invoke(kind, path, pin, store, maximum=maximum)
    assert not calls and store.consumed == 0


@pytest.mark.parametrize("mode", [0o444, 0o555])
def test_protected_source_retains_existing_outside_root_policy(
    tmp_path, monkeypatch, mode
):
    path, pin, store, _, _, _, _, _, _ = setup(tmp_path, monkeypatch, b"source")
    path.chmod(mode)
    store.roots = ()
    assert store.read(pin, source=True) == b"source"
    assert store.consumed == 6


@pytest.mark.parametrize("expected_too", [False, True])
def test_real_qualified_mutable_source_uses_exact_existing_metadata_predicate(
    tmp_path, monkeypatch, expected_too
):
    path, pin, store, _, calls, _, _, _, _ = setup(tmp_path, monkeypatch, b"source")
    path.chmod(0o664)
    qualified = r.file_identity(path.stat())
    store.owner_uid += 1
    store.roots = ()
    expected = (
        {"unrelated": "old qualified branch ignores this"} if expected_too else None
    )
    assert (
        store.read(
            pin, source=True, qualified_metadata=qualified, expected_metadata=expected
        )
        == b"source"
    )
    assert store.consumed == 6
    qualified["ctime_ns"] += 1
    prior = len(calls)
    with pytest.raises(r.RuntimeRefusal, match="store-input-protection"):
        store.read(
            pin, source=True, qualified_metadata=qualified, expected_metadata=expected
        )
    assert len(calls) == prior and store.consumed == 6


@pytest.mark.parametrize(
    "source,expected,qualified",
    [(False, False, True), (False, True, False), (True, True, False)],
)
def test_metadata_flags_do_not_create_new_ordinary_admission(
    tmp_path, monkeypatch, source, expected, qualified
):
    path, pin, store, _, calls, _, _, _, _ = setup(tmp_path, monkeypatch, b"source")
    metadata = r.file_identity(path.stat())
    with pytest.raises(r.RuntimeRefusal, match="store-input-protection"):
        store.read(
            pin,
            source=source,
            expected_metadata=metadata if expected else None,
            qualified_metadata=metadata if qualified else None,
        )
    assert not calls and store.consumed == 0


def test_interpreter_uses_fixed_endpoint_and_separate_counter(tmp_path, monkeypatch):
    _, pin, store, _, calls, returned, _, _, _ = setup(tmp_path, monkeypatch, b"ELF")
    store.consumed = CAP
    assert store.read(pin, source=True, maximum=64 * 2**20, interpreter=True) == b"ELF"
    assert calls == [3] and returned == [3]
    assert store.interpreter_consumed == 3 and store.consumed == CAP


@pytest.mark.parametrize("kind", ["pinned", "fixed"])
def test_hash_crossing_deadline_cannot_return_success(tmp_path, monkeypatch, kind):
    path, pin, store, now, _, returned, closed, _, _ = setup(
        tmp_path, monkeypatch, b"private"
    )
    actual_sha = r.sha

    def late_hash(raw):
        result = actual_sha(raw)
        now[0] = replace(now[0], wall_ns=100)
        return result

    monkeypatch.setattr(r, "sha", late_hash)
    with pytest.raises(r.RuntimeRefusal, match="store-deadline"):
        invoke(kind, path, pin, store)
    assert store.consumed == sum(returned) == 7 and len(closed) == 1
