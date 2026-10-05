"""Conditional transport only; temp payloads and synthetic window/source premises."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import PosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import strength_freshness_cpu_capture_request as q
from scripts import strength_freshness_cpu_observed_before as b
from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r

CAP = 32 * 2**20
SUBCAP = 8 * 2**20
FAIL = (r.RuntimeRefusal, q.RequestRefusal, b.BeforeRefusal, ValueError)


def fixture(tmp_path, monkeypatch, *, consumed=0, raw=b"private-data" * 10000):
    root = tmp_path.resolve()
    root.chmod(0o700)
    path = root / "proof.json"
    path.write_bytes(raw)
    path.chmod(0o444)
    pin = {"path": str(path), "sha256": r.sha(raw), "bytes": len(raw)}
    clock = [o.Clock("boot", 10, 10)]
    store = r.ProtectedStore(
        roots=(root,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: clock[0].monotonic_ns,
    )
    store.bind_deadline(o.Clock("boot", 1, 1), 100, 100, lambda: clock[0])
    store.consumed = consumed
    window = {
        "started": dict(clock[0].__dict__),
        "deadline_monotonic_ns": 90,
        "deadline_wall_ns": 90,
        "metadata_bytes": SUBCAP,
        "runtime_bytes": 24 * 2**20,
    }
    calls = []
    io = SimpleNamespace(**vars(os))

    def read(fd, count):
        data = os.read(fd, count)
        calls.append((count, len(data)))
        return data

    io.read = read
    monkeypatch.setattr(r, "os", io)
    reader = q._conditional_store_reader(
        store, phase="preflight-before", window=window, source_pins=[]
    )
    return store, reader, path, pin, clock, window, calls, io


def test_live_counter_one_reservation_and_same_reader(tmp_path, monkeypatch):
    store, reader, path, pin, _, window, calls, _ = fixture(tmp_path, monkeypatch)
    assert type(reader) is q.PinnedReader
    assert (
        q._conditional_store_reader(
            store, phase="preflight-before", window=deepcopy(window), source_pins=[]
        )
        is reader
    )
    assert reader.read(pin) == path.read_bytes()
    assert reader.consumed == store._metadata_used == store.consumed == pin["bytes"]
    assert sum(n for _, n in calls) == pin["bytes"]
    store._reserve_observed_metadata()
    assert store._metadata_capacity == SUBCAP and reader.consumed == pin["bytes"]


def test_one_shared_reservation_across_immutable_conditional_slots(
    tmp_path, monkeypatch
):
    store, first, _, pin, clock, window, _, _ = fixture(
        tmp_path, monkeypatch, raw=b"data"
    )
    first.read(pin)
    clock[0] = o.Clock("boot", 120, 120)
    store.bind_deadline(o.Clock("boot", 1, 1), 200, 200, store.clock_source)
    later = {
        **window,
        "started": dict(clock[0].__dict__),
        "deadline_monotonic_ns": 190,
        "deadline_wall_ns": 190,
    }
    second = q._conditional_store_reader(
        store, phase="dummy-final", window=later, source_pins=[]
    )
    assert second is not first and first.deadline == 90 / 1e9
    assert second.consumed == first.consumed == 4
    second.read(pin)
    assert second.consumed == first.consumed == store.consumed == 8
    with pytest.raises(FAIL):
        first.read(pin)
    assert store.consumed == 8


@pytest.mark.parametrize("available", [1, 65536, 65537])
def test_reservation_clips_to_actual_owner_remaining_without_probe(
    tmp_path, monkeypatch, available
):
    store, reader, _, pin, _, _, calls, _ = fixture(
        tmp_path, monkeypatch, consumed=CAP - available, raw=b"p" * available
    )
    reader.read(pin)
    assert store.consumed == CAP and reader.consumed == available
    assert (
        store._metadata_capacity == available and sum(x for x, _ in calls) == available
    )
    with pytest.raises(FAIL):
        reader.read(pin)
    assert store.consumed == CAP


def test_direct_reads_cannot_spend_reserved_remainder(tmp_path, monkeypatch):
    store, reader, _, pin, _, _, calls, _ = fixture(
        tmp_path, monkeypatch, consumed=CAP - SUBCAP, raw=b"data"
    )
    with pytest.raises(r.RuntimeRefusal, match="store-byte-budget"):
        store.read(pin)
    assert not calls and reader.consumed == 0
    reader.read(pin)
    assert store.consumed == CAP - SUBCAP + 4 and reader.consumed == 4


def test_direct_and_reserved_reads_each_charge_once(tmp_path, monkeypatch):
    store, reader, _, pin, _, _, _, _ = fixture(tmp_path, monkeypatch, raw=b"data")
    store.read(pin)
    assert store.consumed == 4 and reader.consumed == 0
    reader.read(pin)
    assert store.consumed == 8 and reader.consumed == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("_metadata_started", False),
        ("_metadata_capacity", 4 * 2**20),
        ("_metadata_capacity", float(SUBCAP)),
        ("consumed", 1),
        ("_metadata_used", 1),
        ("roots", ()),
        ("original_clock", object()),
    ],
)
def test_owner_context_loss_poisons_all_ordinary_access(
    tmp_path, monkeypatch, field, value
):
    store, reader, _, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    original = getattr(store, field)
    setattr(store, field, value)
    with pytest.raises(FAIL):
        reader.read(pin)
    setattr(store, field, original)
    assert store._ordinary_poisoned
    with pytest.raises(FAIL):
        store.read(pin)
    with pytest.raises(FAIL):
        reader.read(pin)
    assert not calls


@pytest.mark.parametrize(
    "operation", ["read", "check", "charge", "consumed", "directory", "inventory"]
)
@pytest.mark.parametrize("value", [None, object()])
def test_lost_original_binding_never_revives_standalone(
    tmp_path, monkeypatch, operation, value
):
    store, reader, path, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    reader._original_store_binding = value
    actions = {
        "read": lambda: reader.read(pin),
        "check": reader.check,
        "charge": lambda: reader.charge(1),
        "consumed": lambda: reader.consumed,
        "directory": lambda: reader.directory(path.parent),
        "inventory": lambda: reader.inventory(path.parent),
    }
    with pytest.raises(FAIL):
        actions[operation]()
    assert store._ordinary_poisoned and not calls


@pytest.mark.parametrize(
    "method", ["check", "_finish_block", "_open_block", "_metadata_scope", "read"]
)
def test_replaced_owner_method_cannot_be_noop(tmp_path, monkeypatch, method):
    store, reader, _, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    setattr(store, method, lambda *args, **kwargs: None)
    with pytest.raises(FAIL):
        reader.read(pin)
    assert store._ordinary_poisoned and not calls


@pytest.mark.parametrize(
    "mutation", ["drop", "amount", "foreign", "reenter", "counter"]
)
def test_bad_settlement_poison_prevents_later_resurrection(
    tmp_path, monkeypatch, mutation
):
    store, reader, _, pin, _, _, calls, io = fixture(
        tmp_path, monkeypatch, raw=b"bytes"
    )
    old = io.read

    def changed(fd, count):
        raw = old(fd, count)
        if mutation == "drop":
            store._ordinary_block = None
        elif mutation == "amount":
            assert store._ordinary_block is not None
            store._ordinary_block.amount += 1
        elif mutation == "foreign":
            store._ordinary_block = r._StoreBlock(count, None)
        elif mutation == "counter":
            store.consumed += 1
        else:
            with pytest.raises(r.RuntimeRefusal):
                store._open_block(None, 1, 1)
        return raw

    io.read = changed
    with pytest.raises(FAIL):
        reader.read(pin)
    assert store._ordinary_poisoned and len(calls) == 1
    with pytest.raises(FAIL):
        store.read(pin)
    assert len(calls) == 1


def test_duplicate_settlement_and_abort_refuse(tmp_path, monkeypatch):
    store, reader, _, _, _, _, _, _ = fixture(tmp_path, monkeypatch)
    assert reader._store_binding is not None
    view = reader._store_binding.view
    permit = store._open_block(view, 3, 3)
    store._finish_block(permit, 2)
    assert store.consumed == reader.consumed == 2
    with pytest.raises(r.RuntimeRefusal):
        store._finish_block(permit, 2)
    assert store._ordinary_poisoned and store.consumed == 2


@pytest.mark.parametrize("fault", ["hash", "wall", "boot", "IO"])
def test_failed_bytes_retained_and_view_sticky(tmp_path, monkeypatch, fault):
    store, reader, _, pin, clock, _, calls, io = fixture(tmp_path, monkeypatch)
    old = io.read

    def failed(fd, count):
        if calls and fault == "IO":
            raise OSError("private-IO")
        raw = old(fd, count)
        if fault == "wall":
            clock[0] = replace(clock[0], wall_ns=90)
        if fault == "boot":
            clock[0] = replace(clock[0], boot_id="other")
        return raw

    io.read = failed
    if fault == "hash":
        pin["sha256"] = "0" * 64
    with pytest.raises((OSError, *FAIL)):
        reader.read(pin)
    assert store.consumed == sum(n for _, n in calls) > 0
    previous = len(calls)
    clock[0] = o.Clock("boot", 10, 10)
    assert reader._store_binding is not None
    reader._store_binding.view.failed = False
    with pytest.raises(FAIL):
        reader.read(pin)
    assert len(calls) == previous


@pytest.mark.parametrize("field", ["metadata_bytes", "runtime_bytes"])
def test_float_budget_alias_refuses(tmp_path, monkeypatch, field):
    store, _, _, _, _, window, calls, _ = fixture(tmp_path, monkeypatch)
    changed = {**window, field: float(window[field])}
    with pytest.raises(FAIL):
        q._conditional_store_reader(
            store, phase="preflight-before", window=changed, source_pins=[]
        )
    assert not calls


@pytest.mark.parametrize("phase", ["unknown", "helper-frame", "dummy-final"])
def test_unknown_reordered_phase_and_window_rebind(tmp_path, monkeypatch, phase):
    store, reader, _, _, _, window, calls, _ = fixture(tmp_path, monkeypatch)
    if phase == "dummy-final":
        window = {**window, "deadline_monotonic_ns": 101}
    with pytest.raises(FAIL):
        q._conditional_store_reader(store, phase=phase, window=window, source_pins=[])
    assert reader.consumed == 0 and not calls


@pytest.mark.parametrize(
    "phase", ["preflight-before", "dummy-final", "helper-frame", "anything"]
)
def test_actual_factory_has_no_typed_or_hash_unlock(phase):
    for fake in [None, {}, SimpleNamespace(approved_intent_sha256="a" * 64)]:
        with pytest.raises(
            q.RequestRefusal, match="outer-reader-authority-unavailable"
        ):
            q._outer_pinned_reader(fake, phase=phase, window={})


def test_scope_adds_restrictions_and_keeps_pinned_mode444(tmp_path, monkeypatch):
    store, reader, path, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch, raw=b"x")
    path.chmod(0o555)
    with pytest.raises(q.RequestRefusal, match="input-protection"):
        reader.read(pin)
    assert not calls
    assert (
        store.read(pin, source=True) == b"x"
    )  # Existing direct source policy is separate.


def test_inventory_names_are_charged_and_bad_name_is_sticky(tmp_path, monkeypatch):
    store, reader, path, _, _, _, calls, _ = fixture(tmp_path, monkeypatch, raw=b"x")
    with pytest.raises(q.RequestRefusal, match="proof-inventory-name"):
        reader.inventory(path.parent)
    assert not calls and store.consumed == len(path.name.encode())
    assert store._metadata_used == store.consumed
    with pytest.raises(FAIL):
        reader.check()


def test_directory_failure_is_sticky_without_payload(tmp_path, monkeypatch):
    store, reader, path, _, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(FAIL):
        reader.directory(path.parent / "unknown")
    with pytest.raises(FAIL):
        reader.check()
    assert not calls and store.consumed == 0


def test_standalone_reader_has_single_counter(tmp_path):
    p = tmp_path / "ordinary"
    p.write_bytes(b"hello")
    p.chmod(0o444)
    reader = q.PinnedReader(deadline=100, owner_uid=os.getuid(), monotonic=lambda: 1)
    assert (
        reader.read(
            {"path": str(p.resolve()), "bytes": 5, "sha256": q.digest(b"hello")}
        )
        == b"hello"
    )
    assert reader.consumed == 5 and reader._binding_identity() is None


@pytest.mark.parametrize(
    "field", ["deadline", "owner_uid", "monotonic", "read", "charge"]
)
def test_reader_context_or_method_change_poison(tmp_path, monkeypatch, field):
    store, reader, _, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    replacements = {
        "deadline": reader.deadline + 1,
        "owner_uid": reader.owner_uid + 1,
        "monotonic": lambda: 0,
        "read": lambda *a, **k: b"fake",
        "charge": lambda *a: None,
    }
    setattr(reader, field, replacements[field])
    with pytest.raises(FAIL):
        q._PINNED_METHODS["check"](reader)
    assert store._ordinary_poisoned and not calls
    with pytest.raises(FAIL):
        store.read(pin)


def test_forked_pid_cannot_spend_inherited_counter(tmp_path, monkeypatch):
    store, reader, _, pin, _, _, calls, io = fixture(tmp_path, monkeypatch)
    io.getpid = lambda: os.getpid() + 1
    with pytest.raises(FAIL):
        reader.read(pin)
    assert store._ordinary_poisoned and not calls


def test_source_scope_is_exact_and_not_a_mutable_alias(tmp_path, monkeypatch):
    store, _, path, pin, clock, window, _, _ = fixture(
        tmp_path, monkeypatch, raw=b"registered"
    )
    other = path.parent / "outside"
    other.mkdir(mode=0o700)
    source = other / "source.py"
    source.write_bytes(b"source")
    source.chmod(0o444)
    source_pin = {"path": str(source), "sha256": r.sha(b"source"), "bytes": 6}
    fresh = r.ProtectedStore(
        roots=(path.parent,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=store.monotonic_ns,
    )
    fresh.bind_deadline(o.Clock("boot", 1, 1), 100, 100, store.clock_source)
    sources = [source_pin]
    reader = q._conditional_store_reader(
        fresh, phase="preflight-before", window=window, source_pins=sources
    )
    sources.clear()
    assert reader.read(source_pin) == b"source"
    source_pin["sha256"] = "0" * 64
    with pytest.raises(FAIL):
        reader.read(source_pin)
    assert fresh.consumed == 6


def test_source_mode555_is_not_inherited_by_bound_reader(tmp_path, monkeypatch):
    store, _, path, _, _, window, _, _ = fixture(tmp_path, monkeypatch, raw=b"x")
    other = path.parent / "outside"
    other.mkdir(mode=0o700)
    source = other / "source.py"
    source.write_bytes(b"source")
    source.chmod(0o555)
    source_pin = {"path": str(source), "sha256": r.sha(b"source"), "bytes": 6}
    fresh = r.ProtectedStore(
        roots=(path.parent,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=store.monotonic_ns,
    )
    fresh.bind_deadline(o.Clock("boot", 1, 1), 100, 100, store.clock_source)
    reader = q._conditional_store_reader(
        fresh, phase="preflight-before", window=window, source_pins=[source_pin]
    )
    with pytest.raises(q.RequestRefusal, match="input-protection"):
        reader.read(source_pin)
    assert fresh.consumed == 0


def test_view_replacement_does_not_create_new_same_slot_reader(tmp_path, monkeypatch):
    store, reader, _, _, _, window, calls, _ = fixture(tmp_path, monkeypatch)
    assert reader._store_binding is not None
    reader._store_binding.view.reader = None
    with pytest.raises(FAIL):
        q._conditional_store_reader(
            store, phase="preflight-before", window=window, source_pins=[]
        )
    assert store._ordinary_poisoned and not calls


def root_reported(value):
    """Synthetic root stat premise, while payload bytes use real temporary FDs."""
    fields = {
        name: getattr(value, name) for name in dir(value) if name.startswith("st_")
    }
    fields["st_uid"] = fields["st_gid"] = 0
    return SimpleNamespace(**fields)


class ReportedRootPath(PosixPath):
    def lstat(self) -> Any:
        return root_reported(super().lstat())


def test_real_owner_reader_and_read_guard_composition_no_authority(
    tmp_path, monkeypatch
):
    root = ReportedRootPath(str(tmp_path.resolve()))
    root.chmod(0o700)
    path = root / "proof.json"
    path.write_bytes(b"private")
    path.chmod(0o444)
    clock = [o.Clock("boot", 10, 10)]
    store = r.ProtectedStore(
        roots=(root,),
        deadline_ns=100,
        owner_uid=0,
        monotonic_ns=lambda: clock[0].monotonic_ns,
    )
    store.bind_deadline(o.Clock("boot", 1, 1), 100, 100, lambda: clock[0])
    io = SimpleNamespace(**vars(os))
    actual = []

    def read(fd, n):
        result = os.read(fd, n)
        actual.append(len(result))
        return result

    io.read = read
    io.fstat = lambda fd: root_reported(os.fstat(fd))
    monkeypatch.setattr(r, "os", io)
    monkeypatch.setattr(q, "Path", ReportedRootPath)
    monkeypatch.setattr(
        b,
        "time",
        SimpleNamespace(
            monotonic_ns=lambda: clock[0].monotonic_ns, time_ns=lambda: clock[0].wall_ns
        ),
    )
    window = {
        "started": dict(clock[0].__dict__),
        "deadline_monotonic_ns": 90,
        "deadline_wall_ns": 90,
        "metadata_bytes": SUBCAP,
        "runtime_bytes": 24 * 2**20,
    }
    reader = q._conditional_store_reader(
        store, phase="preflight-before", window=window, source_pins=[]
    )
    guard = b._Read(
        reader, {"limits": window, "admission_clock": dict(clock[0].__dict__)}
    )
    pin = {"path": str(path), "sha256": q.digest(b"private"), "bytes": 7}
    assert guard.read(pin, root=root) == b"private"
    assert guard.read(pin, root=root) == b"private"  # parse cache has no payload debit
    assert guard.read(pin, root=root, fresh=True) == b"private"
    assert actual == [7, 7] and store.consumed == reader.consumed == 14
    assert type(reader) is q.PinnedReader and guard._binding is not None
    reader._store_binding = None
    with pytest.raises(FAIL):
        guard.read(pin, fresh=True)
    assert actual == [7, 7] and store._ordinary_poisoned


@pytest.mark.parametrize("field", ["owner", "view", "methods"])
def test_mutated_binding_poisons_original_not_foreign(tmp_path, monkeypatch, field):
    original_dir = tmp_path / "original"
    foreign_dir = tmp_path / "foreign"
    original_dir.mkdir()
    foreign_dir.mkdir()
    owner, reader, _, pin, _, _, _, _ = fixture(original_dir, monkeypatch, raw=b"first")
    foreign, other, _, other_pin, _, _, _, _ = fixture(
        foreign_dir, monkeypatch, raw=b"second"
    )
    assert reader._store_binding is not None and other._store_binding is not None
    if field == "owner":
        reader._store_binding.owner = foreign
    elif field == "view":
        reader._store_binding.view = other._store_binding.view
    else:
        reader._store_binding.methods = dict(reader._store_binding.methods)
    with pytest.raises(FAIL):
        reader.read(pin)
    assert owner._ordinary_poisoned and not foreign._ordinary_poisoned
    with pytest.raises(FAIL):
        owner.read(pin)
    assert other.read(other_pin) == b"second"
    assert foreign.consumed == 6 and not other._store_binding.view.failed


def test_poison_boolean_reset_does_not_revive_account(tmp_path, monkeypatch):
    store, reader, _, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    assert reader._store_binding is not None
    permit = store._open_block(reader._store_binding.view, 1, 1)
    store._finish_block(permit, 0)
    with pytest.raises(r.RuntimeRefusal):
        store._finish_block(permit, 0)
    assert store._ordinary_failure is not None
    store._ordinary_poisoned = False
    with pytest.raises(FAIL):
        store.read(pin)
    with pytest.raises(FAIL):
        reader.read(pin)
    assert not calls


def test_last_clock_reset_is_sticky_without_touching_interpreter(tmp_path, monkeypatch):
    store, reader, _, pin, clock, _, calls, _ = fixture(
        tmp_path, monkeypatch, raw=b"ELF"
    )
    clock[0] = o.Clock("boot", 20, 20)
    reader.check()
    store.last_clock = None
    with pytest.raises(FAIL):
        reader.check()
    assert store._ordinary_failure is not None and not calls
    # Excluded legacy interpreter transport keeps its independent behavior/counter.
    assert store.read(pin, source=True, interpreter=True, maximum=64 * 2**20) == b"ELF"
    assert store.interpreter_consumed == 3 and store.consumed == 0
    with pytest.raises(FAIL):
        reader.read(pin)


def test_other_thread_cannot_spend_same_process_owner(tmp_path, monkeypatch):
    import threading

    store, reader, _, pin, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    errors = []

    def other():
        try:
            reader.read(pin)
        except Exception as error:
            errors.append(type(error))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive() and errors == [r.RuntimeRefusal]
    assert store._ordinary_poisoned and not calls
