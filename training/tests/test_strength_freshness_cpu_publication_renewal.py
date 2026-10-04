"""Issued fake-kernel observations + private publication files; no target access."""

from copy import copy, deepcopy
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import strength_freshness_cpu_publication_renewal as m
from scripts import strength_freshness_cpu_readonly as ro
from scripts import strength_freshness_cpu_observed_kernel as k
from scripts import strength_freshness_cpu_observed_registration as regmod
from tests.test_strength_freshness_cpu_observed_kernel import observed_kernel_fixture

MODEL = "sha256-" + "a" * 64
PROOF = "b" * 64
SECRET = "private-publication-sentinel"


class Publications:
    """Explicit synthetic IO seam: complete versioned, charged publication reads."""

    def __init__(self, parsed, io):
        self.io, self.parsed = io, parsed
        self.reg = parsed.private_copy()
        self.rows: dict[str, Any] = {}
        self.versions = dict.fromkeys(m.STREAMS, 1)
        self.calls: list[str] = []
        self.fault: str | None = None
        self.mutate_audit: Callable[[dict[str, Any]], None] | None = None
        self.raw_override: bytes | None = None
        self.wall = self.now()["wall_ns"]
        owners = self.reg["policy"]["expected_processes"]
        self.rows["coordinator"] = dict(
            schema_version=1,
            timestamp_ns=self.wall - 1,
            coordinator_pid=owners["coordinator"]["pid"],
            state="running",
            draining=False,
            failure=None,
            hardware_failure_class=None,
            hardware_failure_reason=None,
            workers={
                role: dict(
                    pid=owners[role]["pid"],
                    role="learner"
                    if role == "learner"
                    else "arena"
                    if role == "arena-promotion"
                    else "actor",
                    state="running",
                    restart_count=owners[role]["restarts"],
                    failure_reason=None,
                    failure_class=None,
                    failure_exit_code=None,
                )
                for role in m.records.WORKERS
            },
        )
        for name in m.STREAMS[1:]:
            parent = self.reg["publication_writers"][name]["owner_role"]
            self.rows[name] = dict(
                schema_version=1,
                worker=name,
                pid=owners[parent]["pid"],
                heartbeat_ns=self.wall - 1,
                progress_ns=1,
                progress=10,
                phase="training"
                if parent == "learner"
                else "arena"
                if parent == "arena-promotion"
                else "shared_cohorts",
                private_note=SECRET,
            )
            if name == "actor-cpu-ring4":
                self.rows[name].update(
                    phase="selfplay", model_role="champion", model_version=MODEL
                )
            if name in m.records.COHORTS:
                self.rows[name]["phase"] = "selfplay"
                self.rows[name].update(
                    model_role="champion",
                    requested_model_role="champion",
                    model_version=MODEL,
                    cumulative_games=2,
                    cohort=0,
                )
        io.read_publication = self.read

    def now(self):
        return dict(self.io.clock().value)

    def read(self, key):
        name = next(
            name
            for name, spec in self.reg["publication_writers"].items()
            if spec["key"] == key
        )
        self.calls.append(name)
        if self.fault == name:
            raise ro.ReadRefusal(SECRET)
        raw = (
            self.raw_override
            if self.raw_override is not None
            else regmod.encoded(self.rows[name])
        )
        self.io._consumed += len(raw)
        if self.io._consumed > self.io.maximum_bytes:
            raise ro.ReadRefusal("capture-byte-budget")
        spec = self.reg["publication_writers"][name]
        st = dict(
            device=42,
            inode=1000 + m.STREAMS.index(name) * 100 + self.versions[name],
            mode=spec["mode"],
            uid=spec["uid"],
            gid=spec["gid"],
            bytes=len(raw),
            mtime_ns=self.versions[name],
            ctime_ns=self.versions[name],
            links=1,
        )
        a = dict(
            operation="read-publication",
            subject=key,
            read_start=self.now(),
            read_end=self.now(),
            raw={"sha256": regmod.sha(raw), "bytes": len(raw)},
            named_before=deepcopy(st),
            stat_before=deepcopy(st),
            stat_after=deepcopy(st),
            named_after=deepcopy(st),
            offset=0,
            end_offset=len(raw),
        )
        if self.mutate_audit:
            self.mutate_audit(a)
        return ro.Observation(raw, a)

    def renew(self, names=m.STREAMS):
        stamp = self.now()["wall_ns"]
        for name in names:
            self.rows[name][
                "timestamp_ns" if name == "coordinator" else "heartbeat_ns"
            ] = stamp
            self.versions[name] += 1


def fixture(monkeypatch, champions=None):
    parsed, io, window, premises, observer = observed_kernel_fixture(monkeypatch)
    pubs = Publications(parsed, io)
    tracker = m.PublicationRenewalTracker(
        parsed,
        io,
        expected_window=window,
        external_premises=premises,
        verified_champions={MODEL: PROOF} if champions is None else champions,
    )
    return tracker, pubs, observer, parsed, io, window, premises


def fence(monkeypatch):
    tracker, pubs, observer, *rest = fixture(monkeypatch)
    tracker.fence(observer.capture_current())
    return tracker, pubs, observer, *rest


def ready(monkeypatch):
    tracker, pubs, observer, *rest = fence(monkeypatch)
    pubs.renew()
    assert tracker.observe(observer.capture_current()).renewed == 34
    return tracker, pubs, observer, *rest


def test_issued_kernel_to_all34_renewal_preserves_old_progress_and_has_no_grant(
    monkeypatch,
):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    result = tracker.finish(observer.capture_current())
    summary = result.safe_summary()
    assert pubs.calls == list(m.STREAMS) * 2
    assert summary["publication_renewed"] and summary["streams"] == 34
    assert all(
        not summary[key]
        for key in (
            "physical_work_proven",
            "historical_lifetime_proven",
            "writer_qualified",
            "preservation_passed",
            "execution_authorized",
        )
    )
    assert SECRET not in repr(result) + json.dumps(summary)
    assert "/private-" not in json.dumps(summary)
    assert all(
        row["counters"].get("progress_ns", 1) == 1
        for row in result.private_copy()["latest"].values()
    )


def test_same_files_are_pending_and_finish_is_terminal_incomplete(monkeypatch):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    assert tracker.observe(observer.capture_current()).renewed == 0
    with pytest.raises(m.RenewalRefusal, match="incomplete"):
        tracker.finish(observer.capture_current())
    before = len(pubs.calls)
    pubs.renew()
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        tracker.observe(observer.capture_current())
    assert len(pubs.calls) == before


@pytest.mark.parametrize("missing", ["coordinator", "learner", "actor-gpu-6-cohort-3"])
def test_one_unrenewed_stream_is_not_hidden(monkeypatch, missing):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    pubs.renew(tuple(name for name in m.STREAMS if name != missing))
    assert tracker.observe(observer.capture_current()).renewed == 33
    with pytest.raises(m.RenewalRefusal, match="incomplete"):
        tracker.finish(observer.capture_current())


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "unknown",
        "error",
        "nonfinite",
        "event",
        "candidate",
        "wrong-pid",
        "wrong-worker",
        "future",
        "bool-schema",
        "bad-restarts",
    ],
)
def test_baseline_negative_is_sticky_and_private(monkeypatch, fault):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    row = pubs.rows["actor-gpu-1-cohort-0"]
    if fault in {"failed", "unknown"}:
        row["phase"] = fault
    elif fault == "error":
        row["error"] = SECRET
    elif fault == "nonfinite":
        row["nonfinite_loss_count"] = 1
    elif fault == "event":
        row["event"] = SECRET
    elif fault == "candidate":
        row["requested_model_role"] = "candidate"
    elif fault == "wrong-pid":
        row["pid"] += 1
    elif fault == "wrong-worker":
        row["worker"] = SECRET
    elif fault == "future":
        row["heartbeat_ns"] += 10**15
    elif fault == "bool-schema":
        row["schema_version"] = True
    elif fault == "bad-restarts":
        pubs.rows["coordinator"]["workers"]["learner"]["restart_count"] += 1
    with pytest.raises(m.RenewalRefusal) as error:
        tracker.fence(observer.capture_current())
    assert SECRET not in str(error.value)
    calls = len(pubs.calls)
    row.pop("error", None)
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        tracker.fence(observer.capture_current())
    assert len(pubs.calls) == calls


@pytest.mark.parametrize(
    "fault", ["step", "progress", "games", "stamp", "progress_ns", "missing-counter"]
)
def test_later_counter_regression_refuses_even_after_ready(monkeypatch, fault):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    row = pubs.rows["actor-gpu-1-cohort-0"]
    if fault == "stamp":
        row["heartbeat_ns"] -= 1
    elif fault == "progress_ns":
        row["progress_ns"] = 0
    elif fault == "games":
        row["cumulative_games"] -= 1
    elif fault == "missing-counter":
        row.pop("progress")
    elif fault == "step":
        pubs.rows["learner"]["step"] = 100
        pubs.renew()
        tracker.observe(observer.capture_current())
        pubs.rows["learner"]["step"] = 99
    else:
        row["progress"] -= 1
    with pytest.raises(m.RenewalRefusal):
        tracker.observe(observer.capture_current())


@pytest.mark.parametrize(
    "fault",
    ["raw", "kind", "stat", "device", "uid", "mode", "late-wall", "regressed-clock"],
)
def test_actual_observation_bindings_cannot_be_relabelled(monkeypatch, fault):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    pubs.renew()

    def change(a):
        if fault == "raw":
            a["raw"]["sha256"] = "0" * 64
        elif fault == "kind":
            a["operation"] = "read"
        elif fault == "stat":
            a["named_before"]["inode"] += 1
        elif fault in {"device", "uid", "mode"}:
            for field in ("named_before", "stat_before", "stat_after", "named_after"):
                a[field][fault] += 1
        elif fault == "late-wall":
            a["read_end"]["wall_ns"] = tracker._window["deadline"]["wall_ns"]
        else:
            a["read_start"]["monotonic_ns"] = 1

    pubs.mutate_audit = change
    with pytest.raises(m.RenewalRefusal):
        tracker.observe(observer.capture_current())


@pytest.mark.parametrize("failure", ["publication", "clock"])
def test_io_refusal_cannot_be_retried_after_clean_data(monkeypatch, failure):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    pubs.renew()
    snap = observer.capture_current()
    if failure == "publication":
        pubs.fault = "learner"
    else:
        original = pubs.io.clock

        def clock():
            if len(pubs.calls) >= 68:
                raise ro.ReadRefusal(SECRET)
            return original()

        monkeypatch.setattr(pubs.io, "clock", clock)
    with pytest.raises(m.RenewalRefusal) as error:
        tracker.observe(snap)
    assert SECRET not in str(error.value)
    calls = len(pubs.calls)
    pubs.fault = None
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        tracker.observe(snap)
    assert len(pubs.calls) == calls


def test_refence_and_unissued_snapshot_refuse_before_publication_io(monkeypatch):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    with pytest.raises(m.RenewalRefusal):
        tracker.fence(copy(observer.capture_current()))
    assert pubs.calls == []
    tracker, pubs, observer, *_ = fence(monkeypatch)
    before = len(pubs.calls)
    with pytest.raises(m.RenewalRefusal, match="fence-once"):
        tracker.fence(observer.capture_current())
    assert len(pubs.calls) == before


def test_old_final_snapshot_cannot_supply_postpublication_owner_bracket(monkeypatch):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    old = observer.capture_current()
    pubs.renew()
    tracker.observe(observer.capture_current())
    with pytest.raises(m.RenewalRefusal):
        tracker.finish(old)


@pytest.mark.parametrize("field", ["deadline", "maximum_bytes", "_consumed"])
def test_io_budget_cannot_reset_or_extend(monkeypatch, field):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    snap = observer.capture_current()
    setattr(pubs.io, field, 0 if field == "_consumed" else getattr(pubs.io, field) + 1)
    before = len(pubs.calls)
    with pytest.raises(m.RenewalRefusal, match="context-changed"):
        tracker.observe(snap)
    assert len(pubs.calls) == before


def test_round256_remains_bound(monkeypatch):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    tracker._round = 256  # Boundary-only test; no claim of a real256-round capture.
    with pytest.raises(m.RenewalRefusal, match="round-bound"):
        tracker.observe(observer.capture_current())
    assert len(pubs.calls) == 34


@pytest.mark.parametrize("raw", [b"{", b"\xff", b'{"x":NaN}', b'{"x":1,"x":2}'])
def test_malformed_private_record_refuses_without_raw_error(monkeypatch, raw):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    pubs.raw_override = raw
    with pytest.raises(m.RenewalRefusal) as error:
        tracker.fence(observer.capture_current())
    assert str(error.value) == "renewal-observation-refused"


def test_ready_is_not_a_cached_success_when_later_stream_fails(monkeypatch):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    pubs.rows["actor-gpu-6-cohort-3"]["error"] = SECRET
    with pytest.raises(m.RenewalRefusal, match="negative"):
        tracker.observe(observer.capture_current())
    assert pubs.calls == list(m.STREAMS) * 3
    pubs.rows["actor-gpu-6-cohort-3"].pop("error")
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        tracker.finish(observer.capture_current())


def test_changed_version_without_new_producer_stamp_is_only_pending(monkeypatch):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    for name in m.STREAMS:
        pubs.versions[name] += 1
        pubs.rows[name]["private_note"] = "updated cached details"
    assert tracker.observe(observer.capture_current()).renewed == 0


def test_finish_rechecks_freshness_at_actual_final_clock(monkeypatch):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    # Isolate final freshness predicate inside an already-original-window capture.
    tracker._reg["policy"]["maximum_age_ns"] = 0
    with pytest.raises(m.RenewalRefusal, match="final-age"):
        tracker.finish(observer.capture_current())


def test_wrong_observer_issuer_refuses_before_publication_reads(monkeypatch):
    tracker, pubs, observer, parsed, io, window, premises = fence(monkeypatch)
    second = k.ObservedKernelObserver(
        parsed, io, expected_window=window, external_premises=premises
    )
    snapshot = second.capture_current()
    before = len(pubs.calls)
    with pytest.raises(m.RenewalRefusal):
        tracker.observe(snapshot)
    assert len(pubs.calls) == before


def test_safe_proof_retains_actual_audits_without_private_extra_fields(monkeypatch):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    pubs.mutate_audit = lambda audit: audit.update(private_path=SECRET)
    baseline = tracker.fence(observer.capture_current())
    assert len(baseline.safe_proof()["observations"]) == 34
    pubs.renew()
    tracker.observe(observer.capture_current())
    result = tracker.finish(observer.capture_current())
    proof = result.safe_proof()
    assert len(proof["observations"]) == 68
    assert len(proof["kernel_brackets"]) == 3
    assert (
        m.digest(proof["observations"]) == result.safe_summary()["observations_sha256"]
    )
    assert SECRET not in json.dumps(proof)
    assert set(proof["observations"][0]) == {
        "operation",
        "read_start",
        "read_end",
        "raw",
        "named_before",
        "stat_before",
        "stat_after",
        "named_after",
        "offset",
        "end_offset",
        "stream",
        "round",
        "audit_sha256",
        "producer_stamp_ns",
        "counters",
        "teacher",
    }
    proof["observations"][0]["raw"]["sha256"] = "x"
    assert result.safe_proof()["observations"][0]["raw"]["sha256"] != "x"


def test_safe_proof_limit_refuses_without_truncation(monkeypatch):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    monkeypatch.setattr(m, "MAX_PROOF_BYTES", 100)
    with pytest.raises(m.RenewalRefusal, match="proof-byte-bound"):
        tracker.fence(observer.capture_current())
    assert len(pubs.calls) == 1


def test_new_counter_is_allowed_but_its_later_disappearance_is_not(monkeypatch):
    tracker, pubs, observer, *_ = fence(monkeypatch)
    pubs.renew()
    pubs.rows["learner"]["step"] = 100
    assert tracker.observe(observer.capture_current()).renewed == 34
    pubs.rows["learner"].pop("step")
    with pytest.raises(m.RenewalRefusal, match="counter-regression"):
        tracker.observe(observer.capture_current())


class LocalPublicationSystem(ro.System):
    """Actual temp-file snapshot reads, synthetic clock/kernel, no host fallback."""

    def __init__(self, root, backend):
        self.root, self.backend = root.resolve(), backend
        self.paths = []

    def monotonic_ns(self):
        return self.backend.monotonic_ns()

    def publication_file(self, path, maximum, deadline, *, charge, allowance):
        assert Path(path).is_relative_to(self.root)
        self.paths.append(path)
        return super().publication_file(
            path, maximum, deadline, charge=charge, allowance=allowance
        )


def actual_publication_fixture(monkeypatch, tmp_path):
    import os

    root = tmp_path.resolve()

    def transform(reg, backend):
        for role in reg["kernel_context"]["credentials"]:
            reg["kernel_context"]["credentials"][role] = dict.fromkeys(
                regmod.UID_FIELDS, os.getuid()
            )
        for name, spec in reg["publication_writers"].items():
            reg["scope"]["files"][spec["key"]]["path"] = str(root / (name + ".json"))
            spec.update(uid=os.getuid(), gid=os.getgid())

    parsed, io, window, premises, observer = observed_kernel_fixture(
        monkeypatch, transform
    )
    pubs = Publications(parsed, io)
    del io.read_publication  # Restore the actual ReadOnlyIO entrypoint.
    system = LocalPublicationSystem(root, io._backend)
    io._backend.publication_file = system.publication_file

    def write():
        for name, row in pubs.rows.items():
            target = root / (name + ".json")
            temporary = root / (name + ".tmp")
            temporary.write_bytes(regmod.encoded(row))
            temporary.chmod(0o600)
            temporary.replace(target)

    tracker = m.PublicationRenewalTracker(
        parsed,
        io,
        expected_window=window,
        external_premises=premises,
        verified_champions={MODEL: PROOF},
    )
    write()
    return tracker, pubs, observer, system, write


def test_actual_private_files_to_actual_issued_kernel_to_renewal(monkeypatch, tmp_path):
    tracker, pubs, observer, system, write = actual_publication_fixture(
        monkeypatch, tmp_path
    )
    initial_consumed = pubs.io._consumed
    fence = tracker.fence(observer.capture_current())
    pubs.renew()
    write()
    assert tracker.observe(observer.capture_current()).renewed == 34
    proof = tracker.finish(observer.capture_current()).safe_proof()
    assert len(system.paths) == 68 and len(set(system.paths)) == 34
    assert pubs.io._consumed - initial_consumed >= sum(
        a["raw"]["bytes"] for a in proof["observations"]
    )
    assert all(
        a["named_before"] == a["stat_before"] == a["stat_after"] == a["named_after"]
        for a in proof["observations"]
    )
    for a, b in zip(
        fence.safe_proof()["observations"], proof["observations"][34:], strict=True
    ):
        assert a["stat_before"]["inode"] != b["stat_before"]["inode"]
    assert SECRET not in json.dumps(proof)
    assert proof["publication_renewed"] and not proof["physical_work_proven"]


def test_actual_file_mode_change_refuses_then_clean_retry_stays_closed(
    monkeypatch, tmp_path
):
    tracker, pubs, observer, system, write = actual_publication_fixture(
        monkeypatch, tmp_path
    )
    tracker.fence(observer.capture_current())
    pubs.renew()
    write()
    path = tmp_path / "learner.json"
    path.chmod(0o644)
    snapshot = observer.capture_current()
    with pytest.raises(m.RenewalRefusal, match="publication-access"):
        tracker.observe(snapshot)
    path.chmod(0o600)
    reads = len(system.paths)
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        tracker.observe(snapshot)
    assert len(system.paths) == reads


@pytest.mark.parametrize(
    "stream,phase",
    [
        ("actor-gpu-1", "selfplay"),
        ("actor-gpu-1-cohort-0", "shared_cohorts"),
        ("actor-cpu-ring4", "shared_cohorts"),
    ],
)
def test_source_phase_must_match_exact_registered_publication_kind(
    monkeypatch, stream, phase
):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    pubs.rows[stream]["phase"] = phase
    with pytest.raises(m.RenewalRefusal, match="unknown-phase"):
        tracker.fence(observer.capture_current())


def test_safe_baseline_and_intermediate_stamps_counters_and_teacher_are_retained(
    monkeypatch,
):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    baseline = tracker.fence(observer.capture_current()).safe_proof()
    pubs.renew()
    tracker.observe(observer.capture_current())
    pubs.renew()
    pubs.rows["actor-gpu-1-cohort-0"]["cumulative_games"] += 1
    tracker.observe(observer.capture_current())
    proof = tracker.finish(observer.capture_current()).safe_proof()
    name = "actor-gpu-1-cohort-0"
    chain = [row for row in proof["observations"] if row["stream"] == name]
    assert len(chain) == 3
    assert chain[0] == next(
        row for row in baseline["observations"] if row["stream"] == name
    )
    assert (
        chain[0]["producer_stamp_ns"]
        < chain[1]["producer_stamp_ns"]
        < chain[2]["producer_stamp_ns"]
    )
    assert [row["counters"]["cumulative_games"] for row in chain] == [2, 2, 3]
    assert all(
        row["teacher"] == {"model_identity": MODEL, "proof_sha256": PROOF}
        for row in chain
    )
    assert all(row["counters"]["progress_ns"] == 1 for row in chain)
    assert SECRET not in json.dumps(proof)


@pytest.mark.parametrize("stage", ["baseline", "renewal"])
@pytest.mark.parametrize(
    "fault",
    [
        "candidate",
        "unverified",
        "missing-role",
        "missing-version",
        "requested-candidate",
        "requested-null",
    ],
)
def test_cpu_scalar_teacher_negative_is_sticky(monkeypatch, stage, fault):
    tracker, pubs, observer, *_ = fixture(monkeypatch)
    if stage == "renewal":
        tracker.fence(observer.capture_current())
        pubs.renew()
    row = pubs.rows["actor-cpu-ring4"]
    if fault == "candidate":
        row["model_role"] = "candidate"
    elif fault == "unverified":
        row["model_version"] = "sha256-" + "c" * 64
    elif fault == "missing-role":
        row.pop("model_role")
    elif fault == "missing-version":
        row.pop("model_version")
    elif fault == "requested-candidate":
        row["requested_model_role"] = "candidate"
    else:
        row["requested_model_role"] = None
    operation = tracker.fence if stage == "baseline" else tracker.observe
    snapshot = observer.capture_current()
    with pytest.raises(m.RenewalRefusal, match="publication-champion-proof"):
        operation(snapshot)
    reads = len(pubs.calls)
    row.update(
        model_role="champion", model_version=MODEL, requested_model_role="champion"
    )
    with pytest.raises(m.RenewalRefusal, match="chain-refused"):
        operation(snapshot)
    assert len(pubs.calls) == reads


@pytest.mark.parametrize("requested", [False, True])
def test_cpu_scalar_old_and_new_verified_teacher_facts_are_retained(
    monkeypatch, requested
):
    newer, newer_proof = "sha256-" + "c" * 64, "d" * 64
    tracker, pubs, observer, *_ = fixture(
        monkeypatch, {MODEL: PROOF, newer: newer_proof}
    )
    if requested:
        pubs.rows["actor-cpu-ring4"]["requested_model_role"] = "champion"
    baseline = tracker.fence(observer.capture_current()).safe_proof()
    pubs.renew()
    pubs.rows["actor-cpu-ring4"]["model_version"] = newer
    assert tracker.observe(observer.capture_current()).renewed == 34
    proof = tracker.finish(observer.capture_current()).safe_proof()
    before = next(
        row for row in baseline["observations"] if row["stream"] == "actor-cpu-ring4"
    )
    after = next(
        row for row in proof["observations"][34:] if row["stream"] == "actor-cpu-ring4"
    )
    assert before["teacher"] == {"model_identity": MODEL, "proof_sha256": PROOF}
    assert after["teacher"] == {"model_identity": newer, "proof_sha256": newer_proof}
    assert not proof["physical_work_proven"] and not proof["preservation_passed"]
