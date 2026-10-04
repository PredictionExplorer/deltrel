"""Actual private files + real helpers, with explicitly synthetic kernel authority."""

from copy import copy, deepcopy
import json
import os

import pytest

from scripts import strength_freshness_cpu_before_measurement as m
from scripts import strength_freshness_cpu_readonly as ro
from scripts import strength_freshness_cpu_window_contract as contract
from tests.test_strength_freshness_cpu_observed_kernel import observed_kernel_fixture
from tests.test_strength_freshness_cpu_publication_renewal import Publications, ready
from tests.test_strength_freshness_cpu_collect_records import pointer
from tests.test_strength_freshness_cpu_learner_window import metric

MODEL = "sha256-" + "a" * 64
CHAMPIONS = {MODEL: "b" * 64}
SECRET = "private-before-measurement-sentinel"


def measurement_fixture(tmp_path, monkeypatch):
    folder = tmp_path.resolve()
    metrics = folder / "metrics.jsonl"
    metrics.write_bytes(b"")
    metrics.chmod(0o600)

    def transform(reg, backend):
        reg["scope"]["tails"]["metrics"]["path"] = str(metrics)
        reg["learner_writer"]["binding"].update(
            metrics_path=str(metrics), output_root=str(folder)
        )
        st = metrics.stat()
        reg["learner_writer"]["binding"]["file_identity"] = {
            "path": str(metrics),
            "device": st.st_dev,
            "inode": st.st_ino,
            "uid": st.st_uid,
            "gid": st.st_gid,
            "mode": 0o600,
        }
        for role, uids in reg["kernel_context"]["credentials"].items():
            uids.update(dict.fromkeys(uids, os.geteuid()))
        for name, entry in reg["publication_writers"].items():
            path = folder / (name + ".json")
            reg["scope"]["files"][entry["key"]]["path"] = str(path)
            entry.update(uid=os.geteuid(), gid=os.getegid())
        policy = reg["policy"]["static"]
        profile = b"synthetic registered profile\n"
        manifest = b"synthetic source inventory\n"
        policy.update(
            profile_sha256=m.append.sha(profile),
            source_manifest_sha256=m.append.sha(manifest),
        )
        backend.data.update(profile=profile, source_manifest=manifest)
        profile_path = reg["scope"]["files"][reg["keys"]["profile"]]["path"]
        backend.data["profile_authority"] = (
            m.append.sha(profile) + "  " + profile_path + "\n"
        ).encode()
        backend.data["run_source"] = backend.data["release_source"] = (
            policy["source_commit"] + "\n"
        ).encode()
        backend.data["run"] = m.append.encoded(
            {
                "schema_version": 1,
                "created_ns": 1,
                "run_id": policy["run_id"],
                "generation_family": policy["generation_family"],
            }
        )
        backend.data["continuation"] = m.append.encoded(
            {
                "schema_version": 1,
                "phase": "continuation",
                "continuation_started_ns": policy["continuation_started_ns"],
                "plan_sha256": "a" * 64,
                "profile": profile_path,
                "provisioned_gpus": 8,
                "attempts": [
                    {
                        "pid": reg["policy"]["expected_processes"]["coordinator"][
                            "pid"
                        ],
                        "started_ns": policy["continuation_started_ns"],
                        "status": "running",
                    }
                ],
            }
        )
        champion = pointer()
        champion.update(
            run_id=policy["run_id"], generation_family=policy["generation_family"]
        )
        backend.data["champion"] = m.append.encoded(champion)
        for name, props in backend.props.items():
            for key in m.support.DYNAMIC + m.support.SERVICE_DYNAMIC:
                props.setdefault(key, "0")
            props.update(Job="", NeedDaemonReload="no")

    parsed, io, window, premises, _unused = observed_kernel_fixture(
        monkeypatch, transform
    )
    backend = io._backend
    backend.raw[1] = {"pid": 1, "start_ticks": 1, "ppid": 0, "cgroup": "/init.scope"}
    original_command = backend.command

    def command(argv, deadline, *, charge, allowance):
        if argv[:2] == ("systemctl", "list-jobs"):
            assert ro._readonly_vector(argv)
            backend.touch("command", argv, deadline)
            charge(0)
            return {"stdout": b"", "stderr": b"", "returncode": 0}
        return original_command(argv, deadline, charge=charge, allowance=allowance)

    backend.command = command
    actual_files = ro.System()
    actual_files.monotonic_ns = backend.monotonic_ns
    original_publication = io.read_publication
    pubs = Publications(parsed, io)
    io.read_publication = original_publication
    reg = parsed.private_copy()
    for row in pubs.rows.values():
        row["private_note"] = SECRET
    pubs.rows["learner"]["step"] = reg["policy"]["recipe"]["bootstrap_step"]

    def publish():
        # Test writer only. Production Session has no writer/callback interface.
        stamp = backend.wall_ns()
        for name, row in pubs.rows.items():
            row["timestamp_ns" if name == "coordinator" else "heartbeat_ns"] = stamp
            path = folder / (name + ".json")
            replacement = path.with_suffix(".next")
            replacement.write_bytes(m.append.encoded(row))
            replacement.chmod(0o600)
            replacement.replace(path)

    publish()
    backend.publication_file = actual_files.publication_file
    backend.append_file = actual_files.append_file
    writer = reg["learner_writer"]["binding"]
    bindings = {
        "nonce": "c" * 32,
        "boot_id": window["phase_start"]["boot_id"],
        "request_sha256": "d" * 64,
        "launch_sha256": "e" * 64,
        "registration_sha256": parsed.sha256,
        "policy_sha256": m.digest(reg["policy"]),
        "source_pins_sha256": "f" * 64,
        "external_premises_sha256": m.digest(premises),
        "writer_evidence_sha256": reg["learner_writer"]["evidence_pin"]["sha256"],
        "recipe_sha256": m.digest(reg["policy"]["recipe"]),
        "verified_champions_sha256": m.digest(CHAMPIONS),
        "expected_window_sha256": m.digest(window),
        "source_contract_sha256": writer["source_contract_sha256"],
        "access_sha256": writer["access_sha256"],
        "writer_qualification_sha256": writer["qualification_sha256"],
        "source_qualification_sha256": premises["source_context"][
            "qualification_sha256"
        ],
    }
    session = m.BeforeMeasurementSession(
        parsed,
        io,
        expected_window=window,
        external_premises=premises,
        verified_champions=CHAMPIONS,
        bindings=bindings,
    )

    def append_rows(count=2, *, negative=False):
        recipe = reg["policy"]["recipe"]
        raw = []
        for n in range(1, count + 1):
            backend.advance(0.001)
            row = metric(recipe["bootstrap_step"] + n, 1)
            row.update(timestamp_ns=backend.wall_ns(), **recipe)
            row["ema"] = {"decay": recipe["ema_decay"], "num_updates": n}
            row["private_note"] = SECRET
            row["losses"][SECRET] = 0.25
            if negative and n == 1:
                row["nonfinite_gradient_count"] = 1
            raw.append(m.append.encoded(row))
        with metrics.open("ab") as f:
            for item in raw:
                f.write(item)
            f.flush()
            os.fsync(f.fileno())
        pubs.rows["learner"]["step"] = recipe["bootstrap_step"] + count
        backend.advance(0.001)
        publish()
        return raw

    return (
        session,
        append_rows,
        publish,
        metrics,
        parsed,
        io,
        window,
        premises,
        bindings,
        pubs,
    )


def test_actual_file_measurement_owns_all_evidence_without_authority(
    tmp_path, monkeypatch
):
    s, write, _, _, reg, io, *_ = measurement_fixture(tmp_path, monkeypatch)
    pending = s.begin()
    assert pending["status"] == "pending" and not pending["renewal_complete"]
    write()
    result = s.finish()
    bodies = result.private_copy()
    assert len(bodies["capture_value"]["common"]["progress"]["metrics"]) == 2
    assert SECRET not in json.dumps(bodies) and SECRET not in repr(result)
    assert result.safe_summary()["execution_qualified"] is False
    assert io.maximum_bytes == 24 * 2**20 and io._consumed < io.maximum_bytes
    encoded = contract.encode_before_bodies(
        bodies["capture_value"],
        bodies["provenance_value_without_capture_pin"],
        expected=s._expected,
        registration=reg,
        output_root=str(tmp_path.resolve()),
    )
    assert encoded


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_natural_work_is_terminal_incomplete(tmp_path, monkeypatch, count):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write(count)
    with pytest.raises(m.BeforeIncomplete):
        s.finish()
    used = s._io._consumed
    with pytest.raises(m.BeforeRefusal, match="before-chain-refused"):
        s.finish()
    assert s._io._consumed == used


def test_negative_record_cannot_be_polled_away(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write(negative=True)
    with pytest.raises(m.BeforeRefusal):
        s.finish()
    used = s._io._consumed
    with pytest.raises(m.BeforeRefusal):
        s.observe()
    assert s._io._consumed == used


def test_all_qualifying_metrics_are_retained_not_first_two(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write(5)
    measured = s.finish().private_copy()["capture_value"]
    metrics = measured["common"]["progress"]["metrics"]
    refs = measured["metric_window"]["append_proof"]["qualifying_records"]
    assert len(metrics) == len(refs) == 5
    assert [x["record_ref"] for x in metrics] == [
        {k: x[k] for k in ("start", "end", "sha256")} for x in refs
    ]


def test_excess_selected_metrics_cannot_choose_a_shorter_suffix(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write(101)
    with pytest.raises(m.BeforeRefusal, match="before-metric-count"):
        s.finish()


def test_negative_in_pre_fence_prefix_is_retained(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    write(negative=True)
    s.begin()
    with pytest.raises(m.BeforeRefusal):
        s.finish()


def test_growth_during_final_metadata_tail_cannot_choose_another_eof(
    tmp_path, monkeypatch
):
    s, write, _, metrics, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    actual = s._support.capture

    def grow(**kwargs):
        value = actual(**kwargs)
        with metrics.open("ab") as f:
            f.write(b"new-unread-partial")
        return value

    monkeypatch.setattr(s._support, "capture", grow)
    with pytest.raises(m.BeforeIncomplete, match="before-growth-after-fixed-end"):
        s.finish()
    assert s._e is not None
    end = s._e["value"]["size_at_fstat"]
    assert end < metrics.stat().st_size
    consumed = s._io._consumed
    with pytest.raises(m.BeforeRefusal):
        s.finish()
    assert s._e["value"]["size_at_fstat"] == end and s._io._consumed == consumed


def test_mid_row_end_is_incomplete_without_endpoint_retry(tmp_path, monkeypatch):
    s, write, _, metrics, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    with metrics.open("ab") as f:
        f.write(b'{"schema_version":')
    with pytest.raises(m.BeforeIncomplete, match="unfinished-final-line"):
        s.finish()


@pytest.mark.parametrize("fault", ["source", "profile", "owner", "gpu", "publication"])
def test_final_observed_fact_change_refuses(tmp_path, monkeypatch, fault):
    s, write, publish, _, _, io, _, _, _, pubs = measurement_fixture(
        tmp_path, monkeypatch
    )
    s.begin()
    write()
    if fault in {"source", "profile"}:
        io._backend.data[fault] += b"changed"
    elif fault == "owner":
        io._backend.raw[s._reg["policy"]["expected_processes"]["learner"]["pid"]][
            "start_ticks"
        ] += 1
    elif fault == "gpu":
        io._backend.fault = "foreign-gpu"
    else:
        pubs.rows["learner"]["failure"] = SECRET
        publish()
    with pytest.raises(m.BeforeRefusal) as caught:
        s.finish()
    assert SECRET not in str(caught.value)
    consumed = io._consumed
    with pytest.raises(m.BeforeRefusal):
        s.observe()
    assert io._consumed == consumed


def test_cross_component_clock_regression_is_not_sorted_away(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    actual = s._ranges
    calls = [0]

    def ranges(end):
        result = actual(end)
        calls[0] += 1
        if calls[0] == 2:
            s._io._backend.wall_jump -= 50000
        return result

    monkeypatch.setattr(s, "_ranges", ranges)
    with pytest.raises(m.BeforeRefusal, match="before-observation-clock"):
        s.finish()


def test_final_clock_owner_guard_and_coverage_end_remain_distinct(
    tmp_path, monkeypatch
):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    data = s.finish().private_copy()
    capture = data["capture_value"]
    provenance = data["provenance_value_without_capture_pin"]
    e = capture["metric_window"]["end_fence"]["fence"]["audit"]["read_end"]
    guard = provenance["metric_spans"]["end_guard"]["span"]["audit"]["read_end"]
    owner = provenance["kernel_projections"][-1]
    assert (
        e["monotonic_ns"] < guard["monotonic_ns"] < owner["read_start"]["monotonic_ns"]
    )
    assert owner["read_end"]["monotonic_ns"] < capture["read_end"]["monotonic_ns"]
    assert (
        capture["metric_window"]["append_proof"]["coverage_end_monotonic_ns"]
        == e["monotonic_ns"]
    )
    assert provenance["audit_inventory"][-1]["operation"] == "clock"
    assert provenance["audit_inventory"][-1]["read_end"] == capture["read_end"]


def test_session_context_and_finished_state_cannot_be_reused(tmp_path, monkeypatch):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    assert s.observe()["renewal_complete"]
    s.finish()
    used = s._io._consumed
    for action in (s.begin, s.observe, s.finish):
        with pytest.raises(m.BeforeRefusal):
            action()
    assert s._io._consumed == used


def test_unrenewed_stream_is_incomplete_even_with_two_appends(tmp_path, monkeypatch):
    s, write, _, _, _, io, _, _, _, pubs = measurement_fixture(tmp_path, monkeypatch)
    missing = "actor-gpu-1-cohort-0"
    spec = s._reg["publication_writers"][missing]
    from pathlib import Path

    path = Path(s._reg["scope"]["files"][spec["key"]]["path"])
    original = path.read_bytes()
    s.begin()
    write()
    path.write_bytes(original)
    with pytest.raises(m.BeforeIncomplete, match="renewal-incomplete"):
        s.finish()
    assert pubs.rows["learner"]["step"] > s._reg["policy"]["recipe"]["bootstrap_step"]
    used = io._consumed
    with pytest.raises(m.BeforeRefusal):
        s.observe()
    assert io._consumed == used


@pytest.mark.parametrize("fault", ["io", "deadline", "budget"])
def test_original_session_io_limits_cannot_be_replaced(tmp_path, monkeypatch, fault):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    original = s._io
    if fault == "io":
        s._io = copy(original)
    elif fault == "deadline":
        original.deadline += 1
    else:
        original.maximum_bytes += 1
    used, calls = original._consumed, len(original._backend.calls)
    with pytest.raises(m.BeforeRefusal):
        s.finish()
    assert original._consumed == used and len(original._backend.calls) == calls


def test_unchanged_source_cache_stats_are_present_as_observations(
    tmp_path, monkeypatch
):
    s, write, *_ = measurement_fixture(tmp_path, monkeypatch)
    s.begin()
    write()
    value = s.finish().private_copy()
    projections = value["provenance_value_without_capture_pin"]["kernel_projections"]
    for projected in projections:
        sources = projected["source_facts"]
        assert set(sources["cached_references"]) == set(s._reg["cached_references"])
        for name, row in sources["cached_references"].items():
            assert row["content_hashed"] is False
            assert (
                row["literal_stat"] == s._reg["cached_references"][name]["literal_stat"]
            )
            assert (
                row["resolved_stat"]
                == s._reg["cached_references"][name]["resolved_stat"]
            )
        assert projected["measurement_sha256"] != m.digest(projected)
    first = value["provenance_value_without_capture_pin"]["audit_inventory"][0]
    assert value["capture_value"]["read_start"] == first["read_start"]


def test_tracker_private_records_require_finished_owned_chain(monkeypatch):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    with pytest.raises(m.renewal.RenewalRefusal, match="before-finish"):
        tracker.private_records()
    finished = tracker.finish(observer.capture_current())
    before = deepcopy(finished.safe_proof())
    calls = len(pubs.calls)
    copied = tracker.private_records()
    assert set(copied) == set(m.renewal.STREAMS)
    assert all(
        x["row_sha256"] == m.renewal.digest(x["row"])
        and x["raw_sha256"] == x["audit"]["raw"]["sha256"]
        for x in copied.values()
    )
    copied["learner"]["row"].clear()
    assert tracker.private_records()["learner"]["row"]
    assert finished.safe_proof() == before and len(pubs.calls) == calls
    assert any(x["operation"] == "clock" for x in tracker.private_audits())


@pytest.mark.parametrize("fault", ["row", "audit", "io", "failure"])
def test_tracker_private_access_refuses_tamper_or_failure(monkeypatch, fault):
    tracker, pubs, observer, *_ = ready(monkeypatch)
    tracker.finish(observer.capture_current())
    calls = len(pubs.calls)
    if fault == "row":
        tracker._latest["learner"]["row"]["private_note"] = "changed"
    elif fault == "audit":
        tracker._latest["learner"]["audit"]["raw"]["bytes"] += 1
    elif fault == "io":
        tracker._io = copy(tracker._io)
    else:
        tracker._failure = "observed-negative"
    with pytest.raises(ValueError):
        tracker.private_records()
    assert len(pubs.calls) == calls


def test_pending_operation_clocks_are_actual_detached_audits(tmp_path, monkeypatch):
    s, write, _, _, parsed, io, window, premises, bindings, _ = measurement_fixture(
        tmp_path, monkeypatch
    )
    calls, consumed = len(io._backend.calls), io._consumed
    other = m.BeforeMeasurementSession(
        parsed,
        io,
        expected_window=window,
        external_premises=premises,
        verified_champions=CHAMPIONS,
        bindings=bindings,
    )
    assert len(io._backend.calls) == calls and io._consumed == consumed
    assert not other._inventory  # Constructor performs no unretained observation.
    before = s.begin()
    assert before["read_start"] == s._inventory[0]["read_start"]
    assert before["read_end"] == s._inventory[-1]["read_end"]
    old_end = deepcopy(before["read_end"])
    before["read_end"]["wall_ns"] = 0
    assert s._last_clock == old_end
    write()
    offset = len(s._inventory)
    pending = s.observe()
    assert pending["read_start"] == s._inventory[offset]["read_start"]
    assert pending["read_end"] == s._inventory[-1]["read_end"]
    m.append.order(old_end, pending["read_start"])
    assert pending["status"] == "pending" and not pending["execution_authorized"]
    pending["read_start"]["wall_ns"] = 0
    assert s._inventory[offset]["read_start"]["wall_ns"] > 0
