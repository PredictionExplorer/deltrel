"""Local contract and unconditional-launch refusal; no process execution grant."""

from __future__ import annotations

import copy
import subprocess
import sys
from typing import Any

import pytest

from scripts import strength_freshness_cpu_observed_capture_cli as m


PIN = {"path": "/input/launch.json", "sha256": "a" * 64, "bytes": 123}
PYTHON = "/qualified/.venv/bin/python"
CONTROL = "/qualified/control"
DEADLINE = 123_000_000_000


def spec(**changes):
    fields: dict[str, Any] = dict(
        python=PYTHON,
        control_root=CONTROL,
        launch_path=PIN["path"],
        launch_sha256=PIN["sha256"],
        deadline_monotonic_ns=DEADLINE,
    )
    fields.update(changes)
    return m.ObservedCaptureSpec(**fields)


def test_fixed_before_contract_has_one_separate_module_and_same_resources():
    new = spec()
    old = m.outer.CaptureSpec(
        "before", PYTHON, CONTROL, PIN["path"], PIN["sha256"], DEADLINE
    )
    expected = copy.deepcopy(old.contract())
    argv = list(expected["argv"])
    argv[5] = m.MODULE
    expected["argv"] = tuple(argv)
    assert new.contract() == expected
    assert not isinstance(new, m.outer.CaptureSpec)
    assert m.observed_collector_contract_sha256(
        PYTHON, CONTROL, PIN, DEADLINE
    ) == m.outer.digest(expected)
    assert m.completion.collector_contract_sha256(
        PYTHON, CONTROL, PIN, DEADLINE
    ) == m.outer.digest(old.contract())
    assert new.argv()[5] != old.argv()[5]


@pytest.mark.parametrize(
    "change",
    [
        {"python": "/qualified/python3"},
        {"control_root": "relative"},
        {"control_root": "/qualified/../control"},
        {"launch_path": "/input/../launch.json"},
        {"launch_sha256": "x" * 64},
        {"deadline_monotonic_ns": 0},
        {"deadline_monotonic_ns": True},
    ],
)
def test_spec_refuses_noncanonical_contract(change):
    with pytest.raises(m.outer.OuterRefusal):
        spec(**change).contract()


@pytest.mark.parametrize("bad", [True, 0, -1, "123"])
def test_pin_count_is_not_a_typed_shortcut(bad):
    with pytest.raises(m.completion.CompletionRefusal):
        m.observed_collector_contract_sha256(
            PYTHON, CONTROL, {**PIN, "bytes": bad}, DEADLINE
        )


@pytest.mark.parametrize(
    "argv",
    [None, [], ["--help"], ["--qualified", "true"], ["--launch", "/private/sentinel"]],
)
def test_actual_cli_refuses_without_observing_inputs(argv, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not inspect or collect")

    monkeypatch.setattr(m, "collect_before_artifacts", forbidden)
    monkeypatch.setattr(m, "open", forbidden, raising=False)
    assert m.main(argv) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == m.ACTUAL_LAUNCH_REFUSAL + "\n"
    assert "sentinel" not in output.err


def test_module_entrypoint_cannot_be_enabled_by_json_or_flags():
    run = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            m.MODULE,
            "--launch",
            "/private/sentinel",
            "--qualified",
            "true",
        ],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert run.returncode == 2 and run.stdout == b""
    assert run.stderr.decode() == m.ACTUAL_LAUNCH_REFUSAL + "\n"


def producer_fixture(tmp_path, monkeypatch, *, count=2):
    """Real private file IO, synthetic kernel/source admission and test writer."""
    from types import SimpleNamespace

    from tests.test_strength_freshness_cpu_before_measurement import measurement_fixture

    fixture = measurement_fixture(tmp_path, monkeypatch)
    (
        session,
        write,
        _publish,
        _path,
        registration,
        io,
        window,
        premises,
        bindings,
        _pubs,
    ) = fixture
    backend = io._backend
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        backend.advance(seconds)
        if len(sleeps) == 1:
            # This writer is a test fixture, never an argument to the producer.
            write(count)

    monkeypatch.setattr(
        m,
        "time",
        SimpleNamespace(
            monotonic_ns=backend.monotonic_ns,
            time_ns=backend.wall_ns,
            sleep=sleep,
        ),
    )
    kwargs = dict(
        expected_window=window,
        external_premises=premises,
        verified_champions=session._champions,
        bindings=bindings,
        expected=session._expected,
        output_root=str(tmp_path.resolve()),
    )
    return registration, io, kwargs, sleeps, backend


def test_local_producer_owns_real_helpers_and_does_not_publish(tmp_path, monkeypatch):
    import json

    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    result = m.collect_before_artifacts(registration, io, **kwargs)
    capture = json.loads(result["capture"])
    assert len(capture["common"]["progress"]["metrics"]) == 2
    assert sleeps == [m.POLL_SECONDS]
    assert not (tmp_path / "r3-before.json").exists()
    assert not (tmp_path / "r3-before.receipt.json").exists()
    assert "private-before-measurement-sentinel" not in repr(result)
    assert io.maximum_bytes == 24 * 2**20


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_work_does_not_request_another_window(
    tmp_path, monkeypatch, count
):
    registration, io, kwargs, sleeps, _ = producer_fixture(
        tmp_path, monkeypatch, count=count
    )
    original_deadline = io.deadline
    with pytest.raises(
        m.ProducerRefusal, match="observed-producer-incomplete-or-refused"
    ):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert sleeps == [m.POLL_SECONDS] and io.deadline == original_deadline
    assert not (tmp_path / "r3-before.json").exists()


@pytest.mark.parametrize("field", ["bindings", "expected_window", "verified_champions"])
def test_inconsistent_independent_inputs_refuse_before_io(tmp_path, monkeypatch, field):
    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    kwargs[field] = {}
    original = io._consumed
    with pytest.raises(m.ProducerRefusal):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert io._consumed == original and not sleeps


def test_late_encoding_cannot_return_a_success_body(tmp_path, monkeypatch):
    from scripts import strength_freshness_cpu_window_contract as bodies

    registration, io, kwargs, sleeps, backend = producer_fixture(tmp_path, monkeypatch)
    original = bodies.encode_before_bodies

    def late(*args, **kw):
        result = original(*args, **kw)
        backend.advance(200)
        return result

    monkeypatch.setattr(bodies, "encode_before_bodies", late)
    with pytest.raises(m.ProducerRefusal, match="observed-producer-original-deadline"):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert sleeps == [m.POLL_SECONDS]
    assert not (tmp_path / "r3-before.json").exists()


@pytest.mark.parametrize("axis", ["monotonic_ns", "time_ns"])
def test_clock_regression_during_encoding_refuses(tmp_path, monkeypatch, axis):
    from scripts import strength_freshness_cpu_window_contract as bodies

    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    original = bodies.encode_before_bodies
    clock = getattr(m.time, axis)

    def backwards(*args, **kw):
        result = original(*args, **kw)
        observed = clock()
        setattr(m.time, axis, lambda: observed - 10**9)
        return result

    monkeypatch.setattr(bodies, "encode_before_bodies", backwards)
    with pytest.raises(m.ProducerRefusal, match="observed-producer-clock-regression"):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert sleeps == [m.POLL_SECONDS]


@pytest.mark.parametrize("value", [True, 0, 1.5])
def test_control_clocks_are_literal_nanosecond_integers(tmp_path, monkeypatch, value):
    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    m.time.monotonic_ns = lambda: value
    original = io._consumed
    with pytest.raises(m.ProducerRefusal, match="observed-producer-clock-type"):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert io._consumed == original and not sleeps


@pytest.mark.parametrize("axis", ["monotonic_ns", "time_ns"])
def test_one_bad_control_sample_after_begin_cannot_recover_unnoticed(
    tmp_path, monkeypatch, axis
):
    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    original_begin = m.measurement.BeforeMeasurementSession.begin
    original_observe = m.measurement.BeforeMeasurementSession.observe
    field = "wall_ns" if axis == "time_ns" else axis
    calls = {"control": 0, "observe": 0}

    def begin(session):
        pending = original_begin(session)
        actual = getattr(m.time, axis)

        def sample():
            calls["control"] += 1
            return pending["read_end"][field] - 1 if calls["control"] == 1 else actual()

        setattr(m.time, axis, sample)
        return pending

    def observe(session):
        calls["observe"] += 1
        return original_observe(session)

    monkeypatch.setattr(m.measurement.BeforeMeasurementSession, "begin", begin)
    monkeypatch.setattr(m.measurement.BeforeMeasurementSession, "observe", observe)
    with pytest.raises(m.ProducerRefusal, match="observed-producer-clock-regression"):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert calls == {"control": 1, "observe": 0} and not sleeps


@pytest.mark.parametrize("axis", ["monotonic_ns", "time_ns"])
def test_control_sample_before_io_must_precede_its_start_even_if_end_recovers(
    tmp_path, monkeypatch, axis
):
    registration, io, kwargs, sleeps, _ = producer_fixture(tmp_path, monkeypatch)
    original_begin = m.measurement.BeforeMeasurementSession.begin
    original_observe = m.measurement.BeforeMeasurementSession.observe
    field = "wall_ns" if axis == "time_ns" else axis
    calls = {"control": 0}
    observed = {}

    def begin(session):
        pending = original_begin(session)
        actual = getattr(m.time, axis)

        def sample():
            calls["control"] += 1
            value = actual()
            if calls["control"] == 2:
                observed["control"] = value + 1_000_000
                return value + 1_000_000
            return value

        setattr(m.time, axis, sample)
        return pending

    def observe(session):
        pending = original_observe(session)
        observed["start"] = pending["read_start"][field]
        observed["end"] = pending["read_end"][field]
        return pending

    monkeypatch.setattr(m.measurement.BeforeMeasurementSession, "begin", begin)
    monkeypatch.setattr(m.measurement.BeforeMeasurementSession, "observe", observe)
    with pytest.raises(
        m.ProducerRefusal, match="observed-producer-incomplete-or-refused"
    ):
        m.collect_before_artifacts(registration, io, **kwargs)
    assert observed["start"] < observed["control"] <= observed["end"]
    assert not sleeps
