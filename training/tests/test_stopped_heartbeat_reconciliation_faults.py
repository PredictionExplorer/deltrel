"""Fault checks for telemetry-only recovery at an immutable stopped boundary."""

import json
import os
from pathlib import Path

import pytest

from scripts import graceful_training_deploy as deploy
from test_stopped_heartbeat_reconciliation import reconciliation_fixture, write


def run(case):
    return deploy.reconcile_stopped_heartbeat(
        case.root, case.evidence, expected_workers=case.workers
    )


def write_metrics(case):
    case.metrics.write_text("".join(json.dumps(row) + "\n" for row in case.rows))


def assert_rejected_without_writes(case):
    heartbeat = case.root / "status/learner.heartbeat.json"
    original = heartbeat.read_bytes()
    checkpoint = case.checkpoint.read_bytes()
    pointer = (case.root / "learner/recovery.json").read_bytes()
    with pytest.raises((RuntimeError, ValueError, OSError)):
        run(case)
    assert heartbeat.read_bytes() == original
    assert case.checkpoint.read_bytes() == checkpoint
    assert (case.root / "learner/recovery.json").read_bytes() == pointer


@pytest.mark.parametrize(
    "defect", ["duplicate", "nonfinite", "overflow", "nonobject", "utf8", "partial"]
)
def test_malformed_metric_evidence_never_authorizes_reconciliation(tmp_path, defect):
    case = reconciliation_fixture(tmp_path)
    if defect == "duplicate":
        line = json.dumps(case.rows[0])
        line = line[:-1] + ', "step": 25}\n'
        case.metrics.write_text(
            line + "".join(json.dumps(row) + "\n" for row in case.rows[1:])
        )
    elif defect == "nonfinite":
        with case.metrics.open("a") as stream:
            stream.write('{"worker":"learner","norm":NaN}\n')
    elif defect == "overflow":
        with case.metrics.open("a") as stream:
            stream.write('{"worker":"learner","norm":1e999}\n')
    elif defect == "nonobject":
        with case.metrics.open("a") as stream:
            stream.write("[]\n")
    elif defect == "utf8":
        with case.metrics.open("ab") as stream:
            stream.write(b"\xff\n")
    else:
        with case.metrics.open("a") as stream:
            stream.write('{"unfinished":')
    assert_rejected_without_writes(case)


@pytest.mark.parametrize(
    "index,field,value",
    [
        (0, "step", 25.0),
        (0, "examples_consumed", 12800.0),
        (1, "window_batches_consumed_this_spin", True),
        (1, "epoch", 7.0),
        (2, "schema_version", True),
        (2, "checkpoint_bytes", 1.0),
    ],
)
def test_selected_metric_numbers_require_exact_types(tmp_path, index, field, value):
    case = reconciliation_fixture(tmp_path)
    case.rows[index][field] = value
    write_metrics(case)
    assert_rejected_without_writes(case)


@pytest.mark.parametrize(
    "field", ["window_batches_allocated", "window_selection_max_shard_id"]
)
def test_missing_window_identity_on_both_sides_is_not_a_match(tmp_path, field):
    case = reconciliation_fixture(tmp_path)
    del case.rows[1][field]
    del case.heartbeat[field]
    write_metrics(case)
    write(case.root / "status/learner.heartbeat.json", case.heartbeat)
    assert_rejected_without_writes(case)


@pytest.mark.parametrize(
    "record",
    [
        {"worker": "learner", "event": "progress", "step": 27.0},
        {"worker": "learner", "event": "progress", "examples_consumed": "999999"},
        {"worker": "learner", "event": "progress", "step": True},
    ],
)
def test_noncanonical_later_progress_is_not_silently_ignored(tmp_path, record):
    case = reconciliation_fixture(tmp_path)
    case.rows.append(record)
    write_metrics(case)
    assert_rejected_without_writes(case)


@pytest.mark.parametrize(
    "logical", ["status/learner.heartbeat.json", "status/coordinator.json", "run.json"]
)
def test_control_schema_types_are_strict(tmp_path, logical):
    case = reconciliation_fixture(tmp_path)
    path = case.root / logical
    value = json.loads(path.read_text())
    value["schema_version"] = True
    write(path, value)
    assert_rejected_without_writes(case)


def test_missing_family_on_both_controls_cannot_disable_checkpoint_identity_check(
    tmp_path,
):
    case = reconciliation_fixture(tmp_path)
    for logical in ("run.json", "learner/recovery.json"):
        path = case.root / logical
        payload = json.loads(path.read_text())
        del payload["generation_family"]
        write(path, payload)
    assert_rejected_without_writes(case)


@pytest.mark.parametrize("folder", ["status", "learner/recovery"])
def test_parent_directory_symlink_cannot_redirect_reconciliation(tmp_path, folder):
    case = reconciliation_fixture(tmp_path)
    original = case.root / folder
    outside = tmp_path / "outside"
    original.rename(outside)
    original.symlink_to(outside, target_is_directory=True)
    assert_rejected_without_writes(case)


@pytest.mark.parametrize(
    "defect", ["metrics_append", "metrics_same_length", "checkpoint", "profile"]
)
def test_source_race_after_durable_proof_is_detected_before_telemetry_write(
    tmp_path, monkeypatch, defect
):
    case = reconciliation_fixture(tmp_path)
    heartbeat = case.root / "status/learner.heartbeat.json"
    before = heartbeat.read_bytes()
    credit = case.root / "learner/utd-segment.json"
    credit.write_text('{"credit":"must remain unchanged"}\n')
    credit_bytes = credit.read_bytes()
    original = deploy.atomic_json
    frozen_path = case.metrics if defect == "metrics_same_length" else case.checkpoint
    frozen_stat = frozen_path.stat()
    original_stat = Path.stat

    def unchanged_stat(self, *args, **kwargs):
        return (
            frozen_stat if self == frozen_path else original_stat(self, *args, **kwargs)
        )

    if defect in ("metrics_same_length", "checkpoint"):
        monkeypatch.setattr(Path, "stat", unchanged_stat)

    def publish_then_race(path, payload):
        original(path, payload)
        if path.parent != case.evidence:
            return
        if defect == "metrics_append":
            with case.metrics.open("a") as stream:
                stream.write('{"worker":"learner","step":27}\n')
        elif defect == "metrics_same_length":
            contents = case.metrics.read_bytes()
            changed = contents.replace(b'"step": 25', b'"step": 24', 1)
            assert changed != contents and len(changed) == len(contents)
            case.metrics.write_bytes(changed)
        elif defect == "checkpoint":
            contents = case.checkpoint.read_bytes()
            case.checkpoint.write_bytes(b"!" + contents[1:])
        else:
            case.profile.write_text(case.profile.read_text() + "\n")

    monkeypatch.setattr(deploy, "atomic_json", publish_then_race)
    with pytest.raises(RuntimeError, match="changed|checksum"):
        run(case)
    assert heartbeat.read_bytes() == before
    assert credit.read_bytes() == credit_bytes


@pytest.mark.parametrize("after_write", [False, True])
def test_crash_between_proof_and_correction_has_an_idempotent_retry(
    tmp_path, monkeypatch, after_write
):
    case = reconciliation_fixture(tmp_path, missing=3)
    heartbeat = case.root / "status/learner.heartbeat.json"
    before = heartbeat.read_bytes()
    checkpoint_bytes = case.checkpoint.read_bytes()
    pointer_bytes = (case.root / "learner/recovery.json").read_bytes()
    original = deploy.atomic_json

    def crash(path, payload):
        if path == heartbeat:
            if after_write:
                original(path, payload)
            raise RuntimeError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr(deploy, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(case)
    if not after_write:
        assert heartbeat.read_bytes() == before
    assert len(list(case.evidence.glob("stopped-heartbeat-reconciliation-*.json"))) == 1
    monkeypatch.setattr(deploy, "atomic_json", original)
    result = run(case)
    if after_write:
        assert result is None
    else:
        assert result["reconciled_batches"] == 3
    assert (
        json.loads(heartbeat.read_text())["examples_consumed"]
        == case.pointer["examples_consumed"]
    )
    assert run(case) is None
    assert case.checkpoint.read_bytes() == checkpoint_bytes
    assert (case.root / "learner/recovery.json").read_bytes() == pointer_bytes


def test_reconciled_counter_does_not_hide_a_tampered_proof(tmp_path):
    case = reconciliation_fixture(tmp_path)
    result = run(case)
    proof = Path(result["proof"])
    proof.chmod(0o644)
    proof.write_text(proof.read_text() + "\n")
    assert_rejected_without_writes(case)


def test_live_process_identifier_never_authorizes_reconciliation(tmp_path, monkeypatch):
    case = reconciliation_fixture(tmp_path)
    case.heartbeat["pid"] = os.getpid()
    write(case.root / "status/learner.heartbeat.json", case.heartbeat)
    original = Path.exists
    live = Path(f"/proc/{os.getpid()}")
    monkeypatch.setattr(Path, "exists", lambda self: self == live or original(self))
    assert_rejected_without_writes(case)
