"""Tiny real proof files exercise copying; only the DR snapshot boundary is fake."""

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from deltreltrain import strength_freshness_guard as core
from scripts import backup_active_profile as backup
from scripts import strength_freshness_linux as linux
from scripts import training_disaster_recovery as dr


class LocalIO(linux.LinuxIO):
    def command(self, *args, **kwargs):
        pytest.fail("proof tests must not issue host commands")

    def boot_id(self):
        return "proof-test-boot"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(linux.encoded(value))
    return linux._file_pin(path)


@pytest.fixture
def proof(tmp_path, monkeypatch):
    host = linux.LinuxHost.__new__(linux.LinuxHost)
    host.io = LocalIO()
    host.execute = True
    host.state, host.root = tmp_path / "guard", tmp_path / "run"
    core.Journal(host.state).create()
    host.root.mkdir()
    host.plan = {
        "attempt_id": "proof-fixture",
        "run_identity": {
            "run_id": "run",
            "generation_family": "family",
            "created_ns": 1,
        },
        "runtime_root": str(tmp_path / "runtime/training"),
        "probe_output": str(tmp_path / "probe"),
    }
    archive, backup_root = tmp_path / "proofs", tmp_path / "dr"
    archive.mkdir()
    backup_root.mkdir()
    original = host.root / "learner/recovery/checkpoint.pt"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"tiny checkpoint fixture; no model inference")
    checkpoint = linux._file_pin(original)
    retained = tmp_path / "retained" / checkpoint["sha256"]
    retained.parent.mkdir()
    os.link(original, retained)
    retained_pin = linux._file_pin(retained)
    boundary = {
        "checkpoint": checkpoint,
        "checkpoint_retention": retained_pin,
        "step": 42,
    }
    core.Journal(host.state).save("state.json", {"nonce": "n", "boundary": boundary})
    metadata = write(host.state / "mutable-status.json", {"value": "before"})
    helper_journal = write(
        host.state / "linux-support-transitions/one.intent.json",
        {"stage": "after", "nonce": "n"},
    )
    probe_metadata = write(
        Path(host.plan["probe_output"]) / "result.json",
        {"status": "fixture-only", "work": 0},
    )
    (Path(host.plan["probe_output"]) / "observations.jsonl").write_bytes(
        b'{"fixture":"CPU only"}\n'
    )
    implementation = []
    for name in (
        "strength_freshness_guard.py",
        "strength_freshness_linux.py",
        "strength_freshness_units.py",
    ):
        path = tmp_path / "implementation" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(
            ("# tiny pinned implementation fixture: " + name + "\n").encode()
        )
        implementation.append(linux._file_pin(path))
    core_plan = write(tmp_path / "core-plan.json", host.plan)
    qualification = write(
        tmp_path / "qualification.json",
        {"status": "CPU-fixture-only", "execution_qualified": False},
    )
    teacher = write(tmp_path / "teacher-proof.json", {"proof": "fixture-only"})
    host.manifest = {
        "proof_archive_root": str(archive),
        "backup_root": str(backup_root),
        "backup_mount": str(backup_root),
        "proof_inputs": [teacher, checkpoint],
        "implementation_pins": implementation,
        "guard_plan": core_plan,
        "qualification": qualification,
        "backup_worker_unit": "proof-backup.service",
    }
    host.manifest_path = tmp_path / "adapter-manifest.json"
    write(host.manifest_path, host.manifest)
    host.manifest_sha256 = linux.checksum(host.manifest_path.read_bytes())
    prepared = host.prepared_directory()
    prepared.mkdir(mode=0o700)
    authorization = write(
        prepared / linux.AUTHORIZATION,
        {
            "manifest_sha256": host.manifest_sha256,
            "core_plan_sha256": core.digest(host.plan),
        },
    )
    closure = write(
        host.state / "proof-closure.json",
        {"required": [teacher["sha256"], core.digest(boundary)]},
    )
    request = {
        "guard_plan_sha256": core.digest(host.plan),
        "nonce": "n",
        "support_stage": "after",
        "required_proof_sha256": [
            teacher["sha256"],
            checkpoint["sha256"],
            core.digest(boundary),
        ],
        "proof_closure_sha256": closure["sha256"],
        "deadline_wall_ns": time.time_ns() + 60_000_000_000,
    }
    write(host.state / "linux-backup-request.json", request)
    snapshot = backup_root / "snapshots/run/catalog.json"
    calls = []

    def capture(run_root, destination, *, expected_backup_mount):
        assert run_root == host.root and destination == backup_root
        assert expected_backup_mount == backup_root
        calls.append("DR snapshot boundary")
        pin = write(snapshot, {"run_id": "run", "generation_family": "family"})
        write(dr._snapshot_commit_path(snapshot), {"sha256": pin["sha256"]})
        return snapshot

    def envelope(path, destination):
        assert path == snapshot and destination == backup_root
        raw = path.read_bytes()
        return host.io.json(path, host.io.now() + 5), {}, raw, linux.checksum(raw)

    monkeypatch.setattr(backup, "backup_active_profile", capture)
    monkeypatch.setattr(dr, "_snapshot_envelope", envelope)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    expected_small = [
        metadata,
        helper_journal,
        probe_metadata,
        authorization,
        closure,
        core_plan,
        qualification,
        linux._file_pin(host.manifest_path),
        *implementation,
    ]
    return SimpleNamespace(
        host=host,
        request=request,
        archive=archive / "proof-fixture",
        original=original,
        retained=retained,
        metadata=metadata,
        expected_small=expected_small,
        calls=calls,
    )


def test_mutating_guard_metadata_during_copy_cannot_change_frozen_objects(
    proof, monkeypatch
):
    p = proof
    original_open = Path.open
    changed = False
    source = Path(p.metadata["path"])
    state_path = p.host.state / "state.json"
    original_state = state_path.read_bytes()
    state_sha = linux.checksum(original_state)

    def open_file(path, mode="r", *args, **kwargs):
        nonlocal changed
        if path.parent == p.archive / "objects" and mode == "xb" and not changed:
            changed = True
            with original_open(source, "wb") as stream:
                stream.write(
                    b'{"value":"changed after freeze, before object copying"}\n'
                )
            advanced = json.loads(original_state)
            advanced["revision"] = 2
            with original_open(state_path, "wb") as stream:
                stream.write(linux.encoded(advanced))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    result = linux._backup_worker(p.host, p.request)
    assert changed and result["core_result"]["status"] == "committed"
    frozen = p.archive / "frozen" / p.metadata["sha256"]
    obj = p.archive / "objects" / p.metadata["sha256"]
    assert frozen.read_bytes() == obj.read_bytes() == linux.encoded({"value": "before"})
    assert source.read_bytes() != frozen.read_bytes()
    assert frozen.stat().st_ino != source.stat().st_ino
    assert not frozen.stat().st_mode & 0o222
    assert state_path.read_bytes() != original_state
    assert (p.archive / "frozen" / state_sha).read_bytes() == original_state
    assert (p.archive / "objects" / state_sha).read_bytes() == original_state


def test_retained_checkpoint_survives_original_unlink_and_all_control_dependencies_are_copied(
    proof,
):
    p = proof
    expected_checkpoint = p.retained.read_bytes()
    p.original.unlink()
    result = linux._backup_worker(p.host, p.request)
    objects = {pin["sha256"]: pin for pin in result["retained_objects"]}
    required = set(p.request["required_proof_sha256"])
    assert required <= set(objects)
    assert {pin["sha256"] for pin in p.expected_small} <= set(objects)
    checkpoint_sha = linux.checksum(expected_checkpoint)
    copied = Path(objects[checkpoint_sha]["path"])
    assert copied.read_bytes() == expected_checkpoint
    assert copied.stat().st_ino != p.retained.stat().st_ino
    for pin in result["retained_objects"]:
        file = Path(pin["path"])
        assert linux._file_pin(file) == pin
        assert not file.stat().st_mode & 0o222
    manifest = p.host.io.json(p.archive / "manifest.json", p.host.io.now() + 5)
    assert manifest["objects"] == result["retained_objects"]
    assert set(manifest["required_core_sha256"]) == required
    assert set(result["core_result"]["verified_artifact_sha256"]) == required
    assert p.calls == ["DR snapshot boundary"]


def test_interrupted_object_copy_never_publishes_success(proof, monkeypatch):
    p = proof
    original_open = Path.open
    interrupted = []

    @contextmanager
    def failed_writer(path, *args, **kwargs):
        with original_open(path, "xb", *args, **kwargs) as stream:

            class Writer:
                def write(self, data):
                    stream.write(data[:3])
                    interrupted.append(path)
                    raise OSError("injected partial archive copy")

            yield Writer()

    def open_file(path, mode="r", *args, **kwargs):
        if path.parent == p.archive / "objects" and mode == "xb":
            return failed_writer(path, *args, **kwargs)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError, match="partial archive"):
        linux._backup_worker(p.host, p.request)
    assert interrupted and interrupted[0].stat().st_size == 3
    assert not (p.archive / "manifest.json").exists()
    assert not (p.host.state / "linux-backup-result.json").exists()
    assert not p.calls
    with pytest.raises(core.Refusal, match="archive-exists"):
        linux._backup_worker(p.host, p.request)


def test_finished_proof_copy_is_not_success_when_DR_snapshot_fails(proof, monkeypatch):
    p = proof

    def fail(*args, **kwargs):
        raise OSError("injected DR failure")

    monkeypatch.setattr(backup, "backup_active_profile", fail)
    with pytest.raises(OSError, match="DR failure"):
        linux._backup_worker(p.host, p.request)
    assert (p.archive / "manifest.json").is_file()
    assert not (p.host.state / "linux-backup-result.json").exists()


def completed_child(proof, monkeypatch):
    host = proof.host
    process = core.Process(os.getpid(), 12345)
    active = core.Unit(
        host.manifest["backup_worker_unit"],
        "b" * 64,
        "/system.slice/proof-backup.service",
        invocation_id="a" * 32,
        main=process,
        members=(process,),
        active="active",
        substate="running",
        entered_monotonic=host.io.now() - 1,
    )
    observations = []

    def unit(*args):
        observations.append("child writes its own started receipt")
        return active, {}

    monkeypatch.setattr(host, "_unit", unit)
    monkeypatch.setattr(host, "_jobs", lambda _: {})
    monkeypatch.setattr(
        host,
        "_lease",
        lambda _: {
            "plan_sha256": core.digest(host.plan),
            "nonce": "n",
            "boot_id": host.io.boot_id(),
        },
    )
    result = linux.worker(host, "backup", proof.request)
    assert observations == ["child writes its own started receipt"]
    # This is the first parent observation, already inactive with cleared InvocationID.
    host._exit_pids = {active.name: process.pid}
    dead = replace(
        active,
        main=None,
        members=(),
        active="inactive",
        substate="dead",
        invocation_id="",
        result="success",
        exit_code=0,
    )
    return result, dead


def test_clean_fast_child_is_verifiable_without_any_live_parent_poll(
    proof, monkeypatch
):
    result, dead = completed_child(proof, monkeypatch)
    assert dead.dead and dead.invocation_id == ""
    observed = proof.host._backup_result(proof.host.io.now() + 5, dead)
    assert observed == result["core_result"]


@pytest.mark.parametrize(
    "fault", ["pid", "start-time", "request", "nonce", "boot", "exit", "still-live"]
)
def test_fast_exit_proof_still_requires_exact_start_request_and_clean_death(
    proof, monkeypatch, fault
):
    _, dead = completed_child(proof, monkeypatch)
    path = proof.host.state / "linux-backup-started.json"
    started = proof.host.io.json(path, proof.host.io.now() + 5)
    if fault == "pid":
        proof.host._exit_pids[dead.name] += 1
    elif fault == "start-time":
        dead = replace(dead, entered_monotonic=dead.entered_monotonic + 1)
    elif fault == "request":
        started["request_sha256"] = "0" * 64
    elif fault == "nonce":
        started["nonce"] = "foreign"
    elif fault == "boot":
        started["boot_id"] = "old-boot"
    elif fault == "exit":
        dead = replace(dead, result="exit-code", exit_code=1)
    else:
        dead = replace(
            dead,
            active="active",
            main=core.Process(999, 9),
            members=(core.Process(999, 9),),
        )
    write(path, started)
    if fault == "still-live":
        assert proof.host._backup_result(proof.host.io.now() + 5, dead) is None
    else:
        with pytest.raises(core.Refusal):
            proof.host._backup_result(proof.host.io.now() + 5, dead)
