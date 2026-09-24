from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil

import pytest

from scripts import preserve_replay_snapshot as preservation


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fixture(tmp_path):
    root = tmp_path / "run"
    for logical in (
        "replay/shards",
        "learner/recovery",
        "learner/checkpoints",
        "learner/manifests",
        "status",
        "arena",
    ):
        (root / logical).mkdir(parents=True, exist_ok=True)
    write_json(
        root / "run.json", {"run_id": "run-test", "generation_family": "family-test"}
    )
    write_json(
        root / "status/coordinator.json",
        {
            "state": "stopped",
            "failure": None,
            "coordinator_pid": None,
            "workers": {
                "learner": {"state": "stopped", "last_exit_code": 0, "pid": None}
            },
        },
    )
    write_json(
        root / "status/learner.heartbeat.json",
        {"phase": "stopped", "pid": None, "step": 25, "examples_consumed": 125},
    )
    data = b"final checkpoint"
    checksum = digest(data)
    (root / f"learner/recovery/sha256-{checksum}.pt").write_bytes(data)
    write_json(
        root / "learner/recovery.json",
        {
            "checkpoint": f"recovery/sha256-{checksum}.pt",
            "checkpoint_sha256": checksum,
            "checkpoint_bytes": len(data),
            "step": 25,
            "examples_consumed": 125,
            "run_id": "run-test",
            "generation_family": "family-test",
        },
    )
    model = b"model weights"
    (root / f"learner/checkpoints/sha256-{digest(model)}.pt").write_bytes(model)
    manifest = b'{"model":"frozen"}'
    (root / f"learner/manifests/manifest-{digest(manifest)}.json").write_bytes(manifest)
    write_json(
        root / "arena/pending.resume.json", {"game_states": [{"actions": [1, 2]}]}
    )
    (root / "arena/events.jsonl").write_text('{"event":"saved"}\n')
    (root / "continuous-migrations.jsonl").write_text('{"source":"old"}\n')
    (root / "source-commit.txt").write_text("a" * 40 + "\n")
    (root / "learner/metrics.jsonl").write_text("large unneeded history\n")
    with sqlite3.connect(root / "replay/manifest.sqlite3") as db:
        db.executescript(
            "CREATE TABLE shards(id INTEGER PRIMARY KEY,relative_path TEXT,checksum_sha256 TEXT,sample_count INTEGER,ring INTEGER,state TEXT,variant TEXT,segment TEXT); CREATE TABLE run_counters(run_id TEXT,committed_samples INTEGER);"
        )
        db.execute("INSERT INTO run_counters VALUES ('run-test',17)")
        for index, state in enumerate(("ready", "superseded"), start=1):
            contents = f"immutable {state} payload".encode()
            logical = f"shards/shard-{index}.npz"
            (root / "replay" / logical).write_bytes(contents)
            db.execute(
                "INSERT INTO shards VALUES (?,?,?,?,?,?,?,?)",
                (
                    index,
                    logical,
                    digest(contents),
                    index * 4,
                    10,
                    state,
                    "pie-classic",
                    "pie",
                ),
            )
    return root, tmp_path / "archive"


def test_preservation_links_payloads_copies_database_and_survives_source_gc(tmp_path):
    root, target = fixture(tmp_path)
    result = preservation.preserve_stopped_snapshot(root, target)
    assert result["checkpoint_step"] == 25 and result["replay_shards"] == 2
    assert result["payload_checksum_verification"] == "deferred-to-offline-verify"
    assert not (target / "learner/metrics.jsonl").exists()
    for shard in (root / "replay/shards").iterdir():
        archived = target / shard.relative_to(root)
        assert archived.stat().st_ino == shard.stat().st_ino
        shard.unlink()
    assert (root / "replay/manifest.sqlite3").stat().st_ino != (
        target / "replay/manifest.sqlite3"
    ).stat().st_ino
    write_json(root / "learner/recovery.json", {"changed": True})
    with sqlite3.connect(root / "replay/manifest.sqlite3") as db:
        db.execute("DELETE FROM shards")
    assert preservation.verify_snapshot(target)["status"] == "verified"
    assert (target / "arena/pending.resume.json").stat().st_ino != (
        root / "arena/pending.resume.json"
    ).stat().st_ino
    with pytest.raises(FileExistsError):
        preservation.preserve_stopped_snapshot(root, target)


def measurement_fixture(tmp_path, *, terminal=True):
    from deltreltrain.measurement_scheduling import MeasurementServiceLedger
    from deltreltrain.runtime import append_jsonl, load_run_identity

    root, target = fixture(tmp_path)
    write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": "run-test",
            "generation_family": "family-test",
            "created_ns": 1,
        },
    )
    journal = root / "metrics/coordinator.jsonl"
    journal.parent.mkdir()
    journal.write_bytes(b"")
    options = dict(
        path=root / "arena/measurement-service.json",
        coordinator_events=journal,
        request_path=root / "status/arena-gpu-pause.json",
        run_identity=load_run_identity(root / "run.json"),
    )
    service = MeasurementServiceLedger(**options)

    def event(token, kind, seconds):
        append_jsonl(
            journal, {"token": token, "event": kind, "timestamp_ns": seconds * 10**9}
        )

    service.register_lease("promotion", "promotion", 0.2)
    event("promotion", "pause_lease_ready", 100)
    event("promotion", "pause_lease_released", 200)
    service.refresh()

    def model(label):
        weights = label.encode()
        checkpoint = root / f"learner/checkpoints/sha256-{digest(weights)}.pt"
        checkpoint.write_bytes(weights)
        payload = {
            "model_identity": "sha256-" + digest(weights),
            "checkpoint": "../checkpoints/" + checkpoint.name,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        manifest = root / f"learner/manifests/manifest-{digest(encoded)}.json"
        manifest.write_bytes(encoded)
        return manifest, payload["model_identity"]

    candidate, candidate_id = model("candidate")
    anchor, anchor_id = model("epoch anchor")
    service.pin_job(
        candidate_id, anchor_id, candidate_manifest=candidate, baseline_manifest=anchor
    )
    write_json(
        root / "strength-epoch.json",
        {"anchor_identity": anchor_id, "anchor_manifest": str(anchor)},
    )
    service.register_lease("measurement", "measurement", 0.2)
    event("measurement", "pause_lease_ready", 300)
    service.refresh()
    if terminal:
        event("measurement", "pause_lease_released", 350)
    return root, target, options, candidate


def test_later_stopped_archive_restores_measurement_debt_and_all_pinned_dependencies(
    tmp_path,
):
    from deltreltrain.measurement_scheduling import MeasurementServiceLedger

    root, target, options, _ = measurement_fixture(tmp_path)
    before = options["path"].read_bytes()
    preservation.preserve_stopped_snapshot(root, target)
    assert options["path"].read_bytes() == before
    assert (target / "arena/measurement-service.json").read_bytes() == before
    assert (target / "metrics/coordinator.jsonl").read_bytes() == options[
        "coordinator_events"
    ].read_bytes()
    root.rename(tmp_path / "lost-live-run")
    assert preservation.verify_snapshot(target)["status"] == "verified"
    inventory = json.loads((target / "inventory.json").read_text())
    for logical in inventory["files"]:
        destination = root / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target / logical, destination)
    resumed = MeasurementServiceLedger(**options)
    assert resumed.state["promotion_gpu_ns"] == 100 * 10**9
    assert resumed.state["measurement_gpu_ns"] == 50 * 10**9
    assert resumed.state["debt_ns"] == -20 * 10**9
    assert not resumed.state["leases"]
    resumed.register_lease("after-stopped-restore", "promotion", 0.2)


@pytest.mark.parametrize("missing", ["journal", "manifest"])
def test_preserved_inventory_cannot_omit_measurement_recovery_dependencies(
    tmp_path, missing
):
    root, target, _, candidate = measurement_fixture(tmp_path)
    preservation.preserve_stopped_snapshot(root, target)
    inventory_path = target / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    logical = (
        "metrics/coordinator.jsonl"
        if missing == "journal"
        else str(candidate.relative_to(root))
    )
    del inventory["files"][logical]
    inventory_path.chmod(0o644)
    write_json(inventory_path, inventory)
    marker_path = target / "complete.json"
    marker = json.loads(marker_path.read_text())
    marker["inventory_sha256"] = digest(inventory_path.read_bytes())
    marker["files"] = len(inventory["files"])
    marker_path.chmod(0o644)
    write_json(marker_path, marker)
    with pytest.raises(preservation.PreservationError, match="measurement.*missing"):
        preservation.verify_snapshot(target)


def test_clean_stop_with_unsettled_measurement_is_not_published_as_restorable(tmp_path):
    root, target, options, _ = measurement_fixture(tmp_path, terminal=False)
    before = options["path"].read_bytes()
    with pytest.raises(preservation.PreservationError, match="terminal journal event"):
        preservation.preserve_stopped_snapshot(root, target)
    assert not target.exists()
    assert options["path"].read_bytes() == before


@pytest.mark.parametrize(
    "mutation", ["running", "worker_error", "learner_lag", "live_pid"]
)
def test_rejects_unproven_stop_boundary_before_creating_archive(tmp_path, mutation):
    root, target = fixture(tmp_path)
    coordinator = json.loads((root / "status/coordinator.json").read_text())
    if mutation == "running":
        coordinator["state"] = "running"
    elif mutation == "worker_error":
        coordinator["workers"]["learner"]["last_exit_code"] = -9
    elif mutation == "live_pid":
        coordinator["coordinator_pid"] = os.getpid()
    else:
        heartbeat = json.loads((root / "status/learner.heartbeat.json").read_text())
        heartbeat["step"] += 1
        write_json(root / "status/learner.heartbeat.json", heartbeat)
    write_json(root / "status/coordinator.json", coordinator)
    with pytest.raises(preservation.PreservationError):
        preservation.preserve_stopped_snapshot(root, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.npz",
        "/tmp/outside.npz",
        "shards/../outside.npz",
        "shards//shard-1.npz",
    ],
)
def test_manifest_paths_cannot_escape_or_alias_replay_shards(tmp_path, bad_path):
    root, target = fixture(tmp_path)
    with sqlite3.connect(root / "replay/manifest.sqlite3") as db:
        db.execute("UPDATE shards SET relative_path=? WHERE id=1", (bad_path,))
    with pytest.raises(preservation.PreservationError):
        preservation.preserve_stopped_snapshot(root, target)
    assert not target.exists()
    assert len(list((root / "replay/shards").iterdir())) == 2


@pytest.mark.parametrize("location", ["shard", "ancestor", "checkpoint"])
def test_source_symlinks_are_never_followed(tmp_path, location):
    root, target = fixture(tmp_path)
    if location == "shard":
        source = root / "replay/shards/shard-1.npz"
        outside = tmp_path / "outside.npz"
        source.rename(outside)
        source.symlink_to(outside)
    elif location == "ancestor":
        source = root / "replay/shards"
        outside = tmp_path / "outside"
        source.rename(outside)
        source.symlink_to(outside, target_is_directory=True)
    else:
        source = next((root / "learner/recovery").iterdir())
        outside = tmp_path / "outside.pt"
        source.rename(outside)
        source.symlink_to(outside)
    with pytest.raises(preservation.PreservationError):
        preservation.preserve_stopped_snapshot(root, target)
    assert outside.exists() and not target.exists()


def test_missing_superseded_shard_is_not_silently_dropped(tmp_path):
    root, target = fixture(tmp_path)
    (root / "replay/shards/shard-2.npz").unlink()
    with pytest.raises(FileNotFoundError):
        preservation.preserve_stopped_snapshot(root, target)
    assert not target.exists()


def test_offline_verification_detects_payload_corruption_against_archived_ledger(
    tmp_path,
):
    root, target = fixture(tmp_path)
    preservation.preserve_stopped_snapshot(root, target)
    (target / "replay/shards/shard-2.npz").write_bytes(b"corrupt")
    with pytest.raises(preservation.PreservationError, match="checksum failed"):
        preservation.verify_snapshot(target)


def test_failure_cleans_only_owned_staging_and_never_overwrites_racing_destination(
    tmp_path, monkeypatch
):
    root, target = fixture(tmp_path)
    original = preservation._publish_directory

    def racing(stage, destination):
        destination.mkdir()
        original(stage, destination)

    monkeypatch.setattr(preservation, "_publish_directory", racing)
    with pytest.raises(FileExistsError):
        preservation.preserve_stopped_snapshot(root, target)
    assert target.is_dir() and not list(target.iterdir())
    assert not list(tmp_path.glob(".archive.*"))
    assert len(list((root / "replay/shards").iterdir())) == 2


def test_source_mutation_before_publish_aborts_without_deleting_payloads(
    tmp_path, monkeypatch
):
    root, target = fixture(tmp_path)
    original = preservation._stopped_boundary
    calls = 0

    def changed(directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            with (root / "continuous-migrations.jsonl").open("a") as stream:
                stream.write("changed\n")
        return original(directory)

    monkeypatch.setattr(preservation, "_stopped_boundary", changed)
    with pytest.raises(preservation.PreservationError, match="source controls"):
        preservation.preserve_stopped_snapshot(root, target)
    assert not target.exists() and len(list((root / "replay/shards").iterdir())) == 2


def test_archive_parent_must_share_the_source_filesystem(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path)
    parent = tmp_path / "other-device"
    parent.mkdir()
    original = Path.stat

    def different_device(self, **kwargs):
        actual = original(self, **kwargs)
        if self == parent:
            values = list(actual)
            values[2] += 1
            return os.stat_result(values)
        return actual

    monkeypatch.setattr(Path, "stat", different_device)
    with pytest.raises(preservation.PreservationError, match="same filesystem"):
        preservation.preserve_stopped_snapshot(root, parent / "archive")


def test_sqlite_backup_includes_committed_wal_pages_without_linking_wal(tmp_path):
    root, target = fixture(tmp_path)
    with sqlite3.connect(root / "replay/manifest.sqlite3") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("UPDATE run_counters SET committed_samples=123")
        db.commit()
        assert (root / "replay/manifest.sqlite3-wal").exists()
        preservation.preserve_stopped_snapshot(root, target)
        with sqlite3.connect(target / "replay/manifest.sqlite3") as archived:
            assert archived.execute(
                "SELECT committed_samples FROM run_counters"
            ).fetchone() == (123,)
        assert not (target / "replay/manifest.sqlite3-wal").exists()
    assert preservation.verify_snapshot(target)["status"] == "verified"


def test_preservation_includes_active_allocation_gate_dependencies(
    tmp_path, monkeypatch
):
    from test_allocation_gate_disaster_recovery import prepared

    run, _, gate = prepared(tmp_path, monkeypatch)
    write_json(
        run.root / "status/coordinator.json",
        {
            "state": "stopped",
            "failure": None,
            "coordinator_pid": None,
            "workers": {
                "learner": {"state": "stopped", "last_exit_code": 0, "pid": None}
            },
        },
    )
    write_json(
        run.root / "status/learner.heartbeat.json",
        {
            "phase": "stopped",
            "pid": None,
            "step": 10,
            "examples_consumed": 40,
        },
    )
    target = tmp_path / "preserved"
    preservation.preserve_stopped_snapshot(run.root, target)
    assert (target / gate["baseline_profile"]["path"]).is_file()
    assert preservation.verify_snapshot(target)["status"] == "verified"
