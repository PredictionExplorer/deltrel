"""Controlled activation evidence survives loss of the original run tree."""

from dataclasses import replace
import hashlib
import json

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from scripts.validate_continuous_profile import validate_continuous_config
from deltreltrain import search_allocation_gate as gate
from deltreltrain.config import load_config
from test_graph_cache_controlled_activation import controlled_admission_fixture
from test_search_allocation_gate import reference, write_json
from test_training_disaster_recovery import _fixture, _snapshot, _snapshot_payload


@pytest.mark.parametrize(
    "reference",
    [
        None,
        {},
        {"path": "../outside.json", "sha256": "a" * 64},
        {"path": "evidence.json", "sha256": "bad"},
    ],
)
def test_controlled_receipt_rejects_invalid_live_dependency(reference):
    with pytest.raises(recovery.DisasterRecoveryError):
        recovery._controlled_graph_activation_references(
            {
                "format": "deltreltrain.graph-cache-controlled-activation",
                "schema_version": 1,
                "live_workload_evidence": reference,
            }
        )


def test_controlled_receipt_enumerates_only_its_pinned_live_dependency():
    payload = {
        "format": "deltreltrain.graph-cache-controlled-activation",
        "schema_version": 1,
        "live_workload_evidence": {"path": "evidence/live.json", "sha256": "a" * 64},
    }
    assert recovery._controlled_graph_activation_references(payload) == [
        ("evidence/live.json", "a" * 64, "status-json")
    ]
    payload["schema_version"] = True
    with pytest.raises(recovery.DisasterRecoveryError, match="controlled activation"):
        recovery._controlled_graph_activation_references(payload)


def prepared(tmp_path, monkeypatch):
    run = _fixture(tmp_path)
    _, target, root, envelope = controlled_admission_fixture(
        tmp_path,
        monkeypatch,
        root=run.root,
        run_id="run-disaster-test",
        family="family-disaster-test",
    )
    # Neither file may piggyback on generic status-tree backup enumeration.
    directory = root / "activation-evidence"
    directory.mkdir()
    receipt_path = root / envelope["graph_cache_controlled_activation"]["path"]
    receipt = json.loads(receipt_path.read_text())
    old_live = root / receipt["live_workload_evidence"]["path"]
    live_path = directory / old_live.name
    old_live.rename(live_path)
    receipt["live_workload_evidence"] = reference(root, live_path)
    moved_receipt = directory / receipt_path.name
    receipt_path.unlink()
    write_json(moved_receipt, receipt)
    envelope["graph_cache_controlled_activation"] = reference(root, moved_receipt)
    write_json(gate.allocation_gate_path(target), envelope)
    run.profile.write_text(yaml.safe_dump(target.as_dict()))
    digest = hashlib.sha256(run.profile.read_bytes()).hexdigest()
    (root / "profile.sha256").write_text(f"{digest}  {run.profile.name}\n")
    validate_continuous_config(load_config(run.profile))
    return (
        run,
        target,
        [
            envelope["graph_cache_controlled_activation"],
            receipt["live_workload_evidence"],
        ],
    )


def test_controlled_activation_dependencies_survive_original_root_loss(
    tmp_path, monkeypatch
):
    run, target, references = prepared(tmp_path, monkeypatch)
    original = {
        ref["path"]: (run.root / ref["path"]).read_bytes() for ref in references
    }
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for ref in references:
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
        assert catalog[ref["path"]]["kind"] == "status-json"
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    run.root.rename(tmp_path / "lost-source")
    recovery.restore_snapshot(snapshot, run.root)
    for logical, content in original.items():
        assert (run.root / logical).read_bytes() == content
    restored = load_config(run.root / run.profile.name)
    assert restored == target
    validate_continuous_config(restored)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_backup_rechecks_receipt_and_live_evidence_after_cached_admission(
    tmp_path, monkeypatch, index, damage
):
    run, _, references = prepared(tmp_path, monkeypatch)
    path = run.root / references[index]["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(recovery.DisasterRecoveryError, match="allocation evidence"):
        _snapshot(run, tmp_path / "backup")
    assert not (tmp_path / "backup/latest.json").exists()


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "mismatched_hash"])
def test_verify_and_restore_require_receipt_and_live_catalog_dependencies(
    tmp_path, monkeypatch, index, damage
):
    run, _, references = prepared(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    logical = references[index]["path"]
    original = recovery._snapshot_envelope

    def damaged_catalog(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        if damage == "missing":
            del catalog[logical]
        else:
            other = catalog[references[1 - index]["path"]]
            catalog[logical] = replace(
                catalog[logical], sha256=other.sha256, bytes=other.bytes
            )
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", damaged_catalog)
    with pytest.raises(recovery.DisasterRecoveryError, match="missing or mismatched"):
        recovery.verify_snapshot(snapshot)
    destination = tmp_path / "must-not-restore"
    with pytest.raises(recovery.DisasterRecoveryError):
        recovery.restore_snapshot(snapshot, destination)
    assert not destination.exists()


@pytest.mark.parametrize("index", [0, 1])
def test_corrupt_controlled_activation_object_prevents_restore(
    tmp_path, monkeypatch, index
):
    run, _, references = prepared(tmp_path, monkeypatch)
    backup = tmp_path / "backup"
    snapshot = _snapshot(run, backup)
    stored = recovery._object_path(backup, references[index]["sha256"])
    stored.chmod(0o644)
    stored.write_bytes(b"corrupted controlled activation evidence")
    with pytest.raises(recovery.DisasterRecoveryError, match="byte length|SHA-256"):
        recovery.verify_snapshot(snapshot)
    destination = tmp_path / "must-not-restore"
    with pytest.raises(recovery.DisasterRecoveryError):
        recovery.restore_snapshot(snapshot, destination)
    assert not destination.exists()
