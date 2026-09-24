import copy
import hashlib
import json

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from scripts.prepare_efficiency_scheduling_gate import prepare_scheduling_gate
from deltreltrain import search_allocation_gate as gate
from deltreltrain.efficiency_scheduling import efficiency_scheduling_config
from deltreltrain.measurement_scheduling import MeasurementServiceLedger
from deltreltrain.runtime import append_jsonl, atomic_json, load_run_identity
from test_auxiliary_gate import disaster_fixture
from test_search_allocation_gate import reference
from test_training_disaster_recovery import (
    _content_addressed_file,
    _encoded,
    _fixture,
    _snapshot,
    _snapshot_payload,
)


def scheduling_disaster_fixture(tmp_path, monkeypatch):
    run, source, _, _, refs = disaster_fixture(tmp_path, monkeypatch)
    source_path = run.root / "scheduling-source.yaml"
    source_path.write_bytes(run.profile.read_bytes())
    target = efficiency_scheduling_config(source)
    target_path = run.root / "scheduling-target.yaml"
    target_path.write_text(yaml.safe_dump(target.as_dict()))
    result = prepare_scheduling_gate(source_path, target_path)
    run.profile.write_bytes(target_path.read_bytes())
    (run.root / "profile.sha256").write_text(
        f"{reference(run.root, run.profile)['sha256']}  {run.profile.name}\n"
    )
    refs = [*refs, result["source_profile"], result["source_gate"], result["receipt"]]
    return run, target, result, refs


def test_backup_restores_scheduling_and_every_prior_admission_layer(
    tmp_path, monkeypatch
):
    run, target, _, refs = scheduling_disaster_fixture(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    for ref in refs:
        assert catalog[ref["path"]]["sha256"] == ref["sha256"]
    run.root.rename(tmp_path / "lost-source")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    recovery.restore_snapshot(snapshot, run.root)
    gate.validate_production_ring_allocations(target, _fresh=True)


@pytest.mark.parametrize("missing_index", range(12))
def test_offline_verifier_requires_entire_nested_admission_closure(
    tmp_path, monkeypatch, missing_index
):
    run, _, _, refs = scheduling_disaster_fixture(tmp_path, monkeypatch)
    snapshot = _snapshot(run, tmp_path / "backup")
    original = recovery._snapshot_envelope

    def damaged(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        catalog = dict(catalog)
        del catalog[refs[missing_index]["path"]]
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", damaged)
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="allocation evidence dependency is missing",
    ):
        recovery.verify_snapshot(snapshot)


def test_offline_closure_rejects_nested_scheduling_and_rewritten_inheritance(
    tmp_path, monkeypatch
):
    run, target, result, _ = scheduling_disaster_fixture(tmp_path, monkeypatch)
    envelope = json.loads(gate.allocation_gate_path(target).read_text())

    def reader(path, digest):
        return gate._json(gate._read_ref(run.root, {"path": path, "sha256": digest})[1])

    def chained(path, digest):
        data = reader(path, digest)
        if path == result["source_gate"]["path"]:
            data["efficiency_scheduling_transition"] = result["receipt"]
        return data

    with pytest.raises(recovery.DisasterRecoveryError, match="cannot be chained"):
        recovery._allocation_gate_dependency_closure(envelope, chained)
    changed = copy.deepcopy(envelope)
    changed["groups"] = []
    with pytest.raises(recovery.DisasterRecoveryError, match="changed its source"):
        recovery._allocation_gate_dependency_closure(changed, reader)


def measurement_fixture(tmp_path):
    run = _fixture(tmp_path)
    identity = load_run_identity(run.root / "run.json")
    events = run.root / "metrics/coordinator.jsonl"
    events.parent.mkdir()
    events.write_bytes(b"")
    options = dict(
        path=run.root / "arena/measurement-service.json",
        coordinator_events=events,
        request_path=run.root / "status/arena-gpu-pause.json",
        run_identity=identity,
    )
    ledger = MeasurementServiceLedger(**options)

    def event(token, name, seconds):
        append_jsonl(
            events, {"token": token, "event": name, "timestamp_ns": int(seconds * 1e9)}
        )

    ledger.register_lease("settled", "promotion", 0.2)
    event("settled", "pause_lease_ready", 100)
    event("settled", "pause_lease_released", 300)
    ledger.refresh()
    original = json.loads(run.manifest.read_text())
    checkpoint, digest = _content_addressed_file(
        run.root / "learner/checkpoints",
        prefix="sha256-",
        suffix=".pt",
        data=b"independent measurement candidate",
    )
    candidate = {
        **original,
        "model_identity": "sha256-" + digest,
        "model_version": "sha256-" + digest,
        "checkpoint": "../checkpoints/" + checkpoint.name,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    manifest, _ = _content_addressed_file(
        run.root / "learner/manifests",
        prefix="manifest-",
        suffix=".json",
        data=_encoded(candidate),
    )
    ledger.pin_job(
        candidate["model_identity"],
        original["model_identity"],
        candidate_manifest=manifest,
        baseline_manifest=run.manifest,
    )
    anchor, _ = _content_addressed_file(
        run.root / "learner/manifests",
        prefix="manifest-",
        suffix=".json",
        data=_encoded({**original, "created_ns": original["created_ns"] + 1}),
    )
    atomic_json(
        run.root / "strength-epoch.json",
        {
            "schema_version": 1,
            "anchor_identity": original["model_identity"],
            "anchor_manifest": str(anchor),
        },
    )
    ledger.register_lease("pending", "measurement", 0.2)
    event("pending", "pause_lease_ready", 400)
    ledger.refresh()
    return run, ledger, options, event, manifest, anchor


@pytest.mark.parametrize("release_captured", [False, True])
def test_snapshot_captures_consistent_journal_prefix_and_restore_preserves_debt(
    tmp_path, release_captured
):
    run, ledger, options, event, manifest, anchor = measurement_fixture(tmp_path)
    if release_captured:
        event("pending", "pause_lease_released", 500)
    with options["coordinator_events"].open("ab") as stream:
        stream.write(b'{"unfinished":')
    before = options["path"].read_bytes()
    snapshot = _snapshot(run, tmp_path / "backup")
    catalog = _snapshot_payload(snapshot)["catalog"]
    assert (
        catalog["arena/measurement-service.json"]["sha256"]
        == hashlib.sha256(before).hexdigest()
    )
    assert catalog["metrics/coordinator.jsonl"]["kind"] == "coordinator-journal"
    assert str(manifest.relative_to(run.root)) in catalog
    assert str(anchor.relative_to(run.root)) in catalog
    restored = tmp_path / "restored"
    recovery.restore_snapshot(snapshot, restored, relocate_profile=True)
    captured = (restored / "metrics/coordinator.jsonl").read_bytes()
    assert captured.endswith(b"\n") and b"unfinished" not in captured
    receipt = json.loads(
        (restored / "arena/measurement-service-restore.json").read_text()
    )
    assert (
        receipt["ledger_sha256"]
        == hashlib.sha256(
            (restored / "arena/measurement-service.json").read_bytes()
        ).hexdigest()
    )
    rebuilt = MeasurementServiceLedger(
        path=restored / "arena/measurement-service.json",
        coordinator_events=restored / "metrics/coordinator.jsonl",
        request_path=restored / "status/arena-gpu-pause.json",
        run_identity=load_run_identity(restored / "run.json"),
    )
    assert rebuilt.state["promotion_gpu_ns"] == 200 * 10**9
    assert rebuilt.state["measurement_gpu_ns"] == (
        100 * 10**9 if release_captured else 0
    )
    assert rebuilt.state["debt_ns"] == (-40 * 10**9 if release_captured else 40 * 10**9)
    assert rebuilt.state["leases"] == {}
    assert rebuilt.state["accounting_complete"] is release_captured
    assert rebuilt.pinned_job == ledger.pinned_job
    assert rebuilt.state["candidate_manifest"].startswith(str(restored))


@pytest.mark.parametrize("damage", ["journal", "candidate", "anchor"])
def test_offline_verifier_requires_journal_and_pinned_model_dependencies(
    tmp_path, monkeypatch, damage
):
    run, _, _, _, manifest, anchor = measurement_fixture(tmp_path)
    snapshot = _snapshot(run, tmp_path / "backup")
    logical = (
        "metrics/coordinator.jsonl"
        if damage == "journal"
        else str((manifest if damage == "candidate" else anchor).relative_to(run.root))
    )
    original = recovery._snapshot_envelope

    def damaged(*args, **kwargs):
        payload, catalog, data, digest = original(*args, **kwargs)
        catalog = dict(catalog)
        del catalog[logical]
        return payload, catalog, data, digest

    monkeypatch.setattr(recovery, "_snapshot_envelope", damaged)
    with pytest.raises(
        recovery.DisasterRecoveryError, match="journal|manifest|catalog"
    ):
        recovery.verify_snapshot(snapshot)


def test_capture_rejects_journal_prefix_replacement_without_changing_live_ledger(
    tmp_path,
):
    run, _, options, _, _, _ = measurement_fixture(tmp_path)
    before = options["path"].read_bytes()
    data = options["coordinator_events"].read_bytes()
    options["coordinator_events"].write_bytes(data.replace(b"settled", b"changed"))
    with pytest.raises(recovery.DisasterRecoveryError, match="journal prefix"):
        _snapshot(run, tmp_path / "backup")
    assert options["path"].read_bytes() == before
