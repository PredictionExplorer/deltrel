"""Deployment must prove durable state and fresh workers before reporting success."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import graceful_training_deploy as deploy

WORKERS = ["learner", "actor-gpu-1"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def deployment(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    source_release = tmp_path / "source-release"
    target_release = tmp_path / "target-release"
    for release in (source_release, target_release):
        (release / "training").mkdir(parents=True)
    (target_release / "SOURCE_COMMIT").write_text("b" * 40)
    source = root / "source.yaml"
    source.write_text("{}\n")
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("{}\n")
    smoke = tmp_path / "smoke.py"
    smoke.write_text("raise AssertionError('tests never execute smoke')\n")
    plan = {
        "run_root": str(root),
        "evidence_root": str(tmp_path / "evidence"),
        "target_release": str(target_release),
        "source_release": str(source_release),
        "source_profile": str(source),
        "candidate_profile": str(candidate),
        "target_profile_name": "target.yaml",
        "main_unit": "training.service",
        "source_commit": "a" * 40,
        "target_commit": "b" * 40,
        "source_profile_sha256": deploy.digest(source),
        "candidate_profile_sha256": deploy.digest(candidate),
        "smoke_script": str(smoke),
        "smoke_script_sha256": deploy.digest(smoke),
        "preserved_snapshot": str(tmp_path / "preserved"),
        "expected_workers": WORKERS,
    }
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    (root / "profile.sha256").write_text(f"{deploy.digest(source)}  {source}\n")
    (root / "source-commit.txt").write_text("a" * 40)
    result = deploy.Deployment(plan_path)
    result.test_plan_path = plan_path
    return result


def stopped_files(
    deployment, *, learner_step=5, checkpoint_step=5, missing=False, bad_exit=False
):
    root = deployment.root
    payload = b"exact immutable checkpoint bytes"
    checkpoint_file = root / "learner/recovery/checkpoint.pt"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(payload)
    checkpoint = {
        "step": checkpoint_step,
        "examples_consumed": checkpoint_step * 512,
        "checkpoint": "recovery/checkpoint.pt",
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
    }
    workers = {
        name: {"pid": None, "state": "stopped", "last_exit_code": 0} for name in WORKERS
    }
    if missing:
        workers.pop("actor-gpu-1")
    if bad_exit:
        workers["actor-gpu-1"]["last_exit_code"] = -9
    write(
        root / "status/coordinator.json",
        {"state": "stopped", "failure": None, "workers": workers},
    )
    write(
        root / "status/learner.heartbeat.json",
        {"step": learner_step, "examples_consumed": learner_step * 512},
    )
    write(root / "learner/recovery.json", checkpoint)
    return checkpoint


def mock_stop(deployment, monkeypatch):
    commands = []
    monkeypatch.setattr(
        deployment, "run", lambda command, name, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        deployment,
        "show",
        lambda unit, field: "0" if field == "MainPID" else "inactive",
    )
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *args, **kwargs: "")
    return commands


def test_invalid_candidate_cannot_stop_or_install_any_service(deployment, monkeypatch):
    monkeypatch.setattr(
        deployment,
        "show",
        lambda unit, field: {
            "ActiveState": "active",
            "WorkingDirectory": str(deployment.old_release / "training"),
        }.get(field, ""),
    )
    calls = []
    monkeypatch.setattr(
        deployment, "run", lambda command, name, **kwargs: calls.append((name, command))
    )

    def script(name, args, label, *rest, **kwargs):
        calls.append((label, name))
        raise RuntimeError("candidate gate rejected")

    monkeypatch.setattr(deployment, "script", script)
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kwargs: pytest.fail("failed preparation stopped training"),
    )
    with pytest.raises(RuntimeError, match="candidate gate rejected"):
        deployment.execute()
    assert not (deployment.base / "prepared.json").exists()
    assert not (deployment.base / "complete.json").exists()
    assert all(
        not isinstance(command, list) or "systemctl" not in command
        for _, command in calls
    )


@pytest.mark.parametrize(
    "change,error",
    [("target", "unsafe target profile"), ("candidate", "candidate profile changed")],
)
def test_plan_path_and_profile_authority_are_checked_before_execution(
    deployment, change, error
):
    plan = json.loads(deployment.test_plan_path.read_text())
    if change == "target":
        plan["target_profile_name"] = "../escape.yaml"
        write(deployment.test_plan_path, plan)
    else:
        deployment.candidate.write_text("changed bytes")
    with pytest.raises(RuntimeError, match=error):
        deploy.Deployment(deployment.test_plan_path)
    assert not (deployment.base / "stop-requested.json").exists()


@pytest.mark.parametrize(
    "options,error",
    [
        ({"learner_step": 6}, "final learner progress was not checkpointed"),
        ({"bad_exit": True}, "worker.*exit cleanly"),
        ({"missing": True}, "worker|inventory"),
    ],
)
def test_forward_stop_never_reports_an_unclean_or_incomplete_boundary(
    deployment, monkeypatch, options, error
):
    stopped_files(deployment, **options)
    mock_stop(deployment, monkeypatch)
    with pytest.raises(RuntimeError, match=error):
        deployment.stop(strict=True)
    assert not (deployment.base / "stopped.json").exists()
    assert not (deployment.base / "complete.json").exists()


def test_clean_stop_pins_exact_latest_checkpoint_bytes(deployment, monkeypatch):
    checkpoint = stopped_files(deployment)
    commands = mock_stop(deployment, monkeypatch)
    boundary = deployment.stop(strict=True)
    assert boundary["checkpoint"] == checkpoint
    assert commands == [["sudo", "systemctl", "stop", deployment.main]]
    assert (
        json.loads((deployment.base / "stopped.json").read_text())["checkpoint"]
        == checkpoint
    )
    file = deployment.root / "learner" / checkpoint["checkpoint"]
    file.write_bytes(b"x" * checkpoint["checkpoint_bytes"])
    with pytest.raises(RuntimeError, match="checkpoint verification failed"):
        deployment.stop(strict=True)


def readiness_fixture(deployment, monkeypatch, defect=None):
    class Clock:
        now = 1.0

        def monotonic(self):
            return self.now

        def time_ns(self):
            return 10**12 + int(self.now * 10**9)

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(deploy.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(deploy.time, "time_ns", clock.time_ns)
    monkeypatch.setattr(deploy.time, "sleep", clock.sleep)
    checkpoint = {
        "checkpoint": "recovery/checkpoint.pt",
        "step": 5,
        "examples_consumed": 2560,
    }
    launched = 10**12
    expected = "recovery:" + str(
        (deployment.root / "learner" / checkpoint["checkpoint"]).resolve()
    )
    paths = {
        name: deployment.root / "status" / f"{name}.heartbeat.json" for name in WORKERS
    }
    for path in paths.values():
        write(path, {})
    monkeypatch.setattr(
        deployment,
        "show",
        lambda unit, field: "777" if field == "MainPID" else "active",
    )
    observations = []
    monkeypatch.setattr(
        deployment, "save", lambda name, payload: observations.append(deepcopy(payload))
    )

    def read(path):
        now = clock.time_ns()
        heartbeat = {
            "pid": 111,
            "heartbeat_ns": now,
            "step": 6,
            "resumed_from": expected,
        }
        if defect == "wrong_resume":
            heartbeat["resumed_from"] = "recovery:other-checkpoint"
        if defect == "stale_learner":
            heartbeat["heartbeat_ns"] = launched - 1
        if defect == "future_learner":
            heartbeat["heartbeat_ns"] = now + 10**12
        if Path(path) == paths["learner"]:
            return heartbeat
        if Path(path) == paths["actor-gpu-1"]:
            return {
                "pid": 333 if defect == "wrong_pid" else 222,
                "heartbeat_ns": now,
                "inference": {
                    "completed_requests": 3,
                    "failed_requests": int(defect == "inference_failure"),
                },
            }
        if Path(path) == deployment.root / "status/coordinator.json":
            workers = {
                "learner": {
                    "pid": 111,
                    "heartbeat": str(paths["learner"]),
                    "state": "running",
                    "restart_count": 0,
                },
                "actor-gpu-1": {
                    "pid": 222,
                    "heartbeat": str(paths["actor-gpu-1"]),
                    "state": "running",
                    "restart_count": int(defect == "restarted"),
                },
            }
            if defect == "missing_worker":
                workers.pop("actor-gpu-1")
            return {
                "timestamp_ns": launched - 1 if defect == "stale_coordinator" else now,
                "state": "running",
                "coordinator_pid": 777,
                "workers": workers,
            }
        raise AssertionError(f"unplanned read: {path}")

    monkeypatch.setattr(deploy, "read", read)
    return checkpoint, launched, clock, observations


def test_readiness_requires_sustained_fresh_workers_and_exact_resume(
    deployment, monkeypatch
):
    checkpoint, launched, clock, observations = readiness_fixture(
        deployment, monkeypatch
    )
    result = deployment.wait_ready(checkpoint, launched)
    assert result["ready"] is True
    assert clock.now >= 61
    assert len(observations) >= 7


@pytest.mark.parametrize(
    "defect,error",
    [
        ("wrong_resume", "different checkpoint"),
        ("wrong_pid", "sustained readiness"),
        ("stale_learner", "sustained readiness"),
        ("future_learner", "sustained readiness"),
        ("stale_coordinator", "sustained readiness"),
        ("missing_worker", "sustained readiness"),
        ("restarted", "worker failed or restarted"),
        ("inference_failure", "inference request failed"),
    ],
)
def test_stale_incomplete_or_failed_startup_never_qualifies(
    deployment, monkeypatch, defect, error
):
    checkpoint, launched, _, _ = readiness_fixture(deployment, monkeypatch, defect)
    with pytest.raises(RuntimeError, match=error):
        deployment.wait_ready(checkpoint, launched)


@pytest.mark.parametrize("marker", [None, "complete.json", "recovered.json"])
def test_exec_stop_post_does_not_touch_unprepared_or_finished_training(
    deployment, monkeypatch, marker
):
    if marker is not None:
        write(deployment.base / "prepared.json", {"plan": deployment.plan})
        write(deployment.base / marker, {"status": "complete"})
    monkeypatch.setattr(
        deployment, "stop", lambda **kwargs: pytest.fail("unnecessary training stop")
    )
    monkeypatch.setattr(
        deployment,
        "restore_support",
        lambda: pytest.fail("unexpected support mutation"),
    )
    deployment.recover()


def test_interruption_before_main_stop_restores_support_without_restarting_training(
    deployment, monkeypatch
):
    write(deployment.base / "prepared.json", {"plan": deployment.plan})
    restored = []
    monkeypatch.setattr(deployment, "restore_support", lambda: restored.append(True))
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kwargs: pytest.fail("main training was never stopped"),
    )
    deployment.recover()
    assert restored == [True]


def journal_fixture(deployment, *, trailing_newline=True):
    from types import SimpleNamespace
    from scripts import deployment_metadata as metadata

    checkpoint = stopped_files(deployment)
    root = deployment.root
    log = root / "continuous-migrations.jsonl"
    log.write_bytes(b'{"old":true}' + (b"\n" if trailing_newline else b""))
    write(root / "learner/utd-segment.json", {"scope": "old"})
    write(root / "strength-epoch.json", {"epoch": "old"})
    plan = SimpleNamespace(
        run_root=root,
        target_profile=deployment.target,
        target_profile_checksum=deployment.target.with_suffix(".sha256"),
        target_profile_bytes=b"new: profile\n",
        profile_sha256_bytes=b"new-profile-checksum\n",
        source_commit_bytes=b"b" * 40 + b"\n",
        migration_record={
            "recovery_pointer_sha256": deploy.digest(root / "learner/recovery.json"),
            "new": True,
        },
        utd_segment_payload={"scope": "new"},
        utd_segment_path=root / "learner/utd-segment.json",
        strength_epoch_payload={"epoch": "new"},
        strength_epoch_path=root / "strength-epoch.json",
        heartbeat_step=checkpoint["step"],
        learner_step=checkpoint["step"],
        output=lambda **kwargs: {"migration": "planned"},
    )
    intent = deployment.base / "metadata-intent-test.json"
    metadata.record_intent(plan, intent)
    return metadata, plan, intent, read_journal(intent)


def read_journal(path):
    import base64

    value = json.loads(path.read_text())
    return [
        (
            row,
            base64.b64decode(row["before"]) if row["before"] is not None else None,
            base64.b64decode(row["after"]) if row["after"] is not None else None,
        )
        for row in value["writes"]
    ]


def set_after(root, rows, predicate=lambda row: True):
    for row, _, after in rows:
        if predicate(row):
            path = root / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(after)


def test_partial_metadata_transaction_repairs_only_owned_files(deployment):
    metadata, _, intent, rows = journal_fixture(deployment)
    checkpoint = deployment.root / "learner/recovery/checkpoint.pt"
    checkpoint_before = checkpoint.read_bytes()
    set_after(
        deployment.root,
        rows,
        lambda row: row["path"] in ("target.yaml", "learner/utd-segment.json"),
    )
    assert metadata.repair_interrupted_intent(intent, deployment.root) == "repaired"
    for row, before, _ in rows:
        path = deployment.root / row["path"]
        assert path.read_bytes() == before if before is not None else not path.exists()
    assert checkpoint.read_bytes() == checkpoint_before
    assert metadata.repair_interrupted_intent(intent, deployment.root) == "repaired"


@pytest.mark.parametrize("trailing_newline", [False, True])
def test_owned_partial_log_append_is_repairable_with_either_previous_ending(
    deployment, trailing_newline
):
    metadata, plan, intent, rows = journal_fixture(
        deployment, trailing_newline=trailing_newline
    )
    from scripts.migrate_continuous_profile import _append_jsonl

    log = deployment.root / "continuous-migrations.jsonl"
    _append_jsonl(log, plan.migration_record)
    expected = next(
        after for row, _, after in rows if row["path"] == "continuous-migrations.jsonl"
    )
    assert log.read_bytes() == expected
    log.write_bytes(expected[:-5])
    assert metadata.repair_interrupted_intent(intent, deployment.root) == "repaired"
    before = next(
        before
        for row, before, _ in rows
        if row["path"] == "continuous-migrations.jsonl"
    )
    assert log.read_bytes() == before


def test_unrecognized_metadata_bytes_cannot_be_overwritten_by_recovery(deployment):
    metadata, _, intent, rows = journal_fixture(deployment)
    set_after(
        deployment.root, rows, lambda row: row["path"] == "learner/utd-segment.json"
    )
    changed = deployment.root / "source-commit.txt"
    changed.write_bytes(b"unrelated newer authority")
    actual = {
        row["path"]: (deployment.root / row["path"]).read_bytes()
        for row, _, _ in rows
        if (deployment.root / row["path"]).exists()
    }
    with pytest.raises(ValueError, match="authorized transaction states"):
        metadata.repair_interrupted_intent(intent, deployment.root)
    assert all(
        (deployment.root / path).read_bytes() == value for path, value in actual.items()
    )


def test_partial_migration_never_reverts_after_learned_state_advances(deployment):
    metadata, _, intent, rows = journal_fixture(deployment)
    set_after(
        deployment.root, rows, lambda row: row["path"] == "learner/utd-segment.json"
    )
    pointer = deployment.root / "learner/recovery.json"
    value = json.loads(pointer.read_text())
    value["step"] += 1
    write(pointer, value)
    with pytest.raises(ValueError, match="learned state advanced"):
        metadata.repair_interrupted_intent(intent, deployment.root)
    assert json.loads((deployment.root / "learner/utd-segment.json").read_text()) == {
        "scope": "new"
    }
    assert json.loads(pointer.read_text())["step"] == value["step"]


def test_fully_applied_intent_is_committed_without_restoring_newer_state(deployment):
    metadata, _, intent, rows = journal_fixture(deployment)
    set_after(deployment.root, rows)
    pointer = deployment.root / "learner/recovery.json"
    write(pointer, {"step": 100, "checkpoint": "newest"})
    assert metadata.repair_interrupted_intent(intent, deployment.root) == "committed"
    assert json.loads(pointer.read_text()) == {"step": 100, "checkpoint": "newest"}
    assert deployment.target.read_bytes() == b"new: profile\n"


@pytest.mark.parametrize(
    "corruption", ["checkpoint_path", "escape", "duplicates", "empty"]
)
def test_journal_cannot_authorize_checkpoint_writes_or_ambiguous_paths(
    deployment, corruption
):
    import base64

    metadata, _, intent, rows = journal_fixture(deployment)
    value = json.loads(intent.read_text())
    checkpoint = deployment.root / "learner/recovery/checkpoint.pt"
    before = checkpoint.read_bytes()
    if corruption == "checkpoint_path":
        value["writes"].append(
            {
                "path": "learner/recovery/checkpoint.pt",
                "before": base64.b64encode(b"old dangerous bytes").decode(),
                "after": base64.b64encode(before).decode(),
                "before_mode": 0o644,
                "after_mode": 0o644,
            }
        )
    elif corruption == "escape":
        value["writes"][0]["path"] = "../escape.yaml"
    elif corruption == "duplicates":
        value["writes"].append(deepcopy(value["writes"][0]))
    else:
        value["writes"] = []
    write(intent, value)
    with pytest.raises(ValueError):
        metadata.repair_interrupted_intent(intent, deployment.root)
    assert checkpoint.read_bytes() == before
    assert not deployment.target.exists()


def test_signal_interrupt_is_catchable_and_migration_intent_precedes_writes(
    deployment, monkeypatch
):
    import signal
    from scripts import deployment_metadata as metadata

    _, plan, initial, _ = journal_fixture(deployment)
    initial.unlink()
    monkeypatch.setattr(deploy, "plan_migration", lambda request: plan)
    expected = deployment.base / "metadata-intent-target.yaml.json"

    def interrupted_apply(_plan):
        assert expected.is_file()
        assert json.loads(expected.read_text())["status"] == "pending"
        deploy.interrupt(signal.SIGTERM, None)

    monkeypatch.setattr(deploy, "apply_migration", interrupted_apply)
    assert issubclass(deploy.DeploymentInterrupted, Exception)
    with pytest.raises(deploy.DeploymentInterrupted, match="signal"):
        deployment.migrate(
            deployment.source, deployment.candidate, "target.yaml", "a" * 40
        )
    assert metadata.repair_interrupted_intent(expected, deployment.root) == "repaired"


def test_recovery_records_progress_lost_before_an_unclean_stop(deployment, monkeypatch):
    stopped_files(deployment, learner_step=7, checkpoint_step=5)
    mock_stop(deployment, monkeypatch)
    boundary = deployment.stop(strict=False)
    assert boundary["discarded_uncheckpointed_steps"] == 2
    assert boundary["discarded_uncheckpointed_examples"] == 1024


def test_failed_smoke_does_not_migrate_install_or_publish_success(
    deployment, monkeypatch
):
    from scripts import preserve_replay_snapshot

    checkpoint = stopped_files(deployment)
    events = []
    monkeypatch.setattr(deployment, "prepare", lambda: events.append("prepared"))
    monkeypatch.setattr(deployment, "drain_support", lambda: events.append("drained"))
    monkeypatch.setattr(deployment, "validate_prepared", lambda: None)
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kwargs: {"checkpoint": checkpoint, "progress_available": True},
    )
    monkeypatch.setattr(
        preserve_replay_snapshot,
        "preserve_stopped_snapshot",
        lambda *args: {"preserved": True},
    )
    monkeypatch.setattr(
        deployment,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("smoke failed")),
    )
    monkeypatch.setattr(
        deployment, "migrate", lambda *args: pytest.fail("migrated after failed smoke")
    )
    monkeypatch.setattr(
        deployment,
        "install_units",
        lambda *args: pytest.fail("installed after failed smoke"),
    )
    with pytest.raises(RuntimeError, match="smoke failed"):
        deployment.execute()
    assert events == ["prepared", "drained"]
    assert not (deployment.base / "complete.json").exists()


def test_recovery_resumes_original_authority_after_a_pre_migration_failure(
    deployment, monkeypatch
):
    checkpoint = stopped_files(deployment)
    write(deployment.base / "prepared.json", {"plan": deployment.plan})
    write(deployment.base / "stop-requested.json", {})
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kwargs: {"checkpoint": checkpoint, "progress_available": True},
    )
    monkeypatch.setattr(
        deployment, "migrate", lambda *args: pytest.fail("unnecessary migration")
    )
    calls = []
    monkeypatch.setattr(
        deployment,
        "install_units",
        lambda profile, **kwargs: calls.append(("install", profile, kwargs)),
    )
    monkeypatch.setattr(
        deployment,
        "start",
        lambda profile, resume: (
            calls.append(("start", profile, resume)) or {"ready": True}
        ),
    )
    monkeypatch.setattr(
        deployment, "restore_support", lambda: calls.append(("support",))
    )
    deployment.recover()
    assert calls == [
        ("install", deployment.source, {"original": True}),
        ("start", deployment.source, checkpoint),
        ("support",),
    ]
    assert (
        json.loads((deployment.base / "recovered.json").read_text())["status"]
        == "recovered"
    )
    calls.clear()
    deployment.recover()
    assert calls == []


def test_recovery_after_completed_inverse_migration_does_not_migrate_twice(
    deployment, monkeypatch
):
    checkpoint = stopped_files(deployment)
    rollback = deployment.root / "compatible-rollback.yaml"
    rollback.write_bytes(deployment.source.read_bytes())
    (deployment.root / "profile.sha256").write_text(
        f"{deploy.digest(rollback)}  {rollback}\n"
    )
    (deployment.root / "source-commit.txt").write_text(deployment.plan["target_commit"])
    write(deployment.base / "prepared.json", {"plan": deployment.plan})
    write(deployment.base / "stop-requested.json", {})
    write(deployment.base / "recovery-intent.json", {"profile_name": rollback.name})
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kwargs: {"checkpoint": checkpoint, "progress_available": True},
    )
    monkeypatch.setattr(
        deployment,
        "migrate",
        lambda *args: pytest.fail("already applied inverse migration repeated"),
    )
    calls = []
    monkeypatch.setattr(
        deployment,
        "install_units",
        lambda profile, **kwargs: calls.append(("install", profile)),
    )
    monkeypatch.setattr(
        deployment,
        "start",
        lambda profile, resume: (
            calls.append(("start", profile, resume)) or {"ready": True}
        ),
    )
    monkeypatch.setattr(deployment, "restore_support", lambda: None)
    deployment.recover()
    assert calls == [("install", rollback), ("start", rollback, checkpoint)]
    assert json.loads((deployment.base / "recovered.json").read_text())[
        "profile"
    ] == str(rollback)


def test_recovery_rejects_a_changed_plan_before_any_service_mutation(
    deployment, monkeypatch
):
    write(deployment.base / "prepared.json", {"plan": deepcopy(deployment.plan)})
    write(deployment.base / "stop-requested.json", {})
    changed_plan = deepcopy(deployment.plan)
    changed_plan["main_unit"] = "different-workload.service"
    write(deployment.test_plan_path, changed_plan)
    changed = deploy.Deployment(deployment.test_plan_path)
    monkeypatch.setattr(
        changed,
        "stop",
        lambda **kwargs: pytest.fail("changed plan stopped another workload"),
    )
    monkeypatch.setattr(
        changed, "restore_support", lambda: pytest.fail("changed plan altered support")
    )
    with pytest.raises(RuntimeError, match="plan"):
        changed.recover()


@pytest.mark.parametrize("target_was_activated", [False, True])
def test_early_startup_failure_recovers_checkpoint_without_inventing_progress(
    deployment, monkeypatch, target_was_activated
):
    checkpoint = stopped_files(deployment)
    checkpoint_file = deployment.root / "learner" / checkpoint["checkpoint"]
    immutable_before = checkpoint_file.read_bytes()
    raw = {"phase": "initializing", "pid": 999_999_999, "heartbeat_ns": 1234}
    write(deployment.root / "status/learner.heartbeat.json", raw)
    write(deployment.base / "prepared.json", {"plan": deployment.plan})
    write(deployment.base / "stop-requested.json", {})
    if target_was_activated:
        deployment.target.write_bytes(deployment.candidate.read_bytes())
        (deployment.root / "profile.sha256").write_text(
            f"{deploy.digest(deployment.target)}  {deployment.target}\n"
        )
        (deployment.root / "source-commit.txt").write_text(
            deployment.plan["target_commit"]
        )
    mock_stop(deployment, monkeypatch)
    calls = []

    def migrate(source, candidate, name, commit, *, recovery=False):
        assert target_was_activated and recovery is True
        assert source == deployment.target and candidate == deployment.source
        assert commit == deployment.plan["target_commit"]
        derived = json.loads(
            (deployment.root / "status/learner.heartbeat.json").read_text()
        )
        assert derived["step"] == checkpoint["step"]
        assert derived["observed_progress_available"] is False
        destination = deployment.root / name
        destination.write_bytes(candidate.read_bytes())
        calls.append(("migrate", destination))
        return {"status": "applied"}

    monkeypatch.setattr(deployment, "migrate", migrate)
    monkeypatch.setattr(deployment, "install_units", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        deployment,
        "start",
        lambda profile, resume: (
            calls.append(("start", profile, resume)) or {"ready": True}
        ),
    )
    monkeypatch.setattr(deployment, "restore_support", lambda: None)
    deployment.recover()
    report = json.loads((deployment.base / "recovered.json").read_text())
    assert report["status"] == "recovered"
    assert report["stopped_boundary"]["progress_available"] is False
    assert report["stopped_boundary"]["discarded_uncheckpointed_steps"] is None
    assert report["stopped_boundary"]["discarded_uncheckpointed_examples"] is None
    assert (
        json.loads((deployment.base / "early-startup-heartbeat.json").read_text())[
            "learner"
        ]
        == raw
    )
    derived = json.loads(
        (deployment.root / "status/learner.heartbeat.json").read_text()
    )
    assert derived["observed_progress_available"] is False
    assert derived["derived_from_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert calls[-1][0] == "start" and calls[-1][2] == checkpoint
    assert checkpoint_file.read_bytes() == immutable_before
