from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from test_strength_recovery import source as source
from test_strength_recovery_continuation import screen as screen
from deltreltrain import strength_freshness as f
from deltreltrain.checkpoint import sha256_file
from deltreltrain.contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH_WIRE,
    SEARCH_ALGORITHM_ID,
)
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import completed_screen
from scripts import activate_strength_freshness as op
from scripts import run_strength_recovery_continuation as continuation
from scripts import migrate_continuous_profile as migration
from scripts.active_profile import resolve_active_profile, validate_profile_for_monitor


@pytest.fixture
def registered(screen, tmp_path, monkeypatch):
    old = continuation.prepare(screen, source_commit="a" * 40)
    profile = continuation.handoff(screen, old)
    started = completed_screen(screen)["resource_released_ns"]
    atomic_json(
        screen / continuation.STATE,
        {
            "plan_sha256": old["plan_sha256"],
            "continuation_started_ns": started,
            "profile": str(profile),
            "phase": "stopped",
            "attempts": [{"status": "old-stopped", "pid": 99999999}],
        },
    )
    runtime = tmp_path / "qualified-r4/training"
    binpath = runtime / ".venv/bin"
    binpath.mkdir(parents=True)
    binary = tmp_path / "qualified-host-python"
    binary.write_bytes(b"qualified python fixture")
    binary.chmod(0o755)
    (binpath / "python").symlink_to(binary)
    (binpath / "deltreltrain-orchestrate").write_bytes(
        b"qualified orchestrator fixture"
    )
    (binpath / "deltreltrain-orchestrate").chmod(0o755)
    (runtime / ".venv/pyvenv.cfg").write_text("home = /qualified\n")
    module = runtime / "deltreltrain/__init__.py"
    module.parent.mkdir()
    module.write_text("# frozen qualified package\n")
    native = runtime / ".venv/lib/native.abi3.so"
    native.parent.mkdir()
    native.write_bytes(b"qualified native fixture")
    wrapper = native.with_name("__init__.py")
    wrapper.write_text("# qualified native wrapper\n")
    sums = runtime.parent / "SOURCE_SHA256SUMS"
    sums.write_text(f"{sha256_file(module)}  training/deltreltrain/__init__.py\n")
    (runtime.parent / "SOURCE_COMMIT").write_text("b" * 40 + "\n")
    qualification = tmp_path / "cpu-qualification.json"
    atomic_json(
        qualification,
        {
            "status": "qualified-cpu-native-only",
            "source_commit": "b" * 40,
            "release": str(runtime.parent),
            "source_manifest_sha256": sha256_file(sums),
            "environment": {
                "native_rules": RULES_HASH_WIRE,
                "native_features": FEATURE_SCHEMA_HASH,
                "native_search": SEARCH_ALGORITHM_ID,
                "native_binaries": [op.artifact(native)],
                "training_module": str(module),
                "native_file": str(wrapper),
            },
        },
    )
    plan = op.prepare(
        screen,
        runtime_root=runtime,
        qualification=qualification,
        after_ns=time.time_ns(),
    )
    monkeypatch.setattr(continuation, "require_idle_gpus", lambda: None)
    return SimpleNamespace(
        root=screen, runtime=runtime, qualification=qualification, plan=plan, old=old
    )


def cuda_receipt(item, **overrides):
    plan = item.plan
    runtime = plan["runtime"]
    cpu = json.loads(f.pinned(runtime["qualification"]))
    value = {
        "schema_version": 1,
        "format": "deltreltrain.strength-freshness-cuda-qualification",
        "status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "source_commit": runtime["source_commit"],
        "source_manifest_sha256": runtime["source_checksums"]["sha256"],
        "profile_sha256": plan["target_profile"]["sha256"],
        "recovery_pointer_sha256": sha256_file(item.root / "learner/recovery.json"),
        "execution_pins": runtime["execution_pins"],
        "pyvenv": runtime["pyvenv"],
        **{
            key: cpu["environment"][key]
            for key in ("training_module", "native_file", "native_binaries")
        },
        "checks": {
            key: True
            for key in (
                "cuda_available",
                "learner_resume_step",
                "ema_preserved",
                "native_cuda_search",
                "all_workers_released",
            )
        },
        "fixture_only": True,
    }
    value.update(overrides)
    path = item.root.parent / "cuda-fixture.json"
    atomic_json(path, value)
    return {"cuda_receipt": path, "cuda_sha256": sha256_file(path)}


def test_preparation_changes_exactly_two_fields_and_keeps_authority(registered):
    r = registered
    plan, old, new = f.validate_registration(r.root)
    assert new == replace(
        old,
        learner=replace(
            old.learner,
            champion_only_replay_freshness=True,
            protected_champion_after_ns=plan["activation_after_ns"],
        ),
    )
    assert [
        row[0] for row in migration._validate_profile_pair(old, new, run_root=r.root)
    ] == ["learner." + name for name in f.FIELDS]
    assert resolve_active_profile(r.root).path.name == f.SOURCE
    assert (r.root / "source-commit.txt").read_text().strip() == "a" * 40
    assert not (r.root / op.JOURNAL).exists()
    assert (
        op.prepare(
            r.root,
            runtime_root=r.runtime,
            qualification=r.qualification,
            after_ns=plan["activation_after_ns"],
        )
        == plan
    )
    with pytest.raises(ValueError, match="never recompute"):
        op.prepare(
            r.root,
            runtime_root=r.runtime,
            qualification=r.qualification,
            after_ns=time.time_ns(),
        )


@pytest.mark.parametrize(
    "change",
    [
        "unsealed",
        "wrong-profile",
        "wrong-source",
        "runtime-code",
        "runtime-python",
        "coupled-field",
    ],
)
def test_registration_and_qualified_runtime_refuse_drift(registered, change):
    r = registered
    if change == "unsealed":
        (r.root / continuation.SEAL).unlink()
    elif change == "wrong-profile":
        (r.root / f.INPUT).chmod(0o644)
        (r.root / f.INPUT).write_text((r.root / f.SOURCE).read_text())
    elif change == "wrong-source":
        (r.runtime.parent / "SOURCE_COMMIT").write_text("c" * 40)
    elif change == "runtime-code":
        (r.runtime / "deltreltrain/__init__.py").write_text("changed")
    elif change == "runtime-python":
        (r.runtime / ".venv/bin/python").unlink()
        (r.runtime / ".venv/bin/python").symlink_to(
            r.runtime / ".venv/bin/deltreltrain-orchestrate"
        )
    else:
        _, old, new = f.validate_registration(r.root)
        with pytest.raises(migration.MigrationError):
            migration._validate_profile_pair(
                old,
                replace(new, train=replace(new.train, ema_decay=0.99)),
                run_root=r.root,
            )
        return
    with pytest.raises((ValueError, OSError)):
        op.apply(r.root)
    assert not (r.root / op.JOURNAL).exists()
    assert (r.root / "source-commit.txt").read_text().strip() == "a" * 40


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"status": "failed"},
        {"source_commit": "c" * 40},
        {"recovery_pointer_sha256": "0" * 64},
        {"checks": {"cuda_available": True}},
    ],
)
def test_no_durable_intent_without_actual_exact_cuda_receipt(registered, override):
    r = registered
    before = {
        name: (r.root / name).read_bytes()
        for name in (
            "profile.sha256",
            "source-commit.txt",
            "continuous-migrations.jsonl",
        )
    }
    kwargs = cuda_receipt(r, **override) if override else {}
    with pytest.raises(ValueError, match="CUDA"):
        op.apply(r.root, **kwargs)
    assert not (r.root / op.JOURNAL).exists()
    assert not (r.root / op.BOUNDARY).exists()
    assert all((r.root / name).read_bytes() == data for name, data in before.items())


def test_retained_boundary_rejects_symlink_escape_before_intent(registered, tmp_path):
    r = registered
    outside = tmp_path / "unowned"
    outside.mkdir()
    (r.root / f.PROVENANCE / "checkpoints").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        op.apply(r.root, **cuda_receipt(r))
    assert not list(outside.iterdir())
    assert not (r.root / op.JOURNAL).exists()


def test_apply_preserves_checkpoint_control_and_history_and_changes_active_source(
    registered,
):
    r = registered
    (r.root / "arena").mkdir(exist_ok=True)
    (r.root / "arena/inflight.resume.json").write_text('{"game": "unchanged"}')
    (r.root / "status/work-schedule.json").write_text('{"credits": 123}')
    (r.root / "learner/cadence.json").write_text(
        '{"candidate_interval_examples": 3000000}'
    )
    unchanged = {p: p.read_bytes() for p in op._unchanged_files(r.root)}
    original_controls = {
        p: p.read_bytes()
        for p in (
            r.root / continuation.SEAL,
            r.root / continuation.PLAN,
            r.root / "ablation.json",
        )
    }
    before_ledger = (r.root / "continuous-migrations.jsonl").read_bytes()
    runtime_before = {p: sha256_file(p) for p in r.runtime.rglob("*") if p.is_file()}
    result = op.apply(r.root, **cuda_receipt(r))
    assert result["status"] == "committed-recover-forward-only"
    boundary = f.document(r.root / op.BOUNDARY)
    checkpoint = Path(boundary["checkpoint"]["path"])
    original = r.root / "learner/recovery" / checkpoint.name
    assert checkpoint.stat().st_ino == original.stat().st_ino
    assert sha256_file(checkpoint) == boundary["checkpoint"]["sha256"]
    assert all(
        p.read_bytes() == data for p, data in {**unchanged, **original_controls}.items()
    )
    assert all(sha256_file(p) == sha for p, sha in runtime_before.items())
    ledger = (r.root / "continuous-migrations.jsonl").read_bytes()
    assert (
        ledger.startswith(before_ledger)
        and len(ledger[len(before_ledger) :].splitlines()) == 1
    )
    _, classification = validate_profile_for_monitor(
        r.root, resolve_active_profile(r.root)
    )
    assert classification == "registered_champion_only_freshness"
    assert (
        result["original_source_commit"] == "a" * 40
        and result["active_source_commit"] == "b" * 40
    )
    assert not boundary["migration_record"].get("utd_segment")
    from deltreltrain.strength_recovery_archive import backup_files

    closure = backup_files(r.root)
    assert (
        closure[str(checkpoint.relative_to(r.root))] == boundary["checkpoint"]["sha256"]
    )
    assert (
        closure[f.PROVENANCE + "/r4-cuda-qualification.json"]
        == boundary["cuda_qualification"]["sha256"]
    )
    assert op.apply(r.root) == result
    original.unlink()
    assert checkpoint.is_file()  # Retention survives ordinary recovery GC.


@pytest.mark.parametrize("field", ["changes", "step", "to_source_commit"])
def test_second_migration_tampering_is_refused(registered, field):
    r = registered
    op.apply(r.root, **cuda_receipt(r))
    path = r.root / "continuous-migrations.jsonl"
    rows = path.read_text().splitlines()
    row = json.loads(rows[-1])
    row[field] = [] if field == "changes" else 999 if field == "step" else "a" * 40
    path.write_text("\n".join([*rows[:-1], json.dumps(row)]) + "\n")
    with pytest.raises(ValueError, match="boundary|second migration"):
        f.validate_installed(r.root)


@pytest.mark.parametrize("point", ["before-write", "partial-ledger", "after-write"])
def test_interrupted_apply_completes_original_forward_transaction(
    registered, monkeypatch, point
):
    r = registered
    original_apply = migration.apply_migration

    class Crash(BaseException):
        pass

    def crash(plan, **kwargs):
        if point == "after-write":
            original_apply(plan, **kwargs)
        elif point == "partial-ledger":
            migration._atomic_write_bytes(
                plan.target_profile,
                plan.target_profile_bytes,
                mode=0o444,
                overwrite=False,
            )
            with (r.root / "continuous-migrations.jsonl").open("ab") as stream:
                stream.write(migration._json_bytes(plan.migration_record)[:30])
        raise Crash()

    monkeypatch.setattr(migration, "apply_migration", crash)
    with pytest.raises(Crash):
        op.apply(r.root, **cuda_receipt(r))
    journal = f.document(r.root / op.JOURNAL)
    boundary_bytes = (r.root / op.BOUNDARY).read_bytes()
    result = op.repair(r.root)
    assert result["status"] == "committed-recover-forward-only"
    assert (r.root / op.BOUNDARY).read_bytes() == boundary_bytes
    assert f.document(r.root / op.JOURNAL)["writes"] == journal["writes"]
    assert (r.root / "source-commit.txt").read_text().strip() == "b" * 40
    f.validate_installed(r.root)


def test_repair_refuses_changed_boundary_or_extra_owned_write(registered, monkeypatch):
    r = registered

    class Crash(BaseException):
        pass

    monkeypatch.setattr(
        migration, "apply_migration", lambda _, **k: (_ for _ in ()).throw(Crash())
    )
    with pytest.raises(Crash):
        op.apply(r.root, **cuda_receipt(r))
    path = r.root / op.JOURNAL
    journal = f.document(path)
    journal["writes"][0]["after"] = op.transaction._encode(b"wrong")
    atomic_json(path, journal)
    with pytest.raises(ValueError, match="after-bytes"):
        op.repair(r.root)
    assert resolve_active_profile(r.root).path.name == f.SOURCE


def test_interruption_after_boundary_before_intent_is_exactly_resumable(
    registered, monkeypatch
):
    r = registered
    original_intent = op.transaction.record_intent

    class Crash(BaseException):
        pass

    monkeypatch.setattr(
        op.transaction, "record_intent", lambda *a: (_ for _ in ()).throw(Crash())
    )
    kwargs = cuda_receipt(r)
    with pytest.raises(Crash):
        op.apply(r.root, **kwargs)
    before = (r.root / op.BOUNDARY).read_bytes()
    assert not (r.root / op.JOURNAL).exists()
    assert resolve_active_profile(r.root).path.name == f.SOURCE
    monkeypatch.setattr(op.transaction, "record_intent", original_intent)
    result = op.apply(r.root, **kwargs)
    assert result["status"] == "committed-recover-forward-only"
    assert (r.root / op.BOUNDARY).read_bytes() == before


@pytest.mark.parametrize("point", ["ledger", "profile.sha256", "source-commit.txt"])
def test_write_failure_never_truncates_registered_history(
    registered, monkeypatch, point
):
    r = registered
    original_write, original_append = (
        migration._atomic_write_bytes,
        migration._append_jsonl,
    )

    def write(path, *args, **kwargs):
        original_write(path, *args, **kwargs)
        if path == r.root / point:
            raise OSError("injected post-write failure")

    def append(path, row):
        original_append(path, row)
        if point == "ledger":
            raise OSError("injected post-append failure")

    monkeypatch.setattr(migration, "_atomic_write_bytes", write)
    monkeypatch.setattr(migration, "_append_jsonl", append)
    with pytest.raises(migration.MigrationError, match="retained for forward recovery"):
        op.apply(r.root, **cuda_receipt(r))
    after_failure = (r.root / "continuous-migrations.jsonl").read_bytes()
    assert len(after_failure.splitlines()) == 2
    monkeypatch.setattr(migration, "_atomic_write_bytes", original_write)
    monkeypatch.setattr(migration, "_append_jsonl", original_append)
    op.repair(r.root)
    assert (r.root / "continuous-migrations.jsonl").read_bytes() == after_failure
    f.validate_installed(r.root)


def test_repair_refuses_advanced_checkpoint_or_live_coordinator(
    registered, monkeypatch
):
    r = registered

    class Crash(BaseException):
        pass

    monkeypatch.setattr(
        migration, "apply_migration", lambda _, **k: (_ for _ in ()).throw(Crash())
    )
    with pytest.raises(Crash):
        op.apply(r.root, **cuda_receipt(r))
    pointer = r.root / "learner/recovery.json"
    original = pointer.read_bytes()
    pointer.write_bytes(original + b" ")
    with pytest.raises(ValueError):
        op.repair(r.root)
    pointer.write_bytes(original)
    import os

    atomic_json(r.root / "coordinator.lock", {"pid": os.getpid()})
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        op.repair(r.root)


def test_committed_history_cannot_automatically_migrate_back_to_r3(registered):
    r = registered
    op.apply(r.root, **cuda_receipt(r))
    _, old, new = f.validate_registration(r.root)
    with pytest.raises(migration.MigrationError, match="freshness"):
        migration._validate_profile_pair(new, old, run_root=r.root)
    ledger = r.root / "continuous-migrations.jsonl"
    ledger.write_bytes(
        ledger.read_bytes() + ledger.read_bytes().splitlines()[-1] + b"\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        f.validate_installed(r.root)


@pytest.mark.parametrize(
    "changed", ["status/work-schedule.json", "learner/cadence.json", "replay-counter"]
)
def test_forward_repair_refuses_changed_scheduler_cadence_or_replay_credit(
    registered, monkeypatch, changed
):
    r = registered
    for name in ("status/work-schedule.json", "learner/cadence.json"):
        (r.root / name).write_text('{"counter": 3}')

    class Crash(BaseException):
        pass

    monkeypatch.setattr(
        migration, "apply_migration", lambda *a, **k: (_ for _ in ()).throw(Crash())
    )
    with pytest.raises(Crash):
        op.apply(r.root, **cuda_receipt(r))
    if changed == "replay-counter":
        import sqlite3

        with sqlite3.connect(r.root / "replay/manifest.sqlite3") as connection:
            connection.execute(
                "UPDATE run_counters SET committed_samples=committed_samples+1"
            )
    else:
        (r.root / changed).write_text('{"counter": 4}')
    with pytest.raises(ValueError):
        op.repair(r.root)
    assert resolve_active_profile(r.root).path.name == f.SOURCE


def test_runtime_environment_and_launch_keep_literal_venv_and_original_clock(
    registered, monkeypatch
):
    r = registered
    op.apply(r.root, **cuda_receipt(r))
    old_state = f.document(r.root / continuation.STATE)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(key, "/unqualified")
    captured = {}

    class Latch:
        def install(self):
            pass

        def is_set(self):
            return True

    class Process:
        pid = 99999999
        returncode = 0

        def poll(self):
            return None

    def launch(command, **kwargs):
        captured.update(command=command, environment=kwargs["env"])
        return Process()

    monkeypatch.setattr(continuation, "SignalLatch", Latch)
    monkeypatch.setattr(continuation.subprocess, "Popen", launch)
    monkeypatch.setattr(
        continuation,
        "_terminate",
        lambda *a, **k: {"clean": True, "process_group_released": True},
    )
    state = op.run(r.root)
    assert captured["command"][:2] == [
        str(r.runtime / ".venv/bin/python"),
        str(r.runtime / ".venv/bin/deltreltrain-orchestrate"),
    ]
    env = captured["environment"]
    assert env["PYTHONPATH"] == str(r.runtime)
    assert env["VIRTUAL_ENV"] == str(r.runtime / ".venv")
    assert not any(
        key in env
        for key in (
            "PYTHONHOME",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
        )
    )
    assert state["continuation_started_ns"] == old_state["continuation_started_ns"]
    assert state["plan_sha256"] == r.old["plan_sha256"]
    assert state["attempts"][:-1] == old_state["attempts"]
    assert state["profile"] == str(r.root / f.INSTALLED)
