"""A pre-intent retry adds evidence; it never rewrites prior authority or proof."""

import fcntl
import os
from pathlib import Path
import shutil

import pytest
import torch

from test_strength_freshness_transition import (
    cuda_receipt,
    registered as registered,
    screen as screen,
    source as source,
)
from deltreltrain import strength_freshness as f
from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    load_recovery_pointer,
    sha256_file,
    write_recovery_checkpoint,
)
from deltreltrain.config import load_config
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import digest
from deltreltrain.strength_recovery_archive import backup_files
from deltreltrain.training import build_scheduler
from scripts import activate_strength_freshness as op
from scripts import migrate_continuous_profile as migration


class Crash(BaseException):
    pass


def retain(item, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            op.transaction, "record_intent", lambda *a: (_ for _ in ()).throw(Crash())
        )
        with pytest.raises(Crash):
            op.apply(item.root, **cuda_receipt(item))
    assert not op.current_journal_path(item.root).exists()
    return sha256_file(op.current_boundary_path(item.root))


def advance_r3(item):
    """A tiny fixture checkpoint moves forward using the real publication API."""
    root = item.root
    identity = f.document(root / "run.json")
    info = load_recovery_pointer(
        root / "learner/recovery.json",
        expected_run_id=identity["run_id"],
        expected_generation_family=identity["generation_family"],
    )
    saved = torch.load(info.checkpoint, weights_only=True)
    config = load_config(root / f.SOURCE)
    model = GraphResTNet(config.model)
    model.load_state_dict(saved["model"])
    optimizer = build_optimizer(model, config.optimizer)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    scheduler.load_state_dict(saved["scheduler"])
    ema = ExponentialMovingAverage(model, decay=saved["ema"]["decay"])
    ema.load_state_dict(saved["ema"])
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    reserved = {
        "training_step_version",
        "run_id",
        "generation_family",
        "examples_consumed",
        "global_batch_size",
        "utd_segment",
    }
    extra = saved["extra"]
    write_recovery_checkpoint(
        root / "learner",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=saved["step"] + 1,
        epoch=saved["epoch"],
        config=config.as_dict(),
        run_id=extra["run_id"],
        generation_family=extra["generation_family"],
        examples_consumed=extra["examples_consumed"] + 1,
        global_batch_size=extra["global_batch_size"],
        utd_segment=extra.get("utd_segment"),
        extra={k: v for k, v in extra.items() if k not in reserved},
    )
    heartbeat = f.document(root / "status/learner.heartbeat.json")
    heartbeat.update(
        step=saved["step"] + 1, examples_consumed=extra["examples_consumed"] + 1
    )
    atomic_json(root / "status/learner.heartbeat.json", heartbeat)


def prepare_again(item, checksum, attempt="retry-one"):
    return op.reprepare(item.root, attempt_id=attempt, boundary_sha256=checksum)


def test_new_boundary_keeps_old_evidence_clocks_history_and_relocates(
    registered, monkeypatch, tmp_path
):
    r = registered
    checksum = retain(r, monkeypatch)
    old = {p: p.read_bytes() for p in (r.root / f.PROVENANCE).rglob("*") if p.is_file()}
    old[r.root / op.BOUNDARY] = (r.root / op.BOUNDARY).read_bytes()
    plans = {
        p: p.read_bytes()
        for p in (
            r.root / f.PLAN,
            r.root / f.CONTINUATION_PLAN,
            r.root / "strength-screen-seal.json",
            r.root / f.INPUT,
        )
    }
    ledger = (r.root / "continuous-migrations.jsonl").read_bytes()
    advance_r3(r)
    with pytest.raises(ValueError, match="boundary changed"):
        op.apply(r.root, **cuda_receipt(r))
    receipt = prepare_again(r, checksum)
    assert receipt["generation"] == 1
    assert receipt["activation_after_ns"] == r.plan["activation_after_ns"]
    assert receipt["continuation_started_ns"] == r.plan["continuation_started_ns"]
    assert (r.root / "continuous-migrations.jsonl").read_bytes() == ledger
    assert all(p.read_bytes() == data for p, data in {**old, **plans}.items())
    assert prepare_again(r, checksum) == receipt
    assert not op.current_boundary_path(r.root).exists()
    assert op.inspect_authority(r.root)["phase"] == "r3"
    before_apply = backup_files(r.root)
    assert op.BOUNDARY in before_apply
    committed = op.apply(r.root, **cuda_receipt(r))
    active = op.current_boundary_path(r.root)
    assert active == r.root / f.ATTEMPTS / "000001" / op.BOUNDARY
    assert committed["boundary_sha256"] == sha256_file(active)
    assert committed["boundary_path"] == str(active.relative_to(r.root))
    assert all(p.read_bytes() == data for p, data in {**old, **plans}.items())
    assert (
        len(
            (r.root / "continuous-migrations.jsonl")
            .read_bytes()[len(ledger) :]
            .splitlines()
        )
        == 1
    )
    closure = backup_files(r.root)
    assert before_apply.items() <= closure.items()
    assert (
        len([p for p in closure if p.startswith(f.PROVENANCE + "/checkpoints/")]) == 2
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(r.root, relocated)
    original_read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: (
            pytest.fail("binary read_bytes") if p.suffix == ".pt" else original_read(p)
        ),
    )
    assert backup_files(relocated) == closure


@pytest.mark.parametrize("point", ["before", "after"])
def test_repreparation_publication_crash_is_idempotent(registered, monkeypatch, point):
    r = registered
    checksum = retain(r, monkeypatch)
    advance_r3(r)
    publish = op._publish_repreparation

    def crash(*args):
        if point == "after":
            publish(*args)
        else:
            args[1].parent.mkdir(parents=True)
        raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(op, "_publish_repreparation", crash)
        with pytest.raises(Crash):
            prepare_again(r, checksum)
    prior = f.repreparations(r.root, r.plan)
    assert len(prior) == (point == "after")
    result = prepare_again(r, checksum)
    assert result["generation"] == 1
    assert not prior or prior[0] == result
    assert prepare_again(r, checksum) == result


@pytest.mark.parametrize(
    "mutation",
    [
        "intent",
        "committed",
        "installed",
        "history",
        "source",
        "profile",
        "boundary-hash",
        "cuda",
        "symlink",
    ],
)
def test_repreparation_refuses_intent_unknown_authority_and_tamper(
    registered, monkeypatch, mutation
):
    r = registered
    checksum = retain(r, monkeypatch)
    if mutation == "intent":
        atomic_json(r.root / op.JOURNAL, {"status": "pending"})
    elif mutation == "committed":
        op.apply(r.root, **cuda_receipt(r))
    elif mutation == "installed":
        (r.root / f.INSTALLED).write_bytes((r.root / f.INPUT).read_bytes())
    elif mutation == "history":
        with (r.root / "continuous-migrations.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
    elif mutation == "source":
        (r.root / "source-commit.txt").write_text("c" * 40 + "\n")
    elif mutation == "profile":
        (r.root / "profile.sha256").write_text("0" * 64 + "  changed.yaml\n")
    elif mutation == "boundary-hash":
        checksum = "0" * 64
    elif mutation == "cuda":
        path = r.root / f.PROVENANCE / "r4-cuda-qualification.json"
        path.chmod(0o644)
        path.write_text("{}")
    else:
        (r.root / f.ATTEMPTS).symlink_to(r.root.parent, target_is_directory=True)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        prepare_again(r, checksum)
    assert not list((r.root / f.PROVENANCE).glob("attempts/*/reprepare.json"))


@pytest.mark.parametrize("lock", ["operator", "supervisor", "coordinator"])
def test_repreparation_refuses_concurrent_owners(registered, monkeypatch, lock):
    r = registered
    checksum = retain(r, monkeypatch)
    if lock == "operator":
        with op._operation_lock(r.root), pytest.raises(BlockingIOError):
            prepare_again(r, checksum)
    elif lock == "supervisor":
        with (r.root / ".strength-continuation.lock").open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                prepare_again(r, checksum)
    else:
        atomic_json(r.root / "coordinator.lock", {"pid": os.getpid(), "created_ns": 1})
        with pytest.raises(migration.MigrationError, match="live"):
            prepare_again(r, checksum)
    assert not (r.root / f.ATTEMPTS).exists()


def test_multiple_generations_refuse_reused_ids_and_chain_tamper(
    registered, monkeypatch
):
    r = registered
    checksum = retain(r, monkeypatch)
    prepare_again(r, checksum)
    second = retain(r, monkeypatch)
    advance_r3(r)
    receipt = prepare_again(r, second, "retry-two")
    assert receipt["generation"] == 2
    assert len(f.repreparations(r.root, r.plan)) == 2
    with pytest.raises(ValueError, match="identity"):
        prepare_again(r, checksum)
    path = r.root / f.attempt_directory(2) / f.REPREPARE
    receipt["previous_receipt_sha256"] = "0" * 64
    receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="chain"):
        op.current_boundary_path(r.root)
    with pytest.raises(ValueError, match="chain"):
        backup_files(r.root)


def test_repreparation_schema_boolean_is_not_a_version(registered, monkeypatch):
    r = registered
    checksum = retain(r, monkeypatch)
    receipt = prepare_again(r, checksum)
    receipt["schema_version"] = True
    receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    atomic_json(r.root / f.attempt_directory(1) / f.REPREPARE, receipt)
    with pytest.raises(ValueError, match="chain"):
        op.current_boundary_path(r.root)
    with pytest.raises(ValueError, match="chain"):
        backup_files(r.root)


@pytest.mark.parametrize("control", ["boundary", "receipt"])
def test_inspection_refuses_dangling_authority_symlinks(registered, control):
    r = registered
    if control == "receipt":
        op.apply(r.root, **cuda_receipt(r))
        path = op.current_receipt_path(r.root)
        path.unlink()
    else:
        path = op.current_boundary_path(r.root)
    path.symlink_to(r.root / "missing-control")
    with pytest.raises(ValueError, match="symbolic links"):
        op.inspect_authority(r.root)


@pytest.mark.parametrize(
    "point",
    [
        "r3",
        "pre-intent",
        "pending-before",
        "pending-partial",
        "pending-after",
        "committed",
    ],
)
def test_inspection_is_read_only_and_classifies_only_exact_authority(
    registered, monkeypatch, point
):
    r = registered
    if point == "pre-intent":
        retain(r, monkeypatch)
        advance_r3(r)
    elif point.startswith("pending"):
        original = migration.apply_migration

        def crash(plan, **kwargs):
            if point == "pending-after":
                original(plan, **kwargs)
            elif point == "pending-partial":
                with (r.root / "continuous-migrations.jsonl").open("ab") as stream:
                    stream.write(migration._json_bytes(plan.migration_record)[:17])
            raise Crash()

        with monkeypatch.context() as patch:
            patch.setattr(migration, "apply_migration", crash)
            with pytest.raises(Crash):
                op.apply(r.root, **cuda_receipt(r))
    elif point == "committed":
        op.apply(r.root, **cuda_receipt(r))
    before = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in r.root.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        op.continuation,
        "require_idle_gpus",
        lambda: pytest.fail("inspection queried GPUs"),
    )
    result = op.inspect_authority(r.root)
    assert result["phase"] == (
        "r4"
        if point == "committed"
        else "pending"
        if point.startswith("pending")
        else "r3"
    )
    assert len(result["evidence_sha256"]) == 64
    after = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in r.root.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-intent",
        "unknown-bytes",
        "forged-after",
        "wrong-root",
        "advanced-boundary",
    ],
)
def test_inspection_never_downgrades_unknown_intent_to_r3(
    registered, monkeypatch, mutation
):
    r = registered
    with monkeypatch.context() as patch:
        patch.setattr(
            migration, "apply_migration", lambda *a, **k: (_ for _ in ()).throw(Crash())
        )
        with pytest.raises(Crash):
            op.apply(r.root, **cuda_receipt(r))
    journal = f.document(r.root / op.JOURNAL)
    if mutation == "missing-intent":
        (r.root / op.JOURNAL).unlink()
        (r.root / f.INSTALLED).write_text("unknown")
    elif mutation == "unknown-bytes":
        (r.root / "source-commit.txt").write_text("unknown")
    elif mutation == "forged-after":
        journal["writes"][0]["after"] = op.transaction._encode(b"forged")
        atomic_json(r.root / op.JOURNAL, journal)
    elif mutation == "wrong-root":
        journal["run_root"] = str(r.root.parent)
        atomic_json(r.root / op.JOURNAL, journal)
    else:
        advance_r3(r)
    with pytest.raises((OSError, ValueError)):
        op.inspect_authority(r.root)
