from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from deltreltrain.config import (
    ConfigError,
    LearnerConfig,
    SchedulerConfig,
    TrainConfig,
    load_config,
)
from deltreltrain.learner import LearnerLoop
from deltreltrain.lr_governor import LearningRateGovernorState, apply_governor
from deltreltrain.reuse_clock import ReuseClock, checkpoint_clock, continuation_clock
from deltreltrain.training import build_scheduler


def subject(target=1.5, reference=None):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train = TrainConfig(
        ema_decay=0.9, scheduler=SchedulerConfig(warmup_steps=0, total_steps=100)
    )
    learner = LearnerConfig(
        target_updates_per_new_sample=target, reuse_clock_reference_target=reference
    )
    loop = object.__new__(LearnerLoop)
    loop.model, loop.optimizer = model, optimizer
    model.config = SimpleNamespace(auxiliary_predictions=False)
    loop.scheduler = build_scheduler(optimizer, train.scheduler)
    loop.ema = ExponentialMovingAverage(model, decay=train.ema_decay)
    loop.train_config, loop.learner_config = train, learner
    loop.serialized_config = {"train": asdict(train), "learner": asdict(learner)}
    loop.step, loop.world_size = 0, 1
    loop.rank = 0
    loop.gradient_clipper = None
    loop.run_identity = SimpleNamespace(run_id="reuse-test", generation_family="reuse")
    loop.metrics = SimpleNamespace(append=lambda _event: None)
    loop._lr_governor = LearningRateGovernorState.from_scheduler(loop.scheduler)
    loop._reuse_clock = None
    return loop


def advance(loop, count):
    for _ in range(count):
        loop.optimizer.zero_grad()
        loop.model(torch.ones(1, 2)).sum().backward()
        loop.optimizer.step()
        loop.scheduler.step()
        loop.ema.update(loop.model)
        loop.step += 1


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_reuse_continuation_preserves_full_state_and_data_clocks(tmp_path):
    source = subject()
    advance(source, 7)
    before_lr = source.optimizer.param_groups[0]["lr"]
    path = save_checkpoint(
        tmp_path / "source.pt",
        model=source.model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        ema=source.ema,
        step=source.step,
        config=source.serialized_config,
    )
    treatment = subject(2.0, 1.5)
    metadata = load_checkpoint(
        path,
        model=treatment.model,
        optimizer=treatment.optimizer,
        scheduler=treatment.scheduler,
        ema=treatment.ema,
    )
    treatment.step = metadata["step"]
    treatment._configure_reuse_clock(metadata)
    apply_governor(treatment.optimizer, treatment.scheduler, treatment._lr_governor)
    assert_tree_equal(source.model.state_dict(), treatment.model.state_dict())
    assert_tree_equal(source.optimizer.state_dict(), treatment.optimizer.state_dict())
    assert_tree_equal(source.ema.shadow, treatment.ema.shadow)
    assert source.ema.num_updates == treatment.ema.num_updates
    assert treatment.optimizer.param_groups[0]["lr"] == before_lr
    assert treatment.ema.decay**4 == pytest.approx(source.ema.decay**3)
    advance(source, 3)
    advance(treatment, 4)
    assert treatment.optimizer.param_groups[0]["lr"] == pytest.approx(
        source.optimizer.param_groups[0]["lr"]
    )
    assert treatment._reuse_clock.reference_age(treatment.scheduler.last_epoch) == 10

    checkpoint = save_checkpoint(
        tmp_path / "treatment.pt",
        model=treatment.model,
        optimizer=treatment.optimizer,
        scheduler=treatment.scheduler,
        ema=treatment.ema,
        step=treatment.step,
        config=treatment.serialized_config,
        extra={"reuse_clock": treatment._reuse_clock.as_dict()},
    )
    resumed = subject(2.0, 1.5)
    resumed.ema.decay = treatment.ema.decay
    metadata = load_checkpoint(
        checkpoint,
        model=resumed.model,
        optimizer=resumed.optimizer,
        scheduler=resumed.scheduler,
        ema=resumed.ema,
    )
    resumed.step = metadata["step"]
    resumed._configure_reuse_clock(metadata)
    assert resumed._reuse_clock == treatment._reuse_clock
    advance(resumed, 1)
    advance(treatment, 1)
    assert_tree_equal(resumed.model.state_dict(), treatment.model.state_dict())
    assert_tree_equal(resumed.optimizer.state_dict(), treatment.optimizer.state_dict())
    assert_tree_equal(resumed.ema.state_dict(), treatment.ema.state_dict())


def test_reuse_clock_reanchors_reversal_without_learning_rate_jump():
    clock = ReuseClock(1.5, 2.0, 100, 100, 100.0, 0.9999)
    reverted = continuation_clock(
        previous=clock,
        previous_target=2.0,
        reference_target=1.5,
        target=1.5,
        step=140,
        scheduler_step=140,
        reference_ema_decay=0.9999,
    )
    assert reverted.reference_age(140) == 130
    assert reverted.reference_age(150) == 140
    assert reverted.ema_decay == 0.9999


@pytest.mark.parametrize("updates", [True, -1, float("nan"), float("inf"), 0])
def test_invalid_reference_rejected(updates):
    with pytest.raises(ConfigError, match="reuse clock"):
        LearnerConfig(
            target_updates_per_new_sample=2, reuse_clock_reference_target=updates
        )


def test_corrupt_or_mismatched_checkpoint_clock_fails_closed():
    clock = ReuseClock(1.5, 2, 10, 10, 10, 0.9)
    metadata = {
        "step": 20,
        "extra": {"reuse_clock": clock.as_dict()},
        "config": {
            "learner": {
                "reuse_clock_reference_target": 1.5,
                "target_updates_per_new_sample": 1.5,
            }
        },
    }
    with pytest.raises(ValueError, match="differs"):
        checkpoint_clock(metadata)
    with pytest.raises(ValueError, match="reference clock"):
        continuation_clock(
            previous=clock,
            previous_target=2,
            reference_target=1,
            target=2,
            step=20,
            scheduler_step=20,
            reference_ema_decay=0.9,
        )
    with pytest.raises(ValueError, match="precedes"):
        clock.reference_age(9)


def test_disabled_clock_does_not_change_profile_authority():
    from pathlib import Path

    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    assert "reuse_clock_reference_target" not in config.as_dict()["learner"]


def save_subject(loop, path):
    return save_checkpoint(
        path,
        model=loop.model,
        optimizer=loop.optimizer,
        scheduler=loop.scheduler,
        ema=loop.ema,
        step=loop.step,
        config=loop.serialized_config,
        extra={
            **loop._checkpoint_extra(),
            "run_id": loop.run_identity.run_id,
            "generation_family": loop.run_identity.generation_family,
            "examples_consumed": loop.step * loop.train_config.global_batch_size(1),
        },
    )


def complete_state(loop):
    return deepcopy(
        {
            "model": loop.model.state_dict(),
            "optimizer": loop.optimizer.state_dict(),
            "scheduler": loop.scheduler.state_dict(),
            "ema": loop.ema.state_dict(),
        }
    )


def test_real_resume_preserves_state_across_reuse_restart_and_reversal(tmp_path):
    source = subject()
    advance(source, 7)
    # The plateau multiplier is independent of the new data clock.
    source._lr_governor = source._lr_governor.with_multiplier(
        0.5, scaled_champion_identity="champion"
    )
    apply_governor(source.optimizer, source.scheduler, source._lr_governor)
    checkpoint = save_subject(source, tmp_path / "source.pt")
    treatment = subject(2.0, 1.5)
    treatment.resume(checkpoint)
    assert_tree_equal(source.model.state_dict(), treatment.model.state_dict())
    assert_tree_equal(source.optimizer.state_dict(), treatment.optimizer.state_dict())
    assert_tree_equal(source.ema.shadow, treatment.ema.shadow)
    assert treatment.ema.num_updates == source.ema.num_updates
    assert treatment._lr_governor == source._lr_governor
    assert treatment.ema.decay**4 == pytest.approx(source.ema.decay**3)
    advance(source, 3)
    advance(treatment, 4)
    assert treatment.optimizer.param_groups[0]["lr"] == pytest.approx(
        source.optimizer.param_groups[0]["lr"]
    )

    checkpoint = save_subject(treatment, tmp_path / "treatment.pt")
    restarted = subject(2.0, 1.5)
    restarted.resume(checkpoint)
    assert_tree_equal(complete_state(treatment), complete_state(restarted))
    assert restarted._reuse_clock == treatment._reuse_clock

    reverted = subject(1.5, 1.5)
    reverted.resume(checkpoint)
    assert_tree_equal(treatment.model.state_dict(), reverted.model.state_dict())
    assert_tree_equal(treatment.optimizer.state_dict(), reverted.optimizer.state_dict())
    assert_tree_equal(treatment.ema.shadow, reverted.ema.shadow)
    assert treatment.ema.num_updates == reverted.ema.num_updates
    assert reverted.ema.decay == source.ema.decay
    assert reverted._reuse_clock.reference_age(reverted.scheduler.last_epoch) == 10
    advance(reverted, 3)
    advance(treatment, 4)
    assert reverted.optimizer.param_groups[0]["lr"] == pytest.approx(
        treatment.optimizer.param_groups[0]["lr"]
    )
    # A second restart of the reverse continuation keeps its accumulated age.
    checkpoint = save_subject(reverted, tmp_path / "reverted.pt")
    reversed_restart = subject(1.5, 1.5)
    reversed_restart.resume(checkpoint)
    assert_tree_equal(complete_state(reverted), complete_state(reversed_restart))
    assert (
        reversed_restart._reuse_clock.reference_age(
            reversed_restart.scheduler.last_epoch
        )
        == 13
    )


@pytest.mark.parametrize(
    "mutation",
    ["reference", "scheduler", "ema", "batch", "missing_clock", "anchor", "actual_ema"],
)
def test_resume_rejects_invalid_reuse_before_mutating_any_state(tmp_path, mutation):
    source = subject()
    advance(source, 7)
    source_path = save_subject(source, tmp_path / "source.pt")
    treatment = subject(2.0, 1.5)
    treatment.resume(source_path)
    advance(treatment, 4)
    path = save_subject(treatment, tmp_path / "treatment.pt")
    target = subject(2.0, 1.5)
    # Existing optimizer moments/EMA history make partial loads observable.
    advance(target, 2)
    if mutation == "reference":
        target.learner_config = replace(
            target.learner_config, reuse_clock_reference_target=1.0
        )
    elif mutation in ("scheduler", "ema", "batch"):
        if mutation == "scheduler":
            target.train_config = replace(
                target.train_config,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=200),
            )
        elif mutation == "ema":
            target.train_config = replace(target.train_config, ema_decay=0.95)
        else:
            target.train_config = replace(target.train_config, per_rank_batch_size=4)
        target.serialized_config["train"] = asdict(target.train_config)
    else:
        payload = torch.load(path, weights_only=True)
        if mutation == "missing_clock":
            del payload["extra"]["reuse_clock"]
        elif mutation == "anchor":
            payload["extra"]["reuse_clock"]["anchor_scheduler_step"] = 1_000
        else:
            payload["ema"]["decay"] = 0.5
        torch.save(payload, path)
    before = complete_state(target)
    before_step = target.step
    with pytest.raises(ValueError, match="reuse|LR/EMA/batch"):
        target.resume(path)
    assert_tree_equal(before, complete_state(target))
    assert target.step == before_step
