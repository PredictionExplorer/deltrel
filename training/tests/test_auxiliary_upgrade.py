from dataclasses import asdict, replace
import hashlib

import pytest
import torch

from deltreltrain.auxiliary_upgrade import is_auxiliary_parameter
from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from deltreltrain.gradient_clipping import GradientClipper, GradientClippingConfig
from deltreltrain.model import GraphResTNet, ModelConfig
from deltreltrain.optim import (
    OptimizerConfig,
    build_optimizer,
    optimizer_checkpoint_contract,
)


def state(*, auxiliary=False, kind="muon_adamw", clipping="global"):
    config = ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=2,
        kv_heads=1,
        auxiliary_predictions=auxiliary,
    )
    model = GraphResTNet(config)
    optimizer = build_optimizer(model, OptimizerConfig(kind=kind))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    ema = ExponentialMovingAverage(model, decay=0.99)
    clipper = GradientClipper(
        model.named_parameters(),
        max_norm=1.0,
        config=GradientClippingConfig(mode=clipping),
    )
    return model, optimizer, scheduler, ema, clipper


def update(model, optimizer, scheduler, ema, clipper):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    clipper.apply_(clipper.measure())
    optimizer.step()
    scheduler.step()
    ema.update(model)
    optimizer.zero_grad(set_to_none=True)


def by_name(optimizer):
    contract = optimizer_checkpoint_contract(optimizer)
    saved = optimizer.state_dict()
    return {
        name: saved["state"].get(index, {})
        for route, group in zip(contract["groups"], saved["param_groups"], strict=True)
        for name, index in zip(route["parameter_names"], group["params"], strict=True)
    }


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", ["muon_adamw", "adamw"])
@pytest.mark.parametrize("clipping", ["global", "adagc"])
def test_upgrade_preserves_existing_weights_moments_and_clocks(
    tmp_path, kind, clipping
):
    old, optimizer, scheduler, ema, clipper = state(kind=kind, clipping=clipping)
    update(old, optimizer, scheduler, ema, clipper)
    source = save_checkpoint(
        tmp_path / "original.pt",
        model=old,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
        step=413000,
        epoch=23,
        config={"model": asdict(old.config)},
        extra={"examples_consumed": 211456000, "run_id": "test"},
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    new, new_optimizer, new_scheduler, new_ema, new_clipper = state(
        auxiliary=True, kind=kind, clipping=clipping
    )
    new_initial = {
        k: v.clone() for k, v in new.state_dict().items() if is_auxiliary_parameter(k)
    }
    metadata = load_checkpoint(
        source,
        model=new,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        ema=new_ema,
        gradient_clipper=new_clipper,
        expected_model_config=asdict(new.config),
        allow_auxiliary_upgrade=True,
    )
    assert metadata["step"] == 413000 and metadata["epoch"] == 23
    assert metadata["extra"]["examples_consumed"] == 211456000
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    for name, value in old.state_dict().items():
        equal(new.state_dict()[name], value)
        equal(new_ema.shadow[name], ema.shadow[name])
    equal(new_scheduler.state_dict(), scheduler.state_dict())
    assert new_ema.num_updates == ema.num_updates
    assert new_clipper.updates == clipper.updates
    for name, value in clipper.state_dict()["ema_norms"].items():
        equal(new_clipper.state_dict()["ema_norms"][name], value)
    old_states, new_states = by_name(optimizer), by_name(new_optimizer)
    for name in old_states:
        equal(new_states[name], old_states[name])
    for name, value in new_initial.items():
        equal(new.state_dict()[name], value)
        assert new_states[name] == {}
    update(new, new_optimizer, new_scheduler, new_ema, new_clipper)
    assert all(by_name(new_optimizer)[name] for name in new_initial)


def test_upgrade_is_explicit_and_rejects_other_architecture_changes(tmp_path):
    old, optimizer, scheduler, ema, clipper = state()
    source = save_checkpoint(
        tmp_path / "original.pt",
        model=old,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
        step=7,
        config={"model": asdict(old.config)},
    )
    new, new_optimizer, _, _, new_clipper = state(auxiliary=True)
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(source, model=new, expected_model_config=asdict(new.config))
    with pytest.raises(ValueError, match="only an additive"):
        load_checkpoint(
            source,
            model=new,
            optimizer=new_optimizer,
            gradient_clipper=new_clipper,
            expected_model_config=asdict(replace(new.config, rrt_groups=2)),
            allow_auxiliary_upgrade=True,
        )
    with pytest.raises(ValueError, match="incompatible"):
        inspect_checkpoint(source, expected_model_config=asdict(new.config))
    assert (
        inspect_checkpoint(
            source,
            expected_model_config=asdict(new.config),
            allow_auxiliary_upgrade=True,
        )["step"]
        == 7
    )


def test_legacy_config_authority_excludes_new_default_fields():
    from deltreltrain.config import load_config
    from deltreltrain.auxiliary_upgrade import AUXILIARY_LOSSES

    config = load_config("configs/small.yaml")
    payload = config.as_dict()
    assert "auxiliary_predictions" not in payload["model"]
    assert not set(AUXILIARY_LOSSES) & payload["loss"].keys()


def test_learner_records_only_observed_supervision_and_retains_first_step():
    from types import SimpleNamespace
    from deltreltrain.learner import LearnerLoop

    learner = object.__new__(LearnerLoop)
    learner.step = 100
    learner._lr_governor = SimpleNamespace(as_dict=lambda: {})
    learner._record_auxiliary_supervision({"opponent_reply_available": 0.0})
    assert "auxiliary_supervision" not in learner._checkpoint_extra()
    learner._record_auxiliary_supervision({"final_shores_available": 4.0})
    learner.step = 110
    learner._record_auxiliary_supervision(
        {"final_shores_available": 8.0, "opponent_reply_available": 2.0}
    )
    assert learner._checkpoint_extra()["auxiliary_supervision"] == {
        "final_shores": 100,
        "opponent_reply": 110,
    }
