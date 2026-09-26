from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from deltreltrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH
from deltreltrain.gradient_clipping import GradientClipper, GradientClippingConfig
from deltreltrain.losses import LossWeights
from deltreltrain.optim import OptimizerConfig, build_optimizer
from deltreltrain.rebrand import (
    SOURCE_FEATURE_HASH,
    SOURCE_RULES_HASH,
    migrate_checkpoint,
    migrate_payload,
)
from deltreltrain.topology import coordinate_label, get_topology


def _model(names: tuple[str, str, str]) -> nn.Module:
    model = nn.Module()
    for name, outputs in zip(names, (102, 52, 12), strict=True):
        model.add_module(f"final_{name}_head", nn.Linear(4, outputs))
    return model


def _source(tmp_path: Path) -> tuple[Path, nn.Module, torch.optim.Optimizer]:
    model = _model(("coast", "channels", "corners"))
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    clipper = GradientClipper(
        model.named_parameters(), config=GradientClippingConfig(mode="adagc")
    )
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    clipper.apply_(clipper.measure())
    optimizer.step()
    ema = ExponentialMovingAverage(model)
    path = save_checkpoint(
        tmp_path / "source.pt",
        model=model,
        optimizer=optimizer,
        ema=ema,
        step=7,
        gradient_clipper=clipper,
        config={
            "loss": {"final_coast": 0.1, "final_channels": 0.2, "final_corners": 0.3}
        },
    )
    payload = torch.load(path, weights_only=True)
    payload.update(
        format="previous.checkpoint",
        rules_schema="previous.rules.v3",
        rules_hash=SOURCE_RULES_HASH,
        rules_hash_wire=f"fnv1a64:{SOURCE_RULES_HASH:016x}",
        feature_schema_hash=SOURCE_FEATURE_HASH,
        action_layout_schema="previous.action-layout.nodes-only.v1",
    )
    payload["gradient_clipping"]["format"] = "previous.gradient-clipping"
    torch.save(payload, path)
    return path, model, optimizer


def test_migration_preserves_weights_ema_optimizer_and_progress(tmp_path: Path) -> None:
    source, original, original_optimizer = _source(tmp_path)
    original_bytes = source.read_bytes()
    destination = migrate_checkpoint(source, tmp_path / "deltrel.pt")
    model = _model(("shores", "networks", "capes"))
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    ema = ExponentialMovingAverage(model)
    clipper = GradientClipper(
        model.named_parameters(), config=GradientClippingConfig(mode="adagc")
    )
    metadata = load_checkpoint(
        destination, model=model, optimizer=optimizer, ema=ema, gradient_clipper=clipper
    )
    assert metadata["step"] == 7
    assert source.read_bytes() == original_bytes
    for before, after in zip(original.parameters(), model.parameters(), strict=True):
        assert torch.equal(before, after)
    for before, after in zip(
        original_optimizer.state.values(), optimizer.state.values(), strict=True
    ):
        for name, tensor in before.items():
            assert torch.equal(tensor, after[name])
    for name, tensor in model.state_dict().items():
        assert torch.equal(ema.shadow[name], tensor)
    payload = torch.load(destination, weights_only=True)
    assert payload["rules_hash"] == RULES_HASH
    assert payload["feature_schema_hash"] == FEATURE_SCHEMA_HASH
    loss = LossWeights(**payload["config"]["loss"])
    assert (loss.final_shores, loss.final_networks, loss.final_capes) == (0.1, 0.2, 0.3)
    assert clipper.updates == 1
    source_clipping = torch.load(source, weights_only=True)["gradient_clipping"]
    for before, after in zip(
        source_clipping["ema_norms"].values(),
        clipper.state_dict()["ema_norms"].values(),
        strict=True,
    ):
        assert torch.equal(before, after)
    assert payload["extra"]["deltrel_identity_migration"]["tensor_values_unchanged"]
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, destination)
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, source)


@pytest.mark.parametrize("field", ["rules_hash", "feature_schema_hash", "version"])
def test_migration_rejects_unknown_contract(tmp_path: Path, field: str) -> None:
    source, _, _ = _source(tmp_path)
    payload = torch.load(source, weights_only=True)
    payload[field] = 0
    with pytest.raises(ValueError, match="exact preceding"):
        migrate_payload(payload)


def test_migration_rejects_incomplete_heads_and_corrupt_routing(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    payload = torch.load(source, weights_only=True)
    del payload["model"]["final_channels_head.weight"]
    with pytest.raises(ValueError, match="incomplete"):
        migrate_payload(payload)
    payload = torch.load(source, weights_only=True)
    payload["optimizer_routing"]["routing_hash"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        migrate_payload(payload)


def test_headless_checkpoints_migrate_only_unambiguous_disabled_count_losses(
    tmp_path: Path,
) -> None:
    source, _, _ = _source(tmp_path)
    payload = torch.load(source, weights_only=True)
    payload["model"] = {"linear.weight": torch.ones(4, 4)}
    payload["optimizer_routing"] = None
    payload["config"]["loss"] = {
        "final_coast": 0.0,
        "final_channels": 0.0,
        "final_corners": 0.0,
    }
    converted = migrate_payload(payload)
    assert LossWeights(**converted["config"]["loss"]).final_networks == 0.0
    payload["config"]["loss"]["final_channels"] = 0.5
    with pytest.raises(ValueError, match="matching model heads"):
        migrate_payload(payload)


@pytest.mark.parametrize("rings", [4, 6, 8, 10])
def test_polar_coordinates_are_unique_round_trip_and_ordered(rings: int) -> None:
    topology = get_topology(rings)
    assert len(set(topology.labels)) == topology.n
    assert [
        topology.labels[topology.idx(s, rings, 0)] for s in range(5)
    ] == [f"{arm}{rings % 10}0" for arm in "ABCDE"]
    for node, label in enumerate(topology.labels):
        assert len(label) == 3
        assert label[0] in "ABCDE" and label[1:].isdecimal()
        assert topology.label_to_id(label) == node
        assert topology.label_to_id(f" {label.lower()}\n") == node
        assert topology.label_to_id(f"\ufeff{label.lower()}\u00a0") == node
        assert coordinate_label(node, rings) == label
        assert ord(label[0]) - ord("A") == int(topology.sector_of[node])
        assert (int(label[1]) or 10) == int(topology.ring_of[node])
        assert int(label[2]) == int(topology.pos_of[node])
    assert topology.labels == get_topology(10).labels[:topology.n]
    for invalid in ("A", "F10", "A 10", "A0", "A11", "Z999", "A1"):
        with pytest.raises(ValueError, match="unknown node label"):
            topology.label_to_id(invalid)
    with pytest.raises(ValueError, match="unknown node label"):
        topology.label_to_id(f"\u0085{topology.labels[0]}")


def test_polar_coordinate_validation_and_size_independent_origin() -> None:
    assert [coordinate_label(0, rings) for rings in (4, 6, 8, 10)] == [
        "A10",
        "A10",
        "A10",
        "A10",
    ]
    for invalid in (-1, 0.5, True, 50):
        with pytest.raises(ValueError):
            coordinate_label(invalid, 4)  # type: ignore[arg-type]
    for rings in (0, 5, 12, True):
        with pytest.raises((ValueError, TypeError)):
            coordinate_label(0, rings)
