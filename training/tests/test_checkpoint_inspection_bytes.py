from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from deltreltrain import checkpoint as checkpoints
from deltreltrain.config import GameConfig
from deltreltrain.model import ModelConfig


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    path = checkpoints.save_checkpoint(
        tmp_path / "tiny.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=checkpoints.ExponentialMovingAverage(model, decay=0.9),
        step=7,
        epoch=2,
        config={
            "model": asdict(
                ModelConfig(width=16, rrt_groups=1, attention_heads=4, kv_heads=1)
            ),
            "game": asdict(GameConfig()),
        },
        extra={"run_id": "tiny-run", "generation_family": "tiny-family"},
    )
    assert path.stat().st_size < 1024**2
    return path


def inspect_bytes(data: bytes, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "expected_bytes": len(data),
        "expected_sha256": hashlib.sha256(data).hexdigest(),
    }
    arguments.update(overrides)
    return checkpoints.inspect_checkpoint_bytes(data, **arguments)


def serialized(payload: Any) -> bytes:
    with io.BytesIO() as stream:
        torch.save(payload, stream)
        return stream.getvalue()


def test_real_bytes_and_path_metadata_agree(checkpoint_file: Path) -> None:
    raw = checkpoint_file.read_bytes()
    expected: dict[str, Any] = {
        "expected_model_config": {
            "width": 16,
            "rrt_groups": 1,
            "attention_heads": 4,
            "kv_heads": 1,
        },
        "expected_game_config": asdict(GameConfig()),
        "expected_run_id": "tiny-run",
        "expected_generation_family": "tiny-family",
    }
    metadata = inspect_bytes(raw, **expected)
    assert metadata == checkpoints.inspect_checkpoint(checkpoint_file, **expected)
    assert metadata["step"] == 7
    assert metadata["epoch"] == 2
    assert metadata["has_optimizer"] is True
    assert metadata["has_scheduler"] is True
    assert metadata["has_ema"] is True
    assert metadata["scheduler_step"] == 0
    assert metadata["ema_decay"] == 0.9


def test_bytes_use_one_fixed_cpu_load_without_path_reopen(
    checkpoint_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = checkpoint_file.read_bytes()
    original_load = torch.load
    calls: list[tuple[io.BytesIO, dict[str, Any], Any]] = []

    def load(source: io.BytesIO, **kwargs: Any) -> Any:
        assert type(source) is io.BytesIO
        assert source.getvalue() == raw
        assert kwargs == {"map_location": "cpu", "weights_only": True, "mmap": False}
        payload = original_load(source, **kwargs)
        assert all(value.device.type == "cpu" for value in payload["model"].values())
        calls.append((source, kwargs, payload))
        return payload

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("bytes inspection must not reopen a path")

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=load))
    monkeypatch.setattr(checkpoints, "Path", forbidden)
    monkeypatch.setattr(checkpoints, "verify_file", forbidden)
    metadata = inspect_bytes(raw)
    assert len(calls) == 1
    stream, _, payload = calls[0]
    assert stream.closed
    assert metadata["config"] is payload["config"]
    assert metadata["extra"] is payload["extra"]


class BytesSubclass(bytes):
    pass


@pytest.mark.parametrize(
    "data",
    [
        "tiny.pt",
        Path("tiny.pt"),
        bytearray(b"x"),
        memoryview(b"x"),
        BytesSubclass(b"x"),
        io.BytesIO(b"x"),
    ],
)
def test_only_exact_immutable_bytes_are_accepted(
    data: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid input reached a loader or path operation")

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=forbidden))
    monkeypatch.setattr(checkpoints, "Path", forbidden)
    monkeypatch.setattr(checkpoints, "verify_file", forbidden)
    with pytest.raises(TypeError, match="^checkpoint data must be immutable bytes$"):
        checkpoints.inspect_checkpoint_bytes(
            data, expected_bytes=1, expected_sha256="0" * 64
        )


@pytest.mark.parametrize("length", [None, True, 1.0, 0, -1])
def test_invalid_length_pin_refuses_before_loading(
    length: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid length pin reached deserialization")

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=forbidden))
    with pytest.raises(ValueError, match="positive integer"):
        inspect_bytes(b"x", expected_bytes=length)


@pytest.mark.parametrize(
    "digest", [None, True, b"0" * 64, "0" * 63, "A" * 64, "z" * 64]
)
def test_invalid_digest_pin_refuses_before_loading(
    digest: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid digest pin reached deserialization")

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=forbidden))
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        inspect_bytes(b"x", expected_sha256=digest)


@pytest.mark.parametrize("kind", ["length", "digest"])
def test_mismatched_binding_refuses_before_loading(
    checkpoint_file: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    raw = checkpoint_file.read_bytes()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("mismatched binding reached deserialization")

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=forbidden))
    if kind == "length":
        with pytest.raises(ValueError, match="^checkpoint byte length failed$"):
            inspect_bytes(raw, expected_bytes=len(raw) + 1)
    else:
        with pytest.raises(ValueError, match="^checkpoint SHA-256 failed$"):
            inspect_bytes(raw, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("format", "startrain.checkpoint", "not a deltreltrain checkpoint"),
        ("version", 999, "unsupported checkpoint version"),
        ("rules_schema", "wrong", "rules schema"),
        ("rules_hash", "wrong", "rules hash"),
        ("rules_hash_wire", "wrong", "rules hash identifier"),
        ("feature_schema_hash", "wrong", "feature schema hash"),
        ("action_layout_schema", "wrong", "action layout schema"),
        ("action_layout_version", 999, "action layout version"),
        ("model_schema_version", 999, "model schema"),
        ("model", [], "model state"),
        ("extra", [], "extra metadata"),
        ("optimizer_routing", [], "optimizer routing"),
        ("ema", {"version": 1, "shadow": []}, "EMA state"),
        ("step", True, "step is invalid"),
        ("epoch", 2.5, "epoch is invalid"),
        ("config", [], "configuration is missing"),
    ],
)
def test_repinned_invalid_payload_reaches_real_validator(
    checkpoint_file: Path, field: str, value: Any, reason: str
) -> None:
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    payload[field] = value
    raw = serialized(payload)
    with pytest.raises(ValueError, match=reason):
        inspect_bytes(raw)


@pytest.mark.parametrize(
    "expectation,reason",
    [
        ({"expected_model_config": {"width": 32}}, "model/feature"),
        ({"expected_game_config": {"mode": "classic"}}, "game/rules"),
        ({"expected_run_id": "wrong"}, "run_id"),
        ({"expected_generation_family": "wrong"}, "generation family"),
    ],
)
def test_real_expected_metadata_mismatches_refuse(
    checkpoint_file: Path, expectation: dict[str, Any], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        inspect_bytes(checkpoint_file.read_bytes(), **expectation)


def test_missing_required_payload_field_refuses(checkpoint_file: Path) -> None:
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    del payload["model"]
    with pytest.raises(ValueError, match="payload is incomplete"):
        inspect_bytes(serialized(payload))


@pytest.mark.parametrize(
    "option", ["map_location", "allow_auxiliary_upgrade", "pickle_module", "mmap"]
)
def test_bytes_api_has_no_loader_or_upgrade_overrides(
    checkpoint_file: Path, option: str
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        inspect_bytes(checkpoint_file.read_bytes(), **{option: None})


def test_correctly_pinned_truncated_archive_does_not_fall_back(
    checkpoint_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = checkpoint_file.read_bytes()
    original_load = torch.load
    calls = []

    def load(source: io.BytesIO, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_load(source, **kwargs)

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=load))
    with pytest.raises(RuntimeError, match="PytorchStreamReader"):
        inspect_bytes(raw[: len(raw) // 2])
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": False}]


@pytest.mark.parametrize("explicit_location", [False, True])
def test_path_loader_kwargs_order_and_metadata_aliasing_are_preserved(
    checkpoint_file: Path, monkeypatch: pytest.MonkeyPatch, explicit_location: bool
) -> None:
    raw = checkpoint_file.read_bytes()
    original_load, original_verify = torch.load, checkpoints.verify_file
    events = []
    loaded = []
    location: str | torch.device = torch.device("cpu") if explicit_location else "cpu"

    def verify(*args: Any, **kwargs: Any) -> None:
        events.append("verify")
        original_verify(*args, **kwargs)

    def load(source: Path, **kwargs: Any) -> Any:
        events.append("load")
        assert source == checkpoint_file
        assert kwargs == {"map_location": location, "weights_only": True}
        payload = original_load(source, **kwargs)
        loaded.append(payload)
        return payload

    monkeypatch.setattr(checkpoints, "torch", SimpleNamespace(load=load))
    monkeypatch.setattr(checkpoints, "verify_file", verify)
    arguments: dict[str, Any] = {"map_location": location} if explicit_location else {}
    result = checkpoints.inspect_checkpoint(
        checkpoint_file,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_bytes=len(raw),
        **arguments,
    )
    assert events == ["verify", "load"]
    assert result["config"] is loaded[0]["config"]
    assert result["extra"] is loaded[0]["extra"]


def test_path_auxiliary_dual_fault_keeps_common_validation_first(
    checkpoint_file: Path,
) -> None:
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    payload["version"] = 999
    checkpoint_file.write_bytes(serialized(payload))
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        checkpoints.inspect_checkpoint(checkpoint_file, allow_auxiliary_upgrade=True)


def test_metadata_does_not_claim_model_state_dict_loadability(
    checkpoint_file: Path,
) -> None:
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    payload["model"] = {}
    assert inspect_bytes(serialized(payload))["step"] == 7
