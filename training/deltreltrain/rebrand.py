"""Explicit, lossless migration of the immediately preceding model identity.

Normal loaders remain strict. This opt-in boundary recognizes only the previous
numeric fingerprints; gameplay tensors and learned values do not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, inspect_checkpoint
from .contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    RULES_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)

# Exactly the pre-rebrand production contracts, never a permissive alias.
SOURCE_RULES_HASH = 0xA5D932B0EF8354E8
SOURCE_FEATURE_HASH = 0xCB0E1E89A6CE3540
_HEAD_BY_OUTPUT = {
    102: "final_shores_head",
    52: "final_networks_head",
    12: "final_capes_head",
}


def _parameter_renames(model: dict[str, Any]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    targets: set[str] = set()
    for name, tensor in model.items():
        if not name.startswith("final_") or not name.endswith("_head.weight"):
            continue
        if not isinstance(tensor, Tensor) or tensor.ndim != 2:
            raise ValueError("invalid final-count head weight")
        head = _HEAD_BY_OUTPUT.get(tensor.shape[0])
        if head is None or head in targets:
            raise ValueError("unrecognized or ambiguous final-count head")
        targets.add(head)
        source = name.removesuffix(".weight")
        bias = model.get(f"{source}.bias")
        if not isinstance(bias, Tensor) or bias.shape != (tensor.shape[0],):
            raise ValueError("invalid final-count head bias")
        for suffix in ("weight", "bias"):
            replacements[f"{source}.{suffix}"] = f"{head}.{suffix}"
        replacements[source.removesuffix("_head")] = head.removesuffix("_head")
    if targets and targets != set(_HEAD_BY_OUTPUT.values()):
        raise ValueError("incomplete final-count heads")
    return replacements


def _loss_renames(payload: dict[str, Any], names: dict[str, str]) -> None:
    config = payload.get("config", {})
    loss = config.get("loss", {}) if isinstance(config, dict) else {}
    if not isinstance(loss, dict):
        raise ValueError("checkpoint loss configuration must be a dictionary")
    targets = [head.removesuffix("_head") for head in _HEAD_BY_OUTPUT.values()]
    unknown = [
        key
        for key in loss
        if key.startswith("final_") and key not in names and key not in targets
    ]
    if not unknown:
        return
    # Older models without auxiliary heads can still carry the three default
    # disabled loss weights. Their zero values make this conversion unambiguous.
    if len(unknown) != 3 or any(loss[key] != 0 for key in unknown):
        raise ValueError("nonzero final-count losses require matching model heads")
    names.update(zip(unknown, targets, strict=True))


def _rewrite(value: Any, names: dict[str, str], package: str) -> Any:
    """Copy containers while retaining tensor storage and primitive values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            new_key = names.get(key, key) if isinstance(key, str) else key
            if new_key in result:
                raise ValueError("migration would overwrite a metadata key")
            if key == "format" and isinstance(item, str) and item.startswith(package):
                item = f"deltreltrain.{item.removeprefix(package)}"
            result[new_key] = _rewrite(item, names, package)
        return result
    if isinstance(value, list):
        return [_rewrite(item, names, package) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite(item, names, package) for item in value)
    if isinstance(value, str):
        return names.get(value, value)
    return value


def _routing_hash(routing: dict[str, Any], model: dict[str, Any]) -> str:
    groups = []
    for group in routing["groups"]:
        parameters = []
        for name in group["parameter_names"]:
            tensor = model.get(name)
            if not isinstance(tensor, Tensor):
                raise ValueError("optimizer refers to a missing model parameter")
            parameters.append(
                {"name": name, "shape": list(tensor.shape), "elements": tensor.numel()}
            )
        groups.append(
            {
                "name": group["name"],
                "algorithm": group["algorithm"],
                "weight_decay": float(group["weight_decay"]),
                "parameters": parameters,
            }
        )
    canonical = {
        key: routing[key]
        for key in (
            "schema_version",
            "requested_kind",
            "implementation",
            "fallback_used",
            "optimizer_config",
        )
    }
    canonical["groups"] = groups
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"sha256-{digest}"


def migrate_payload(payload: Any) -> dict[str, Any]:
    """Re-identify only the known production contract, without changing tensors."""
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    expected = {
        "version": CHECKPOINT_VERSION,
        "rules_hash": SOURCE_RULES_HASH,
        "rules_hash_wire": f"fnv1a64:{SOURCE_RULES_HASH:016x}",
        "feature_schema_hash": SOURCE_FEATURE_HASH,
        "model_schema_version": 3,
        "action_layout_version": 1,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("only the exact preceding production contract can migrate")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint model state is missing")
    source_format = payload.get("format")
    if not isinstance(source_format, str) or not source_format.endswith(".checkpoint"):
        raise ValueError("checkpoint format identity is invalid")
    source_package = source_format.removesuffix("checkpoint")
    renames = _parameter_renames(payload["model"])
    _loss_renames(payload, renames)
    routing = payload.get("optimizer_routing")
    if routing is not None:
        if not isinstance(routing, dict):
            raise ValueError("invalid optimizer routing")
        if _routing_hash(routing, payload["model"]) != routing.get("routing_hash"):
            raise ValueError("source optimizer routing checksum is invalid")
    converted = _rewrite(payload, renames, source_package)
    converted.update(
        format=CHECKPOINT_FORMAT,
        rules_schema=RULES_SCHEMA_ID,
        rules_hash=RULES_HASH,
        rules_hash_wire=RULES_HASH_WIRE,
        feature_schema_hash=FEATURE_SCHEMA_HASH,
        action_layout_schema=ACTION_LAYOUT_SCHEMA_ID,
    )
    if routing is not None:
        converted["optimizer_routing"]["routing_hash"] = _routing_hash(
            converted["optimizer_routing"], converted["model"]
        )
    extra = converted.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("checkpoint extra metadata must be a dictionary")
    extra["deltrel_identity_migration"] = {
        "source_rules_hash": f"fnv1a64:{SOURCE_RULES_HASH:016x}",
        "source_feature_hash": f"{SOURCE_FEATURE_HASH:016x}",
        "tensor_values_unchanged": True,
    }
    return converted


def migrate_checkpoint(source: Path, destination: Path) -> Path:
    """Validate and publish a new file; never overwrite the source or output."""
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("migration requires a new destination path")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    converted = migrate_payload(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pt") as file:
        torch.save(converted, file)
        file.flush()
        os.fsync(file.fileno())
        inspect_checkpoint(file.name)
        # Atomic publication that fails if another writer claimed the path.
        os.link(file.name, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(migrate_checkpoint(args.source, args.destination))


if __name__ == "__main__":
    main()
