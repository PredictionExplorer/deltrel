"""Add prediction heads without discarding the learned training state.

This is a one-way, explicitly requested architecture extension. Original tensors,
optimizer moments, scheduler clocks, EMA history and progress remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .gradient_clipping import GradientClipper
from .model import ModelConfig
from .optim import optimizer_checkpoint_contract

AUXILIARY_LOSSES = (
    "opponent_reply",
    "second_stone",
    "final_peries",
    "final_stars",
    "final_quarks",
)
AUXILIARY_PREFIXES = tuple(
    name + "."
    for name in (
        "opponent_reply_head",
        "opponent_reply_swap_head",
        "second_stone_head",
        "final_peries_head",
        "final_stars_head",
        "final_quarks_head",
    )
)


def is_auxiliary_parameter(name: str) -> bool:
    return name.startswith(AUXILIARY_PREFIXES)


def is_auxiliary_extension(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Only adding the optional heads is compatible; no trunk change is allowed."""
    from .checkpoint import normalize_model_config

    old = normalize_model_config(before)
    new = normalize_model_config(after)
    if old.pop("auxiliary_predictions") is not False:
        return False
    return new.pop("auxiliary_predictions") is True and old == new


def upgrade_checkpoint_payload(
    payload: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    gradient_clipper: GradientClipper | None,
    expected_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an extended payload; never mutate a source checkpoint or tensor."""
    from .checkpoint import normalize_model_config

    old_config = payload["config"]["model"]
    if normalize_model_config(old_config) == normalize_model_config(
        expected_model_config
    ):
        return payload
    if not is_auxiliary_extension(old_config, expected_model_config):
        raise ValueError("only an additive auxiliary prediction upgrade is supported")
    actual_config = getattr(model, "config", None)
    if not isinstance(actual_config, ModelConfig) or asdict(
        actual_config
    ) != normalize_model_config(expected_model_config):
        raise ValueError("auxiliary upgrade target model configuration differs")
    target = model.state_dict()
    original = payload["model"]
    added = set(target) - set(original)
    if (
        not added
        or set(original) - set(target)
        or any(not is_auxiliary_parameter(k) for k in added)
    ):
        raise ValueError("auxiliary upgrade changed existing model keys")
    if added != {k for k in target if is_auxiliary_parameter(k)}:
        raise ValueError("source checkpoint contains an incomplete auxiliary extension")
    for name, value in original.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target[name].shape
            or value.dtype != target[name].dtype
        ):
            raise ValueError(f"auxiliary upgrade changed existing model tensor: {name}")

    result = dict(payload)
    result["model"] = dict(original) | {k: target[k].detach().clone() for k in added}
    ema = payload["ema"]
    if not isinstance(ema, Mapping) or set(ema["shadow"]) != {
        k for k, v in original.items() if v.is_floating_point()
    }:
        raise ValueError("auxiliary upgrade requires complete original EMA weights")
    result["ema"] = dict(ema) | {
        "shadow": dict(ema["shadow"])
        | {
            k: target[k].detach().float().clone()
            for k in added
            if target[k].is_floating_point()
        }
    }

    if payload.get("optimizer") is not None:
        if optimizer is None:
            raise ValueError(
                "auxiliary training-state upgrade requires the target optimizer"
            )
        result["optimizer"], result["optimizer_routing"] = _extend_optimizer(
            payload["optimizer"], payload.get("optimizer_routing"), optimizer, added
        )
    clipping = payload.get("gradient_clipping")
    if clipping is not None:
        if gradient_clipper is None:
            raise ValueError("auxiliary upgrade requires the target gradient clipper")
        template = gradient_clipper.state_dict()
        old_rows = {r["name"]: r for r in clipping["parameters"]}
        new_rows = {r["name"]: r for r in template["parameters"]}
        if set(new_rows) - set(old_rows) != added or any(
            new_rows.get(k) != v for k, v in old_rows.items()
        ):
            raise ValueError(
                "auxiliary upgrade changed gradient clipping parameter history"
            )
        for key in ("format", "version", "config", "max_norm"):
            if clipping[key] != template[key]:
                raise ValueError(
                    "auxiliary upgrade changed gradient clipping configuration"
                )
        result["gradient_clipping"] = dict(clipping) | {
            "parameters": template["parameters"],
            "ema_norms": dict(clipping["ema_norms"])
            | {k: v for k, v in template["ema_norms"].items() if k in added},
        }
        gradient_clipper.validate_state_dict(result["gradient_clipping"])
    result["config"] = dict(payload["config"]) | {
        "model": normalize_model_config(expected_model_config)
    }
    result["extra"] = dict(payload["extra"]) | {
        "auxiliary_upgrade": {
            "schema_version": 1,
            "source_step": payload["step"],
            "added_parameters": sorted(added),
            "preserved_parameter_tensors": len(original),
            "preserved_optimizer_state_tensors": len(
                (payload.get("optimizer") or {}).get("state", {})
            ),
        }
    }
    return result


def _extend_optimizer(
    saved: dict[str, Any],
    source_contract: object,
    optimizer: torch.optim.Optimizer,
    added: set[str],
) -> tuple[dict[str, Any], dict[str, object]]:
    target_contract = optimizer_checkpoint_contract(optimizer)
    if not isinstance(source_contract, dict) or target_contract is None:
        raise ValueError("auxiliary upgrade needs named optimizer routing contracts")
    for key in (
        "schema_version",
        "requested_kind",
        "implementation",
        "fallback_used",
        "optimizer_config",
    ):
        if source_contract.get(key) != target_contract.get(key):
            raise ValueError("auxiliary upgrade changed optimizer configuration")
    old_routes = source_contract["groups"]
    new_routes = target_contract["groups"]
    target_state = optimizer.state_dict()
    if not isinstance(new_routes, list) or not (
        len(old_routes)
        == len(new_routes)
        == len(saved["param_groups"])
        == len(target_state["param_groups"])
    ):
        raise ValueError("auxiliary upgrade changed optimizer groups")
    state: dict[int, Any] = {}
    groups = []
    covered_old: set[int] = set()
    covered_new: set[str] = set()
    for old_route, new_route, old_group, new_group in zip(
        old_routes,
        new_routes,
        saved["param_groups"],
        target_state["param_groups"],
        strict=True,
    ):
        for key in ("name", "algorithm", "weight_decay"):
            if old_route[key] != new_route[key]:
                raise ValueError("auxiliary upgrade rerouted an optimizer group")
        old_names = old_route["parameter_names"]
        new_names = new_route["parameter_names"]
        if [n for n in new_names if n not in added] != old_names:
            raise ValueError("auxiliary upgrade rerouted existing optimizer parameters")
        old_ids = dict(zip(old_names, old_group["params"], strict=True))
        new_ids = dict(zip(new_names, new_group["params"], strict=True))
        if len(old_ids) != len(old_names) or len(new_ids) != len(new_names):
            raise ValueError("auxiliary upgrade found duplicate optimizer names")
        for name, old_id in old_ids.items():
            if old_id in covered_old:
                raise ValueError("auxiliary upgrade found duplicate optimizer ids")
            covered_old.add(old_id)
            if old_id in saved["state"]:
                state[new_ids[name]] = saved["state"][old_id]
        covered_new.update(set(new_names) - set(old_names))
        groups.append(deepcopy(old_group) | {"params": list(new_group["params"])})
    if set(saved["state"]) - covered_old or covered_new != added:
        raise ValueError("auxiliary upgrade optimizer coverage differs")
    return {"state": state, "param_groups": groups}, target_contract
