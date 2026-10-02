"""Bounded, read-only shared-trunk loss-gradient diagnostics on frozen rows."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math

import torch

from deltreltrain.config import ExperimentConfig
from deltreltrain.losses import compute_losses
from deltreltrain.replay import ReplayBatch


def gradient_geometry(vectors: dict[str, torch.Tensor]) -> dict[str, object]:
    """Report norms and pairwise cosines; an unavailable/zero head has no angle."""
    norms = {name: float(vector.norm()) for name, vector in vectors.items()}
    if any(not math.isfinite(value) for value in norms.values()):
        raise FloatingPointError("non-finite per-head gradient")
    cosines = {}
    names = list(vectors)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            denominator = norms[first] * norms[second]
            cosine = (
                float(torch.dot(vectors[first], vectors[second])) / denominator
                if denominator
                else None
            )
            cosines[f"{first}/{second}"] = (
                max(-1.0, min(1.0, cosine)) if cosine is not None else None
            )
    return {"weighted_gradient_norms": norms, "pairwise_cosines": cosines}


def per_head_gradient_conflicts(
    model: torch.nn.Module,
    batch: ReplayBatch,
    config: ExperimentConfig,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, object]:
    """No optimizer step or .grad mutation; at most 32 rows, one graph per call.

    Measure the shared representation, excluding output heads. Keeping detached
    gradient vectors on CPU bounds GPU storage independently of head count.
    Cosines are descriptive for this fixed batch, never an adoption criterion.
    """
    rows = int(batch.targets.policy.shape[0])
    if not 1 <= rows <= 32:
        raise ValueError("gradient diagnostic requires 1..32 frozen rows")
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "head" not in name
    ]
    if not named:
        raise ValueError("gradient diagnostic found no shared parameters")
    parameters = tuple(parameter for _, parameter in named)
    before_mode = model.training
    model.eval()
    try:
        moved = batch.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            output = model(*moved.inputs.model_args())
            losses = compute_losses(
                output,
                moved.targets,
                legal_action_mask=moved.inputs.legal_action_mask,
                node_mask=moved.inputs.node_mask,
                weights=config.loss,
                validate_targets=False,
            )
        weights = {
            field.name: getattr(config.loss, field.name)
            for field in fields(config.loss)
            if getattr(config.loss, field.name) > 0 and field.name in losses
        }
        vectors = {}
        for index, (name, weight) in enumerate(weights.items()):
            gradients = torch.autograd.grad(
                losses[name] * weight,
                parameters,
                retain_graph=index + 1 < len(weights),
                allow_unused=True,
            )
            vectors[name] = torch.cat(
                [
                    gradient.detach().float().cpu().reshape(-1)
                    if gradient is not None
                    else torch.zeros(parameter.numel())
                    for parameter, gradient in zip(parameters, gradients, strict=True)
                ]
            )
        geometry = gradient_geometry(vectors)
        return {
            "schema_version": 1,
            "rows": rows,
            "weights": "raw",
            "scope": "shared-parameters-excluding-output-heads",
            "parameter_names_sha256": hashlib.sha256(
                json.dumps([name for name, _ in named]).encode()
            ).hexdigest(),
            "parameter_elements": sum(parameter.numel() for parameter in parameters),
            "head_coefficients": weights,
            **geometry,
            "diagnostic_only": True,
        }
    finally:
        model.train(before_mode)
