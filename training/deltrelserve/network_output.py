"""Bounded full-head inspection of one root; never modifies search behavior."""

from __future__ import annotations

from typing import Any, Mapping
import torch
from torch import Tensor

from .schemas import AnalyzeRequest, NetworkOutput


def network_output_payload(
    values: Mapping[str, Tensor | None],
    request: AnalyzeRequest,
    *,
    auxiliary_ready: bool,
    swap_recommended: bool,
) -> dict[str, Any]:
    """Preserve every output, including inapplicable future-turn diagnostics.

    Ownership classes are current/opponent/unclaimed. Count rows are current/
    opponent. The reply head's final swap slot is an unconditional model output;
    actual swapping only applies after an opening in a pie-rule game.
    """
    nodes = len(request.stones)
    shapes = {
        "policy": [nodes],
        "outcome": [2],
        "score_margin": [303],
        "ownership": [nodes, 3],
        "alive": [nodes],
        "soft_policy": [nodes],
        "opponent_reply": [nodes + 1],
        "second_stone": [nodes],
        "final_shores": [2, 51],
        "final_networks": [2, 26],
        "final_capes": [2, 6],
    }
    if set(values) != {f"{name}_logits" for name in shapes}:
        raise ValueError("model output head set is incompatible")
    absent = values["final_shores_logits"] is None
    empty = torch.tensor([stone == -1 for stone in request.stones], dtype=torch.bool)
    heads: dict[str, Any] = {}
    for index, (name, shape) in enumerate(shapes.items()):
        tensor = values[f"{name}_logits"]
        if index >= 6 and absent:
            if tensor is not None:
                raise ValueError("model auxiliary heads are incomplete")
            heads[name] = None
            continue
        if tensor is None or tuple(tensor.shape) != (1, *shape):
            raise ValueError(f"model {name} shape is incompatible")
        logits = tensor[0].detach().to(device="cpu", dtype=torch.float64)
        mask = torch.ones(shape, dtype=torch.bool)
        if name in {"policy", "soft_policy", "second_stone"}:
            mask = empty.clone()
        elif name == "opponent_reply":
            mask = torch.cat((empty, torch.ones(1, dtype=torch.bool)))
        elif name == "final_shores":
            mask = (torch.arange(51) <= 5 * request.rings).expand(2, -1)
        elif name == "final_networks":
            mask = (torch.arange(26) <= (5 * request.rings) // 2).expand(2, -1)
        if not bool(torch.isfinite(logits[mask]).all()):
            raise ValueError(f"model {name} contains nonfinite active logits")
        activation = "sigmoid" if name == "alive" else "softmax"
        probabilities = (
            logits.sigmoid()
            if activation == "sigmoid"
            else logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        )
        applicable = True
        if name == "second_stone":
            applicable = (
                request.mode == "double"
                and not request.opening
                and request.moves_left == 2
                and not swap_recommended
            )
        elif name == "opponent_reply":
            applicable = int(empty.sum()) > request.moves_left
        heads[name] = {
            "shape": shape,
            "activation": activation,
            "logits": [
                float(value) if active else None
                for value, active in zip(
                    logits.flatten().tolist(), mask.flatten().tolist(), strict=True
                )
            ],
            "probabilities": probabilities.flatten().tolist(),
            "mask": mask.flatten().tolist(),
            "applicable": applicable,
        }
    return NetworkOutput.model_validate(
        {
            "schema_version": 1,
            "perspective": request.to_move,
            "node_count": nodes,
            "auxiliary_status": "absent"
            if absent
            else "ready"
            if auxiliary_ready
            else "untrained",
            "heads": heads,
        }
    ).model_dump()
