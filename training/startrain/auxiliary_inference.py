"""Compact root-only auxiliary beliefs, kept out of search utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AuxiliaryPrediction:
    """All player pairs are ordered current player, then opponent."""

    final_peries: list[float]
    final_stars: list[float]
    final_quarks: list[float]
    quark_bonus_probability: list[float]
    opponent_reply_probabilities: list[float]
    second_stone_probabilities: list[float]


def pack_auxiliary_logits(output: Any, rows: int, nodes: int) -> Tensor:
    tensors = []
    for name, shape in (
        ("opponent_reply_logits", (rows, nodes + 1)),
        ("second_stone_logits", (rows, nodes)),
        ("final_peries_logits", (rows, 2, 51)),
        ("final_stars_logits", (rows, 2, 26)),
        ("final_quarks_logits", (rows, 2, 6)),
    ):
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"model {name} violates auxiliary inference schema")
        tensors.append(value.float().reshape(rows, -1))
    return torch.cat(tensors, dim=1)


def unpack_auxiliary_prediction(packed: bytes, nodes: int) -> AuxiliaryPrediction:
    # Own the array: cache bytes are immutable and must never become writable views.
    values = torch.from_numpy(np.frombuffer(packed, dtype=np.float32).copy())
    if values.numel() != 2 * nodes + 167:
        raise ValueError("cached auxiliary predictions have invalid shape")
    if torch.isnan(values).any() or torch.isposinf(values).any():
        raise ValueError("non-finite auxiliary predictions")
    reply, second, peries, stars, quarks = torch.split(
        values, [nodes + 1, nodes, 102, 52, 12]
    )

    def probabilities(logits: Tensor) -> Tensor:
        result = logits.softmax(dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("auxiliary prediction has no finite probability support")
        return result

    def expectation(logits: Tensor, width: int) -> list[float]:
        probs = probabilities(logits.reshape(2, width))
        return (probs * torch.arange(width)).sum(dim=-1).tolist()

    quark_probabilities = probabilities(quarks.reshape(2, 6))
    return AuxiliaryPrediction(
        final_peries=expectation(peries, 51),
        final_stars=expectation(stars, 26),
        final_quarks=(quark_probabilities * torch.arange(6)).sum(dim=-1).tolist(),
        quark_bonus_probability=quark_probabilities[:, 3:].sum(dim=-1).tolist(),
        opponent_reply_probabilities=probabilities(reply).tolist(),
        second_stone_probabilities=probabilities(second).tolist(),
    )
