"""Inference-only source/class message factoring; checkpoint parameters stay put.

The mean operator's message depends on the source node and one of three edge
classes, not its destination. Materialize that small table once, then gather and
reduce it without a [batch, nodes, degree, channels] edge tensor on CUDA.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as functional


def _reference_mean(
    table: Tensor, indices: Tensor, edge_types: Tensor, mask: Tensor
) -> Tensor:
    batch, nodes, classes, channels = table.shape
    if bool(
        (
            (indices < 0)
            | (indices >= nodes)
            | (edge_types < 0)
            | (edge_types >= classes)
        ).any()
    ):
        raise ValueError("source/class message indices are out of bounds")
    codes = indices * classes + edge_types
    flat = table.reshape(batch, nodes * classes, channels)
    selected = flat.gather(
        1, codes.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, channels)
    )
    selected = selected.reshape(batch, indices.shape[1], indices.shape[2], channels)
    weights = mask.unsqueeze(-1).to(table.dtype)
    return (selected * weights).sum(2) / weights.sum(2).clamp_min(1)


@torch.library.custom_op("deltreltrain::source_class_mean", mutates_args=())
def _source_class_mean(
    table: Tensor, indices: Tensor, edge_types: Tensor, mask: Tensor
) -> Tensor:
    if table.is_cuda and table.dtype in (torch.float16, torch.bfloat16, torch.float32):
        from .local_message_triton import source_class_mean_cuda

        return source_class_mean_cuda(table, indices, edge_types, mask)
    return _reference_mean(table, indices, edge_types, mask)


@_source_class_mean.register_fake
def _source_class_mean_fake(
    table: Tensor, indices: Tensor, edge_types: Tensor, mask: Tensor
) -> Tensor:
    return table.new_empty((table.shape[0], indices.shape[1], table.shape[-1]))


def source_class_aggregate(
    projected: Tensor,
    embedding: Tensor,
    indices: Tensor,
    edge_types: Tensor,
    mask: Tensor,
) -> Tensor:
    # The opaque consumer prevents Inductor from inlining this nonlinearity into
    # every destination edge. It runs classes*N times, not degree*N times.
    table = functional.silu(projected.unsqueeze(2) + embedding[None, None, :, :])
    return _source_class_mean(table, indices, edge_types, mask)
