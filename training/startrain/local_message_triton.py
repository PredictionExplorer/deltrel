"""Bounded fused gather/mean for materialized source/class inference messages.

Imported lazily only on CUDA. No autograd registration: the model rejects use
of this execution path during training or gradient-enabled evaluation.
"""

from __future__ import annotations

import torch
from typing import Any, cast

# Triton ships with CUDA PyTorch, but is intentionally absent from CPU/macOS
# installations. This module itself is imported only by the CUDA implementation.
import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]


@triton.jit
def _mean_kernel(
    table,
    indices,
    edge_types,
    mask,
    output,
    N: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    TS2: tl.constexpr,
    TS3: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    ES0: tl.constexpr,
    ES1: tl.constexpr,
    ES2: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    MS2: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    batch, node = rows // N, rows % N
    total = tl.full((BLOCK_ROWS, BLOCK_CHANNELS), 0, tl.float32)
    count = tl.full((BLOCK_ROWS,), 0, tl.float32)
    # @triton.jit rewrites this DSL loop; static_range is not a Python iterator.
    for edge in tl.static_range(D):  # pyright: ignore[reportGeneralTypeIssues]
        source = tl.load(
            indices + batch * IS0 + node * IS1 + edge * IS2, rows < ROWS, 0
        )
        kind = tl.load(
            edge_types + batch * ES0 + node * ES1 + edge * ES2, rows < ROWS, 0
        )
        active = tl.load(
            mask + batch * MS0 + node * MS1 + edge * MS2, rows < ROWS, False
        )
        valid = (source >= 0) & (source < N) & (kind >= 0) & (kind < C)
        addresses = (
            table
            + batch[:, None] * TS0
            + source[:, None] * TS1
            + kind[:, None] * TS2
            + channels[None, :] * TS3
        )
        values = tl.load(
            addresses,
            (rows[:, None] < ROWS)
            & (channels[None, :] < K)
            & active[:, None]
            & valid[:, None],
            0,
        ).to(tl.float32)
        # Invalid active edges fail downstream finite-output checks without an
        # out-of-bounds device read. Normal input topology is validated upstream.
        total += tl.where(active[:, None] & ~valid[:, None], float("nan"), values)
        count += active.to(tl.float32)
    # Preserve the public sum dtype before division (also relevant to a model
    # explicitly converted to BF16, although production keeps FP32 weights).
    total = total.to(table.dtype.element_ty).to(tl.float32)
    result = total / tl.maximum(count[:, None], 1)
    tl.store(
        output + rows[:, None] * K + channels[None, :],
        result,
        (rows[:, None] < ROWS) & (channels[None, :] < K),
    )


def source_class_mean_cuda(
    table: torch.Tensor,
    indices: torch.Tensor,
    edge_types: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch, nodes, classes, channels = table.shape
    if (
        indices.shape != mask.shape
        or indices.shape != edge_types.shape
        or indices.shape[:2] != (batch, nodes)
    ):
        raise ValueError("source/class aggregation shapes disagree")
    output = table.new_empty((batch, nodes, channels))
    cast(Any, _mean_kernel)[
        (triton.cdiv(batch * nodes, 16), triton.cdiv(channels, 128))
    ](
        table,
        indices,
        edge_types,
        mask,
        output,
        nodes,
        classes,
        channels,
        indices.shape[2],
        batch * nodes,
        *table.stride(),
        *indices.stride(),
        *edge_types.stride(),
        *mask.stride(),
        BLOCK_ROWS=16,
        BLOCK_CHANNELS=128,
        num_warps=4,
    )
    return output
