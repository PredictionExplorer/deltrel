"""Frozen schema-v3 encoder of the previous lineage.

The lineage-transfer importer and the cross-schema arena encode the teacher
checkpoint's inputs with this module; production training uses
:mod:`deltreltrain.features`. The layout is exactly the fifteen node planes and
seventeen global scalars of ``deltreltrain/features/v3`` and must never change.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .contracts import LEGACY_FEATURE_SCHEMA_HASH, LEGACY_FEATURE_SCHEMA_VERSION
from .contracts import SCORE_MARGIN_MAX
from .features import DoubleDeltrelPosition, EncodedBatch, EncodedPosition, collate_encoded
from .scoring import EMPTY, score_position
from .topology import MAX_RINGS, get_topology

LEGACY_NODE_FEATURE_NAMES = (
    "empty",
    "current_stone",
    "opponent_stone",
    "owner_current",
    "owner_opponent",
    "owner_unclaimed",
    "alive_current",
    "alive_opponent",
    "is_shore",
    "is_cape",
    "ring_fraction",
    "arm_distance_fraction",
    "degree_fraction",
    "is_bridge",
    "legal",
)
LEGACY_GLOBAL_FEATURE_NAMES = (
    "rings_fraction",
    "occupancy_fraction",
    "current_stone_fraction",
    "opponent_stone_fraction",
    "moves_left_fraction",
    "opening",
    "terminal",
    "current_total_scaled",
    "opponent_total_scaled",
    "score_margin_scaled",
    "current_shores_fraction",
    "opponent_shores_fraction",
    "current_capes_fraction",
    "opponent_capes_fraction",
    "current_networks_fraction",
    "opponent_networks_fraction",
    "contested_shores_fraction",
)
LEGACY_NODE_FEATURE_DIM = len(LEGACY_NODE_FEATURE_NAMES)
LEGACY_GLOBAL_FEATURE_DIM = len(LEGACY_GLOBAL_FEATURE_NAMES)


def encode_legacy_position(
    position: DoubleDeltrelPosition,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedPosition:
    """Encode schema v3 from the six legacy semantic fields only.

    Variant, history, and context fields are ignored: the previous lineage
    never observed them. ``moves_left`` keeps its historical ``/ 2`` scale.
    """

    topology = get_topology(position.rings)
    stones = position.stones.detach().to(device="cpu", dtype=torch.int8)
    score = score_position(topology, stones)
    current = position.to_move
    opponent = 1 - current

    empty = stones == EMPTY
    current_stone = stones == current
    opponent_stone = stones == opponent
    owner = score.node_owner
    alive = score.alive_stone
    degrees = topology.neighbor_mask.sum(dim=1)
    ring = topology.ring_of.to(torch.float32)
    position_on_ring = topology.pos_of.to(torch.float32)
    arm_distance = torch.minimum(position_on_ring, ring - position_on_ring) / ring
    legal = empty & (not position.terminal)

    node_features = torch.stack(
        (
            empty,
            current_stone,
            opponent_stone,
            owner == current,
            owner == opponent,
            owner == -1,
            alive & current_stone,
            alive & opponent_stone,
            topology.is_shore,
            topology.is_cape,
            ring / float(position.rings),
            arm_distance,
            degrees.to(torch.float32) / float(topology.max_degree),
            topology.ring_of == 1,
            legal,
        ),
        dim=-1,
    ).to(dtype=dtype)

    occupied = topology.n - int(empty.sum())
    current_count = int(current_stone.sum())
    opponent_count = int(opponent_stone.sum())
    current_score = score.players[current]
    opponent_score = score.players[opponent]
    score_scale = float(SCORE_MARGIN_MAX)
    deltrel_scale = max(1.0, topology.shore_count / 2.0)
    global_features = torch.tensor(
        (
            position.rings / MAX_RINGS,
            occupied / topology.n,
            current_count / topology.n,
            opponent_count / topology.n,
            position.moves_left / 2.0,
            float(position.opening),
            float(position.terminal),
            current_score.total / score_scale,
            opponent_score.total / score_scale,
            (current_score.total - opponent_score.total) / score_scale,
            current_score.shores / topology.shore_count,
            opponent_score.shores / topology.shore_count,
            current_score.capes / 5.0,
            opponent_score.capes / 5.0,
            current_score.networks / deltrel_scale,
            opponent_score.networks / deltrel_scale,
            score.contested_shores / topology.shore_count,
        ),
        dtype=dtype,
    )

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return EncodedPosition(
        topology=topology,
        node_features=node_features.to(target_device),
        global_features=global_features.to(target_device),
        legal_node_mask=legal.to(target_device),
        score=score,
    )


def encode_legacy_batch(
    positions: Sequence[DoubleDeltrelPosition],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> EncodedBatch:
    return collate_encoded(
        [
            encode_legacy_position(position, dtype=dtype, device=device)
            for position in positions
        ],
        node_feature_dim=LEGACY_NODE_FEATURE_DIM,
    )


assert LEGACY_FEATURE_SCHEMA_VERSION == 3
assert LEGACY_FEATURE_SCHEMA_HASH == 0x49336C937D534865
