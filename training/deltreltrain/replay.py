"""Validated schema-v5 replay samples and pickle-free heterogeneous shards.

Schema v5 stores the full rules-v3 semantic key (variant, swap state, retained
placement history), the evaluation context (``history_known``, ``pda``), and
optional teacher soft targets attached by the lineage-transfer importer. The
previous lineage's schema-v4 shards can be decoded through an upgrade adapter
(standard variant, unknown history) by that importer only; committed replay is
always v5.
"""

from __future__ import annotations

import json
import numbers
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import relocate_sample_actions
from .contracts import (
    ACTION_LAYOUT_VERSION,
    ALL_TARGETS,
    FEATURE_SCHEMA_HASH,
    LEGACY_FEATURE_SCHEMA_HASH,
    LEGACY_RULES_HASH,
    LEGACY_RULES_HASH_WIRE,
    LEGACY_RULES_SCHEMA_ID,
    MAX_HANDICAP,
    MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    MODE_INDEX,
    MODES,
    OUTCOME_LOSS,
    OUTCOME_WIN,
    RULES_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
    SOFT_POLICY_TEMPERATURE,
    TARGET_ALIVE,
    TARGET_OUTCOME,
    TARGET_OWNERSHIP,
    TARGET_POLICY,
    TARGET_SCORE_MARGIN,
    TARGET_SOFT_POLICY,
    TARGET_TEACHER,
)
from .features import (
    DoubleDeltrelPosition,
    EncodedBatch,
    collate_encoded,
    encode_position,
    variant_label,
    variant_segment,
)
from .losses import TrainingTargets
from .policy_batch_metrics import PolicyBatchMetrics
from .runtime import validate_identifier
from .scoring import ScoreResult
from .symmetry import D5Transform
from .topology import get_topology

REPLAY_SCHEMA_VERSION = 5
LEGACY_REPLAY_SCHEMA_VERSION = 4
REPLAY_SHARD_FORMAT = "deltreltrain.replay.npz"
MISSING_OUTCOME = -1
MISSING_OWNERSHIP = -100
MISSING_ALIVE = 255
SCORE_MARGIN_BINS = SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1
LEGACY_UPGRADE_PROVENANCE = "upgraded=rules-v2-features-v3"
ClinchAuxiliaryTargets = Literal["synthetic", "outcome_only"]

_LEGACY_SAMPLE_ARRAY_NAMES = (
    "rings",
    "node_offsets",
    "stones",
    "to_move",
    "moves_left",
    "opening",
    "terminal",
    "policy",
    "soft_policy",
    "target_mask",
    "outcome",
    "final_scores",
    "final_capes",
    "final_ownership",
    "final_alive",
    "search_provenance",
    "policy_provenance",
    "run_id",
    "generation_family",
    "actor_id",
    "generation",
    "game_id",
    "ply",
    "model_identity",
    "soft_policy_temperature",
    "rules_hash",
    "feature_schema_hash",
    "weight",
    "policy_weight",
)
_VARIANT_SAMPLE_ARRAY_NAMES = (
    "mode",
    "handicap",
    "pie",
    "swap_available",
    "swapped",
    "history_known",
    "pda",
    "history_flags",
    "teacher_policy",
    "teacher_outcome",
    "teacher_score_margin",
)
_REPLAY_SAMPLE_ARRAY_NAMES = _LEGACY_SAMPLE_ARRAY_NAMES + _VARIANT_SAMPLE_ARRAY_NAMES
REPLAY_AUXILIARY_TARGETS_VERSION = 1
_AUXILIARY_SAMPLE_ARRAY_NAMES = (
    "opponent_reply",
    "opponent_reply_swap",
    "opponent_reply_ply",
    "second_stone",
    "second_stone_ply",
    "final_shores",
    "final_networks",
)


@lru_cache(maxsize=4096)
def _final_components(
    rings: int,
    ownership_bytes: bytes,
    alive_bytes: bytes | None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Decode once per distinct game ending, never walk graphs in collation."""
    topology = get_topology(rings)
    owners = np.frombuffer(ownership_bytes, dtype=np.int8)
    perimeter_owners = owners[topology.is_shore.numpy()]
    shores = (
        int(np.count_nonzero(perimeter_owners == 0)),
        int(np.count_nonzero(perimeter_owners == 1)),
    )
    if alive_bytes is None:
        return shores, None
    alive = np.frombuffer(alive_bytes, dtype=np.uint8).astype(bool)
    offsets, neighbors = topology.adjacency_offsets.numpy(), topology.adjacency.numpy()
    visited = np.zeros(topology.n, dtype=bool)
    networks = [0, 0]
    for start in np.flatnonzero(alive):
        color = int(owners[start])
        if visited[start] or color not in (0, 1):
            continue
        networks[color] += 1
        visited[start] = True
        stack = [int(start)]
        while stack:
            node = stack.pop()
            for neighbor in neighbors[offsets[node] : offsets[node + 1]]:
                if (
                    alive[neighbor]
                    and not visited[neighbor]
                    and owners[neighbor] == color
                ):
                    visited[neighbor] = True
                    stack.append(int(neighbor))
    return shores, (networks[0], networks[1])


class ReplaySchemaError(ValueError):
    pass


def _checked_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ReplaySchemaError(f"{name} must be an integer before conversion")
    return int(value)


def _checked_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ReplaySchemaError(f"{name} must be bool")
    return bool(value)


def _integer_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ReplaySchemaError(f"{name} must contain integers before conversion")
    if array.shape != shape:
        raise ReplaySchemaError(f"{name} must have shape {shape}")
    limits = np.iinfo(dtype)
    if array.size and (np.min(array) < limits.min or np.max(array) > limits.max):
        raise ReplaySchemaError(f"{name} cannot be represented as {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def _float_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ReplaySchemaError(f"{name} must be numeric")
    if array.shape != shape:
        raise ReplaySchemaError(f"{name} must have shape {shape}")
    array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ReplaySchemaError(f"{name} must be finite and non-negative")
    return array


def _distribution(
    name: str,
    value: object | None,
    *,
    shape: tuple[int, ...],
    available: bool,
) -> np.ndarray:
    """Validate a stored teacher distribution (float16, unit legal mass)."""

    if value is None:
        if available:
            raise ReplaySchemaError(f"{name} is required when teacher targets are set")
        return np.zeros(shape, dtype=np.float16)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ReplaySchemaError(f"{name} must be numeric")
    if array.shape != shape:
        raise ReplaySchemaError(f"{name} must have shape {shape}")
    array = np.ascontiguousarray(array, dtype=np.float16)
    if not np.isfinite(array.astype(np.float32)).all() or (array < 0).any():
        raise ReplaySchemaError(f"{name} must be finite and non-negative")
    if not available:
        if np.any(array != 0):
            raise ReplaySchemaError(f"unavailable {name} must be all zero")
    else:
        mass = float(array.astype(np.float64).sum())
        if not np.isclose(mass, 1.0, atol=5e-3, rtol=0.0):
            raise ReplaySchemaError(f"{name} must sum to one")
    return array


def katago_soft_policy_target(
    policy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float = SOFT_POLICY_TEMPERATURE,
) -> np.ndarray:
    """KataGo auxiliary target: normalize ``policy ** (1 / T)`` at T=4."""

    if temperature != SOFT_POLICY_TEMPERATURE:
        raise ReplaySchemaError("soft-policy temperature must be exactly 4")
    values = np.where(legal_mask, np.asarray(policy, dtype=np.float64), 0.0)
    values = np.power(values, 1.0 / temperature)
    mass = float(values.sum())
    if mass <= 0:
        raise ReplaySchemaError("soft-policy source has no legal mass")
    return (values / mass).astype(np.float32)


@dataclass(slots=True)
class ReplaySample:
    rings: int
    stones: np.ndarray
    to_move: int
    moves_left: int
    opening: bool
    terminal: bool
    policy: np.ndarray
    soft_policy: np.ndarray
    target_mask: int
    outcome: int
    final_scores: np.ndarray
    final_capes: np.ndarray
    final_ownership: np.ndarray
    final_alive: np.ndarray
    search_provenance: str
    policy_provenance: str
    run_id: str = "manual"
    generation_family: str = "manual"
    actor_id: str = "manual"
    generation: int = 0
    game_id: str = field(default_factory=lambda: f"manual-{uuid.uuid4().hex}")
    ply: int = 0
    model_identity: str = "manual"
    soft_policy_temperature: float = SOFT_POLICY_TEMPERATURE
    rules_hash: int = RULES_HASH
    feature_schema_hash: int = FEATURE_SCHEMA_HASH
    weight: float = 1.0
    policy_weight: float = 1.0
    schema_version: int = REPLAY_SCHEMA_VERSION
    # Rules v3: variant, swap state, retained history, and evaluation context.
    mode: str = "double"
    handicap: int = 1
    pie: bool = False
    swap_available: bool = False
    swapped: bool = False
    history_known: bool = True
    pda: int = 0
    history_flags: np.ndarray | None = None
    # Teacher soft targets (lineage transfer); present iff TARGET_TEACHER is set.
    teacher_policy: np.ndarray | None = None
    teacher_outcome: np.ndarray | None = None
    teacher_score_margin: np.ndarray | None = None
    # Optional schema-v5 capability; policy labels refer to later observed
    # decisions, with swap occupying the final opponent-reply slot.
    opponent_reply: np.ndarray | None = None
    opponent_reply_ply: int = -1
    second_stone: np.ndarray | None = None
    second_stone_ply: int = -1
    # Fixed player order in replay; collation changes to current/opponent order.
    final_shores: np.ndarray | None = None
    final_networks: np.ndarray | None = None

    def __post_init__(self) -> None:
        schema_version = _checked_int("schema_version", self.schema_version)
        if schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplaySchemaError(f"sample schema {schema_version} is unsupported")
        rings = _checked_int("rings", self.rings)
        topology = get_topology(rings)
        stones = _integer_array(
            "stones", self.stones, shape=(topology.n,), dtype=np.dtype(np.int8)
        )
        if not np.isin(stones, (-1, 0, 1)).all():
            raise ReplaySchemaError("stones must contain only -1, 0, or 1")
        to_move = _checked_int("to_move", self.to_move)
        moves_left = _checked_int("moves_left", self.moves_left)
        opening = _checked_bool("opening", self.opening)
        terminal = _checked_bool("terminal", self.terminal)
        mode = self.mode
        if isinstance(mode, (np.integer, int)) and not isinstance(mode, bool):
            index = int(mode)
            if index < 0 or index >= len(MODES):
                raise ReplaySchemaError("mode index is invalid")
            mode = MODES[index]
        if not isinstance(mode, str) or mode not in MODES:
            raise ReplaySchemaError("mode must be classic or double")
        handicap = _checked_int("handicap", self.handicap)
        if not 1 <= handicap <= MAX_HANDICAP:
            raise ReplaySchemaError(f"handicap must be in 1..{MAX_HANDICAP}")
        pie = _checked_bool("pie", self.pie)
        swap_available = _checked_bool("swap_available", self.swap_available)
        swapped = _checked_bool("swapped", self.swapped)
        history_known = _checked_bool("history_known", self.history_known)
        pda = _checked_int("pda", self.pda)
        if abs(pda) > MAX_PLAYOUT_DOUBLING_ADVANTAGE:
            raise ReplaySchemaError("pda is outside the supported range")
        if self.history_flags is None:
            history_flags = np.zeros(topology.n, dtype=np.uint8)
        else:
            history_flags = _integer_array(
                "history_flags",
                self.history_flags,
                shape=(topology.n,),
                dtype=np.dtype(np.uint8),
            )

        # Reuse the exact semantic-key validator before storing narrowed dtypes.
        try:
            position = DoubleDeltrelPosition.from_sequence(
                rings=rings,
                stones=stones,
                to_move=to_move,
                moves_left=moves_left,
                opening=opening,
                terminal=terminal,
                mode=mode,
                handicap=handicap,
                pie=pie,
                swap_available=swap_available,
                swapped=swapped,
                history_flags=history_flags,
                history_known=history_known,
                pda=pda,
            )
        except (TypeError, ValueError) as exc:
            raise ReplaySchemaError(str(exc)) from exc
        # The opening derives its history from the stones; store the derived
        # flags so shards are self-consistent.
        history_flags = position.history_flags().numpy().astype(np.uint8)

        target_mask = _checked_int("target_mask", self.target_mask)
        if target_mask < 0 or target_mask & ~(ALL_TARGETS | TARGET_TEACHER):
            raise ReplaySchemaError("target_mask contains unknown bits")
        policy = _float_array("policy", self.policy, shape=(topology.n,))
        soft_policy = _float_array("soft_policy", self.soft_policy, shape=(topology.n,))
        outcome = _checked_int("outcome", self.outcome)
        final_scores = _integer_array(
            "final_scores", self.final_scores, shape=(2,), dtype=np.dtype(np.int16)
        )
        final_capes = _integer_array(
            "final_capes", self.final_capes, shape=(2,), dtype=np.dtype(np.int8)
        )
        final_ownership = _integer_array(
            "final_ownership",
            self.final_ownership,
            shape=(topology.n,),
            dtype=np.dtype(np.int8),
        )
        final_alive = _integer_array(
            "final_alive",
            self.final_alive,
            shape=(topology.n,),
            dtype=np.dtype(np.uint8),
        )
        has_teacher = bool(target_mask & TARGET_TEACHER)
        teacher_policy = _distribution(
            "teacher_policy",
            self.teacher_policy,
            shape=(topology.n,),
            available=has_teacher and not terminal,
        )
        teacher_outcome = _distribution(
            "teacher_outcome",
            self.teacher_outcome,
            shape=(2,),
            available=has_teacher,
        )
        teacher_score_margin = _distribution(
            "teacher_score_margin",
            self.teacher_score_margin,
            shape=(SCORE_MARGIN_BINS,),
            available=has_teacher,
        )

        if not isinstance(self.search_provenance, str) or not self.search_provenance:
            raise ReplaySchemaError("search_provenance must be a non-empty string")
        if not isinstance(self.policy_provenance, str) or not self.policy_provenance:
            raise ReplaySchemaError("policy_provenance must be a non-empty string")
        try:
            run_id = validate_identifier("run_id", self.run_id)
            generation_family = validate_identifier(
                "generation_family", self.generation_family
            )
            actor_id = validate_identifier("actor_id", self.actor_id)
            game_id = validate_identifier("game_id", self.game_id)
            model_identity = validate_identifier("model_identity", self.model_identity)
        except ValueError as exc:
            raise ReplaySchemaError(str(exc)) from exc
        generation = _checked_int("generation", self.generation)
        ply = _checked_int("ply", self.ply)
        if generation < 0 or ply < 0:
            raise ReplaySchemaError("generation and ply must be non-negative")
        if float(self.soft_policy_temperature) != SOFT_POLICY_TEMPERATURE:
            raise ReplaySchemaError("soft_policy_temperature must be exactly 4")
        rules_hash = _checked_int("rules_hash", self.rules_hash)
        feature_hash = _checked_int("feature_schema_hash", self.feature_schema_hash)
        if rules_hash != RULES_HASH:
            raise ReplaySchemaError("rules hash does not match the Rust contract")
        if feature_hash != FEATURE_SCHEMA_HASH:
            raise ReplaySchemaError("feature schema hash is incompatible")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ReplaySchemaError("sample weight must be finite and positive")
        policy_weight = float(self.policy_weight)
        if not np.isfinite(policy_weight) or policy_weight < 0:
            raise ReplaySchemaError(
                "sample policy_weight must be finite and non-negative"
            )

        legal = np.zeros(topology.n, dtype=np.bool_)
        if not terminal:
            legal[:] = stones == -1
        self._validate_policy(
            "policy", policy, legal, bool(target_mask & TARGET_POLICY)
        )
        self._validate_policy(
            "soft_policy",
            soft_policy,
            legal,
            bool(target_mask & TARGET_SOFT_POLICY),
        )
        if has_teacher and not terminal:
            if np.any(teacher_policy[~legal].astype(np.float32) > 1e-6):
                raise ReplaySchemaError(
                    "teacher_policy has support on an illegal action"
                )
        if target_mask & TARGET_SOFT_POLICY:
            if not target_mask & TARGET_POLICY:
                raise ReplaySchemaError("soft policy requires a policy target")
            expected = katago_soft_policy_target(policy, legal)
            if not np.allclose(soft_policy, expected, atol=2e-6, rtol=2e-6):
                raise ReplaySchemaError(
                    "soft policy is not the T=4 exponent-1/4 target"
                )
        if terminal and target_mask & (TARGET_POLICY | TARGET_SOFT_POLICY):
            raise ReplaySchemaError("terminal samples cannot carry decision policies")

        if target_mask & (TARGET_OUTCOME | TARGET_SCORE_MARGIN):
            if outcome not in (OUTCOME_LOSS, OUTCOME_WIN):
                raise ReplaySchemaError("available outcome must be loss=0 or win=1")
            if not np.logical_and(final_capes >= 0, final_capes <= 5).all():
                raise ReplaySchemaError("final capes must be in 0..5")
            expected_leader = _leader_from_scores(final_scores, final_capes)
            if expected_leader == -1:
                raise ReplaySchemaError("terminal replay result cannot be tied")
            expected_outcome = (
                OUTCOME_WIN if expected_leader == to_move else OUTCOME_LOSS
            )
            if outcome != expected_outcome:
                raise ReplaySchemaError(
                    "outcome disagrees with totals and cape tiebreak"
                )
        elif outcome != MISSING_OUTCOME:
            raise ReplaySchemaError("unavailable final result must use outcome=-1")
        if terminal and not target_mask & TARGET_OUTCOME:
            raise ReplaySchemaError("terminal replay samples require a binary outcome")

        if target_mask & TARGET_SCORE_MARGIN:
            margin = int(final_scores[to_move]) - int(final_scores[1 - to_move])
            if not SCORE_MARGIN_MIN <= margin <= SCORE_MARGIN_MAX:
                raise ReplaySchemaError("score margin is outside [-151, 151]")
        if target_mask & TARGET_OWNERSHIP:
            if not np.isin(final_ownership, (-1, 0, 1)).all():
                raise ReplaySchemaError("final ownership must contain -1, 0, or 1")
        elif not np.all(final_ownership == MISSING_OWNERSHIP):
            raise ReplaySchemaError("missing ownership must use -100")
        if target_mask & TARGET_ALIVE:
            if not np.isin(final_alive, (0, 1)).all():
                raise ReplaySchemaError("final alive target must be binary")
        elif not np.all(final_alive == MISSING_ALIVE):
            raise ReplaySchemaError("missing alive target must use 255")

        for name, size in (
            ("opponent_reply", topology.n + 1),
            ("second_stone", topology.n),
        ):
            value = getattr(self, name)
            destination = _checked_int(f"{name}_ply", getattr(self, f"{name}_ply"))
            setattr(self, f"{name}_ply", destination)
            if value is None:
                if destination != -1:
                    raise ReplaySchemaError(
                        f"missing {name} requires destination ply=-1"
                    )
                continue
            value = _float_array(name, value, shape=(size,)).copy()
            if terminal or destination <= ply:
                raise ReplaySchemaError(
                    f"{name} must refer to a later observed decision"
                )
            if not np.isclose(float(value.sum()), 1.0, atol=2e-6, rtol=2e-6):
                raise ReplaySchemaError(f"{name} must sum to one")
            if np.any(value[: topology.n][~legal] > 1e-6):
                raise ReplaySchemaError(f"{name} has support on an occupied node")
            if name == "second_stone" and (
                mode != "double" or opening or moves_left != 2
            ):
                raise ReplaySchemaError(
                    "second_stone requires the first placement of a regular double turn"
                )
            if (
                name == "opponent_reply"
                and value[-1] > 0
                and not (pie and opening and handicap == 1 and to_move == 0)
            ):
                raise ReplaySchemaError(
                    "future swap reply requires the pie opening turn"
                )
            setattr(self, name, value)
            setattr(self, f"{name}_ply", destination)

        derived_shores: tuple[int, int] | None = None
        derived_networks: tuple[int, int] | None = None
        if target_mask & TARGET_OWNERSHIP:
            derived_shores, derived_networks = _final_components(
                rings,
                final_ownership.tobytes(),
                final_alive.tobytes() if target_mask & TARGET_ALIVE else None,
            )
        for name, derived, maximum in (
            ("final_shores", derived_shores, topology.shore_count),
            ("final_networks", derived_networks, topology.shore_count // 2),
        ):
            if derived is None:
                # These are cached views of spatial targets. Existing callers
                # remove those targets with dataclasses.replace; carrying the
                # old cache must not restore a masked target or reject the row.
                setattr(self, name, None)
                continue
            supplied = getattr(self, name)
            if supplied is not None:
                supplied = _integer_array(
                    name, supplied, shape=(2,), dtype=np.dtype(np.int8)
                )
                if not np.array_equal(supplied, derived):
                    raise ReplaySchemaError(
                        f"{name} disagrees with final spatial targets"
                    )
            if derived is not None and any(
                value < 0 or value > maximum for value in derived
            ):
                raise ReplaySchemaError(f"{name} exceeds the board's feasible count")
            setattr(
                self,
                name,
                np.asarray(derived, dtype=np.int8) if derived is not None else None,
            )

        self.rings = rings
        self.stones = stones
        self.to_move = to_move
        self.moves_left = moves_left
        self.opening = opening
        self.terminal = terminal
        self.policy = policy
        self.soft_policy = soft_policy
        self.target_mask = target_mask
        self.outcome = outcome
        self.final_scores = final_scores
        self.final_capes = final_capes
        self.final_ownership = final_ownership
        self.final_alive = final_alive
        self.soft_policy_temperature = SOFT_POLICY_TEMPERATURE
        self.rules_hash = rules_hash
        self.feature_schema_hash = feature_hash
        self.weight = weight
        self.policy_weight = policy_weight
        self.schema_version = schema_version
        self.run_id = run_id
        self.generation_family = generation_family
        self.actor_id = actor_id
        self.generation = generation
        self.game_id = game_id
        self.ply = ply
        self.model_identity = model_identity
        self.mode = mode
        self.handicap = handicap
        self.pie = pie
        self.swap_available = swap_available
        self.swapped = swapped
        self.history_known = history_known
        self.pda = pda
        self.history_flags = history_flags
        self.teacher_policy = teacher_policy
        self.teacher_outcome = teacher_outcome
        self.teacher_score_margin = teacher_score_margin

    @staticmethod
    def _validate_policy(
        name: str,
        values: np.ndarray,
        legal: np.ndarray,
        available: bool,
    ) -> None:
        if not available:
            if np.any(values != 0):
                raise ReplaySchemaError(f"unavailable {name} must be all zero")
            return
        if np.any(values[~legal] > 1e-8):
            raise ReplaySchemaError(f"{name} has support on an illegal action")
        mass = float(values[legal].sum())
        if not np.isclose(mass, 1.0, atol=1e-5, rtol=1e-5):
            raise ReplaySchemaError(f"{name} legal mass must sum to one")

    @classmethod
    def from_position(
        cls,
        position: DoubleDeltrelPosition,
        *,
        policy: np.ndarray | None,
        final_score: ScoreResult | None,
        search_provenance: str,
        policy_provenance: str,
        include_spatial_targets: bool = True,
        clinch_auxiliary_targets: ClinchAuxiliaryTargets = "synthetic",
        weight: float = 1.0,
        policy_weight: float = 1.0,
        run_id: str = "manual",
        generation_family: str = "manual",
        actor_id: str = "manual",
        generation: int = 0,
        game_id: str | None = None,
        ply: int = 0,
        model_identity: str = "manual",
        teacher_policy: np.ndarray | None = None,
        teacher_outcome: np.ndarray | None = None,
        teacher_score_margin: np.ndarray | None = None,
        opponent_reply: np.ndarray | None = None,
        opponent_reply_ply: int = -1,
        second_stone: np.ndarray | None = None,
        second_stone_ply: int = -1,
    ) -> "ReplaySample":
        if clinch_auxiliary_targets not in ("synthetic", "outcome_only"):
            raise ReplaySchemaError(
                "clinch_auxiliary_targets must be synthetic or outcome_only"
            )
        topology = get_topology(position.rings)
        target_mask = 0
        if policy is None:
            policy_array = np.zeros(topology.n, dtype=np.float32)
            soft_policy = np.zeros_like(policy_array)
        else:
            if position.terminal:
                raise ReplaySchemaError("terminal positions cannot have policy targets")
            policy_array = np.asarray(policy, dtype=np.float32)
            legal = position.stones.detach().cpu().numpy() == -1
            soft_policy = katago_soft_policy_target(policy_array, legal)
            target_mask |= TARGET_POLICY | TARGET_SOFT_POLICY

        if final_score is None:
            outcome = MISSING_OUTCOME
            final_scores = np.zeros(2, dtype=np.int16)
            final_capes = np.zeros(2, dtype=np.int8)
            final_ownership = np.full(topology.n, MISSING_OWNERSHIP, dtype=np.int8)
            final_alive = np.full(topology.n, MISSING_ALIVE, dtype=np.uint8)
        else:
            if final_score.leader not in (0, 1):
                raise ReplaySchemaError("final score cannot be tied")
            outcome = (
                OUTCOME_WIN if final_score.leader == position.to_move else OUTCOME_LOSS
            )
            final_scores = np.asarray(
                [player.total for player in final_score.players], dtype=np.int16
            )
            final_capes = np.asarray(
                [player.capes for player in final_score.players], dtype=np.int8
            )
            target_mask |= TARGET_OUTCOME
            if clinch_auxiliary_targets == "synthetic":
                target_mask |= TARGET_SCORE_MARGIN
            if clinch_auxiliary_targets == "synthetic" and include_spatial_targets:
                final_ownership = final_score.node_owner.numpy()
                final_alive = final_score.alive_stone.numpy().astype(np.uint8)
                target_mask |= TARGET_OWNERSHIP | TARGET_ALIVE
            else:
                final_ownership = np.full(topology.n, MISSING_OWNERSHIP, dtype=np.int8)
                final_alive = np.full(topology.n, MISSING_ALIVE, dtype=np.uint8)
        if teacher_outcome is not None or teacher_score_margin is not None:
            target_mask |= TARGET_TEACHER
        return cls(
            rings=position.rings,
            stones=position.stones.detach().cpu().numpy(),
            to_move=position.to_move,
            moves_left=position.moves_left,
            opening=position.opening,
            terminal=position.terminal,
            policy=policy_array,
            soft_policy=soft_policy,
            target_mask=target_mask,
            outcome=outcome,
            final_scores=final_scores,
            final_capes=final_capes,
            final_ownership=final_ownership,
            final_alive=final_alive,
            search_provenance=search_provenance,
            policy_provenance=policy_provenance,
            run_id=run_id,
            generation_family=generation_family,
            actor_id=actor_id,
            generation=generation,
            game_id=game_id or f"manual-{uuid.uuid4().hex}",
            ply=ply,
            model_identity=model_identity,
            weight=weight,
            policy_weight=policy_weight,
            mode=position.mode,
            handicap=position.handicap,
            pie=position.pie,
            swap_available=position.swap_available,
            swapped=position.swapped,
            history_known=position.history_known,
            pda=position.pda,
            history_flags=position.history_flags().numpy(),
            teacher_policy=teacher_policy,
            teacher_outcome=teacher_outcome,
            teacher_score_margin=teacher_score_margin,
            opponent_reply=opponent_reply,
            opponent_reply_ply=opponent_reply_ply,
            second_stone=second_stone,
            second_stone_ply=second_stone_ply,
        )

    def to_position(self) -> DoubleDeltrelPosition:
        return DoubleDeltrelPosition.from_sequence(
            rings=self.rings,
            stones=self.stones,
            to_move=self.to_move,
            moves_left=self.moves_left,
            opening=self.opening,
            terminal=self.terminal,
            mode=self.mode,
            handicap=self.handicap,
            pie=self.pie,
            swap_available=self.swap_available,
            swapped=self.swapped,
            history_flags=self.history_flags,
            history_known=self.history_known,
            pda=self.pda,
        )

    @property
    def variant_label(self) -> str:
        return variant_label(self.mode, self.handicap, self.pie)

    @property
    def segment(self) -> str:
        return variant_segment(self.mode, self.handicap, self.pie)

    @property
    def has_teacher(self) -> bool:
        return bool(self.target_mask & TARGET_TEACHER)

    def outcome_targets(self) -> tuple[int, int]:
        """Return current-player binary outcome and score margin."""

        if not self.target_mask & (TARGET_OUTCOME | TARGET_SCORE_MARGIN):
            raise ReplaySchemaError("final outcome targets are unavailable")
        margin = int(self.final_scores[self.to_move]) - int(
            self.final_scores[1 - self.to_move]
        )
        return self.outcome, margin


def _leader_from_scores(scores: np.ndarray, capes: np.ndarray) -> int:
    if scores[0] != scores[1]:
        return 0 if scores[0] > scores[1] else 1
    if capes[0] != capes[1]:
        return 0 if capes[0] > capes[1] else 1
    return -1


def augment_sample(sample: ReplaySample, transform: D5Transform) -> ReplaySample:
    topology = get_topology(sample.rings)
    permutation = topology.d5_permutation(
        transform.rotation, transform.reflected
    ).numpy()

    def nodes(values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values)
        output[permutation] = values
        return output

    assert sample.history_flags is not None
    assert sample.teacher_policy is not None
    assert sample.teacher_outcome is not None
    assert sample.teacher_score_margin is not None
    augmented = ReplaySample(
        rings=sample.rings,
        stones=nodes(sample.stones),
        to_move=sample.to_move,
        moves_left=sample.moves_left,
        opening=sample.opening,
        terminal=sample.terminal,
        policy=nodes(sample.policy),
        soft_policy=nodes(sample.soft_policy),
        target_mask=sample.target_mask,
        outcome=sample.outcome,
        final_scores=sample.final_scores.copy(),
        final_capes=sample.final_capes.copy(),
        # Validate the existing game ending (cached), then apply the exact D5
        # permutation below. Component counts are invariant under that map.
        final_ownership=sample.final_ownership.copy(),
        final_alive=sample.final_alive.copy(),
        search_provenance=sample.search_provenance,
        policy_provenance=sample.policy_provenance,
        run_id=sample.run_id,
        generation_family=sample.generation_family,
        actor_id=sample.actor_id,
        generation=sample.generation,
        game_id=sample.game_id,
        ply=sample.ply,
        model_identity=sample.model_identity,
        weight=sample.weight,
        policy_weight=sample.policy_weight,
        mode=sample.mode,
        handicap=sample.handicap,
        pie=sample.pie,
        swap_available=sample.swap_available,
        swapped=sample.swapped,
        history_known=sample.history_known,
        pda=sample.pda,
        history_flags=nodes(sample.history_flags),
        teacher_policy=nodes(sample.teacher_policy),
        teacher_outcome=sample.teacher_outcome.copy(),
        teacher_score_margin=sample.teacher_score_margin.copy(),
        opponent_reply=(
            np.concatenate(
                (nodes(sample.opponent_reply[:-1]), sample.opponent_reply[-1:])
            )
            if sample.opponent_reply is not None
            else None
        ),
        opponent_reply_ply=sample.opponent_reply_ply,
        second_stone=nodes(sample.second_stone)
        if sample.second_stone is not None
        else None,
        second_stone_ply=sample.second_stone_ply,
        final_shores=sample.final_shores,
        final_networks=sample.final_networks,
    )
    augmented.final_ownership = nodes(sample.final_ownership)
    augmented.final_alive = nodes(sample.final_alive)
    return augmented


def _offsets(lengths: Sequence[int]) -> np.ndarray:
    output = np.zeros(len(lengths) + 1, dtype=np.int64)
    output[1:] = np.cumsum(lengths, dtype=np.int64)
    return output


def shard_metadata(sample_count: int) -> dict[str, object]:
    return {
        "format": REPLAY_SHARD_FORMAT,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "rules_schema": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH,
        "rules_hash_wire": RULES_HASH_WIRE,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "sample_count": sample_count,
        "soft_policy_temperature": SOFT_POLICY_TEMPERATURE,
    }


def write_replay_shard(
    destination: str | Path,
    samples: Sequence[ReplaySample],
    *,
    compressed: bool = True,
) -> Path:
    if not samples:
        raise ValueError("cannot write an empty replay shard")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_offsets = _offsets([sample.stones.size for sample in samples])
    metadata = shard_metadata(len(samples))
    metadata["auxiliary_targets_version"] = REPLAY_AUXILIARY_TARGETS_VERSION

    def required(values: Sequence[np.ndarray | None], name: str) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for value in values:
            if value is None:
                raise ReplaySchemaError(f"{name} is missing on a validated sample")
            output.append(value)
        return output

    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "rings": np.asarray([sample.rings for sample in samples], dtype=np.int8),
        "node_offsets": node_offsets,
        "stones": np.concatenate([sample.stones for sample in samples]),
        "to_move": np.asarray([sample.to_move for sample in samples], dtype=np.int8),
        "moves_left": np.asarray(
            [sample.moves_left for sample in samples], dtype=np.int8
        ),
        "opening": np.asarray([sample.opening for sample in samples], dtype=np.bool_),
        "terminal": np.asarray([sample.terminal for sample in samples], dtype=np.bool_),
        "policy": np.concatenate([sample.policy for sample in samples]),
        "soft_policy": np.concatenate([sample.soft_policy for sample in samples]),
        "target_mask": np.asarray(
            [sample.target_mask for sample in samples], dtype=np.uint16
        ),
        "outcome": np.asarray([sample.outcome for sample in samples], dtype=np.int8),
        "final_scores": np.stack([sample.final_scores for sample in samples]),
        "final_capes": np.stack([sample.final_capes for sample in samples]),
        "final_ownership": np.concatenate(
            [sample.final_ownership for sample in samples]
        ),
        "final_alive": np.concatenate([sample.final_alive for sample in samples]),
        "search_provenance": np.asarray(
            [sample.search_provenance for sample in samples], dtype=np.str_
        ),
        "policy_provenance": np.asarray(
            [sample.policy_provenance for sample in samples], dtype=np.str_
        ),
        "run_id": np.asarray([sample.run_id for sample in samples], dtype=np.str_),
        "generation_family": np.asarray(
            [sample.generation_family for sample in samples], dtype=np.str_
        ),
        "actor_id": np.asarray([sample.actor_id for sample in samples], dtype=np.str_),
        "generation": np.asarray(
            [sample.generation for sample in samples], dtype=np.int64
        ),
        "game_id": np.asarray([sample.game_id for sample in samples], dtype=np.str_),
        "ply": np.asarray([sample.ply for sample in samples], dtype=np.int32),
        "model_identity": np.asarray(
            [sample.model_identity for sample in samples], dtype=np.str_
        ),
        "soft_policy_temperature": np.asarray(
            [sample.soft_policy_temperature for sample in samples],
            dtype=np.float32,
        ),
        "rules_hash": np.asarray(
            [sample.rules_hash for sample in samples], dtype=np.uint64
        ),
        "feature_schema_hash": np.asarray(
            [sample.feature_schema_hash for sample in samples], dtype=np.uint64
        ),
        "weight": np.asarray([sample.weight for sample in samples], dtype=np.float32),
        "policy_weight": np.asarray(
            [sample.policy_weight for sample in samples], dtype=np.float32
        ),
        "mode": np.asarray(
            [MODE_INDEX[sample.mode] for sample in samples], dtype=np.int8
        ),
        "handicap": np.asarray([sample.handicap for sample in samples], dtype=np.int8),
        "pie": np.asarray([sample.pie for sample in samples], dtype=np.bool_),
        "swap_available": np.asarray(
            [sample.swap_available for sample in samples], dtype=np.bool_
        ),
        "swapped": np.asarray([sample.swapped for sample in samples], dtype=np.bool_),
        "history_known": np.asarray(
            [sample.history_known for sample in samples], dtype=np.bool_
        ),
        "pda": np.asarray([sample.pda for sample in samples], dtype=np.int8),
        "history_flags": np.concatenate(
            required([sample.history_flags for sample in samples], "history_flags")
        ),
        "teacher_policy": np.concatenate(
            required([sample.teacher_policy for sample in samples], "teacher_policy")
        ),
        "teacher_outcome": np.stack(
            required([sample.teacher_outcome for sample in samples], "teacher_outcome")
        ),
        "teacher_score_margin": np.stack(
            required(
                [sample.teacher_score_margin for sample in samples],
                "teacher_score_margin",
            )
        ),
        "opponent_reply": np.concatenate(
            [
                s.opponent_reply[:-1]
                if s.opponent_reply is not None
                else np.zeros(s.stones.size, np.float32)
                for s in samples
            ]
        ),
        "opponent_reply_swap": np.asarray(
            [
                s.opponent_reply[-1] if s.opponent_reply is not None else 0.0
                for s in samples
            ],
            dtype=np.float32,
        ),
        "opponent_reply_ply": np.asarray(
            [s.opponent_reply_ply for s in samples], dtype=np.int32
        ),
        "second_stone": np.concatenate(
            [
                s.second_stone
                if s.second_stone is not None
                else np.zeros(s.stones.size, np.float32)
                for s in samples
            ]
        ),
        "second_stone_ply": np.asarray(
            [s.second_stone_ply for s in samples], dtype=np.int32
        ),
        "final_shores": np.stack(
            [
                s.final_shores
                if s.final_shores is not None
                else np.full(2, -1, np.int8)
                for s in samples
            ]
        ),
        "final_networks": np.stack(
            [
                s.final_networks if s.final_networks is not None else np.full(2, -1, np.int8)
                for s in samples
            ]
        ),
    }
    writer = np.savez_compressed if compressed else np.savez
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


@dataclass(frozen=True, slots=True)
class DecodedReplayShard:
    """A validated shard whose NPZ members have each been materialized once.

    ``legacy`` marks a schema-v4 shard of the previous lineage decoded through
    the upgrade adapter: its samples become standard-variant rules-v3 samples
    with unknown history and the current contract hashes.
    """

    source: Path
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    sample_count: int
    legacy: bool = False

    def __len__(self) -> int:
        return self.sample_count

    def sample(self, index: int) -> ReplaySample:
        if index < 0:
            index += self.sample_count
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        arrays = self.arrays
        node_offsets = arrays["node_offsets"]
        node_slice = slice(int(node_offsets[index]), int(node_offsets[index + 1]))
        common = dict(
            rings=arrays["rings"][index],
            stones=arrays["stones"][node_slice].copy(),
            to_move=arrays["to_move"][index],
            moves_left=arrays["moves_left"][index],
            opening=arrays["opening"][index],
            terminal=arrays["terminal"][index],
            policy=arrays["policy"][node_slice].copy(),
            soft_policy=arrays["soft_policy"][node_slice].copy(),
            target_mask=arrays["target_mask"][index],
            outcome=arrays["outcome"][index],
            final_scores=arrays["final_scores"][index].copy(),
            final_capes=arrays["final_capes"][index].copy(),
            final_ownership=arrays["final_ownership"][node_slice].copy(),
            final_alive=arrays["final_alive"][node_slice].copy(),
            search_provenance=str(arrays["search_provenance"][index]),
            policy_provenance=str(arrays["policy_provenance"][index]),
            run_id=str(arrays["run_id"][index]),
            generation_family=str(arrays["generation_family"][index]),
            actor_id=str(arrays["actor_id"][index]),
            generation=int(arrays["generation"][index]),
            game_id=str(arrays["game_id"][index]),
            ply=int(arrays["ply"][index]),
            model_identity=str(arrays["model_identity"][index]),
            soft_policy_temperature=float(arrays["soft_policy_temperature"][index]),
            weight=float(arrays["weight"][index]),
            policy_weight=float(arrays["policy_weight"][index]),
        )
        if self.legacy:
            # Upgrade adapter: standard variant, unknown history, rehashed.
            provenance = common["search_provenance"]
            if LEGACY_UPGRADE_PROVENANCE not in provenance:
                common["search_provenance"] = (
                    f"{provenance};{LEGACY_UPGRADE_PROVENANCE}"
                )
            return ReplaySample(
                **common,
                rules_hash=RULES_HASH,
                feature_schema_hash=FEATURE_SCHEMA_HASH,
                history_known=False,
            )
        auxiliary: dict[str, Any] = {}
        if "opponent_reply" in arrays:
            for name in ("opponent_reply", "second_stone"):
                destination = _checked_int(f"{name}_ply", arrays[f"{name}_ply"][index])
                values = arrays[name][node_slice].copy()
                if name == "opponent_reply":
                    values = np.concatenate(
                        (values, arrays["opponent_reply_swap"][index : index + 1])
                    )
                if destination == -1 and np.any(values):
                    raise ReplaySchemaError(f"missing {name} must be all zero")
                auxiliary[name] = values if destination != -1 else None
                auxiliary[f"{name}_ply"] = destination
            for name in ("final_shores", "final_networks"):
                values = arrays[name][index].copy()
                if not np.issubdtype(values.dtype, np.integer):
                    raise ReplaySchemaError(f"{name} must contain integer counts")
                if np.any(values == -1) and not np.all(values == -1):
                    raise ReplaySchemaError(f"missing {name} must use two -1 sentinels")
                required = TARGET_OWNERSHIP | (
                    TARGET_ALIVE if name == "final_networks" else 0
                )
                if (
                    not np.all(values == -1)
                    and int(arrays["target_mask"][index]) & required != required
                ):
                    raise ReplaySchemaError(
                        f"stored {name} requires available spatial targets"
                    )
                auxiliary[name] = None if np.all(values == -1) else values
        return ReplaySample(
            **common,
            rules_hash=arrays["rules_hash"][index],
            feature_schema_hash=arrays["feature_schema_hash"][index],
            mode=MODES[int(arrays["mode"][index])],
            handicap=int(arrays["handicap"][index]),
            pie=bool(arrays["pie"][index]),
            swap_available=bool(arrays["swap_available"][index]),
            swapped=bool(arrays["swapped"][index]),
            history_known=bool(arrays["history_known"][index]),
            pda=int(arrays["pda"][index]),
            history_flags=arrays["history_flags"][node_slice].copy(),
            teacher_policy=arrays["teacher_policy"][node_slice].copy(),
            teacher_outcome=arrays["teacher_outcome"][index].copy(),
            teacher_score_margin=arrays["teacher_score_margin"][index].copy(),
            **auxiliary,
        )

    def samples(self, indices: Sequence[int] | None = None) -> list[ReplaySample]:
        requested = range(self.sample_count) if indices is None else indices
        return [self.sample(index) for index in requested]


def _legacy_shard_metadata() -> dict[str, object]:
    return {
        "format": REPLAY_SHARD_FORMAT,
        "schema_version": LEGACY_REPLAY_SCHEMA_VERSION,
        "rules_schema": LEGACY_RULES_SCHEMA_ID,
        "rules_hash": LEGACY_RULES_HASH,
        "rules_hash_wire": LEGACY_RULES_HASH_WIRE,
        "feature_schema_hash": LEGACY_FEATURE_SCHEMA_HASH,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "soft_policy_temperature": SOFT_POLICY_TEMPERATURE,
    }


def decode_replay_shard(
    source: str | Path,
    *,
    allow_legacy: bool = False,
) -> DecodedReplayShard:
    """Load every compressed NPZ member once and validate its column layout.

    With ``allow_legacy`` a schema-v4 shard of the previous lineage decodes
    through the upgrade adapter; the replay store never commits such shards,
    the lineage-transfer importer rewrites them as v5.
    """

    path = Path(source)
    with np.load(path, allow_pickle=False) as shard:
        if "metadata" not in shard.files:
            raise ReplaySchemaError("replay shard is missing arrays: metadata")
        metadata = json.loads(str(shard["metadata"].item()))
        legacy = metadata.get("schema_version") == LEGACY_REPLAY_SCHEMA_VERSION
        if legacy and not allow_legacy:
            raise ReplaySchemaError(
                "incompatible shard metadata: schema_version (legacy schema-v4 "
                "shards are only readable through the lineage-transfer importer)"
            )
        names = _LEGACY_SAMPLE_ARRAY_NAMES if legacy else _REPLAY_SAMPLE_ARRAY_NAMES
        auxiliary_version = metadata.get("auxiliary_targets_version")
        auxiliary_present = set(_AUXILIARY_SAMPLE_ARRAY_NAMES).intersection(shard.files)
        if auxiliary_version is not None:
            if (
                legacy
                or type(auxiliary_version) is not int
                or auxiliary_version != REPLAY_AUXILIARY_TARGETS_VERSION
            ):
                raise ReplaySchemaError(
                    "unsupported replay auxiliary targets capability"
                )
            names += _AUXILIARY_SAMPLE_ARRAY_NAMES
        elif auxiliary_present:
            raise ReplaySchemaError(
                "auxiliary replay arrays require a versioned capability"
            )
        missing = set(names).difference(shard.files)
        if missing:
            raise ReplaySchemaError(
                f"replay shard is missing arrays: {', '.join(sorted(missing))}"
            )
        # NpzFile is lazy and does not cache __getitem__ results. Materializing
        # these columns here prevents each sample from reopening and inflating
        # the same compressed ZIP members.
        arrays = {name: shard[name] for name in names}

    expected_metadata = _legacy_shard_metadata() if legacy else shard_metadata(0)
    expected_metadata.pop("sample_count", None)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ReplaySchemaError(f"incompatible shard metadata: {key}")
    count = _checked_int("sample_count", metadata.get("sample_count"))
    if count <= 0:
        raise ReplaySchemaError("shard sample_count must be positive")
    _validate_decoded_shard_arrays(arrays, count=count, legacy=legacy)
    return DecodedReplayShard(path, metadata, arrays, count, legacy=legacy)


def _validate_decoded_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    count: int,
    legacy: bool = False,
) -> None:
    per_sample = [
        "rings",
        "to_move",
        "moves_left",
        "opening",
        "terminal",
        "target_mask",
        "outcome",
        "search_provenance",
        "policy_provenance",
        "run_id",
        "generation_family",
        "actor_id",
        "generation",
        "game_id",
        "ply",
        "model_identity",
        "soft_policy_temperature",
        "rules_hash",
        "feature_schema_hash",
        "weight",
        "policy_weight",
    ]
    if not legacy:
        per_sample.extend(
            (
                "mode",
                "handicap",
                "pie",
                "swap_available",
                "swapped",
                "history_known",
                "pda",
            )
        )
    if any(arrays[name].shape != (count,) for name in per_sample):
        raise ReplaySchemaError("shard sample count does not match arrays")
    if arrays["final_scores"].shape != (count, 2) or arrays["final_capes"].shape != (
        count,
        2,
    ):
        raise ReplaySchemaError("shard result columns have invalid shapes")
    if not legacy:
        if arrays["teacher_outcome"].shape != (count, 2) or arrays[
            "teacher_score_margin"
        ].shape != (count, SCORE_MARGIN_BINS):
            raise ReplaySchemaError("shard teacher columns have invalid shapes")
    if "opponent_reply" in arrays:
        for name in ("opponent_reply_ply", "second_stone_ply", "opponent_reply_swap"):
            if arrays[name].shape != (count,):
                raise ReplaySchemaError(
                    "shard auxiliary policy columns have invalid shapes"
                )
        for name in ("final_shores", "final_networks"):
            if arrays[name].shape != (count, 2):
                raise ReplaySchemaError(
                    "shard auxiliary count columns have invalid shapes"
                )

    node_offsets = arrays["node_offsets"]
    if node_offsets.shape != (count + 1,):
        raise ReplaySchemaError("shard offsets have invalid shapes")
    if int(node_offsets[0]) != 0 or np.any(np.diff(node_offsets) < 0):
        raise ReplaySchemaError("shard offsets must be monotonic and zero-based")
    node_values = int(node_offsets[-1])
    node_columns = ["stones", "final_ownership", "final_alive"]
    if not legacy:
        node_columns.extend(("history_flags", "teacher_policy"))
    if "opponent_reply" in arrays:
        node_columns.extend(("opponent_reply", "second_stone"))
    if any(arrays[name].shape != (node_values,) for name in node_columns):
        raise ReplaySchemaError("shard node columns disagree with node offsets")
    if any(arrays[name].shape != (node_values,) for name in ("policy", "soft_policy")):
        raise ReplaySchemaError("shard policy columns disagree with node offsets")
    if len(np.unique(arrays["rings"])) != 1:
        raise ReplaySchemaError("replay shard must be ring-homogeneous")
    if not legacy:
        variants = {
            (int(mode), int(handicap), bool(pie))
            for mode, handicap, pie in zip(
                arrays["mode"], arrays["handicap"], arrays["pie"], strict=True
            )
        }
        if len(variants) != 1:
            raise ReplaySchemaError("replay shard must be variant-homogeneous")


def read_replay_shard(
    source: str | Path, *, allow_legacy: bool = False
) -> list[ReplaySample]:
    return decode_replay_shard(source, allow_legacy=allow_legacy).samples()


class ReplayDataset(Dataset[ReplaySample]):
    def __init__(self, samples: Sequence[ReplaySample]) -> None:
        self.samples = list(samples)
        self.rings = [sample.rings for sample in self.samples]

    @classmethod
    def from_shards(cls, paths: Sequence[str | Path]) -> "ReplayDataset":
        samples: list[ReplaySample] = []
        for path in paths:
            samples.extend(read_replay_shard(path))
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReplaySample:
        return self.samples[index]


def _geometry_tensors(inputs: EncodedBatch) -> tuple[torch.Tensor, ...]:
    return (
        inputs.rings,
        inputs.neighbor_index,
        inputs.neighbor_mask,
        inputs.neighbor_edge_type,
        inputs.node_mask,
    )


def _geometry_versions(inputs: EncodedBatch) -> tuple[int, ...] | None:
    try:
        return tuple(tensor._version for tensor in _geometry_tensors(inputs))
    except RuntimeError:
        # Inference tensors have no mutation counters. They cannot certify a
        # training optimization whose safety depends on unchanged masks.
        return None


@dataclass(frozen=True, slots=True)
class _HomogeneousGeometry:
    """CPU-validated geometry plus cheap, device-independent mutation guards."""

    ring: int
    inputs: EncodedBatch = field(repr=False, compare=False)
    versions: tuple[int, ...]
    shape: tuple[int, int]

    def matches(self, inputs: EncodedBatch) -> bool:
        return (
            (inputs.batch_size, inputs.max_nodes) == self.shape
            and all(
                current is original
                for current, original in zip(
                    _geometry_tensors(inputs),
                    _geometry_tensors(self.inputs),
                    strict=True,
                )
            )
            and _geometry_versions(inputs) == self.versions
        )

    def __reduce__(self) -> tuple[object, tuple[EncodedBatch]]:
        # DataLoader IPC reconstructs tensor identities/version counters. Recheck
        # CPU values in the receiving process instead of trusting old counters.
        return _validate_homogeneous_geometry, (self.inputs,)


def _bind_homogeneous_geometry(
    inputs: EncodedBatch, ring: int | None
) -> _HomogeneousGeometry | None:
    """Bind previously validated, value-preserving copies without device reads."""

    versions = _geometry_versions(inputs)
    if ring is None or versions is None:
        return None
    return _HomogeneousGeometry(
        ring, inputs, versions, (inputs.batch_size, inputs.max_nodes)
    )


def _validate_homogeneous_geometry(
    inputs: EncodedBatch,
) -> _HomogeneousGeometry | None:
    """Certify canonical, unpadded CPU collation; other batches stay general."""

    tensors = _geometry_tensors(inputs)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        return None
    if inputs.node_features.ndim != 3 or inputs.batch_size == 0:
        return None
    batch_size, nodes = inputs.batch_size, inputs.max_nodes
    if inputs.rings.shape != (batch_size,) or inputs.rings.dtype != torch.long:
        return None
    ring = int(inputs.rings[0])
    if not bool((inputs.rings == ring).all()):
        return None
    try:
        topology = get_topology(ring)
    except ValueError:
        return None
    if nodes != topology.n:
        return None
    for actual, canonical in (
        (inputs.neighbor_index, topology.neighbor_index),
        (inputs.neighbor_mask, topology.neighbor_mask),
        (inputs.neighbor_edge_type, topology.neighbor_edge_type),
        (inputs.node_mask, torch.ones(nodes, dtype=torch.bool)),
    ):
        if actual.shape != (batch_size, *canonical.shape):
            return None
        if actual.dtype != canonical.dtype or not torch.equal(actual[0], canonical):
            return None
        # Native collation broadcasts one immutable topology row already.
        if actual.stride(0) != 0 and not torch.equal(
            actual, actual[:1].expand_as(actual)
        ):
            return None
    return _bind_homogeneous_geometry(inputs, ring)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    inputs: EncodedBatch
    targets: TrainingTargets
    feature_path: str = "python"
    # Original rules survive resolved pie openings, whose encoded features no
    # longer distinguish them from standard play. This is diagnostic metadata.
    variant_labels: tuple[str, ...] | None = None
    policy_metrics: PolicyBatchMetrics | None = None
    _homogeneous_geometry: _HomogeneousGeometry | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def homogeneous_ring(self) -> int | None:
        """Validated geometry metadata, or None after any static-input mutation."""

        proof = self._homogeneous_geometry
        return proof.ring if proof is not None and proof.matches(self.inputs) else None

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "ReplayBatch":
        inputs = self.inputs.to(
            device,
            feature_dtype=feature_dtype,
            non_blocking=non_blocking,
        )
        return ReplayBatch(
            inputs=inputs,
            targets=self.targets.to(device, non_blocking=non_blocking),
            feature_path=self.feature_path,
            variant_labels=self.variant_labels,
            policy_metrics=self.policy_metrics,
            _homogeneous_geometry=_bind_homogeneous_geometry(
                inputs, self.homogeneous_ring
            ),
        )

    def pin_memory(self) -> "ReplayBatch":
        inputs = self.inputs.pin_memory(pin_topology=False)
        return ReplayBatch(
            inputs=inputs,
            targets=self.targets.pin_memory(),
            feature_path=self.feature_path,
            variant_labels=self.variant_labels,
            policy_metrics=self.policy_metrics,
            _homogeneous_geometry=_bind_homogeneous_geometry(
                inputs, self.homogeneous_ring
            ),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        self.inputs.record_stream(stream)
        self.targets.record_stream(stream)


def collate_replay_samples(
    samples: Sequence[ReplaySample],
    *,
    prefer_native: bool = True,
) -> ReplayBatch:
    if not samples:
        raise ValueError("cannot collate an empty replay batch")
    positions = [sample.to_position() for sample in samples]
    inputs: EncodedBatch | None = None
    feature_path = "python"
    if prefer_native:
        # Keep this local to avoid a features -> native -> replay import cycle.
        from .native import encode_native_semantic_batch

        inputs = encode_native_semantic_batch(positions)
        if inputs is not None:
            feature_path = "rust"
    if inputs is None:
        inputs = collate_encoded([encode_position(position) for position in positions])
    batch_size = len(samples)
    max_nodes = inputs.max_nodes
    policy = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    soft_policy = torch.zeros_like(policy)
    ownership = torch.full((batch_size, max_nodes), -100, dtype=torch.long)
    alive = torch.full((batch_size, max_nodes), -1.0, dtype=torch.float32)
    outcome = torch.zeros(batch_size, dtype=torch.long)
    margin = torch.zeros(batch_size, dtype=torch.long)
    clinch = torch.zeros(batch_size, dtype=torch.bool)
    opponent_reply = torch.zeros((batch_size, max_nodes + 1), dtype=torch.float32)
    second_stone = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    auxiliary_counts = {
        name: torch.zeros((batch_size, 2), dtype=torch.long)
        for name in ("final_shores", "final_networks", "final_capes")
    }
    auxiliary_masks = {
        name: torch.zeros(batch_size, dtype=torch.bool)
        for name in ("opponent_reply", "second_stone", *auxiliary_counts)
    }
    any_teacher = any(sample.has_teacher for sample in samples)
    teacher_policy = (
        torch.zeros((batch_size, max_nodes), dtype=torch.float32)
        if any_teacher
        else None
    )
    teacher_outcome = (
        torch.zeros((batch_size, 2), dtype=torch.float32) if any_teacher else None
    )
    teacher_margin = (
        torch.zeros((batch_size, SCORE_MARGIN_BINS), dtype=torch.float32)
        if any_teacher
        else None
    )
    teacher_mask = torch.zeros(batch_size, dtype=torch.bool) if any_teacher else None

    masks = {
        "policy": torch.zeros(batch_size, dtype=torch.bool),
        "outcome": torch.zeros(batch_size, dtype=torch.bool),
        "margin": torch.zeros(batch_size, dtype=torch.bool),
        "ownership": torch.zeros(batch_size, dtype=torch.bool),
        "alive": torch.zeros(batch_size, dtype=torch.bool),
        "soft": torch.zeros(batch_size, dtype=torch.bool),
    }
    for index, sample in enumerate(samples):
        nodes = get_topology(sample.rings).n
        if sample.opponent_reply is not None:
            opponent_reply[index, :nodes] = torch.from_numpy(sample.opponent_reply[:-1])
            # The swap slot follows batch padding, not this sample's last node.
            opponent_reply[index, max_nodes] = float(sample.opponent_reply[-1])
            auxiliary_masks["opponent_reply"][index] = True
        if sample.second_stone is not None:
            second_stone[index, :nodes] = torch.from_numpy(sample.second_stone)
            auxiliary_masks["second_stone"][index] = True
        order = [sample.to_move, 1 - sample.to_move]
        for name in auxiliary_counts:
            values = getattr(sample, name)
            available = values is not None and (
                name != "final_capes" or bool(sample.target_mask & TARGET_SCORE_MARGIN)
            )
            if available:
                assert values is not None
                auxiliary_counts[name][index] = torch.from_numpy(
                    values[order].astype(np.int64)
                )
                auxiliary_masks[name][index] = True
        policy[index] = relocate_sample_actions(
            torch.from_numpy(sample.policy),
            sample_nodes=nodes,
            batch_max_nodes=max_nodes,
            fill_value=0.0,
        )
        soft_policy[index] = relocate_sample_actions(
            torch.from_numpy(sample.soft_policy),
            sample_nodes=nodes,
            batch_max_nodes=max_nodes,
            fill_value=0.0,
        )
        masks["policy"][index] = bool(sample.target_mask & TARGET_POLICY)
        masks["soft"][index] = bool(sample.target_mask & TARGET_SOFT_POLICY)
        clinch[index] = "final=clinch-loser-fill" in sample.search_provenance
        if sample.target_mask & (TARGET_OUTCOME | TARGET_SCORE_MARGIN):
            sample_outcome, sample_margin = sample.outcome_targets()
            outcome[index] = sample_outcome
            margin[index] = sample_margin
        masks["outcome"][index] = bool(sample.target_mask & TARGET_OUTCOME)
        masks["margin"][index] = bool(sample.target_mask & TARGET_SCORE_MARGIN)
        if sample.target_mask & TARGET_OWNERSHIP:
            absolute_owner = torch.from_numpy(sample.final_ownership)
            ownership[index, :nodes] = torch.where(
                absolute_owner == -1,
                torch.tensor(2),
                torch.where(
                    absolute_owner == sample.to_move,
                    torch.tensor(0),
                    torch.tensor(1),
                ),
            )
            masks["ownership"][index] = True
        if sample.target_mask & TARGET_ALIVE:
            alive[index, :nodes] = torch.from_numpy(sample.final_alive).float()
            masks["alive"][index] = True
        if sample.has_teacher and teacher_mask is not None:
            assert teacher_policy is not None
            assert teacher_outcome is not None
            assert teacher_margin is not None
            assert sample.teacher_policy is not None
            assert sample.teacher_outcome is not None
            assert sample.teacher_score_margin is not None
            teacher_policy[index] = relocate_sample_actions(
                torch.from_numpy(sample.teacher_policy.astype(np.float32)),
                sample_nodes=nodes,
                batch_max_nodes=max_nodes,
                fill_value=0.0,
            )
            teacher_outcome[index] = torch.from_numpy(
                sample.teacher_outcome.astype(np.float32)
            )
            teacher_margin[index] = torch.from_numpy(
                sample.teacher_score_margin.astype(np.float32)
            )
            teacher_mask[index] = True

    return ReplayBatch(
        inputs=inputs,
        targets=TrainingTargets(
            policy=policy,
            outcome=outcome,
            score_margin=margin,
            ownership=ownership,
            alive=alive,
            soft_policy=soft_policy,
            policy_mask=masks["policy"],
            outcome_mask=masks["outcome"],
            score_margin_mask=masks["margin"],
            ownership_mask=masks["ownership"],
            alive_mask=masks["alive"],
            soft_policy_mask=masks["soft"],
            sample_weight=torch.tensor(
                [sample.weight for sample in samples], dtype=torch.float32
            ),
            policy_weight=torch.tensor(
                [sample.policy_weight for sample in samples], dtype=torch.float32
            ),
            clinch_mask=clinch,
            teacher_policy=teacher_policy,
            teacher_outcome=teacher_outcome,
            teacher_score_margin=teacher_margin,
            teacher_mask=teacher_mask,
            opponent_reply=opponent_reply,
            opponent_reply_mask=auxiliary_masks["opponent_reply"],
            second_stone=second_stone,
            second_stone_mask=auxiliary_masks["second_stone"],
            final_shores=auxiliary_counts["final_shores"],
            final_shores_mask=auxiliary_masks["final_shores"],
            final_networks=auxiliary_counts["final_networks"],
            final_networks_mask=auxiliary_masks["final_networks"],
            final_capes=auxiliary_counts["final_capes"],
            final_capes_mask=auxiliary_masks["final_capes"],
        ),
        feature_path=feature_path,
        variant_labels=tuple(sample.variant_label for sample in samples),
        policy_metrics=PolicyBatchMetrics.from_samples(samples),
        _homogeneous_geometry=_validate_homogeneous_geometry(inputs),
    )
