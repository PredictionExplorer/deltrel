"""Versioned wire schemas for full-strength Deltrel analysis (rules v3).

Schema v3 carries the rule variant (mode, handicap, pie), the swap state, the
retained placement history the network was trained on, and an optional
playout-doubling advantage for the side to move. The response adds the swap
recommendation for pie games and the search's root value.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deltreltrain.contracts import (
    MAX_HANDICAP,
    MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    RULES_HASH_WIRE,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
)
from deltreltrain.features import DoubleDeltrelPosition
from deltreltrain.topology import MAX_NODES, get_topology

API_SCHEMA_VERSION = 3

StrictRing = Annotated[int, Field(strict=True)]
StrictPlayer = Annotated[int, Field(strict=True, ge=0, le=1)]
StrictMoves = Annotated[int, Field(strict=True, ge=0, le=MAX_HANDICAP)]
StrictHandicap = Annotated[int, Field(strict=True, ge=1, le=MAX_HANDICAP)]
StrictPda = Annotated[
    int,
    Field(
        strict=True,
        ge=-MAX_PLAYOUT_DOUBLING_ADVANTAGE,
        le=MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    ),
]
StrictNode = Annotated[int, Field(strict=True, ge=0)]
StrictSeed = Annotated[int, Field(strict=True, ge=0, le=(1 << 64) - 1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
UnitValue = Annotated[float, Field(strict=True, ge=-1.0, le=1.0)]
NonnegativeFinite = Annotated[float, Field(strict=True, ge=0.0)]
ScoreMarginExpectation = Annotated[
    float,
    Field(strict=True, ge=SCORE_MARGIN_MIN, le=SCORE_MARGIN_MAX),
]
RulesHash = Annotated[
    str,
    Field(strict=True, pattern=f"^{re.escape(RULES_HASH_WIRE)}$"),
]
ScoreSupportMin = Annotated[
    int,
    Field(strict=True, ge=SCORE_MARGIN_MIN, le=SCORE_MARGIN_MIN),
]
ScoreSupportMax = Annotated[
    int,
    Field(strict=True, ge=SCORE_MARGIN_MAX, le=SCORE_MARGIN_MAX),
]


class SearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    simulations: PositiveInt
    max_considered: PositiveInt
    seed: StrictSeed


class PlacementHistory(BaseModel):
    """Retained placement sets; every entry is a dense node id."""

    model_config = ConfigDict(extra="forbid", strict=True)

    current_turn: list[StrictNode]
    previous_turn: list[StrictNode]
    own_previous_turn: list[StrictNode]
    handicap_stones: list[StrictNode]

    def node_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        return (
            list(self.current_turn),
            list(self.previous_turn),
            list(self.own_previous_turn),
            list(self.handicap_stones),
        )


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[3]
    rules_hash: RulesHash
    rings: StrictRing
    stones: list[Literal[-1, 0, 1]]
    to_move: StrictPlayer
    moves_left: StrictMoves
    opening: bool
    terminal: Literal[False]
    mode: Literal["classic", "double"] = "double"
    handicap: StrictHandicap = 1
    pie: bool = False
    swap_available: bool = False
    swapped: bool = False
    # ``None`` means the client cannot supply history (an imported position);
    # the network then sees ``history_known = 0`` and empty history planes.
    history: PlacementHistory | None = None
    pda: StrictPda = 0
    search: SearchBudget
    # Additive opt-in keeps strict schema-v3 clients' existing response shape.
    include_predictions: bool = False
    # Complete root head diagnostics are opt-in and never streamed for leaves.
    include_network_output: bool = False

    @model_validator(mode="after")
    def validate_semantic_state(self) -> "AnalyzeRequest":
        if any(type(stone) is not int for stone in self.stones):
            raise ValueError("stones must contain strict integers")
        nodes = get_topology(self.rings).n
        if len(self.stones) != nodes:
            raise ValueError(f"stones must contain exactly {nodes} entries")
        try:
            self.position()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Deltrel state: {exc}") from exc
        return self

    def position(self) -> DoubleDeltrelPosition:
        """The validated semantic position (history planes when supplied)."""

        nodes = len(self.stones)

        def mask(values: list[int] | None) -> torch.Tensor | None:
            if values is None:
                return None
            if any(node >= nodes for node in values) or len(set(values)) != len(values):
                raise ValueError("history nodes must be unique dense node ids")
            output = torch.zeros(nodes, dtype=torch.bool)
            if values:
                output[torch.tensor(values, dtype=torch.long)] = True
            return output

        sets = self.history.node_sets() if self.history is not None else None
        return DoubleDeltrelPosition(
            rings=self.rings,
            stones=torch.tensor(self.stones, dtype=torch.int8),
            to_move=self.to_move,
            moves_left=self.moves_left,
            opening=self.opening,
            terminal=self.terminal,
            mode=self.mode,
            handicap=self.handicap,
            pie=self.pie,
            swap_available=self.swap_available,
            swapped=self.swapped,
            current_turn=mask(sets[0]) if sets is not None else None,
            previous_turn=mask(sets[1]) if sets is not None else None,
            own_previous_turn=mask(sets[2]) if sets is not None else None,
            handicap_stones=mask(sets[3]) if sets is not None else None,
            history_known=self.history is not None,
            pda=self.pda,
        )


class AtomicAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: StrictNode
    kind: Literal["place"]
    node: StrictNode

    @model_validator(mode="after")
    def validate_node_code(self) -> "AtomicAction":
        if self.code != self.node:
            raise ValueError("placement code and node must match")
        return self


class VariantDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["classic", "double"]
    handicap: StrictHandicap
    pie: bool

    @model_validator(mode="after")
    def validate_family(self) -> "VariantDescriptor":
        if self.pie and self.handicap != 1:
            raise ValueError("handicap games cannot use the pie rule")
        return self


class OutcomeBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: Probability
    win: Probability

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "OutcomeBelief":
        if not math.isclose(self.loss + self.win, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("binary outcome probabilities must sum to one")
        return self


class ScoreBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support_min: ScoreSupportMin
    support_max: ScoreSupportMax
    expected_margin: ScoreMarginExpectation
    probabilities: list[Probability]

    @model_validator(mode="after")
    def validate_score_belief(self) -> "ScoreBelief":
        expected_bins = SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1
        if len(self.probabilities) != expected_bins:
            raise ValueError(f"score probabilities must contain {expected_bins} bins")
        if not math.isfinite(self.expected_margin) or not (
            SCORE_MARGIN_MIN <= self.expected_margin <= SCORE_MARGIN_MAX
        ):
            raise ValueError("expected score margin is outside its support")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("score probabilities must sum to one")
        return self


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: NonnegativeFinite
    model_reload: NonnegativeFinite
    inference_search: NonnegativeFinite
    total: NonnegativeFinite


class FinalPlayerPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    player: StrictPlayer
    shores: Annotated[float, Field(ge=0.0, le=50.0, allow_inf_nan=False)]
    networks: Annotated[float, Field(ge=0.0, le=25.0, allow_inf_nan=False)]
    corners: Annotated[float, Field(ge=0.0, le=5.0, allow_inf_nan=False)]
    corner_bonus_probability: Probability


class FutureMovePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    player: StrictPlayer
    kind: Literal["place", "swap"]
    node: Annotated[int, Field(strict=True, ge=0, lt=MAX_NODES)] | None
    probability: Probability

    @model_validator(mode="after")
    def validate_action(self) -> "FutureMovePrediction":
        if (self.kind == "place") != (self.node is not None):
            raise ValueError("future placement requires a node; swap must omit it")
        return self


class AuxiliaryPredictions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    perspective: StrictPlayer
    final_basis: Literal["official_end"]
    final_counts: list[FinalPlayerPrediction]
    opponent_reply: FutureMovePrediction | None
    second_stone: FutureMovePrediction | None

    @model_validator(mode="after")
    def validate_players(self) -> "AuxiliaryPredictions":
        if [item.player for item in self.final_counts] != [0, 1]:
            raise ValueError("final predictions must name both players in order")
        if (
            self.opponent_reply is not None
            and self.opponent_reply.player != 1 - self.perspective
        ):
            raise ValueError("opponent reply has the wrong player")
        if self.second_stone is not None and (
            self.second_stone.player != self.perspective
            or self.second_stone.kind != "place"
        ):
            raise ValueError("second stone must be the current player's placement")
        return self


class NetworkHead(BaseModel):
    """Flattened row-major model head; null logits represent structural masks."""

    model_config = ConfigDict(extra="forbid", strict=True)
    shape: Annotated[
        list[Annotated[int, Field(strict=True, ge=1, le=303)]],
        Field(min_length=1, max_length=2),
    ]
    activation: Literal["softmax", "sigmoid"]
    logits: Annotated[
        list[Annotated[float, Field(allow_inf_nan=False)] | None],
        Field(min_length=1, max_length=825),
    ]
    probabilities: Annotated[list[Probability], Field(min_length=1, max_length=825)]
    mask: Annotated[list[bool], Field(min_length=1, max_length=825)]
    applicable: bool

    @model_validator(mode="after")
    def validate_values(self) -> "NetworkHead":
        size = math.prod(self.shape)
        if not len(self.logits) == len(self.probabilities) == len(self.mask) == size:
            raise ValueError("network head dimensions do not match its values")
        for logit, probability, active in zip(
            self.logits, self.probabilities, self.mask, strict=True
        ):
            if (active and logit is None) or (
                not active and (logit is not None or probability != 0)
            ):
                raise ValueError("network head masks and values disagree")
        if self.activation == "softmax":
            width = self.shape[-1]
            for start in range(0, size, width):
                row = self.probabilities[start : start + width]
                if not math.isclose(math.fsum(row), 1, abs_tol=1e-6):
                    raise ValueError("network head probabilities are not normalized")
                logits = self.logits[start : start + width]
                active_logits = [x for x in logits if x is not None]
                maximum = max(active_logits)
                denominator = math.fsum(math.exp(x - maximum) for x in active_logits)
                for logit, probability in zip(logits, row, strict=True):
                    expected = (
                        0.0
                        if logit is None
                        else math.exp(logit - maximum) / denominator
                    )
                    if not math.isclose(probability, expected, abs_tol=1e-6):
                        raise ValueError(
                            "network head activation disagrees with logits"
                        )
        else:
            for logit, probability in zip(self.logits, self.probabilities, strict=True):
                if logit is None:
                    raise ValueError("alive logits cannot be masked")
                # Stable sigmoid avoids overflow on valid extreme logits.
                expected = (
                    1 / (1 + math.exp(-logit))
                    if logit >= 0
                    else math.exp(logit) / (1 + math.exp(logit))
                )
                if not math.isclose(probability, expected, abs_tol=1e-6):
                    raise ValueError("alive activation disagrees with logits")
        return self


class NetworkHeads(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    policy: NetworkHead
    outcome: NetworkHead
    score_margin: NetworkHead
    ownership: NetworkHead
    alive: NetworkHead
    soft_policy: NetworkHead
    opponent_reply: NetworkHead | None
    second_stone: NetworkHead | None
    final_shores: NetworkHead | None
    final_networks: NetworkHead | None
    final_capes: NetworkHead | None


class NetworkOutput(BaseModel):
    """Complete root outputs, all player-relative rows current then opponent."""

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    perspective: StrictPlayer
    node_count: Literal[50, 105, 180, 275]
    auxiliary_status: Literal["ready", "untrained", "absent"]
    heads: NetworkHeads

    @model_validator(mode="after")
    def validate_head_contract(self) -> "NetworkOutput":
        shapes = {
            "policy": [self.node_count],
            "outcome": [2],
            "score_margin": [303],
            "ownership": [self.node_count, 3],
            "alive": [self.node_count],
            "soft_policy": [self.node_count],
            "opponent_reply": [self.node_count + 1],
            "second_stone": [self.node_count],
            "final_shores": [2, 51],
            "final_networks": [2, 26],
            "final_capes": [2, 6],
        }
        for index, (name, shape) in enumerate(shapes.items()):
            head = getattr(self.heads, name)
            absent = index >= 6 and self.auxiliary_status == "absent"
            if absent:
                if head is not None:
                    raise ValueError("absent auxiliary head cannot have values")
                continue
            if head is None or head.shape != shape:
                raise ValueError("network head has an incompatible shape")
            if head.activation != ("sigmoid" if name == "alive" else "softmax"):
                raise ValueError("network head activation is incompatible")
        return self


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3]
    request_id: str
    action: AtomicAction
    root_actions: list[AtomicAction]
    root_policy: list[Probability]
    root_q: list[UnitValue]
    root_visits: list[NonnegativeInt]
    outcome: OutcomeBelief
    value: UnitValue
    search_value: UnitValue
    # Visit-weighted search value of the root (the opener's optimal-swap
    # payoff while the pie decision is pending).
    root_value: UnitValue
    score_belief: ScoreBelief
    variant: VariantDescriptor
    swap_available: bool
    # True when the responder should take the pie swap instead of ``action``.
    swap_recommended: bool
    history_known: bool
    model_version: str
    model_step: NonnegativeInt
    timing_ms: Timing
    predictions: AuxiliaryPredictions | None = None
    network_output: NetworkOutput | None = None

    @model_validator(mode="after")
    def validate_response_shapes(self) -> "AnalyzeResponse":
        if self.swap_recommended and not self.swap_available:
            raise ValueError("a swap can only be recommended while it is available")
        if (
            self.swap_recommended
            and self.predictions is not None
            and self.predictions.second_stone is not None
        ):
            raise ValueError(
                "a swap ends the turn and cannot have a second-stone forecast"
            )
        if self.swap_available and not self.variant.pie:
            raise ValueError("swaps are available only in pie games")
        width = len(self.root_actions)
        if width == 0 or not (
            len(self.root_policy) == len(self.root_q) == len(self.root_visits) == width
        ):
            raise ValueError("root action statistics have inconsistent shapes")
        if self.action not in self.root_actions:
            raise ValueError("selected action must appear in root_actions")
        if any(visit < 0 for visit in self.root_visits):
            raise ValueError("root visits must be non-negative")
        if not math.isclose(sum(self.root_policy), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("root policy must sum to one")
        expected_value = self.outcome.win - self.outcome.loss
        if not math.isclose(self.value, expected_value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("value must equal P(win)-P(loss)")
        return self
