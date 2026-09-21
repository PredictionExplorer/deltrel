"""Stable tensor-only inference and ONNX export surface."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .features import EncodedBatch
from .model import GraphResTNet
from .topology import MAX_NODES

ONNX_INPUT_NAMES = (
    "node_features",
    "global_features",
    "neighbor_index",
    "neighbor_mask",
    "neighbor_edge_type",
    "node_mask",
    "legal_action_mask",
    "rings",
)
ONNX_OUTPUT_NAMES = (
    "policy_logits",
    "outcome_logits",
    "score_margin_logits",
    "ownership_logits",
    "alive_logits",
    "soft_policy_logits",
)
ONNX_AUXILIARY_OUTPUT_NAMES = (
    "opponent_reply_logits",
    "second_stone_logits",
    "final_shores_logits",
    "final_networks_logits",
    "final_capes_logits",
)


class ONNXDeltrelModel(nn.Module):
    """Tuple-returning wrapper that avoids Python dataclasses at export time.

    The per-sample ``rings`` input selects the D5-invariant relation table of
    every board size inside the graph, so browser clients feed the same eight
    tensors the training encoders produce.
    """

    def __init__(self, model: GraphResTNet, *, include_auxiliary: bool = False) -> None:
        super().__init__()
        if include_auxiliary and not model.config.auxiliary_predictions:
            raise ValueError(
                "auxiliary export requires a model trained with those heads"
            )
        self.model = model
        self.include_auxiliary = include_auxiliary

    def forward(
        self,
        node_features: Tensor,
        global_features: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
        neighbor_edge_type: Tensor,
        node_mask: Tensor,
        legal_action_mask: Tensor,
        rings: Tensor,
    ) -> tuple[Tensor, ...]:
        output = self.model(
            node_features,
            global_features,
            neighbor_index,
            neighbor_mask,
            neighbor_edge_type,
            node_mask,
            legal_action_mask,
            rings,
            include_auxiliary=self.include_auxiliary,
        )
        # Six-output artifacts remain compatible. Direct champion publication
        # can retain every supervised auxiliary head without changing weights.
        return tuple(output if self.include_auxiliary else output[:6])  # type: ignore[return-value]


def export_onnx(
    model: GraphResTNet,
    example_batch: EncodedBatch,
    destination: str | Path,
    *,
    opset_version: int = 18,
    include_auxiliary: bool = False,
) -> Path:
    """Export variable batch/node axes using the stable ONNX exporter."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ONNXDeltrelModel(model, include_auxiliary=include_auxiliary)
    was_training = model.training
    wrapper.eval()
    batch = torch.export.Dim("batch")
    # Padded node axes never exceed the largest board; the bound lets the
    # relation-table gather export without a specialization guard.
    nodes = torch.export.Dim("nodes", max=MAX_NODES)
    degree = torch.export.Dim("degree")
    dynamic_shapes = (
        {0: batch, 1: nodes},
        {0: batch},
        {0: batch, 1: nodes, 2: degree},
        {0: batch, 1: nodes, 2: degree},
        {0: batch, 1: nodes, 2: degree},
        {0: batch, 1: nodes},
        {0: batch, 1: nodes},
        {0: batch},
    )
    try:
        # This artifact is inference-only. Do not capture the training-only
        # additive-mask gradient carrier or change parameter trainability.
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example_batch.model_args(),
                str(destination),
                input_names=list(ONNX_INPUT_NAMES),
                output_names=list(
                    ONNX_OUTPUT_NAMES
                    + (ONNX_AUXILIARY_OUTPUT_NAMES if include_auxiliary else ())
                ),
                dynamic_shapes=dynamic_shapes,
                opset_version=opset_version,
                dynamo=True,
                # Browsers verify/download one immutable model file. Sidecar
                # tensors would bypass that checksum and break atomic publish.
                external_data=False,
            )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("ONNX export requires the optional onnx package") from exc
    finally:
        model.train(was_training)
    return destination
