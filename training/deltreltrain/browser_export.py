"""Export a verified trained champion for browser inference, without training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .auxiliary_upgrade import auxiliary_heads_ready
from .checkpoint import (
    extract_verified_manifest_config,
    load_ema_checkpoint,
    load_model_manifest,
    sha256_file,
    verify_file,
)
from .contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    MAX_HANDICAP,
    MODES,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)
from .distill import (
    BROWSER_MANIFEST_FORMAT,
    BROWSER_MANIFEST_SCHEMA_VERSION,
    BrowserSearchConfig,
    _browser_tensor_schema,
    validate_browser_onnx,
)
from .export import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    ONNX_AUXILIARY_OUTPUT_NAMES,
    ONNXDeltrelModel,
    export_onnx,
)
from .features import (
    DoubleDeltrelPosition,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    encode_batch,
)
from .model import GraphResTNet, MODEL_SCHEMA_VERSION
from .runtime import atomic_json
from .topology import SUPPORTED_RINGS


def validation_cases(
    fixture_path: Path,
) -> list[tuple[str, int, DoubleDeltrelPosition]]:
    """Actual legal traces cover all sizes, both modes, handicap, and pie swaps."""
    fixture = json.loads(fixture_path.read_text())
    if fixture["rules"]["hash"] != RULES_HASH_WIRE:
        raise ValueError("conformance fixture rules are incompatible")
    cases = []
    for game in fixture["games"]:
        config, states = game["config"], game["states"]
        for index in sorted({0, 1, 2, 3, len(states) - 2}):
            state = states[index]

            def mask(name: str) -> torch.Tensor:
                result = torch.zeros(len(state["stones"]), dtype=torch.bool)
                result[state[name]] = True
                return result

            position = DoubleDeltrelPosition(
                rings=config["rings"],
                stones=torch.tensor(state["stones"], dtype=torch.int8),
                to_move=state["toMove"],
                moves_left=state["movesLeft"],
                opening=state["opening"],
                terminal=False,
                mode=config["mode"],
                handicap=config["handicap"],
                pie=config["pieRule"],
                swap_available=state["canSwap"],
                swapped=state["swapped"],
                current_turn=mask("currentTurnMoves"),
                previous_turn=mask("previousTurnMoves"),
                own_previous_turn=mask("ownPreviousTurnMoves"),
                handicap_stones=mask("handicapStones"),
                history_known=True,
                pda=0,
            )
            cases.append((game["id"], index, position))
    return cases


def strip_export_debug_metadata(model_path: Path) -> None:
    """Remove nonsemantic exporter stack traces/local paths before publication."""
    import onnx

    model = onnx.load(model_path)

    def clean(message: Any) -> None:
        for field, value in list(message.ListFields()):
            if field.name in {"doc_string", "metadata_props"}:
                message.ClearField(field.name)
            elif field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else [value]:
                    clean(child)

    clean(model)
    onnx.save_model(model, model_path, save_as_external_data=False)


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    logits = values.astype(np.float64)
    if name == "alive_logits":
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def export_browser_champion(
    champion: str | Path,
    destination: str | Path,
    *,
    fixture_path: str | Path | None = None,
    simulations: int = 512,
    max_considered: int = 16,
) -> dict[str, Any]:
    """Create an immutable, fully verified single-file FP32 browser release.

    Source archives are read-only. The original EMA is verified against its
    content-addressed champion publication. Full inference precision is retained;
    no random replacement, optimizer step, distillation, or retraining occurs.
    """
    import onnxruntime as ort

    search = BrowserSearchConfig(
        simulations=simulations,
        max_considered=max_considered,
        maximum_simulations=max(4_096, simulations),
        maximum_max_considered=max(64, max_considered),
        score_utility_weight=0.05,
        seed_contract="native-search-batch-v1",
    )
    manifest = load_model_manifest(champion)
    if manifest.role != "champion":
        raise ValueError("direct browser export requires a published champion pointer")
    verified = extract_verified_manifest_config(manifest)
    model = GraphResTNet(verified.model).eval()
    metadata = load_ema_checkpoint(
        manifest.checkpoint,
        model=model,
        expected_model_config=verified.model_config,
        expected_game_config=verified.game_config,
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
    )
    if metadata["step"] != manifest.model_step:
        raise ValueError("champion step differs from its checkpoint")
    include_auxiliary = bool(verified.model.auxiliary_predictions)
    ready = auxiliary_heads_ready(verified.model, metadata)
    names = ONNX_OUTPUT_NAMES + (
        ONNX_AUXILIARY_OUTPUT_NAMES if include_auxiliary else ()
    )
    source_fixture = (
        Path(fixture_path)
        if fixture_path
        else Path(__file__).resolve().parents[2]
        / "testdata/deltrel/conformance-v3.json"
    )
    cases = validation_cases(source_fixture)
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(
            "browser releases are immutable; choose a new destination"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference_model = ONNXDeltrelModel(model, include_auxiliary=include_auxiliary)
        expected = []
        with torch.inference_mode():
            for _, _, position in cases:
                expected.append(
                    [
                        x.float().numpy().copy()
                        for x in reference_model(*encode_batch([position]).model_args())
                    ]
                )
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.", dir=target.parent
        ) as temporary:
            stage = Path(temporary)
            temporary_onnx = stage / "export.onnx"
            # Batch two prevents accidental exporter specialization to batch one.
            example = encode_batch([cases[0][2], cases[1][2]])
            export_onnx(
                model, example, temporary_onnx, include_auxiliary=include_auxiliary
            )
            strip_export_debug_metadata(temporary_onnx)
            validate_browser_onnx(
                temporary_onnx, include_auxiliary=include_auxiliary, precision="float32"
            )
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(temporary_onnx), options, providers=["CPUExecutionProvider"]
            )
            errors = {name: 0.0 for name in names}
            maximum_margin_error = 0.0
            timings = []
            for (game, index, position), reference in zip(cases, expected, strict=True):
                batch = encode_batch([position])
                feeds = {
                    name: tensor.numpy()
                    for name, tensor in zip(
                        ONNX_INPUT_NAMES, batch.model_args(), strict=True
                    )
                }
                started = time.perf_counter()
                actual = session.run(None, feeds)
                timings.append(1000 * (time.perf_counter() - started))
                for name, before, after in zip(names, reference, actual, strict=True):
                    if not isinstance(after, np.ndarray):
                        raise ValueError(f"exported {name} is not a dense tensor")
                    if after.shape != before.shape or not np.isfinite(after).all():
                        raise ValueError(
                            f"exported {name} shape/finiteness mismatch in {game}:{index}"
                        )
                    prior = _probabilities(before, name)
                    current = _probabilities(after, name)
                    error = float(np.max(np.abs(prior - current)))
                    errors[name] = max(errors[name], error)
                    if error > 0.0001:
                        raise ValueError(
                            f"FP32 probability parity exceeded for {name}: {error}"
                        )
                    if name == "score_margin_logits":
                        support = np.arange(-151, 152)
                        maximum_margin_error = max(
                            maximum_margin_error,
                            float(abs(((prior - current) * support).sum())),
                        )
            if maximum_margin_error > 0.005:
                raise ValueError("FP32 expected margin drift exceeds 0.005 points")
            # Dynamic batch equivalence is part of the browser inference contract.
            pair = encode_batch([cases[0][2], cases[1][2]])
            pair_inputs = {
                name: tensor.numpy()
                for name, tensor in zip(
                    ONNX_INPUT_NAMES, pair.model_args(), strict=True
                )
            }
            batched = session.run(None, pair_inputs)
            for row in range(2):
                single = session.run(
                    None,
                    {
                        name: values[row : row + 1]
                        for name, values in pair_inputs.items()
                    },
                )
                for combined, individual in zip(batched, single, strict=True):
                    if not isinstance(combined, np.ndarray) or not isinstance(
                        individual, np.ndarray
                    ):
                        raise ValueError(
                            "browser model batch outputs must be dense tensors"
                        )
                    np.testing.assert_allclose(
                        combined[row : row + 1], individual, atol=0.0001, rtol=0.0001
                    )
            onnx_sha = sha256_file(temporary_onnx)
            version = f"deltrel-champion-{manifest.model_step}-{onnx_sha[:12]}"
            model_path = stage / f"{version}.fp32.onnx"
            temporary_onnx.rename(model_path)
            checkpoint = stage / manifest.checkpoint.name
            shutil.copy2(manifest.checkpoint, checkpoint)
            verify_file(
                checkpoint,
                expected_sha256=manifest.checkpoint_sha256,
                expected_bytes=manifest.checkpoint_bytes,
            )
            validation = {
                "positions": len(cases),
                "maximum_probability_error": errors,
                "maximum_expected_margin_error": maximum_margin_error,
                "native_cpu_maximum_forward_ms": max(timings),
                "onnx_external_data": False,
            }
            payload = {
                "format": BROWSER_MANIFEST_FORMAT,
                "schema_version": BROWSER_MANIFEST_SCHEMA_VERSION,
                "model_version": version,
                "created_ns": time.time_ns(),
                "rules": {
                    "schema_id": RULES_SCHEMA_ID,
                    "hash": RULES_HASH_WIRE,
                    "mode": "double",
                    "pie_rule": False,
                    "handicap": 1,
                    "rings": list(SUPPORTED_RINGS),
                    "variants": {
                        "modes": list(MODES),
                        "handicap_min": 1,
                        "handicap_max": MAX_HANDICAP,
                        "pie_allowed": True,
                    },
                },
                "features": {
                    "schema_id": EXTERNAL_FEATURE_SCHEMA_ID,
                    "version": FEATURE_SCHEMA_VERSION,
                    "hash": f"{FEATURE_SCHEMA_HASH:016x}",
                    "node_feature_count": NODE_FEATURE_DIM,
                    "global_feature_count": GLOBAL_FEATURE_DIM,
                },
                "actions": {
                    "schema_id": ACTION_LAYOUT_SCHEMA_ID,
                    "types": ["place", "swap"],
                },
                "outcome": {"classes": ["loss", "win"], "value": "P(win)-P(loss)"},
                "architecture": {
                    "name": "GraphResTNet",
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "all_size": True,
                    "parameter_count": model.parameter_count(),
                    "config": asdict(verified.model),
                },
                "precision": "float32",
                "weights": "ema",
                "artifacts": {
                    "onnx": {
                        "file": model_path.name,
                        "sha256": onnx_sha,
                        "bytes": model_path.stat().st_size,
                        "opset": 18,
                    },
                    "checkpoint": {
                        "file": checkpoint.name,
                        "sha256": manifest.checkpoint_sha256,
                        "bytes": manifest.checkpoint_bytes,
                    },
                },
                "tensors": _browser_tensor_schema(
                    verified.model,
                    include_auxiliary=include_auxiliary,
                    precision="float32",
                ),
                "recommended_local_search": search.manifest_fields(),
                "training": {
                    "kind": "direct-champion-export",
                    "steps": manifest.model_step,
                    "source_model_identity": manifest.model_identity,
                    "source_checkpoint_sha256": manifest.checkpoint_sha256,
                    "auxiliary_predictions_ready": ready,
                    "validation": validation,
                },
            }
            atomic_json(stage / "browser.json", payload)
            stage.rename(target)
        verify_file(
            manifest.checkpoint,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
        )
        return {
            "manifest": str(target / "browser.json"),
            "onnx": str(target / model_path.name),
            "bytes": payload["artifacts"]["onnx"]["bytes"],
            "sha256": onnx_sha,
            "model_version": version,
            "validation": validation,
        }
    finally:
        torch.set_num_threads(thread_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conformance", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_browser_champion(
                args.champion, args.output, fixture_path=args.conformance
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
