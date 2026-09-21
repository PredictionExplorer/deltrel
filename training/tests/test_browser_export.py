from dataclasses import replace
import json
from pathlib import Path

import onnx
import pytest
import torch

from deltreltrain.browser_export import export_browser_champion, validation_cases
from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    load_model_manifest,
    sha256_file,
    write_model_pointer,
)
from deltreltrain.config import load_config
from deltreltrain.distill import BrowserSearchConfig, validate_browser_onnx
from deltreltrain.learner import ImmutableModelPublisher
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.runtime import RunIdentity
from deltreltrain.training import build_scheduler


def champion_fixture(root: Path, auxiliary: bool) -> Path:
    experiment = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    experiment = replace(
        experiment,
        model=replace(
            experiment.model,
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
            auxiliary_predictions=auxiliary,
        ),
    )
    torch.manual_seed(71)
    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, experiment.optimizer)
    publisher = ImmutableModelPublisher(
        root / "publication",
        RunIdentity(root / "run.json", "browser-test", "browser-test-family", 1),
    )
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, experiment.train.scheduler),
        ema=ExponentialMovingAverage(model, decay=experiment.train.ema_decay),
        step=7,
        epoch=1,
        config=experiment.as_dict(),
    )
    pointer = publisher.root / "champion.json"
    write_model_pointer(
        pointer, candidate, role="champion", promotion_result="bootstrap"
    )
    return pointer


@pytest.mark.parametrize("auxiliary", [False, True])
def test_direct_champion_export_preserves_source_and_embeds_all_weights(
    tmp_path: Path, auxiliary: bool
) -> None:
    source = champion_fixture(tmp_path, auxiliary)
    champion = load_model_manifest(source)
    original = sha256_file(champion.checkpoint)
    result = export_browser_champion(source, tmp_path / "browser")
    assert sha256_file(champion.checkpoint) == original
    payload = json.loads(Path(result["manifest"]).read_text())
    assert payload["training"]["kind"] == "direct-champion-export"
    assert payload["training"]["steps"] == 7
    assert payload["training"]["source_checkpoint_sha256"] == original
    assert payload["training"]["auxiliary_predictions_ready"] == auxiliary
    assert payload["recommended_local_search"]["simulations"] == 8
    assert payload["recommended_local_search"]["maximum_simulations"] == 64
    assert payload["recommended_local_search"]["maximum_max_considered"] == 8
    assert len(payload["tensors"]["outputs"]) == (11 if auxiliary else 6)
    assert result["validation"]["positions"] == 50
    assert result["validation"]["maximum_expected_margin_error"] < 0.25
    exported = onnx.load(result["onnx"])
    assert not any(tensor.external_data for tensor in exported.graph.initializer)
    assert not exported.metadata_props
    assert all(
        not node.doc_string and not node.metadata_props for node in exported.graph.node
    )
    assert str(tmp_path).encode() not in Path(result["onnx"]).read_bytes()
    validate_browser_onnx(result["onnx"], include_auxiliary=auxiliary)
    assert not list((tmp_path / "browser").glob("*.data"))
    with pytest.raises(FileExistsError):
        export_browser_champion(source, tmp_path / "browser")


def test_export_rejects_unpublished_candidates_and_bad_search_limits(
    tmp_path: Path,
) -> None:
    source = champion_fixture(tmp_path, False)
    with pytest.raises(ValueError, match="published champion"):
        export_browser_champion(
            source.with_name("candidate.json"), tmp_path / "browser"
        )
    for fields in (
        {"maximum_simulations": 7, "simulations": 8},
        {"maximum_max_considered": 2, "max_considered": 4},
        {"maximum_simulations": 1025},
    ):
        with pytest.raises(ValueError):
            BrowserSearchConfig(**fields)


def test_parity_cases_cover_variant_transitions_and_near_full_positions() -> None:
    cases = validation_cases(
        Path(__file__).parents[2] / "testdata/deltrel/conformance-v3.json"
    )
    positions = [position for _, _, position in cases]
    assert {p.rings for p in positions} == {4, 6, 8, 10}
    assert {p.mode for p in positions} == {"classic", "double"}
    assert any(p.swap_available for p in positions)
    assert any(p.swapped for p in positions)
    assert any(
        p.handicap > 1 and p.opening and p.moves_left < p.handicap for p in positions
    )
    assert any(int((p.stones == -1).sum()) == 1 for p in positions)
