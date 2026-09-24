from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from deltreltrain.config import load_config
from scripts.prepare_efficiency_profile import prepare_profile


def profile(tmp_path):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/h100-8gpu-throughput.yaml").read_text()
    )
    raw["orchestration"]["historical_evaluation"] = {
        "enabled": True,
        "measure_direct_predecessor": True,
    }
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(raw))
    return source


def test_preparer_preserves_training_and_search_and_publishes_no_live_state(tmp_path):
    source = profile(tmp_path)
    original_bytes = source.read_bytes()
    before = load_config(source)
    output = tmp_path / "prepared.yaml"
    result = prepare_profile(source, output)
    after = load_config(output)
    assert after.orchestration.historical_evaluation.measurement_service_fraction == 0.2
    assert after.orchestration.model_refresh.history_horizon_enabled
    assert (
        replace(
            after,
            orchestration=replace(
                after.orchestration,
                historical_evaluation=before.orchestration.historical_evaluation,
                model_refresh=before.orchestration.model_refresh,
            ),
        )
        == before
    )
    assert source.read_bytes() == original_bytes
    assert result["deployment_performed"] is False
    assert set(path.name for path in tmp_path.iterdir()) == {
        "source.yaml",
        "prepared.yaml",
    }
    with pytest.raises(FileExistsError):
        prepare_profile(source, output)


def test_preparer_refuses_disabled_independent_measurement(tmp_path):
    source = profile(tmp_path)
    raw = yaml.safe_load(source.read_text())
    raw["orchestration"]["historical_evaluation"]["enabled"] = False
    source.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="configured independent measurement"):
        prepare_profile(source, tmp_path / "prepared.yaml")
    assert not (tmp_path / "prepared.yaml").exists()
