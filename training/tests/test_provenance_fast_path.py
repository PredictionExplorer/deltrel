"""Current provenance needs no enumeration of historical default combinations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import deltreltrain.orchestration as module
from deltreltrain.config import load_config
from deltreltrain.runtime import load_or_create_run_identity


def fixture(tmp_path):
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-autonomous.yaml"
    )
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            directories=replace(config.orchestration.directories, root=str(tmp_path)),
        ),
    )
    directories = module.RunDirectories.from_experiment(config)
    directories.create()
    identity = load_or_create_run_identity(directories.run_identity)
    module.ensure_autonomous_provenance(config, directories, identity)
    return config, directories, identity


def test_exact_current_provenance_skips_legacy_enumeration(tmp_path, monkeypatch):
    config, directories, identity = fixture(tmp_path)
    original = directories.autonomous_provenance.read_bytes()

    def unexpected(_config):
        raise AssertionError("matching current provenance needs no legacy hashes")

    monkeypatch.setattr(module, "_compatible_autonomous_config_sha256s", unexpected)
    module.ensure_autonomous_provenance(config, directories, identity)
    assert directories.autonomous_provenance.read_bytes() == original


@pytest.mark.parametrize("field", ["run_id", "train_seed", "external_weights"])
def test_current_hash_does_not_bypass_other_provenance_fields(
    tmp_path, monkeypatch, field
):
    config, directories, identity = fixture(tmp_path)
    payload = json.loads(directories.autonomous_provenance.read_text())
    payload[field] = "changed"
    directories.autonomous_provenance.write_text(json.dumps(payload))
    monkeypatch.setattr(
        module,
        "_compatible_autonomous_config_sha256s",
        lambda _: {payload["config_sha256"]},
    )
    with pytest.raises(ValueError, match="frozen run profile"):
        module.ensure_autonomous_provenance(config, directories, identity)


@pytest.mark.parametrize("recognized", [True, False])
def test_different_hash_still_uses_legacy_compatibility(
    tmp_path, monkeypatch, recognized
):
    config, directories, identity = fixture(tmp_path)
    payload = json.loads(directories.autonomous_provenance.read_text())
    payload["config_sha256"] = "legacy-hash"
    original = json.dumps(payload).encode()
    directories.autonomous_provenance.write_bytes(original)
    observed = []

    def compatible(value):
        observed.append(value)
        return {"legacy-hash"} if recognized else set()

    monkeypatch.setattr(module, "_compatible_autonomous_config_sha256s", compatible)
    if recognized:
        module.ensure_autonomous_provenance(config, directories, identity)
    else:
        with pytest.raises(ValueError, match="frozen run profile"):
            module.ensure_autonomous_provenance(config, directories, identity)
    assert observed == [config]
    assert directories.autonomous_provenance.read_bytes() == original
