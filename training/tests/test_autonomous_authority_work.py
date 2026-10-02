"""Exact legacy authority without copying every configuration for every mask."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deltreltrain import config_compatibility, orchestration
from deltreltrain.config import load_config
from deltreltrain.runtime import load_or_create_run_identity


def parent_hashes(sources):
    """Frozen mask algorithm, including its original JSON normalization."""
    paths = (
        ("selfplay", "variants", "handicap_classic_share"),
        ("arena", "segment_handicap_classic_share"),
    )
    hashes = set()
    for source in sources:
        for mask in range(1 << len(paths)):
            payload = json.loads(json.dumps(source))
            for bit, path in enumerate(paths):
                if not mask & (1 << bit):
                    continue
                parent = payload
                for key in path[:-1]:
                    parent = parent[key]
                if type(parent[path[-1]]) is not float or parent[path[-1]] != 0.0:
                    break
                del parent[path[-1]]
            else:
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                hashes.add(hashlib.sha256(encoded.encode()).hexdigest())
    return hashes


@pytest.mark.parametrize("share", [0.0, 0, False, "0", None, 0.5, -0.0])
@pytest.mark.parametrize("side", ["selfplay", "arena"])
def test_autonomous_mask_copy_preserves_parent_hashes_and_sources(
    monkeypatch, share, side
):
    source = {
        "selfplay": {"variants": {"handicap_classic_share": 0.0}},
        "arena": {"segment_handicap_classic_share": 0.0},
        "future": {"values": [False, 0, 0.0, (1, 2)], 3: {"unknown": [4]}},
    }
    parent = source[side]["variants"] if side == "selfplay" else source[side]
    parent[
        "handicap_classic_share"
        if side == "selfplay"
        else "segment_handicap_classic_share"
    ] = share
    previous = deepcopy(source)
    previous["future"]["epoch"] = "previous"
    sources = (source, previous, deepcopy(source))
    before = deepcopy(sources)
    monkeypatch.setattr(
        config_compatibility, "compatible_config_epoch_payloads", lambda _: sources
    )
    config = SimpleNamespace(as_dict=lambda: source)
    assert orchestration._compatible_autonomous_config_sha256s(config) == parent_hashes(
        sources
    )
    assert sources == before
    assert isinstance(source["future"]["values"][-1], tuple)
    assert 3 in source["future"]


@pytest.mark.parametrize(
    "tamper",
    [
        {"external_weights": True},
        {"external_replay": True},
        {"external_positions": True},
        {"run_id": "another-run"},
        {"train_seed": -1},
        {"extra_authority": True},
    ],
)
def test_provenance_metadata_mismatch_never_enumerates_legacy_configs(
    tmp_path, monkeypatch, tamper
):
    configured = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-autonomous.yaml"
    )
    configured = replace(
        configured,
        orchestration=replace(
            configured.orchestration,
            directories=replace(
                configured.orchestration.directories, root=str(tmp_path)
            ),
        ),
    )
    directories = orchestration.RunDirectories.from_experiment(configured)
    directories.create()
    identity = load_or_create_run_identity(directories.run_identity)
    orchestration.ensure_autonomous_provenance(configured, directories, identity)
    payload = json.loads(directories.autonomous_provenance.read_text())
    payload.update(config_sha256="a" * 64, **tamper)
    original = json.dumps(payload, indent=4).encode()
    directories.autonomous_provenance.write_bytes(original)

    def forbidden(_):
        pytest.fail("non-config provenance cannot be repaired by a legacy config hash")

    monkeypatch.setattr(
        orchestration, "_compatible_autonomous_config_sha256s", forbidden
    )
    with pytest.raises(ValueError, match="frozen run profile"):
        orchestration.ensure_autonomous_provenance(configured, directories, identity)
    assert directories.autonomous_provenance.read_bytes() == original


def test_canonical_autonomous_epochs_keep_all_parent_hashes(monkeypatch):
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-autonomous.yaml"
    )
    sources = config_compatibility.compatible_config_epoch_payloads(config.as_dict())
    assert len(sources) == 1472
    monkeypatch.setattr(
        config_compatibility, "compatible_config_epoch_payloads", lambda _: sources
    )
    expected = parent_hashes(sources)
    actual = orchestration._compatible_autonomous_config_sha256s(config)
    assert len(actual) == 5888
    assert actual == expected
