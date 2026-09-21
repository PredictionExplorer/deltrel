import argparse
import json
from pathlib import Path

import pytest
import yaml

from scripts.benchmark_replay_refresh_cost import (
    freeze_selection,
    load_selection,
    measure_arm,
    parse_arm,
)
from deltreltrain.config import load_config
from test_replay_game_revisions import publication as publication, _samples


@pytest.mark.parametrize("value", ["0:2", "4:0", "33:2", "4:9", "4", "4:2:1"])
def test_loader_arms_are_bounded(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_arm(value)


def prepare(publication, tmp_path):
    store, identity, generation, kwargs = publication
    record = store.append_game_revision(
        _samples(identity, generation, 16), finalized=False, **kwargs
    ).record
    identity.path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": identity.run_id,
                "generation_family": identity.generation_family,
                "created_ns": 1,
            }
        )
    )
    (tmp_path / "learner").mkdir()
    (tmp_path / "learner/recovery.json").write_text(
        json.dumps({"step": 10, "epoch": 0})
    )
    config_path = tmp_path / "config.yaml"
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    config_path.write_text(yaml.safe_dump(config.as_dict()))
    output = tmp_path / "frozen"
    before = store.connection.total_changes
    freeze_selection(config_path, store.root, output, 16)
    assert store.connection.total_changes == before
    return record, output, config_path


def test_freeze_keeps_payload_alive_after_production_gc(publication, tmp_path):
    record, directory, _ = prepare(publication, tmp_path)
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 16, final=True), finalized=True, **kwargs
    )
    store.collect_garbage(
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        retain_shards_per_ring=1,
        dry_run=False,
    )
    assert not record.path.exists()
    selection, plan = load_selection(directory)
    assert selection.sample_count == 16 and plan["ring"] == 4
    assert selection.spans[0].record.path.is_file()
    assert selection.spans[0].record.shard_id == record.shard_id


def test_freeze_rejects_production_output_and_existing_artifacts(publication, tmp_path):
    _, output, config = prepare(publication, tmp_path)
    store = publication[0]
    for destination in (output, store.root / "bad"):
        with pytest.raises(ValueError):
            freeze_selection(config, store.root, destination, 16)


def test_frozen_metadata_cannot_escape_owned_directory(publication, tmp_path):
    _, directory, _ = prepare(publication, tmp_path)
    path = directory / "selection.json"
    value = json.loads(path.read_text())
    value["spans"][0]["record"]["path"] = str(tmp_path / "outside.npz")
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="escapes"):
        load_selection(directory)


def test_real_persistent_pool_rebinds_same_rows_and_exposes_speculation(
    publication, tmp_path
):
    _, directory, _ = prepare(publication, tmp_path)
    selection, _ = load_selection(directory)
    reports = [
        measure_arm(
            selection,
            workers=1,
            prefetch=prefetch,
            batch_size=2,
            batches=8,
            consume=2,
            cycles=2,
            pace_seconds=0.01,
            pin_memory=False,
        )
        for prefetch in (1, 2)
    ]
    assert [x["first_features_sha256"] for x in reports[0]["cycles"]] == [
        x["first_features_sha256"] for x in reports[1]["cycles"]
    ]
    for report in reports:
        assert report["cycles"][0]["worker_pids"] == report["cycles"][1]["worker_pids"]
        for cycle in report["cycles"]:
            assert cycle["consumed_batches"] == 2
            assert 0 <= cycle["unconsumed_issued_batches"] <= report["prefetch_factor"]
            assert cycle["quiesce_seconds"] >= 0
        for pid in report["cycles"][0]["worker_pids"]:
            # The helper owns and shuts down every pool, including cold starts.
            assert not Path(f"/proc/{pid}").exists()
