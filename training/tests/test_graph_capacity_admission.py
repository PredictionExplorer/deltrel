from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import pytest
import yaml

from startrain.config import ActorPipelineConfig
from startrain import search_allocation_gate as gate
from test_graph_cache_capacity_benchmark import fixture_report
from test_search_allocation_gate import fixture, reference, update_plan, write_json


def graph_admission_fixture(tmp_path, monkeypatch, *, root=None, **kwargs):
    base, target, root, envelope = fixture(tmp_path, monkeypatch, root=root, **kwargs)
    inference = replace(
        base.orchestration.model_refresh.inference,
        cuda_graph_max_entries=16,
        cuda_graph_max_bytes=48 * 1024**3,
        small_batch_graph_buckets=True,
    )
    gpus = tuple(
        replace(
            gpu, actor_cohorts=4, actor_pipeline=ActorPipelineConfig(cuda_graphs=True)
        )
        if gpu.role == "actor"
        else gpu
        for gpu in base.orchestration.gpus
    )
    orchestration = replace(
        base.orchestration,
        gpus=gpus,
        model_refresh=replace(base.orchestration.model_refresh, inference=inference),
    )
    base = replace(base, orchestration=orchestration)
    target = replace(
        target,
        orchestration=replace(
            orchestration,
            model_refresh=replace(
                orchestration.model_refresh,
                inference=replace(inference, cuda_graph_max_entries=32),
            ),
        ),
    )
    baseline_path = root / envelope["baseline_profile"]["path"]
    baseline_path.write_text(yaml.safe_dump(base.as_dict()))
    envelope["baseline_profile"] = reference(root, baseline_path)
    raw_sha = envelope["baseline_profile"]["sha256"]
    canonical = hashlib.sha256(
        json.dumps(base.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actor = next(gpu for gpu in gpus if gpu.role == "actor")
    envelope["graph_cache_reports"] = []
    template, _ = fixture_report()
    for group in envelope["groups"]:
        selection_path = root / group["selection"]["path"]
        selection = json.loads(selection_path.read_text())
        selection["config_sha256"] = raw_sha
        write_json(selection_path, selection)
        group["selection"] = reference(root, selection_path)
        search_path = root / group["report"]["path"]
        search = json.loads(search_path.read_text())
        search["plan"].update(
            config_sha256=raw_sha,
            profile_inference=asdict(inference),
            selection_sha256=group["selection"]["sha256"],
        )
        update_plan(search)
        write_json(search_path, search)
        group["report"] = reference(root, search_path)
        model = json.loads((root / group["model_manifest"]["path"]).read_text())
        report = deepcopy(template)
        report["plan"].update(
            config_sha256=raw_sha,
            source_config_sha256=raw_sha,
            source_config_canonical_sha256=canonical,
            model_identity=model["model_identity"],
            checkpoint_sha256=model["checkpoint_sha256"],
            manifest_sha256=group["model_manifest"]["sha256"],
            profile_inference=asdict(replace(inference, cuda_graphs=True)),
            inference_configuration=asdict(replace(inference, cuda_graphs=True)),
            actor_gpu_id=actor.gpu_id,
            actor_configuration=asdict(actor),
            precision=base.train.precision,
            compile=base.train.compile,
            compile_dynamic=base.orchestration.model_refresh.inference_compile_dynamic,
            compile_mode=base.orchestration.model_refresh.inference_compile_mode,
        )
        report["plan_sha256"] = hashlib.sha256(
            json.dumps(report["plan"], sort_keys=True).encode()
        ).hexdigest()
        report_path = root / f"status/graph-cache-{group['role']}.json"
        write_json(report_path, report)
        envelope["graph_cache_reports"].append(reference(root, report_path))
    envelope["target_config_sha256"] = gate.canonical_config_sha256(target)
    write_json(gate.allocation_gate_path(target), envelope)
    return base, target, root, envelope


def test_exact_capacity_change_requires_both_original_quality_and_new_execution_evidence(
    tmp_path, monkeypatch
):
    _, target, root, envelope = graph_admission_fixture(tmp_path, monkeypatch)
    gate.validate_production_ring_allocations(target)
    # A cached pass must continue monitoring the newly required report bytes.
    report_path = root / envelope["graph_cache_reports"][0]["path"]
    report_path.write_text(report_path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "model",
        "bytes",
        "prediction",
        "topology",
        "dynamic",
        "mode",
        "precision",
    ],
)
def test_capacity_evidence_cannot_authorize_other_inference_changes(
    tmp_path, monkeypatch, change
):
    _, target, root, envelope = graph_admission_fixture(tmp_path, monkeypatch)
    if change == "missing":
        envelope["graph_cache_reports"].pop()
    elif change == "duplicate":
        envelope["graph_cache_reports"][1] = envelope["graph_cache_reports"][0]
    elif change in ("model", "precision"):
        path = root / envelope["graph_cache_reports"][0]["path"]
        report = json.loads(path.read_text())
        if change == "model":
            report["plan"]["model_identity"] = "sha256-" + "f" * 64
        else:
            report["plan"]["precision"] = "fp32"
        report["plan_sha256"] = hashlib.sha256(
            json.dumps(report["plan"], sort_keys=True).encode()
        ).hexdigest()
        write_json(path, report)
        envelope["graph_cache_reports"][0] = reference(root, path)
    elif change == "topology":
        changed = tuple(
            replace(gpu, actor_cohorts=3) if gpu.role == "actor" else gpu
            for gpu in target.orchestration.gpus
        )
        target = replace(
            target, orchestration=replace(target.orchestration, gpus=changed)
        )
    elif change in ("dynamic", "mode"):
        refresh = target.orchestration.model_refresh
        refresh = replace(
            refresh,
            **(
                {"inference_compile_dynamic": not refresh.inference_compile_dynamic}
                if change == "dynamic"
                else {"inference_compile_mode": "reduce-overhead"}
            ),
        )
        target = replace(
            target, orchestration=replace(target.orchestration, model_refresh=refresh)
        )
    else:
        inference = target.orchestration.model_refresh.inference
        inference = replace(
            inference,
            **(
                {"cuda_graph_max_bytes": 49 * 1024**3}
                if change == "bytes"
                else {"deduplicate": not inference.deduplicate}
            ),
        )
        target = replace(
            target,
            orchestration=replace(
                target.orchestration,
                model_refresh=replace(
                    target.orchestration.model_refresh, inference=inference
                ),
            ),
        )
    envelope["target_config_sha256"] = gate.canonical_config_sha256(target)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError):
        gate.validate_production_ring_allocations(target)


def test_disabled_clinch_and_old_capacity_keep_original_gate_authority(
    tmp_path, monkeypatch
):
    _, target, _, _ = fixture(tmp_path, monkeypatch)
    gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize("reports", [None, [], {}])
def test_unchanged_capacity_rejects_any_unused_report_extension(
    tmp_path, monkeypatch, reports
):
    _, target, _, envelope = fixture(tmp_path, monkeypatch)
    envelope["graph_cache_reports"] = reports
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="require a cache-capacity treatment"):
        gate.validate_production_ring_allocations(target)
