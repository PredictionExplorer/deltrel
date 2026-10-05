from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from deltreltrain import champion_migration as migration
from deltreltrain.adaptive_promotion import next_allocation, plan_targets
from deltreltrain.arena import ArenaGame, ArenaPair, summarize_completed_arena_pairs
from deltreltrain.balanced_evaluation import balanced_opening_seed, cell_variant
from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    load_model_manifest,
    save_checkpoint,
    sha256_file,
)
from deltreltrain.config import ArenaConfig, load_config
from deltreltrain.config_compatibility import without_search_execution_defaults
from deltreltrain.gradient_clipping import GradientClipper, GradientClippingConfig
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.rebrand import SOURCE_FEATURE_HASH, SOURCE_RULES_HASH, _routing_hash
from deltreltrain.training import build_scheduler


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _rename(value: Any) -> Any:
    replacements = {
        new + suffix: old + suffix
        for new, old in (
            ("final_shores", "final_coast"),
            ("final_networks", "final_channels"),
            ("final_capes", "final_corners"),
        )
        for suffix in ("", "_head", "_head.weight", "_head.bias")
    }
    if isinstance(value, dict):
        return {
            replacements.get(key, key): _rename(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rename(item) for item in value)
    if isinstance(value, str):
        return replacements.get(value, value.replace("deltreltrain.", "startrain."))
    return value


@dataclass
class Publication:
    pointer: Path
    proof: Path
    checkpoint: Path
    config: Any

    def rebind(self) -> dict[str, str]:
        """Re-pin synthetic evidence so negative tests exercise semantics, too."""
        docs = {
            p.name: json.loads(p.read_text())
            for p in self.proof.glob("*.json")
            if p.name
            not in (
                "manifest.json",
                "assessment.json",
                "verification.json",
                "capture.json",
            )
        }
        source_paths = {
            "published-champion-pointer.json": self.pointer,
            "candidate-manifest.json": self.pointer.parent
            / docs["published-champion-pointer.json"]["manifest"],
            "result.json": self.pointer.parent.parent / "arena/fixture.json",
        }
        capture = {
            "source_commit": "1" * 40,
            "files": [
                {
                    "path": name,
                    "source": str(source_paths.get(name, self.proof / name)),
                    "sha256": sha256_file(self.proof / name),
                    "bytes": (self.proof / name).stat().st_size,
                }
                for name in sorted(docs)
            ],
        }
        capture["sha256"] = hashlib.sha256(_bytes(capture).rstrip(b"\n")).hexdigest()
        _write(self.proof / "capture.json", capture)
        result = docs["result.json"]
        games = len(result["games"])
        verification = {
            "capture_sha256": capture["sha256"],
            "source_commit": capture["source_commit"],
            "published_champion_pointer": docs["published-champion-pointer.json"],
            "promotion": result["promotion"],
            "pairs": len(result["pairs"]),
            "games": games,
            "resume_scope": {
                "boundary_finished_games": games,
                "finished_games_checked": games,
                "game_states": games,
                "unfinished_games": 0,
            },
        }
        _write(self.proof / "verification.json", verification)
        _write(
            self.proof / "assessment.json",
            {
                "status": "VERIFIED_CONCLUSIVE_PROMOTION",
                "verification_sha256": sha256_file(self.proof / "verification.json"),
            },
        )
        inventory = {
            "schema_version": 1,
            "status": "frozen",
            "files": [
                {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in sorted(self.proof.glob("*.json"))
                if p.name != "manifest.json"
            ],
        }
        _write(self.proof / "manifest.json", inventory)
        return {
            "expected_pointer_sha256": sha256_file(self.pointer),
            "expected_proof_manifest_sha256": sha256_file(self.proof / "manifest.json"),
            "expected_contract_identity": result["evaluation_contract"]["identity"],
        }

    def migrate(self, destination: Path) -> dict[str, Any]:
        return migration.migrate_verified_champion(
            self.pointer, self.proof, destination, **self.rebind()
        )


@pytest.fixture
def publication(tmp_path: Path) -> Publication:
    torch.set_num_threads(1)
    experiment = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    experiment = replace(
        experiment,
        model=replace(
            experiment.model,
            width=8,
            rrt_groups=1,
            attention_heads=2,
            kv_heads=1,
            auxiliary_predictions=True,
        ),
        optimizer=replace(
            experiment.optimizer, kind="muon_adamw", fallback_to_adamw=False
        ),
        train=replace(experiment.train, compile=False),
    )
    torch.manual_seed(41)
    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, experiment.optimizer)
    scheduler = build_scheduler(optimizer, experiment.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=0.99)
    clipper = GradientClipper(
        model.named_parameters(), config=GradientClippingConfig(mode="adagc")
    )
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.01 + index / 10000)
    clipper.apply_(clipper.measure())
    optimizer.step()
    scheduler.step()
    # EMA intentionally remains distinct from the updated raw weights.
    initial = tmp_path / "raw.pt"
    save_checkpoint(
        initial,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
        step=7,
        epoch=3,
        config=experiment.as_dict(),
        extra={
            "run_id": "test-run",
            "generation_family": "test-family",
            "examples_consumed": 448,
            "global_batch_size": 64,
            "utd_segment": {"version": 1, "start_step": 0},
        },
    )
    payload = _rename(torch.load(initial, weights_only=True))
    payload.update(
        rules_schema="edgeconnect.star.rules.v3",
        rules_hash=SOURCE_RULES_HASH,
        rules_hash_wire=f"fnv1a64:{SOURCE_RULES_HASH:016x}",
        feature_schema_hash=SOURCE_FEATURE_HASH,
        action_layout_schema="edgeconnect.star.action-layout.nodes-only.v1",
    )
    payload["optimizer_routing"]["routing_hash"] = _routing_hash(
        payload["optimizer_routing"], payload["model"]
    )
    assert {group["algorithm"] for group in payload["optimizer_routing"]["groups"]} == {
        "muon",
        "adamw",
    }
    root = tmp_path / "legacy/learner"
    (root / "checkpoints").mkdir(parents=True)
    staged = root / "checkpoints/temporary.pt"
    torch.save(payload, staged)
    checkpoint_hash = sha256_file(staged)
    checkpoint = staged.with_name(f"sha256-{checkpoint_hash}.pt")
    staged.rename(checkpoint)
    manifest = {
        "format": "startrain.model-manifest",
        "schema_version": 3,
        "model_identity": "sha256-" + checkpoint_hash,
        "model_version": "sha256-" + checkpoint_hash,
        "model_step": 7,
        "checkpoint": "../checkpoints/" + checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "run_id": "test-run",
        "generation_family": "test-family",
        "weights": "ema",
        "model_schema_version": 3,
        "rules_hash": f"fnv1a64:{SOURCE_RULES_HASH:016x}",
        "feature_schema_hash": f"{SOURCE_FEATURE_HASH:016x}",
        "created_ns": 100,
    }
    manifest_hash = hashlib.sha256(_bytes(manifest)).hexdigest()
    manifest_path = root / "manifests" / f"manifest-{manifest_hash}.json"
    _write(manifest_path, manifest)
    pointer = {
        "format": "startrain.model-pointer",
        "schema_version": 2,
        "role": "champion",
        "manifest": "manifests/" + manifest_path.name,
        "manifest_sha256": manifest_hash,
        "manifest_bytes": manifest_path.stat().st_size,
        "model_identity": manifest["model_identity"],
        "model_step": 7,
        "run_id": "test-run",
        "generation_family": "test-family",
        "promotion_result": "../arena/fixture.json",
        "updated_ns": 200,
    }
    _write(root / "champion.json", pointer)
    baseline = {
        **manifest,
        "model_identity": "sha256-" + "0" * 64,
        "model_version": "sha256-" + "0" * 64,
        "checkpoint_sha256": "0" * 64,
        "checkpoint": "../checkpoints/sha256-" + "0" * 64 + ".pt",
        "model_step": 6,
    }
    config = ArenaConfig(
        rings=(10,),
        balanced_cells=True,
        variant_policy="pie_even",
        allocation_policy="adaptive_pie",
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=40,
        continuation_pairs_per_ring=4,
    )
    pairs: list[ArenaPair] = []
    history = []
    previous = None
    while True:
        summary = cast(
            dict[str, Any],
            summarize_completed_arena_pairs(
                pairs,
                config,
                completed_allocation_targets=plan_targets(previous, config)
                if previous
                else None,
            ),
        )
        old_contract = dict(summary["evaluation_contract"])
        old_contract.update(
            rules_hash=manifest["rules_hash"], rules_schema="edgeconnect.star.rules.v3"
        )
        old_contract.pop("identity")
        old_contract["identity"] = (
            "sha256-" + hashlib.sha256(_bytes(old_contract).rstrip(b"\n")).hexdigest()
        )
        summary["evaluation_contract"] = old_contract
        if summary["promotion"]["decision"] == "promote":
            break
        plan = next_allocation(config, previous_plan=previous, summary=summary)
        assert plan is not None
        record = {
            **plan,
            "plan_index": len(history),
            "decision_pairs": [asdict(pair) for pair in pairs],
            "decision_pairs_sha256": hashlib.sha256(
                _bytes([asdict(pair) for pair in pairs]).rstrip(b"\n")
            ).hexdigest(),
            "decision_summary": summary,
        }
        history.append(record)
        pairs = []
        for cell, count in plan_targets(plan, config).items():
            category = cell.split("/", 1)[1]
            for index in range(count):
                variant = cell_variant(category, index, config)
                pairs.append(
                    ArenaPair(
                        ring=10,
                        pair=index,
                        opening_seed=balanced_opening_seed(
                            config.seed, 10, variant, index
                        ),
                        opening_action=None,
                        forced_opening=False,
                        outcomes=(1, 1),
                        variant=variant.label,
                        segment=variant.segment,
                    )
                )
        previous = plan
    games = [
        asdict(
            ArenaGame(
                ring=pair.ring,
                pair=pair.pair,
                candidate_player=seat,
                opening_seed=pair.opening_seed,
                opening_action=None,
                forced_opening=False,
                winner=seat,
                outcome=1,
                searched_moves=0,
                variant=pair.variant,
                segment=pair.segment,
            )
        )
        for pair in pairs
        for seat in (0, 1)
    ]
    summary["promotion"]["allocation_boundary_complete"] = True
    result = {
        **summary,
        "candidate": manifest["model_identity"],
        "baseline": baseline["model_identity"],
        "candidate_step": 7,
        "champion_step": 6,
        "result_kind": "promotion",
        "terminal": True,
        "conclusive": True,
        "pairs": [asdict(pair) for pair in pairs],
        "games": games,
    }
    common = {
        "run_id": "test-run",
        "generation_family": "test-family",
        "candidate_identity": result["candidate"],
        "baseline_identity": result["baseline"],
    }
    state = {
        "candidate": result["candidate"],
        "baseline": result["baseline"],
        "config": without_search_execution_defaults({"arena": asdict(config)})["arena"],
        "games": games,
        "pairs": result["pairs"],
        "game_states": [{"result": game, "actions": []} for game in games],
    }
    proof = tmp_path / "proof"
    for name, value in {
        "result.json": result,
        "allocation.json": {
            **common,
            "evaluation_contract": old_contract,
            "plan_history": history,
        },
        "resume.json": {**common, "arena_state": state},
        "run.json": {"run_id": "test-run", "generation_family": "test-family"},
        "candidate-manifest.json": manifest,
        "champion-manifest.json": baseline,
        "published-champion-pointer.json": pointer,
    }.items():
        _write(proof / name, value)
    fixture = Publication(root / "champion.json", proof, checkpoint, experiment)
    fixture.rebind()
    return fixture


def test_migration_preserves_mixed_optimizer_ema_counters_and_relocatable_proof(
    publication: Publication, tmp_path: Path
) -> None:
    original = publication.checkpoint.read_bytes()
    proof_bytes = {p.name: p.read_bytes() for p in publication.proof.iterdir()}
    result = publication.migrate(tmp_path / "published")
    output = Path(result["destination"])
    manifest = load_model_manifest(output / "champion.json")
    assert manifest.model_step == 7 and manifest.role == "champion"
    assert manifest.model_identity != result["source_model_identity"]
    pointer = json.loads((output / "champion.json").read_text())
    assert "promotion_result" not in pointer
    raw = torch.load(publication.checkpoint, weights_only=True)
    migrated = torch.load(manifest.checkpoint, weights_only=True)
    assert migrated["step"] == 7 and migrated["epoch"] == 3
    assert migrated["scheduler"] == raw["scheduler"]
    assert migrated["extra"]["examples_consumed"] == 448
    assert migrated["extra"]["utd_segment"] == raw["extra"]["utd_segment"]
    assert any(
        not torch.equal(raw["model"][name], raw["ema"]["shadow"][name])
        for name in raw["model"]
    )
    model = GraphResTNet(publication.config.model)
    optimizer = build_optimizer(model, publication.config.optimizer)
    scheduler = build_scheduler(optimizer, publication.config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=0.99)
    clipper = GradientClipper(
        model.named_parameters(), config=GradientClippingConfig(mode="adagc")
    )
    load_checkpoint(
        manifest.checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
    )
    assert clipper.updates == 1
    for before, after in zip(
        raw["model"].values(), model.state_dict().values(), strict=True
    ):
        assert torch.equal(before, after)
    for before, after in zip(
        raw["ema"]["shadow"].values(), ema.shadow.values(), strict=True
    ):
        assert torch.equal(before, after)
    assert migration._equal_payload(raw["optimizer"], migrated["optimizer"]) > 0
    assert publication.checkpoint.read_bytes() == original
    for name, data in proof_bytes.items():
        assert (output / "provenance/proof" / name).read_bytes() == data
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    manifest = load_model_manifest(relocated / "champion.json")
    assert manifest.checkpoint.is_relative_to(relocated)
    old_pointer = relocated / "provenance/original/learner/champion.json"
    payload = json.loads(old_pointer.read_text())
    assert (
        old_pointer.parent / payload["promotion_result"]
    ).resolve().read_bytes() == proof_bytes["result.json"]
    inventory = json.loads((relocated / "publication.json").read_text())
    for item in inventory["files"]:
        path = relocated / item["path"]
        assert (
            sha256_file(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        )
    receipt = json.loads(
        (relocated / inventory["migration_receipt"]["path"]).read_text()
    )
    assert receipt["source_model_identity"] == result["source_model_identity"]
    assert receipt["destination_model_identity"] == manifest.model_identity
    assert receipt["preservation"]["all_payload_leaves_equal_after_approved_mapping"]


@pytest.mark.parametrize(
    "case",
    [
        "candidate",
        "inconclusive",
        "wrong_candidate",
        "wrong_baseline",
        "wrong_contract",
        "altered_statistic",
        "veto",
        "verification_binding",
    ],
)
def test_wrong_publication_or_proof_fails_before_migration(
    publication: Publication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    result_path = publication.proof / "result.json"
    result = json.loads(result_path.read_text())
    if case == "candidate":
        pointer = json.loads(publication.pointer.read_text())
        pointer["role"] = "candidate"
        _write(publication.pointer, pointer)
        _write(publication.proof / "published-champion-pointer.json", pointer)
    elif case == "inconclusive":
        result["conclusive"] = False
        result["promotion"]["decision"] = "reject_max_pairs"
    elif case == "wrong_candidate":
        result["candidate"] = result["baseline"]
    elif case == "wrong_baseline":
        result["baseline"] = "sha256-" + "f" * 64
    elif case == "wrong_contract":
        result["evaluation_contract"]["simulations"] += 1
    elif case == "altered_statistic":
        result["promotion"]["statistical_test"]["promotion"]["e_value"] = (
            math.nextafter(
                result["promotion"]["statistical_test"]["promotion"]["e_value"],
                math.inf,
            )
        )
    elif case == "veto":
        result["promotion"]["cell_vetoes"] = ["r10/classic-pie"]
    _write(result_path, result)
    pins = publication.rebind()
    if case == "verification_binding":
        value = json.loads((publication.proof / "verification.json").read_text())
        value["capture_sha256"] = "f" * 64
        _write(publication.proof / "verification.json", value)
    monkeypatch.setattr(
        migration,
        "migrate_checkpoint",
        lambda *args: pytest.fail("invalid evidence reached migration"),
    )
    with pytest.raises(ValueError):
        migration.migrate_verified_champion(
            publication.pointer, publication.proof, tmp_path / "out", **pins
        )
    assert not (tmp_path / "out").exists()


def test_tampering_symlinks_and_existing_destination_are_rejected(
    publication: Publication, tmp_path: Path
) -> None:
    pins = publication.rebind()
    result = publication.proof / "result.json"
    data = result.read_bytes()
    result.write_bytes(data + b" ")
    with pytest.raises(ValueError):
        migration.migrate_verified_champion(
            publication.pointer, publication.proof, tmp_path / "out", **pins
        )
    result.write_bytes(data)
    result.rename(tmp_path / "outside.json")
    result.symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        migration.migrate_verified_champion(
            publication.pointer, publication.proof, tmp_path / "out", **pins
        )
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        migration.migrate_verified_champion(
            publication.pointer, publication.proof, target, **pins
        )
    assert (target / "keep").read_text() == "preserve"


@pytest.mark.parametrize(
    "failure",
    ["tensor_corruption", "source_race", "destination_race", "interrupt_commit"],
)
def test_faults_never_overwrite_sources_or_publish_partial_champion(
    publication: Publication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    pins = publication.rebind()
    source_hash = sha256_file(publication.checkpoint)
    target = tmp_path / "out"
    original_migrate = migration.migrate_checkpoint
    original_commit = migration._commit
    if failure in ("tensor_corruption", "source_race"):

        def faulty(source: Path, destination: Path) -> Path:
            result = original_migrate(source, destination)
            if failure == "tensor_corruption":
                payload = torch.load(destination, weights_only=True)
                next(iter(payload["ema"]["shadow"].values())).add_(1)
                torch.save(payload, destination)
            else:
                publication.pointer.write_bytes(publication.pointer.read_bytes() + b" ")
            return result

        monkeypatch.setattr(migration, "migrate_checkpoint", faulty)
    elif failure == "destination_race":

        def occupied(stage: Path, destination: Path) -> None:
            destination.mkdir()
            (destination / "keep").write_text("other writer")
            original_commit(stage, destination)

        monkeypatch.setattr(migration, "_commit", occupied)
    else:
        original_link = migration.os.link

        def interrupted(
            source: Any, destination: Any, *args: Any, **kwargs: Any
        ) -> None:
            if Path(destination) == target / "champion.json":
                assert (target / "publication.json").exists()
                assert not Path(destination).exists()
                raise KeyboardInterrupt
            original_link(source, destination, *args, **kwargs)

        monkeypatch.setattr(migration.os, "link", interrupted)
    with pytest.raises((ValueError, FileExistsError, KeyboardInterrupt)):
        migration.migrate_verified_champion(
            publication.pointer, publication.proof, target, **pins
        )
    assert sha256_file(publication.checkpoint) == source_hash
    assert not (target / "champion.json").exists()
    assert not list(tmp_path.glob(".out.migration-*"))
    if failure == "destination_race":
        assert (target / "keep").read_text() == "other writer"
    else:
        assert not target.exists()


def test_only_display_elo_allows_platform_rounding() -> None:
    number = 227.80629512192303
    rounded = math.nextafter(number, math.inf)
    assert migration._summary_equal(
        {"aggregate": {"elo_difference": number}},
        {"aggregate": {"elo_difference": rounded}},
    )
    assert migration._summary_equal(
        {"per_cell": {"r10/classic-pie": {"anytime_elo_interval": [number, None]}}},
        {"per_cell": {"r10/classic-pie": {"anytime_elo_interval": [rounded, None]}}},
    )
    assert not migration._summary_equal(
        {"evaluation_contract": {"elo_difference": number}},
        {"evaluation_contract": {"elo_difference": rounded}},
    )
    assert not migration._summary_equal({"score_rate": number}, {"score_rate": rounded})
    assert not migration._summary_equal({"e_value": number}, {"e_value": rounded})
    assert not migration._summary_equal(
        {"confidence_sequence": [number]}, {"confidence_sequence": [rounded]}
    )
    assert not migration._summary_equal(
        {"elo_difference": number}, {"elo_difference": number + 1e-8}
    )
    assert not migration._summary_equal(
        {"aggregate": {"elo_difference": 1.0}}, {"aggregate": {"elo_difference": True}}
    )
    assert not migration._summary_equal(
        {"aggregate": {"elo_difference": number}},
        {"aggregate": {"elo_difference": math.inf}},
    )


def test_failure_after_pointer_visibility_preserves_complete_publication(
    publication: Publication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out"
    original = migration._fsync_directory
    synchronized: list[Path] = []

    def fail_after_publication(path: Path) -> None:
        synchronized.append(path)
        if path == target and (target / "champion.json").exists():
            raise OSError("injected durability failure")
        original(path)

    monkeypatch.setattr(migration, "_fsync_directory", fail_after_publication)
    with pytest.raises(migration.PublicationDurabilityError, match="retained"):
        publication.migrate(target)
    assert load_model_manifest(target / "champion.json").model_step == 7
    assert target.parent in synchronized
    assert target / "provenance/proof" in synchronized
    inventory = json.loads((target / "publication.json").read_text())
    assert all(
        sha256_file(target / item["path"]) == item["sha256"]
        for item in inventory["files"]
    )


def test_output_inside_source_is_rejected_without_creating_directories(
    publication: Publication,
) -> None:
    target = publication.pointer.parent / "new-parent/new-output"
    with pytest.raises(ValueError, match="inside"):
        publication.migrate(target)
    assert not target.parent.exists()
