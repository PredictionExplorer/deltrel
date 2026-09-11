"""Real native search parity and immutable replay selection for cost probes."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from scripts import benchmark_selfplay_search_cost as benchmark
from startrain.config import load_config
from startrain.native import positions_from_native
from startrain.replay import ReplaySample
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayConfig, VariantMixtureConfig
from test_native_inference_keys import adapter


@pytest.fixture
def native():
    return pytest.importorskip("star_native")


def sample(
    native,
    *,
    ring=4,
    mode="double",
    handicap=1,
    pie=False,
    placed=5,
    pda=0,
    game="game",
    generation=0,
):
    state = native.StateBatch(ring, 1, mode=mode, handicap=handicap, pie=pie)
    state.apply_many([0] * placed, list(range(placed)))
    position = positions_from_native(state.data(), pda=[pda])[0]
    policy = (position.stones.numpy() == -1).astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=None,
        search_provenance="fixture:final=pending-policy",
        policy_provenance="fixture-policy",
        run_id="run-test",
        generation_family="family-test",
        actor_id="actor-test",
        generation=generation,
        game_id=game,
        ply=0,
        model_identity="sha256-test",
    )


@pytest.mark.parametrize(
    "mode,handicap,pie,placed",
    [
        ("double", 1, False, 5),
        ("classic", 1, False, 8),
        ("double", 4, False, 9),
        ("classic", 3, False, 8),
        ("double", 1, True, 1),
        ("classic", 1, True, 20),
    ],
)
def test_semantic_import_preserves_every_input_and_pda(
    native, mode, handicap, pie, placed
):
    original = native.StateBatch(10, 1, mode=mode, handicap=handicap, pie=pie)
    original.apply_many([0] * placed, list(range(placed)))
    row = sample(
        native, ring=10, mode=mode, handicap=handicap, pie=pie, placed=placed, pda=-2
    )
    imported = benchmark.semantic_states(native, [row])
    for name in (
        "zero_bits",
        "one_bits",
        "to_move",
        "moves_left",
        "opening",
        "mode",
        "handicap",
        "pie",
        "swapped",
        "swap_available",
        "current_turn_bits",
        "previous_turn_bits",
        "own_previous_turn_bits",
        "handicap_bits",
    ):
        assert getattr(original.data(), name) == getattr(imported.data(), name), name
    side = row.to_move
    seats = [(-2, 2) if side == 0 else (2, -2)]
    left = native.SearchBatch(
        original, simulations=1, pda_by_seat=seats
    ).root_requests()
    right = native.SearchBatch(
        imported, simulations=1, pda_by_seat=seats
    ).root_requests()
    assert left.inference_keys() == right.inference_keys()


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
@pytest.mark.parametrize("pair", [(32, 384), (16, 192), (8, 96), (4, 48)])
def test_all_budget_pairs_preserve_handicap_ratio_across_rings(native, ring, pair):
    base = SelfPlayConfig(max_considered=32, max_considered_ring_exponent=1)
    for magnitude in range(4):
        left = sample(native, ring=ring, pda=magnitude)
        right = sample(native, ring=ring, pda=-magnitude)
        for full in (False, True):
            high, considered, _ = benchmark.budgets(base, left, pair, full)
            low, _, _ = benchmark.budgets(base, right, pair, full)
            assert high == low * 2**magnitude
            assert considered == replace(base, rings=ring).considered_actions()


def test_clipped_handicap_arm_is_rejected(native):
    row = sample(native, ring=10, pda=3)
    with pytest.raises(ValueError, match="clips"):
        benchmark.budgets(SelfPlayConfig(), row, (32, 128), True)


@pytest.mark.parametrize("max_rows", [3, 16])
def test_actual_batched_native_widths_preserve_results_and_global_cap(native, max_rows):
    samples = [
        sample(
            native,
            placed=5 + i,
            game=f"game-{i}",
            pda=(i % 3) - 1,
            handicap=3 if i % 2 else 1,
        )
        for i in range(8)
    ]
    base = SelfPlayConfig(
        simulation_ring_exponent=0,
        score_utility_weight=0.05,
        variants=VariantMixtureConfig(
            score_utility_weight_by_segment={"handicap": 0.25}
        ),
    )
    inference = adapter(cache_max_entries=128)
    try:
        baseline = benchmark.run_arm(
            native,
            inference,
            samples,
            base,
            (2, 16),
            full=True,
            width=1,
            max_rows=max_rows,
            seed=17,
            deadline=time.monotonic() + 30,
        )
        inference.model.rows.clear()
        prefetched = benchmark.run_arm(
            native,
            inference,
            samples,
            base,
            (2, 16),
            full=True,
            width=8,
            max_rows=max_rows,
            seed=17,
            deadline=time.monotonic() + 30,
        )
        assert baseline["positions"] == prefetched["positions"]
        assert all(count <= max_rows for count in inference.model.rows)
        assert any(count > 1 for count in inference.model.rows)
        assert (
            baseline["simulations"]
            == prefetched["simulations"]
            == sum(row["simulations"] for row in baseline["positions"])
        )
        # Same roots separately preserve each score-utility context and seed.
        for index, row in enumerate(samples):
            single = benchmark.run_arm(
                native,
                inference,
                [row],
                base,
                (2, 16),
                full=True,
                width=1,
                max_rows=max_rows,
                seed=17,
                deadline=time.monotonic() + 30,
            )
            assert single["positions"][0] == baseline["positions"][index]
    finally:
        inference.close()


def test_regret_explicitly_excludes_imputed_reference_q():
    reference = {
        "id": "a",
        "cell": [4, "standard-double", "early"],
        "actions": [1, 2, 3],
        "selected_action": 1,
        "selected_value": 0.4,
        "swap": False,
        "visits": [4, 0, 2],
        "q_values": [0.4, 0.99, 0.2],
        "policy_target": [0.4, 0.4, 0.2],
    }
    candidate = {**reference, "selected_action": 2}
    result = benchmark.compare(candidate, reference)
    assert not result["candidate_action_reference_visited"]
    assert result["reference_visited_q_gap"] is None
    assert result["reference_policy_mass_on_visited"] == pytest.approx(0.6)
    assessed = benchmark.compare({**candidate, "selected_action": 3}, reference)
    assert assessed["reference_visited_q_gap"] == pytest.approx(0.2)
    assert benchmark.compare(reference, reference)["policy_l1"] == 0
    summary = benchmark.summarize_comparisons([result, assessed])
    assert summary["reference_q_assessed_fraction"] == 0.5
    assert summary["mean_reference_visited_q_gap"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical"):
        benchmark.compare({**candidate, "actions": [3, 2, 1]}, reference)


def replay_fixture(tmp_path, native):
    root = tmp_path / "replay"
    identity = RunIdentity(tmp_path / "run.json", "run-test", "family-test", 1)
    with ReplayStore(root) as store:
        generation = store.lease_generation(identity, "actor-test")
        for label in benchmark.MODES:
            segment, mode = label.split("-")
            rows = [
                sample(
                    native,
                    mode=mode,
                    handicap=3 if segment == "handicap" else 1,
                    pie=segment == "pie",
                    placed=placed,
                    game=f"{label}-{placed}",
                    generation=generation,
                )
                for placed in (5, 20, 38)
            ]
            store.append(
                rows,
                phase_min=0,
                phase_max=0,
                model_version="sha256-test",
                model_identity="sha256-test",
                model_step=10,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id="actor-test",
                generation=generation,
            )
    config_file = tmp_path / "profile.yaml"
    config_file.write_text("immutable frozen fixture")
    config = load_config(Path("configs/small.yaml"))
    manifest = SimpleNamespace(
        model_step=10,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        model_identity="sha256-test",
        manifest_sha256="manifest-test",
    )
    args = SimpleNamespace(
        replay_root=root,
        output=tmp_path / "frozen",
        rings=[4],
        per_cell=1,
        max_shards=32,
        seed=17,
        config=config_file,
    )
    return args, config, manifest


def test_freeze_reads_actual_store_without_mutation_and_pins_balanced_samples(
    tmp_path, native
):
    args, config, manifest = replay_fixture(tmp_path, native)
    before = {
        str(path.relative_to(args.replay_root)): benchmark.digest(path.read_bytes())
        for path in args.replay_root.rglob("*")
        if path.is_file() and not path.name.endswith(("-shm", "-wal"))
    }
    report = benchmark.freeze_positions(args, config, manifest)
    samples, metadata = benchmark.read_frozen(
        args.output, benchmark.digest(args.config.read_bytes()), manifest
    )
    assert report["model_identity"] == metadata["model_identity"] == "sha256-test"
    assert len(samples) == 18
    assert len({benchmark.cell(row) for row in samples}) == 18
    assert metadata["missing_payloads_skipped"] == 0
    after = {
        str(path.relative_to(args.replay_root)): benchmark.digest(path.read_bytes())
        for path in args.replay_root.rglob("*")
        if path.is_file() and not path.name.endswith(("-shm", "-wal"))
    }
    assert before == after
    with pytest.raises(FileExistsError):
        benchmark.freeze_positions(args, config, manifest)
    with pytest.raises(ValueError, match="pinned"):
        benchmark.read_frozen(args.output, "changed-profile", manifest)
    next(args.output.glob("*.npz")).write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="pinned"):
        benchmark.read_frozen(
            args.output, benchmark.digest(args.config.read_bytes()), manifest
        )


def test_incomplete_or_corrupt_replay_never_publishes_completed_freeze(
    tmp_path, native
):
    args, config, manifest = replay_fixture(tmp_path, native)
    args.per_cell = 2
    with pytest.raises(ValueError, match="insufficient"):
        benchmark.freeze_positions(args, config, manifest)
    assert not args.output.exists()
    args.per_cell = 1
    next((args.replay_root / "shards").glob("*.npz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.freeze_positions(args, config, manifest)
    assert not args.output.exists()


def test_atomic_output_refuses_overwrite_and_dangling_symlink(tmp_path):
    output = tmp_path / "report.json"
    benchmark.publish_new(output, benchmark.json_bytes({"status": "passed"}))
    assert json.loads(output.read_bytes()) == {"status": "passed"}
    with pytest.raises(FileExistsError):
        benchmark.publish_new(output, b"overwritten")
    output.unlink()
    output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        benchmark.publish_new(output, b"overwritten")


def test_complete_worker_writes_finite_pinned_artifact_with_batched_real_native(
    tmp_path, native, monkeypatch, capsys
):
    args, config, manifest = replay_fixture(tmp_path, native)
    args.config = Path("configs/small.yaml")
    benchmark.freeze_positions(args, config, manifest)
    manifest.checkpoint_sha256 = "checkpoint-test"
    monkeypatch.setattr(benchmark, "load_model_manifest", lambda _: manifest)
    monkeypatch.setattr(
        benchmark,
        "load_manifest_evaluator",
        lambda *_, **__: adapter(cache_max_entries=128),
    )
    output = tmp_path / "report.json"
    arguments = [
        "run",
        "--config",
        str(args.config),
        "--checkpoint",
        "immutable.json",
        "--positions",
        str(args.output),
        "--pairs",
        "4:16",
        "2:8",
        "--widths",
        "1",
        "8",
        "--max-rows",
        "8",
        "--timeout-seconds",
        "30",
        "--output",
        str(output),
    ]
    assert benchmark.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert (
        benchmark.main(
            [*arguments, "--execute", "--worker", "--pinned-plan", plan["plan_sha256"]]
        )
        == 0
    )
    artifact = json.loads(output.read_bytes())
    assert artifact["status"] == "passed"
    assert len(artifact["results"]) == 33
    assert len(artifact["weighted_costs"]) == 4
    assert all(record["inference"]["neural_rows"] > 0 for record in artifact["results"])
    assert all(
        record["evaluator_batches"] < record["requested_rows"]
        for record in artifact["results"]
    )
    # Plan-only and execute shared exactly the same pinned input identities.
    assert artifact["plan_sha256"] == plan["plan_sha256"]
    baseline = next(
        record
        for record in artifact["results"]
        if record["stage"] == "measured"
        and record["raw_pair"] == [4, 16]
        and record["full"]
        and record["first_visit_batch_size"] == 1
    )
    assert baseline["quality"]["action_agreement"] == 1
    assert baseline["quality"]["mean_policy_l1"] == 0


@pytest.mark.parametrize(
    "change", ["model_step=11", "state='superseded'", "rules_hash='wrong'"]
)
def test_frozen_selection_respects_immutable_step_ready_and_contract_filters(
    tmp_path, native, change
):
    args, config, manifest = replay_fixture(tmp_path, native)
    import sqlite3

    with sqlite3.connect(args.replay_root / "manifest.sqlite3") as connection:
        connection.execute("UPDATE shards SET " + change)
    with pytest.raises(ValueError, match="insufficient"):
        benchmark.freeze_positions(args, config, manifest)
    assert not args.output.exists()


def actor_override_config(base):
    from startrain.config import ActorPipelineConfig, GPUWorkerConfig

    workers = (
        GPUWorkerConfig(gpu_id=0, role="learner", cpu_threads=1),
        GPUWorkerConfig(
            gpu_id=3,
            role="actor",
            cpu_threads=1,
            actor_batch_size=64,
            actor_pipeline=ActorPipelineConfig(cuda_graphs=True),
        ),
        GPUWorkerConfig(
            gpu_id=7,
            role="actor",
            cpu_threads=1,
            actor_batch_size=32,
            actor_pipeline=ActorPipelineConfig(cuda_graphs=False),
        ),
    )
    return replace(
        base,
        train=replace(base.train, compile=True),
        orchestration=replace(
            base.orchestration,
            gpus=workers,
            model_refresh=replace(
                base.orchestration.model_refresh,
                inference=replace(
                    base.orchestration.model_refresh.inference, cuda_graphs=False
                ),
            ),
        ),
    )


def test_profile_runtime_resolves_first_actor_and_explicit_gpu_pipeline_without_mutating_raw():
    base = actor_override_config(load_config(Path("configs/small.yaml")))
    effective, selected = benchmark.resolve_benchmark_config(base, None, "profile")
    assert selected.gpu_id == 3
    assert effective.orchestration.model_refresh.inference.cuda_graphs
    assert effective.train.compile is True
    assert not base.orchestration.model_refresh.inference.cuda_graphs
    assert effective.selfplay.full_simulations == base.selfplay.full_simulations
    assert effective.selfplay.variants == base.selfplay.variants
    assert effective.learner == base.learner
    assert effective.model == base.model
    other, selected = benchmark.resolve_benchmark_config(base, 7, "profile")
    assert selected.gpu_id == 7
    assert not other.orchestration.model_refresh.inference.cuda_graphs
    eager, selected = benchmark.resolve_benchmark_config(base, 3, "eager")
    assert selected.gpu_id == 3
    assert not eager.orchestration.model_refresh.inference.cuda_graphs
    assert eager.train.compile is False
    for invalid in (0, 99, -1):
        with pytest.raises(ValueError, match="actor GPU"):
            benchmark.resolve_benchmark_config(base, invalid, "profile")


def test_profile_plan_and_worker_use_effective_actor_flags_keep_original_freeze_hash(
    tmp_path, native, monkeypatch, capsys
):
    args, raw_config, manifest = replay_fixture(tmp_path, native)
    args.config = Path("configs/small.yaml")
    benchmark.freeze_positions(args, raw_config, manifest)
    source_hash = benchmark.digest(args.config.read_bytes())
    config = actor_override_config(raw_config)
    manifest.checkpoint_sha256 = "checkpoint-test"
    monkeypatch.setattr(benchmark, "load_config", lambda _: config)
    monkeypatch.setattr(benchmark, "load_model_manifest", lambda _: manifest)
    calls = []

    def load_evaluator(effective, *_args, **_kwargs):
        calls.append(effective)
        return adapter(cache_max_entries=128)

    monkeypatch.setattr(benchmark, "load_manifest_evaluator", load_evaluator)
    output = tmp_path / "effective-report.json"
    arguments = [
        "run",
        "--config",
        str(args.config),
        "--checkpoint",
        "immutable.json",
        "--positions",
        str(args.output),
        "--pairs",
        "4:16",
        "--widths",
        "1",
        "--waves",
        "full",
        "--runtime",
        "profile",
        "--actor-gpu-id",
        "3",
        "--timeout-seconds",
        "30",
        "--output",
        str(output),
    ]
    assert benchmark.main(arguments) == 0
    planned = json.loads(capsys.readouterr().out)
    assert calls == []
    plan = planned["plan"]
    assert plan["config_sha256"] == source_hash
    assert plan["actor_gpu_id"] == 3
    assert plan["actor_pipeline"]["cuda_graphs"] is True
    assert plan["profile_inference"]["cuda_graphs"] is False
    assert plan["effective_inference"]["cuda_graphs"] is True
    assert plan["effective_compile"] is True
    assert plan["graph_registry_model_capacity"] == 3
    assert plan["effective_inference"]["cuda_graph_max_bytes"] == max(
        1, plan["profile_inference"]["cuda_graph_max_bytes"] // 3
    )
    assert (
        benchmark.main(
            [
                *arguments,
                "--worker",
                "--execute",
                "--pinned-plan",
                planned["plan_sha256"],
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0].orchestration.model_refresh.inference.cuda_graphs
    assert calls[0].train.compile is True
    assert json.loads(output.read_bytes())["plan_sha256"] == planned["plan_sha256"]
    assert benchmark.digest(args.config.read_bytes()) == source_hash


def test_graph_memory_cap_matches_selected_actor_registry_capacity():
    base = actor_override_config(load_config(Path("configs/small.yaml")))
    gpus = tuple(
        replace(gpu, actor_cohorts=4) if gpu.gpu_id == 3 else gpu
        for gpu in base.orchestration.gpus
    )
    refresh = base.orchestration.model_refresh
    source = replace(
        base,
        orchestration=replace(
            base.orchestration,
            gpus=gpus,
            model_refresh=replace(
                refresh,
                inference=replace(
                    refresh.inference,
                    shared_batching=True,
                    cuda_graph_max_bytes=48 * 1024**3,
                ),
            ),
        ),
    )
    effective, selected = benchmark.resolve_benchmark_config(source, 3, "profile")
    assert selected.actor_cohorts == 4
    assert (
        effective.orchestration.model_refresh.inference.cuda_graph_max_bytes
        == 8 * 1024**3
    )
    assert (
        source.orchestration.model_refresh.inference.cuda_graph_max_bytes
        == 48 * 1024**3
    )
    other, _ = benchmark.resolve_benchmark_config(source, 7, "profile")
    assert (
        other.orchestration.model_refresh.inference.cuda_graph_max_bytes == 16 * 1024**3
    )


def test_warm_protocol_shares_identical_full_allocations_and_rejects_new_captures(
    native, monkeypatch
):
    rows = [sample(native)]
    base = SelfPlayConfig(simulation_ring_exponent=0)
    calls = []
    capture_during_measurement = False

    def fake_arm(
        _native, _evaluator, samples, _base, pair, *, full, width, reset="all", **_
    ):
        calls.append((pair, full, reset))
        amount = pair[1] if full else pair[0]
        return {
            "raw_pair": pair,
            "full": full,
            "first_visit_batch_size": width,
            "seconds": 1.0,
            "requested_rows": amount + 1,
            "inference": {
                "graph_captures": int(reset == "all" or capture_during_measurement)
            },
            "positions": [
                {
                    "id": benchmark.sample_id(samples[0]),
                    "cell": benchmark.cell(samples[0]),
                    "selected_action": 1,
                    "selected_value": 0.25,
                    "swap": False,
                    "actions": [1, 2],
                    "visits": [amount, 0],
                    "q_values": [0.25, 0.25],
                    "policy_target": [0.5, 0.5],
                }
            ],
        }

    monkeypatch.setattr(benchmark, "run_arm", fake_arm)
    results = benchmark.measure_sweep(
        native,
        object(),
        rows,
        base,
        [(2, 16), (1, 16)],
        [1],
        ["full", "fast"],
        max_rows=8,
        seed=17,
        deadline=time.monotonic() + 10,
        repeats=2,
    )
    assert len(calls) == 10  # one reference + three unique (warmup + two measured).
    measured = [r for r in results if r["stage"] == "measured"]
    assert len(measured) == 8  # Full-arm aliases are explicit, not extra execution.
    full = [r for r in measured if r["full"]]
    assert len({r["canonical_arm"] for r in full}) == 1
    assert all(r["shared_measurement"] for r in full)
    assert all(r["timing_admissible"] for r in measured)
    assert sum(reset == "predictions" for _, _, reset in calls) == 6
    capture_during_measurement = True
    rejected = benchmark.measure_sweep(
        native,
        object(),
        rows,
        base,
        [(2, 16)],
        [1],
        ["full", "fast"],
        max_rows=8,
        seed=17,
        deadline=time.monotonic() + 10,
        repeats=2,
    )
    assert all(not r["timing_admissible"] for r in rejected if r["stage"] == "measured")
    costs = benchmark.weighted_costs(rejected, base, [(2, 16)], [1], None, None)
    assert costs[0]["expected_seconds_per_root"] is None
    assert costs[0]["expected_full_targets_per_second"] is None


def test_cap_randomization_projection_preserves_expected_policy_share():
    base = SelfPlayConfig(
        fast_probability=0.65, full_probability=0.35, fast_policy_weight=0.2
    )
    records = []
    for pair, fast in [((32, 384), 53), ((8, 384), 13)]:
        for full in [True, False]:
            for repeat in range(3):
                amount = 640 if full else fast
                records.append(
                    {
                        "ring": 10,
                        "raw_pair": pair,
                        "full": full,
                        "first_visit_batch_size": 1,
                        "stage": "measured",
                        "repeat": repeat,
                        "positions": [{}],
                        "seconds": (amount + 1) / 1000,
                        "requested_rows": amount + 1,
                        "timing_admissible": True,
                    }
                )
    baseline, candidate = benchmark.weighted_costs(
        records, base, [(32, 384), (8, 384)], [1], 0.125, None
    )
    assert baseline["full_probability"] == 0.35
    assert candidate["full_probability"] == 0.125
    assert candidate["fast_policy_weight"] == pytest.approx(0.05306122448979592)
    assert candidate["expected_weighted_full_policy_share"] == pytest.approx(
        0.7291666666666666
    )
    assert (
        candidate["expected_weighted_full_policy_share"]
        == baseline["expected_weighted_full_policy_share"]
    )
    assert (
        candidate["expected_full_targets_per_second"]
        > baseline["expected_full_targets_per_second"]
    )
    assert candidate["repeats_per_wave"] == {"True": 3, "False": 3}
    assert base.fast_probability == 0.65 and base.fast_policy_weight == 0.2


def test_independent_holdout_excludes_prior_games_and_requires_real_swap_positions(
    tmp_path, native, monkeypatch
):
    args, config, manifest = replay_fixture(tmp_path, native)
    previous = benchmark.freeze_positions(args, config, manifest)
    previous_directory = args.output
    args.output = tmp_path / "holdout"
    args.exclude_positions = [previous_directory]
    args.swap_per_mode = 1
    with pytest.raises(ValueError, match="insufficient"):
        benchmark.freeze_positions(args, config, manifest)
    assert not args.output.exists()
    with ReplayStore(args.replay_root) as store:
        generation = int(
            store.connection.execute(
                "SELECT generation FROM actor_generations WHERE actor_id='actor-test'"
            ).fetchone()[0]
        )
        for label in benchmark.MODES:
            segment, mode = label.split("-")
            placed_values = (5, 20, 38, 1) if segment == "pie" else (5, 20, 38)
            samples = [
                sample(
                    native,
                    mode=mode,
                    handicap=3 if segment == "handicap" else 1,
                    pie=segment == "pie",
                    placed=placed,
                    game=f"fresh-{label}-{placed}",
                    generation=generation,
                )
                for placed in placed_values
            ]
            store.append(
                samples,
                phase_min=0,
                phase_max=0,
                model_version="sha256-test",
                model_identity="sha256-test",
                model_step=10,
                run_id="run-test",
                generation_family="family-test",
                actor_id="actor-test",
                generation=generation,
            )
    holdout = benchmark.freeze_positions(args, config, manifest)
    old_games = {row["id"].rsplit("/", 1)[0] for row in previous["positions"]}
    new_games = {row["id"].rsplit("/", 1)[0] for row in holdout["positions"]}
    assert old_games.isdisjoint(new_games)
    assert holdout["excluded_games"] == 18
    assert len(holdout["swap_position_ids"]) == 2
    decoded, _ = benchmark.read_frozen(
        args.output, benchmark.digest(args.config.read_bytes()), manifest
    )
    assert len(decoded) == 20
    swaps = [
        row
        for row in decoded
        if benchmark.sample_id(row) in holdout["swap_position_ids"]
    ]
    assert {row.mode for row in swaps} == {"classic", "double"}
    assert all(row.swap_available for row in swaps)
    manifest.checkpoint_sha256 = "checkpoint-test"
    monkeypatch.setattr(benchmark, "load_config", lambda _: config)
    monkeypatch.setattr(benchmark, "load_model_manifest", lambda _: manifest)
    monkeypatch.setattr(
        benchmark,
        "load_manifest_evaluator",
        lambda *_, **__: adapter(cache_max_entries=128),
    )
    output = tmp_path / "holdout-report.json"
    assert (
        benchmark.main(
            [
                "run",
                "--config",
                str(args.config),
                "--checkpoint",
                "immutable.json",
                "--positions",
                str(args.output),
                "--pairs",
                "4:16",
                "2:16",
                "--widths",
                "1",
                "--max-rows",
                "8",
                "--repeats",
                "2",
                "--timeout-seconds",
                "30",
                "--candidate-full-probability",
                ".125",
                "--execute",
                "--worker",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_bytes())
    assert result["status"] == "passed"
    assert result["plan"]["swap_probe_count"] == 2
    assert {row["dataset"] for row in result["results"]} == {"balanced", "swap-probes"}
    assert all(
        len(row["positions"]) == 18
        for row in result["results"]
        if row["dataset"] == "balanced"
    )
    assert all(
        len(row["positions"]) == 2
        for row in result["results"]
        if row["dataset"] == "swap-probes"
    )
    assert all(
        not row["full"]
        for row in result["results"]
        if row["dataset"] == "swap-probes" and row["stage"] == "measured"
    )
    assert (
        len(result["weighted_costs"]) == 2
    )  # Diagnostic swaps never enter cost projections.
