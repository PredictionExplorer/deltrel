"""A graceful stop must publish one coherent durable learner counter pair."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deltreltrain import cli
from deltreltrain.config import load_config
from deltreltrain.runtime import HeartbeatReporter
from test_learner_shutdown_durability import recovery_pointer, trained_fixture


def test_one_real_update_after_utd_wait_keeps_heartbeat_and_recovery_in_sync(
    tmp_path, monkeypatch
):
    with trained_fixture(tmp_path) as learner:
        heartbeat_path = tmp_path / "learner.heartbeat.json"
        heartbeat = HeartbeatReporter(
            heartbeat_path, worker="learner", interval_seconds=60
        )
        credit_arrived = False
        observed = []

        def progress(**details):
            nonlocal credit_arrived
            heartbeat.advance(**details)
            if details["phase"] == "update_to_data_wait":
                observed.append(("wait", learner.step, learner.examples_consumed))
                credit_arrived = True
            elif details["phase"] == "training":
                current = json.loads(heartbeat_path.read_text())
                observed.append(
                    ("training", current["step"], current["examples_consumed"])
                )

        monkeypatch.setattr(learner, "_utd_step_budget", lambda: int(credit_arrived))
        heartbeat.start()
        try:
            learner.run(
                steps=4, stop_requested=lambda: learner.step >= 1, progress=progress
            )
        finally:
            heartbeat.close(final_phase="stopped")
        batch = learner.train_config.global_batch_size(learner.world_size)
        assert observed == [("wait", 0, 0), ("training", 1, batch)]
        saved = json.loads(heartbeat_path.read_text())
        recovery = recovery_pointer(learner)
        assert saved["phase"] == "stopped"
        assert saved["step"] == recovery["step"] == learner.step == 1
        assert (
            saved["examples_consumed"]
            == recovery["examples_consumed"]
            == learner.examples_consumed
            == batch
        )


@pytest.mark.parametrize("stopping", [True, False])
def test_cli_final_heartbeat_uses_authoritative_counters_even_without_last_progress(
    tmp_path, monkeypatch, stopping
):
    experiment = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    experiment = replace(
        experiment, learner=replace(experiment.learner, resume_latest=False)
    )
    monkeypatch.setattr(cli, "load_config", lambda _path: experiment)
    for name, value in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    stopped = False
    monkeypatch.setattr(
        cli,
        "SignalLatch",
        lambda: SimpleNamespace(install=lambda: None, is_set=lambda: stopped),
    )

    class Learner:
        step = 41
        examples_consumed = 41 * 512
        epoch = 3

        def run(self, *, progress, **_kwargs):
            nonlocal stopped
            progress(
                phase="update_to_data_wait",
                step=self.step,
                examples_consumed=self.examples_consumed,
            )
            self.step += 1
            self.examples_consumed += 512
            stopped = stopping
            # Model a clean stop before a final progress notification. The CLI
            # still owns the authoritative counters and must publish them.
            return self.step

    learner = Learner()
    monkeypatch.setattr(
        cli.LearnerLoop, "from_experiment", lambda *_args, **_kwargs: learner
    )
    identity = tmp_path / "run.json"
    identity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "generation_family": "family",
                "created_ns": 1,
            }
        )
    )
    heartbeat = tmp_path / "heartbeat.json"
    cli.train_main(
        [
            "--config",
            "unused.yaml",
            "--run-identity",
            str(identity),
            "--replay-store",
            str(tmp_path / "replay"),
            "--output",
            str(tmp_path / "learner"),
            "--heartbeat",
            str(heartbeat),
            "--device",
            "cpu",
        ]
    )
    saved = json.loads(heartbeat.read_text())
    assert saved["phase"] == ("stopped" if stopping else "completed")
    assert (saved["step"], saved["examples_consumed"]) == (
        42,
        42 * 512,
    )
