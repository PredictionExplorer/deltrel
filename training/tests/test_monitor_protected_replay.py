"""Selected teacher replay is visible independently of actor generation."""

import json

import pytest

from scripts import monitor_run as monitor
from test_monitor_run import _fixture, _healthy_dependencies


@pytest.mark.parametrize("cap_respected", [True, False])
def test_monitor_reads_protected_selection_separately_from_latest_loss(
    tmp_path, monkeypatch, cap_respected
):
    now = 10_000_000_000
    root = _fixture(tmp_path, now_ns=now)
    _healthy_dependencies(monkeypatch)
    path = root / "learner" / "metrics.jsonl"
    loss = json.loads(path.read_text())
    selection = {
        "selected_rows_by_model_identity": {"champion": 25, "candidate": 75},
        "protected_champion_replay": {
            "enabled": True,
            "scope": "selected_replay_window_not_consumed_updates",
            "model_identity": "champion",
            "model_step": 1,
            "selected_rows": 25,
            "selected_fraction": 0.25,
            "max_fraction": 0.25,
            "cap_respected": cap_respected,
        },
    }
    allocation = {
        "event": "replay_window_allocated",
        "timestamp_ns": now - 2_000_000_000,
        "replay_selection": selection,
    }
    path.write_text(json.dumps(allocation) + "\n" + json.dumps(loss) + "\n")
    actor_path = root / "metrics" / "actor-gpu-1.jsonl"
    actor = json.loads(actor_path.read_text())
    actor.update(
        replay_eligible_at_commit=None,
        ordinary_replay_eligible_at_commit=False,
        protected_champion_candidate_at_commit=True,
        replay_eligibility_status="protected_pending_selection",
        protected_pending_selection_samples=100,
    )
    actor_path.write_text(json.dumps(actor) + "\n")
    snapshot = monitor.collect_snapshot(root, now_ns=now)
    assert snapshot["learner"]["replay_selection"] == selection
    assert snapshot["learner"]["replay_selection_age_seconds"] == 2.0
    assert snapshot["learner"]["losses"] == loss["losses"]
    latest = snapshot["actors"]["latest"][0]
    assert latest["protected_pending_selection_samples"] == 100
    assert latest["replay_eligible_at_commit"] is None
    assert latest["ordinary_replay_eligible_at_commit"] is False
    codes = {warning["code"] for warning in snapshot["warnings"]}
    assert ("protected_champion_replay_cap" in codes) is (not cap_respected)
