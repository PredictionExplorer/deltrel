from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import pytest

from deltreltrain.config import ConfigError, HistoricalEvaluationConfig
from deltreltrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_measurement_scheduler_defaults,
)
from deltreltrain.measurement_scheduling import (
    MEASUREMENT_RESTORE_POLICY,
    MeasurementServiceLedger,
    measurement_service_status,
)
from deltreltrain.runtime import RunIdentity, append_jsonl


def ledger_case(tmp_path):
    events = tmp_path / "coordinator.jsonl"
    events.write_text("")
    identity = RunIdentity(tmp_path / "run.json", "run", "family", 1)
    options = dict(
        path=tmp_path / "measurement-service.json",
        coordinator_events=events,
        request_path=tmp_path / "pause.json",
        run_identity=identity,
    )
    return MeasurementServiceLedger(**options), options


def event(options, token, name, seconds):
    append_jsonl(
        options["coordinator_events"],
        {"token": token, "event": name, "timestamp_ns": int(seconds * 1e9)},
    )


def completed_lease(ledger, options, token, kind, start, end, share=0.2):
    ledger.register_lease(token, kind, share)
    event(options, token, "pause_lease_ready", start)
    event(options, token, "pause_lease_released", end)
    ledger.refresh()


def test_service_counts_actual_gpu_hold_not_requested_or_session_time(tmp_path):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("p", "promotion", 0.2)
    event(options, "p", "pause_lease_requested", 1)
    event(options, "p", "pause_lease_ready", 900)
    ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == 0
    event(options, "p", "pause_lease_released", 910)
    ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == 10_000_000_000
    assert ledger.state["debt_ns"] == 2_000_000_000
    # A failed acquisition earns no credit.
    ledger.register_lease("cancel", "promotion", 0.2)
    event(options, "cancel", "pause_lease_released", 1910)
    ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == 10_000_000_000


def test_crash_mid_lease_reconciles_coordinator_release_exactly_once(tmp_path):
    ledger, options = ledger_case(tmp_path)
    ledger.pin_job("candidate", "anchor")
    ledger.register_lease("m", "measurement", 0.2)
    event(options, "m", "pause_lease_ready", 10)
    ledger.refresh()
    resumed = MeasurementServiceLedger(**options)
    assert resumed.pinned_job == {"candidate": "candidate", "baseline": "anchor"}
    with pytest.raises(ValueError, match="unreconciled"):
        resumed.register_lease("p", "promotion", 0.2)
    # Recovery is recorded by the coordinator, not the killed evaluator.
    event(options, "m", "pause_lease_released", 47)
    resumed.refresh()
    again = MeasurementServiceLedger(**options)
    again.refresh()
    assert again.state["measurement_gpu_ns"] == 37_000_000_000
    assert again.state["debt_ns"] == -29_600_000_000


def test_fraction_changes_apply_only_to_future_lease_credit(tmp_path):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "p1", "promotion", 1, 101)
    completed_lease(ledger, options, "p2", "promotion", 102, 202, share=0.4)
    assert ledger.state["debt_ns"] == 60_000_000_000


@pytest.mark.parametrize("became_ready", [False, True])
def test_actor_restart_release_settles_legacy_coordinator_once(tmp_path, became_ready):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("p", "promotion", 0.2)
    if became_ready:
        event(options, "p", "pause_lease_ready", 1)
    event(options, "p", "pause_target_restarted", 11)
    # New coordinators additionally emit canonical release; never double count.
    event(options, "p", "pause_lease_released", 11)
    ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == (10 * 10**9 if became_ready else 0)
    ledger.register_lease("next", "measurement", 0.2)


def test_rejected_unadmitted_lease_earns_no_credit_and_does_not_block_next_request(
    tmp_path,
):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("p", "promotion", 0.2)
    event(options, "p", "pause_request_rejected", 100)
    ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == 0
    assert ledger.state["debt_ns"] == 0
    ledger.register_lease("next", "measurement", 0.2)


@pytest.mark.parametrize("enabled,pause", [(False, False), (True, False)])
def test_protected_service_requires_coordinator_and_shared_gpu(enabled, pause):
    from deltreltrain.config import OrchestrationConfig, PromotionConfig

    with pytest.raises(ConfigError, match="measurement reservation"):
        OrchestrationConfig(
            enabled=False,
            promotion=PromotionConfig(enabled=enabled, pause_sharing_mode=pause),
            historical_evaluation=HistoricalEvaluationConfig(
                enabled=True,
                measure_direct_predecessor=True,
                measurement_service_fraction=0.2,
            ),
        )


def test_durable_debt_services_measurement_under_endless_promotion_arrivals(tmp_path):
    ledger, options = ledger_case(tmp_path)
    now = 1
    kinds = []
    for index in range(25):
        ledger = MeasurementServiceLedger(**options)
        selected = ledger.should_serve(
            due=True, now_ns=now * 10**9, max_wait_seconds=3600
        )
        kind = "measurement" if selected else "promotion"
        kinds.append(kind)
        completed_lease(ledger, options, str(index), kind, now, now + 10)
        now += 310  # Actor catch-up is not evaluation service.
    assert kinds.count("measurement") == 5
    assert ledger.state["measurement_gpu_ns"] == 50 * 10**9
    assert ledger.state["promotion_gpu_ns"] == 200 * 10**9
    assert ledger.state["debt_ns"] == 0


def test_wait_deadline_survives_restart_even_with_excess_background_credit(tmp_path):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "m", "measurement", 1, 101)
    assert not ledger.should_serve(due=True, now_ns=102 * 10**9, max_wait_seconds=600)
    resumed = MeasurementServiceLedger(**options)
    assert resumed.should_serve(due=True, now_ns=602 * 10**9, max_wait_seconds=600)
    assert not resumed.should_serve(due=False, now_ns=603 * 10**9, max_wait_seconds=600)


def test_idle_measurement_queue_cannot_bank_hours_of_future_gpu_monopoly(tmp_path):
    ledger, options = ledger_case(tmp_path)
    for index in range(12):
        completed_lease(
            ledger, options, str(index), "promotion", 1 + index * 600, 301 + index * 600
        )
        assert not ledger.should_serve(
            due=False, now_ns=(302 + index * 600) * 10**9, max_wait_seconds=3600
        )
    assert ledger.state["debt_ns"] == 0
    assert ledger.state["promotion_gpu_ns"] == 3600 * 10**9
    assert ledger.should_serve(due=True, now_ns=8000 * 10**9, max_wait_seconds=3600)
    completed_lease(ledger, options, "m", "measurement", 8000, 8300)
    assert not ledger.should_serve(due=True, now_ns=8600 * 10**9, max_wait_seconds=3600)


def test_backlogged_job_completion_preserves_service_debt(tmp_path):
    ledger, options = ledger_case(tmp_path)
    ledger.pin_job("one", "anchor")
    ledger.should_serve(due=True, now_ns=10**9, max_wait_seconds=3600)
    completed_lease(ledger, options, "m", "measurement", 1, 301)
    ledger.complete_job()
    ledger.pin_job("two", "anchor")
    assert not ledger.should_serve(due=True, now_ns=601 * 10**9, max_wait_seconds=3600)
    assert ledger.state["debt_ns"] == -240 * 10**9


def test_partial_append_is_not_consumed_and_corrupt_complete_record_fails_closed(
    tmp_path,
):
    ledger, options = ledger_case(tmp_path)
    path = options["coordinator_events"]
    path.write_bytes(b'{"event":"unrelated"}')
    ledger.refresh()
    assert ledger.state["journal_offset"] == 0
    with path.open("ab") as stream:
        stream.write(b"\ninvalid\n")
    before = options["path"].read_bytes()
    with pytest.raises(ValueError):
        ledger.refresh()
    assert options["path"].read_bytes() == before
    assert ledger.state["journal_offset"] == 0


@pytest.mark.parametrize("mutation", ["truncate", "rewrite"])
def test_accounted_journal_cannot_be_silently_replaced(tmp_path, mutation):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "p", "promotion", 1, 2)
    path = options["coordinator_events"]
    original = path.read_bytes()
    path.write_bytes(
        b""
        if mutation == "truncate"
        else original.replace(b"1000000000", b"9000000000")
    )
    with pytest.raises(ValueError, match="journal changed"):
        MeasurementServiceLedger(**options)


def test_restore_with_identical_journal_bytes_accepts_new_inode(tmp_path):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "p", "promotion", 1, 2)
    path = options["coordinator_events"]
    content = path.read_bytes()
    path.unlink()
    path.write_bytes(content)
    assert MeasurementServiceLedger(**options).state["promotion_gpu_ns"] == 10**9


def test_duplicate_ready_and_negative_interval_fail_without_partial_accounting(
    tmp_path,
):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("p", "promotion", 0.2)
    event(options, "p", "pause_lease_ready", 3)
    event(options, "p", "pause_lease_ready", 4)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.refresh()
    assert ledger.state["promotion_gpu_ns"] == 0
    options["coordinator_events"].write_text("")
    event(options, "p", "pause_lease_ready", 3)
    event(options, "p", "pause_lease_released", 2)
    with pytest.raises(ValueError, match="backwards"):
        ledger.refresh()


def test_corrupt_or_foreign_ledger_is_not_reset(tmp_path):
    ledger, options = ledger_case(tmp_path)
    state = ledger.state
    state["run_id"] = "foreign"
    options["path"].write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity"):
        MeasurementServiceLedger(**options)


def test_service_status_reads_settled_time_and_never_mutates_ledger(tmp_path):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "p", "promotion", 1, 101)
    completed_lease(ledger, options, "m", "measurement", 102, 127)
    directory = tmp_path / "arena"
    directory.mkdir()
    destination = directory / "measurement-service.json"
    destination.write_bytes(options["path"].read_bytes())
    before = destination.read_bytes()
    status = measurement_service_status(
        tmp_path, options["run_identity"], now_ns=200 * 10**9, target_fraction=0.2
    )
    assert status["promotion_gpu_seconds"] == 100
    assert status["measurement_gpu_seconds"] == 25
    assert status["observed_fraction"] == 0.2
    assert status["debt_seconds"] == 0
    assert status["last_service_age_seconds"] == 98
    assert destination.read_bytes() == before
    destination.write_text("{broken")
    assert (
        measurement_service_status(
            tmp_path, options["run_identity"], now_ns=200 * 10**9
        )["status"]
        == "invalid"
    )


def test_pinned_job_cannot_be_superseded_and_protects_manifests(tmp_path):
    ledger, _ = ledger_case(tmp_path)
    ledger.pin_job(
        "one",
        "anchor",
        candidate_manifest=tmp_path / "one.json",
        baseline_manifest=tmp_path / "anchor.json",
    )
    assert ledger.state["candidate_manifest"] == str(tmp_path / "one.json")
    with pytest.raises(ValueError, match="unfinished"):
        ledger.pin_job("newer", "anchor")
    ledger.complete_job()
    assert "candidate_manifest" not in ledger.state
    ledger.pin_job("newer", "anchor")


def restore_receipt(options):
    ledger = options["path"].read_bytes()
    journal = options["coordinator_events"].read_bytes()
    document = {
        "schema_version": 1,
        "run_id": "run",
        "generation_family": "family",
        "policy": MEASUREMENT_RESTORE_POLICY,
        "ledger_sha256": hashlib.sha256(ledger).hexdigest(),
        "journal_bytes": len(journal),
        "journal_sha256": hashlib.sha256(journal).hexdigest(),
        "unsettled_tokens": sorted(json.loads(ledger)["leases"]),
        "restored_ns": 500 * 10**9,
    }
    path = options["path"].with_name("measurement-service-restore.json")
    path.write_text(json.dumps(document))
    return path, document


def test_verified_disaster_restore_preserves_debt_and_marks_unknown_open_interval(
    tmp_path,
):
    ledger, options = ledger_case(tmp_path)
    completed_lease(ledger, options, "p", "promotion", 1, 101)
    ledger.pin_job("champion", "anchor")
    ledger.register_lease("m", "measurement", 0.2)
    event(options, "m", "pause_lease_ready", 102)
    ledger.refresh()
    restore_receipt(options)
    restored = MeasurementServiceLedger(**options)
    assert restored.state["debt_ns"] == 20 * 10**9
    assert restored.state["promotion_gpu_ns"] == 100 * 10**9
    assert restored.state["measurement_gpu_ns"] == 0
    assert restored.state["accounting_complete"] is False
    assert restored.pinned_job == {"candidate": "champion", "baseline": "anchor"}
    assert restored.state["leases"] == {}
    assert (
        restored.state["uncredited_restored_leases"][0]["leases"]["m"]["ready_ns"]
        == 102 * 10**9
    )
    # Crash after atomic consumption, before receipt removal: apply only once.
    again = MeasurementServiceLedger(**options)
    assert len(again.state["applied_restore_receipts"]) == 1
    assert len(again.state["uncredited_restored_leases"]) == 1
    again.register_lease("new", "promotion", 0.2)


def test_restore_consumes_captured_release_before_marking_unknown_time(tmp_path):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("m", "measurement", 0.2)
    event(options, "m", "pause_lease_ready", 1)
    ledger.refresh()
    event(options, "m", "pause_lease_released", 11)
    restore_receipt(options)
    restored = MeasurementServiceLedger(**options)
    assert restored.state["accounting_complete"] is True
    assert restored.state["measurement_gpu_ns"] == 10 * 10**9
    assert restored.state["debt_ns"] == -8 * 10**9


@pytest.mark.parametrize(
    "key,value",
    [
        ("ledger_sha256", "f" * 64),
        ("journal_sha256", "f" * 64),
        ("unsettled_tokens", ["unregistered"]),
        ("run_id", "foreign"),
    ],
)
def test_restore_receipt_cannot_discard_unbound_accounting(tmp_path, key, value):
    ledger, options = ledger_case(tmp_path)
    ledger.register_lease("m", "measurement", 0.2)
    path, document = restore_receipt(options)
    document[key] = value
    path.write_text(json.dumps(document))
    before = options["path"].read_bytes()
    with pytest.raises(ValueError):
        MeasurementServiceLedger(**options)
    assert options["path"].read_bytes() == before


@pytest.mark.parametrize(
    "value", [True, False, "0.2", None, -0.1, 1, float("nan"), float("inf")]
)
def test_reservation_fraction_rejects_invalid_values(value):
    with pytest.raises(ConfigError, match="measurement_service_fraction"):
        HistoricalEvaluationConfig(measurement_service_fraction=value)


def test_enabled_reservation_requires_measurement_and_keeps_defaults_compatible():
    with pytest.raises(ConfigError, match="requires"):
        HistoricalEvaluationConfig(measurement_service_fraction=0.2)
    config = HistoricalEvaluationConfig(
        enabled=True, measure_direct_predecessor=True, measurement_service_fraction=0.2
    )
    assert config.measurement_max_wait_seconds == 3600
    payload = {
        "orchestration": {
            "historical_evaluation": {
                "measurement_service_fraction": 0.0,
                "measurement_max_wait_seconds": 3600.0,
            }
        }
    }
    legacy = {"orchestration": {"historical_evaluation": {}}}
    assert without_measurement_scheduler_defaults(payload) == legacy
    assert legacy in compatible_config_epoch_payloads(payload)
    payload["orchestration"]["historical_evaluation"][
        "measurement_service_fraction"
    ] = 0.2
    assert all(
        p["orchestration"]["historical_evaluation"]["measurement_service_fraction"]
        == 0.2
        for p in compatible_config_epoch_payloads(payload)
    )


@pytest.mark.parametrize("value", [True, None, "1", 0, -1, float("inf"), float("nan")])
def test_wait_deadline_requires_finite_positive_seconds(value):
    with pytest.raises(ConfigError, match="measurement_max_wait_seconds"):
        replace(HistoricalEvaluationConfig(), measurement_max_wait_seconds=value)
