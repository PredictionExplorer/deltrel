"""CPU-only receipt/transport/semantic guards; no service or GPU operations."""

import copy
import json
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import qualify_cloud_gpu_probe as probe
from test_deltrelserve import request_payload, response_payload


@pytest.fixture
def challenge():
    return {
        "format": "deltrelserve.gpu-qualification-challenge",
        "schema_version": 1,
        "run_id": "test-572377",
        "plan_sha256": "a" * 64,
        "boot_id": "boot-one",
        "challenge": "b" * 64,
        "probe_unit": "deltrelserve-qualification-test-572377-probe.service",
        "model_identity": "sha256-" + "c" * 64,
        "model_step": 572377,
        "issued_monotonic": 10,
        "deadline_monotonic": 360,
        "required_checks": list(probe.REQUIRED),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("run_id", "../other"),
        ("plan_sha256", "wrong"),
        ("challenge", "d" * 63),
        ("boot_id", "other"),
        ("probe_unit", "other.service"),
        ("model_identity", "candidate"),
        ("model_step", 566428),
        ("required_checks", ["semantic_16"]),
        ("issued_monotonic", 50),
        ("deadline_monotonic", 12),
        ("deadline_monotonic", float("nan")),
    ],
)
def test_challenge_fails_closed_for_wrong_binding_or_deadline(challenge, field, value):
    challenge[field] = value
    with pytest.raises(probe.Refusal):
        probe.validate_challenge(
            challenge, boot="boot-one", invocation="e" * 32, now=20
        )


def test_invocation_is_real_systemd_identity(challenge):
    assert (
        probe.validate_challenge(
            challenge, boot="boot-one", invocation="e" * 32, now=20
        )
        == challenge
    )
    with pytest.raises(probe.Refusal, match="invocation"):
        probe.validate_challenge(challenge, boot="boot-one", invocation="", now=20)


def test_receipts_never_replace_prior_file_or_follow_symlink(tmp_path):
    target = tmp_path / "receipt.json"
    probe.atomic_new(target, {"first": True})
    with pytest.raises(FileExistsError):
        probe.atomic_new(target, {"first": False})
    assert json.loads(target.read_text()) == {"first": True}
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(FileExistsError):
        probe.atomic_new(alias, {"replacement": True})
    assert json.loads(target.read_text()) == {"first": True}


def test_evidence_is_relative_and_bounded(tmp_path):
    evidence = probe.Evidence(tmp_path)
    with pytest.raises(probe.Refusal):
        evidence.save("../escape.json", {})
    pin = evidence.save("first.json", {"ok": True})
    assert pin["path"] == "probe-evidence/first.json"
    evidence.total = probe.MAX_EVIDENCE
    with pytest.raises(probe.Refusal, match="evidence-budget"):
        evidence.save("second.json", {})


@pytest.fixture
def wire(challenge):
    payload = cast(dict[str, Any], request_payload())
    body = response_payload()
    body.update(
        request_id="probe-request",
        model_step=572377,
        model_version=challenge["model_identity"],
    )
    applied = []
    state = SimpleNamespace(
        node_count=len(payload["stones"]),
        apply_many=lambda rows, actions: applied.append((rows, actions)),
    )
    return payload, body, state, applied


def test_response_schema_identity_budget_and_native_legality(challenge, wire):
    payload, body, state, applied = wire
    assert (
        probe.validate_response(body, payload, state, challenge, "probe-request") == 0
    )
    assert applied == [([0], [0])]


@pytest.mark.parametrize(
    "fault",
    [
        "model",
        "step",
        "request",
        "budget",
        "variant",
        "history",
        "nonfinite",
        "probability",
        "occupied",
        "network",
        "predictions",
        "native",
    ],
)
def test_invalid_required_response_fields_cannot_pass(challenge, wire, fault):
    payload, body, state, applied = wire
    if fault == "model":
        body["model_version"] = "sha256-" + "d" * 64
    elif fault == "step":
        body["model_step"] = 566428
    elif fault == "request":
        body["request_id"] = "another-request"
    elif fault == "budget":
        body["root_visits"] = [1, 1]
    elif fault == "variant":
        body["variant"]["mode"] = "classic"
    elif fault == "history":
        body["history_known"] = False
    elif fault == "nonfinite":
        body["root_q"][0] = float("nan")
    elif fault == "probability":
        body["root_policy"] = [0.2, 0.2]
    elif fault == "occupied":
        payload["stones"][0] = 1
    elif fault == "network":
        payload["include_network_output"] = True
    elif fault == "predictions":
        payload["include_predictions"] = True
    else:

        def illegal(*args):
            raise ValueError("illegal")

        state.apply_many = illegal
    with pytest.raises(probe.Refusal):
        probe.validate_response(body, payload, state, challenge, "probe-request")
    assert not applied


@pytest.mark.parametrize(
    "fault", ["duplicate", "regression", "incomplete", "error", "budget"]
)
def test_stream_must_complete_exactly_once_monotonically(wire, fault):
    payload, body, _, _ = wire
    events = [
        {"type": "progress", "total_simulations": 4, "completed_simulations": n}
        for n in (0, 2, 4)
    ]
    events.append({"type": "result", "result": body})
    if fault == "duplicate":
        events.append(copy.deepcopy(events[-1]))
    elif fault == "regression":
        events[1]["completed_simulations"] = 5
    elif fault == "incomplete":
        events.pop(2)
    elif fault == "error":
        events.insert(1, {"type": "error"})
    else:
        events[0]["total_simulations"] = 8
    with pytest.raises(probe.Refusal):
        probe.decode_http(
            {"status": 200, "body": "\n".join(json.dumps(e) for e in events)},
            payload,
            True,
        )


def test_http_deadline_is_parent_enforced_and_secret_never_in_argv(monkeypatch):
    captured = {}
    monkeypatch.setattr(probe.time, "monotonic", lambda: 100.0)

    def hung(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(probe.subprocess, "run", hung)
    with pytest.raises(probe.Refusal, match="http-deadline") as failure:
        probe.http_request({}, "PRIVATE-TOKEN", "id", True, 102.0)
    assert captured["timeout"] == 2.0
    assert "PRIVATE-TOKEN" not in repr(captured["argv"])
    assert captured["stderr"] == subprocess.DEVNULL
    assert "PRIVATE-TOKEN" not in str(failure.value)


def test_optional_deep_requires_conservative_measured_budget():
    assert not probe.optional_deep_fits(200, 20)
    assert probe.optional_deep_fits(256, 20)
    assert not probe.optional_deep_fits(134, 1)


def test_failed_probe_writes_bound_failure_without_false_pass(
    challenge, tmp_path, monkeypatch
):
    monkeypatch.setattr(probe.time, "monotonic", lambda: 20.0)

    def failure():
        raise probe.Refusal("synthetic-case-failed")

    monkeypatch.setattr(probe, "semantic_cases", failure)
    result = probe.run_probe(challenge, tmp_path, "e" * 32, "PRIVATE")
    started = json.loads((tmp_path / "probe-started.json").read_text())
    assert result["invocation_id"] == started["invocation_id"] == "e" * 32
    assert all(result[key] == challenge[key] for key in probe.BINDINGS)
    assert result["started_monotonic"] == started["started_monotonic"]
    assert all(check["status"] == "failed" for check in result["checks"].values())
    assert all(
        check["evidence_sha256"] in {pin["sha256"] for pin in result["evidence"]}
        for check in result["checks"].values()
    )
    assert "PRIVATE" not in (tmp_path / "probe-result.json").read_text()


def test_complete_probe_receipt_is_accepted_with_cleared_systemd_invocation(
    challenge, tmp_path, monkeypatch
):
    from scripts import qualify_cloud_gpu_window as controller

    monkeypatch.setattr(probe.time, "monotonic", lambda: 20.0)
    monkeypatch.setattr(probe, "optional_deep_fits", lambda *args: False)

    def case(*args, simulations=32, **kwargs):
        payload = cast(dict[str, Any], request_payload())
        payload["search"]["simulations"] = simulations
        return SimpleNamespace(node_count=50, apply_many=lambda *args: None), payload

    def http(payload, token, request_id, streaming, deadline):
        body = response_payload()
        simulations = payload["search"]["simulations"]
        body.update(
            request_id=request_id,
            model_step=572377,
            model_version=challenge["model_identity"],
            root_visits=[simulations - 1, 1],
        )
        events = [
            {
                "type": "progress",
                "total_simulations": simulations,
                "completed_simulations": simulations,
            },
            {"type": "result", "result": body},
        ]
        return {
            "status": 200,
            "body": "\n".join(json.dumps(e) for e in events)
            if streaming
            else json.dumps(body),
        }

    monkeypatch.setattr(probe, "make_case", case)
    monkeypatch.setattr(probe, "semantic_cases", lambda: [case() for _ in range(16)])
    monkeypatch.setattr(probe, "http_request", http)
    probe.atomic_new(tmp_path / "probe-challenge.json", challenge)
    result = probe.run_probe(challenge, tmp_path, "e" * 32, "PRIVATE")
    state = {
        "probe_challenge_sha256": controller.digest(
            (tmp_path / "probe-challenge.json").read_bytes()
        ),
        "prior_invocations": {"probe": ""},
        "stage_deadline": 360,
    }
    current = {
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "Job": "0",
        "InvocationID": "",
        "Result": "success",
    }
    accepted = controller._probe_result(
        tmp_path, state, {}, current, cast(Any, SimpleNamespace(now=lambda: 25.0))
    )
    assert accepted == result
    assert all(v["status"] == "passed" for v in result["checks"].values())
    assert result["optional_checks"]["deep_8"]["status"] == "not_run"


def test_wrong_native_search_is_rejected_without_request(monkeypatch):
    monkeypatch.setattr(
        probe.importlib,
        "import_module",
        lambda _: SimpleNamespace(native_search_algorithm_id=lambda: "old-v2"),
    )
    with pytest.raises(probe.Refusal, match="native-search-identity"):
        probe.make_case(4, "classic")


@pytest.mark.native
def test_real_native_semantic_cases_cover_all_cells_and_round_trip():
    pytest.importorskip("deltrel_native")
    from deltrelserve.schemas import AnalyzeRequest

    cases = probe.semantic_cases()
    assert len(cases) == 16
    payloads = [AnalyzeRequest.model_validate(payload) for _, payload in cases]
    assert {(p.rings, p.mode) for p in payloads[:8]} == {
        (r, m) for r in (4, 6, 8, 10) for m in ("classic", "double")
    }
    assert sum(p.swap_available for p in payloads) == 2
    assert sum(p.swapped for p in payloads) == 2
    assert {(p.mode, p.pda) for p in payloads if p.handicap == 9} == {
        (m, p) for m in ("classic", "double") for p in (-3, 3)
    }
