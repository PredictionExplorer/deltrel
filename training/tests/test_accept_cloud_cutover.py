"""Hermetic public acceptance guards: no network, service or GPU action."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import py_compile
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts import accept_cloud_cutover as acceptance
from test_deltrelserve import request_payload, response_payload


def test_unpinned_sibling_is_refused_before_execution(tmp_path, monkeypatch):
    marker = tmp_path / "unverified-executed"
    sibling = tmp_path / "qualify_cloud_gpu_probe.py"
    sibling.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(
        acceptance, "__file__", str(tmp_path / "accept_cloud_cutover.py")
    )
    with pytest.raises(RuntimeError, match="probe-helper-checksum"):
        acceptance._load_probe_helpers()
    assert not marker.exists()


def test_verified_sibling_snapshot_ignores_matching_stale_bytecode(
    tmp_path, monkeypatch
):
    source = Path(acceptance.__file__).with_name("qualify_cloud_gpu_probe.py")
    approved = source.read_bytes()
    sibling = tmp_path / source.name
    stale = b"raise RuntimeError('stale-cached-code-executed')\n"
    sibling.write_bytes(stale + b" " * (len(approved) - len(stale)))
    os.utime(sibling, (1720000000, 1720000000))
    py_compile.compile(
        str(sibling),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    sibling.write_bytes(approved)
    os.utime(sibling, (1720000000, 1720000000))
    monkeypatch.setattr(
        acceptance, "__file__", str(tmp_path / "accept_cloud_cutover.py")
    )
    helper = acceptance._load_probe_helpers()
    assert helper.__file__ == str(sibling)
    assert callable(helper.make_case) and callable(helper.validate_response)


@pytest.fixture
def challenge():
    return {
        "format": "deltrelserve.cloud-cutover-challenge",
        "schema_version": 1,
        "run_id": "test-572377",
        "plan_sha256": "a" * 64,
        "boot_id": "boot-one",
        "challenge": "b" * 64,
        "acceptance_unit": "deltrelserve-cutover-test-572377-acceptance.service",
        "model_identity": "sha256-" + "c" * 64,
        "model_step": 572377,
        "issued_monotonic": 10.0,
        "deadline_monotonic": 360.0,
    }


@pytest.fixture
def wire(challenge):
    from deltrelserve.network_output import network_output_payload
    from deltrelserve.schemas import AnalyzeRequest

    torch.set_num_threads(1)
    payload = request_payload()
    stones = [-1] * 275
    for node in (0, 3, 4, 7, 8, 11):
        stones[node] = 0
    for node in (1, 2, 5, 6, 9, 10):
        stones[node] = 1
    payload.update(
        rings=10,
        stones=stones,
        opening=False,
        pie=True,
        moves_left=1,
        to_move=0,
        include_predictions=True,
        include_network_output=True,
        history={
            "current_turn": [11],
            "previous_turn": [9, 10],
            "own_previous_turn": [7, 8],
            "handicap_stones": [0],
        },
        search={"simulations": 544, "max_considered": 16, "seed": 572477},
    )
    body = response_payload()
    body.update(
        request_id=challenge["challenge"][:24] + "-standard-1",
        model_step=572377,
        model_version=challenge["model_identity"],
        root_visits=[543, 1],
        action={"code": 12, "kind": "place", "node": 12},
        root_actions=[{"code": n, "kind": "place", "node": n} for n in (12, 13)],
        variant={"mode": "double", "handicap": 1, "pie": True},
        predictions={
            "perspective": 0,
            "final_basis": "official_end",
            "final_counts": [
                {
                    "player": p,
                    "shores": 1.0,
                    "networks": 1.0,
                    "corners": 1.0,
                    "corner_bonus_probability": 0.1,
                }
                for p in (0, 1)
            ],
            "opponent_reply": None,
            "second_stone": None,
        },
    )
    shapes = {
        "policy": [275],
        "outcome": [2],
        "score_margin": [303],
        "ownership": [275, 3],
        "alive": [275],
        "soft_policy": [275],
        "opponent_reply": [276],
        "second_stone": [275],
        "final_shores": [2, 51],
        "final_networks": [2, 26],
        "final_capes": [2, 6],
    }
    body["network_output"] = network_output_payload(
        {name + "_logits": torch.zeros(1, *shape) for name, shape in shapes.items()},
        AnalyzeRequest.model_validate(payload),
        auxiliary_ready=True,
        swap_recommended=False,
    )
    applied = []
    state = SimpleNamespace(
        node_count=275, apply_many=lambda rows, actions: applied.append((rows, actions))
    )
    return payload, body, state, applied


def streamed(body, progress=(0, 272, 544)):
    return {
        "status": 200,
        "body": "".join(
            json.dumps(event) + "\n"
            for event in [
                *(
                    {
                        "type": "progress",
                        "completed_simulations": n,
                        "total_simulations": 544,
                    }
                    for n in progress
                ),
                {"type": "result", "result": body},
            ]
        ),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("format", "wrong"),
        ("run_id", "../other"),
        ("plan_sha256", "wrong"),
        ("challenge", "b" * 63),
        ("boot_id", "other-boot"),
        ("acceptance_unit", "other.service"),
        ("model_identity", "candidate"),
        ("model_step", 566428),
        ("model_step", True),
        ("issued_monotonic", 21),
        ("issued_monotonic", float("nan")),
        ("deadline_monotonic", 25),
        ("deadline_monotonic", 371),
        ("deadline_monotonic", float("inf")),
        ("deadline_monotonic", 2**2048),
    ],
)
def test_challenge_binding_and_deadlines_fail_closed(challenge, field, value):
    challenge[field] = value
    with pytest.raises(acceptance.Refusal):
        acceptance.validate_challenge(
            challenge, boot="boot-one", invocation="e" * 32, now=20
        )


def test_real_invocation_and_exact_challenge_fields_required(challenge):
    assert (
        acceptance.validate_challenge(
            challenge, boot="boot-one", invocation="e" * 32, now=20
        )
        == challenge
    )
    with pytest.raises(acceptance.Refusal, match="invocation"):
        acceptance.validate_challenge(challenge, boot="boot-one", invocation="", now=20)
    challenge["token"] = "PRIVATE"
    with pytest.raises(acceptance.Refusal, match="fields"):
        acceptance.validate_challenge(
            challenge, boot="boot-one", invocation="e" * 32, now=20
        )


def test_child_sends_fixed_public_origin_without_auth_or_redirects(
    wire, monkeypatch, capsys
):
    payload, body, _, _ = wire
    captured = {}
    raw = streamed(body)["body"].encode()

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            captured["limit"] = limit
            return raw

    class Opener:
        def open(self, request, timeout):
            captured.update(request=request, timeout=timeout)
            return Reply()

    def opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(acceptance.urllib.request, "build_opener", opener)
    monkeypatch.setattr(
        acceptance.sys,
        "stdin",
        io.StringIO(json.dumps({"payload": payload, "request_id": body["request_id"]})),
    )
    monkeypatch.setenv("DELTRELSERVE_BEARER_TOKEN", "PRIVATE-NEVER-SENT")
    monkeypatch.setenv("HTTPS_PROXY", "http://private:credential@unused.invalid")
    acceptance._http_worker()
    request = captured["request"]
    assert request.full_url == "https://deltrel.com/v2/move"
    assert request.get_method() == "POST"
    assert request.get_header("Origin") == "https://deltrel.com"
    assert request.get_header("Accept") == "application/x-ndjson"
    assert request.get_header("Authorization") is None
    assert request.get_header("Cookie") is None
    assert json.loads(request.data) == payload
    assert captured["handlers"][0].proxies == {}
    with pytest.raises(acceptance.Refusal, match="redirect"):
        captured["handlers"][1].redirect_request(
            None, None, 302, "redirect", {}, "https://other.invalid"
        )
    output = capsys.readouterr().out
    assert set(json.loads(output)) == {"status", "body"}
    assert "PRIVATE" not in output and "credential" not in output


@pytest.mark.parametrize(
    "field,value",
    [("token", "PRIVATE"), ("headers", {}), ("url", "https://other.invalid")],
)
def test_child_cannot_accept_credentials_headers_or_url(
    wire, monkeypatch, field, value
):
    payload, body, _, _ = wire
    request = {"payload": payload, "request_id": body["request_id"], field: value}
    monkeypatch.setattr(acceptance.sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(
        acceptance.urllib.request,
        "build_opener",
        lambda *args: pytest.fail("unexpected HTTP"),
    )
    with pytest.raises(acceptance.Refusal, match="http-input"):
        acceptance._http_worker()


@pytest.mark.parametrize("fault", ["credential", "deep", "rings", "no_heads"])
def test_child_payload_cannot_expand_the_declared_request(wire, fault):
    payload, _, _, _ = wire
    if fault == "credential":
        payload["credential"] = "PRIVATE"
    elif fault == "deep":
        payload["search"]["simulations"] = 4096
    elif fault == "rings":
        payload["rings"] = 4
    else:
        payload["include_network_output"] = False
    with pytest.raises(acceptance.Refusal, match="native-case-contract"):
        acceptance.validate_request(payload)


def test_parent_enforces_total_http_deadline_and_has_no_secret_input(wire, monkeypatch):
    payload, body, _, _ = wire
    captured = {}
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 100.0)

    def blocked(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(acceptance.subprocess, "run", blocked)
    with pytest.raises(acceptance.Refusal, match="http-deadline"):
        acceptance.http_request(payload, body["request_id"], 102.0)
    assert captured["timeout"] == 2.0
    assert captured["stderr"] == subprocess.DEVNULL
    assert set(json.loads(captured["input"])) == {"payload", "request_id"}
    with pytest.raises(acceptance.Refusal, match="http-deadline"):
        acceptance.http_request(payload, body["request_id"], float("nan"))


@pytest.mark.parametrize(
    "fault",
    [
        "early_result",
        "trailing",
        "unknown",
        "bool",
        "negative",
        "overshoot",
        "regression",
        "incomplete",
        "duplicate_result",
    ],
)
def test_stream_requires_complete_ordered_bounded_progress(wire, fault):
    payload, body, _, _ = wire
    raw = streamed(body)
    events = [json.loads(line) for line in raw["body"].splitlines()]
    if fault == "early_result":
        events = [events[-1], *events[:-1]]
    elif fault == "trailing":
        events.append(events[0])
    elif fault == "unknown":
        events.insert(1, {"type": "unknown"})
    elif fault == "bool":
        events[1]["completed_simulations"] = True
    elif fault == "negative":
        events[0]["completed_simulations"] = -1
    elif fault == "overshoot":
        events[1]["completed_simulations"] = 545
    elif fault == "regression":
        events[1]["completed_simulations"] = -1
    elif fault == "incomplete":
        events.pop(2)
    else:
        events.append(events[-1])
    raw["body"] = "".join(json.dumps(event) + "\n" for event in events)
    with pytest.raises(acceptance.Refusal):
        acceptance.strict_stream(raw, payload)


def test_success_binds_one_request_receipts_and_full_evidence(
    challenge, wire, tmp_path, monkeypatch
):
    payload, body, state, applied = wire
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 20.0)

    def case(*args, **kwargs):
        assert args == (10, "double")
        assert kwargs == {
            "pie": True,
            "stage": "midgame",
            "seed": 100,
            "simulations": 544,
            "heads": True,
            "placements": 12,
        }
        return state, payload

    calls = []

    def http(actual, request_id, deadline):
        calls.append((actual, request_id, deadline))
        return streamed(body)

    monkeypatch.setattr(acceptance, "make_case", case)
    monkeypatch.setattr(acceptance, "http_request", http)
    result = acceptance.run_acceptance(challenge, tmp_path, "e" * 32)
    started = json.loads((tmp_path / "acceptance-started.json").read_text())
    assert len(calls) == 1 and calls[0][2] == 355.0
    assert applied == [([0], [12])]
    assert result["checks"]["standard_1"]["status"] == "passed"
    assert result["invocation_id"] == started["invocation_id"] == "e" * 32
    assert all(
        result[key] == started[key] == challenge[key] for key in acceptance.BINDINGS
    )
    assert result["started_monotonic"] == started["started_monotonic"] == 20.0
    pin = result["evidence"][0]
    evidence = (tmp_path / pin["path"]).read_bytes()
    assert len(evidence) == pin["bytes"] <= 4 * 1024 * 1024
    assert (
        hashlib.sha256(evidence).hexdigest()
        == pin["sha256"]
        == result["checks"]["standard_1"]["evidence_sha256"]
    )
    record = json.loads(evidence)
    assert record["request"] == payload and record["response"] == body
    assert record["events"][-1]["result"] == body
    assert record["http_response"] == streamed(body)
    assert (
        record["public_url"] == acceptance.URL and record["origin"] == acceptance.ORIGIN
    )
    with pytest.raises(acceptance.Refusal, match="replay"):
        acceptance.run_acceptance(challenge, tmp_path, "f" * 32)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "model",
        "visits",
        "probability",
        "native",
        "nonfinite",
        "expired",
        "case",
        "evidence_size",
    ],
)
def test_failed_validation_writes_bound_failure(
    challenge, wire, tmp_path, monkeypatch, fault
):
    payload, body, state, _ = wire
    now = [20.0]
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: now[0])

    def case(*args, **kwargs):
        if fault == "case":
            raise RuntimeError("PRIVATE detail must not escape")
        return state, payload

    if fault == "model":
        body["model_version"] = "sha256-" + "d" * 64
    elif fault == "visits":
        body["root_visits"] = [1, 1]
    elif fault == "probability":
        body["outcome"]["win"] = 0.1
    elif fault == "native":

        def illegal(*args):
            raise RuntimeError("illegal")

        state.apply_many = illegal
    elif fault == "nonfinite":
        body["root_q"][0] = float("nan")
    elif fault == "evidence_size":
        monkeypatch.setattr(acceptance, "MAX_EVIDENCE", 1000)

    def http(*args):
        if fault == "expired":
            now[0] = 361.0
        return streamed(body)

    monkeypatch.setattr(acceptance, "make_case", case)
    monkeypatch.setattr(acceptance, "http_request", http)
    result = acceptance.run_acceptance(challenge, tmp_path, "e" * 32)
    assert result["checks"]["standard_1"]["status"] == "failed"
    assert result["challenge"] == challenge["challenge"]
    assert result["failure"] is not None
    assert "PRIVATE" not in (tmp_path / "acceptance-result.json").read_text()
    pin = result["evidence"][0]
    assert (
        hashlib.sha256((tmp_path / pin["path"]).read_bytes()).hexdigest()
        == pin["sha256"]
    )
    assert pin["bytes"] <= acceptance.MAX_EVIDENCE


def test_insufficient_budget_and_orphan_result_do_not_send_http(
    challenge, tmp_path, monkeypatch
):
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 350.0)
    monkeypatch.setattr(
        acceptance, "make_case", lambda *args, **kwargs: pytest.fail("no case budget")
    )
    result = acceptance.run_acceptance(challenge, tmp_path, "e" * 32)
    assert result["failure"] == "insufficient-acceptance-budget"
    other = tmp_path / "other"
    other.mkdir()
    (other / "acceptance-result.json").write_text("previous")
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 20.0)
    with pytest.raises(acceptance.Refusal, match="replay"):
        acceptance.run_acceptance(challenge, other, "e" * 32)
    assert not (other / "acceptance-started.json").exists()


def test_root_cpu_guard_rejects_before_request(tmp_path, monkeypatch):
    path = tmp_path / "challenge.json"
    path.write_text("{}")
    monkeypatch.setattr(acceptance.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        acceptance, "run_acceptance", lambda *args: pytest.fail("must not run")
    )
    with pytest.raises(acceptance.Refusal, match="root-cpu"):
        acceptance.main(["--challenge", str(path)])


def test_main_uses_system_boot_and_invocation(challenge, tmp_path, monkeypatch, capsys):
    path = tmp_path / "challenge.json"
    path.write_text(json.dumps(challenge))
    monkeypatch.setattr(acceptance.os, "geteuid", lambda: 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("INVOCATION_ID", "e" * 32)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 20.0)
    original_read = Path.read_text

    def read_text(self, *args, **kwargs):
        return (
            "boot-one\n"
            if str(self) == "/proc/sys/kernel/random/boot_id"
            else original_read(self, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "read_text", read_text)
    called = []

    def run(value, directory, invocation):
        called.append((value, directory, invocation))
        return {"checks": {"standard_1": {"status": "passed"}}}

    monkeypatch.setattr(acceptance, "run_acceptance", run)
    acceptance.main(["--challenge", str(path)])
    assert called == [(challenge, tmp_path, "e" * 32)]
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    monkeypatch.delenv("INVOCATION_ID")
    with pytest.raises(acceptance.Refusal, match="invocation"):
        acceptance.main(["--challenge", str(path)])
    assert len(called) == 1
    monkeypatch.setattr(acceptance.os, "geteuid", lambda: 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(acceptance.Refusal, match="root-cpu"):
        acceptance.main(["--challenge", str(path)])


def test_receipts_do_not_overwrite_or_follow_symlinks(tmp_path):
    receipt = tmp_path / "receipt.json"
    acceptance.atomic_new(receipt, {"original": True})
    with pytest.raises(FileExistsError):
        acceptance.atomic_new(receipt, {})
    alias = tmp_path / "alias.json"
    alias.symlink_to(receipt)
    with pytest.raises(FileExistsError):
        acceptance.atomic_new(alias, {})
    assert json.loads(receipt.read_text()) == {"original": True}


@pytest.mark.native
def test_actual_native_v3_standard_case_is_legal_and_declared():
    pytest.importorskip("deltrel_native")
    state, payload = acceptance.make_case(
        10,
        "double",
        pie=True,
        stage="midgame",
        seed=100,
        simulations=544,
        heads=True,
        placements=12,
    )
    acceptance.validate_request(payload)
    assert state.node_count == 275 and sum(x != -1 for x in payload["stones"]) == 12
    assert (
        not payload["terminal"]
        and not payload["opening"]
        and not payload["swap_available"]
    )
