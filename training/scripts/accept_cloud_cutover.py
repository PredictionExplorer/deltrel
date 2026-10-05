"""One challenged public-route acceptance request, with no credentials or GPU use.

The parent is a root, CPU-only systemd service. A disposable stdlib-only HTTP
child sends one fixed-origin Standard request; its parent enforces a total
timeout even if the peer keeps streaming. Receipts are atomic and no-clobber.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
from typing import Any, TypeGuard
import urllib.request

PROBE_HELPER_SHA256 = "325b994c007fdcb11a9f6f9d85fc353a51445938b8bfb40dafcd879a13d0dc24"


def _load_probe_helpers() -> Any:
    # The sealed operator bundle carries this pinned sibling. The serving
    # release intentionally remains unchanged and need not package newer scripts.
    path = Path(__file__).with_name("qualify_cloud_gpu_probe.py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PROBE_HELPER_SHA256:
        raise RuntimeError("probe-helper-checksum-mismatch")
    module = types.ModuleType("_cutover_probe_helpers")
    module.__file__ = str(path)
    # Compile the verified snapshot, never a later read or timestamp-matching pyc.
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


_helpers = _load_probe_helpers()
Refusal = _helpers.Refusal
atomic_new = _helpers.atomic_new
decode_http = _helpers.decode_http
encoded = _helpers.encoded
make_case = _helpers.make_case
validate_response = _helpers.validate_response

URL = "https://deltrel.com/v2/move"
ORIGIN = "https://deltrel.com"
MAX_RESPONSE = 2 * 1024 * 1024
MAX_EVIDENCE = 4 * 1024 * 1024
RECEIPT_RESERVE = 5.0
BINDINGS = (
    "run_id",
    "plan_sha256",
    "boot_id",
    "challenge",
    "acceptance_unit",
    "model_identity",
    "model_step",
    "issued_monotonic",
    "deadline_monotonic",
)
SAFE_FAILURES = frozenset(
    {
        "acceptance-deadline",
        "insufficient-acceptance-budget",
        "evidence-budget",
        "http-deadline",
        "http-worker-failed",
        "http-worker-output",
        "http-status",
        "native-case-contract",
        "native-search-identity",
        "native-case-state",
        "response-schema",
        "response-identity",
        "response-budget",
        "response-state",
        "response-diagnostics",
        "response-illegal-placement",
        "response-native-illegal",
        "stream-error",
        "stream-budget",
        "stream-completion",
        "stream-format",
        "stream-order",
        "stream-progress",
        "response-size",
    }
)


def _finite(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_challenge(value: Any, *, boot: str, invocation: str, now: float) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "format",
        "schema_version",
        *BINDINGS,
    }:
        raise Refusal("challenge-fields")
    if (
        value["format"] != "deltrelserve.cloud-cutover-challenge"
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise Refusal("challenge-version")
    for field in ("plan_sha256", "challenge"):
        if not isinstance(value[field], str) or not re.fullmatch(
            r"[a-f0-9]{64}", value[field]
        ):
            raise Refusal("challenge-binding")
    if not re.fullmatch(r"[a-f0-9]{32}", invocation):
        raise Refusal("invocation-required")
    if not boot or value["boot_id"] != boot:
        raise Refusal("challenge-boot")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,48}", run_id
    ):
        raise Refusal("challenge-run")
    if value["acceptance_unit"] != f"deltrelserve-cutover-{run_id}-acceptance.service":
        raise Refusal("challenge-unit")
    if (
        not isinstance(value["model_identity"], str)
        or not re.fullmatch(r"sha256-[a-f0-9]{64}", value["model_identity"])
        or type(value["model_step"]) is not int
        or value["model_step"] != 572377
    ):
        raise Refusal("challenge-model")
    issued, deadline = value["issued_monotonic"], value["deadline_monotonic"]
    if (
        not _finite(now)
        or not _finite(issued)
        or not _finite(deadline)
        or not issued <= now < deadline - RECEIPT_RESERVE
        or not 0 < deadline - issued <= 360
    ):
        raise Refusal("challenge-deadline")
    return value


def validate_request(payload: Any) -> None:
    """Constrain the child transport to the single predeclared public request."""
    if not isinstance(payload, dict):
        raise Refusal("native-case-contract")
    expected = {
        "schema_version": 3,
        "rules_hash": "fnv1a64:46e4fbcff4e17fd3",
        "rings": 10,
        "mode": "double",
        "handicap": 1,
        "pie": True,
        "opening": False,
        "terminal": False,
        "swap_available": False,
        "swapped": False,
        "pda": 0,
        "include_predictions": True,
        "include_network_output": True,
    }
    if set(payload) != {
        *expected,
        "stones",
        "to_move",
        "moves_left",
        "history",
        "search",
    }:
        raise Refusal("native-case-contract")
    if any(
        type(payload.get(key)) is not type(value) or payload[key] != value
        for key, value in expected.items()
    ):
        raise Refusal("native-case-contract")
    search = payload.get("search")
    if (
        not isinstance(search, dict)
        or set(search) != {"simulations", "max_considered", "seed"}
        or type(search["simulations"]) is not int
        or search["simulations"] != 544
        or type(search["max_considered"]) is not int
        or search["max_considered"] != 16
        or type(search["seed"]) is not int
        or search["seed"] != 572477
    ):
        raise Refusal("native-case-contract")
    stones, history = payload.get("stones"), payload.get("history")
    history_fields = {
        "current_turn",
        "previous_turn",
        "own_previous_turn",
        "handicap_stones",
    }
    if (
        not isinstance(stones, list)
        or len(stones) != 275
        or any(type(stone) is not int or stone not in (-1, 0, 1) for stone in stones)
        or sum(stone != -1 for stone in stones) != 12
        or type(payload["to_move"]) is not int
        or payload["to_move"] not in (0, 1)
        or type(payload["moves_left"]) is not int
        or payload["moves_left"] not in (1, 2)
        or not isinstance(history, dict)
        or set(history) != history_fields
        or any(
            not isinstance(nodes, list)
            or len(nodes) > 275
            or any(type(node) is not int or not 0 <= node < 275 for node in nodes)
            for nodes in history.values()
        )
    ):
        raise Refusal("native-case-contract")


def http_request(payload: dict, request_id: str, deadline: float) -> dict:
    validate_request(payload)
    if not _finite(deadline):
        raise Refusal("http-deadline")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Refusal("http-deadline")
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--http-worker"],
            input=json.dumps(
                {"payload": payload, "request_id": request_id}, allow_nan=False
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=min(185.0, remaining),
            check=True,
        )
    except subprocess.TimeoutExpired:
        raise Refusal("http-deadline") from None
    except (subprocess.SubprocessError, OSError):
        raise Refusal("http-worker-failed") from None
    try:
        raw = json.loads(child.stdout)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"status", "body"}
            or type(raw["status"]) is not int
            or not isinstance(raw["body"], str)
        ):
            raise ValueError
        if len(raw["body"].encode("utf-8")) > MAX_RESPONSE:
            raise Refusal("response-size")
        return raw
    except (ValueError, TypeError):
        raise Refusal("http-worker-output") from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise Refusal("redirect-refused")


def _http_worker() -> None:
    request = json.load(sys.stdin)
    if (
        not isinstance(request, dict)
        or set(request) != {"payload", "request_id"}
        or not isinstance(request["request_id"], str)
        or not re.fullmatch(r"[a-z0-9-]{1,96}", request["request_id"])
    ):
        raise Refusal("http-input")
    validate_request(request["payload"])
    headers = {
        "Origin": ORIGIN,
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson",
        "X-Request-ID": request["request_id"],
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(request["payload"], allow_nan=False).encode(),
        headers=headers,
        method="POST",
    )
    # No environment proxies, cookies, authentication handler or redirected URL.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=180) as response:
        body = response.read(MAX_RESPONSE + 1)
        if len(body) > MAX_RESPONSE:
            raise Refusal("response-size")
        print(json.dumps({"status": response.status, "body": body.decode("utf-8")}))


def strict_stream(raw: dict, payload: dict) -> tuple[dict, list]:
    if raw.get("status") != 200:
        raise Refusal("http-status")
    text = raw.get("body")
    if (
        not isinstance(text, str)
        or not text.endswith("\n")
        or len(text.encode("utf-8")) > MAX_RESPONSE
    ):
        raise Refusal("stream-format")
    previous = -1
    terminal = False
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            raise Refusal("stream-format") from None
        if terminal or not isinstance(event, dict):
            raise Refusal("stream-order")
        if event.get("type") == "progress":
            completed, total = (
                event.get("completed_simulations"),
                event.get("total_simulations"),
            )
            if (
                set(event) != {"type", "completed_simulations", "total_simulations"}
                or type(completed) is not int
                or type(total) is not int
                or total != 544
                or not 0 <= completed <= total
                or completed < previous
            ):
                raise Refusal("stream-progress")
            previous = completed
        elif event.get("type") == "result":
            if (
                set(event) != {"type", "result"}
                or not isinstance(event["result"], dict)
                or previous != 544
            ):
                raise Refusal("stream-completion")
            terminal = True
        elif event.get("type") == "error":
            raise Refusal("stream-error")
        else:
            raise Refusal("stream-format")
    if not terminal:
        raise Refusal("stream-completion")
    return decode_http(raw, payload, True)


def _safe_failure(error: Exception) -> str:
    return (
        str(error)
        if isinstance(error, Refusal) and str(error) in SAFE_FAILURES
        else "acceptance-failed"
    )


def run_acceptance(challenge: dict, directory: Path, invocation: str) -> dict:
    names = ("acceptance-started.json", "acceptance-result.json", "acceptance-evidence")
    if any(os.path.lexists(directory / name) for name in names):
        raise Refusal("acceptance-replay")
    started = time.monotonic()
    if started >= challenge["deadline_monotonic"] - RECEIPT_RESERVE:
        raise Refusal("challenge-deadline")
    base = {
        "schema_version": 1,
        **{key: challenge[key] for key in BINDINGS},
        "invocation_id": invocation,
        "started_monotonic": started,
    }
    atomic_new(
        directory / names[0],
        {"format": "deltrelserve.cloud-cutover-acceptance-started", **base},
    )
    evidence_directory = directory / "acceptance-evidence"
    evidence_directory.mkdir(mode=0o700)
    request_id = challenge["challenge"][:24] + "-standard-1"
    record: dict[str, Any] = {
        "public_url": URL,
        "origin": ORIGIN,
        "request_id": request_id,
    }
    failure = None
    work_deadline = challenge["deadline_monotonic"] - RECEIPT_RESERVE
    try:
        if work_deadline - time.monotonic() < 10:
            raise Refusal("insufficient-acceptance-budget")
        state, payload = make_case(
            10,
            "double",
            pie=True,
            stage="midgame",
            seed=100,
            simulations=544,
            heads=True,
            placements=12,
        )
        validate_request(payload)
        record["request"] = payload
        if work_deadline - time.monotonic() < 10:
            raise Refusal("insufficient-acceptance-budget")
        raw = http_request(payload, request_id, work_deadline)
        record["http_response"] = raw
        body, events = strict_stream(raw, payload)
        record.update(response=body, events=events)
        record["native_legal_action"] = validate_response(
            body, payload, state, challenge, request_id
        )
        if time.monotonic() > work_deadline:
            raise Refusal("acceptance-deadline")
    except Exception as error:
        failure = _safe_failure(error)
    record.update(status="passed" if failure is None else "failed", failure=failure)
    try:
        raw_record = encoded(record)
    except (ValueError, TypeError):
        # Preserve the original response text on malformed/nonfinite JSON, while
        # never attempting to serialize its rejected parsed numeric objects.
        record.pop("response", None)
        record.pop("events", None)
        raw_record = encoded(record)
    if len(raw_record) > MAX_EVIDENCE:
        failure = "evidence-budget"
        record = {
            "status": "failed",
            "failure": failure,
            "request_id": request_id,
            "omitted_evidence_bytes": len(raw_record),
            "omitted_evidence_sha256": hashlib.sha256(raw_record).hexdigest(),
        }
        raw_record = encoded(record)
    pin = {
        "path": "acceptance-evidence/standard_1.json",
        "sha256": hashlib.sha256(raw_record).hexdigest(),
        "bytes": len(raw_record),
    }
    atomic_new(directory / pin["path"], record)
    completed = time.monotonic()
    if completed > challenge["deadline_monotonic"]:
        failure = "acceptance-deadline"
    result = {
        "format": "deltrelserve.cloud-cutover-acceptance",
        **base,
        "completed_monotonic": completed,
        "checks": {
            "standard_1": {
                "status": "passed" if failure is None else "failed",
                "evidence_sha256": pin["sha256"],
            }
        },
        "evidence": [pin],
        "failure": failure,
    }
    atomic_new(directory / names[1], result)
    return result


def main(argv=None) -> None:
    if argv is None and sys.argv[1:] == ["--http-worker"]:
        _http_worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.challenge.absolute()
    if (
        os.geteuid() != 0
        or not path.is_file()
        or path.resolve() != path
        or path.stat().st_size > 2**20
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or Path("/dev/nvidiactl").exists()
    ):
        raise Refusal("root-cpu-only-challenge-required")
    invocation = os.environ.get("INVOCATION_ID", "")
    challenge = validate_challenge(
        json.loads(path.read_bytes()),
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        invocation=invocation,
        now=time.monotonic(),
    )
    result = run_acceptance(challenge, path.parent, invocation)
    if result["checks"]["standard_1"]["status"] != "passed":
        raise Refusal("public-acceptance-failed")
    print(json.dumps({"status": "passed", "check": "standard_1"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"failure": _safe_failure(error)}), file=sys.stderr)
        raise SystemExit(1) from None
