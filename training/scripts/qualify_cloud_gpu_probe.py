"""CPU-only HTTP client for one challenged, exclusive cloud GPU qualification.

No model is loaded and no Torch device or CUDA API is called. Native CPU states
establish legality; the existing response schema validates network probabilities.
HTTP workers use only stdlib, receive credentials through stdin, and have a hard
parent-enforced timeout, including a peer which keeps sending incomplete lines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import math
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, TypeGuard
import urllib.request

REQUIRED = ["semantic_16", "standard_1", "standard_8", "deep_1"]
BINDINGS = (
    "run_id",
    "plan_sha256",
    "boot_id",
    "challenge",
    "probe_unit",
    "model_identity",
    "model_step",
)
MAX_RESPONSE = 2 * 1024 * 1024
MAX_EVIDENCE = 16 * 1024 * 1024
URL = "http://127.0.0.1:8081/v2/move"


class Refusal(RuntimeError):
    """Constant safe error code; never includes HTTP headers or credentials."""


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def atomic_new(path: Path, value: Any) -> bytes:
    """Atomic, fsynced publication which cannot replace an existing receipt."""
    raw = encoded(value)
    fd, name = tempfile.mkstemp(prefix=".probe-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return raw


def _number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_challenge(value: Any, *, boot: str, invocation: str, now: float) -> dict:
    if not isinstance(value, dict):
        raise Refusal("invalid-challenge")
    if (
        value.get("format") != "deltrelserve.gpu-qualification-challenge"
        or value.get("schema_version") != 1
    ):
        raise Refusal("challenge-version")
    for key in ("plan_sha256", "challenge"):
        if not isinstance(value.get(key), str) or not re.fullmatch(
            r"[a-f0-9]{64}", value[key]
        ):
            raise Refusal("challenge-binding")
    if not re.fullmatch(r"[a-f0-9]{32}", invocation):
        raise Refusal("invocation-required")
    if value.get("boot_id") != boot:
        raise Refusal("challenge-boot")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,48}", run_id
    ):
        raise Refusal("challenge-run")
    if value.get("probe_unit") != f"deltrelserve-qualification-{run_id}-probe.service":
        raise Refusal("challenge-unit")
    if (
        not re.fullmatch(r"sha256-[a-f0-9]{64}", str(value.get("model_identity")))
        or type(value.get("model_step")) is not int
        or value["model_step"] != 572377
    ):
        raise Refusal("challenge-model")
    if value.get("required_checks") != REQUIRED:
        raise Refusal("challenge-checks")
    issued, deadline = value.get("issued_monotonic"), value.get("deadline_monotonic")
    if (
        not _number(issued)
        or not _number(deadline)
        or not issued <= now < deadline - 5
        or not 0 < deadline - issued <= 360
    ):
        raise Refusal("challenge-deadline")
    return value


class Evidence:
    def __init__(self, directory: Path):
        self.directory = directory
        (directory / "probe-evidence").mkdir(mode=0o700)
        self.pins: list[dict] = []
        self.total = 0

    def save(self, name: str, value: Any) -> dict:
        pure = PurePosixPath(name)
        if len(pure.parts) != 1 or pure.suffix != ".json":
            raise Refusal("unsafe-evidence-name")
        raw = encoded(value)
        if self.total + len(raw) > MAX_EVIDENCE or len(self.pins) >= 64:
            raise Refusal("evidence-budget")
        relative = "probe-evidence/" + name
        atomic_new(self.directory / relative, value)
        pin = {
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        self.pins.append(pin)
        self.total += len(raw)
        return pin


def make_case(
    rings: int,
    mode: str,
    *,
    handicap=1,
    pie=False,
    stage="midgame",
    pda=0,
    seed=0,
    simulations=32,
    heads=True,
    placements=None,
):
    native = importlib.import_module("deltrel_native")
    from deltreltrain.contracts import RULES_HASH_WIRE

    if native.native_search_algorithm_id() != "gumbel-completed-q-v3-conditional-keep":
        raise Refusal("native-search-identity")
    state = native.StateBatch(rings, 1, mode=mode, handicap=handicap, pie=pie)
    order = list(range(state.node_count))
    random.Random(572377 + seed).shuffle(order)
    count = (
        placements
        if placements is not None
        else (1 if pie else 13 if handicap == 9 else 8 + seed % 4)
    )
    for action in order[:count]:
        state.apply_many([0], [action])
    if stage == "swapped":
        state.apply_many([0], [state.node_count])
    data = state.data()

    def members(words):
        return [
            n for n in range(state.node_count) if int(words[n // 64]) & (1 << (n % 64))
        ]

    stones = [-1] * state.node_count
    for player, words in enumerate((data.zero_bits, data.one_bits)):
        for node in members(words):
            stones[node] = player
    payload = {
        "schema_version": 3,
        "rules_hash": RULES_HASH_WIRE,
        "rings": rings,
        "stones": stones,
        "mode": mode,
        "handicap": handicap,
        "pda": pda,
        **{
            key: bool(getattr(data, key)[0])
            for key in ("opening", "terminal", "pie", "swap_available", "swapped")
        },
        **{key: int(getattr(data, key)[0]) for key in ("to_move", "moves_left")},
        "history": {
            key: members(getattr(data, source))
            for key, source in (
                ("current_turn", "current_turn_bits"),
                ("previous_turn", "previous_turn_bits"),
                ("own_previous_turn", "own_previous_turn_bits"),
                ("handicap_stones", "handicap_bits"),
            )
        },
        "search": {
            "simulations": simulations,
            "max_considered": 64 if simulations == 4096 else 16,
            "seed": 572377 + seed,
        },
        "include_predictions": True,
        "include_network_output": heads,
    }
    if (
        payload["terminal"]
        or stage == "pending"
        and not payload["swap_available"]
        or stage == "swapped"
        and not payload["swapped"]
    ):
        raise Refusal("native-case-state")
    return state, payload


def semantic_cases():
    specs = [(r, m, {}) for r in (4, 6, 8, 10) for m in ("classic", "double")]
    specs += [
        (10, m, {"pie": True, "stage": s})
        for m in ("classic", "double")
        for s in ("pending", "swapped")
    ]
    specs += [
        (10, m, {"handicap": 9, "pda": p})
        for m in ("classic", "double")
        for p in (-3, 3)
    ]
    return [
        make_case(r, m, seed=i, simulations=32 if i < 8 else 64, **extra)
        for i, (r, m, extra) in enumerate(specs)
    ]


def validate_response(body, payload, state, binding, request_id):
    from deltrelserve.schemas import AnalyzeResponse

    try:
        response = AnalyzeResponse.model_validate(body)
    except Exception:
        raise Refusal("response-schema") from None
    if (
        response.model_step != binding["model_step"]
        or response.model_version != binding["model_identity"]
        or response.request_id != request_id
    ):
        raise Refusal("response-identity")
    if sum(response.root_visits) != payload["search"]["simulations"]:
        raise Refusal("response-budget")
    if (
        (response.variant.mode, response.variant.handicap, response.variant.pie)
        != (payload["mode"], payload["handicap"], payload["pie"])
        or response.swap_available != payload["swap_available"]
        or not response.history_known
    ):
        raise Refusal("response-state")
    if (
        payload.get("include_predictions")
        and response.predictions is None
        or payload.get("include_network_output")
        and response.network_output is None
    ):
        raise Refusal("response-diagnostics")
    if payload["stones"][response.action.node] != -1:
        raise Refusal("response-illegal-placement")
    action = state.node_count if response.swap_recommended else response.action.code
    try:
        state.apply_many([0], [action])
    except Exception:
        raise Refusal("response-native-illegal") from None
    return action


def decode_http(raw: dict, payload: dict, streaming: bool):
    if raw.get("status") != 200:
        raise Refusal("http-status")
    text = raw["body"]
    if not streaming:
        return json.loads(text), []
    events = [json.loads(line) for line in text.splitlines() if line]
    progress, results = [], []
    for event in events:
        if event["type"] == "error":
            raise Refusal("stream-error")
        if event["type"] == "progress":
            if event["total_simulations"] != payload["search"]["simulations"]:
                raise Refusal("stream-budget")
            progress.append(event["completed_simulations"])
        elif event["type"] == "result":
            results.append(event["result"])
    if (
        len(results) != 1
        or not progress
        or progress != sorted(progress)
        or progress[-1] != payload["search"]["simulations"]
    ):
        raise Refusal("stream-completion")
    return results[0], events


def http_request(payload, token, request_id, streaming, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Refusal("http-deadline")
    request = {
        "payload": payload,
        "token": token,
        "request_id": request_id,
        "streaming": streaming,
    }
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--http-worker"],
            input=json.dumps(request),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=min(185, remaining),
            check=True,
        )
    except subprocess.TimeoutExpired:
        raise Refusal("http-deadline") from None
    except (subprocess.SubprocessError, OSError):
        raise Refusal("http-worker-failed") from None
    return json.loads(child.stdout)


def _http_worker():
    request = json.load(sys.stdin)
    headers = {
        "Authorization": "Bearer " + request["token"],
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson"
        if request["streaming"]
        else "application/json",
        "X-Request-ID": request["request_id"],
    }

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise Refusal("redirect-refused")

    req = urllib.request.Request(
        URL, data=json.dumps(request["payload"]).encode(), headers=headers
    )
    with urllib.request.build_opener(NoRedirect).open(req, timeout=180) as response:
        body = response.read(MAX_RESPONSE + 1)
        if len(body) > MAX_RESPONSE:
            raise Refusal("response-size")
        print(json.dumps({"status": response.status, "body": body.decode()}))


def optional_deep_fits(remaining, single_seconds):
    return remaining >= max(120, single_seconds * 12) + 15


def run_probe(challenge: dict, directory: Path, invocation: str, token: str):
    binding = {key: challenge[key] for key in BINDINGS}
    started = time.monotonic()
    base = {
        "schema_version": 1,
        **binding,
        "invocation_id": invocation,
        "started_monotonic": started,
    }
    atomic_new(
        directory / "probe-started.json",
        {"format": "deltrelserve.gpu-qualification-probe-started", **base},
    )
    evidence = Evidence(directory)
    checks, optional = {}, {}
    work_deadline = challenge["deadline_monotonic"] - 5
    failure = None
    elapsed = 0.0

    def batch(name, cases, concurrent, minimum):
        if work_deadline - time.monotonic() < minimum:
            raise Refusal("insufficient-required-budget")
        began = time.monotonic()
        records = []

        def request(index_case):
            index, (state, payload) = index_case
            request_id = challenge["challenge"][:16] + "-" + name + "-" + str(index)
            streaming = index % 2 == 1 if name == "semantic_16" else True
            raw = http_request(payload, token, request_id, streaming, work_deadline)
            body, events = decode_http(raw, payload, streaming)
            return (
                index,
                payload,
                body,
                events,
                validate_response(body, payload, state, binding, request_id),
            )

        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            outputs = (
                map(request, enumerate(cases))
                if concurrent == 1
                else pool.map(request, enumerate(cases))
            )
            for index, payload, body, events, action in outputs:
                pin = evidence.save(
                    f"{name}-{index:02}.json",
                    {
                        "request": payload,
                        "response": body,
                        "events": events,
                        "native_legal_action": action,
                    },
                )
                records.append(pin)
        elapsed = time.monotonic() - began
        summary = evidence.save(
            name + ".json",
            {
                "status": "passed",
                "concurrency": concurrent,
                "requests": len(cases),
                "seconds": elapsed,
                "simulations_each": cases[0][1]["search"]["simulations"]
                if name != "semantic_16"
                else [32, 64],
                "records": records,
            },
        )
        return {"status": "passed", "evidence_sha256": summary["sha256"]}, elapsed

    try:
        checks["semantic_16"], _ = batch("semantic_16", semantic_cases(), 1, 30)
        for name, count, simulations, minimum in (
            ("standard_1", 1, 544, 10),
            ("standard_8", 8, 544, 30),
            ("deep_1", 1, 4096, 45),
        ):
            cases = [
                make_case(
                    10,
                    "double",
                    pie=True,
                    seed=100 + index,
                    simulations=simulations,
                    heads=False,
                    placements=12 + index % 12,
                )
                for index in range(count)
            ]
            checks[name], elapsed = batch(name, cases, count, minimum)
    except Exception as exc:
        failure = str(exc) if isinstance(exc, Refusal) else type(exc).__name__
    if failure is None:
        if optional_deep_fits(work_deadline - time.monotonic(), elapsed):
            try:
                optional["deep_8"], _ = batch(
                    "deep_8",
                    [
                        make_case(
                            10,
                            "double",
                            pie=True,
                            seed=200 + index,
                            simulations=4096,
                            heads=False,
                            placements=12 + index % 12,
                        )
                        for index in range(8)
                    ],
                    8,
                    max(120, elapsed * 12) + 15,
                )
            except Exception as exc:
                reason = str(exc) if isinstance(exc, Refusal) else type(exc).__name__
                pin = evidence.save(
                    "deep_8-failed.json", {"status": "failed", "reason": reason}
                )
                optional["deep_8"] = {
                    "status": "failed",
                    "evidence_sha256": pin["sha256"],
                }
        else:
            optional["deep_8"] = {
                "status": "not_run",
                "reason": "insufficient-projected-budget",
            }
    for name in REQUIRED:
        if name not in checks:
            pin = evidence.save(
                name + "-failed.json",
                {"status": "failed", "reason": failure or "not-completed"},
            )
            checks[name] = {"status": "failed", "evidence_sha256": pin["sha256"]}
    completed = time.monotonic()
    if completed > challenge["deadline_monotonic"]:
        raise Refusal("result-deadline")
    result = {
        "format": "deltrelserve.gpu-qualification-probe-result",
        **base,
        "completed_monotonic": completed,
        "checks": checks,
        "optional_checks": optional,
        "evidence": evidence.pins,
        "failure": failure,
        "interpretation": "Protocol, legality and runtime qualification; not Elo or a production capacity guarantee.",
    }
    atomic_new(directory / "probe-result.json", result)
    return result


def main(argv=None):
    if argv is None and sys.argv[1:] == ["--http-worker"]:
        _http_worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.challenge
    if (
        os.geteuid() != 0
        or path.is_symlink()
        or not path.is_file()
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
    token = os.environ.get("DELTRELSERVE_BEARER_TOKEN")
    if not token:
        raise Refusal("protected-credential-unavailable")
    result = run_probe(challenge, path.parent, invocation, token)
    if any(item["status"] != "passed" for item in result["checks"].values()):
        raise Refusal("required-qualification-check-failed")
    print(
        json.dumps(
            {
                "status": "passed",
                "required_checks": REQUIRED,
                "optional_checks": result["optional_checks"],
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "failure": str(error)
                    if isinstance(error, Refusal)
                    else type(error).__name__
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
