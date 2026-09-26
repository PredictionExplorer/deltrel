"""Exercise real, distinct game positions through the public serving protocol.

Run on an operator machine with the native extension. Credentials are read only
from DELTRELSERVE_BEARER_TOKEN; neither credentials nor position bodies are logged.
This measures a small deterministic workload, not a production capacity promise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from deltrelserve.schemas import AnalyzeRequest, AnalyzeResponse
from deltreltrain.contracts import RULES_HASH_WIRE
from deltreltrain.native import load_deltrel_native, positions_from_native


def position_request(index: int, rings: int, simulations: int) -> dict[str, Any]:
    native = load_deltrel_native(required=True)
    assert native is not None
    states = native.StateBatch(rings, 1, mode="double", pie=True)
    moves = list(range(states.node_count))
    random.Random(7300 + index).shuffle(moves)
    for action in moves[: 12 + index % 12]:
        states.apply_many([0], [action])
    position = positions_from_native(states.data())[0]
    payload = {
        "schema_version": 3,
        "rules_hash": RULES_HASH_WIRE,
        "rings": rings,
        "stones": position.stones.tolist(),
        **{
            name: getattr(position, name)
            for name in (
                "to_move", "moves_left", "opening", "terminal", "mode",
                "handicap", "pie", "swap_available", "swapped", "pda",
            )
        },
        "history": {
            name: getattr(position, name).nonzero().flatten().tolist()
            for name in (
                "current_turn", "previous_turn", "own_previous_turn", "handicap_stones"
            )
        },
        "search": {
            "simulations": simulations,
            "max_considered": 16 if simulations <= 544 else 64,
            "seed": 9900 + index,
        },
        "include_predictions": True,
    }
    return AnalyzeRequest.model_validate(payload).model_dump()


def run_request(url: str, token: str, index: int, payload: dict[str, Any]) -> dict[str, Any]:
    request_id = f"cloud-benchmark-{time.time_ns()}-{index}"
    request = urllib.request.Request(
        url.rstrip("/") + "/v2/move",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "X-Request-ID": request_id,
        },
    )
    started = time.perf_counter()
    progress: list[int] = []
    result = None
    with urllib.request.urlopen(request, timeout=205) as response:
        for line in response:
            event = json.loads(line)
            if event["type"] == "error":
                raise RuntimeError(event["error"]["code"])
            if event["type"] == "progress":
                assert event["total_simulations"] == payload["search"]["simulations"]
                progress.append(event["completed_simulations"])
            elif event["type"] == "result":
                result = AnalyzeResponse.model_validate(event["result"])
    assert result is not None and result.request_id == request_id
    assert progress and progress == sorted(progress)
    assert progress[-1] == payload["search"]["simulations"]
    if result.action.kind == "place":
        assert payload["stones"][result.action.node] == -1
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "progress_events": len(progress),
        "model_step": result.model_step,
        "action": result.action.model_dump(),
        "win_probability": result.outcome.win,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--concurrency", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--rings", type=int, choices=(4, 6, 8, 10), default=10)
    parser.add_argument("--simulations", type=int, default=544)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    token = os.environ["DELTRELSERVE_BEARER_TOKEN"]
    payloads = [
        position_request(index + args.seed_offset, args.rings, args.simulations)
        for index in range(args.concurrency)
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_request, args.url, token, index, payload)
            for index, payload in enumerate(payloads)
        ]
        results = [future.result() for future in futures]
    print(json.dumps({
        "concurrency": args.concurrency,
        "rings": args.rings,
        "simulations": args.simulations,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }), flush=True)


if __name__ == "__main__":
    main()
