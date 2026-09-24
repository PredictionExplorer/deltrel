"""Live champion progress shares the original search, limits and cancellation."""

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from deltrelserve.app import create_app
from deltrelserve.config import LimitConfig, SecurityConfig
from deltrelserve.runtime import AnalysisError
from deltrelserve.schemas import AnalyzeResponse
from test_deltrelserve import (
    FakeService,
    request_payload,
    response_payload,
    server_config,
)


STREAM_HEADERS = {"Accept": "application/x-ndjson", "X-Request-ID": "live-search"}


class ProgressService(FakeService):
    def __init__(self, error=None):
        super().__init__()
        self.calls = 0
        self.error = error

    def analyze(self, request, cancellation, progress=None):
        self.calls += 1
        if progress is not None:
            for completed in range(request.search.simulations + 1):
                progress(completed, request.search.simulations)
                assert not cancellation.wait(0.02)
        if self.error is not None:
            raise self.error
        return response_payload()


@pytest.mark.parametrize("path", ["/v2/analyze", "/v2/move"])
def test_streams_progress_and_one_validated_legacy_result(tmp_path, path):
    service = ProgressService()
    with TestClient(create_app(server_config(tmp_path), service=service)) as client:
        response = client.post(path, json=request_payload(), headers=STREAM_HEADERS)
        legacy = client.post(
            path, json=request_payload(), headers={"X-Request-ID": "live-search"}
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-request-id"] == "live-search"
    assert response.content.endswith(b"\n")
    events = [json.loads(line) for line in response.iter_lines()]
    updates = events[:-1]
    assert len(updates) >= 2
    assert all(event["type"] == "progress" for event in updates)
    completed = [event["completed_simulations"] for event in updates]
    assert completed == sorted(set(completed))
    assert completed[-1] == 4
    assert all(event["total_simulations"] == 4 for event in updates)
    assert events[-1]["type"] == "result"
    result = events[-1]["result"]
    AnalyzeResponse.model_validate(result)
    assert {k: v for k, v in result.items() if k != "timing_ms"} == {
        k: v for k, v in legacy.json().items() if k != "timing_ms"
    }
    assert service.calls == 2


@pytest.mark.parametrize(
    "error,code,private",
    [
        (
            AnalysisError("model_unavailable", "model is unavailable", status_code=503),
            "model_unavailable",
            False,
        ),
        (RuntimeError("private model path"), "internal_error", True),
    ],
)
def test_stream_errors_have_one_terminal_event_and_redact_internal_details(
    tmp_path, error, code, private
):
    with TestClient(
        create_app(server_config(tmp_path), service=ProgressService(error))
    ) as client:
        response = client.post(
            "/v2/move", json=request_payload(), headers=STREAM_HEADERS
        )
    events = [json.loads(line) for line in response.iter_lines()]
    assert all(event["type"] == "progress" for event in events[:-1])
    assert events[-1]["type"] == "error"
    assert events[-1]["error"]["code"] == code
    assert events[-1]["error"]["request_id"] == "live-search"
    if private:
        assert "private model path" not in response.text


def test_stream_keeps_auth_and_budget_rejections_as_http_json(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DELTRELSERVE_TOKEN", "secret")
    service = ProgressService()
    config = server_config(
        tmp_path, security=SecurityConfig(bearer_token_env="TEST_DELTRELSERVE_TOKEN")
    )
    with TestClient(create_app(config, service=service)) as client:
        unauthorized = client.post(
            "/v2/move", json=request_payload(), headers=STREAM_HEADERS
        )
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "unauthorized"
        payload = request_payload()
        payload["search"]["simulations"] = 17
        rejected = client.post(
            "/v2/move",
            json=payload,
            headers={**STREAM_HEADERS, "Authorization": "Bearer secret"},
        )
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "search_budget_exceeded"
    assert service.calls == 0


class HeldProgressService(FakeService):
    """Keep the worker alive after cancellation to test deferred slot release."""

    def __init__(self):
        super().__init__()
        self.cancelled = threading.Event()
        self.finish = threading.Event()

    def analyze(self, request, cancellation, progress=None):
        if progress is not None:
            progress(1, request.search.simulations)
        cancellation.wait(2)
        if cancellation.is_set():
            self.cancelled.set()
        self.finish.wait(2)
        return response_payload()


def asgi_exchange(app, *, on_body, disconnected):
    supplied = False

    async def receive():
        nonlocal supplied
        if not supplied:
            supplied = True
            return {
                "type": "http.request",
                "body": json.dumps(request_payload()).encode(),
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            await on_body(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v2/move",
        "query_string": b"",
        "root_path": "",
        "server": ("localhost", 80),
        "client": ("127.0.0.1", 1234),
        "headers": [
            (b"content-type", b"application/json"),
            (b"accept", b"application/x-ndjson"),
        ],
    }
    return app(scope, receive, send)


def test_live_asgi_disconnect_cancels_worker_and_holds_slot_until_it_stops(tmp_path):
    service = HeldProgressService()
    app = create_app(server_config(tmp_path), service=service)

    async def run():
        first_progress = asyncio.Event()
        disconnected = asyncio.Event()
        bodies = []

        async def on_body(body):
            bodies.append(body)
            if b'"type":"progress"' in body:
                first_progress.set()

        task = asyncio.create_task(
            asgi_exchange(app, on_body=on_body, disconnected=disconnected)
        )
        try:
            await asyncio.wait_for(first_progress.wait(), 1)
            assert not task.done()
            assert b'"type":"result"' not in b"".join(bodies)
            disconnected.set()
            assert await asyncio.to_thread(service.cancelled.wait, 1)
            assert app.state.analysis_slots.locked()
        finally:
            service.finish.set()
            disconnected.set()
            await asyncio.wait_for(task, 2)
        # The native worker's completion, not the HTTP disconnect, frees the slot.
        await asyncio.wait_for(app.state.analysis_slots.acquire(), 1)
        app.state.analysis_slots.release()

    asyncio.run(run())


def test_stream_timeout_still_runs_when_reader_is_backpressured(tmp_path):
    service = HeldProgressService()
    config = server_config(
        tmp_path,
        limits=LimitConfig(
            max_concurrency=1,
            max_request_bytes=4096,
            request_timeout_seconds=0.12,
            queue_timeout_seconds=0.05,
        ),
    )
    app = create_app(config, service=service)

    async def run():
        first_progress = asyncio.Event()
        release_reader = asyncio.Event()
        disconnected = asyncio.Event()
        bodies = []

        async def on_body(body):
            bodies.append(body)
            if b'"type":"progress"' in body:
                first_progress.set()
                await release_reader.wait()

        task = asyncio.create_task(
            asgi_exchange(app, on_body=on_body, disconnected=disconnected)
        )
        try:
            await asyncio.wait_for(first_progress.wait(), 1)
            assert await asyncio.to_thread(service.cancelled.wait, 1)
            assert app.state.analysis_slots.locked()
        finally:
            service.finish.set()
            release_reader.set()
            await asyncio.wait_for(task, 2)
        events = [json.loads(line) for line in b"".join(bodies).splitlines()]
        assert events[-1]["type"] == "error"
        assert events[-1]["error"]["code"] == "analysis_timeout"
        await asyncio.wait_for(app.state.analysis_slots.acquire(), 1)
        app.state.analysis_slots.release()

    asyncio.run(run())
