"""HTTP backpressure must bound GPU work without losing slots on disconnect."""

import asyncio
import threading
from contextlib import suppress

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from deltrelserve.app import (
    _AdmittedStreamingResponse,
    _admit,
    _run_bounded,
    _stream_bounded,
    create_app,
)
from deltrelserve.config import (
    LimitConfig,
    SearchConfig,
    SecurityConfig,
    ServerConfigError,
    load_server_config,
)
from deltrelserve.runtime import AnalysisError
from deltrelserve.schemas import AnalyzeRequest
from test_deltrelserve import (
    FakeService,
    request_payload,
    response_payload,
    server_config,
)
from test_deltrelserve_streaming import HeldProgressService


class BlockingService(FakeService):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.entered = threading.Event()
        self.finish = threading.Event()

    def analyze(self, request, cancellation, progress=None):
        self.calls += 1
        self.entered.set()
        assert self.finish.wait(3)
        return response_payload()


async def wait_until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("accept", ["application/json", "application/x-ndjson"])
def test_queue_full_is_retryable_http_before_stream_starts(tmp_path, accept):
    service = BlockingService()
    app = create_app(
        server_config(
            tmp_path,
            limits=LimitConfig(
                max_concurrency=1,
                max_queued_requests=1,
                request_timeout_seconds=2,
                queue_timeout_seconds=1,
            ),
        ),
        service=service,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/v2/move", json=request_payload()))
            second = None
            try:
                assert await asyncio.to_thread(service.entered.wait, 1)
                second = asyncio.create_task(
                    client.post("/v2/move", json=request_payload())
                )
                await wait_until(lambda: app.state.analysis_waiting == 1)
                rejected = await client.post(
                    "/v2/move", json=request_payload(), headers={"Accept": accept}
                )
                assert rejected.status_code == 503
                assert rejected.headers["retry-after"] == "1"
                assert rejected.headers["content-type"] == "application/json"
                assert rejected.json()["error"]["code"] == "service_busy"
                assert service.calls == 1
                capacity = (await client.get("/v2/health")).json()["capacity"]
                assert capacity == {
                    "active_requests": 1,
                    "queued_requests": 1,
                    "max_concurrency": 1,
                    "max_queued_requests": 1,
                }
            finally:
                service.finish.set()
                assert (await first).status_code == 200
                if second is not None:
                    assert (await second).status_code == 200
            assert service.calls == 2
            assert app.state.analysis_requests == 0
            assert app.state.analysis_waiting == 0
            assert not app.state.analysis_jobs

    asyncio.run(run())


def test_queue_timeout_is_json_with_retry_hint_even_for_stream_clients(tmp_path):
    app = create_app(
        server_config(
            tmp_path, limits=LimitConfig(max_concurrency=1, queue_timeout_seconds=0.02)
        ),
        service=FakeService(),
    )

    async def run():
        await app.state.analysis_slots.acquire()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                rejected = await client.post(
                    "/v2/analyze",
                    json=request_payload(),
                    headers={"Accept": "application/x-ndjson"},
                )
            assert rejected.status_code == 503
            assert rejected.json()["error"]["code"] == "service_busy"
            assert rejected.headers["retry-after"] == "1"
            assert app.state.analysis_requests == 0
            assert app.state.analysis_slots.locked()
        finally:
            app.state.analysis_slots.release()

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["disconnect", "cancel", "shutdown"])
def test_waiting_requests_leave_queue_without_starting_native_work(tmp_path, reason):
    app = create_app(server_config(tmp_path), service=FakeService())

    async def run():
        disconnected = asyncio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        request = Request({"type": "http"}, receive=receive)
        await app.state.analysis_slots.acquire()
        pending = asyncio.create_task(_admit(app, request))
        try:
            await wait_until(lambda: app.state.analysis_waiting == 1)
            if reason == "disconnect":
                disconnected.set()
            elif reason == "cancel":
                pending.cancel()
            else:
                app.state.analysis_stopping.set()
            if reason == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(AnalysisError) as exc:
                    await asyncio.wait_for(pending, 1)
                assert exc.value.code == (
                    "client_disconnected"
                    if reason == "disconnect"
                    else "service_unavailable"
                )
            assert app.state.analysis_waiting == 0
            assert app.state.analysis_requests == 0
            assert not app.state.analysis_jobs
            assert app.state.analysis_slots.locked()
        finally:
            app.state.analysis_slots.release()

    asyncio.run(run())


def test_simultaneous_admission_and_disconnect_returns_acquired_slot(tmp_path):
    app = create_app(server_config(tmp_path), service=FakeService())

    async def run():
        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(AnalysisError, match="disconnected"):
            await _admit(app, Request({"type": "http"}, receive=receive))
        assert app.state.analysis_requests == 0
        assert not app.state.analysis_slots.locked()
        await app.state.analysis_slots.acquire()
        assert app.state.analysis_slots.locked()
        app.state.analysis_slots.release()

    asyncio.run(run())


@pytest.mark.parametrize("field", ["request_timeout_seconds", "queue_timeout_seconds"])
@pytest.mark.parametrize("value", [False, "1", float("nan"), float("inf"), 0, -1])
def test_service_timeouts_are_finite_positive_numbers(field, value):
    with pytest.raises(ServerConfigError):
        LimitConfig(**{field: value})


@pytest.mark.parametrize("value", [False, 1.5, -1])
def test_waiting_queue_bound_is_a_nonnegative_integer(value):
    with pytest.raises(ServerConfigError):
        LimitConfig(max_queued_requests=value)


def test_cloud_config_matches_public_search_presets_and_bounds_resources():
    config = load_server_config("configs/deltrelserve-cloud.yaml")
    assert config.search == SearchConfig()
    assert config.search.named_presets() == {
        "standard": {"simulations": 544, "max_considered": 16},
        "deep": {"simulations": 4096, "max_considered": 64},
    }
    assert config.limits.max_concurrency == 8
    assert config.limits.max_queued_requests == 16
    assert config.inference.shared_batching
    assert config.host == "127.0.0.1"
    assert config.security.cors_allow_origins == ()
    assert config.security.bearer_token_env == "DELTRELSERVE_BEARER_TOKEN"


def test_diagnostic_health_and_schema_require_token_but_readiness_is_minimal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_CLOUD_TOKEN", "secret")
    config = server_config(
        tmp_path, security=SecurityConfig(bearer_token_env="TEST_CLOUD_TOKEN")
    )
    with TestClient(create_app(config, service=FakeService())) as client:
        for path in ("/v2/health", "/v2/openapi.json"):
            assert client.get(path).status_code == 401
            assert (
                client.get(path, headers={"Authorization": "Bearer wrong"}).status_code
                == 401
            )
            assert (
                client.get(path, headers={"Authorization": "Bearer secret"}).status_code
                == 200
            )
        readiness = client.get("/healthz")
        assert readiness.status_code == 200
        assert readiness.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "failed_message", ["http.response.start", "http.response.body"]
)
def test_asgi_send_failure_releases_unstarted_stream_or_cancels_running_worker(
    tmp_path, failed_message
):
    service = HeldProgressService()
    app = create_app(server_config(tmp_path), service=service)

    async def run():
        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == failed_message:
                raise OSError("client closed connection")

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        request = Request(scope, receive=receive)
        request.state.request_id = "send-failure"
        admission = await _admit(app, request)
        response = _AdmittedStreamingResponse(
            _stream_bounded(
                app,
                service,
                AnalyzeRequest.model_validate(request_payload()),
                request,
                admission,
            ),
            admission=admission,
        )
        try:
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
            if failed_message == "http.response.body":
                assert await asyncio.to_thread(service.cancelled.wait, 1)
                assert app.state.analysis_slots.locked()
            else:
                assert not app.state.analysis_slots.locked()
        finally:
            service.finish.set()
        await wait_until(lambda: app.state.analysis_requests == 0)
        assert not app.state.analysis_jobs
        # Exactly one slot is returned even when both cleanup paths run.
        await app.state.analysis_slots.acquire()
        assert app.state.analysis_slots.locked()
        app.state.analysis_slots.release()

    asyncio.run(run())


def test_shutdown_waits_for_cancelled_native_worker_before_closing_model(tmp_path):
    class DrainingService(HeldProgressService):
        closed = False

        def shutdown(self):
            assert self.finish.is_set()
            self.closed = True

    service = DrainingService()
    app = create_app(server_config(tmp_path), service=service)

    async def run():
        async def receive():
            await asyncio.Event().wait()

        context = app.router.lifespan_context(app)
        await context.__aenter__()
        request = Request({"type": "http"}, receive=receive)
        admission = await _admit(app, request)
        pending = asyncio.create_task(
            _run_bounded(
                app,
                service,
                AnalyzeRequest.model_validate(request_payload()),
                request,
                admission,
            )
        )
        shutdown = None
        try:
            await wait_until(lambda: bool(app.state.analysis_jobs))
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
            assert await asyncio.to_thread(service.cancelled.wait, 1)
            shutdown = asyncio.create_task(context.__aexit__(None, None, None))
            await wait_until(app.state.analysis_stopping.is_set)
            assert not shutdown.done()
            assert not service.closed
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                health = await client.get("/healthz")
                assert health.status_code == 503
                assert health.json() == {"status": "unavailable"}
                rejected = await client.post("/v2/move", json=request_payload())
                assert rejected.status_code == 503
                assert rejected.json()["error"]["code"] == "service_unavailable"
        finally:
            service.finish.set()
            if shutdown is not None:
                await asyncio.wait_for(shutdown, 1)
            else:
                await context.__aexit__(None, None, None)
        assert service.closed
        assert not app.state.analysis_jobs
        assert app.state.analysis_requests == 0

    asyncio.run(run())
