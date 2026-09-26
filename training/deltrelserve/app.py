"""FastAPI application with bounded, authenticated native model analysis."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from deltreltrain.contracts import (
    MAX_HANDICAP,
    MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    MODES,
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    RULES_VERSION,
)
from deltreltrain.model import MODEL_SCHEMA_VERSION

from .config import SERVER_CONFIG_SCHEMA_VERSION, ServerConfig, load_server_config
from .runtime import AnalysisError, NativeAnalysisService
from .schemas import API_SCHEMA_VERSION, AnalyzeRequest, AnalyzeResponse

SERVICE_VERSION = "2.0.0"
_LOGGER = logging.getLogger("deltrelserve")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class AnalysisServiceProtocol(Protocol):
    def startup(self) -> None: ...

    def health(self) -> dict[str, object]: ...

    def analyze(
        self,
        request: AnalyzeRequest,
        cancellation: threading.Event,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, object]: ...


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, *, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        request_id = _request_id_from_headers(headers)
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0:
                await _send_error(
                    send,
                    status_code=400,
                    code="invalid_content_length",
                    message="Content-Length must be a non-negative integer",
                    request_id=request_id,
                )
                return
            if declared > self.maximum_bytes:
                await _send_error(
                    send,
                    status_code=413,
                    code="request_too_large",
                    message="request body exceeds the configured size limit",
                    request_id=request_id,
                )
                return
        received = 0
        buffered: list[dict[str, Any]] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.maximum_bytes:
                    await _send_error(
                        send,
                        status_code=413,
                        code="request_too_large",
                        message="request body exceeds the configured size limit",
                        request_id=request_id,
                    )
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        async def replay_receive() -> dict[str, Any]:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


async def _send_error(
    send: Any,
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
) -> None:
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": None,
                "request_id": request_id,
            }
        },
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _request_id_from_headers(headers: dict[bytes, bytes]) -> str:
    raw = headers.get(b"x-request-id")
    if raw is not None and 0 < len(raw) <= 128:
        try:
            supplied = raw.decode("ascii")
        except UnicodeDecodeError:
            supplied = ""
        else:
            if _REQUEST_ID.fullmatch(supplied):
                return supplied
    return uuid.uuid4().hex


def create_app(
    config: ServerConfig | str | Path,
    *,
    service: AnalysisServiceProtocol | None = None,
) -> FastAPI:
    settings = (
        config if isinstance(config, ServerConfig) else load_server_config(config)
    )
    analysis_service = service or NativeAnalysisService(settings)
    bearer_token = settings.security.bearer_token()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup = getattr(analysis_service, "startup", None)
        if callable(startup):
            await asyncio.to_thread(startup)
        try:
            yield
        finally:
            app.state.analysis_stopping.set()
            # A cancelled HTTP response may still own native inference. Drain
            # those workers before closing their shared model and batch broker.
            jobs = dict(app.state.analysis_jobs)
            for cancellation in jobs.values():
                cancellation.set()
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)
            shutdown = getattr(analysis_service, "shutdown", None)
            if callable(shutdown):
                await asyncio.to_thread(shutdown)

    app = FastAPI(
        title="Double Deltrel model service",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v2/openapi.json",
    )
    app.state.config = settings
    app.state.analysis_service = analysis_service
    app.state.analysis_slots = asyncio.Semaphore(settings.limits.max_concurrency)
    app.state.analysis_requests = 0
    app.state.analysis_waiting = 0
    app.state.analysis_jobs = {}
    app.state.analysis_stopping = asyncio.Event()
    app.add_middleware(
        RequestSizeLimitMiddleware,
        maximum_bytes=settings.limits.max_request_bytes,
    )
    if settings.security.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.security.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )

    def error_response(
        request: Request,
        *,
        status_code: int,
        code: str,
        message: str,
        details: object | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            headers={"Retry-After": "1"} if status_code in (503, 504) else None,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Any]],
    ) -> Any:
        supplied = request.headers.get("x-request-id")
        request_id = (
            supplied
            if supplied is not None and _REQUEST_ID.fullmatch(supplied)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        if bearer_token is not None and (
            request.method == "POST"
            and request.url.path in ("/v2/analyze", "/v2/move")
            or request.method in ("GET", "HEAD")
            and request.url.path in ("/v2/health", "/v2/openapi.json")
        ):
            authorization = request.headers.get("authorization")
            candidate = (
                authorization.removeprefix("Bearer ")
                if authorization is not None and authorization.startswith("Bearer ")
                else ""
            )
            if not candidate or not hmac.compare_digest(candidate, bearer_token):
                response = error_response(
                    request,
                    status_code=401,
                    code="unauthorized",
                    message="a valid bearer token is required",
                )
                response.headers["X-Request-ID"] = request_id
                response.headers["Cache-Control"] = "no-store"
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AnalysisError)
    async def analysis_error_handler(
        request: Request, error: AnalysisError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "location": [str(part) for part in item["loc"]],
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return error_response(
            request,
            status_code=422,
            code="invalid_request",
            message=f"request does not match the v{API_SCHEMA_VERSION} analysis schema",
            details=details,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, error: Exception
    ) -> JSONResponse:
        _LOGGER.error(
            "unhandled analysis service error",
            exc_info=(type(error), error, error.__traceback__),
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="the analysis service encountered an internal error",
        )

    @app.get("/healthz")
    async def readiness() -> JSONResponse:
        ready = bool(analysis_service.health().get("ready", True)) and not (
            app.state.analysis_stopping.is_set()
        )
        return JSONResponse(
            {"status": "ok" if ready else "unavailable"},
            status_code=200 if ready else 503,
            headers=None if ready else {"Retry-After": "1"},
        )

    @app.get("/v2/health")
    async def health() -> JSONResponse:
        model_health = analysis_service.health()
        stopping = app.state.analysis_stopping.is_set()
        ready = bool(model_health.get("ready", True)) and not stopping
        degraded = bool(model_health.get("last_reload_error"))
        payload = {
            "status": (
                "stopping"
                if stopping
                else "degraded"
                if degraded
                else ("ok" if ready else "starting")
            ),
            "service_version": SERVICE_VERSION,
            "api_schema_version": API_SCHEMA_VERSION,
            "network_output_schema_version": 1,
            "server_config_schema_version": SERVER_CONFIG_SCHEMA_VERSION,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "device": settings.device,
            "model": model_health,
            "capacity": {
                "active_requests": app.state.analysis_requests
                - app.state.analysis_waiting,
                "queued_requests": app.state.analysis_waiting,
                "max_concurrency": settings.limits.max_concurrency,
                "max_queued_requests": settings.limits.max_queued_requests,
            },
            "search": {
                "defaults": {
                    "simulations": settings.search.default_simulations,
                    "max_considered": settings.search.default_max_considered,
                },
                "maximums": {
                    "simulations": settings.search.maximum_simulations,
                    "max_considered": settings.search.maximum_max_considered,
                },
                "presets": settings.search.named_presets(),
            },
            "rules": {
                "schema_id": RULES_SCHEMA_ID,
                "version": RULES_VERSION,
                "hash": RULES_HASH_WIRE,
            },
            "features": {
                "schema_id": EXTERNAL_FEATURE_SCHEMA_ID,
                "version": FEATURE_SCHEMA_VERSION,
                "hash": f"{FEATURE_SCHEMA_HASH:016x}",
            },
            "actions": {
                "schema_id": ACTION_LAYOUT_SCHEMA_ID,
                "types": ["place", "swap"],
            },
            "variants": {
                "modes": list(MODES),
                "handicap": {"min": 1, "max": MAX_HANDICAP},
                "pie": True,
                "history": "optional",
                "playout_doubling_advantage": {
                    "min": -MAX_PLAYOUT_DOUBLING_ADVANTAGE,
                    "max": MAX_PLAYOUT_DOUBLING_ADVANTAGE,
                },
                "swap_dead_zone": settings.search.swap_dead_zone,
            },
            "outcomes": {
                "classes": ["loss", "win"],
                "value": "P(win)-P(loss)",
            },
        }
        return JSONResponse(
            payload,
            status_code=200 if ready else 503,
            headers=None if ready else {"Retry-After": "1"},
        )

    async def analyze(
        payload: AnalyzeRequest,
        request: Request,
    ) -> AnalyzeResponse | StreamingResponse:
        if payload.search.simulations > settings.search.maximum_simulations:
            raise AnalysisError(
                "search_budget_exceeded",
                "simulations exceed the configured maximum",
                status_code=422,
                details={"maximum_simulations": settings.search.maximum_simulations},
            )
        if payload.search.max_considered > settings.search.maximum_max_considered:
            raise AnalysisError(
                "search_budget_exceeded",
                "max_considered exceeds the configured maximum",
                status_code=422,
                details={
                    "maximum_max_considered": (settings.search.maximum_max_considered)
                },
            )
        admission = await _admit(app, request)
        if any(
            value.split(";", 1)[0].strip().lower() == "application/x-ndjson"
            for value in request.headers.get("accept", "").split(",")
        ):
            return _AdmittedStreamingResponse(
                _stream_bounded(app, analysis_service, payload, request, admission),
                admission=admission,
                media_type="application/x-ndjson",
                headers={"X-Accel-Buffering": "no"},
            )
        result, queue_ms = await _run_bounded(
            app,
            analysis_service,
            payload,
            request,
            admission,
        )
        return _analysis_response(result, queue_ms, request.state.request_id)

    app.post(
        "/v2/analyze",
        response_model=AnalyzeResponse,
        response_model_exclude_unset=True,
        response_model_exclude_none=False,
    )(analyze)
    app.post(
        "/v2/move",
        response_model=AnalyzeResponse,
        response_model_exclude_unset=True,
        response_model_exclude_none=False,
        include_in_schema=False,
    )(analyze)
    return app


def _analysis_response(
    result: dict[str, object], queue_ms: float, request_id: str
) -> AnalyzeResponse:
    result = dict(result)
    result["request_id"] = request_id
    raw_timing = result.get("timing_ms")
    if not isinstance(raw_timing, dict):
        raise AnalysisError(
            "native_search_error",
            "analysis service returned malformed timing metrics",
        )
    timing: dict[str, float] = {}
    for name, value in raw_timing.items():
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            raise AnalysisError(
                "native_search_error",
                "analysis service returned malformed timing metrics",
            )
        timing[name] = float(value)
    if "total" not in timing:
        raise AnalysisError(
            "native_search_error", "analysis service omitted total timing"
        )
    timing["queue"] = queue_ms
    timing["total"] += queue_ms
    result["timing_ms"] = timing
    return AnalyzeResponse.model_validate(result)


class _LatestProgress:
    """One overwriteable update; slow readers never queue simulation events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[int, int] | None = None

    def publish(self, completed: int, total: int) -> None:
        with self._lock:
            self._latest = (completed, total)

    def take(self) -> tuple[int, int] | None:
        with self._lock:
            latest, self._latest = self._latest, None
            return latest


class _Admission:
    """One request owns its slot until its native worker actually finishes."""

    def __init__(self, app: FastAPI, queued_at: float) -> None:
        self.app = app
        self.queue_ms = (time.perf_counter() - queued_at) * 1_000.0
        self.deadline = queued_at + app.state.config.limits.request_timeout_seconds
        self.started = False
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.app.state.analysis_requests -= 1
            self.app.state.analysis_slots.release()


class _AdmittedStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content: AsyncGenerator[bytes, None],
        *,
        admission: _Admission,
        **kwargs: Any,
    ) -> None:
        super().__init__(content, **kwargs)
        self.content = content
        self.admission = admission

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.content.aclose()
            finally:
                # A disconnect/send failure can prevent the async generator from
                # ever starting. Once started, _run_bounded owns deferred release.
                if not self.admission.started:
                    self.admission.release()


async def _admit(app: FastAPI, request: Request) -> _Admission:
    """Bound waiting requests before committing either JSON or stream headers."""

    settings: ServerConfig = app.state.config
    semaphore: asyncio.Semaphore = app.state.analysis_slots
    stopping: asyncio.Event = app.state.analysis_stopping
    if stopping.is_set():
        raise AnalysisError(
            "service_unavailable", "analysis service is shutting down", status_code=503
        )
    if app.state.analysis_requests >= (
        settings.limits.max_concurrency + settings.limits.max_queued_requests
    ):
        raise AnalysisError(
            "service_busy", "analysis waiting queue is full", status_code=503
        )
    # These counters are confined to the event loop; no await can interleave
    # checking the capacity with reserving a place in its bounded queue.
    app.state.analysis_requests += 1
    app.state.analysis_waiting += 1
    queued = time.perf_counter()
    acquire = asyncio.create_task(semaphore.acquire())
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    shutdown = asyncio.create_task(stopping.wait())
    tasks = {acquire, disconnect, shutdown}
    handed_off = False
    try:
        try:
            done, _ = await asyncio.wait(
                tasks,
                timeout=min(
                    settings.limits.queue_timeout_seconds,
                    settings.limits.request_timeout_seconds,
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown in done:
                raise AnalysisError(
                    "service_unavailable",
                    "analysis service is shutting down",
                    status_code=503,
                )
            if disconnect in done:
                raise AnalysisError(
                    "client_disconnected",
                    "client disconnected while queued",
                    status_code=499,
                )
            if acquire not in done:
                raise AnalysisError(
                    "service_busy",
                    "analysis concurrency limit is saturated",
                    status_code=503,
                )
            acquire.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        handed_off = True
        return _Admission(app, queued)
    finally:
        app.state.analysis_waiting -= 1
        if not handed_off:
            app.state.analysis_requests -= 1
            # Acquisition and disconnection/cancellation can race. Return the
            # slot even when both completed in the same event-loop iteration.
            if (
                acquire.done()
                and not acquire.cancelled()
                and acquire.exception() is None
            ):
                semaphore.release()


def _stream_event(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()


async def _stream_bounded(
    app: FastAPI,
    service: AnalysisServiceProtocol,
    payload: AnalyzeRequest,
    request: Request,
    admission: _Admission,
) -> AsyncGenerator[bytes, None]:
    progress = _LatestProgress()
    task = asyncio.create_task(
        _run_bounded(app, service, payload, request, admission, progress.publish)
    )
    try:
        while not task.done():
            # At most 20 updates/second, independent of simulation count. The
            # worker and its absolute timeout keep running under backpressure.
            await asyncio.wait({task}, timeout=0.05)
            update = progress.take()
            if update is not None:
                completed, total = update
                yield _stream_event(
                    {
                        "type": "progress",
                        "completed_simulations": completed,
                        "total_simulations": total,
                    }
                )
        update = progress.take()
        if update is not None:
            completed, total = update
            yield _stream_event(
                {
                    "type": "progress",
                    "completed_simulations": completed,
                    "total_simulations": total,
                }
            )
        result, queue_ms = task.result()
        validated = _analysis_response(result, queue_ms, request.state.request_id)
        yield _stream_event(
            {
                "type": "result",
                "result": validated.model_dump(mode="json", exclude_unset=True),
            }
        )
    except AnalysisError as error:
        yield _stream_event(
            {
                "type": "error",
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "request_id": request.state.request_id,
                },
            }
        )
    except Exception:
        _LOGGER.exception("unhandled streaming analysis service error")
        yield _stream_event(
            {
                "type": "error",
                "error": {
                    "code": "internal_error",
                    "message": "the analysis service encountered an internal error",
                    "details": None,
                    "request_id": request.state.request_id,
                },
            }
        )
    finally:
        if not task.done():
            task.cancel()
        # Always observe the task and let it register deferred slot release
        # before returning, including client disconnects during a yielded chunk.
        with suppress(asyncio.CancelledError, Exception):
            await task


async def _run_bounded(
    app: FastAPI,
    service: AnalysisServiceProtocol,
    payload: AnalyzeRequest,
    request: Request,
    admission: _Admission,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, object], float]:
    queue_ms = admission.queue_ms
    admission.started = True
    if app.state.analysis_stopping.is_set():
        admission.release()
        raise AnalysisError(
            "service_unavailable", "analysis service is shutting down", status_code=503
        )
    if time.perf_counter() >= admission.deadline:
        admission.release()
        raise AnalysisError(
            "analysis_timeout",
            "analysis exceeded the configured request timeout",
            status_code=504,
        )
    cancellation = threading.Event()
    loop = asyncio.get_running_loop()
    try:
        future = (
            loop.run_in_executor(None, service.analyze, payload, cancellation)
            if progress is None
            else loop.run_in_executor(
                None, service.analyze, payload, cancellation, progress
            )
        )
    except BaseException:
        admission.release()
        raise
    app.state.analysis_jobs[future] = cancellation

    def release(completed: asyncio.Future[Any]) -> None:
        if not completed.cancelled():
            completed.exception()
        app.state.analysis_jobs.pop(completed, None)
        admission.release()

    # Release is tied exclusively to native completion, including timeout,
    # disconnect, exception, and task cancellation paths.
    future.add_done_callback(release)
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        remaining = max(0.0, admission.deadline - time.perf_counter())
        done, _ = await asyncio.wait(
            {future, disconnect},
            timeout=remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if future in done:
            return future.result(), queue_ms
        cancellation.set()
        if disconnect in done and disconnect.result():
            raise AnalysisError(
                "client_disconnected",
                "client disconnected during analysis",
                status_code=499,
            )
        raise AnalysisError(
            "analysis_timeout",
            "analysis exceeded the configured request timeout",
            status_code=504,
        )
    except asyncio.CancelledError:
        cancellation.set()
        raise
    finally:
        disconnect.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect


async def _wait_for_disconnect(request: Request) -> bool:
    while True:
        # FastAPI has already consumed and validated the request body. Waiting
        # directly on receive avoids polling and remains normally cancellable;
        # is_disconnected uses an AnyIO cancel scope that can swallow a task's
        # concurrent cancellation while admission is cleaning up its watchers.
        if (await request.receive())["type"] == "http.disconnect":
            return True
