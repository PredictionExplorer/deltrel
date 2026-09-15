from __future__ import annotations

import email.utils
import http.client
import io
import json
import threading
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

import scripts.lambda_capacity_watch as watch


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


class Response(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, result=b'{"data": []}') -> None:
        self.result = result
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []

    def open(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.result, BaseException):
            raise self.result
        return Response(self.result)


class TimedOpener(Opener):
    def __init__(self, clock: Clock) -> None:
        super().__init__()
        self.clock = clock
        self.starts: list[float] = []

    def open(self, request, *, timeout):
        self.starts.append(self.clock())
        return super().open(request, timeout=timeout)


def _http_error(status: int, body: bytes, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    headers["Authorization"] = "Bearer response-header-secret"
    return urllib.error.HTTPError(
        "https://untrusted.invalid/?secret=response-url-secret",
        status,
        "response-reason-secret",
        headers,
        io.BytesIO(body),
    )


def test_requests_are_spaced_without_real_sleep_and_keep_official_origin():
    clock = Clock()
    opener = TimedOpener(clock)
    api = watch.LambdaAPI(
        "fake-test-credential", opener=opener, clock=clock, sleep=clock.sleep
    )
    api.request("GET", "/instances")
    clock.now += 0.25
    api.request("GET", "/instance-types")
    clock.now += 3
    api.request("GET", "/ssh-keys")

    assert all(b - a >= 2 for a, b in zip(opener.starts, opener.starts[1:]))
    assert sum(clock.sleeps) == 1.75
    assert [request.full_url for request in opener.requests] == [
        "https://cloud.lambda.ai/api/v1/instances",
        "https://cloud.lambda.ai/api/v1/instance-types",
        "https://cloud.lambda.ai/api/v1/ssh-keys",
    ]
    assert all(
        request.get_header("Authorization") == "Bearer fake-test-credential"
        for request in opener.requests
    )
    assert opener.timeouts == [30, 30, 30]


def test_explicit_launch_requests_keep_fifteen_second_gap_across_get_requests():
    clock = Clock()
    opener = TimedOpener(clock)
    api = watch.LambdaAPI("fake-key", opener=opener, clock=clock, sleep=clock.sleep)
    payload = {"name": "test-instance"}
    api.request("POST", "/instance-operations/launch", payload)
    api.request("GET", "/instances")
    api.request("POST", "/instance-operations/launch", payload)
    api.request("GET", "/instance-types")

    methods = [request.get_method() for request in opener.requests]
    assert methods == ["POST", "GET", "POST", "GET"]
    assert all(b - a >= 2 for a, b in zip(opener.starts, opener.starts[1:]))
    launch_starts = [
        start for method, start in zip(methods, opener.starts) if method == "POST"
    ]
    assert launch_starts[1] - launch_starts[0] >= 15
    # Inventory requests should retain their shorter pacing between launches.
    assert opener.starts[1] - opener.starts[0] == 2
    assert opener.starts[3] - opener.starts[2] == 2


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "https://untrusted.invalid/instances"),
        ("GET", "//untrusted.invalid/instances"),
        ("GET", "/instances?redirect=https://untrusted.invalid"),
        ("POST", "/instances"),
        ("DELETE", "/instances"),
    ],
)
def test_unsupported_operations_never_send_credentials(method, path):
    opener = Opener()
    with pytest.raises(watch.WatchError, match="Unsupported API operation"):
        watch.LambdaAPI("fake-test-credential", opener=opener).request(method, path)
    assert opener.requests == []


@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
def test_redirects_never_forward_authorization(monkeypatch, redirect_status):
    received: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append((self.path, self.headers.get("Authorization")))
            if self.path == "/api/v1/instances":
                self.send_response(redirect_status)
                self.send_header("Location", "/credential-sink")
            else:
                self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    # Only a fake key reaches this loopback test server. Using a real opener
    # exercises urllib's redirect handler rather than a mock of that handler.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(
            watch, "API_BASE", f"http://127.0.0.1:{server.server_port}/api/v1"
        )
        with pytest.raises(watch.APIError) as error:
            watch.LambdaAPI("loopback-fake-key").request("GET", "/instances")
        assert error.value.status == redirect_status
        assert received == [("/api/v1/instances", "Bearer loopback-fake-key")]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "body",
    [
        b"{broken",
        b"\xff",
        b"[]",
        b"null",
        b"1",
        b'"value"',
        b" " * (4 * 1024 * 1024 + 1),
    ],
    ids=[
        "malformed",
        "invalid-encoding",
        "array",
        "null",
        "number",
        "string",
        "oversize",
    ],
)
def test_unusable_success_body_is_ambiguous_and_is_not_retried(body):
    opener = Opener(body)
    with pytest.raises(watch.APIError) as error:
        watch.LambdaAPI("fake-key", opener=opener).request(
            "POST", "/instance-operations/launch", {"name": "one-instance"}
        )
    assert error.value.status is None
    assert len(opener.requests) == 1
    assert opener.requests[0].get_method() == "POST"


@pytest.mark.parametrize("retry_after", ["123", "http-date"])
def test_429_preserves_retry_after_and_closes_error_body(monkeypatch, retry_after):
    now = 1_800_000_000
    monkeypatch.setattr(watch.time, "time", lambda: now)
    header = (
        email.utils.formatdate(now + 123, usegmt=True)
        if retry_after == "http-date"
        else retry_after
    )
    http_error = _http_error(429, b'{"error":{"code":"too-many-requests"}}', header)
    body = http_error.fp
    opener = Opener(http_error)
    with pytest.raises(watch.APIError) as error:
        watch.LambdaAPI("fake-key", opener=opener).request("GET", "/instances")
    assert error.value.status == 429
    assert error.value.retry_after == 123
    assert body.closed
    assert len(opener.requests) == 1


@pytest.mark.parametrize("header", [None, "garbage", "-5", "nan", "inf"])
def test_invalid_retry_after_does_not_create_an_unbounded_wait(header):
    assert watch.retry_after_seconds(header, 1_800_000_000) == 0


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":{"code":"response-body-secret","message":"private"}}',
        b'{"error": "response-body-secret"}',
        b"response-body-secret",
        b"[]",
    ],
)
def test_http_errors_never_expose_response_secrets(body):
    with pytest.raises(watch.APIError) as error:
        watch.LambdaAPI(
            "request-key-secret", opener=Opener(_http_error(503, body))
        ).request("GET", "/instances")
    assert error.value.status == 503
    assert str(error.value) == "Lambda API HTTP 503"
    assert "secret" not in repr(error.value)
    assert error.value.__suppress_context__ is True


def test_incomplete_http_error_body_is_closed_and_sanitized():
    class IncompleteBody(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b"private partial error body", 100)

    body = IncompleteBody()
    headers = Message()
    headers["Retry-After"] = "120"
    http_error = urllib.error.HTTPError(
        "https://cloud.lambda.ai/api/v1/instance-operations/launch",
        503,
        "Service Unavailable",
        headers,
        body,
    )
    opener = Opener(http_error)
    with pytest.raises(watch.APIError) as error:
        watch.LambdaAPI("fake-key", opener=opener).request(
            "POST", "/instance-operations/launch", {"name": "one-instance"}
        )
    assert error.value.status == 503
    assert error.value.retry_after == 120
    assert "private" not in str(error.value)
    assert body.closed
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("private timeout detail"),
        urllib.error.URLError("private transport detail"),
        http.client.IncompleteRead(b"partial private response", 100),
        http.client.RemoteDisconnected("private disconnect detail"),
    ],
)
def test_transport_failure_never_retries_a_launch(failure):
    opener = Opener(failure)
    with pytest.raises(watch.APIError) as error:
        watch.LambdaAPI("fake-key", opener=opener).request(
            "POST", "/instance-operations/launch", {"name": "one-instance"}
        )
    assert error.value.status is None
    assert "private" not in str(error.value)
    assert len(opener.requests) == 1


def test_missing_api_key_fails_before_any_request(monkeypatch, caplog):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)

    def unexpected_opener(*args, **kwargs):
        pytest.fail("Missing credentials must fail before initializing HTTP transport")

    monkeypatch.setattr(watch.urllib.request, "build_opener", unexpected_opener)
    assert watch.main(["--list-options"]) == 2
    assert "LAMBDA_API_KEY" in caplog.text


@pytest.mark.parametrize("option", ["--api-base", "--endpoint", "--api-key"])
def test_cli_cannot_override_api_origin_or_supply_key_in_arguments(option):
    with pytest.raises(SystemExit) as error:
        watch.main([option, "unsafe-value"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["--list-options", "--dry-run"])
@pytest.mark.parametrize("interactive", [False, True])
def test_read_only_cli_modes_never_launch(
    monkeypatch, tmp_path, capsys, mode, interactive
):
    class ReadOnlyOpener(Opener):
        def open(self, request, *, timeout):
            assert request.get_method() == "GET", "Read-only mode attempted a launch"
            assert request.data is None
            path = urlsplit(request.full_url).path
            if path == "/api/v1/instance-types":
                self.result = json.dumps(
                    {
                        "data": {
                            watch.TARGET_TYPE: {
                                "instance_type": {
                                    "name": watch.TARGET_TYPE,
                                    "price_cents_per_hour": 5352,
                                    "specs": {"gpus": 8},
                                },
                                "regions_with_capacity_available": [
                                    {"name": "us-east-3"},
                                    {"name": "us-west-1"},
                                ],
                            }
                        }
                    }
                ).encode()
            elif path == "/api/v1/ssh-keys":
                self.result = b'{"data":[{"name":"training-key"}]}'
            elif path == "/api/v1/instances":
                self.result = b'{"data":[]}'
            else:
                pytest.fail(f"Unexpected API operation: {path}")
            return super().open(request, timeout=timeout)

    clock, opener = Clock(), ReadOnlyOpener()
    api_type = watch.LambdaAPI
    if interactive:
        monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
        monkeypatch.setattr(watch.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(watch.getpass, "getpass", lambda prompt: "fake-cli-key")
    else:
        monkeypatch.setenv("LAMBDA_API_KEY", "fake-cli-key")
    monkeypatch.setattr(
        watch,
        "LambdaAPI",
        lambda key: api_type(key, opener=opener, clock=clock, sleep=clock.sleep),
    )
    assert (
        watch.main(
            [
                mode,
                "--ssh-key",
                "training-key",
                "--state-file",
                str(tmp_path / "state.json"),
            ]
        )
        == 0
    )
    assert opener.requests
    assert all(request.get_method() == "GET" for request in opener.requests)
    assert all(
        request.get_header("Authorization") == "Bearer fake-cli-key"
        for request in opener.requests
    )
    output = capsys.readouterr().out
    assert "fake-cli-key" not in output
    if (tmp_path / "state.json").exists():
        assert "fake-cli-key" not in (tmp_path / "state.json").read_text()
    if mode == "--list-options":
        options = json.loads(output)
        assert options["available_regions"] == ["us-east-3", "us-west-1"]
        assert options["ssh_key_names"] == ["training-key"]
        assert options["hourly_usd"] == 53.52
