from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import scripts.lambda_capacity_watch as watch


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay

    def advance(self, delay: float = 3600) -> None:
        self.now += delay


def _instance(name: str, *, region_name: str = "us-east-3", **overrides) -> dict:
    return {
        "id": "reserved-instance-1",
        "name": name,
        "status": "booting",
        "region": {"name": region_name},
        "instance_type": {"name": watch.TARGET_TYPE},
        **overrides,
    }


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.instances: list[dict] = []
        self.regions = ["us-east-3"]
        self.price = 5352
        self.launch: dict | Exception | Callable = {
            "data": {"instance_ids": ["reserved-instance-1"]}
        }
        self.failures: dict[tuple[str, str], Exception] = {}

    def request(self, method: str, path: str, payload=None) -> dict:
        # Accept an API-relative path with or without the leading version.
        path = path.removeprefix("/api/v1")
        self.calls.append((method, path, payload))
        if (method, path) in self.failures:
            raise self.failures[method, path]
        if (method, path) == ("GET", "/instances"):
            return {"data": self.instances}
        if (method, path) == ("GET", "/ssh-keys"):
            return {"data": [{"name": "training-key", "id": "ssh-key-1"}]}
        if (method, path) == ("GET", "/instance-types"):
            return {
                "data": {
                    watch.TARGET_TYPE: {
                        "instance_type": {
                            "name": watch.TARGET_TYPE,
                            "price_cents_per_hour": self.price,
                            "specs": {"gpus": 8},
                        },
                        "regions_with_capacity_available": [
                            {"name": name} for name in self.regions
                        ],
                    }
                }
            }
        if (method, path) == ("POST", "/instance-operations/launch"):
            if isinstance(self.launch, Exception):
                raise self.launch
            if callable(self.launch):
                return self.launch(payload)
            return self.launch
        raise AssertionError(f"Unexpected API call: {method} {path}")

    @property
    def launches(self) -> list[dict]:
        return [
            payload
            for method, path, payload in self.calls
            if method == "POST" and path == "/instance-operations/launch"
        ]


def _watcher(tmp_path: Path, api: FakeAPI, clock: Clock, **config):
    return watch.Watcher(
        watch.Config(ssh_key="training-key", **config),
        api,
        tmp_path / "state.json",
        clock=clock,
        sleep=clock.sleep,
        jitter=lambda *args: 0.0,
    )


def test_launch_is_single_instance_and_intent_is_durable_before_request(tmp_path):
    api, clock = FakeAPI(), Clock()

    def launch(payload):
        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["phase"] == "pending"
        assert payload["name"] in json.dumps(saved)
        assert payload["region_name"] in json.dumps(saved)
        assert payload["instance_type_name"] == watch.TARGET_TYPE
        assert payload["ssh_key_names"] == ["training-key"]
        assert "quantity" not in payload
        assert "file_system_names" not in payload
        return {"data": {"instance_ids": ["reserved-instance-1"]}}

    api.launch = launch
    watcher = _watcher(tmp_path, api, clock)
    assert watcher.check() is True
    assert len(api.launches) == 1

    # A successful launch must survive a process restart even if listing has
    # not caught up with the accepted launch yet.
    api.instances = []
    restarted = _watcher(tmp_path, api, clock)
    assert restarted.check() is True
    assert len(api.launches) == 1


def test_any_region_can_be_selected(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.regions = ["europe-central-1"]
    assert _watcher(tmp_path, api, clock).check() is True
    assert api.launches[0]["region_name"] == "europe-central-1"


def test_region_allowlist_is_respected(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.regions = ["europe-central-1", "us-east-3"]
    assert _watcher(tmp_path, api, clock, regions=("us-east-3",)).check() is True
    assert api.launches[0]["region_name"] == "us-east-3"


def test_unavailable_capacity_makes_no_launch_request(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.regions = []
    assert _watcher(tmp_path, api, clock).check() is False
    assert api.launches == []


def test_no_allowed_region_makes_no_launch_request(tmp_path):
    api, clock = FakeAPI(), Clock()
    assert (
        _watcher(tmp_path, api, clock, regions=("europe-central-1",)).check() is False
    )
    assert api.launches == []


def test_price_cap_uses_whole_instance_hourly_price(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.price = 6001
    assert _watcher(tmp_path, api, clock, max_hourly_cents=6000).check() is False
    assert api.launches == []


def test_price_equal_to_cap_is_allowed(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.price = 6000
    assert _watcher(tmp_path, api, clock, max_hourly_cents=6000).check() is True
    assert len(api.launches) == 1


def test_dry_run_never_launches_or_leaves_pending_intent(tmp_path):
    api, clock = FakeAPI(), Clock()
    watcher = _watcher(tmp_path, api, clock, dry_run=True)
    assert watcher.check() is True
    assert api.launches == []
    assert watcher.state["phase"] != "pending"


@pytest.mark.parametrize("interval", [0, -1, 119.99, float("nan"), float("inf")])
def test_invalid_or_aggressive_poll_interval_is_rejected(tmp_path, interval):
    api, clock = FakeAPI(), Clock()
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock, interval_seconds=interval)
    assert api.calls == []


def test_polling_does_not_repeat_requests_before_due_time(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.regions = []
    watcher = _watcher(tmp_path, api, clock)
    assert watcher.check() is False
    count = len(api.calls)
    assert watcher.state["next_check_at"] >= clock() + 120
    assert watcher.check() is False
    assert _watcher(tmp_path, api, clock).check() is False
    assert len(api.calls) == count


@pytest.mark.parametrize("status", [429, 500, 503, None])
def test_transient_get_failure_schedules_persistent_backoff(tmp_path, status):
    api, clock = FakeAPI(), Clock()
    api.failures["GET", "/instance-types"] = watch.APIError(status, retry_after=900)
    watcher = _watcher(tmp_path, api, clock)
    assert watcher.check() is False
    assert watcher.state["next_check_at"] >= clock() + 900
    count = len(api.calls)
    clock.advance(899)
    assert _watcher(tmp_path, api, clock).check() is False
    assert len(api.calls) == count
    assert api.launches == []


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors_stop_instead_of_retrying(tmp_path, status):
    api, clock = FakeAPI(), Clock()
    api.failures["GET", "/ssh-keys"] = watch.APIError(status)
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.launches == []


def test_quota_failure_stops_after_one_launch_attempt(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(400, "global/quota-exceeded")
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert len(api.launches) == 1


def test_explicit_insufficient_capacity_allows_later_retry(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(400, watch.CAPACITY_ERROR)
    watcher = _watcher(tmp_path, api, clock)
    assert watcher.check() is False
    assert len(api.launches) == 1
    clock.advance()
    api.launch = {"data": {"instance_ids": ["reserved-instance-1"]}}
    assert _watcher(tmp_path, api, clock).check() is True
    assert len(api.launches) == 2


@pytest.mark.parametrize(
    "launch_response",
    [
        pytest.param(lambda: watch.APIError(None), id="network-interruption"),
        pytest.param(lambda: watch.APIError(500), id="server-error"),
        pytest.param(lambda: watch.APIError(503), id="service-unavailable"),
        pytest.param(lambda: watch.APIError(429), id="ambiguous-rate-limit"),
        pytest.param(lambda: {}, id="missing-data"),
        pytest.param(lambda: {"data": {}}, id="missing-instance-ids"),
        pytest.param(lambda: {"data": {"instance_ids": []}}, id="no-instance-id"),
        pytest.param(
            lambda: {"data": {"instance_ids": ["one", "two"]}},
            id="unexpected-instance-count",
        ),
    ],
)
def test_ambiguous_launch_never_retries_even_after_restart(tmp_path, launch_response):
    api, clock = FakeAPI(), Clock()
    api.launch = launch_response()
    watcher = _watcher(tmp_path, api, clock)
    assert watcher.check() is False
    assert len(api.launches) == 1
    clock.advance()
    for _ in range(3):
        assert _watcher(tmp_path, api, clock).check() is False
        clock.advance()
    assert len(api.launches) == 1


def test_ambiguous_launch_reconciles_exact_existing_instance(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(None)
    assert _watcher(tmp_path, api, clock).check() is False
    name = api.launches[0]["name"]
    api.instances = [_instance(name)]
    clock.advance()
    assert _watcher(tmp_path, api, clock).check() is True
    assert len(api.launches) == 1


def test_interruption_during_post_preserves_pending_intent(tmp_path):
    api, clock = FakeAPI(), Clock()

    def interrupt(_payload):
        raise KeyboardInterrupt

    api.launch = interrupt
    with pytest.raises(KeyboardInterrupt):
        _watcher(tmp_path, api, clock).check()
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["phase"] == "pending"
    clock.advance()
    api.launch = {"data": {"instance_ids": ["must-not-be-used"]}}
    assert _watcher(tmp_path, api, clock).check() is False
    assert len(api.launches) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"region": {"name": "another-region"}},
        {"instance_type": {"name": "gpu_8x_h100_sxm5"}},
        {"status": "terminated"},
    ],
)
def test_pending_name_collision_is_not_accepted_as_success(tmp_path, overrides):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(None)
    assert _watcher(tmp_path, api, clock).check() is False
    api.instances = [_instance(api.launches[0]["name"], **overrides)]
    clock.advance()
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert len(api.launches) == 1


def test_duplicate_named_instances_stop_without_another_launch(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(None)
    assert _watcher(tmp_path, api, clock).check() is False
    name = api.launches[0]["name"]
    api.instances = [_instance(name), _instance(name, id="second-instance")]
    clock.advance()
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert len(api.launches) == 1


@pytest.mark.parametrize("contents", ["not json", "[]", "null", "{}"])
def test_malformed_existing_state_fails_before_any_api_call(tmp_path, contents):
    api, clock = FakeAPI(), Clock()
    (tmp_path / "state.json").write_text(contents)
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.calls == []


def test_state_symlink_is_rejected_without_touching_destination(tmp_path):
    target = tmp_path / "other.json"
    target.write_text("original")
    (tmp_path / "state.json").symlink_to(target)
    api, clock = FakeAPI(), Clock()
    with pytest.raises((watch.WatchError, OSError)):
        _watcher(tmp_path, api, clock).check()
    assert target.read_text() == "original"
    assert api.calls == []


def test_state_lock_excludes_second_watcher_and_releases_cleanly(tmp_path):
    path = tmp_path / "state.json"
    with watch.state_lock(path):
        with pytest.raises(watch.WatchError, match="Another watcher"):
            with watch.state_lock(path):
                pytest.fail("A second watcher acquired the same reservation lock")
    with watch.state_lock(path):
        pass


def test_lock_symlink_is_rejected_without_modifying_destination(tmp_path):
    path = tmp_path / "state.json"
    target = tmp_path / "other.lock"
    target.write_text("original")
    path.with_name(path.name + ".lock").symlink_to(target)
    with pytest.raises((watch.WatchError, OSError)):
        with watch.state_lock(path):
            pytest.fail("A symbolic link was accepted as the reservation lock")
    assert target.read_text() == "original"


def test_intent_write_failure_prevents_launch(tmp_path, monkeypatch):
    api, clock = FakeAPI(), Clock()
    original = watch.write_state

    def fail_pending(path, value):
        if value["phase"] == "pending":
            raise OSError("disk is full")
        original(path, value)

    monkeypatch.setattr(watch, "write_state", fail_pending)
    with pytest.raises(OSError, match="disk is full"):
        _watcher(tmp_path, api, clock).check()
    assert api.launches == []


def test_save_failure_after_success_keeps_restart_read_only(tmp_path, monkeypatch):
    api, clock = FakeAPI(), Clock()
    original = watch.write_state

    def fail_reserved(path, value):
        if value["phase"] == "reserved":
            raise OSError("disk is full")
        original(path, value)

    monkeypatch.setattr(watch, "write_state", fail_reserved)
    with pytest.raises(OSError, match="disk is full"):
        _watcher(tmp_path, api, clock).check()
    assert len(api.launches) == 1
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "pending"
    monkeypatch.setattr(watch, "write_state", original)
    clock.advance()
    assert _watcher(tmp_path, api, clock).check() is False
    assert len(api.launches) == 1


def test_compatible_existing_instance_avoids_launch(tmp_path):
    api, clock = FakeAPI(), Clock()
    api.instances = [_instance("edgeconnect-b200-reserve")]
    assert _watcher(tmp_path, api, clock).check() is True
    assert api.launches == []


def test_instance_appearing_after_capacity_lookup_avoids_launch(tmp_path, monkeypatch):
    api, clock = FakeAPI(), Clock()
    original = api.request
    inventories = 0

    def appear_before_launch(method, path, payload=None):
        nonlocal inventories
        if method == "GET" and path == "/instances":
            inventories += 1
            if inventories == 2:
                api.instances = [_instance("edgeconnect-b200-reserve")]
        return original(method, path, payload)

    monkeypatch.setattr(api, "request", appear_before_launch)
    assert _watcher(tmp_path, api, clock).check() is True
    assert inventories == 2
    assert api.launches == []


@pytest.mark.parametrize(
    "changed",
    [
        {"launch": {"region_name": "us-east-3"}},
        {"instance_ids": ["reserved-instance-1"]},
    ],
)
def test_watching_state_with_evidence_of_a_launch_fails_closed(tmp_path, changed):
    api, clock = FakeAPI(), Clock()
    watcher = _watcher(tmp_path, api, clock)
    watcher.state.update(changed)
    (tmp_path / "state.json").write_text(json.dumps(watcher.state))
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.calls == []


@pytest.mark.parametrize("missing", ["launch", "instance_ids", "created_at"])
def test_incomplete_state_cannot_authorize_another_purchase(tmp_path, missing):
    api, clock = FakeAPI(), Clock()
    watcher = _watcher(tmp_path, api, clock)
    del watcher.state[missing]
    (tmp_path / "state.json").write_text(json.dumps(watcher.state))
    with pytest.raises(watch.WatchError, match="incomplete"):
        _watcher(tmp_path, api, clock).check()
    assert api.calls == []


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "another-name"},
        {"instance_type_name": "gpu_8x_h100_sxm5"},
        {"ssh_key_names": ["another-key"]},
    ],
)
def test_pending_request_must_match_persisted_purchase_config(tmp_path, changed):
    api, clock = FakeAPI(), Clock()
    api.launch = watch.APIError(None)
    assert _watcher(tmp_path, api, clock).check() is False
    path = tmp_path / "state.json"
    saved = json.loads(path.read_text())
    saved["launch"].update(changed)
    path.write_text(json.dumps(saved))
    api.calls.clear()
    clock.advance()
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.calls == []


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        {},
        {watch.TARGET_TYPE: None},
        {
            watch.TARGET_TYPE: {
                "instance_type": {},
                "regions_with_capacity_available": [],
            }
        },
    ],
)
def test_malformed_capacity_fails_without_launch(tmp_path, monkeypatch, malformed):
    api, clock = FakeAPI(), Clock()
    original = api.request

    def bad_capacity(method, path, payload=None):
        if method == "GET" and path == "/instance-types":
            return {"data": malformed}
        return original(method, path, payload)

    monkeypatch.setattr(api, "request", bad_capacity)
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.launches == []


def test_capacity_quote_must_be_for_exact_target_type(tmp_path, monkeypatch):
    api, clock = FakeAPI(), Clock()
    original = api.request

    def wrong_type_quote(method, path, payload=None):
        response = original(method, path, payload)
        if method == "GET" and path == "/instance-types":
            response["data"][watch.TARGET_TYPE]["instance_type"]["name"] = (
                "gpu_8x_h100_sxm5"
            )
        return response

    monkeypatch.setattr(api, "request", wrong_type_quote)
    with pytest.raises(watch.WatchError):
        _watcher(tmp_path, api, clock).check()
    assert api.launches == []
