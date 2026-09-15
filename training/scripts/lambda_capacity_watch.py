#!/usr/bin/env python3
"""Politely watch Lambda capacity and launch one 8x B200 instance.

Python 3.11+, standard library only, on Linux or macOS. See
training/docs/lambda-capacity-watch.md for setup and recovery instructions.
"""

from __future__ import annotations

import argparse
import email.utils
import fcntl
import getpass
import http.client
import json
import logging
import math
import os
import random
import re
import signal
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypeGuard

API_BASE = "https://cloud.lambda.ai/api/v1"
TARGET_TYPE = "gpu_8x_b200_sxm6"
DEFAULT_STATE = Path.home() / ".local/state/edgeconnect/lambda-b200-watch.json"
CAPACITY_ERROR = "instance-operations/launch/insufficient-capacity"
REJECTED_ERRORS = {
    CAPACITY_ERROR,
    "instance-operations/launch/file-system-in-wrong-region",
    "global/invalid-parameters",
    "global/object-does-not-exist",
    "global/quota-exceeded",
    "global/invalid-api-key",
    "global/account-inactive",
    "global/invalid-address",
}
LOG = logging.getLogger("lambda-capacity-watch")


class WatchError(RuntimeError):
    """An actionable configuration, state, or API error; do not loop blindly."""


class APIError(WatchError):
    def __init__(
        self, status: int | None, code: str | None = None, retry_after: float = 0
    ) -> None:
        self.status = status
        self.code = code
        self.retry_after = retry_after
        # Do not log arbitrary response bodies, URLs, or authentication headers.
        detail = f" ({code})" if code in REJECTED_ERRORS else ""
        super().__init__(f"Lambda API HTTP {status or 'transport failure'}{detail}")


def retry_after_seconds(value: str | None, now: float) -> float:
    if value is None:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - now
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Authorization must never be forwarded to another URL.
        return None


class LambdaAPI:
    """No automatic HTTP retries, especially for the non-idempotent launch POST."""

    def __init__(
        self,
        api_key: str,
        *,
        opener: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key or any(character.isspace() for character in api_key):
            raise WatchError("Set LAMBDA_API_KEY to a valid key in your environment.")
        self._key = api_key
        self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._clock, self._sleep = clock, sleep
        self._last_request: float | None = None
        self._last_launch: float | None = None

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if (method, path) not in {
            ("GET", "/instance-types"),
            ("GET", "/instances"),
            ("GET", "/ssh-keys"),
            ("POST", "/instance-operations/launch"),
        }:
            raise WatchError("Unsupported API operation.")
        delay = 0.0
        if self._last_request is not None:
            delay = max(delay, 2 - (self._clock() - self._last_request))
        if method == "POST" and self._last_launch is not None:
            delay = max(delay, 15 - (self._clock() - self._last_launch))
        if delay > 0:
            self._sleep(delay)
        self._last_request = self._clock()
        if method == "POST":
            self._last_launch = self._last_request
        request = urllib.request.Request(
            API_BASE + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "EdgeConnect-capacity-watch/1.0",
            },
            method=method,
        )
        try:
            with self._opener.open(request, timeout=30) as response:
                data = response.read(4 * 1024 * 1024 + 1)
                if not 200 <= response.status < 300:
                    raise APIError(response.status)
                if len(data) > 4 * 1024 * 1024:
                    raise APIError(None)
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise APIError(None)
                return value
        except urllib.error.HTTPError as error:
            try:
                value = json.loads(error.read(65536))
                code = value.get("error", {}).get("code")
                code = code if isinstance(code, str) else None
            except (ValueError, AttributeError, OSError, http.client.HTTPException):
                code = None
            delay = retry_after_seconds(error.headers.get("Retry-After"), time.time())
            error.close()
            raise APIError(error.code, code, delay) from None
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
            raise APIError(None) from None


@dataclass(frozen=True)
class Config:
    ssh_key: str
    regions: tuple[str, ...] = ()
    max_hourly_cents: int = 6000
    interval_seconds: float = 300
    name: str = "edgeconnect-b200-reserve"
    dry_run: bool = False

    def validate(self) -> None:
        if (
            not self.ssh_key
            or len(self.ssh_key) > 128
            or any(ord(character) < 32 for character in self.ssh_key)
        ):
            raise WatchError("Specify the name of one existing Lambda SSH key.")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.name) is None:
            raise WatchError(
                "Instance name must be 1–64 letters, numbers, '.', '_' or '-'."
            )
        if any(
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", r) is None for r in self.regions
        ):
            raise WatchError("Invalid region name.")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 120:
            raise WatchError("Polling interval must be at least 120 seconds.")
        if type(self.max_hourly_cents) is not int or self.max_hourly_cents <= 0:
            raise WatchError("Hourly price ceiling must be positive.")

    def identity(self) -> dict[str, Any]:
        # Operational settings may change on restart; purchase identity may not.
        return {
            "instance_type": TARGET_TYPE,
            "ssh_key": self.ssh_key,
            "regions": sorted(set(self.regions)),
            "max_hourly_cents": self.max_hourly_cents,
            "name": self.name,
        }


def _prepare_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise WatchError("State directory must be a real directory.")


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
    _prepare_parent(path)
    descriptor = os.open(
        path.with_name(path.name + ".lock"),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WatchError("State lock must be a regular file.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WatchError("Another watcher already owns this state file.") from None
        yield
    finally:
        os.close(descriptor)


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise WatchError("State must be a regular file.")
        try:
            value = json.loads(stream.read(1024 * 1024 + 1))
        except ValueError:
            raise WatchError(
                "State is corrupt; inspect it rather than deleting it blindly."
            ) from None
    if not isinstance(value, dict):
        raise WatchError("State is not a JSON object.")
    return value


def write_state(path: Path, value: dict[str, Any]) -> None:
    _prepare_parent(path)
    if path.is_symlink():
        raise WatchError("State may not be a symbolic link.")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def target_capacity(response: dict[str, Any]) -> tuple[int, list[str]]:
    try:
        entry = response["data"][TARGET_TYPE]
        instance_type = entry["instance_type"]
        price = instance_type["price_cents_per_hour"]
        gpus = instance_type["specs"]["gpus"]
        regions = [row["name"] for row in entry["regions_with_capacity_available"]]
        if (
            instance_type["name"] != TARGET_TYPE
            or type(price) is not int
            or price <= 0
            or type(gpus) is not int
            or gpus != 8
        ):
            raise ValueError
        if any(
            not isinstance(r, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", r) is None
            for r in regions
        ):
            raise ValueError
        return price, sorted(set(regions))
    except (KeyError, TypeError, ValueError):
        raise WatchError(
            "Lambda did not return a valid 8× B200 type, price, and capacity list."
        ) from None


def ssh_key_names(response: dict[str, Any]) -> list[str]:
    keys = response.get("data")
    if not isinstance(keys, list) or any(
        not isinstance(key, dict)
        or not isinstance(key.get("name"), str)
        or not key["name"]
        for key in keys
    ):
        raise WatchError("Lambda returned an invalid SSH key inventory.")
    return [key["name"] for key in keys]


class Watcher:
    def __init__(
        self,
        config: Config,
        api: LambdaAPI,
        state_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        config.validate()
        self.config, self.api, self.state_path = config, api, state_path
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        self.preflight_done = False
        self.state = read_state(state_path) or {}
        if self.state:
            self._validate_state()
        else:
            # A valid empty object is corrupt state, not permission to buy again.
            if state_path.exists():
                raise WatchError("Empty state file; inspect its launch history first.")
            self.state = {
                "version": 1,
                "config": config.identity(),
                "phase": "watching",
                "created_at": clock(),
                "next_check_at": 0,
                "failures": 0,
                "launch": None,
                "instance_ids": [],
            }
            self._save()

    def _validate_state(self) -> None:
        value = self.state
        if (
            not {
                "version",
                "config",
                "phase",
                "created_at",
                "next_check_at",
                "failures",
                "launch",
                "instance_ids",
            }
            <= value.keys()
        ):
            raise WatchError("State is incomplete; inspect its launch history first.")
        if (
            type(value.get("version")) is not int
            or value["version"] != 1
            or value.get("config") != self.config.identity()
        ):
            raise WatchError(
                "State/configuration mismatch; use the original purchase settings."
            )
        if value.get("phase") not in ("watching", "pending", "reserved"):
            raise WatchError("State has an invalid phase.")
        next_check = value.get("next_check_at")
        failures = value.get("failures")
        if (
            not isinstance(next_check, (int, float))
            or isinstance(next_check, bool)
            or not math.isfinite(next_check)
            or next_check < 0
        ):
            raise WatchError("State has an invalid polling deadline.")
        if type(failures) is not int or not 0 <= failures <= 100:
            raise WatchError("State has an invalid backoff count.")
        launch = value.get("launch")
        if value["phase"] != "reserved" and (
            "reserved_at" in value or "region" in value
        ):
            raise WatchError(
                "State contains a prior reservation; refusing another launch."
            )
        if value["phase"] == "watching" and (
            launch is not None or value.get("instance_ids") != []
        ):
            raise WatchError(
                "Watching state contains a prior launch; inspect the Lambda console."
            )
        if value["phase"] == "pending" and (
            launch is None or value.get("instance_ids") != []
        ):
            raise WatchError(
                "Pending launch state is incomplete; inspect the Lambda console."
            )
        if launch is not None:
            if not isinstance(launch, dict):
                raise WatchError("Launch state is malformed.")
            region = launch.get("region_name")
            expected = {
                "region_name": region,
                "instance_type_name": TARGET_TYPE,
                "ssh_key_names": [self.config.ssh_key],
                "name": self.config.name,
            }
            if (
                launch != expected
                or not isinstance(region, str)
                or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", region) is None
                or (self.config.regions and region not in self.config.regions)
            ):
                raise WatchError("Saved launch differs from the configured purchase.")
        if value["phase"] == "reserved" and not self._valid_ids(
            value.get("instance_ids")
        ):
            raise WatchError("Reserved state has invalid instance IDs.")

    @staticmethod
    def _valid_ids(value: Any) -> TypeGuard[list[str]]:
        return (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
            and re.fullmatch(r"[A-Za-z0-9-]{1,128}", value[0]) is not None
        )

    def _save(self) -> None:
        write_state(self.state_path, self.state)

    def _backoff(self, error: APIError) -> None:
        self.state["failures"] = min(100, self.state["failures"] + 1)
        delay = min(
            3600, self.config.interval_seconds * 2 ** min(self.state["failures"], 5)
        )
        delay = max(delay + self.jitter(0, 30), error.retry_after)
        self.state["next_check_at"] = max(
            self.state["next_check_at"], self.clock() + delay
        )
        self._save()
        LOG.warning("%s; next check in at least %.0f seconds.", error, delay)

    def _existing(self) -> dict[str, Any] | None:
        rows = self.api.request("GET", "/instances").get("data")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not self._valid_ids([row.get("id")])
            or (row.get("name") is not None and not isinstance(row["name"], str))
            for row in rows
        ):
            raise WatchError("Invalid instance inventory; refusing to launch.")
        matches = [row for row in rows if row.get("name") == self.config.name]
        if not matches:
            return None
        if len(matches) != 1:
            raise WatchError(
                "Multiple instances share this watcher name; inspect the console."
            )
        instance = matches[0]
        try:
            region = instance["region"]["name"]
            compatible = (
                instance["instance_type"]["name"] == TARGET_TYPE
                and instance["status"] in ("booting", "active", "unhealthy")
                and self._valid_ids([instance["id"]])
                and isinstance(region, str)
                and (not self.config.regions or region in self.config.regions)
            )
            if self.state["phase"] == "pending":
                compatible = (
                    compatible and region == self.state["launch"]["region_name"]
                )
        except (KeyError, TypeError):
            compatible = False
        if not compatible:
            raise WatchError(
                "An incompatible or stopped instance already uses this name."
            )
        return instance

    def _reserved(self, ids: list[str], region: str) -> bool:
        self.state.update(
            phase="reserved", instance_ids=ids, region=region, reserved_at=self.clock()
        )
        self._save()
        LOG.info(
            "Reserved %s in %s: instance %s. Watcher is stopping.",
            TARGET_TYPE,
            region,
            ids[0],
        )
        return True

    def check(self) -> bool:
        """One paced cycle. Pending intent can only reconcile, never launch again."""
        if self.state["phase"] == "reserved":
            LOG.info(
                "Already reserved instance %s; no new launch.",
                self.state["instance_ids"][0],
            )
            return True
        if self.clock() < self.state["next_check_at"]:
            return False
        self.state["next_check_at"] = (
            self.clock() + self.config.interval_seconds + self.jitter(0, 30)
        )
        self._save()  # Preserve cooldown even if the process dies mid-request.
        try:
            if self.state["phase"] == "pending":
                existing = self._existing()
                if existing:
                    return self._reserved([existing["id"]], existing["region"]["name"])
                LOG.warning(
                    "Launch outcome is unconfirmed. Checking inventory only; no further launch requests will be sent. Inspect the console if this persists."
                )
                return False
            if not self.preflight_done:
                keys = ssh_key_names(self.api.request("GET", "/ssh-keys"))
                if self.config.ssh_key not in keys:
                    raise WatchError(
                        "SSH key name was not found in this Lambda workspace."
                    )
                existing = self._existing()
                if existing:
                    return self._reserved([existing["id"]], existing["region"]["name"])
                self.preflight_done = True
            price, regions = target_capacity(self.api.request("GET", "/instance-types"))
            regions = [
                r
                for r in regions
                if not self.config.regions or r in self.config.regions
            ]
            if self.config.dry_run:
                LOG.info(
                    "Dry run: %s costs $%.2f/hour; eligible capacity: %s. No launch sent.",
                    TARGET_TYPE,
                    price / 100,
                    ", ".join(regions) or "none",
                )
                return True
            if price > self.config.max_hourly_cents:
                LOG.info(
                    "Price $%.2f/hour exceeds ceiling $%.2f/hour; waiting.",
                    price / 100,
                    self.config.max_hourly_cents / 100,
                )
                self.state["failures"] = 0
                self._save()
                return False
            if not regions:
                LOG.info(
                    "No matching B200 capacity; next check in at least %.0f seconds.",
                    self.config.interval_seconds,
                )
                self.state["failures"] = 0
                self._save()
                return False
            existing = self._existing()
            if existing:
                return self._reserved([existing["id"]], existing["region"]["name"])
        except APIError as error:
            if error.status is None or error.status == 429 or error.status >= 500:
                self._backoff(error)
                return False
            raise

        region = random.choice(regions)
        payload = {
            "region_name": region,
            "instance_type_name": TARGET_TYPE,
            "ssh_key_names": [self.config.ssh_key],
            "name": self.config.name,
        }
        self.state.update(
            phase="pending",
            launch=payload,
            launch_started_at=self.clock(),
            quoted_hourly_cents=price,
            next_check_at=self.clock()
            + self.config.interval_seconds
            + self.jitter(0, 30),
        )
        self._save()  # Must reach disk BEFORE this non-idempotent POST.
        LOG.info(
            "Launching one %s in %s at quoted $%.2f/hour.",
            TARGET_TYPE,
            region,
            price / 100,
        )
        try:
            response = self.api.request("POST", "/instance-operations/launch", payload)
        except APIError as error:
            if error.status == 400 and error.code == CAPACITY_ERROR:
                self.state.update(phase="watching", launch=None, failures=0)
                self.state["next_check_at"] = max(
                    self.state["next_check_at"], self.clock() + error.retry_after
                )
                self._save()
                LOG.info(
                    "Capacity disappeared before launch; waiting for the next check."
                )
                return False
            if error.status in (400, 401, 403, 404) and error.code in REJECTED_ERRORS:
                self.state.update(phase="watching", launch=None)
                self._save()
                raise
            self._backoff(error)
            LOG.warning(
                "Launch response is ambiguous. Intent remains saved; subsequent checks are read-only."
            )
            return False
        ids = (
            response.get("data", {}).get("instance_ids")
            if isinstance(response.get("data"), dict)
            else None
        )
        if not self._valid_ids(ids):
            LOG.warning(
                "Launch response has no single valid instance ID; retaining pending intent for read-only reconciliation."
            )
            return False
        return self._reserved(ids, region)

    def run(self) -> None:
        if self.config.dry_run:
            self.check()
            LOG.info("Dry run complete; no launch request was sent.")
            return
        while not self.check():
            # Short sleeps make cancellation responsive and do not add API calls.
            self.sleep(min(60, max(0, self.state["next_check_at"] - self.clock())))


def dollars_to_cents(value: str) -> int:
    try:
        amount = Decimal(value)
        cents = amount * 100
        if not amount.is_finite() or cents <= 0 or cents != cents.to_integral_value():
            raise ValueError
        return int(cents)
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError(
            "Use a positive dollar amount with at most two decimals."
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssh-key", help="Name of one SSH key already in your Lambda workspace"
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Optional region allowlist; default: any region",
    )
    parser.add_argument(
        "--max-hourly-usd",
        type=dollars_to_cents,
        default=6000,
        help="Whole-instance price ceiling, default: 60.00 USD/hour",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300,
        help="Default 300, minimum 120; adds 0–30 seconds jitter",
    )
    parser.add_argument("--name", default="edgeconnect-b200-reserve")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="One read-only availability check, then exit",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List SSH key names, current capacity and price; never launch",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        api_key = os.environ.get("LAMBDA_API_KEY", "")
        if not api_key and sys.stdin.isatty():
            api_key = getpass.getpass("Lambda API key (hidden): ")
        api = LambdaAPI(api_key)
        if args.list_options:
            price, regions = target_capacity(api.request("GET", "/instance-types"))
            keys = ssh_key_names(api.request("GET", "/ssh-keys"))
            print(
                json.dumps(
                    {
                        "instance_type": TARGET_TYPE,
                        "hourly_usd": price / 100,
                        "available_regions": regions,
                        "ssh_key_names": keys,
                    },
                    indent=2,
                )
            )
            return 0
        config = Config(
            args.ssh_key or "",
            tuple(args.region),
            args.max_hourly_usd,
            args.interval_seconds,
            args.name,
            args.dry_run,
        )
        config.validate()
        path = args.state_file.expanduser().absolute()
        LOG.info(
            "Watching %s in %s; ceiling $%.2f/hour; state %s. A successful launch is billable until terminated.",
            TARGET_TYPE,
            ", ".join(config.regions) or "any region",
            config.max_hourly_cents / 100,
            path,
        )
        with state_lock(path):
            Watcher(config, api, path).run()
        return 0
    except KeyboardInterrupt:
        LOG.info(
            "Stopped. Launch state was preserved; reuse the same state file on restart."
        )
        return 130
    except (WatchError, OSError) as error:
        LOG.error("%s", error)
        return 2


def _stop(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    raise SystemExit(main())
