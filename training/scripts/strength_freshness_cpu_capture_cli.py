"""Pinned stdlib measurement launcher; never a target qualification grant.

Run only from an independently reviewed immutable control tree and outer finite
launcher: python -S -E -B -m scripts.strength_freshness_cpu_capture_cli ... .
The launch hash, source/bootstrap/import closure and target registration require
external approval. An immutable manifest cannot authorize itself.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time
from typing import Any

from scripts import strength_freshness_cpu_capture_request as requests
from scripts import strength_freshness_cpu_collector as collector
from scripts import strength_freshness_cpu_readonly as readonly
from scripts import strength_freshness_cpu_preservation as preservation

LAUNCH = "strength-preservation-capture-launch-v1"
PROVENANCE = "strength-preservation-capture-provenance-bundle-v1"
PROVENANCE_LIMIT = 4 * 2**20
COMMON = {
    "format",
    "schema_version",
    "phase",
    "request",
    "registration",
    "physical_kind",
}


class LaunchRefusal(ValueError):
    """Only an authored fixed reason code is public."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise LaunchRefusal(reason)


def encoded(value: object) -> bytes:
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        require(nodes <= 250000 and depth <= 64, "publication-structure-limit")
        if isinstance(item, dict):
            require(all(isinstance(k, str) for k in item), "publication-key")
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((v, depth + 1) for v in item)
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise LaunchRefusal("publication-json") from None


def file_pin(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": requests.digest(raw), "bytes": len(raw)}


def launch(raw: bytes) -> dict[str, Any]:
    value = requests.records.parse_json(raw)
    phase = value.get("phase")
    require(phase in {"before", "after"}, "launch-phase")
    require(
        set(value)
        == COMMON
        | (
            {"input_root", "verified_champions"}
            if phase == "before"
            else {"plan", "anchor"}
        ),
        "launch-fields",
    )
    require(
        value["format"] == LAUNCH
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1,
        "launch-format",
    )
    require(
        value["physical_kind"] in {"learner_metrics", "actor_broker"},
        "launch-physical-kind",
    )
    for key in ("request", "registration") + (
        () if phase == "before" else ("plan", "anchor")
    ):
        requests.pin(value[key])
    if phase == "before":
        requests.canonical(value["input_root"])
        verified = value["verified_champions"]
        require(
            isinstance(verified, dict)
            and 0 < len(verified) <= 32
            and all(
                isinstance(k, str)
                and re.fullmatch(r"sha256-[0-9a-f]{64}", k)
                and requests.checksum(v)
                for k, v in verified.items()
            ),
            "launch-champion-map",
        )
    return value


def custom_modules():
    """Exact direct/transitive custom modules executed by this entrypoint."""
    return (
        sys.modules[__name__],
        requests,
        requests.completion,
        requests.records,
        requests.lifecycle,
        preservation,
        collector,
        collector.identities,
        collector.identities.facts,
        collector.processes,
        collector.processes.auxiliary,
        collector.supports,
        readonly,
    )


def verify_sources(reader, value):
    requests.executed_sources(value, modules=custom_modules())
    for expected in value["source_pins"]:
        reader.read(expected)


class SystemRuntime:
    owner_uid = 0

    def monotonic_ns(self):
        return time.monotonic_ns()

    def wall_ns(self):
        return time.time_ns()

    def clock(self, reader):
        reader.check()
        fd = os.open(
            "/proc/sys/kernel/random/boot_id",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            raw = os.read(fd, 129)
        finally:
            os.close(fd)
        reader.charge(len(raw))
        reader.check()
        require(len(raw) <= 128, "bootstrap-boot-size")
        try:
            boot = raw.decode("ascii").strip()
        except UnicodeError:
            raise LaunchRefusal("bootstrap-boot-format") from None
        require(
            re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", boot),
            "bootstrap-boot-format",
        )
        return {
            "boot_id": boot,
            "monotonic_ns": self.monotonic_ns(),
            "wall_ns": self.wall_ns(),
        }

    def make_io(self, scope, deadline):
        return readonly.ReadOnlyIO(
            scope, deadline=deadline, maximum_bytes=requests.RUNTIME_BUDGET
        )


def restamp(measurement, *, now, policy, verified_champions, admission=None):
    """No producer evidence is rewritten; project only unchanged support ages."""
    output = copy.deepcopy(measurement)
    provenance = output["provenance"]
    old = provenance["read_end"]
    requests.clock(now)
    require(
        old["boot_id"] == now["boot_id"]
        and all(
            type(old[k]) is int and old[k] <= now[k]
            for k in ("monotonic_ns", "wall_ns")
        ),
        "final-clock-regressed",
    )
    capture = output["capture"]
    require(
        capture["clock"] == collector.identities.capture_clock(old),
        "core-clock-binding",
    )
    extra = (now["monotonic_ns"] - old["monotonic_ns"] + 999_999_999) // 1_000_000_000
    for row in capture["support"].values():
        if row["running_seconds"] is not None:
            row["running_seconds"] += extra
        if row["job"] is not None:
            row["job"]["age_seconds"] += extra
    capture["clock"] = collector.identities.capture_clock(now)
    preservation._snapshot(policy, capture, verified_champions)
    verified = None
    if admission is not None:
        preservation._physical_work(policy, capture, admission.cleanup_clock)
        verified = preservation.verify_preservation(
            policy,
            admission.before,
            capture,
            verified_champions=verified_champions,
            attempt=admission.anchor.as_dict(),
            cleanup_clock=admission.cleanup_clock,
            audit_clock=capture["clock"],
        )
    return (
        output,
        {
            "core_measurement_clock": old,
            "final_clock": dict(now),
            "added_support_age_seconds": extra,
        },
        verified,
    )


def _directory_identity(fd):
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino, st.st_uid, stat.S_IMODE(st.st_mode))


def publish_files(parent: Path, entries, *, owner_uid: int, check):
    """Publish immutable evidence in order; never retract a visible artifact.

    The final capture is the commit marker. Earlier artifacts survive any crash;
    neither an orphan receipt nor a retry can overwrite or authorize completion.
    """
    require(
        parent == requests.canonical(str(parent)) and parent.resolve() == parent,
        "output-parent-canonical",
    )
    names = [name for name, _ in entries]
    require(
        len(names) == len(set(names))
        and all(
            re.fullmatch(
                r"r3-(?:before|after)(?:\.provenance-[0-9a-f]{64}|\.receipt)?\.json", n
            )
            for n in names
        ),
        "output-name",
    )
    check()
    fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temp = None
    try:
        identity = _directory_identity(fd)
        require(identity[2:] == (owner_uid, 0o700), "output-parent-protection")

        def directory_check():
            check()
            st = parent.lstat()
            require(
                stat.S_ISDIR(st.st_mode)
                and (st.st_dev, st.st_ino, st.st_uid, stat.S_IMODE(st.st_mode))
                == identity
                and parent.resolve() == parent,
                "output-parent-raced",
            )

        directory_check()
        for name in names:
            try:
                os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise LaunchRefusal("output-already-exists")
        for name, raw in entries:
            directory_check()
            temp = "." + name + ".tmp-" + secrets.token_hex(16)
            out = os.open(
                temp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o400,
                dir_fd=fd,
            )
            with os.fdopen(out, "wb") as stream:
                for offset in range(0, len(raw), 65536):
                    check()
                    stream.write(raw[offset : offset + 65536])
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            directory_check()
            os.link(temp, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            os.fsync(fd)
            os.unlink(temp, dir_fd=fd)
            temp = None
            os.fsync(fd)
            directory_check()
    finally:
        if temp is not None:
            try:
                os.unlink(temp, dir_fd=fd)
            except FileNotFoundError:
                pass
        os.close(fd)


def _sidecar(
    value, measurement, capture_pin, request_sha, registration_sha, derivations, now
):
    return {
        "format": requests.RECEIPT,
        "schema_version": 2,
        "status": "complete",
        "request_sha256": request_sha,
        "registration_sha256": registration_sha,
        "encoding_contract_sha256": measurement["provenance"][
            "encoding_contract_sha256"
        ],
        "source_pins": value["source_pins"],
        "read_start": measurement["provenance"]["read_start"],
        "read_end": dict(now),
        "capture_pin": capture_pin,
        "raw_inventory_sha256": derivations.pop("raw_inventory_sha256"),
        "derivations": derivations,
        "restart_counter_scopes": derivations.pop("_counter_scopes"),
        "refusals": [],
    }


def unpublished(reader, parent, stem):
    """An old receipt, capture, provenance or staging fragment is not a retry."""
    count = 0
    with os.scandir(parent) as entries:
        for entry in entries:
            reader.check()
            count += 1
            require(count <= 4096, "output-inventory-limit")
            reader.charge(len(entry.name.encode()))
            require(
                entry.name not in {stem + ".json", stem + ".receipt.json"}
                and not entry.name.startswith(
                    (stem + ".provenance-", "." + stem + ".")
                ),
                "output-already-exists",
            )


def _publish_refusal(
    error,
    *,
    reader,
    rt,
    value,
    config,
    expected_sha,
    output_path,
    registration,
    admission_clock,
    check,
):
    """Best effort only inside the same admitted scope and original budget.

    No failure closure is claimed if expired, out of budget or storage fails.
    In particular this never authorizes a retry or a capture commit marker.
    """
    try:
        known = isinstance(
            error,
            (
                collector.CaptureRefusal,
                readonly.ReadRefusal,
                requests.RequestRefusal,
                LaunchRefusal,
            ),
        )
        if known and (
            "budget" in str(error)
            or "deadline" in str(error)
            or "output-size" in str(error)
            or "structure-limit" in str(error)
        ):
            return
        check()
        require(reader.consumed <= requests.METADATA_BUDGET, "metadata-budget")
        require(
            not output_path.exists() and not output_path.is_symlink(),
            "output-already-exists",
        )
        receipt_path = output_path.with_name(output_path.stem + ".receipt.json")
        require(
            not receipt_path.exists() and not receipt_path.is_symlink(),
            "output-already-exists",
        )
        partial = (
            copy.deepcopy(error.provenance)
            if isinstance(error, collector.CaptureRefusal)
            else {}
        )
        if isinstance(error, readonly.ReadRefusal) and error.audit:
            partial["failed_operation"] = copy.deepcopy(error.audit)
        now = rt.clock(reader)
        requests.effective_deadline(value, now)
        raw_inventory = partial.get("raw_inventory", [])
        raw_hash = preservation.digest(raw_inventory)
        started = partial.get("read_start", admission_clock)
        binding = {
            "phase": value["phase"],
            "request_sha256": config["request"]["sha256"],
            "registration_file_sha256": config["registration"]["sha256"],
            "encoding_contract_sha256": registration["encoding_contract_sha256"],
            "source_pins": value["source_pins"],
            "capture_pin": None,
            "read_start": started,
            "read_end": now,
        }
        provenance = {
            "format": PROVENANCE,
            "schema_version": 1,
            "status": "refused",
            "binding": binding,
            "raw_inventory": raw_inventory,
            "raw_inventory_sha256": raw_hash,
            "support_witnesses": {},
            "partial_core_provenance": partial,
            "metadata_audit": copy.deepcopy(reader.audit),
            "refusals": ["capture-cli-refused"],
            "execution_qualified": False,
        }
        raw = encoded(provenance)
        require(len(raw) <= PROVENANCE_LIMIT, "provenance-output-size")
        provenance_path = output_path.with_name(
            output_path.stem + ".provenance-" + requests.digest(raw) + ".json"
        )
        provenance_pin = file_pin(provenance_path, raw)
        receipt = {
            "format": requests.RECEIPT,
            "schema_version": 2,
            "status": "refused",
            "request_sha256": config["request"]["sha256"],
            "registration_sha256": config["registration"]["sha256"],
            "encoding_contract_sha256": registration["encoding_contract_sha256"],
            "source_pins": value["source_pins"],
            "read_start": started,
            "read_end": now,
            "capture_pin": None,
            "raw_inventory_sha256": raw_hash,
            "derivations": {
                "support_witnesses": {},
                "provenance_pin": provenance_pin,
                "launch_sha256": expected_sha,
            },
            "restart_counter_scopes": registration["counter_scopes"],
            "refusals": ["capture-cli-refused"],
        }
        receipt_raw = encoded(receipt)
        require(
            set(receipt) == requests.RECEIPT_FIELDS
            and len(receipt_raw) <= requests.FILE_LIMIT,
            "receipt-output-size",
        )
        publish_files(
            output_path.parent,
            [(provenance_path.name, raw), (receipt_path.name, receipt_raw)],
            owner_uid=rt.owner_uid,
            check=check,
        )
    except Exception:
        # The outer qualified launcher must retain this process's fixed failure.
        # Previously published artifacts are never deleted or overwritten here.
        return


def execute_launch(
    path: str, expected_sha: str, deadline_monotonic_ns: int, *, runtime=None
):
    """Authority remains with the independently reviewed outer launcher.

    runtime is the local fault-test syscall seam; the CLI exposes no override.
    """
    rt = SystemRuntime() if runtime is None else runtime
    require(
        requests.checksum(expected_sha) and type(deadline_monotonic_ns) is int,
        "launch-arguments",
    )
    require(
        0 < deadline_monotonic_ns - rt.monotonic_ns() <= 120 * 10**9,
        "launch-original-window",
    )
    reader = requests.PinnedReader(
        deadline=deadline_monotonic_ns / 1e9,
        owner_uid=rt.owner_uid,
        monotonic=lambda: rt.monotonic_ns() / 1e9,
    )
    path_obj = requests.canonical(path)
    reader.directory(path_obj.parent)
    size = path_obj.lstat().st_size
    launch_pin = {"path": path, "sha256": expected_sha, "bytes": size}
    raw = reader.read(launch_pin)
    config = launch(raw)
    require(
        all(
            requests.canonical(config[k]["path"]).parent == path_obj.parent
            for k in ("request", "registration")
        ),
        "launch-input-parent",
    )
    value = requests.request(reader.read(config["request"]))
    require(
        value["phase"] == config["phase"]
        and value["limits"]["deadline_monotonic_ns"] == deadline_monotonic_ns,
        "launch-request-binding",
    )
    registration = requests.validate_registration(
        value, reader.read(config["registration"])
    )
    require(
        config["registration"]["sha256"] == value["registration_sha256"],
        "launch-registration-binding",
    )
    now = rt.clock(reader)
    reader.deadline = min(reader.deadline, requests.effective_deadline(value, now))
    verify_sources(reader, value)
    admission = None
    if value["phase"] == "before":
        output_root = requests.canonical(config["input_root"])
        require(output_root == path_obj.parent, "before-output-root")
        verified = config["verified_champions"]
        output_path = output_root / "r3-before.json"
    else:
        admission_now = rt.clock(reader)
        reader.deadline = min(
            reader.deadline, requests.effective_deadline(value, admission_now)
        )
        admission = requests.admit_after(
            value=value,
            registration=registration,
            reader=reader,
            plan_pin=config["plan"],
            anchor_pin=config["anchor"],
            now=admission_now,
        )
        verified = admission.verified_champions
        output_path = requests.canonical(admission.after_path)
        output_root = output_path.parent
    directory = reader.directory(output_root)
    unpublished(reader, output_root, output_path.stem)
    io = rt.make_io(
        collector.identities.scope_from(registration["scope"]), reader.deadline
    )
    require(
        io.deadline <= reader.deadline and io.maximum_bytes == requests.RUNTIME_BUDGET,
        "runtime-budget-binding",
    )

    def budget_check():
        reader.check()
        require(
            rt.monotonic_ns() < deadline_monotonic_ns
            and rt.wall_ns() < value["limits"]["deadline_wall_ns"],
            "publication-original-deadline",
        )

    measurement = None
    publishing = False
    try:
        measurement = collector.collect_capture(
            registration,
            io,
            verified_champions=verified,
            phase=value["phase"],
            cleanup_clock=None if admission is None else admission.cleanup_clock,
            physical_kind=config["physical_kind"],
            previous_support_witnesses=None
            if admission is None
            else admission.support_witnesses,
        )
        # Re-read exact protected metadata and actual module paths; no new authority.
        reader.read(launch_pin)
        reader.read(config["request"])
        reader.read(config["registration"])
        verify_sources(reader, value)
        if admission is not None:
            reader.read(config["plan"])
            reader.read(config["anchor"])
            admission.proof.recheck()
        require(reader.directory(output_root) == directory, "output-parent-raced")
        final_observation = io.clock()
        now = final_observation.value
        reader.deadline = min(reader.deadline, requests.effective_deadline(value, now))
        require(now["monotonic_ns"] / 1e9 < reader.deadline, "final-deadline")
        measurement, projection, verification = restamp(
            measurement,
            now=now,
            policy=registration["policy"],
            verified_champions=verified,
            admission=admission,
        )
        capture = measurement["capture"]
        output = (
            capture
            if admission is None
            else {
                "format": "strength-freshness-cpu-r3-after-v1",
                "plan_sha256": value["plan_sha256"],
                "anchor_sha256": value["anchor_sha256"],
                "capture": capture,
            }
        )
        capture_bytes = encoded(output)
        require(len(capture_bytes) <= requests.FILE_LIMIT, "capture-output-size")
        capture_pin = file_pin(output_path, capture_bytes)
        core_provenance = copy.deepcopy(measurement["provenance"])
        inventory = core_provenance.pop("raw_inventory")
        core_provenance["raw_inventory_count"] = len(inventory)
        inventory.append(copy.deepcopy(dict(final_observation.audit)))
        inventory_sha = preservation.digest(inventory)
        provenance = {
            "format": PROVENANCE,
            "schema_version": 1,
            "binding": {
                "phase": value["phase"],
                "request_sha256": config["request"]["sha256"],
                "registration_file_sha256": config["registration"]["sha256"],
                "encoding_contract_sha256": registration["encoding_contract_sha256"],
                "source_pins": value["source_pins"],
                "capture_pin": capture_pin,
                "read_start": measurement["provenance"]["read_start"],
                "read_end": dict(now),
            },
            "raw_inventory": inventory,
            "raw_inventory_sha256": inventory_sha,
            "support_witnesses": measurement["provenance"]["support_witnesses"],
            "core_provenance": core_provenance,
            "metadata_audit": copy.deepcopy(reader.audit),
            "final_clock_projection": projection,
            "preservation_verification": verification,
        }
        provenance_bytes = encoded(provenance)
        require(len(provenance_bytes) <= PROVENANCE_LIMIT, "provenance-output-size")
        stem = output_path.stem
        provenance_path = output_root / (
            stem + ".provenance-" + requests.digest(provenance_bytes) + ".json"
        )
        provenance_pin = file_pin(provenance_path, provenance_bytes)
        derivations = {
            "support_witnesses": copy.deepcopy(
                measurement["provenance"]["support_witnesses"]
            ),
            "provenance_pin": provenance_pin,
            "core_measurement_clock": projection["core_measurement_clock"],
            "final_clock_projection": projection,
            "launch_sha256": expected_sha,
            "raw_inventory_sha256": inventory_sha,
            "_counter_scopes": registration["counter_scopes"],
        }
        receipt = _sidecar(
            value,
            measurement,
            capture_pin,
            config["request"]["sha256"],
            config["registration"]["sha256"],
            derivations,
            now,
        )
        require(set(receipt) == requests.RECEIPT_FIELDS, "receipt-fields")
        receipt_bytes = encoded(receipt)
        require(len(receipt_bytes) <= requests.FILE_LIMIT, "receipt-output-size")
        receipt_path = output_root / (stem + ".receipt.json")

        budget_check()
        publishing = True
        publish_files(
            output_root,
            [
                (provenance_path.name, provenance_bytes),
                (receipt_path.name, receipt_bytes),
                (output_path.name, capture_bytes),
            ],
            owner_uid=rt.owner_uid,
            check=budget_check,
        )
        return {
            "status": "complete",
            "capture_pin": capture_pin,
            "receipt_pin": file_pin(receipt_path, receipt_bytes),
            "provenance_pin": provenance_pin,
            "execution_qualified": False,
            "cuda_qualified": False,
        }
    except Exception as error:
        if not publishing:
            _publish_refusal(
                error,
                reader=reader,
                rt=rt,
                value=value,
                config=config,
                expected_sha=expected_sha,
                output_path=output_path,
                registration=registration,
                admission_clock=now,
                check=budget_check,
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--deadline-monotonic-ns", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        require(
            sys.platform == "linux"
            and os.geteuid() == 0
            and sys.flags.no_site
            and sys.flags.ignore_environment
            and sys.flags.dont_write_bytecode,
            "qualified-bootstrap-mode-required",
        )
        require(
            not any(
                n == "torch"
                or n.startswith(("torch.", "deltreltrain", "startrain"))
                or n
                in {
                    "scripts.strength_freshness_linux",
                    "scripts.strength_freshness_cpu_qualification",
                }
                for n in sys.modules
            ),
            "forbidden-bootstrap-import",
        )
        result = execute_launch(
            args.launch, args.launch_sha256, args.deadline_monotonic_ns
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        # No raw exception strings, private process data or partial verdicts.
        print('{"status":"refused","reason":"capture-cli-refused"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
