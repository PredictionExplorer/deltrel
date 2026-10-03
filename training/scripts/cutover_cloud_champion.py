"""Journaled final cloud cutover with an independently enabled rollback guard.

The existing production unit and credential file never change. Only the pinned
production YAML, current symlink and their captured enablement are transactional.
A new boot never resumes qualification or trusts old PIDs/monotonic deadlines:
it runs a separately recorded bounded recovery, to old unless acceptance committed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
import types
from typing import Any, cast

# The qualified serving release predates this operator. Deploy this helper as
# an unchanged sibling, and execute only its independently reviewed byte snapshot.
HOST_HELPER_SHA256 = "6bd5448b5fcb9d334fad8e922f4b38c3be75e8fe5dd0a337164e37d996fcdb6c"


def _load_host() -> Any:
    path = Path(__file__).with_name("qualify_cloud_gpu_window.py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != HOST_HELPER_SHA256:
        raise RuntimeError("cutover host helper checksum mismatch")
    module = types.ModuleType("_qualified_cutover_host")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return cast(Any, module)


q = _load_host()

FORMAT = "deltrelserve.cloud-cutover"
PUBLIC = "https://deltrel.com/v2/move"
BINDINGS = (
    "run_id",
    "plan_sha256",
    "boot_id",
    "challenge",
    "acceptance_unit",
    "model_identity",
    "model_step",
)
Refusal = q.Refusal


def pin_bytes(pin: dict[str, Any]) -> bytes:
    path = Path(pin["path"])
    q._absolute(str(path))
    q._sha(pin["sha256"])
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != pin["bytes"]
        or path.stat().st_size > 2**20
    ):
        raise Refusal("unsafe-cutover-control-artifact")
    data = path.read_bytes()
    if q.digest(data) != pin["sha256"]:
        raise Refusal("cutover-control-artifact-changed")
    return data


def production_yaml(qualified: bytes) -> bytes:
    # The actually qualified YAML is pinned; preserve every other byte, including
    # credential variable, model/profile, precision, search and batching settings.
    line = b"\nport: 8081\n"
    if qualified.count(line) != 1:
        raise Refusal("qualified-config-port-line-not-unique")
    return qualified.replace(line, b"\nport: 8080\n", 1)


def load_plan(path: Path, checksum: str) -> dict[str, Any]:
    q._sha(checksum)
    if q.digest(path.read_bytes()) != checksum:
        raise Refusal("cutover-plan-hash-mismatch")
    p = q.read_json(path)
    if (
        p.get("format") != FORMAT
        or p.get("schema_version") != 1
        or p.get("approved_for_cutover") is not True
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,48}", p.get("run_id", "")) is None
    ):
        raise Refusal("explicit-cutover-plan-required")
    for name in (
        "state_directory",
        "current_link",
        "production_config",
        "enablement_directory",
    ):
        q._absolute(p[name])
    if p["enablement_directory"] != "/etc/systemd/system/multi-user.target.wants":
        raise Refusal("standard-persistent-boot-target-required")
    qualified = {
        name: json.loads(pin_bytes(pin)) for name, pin in p["qualification"].items()
    }
    execution, state, result, verification = (
        qualified[k] for k in ("plan", "state", "result", "verification")
    )
    if (
        state.get("qualification_passed") is not True
        or state.get("restored") is not True
        or state["plan_sha256"] != p["qualification"]["plan"]["sha256"]
        or state.get("probe_result_sha256") != p["qualification"]["result"]["sha256"]
        or verification.get("state_sha256") != p["qualification"]["state"]["sha256"]
        or verification.get("probe_result_sha256") != state["probe_result_sha256"]
        or verification.get("status")
        != "passed-exclusive-gpu-qualification-and-independent-old-production-restoration"
        or any(
            verification.get(k) is not True
            for k in (
                "authenticated_loopback_and_public_tls_healthy",
                "stage_probe_inactive_no_jobs",
                "port8081_connection_refused",
                "watchdog_inactive",
            )
        )
        or set(result.get("checks", {})) != set(q.REQUIRED_CHECKS)
        or any(row.get("status") != "passed" for row in result["checks"].values())
    ):
        raise Refusal("successful-closed-gpu-qualification-required")
    old, new = copy.deepcopy(execution["production"]), copy.deepcopy(execution["stage"])
    if (
        old["unit"] != "deltrelserve.service"
        or old["model_step"] != 566428
        or new["model_step"] != 572377
        or result["model_identity"] != new["model_identity"]
        or result["model_step"] != 572377
        or p["current_link"] != execution["current_link"]
        or p["production_config"] != old["config"]
        or str((Path(p["current_link"]).parent / p["old_link_target"]).resolve())
        != old["release"]
    ):
        raise Refusal("cutover-qualified-identities-mismatch")
    qualified_config = next(row for row in new["files"] if row["path"] == new["config"])
    original_config = next(row for row in old["files"] if row["path"] == old["config"])
    if (
        p["qualified_gpu_config"]["sha256"] != qualified_config["sha256"]
        or p["old_config"]["sha256"] != original_config["sha256"]
        or pin_bytes(p["candidate_config"])
        != production_yaml(pin_bytes(p["qualified_gpu_config"]))
    ):
        raise Refusal("cutover-config-diff-exceeds-port-only")
    pin_bytes(p["old_config"])
    for role in (old, new):
        role["files"] = [
            row for row in role["files"] if row["path"] != p["production_config"]
        ]
    # Runtime/model pins come from the successful GPU stage, but the production
    # service identity, credentials and public/loopback endpoints stay unchanged.
    for key in ("unit", "properties", "exec_start", "env_files", "health", "config"):
        new[key] = copy.deepcopy(old[key])
    unit_pin = next(
        row for row in old["files"] if row["path"] == old["properties"]["FragmentPath"]
    )
    if unit_pin not in new["files"]:
        new["files"].append(unit_pin)
    p["old"], p["new"] = old, new
    if (
        p["original_enablement"] not in ("enabled", "disabled")
        or type(p["initial_pid"]) is not int
        or p["initial_pid"] <= 0
        or re.fullmatch(r"[a-f0-9]{32}", p["initial_invocation"]) is None
    ):
        raise Refusal("production-ownership-enablement-required")
    for role, suffix in (("guard", "guard"), ("acceptance", "acceptance")):
        unit = p[role]
        if (
            unit["unit"] != f"deltrelserve-cutover-{p['run_id']}-{suffix}.service"
            or set(unit["properties"]) != set(q.IDENTITY_PROPERTIES)
            or unit["properties"]["User"] != "root"
            or unit["properties"]["Type"] != "exec"
            or unit["properties"]["Restart"] != "no"
            or unit["properties"]["KillMode"] != "control-group"
            or unit["properties"]["SendSIGKILL"] != "yes"
        ):
            raise Refusal("cutover-owned-unit-required")
        if unit["env_files"] or unit["properties"]["EnvironmentFiles"]:
            raise Refusal("cutover-auxiliary-units-cannot-load-credentials")
        if unit["properties"]["FragmentPath"] not in {
            row["path"] for row in unit["files"]
        }:
            raise Refusal("cutover-unit-file-not-pinned")
    if (
        p["guard"]["properties"]["RuntimeMaxUSec"] != "15min 10s"
        or p["guard"]["properties"]["TimeoutStopUSec"] != "5s"
        or p["acceptance"]["properties"]["RuntimeMaxUSec"] != "3min 20s"
        or p["acceptance"]["properties"]["TimeoutStopUSec"] != "5s"
        or p["acceptance"]["properties"]["PrivateDevices"] != "yes"
    ):
        raise Refusal("cutover-auxiliary-bounds-invalid")
    if {row["url"] for row in old["health"]} != {
        "http://127.0.0.1:8080/v2/health",
        "https://deltrel.com/v2/health",
    }:
        raise Refusal("both-production-health-checks-required")
    for request in old["health"]:
        if request["url"] not in (
            "http://127.0.0.1:8080/v2/health",
            "https://deltrel.com/v2/health",
        ) or (
            request.get("env_file")
            and request["url"] != "http://127.0.0.1:8080/v2/health"
        ):
            raise Refusal("cutover-health-target-invalid")
    for dependency in p["dependencies"]:
        pin_bytes(dependency)
    if not {str(Path(__file__).resolve()), str(Path(q.__file__).resolve())} <= {
        str(Path(row["path"]).resolve()) for row in p["dependencies"]
    }:
        raise Refusal("cutover-executed-dependencies-not-pinned")
    return p


class Host(q.Host):
    def enablement(self, unit: str, deadline: float) -> str:
        return self.command(
            ["systemctl", "show", unit, "--value", "--property=UnitFileState"], deadline
        ).strip()

    def enabled(self, verb: str, unit: str, directory: str, deadline: float) -> None:
        self.command(["systemctl", verb, unit], deadline)
        self.confirm_enablement(
            unit, "enabled" if verb == "enable" else "disabled", directory, deadline
        )

    def confirm_enablement(
        self, unit: str, wanted: str, directory: str, deadline: float
    ) -> None:
        fsync_directory(Path(directory))
        link = Path(directory) / unit
        if wanted == "enabled":
            expected = Path(self.unit(unit, deadline)["FragmentPath"])
            if not link.is_symlink() or link.resolve(strict=True) != expected.resolve(
                strict=True
            ):
                raise Refusal("boot-enablement-link-not-confirmed")
        elif os.path.lexists(link):
            raise Refusal("boot-disable-link-still-present")
        if self.enablement(unit, deadline) != wanted:
            raise Refusal("enablement-transition-unconfirmed")

    def pair(self, p: dict[str, Any]) -> tuple[str, str]:
        config, link = Path(p["production_config"]), Path(p["current_link"])
        if config.is_symlink() or not config.is_file() or not link.is_symlink():
            raise Refusal("production-pair-file-type-changed")
        info = config.stat()
        if any(
            value != p["config_metadata"][key]
            for key, value in (
                ("uid", info.st_uid),
                ("gid", info.st_gid),
                ("mode", stat.S_IMODE(info.st_mode)),
            )
        ):
            raise Refusal("production-config-metadata-changed")
        return q.digest(config.read_bytes()), os.readlink(link)

    def replace_config(self, p: dict[str, Any], data: bytes) -> None:
        path = Path(p["production_config"])
        fd, name = tempfile.mkstemp(prefix=".cutover-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                os.fchown(
                    stream.fileno(),
                    p["config_metadata"]["uid"],
                    p["config_metadata"]["gid"],
                )
                os.fchmod(stream.fileno(), p["config_metadata"]["mode"])
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            fsync_directory(path.parent)
        finally:
            Path(name).unlink(missing_ok=True)

    def replace_link(self, p: dict[str, Any], target: str) -> None:
        link = Path(p["current_link"])
        temporary = link.with_name(".cutover-" + secrets.token_hex(16))
        try:
            os.symlink(target, temporary)
            os.replace(temporary, link)
            fsync_directory(link.parent)
        finally:
            temporary.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def desired(p: dict[str, Any], role: str) -> tuple[str, str]:
    return p["old_config" if role == "old" else "candidate_config"]["sha256"], p[
        "old_link_target"
    ] if role == "old" else p["new"]["release"]


def recognize(p: dict[str, Any], host: Host) -> tuple[str, str]:
    actual = host.pair(p)
    if actual[0] not in {desired(p, r)[0] for r in ("old", "new")} or actual[1] not in {
        desired(p, r)[1] for r in ("old", "new")
    }:
        raise Refusal("unrecognized-production-pair-refuse-overwrite")
    return actual


def save(directory: Path, state: dict[str, Any], host: Host, event: str) -> None:
    state["events"].append(
        {
            "event": event,
            "boot_id": host.boot_id(),
            "monotonic": host.now(),
            "utc_ns": time.time_ns(),
        }
    )
    q.write_json(directory / "state.json", state)


def state_read(directory: Path, checksum: str) -> dict[str, Any]:
    state = q.read_json(directory / "state.json")
    if (
        state.get("format") != FORMAT
        or state.get("schema_version") != 1
        or state["plan_sha256"] != checksum
        or state["operator_sha256"] != q.digest(Path(__file__).read_bytes())
        or state["host_helper_sha256"] != q.digest(Path(q.__file__).read_bytes())
    ):
        raise Refusal("cutover-state-source-or-plan-mismatch")
    return state


def current_state(directory: Path, checksum: str, host: Host) -> dict[str, Any]:
    state = state_read(directory, checksum)
    if state["active_boot_id"] != host.boot_id():
        raise Refusal("new-boot-requires-guard-recovery")
    return state


def set_pair(p, role, directory, state, host):
    recognize(p, host)
    target = desired(p, role)
    state["pair_write_intent"] = role
    save(directory, state, host, "pair-write-intent-" + role)
    host.replace_config(
        p, pin_bytes(p["old_config" if role == "old" else "candidate_config"])
    )
    save(directory, state, host, "config-written-" + role)
    recognize(p, host)
    host.replace_link(p, target[1])
    if host.pair(p) != target:
        raise Refusal("production-pair-write-not-confirmed")
    save(directory, state, host, "pair-written-" + role)


def enabled(p, host, role, value, deadline):
    unit = p[role]["unit"]
    if host.enablement(unit, deadline) != value:
        host.enabled(
            "enable" if value == "enabled" else "disable",
            unit,
            p["enablement_directory"],
            deadline,
        )
    else:
        host.confirm_enablement(unit, value, p["enablement_directory"], deadline)


def admit(directory, checksum, host):
    state = current_state(directory, checksum, host)
    if state["rollback_requested"] or host.now() >= state["rollback_by"]:
        raise Refusal("cutover-fenced-for-rollback")
    return state


def acceptance(p, directory, state, *, historical=False):
    path = directory / "acceptance-result.json"
    result = q.read_json(path)
    started = q.read_json(directory / "acceptance-started.json")
    challenge = q.read_json(directory / "acceptance-challenge.json")
    if (
        q.digest((directory / "acceptance-challenge.json").read_bytes())
        != state["acceptance_challenge_sha256"]
    ):
        raise Refusal("cutover-acceptance-challenge-changed")
    for value, kind in (
        (started, "deltrelserve.cloud-cutover-acceptance-started"),
        (result, "deltrelserve.cloud-cutover-acceptance"),
    ):
        if (
            value.get("format") != kind
            or value.get("schema_version") != 1
            or any(value.get(k) != challenge[k] for k in BINDINGS)
        ):
            raise Refusal("cutover-acceptance-binding-mismatch")
    if (
        result["invocation_id"] != started["invocation_id"]
        or re.fullmatch(r"[a-f0-9]{32}", result["invocation_id"]) is None
        or result["invocation_id"] == state["acceptance_prior_invocation"]
    ):
        raise Refusal("cutover-acceptance-invocation-invalid")
    first, last = result.get("started_monotonic"), result.get("completed_monotonic")
    if (
        not q._finite_number(first)
        or not q._finite_number(last)
        or first != started.get("started_monotonic")
        or not challenge["issued_monotonic"]
        <= first
        <= last
        <= challenge["deadline_monotonic"]
    ):
        raise Refusal("cutover-acceptance-time-invalid")
    if not historical and state["active_boot_id"] != challenge["boot_id"]:
        raise Refusal("acceptance-is-from-another-boot")
    if (
        set(result.get("checks", {})) != {"standard_1"}
        or result["checks"]["standard_1"].get("status") != "passed"
    ):
        raise Refusal("public-standard-acceptance-failed")
    evidence = result.get("evidence")
    hashes = set()
    total = 0
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
        raise Refusal("acceptance-evidence-inventory-invalid")
    seen = set()
    for row in evidence:
        relative = row["path"]
        path = directory / relative
        if (
            not isinstance(relative, str)
            or not relative.startswith("acceptance-evidence/")
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or relative in seen
            or path.resolve() != path
            or not path.is_file()
        ):
            raise Refusal("acceptance-evidence-path-invalid")
        seen.add(relative)
        size = row["bytes"]
        if type(size) is not int or size < 0 or path.stat().st_size != size:
            raise Refusal("acceptance-evidence-size-invalid")
        total += size
        if total > 4 * 2**20:
            raise Refusal("acceptance-evidence-too-large")
        actual = q.digest(path.read_bytes())
        if actual != row["sha256"]:
            raise Refusal("acceptance-evidence-hash-invalid")
        hashes.add(actual)
    if result["checks"]["standard_1"].get("evidence_sha256") not in hashes:
        raise Refusal("acceptance-check-unbound")
    return result


def committed(p, directory, state):
    path = directory / "accepted.json"
    if not path.exists():
        return False
    receipt = q.read_json(path)
    if (
        receipt.get("format") != FORMAT + ".accepted"
        or receipt.get("schema_version") != 1
        or receipt.get("boot_id") != state["original_boot_id"]
        or not q._finite_number(receipt.get("accepted_monotonic"))
        or not state["budget_started"]
        <= receipt["accepted_monotonic"]
        < state["original_rollback_by"]
        or receipt.get("service_invocation") not in state["service_invocations"]
        or receipt.get("plan_sha256") != state["plan_sha256"]
        or receipt.get("pair") != list(desired(p, "new"))
        or receipt.get("result_sha256")
        != q.digest((directory / "acceptance-result.json").read_bytes())
        or receipt.get("authenticated_and_public_health") is not True
    ):
        raise Refusal("cutover-commit-proof-invalid")
    result = acceptance(p, directory, state, historical=True)
    if result["completed_monotonic"] > receipt["accepted_monotonic"]:
        raise Refusal("acceptance-commit-predates-result")
    return True


def _known_service(p, state, host, deadline):
    unit = q._owned(p, "old", host, deadline)
    if (
        not state.get("boot_recovery")
        and not q._dead(unit)
        and unit["InvocationID"] not in state["service_invocations"]
        and not state.get("service_start_intent")
    ):
        raise Refusal("production-invocation-not-owned")
    return unit


def recover(p, checksum, directory, host):
    before = state_read(directory, checksum)
    if before["finished"]:
        return before
    # Fresh boot: no use of old process IDs or monotonic values for authorization.
    deadline = (
        before["deadline"]
        if before["active_boot_id"] == host.boot_id()
        else host.now() + q.TOTAL
    )
    with q.locked(directory, host, deadline):
        state = state_read(directory, checksum)
        if state["active_boot_id"] != host.boot_id():
            state.update(
                active_boot_id=host.boot_id(),
                deadline=host.now() + q.TOTAL,
                rollback_by=host.now(),
                boot_recovery=True,
                finished=False,
            )
            state["recovery_boots"].append(
                {
                    "boot_id": host.boot_id(),
                    "started_monotonic": host.now(),
                    "deadline": state["deadline"],
                }
            )
            save(directory, state, host, "fresh-boot-recovery-no-cutover-resumption")
        if state["finished"]:
            return state
        state["rollback_requested"] = True
        save(directory, state, host, "recovery-fenced")
        try:
            try:
                target = "new" if committed(p, directory, state) else "old"
            except Exception as error:
                # Invalid acceptance evidence refuses new publication; it must
                # never prevent recovery of an otherwise owned transaction.
                target = "old"
                state["invalid_acceptance_commit"] = type(error).__name__
                save(directory, state, host, "invalid-commit-rollback-old")
            recognize(p, host)
            # Stop/cancel only our fixed CPU acceptance unit; never trust probe
            # receipt bytes as authority to release GPU ownership.
            client = q._owned(p, "acceptance", host, state["deadline"])
            if state.get("acceptance_start_intent"):
                if (
                    state.get("acceptance_invocation")
                    and client["InvocationID"]
                    not in ("", state["acceptance_invocation"])
                    and not q._dead(client)
                ):
                    raise Refusal("acceptance-invocation-not-owned")
                host.action("stop", p["acceptance"]["unit"], state["deadline"])
            elif not q._dead(client):
                raise Refusal("unowned-active-acceptance-unit")
            service = _known_service(p, state, host, state["deadline"])
            already = (
                host.pair(p) == desired(p, target)
                and service["Job"] in ("", "0")
                and q._healthy(p, target, host, state["deadline"])
            )
            if not already:
                # Disable is persistent before any old/new partial pair can change.
                enabled(p, host, "old", "disabled", state["deadline"])
                host.action("stop", p["old"]["unit"], state["deadline"])
                cleanup = min(
                    host.now() + q.DRAIN + q.OBSERVATION_RESERVE,
                    state["deadline"] - q.EMPTY - q.RESTORE,
                )
                q._wait(
                    host,
                    lambda: all(
                        q._dead(q._owned(p, r, host, cleanup))
                        for r in ("old", "acceptance")
                    ),
                    cleanup,
                    "cutover-owned-cleanup-timeout",
                )
                gate = min(host.now() + q.EMPTY, state["deadline"] - q.RESTORE)
                q._wait(
                    host,
                    lambda: (
                        host.port_closed(8080, gate)
                        and host.port_closed(8081, gate)
                        and not host.owners(gate)
                    ),
                    gate,
                    "cutover-empty-owner-gate-blocked",
                )
                restore_end = min(host.now() + q.RESTORE, state["deadline"])
                host.check_files(p[target], restore_end)
                set_pair(p, target, directory, state, host)
                if host.owners(restore_end) or not q._dead(
                    q._owned(p, "old", host, restore_end)
                ):
                    raise Refusal("recovery-owner-changed-before-start")
                state["service_start_intent"] = target
                save(directory, state, host, "recovery-service-start-intent")
                host.action("start", p[target]["unit"], restore_end)
                q._wait(
                    host,
                    lambda: q._healthy(p, target, host, restore_end),
                    restore_end,
                    "recovery-health-timeout",
                )
            # The acceptance commit may have preceded a crash before re-enabling.
            # Its proof is verified before finalizing new; absent proof means old.
            enabled(p, host, "old", p["original_enablement"], state["deadline"])
            state["outcome"] = "committed-new" if target == "new" else "rolled-back-old"
            save(directory, state, host, "recovery-complete-" + target)
            enabled(p, host, "guard", "disabled", state["deadline"])
            state["finished"] = True
            save(directory, state, host, "persistent-guard-retired")
        except BaseException as error:
            state["failure"] = (
                str(error) if isinstance(error, Refusal) else type(error).__name__
            )
            save(directory, state, host, "recovery-blocked-no-ownership-waiver")
            raise
        return state


def execute(p, path, checksum, directory, host):
    end = host.now() + 180
    for role in ("old", "new", "guard", "acceptance"):
        host.check_files(p[role], end)
        q._owned(p, role, host, end)
    if (
        host.pair(p) != desired(p, "old")
        or host.enablement(p["old"]["unit"], end) != p["original_enablement"]
        or not q._healthy(p, "old", host, end)
        or not host.port_closed(8081, end)
    ):
        raise Refusal("cutover-initial-state-mismatch")
    initial = q._owned(p, "old", host, end)
    if (
        int(initial["MainPID"]) != p["initial_pid"]
        or initial["InvocationID"] != p["initial_invocation"]
        or initial["Job"] not in ("", "0")
    ):
        raise Refusal("cutover-initial-owner-changed")
    if (
        not all(q._dead(q._owned(p, r, host, end)) for r in ("guard", "acceptance"))
        or host.enablement(p["guard"]["unit"], end) != "disabled"
        or host.enablement(p["acceptance"]["unit"], end) not in ("disabled", "static")
    ):
        raise Refusal("cutover-auxiliary-not-idle-disabled")
    directory.mkdir(mode=0o700)
    fsync_directory(directory.parent)
    now = host.now()
    state = {
        "format": FORMAT,
        "schema_version": 1,
        "plan_sha256": checksum,
        "plan_path": str(path.resolve()),
        "operator_sha256": q.digest(Path(__file__).read_bytes()),
        "host_helper_sha256": q.digest(Path(q.__file__).read_bytes()),
        "original_boot_id": host.boot_id(),
        "active_boot_id": host.boot_id(),
        "controller": host.process(os.getpid()),
        "budget_started": now,
        "rollback_by": now + q.RESTORE_BY,
        "original_rollback_by": now + q.RESTORE_BY,
        "deadline": now + q.TOTAL,
        "rollback_requested": False,
        "finished": False,
        "service_invocations": [initial["InvocationID"]],
        "acceptance_prior_invocation": q._owned(p, "acceptance", host, end)[
            "InvocationID"
        ],
        "events": [],
        "recovery_boots": [],
    }
    save(directory, state, host, "prepared-before-outage")
    try:
        enabled(p, host, "guard", "enabled", now + 15)
        host.action("start", p["guard"]["unit"], now + 15)

        def armed():
            file = directory / "guard-armed.json"
            if not file.exists():
                return False
            proof = q.read_json(file)
            unit = q._owned(p, "guard", host, now + 15)
            return (
                proof.get("plan_sha256") == checksum
                and proof.get("boot_id") == state["active_boot_id"]
                and proof.get("invocation_id") == unit["InvocationID"]
                and unit["ActiveState"] == "active"
                and proof.get("process") == host.process(int(unit["MainPID"]))
                and host.enablement(p["guard"]["unit"], now + 15) == "enabled"
            )

        q._wait(host, armed, now + 15, "persistent-rollback-guard-not-armed")
        with q.locked(directory, host, state["rollback_by"]):
            state = admit(directory, checksum, host)
            current = q._owned(p, "old", host, state["rollback_by"])
            if current["InvocationID"] != initial["InvocationID"] or not q._healthy(
                p, "old", host, state["rollback_by"]
            ):
                raise Refusal("production-owner-changed-after-guard-arm")
            recognize(p, host)
            state["autostart_disable_intent"] = True
            save(directory, state, host, "production-autostart-disable-intent")
            enabled(p, host, "old", "disabled", state["rollback_by"])
            state["stop_deadline"] = min(
                host.now() + q.DRAIN + q.OBSERVATION_RESERVE, state["rollback_by"]
            )
            save(directory, state, host, "production-stop-intent")
            host.action("stop", p["old"]["unit"], state["stop_deadline"])
        q._wait(
            host,
            lambda: q._dead(q._owned(p, "old", host, state["stop_deadline"])),
            state["stop_deadline"],
            "old-production-drain-timeout",
        )
        gate = min(host.now() + q.EMPTY, state["rollback_by"])
        q._wait(
            host,
            lambda: (
                host.port_closed(8080, gate)
                and host.port_closed(8081, gate)
                and not host.owners(gate)
            ),
            gate,
            "cutover-empty-before-new-blocked",
        )
        with q.locked(directory, host, state["rollback_by"]):
            state = admit(directory, checksum, host)
            host.check_files(p["new"], state["rollback_by"])
            set_pair(p, "new", directory, state, host)
            if host.owners(state["rollback_by"]) or not q._dead(
                q._owned(p, "old", host, state["rollback_by"])
            ):
                raise Refusal("cutover-owner-changed-before-new")
            state["service_start_intent"] = "new"
            save(directory, state, host, "new-production-start-intent")
            host.action("start", p["new"]["unit"], state["rollback_by"])
        q._wait(
            host,
            lambda: q._healthy(p, "new", host, state["rollback_by"]),
            state["rollback_by"],
            "new-production-health-timeout",
        )
        with q.locked(directory, host, state["rollback_by"]):
            state = admit(directory, checksum, host)
            current = q._owned(p, "new", host, state["rollback_by"])
            state["service_invocations"].append(current["InvocationID"])
            state.pop("service_start_intent", None)
            challenge = {
                "format": "deltrelserve.cloud-cutover-challenge",
                "schema_version": 1,
                "run_id": p["run_id"],
                "plan_sha256": checksum,
                "boot_id": host.boot_id(),
                "challenge": secrets.token_hex(32),
                "issued_monotonic": host.now(),
                "deadline_monotonic": min(host.now() + 200, state["rollback_by"]),
                "acceptance_unit": p["acceptance"]["unit"],
                "model_identity": p["new"]["model_identity"],
                "model_step": 572377,
            }
            q.write_json(
                directory / "acceptance-challenge.json", challenge, replace=False
            )
            state["acceptance_challenge_sha256"] = q.digest(
                (directory / "acceptance-challenge.json").read_bytes()
            )
            state["acceptance_start_intent"] = True
            save(directory, state, host, "public-standard-acceptance-start-intent")
            host.action("start", p["acceptance"]["unit"], state["rollback_by"])

        def accepted():
            current = q._owned(p, "acceptance", host, challenge["deadline_monotonic"])
            if (
                current["InvocationID"]
                and current["InvocationID"] != state["acceptance_prior_invocation"]
            ):
                with q.locked(directory, host, state["rollback_by"]):
                    fresh = admit(directory, checksum, host)
                    if fresh.get("acceptance_invocation") not in (
                        None,
                        current["InvocationID"],
                    ):
                        raise Refusal("acceptance-owner-changed")
                    fresh["acceptance_invocation"] = current["InvocationID"]
                    save(directory, fresh, host, "acceptance-owner-observed")
            if not (directory / "acceptance-result.json").exists():
                return False
            result = acceptance(p, directory, state)
            if result["completed_monotonic"] > host.now() or current[
                "InvocationID"
            ] not in ("", result["invocation_id"]):
                raise Refusal("acceptance-owner-or-clock-mismatch")
            return q._dead(current)

        q._wait(
            host,
            accepted,
            challenge["deadline_monotonic"],
            "public-standard-acceptance-deadline",
        )
        with q.locked(directory, host, state["rollback_by"]):
            state = admit(directory, checksum, host)
            if host.pair(p) != desired(p, "new") or not q._healthy(
                p, "new", host, state["rollback_by"]
            ):
                raise Refusal("new-state-changed-before-commit")
            acceptance(p, directory, state)
            q.write_json(
                directory / "accepted.json",
                {
                    "format": FORMAT + ".accepted",
                    "schema_version": 1,
                    "plan_sha256": checksum,
                    "pair": list(desired(p, "new")),
                    "result_sha256": q.digest(
                        (directory / "acceptance-result.json").read_bytes()
                    ),
                    "authenticated_and_public_health": True,
                    "service_invocation": q._owned(
                        p, "new", host, state["rollback_by"]
                    )["InvocationID"],
                    "boot_id": host.boot_id(),
                    "accepted_monotonic": host.now(),
                },
                replace=False,
            )
            save(directory, state, host, "new-acceptance-committed-before-enablement")
    except BaseException as error:
        with q.locked(directory, host, state["deadline"]):
            state = state_read(directory, checksum)
            state["cutover_failure"] = (
                str(error) if isinstance(error, Refusal) else type(error).__name__
            )
            save(directory, state, host, "cutover-interrupted")
        raise
    finally:
        recover(p, checksum, directory, host)
    return state_read(directory, checksum)


def guard(p, checksum, directory, host):
    state = state_read(directory, checksum)
    end = (
        state["deadline"]
        if state["active_boot_id"] == host.boot_id()
        else host.now() + q.TOTAL
    )
    unit = q._owned(p, "guard", host, end)
    if (
        int(unit["MainPID"]) != os.getpid()
        or host.enablement(p["guard"]["unit"], end) != "enabled"
    ):
        raise Refusal("persistent-guard-not-owned-enabled")
    host.confirm_enablement(
        p["guard"]["unit"], "enabled", p["enablement_directory"], end
    )
    q.write_json(
        directory / "guard-armed.json",
        {
            "plan_sha256": checksum,
            "boot_id": host.boot_id(),
            "invocation_id": unit["InvocationID"],
            "process": host.process(os.getpid()),
        },
    )
    while (
        state["active_boot_id"] == host.boot_id() and host.now() < state["rollback_by"]
    ):
        state = state_read(directory, checksum)
        if state["finished"]:
            return state
        if (
            state["rollback_requested"]
            or (directory / "accepted.json").exists()
            or host.process(state["controller"]["pid"]) != state["controller"]
        ):
            break
        host.sleep(min(0.5, state["rollback_by"] - host.now()))
    return recover(p, checksum, directory, host)


def guard_checksum(directory: Path, plan_path: Path) -> str:
    file = directory / "state.json"
    if (
        directory.is_symlink()
        or directory.resolve(strict=True) != directory
        or stat.S_IMODE(directory.stat().st_mode) != 0o700
        or directory.stat().st_uid != os.geteuid()
        or file.is_symlink()
        or not file.is_file()
        or stat.S_IMODE(file.stat().st_mode) != 0o600
        or file.stat().st_uid != os.geteuid()
    ):
        raise Refusal("guard-bootstrap-state-not-protected")
    prepared = q.read_json(file)
    checksum = prepared.get("plan_sha256")
    q._sha(checksum)
    state = state_read(directory, checksum)
    if state.get("plan_path") != str(plan_path.resolve(strict=True)):
        raise Refusal("guard-bootstrap-plan-location-mismatch")
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "recover", "guard"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    try:
        if os.geteuid() != 0:
            raise Refusal("root-cutover-operator-required")
        directory = args.state.absolute()
        checksum = args.plan_sha256
        if checksum is None:
            if args.mode != "guard":
                raise Refusal("explicit-reviewed-plan-checksum-required")
            checksum = guard_checksum(directory, args.plan)
        p = load_plan(args.plan.resolve(strict=True), checksum)
        if (
            str(directory) != p["state_directory"]
            or directory.is_symlink()
            or directory.parent.resolve() != directory.parent
        ):
            raise Refusal("cutover-state-location-mismatch")
        host = Host()
        result = (
            execute(p, args.plan, checksum, directory, host)
            if args.mode == "run"
            else guard(p, checksum, directory, host)
            if args.mode == "guard"
            else recover(p, checksum, directory, host)
        )
        print(
            json.dumps(
                {
                    "outcome": result.get("outcome"),
                    "finished": result["finished"],
                    "state": str(directory / "state.json"),
                }
            )
        )
    except BaseException as error:
        print(
            json.dumps(
                {
                    "failure": str(error)
                    if isinstance(error, Refusal)
                    else type(error).__name__
                }
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
