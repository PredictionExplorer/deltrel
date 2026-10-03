"""Owned support-file transactions; no runtime/GPU launch or implicit observation repair.

The Linux adapter supplies bounded I/O and actual unit/cgroup/lease observations.
An explicit pre-admission recovery can only disable owned support boot links and
finish already-authorized stopped file writes/reload. Service control waits for
full guardian admission. Environment contents never enter the journal.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from deltreltrain import strength_freshness_guard as core

FORMAT = "deltreltrain.strength-freshness-support-transaction"
DIRECTORY = "linux-support-transitions"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _names(host) -> list[str]:
    before, after = (host.plan["support_transition"][s] for s in ("before", "after"))
    core.require(
        set(before) == set(after) and 0 < len(before) <= 32, "support-inventory"
    )
    return sorted(before)


def _known_links(host, name: str, deadline: float) -> dict[str, str]:
    spec = host.manifest["units"][name]
    expected = spec["boot_links"]
    core.require(
        isinstance(expected, dict) and len(expected) <= 16, "support-boot-link-policy"
    )
    for path, target in expected.items():
        p = Path(path)
        core.require(
            p.parent.parent == Path("/etc/systemd/system")
            and p.name == name
            and p.parent.name.endswith((".target.wants", ".target.requires"))
            and target == spec["installed_path"],
            "support-persistent-boot-link",
        )
    actual = host.io.boot_links(name, deadline)
    core.require(
        isinstance(actual, dict)
        and set(actual) <= set(expected)
        and all(actual[p] == expected[p] for p in actual),
        "support-unknown-boot-link",
    )
    return actual


def verify_boot_edges(host, name: str, enabled: bool, deadline: float) -> None:
    """Require exact persistent links reachable from the actual default target."""
    core.require(type(enabled) is bool, "support-enablement-type")
    actual = _known_links(host, name, deadline)
    declared = host.manifest["units"][name]["boot_links"]
    core.require(
        actual == (declared if enabled else {}) and (not enabled or bool(declared)),
        "support-boot-enablement",
    )
    topology = host.manifest["boot_topology"]
    default = host.io.command(["systemctl", "get-default"], deadline).strip()
    core.require(default == topology["default_target"], "support-default-target")
    for path in declared:
        target = Path(path).parent.name.rsplit(".", 1)[0]
        chain = topology["target_paths"].get(target)
        core.require(
            isinstance(chain, list)
            and 1 <= len(chain) <= 16
            and chain[0] == default
            and chain[-1] == target
            and len(chain) == len(set(chain))
            and all(
                isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.@-]+\.target", x)
                for x in chain
            ),
            "support-boot-reachability-policy",
        )
        for parent, child in zip(chain, chain[1:]):
            raw = host.io.command(
                [
                    "systemctl",
                    "show",
                    parent,
                    "--all",
                    "--no-pager",
                    "--property=Wants,Requires",
                ],
                deadline,
            )
            rows = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
            core.require(
                set(rows) <= {"Wants", "Requires"}
                and child
                in (rows.get("Wants", "") + " " + rows.get("Requires", "")).split(),
                "support-boot-edge-missing",
            )


def _units(host, deadline: float, *, partial: bool = False):
    jobs = host._jobs(deadline)
    return {
        name: host._unit(name, jobs, deadline, allow_known_partial=partial)[0]
        for name in _names(host)
    }


def require_stopped_support(host, deadline: float, *, partial: bool = False) -> None:
    for name, unit in _units(host, deadline, partial=partial).items():
        core.require(unit.dead and not unit.enabled, "support-not-drained")
        verify_boot_edges(host, name, False, deadline)


def _binding(
    host,
    deadline: float,
    intent: dict[str, Any] | None = None,
    *,
    pre_admission: bool = False,
):
    core.require(host.execute and host.lease_fd is not None, "support-lease-required")
    now = host.clock()
    state = host.io.json(host.state / "state.json", deadline)
    anchor_bytes = host.io.read(host.state / "anchor.json", 2**20, deadline)
    anchor = json.loads(anchor_bytes)
    core.require(
        _sha(anchor_bytes) == state["anchor_sha256"]
        and all(
            state[k] == anchor[k]
            for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
        )
        and state["plan_sha256"] == core.digest(host.plan)
        and not state.get("finished")
        and not state.get("runtime_authority_retired")
        and now.wall_ns >= state["last_wall_ns"]
        and now.wall_ns < state["deadline_wall_ns"],
        "support-original-authority",
    )
    lease = host._lease(deadline)
    process = host.io.process(os.getpid())
    guard, _ = host._unit(
        host.plan["units"]["guard"]["name"], host._jobs(deadline), deadline
    )
    core.require(process is not None, "support-guardian-process")
    current = core.Process(process["pid"], process["start_ticks"])
    core.require(
        lease is not None
        and lease.get("owner") == asdict(current)
        and lease.get("boot_id") == now.boot_id
        and lease.get("nonce") == state["nonce"]
        and lease.get("attempt_id") == state["attempt_id"]
        and lease.get("plan_sha256") == state["plan_sha256"]
        and guard.main == current
        and current in guard.members
        and guard.active == "active"
        and guard.job is None
        and guard.enabled
        and guard.invocation_id
        == lease.get("invocation_id")
        == os.environ.get("INVOCATION_ID"),
        "support-actual-guardian-lease",
    )
    verify_boot_edges(host, guard.name, True, deadline)
    owner = state["owners"].get("guard")
    core.require(owner is not None, "support-no-admitted-predecessor")
    exact = owner == {
        "main": asdict(current),
        "invocation_id": guard.invocation_id,
        "boot_id": now.boot_id,
    }
    if not exact:
        core.require(
            pre_admission and guard.invocation_id != owner["invocation_id"],
            "support-guardian-not-admitted",
        )
        previous = core.Process(**owner["main"])
        observed = host.io.process(previous.pid)
        core.require(
            owner["boot_id"] != now.boot_id
            or observed is None
            or (
                observed["pid"] == previous.pid
                and observed["start_ticks"] != previous.start_ticks
            ),
            "support-old-guardian-alive",
        )
    if intent is not None:
        core.require(
            intent["format"] == FORMAT
            and intent["schema_version"] == 1
            and intent["helper_sha256"] == _sha(Path(__file__).read_bytes())
            and intent["manifest_sha256"] == host.manifest_sha256
            and all(
                intent[k] == state[k]
                for k in ("plan_sha256", "attempt_id", "nonce", "phase")
            )
            and intent["action"] == state["last_action"],
            "support-intent-binding",
        )
        core.require(
            now.wall_ns < intent["deadline_wall_ns"], "support-original-deadline"
        )
        remaining = (intent["deadline_wall_ns"] - now.wall_ns) / 1e9
        deadline = min(deadline, now.monotonic + remaining)
        if now.boot_id == intent["boot_id"]:
            deadline = min(deadline, intent["deadline_monotonic"])
    if now.boot_id == state["active_boot_id"]:
        deadline = min(deadline, state["deadline"])
    core.require(now.monotonic < deadline, "support-original-deadline")
    return state, now, deadline


def _files(host, stage: str) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for name in _names(host):
        spec = host.manifest["units"][name]
        core.require(
            spec["installed_path"] == "/etc/systemd/system/" + name, "support-unit-path"
        )
        for side in ("before", "after"):
            entries = [(spec["installed_path"], spec[side]["unit"], 0o644)]
            entries += [
                (
                    e["path"],
                    {
                        "path": e["source_path"],
                        "sha256": e["sha256"],
                        "bytes": e["bytes"],
                    },
                    e.get("mode", 0o600),
                )
                for e in spec[side]["environment_files"]
            ]
            for target, pin, mode in entries:
                path = Path(target)
                core.require(
                    path.is_absolute()
                    and str(path) == target
                    and ".." not in path.parts
                    and not path.is_relative_to(host.state)
                    and not path.is_relative_to(host.root)
                    and type(mode) is int
                    and mode in (0o600, 0o640, 0o644),
                    "support-file-policy",
                )
                item = found.setdefault(
                    target,
                    {
                        "path": target,
                        "known": [],
                        "desired": None,
                        "owners": [],
                        "sides": [],
                    },
                )
                if side not in item["sides"]:
                    item["sides"].append(side)
                value = {"source": dict(pin), "mode": mode}
                if value not in item["known"]:
                    item["known"].append(value)
                if name not in item["owners"]:
                    item["owners"].append(name)
                if side == stage:
                    core.require(
                        item["desired"] in (None, value), "support-conflicting-target"
                    )
                    item["desired"] = value
    return [found[name] for name in sorted(found)]


def _read_current(host, item, deadline):
    try:
        data = host.io.read(Path(item["path"]), 2**20, deadline)
    except FileNotFoundError:
        core.require(len(item["sides"]) == 1, "support-missing-preserved-file")
        return None, None
    metadata = host.io.file_metadata(Path(item["path"]), deadline)
    core.require(
        metadata["uid"] == os.geteuid()
        and metadata["gid"] == os.getegid()
        and metadata["mode"] in {k["mode"] for k in item["known"]},
        "support-unowned-target-mode",
    )
    core.require(
        any(
            _sha(data) == k["source"]["sha256"] and len(data) == k["source"]["bytes"]
            for k in item["known"]
        ),
        "support-unowned-target-bytes",
    )
    return data, metadata


def _write_files(host, intent, deadline, *, pre_admission):
    require_stopped_support(host, deadline, partial=True)
    for item in intent["files"]:
        _, _, deadline = _binding(host, deadline, intent, pre_admission=pre_admission)
        current = _read_current(host, item, deadline)
        desired = item["desired"]
        if desired is None:
            continue
        pin = desired["source"]
        host.io.pin(pin, deadline)
        data = host.io.read(Path(pin["path"]), 2**20, deadline)
        core.require(
            _sha(data) == pin["sha256"] and len(data) == pin["bytes"],
            "support-staged-bytes-raced",
        )
        jobs = host._jobs(deadline)
        core.require(
            all(
                host._unit(name, jobs, deadline, allow_known_partial=True)[0].dead
                for name in item["owners"]
            ),
            "support-writer-not-drained",
        )
        core.require(
            _read_current(host, item, deadline) == current, "support-target-raced"
        )
        if (
            current[0] != data
            or current[1] is None
            or current[1]["mode"] != desired["mode"]
        ):
            host.io.atomic(
                Path(item["path"]),
                data,
                overwrite=current[0] is not None,
                mode=desired["mode"],
            )
        final, metadata = _read_current(host, item, deadline)
        core.require(
            final == data
            and metadata is not None
            and metadata["mode"] == desired["mode"],
            "support-final-file-mode",
        )
    host.io.command(["systemctl", "daemon-reload"], deadline)
    for name in _names(host):
        unit, metadata = host._unit(name, host._jobs(deadline), deadline)
        expected = host.plan["support_transition"][intent["stage"]][name]
        core.require(
            unit.dead
            and all(
                metadata[k] == expected[k]
                for k in ("definition_sha256", "environment_sha256")
            ),
            "support-installed-stage-differs",
        )


def _finish(
    host, path: Path, intent, deadline, *, pre_admission: bool
) -> dict[str, Any]:
    _, _, deadline = _binding(host, deadline, intent, pre_admission=pre_admission)
    core.require(
        intent["files"] == _files(host, intent["stage"]), "support-intent-file-scope"
    )
    authority = host._authority(deadline)
    role = "r4" if intent["stage"] == "after" else "r3"
    core.require(
        authority.phase == role
        and authority.plan_sha256 == host.plan["freshness_plan_sha256"],
        "support-runtime-authority",
    )
    for item in intent["files"]:
        _read_current(host, item, deadline)
    for name in _names(host):
        _known_links(host, name, deadline)
        host._enable(name, False, deadline)  # No --now; never controls a process here.
        verify_boot_edges(host, name, False, deadline)
    if pre_admission:
        try:
            _units(
                host, deadline
            )  # Coherent metadata already permits full core admission.
            return {"status": "awaiting-guardian-admission", "intent": str(path)}
        except core.Refusal:
            _write_files(host, intent, deadline, pre_admission=True)
            return {
                "status": "files-ready-awaiting-guardian-admission",
                "intent": str(path),
            }
    for name in _names(host):
        host._unit(name, host._jobs(deadline), deadline, allow_known_partial=True)
        host.io.action(
            "stop", name, deadline
        )  # Also cancels queued starts on inactive units.
    while not all(u.dead for u in _units(host, deadline, partial=True).values()):
        _binding(host, deadline, intent)
        core.require(host.io.now() + 0.2 < deadline, "support-drain-deadline")
        host.io.sleep(0.2)
    _write_files(host, intent, deadline, pre_admission=False)
    for name, expected in host.plan["support_transition"][intent["stage"]].items():
        _binding(host, deadline, intent)
        host._enable(name, expected["enabled"], deadline)
        verify_boot_edges(host, name, expected["enabled"], deadline)
    for role in ("r3", "r4", "probe"):
        spec = host.plan["units"][role]
        enabled = spec[
            "committed_enabled" if intent["stage"] == "after" else "initial_enabled"
        ]
        host._enable(spec["name"], enabled, deadline)
        verify_boot_edges(host, spec["name"], enabled, deadline)
    for name, expected in host.plan["support_transition"][intent["stage"]].items():
        if expected["enabled"]:
            _binding(host, deadline, intent)
            host.io.action("start", name, deadline)
    while True:
        ready = True
        for name, expected in host.plan["support_transition"][intent["stage"]].items():
            if not expected["enabled"]:
                continue
            unit = host._unit(name, host._jobs(deadline), deadline)[0]
            if unit.job is not None:
                ready = False
                continue
            core.require(
                unit.result in ("", "success") and unit.exit_code == 0,
                "support-start-failed",
            )
            oneshot = (
                host.manifest["units"][name][intent["stage"]]["properties"].get("Type")
                == "oneshot"
            )
            ready &= unit.active == "active" or (oneshot and unit.dead)
        if ready:
            break
        _binding(host, deadline, intent)
        core.require(host.io.now() + 0.2 < deadline, "support-start-deadline")
        host.io.sleep(0.2)
    result = {
        "format": FORMAT + "-complete",
        "intent_sha256": _sha(host.io.read(path, 2**20, deadline)),
        "stage": intent["stage"],
        "clock": asdict(host.clock()),
        "action": intent["action"],
    }
    host.io.atomic(path.with_suffix(".complete.json"), _bytes(result))
    return result


def recover_transition(
    host, deadline: float, *, pre_admission: bool = False
) -> dict[str, Any] | None:
    folder = host.state / DIRECTORY
    core.require(folder.resolve() == folder, "support-journal-path")
    if not folder.exists():
        return None
    paths = sorted(folder.glob("*.intent.json"))
    core.require(len(paths) <= 32, "support-transaction-budget")
    pending = []
    completed = []
    for path in paths:
        complete = path.with_suffix(".complete.json")
        intent = host.io.json(path, deadline)
        core.require(
            intent.get("format") == FORMAT
            and type(intent.get("schema_version")) is int
            and intent["schema_version"] == 1
            and intent.get("manifest_sha256") == host.manifest_sha256
            and intent.get("plan_sha256") == core.digest(host.plan),
            "support-journal-binding",
        )
        if complete.exists():
            receipt = host.io.json(complete, deadline)
            core.require(
                receipt.get("format") == FORMAT + "-complete"
                and receipt.get("intent_sha256")
                == _sha(host.io.read(path, 2**20, deadline))
                and receipt.get("stage") == intent["stage"],
                "support-completion-binding",
            )
            completed.append((intent, receipt))
        else:
            pending.append((path, intent))
    core.require(len(pending) <= 1, "support-overlapping-transactions")
    if pending:
        path, intent = pending[0]
        return _finish(host, path, intent, deadline, pre_admission=pre_admission)
    state = host.io.json(host.state / "state.json", deadline)
    if not state.get("finished") and not state.get("runtime_authority_retired"):
        matching = [
            (i, r)
            for i, r in completed
            if i["action"] == state.get("last_action")
            and i["phase"] == state.get("phase")
        ]
        core.require(len(matching) <= 1, "support-completion-ambiguity")
        if matching:
            _binding(host, deadline, matching[0][0], pre_admission=pre_admission)
            return matching[0][1]
    return None


def transition(host, stage: str, deadline: float) -> dict[str, Any]:
    core.require(stage in {"before", "after"}, "support-stage")
    state, now, deadline = _binding(host, deadline)
    action = state["last_action"]
    expected_role = "r4" if stage == "after" else "r3"
    core.require(
        action["kind"]
        in {
            "restore-support-and-backup-" + expected_role,
            "restore-support-without-proof-" + expected_role,
            *(["restore-support-before-stop"] if stage == "before" else []),
        }
        and action["deadline"] == deadline,
        "support-action-intent",
    )
    recover_transition(host, deadline)
    files = _files(host, stage)
    for item in files:
        _read_current(host, item, deadline)
        for value in item["known"]:
            host.io.pin(value["source"], deadline)
    folder = host.state / DIRECTORY
    folder.mkdir(mode=0o700, exist_ok=True)
    core.require(folder.resolve() == folder, "support-journal-path")
    descriptor = os.open(host.state, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    identity = core.digest(
        {"action": action, "attempt_id": state["attempt_id"], "nonce": state["nonce"]}
    )
    path = folder / (identity + ".intent.json")
    if path.exists():
        complete = path.with_suffix(".complete.json")
        core.require(complete.exists(), "support-incomplete-action")
        return host.io.json(complete, deadline)
    intent = {
        "format": FORMAT,
        "schema_version": 1,
        "stage": stage,
        "action": action,
        "helper_sha256": _sha(Path(__file__).read_bytes()),
        "manifest_sha256": host.manifest_sha256,
        **{k: state[k] for k in ("plan_sha256", "attempt_id", "nonce", "phase")},
        "boot_id": now.boot_id,
        "deadline_monotonic": deadline,
        "deadline_wall_ns": min(
            state["deadline_wall_ns"],
            now.wall_ns + int((deadline - now.monotonic) * 1e9),
        ),
        "files": files,
    }
    host.io.atomic(path, _bytes(intent))
    return _finish(host, path, intent, deadline, pre_admission=False)
