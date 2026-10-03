from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
import runpy
import subprocess
import sys

import pytest

from scripts import strength_freshness_cpu_capture_request as m
from scripts import strength_freshness_cpu_lifecycle as life


fixture_definitions = runpy.run_path(
    str(Path(__file__).with_name("test_strength_freshness_cpu_preservation.py"))
)


@pytest.fixture
def observations():
    return fixture_definitions["observations"].__wrapped__()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def pinned(path, raw):
    return {"path": str(path), "sha256": m.digest(raw), "bytes": len(raw)}


def request_value(phase="before"):
    return {
        "format": m.FORMAT,
        "schema_version": 2,
        "phase": phase,
        "registration_sha256": "b" * 64,
        "nonce": "a" * 32,
        "source_pins": [{"path": "/input/control.py", "sha256": "c" * 64, "bytes": 12}],
        "limits": {
            "started": {
                "boot_id": "boot",
                "monotonic_ns": 100_000_000_000,
                "wall_ns": 1_800_000_000_000_000_000,
            },
            "deadline_monotonic_ns": 110_000_000_000,
            "deadline_wall_ns": 1_800_000_010_000_000_000,
            "metadata_bytes": m.METADATA_BUDGET,
            "runtime_bytes": m.RUNTIME_BUDGET,
        },
    }


def test_before_is_plan_free_and_integer_clocks_survive():
    value = request_value()
    result = m.request(encoded(value))
    assert result == value
    assert result["limits"]["started"]["wall_ns"] == 1_800_000_000_000_000_000
    now = {
        "boot_id": "boot",
        "monotonic_ns": 101_000_000_000,
        "wall_ns": 1_800_000_001_000_000_000,
    }
    assert m.effective_deadline(result, now) == 110


@pytest.mark.parametrize(
    "change",
    [
        "old-version",
        "before-plan",
        "phase",
        "nonce",
        "float-ns",
        "renew-time",
        "bytes",
        "source-alias",
    ],
)
def test_request_refuses_unfrozen_authority(change):
    value = request_value()
    if change == "old-version":
        value["schema_version"] = 1
    if change == "before-plan":
        value["plan_sha256"] = "a" * 64
    if change == "phase":
        value["phase"] = "activate"
    if change == "nonce":
        value["nonce"] = "a" * 31
    if change == "float-ns":
        value["limits"]["started"]["wall_ns"] = 1.8e18
    if change == "renew-time":
        value["limits"]["deadline_monotonic_ns"] += 200_000_000_000
    if change == "bytes":
        value["limits"]["runtime_bytes"] += 1
    if change == "source-alias":
        value["source_pins"] *= 2
    with pytest.raises(m.RequestRefusal):
        m.request(encoded(value))


@pytest.mark.parametrize("axis", ["boot", "wall", "monotonic"])
def test_original_deadline_and_boot_cannot_change(axis):
    value = request_value()
    now = copy.deepcopy(value["limits"]["started"])
    if axis == "boot":
        now["boot_id"] = "new-boot"
    else:
        now[axis + "_ns"] = value["limits"]["deadline_" + axis + "_ns"]
    with pytest.raises(m.RequestRefusal):
        m.effective_deadline(value, now)


def test_real_reader_requires_protected_pinned_file(tmp_path):
    p = tmp_path / "input.json"
    raw = b'{"safe":true}\n'
    p.write_bytes(raw)
    p.chmod(0o444)
    reader = m.PinnedReader(deadline=time.monotonic() + 5, owner_uid=os.geteuid())
    assert reader.read(pinned(p, raw)) == raw
    assert reader.consumed == len(raw)
    p.chmod(0o644)
    with pytest.raises(m.RequestRefusal, match="input-protection-or-size"):
        reader.read(pinned(p, raw))


def test_reader_does_not_follow_symlink_or_wrong_digest(tmp_path):
    p = tmp_path / "input.json"
    p.write_bytes(b"{}\n")
    p.chmod(0o444)
    alias = tmp_path / "alias.json"
    alias.symlink_to(p)
    reader = m.PinnedReader(deadline=time.monotonic() + 5, owner_uid=os.geteuid())
    with pytest.raises(m.RequestRefusal, match="input-symlink"):
        reader.read(pinned(alias, b"{}\n"))
    with pytest.raises(m.RequestRefusal, match="input-hash"):
        reader.read({**pinned(p, b"{}\n"), "sha256": "0" * 64})


def test_fifo_metadata_pin_refuses_without_waiting_for_writer(tmp_path):
    target = tmp_path / "request.json"
    os.mkfifo(target, 0o444)
    code = """
import os,sys,time
sys.path.insert(0,sys.argv[1])
from scripts.strength_freshness_cpu_capture_request import PinnedReader,RequestRefusal
r=PinnedReader(deadline=time.monotonic()+1,owner_uid=os.geteuid())
try:
 r.read({"path":sys.argv[2],"sha256":"a"*64,"bytes":1})
except RequestRefusal as e:
 assert str(e)=="input-protection-or-size"
else:
 raise AssertionError("FIFO was admitted")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-E",
            "-B",
            "-c",
            code,
            str(Path(__file__).resolve().parents[1]),
            str(target),
        ],
        capture_output=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_reader_single_budget_and_deadline(tmp_path):
    now = [100.0]
    reader = m.PinnedReader(
        deadline=102, owner_uid=os.geteuid(), monotonic=lambda: now[0]
    )
    reader.charge(m.METADATA_BUDGET)
    with pytest.raises(m.RequestRefusal, match="metadata-budget"):
        reader.charge(1)
    now[0] = 102
    with pytest.raises(m.RequestRefusal, match="metadata-deadline"):
        reader.check()
    with pytest.raises(m.RequestRefusal, match="reader-deadline"):
        m.PinnedReader(deadline=float("inf"), owner_uid=os.geteuid())


def test_readonly_log_never_creates_missing_directory(tmp_path):
    root = tmp_path / "missing"
    anchor = life.Anchor(
        "test", "a" * 32, "b" * 64, "boot", 100, 1_800_000_000_000_000_000
    )
    reader = m.PinnedReader(deadline=time.monotonic() + 5, owner_uid=os.geteuid())
    with pytest.raises(FileNotFoundError):
        m.ReadOnlyLog(reader, root, anchor)
    assert not root.exists()


def test_readonly_log_rejects_late_duplicate_receipt(tmp_path):
    root = tmp_path / "proof"
    root.mkdir(mode=0o700)
    anchor = life.Anchor(
        "test", "a" * 32, "b" * 64, "boot", 100, 1_800_000_000_000_000_000
    )
    writer = life.RawLog(root, anchor)
    entry = writer.record_once("cleanup-complete", {"test": 1})
    reader = m.PinnedReader(deadline=time.monotonic() + 5, owner_uid=os.geteuid())
    log = m.ReadOnlyLog(reader, root, anchor)
    assert log.load(entry, "cleanup-complete") == {"test": 1}
    writer.record("cleanup-complete", {"test": 2})
    with pytest.raises(m.RequestRefusal, match="proof-singleton-raced"):
        log.recheck()


def test_static_projection_keeps_ignore_errors_unknown_fields_and_order():
    value: dict[str, str] = dict.fromkeys(m.DUMMY_STATIC, "")
    value.update(
        ExecStart="{ path=/bin/sleep ; argv[]=/bin/sleep 1 ; ignore_errors=no ; pid=12 ; code=exited ; status=0 ; extension=keep }",
        Before="z.service a.service",
    )
    static = m.dummy_static(value)
    assert static["Before"] == "a.service z.service"
    assert static["ExecStart"].endswith("ignore_errors=no ; extension=keep")
    assert "pid=12" not in static["ExecStart"]
    changed = {
        **value,
        "ExecStart": value["ExecStart"].replace(
            "ignore_errors=no", "ignore_errors=yes"
        ),
    }
    assert m.dummy_static(changed) != static
    del value["EnvironmentFiles"]
    with pytest.raises(m.RequestRefusal, match="unqualified-dummy-empty-property"):
        m.dummy_static(value)
    assert m.dummy_static(value, allow_empty_environment_files=True) == static


@pytest.mark.parametrize("registered", [False, True])
def test_dummy_empty_exec_start_pre_requires_explicit_permission(registered):
    expected = dict.fromkeys(m.DUMMY_STATIC, "")
    raw = {k: v for k, v in expected.items() if k != "ExecStartPre"}
    if registered:
        assert m.dummy_static(raw, allow_empty_exec_start_pre=True) == expected
        expected["ExecStartPre"] = "/qualified/gate"
        assert m.dummy_static(raw, allow_empty_exec_start_pre=True) != expected
    else:
        with pytest.raises(m.RequestRefusal, match="unqualified-dummy-empty-property"):
            m.dummy_static(raw)


@pytest.mark.parametrize(
    "field", sorted(m.DUMMY_STATIC - {"EnvironmentFiles", "ExecStartPre"})
)
def test_dummy_projection_never_defaults_other_static_fields(field):
    raw = dict.fromkeys(m.DUMMY_STATIC, "")
    del raw[field]
    with pytest.raises(m.RequestRefusal, match="dummy-static-fields"):
        m.dummy_static(
            raw, allow_empty_environment_files=True, allow_empty_exec_start_pre=True
        )


@pytest.fixture
def admitted(observations, request):
    policy, before, _, champions = copy.deepcopy(observations)
    now_ns = before["clock"]["wall_ns"]
    root = Path("/input")
    nonce = "a" * 32
    scratch = Path("/run") / ("edgeconnect-cpuqual-" + nonce)
    files = {}
    sources = []
    for module in (m, m.records, m.supports, m.lifecycle, m.preservation):
        assert isinstance(module.__file__, str)
        path = Path(module.__file__).resolve()
        raw = path.read_bytes()
        sources.append(pinned(path, raw))
        files[str(path)] = raw
    sources.sort(key=lambda x: x["path"])
    registration = {
        "policy": policy,
        "birth_reference": {"boot_id": "boot"},
        "scope": {
            "files": {str(i): {"path": x["path"]} for i, x in enumerate(sources)}
        },
        "source_pins": {
            str(i): {k: x[k] for k in ("sha256", "bytes")}
            for i, x in enumerate(sources)
        },
        "encoding_contract_sha256": "e" * 64,
        "counter_scopes": {"worker": "actual"},
        "empty_property_rules": {
            "EnvironmentFiles": "systemd-255-empty-EnvironmentFiles"
        },
    }

    if getattr(request, "param", None) is not None:
        registration["empty_property_rules"]["ExecStartPre"] = request.param

    def add(path, value):
        raw = value if isinstance(value, bytes) else encoded(value)
        files[str(path)] = raw
        return pinned(path, raw)

    before_pin = add(root / "before.json", before)
    policy_pin = add(root / "policy.json", policy)
    b = request_value()
    b["source_pins"] = sources
    b["registration_sha256"] = m.digest(encoded(registration))
    b["limits"]["started"].update(monotonic_ns=99_000_000_000, wall_ns=now_ns - 10**9)
    bp = add(root / "before-request.json", b)
    rc = {"boot_id": "boot", "monotonic_ns": 100_000_000_000, "wall_ns": now_ns}
    br = {
        "format": m.RECEIPT,
        "schema_version": 2,
        "status": "complete",
        "request_sha256": bp["sha256"],
        "registration_sha256": b["registration_sha256"],
        "encoding_contract_sha256": "e" * 64,
        "source_pins": sources,
        "read_start": rc,
        "read_end": rc,
        "capture_pin": before_pin,
        "raw_inventory_sha256": m.preservation.digest([]),
        "derivations": {"support_witnesses": {}},
        "restart_counter_scopes": registration["counter_scopes"],
        "refusals": [],
    }
    bundle = {
        "format": "strength-preservation-capture-provenance-bundle-v1",
        "schema_version": 1,
        "binding": {
            "phase": "before",
            "request_sha256": bp["sha256"],
            "registration_file_sha256": b["registration_sha256"],
            "encoding_contract_sha256": "e" * 64,
            "source_pins": sources,
            "capture_pin": before_pin,
            "read_start": rc,
            "read_end": rc,
        },
        "raw_inventory": [],
        "raw_inventory_sha256": m.preservation.digest([]),
        "support_witnesses": {},
        "core_provenance": {
            "format": "strength-preservation-capture-provenance-v1",
            "schema_version": 1,
            "phase": "before",
            "registration_sha256": m.preservation.digest(registration),
            "policy_sha256": m.preservation.digest(policy),
            "verified_champions_sha256": m.preservation.digest(champions),
            "encoding_contract_sha256": "e" * 64,
            "support_witnesses": {},
            "raw_inventory_count": 0,
            "support_provenance": [],
            "raw_inventory_sha256": m.preservation.digest([]),
        },
        "metadata_audit": [],
        "final_clock_projection": {},
        "preservation_verification": None,
    }
    bundle_raw = encoded(bundle)
    bundle_pin = add(
        root / ("before.provenance-" + m.digest(bundle_raw) + ".json"), bundle_raw
    )
    br["derivations"]["provenance_pin"] = bundle_pin
    brp = add(root / "before-receipt.json", br)
    prefix = "edgeconnect-cpuqual-" + nonce + "-"
    dispatcher = prefix + "dispatcher.service"
    props: dict[str, str] = dict.fromkeys(m.DUMMY_STATIC, "")
    props.update(
        Id=dispatcher,
        LoadState="loaded",
        FragmentPath="/etc/systemd/system/" + dispatcher,
    )
    definition = add(root / "dispatcher.service", b"[Service]\nExecStart=/bin/true\n")
    stage = {"properties": props, "unit": definition, "environment_files": []}
    units = {
        prefix + role + ".service": {
            "role": role,
            "installed_path": "/etc/systemd/system/" + prefix + role + ".service",
            "before": stage,
            "after": stage,
        }
        for role in ("dispatcher", "cleanup", "observer", "publisher", "watchdog")
    }
    units.update({prefix + f"work{i}.service": {"role": "workload"} for i in range(7)})
    plan = {
        "format": "strength-freshness-cpu-plan-v1",
        "schema_version": 1,
        "nonce": nonce,
        "attempt_id": "trial",
        "boot_id": "boot",
        "addenda_sha256": [m.ADDENDUM],
        "input_root": str(root),
        "scratch_root": str(scratch),
        "source_pins": sources,
        "units": units,
        "preservation": {
            "policy": policy_pin,
            "before": before_pin,
            "before_request": bp,
            "before_receipt": brp,
            "after_path": str(scratch / "external/r3-after.json"),
            "verified_champions": champions,
        },
    }
    pp = add(root / "plan.json", plan)
    anchor = life.Anchor(
        "trial", nonce, pp["sha256"], "boot", 103.0, now_ns + 3 * 10**9
    )
    ap = add(root / "anchor.json", anchor.as_dict())
    proofroot = scratch / "evidence"

    def proof(event, data, token):
        return add(
            proofroot / (event + "-" + str(token) * 32 + ".json"),
            {
                "format": "strength-freshness-cpu-lifecycle-evidence-v1",
                "schema_version": 1,
                "event": event,
                "anchor": anchor.as_dict(),
                "anchor_sha256": life.digest(anchor.as_dict()),
                "data": data,
            },
        )

    owner = {
        "pid": 700,
        "unit": dispatcher,
        "cgroup": "/system.slice/" + dispatcher,
        "invocation_id": "7" * 32,
        "start_monotonic_us": 104_000_000,
    }
    started_clock = {
        "boot_id": "boot",
        "monotonic": 104.0,
        "wall_ns": now_ns + 4 * 10**9,
    }
    raw_start = proof(
        "raw-role-start",
        {"role": "dispatcher", "owner": owner, "clock": started_clock},
        1,
    )
    started = proof(
        "dispatcher-started",
        {
            "raw": raw_start,
            "owner": owner,
            "clock": started_clock,
            "source_sha256": life._source_sha256(),
        },
        2,
    )
    barrier_clock = {
        "boot_id": "boot",
        "monotonic": 105.0,
        "wall_ns": now_ns + 5 * 10**9,
    }
    terminal = {
        **props,
        "Id": dispatcher,
        "ControlGroup": owner["cgroup"],
        "InvocationID": owner["invocation_id"],
        "ExecMainPID": 700,
        "ExecMainStartTimestampMonotonic": 104_000_000,
        "ExecMainExitTimestampMonotonic": 105_000_000,
        "ExecMainCode": "exited",
        "ExecMainStatus": "0",
        "MainPID": 0,
        "ActiveState": "inactive",
        "SubState": "dead",
        "Result": "success",
        "Job": "0",
        "members": [],
        "observed_boot_id": "boot",
        "observed_monotonic": 105.0,
    }
    cleanup_clock = {
        "boot_id": "boot",
        "monotonic": 106.0,
        "wall_ns": now_ns + 6 * 10**9,
    }
    workloads = [n for n, v in units.items() if v["role"] == "workload"]
    facts = {
        "dispatcher_barrier": {
            "started": started,
            "unit": terminal,
            "clock": barrier_clock,
        },
        "units": {
            n: {
                "Id": n,
                "ActiveState": "inactive",
                "SubState": "dead",
                "MainPID": 0,
                "Job": "0",
                "members": [],
            }
            for n in workloads
        },
        "jobs": [],
        "members": [],
        "links": [],
        "leftovers": [],
    }
    raw_cleanup = proof("raw-cleanup", {"clock": cleanup_clock, "facts": facts}, 3)
    cp = proof(
        "cleanup-complete",
        {
            "raw": raw_cleanup,
            "clock": cleanup_clock,
            "expected_workloads": workloads,
            "source_sha256": life._source_sha256(),
        },
        4,
    )
    value = request_value("after")
    value.update(
        registration_sha256=b["registration_sha256"],
        source_pins=sources,
        plan_sha256=pp["sha256"],
        anchor_sha256=life.digest(anchor.as_dict()),
        before_pin=before_pin,
        policy_pin=policy_pin,
        cleanup_pin=cp,
        before_request_pin=bp,
        before_receipt_pin=brp,
    )
    value["limits"]["started"].update(
        monotonic_ns=106_000_000_000, wall_ns=now_ns + 6 * 10**9
    )
    value["limits"].update(
        deadline_monotonic_ns=120_000_000_000, deadline_wall_ns=now_ns + 20 * 10**9
    )
    now = {
        "boot_id": "boot",
        "monotonic_ns": 107_000_000_000,
        "wall_ns": now_ns + 7 * 10**9,
    }

    class MemoryReader(m.PinnedReader):
        def __init__(self):
            super().__init__(deadline=120, owner_uid=0, monotonic=lambda: 107)

        def read(self, expected, *, root=None, limit=m.FILE_LIMIT):
            m.pin(expected, limit=limit)
            self.check()
            if root is not None:
                assert Path(expected["path"]).parent == root
            raw = files[expected["path"]]
            self.charge(len(raw))
            m.require(
                m.digest(raw) == expected["sha256"] and len(raw) == expected["bytes"],
                "input-hash",
            )
            return raw

        def directory(self, root):
            assert root == proofroot
            return (1, 2, 0, 0o700)

        def inventory(self, root):
            assert root == proofroot
            return sorted(Path(x) for x in files if Path(x).parent == root)

    return (
        {
            "value": value,
            "registration": registration,
            "reader": MemoryReader(),
            "plan_pin": pp,
            "anchor_pin": ap,
            "now": now,
        },
        files,
        plan,
        anchor,
        br,
        b,
    )


def test_after_admission_uses_real_semantic_cleanup_chain(admitted):
    args, *_ = admitted
    result = m.admit_after(**args)
    assert result.cleanup_clock["monotonic"] == 106
    assert result.policy == args["registration"]["policy"]
    assert result.support_witnesses == {}
    assert len(result.proof.loaded) == 4


@pytest.mark.parametrize(
    "field", ["before_pin", "before_request_pin", "before_receipt_pin", "policy_pin"]
)
def test_uncommitted_before_sidecar_cannot_be_substituted(admitted, field):
    args, *_ = admitted
    args["value"][field] = {**args["value"][field], "sha256": "0" * 64}
    with pytest.raises(m.RequestRefusal, match="precommitted-input-pin"):
        m.admit_after(**args)


def test_before_request_other_boot_refuses_independently(admitted):
    args, _, _, anchor, receipt, b = admitted
    b["limits"]["started"]["boot_id"] = "oldboot"
    before = json.loads(args["reader"].read(args["value"]["before_pin"]))
    with pytest.raises(m.RequestRefusal, match="before-receipt-boot"):
        m.validate_before_receipt(
            receipt,
            before_request=b,
            before_request_pin=args["value"]["before_request_pin"],
            before_pin=args["value"]["before_pin"],
            registration=args["registration"],
            before=before,
            anchor=anchor,
        )


def test_executing_checker_cannot_be_replaced_by_unrelated_disk_copy(admitted):
    args, *_ = admitted
    original = str(Path(m.__file__).resolve())
    for item in args["value"]["source_pins"]:
        if item["path"] == original:
            item["path"] = "/input/other-checker.py"
    args["value"]["source_pins"].sort(key=lambda item: item["path"])
    for item in args["registration"]["scope"]["files"].values():
        if item["path"] == original:
            item["path"] = "/input/other-checker.py"
    with pytest.raises(m.RequestRefusal, match="executed-module-origin"):
        m.admit_after(**args)


def repin(files, expected, value):
    raw = encoded(value)
    files[expected["path"]] = raw
    expected.update(sha256=m.digest(raw), bytes=len(raw))
    return expected


@pytest.mark.parametrize(
    "admitted,allowed",
    [
        (None, False),
        ("unqualified-empty-ExecStartPre", False),
        ("systemd-255-empty-ExecStartPre", True),
    ],
    indirect=["admitted"],
)
def test_cleanup_command_omission_needs_exact_registered_rule(admitted, allowed):
    args, files, *_ = admitted
    cp = args["value"]["cleanup_pin"]
    completed = json.loads(files[cp["path"]])
    rp = completed["data"]["raw"]
    raw = json.loads(files[rp["path"]])
    del raw["data"]["facts"]["dispatcher_barrier"]["unit"]["ExecStartPre"]
    repin(files, rp, raw)
    repin(files, cp, completed)
    if allowed:
        assert m.admit_after(**args).cleanup_clock["monotonic"] == 106
    else:
        with pytest.raises(m.RequestRefusal, match="unqualified-dummy-empty-property"):
            m.admit_after(**args)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-workload",
        "foreign-dispatcher",
        "static-command",
        "late-cleanup",
        "wrong-source",
        "live-members",
        "pending-job",
    ],
)
def test_hash_valid_cleanup_cannot_grant_its_own_scope(admitted, fault):
    args, files, _plan, _anchor, _receipt, _before_request = admitted
    cp = args["value"]["cleanup_pin"]
    completed = json.loads(files[cp["path"]])
    rp = completed["data"]["raw"]
    raw = json.loads(files[rp["path"]])
    facts = raw["data"]["facts"]
    barrier = facts["dispatcher_barrier"]
    if fault == "missing-workload":
        name = completed["data"]["expected_workloads"].pop()
        del facts["units"][name]
    if fault == "foreign-dispatcher":
        start_pin = barrier["started"]
        start = json.loads(files[start_pin["path"]])
        owner = start["data"]["owner"]
        name = owner["unit"].replace("dispatcher", "other")
        owner.update(unit=name, cgroup="/system.slice/" + name)
        sp = start["data"]["raw"]
        start_raw = json.loads(files[sp["path"]])
        start_raw["data"]["owner"] = owner
        repin(files, sp, start_raw)
        repin(files, start_pin, start)
        barrier["unit"].update(Id=name, ControlGroup=owner["cgroup"])
    if fault == "static-command":
        barrier["unit"]["ExecStart"] = "/unregistered/command"
    if fault == "late-cleanup":
        completed["data"]["clock"]["monotonic"] += 600
        raw["data"]["clock"] = completed["data"]["clock"]
    if fault == "wrong-source":
        completed["data"]["source_sha256"] = "0" * 64
    if fault == "live-members":
        facts["members"] = [{"pid": 55}]
    if fault == "pending-job":
        facts["jobs"] = [{"id": 1}]
    repin(files, rp, raw)
    repin(files, cp, completed)
    with pytest.raises((m.RequestRefusal, life.Refusal)):
        m.admit_after(**args)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-binding",
        "wrong-inventory",
        "wrong-witness",
        "bad-prefix",
        "claims-verification",
        "outside-parent",
    ],
)
def test_before_provenance_is_read_and_bound(admitted, fault):
    args, files, _plan, _anchor, receipt, _before_request = admitted
    pp = receipt["derivations"]["provenance_pin"]
    body = json.loads(files[pp["path"]])
    if fault == "wrong-binding":
        body["binding"]["registration_file_sha256"] = "0" * 64
    if fault == "wrong-inventory":
        body["raw_inventory"].append({"invented": True})
    if fault == "wrong-witness":
        body["support_witnesses"] = {"fake": {}}
    if fault == "bad-prefix":
        body["core_provenance"]["raw_inventory_count"] = 1
    if fault == "claims-verification":
        body["preservation_verification"] = {"passed": True}
    raw = encoded(body)
    parent = "/outside" if fault == "outside-parent" else "/input"
    updated = pinned(
        Path(parent) / ("before.provenance-" + m.digest(raw) + ".json"), raw
    )
    files[updated["path"]] = raw
    receipt["derivations"]["provenance_pin"] = updated
    with pytest.raises(m.RequestRefusal):
        m.read_before_provenance(
            args["reader"],
            receipt=receipt,
            receipt_pin=args["value"]["before_receipt_pin"],
            before_request_pin=args["value"]["before_request_pin"],
            before_pin=args["value"]["before_pin"],
            registration=args["registration"],
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1e999}',
        b'{"x":1,"x":2}',
        '{"x":1}'.encode("utf-16"),
        b"{" + b'"x":[' * 1 + b"[" * 65 + b"]" * 66 + b"}",
    ],
)
def test_provenance_parser_never_relaxes_finite_utf8_depth(raw):
    with pytest.raises(m.RequestRefusal):
        m.provenance_json(raw)


def test_source_projection_matches_linux_without_runtime_import_in_collector():
    from scripts.strength_freshness_linux import PROPERTIES, VARIABLE, stable_properties

    assert m.DUMMY_STATIC == set(PROPERTIES) - VARIABLE
    expected: dict[str, str] = dict.fromkeys(m.DUMMY_STATIC, "")
    expected.update(
        ExecStart="{ path=/bin/x ; argv[]=/bin/x ; ignore_errors=yes ; pid=3 ; code=exited ; future=keep }",
        Before="z.service a.service",
    )
    assert m.dummy_static(expected) == stable_properties(expected)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "future-window",
        "wrong-self-hash",
        "not-absence",
        "unlogged",
        "unretained-manager",
    ],
)
def test_authenticated_witness_has_real_shape_and_prework_window(admitted, fault):
    support_fixtures = runpy.run_path(
        str(Path(__file__).with_name("test_strength_freshness_cpu_collect_support.py"))
    )
    collector, _io, monitors = support_fixtures["support_fixture"]()
    witness = copy.deepcopy(
        collector.capture(monitors=monitors).witnesses["backup.service"]
    )
    args, files, _plan, _anchor, receipt, _before_request = admitted
    rc = receipt["read_end"]
    witness["absence"]["read_start"] = dict(rc)
    witness["absence"]["read_end"] = dict(rc)
    for rows in (witness["proof"]["unit_reads"], witness["proof"]["jobs_reads"]):
        for row in rows:
            row["audit"]["read_start"] = dict(rc)
            row["audit"]["read_end"] = dict(rc)
    if fault == "future-window":
        witness["absence"]["read_end"]["monotonic_ns"] += 1
    if fault == "not-absence":
        witness["absence"]["job"] = 7
    witness["absence"]["raw_sha256"] = m.supports.digest(witness["proof"])
    witness = m.supports._witness(witness["absence"], witness["proof"])
    if fault == "wrong-self-hash":
        witness["sha256"] = "0" * 64
    packet = {"backup.service": witness}
    pp = receipt["derivations"]["provenance_pin"]
    value = json.loads(files[pp["path"]])
    value["support_witnesses"] = value["core_provenance"]["support_witnesses"] = packet
    receipt["derivations"]["support_witnesses"] = packet
    audits = [
        copy.deepcopy(row["audit"])
        for row in witness["proof"]["unit_reads"] + witness["proof"]["jobs_reads"]
    ]
    value["raw_inventory"] = [] if fault == "unlogged" else audits
    value["raw_inventory_sha256"] = m.preservation.digest(value["raw_inventory"])
    receipt["raw_inventory_sha256"] = value["raw_inventory_sha256"]
    value["core_provenance"]["raw_inventory_count"] = len(value["raw_inventory"])
    value["core_provenance"]["raw_inventory_sha256"] = value["raw_inventory_sha256"]
    value["core_provenance"]["support_provenance"] = (
        []
        if fault == "unretained-manager"
        else [
            {
                "format": "strength-support-capture-v1",
                "manager": witness["proof"]["manager"],
                "manager_id": witness["absence"]["manager_id"],
                "clock": witness["absence"]["read_end"],
                "unit_states": {"backup.service": {"Job": "0"}},
                "jobs": {},
                "audit": audits,
            }
        ]
    )
    raw = encoded(value)
    pp = pinned(Path("/input") / ("before.provenance-" + m.digest(raw) + ".json"), raw)
    files[pp["path"]] = raw
    receipt["derivations"]["provenance_pin"] = pp
    kwargs = dict(
        reader=args["reader"],
        receipt=receipt,
        receipt_pin=args["value"]["before_receipt_pin"],
        before_request_pin=args["value"]["before_request_pin"],
        before_pin=args["value"]["before_pin"],
        registration=args["registration"],
    )
    if fault is None:
        assert m.read_before_provenance(**kwargs)["support_witnesses"] == packet
    else:
        with pytest.raises((m.RequestRefusal, m.supports.SupportRefusal)):
            m.read_before_provenance(**kwargs)
