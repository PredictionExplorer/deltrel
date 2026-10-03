"""Fixed helper facade tests; no real host, processes, services or GPU work."""

import json
from pathlib import Path

import pytest

from scripts import strength_freshness_cpu_install_runtime as f

BASE = 10**18
NONCE = "a" * 32
ROOT = "/input"
SCRATCH = "/run/edgeconnect-cpuqual-" + NONCE


def pin(path, data=b"x"):
    return {"path": path, "sha256": f.c.sha(data), "bytes": len(data)}


def owner(pid, ppid):
    return {
        "pid": pid,
        "ppid": ppid,
        "start_ticks": pid * 10,
        "pgid": pid,
        "sid": pid,
        "uid": 0,
        "boot_id": "boot",
        "cgroup": "/scope",
        "pid_namespace_inode": 1,
    }


class Clock:
    def __init__(self):
        self.mono = 100 * 10**9
        self.wall = BASE
        self.hints = {"cleanup-complete": True, "published-candidate": True}
        self.sleep_calls = []

    def clock(self):
        return {"boot_id": "boot", "monotonic_ns": self.mono, "wall_ns": self.wall}

    def hint(self, root, event):
        assert root == Path(SCRATCH) / "evidence"
        return self.hints[event]

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)
        self.mono += round(seconds * 1e9)
        self.wall += round(seconds * 1e9)


class Helper:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.fault = None
        self.alive = True

    def check_alive(self):
        if not self.alive:
            raise RuntimeError("pinned supervisor died")

    def call(self, mode, payload, deadline_ns):
        request = f.c.parse(payload)
        assert request["operation"] == mode
        assert len(payload) <= 32768
        self.calls.append((mode, request, deadline_ns))
        if self.fault == "exception":
            raise RuntimeError("helper unknown outcome")
        if self.fault == "late":
            self.clock.mono = deadline_ns + 1
        if mode == "prepare-install":
            result = {
                "plan_pin": pin(ROOT + "/plan.json"),
                "anchor_pin": pin(ROOT + "/anchor.json"),
                "after_path": SCRATCH + "/external/r3-after.json",
                "authorization_root": ROOT,
                "cleanup_proof_root": SCRATCH + "/evidence",
            }
            if self.fault == "output-path":
                result["after_path"] = "/foreign/result.json"
        elif mode == "inspect-cleanup":
            result = {
                "cleanup_pin": pin(SCRATCH + "/evidence/cleanup.json"),
                "clock": self.clock.clock(),
            }
        elif mode == "audit":
            result = {
                "status": "passed_cpu_scope",
                "plan_sha256": pin(ROOT + "/plan.json")["sha256"],
                "anchor_sha256": pin(ROOT + "/anchor.json")["sha256"],
                "evidence_pins": {"result": pin(SCRATCH + "/evidence/final.json")},
            }
        else:
            result = {"status": "prearm-only-complete"}
        response = {
            "format": f.RESULT,
            "schema_version": 1,
            "operation": mode,
            "status": "incomplete" if self.fault == "incomplete" else "complete",
            "result": result,
            "receipt_pin": pin(ROOT + "/installer-receipts/" + mode + ".json"),
        }
        if self.fault == "receipt-path":
            response["receipt_pin"] = pin("/foreign/receipt")
        return f.c.encoded(response)


def fixture():
    clock = Clock()
    helper = Helper(clock)
    intent = {
        "format": "explicit synthetic admitted runtime",
        "nonce": NONCE,
        "boot_id": "boot",
        "installer": {
            "input_root": ROOT,
            "template_root": "/templates",
            "blueprint": pin("/templates/blueprint.json"),
            "sources": pin("/templates/sources.json"),
        },
    }
    raw = f.c.encoded(intent)
    bundle = {
        "phase": "before",
        "completion_pin": pin(ROOT + "/before-execution.json"),
        "artifacts": {},
        "evidence_pins": {},
    }
    facade = f.Installer(
        raw,
        f.c.sha(raw),
        {"supervisor": owner(10, 1), "operator": owner(20, 10)},
        intent_pin=pin(ROOT + "/intent.json", raw),
        helper_executor=helper,
        _observation=clock,
    )
    return facade, helper, clock, f.c.encoded(bundle)


def prepare(x):
    facade, _helper, clock, bundle = x
    return json.loads(facade.finalize_template_and_install(bundle, clock.clock()))


def test_closed_sequence_reuses_start_and_calls_each_fixed_helper_once():
    x = fixture()
    facade, helper, clock, _ = x
    result = prepare(x)
    clock.mono += 405 * 10**9
    clock.wall += 405 * 10**9
    cleanup = json.loads(facade.await_checked_cleanup())
    clock.mono += 180 * 10**9
    clock.wall += 180 * 10**9
    final = json.loads(facade.final_audit_and_retire())
    assert facade.stage == "audited"
    assert [mode for mode, _, _ in helper.calls] == [
        "prepare-install",
        "inspect-cleanup",
        "audit",
    ]
    assert all(
        r["original_start"]["monotonic_ns"] == 100 * 10**9 for _, r, _ in helper.calls
    )
    assert all(
        r["start_pin"]["path"] == ROOT + "/dummy-start.json" for _, r, _ in helper.calls
    )
    assert cleanup["cleanup_pin"]["path"].startswith(SCRATCH + "/evidence/")
    assert final["plan_sha256"] == result["plan_pin"]["sha256"]
    assert helper.calls[0][2] == 160 * 10**9
    assert helper.calls[-1][2] == 695 * 10**9  # min(now+10, original+598)


@pytest.mark.parametrize(
    "fault", ["exception", "late", "output-path", "receipt-path", "incomplete"]
)
def test_bad_or_unknown_prepare_never_retries_as_new_action(fault):
    x = fixture()
    facade, helper, _clock, _ = x
    helper.fault = fault
    with pytest.raises((ValueError, RuntimeError)):
        prepare(x)
    assert facade.stage == "failed"
    with pytest.raises(ValueError, match="installer-prepare-once"):
        prepare(x)
    assert len(helper.calls) == 1


def test_filename_hint_only_defers_to_semantic_helper_and_is_not_success():
    x = fixture()
    facade, helper, _clock, _ = x
    prepare(x)
    helper.fault = "incomplete"
    with pytest.raises(ValueError, match="installer-helper-incomplete"):
        facade.await_checked_cleanup()
    assert facade.stage != "cleaned"


def test_absent_hint_waits_bounded_without_repeated_heavy_helper_import():
    x = fixture()
    facade, helper, clock, _ = x
    prepare(x)
    clock.mono = 639_900_000_000
    clock.wall = BASE + 539_900_000_000
    clock.hints["cleanup-complete"] = False
    with pytest.raises(ValueError, match="installer-original-deadline"):
        facade.await_checked_cleanup()
    assert len(helper.calls) == 1 and len(clock.sleep_calls) <= 2


def test_wall_clock_exhaustion_never_renews_monotonic_allowance():
    x = fixture()
    facade, helper, clock, _ = x
    prepare(x)
    clock.wall = BASE + 540 * 10**9
    with pytest.raises(ValueError, match="installer-original-deadline"):
        facade.await_checked_cleanup()
    assert len(helper.calls) == 1


def test_prearm_cleanup_only_after_failure_and_uses_original_setup_ceiling():
    x = fixture()
    facade, helper, clock, _ = x
    with pytest.raises(ValueError, match="installer-prearm-stage"):
        facade.prearm_cleanup()
    helper.fault = "exception"
    with pytest.raises(RuntimeError):
        prepare(x)
    helper.fault = None
    clock.mono += 55 * 10**9
    clock.wall += 55 * 10**9
    result = json.loads(facade.prearm_cleanup())
    assert result["status"] == "prearm-only-complete"
    assert helper.calls[-1][2] == 160 * 10**9
    with pytest.raises(ValueError, match="installer-single-helper-call"):
        facade.prearm_cleanup()


def test_helper_request_binds_shared_supervisor_ack_without_inventing_its_clock():
    x = fixture()
    facade, helper, _clock, _ = x
    prepare(x)
    body = helper.calls[0][1]
    value = f.outer.dummy_start_ack(
        outer_intent_sha256=body["intent_pin"]["sha256"],
        nonce=NONCE,
        start_pin=body["start_pin"],
        before_execution_pin=body["before_bundle"]["completion_pin"],
        enclosing=body["enclosing"],
    )
    raw = f.c.encoded(value)
    assert body["ack_pin"] == pin(ROOT + "/dummy-start.ack.json", raw)
    assert (
        "clock" not in value
    )  # Timeliness belongs to the actual supervisor, not a fabricated field.


def test_supervisor_death_revokes_polling_before_a_new_helper_call():
    x = fixture()
    facade, helper, clock, _ = x
    prepare(x)
    helper.alive = False
    clock.hints["cleanup-complete"] = False
    with pytest.raises(RuntimeError, match="pinned supervisor died"):
        facade.await_checked_cleanup()
    assert len(helper.calls) == 1 and clock.sleep_calls == []
