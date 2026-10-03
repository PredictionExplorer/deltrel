"""Finite-core fault tests. All host/GPU observations and actions are fakes."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deltreltrain import strength_freshness_guard as g
from scripts import strength_freshness_guard as cli


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def artifact(path, label="data"):
    return {"path": str(path), "bytes": len(label), "sha256": sha(label)}


def make_plan(tmp_path):
    runtime = tmp_path / "release/training"
    (tmp_path / "live").mkdir()
    python = {
        **artifact(runtime / ".venv/bin/python"),
        "resolved_path": "/usr/bin/python",
    }
    runtime_pins = {
        "source_commit": "a" * 40,
        "source_commit_file": artifact(runtime / "SOURCE_COMMIT"),
        "source_manifest": artifact(runtime / "SOURCE_SHA256SUMS", "manifest"),
        "python": python,
        "pyvenv": artifact(runtime / ".venv/pyvenv.cfg"),
        "training_module": artifact(runtime / "deltreltrain/__init__.py"),
        "native_wrapper": artifact(runtime / ".venv/deltrel_native/__init__.py"),
        "native_binaries": [artifact(runtime / ".venv/deltrel_native/native.so")],
        "rules_hash": "rules",
        "feature_schema_hash": 123,
        "search_algorithm": "search",
        "torch_version": "2.13.0",
        "torch_cuda_build": "13.0",
    }
    plan = {
        "format": g.FORMAT,
        "schema_version": 1,
        "attempt_id": "test-once",
        "freshness_plan_sha256": sha("freshness"),
        "guard_source_sha256": g.source_sha256(),
        "adapter_qualification_sha256": sha("qualified-fake"),
        "probe_source_sha256": sha("probe"),
        "gpu_identity_sha256": sha("hardware"),
        "initial_boot_id": "boot-a",
        "run_root": str(tmp_path / "live"),
        "runtime_root": str(runtime),
        "probe_output": str(tmp_path / "probe"),
        "state_root": str(tmp_path / "guard"),
        "exclusion_path": str(tmp_path / "host.lock"),
        "continuation_started_ns": 10,
        "budget": {
            "total": 2700,
            "stop": 390,
            "probe": 600,
            "cleanup": 90,
            "apply": 120,
            "startup": 900,
            "canary": 120,
            "tail": 480,
        },
        "gpu_uuids": [f"GPU-{x}" for x in range(8)],
        "probe_gpu_uuid": "GPU-0",
        "units": {
            role: {
                "name": role + ".service",
                "definition_sha256": sha(role),
                "cgroup": "/system.slice/" + role + ".service",
                "initial_enabled": role == "r3",
                "committed_enabled": role == "r4",
                "autonomous_retries": 0,
            }
            for role in g.ROLES
        },
        "initial_r3": {
            "invocation_id": "1" * 32,
            "main": {"pid": 101, "start_ticks": 1},
        },
        "admission": {
            "plan_sha256": sha("freshness"),
            "source_commit": runtime_pins["source_commit"],
            "source_manifest_sha256": runtime_pins["source_manifest"]["sha256"],
            "profile_sha256": sha("profile"),
            "execution_pins": [
                python,
                artifact(runtime / ".venv/bin/deltreltrain-orchestrate"),
            ],
            "pyvenv": runtime_pins["pyvenv"],
            "training_module": runtime_pins["training_module"]["path"],
            "native_file": runtime_pins["native_wrapper"]["path"],
            "native_binaries": runtime_pins["native_binaries"],
        },
        "probe_runtime": runtime_pins,
        "probe_inputs": {
            "profile": artifact(tmp_path / "target.yaml", "profile"),
            "batch_manifest": artifact(tmp_path / "batch.json", "batch"),
        },
        "run_identity": {
            "run_id": "same-run",
            "generation_family": "same-family",
            "created_ns": 1,
        },
        "expected_workers": ["learner", "arena", "actor-1", "actor-2"],
        "expected_cohorts": 24,
        "learning_rates": {"muon": 0.0005, "adamw": 0.000008},
        "ema_decay": 0.9999,
    }
    plan["proof_closure_inputs"] = {key: [sha(key)] for key in g.PROOF_INPUTS}
    plan["proof_closure_inputs"].update(
        guard_source=[plan["guard_source_sha256"]],
        probe_source=[plan["probe_source_sha256"]],
        batch_manifest=[sha("batch")],
    )
    support: dict[str, Any] = {
        stage: {
            name + ".service": {
                "definition_sha256": sha(name + stage),
                "environment_sha256": sha("env" + name + stage),
                "enabled": True,
            }
            for name in ("monitor", "report", "backup")
        }
        for stage in ("before", "after")
    }
    support["bindings"] = {
        "monitor_unit_after": "r4.service",
        "profile_authority": str(tmp_path / "live/profile.sha256"),
        "python_after": python["path"],
        "runtime_pythonpath": str(runtime),
        "control_pythonpath": str(tmp_path / "support"),
        "control_cuda_visible_devices": "",
        "guard_before_units": ["r3.service", "r4.service"],
    }
    support["sha256"] = g.digest(support)
    plan["support_transition"] = support
    return plan


class Crash(BaseException):
    pass


class FakeHost:
    def __init__(self, plan, journal):
        self.plan, self.journal = plan, journal
        self.adapter_qualification_sha256 = plan["adapter_qualification_sha256"]
        self.time, self.boot, self.wall_origin = 100.0, "boot-a", 1_000_000_000_000
        self.calls, self.due = [], []
        self.controller_alive, self.auto_progress = True, True
        self.lease = self.guard_ack = self.backup = None
        self.authority = g.Authority(
            "r3", plan["freshness_plan_sha256"], sha("old-authority")
        )
        self.behavior = {}
        self.next_pid = 200
        self.units = {
            role: g.Unit(
                spec["name"],
                spec["definition_sha256"],
                spec["cgroup"],
                enabled=spec["initial_enabled"],
            )
            for role, spec in plan["units"].items()
        }
        process = g.Process(101, 1)
        self.units["r3"] = replace(
            self.units["r3"],
            main=process,
            members=(process,),
            invocation_id="1" * 32,
            active="active",
            substate="running",
            entered_monotonic=99,
        )
        self.owners = tuple(
            g.GPUOwner(
                uuid, process, "r3.service", "1" * 32, "/system.slice/r3.service"
            )
            for uuid in plan["gpu_uuids"]
        )
        self.support = {
            k: {**v, "job": None}
            for k, v in plan["support_transition"]["before"].items()
        }
        self.boundary = {
            "clean_stop": True,
            "preserved": {key: sha(key) for key in g.PRESERVED},
            "run_identity": plan["run_identity"],
            "continuation_started_ns": 10,
            "step": 100,
            "examples_consumed": 51200,
            "replay_committed_samples": 1000,
            "replay_counter_updated_ns": 1,
            "recovery_pointer": artifact(
                Path(plan["run_root"]) / "learner/recovery.json", "pointer"
            ),
            "checkpoint": artifact(
                Path(plan["run_root"]) / "learner/recovery/checkpoint.pt", "weights"
            ),
        }

    def clock(self):
        return g.Clock(self.boot, self.time, self.wall_origin + int(self.time * 1e9))

    def advance(self, seconds=1):
        self.time += seconds
        for when, callback in list(self.due):
            if when <= self.time:
                self.due.remove((when, callback))
                callback()

    def observe(self, deadline):
        assert g.finite(deadline)
        self.calls.append(("observe", deadline))
        progress = None
        for role in ("r4", "r3"):
            if self.units[role].active == "active" and self.auto_progress:
                progress = {
                    "role": role,
                    "invocation_id": self.units[role].invocation_id,
                    "continuation_started_ns": 10,
                    "profile_sha256": self.plan["admission"]["profile_sha256"],
                    "source_commit": self.plan["admission"]["source_commit"],
                    "learning_rates": self.plan["learning_rates"],
                    "ema_decay": 0.9999,
                    "workers": self.plan["expected_workers"],
                    "cohorts": 24,
                    "actor_sources": ["champion"],
                    "finite_metrics": True,
                    "arena_prefix_preserved": True,
                    "replay_committed_samples": 1000 + int(self.time),
                    "step": 100 + int(self.time),
                    "neural_work": int(self.time),
                }
                break
        return g.Observation(
            self.clock(),
            copy.deepcopy(self.units),
            tuple(self.plan["gpu_uuids"]),
            self.plan["gpu_identity_sha256"],
            self.owners,
            self.authority,
            self.controller_alive,
            copy.deepcopy(self.lease),
            copy.deepcopy(self.guard_ack),
            progress,
            copy.deepcopy(self.backup),
            copy.deepcopy(self.support),
        )

    def verify_inputs(self, plan, deadline):
        assert self.time < deadline

    def capture_boundary(self, plan, deadline):
        assert self.time < deadline
        return copy.deepcopy(self.boundary)

    def check_boundary(self, boundary, deadline):
        assert self.time < deadline
        if boundary != self.boundary:
            raise g.Refusal("stopped-boundary-drift")

    def activate(self, role):
        self.next_pid += 1
        process = g.Process(self.next_pid, self.next_pid + 10)
        self.units[role] = replace(
            self.units[role],
            main=process,
            members=(process,),
            invocation_id=f"{self.next_pid:032x}",
            active="active",
            substate="running",
            job=None,
            entered_monotonic=self.time,
        )
        if role != "guard":
            uuids = (
                [self.plan["probe_gpu_uuid"]]
                if role == "probe"
                else self.plan["gpu_uuids"]
            )
            self.owners = tuple(
                g.GPUOwner(
                    uuid,
                    process,
                    self.units[role].name,
                    self.units[role].invocation_id,
                    self.units[role].cgroup,
                )
                for uuid in uuids
            )

    def dead(self, role, *, result="success", exit_code=0):
        self.units[role] = replace(
            self.units[role],
            main=None,
            members=(),
            active="inactive",
            substate="dead",
            job=None,
            result=result,
            exit_code=exit_code,
        )
        self.owners = tuple(p for p in self.owners if p.unit != self.units[role].name)

    def raw_proof(self):
        challenge = self.journal.read("challenge.json")
        unit = self.units["probe"]
        common = {
            key: challenge[key]
            for key in ("attempt_id", "boot_id", "nonce", "probe_unit")
        }
        common.update(
            schema_version=1,
            mode="cuda",
            invocation_id=unit.invocation_id,
            challenge_sha256=hashlib.sha256(
                (self.journal.directory / "challenge.json").read_bytes()
            ).hexdigest(),
            started_monotonic=self.time,
        )
        output = Path(self.plan["probe_output"])
        output.mkdir()
        data = b"FAKE CPU fixture: no actual GPU work occurred"
        (output / "observations.jsonl").write_bytes(data)
        started = {"format": "strength-freshness-probe-started-v1", **common}
        result = {
            "format": "strength-freshness-probe-result-v1",
            **common,
            "completed_monotonic": self.time,
            "mode": "cuda",
            "status": "cuda_observed",
            "admission": challenge["admission"],
            "checks": {key: True for key in g.CHECKS},
            "work": {
                "optimizer_steps": 1,
                "native_searches": 24,
                "native_neural_calls": 24,
            },
            "evidence": [
                {
                    "path": "observations.jsonl",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            ],
        }
        if "mutate_result" in self.behavior:
            self.behavior["mutate_result"](result)
        (output / "started.json").write_text(json.dumps(started))
        (output / "result.json").write_text(json.dumps(result))

    def perform(self, kind, details, deadline):
        assert self.time < deadline
        self.calls.append((kind, deadline, copy.deepcopy(details)))
        callback = self.behavior.get(kind)
        if callback:
            callback()
            return
        if kind == "arm-guard":
            self.activate("guard")
            unit = self.units["guard"]
            assert unit.main is not None
            self.units["guard"] = replace(unit, enabled=True)
            self.lease = {
                "path": self.plan["exclusion_path"],
                "held": True,
                "attempt_id": details["attempt_id"],
                "nonce": details["nonce"],
                "plan_sha256": details["guard_plan_sha256"],
                "boot_id": self.boot,
                "invocation_id": unit.invocation_id,
                "owner": asdict(unit.main),
            }
            self.guard_ack = {
                key: self.lease[key]
                for key in (
                    "plan_sha256",
                    "attempt_id",
                    "nonce",
                    "boot_id",
                    "invocation_id",
                )
            }
            self.guard_ack["deadline"] = details["overall_deadline"]
        elif kind == "pause-support-and-stop-r3":
            self.dead("r3")
            self.units["r3"] = replace(self.units["r3"], enabled=False)
            for value in self.support.values():
                value["enabled"] = False
        elif kind.startswith("start-"):
            role = kind.removeprefix("start-")
            self.activate(role)
            if role == "probe":
                self.raw_proof()
                self.due.append((self.time + 3, lambda: self.dead("probe")))
        elif kind.startswith("stop-"):
            self.dead(kind.removeprefix("stop-"), result="signal", exit_code=15)
        elif kind in ("apply-r4", "repair-r4"):
            self.authority = g.Authority(
                "r4", self.plan["freshness_plan_sha256"], sha("new-authority")
            )
        elif kind == "fence-terminal-restarts":
            for role in ("r3", "r4", "probe"):
                self.units[role] = replace(self.units[role], enabled=False)
            for row in self.support.values():
                row["enabled"] = False
                row["job"] = None
        elif kind.startswith(
            ("restore-support-and-backup-", "restore-support-without-proof-")
        ):
            role = kind[-2:]
            self.support = {
                k: {**v, "job": None}
                for k, v in self.plan["support_transition"][
                    details["support_stage"]
                ].items()
            }
            enabled_key = "committed_enabled" if role == "r4" else "initial_enabled"
            for name in ("r3", "r4", "probe"):
                self.units[name] = replace(
                    self.units[name], enabled=self.plan["units"][name][enabled_key]
                )
            if kind.startswith("restore-support-and-backup-"):
                self.backup = {
                    "status": "committed",
                    "guard_plan_sha256": details["guard_plan_sha256"],
                    "role": role,
                    "closure_verified": True,
                    "catalog_sha256": sha("catalog"),
                    "proof_closure_sha256": details["proof_closure_sha256"],
                    "verified_artifact_sha256": details["required_proof_sha256"],
                }
        elif kind == "restore-support-before-stop":
            self.support = {
                k: {**v, "job": None}
                for k, v in self.plan["support_transition"]["before"].items()
            }
            self.units["r3"] = replace(
                self.units["r3"], enabled=self.plan["units"]["r3"]["initial_enabled"]
            )
        elif kind in ("retire-guard", "retire-guard-only"):
            self.dead("guard")
            self.units["guard"] = replace(self.units["guard"], enabled=False)
            self.lease = None
        else:
            raise AssertionError(kind)


@pytest.fixture
def setup(tmp_path):
    plan = make_plan(tmp_path)
    journal = g.Journal(Path(plan["state_root"]))
    host = FakeHost(plan, journal)
    controller = g.Controller(plan, g.digest(plan), journal, host)
    return plan, journal, host, controller


def drive(controller, host, *, recover=False, limit=40):
    for _ in range(limit):
        host.advance()
        state = controller.tick(recover=recover)
        if state["finished"]:
            return state
    raise AssertionError("finite fake driver did not finish")


def actions(host):
    return [call[0] for call in host.calls if call[0] != "observe"]


def test_success_requires_natural_exit_real_closure_and_guard_retirement(setup):
    plan, journal, host, controller = setup
    controller.begin()
    result = drive(controller, host)
    assert result["outcome"] == "committed-r4"
    assert result["handoff_complete"] and result["finished"]
    assert "stop-probe" not in actions(host)
    receipt = journal.read("cuda-qualification.json")
    assert receipt["checks"] == {
        **{key: True for key in g.CHECKS},
        "all_workers_released": True,
    }
    assert (
        receipt["recovery_pointer_sha256"]
        == host.boundary["recovery_pointer"]["sha256"]
    )
    assert (
        receipt["guard_evidence"]["released_monotonic"]
        >= journal.read("challenge.json")["issued_monotonic"] + 3
    )
    assert set(result["required_proof_sha256"]) <= set(
        host.backup["verified_artifact_sha256"]
    )
    assert host.units["r3"].enabled is False and host.units["r4"].enabled is True
    assert host.units["guard"].dead and not host.lease


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(mode="cpu_validation", status="cpu_validated"),
        lambda x: x.update(nonce="stale"),
        lambda x: x.update(boot_id="old-boot"),
        lambda x: x.update(invocation_id="0" * 32),
        lambda x: x.update(completed_monotonic=float("nan")),
        lambda x: x.update(schema_version=True),
        lambda x: x["work"].update(optimizer_steps=0),
        lambda x: x["work"].update(native_searches=23),
        lambda x: x["work"].update(native_neural_calls=0),
        lambda x: x["checks"].update(all_workers_released=True),
        lambda x: x["checks"].update(ema_preserved=False),
        lambda x: x["evidence"][0].update(sha256="0" * 64),
        lambda x: x["evidence"][0].update(path="../escape"),
    ],
)
def test_bad_probe_restores_r3_without_receipt_or_apply(setup, mutation):
    _, journal, host, controller = setup
    host.behavior["mutate_result"] = mutation
    controller.begin()
    result = drive(controller, host)
    assert result["outcome"] == "restored-r3", result
    assert "stop-probe" in actions(host)
    assert (
        "apply-r4" not in actions(host)
        and not (journal.directory / "cuda-qualification.json").exists()
    )


def test_persistent_controller_loss_recovers_all_the_way_to_backup(setup):
    _, _, host, controller = setup
    controller.begin()
    host.advance()
    controller.tick()  # Stopped R3.
    host.controller_alive = False
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "restored-r3"
    assert actions(host).count("start-r3") == 1


@pytest.mark.parametrize("exception", [RuntimeError, Crash])
def test_apply_durable_intent_crossover_never_restores_r3(setup, exception):
    plan, journal, host, controller = setup

    def fault():
        host.authority = g.Authority(
            "pending", plan["freshness_plan_sha256"], sha("durable-intent")
        )
        raise exception("after journal, before return")

    host.behavior["apply-r4"] = fault
    controller.begin()
    for _ in range(20):
        host.advance()
        try:
            controller.tick()
        except Crash:
            break
        if host.authority.phase == "pending":
            break
    # A separate controller/guard instance consumes persisted state, not the
    # process's last phase or whether its apply call returned successfully.
    replacement = g.Controller(plan, g.digest(plan), journal, host)
    host.controller_alive = False
    result = drive(replacement, host, recover=True)
    assert result["outcome"] == "committed-r4", result
    assert "repair-r4" in actions(host) and "start-r3" not in actions(host)


def test_result_visibility_waits_for_exit_and_pending_stop_job(setup):
    _, journal, host, controller = setup
    controller.begin()
    host.advance()
    controller.tick()
    host.advance()
    controller.tick()  # Probe starts, raw result visible.
    host.due.clear()
    host.advance()
    state = controller.tick()
    assert (
        state["phase"] == "cleaning-probe"
        and not (journal.directory / "cuda-qualification.json").exists()
    )
    host.units["probe"] = replace(host.units["probe"], job={"id": 12, "kind": "stop"})
    host.advance()
    state = controller.tick()
    assert state["phase"] == "cleaning-probe"
    assert not (journal.directory / "cuda-qualification.json").exists()
    host.dead("probe")
    assert drive(controller, host)["outcome"] == "committed-r4"


def test_retirement_failure_does_not_regain_runtime_authority(setup):
    _, _, host, controller = setup
    host.behavior["retire-guard-only"] = lambda: (_ for _ in ()).throw(
        RuntimeError("once")
    )
    state = controller.begin()
    for _ in range(25):
        host.advance()
        state = controller.tick()
        if state.get("retirement_failure"):
            break
    assert state["handoff_complete"] and not state["finished"]
    prior = len(host.calls)
    del host.behavior["retire-guard-only"]
    host.controller_alive = False
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "committed-r4"
    assert set(call[0] for call in host.calls[prior:]) <= {
        "observe",
        "retire-guard-only",
    }


def test_completed_guard_never_regains_runtime_authority_on_new_boot(setup):
    _, _, host, controller = setup
    controller.begin()
    drive(controller, host)
    prior = list(host.calls)
    host.boot = "another-boot"
    controller.tick(recover=True)
    assert host.calls == prior


def test_missing_external_proof_or_support_closure_cannot_complete(setup):
    _, _, host, controller = setup
    controller.begin()
    for _ in range(25):
        host.advance()
        state = controller.tick()
        if state["phase"] == "backup-r4":
            break
    host.backup["verified_artifact_sha256"] = []
    host.advance()
    state = controller.tick()
    assert (
        state["phase"] == "blocked"
        and state["failure"] == "external-proof-closure-incomplete"
    )
    assert not state["handoff_complete"]


@pytest.mark.parametrize("change", ["foreign", "lease", "unit", "authority"])
def test_unknown_ownership_blocks_admission(setup, change):
    _, _, host, controller = setup
    controller.begin()
    if change == "foreign":
        host.owners += (
            g.GPUOwner(
                "GPU-0", g.Process(999, 1), "foreign.service", "f" * 32, "/foreign"
            ),
        )
    elif change == "lease":
        host.lease["nonce"] = "foreign"
    elif change == "unit":
        host.units["r3"] = replace(host.units["r3"], definition_sha256=sha("drift"))
    else:
        host.authority = g.Authority(
            "unknown", host.plan["freshness_plan_sha256"], sha("unknown")
        )
    host.advance()
    state = controller.tick()
    assert state["phase"] == "blocked"
    assert not {"start-probe", "apply-r4", "start-r4"} & set(actions(host))


def test_state_location_source_and_deadline_tampering_refused(setup, tmp_path):
    plan, journal, host, controller = setup
    with pytest.raises(g.Refusal, match="location"):
        g.Controller(plan, g.digest(plan), g.Journal(tmp_path / "elsewhere"), host)
    bad = copy.deepcopy(plan)
    bad["guard_source_sha256"] = sha("other")
    bad["proof_closure_inputs"]["guard_source"] = [bad["guard_source_sha256"]]
    with pytest.raises(g.Refusal, match="source-mismatch"):
        g.Controller(bad, g.digest(bad), journal, host)
    controller.begin()
    state = journal.read("state.json")
    state["deadline"] += 100
    journal.save("state.json", state)
    with pytest.raises(g.Refusal, match="deadline-extended"):
        controller.tick()


def test_read_only_cli_does_not_expose_execution(setup, tmp_path):
    plan, _, host, _ = setup
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    output = cli.validate(p, hashlib.sha256(p.read_bytes()).hexdigest())
    assert output["live_execution_available"] is False
    assert host.calls == []


def until_phase(controller, host, phase):
    state = None
    for _ in range(25):
        host.advance()
        state = controller.tick()
        if state["phase"] == phase:
            return state
    raise AssertionError((phase, state))


def test_delayed_guard_ack_waits_without_stopping_runtime(setup):
    _, _, host, controller = setup
    controller.begin()
    unit, lease, ack = host.units["guard"], host.lease, host.guard_ack
    host.units["guard"] = replace(
        unit,
        main=None,
        members=(),
        invocation_id="",
        active="activating",
        substate="start",
        job={"id": 1, "kind": "start"},
    )
    host.lease = host.guard_ack = None
    for _ in range(10):
        host.advance()
        state = controller.tick()
        assert state["phase"] == "arming" and not state["finished"]
    assert actions(host) == ["arm-guard"]
    host.units["guard"], host.lease, host.guard_ack = unit, lease, ack
    assert drive(controller, host)["outcome"] == "committed-r4"


def test_pending_owned_stop_and_support_job_prevent_probe_admission(setup):
    _, _, host, controller = setup

    def stopping():
        host.units["r3"] = replace(
            host.units["r3"],
            active="deactivating",
            substate="stop",
            job={"id": 7, "kind": "stop"},
            enabled=False,
        )
        host.support["backup.service"]["job"] = {"id": 8, "kind": "stop"}

    host.behavior["pause-support-and-stop-r3"] = stopping
    controller.begin()
    state = until_phase(controller, host, "stopping-r3")
    host.advance()
    state = controller.tick()
    assert state["last_units"]["r3"]["job"]["id"] == 7
    host.dead("r3")
    host.advance()
    state = controller.tick()
    assert state["phase"] == "stopping-r3" and "start-probe" not in actions(host)
    assert state["last_support"]["backup.service"]["job"]["id"] == 8
    host.support["backup.service"]["job"] = None
    completed = drive(controller, host)
    assert completed["outcome"] == "committed-r4"
    assert any(
        event["jobs"]["support"].get("backup.service", {}).get("id") == 8
        for event in completed["events"]
    )


def test_pending_owned_probe_start_waits_for_actual_invocation(setup):
    _, journal, host, controller = setup

    def pending():
        host.units["probe"] = replace(
            host.units["probe"],
            active="activating",
            substate="start",
            job={"id": 9, "kind": "start"},
        )

    host.behavior["start-probe"] = pending
    controller.begin()
    until_phase(controller, host, "probing")
    host.advance()
    state = controller.tick()
    assert state["last_units"]["probe"]["job"]["id"] == 9
    assert "probe" not in state["owners"]
    assert not (journal.directory / "cuda-qualification.json").exists()
    host.activate("probe")
    host.raw_proof()
    host.due.append((host.time + 3, lambda: host.dead("probe")))
    assert drive(controller, host)["outcome"] == "committed-r4"


def test_controller_death_before_stop_restores_partial_support_only(setup):
    _, _, host, controller = setup
    controller.begin()
    host.support["monitor.service"]["enabled"] = False
    host.controller_alive = False
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "aborted-before-stop-r3-unchanged"
    assert set(actions(host)) == {
        "arm-guard",
        "restore-support-before-stop",
        "retire-guard-only",
    }
    assert host.units["r3"].main == g.Process(101, 1)


def test_successful_raw_result_with_failed_exit_restores_r3(setup):
    _, journal, host, controller = setup
    controller.begin()
    until_phase(controller, host, "cleaning-probe")
    host.due.clear()
    host.dead("probe", result="exit-code", exit_code=1)
    result = drive(controller, host)
    assert result["outcome"] == "restored-r3"
    assert not (journal.directory / "cuda-qualification.json").exists()


def test_probe_deadline_does_not_renew_or_admit_late_result(setup):
    _, journal, host, controller = setup
    initial = controller.begin()
    state = until_phase(controller, host, "probing")
    host.advance(state["phase_deadline"] - host.time)
    state = controller.tick()
    assert state["deadline"] == initial["deadline"]
    assert "apply-r4" not in actions(host)
    assert drive(controller, host)["outcome"] == "restored-r3"
    assert not (journal.directory / "cuda-qualification.json").exists()


def test_startup_timeout_is_terminal_without_unbounded_retry(setup):
    _, _, host, controller = setup
    host.auto_progress = False
    initial = controller.begin()
    state = until_phase(controller, host, "starting-r4")
    host.advance(state["phase_deadline"] - host.time)
    state = controller.tick(recover=True)
    # A recovery request does not reissue/renew a running target's startup.
    state = drive(controller, host, recover=True)
    assert state["finished"] and state["failure"] == "phase-timeout-no-implicit-retry"
    assert state["outcome"] == "failed-closed-no-runtime"
    assert not host.owners and host.units["r4"].dead and not host.units["r4"].enabled
    assert state["deadline"] == initial["deadline"]
    assert actions(host).count("start-r4") == 1 and "start-r3" not in actions(host)


def test_boot_recovery_uses_original_expiry_and_no_old_process_ownership(setup):
    _, _, host, controller = setup
    initial = controller.begin()
    until_phase(controller, host, "stopping-r3")
    wall = host.clock().wall_ns
    host.dead("guard")
    host.lease = host.guard_ack = None
    host.boot, host.time, host.wall_origin = "boot-b", 0.0, wall + 10_000_000_000
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "restored-r3"
    assert result["deadline_wall_ns"] == initial["deadline_wall_ns"]
    assert result["deadline"] < g.TOTAL and result["active_boot_id"] == "boot-b"
    assert "start-probe" not in actions(host)
    assert all(owner["boot_id"] == "boot-b" for owner in result["owners"].values())


def test_retirement_after_crash_on_later_boot_has_only_cleanup_authority(setup):
    _, _, host, controller = setup
    controller.begin()
    until_phase(controller, host, "retiring")
    host.dead("guard")
    host.lease = None
    host.boot = "boot-b"
    prior = len(host.calls)
    assert drive(controller, host, recover=True)["outcome"] == "committed-r4"
    assert set(call[0] for call in host.calls[prior:]) <= {
        "observe",
        "retire-guard-only",
    }


def test_retirement_refuses_reused_unit_with_foreign_invocation(setup):
    _, _, host, controller = setup
    controller.begin()
    until_phase(controller, host, "retiring")
    host.activate("guard")
    host.lease = None
    prior = len(host.calls)
    with pytest.raises(g.Refusal, match="retirement-foreign-guard"):
        controller.tick(recover=True)
    assert [call[0] for call in host.calls[prior:]] == ["observe"]


def test_emitted_challenge_matches_real_probe_parser_without_execution(
    setup, monkeypatch
):
    # Only the parser executes; fake ambient identity does not qualify a runtime.
    from scripts import strength_freshness_probe as probe

    plan, journal, host, controller = setup
    host.behavior["start-probe"] = lambda: None
    controller.begin()
    state = until_phase(controller, host, "probing")
    monkeypatch.setattr(probe, "boot_id", lambda: host.boot)
    monkeypatch.setattr(probe.time, "monotonic", lambda: host.time)
    monkeypatch.setattr(probe, "digest", lambda _: plan["probe_source_sha256"])
    monkeypatch.setattr(
        probe.sys, "executable", plan["probe_runtime"]["python"]["path"]
    )
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            __file__=str(Path(plan["runtime_root"]) / "deltreltrain/__init__.py")
        ),
    )
    value, invocation = probe.validate_challenge(
        journal.directory / "challenge.json", state["challenge_file_sha256"]
    )
    assert value == state["challenge"] and invocation == "a" * 32
    assert value["schema_version"] == 1
    assert not Path(plan["probe_output"]).exists()


def test_preexisting_probe_directory_rejected_before_child_start(setup):
    plan, journal, host, controller = setup
    Path(plan["probe_output"]).mkdir()
    controller.begin()
    result = drive(controller, host)
    assert result["outcome"] == "restored-r3"
    assert "start-probe" not in actions(host)
    assert not (journal.directory / "cuda-qualification.json").exists()


def test_systemd_may_clear_invocation_only_after_fully_dead(setup, monkeypatch):
    _, journal, host, controller = setup
    original = host.dead

    def cleared(role, **kwargs):
        original(role, **kwargs)
        host.units[role] = replace(host.units[role], invocation_id="")

    monkeypatch.setattr(host, "dead", cleared)
    controller.begin()
    result = drive(controller, host)
    assert result["outcome"] == "committed-r4"
    retained = result["owners"]["probe"]["invocation_id"]
    assert retained and not host.units["probe"].invocation_id
    assert (
        journal.read("cuda-qualification.json")["guard_evidence"]["probe_invocation_id"]
        == retained
    )


def test_nonempty_different_exited_invocation_still_refused(setup):
    _, journal, host, controller = setup
    controller.begin()
    until_phase(controller, host, "cleaning-probe")
    host.due.clear()
    host.dead("probe")
    host.units["probe"] = replace(host.units["probe"], invocation_id="f" * 32)
    host.advance()
    state = controller.tick()
    assert state["failure"] == "exited-unit-invocation-drift"
    assert not (journal.directory / "cuda-qualification.json").exists()


@pytest.mark.parametrize(
    "key,value", [("torch_version", ""), ("torch_cuda_build", None)]
)
def test_cuda_plan_requires_qualified_torch_and_cuda_build(setup, key, value):
    plan, _, _, _ = setup
    plan["probe_runtime"][key] = value
    with pytest.raises(g.Refusal, match="cuda-runtime-build-required"):
        g.validate_plan(plan)


def test_active_persistent_guard_can_recover_after_reboot(setup):
    _, _, host, controller = setup
    initial = controller.begin()
    until_phase(controller, host, "stopping-r3")
    wall = host.clock().wall_ns
    host.boot, host.time, host.wall_origin = "boot-b", 0.0, wall + 10_000_000_000
    host.activate("guard")
    unit = host.units["guard"]
    host.lease.update(
        boot_id=host.boot, invocation_id=unit.invocation_id, owner=asdict(unit.main)
    )
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "restored-r3"
    assert result["deadline_wall_ns"] == initial["deadline_wall_ns"]
    assert actions(host).count("arm-guard") == 1 and "start-probe" not in actions(host)


def test_backup_timeout_preserves_productive_r4_without_success_claim(setup):
    _, _, host, controller = setup
    host.behavior["restore-support-and-backup-r4"] = lambda: None
    controller.begin()
    state = until_phase(controller, host, "backup-r4")
    host.advance(state["phase_deadline"] - host.time)
    state = controller.tick()
    assert state["phase"] == "proof-incomplete"
    result = drive(controller, host, recover=True)
    assert result["outcome"] == "productive-r4-proof-incomplete"
    assert result["recovery_required"] and not result["handoff_complete"]
    assert host.units["r4"].active == "active" and host.units["r4"].enabled
    assert (
        host.owners
        and "stop-r4" not in actions(host)
        and "start-r3" not in actions(host)
    )
    assert host.backup is None and host.units["guard"].dead


def test_runtime_death_during_backup_prevents_success_and_cleans_ownership(setup):
    _, _, host, controller = setup
    controller.begin()
    until_phase(controller, host, "backup-r4")
    assert host.backup["status"] == "committed"
    host.dead("r4", result="exit-code", exit_code=1)
    result = drive(controller, host)
    assert result["outcome"] == "failed-closed-no-runtime"
    assert result["failure"] == "runtime-failed-during-backup"
    assert not result["handoff_complete"] and not host.owners
    assert not host.units["r3"].enabled and not host.units["r4"].enabled


def test_frequent_observation_does_not_exhaust_bounded_action_history(setup):
    _, _, host, controller = setup
    host.auto_progress = False
    controller.begin()
    initial = until_phase(controller, host, "starting-r4")
    state = initial
    for _ in range(300):
        host.advance()
        state = controller.tick()
        assert state["phase"] == "starting-r4" and not state["finished"]
    assert state["deadline"] == initial["deadline"]
    assert len(state["events"]) < 128 and state["revision"] > 300
    assert state["events"][-1]["observations"] >= 300
    host.auto_progress = True
    assert drive(controller, host)["outcome"] == "committed-r4"
