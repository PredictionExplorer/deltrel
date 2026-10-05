"""Pure setup rendering: no host writes, service calls, forks, or GPU access."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import strength_freshness_cpu_install as install
from scripts import strength_freshness_cpu_lifecycle as life
from scripts import strength_freshness_cpu_qualification as q
from scripts import strength_freshness_cpu_support_case as support
from test_strength_freshness_cpu_qualification import (
    BOOT,
    INPUT,
    NONCE,
    make_plan,
    artifact,
)


def fixture_inputs(monkeypatch):
    p, source = make_plan()
    # Artifact-render tests isolate the separately tested completion validator.
    # The real integration uses its shared completion fixture in a separate test.
    before_raw = b"explicit simulated producer completion"
    p["preservation"]["before_execution"] = artifact(
        INPUT / "before-execution.json", before_raw
    )
    if not any(
        Path(x["path"]).name == "strength_freshness_cpu_completion.py"
        for x in p["source_pins"]
    ):
        p["source_pins"].append(
            artifact(
                Path(p["control_root"])
                / "scripts/strength_freshness_cpu_completion.py",
                b"source",
            )
        )
    name = p["bindings"]["barrier"]
    spec = p["units"][name]
    old = source[spec["before"]["unit"]["path"]]
    env_target = p["scratch_root"] + "/env/support.env"
    p["files"][spec["installed_path"]] = []
    p["files"][env_target] = []
    for stage in ("before", "after"):
        raw = old.replace(
            b"Description=CPU fixture", ("Description=CPU fixture " + stage).encode()
        )
        raw = raw.replace(
            b"[Service]\n", ("[Service]\nEnvironmentFile=" + env_target + "\n").encode()
        )
        pin = artifact(INPUT / (name + "-" + stage), raw)
        source[pin["path"]] = raw
        spec[stage] = dict(spec[stage], unit=pin)
        p["files"][spec["installed_path"]].append(
            {"source": pin, "mode": 0o644, "uid": 0, "gid": 0}
        )
        envraw = ("CPUQUAL_PHASE=" + stage + "\n").encode()
        envpin = artifact(INPUT / (stage + ".env"), envraw)
        source[envpin["path"]] = envraw
        spec[stage]["environment_files"] = [
            {
                "path": env_target,
                "source_path": envpin["path"],
                "sha256": envpin["sha256"],
                "bytes": envpin["bytes"],
                "mode": 0o600,
            }
        ]
        p["files"][env_target].append(
            {"source": envpin, "mode": 0o600, "uid": 0, "gid": 0}
        )
    bindings = p["bindings"]
    scenario = {
        "format": support.FORMAT,
        "schema_version": 1,
        "nonce": NONCE,
        "guard_unit": bindings["support_guard"],
        "unit_target": spec["installed_path"],
        "env_target": env_target,
        "synthetic_plan": {
            "run_root": p["scratch_root"] + "/synthetic-run",
            "state_root": p["scratch_root"] + "/synthetic-support-state",
            "exclusion_path": p["scratch_root"] + "/qualification.lock",
            "freshness_plan_sha256": "f" * 64,
            "units": {
                role: {
                    "name": bindings[binding],
                    "initial_enabled": False,
                    "committed_enabled": False,
                }
                for role, binding in {
                    "guard": "support_guard",
                    "r3": "holder",
                    "r4": "contender",
                    "probe": "dependent",
                }.items()
            },
            "support_transition": {
                stage: {
                    name: {
                        "definition_sha256": spec[stage]["unit"]["sha256"],
                        "environment_sha256": q.linux.env_identity(
                            spec[stage]["properties"]["Environment"],
                            spec[stage]["environment_files"],
                        ),
                        "enabled": True,
                    }
                }
                for stage in ("before", "after")
            },
        },
    }
    scenario_raw = q.encode(scenario)
    scenario_pin = artifact(INPUT / "scenario.json", scenario_raw)
    p["units"][bindings["support_guard"]]["payload"]["scenario"] = scenario_pin
    source[scenario_pin["path"]] = scenario_raw
    artifacts = {
        key: p["preservation"][field]
        for key, field in [
            ("capture", "before"),
            ("request", "before_request"),
            ("receipt", "before_receipt"),
        ]
    }
    calls = []

    def validation(raw, expected, now, *, evidence):
        calls.append((raw, expected, now, evidence))
        return SimpleNamespace(artifacts=artifacts)

    monkeypatch.setattr(install.completion, "validate_before", validation)
    return SimpleNamespace(
        plan=p,
        sources=source,
        before=before_raw,
        artifacts=artifacts,
        calls=calls,
        expected={"outer_intent_sha256": "7" * 64, "nonce": NONCE, "boot_id": BOOT},
    )


def render(x, start=None):
    raw = q.encode(x.plan)
    return install.prepare(
        raw,
        q.sha(raw),
        start or life.Clock(BOOT, 100, 10**18),
        x.sources,
        before_execution=x.before,
        expected_before=x.expected,
        before_evidence={},
        approved_outer_intent_sha256="7" * 64,
    )


def test_deterministic_artifacts_keep_original_clock_and_exact_roles(monkeypatch):
    x = fixture_inputs(monkeypatch)
    first = render(x, life.Clock(BOOT, 100.123456789, 10**18 + 123456789))
    second = render(x, life.Clock(BOOT, 100.123456789, 10**18 + 123456789))
    assert first == second
    assert first.anchor.started_monotonic == 100.123456789
    assert first.anchor.started_wall_ns == 10**18 + 123456789
    assert len(first.plan.value["units"]) == 12
    assert (
        len([v for v in first.installed if v.path.endswith((".service", ".timer"))])
        == 12
    )
    assert (
        len([v for v in first.inputs if v.path.endswith(".authorization.json")]) == 10
    )
    assert len(first.start_order) == 4
    assert [first.plan.value["units"][n]["role"] for n in first.start_order] == [
        "watchdog",
        "observer",
        "publisher",
        "dispatcher",
    ]
    timer = next(a for a in first.installed if "watchdog.timer" in a.path)
    assert b"OnBootSec=490.123456s" in timer.data
    assert x.calls[0][2]["monotonic_ns"] == 100123456789
    assert first.plan.value["preservation"] == x.plan["preservation"]
    for a in first.inputs:
        if a.path.endswith(".authorization.json"):
            auth = json.loads(a.data)
            assert auth["plan_sha256"] == first.plan.checksum
            assert auth["anchor_sha256"] == q.sha(q.encode(asdict(first.anchor)))
            assert a.mode == 0o600
    assert all(
        a.mode == 0o644
        for a in first.installed
        if a.path.endswith((".timer", ".service"))
    )


def test_original_inputs_are_not_modified(monkeypatch):
    x = fixture_inputs(monkeypatch)
    before = q.encode(x.plan), dict(x.sources)
    prepared = render(x, life.Clock(BOOT, 110, 10**18 + 10**10))
    assert (q.encode(x.plan), x.sources) == before
    assert prepared.plan.checksum != prepared.template_sha256
    assert all(a.data is not None for a in prepared.inputs)


@pytest.mark.parametrize(
    "fault",
    [
        "template_sha",
        "before_sha",
        "before_artifact",
        "intent",
        "nonce",
        "boot",
        "source",
        "clock",
        "wall_bool",
        "source_helper",
    ],
)
def test_binding_drift_refuses_before_producing_setup(monkeypatch, fault):
    x = fixture_inputs(monkeypatch)
    start = life.Clock(BOOT, 100, 10**18)
    if fault == "before_sha":
        x.before += b"changed"
    if fault == "before_artifact":
        x.artifacts["capture"] = dict(x.artifacts["capture"], sha256="9" * 64)
    if fault == "intent":
        x.expected["outer_intent_sha256"] = "8" * 64
    if fault == "nonce":
        x.expected["nonce"] = "2" * 32
    if fault == "boot":
        start = life.Clock("other-boot", 100, 10**18)
    if fault == "clock":
        start = life.Clock(BOOT, float("nan"), 10**18)
    if fault == "wall_bool":
        start = life.Clock(BOOT, 100, True)
    if fault == "source_helper":
        x.plan["source_pins"] = [
            p
            for p in x.plan["source_pins"]
            if not p["path"].endswith("strength_freshness_cpu_completion.py")
        ]
    if fault == "source":
        unit = next(iter(x.plan["units"].values()))
        x.sources[unit["before"]["unit"]["path"]] += b"changed"
    with pytest.raises((q.Refusal, life.Refusal)):
        if fault == "template_sha":
            install.prepare(
                q.encode(x.plan),
                "f" * 64,
                start,
                x.sources,
                before_execution=x.before,
                expected_before=x.expected,
                before_evidence={},
                approved_outer_intent_sha256="7" * 64,
            )
        else:
            render(x, start)


def test_producer_completion_refusal_cannot_be_bypassed(monkeypatch):
    x = fixture_inputs(monkeypatch)

    def no_completion(*args, **kwargs):
        raise ValueError("guardian remains alive")

    monkeypatch.setattr(install.completion, "validate_before", no_completion)
    with pytest.raises(ValueError, match="guardian remains"):
        render(x)


@pytest.mark.parametrize(
    "role,duration", [("dispatcher", 330), ("observer", 510), ("publisher", 530)]
)
def test_nominal_runtime_that_cannot_fit_slow_setup_refuses(
    monkeypatch, role, duration
):
    x = fixture_inputs(monkeypatch)
    name, unit = next((n, v) for n, v in x.plan["units"].items() if v["role"] == role)
    oldpin = unit["before"]["unit"]
    raw = x.sources[oldpin["path"]].replace(
        b"RuntimeMaxSec=30s", f"RuntimeMaxSec={duration}s".encode()
    )
    pin = artifact(oldpin["path"], raw)
    x.sources[pin["path"]] = raw
    unit["before"]["unit"] = unit["after"]["unit"] = pin
    x.plan["files"][unit["installed_path"]][0]["source"] = pin
    with pytest.raises(q.Refusal, match="setup-(role-lifetime|dispatcher-work)"):
        render(x)


def test_support_scenario_must_be_real_fixed_scope_not_only_a_matching_hash(
    monkeypatch,
):
    x = fixture_inputs(monkeypatch)
    guard = x.plan["units"][x.plan["bindings"]["support_guard"]]
    pin = guard["payload"]["scenario"]
    scenario = json.loads(x.sources[pin["path"]])
    scenario["env_target"] = "/etc/environment"
    raw = q.encode(scenario)
    guard["payload"]["scenario"] = artifact(pin["path"], raw)
    x.sources[pin["path"]] = raw
    with pytest.raises(q.Refusal, match="support-ordered-fault-targets"):
        render(x)


@pytest.mark.parametrize("fault", ["missing", "mode", "source", "bytes"])
def test_after_only_environment_must_have_exact_registered_variant(monkeypatch, fault):
    x = fixture_inputs(monkeypatch)
    env = x.plan["units"][x.plan["bindings"]["barrier"]]["after"]["environment_files"][
        0
    ]
    variants = x.plan["files"][env["path"]]
    if fault == "missing":
        variants.pop()
    if fault == "mode":
        variants[-1]["mode"] = 0o640
    if fault == "source":
        variants[-1]["source"] = dict(
            variants[-1]["source"], path=str(INPUT / "unrelated.env")
        )
    if fault == "bytes":
        # Preserve the registered descriptor but remove its actual source bytes.
        del x.sources[env["source_path"]]
    with pytest.raises(
        q.Refusal, match="setup-(environment-variant-unregistered|input-byte-pin)"
    ):
        render(x)


@pytest.mark.parametrize(
    "path",
    [
        "plan.json/child",
        "anchor.json/child",
        "before.json",
        "before.json/child",
        "runtime-qualification.json",
        "runtime-qualification.json/child",
    ],
)
def test_source_cannot_shadow_generated_or_preservation_paths(monkeypatch, path):
    x = fixture_inputs(monkeypatch)
    unit = next(v for v in x.plan["units"].values() if v["role"] == "observer")
    old = unit["before"]["unit"]
    pin = dict(old, path=str(INPUT / path))
    unit["before"]["unit"] = unit["after"]["unit"] = pin
    x.plan["files"][unit["installed_path"]][0]["source"] = pin
    x.sources[pin["path"]] = x.sources[old["path"]]
    with pytest.raises(
        q.Refusal,
        match="setup-(readonly-path-alias|artifact-path-prefix|generated-path-alias)",
    ):
        render(x)


def test_regular_input_source_cannot_be_another_sources_directory(monkeypatch):
    x = fixture_inputs(monkeypatch)
    for role, suffix in [("observer", "a"), ("publisher", "a/child")]:
        unit = next(v for v in x.plan["units"].values() if v["role"] == role)
        old = unit["before"]["unit"]
        pin = dict(old, path=str(INPUT / suffix))
        unit["before"]["unit"] = unit["after"]["unit"] = pin
        x.plan["files"][unit["installed_path"]][0]["source"] = pin
        x.sources[pin["path"]] = x.sources[old["path"]]
    with pytest.raises(q.Refusal, match="setup-artifact-path-prefix"):
        render(x)


def test_real_completion_validator_integrates_with_renderer_using_synthetic_facts(
    monkeypatch,
):
    from test_strength_freshness_cpu_completion import valid_completion

    real_validator = install.completion.validate_before
    x = fixture_inputs(monkeypatch)
    monkeypatch.setattr(install.completion, "validate_before", real_validator)
    raw, expected, now, evidence, _ = valid_completion()

    def translated(value):
        if isinstance(value, dict):
            return {k: translated(v) for k, v in value.items()}
        if isinstance(value, list):
            return [translated(v) for v in value]
        if value == "boot":
            return BOOT
        if value == "a" * 32:
            return NONCE
        if isinstance(value, str):
            return value.replace("/input/", str(INPUT) + "/")
        return value

    c = install.completion
    proof = translated(c.parse(raw))
    expected = translated(expected)
    now = translated(now)
    evidence = {
        name: c.encoded(translated(c.parse(data))) for name, data in evidence.items()
    }
    for name, data in evidence.items():
        proof["evidence_pins"][name].update(sha256=c.sha(data), bytes=len(data))
    raw = c.encoded(proof)
    for artifact_name, plan_name in [
        ("capture", "before"),
        ("request", "before_request"),
        ("receipt", "before_receipt"),
    ]:
        x.plan["preservation"][plan_name] = expected["artifacts"][artifact_name]
    x.plan["preservation"]["before_execution"] = artifact(
        INPUT / "before-execution.json", raw
    )
    data = q.encode(x.plan)
    prepared = install.prepare(
        data,
        q.sha(data),
        life.Clock(BOOT, now["monotonic_ns"] / 10**9, now["wall_ns"]),
        x.sources,
        before_execution=raw,
        expected_before=expected,
        before_evidence=evidence,
        approved_outer_intent_sha256=expected["outer_intent_sha256"],
    )
    assert prepared.anchor.started_monotonic == 115
    assert prepared.plan.value["preservation"]["before_execution"]["sha256"] == c.sha(
        raw
    )
    assert prepared.start_order[0].endswith("-watchdog.timer")


def test_fractional_anchor_projection_matches_both_completion_and_outer_budget(
    monkeypatch,
):
    import math
    from scripts import strength_freshness_cpu_outer as outer

    x = fixture_inputs(monkeypatch)
    # This value differs by one nanosecond under decimal-string vs float-product
    # projection. Stored Anchor consumers use the latter; no tolerance is added.
    start = life.Clock(BOOT, 551286.627352024, 10**18)
    prepared = render(x, start)
    anchor_ns = math.floor(prepared.anchor.started_monotonic * 10**9)
    assert anchor_ns == 551286627352023
    assert x.calls[0][2]["monotonic_ns"] == anchor_ns
    launch = outer.Clock(BOOT, anchor_ns + 541 * 10**9, 10**18 + 541 * 10**9)
    expected = install.completion.after_budget(
        asdict(launch), prepared.anchor.as_dict()
    )
    actual = outer.PhaseBudget.after(outer.Clock(BOOT, anchor_ns, 10**18), launch)
    assert actual.compact() == expected
