from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import strength_freshness_cpu_collect_identity as m
from scripts import strength_freshness_cpu_readonly as ro

SHA = "a" * 64
MONITOR = "monitor.service"
TIMER = "report.timer"
SECRET = "private-sentinel-value"


def fingerprint(size=10, inode=10):
    return dict(
        device=os.makedev(8, 1),
        inode=inode,
        mode=0o644,
        uid=0,
        gid=0,
        bytes=size,
        mtime_ns=10,
        ctime_ns=11,
    )


class FakeIO:
    """Explicit memory-only observations; never falls back to host paths/proc."""

    def __init__(self, reg):
        self.scope = m.scope_from(reg["scope"])
        self.data: dict[str, Any] = {key: b"{}" for key in self.scope.files}
        self.data["source"] = b"control source"
        self.refs = copy.deepcopy(reg["cached_references"])
        self.calls = []
        self.links: dict[str, Any] = {}
        self.props: dict[str, Any] = {}
        self.race = False
        self.bad_raw = False

    def observation(self, operation, subject, value, raw=None, **metadata):
        if raw is None:
            raw = m.encoded(value)
        clock = dict(boot_id="boot", monotonic_ns=100, wall_ns=10_000)
        return ro.Observation(
            value,
            dict(
                operation=operation,
                subject=subject,
                read_start=clock,
                read_end=clock,
                raw=dict(sha256=m.sha(raw), bytes=len(raw)),
                **metadata,
            ),
        )

    def read(self, key):
        self.calls.append(("read", key))
        value = self.data[key]
        if isinstance(value, Exception):
            raise value
        st = fingerprint(len(value))
        after = {**st, "ctime_ns": st["ctime_ns"] + int(self.race)}
        obs = self.observation(
            "read", key, value, value, stat_before=st, stat_after=after
        )
        if self.bad_raw:
            obs.audit["raw"]["sha256"] = SHA
        return obs

    def stat_cached(self, key):
        self.calls.append(("stat", key))
        ref = self.refs[key]
        item = self.scope.cached[key]
        return self.observation(
            "stat-cached",
            key,
            dict(
                literal=item.literal,
                resolved=item.resolved,
                literal_stat=ref["literal_stat"],
                resolved_stat=ref["resolved_stat"],
                link=None,
            ),
            content_hashed=False,
        )

    def boot_links(self, name):
        return self.observation("boot-links", name, self.links.get(name, []))

    def query(self, kind, name):
        assert kind == "unit"
        raw = "\n".join(f"{k}={v}" for k, v in self.props[name].items())
        return self.observation("unit", name, raw, raw.encode())

    def clock(self):
        return self.observation(
            "clock", "collector", dict(boot_id="boot", monotonic_ns=100, wall_ns=10000)
        )


def identity_fixture() -> tuple[dict[str, Any], FakeIO, dict[str, Any]]:
    """Return (registration, explicit FakeIO, private process) for composition."""
    roles = {"controller", "coordinator", *m.WORKERS}
    names = {m.RUNTIME: "service", MONITOR: "service", TIMER: "timer"}
    metadata = "run continuation profile_authority profile run_source release_source source_manifest coordinator champion".split()
    cohorts = {
        f"cohort-{i}": {
            "key": f"cohort-{i}",
            "worker": f"cohort-{i}",
            "parent_role": f"actor-gpu-{1 + i % 6}",
        }
        for i in range(24)
    }
    files = {
        k: {"path": "/run/" + k}
        for k in [*metadata, *m.WORKERS, *cohorts, "source", "env"]
    }
    units = {}
    for i, (name, kind) in enumerate(names.items()):
        key = f"fragment-{i}"
        files[key] = {"path": "/etc/systemd/system/" + name}
        units[name] = dict(
            kind="runtime"
            if name == m.RUNTIME
            else "timer"
            if kind == "timer"
            else "long_running",
            fragment=key,
            dropins=[],
            environment_files=[],
            owned_links=[],
        )
    cached = {
        "python": {"literal": "/release/bin/python", "resolved": "/bin/python"},
        "native": {
            "literal": "/release/star_native.so",
            "resolved": "/release/star_native.so",
        },
    }
    refs = {
        key: dict(
            sha256=SHA,
            qualification_sha256="b" * 64,
            literal_stat=fingerprint(),
            resolved_stat=fingerprint(),
        )
        for key in cached
    }
    argv = b"/release/bin/python\0-m\0startrain.worker\0"
    env = b"PYTHONPATH=/release\0TOKEN=" + SECRET.encode() + b"\0"
    spec = dict(
        interpreter="python",
        cwd="/run/training",
        argv_sha256=m.sha(argv),
        argv_bytes=len(argv),
        approved_template_id=SHA,
        environment={
            "PYTHONPATH": dict(
                present=True, value_sha256=m.sha(b"/release"), value_bytes=8
            )
        },
        native_keys=["native"],
        entrypoint_contract_sha256=SHA,
        restart_counter_scope="systemd-unit:NRestarts",
    )
    reg: dict[str, Any] = dict(
        format=m.FORMAT,
        schema_version=1,
        encoding_contract_sha256=m.CONTRACT,
        timer_addendum_sha256=m.facts.TIMER_ADDENDUM_SHA256,
        timer_environment_addendum_sha256=m.facts.TIMER_ENVIRONMENT_ADDENDUM_SHA256,
        scope=dict(
            units=names,
            targets=["multi-user.target"],
            files=files,
            tails={"metrics": {"path": "/run/metrics"}},
            cached=cached,
        ),
        policy=dict(
            static=dict(
                runtime_name=m.RUNTIME,
                source_commit=m.preservation.R3_SOURCE_COMMIT,
                source_manifest_sha256=SHA,
            ),
            workers=sorted(m.WORKERS),
            cohorts=cohorts,
            expected_processes={role: {} for role in roles},
            learner_birth_upper_ns=50,
        ),
        keys={**{k: k for k in metadata}, "metrics": "metrics"},
        units=units,
        heartbeats={k: k for k in m.WORKERS},
        cohorts=cohorts,
        origins={
            role: {
                **copy.deepcopy(spec),
                "restart_counter_scope": "coordinator-worker:restart_count"
                if role in m.WORKERS
                else "systemd-unit:NRestarts",
            }
            for role in {*roles, "monitor"}
        },
        auxiliary_policy={},
        cached_references=refs,
        source_pins={"source": dict(sha256=m.sha(b"control source"), bytes=14)},
        birth_reference=dict(
            boot_id="boot",
            qualification_sha256=SHA,
            offset_lower_ns=1,
            offset_upper_ns=2,
            max_bracket_ns=1,
            bounds={role: 50 for role in roles},
        ),
        boot=dict(default_target="multi-user.target", edges=[]),
        empty_property_rules={},
        counter_scopes=dict(
            controller="owning-unit-NRestarts",
            coordinator="owning-unit-NRestarts",
            workers="coordinator-worker-restart_count",
            monitor="owning-unit-NRestarts",
        ),
    )
    io = FakeIO(reg)
    for name in names:
        props: dict[str, str] = dict.fromkeys(
            m.facts.TIMER_STATIC if name == TIMER else m.facts.SERVICE_STATIC, ""
        )
        props.update(
            Id=name,
            Names=name,
            LoadState="loaded",
            FragmentPath="/etc/systemd/system/" + name,
            DropInPaths="",
            NeedDaemonReload="no",
            UnitFileState="disabled",
        )
        if name != TIMER:
            props.update(
                Environment="TOKEN=" + SECRET,
                PassEnvironment="",
                UnsetEnvironment="",
                EnvironmentFiles="",
            )
        io.props[name] = props
    process = dict(
        pid=50,
        start_ticks=1,
        ppid=1,
        cgroup="/system.slice/" + m.RUNTIME,
        exe="/bin/python",
        cwd="/run/training",
        cmdline=argv,
        environ=env,
        maps=b"100-200 r-xp 0 08:01 10 /release/star_native.so\n",
    )
    return reg, io, process


def collector(reg=None, io=None):
    if reg is None:
        reg, io, _ = identity_fixture()
    return m.IdentityCollector(reg, cast(ro.ReadOnlyIO, io))


def test_real_public_api_composition_and_private_audit():
    reg, io, process = identity_fixture()
    c = collector(reg, io)
    assert c.reg == reg and c.reg is not reg
    assert c.read_json("run") == {}
    assert c.clock()["boot_id"] == "boot"
    assert c.verify_source_pins()["source"]["bytes"] == 14
    assert set(c.verify_cached_references()) == {"python", "native"}
    for role in reg["origins"]:
        assert len(c.origin(role, process)) == 64
    for name in reg["units"]:
        result = c.unit_static(name, c.unit(name), {})
        assert result["enabled"] is False
        assert all(len(result[k]) == 64 for k in result if k.endswith("sha256"))
    public = json.dumps(c.audit)
    assert SECRET not in public and "/release\u0000" not in public
    assert all(
        call[0] != "read" or call[1] not in {"native", "python"} for call in io.calls
    )


@pytest.mark.parametrize(
    "role",
    [
        "learner",
        "arena-promotion",
        "actor-cpu-ring4",
        "actor-gpu-1",
        "controller",
        "coordinator",
        "monitor",
    ],
)
def test_restart_scope_is_bound_by_role(role):
    reg, io, _ = identity_fixture()
    reg["origins"][role]["restart_counter_scope"] = (
        "systemd-unit:NRestarts"
        if role in m.WORKERS
        else "coordinator-worker:restart_count"
    )
    with pytest.raises(m.CollectionRefusal, match="origin-restart-counter-scope"):
        collector(reg, io)
    assert not io.calls


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["source_pins"]["source"].update(bytes=True),
        lambda r: r["source_pins"]["source"].update(sha256="bad"),
        lambda r: r["birth_reference"]["bounds"].update(learner=51),
        lambda r: r["birth_reference"].update(offset_lower_ns=3),
        lambda r: r["cached_references"]["python"]["literal_stat"].update(inode=True),
        lambda r: r["keys"].update(champion="unregistered"),
        lambda r: r["origins"]["learner"].update(native_keys=["unknown"]),
        lambda r: r["origins"]["learner"]["environment"]["PYTHONPATH"].update(
            present="yes"
        ),
        lambda r: r["empty_property_rules"].update(Environment="unqualified"),
        lambda r: r["units"][TIMER].update(kind="oneshot"),
        lambda r: r["boot"].update(default_target="unregistered.target"),
    ],
)
def test_bad_registration_refuses_before_io(change):
    reg, io, _ = identity_fixture()
    change(reg)
    with pytest.raises((m.CollectionRefusal, ro.ReadRefusal)):
        collector(reg, io)
    assert not io.calls


@pytest.mark.parametrize("drift", ["race", "raw", "source", "cache"])
def test_file_and_cached_provenance_drift(drift):
    reg, io, _ = identity_fixture()
    c = collector(reg, io)
    if drift == "race":
        io.race = True
    elif drift == "raw":
        io.bad_raw = True
    elif drift == "source":
        io.data["source"] = b"changed"
    else:
        io.refs["python"]["resolved_stat"]["inode"] += 1
    with pytest.raises(m.CollectionRefusal):
        c.verify_cached_references() if drift == "cache" else c.verify_source_pins()


def test_environment_files_present_missing_and_order():
    reg, io, _ = identity_fixture()
    reg["units"][m.RUNTIME]["environment_files"] = [
        dict(key="env", ignore_missing=True)
    ]
    props = io.props[m.RUNTIME]
    props["EnvironmentFiles"] = "/run/env (ignore_errors=yes)"
    c = collector(reg, io)
    present = c.environment(m.RUNTIME, props)
    io.data["env"] = FileNotFoundError()
    assert c.environment(m.RUNTIME, props) != present
    props["EnvironmentFiles"] = "/run/env (ignore_errors=no)"
    with pytest.raises(m.CollectionRefusal, match="environment-file-options"):
        c.environment(m.RUNTIME, props)


def test_required_missing_and_explicit_property_absence():
    reg, io, _ = identity_fixture()
    reg["units"][m.RUNTIME]["environment_files"] = [
        dict(key="env", ignore_missing=False)
    ]
    io.props[m.RUNTIME]["EnvironmentFiles"] = "/run/env (ignore_errors=no)"
    io.data["env"] = FileNotFoundError()
    with pytest.raises(m.CollectionRefusal, match="required-env-file-missing"):
        collector(reg, io).environment(m.RUNTIME, io.props[m.RUNTIME])
    reg, io, _ = identity_fixture()
    del io.props[m.RUNTIME]["Environment"]
    with pytest.raises(m.CollectionRefusal, match="unqualified-property-absence"):
        collector(reg, io).environment(m.RUNTIME, io.props[m.RUNTIME])
    reg["empty_property_rules"]["Environment"] = "systemd-255-empty-Environment"
    assert len(collector(reg, io).environment(m.RUNTIME, io.props[m.RUNTIME])) == 64


def test_actual_io_boot_link_field_mapping_and_drift():
    reg, io, _ = identity_fixture()
    link = dict(
        path="/etc/systemd/system/multi-user.target.wants/" + m.RUNTIME,
        literal_target="../" + m.RUNTIME,
        resolved_target="/etc/systemd/system/" + m.RUNTIME,
    )
    reg["units"][m.RUNTIME]["owned_links"] = [link]
    io.links[m.RUNTIME] = [
        dict(
            path=link["path"],
            literal=link["literal_target"],
            resolved=link["resolved_target"],
            stat={},
        )
    ]
    io.props[m.RUNTIME]["UnitFileState"] = "enabled"
    c = collector(reg, io)
    assert c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {})["enabled"]
    io.links[m.RUNTIME][0]["path"] += "-alias"
    with pytest.raises(m.CollectionRefusal, match="owned-boot-links-drift"):
        c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {})


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(cmdline=b"secret-unapproved\0"),
        lambda p: p.update(cwd="/different"),
        lambda p: p.update(environ=p["environ"] + b"LD_PRELOAD=secret\0"),
        lambda p: p.update(environ=p["environ"] + b"PYTHONPATH=secret\0"),
        lambda p: p.update(environ=b"PYTHONPATH=secret\0"),
        lambda p: p.update(environ=b"\xff=secret\0"),
        lambda p: p.update(environ=b"TOKEN=secret"),
        lambda p: p.update(maps=None),
        lambda p: p.update(maps=b""),
        lambda p: p.update(maps=b"100-200 r-xp 0 08:01 99 /release/star_native.so\n"),
        lambda p: p.update(
            maps=b"100-200 r-xp 0 08:01 10 /release/star_native.so (deleted)\n"
        ),
        lambda p: p.update(maps=b"100-200 r-xp 0 bad 10 /release/star_native.so\n"),
    ],
)
def test_private_origin_refusals_do_not_expose_values(change):
    reg, io, process = identity_fixture()
    change(process)
    c = collector(reg, io)
    with pytest.raises(m.CollectionRefusal) as failure:
        c.origin("learner", process)
    assert "secret" not in str(failure.value) and SECRET not in json.dumps(c.audit)


def test_absent_and_empty_import_variable_are_different():
    reg, io, p = identity_fixture()
    reg["origins"]["learner"]["environment"]["LD_LIBRARY_PATH"] = dict(
        present=False, value_sha256=None, value_bytes=0
    )
    c = collector(reg, io)
    c.origin("learner", p)
    p["environ"] += b"LD_LIBRARY_PATH=\0"
    with pytest.raises(m.CollectionRefusal, match="environment-origin-drift"):
        c.origin("learner", p)


@pytest.mark.parametrize(
    "raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"{bad-private}", b"[]"]
)
def test_json_refusals_are_fixed(raw):
    with pytest.raises(m.CollectionRefusal) as failure:
        m.strict_json(raw)
    assert "private" not in str(failure.value)


def test_fresh_interpreter_import_has_no_training_or_accelerator_imports():
    training = Path(__file__).resolve().parents[1]
    code = 'import sys;sys.path.insert(0,sys.argv[1]);import scripts.strength_freshness_cpu_collect_identity;assert not any(x.split(".")[0] in {"torch","numpy","deltreltrain","startrain","deltrel_native","star_native"} for x in sys.modules)'
    result = subprocess.run(
        [sys.executable, "-S", "-E", "-c", code, str(training)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize(
    "key", ["timer_addendum_sha256", "timer_environment_addendum_sha256"]
)
def test_both_timer_contract_addenda_are_explicit(key):
    reg, io, _ = identity_fixture()
    reg[key] = "f" * 64
    with pytest.raises(m.CollectionRefusal, match="encoding-contract"):
        collector(reg, io)
    assert io.calls == []


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda p: p.update(NeedDaemonReload="yes"), "daemon-reload"),
        (lambda p: p.update(FragmentPath="/foreign"), "unit-file-binding"),
        (lambda p: p.update(DropInPaths="/foreign/drop.conf"), "unit-file-binding"),
        (lambda p: p.pop("ExecStart"), "missing-unit-property"),
    ],
)
def test_unit_static_unknown_or_mixed_refusal(change, expected):
    reg, io, _ = identity_fixture()
    props = io.props[m.RUNTIME]
    change(props)
    with pytest.raises((m.CollectionRefusal, m.facts.FactViolation), match=expected):
        collector(reg, io).unit_static(m.RUNTIME, props, {})


def test_static_exec_flag_is_preserved_and_dynamic_fields_are_not():
    reg, io, _ = identity_fixture()
    c = collector(reg, io)
    props = io.props[m.RUNTIME]
    props["ExecStart"] = (
        "{ path=/bin/python ; argv[]=python secret ; ignore_errors=no ; pid=1 ; status=0 }"
    )
    first = c.unit_static(m.RUNTIME, props, {})
    props["ExecStart"] = props["ExecStart"].replace("pid=1", "pid=999")
    assert c.unit_static(m.RUNTIME, props, {}) == first
    props["ExecStart"] = props["ExecStart"].replace(
        "ignore_errors=no", "ignore_errors=yes"
    )
    assert (
        c.unit_static(m.RUNTIME, props, {})["definition_sha256"]
        != first["definition_sha256"]
    )


def test_registered_boot_edge_must_be_observed():
    reg, io, _ = identity_fixture()
    reg["scope"]["targets"].append("graphical.target")
    reg["boot"] = dict(
        default_target="graphical.target",
        edges=[
            dict(
                **{"from": "graphical.target", "to": "multi-user.target"},
                relation="Wants",
            )
        ],
    )
    io.scope = m.scope_from(reg["scope"])
    c = collector(reg, io)
    targets = {"graphical.target": {"Wants": "multi-user.target"}}
    assert (
        len(c.unit_static(m.RUNTIME, io.props[m.RUNTIME], targets)["boot_links_sha256"])
        == 64
    )
    with pytest.raises(m.CollectionRefusal, match="missing-boot-proof-edge"):
        c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {})


def test_native_multiple_segments_do_not_duplicate_origin_identity():
    reg, io, process = identity_fixture()
    c = collector(reg, io)
    original = c.origin("learner", process)
    process["maps"] *= 2
    assert c.origin("learner", process) == original


def test_registry_copy_and_all_cache_references_checked_without_payload_reads():
    reg, io, _ = identity_fixture()
    c = collector(reg, io)
    reg["source_pins"]["source"]["sha256"] = "f" * 64
    c.verify_source_pins()
    io.calls.clear()
    c.verify_cached_references()
    assert io.calls == [("stat", "native"), ("stat", "python")]
    assert all(
        a.get("content_hashed") is False
        for a in c.audit
        if a["operation"] == "stat-cached"
    )


@pytest.mark.parametrize("raw", ["MissingEquals", "Id=x\nId=y"])
def test_unit_property_parser_refuses_ambiguous_records(raw):
    with pytest.raises(m.CollectionRefusal):
        m.properties(raw)


@pytest.mark.parametrize(
    "raw,code",
    [
        (b'{"x":1e999}', "nonfinite-json"),
        (b'{"x":-1e999}', "nonfinite-json"),
        (b'{"x":' + b"[" * 65 + b"0" + b"]" * 65 + b"}", "json-depth"),
        (b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}", "invalid-json"),
    ],
)
def test_strict_json_numeric_overflow_and_nesting_are_sanitized(raw, code):
    with pytest.raises(m.CollectionRefusal, match="^" + code + "$"):
        m.strict_json(raw)


def test_strict_json_preserves_finite_numeric_values():
    assert m.strict_json(b'{"x":[1.25,-1e3,0,1e-300]}') == {
        "x": [1.25, -1000.0, 0, 1e-300]
    }


def test_strict_json_accepts_only_runtime_utf8_encoding():
    with pytest.raises(m.CollectionRefusal, match="^invalid-json$"):
        m.strict_json('{"x":1}'.encode("utf-16"))
    assert m.strict_json('{"x":"résumé"}'.encode("utf-8")) == {"x": "résumé"}


@pytest.mark.parametrize("property_name", m.OPTIONAL_EMPTY_EXEC)
def test_missing_optional_exec_requires_registered_rule(property_name):
    reg, io, _ = identity_fixture()
    del io.props[m.RUNTIME][property_name]
    with pytest.raises(
        m.CollectionRefusal, match="unqualified-execution-property-absence"
    ):
        collector(reg, io).unit_static(m.RUNTIME, io.props[m.RUNTIME], {})


@pytest.mark.parametrize("property_name", m.OPTIONAL_EMPTY_EXEC)
def test_registered_empty_exec_retains_observed_absence_and_semantics(property_name):
    reg, io, _ = identity_fixture()
    baseline = collector(reg, io).unit_static(m.RUNTIME, io.props[m.RUNTIME], {})
    reg["empty_property_rules"][property_name] = "systemd-255-empty-" + property_name
    del io.props[m.RUNTIME][property_name]
    original = copy.deepcopy(io.props[m.RUNTIME])
    c = collector(reg, io)
    assert c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {}) == baseline
    assert io.props[m.RUNTIME] == original and property_name not in io.props[m.RUNTIME]
    assert c.derivations["omitted_execution_properties"] == [
        {
            "unit": m.RUNTIME,
            "parsed_properties_sha256": m.sha(m.encoded(original)),
            "observed_present": False,
            "normalization_rules": {
                property_name: "systemd-255-empty-" + property_name
            },
        }
    ]
    assert SECRET not in json.dumps(c.derivations)


@pytest.mark.parametrize("property_name", m.OPTIONAL_EMPTY_EXEC)
def test_registered_rule_never_erases_nonempty_exec(property_name):
    reg, io, _ = identity_fixture()
    reg["empty_property_rules"][property_name] = "systemd-255-empty-" + property_name
    baseline = collector(reg, io).unit_static(m.RUNTIME, io.props[m.RUNTIME], {})
    io.props[m.RUNTIME][property_name] = (
        "{ path=/bin/true ; argv[]=true ; ignore_errors=no ; pid=0 ; status=0 }"
    )
    c = collector(reg, io)
    assert (
        c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {})["definition_sha256"]
        != baseline["definition_sha256"]
    )
    assert c.derivations == {}


@pytest.mark.parametrize(
    "key,value",
    [
        ("ExecStart", "systemd-255-empty-ExecStart"),
        ("ExecReload", "systemd-255-empty-ExecReload"),
        ("ExecStartPre", "systemd-255-empty-ExecStop"),
        ("ExecStop", "systemd-256-empty-ExecStop"),
    ],
)
def test_no_required_unknown_or_mismatched_exec_rule(key, value):
    reg, io, _ = identity_fixture()
    reg["empty_property_rules"][key] = value
    with pytest.raises(m.CollectionRefusal, match="empty-property-rules"):
        collector(reg, io)
    assert io.calls == []


def test_optional_rules_do_not_fill_missing_required_execstart():
    reg, io, _ = identity_fixture()
    reg["empty_property_rules"].update(
        {key: "systemd-255-empty-" + key for key in m.OPTIONAL_EMPTY_EXEC}
    )
    del io.props[m.RUNTIME]["ExecStart"]
    with pytest.raises(m.CollectionRefusal, match="missing-unit-property"):
        collector(reg, io).unit_static(m.RUNTIME, io.props[m.RUNTIME], {})


def test_actual_three_property_omission_is_not_a_timer_environment_rule():
    reg, io, _ = identity_fixture()
    reg["empty_property_rules"].update(
        {key: "systemd-255-empty-" + key for key in m.OPTIONAL_EMPTY_EXEC}
    )
    for key in m.OPTIONAL_EMPTY_EXEC:
        del io.props[m.RUNTIME][key]
    c = collector(reg, io)
    assert c.unit_static(m.RUNTIME, io.props[m.RUNTIME], {})
    assert (
        c.derivations["omitted_execution_properties"][0]["normalization_rules"]
        == reg["empty_property_rules"]
    )
    count = len(c.derivations["omitted_execution_properties"])
    c.unit_static(TIMER, io.props[TIMER], {})
    assert len(c.derivations["omitted_execution_properties"]) == count
