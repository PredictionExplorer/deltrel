"""Helper admission/ABI checks only; no site-enabled or target execution."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import copy
import subprocess
import sys

import pytest

from scripts import strength_freshness_cpu_install_helper as m
from scripts import strength_freshness_cpu_outer as outer


def pin(path, raw=b"x"):
    return {"path": path, "sha256": m.c.sha(raw), "bytes": len(raw)}


def fixture():
    supervisor = outer.ProcessIdentity(
        100, 10, 1, 100, 100, 0, "boot", "/existing.scope", 99
    )
    operator = replace(supervisor, pid=200, start_ticks=20, ppid=100, pgid=200, sid=200)
    start = {"boot_id": "boot", "monotonic_ns": 100 * 10**9, "wall_ns": 10**18}
    spec = {
        "input_root": "/input",
        "template_root": "/templates",
        "blueprint": pin("/templates/blueprint.json"),
        "sources": pin("/templates/sources.json"),
    }
    intent = {"nonce": "a" * 32, "boot_id": "boot", "installer": spec}
    raw = m.c.encoded(intent)
    parent = outer.Handle(operator, 55)
    clock = outer.Clock("boot", 110 * 10**9, 10**18 + 10 * 10**9)
    kernel = SimpleNamespace(
        identity=lambda pid: supervisor if pid == 100 else operator,
        clock=lambda: clock,
        exited=lambda fd: False,
    )
    admission = SimpleNamespace(
        role="helper",
        intent=intent,
        raw_intent=raw,
        approved_intent_sha256=m.c.sha(raw),
        parent=parent,
        kernel=kernel,
        outer={"source_pins": [], "input_root": "/input", "control_root": "/control"},
    )
    bundle = {
        "phase": "before",
        "completion_pin": pin("/input/before-execution.json"),
        "artifacts": {},
        "evidence_pins": {},
    }
    start_raw = m.c.encoded(
        {
            "format": m.START,
            "schema_version": 1,
            "outer_intent_sha256": admission.approved_intent_sha256,
            "before_execution_pin": bundle["completion_pin"],
            "clock": start,
        }
    )
    value = {
        "format": m.REQUEST,
        "schema_version": 1,
        "operation": "prepare-install",
        "intent_pin": pin("/input/intent.json", raw),
        "enclosing": {"operator": asdict(operator), "supervisor": asdict(supervisor)},
        "blueprint_pin": spec["blueprint"],
        "sources_pin": spec["sources"],
        "before_bundle": bundle,
        "original_start": start,
        "start_pin": pin("/input/dummy-start.json", start_raw),
        "prepared_run": None,
    }
    ack = outer.dummy_start_ack(
        outer_intent_sha256=admission.approved_intent_sha256,
        nonce=intent["nonce"],
        start_pin=value["start_pin"],
        before_execution_pin=bundle["completion_pin"],
        enclosing=value["enclosing"],
    )
    value["ack_pin"] = pin("/input/dummy-start.ack.json", m.c.encoded(ack))
    frame = {
        "operation": "prepare-install",
        "parent": asdict(operator),
        "phase_start": asdict(clock),
        "work_deadline": {"monotonic_ns": 158 * 10**9, "wall_ns": 10**18 + 58 * 10**9},
    }
    return value, admission, frame


def test_exact_request_matches_original_intent_parent_start_and_fixed_pins():
    value, admission, frame = fixture()
    assert (
        m.validate_request(m.c.encoded(value), admission, frame, "/input/intent.json")
        == value
    )


@pytest.mark.parametrize(
    "fault",
    [
        "callback",
        "operation",
        "frame-operation",
        "intent",
        "blueprint",
        "sources",
        "parent",
        "supervisor",
        "start-pin",
        "boot",
        "renewed-start",
        "prepared",
        "role",
        "ack",
    ],
)
def test_unbound_or_expanding_request_refuses_before_backend(fault):
    value, admission, frame = fixture()
    if fault == "callback":
        value["callback"] = "os.system"
    elif fault == "operation":
        value["operation"] = "shell"
    elif fault == "frame-operation":
        frame["operation"] = "audit"
    elif fault == "intent":
        value["intent_pin"]["sha256"] = "f" * 64
    elif fault in {"blueprint", "sources"}:
        value[fault + "_pin"] = dict(value[fault + "_pin"], path="/foreign/input")
    elif fault == "parent":
        frame["parent"]["start_ticks"] += 1
    elif fault == "supervisor":
        value["enclosing"]["supervisor"]["start_ticks"] += 1
    elif fault == "start-pin":
        value["start_pin"]["path"] = "/input/later-start.json"
    elif fault == "boot":
        value["original_start"]["boot_id"] = "other"
    elif fault == "renewed-start":
        value["original_start"]["monotonic_ns"] += 20 * 10**9
    elif fault == "prepared":
        value["prepared_run"] = {}
    elif fault == "ack":
        value["ack_pin"]["sha256"] = "0" * 64
    else:
        admission.role = "operator"
    with pytest.raises((m.HelperRefusal, m.c.CompletionRefusal)):
        m.validate_request(m.c.encoded(value), admission, frame, "/input/intent.json")


@pytest.mark.parametrize("operation", ["inspect-cleanup", "audit", "prearm-cleanup"])
def test_other_operations_require_earlier_preparation_except_cleanup_unknown_outcome(
    operation,
):
    value, admission, frame = fixture()
    value["operation"] = frame["operation"] = operation
    if operation == "prearm-cleanup":
        assert (
            m.validate_request(
                m.c.encoded(value), admission, frame, "/input/intent.json"
            )["prepared_run"]
            is None
        )
    else:
        with pytest.raises(m.HelperRefusal, match="prepared-required"):
            m.validate_request(
                m.c.encoded(value), admission, frame, "/input/intent.json"
            )


class MemoryStore:
    def __init__(self, rows):
        self.rows = rows
        self.roots: tuple[Path, ...] = ()
        self.calls = []
        self.expired = False

    def check(self):
        if self.expired:
            raise m.HelperRefusal("fake-deadline")

    def read(self, expected, *, source=False, maximum=2**20):
        self.check()
        self.calls.append((copy.deepcopy(expected), source, maximum))
        raw = self.rows[expected["path"]]
        assert pin(expected["path"], raw) == expected
        assert len(raw) <= maximum
        return raw


def test_reader_allows_only_registered_source_or_closed_input_template_paths():
    _value, admission, frame = fixture()
    src = pin("/control/source.py", b"source")
    admission.outer["source_pins"] = [src]
    admission.reader = MemoryStore(
        {src["path"]: b"source", "/input/execution-evidence/proof.json": b"x"}
    )
    reader = m.Reader(admission, frame, {"plan": {"source_pins": [src]}})
    assert reader.read(src) == b"source"
    expected = pin("/input/execution-evidence/proof.json")
    assert reader.read(expected, root=Path("/input/execution-evidence")) == b"x"
    assert Path("/input/execution-evidence") in admission.reader.roots
    with pytest.raises(m.HelperRefusal, match="read-scope"):
        reader.read(pin("/control/unregistered.py"))
    with pytest.raises(m.HelperRefusal, match="proof-parent"):
        reader.read(expected, root=Path("/input"))
    admission.reader.expired = True
    with pytest.raises(m.HelperRefusal, match="deadline"):
        reader.read(src)


def test_parent_death_revokes_new_helper_io():
    _value, admission, frame = fixture()
    admission.reader = MemoryStore({})
    admission.kernel.exited = lambda fd: True
    reader = m.Reader(admission, frame, {"plan": {"source_pins": []}})
    with pytest.raises(m.HelperRefusal, match="parent-died"):
        reader.check()


def test_actual_import_origin_is_not_a_same_named_qualified_copy():
    _value, admission, frame = fixture()
    actual = Path(m.__file__).resolve()
    raw = actual.read_bytes()
    p = pin(str(actual), raw)
    admission.outer.update(control_root=str(actual.parents[1]), source_pins=[p])
    admission.reader = MemoryStore({p["path"]: raw})
    blueprint = {"plan": {"source_pins": [p]}}
    reader = m.Reader(admission, frame, blueprint)
    m._actual_sources(admission, blueprint, reader, [m])
    shadow = SimpleNamespace(__file__=str(actual.parent / "shadow" / actual.name))
    with pytest.raises(m.HelperRefusal, match="source-pin"):
        m._actual_sources(admission, blueprint, reader, [shadow])


def test_unqualified_entry_never_imports_site_backend_or_leaks_reason():
    code = r"""
import sys,types
from scripts import strength_freshness_cpu_install_helper as helper
def refuse(*args): raise ValueError("PRIVATE_UNAPPROVED_DETAIL")
sys.modules["scripts.strength_freshness_cpu_outer_runtime"]=types.SimpleNamespace(helper_entry_admission=refuse)
sys.argv=["helper","--operation","prepare-install","--authorization","/input/intent.json","--sha256","a"*64]
try: helper.main()
except SystemExit as e: assert e.code==1
else: raise AssertionError("unqualified launch")
assert "torch" not in sys.modules
assert "scripts.strength_freshness_cpu_install_backend" not in sys.modules
"""
    done = subprocess.run(
        [sys.executable, "-S", "-E", "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert done.returncode == 0, done.stderr
    assert "PRIVATE_UNAPPROVED_DETAIL" not in done.stdout + done.stderr
    assert '"reason":"helper-refused"' in done.stdout


def test_actual_helper_finalaudit_shape_is_accepted_by_real_facade(monkeypatch):
    from test_strength_freshness_cpu_install_runtime import (
        fixture as facade_fixture,
        prepare,
    )

    facade, executor, observed, _bundle = x = facade_fixture()
    prepared = prepare(x)
    observed.mono += 405 * 10**9
    observed.wall += 405 * 10**9
    facade.await_checked_cleanup()
    observed.mono += 180 * 10**9
    observed.wall += 180 * 10**9
    original = executor.call
    actual_pin = pin(
        prepared["cleanup_proof_root"] + "/cpu-scope-result-" + "a" * 32 + ".json"
    )

    def call(mode, payload, deadline_ns):
        value = m.c.parse(original(mode, payload, deadline_ns))
        assert mode == "audit"
        value["result"] = m.audit_result(
            actual_pin,
            plan_sha256=prepared["plan_pin"]["sha256"],
            anchor_sha256=prepared["anchor_pin"]["sha256"],
            proof_root=prepared["cleanup_proof_root"],
        )
        return m.c.encoded(value)

    monkeypatch.setattr(executor, "call", call)
    result = m.c.parse(facade.final_audit_and_retire())
    assert result["status"] == "passed_cpu_scope"
    assert result["evidence_pins"] == {"external_audit": actual_pin}
    assert facade.stage == "audited"
    with pytest.raises(m.HelperRefusal, match="evidence-location"):
        m.audit_result(
            pin("/foreign/result.json"),
            plan_sha256="a" * 64,
            anchor_sha256="b" * 64,
            proof_root=prepared["cleanup_proof_root"],
        )


@pytest.mark.parametrize("crash", [False, True])
def test_real_dispatch_finalizer_renderer_backend_and_facade_with_fake_admission(
    monkeypatch,
    tmp_path,
    crash,
):
    """Synthetic admission/BEFORE kernel facts; real code and private-temp writes."""
    import json
    from scripts import strength_freshness_cpu_install as render
    from scripts import strength_freshness_cpu_install_backend as backend
    from scripts import strength_freshness_cpu_install_runtime as facade_module
    from scripts import strength_freshness_cpu_template_finalization as finalizer
    from scripts import (
        strength_freshness_cpu_driver,
    )  # ensure its real closure is loaded
    from test_strength_freshness_cpu_install import fixture_inputs
    from test_strength_freshness_cpu_completion import valid_completion
    from test_strength_freshness_cpu_install_backend import FakeHost

    real_validation = m.c.validate_before
    fixture_data = fixture_inputs(monkeypatch)
    monkeypatch.setattr(m.c, "validate_before", real_validation)
    p, source_bytes = copy.deepcopy(fixture_data.plan), dict(fixture_data.sources)
    root, control = p["input_root"], str(Path(m.__file__).resolve().parents[1])
    old_control = p["control_root"]
    p["control_root"] = control
    modules = [
        module
        for name, module in tuple(sys.modules.items())
        if name.startswith("scripts.")
        and isinstance(getattr(module, "__file__", None), str)
        and Path(str(module.__file__)).resolve().is_relative_to(Path(control))
    ]
    paths = {str(Path(str(module.__file__)).resolve()) for module in modules}
    paths.add(str(Path(strength_freshness_cpu_driver.q.linux.core.__file__).resolve()))
    paths.update(
        str(Path(control) / "scripts" / Path(item["path"]).name)
        for item in p["source_pins"]
    )
    rows = {path: Path(path).read_bytes() for path in sorted(paths)}
    p["source_pins"] = [pin(path, raw) for path, raw in rows.items()]
    for spec in p["units"].values():
        for side in ("before", "after"):
            old = spec[side]["unit"]
            raw = source_bytes[old["path"]].replace(
                old_control.encode(), control.encode()
            )
            new = pin(old["path"], raw)
            source_bytes[old["path"]] = raw
            spec[side]["unit"] = new
            if "Environment" in spec[side]["properties"]:
                spec[side]["properties"]["Environment"] = render.q.environment_string(p)
            for variant in p["files"][spec["installed_path"]]:
                if variant["source"] == old:
                    variant["source"] = new
    guard = p["units"][p["bindings"]["support_guard"]]
    scenario_pin = guard["payload"]["scenario"]
    scenario = json.loads(source_bytes[scenario_pin["path"]])
    for side in ("before", "after"):
        for name, row in scenario["synthetic_plan"]["support_transition"][side].items():
            spec = p["units"][name][side]
            row.update(
                definition_sha256=spec["unit"]["sha256"],
                environment_sha256=render.q.linux.env_identity(
                    spec["properties"]["Environment"], spec["environment_files"]
                ),
            )
    source_bytes[scenario_pin["path"]] = render.q.encode(scenario)
    guard["payload"]["scenario"] = pin(
        scenario_pin["path"], source_bytes[scenario_pin["path"]]
    )

    def add(path, value):
        raw = value if isinstance(value, bytes) else m.c.encoded(value)
        rows[path] = raw
        return pin(path, raw)

    reg = {
        "policy": {},
        "birth_reference": {"boot_id": p["boot_id"]},
        "scope": {
            "files": {
                str(i): {"path": v["path"]} for i, v in enumerate(p["source_pins"])
            }
        },
        "source_pins": {
            str(i): {k: v[k] for k in ("sha256", "bytes")}
            for i, v in enumerate(p["source_pins"])
        },
    }
    registration_pin = add(root + "/registration.json", reg)
    base = 1800000000000000000
    req = {
        "format": "strength-preservation-collect-request-v2",
        "schema_version": 2,
        "phase": "before",
        "registration_sha256": registration_pin["sha256"],
        "nonce": p["nonce"],
        "source_pins": p["source_pins"],
        "limits": {
            "started": {
                "boot_id": p["boot_id"],
                "monotonic_ns": 100 * 10**9,
                "wall_ns": base + 100 * 10**9,
            },
            "deadline_monotonic_ns": 205 * 10**9,
            "deadline_wall_ns": base + 205 * 10**9,
            "metadata_bytes": 8 * 2**20,
            "runtime_bytes": 24 * 2**20,
        },
    }
    request_pin = add(root + "/before-request.json", req)
    policy_pin = add(root + "/policy.json", {})
    launch_pin = add(
        root + "/before-launch.json",
        {
            "format": "strength-preservation-capture-launch-v1",
            "schema_version": 1,
            "phase": "before",
            "request": request_pin,
            "registration": registration_pin,
            "physical_kind": "learner_metrics",
            "input_root": root,
            "verified_champions": p["preservation"]["verified_champions"],
        },
    )
    first = {
        "boot_id": p["boot_id"],
        "monotonic_ns": 107 * 10**9,
        "wall_ns": base + 107 * 10**9,
    }
    last = dict(first, monotonic_ns=109 * 10**9, wall_ns=base + 109 * 10**9)
    capture_pin = add(
        root + "/r3-before.json",
        {
            "clock": {
                "boot_id": p["boot_id"],
                "monotonic": 109.0,
                "wall_ns": last["wall_ns"],
            },
            "synthetic_measurement": True,
        },
    )
    provenance_raw = m.c.encoded({"synthetic_provenance": True})
    provenance_pin = add(
        root + "/r3-before.provenance-" + m.c.sha(provenance_raw) + ".json",
        provenance_raw,
    )
    receipt_pin = add(
        root + "/r3-before.receipt.json",
        {
            "format": "strength-preservation-collector-receipt-v2",
            "schema_version": 2,
            "status": "complete",
            "refusals": [],
            "read_start": first,
            "read_end": last,
            "capture_pin": capture_pin,
            "request_sha256": request_pin["sha256"],
            "registration_sha256": registration_pin["sha256"],
            "source_pins": p["source_pins"],
            "derivations": {
                "launch_sha256": launch_pin["sha256"],
                "provenance_pin": provenance_pin,
            },
        },
    )
    p["preservation"].update(
        policy=policy_pin,
        before_request=request_pin,
        before=capture_pin,
        before_receipt=receipt_pin,
    )
    for key, name in [
        ("before", "r3-before.json"),
        ("before_receipt", "r3-before.receipt.json"),
        ("before_execution", "before-execution.json"),
    ]:
        p["preservation"][key] = {
            "path": root + "/" + name,
            "sha256": "0" * 64,
            "bytes": 1,
        }
    primitive = next(
        item["sha256"]
        for item in p["source_pins"]
        if item["path"].endswith("/strength_freshness_cpu_outer.py")
    )
    blueprint = {
        "format": finalizer.FORMAT,
        "schema_version": 1,
        "plan": p,
        "before_launch": launch_pin,
        "before_registration": registration_pin,
        "primitive_source_sha256": primitive,
        "guardian_contract_sha256": "f" * 64,
    }
    blueprint_pin = add("/templates/blueprint.json", blueprint)
    source_map = {
        logical: add(f"/templates/source-{i}.bin", raw)
        for i, (logical, raw) in enumerate(sorted(source_bytes.items()))
        if Path(logical).is_relative_to(Path(root))
    }
    sources_pin = add("/templates/sources.json", source_map)
    intent = {
        "nonce": p["nonce"],
        "boot_id": p["boot_id"],
        "installer": {
            "input_root": root,
            "template_root": "/templates",
            "blueprint": blueprint_pin,
            "sources": sources_pin,
        },
    }
    raw_intent = m.c.encoded(intent)
    intent_sha = m.c.sha(raw_intent)
    intent_pin = add(root + "/intent.json", raw_intent)
    raw, _, _, evidence, _ = valid_completion()

    def translate(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: translate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [translate(v) for v in value]
        if value == "boot":
            return p["boot_id"]
        if value == "a" * 32:
            return p["nonce"]
        if value == "d" * 64:
            return intent_sha
        if value == "e" * 64:
            return primitive
        if isinstance(value, str):
            return value.replace("/input/", root + "/")
        return value

    proof = translate(m.c.parse(raw))
    proof["artifacts"] = {
        "launch": launch_pin,
        "request": request_pin,
        "registration": registration_pin,
        "capture": capture_pin,
        "receipt": receipt_pin,
        "provenance": provenance_pin,
    }
    collector = m.c.collector_contract_sha256(
        p["python"]["path"], control, launch_pin, 205 * 10**9
    )
    proof["exec_contracts"]["collector"] = collector
    for name, body in evidence.items():
        doc = translate(m.c.parse(body))
        if name == "collector_family":
            for event in ("child_started", "child_released"):
                doc[event]["exec_contract_sha256"] = collector
        proof["evidence_pins"][name] = add(proof["evidence_pins"][name]["path"], doc)
    proof_pin = add(root + "/before-execution.json", proof)
    bundle = {
        "phase": "before",
        "completion_pin": proof_pin,
        "artifacts": proof["artifacts"],
        "evidence_pins": proof["evidence_pins"],
    }
    now = {
        "boot_id": p["boot_id"],
        "monotonic_ns": 115 * 10**9,
        "wall_ns": base + 115 * 10**9,
    }
    start_pin = add(
        root + "/dummy-start.json",
        {
            "format": m.START,
            "schema_version": 1,
            "outer_intent_sha256": intent_sha,
            "before_execution_pin": proof_pin,
            "clock": now,
        },
    )
    ack = outer.dummy_start_ack(
        outer_intent_sha256=intent_sha,
        nonce=p["nonce"],
        start_pin=start_pin,
        before_execution_pin=proof_pin,
        enclosing=proof["enclosing"],
    )
    add(root + "/dummy-start.ack.json", ack)

    class Store(MemoryStore):
        def directory(self, path):
            assert path in self.roots

        def publish(self, path, raw, maximum=2**20):
            assert str(path) not in self.rows and len(raw) <= maximum
            return add(str(path), raw)

    store = Store(rows)
    store.roots = (Path(root), Path("/templates"))
    parent = outer.ProcessIdentity(**proof["enclosing"]["operator"])
    supervisor = outer.ProcessIdentity(**proof["enclosing"]["supervisor"])
    kernel = SimpleNamespace(
        clock=lambda: outer.Clock(**now),
        identity=lambda pid: supervisor if pid == parent.ppid else parent,
        exited=lambda fd: False,
    )
    admission = SimpleNamespace(
        role="helper",
        intent=intent,
        raw_intent=raw_intent,
        approved_intent_sha256=intent_sha,
        parent=outer.Handle(parent, 55),
        kernel=kernel,
        reader=store,
        outer={
            "input_root": root,
            "control_root": control,
            "source_pins": p["source_pins"],
        },
        source_sha256=primitive,
    )
    files = backend.Files._private_test_root(tmp_path)
    for directory in (root, "/run", "/etc/systemd/system"):
        files.path(directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in (start_pin["path"], root + "/dummy-start.ack.json"):
        files.path(path).write_bytes(rows[path])
        files.path(path).chmod(0o444)
    host_box = []

    def factory(prepared, stop):
        host = host_box[0] if host_box else FakeHost(prepared, files)
        if not host_box:
            host_box.append(host)
        return backend.Backend(
            prepared,
            host,
            files,
            intent_sha256=intent_sha,
            work_deadline=stop,
            start_authorization=backend.StartAuthorization(
                pin(
                    root + "/dummy-start.ack.json", rows[root + "/dummy-start.ack.json"]
                ),
                start_pin,
                proof["enclosing"],
            ),
        )

    original_create = files.create
    fired = False

    def create(item, journal):
        nonlocal fired
        original_create(item, journal)
        if crash and not fired:
            fired = True
            raise RuntimeError("injected partial setup")

    monkeypatch.setattr(files, "create", create)

    class Observation:
        def clock(self):
            return dict(now)

        def hint(self, root, event):
            raise AssertionError("prepare/prearm never polls a lifecycle hint")

        def sleep(self, seconds):
            raise AssertionError("prepare/prearm never sleeps in the facade")

    class Executor:
        def check_alive(self) -> None:
            assert not kernel.exited(admission.parent.pidfd)

        def call(self, mode: str, payload: bytes, deadline_ns: int) -> bytes:
            frame = {
                "operation": mode,
                "parent": asdict(parent),
                "phase_start": dict(now),
                "work_deadline": {
                    "monotonic_ns": deadline_ns - 2 * 10**9,
                    "wall_ns": now["wall_ns"]
                    + deadline_ns
                    - now["monotonic_ns"]
                    - 2 * 10**9,
                },
            }
            return m.dispatch(
                payload, admission, frame, intent_pin["path"], _test_factory=factory
            )

    facade = facade_module.Installer(
        raw_intent,
        intent_sha,
        proof["enclosing"],
        intent_pin=intent_pin,
        helper_executor=Executor(),
        _observation=Observation(),
    )
    if crash:
        with pytest.raises(RuntimeError, match="partial setup"):
            facade.finalize_template_and_install(m.c.encoded(bundle), now)
        result = m.c.parse(facade.prearm_cleanup())
        assert (
            result["status"] == "prearm-owned-files-removed"
            and len(result["removed"]) == 1
        )
        assert not any(
            isinstance(x, tuple) and x[0] == "start" for x in host_box[0].trace
        )
    else:
        result = m.c.parse(
            facade.finalize_template_and_install(m.c.encoded(bundle), now)
        )
        assert set(result) == {
            "plan_pin",
            "anchor_pin",
            "after_path",
            "authorization_root",
            "cleanup_proof_root",
        }
        assert facade.stage == "installed"
        assert files.read(result["plan_pin"]["path"], 2**20)
