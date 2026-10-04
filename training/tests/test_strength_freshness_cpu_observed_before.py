"""Real measurement/body/family checks with simulated protected caller transport.

The qualification documents and kernel/parent facts here are deliberately fake;
they do not qualify any host. Measurement uses private temporary files and the
actual internally owned Session. The reader result remains conditional data.
"""

from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import time
from typing import Any

import pytest

from scripts import strength_freshness_cpu_observed_before as m
from scripts import strength_freshness_cpu_before_measurement as measurement
from tests import test_strength_freshness_cpu_before_measurement as fixtures
from tests.test_strength_freshness_cpu_completion import valid_completion


class MemoryReader:
    """Explicit simulated root-owned metadata; unknown paths never reach the host."""

    def __init__(self, files, clock, deadline: float):
        self.files, self.clock, self.deadline = files, clock, deadline
        self.owner_uid, self.consumed = 0, 0
        self.calls = []
        self.audit = []
        self.hook: Any = None

    def check(self):
        m.require(self.clock[0] < int(self.deadline * 1e9), "metadata-deadline")

    def directory(self, path):
        self.check()
        return (1, str(path), 0, 0o700)

    def read(self, pin, *, root=None, limit=m.requests.FILE_LIMIT):
        self.check()
        if root is not None:
            assert Path(pin["path"]).parent == root
        m.require(pin["bytes"] <= limit, "fixture-file-limit")
        self.calls.append(deepcopy(pin))
        if self.hook is not None:
            self.hook(self, pin)
        raw = self.files[pin["path"]]
        count = min(len(raw), max(0, m.requests.METADATA_BUDGET - self.consumed))
        self.consumed += count
        m.require(count == len(raw), "metadata-budget")
        m.require(
            len(raw) == pin["bytes"] and m.completion.sha(raw) == pin["sha256"],
            "fixture-pin-drift",
        )
        self.audit.append({"pin": deepcopy(pin)})
        return raw


def transport(monkeypatch, reader, clock):
    monkeypatch.setattr(m, "_READER_TYPE", MemoryReader)
    monkeypatch.setattr(m, "_BASE_READ", MemoryReader.read)
    monkeypatch.setattr(m, "_BASE_CHECK", MemoryReader.check)
    monkeypatch.setattr(m, "_BASE_DIRECTORY", MemoryReader.directory)
    monkeypatch.setattr(
        m,
        "time",
        SimpleNamespace(monotonic_ns=lambda: clock[0], time_ns=lambda: clock[1]),
    )
    return reader


def current_window(clock, seconds=5):
    return {
        "limits": {
            "started": {
                "boot_id": "boot",
                "monotonic_ns": clock[0],
                "wall_ns": clock[1],
            },
            "deadline_monotonic_ns": clock[0] + seconds * 10**9,
            "deadline_wall_ns": clock[1] + seconds * 10**9,
            "metadata_bytes": m.requests.METADATA_BUDGET,
            "runtime_bytes": m.requests.RUNTIME_BUDGET,
        },
        "admission_clock": {
            "boot_id": "boot",
            "monotonic_ns": clock[0],
            "wall_ns": clock[1],
        },
    }


def control_sources():
    pending, seen, paths = [m], set(), set()
    while pending:
        mod = pending.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        assert isinstance(mod.__file__, str)
        paths.add(str(Path(mod.__file__).resolve()))
        pending.extend(
            x
            for x in vars(mod).values()
            if isinstance(x, ModuleType)
            and x.__name__.startswith("scripts.strength_freshness_")
        )
    root = Path(m.__file__).resolve().parent.parent
    for name in m.runtime.REQUIRED_SOURCE_NAMES | {
        "strength_freshness_cpu_before_measurement.py"
    }:
        paths.add(str(root / "scripts" / name))
    return str(root), [
        {
            "path": p,
            "sha256": m.completion.sha(Path(p).read_bytes()),
            "bytes": Path(p).stat().st_size,
        }
        for p in sorted(paths)
    ]


def reader_fixture(tmp_path, monkeypatch):
    """All positive raw artifacts are produced by real Session/codec validators."""
    files = {}

    def put(path, value, *, raw=False):
        data = value if raw else m.bodies.encoded(value)
        files[path] = data
        return {"path": path, "sha256": m.completion.sha(data), "bytes": len(data)}

    premise_pins = {
        name: put(
            "/input/premise-" + name + ".json",
            {"fixture": name, "real_qualification": False},
        )
        for name in ("source", "writer", "writer_access")
    }
    access_raw = b"synthetic publication access, not qualification\n"
    original_fixture = fixtures.observed_kernel_fixture

    def kernel_fixture(patch, transform):
        def extra(reg, backend):
            transform(reg, backend)
            reg["scope"]["files"]["external-premises"] = {
                "path": "/input/external-premises.json",
                "maximum_bytes": 2**20,
            }
            writer = reg["learner_writer"]["binding"]
            writer["qualification_sha256"] = premise_pins["writer"]["sha256"]
            writer["access_sha256"] = premise_pins["writer_access"]["sha256"]
            for row in reg["publication_writers"].values():
                pin = row["access_qualification_pin"]
                pin.update(sha256=m.completion.sha(access_raw), bytes=len(access_raw))
                files[pin["path"]] = access_raw

        parsed, io, window, premises, unused = original_fixture(patch, extra)
        premises["source_context"]["qualification_sha256"] = premise_pins["source"][
            "sha256"
        ]
        return parsed, io, window, premises, unused

    monkeypatch.setattr(fixtures, "observed_kernel_fixture", kernel_fixture)
    _, write, _, _, reg, io, window, premises, bindings, _ = (
        fixtures.measurement_fixture(tmp_path, monkeypatch)
    )
    control, sources = control_sources()
    for pin in sources:
        files[pin["path"]] = Path(pin["path"]).read_bytes()
    reg_value = reg.private_copy()
    reg_pin = put("/input/registration.json", reg._raw, raw=True)
    writer_pin = reg_value["learner_writer"]["evidence_pin"]
    assert (
        put(writer_pin["path"], reg.private_writer_evidence(), raw=True) == writer_pin
    )
    premises_pin = put("/input/external-premises.json", premises)
    policy_pin = put("/input/policy.json", reg_value["policy"])
    request = {
        "format": m.REQUEST,
        "schema_version": 1,
        "contract": m.CONTRACT,
        "phase": "before",
        "nonce": bindings["nonce"],
        "registration_sha256": reg_pin["sha256"],
        "source_pins": sources,
        "limits": {
            "started": window["phase_start"],
            "deadline_monotonic_ns": window["deadline"]["monotonic_ns"],
            "deadline_wall_ns": window["deadline"]["wall_ns"],
            "metadata_bytes": m.requests.METADATA_BUDGET,
            "runtime_bytes": m.requests.RUNTIME_BUDGET,
        },
    }
    request_pin = put("/input/request.json", request)
    launch = {
        "format": m.LAUNCH,
        "schema_version": 1,
        "contract": m.CONTRACT,
        "phase": "before",
        "request": request_pin,
        "registration": reg_pin,
        "input_root": "/input",
        "verified_champions": fixtures.CHAMPIONS,
        "external_premisesPin": premises_pin,
        "writer_evidencePin": writer_pin,
    }
    launch_pin = put("/input/launch.json", launch)
    bindings.update(
        request_sha256=request_pin["sha256"],
        launch_sha256=launch_pin["sha256"],
        source_pins_sha256=m.bodies.digest(sources),
    )
    session = measurement.BeforeMeasurementSession(
        reg,
        io,
        expected_window=window,
        external_premises=premises,
        verified_champions=fixtures.CHAMPIONS,
        bindings=bindings,
    )
    session.begin()
    write()
    measured = session.finish().private_copy()
    encoded = m.bodies.encode_before_bodies(
        measured["capture_value"],
        measured["provenance_value_without_capture_pin"],
        expected=session._expected,
        registration=reg,
        output_root="/input",
    )
    for name in ("capture", "receipt", "provenance"):
        assert (
            put(encoded["pins"][name]["path"], encoded[name], raw=True)
            == encoded["pins"][name]
        )
    preflight = {
        **window["phase_start"],
        **{
            k: window["phase_start"][k] - 7 * 10**9 for k in ("monotonic_ns", "wall_ns")
        },
    }
    old_raw, old_expected, _, old_evidence, _ = valid_completion()
    boot = preflight["boot_id"]
    namespace = reg_value["kernel_context"]["namespace_expectations"]["self"]["pid"]

    def adjust(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) in (
                {"boot_id", "monotonic_ns", "wall_ns"},
                {"monotonic_ns", "wall_ns"},
            ):
                offset = value["monotonic_ns"] - 100 * 10**9
                return {
                    **({"boot_id": boot} if "boot_id" in value else {}),
                    **{k: preflight[k] + offset for k in ("monotonic_ns", "wall_ns")},
                }
            result = {k: adjust(v) for k, v in value.items()}
            if "pid_namespace_inode" in result:
                result.update(boot_id=boot, pid_namespace_inode=namespace)
            return result
        if isinstance(value, list):
            return [adjust(v) for v in value]
        return value

    expected = adjust(old_expected)
    enclosing = expected["enclosing"]
    caller = {
        **enclosing["supervisor"],
        "pid": 1,
        "ppid": 0,
        "start_ticks": 1,
        "pgid": 1,
        "sid": 1,
    }
    python = {
        "path": "/qualified/bin/python",
        "resolved_path": "/qualified/bin/python-real",
        "sha256": "a" * 64,
        "bytes": 10,
        "metadata": dict.fromkeys(m.runtime.INTERPRETER_STAT_FIELDS, 1),
    }
    python["metadata"].update(size=10, mode=0o755)
    common = {
        "schema_version": 1,
        "status": "passed-real-linux",
        "evidence_kind": "real-linux-process",
        "synthetic": False,
        "boot_id": boot,
        "source_closure_sha256": m.outer.digest(sources),
        "python_sha256": python["sha256"],
        "resource_policy_sha256": m.outer.digest(m.runtime.resource_policy()),
        "raw_evidence_pins": [
            put("/input/simulated-raw.json", {"not_a_real_qualification": True})
        ],
    }
    session_pin = put(
        "/input/session.json",
        {
            **common,
            "format": m.runtime.SESSION,
            "caller_cgroup": caller["cgroup"],
            "pid_namespace_inode": namespace,
            "checks": dict.fromkeys(m.runtime.SESSION_CHECKS, True),
        },
    )
    stat = {**python["metadata"], "mode": 0o644}
    cache = {
        "format": "strength-freshness-qualified-file-cache-v1",
        "status": "independently-qualified",
        "evidence_kind": "real-host-content-hash",
        "synthetic": False,
        "path": "/site/cache.so",
        "sha256": "a" * 64,
        "metadata": stat,
    }
    inv_pin = put(
        "/input/site-inventory.json",
        {
            "format": "strength-freshness-qualified-site-files-v1",
            "schema_version": 1,
            "root": "/site",
            "startup_directories": [
                {"path": "/site", "entries": ["cache.so", "startup.py"]}
            ],
            "startup_files": [
                {"path": "/site/startup.py", "sha256": "b" * 64, "metadata": stat}
            ],
            "cached_files": [
                {
                    "path": "/site/cache.so",
                    "sha256": cache["sha256"],
                    "metadata": stat,
                    "cache_receipt": put("/input/cache-receipt.json", cache),
                }
            ],
        },
    )
    site_pin = put(
        "/input/site.json",
        {
            **common,
            "format": "strength-freshness-real-helper-site-qualification-v1",
            "checks": dict.fromkeys(m.SITE_CHECKS, True),
            "helper_environment_sha256": m.outer.digest(
                {**m.outer.ENV, "PYTHONPATH": control}
            ),
            "site_inventory_pin": inv_pin,
        },
    )
    cfg = {
        "format": m.runtime.ADMISSION,
        "schema_version": 1,
        "nonce": bindings["nonce"],
        "preflight_start": preflight,
        "input_root": "/input",
        "control_root": control,
        "python": python,
        "source_pins": sources,
        "session_qualification": session_pin,
        "helper_site_qualification": site_pin,
        "caller": caller,
        "before_launch": launch_pin,
        "before_request": request_pin,
        "registration": reg_pin,
        "verified_champions": fixtures.CHAMPIONS,
        "resource_policy_sha256": common["resource_policy_sha256"],
    }
    intent = {
        "nonce": bindings["nonce"],
        "boot_id": boot,
        "outer": cfg,
        "observed_before": {
            "contract": m.CONTRACT,
            "premise_evidence_pins": premise_pins,
        },
    }
    intent_pin = put("/input/intent.json", intent)
    source_map = {p["path"]: p for p in sources}
    op = source_map[control + "/scripts/strength_freshness_cpu_outer.py"]
    rt = source_map[control + "/scripts/strength_freshness_cpu_outer_runtime.py"]
    artifacts = {
        "launch": launch_pin,
        "request": request_pin,
        "registration": reg_pin,
        **encoded["pins"],
    }
    expected.update(
        nonce=bindings["nonce"],
        boot_id=boot,
        outer_intent_sha256=intent_pin["sha256"],
        qualified_outer_source_sha256=op["sha256"],
        artifacts=artifacts,
    )
    expected["exec_contracts"] = {
        "guardian": m.outer.guardian_contract_sha256(
            python["path"], control, intent_pin["path"], (op["sha256"], rt["sha256"])
        ),
        "collector": m.producer.observed_collector_contract_sha256(
            python["path"],
            control,
            launch_pin,
            request["limits"]["deadline_monotonic_ns"],
        ),
    }
    binding = {
        "phase": "before",
        **{
            k: expected[k]
            for k in (
                "nonce",
                "boot_id",
                "outer_intent_sha256",
                "qualified_outer_source_sha256",
                "phase_start",
            )
        },
    }
    evidence = {
        name: adjust(m.completion.parse(raw)) for name, raw in old_evidence.items()
    }
    for name, role in (
        ("collector_family", "collector"),
        ("guardian_family", "guardian"),
    ):
        doc = evidence[name]
        doc["binding"] = binding
        doc["budget"] = expected["budget"]
        for field in ("child_started", "child_released"):
            doc[field]["exec_contract_sha256"] = expected["exec_contracts"][role]
    evidence_pins = {
        name: put(
            "/input/execution-evidence/before-" + name.replace("_", "-") + ".json", doc
        )
        for name, doc in evidence.items()
    }
    proof = adjust(m.completion.parse(old_raw))
    proof.update(expected)
    proof["evidence_pins"] = evidence_pins
    proof_pin = put("/input/before-execution.json", proof)
    historical = {
        **preflight,
        **{k: preflight[k] + 15 * 10**9 for k in ("monotonic_ns", "wall_ns")},
    }
    now = {
        **historical,
        **{k: historical[k] + 10**9 for k in ("monotonic_ns", "wall_ns")},
    }
    clock = [now["monotonic_ns"], now["wall_ns"]]
    reader = transport(
        monkeypatch,
        MemoryReader(files, clock, request["limits"]["deadline_monotonic_ns"] / 1e9),
        clock,
    )
    args = {
        "committed": {
            "before_execution": proof_pin,
            "before": artifacts["capture"],
            "before_request": request_pin,
            "before_receipt": artifacts["receipt"],
            "policy": policy_pin,
            "verified_champions": fixtures.CHAMPIONS,
        },
        "approved_outer": {
            "intent_pin": intent_pin,
            "caller_identity": caller,
            "premise_evidence_pins": premise_pins,
        },
        "enclosing": enclosing,
        "producer_sources": sources,
        "control_root": control,
        "python": python["path"],
        "reader": reader,
        "historical_consume_clock": historical,
        "current_request_limits": {"limits": request["limits"], "admission_clock": now},
    }
    return args, files, put, clock


def test_full_measurement_codec_completion_and_reader_are_conditional(
    tmp_path, monkeypatch
):
    args, files, _, _ = reader_fixture(tmp_path, monkeypatch)
    result = m.read_observed_before_artifacts(**args)
    value = result.private_copy()
    assert not value["execution_qualified"] and not value["after_authority"]
    assert not value["opaque_premise_semantics_independently_verified"]
    assert (
        fixtures.SECRET not in repr(result)
        and fixtures.SECRET not in m.bodies.encoded(value).decode()
    )
    counts = {
        path: sum(p["path"] == path for p in args["reader"].calls) for path in files
    }
    assert all(n >= 2 for n in counts.values())
    assert args["reader"].consumed < m.requests.METADATA_BUDGET


@pytest.mark.parametrize(
    "fault", ["deadline", "owner", "ledger", "read", "check", "directory"]
)
def test_retained_reader_context_cannot_be_mutated(monkeypatch, fault):
    clock = [100 * 10**9, 1800000000100000000]
    reader = transport(monkeypatch, MemoryReader({}, clock, 104), clock)
    reader.consumed = 10
    bounded = m._Read(reader, current_window(clock))
    if fault == "deadline":
        reader.deadline += 0.5
    elif fault == "owner":
        reader.owner_uid = 1
    elif fault == "ledger":
        reader.consumed = 0
    else:
        setattr(reader, fault, lambda *a, **k: None)
    with pytest.raises(m.BeforeRefusal):
        bounded.check()
    assert reader.calls == []


@pytest.mark.parametrize(
    "axis,delta", [(0, -1), (1, -1), (0, 5 * 10**9), (1, 5 * 10**9)]
)
def test_actual_raw_clocks_refuse_regression_and_exact_deadline(
    monkeypatch, axis, delta
):
    clock = [100 * 10**9, 1800000000100000000]
    reader = transport(monkeypatch, MemoryReader({}, clock, 105), clock)
    bounded = m._Read(reader, current_window(clock))
    clock[axis] += delta
    with pytest.raises((m.BeforeRefusal, m.completion.CompletionRefusal)):
        bounded.check()
    assert reader.calls == []


def test_caching_wrapper_is_not_production_reader():
    with pytest.raises(m.BeforeRefusal, match="before-reader-type"):
        m._Read(SimpleNamespace(), current_window([100 * 10**9, 1800000000100000000]))


def test_actual_source_origin_shadow_is_not_a_same_named_pin(monkeypatch):
    control, sources = control_sources()
    monkeypatch.setattr(
        m.bodies,
        "__file__",
        "/private/unapproved/strength_freshness_cpu_window_contract.py",
    )
    with pytest.raises(m.BeforeRefusal, match="before-import-origin"):
        m.executed_sources(sources, control)


@pytest.mark.parametrize("kind", ["class", "function"])
def test_from_symbol_imports_outside_old_prefix_are_origin_checked(
    tmp_path, monkeypatch, kind
):
    _, sources = control_sources()
    helper_path = tmp_path / "qualified-helper.py"
    helper_path.write_text("# source-origin fixture only\n")
    helper = ModuleType("scripts.qualified_fixture_helper")
    helper.__file__ = str(helper_path)

    def helper_function():
        return None

    helper_function.__module__ = helper.__name__
    symbol: Any = (
        type("Imported", (), {"__module__": helper.__name__})
        if kind == "class"
        else helper_function
    )
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setattr(m, "_test_imported_symbol", symbol, raising=False)
    pin = {
        "path": str(helper_path),
        "sha256": m.completion.sha(helper_path.read_bytes()),
        "bytes": helper_path.stat().st_size,
    }
    all_sources = sorted([*sources, pin], key=lambda p: p["path"])
    assert str(helper_path) in m.executed_sources(all_sources, "/")
    helper.__file__ = str(tmp_path / "shadow.py")
    with pytest.raises(m.BeforeRefusal, match="before-import-origin"):
        m.executed_sources(all_sources, "/")


def mutate_intent(args, files, put, change):
    intent = m.bodies.parse(files[args["approved_outer"]["intent_pin"]["path"]])
    change(intent)
    args["approved_outer"]["intent_pin"] = put("/input/intent.json", intent)


@pytest.mark.parametrize(
    "fault",
    [
        "old_launch",
        "unknown_launch",
        "old_request",
        "missing_session",
        "missing_raw_qualification",
        "caller",
        "source_list",
        "premise_pin",
    ],
)
def test_independent_input_and_qualification_joins_refuse_before_capture(
    tmp_path, monkeypatch, fault
):
    args, files, put, _ = reader_fixture(tmp_path, monkeypatch)
    if fault in {"old_launch", "unknown_launch"}:
        row = m.bodies.parse(files["/input/launch.json"])
        if fault == "old_launch":
            row["format"] = "strength-preservation-capture-launch-v1"
        else:
            row["unapproved"] = True
        pin = put("/input/launch.json", row)
        mutate_intent(
            args, files, put, lambda doc: doc["outer"].update(before_launch=pin)
        )
    elif fault == "old_request":
        row = m.bodies.parse(files["/input/request.json"])
        row["format"] = m.requests.FORMAT
        pin = put("/input/request.json", row)
        args["committed"]["before_request"] = pin
        mutate_intent(
            args, files, put, lambda doc: doc["outer"].update(before_request=pin)
        )
    elif fault == "missing_session":
        mutate_intent(
            args,
            files,
            put,
            lambda doc: doc["outer"].update(session_qualification=None),
        )
    elif fault == "missing_raw_qualification":
        del files["/input/simulated-raw.json"]
    elif fault == "caller":
        args["approved_outer"]["caller_identity"]["start_ticks"] += 1
    elif fault == "source_list":
        args["producer_sources"].pop()
        mutate_intent(
            args,
            files,
            put,
            lambda doc: doc["outer"].update(source_pins=args["producer_sources"]),
        )
    else:
        args["approved_outer"]["premise_evidence_pins"]["writer"]["sha256"] = "f" * 64
    with pytest.raises(m.BeforeRefusal):
        m.read_observed_before_artifacts(**args)
    assert not any(p["path"] == "/input/r3-before.json" for p in args["reader"].calls)


@pytest.mark.parametrize(
    "fault", ["signal", "nonzero", "unreaped", "residue", "contract", "parent", "late"]
)
def test_fully_rehashed_family_cannot_replace_natural_producer_exit(
    tmp_path, monkeypatch, fault
):
    args, files, put, _ = reader_fixture(tmp_path, monkeypatch)
    proof = m.completion.parse(files["/input/before-execution.json"])
    key = "collector_family"
    pin = proof["evidence_pins"][key]
    doc = m.completion.parse(files[pin["path"]])
    if fault == "signal":
        doc["signals"] = [{"signal": 15}]
    elif fault == "nonzero":
        doc["child_terminal"]["waitid"]["status"] = 2
        doc["child_terminal"]["waitpid"]["status"] = 2 << 8
    elif fault == "unreaped":
        doc["child_terminal"]["reaped"] = False
    elif fault == "residue":
        doc["family_closed"]["retained_children"] = [doc["child_started"]["child"]]
    elif fault == "contract":
        doc["child_released"]["exec_contract_sha256"] = "f" * 64
    elif fault == "parent":
        doc["child_started"]["child"]["ppid"] += 1
    else:
        doc["child_terminal"]["clock"]["monotonic_ns"] += 120 * 10**9
    proof["evidence_pins"][key] = put(pin["path"], doc)
    args["committed"]["before_execution"] = put("/input/before-execution.json", proof)
    with pytest.raises(m.BeforeRefusal, match="before-artifact-refused"):
        m.read_observed_before_artifacts(**args)


@pytest.mark.parametrize(
    "which",
    [
        "capture",
        "provenance",
        "collector_family",
        "raw_qualification",
        "writer_premise",
        "control_source",
    ],
)
def test_final_reread_is_uncached_for_every_evidence_domain(
    tmp_path, monkeypatch, which
):
    args, files, _, _ = reader_fixture(tmp_path, monkeypatch)
    proof = m.completion.parse(files["/input/before-execution.json"])
    paths = {
        "capture": proof["artifacts"]["capture"]["path"],
        "provenance": proof["artifacts"]["provenance"]["path"],
        "collector_family": proof["evidence_pins"]["collector_family"]["path"],
        "raw_qualification": "/input/simulated-raw.json",
        "writer_premise": "/input/premise-writer.json",
        "control_source": args["producer_sources"][0]["path"],
    }
    target = paths[which]
    hit = []

    def changed(reader, pin):
        if (
            pin["path"] == target
            and sum(x["path"] == target for x in reader.calls) == 2
        ):
            hit.append(True)
            reader.files[target] += b"changed"

    args["reader"].hook = changed
    with pytest.raises(m.BeforeRefusal, match="fixture-pin-drift"):
        m.read_observed_before_artifacts(**args)
    assert hit and args["reader"].consumed > 0


@pytest.mark.parametrize(
    "mutation", ["deadline", "ledger", "late_wall", "late_mono", "private_error"]
)
def test_mid_read_context_and_time_refusal_retains_no_authority(
    tmp_path, monkeypatch, mutation
):
    args, _, _, clock = reader_fixture(tmp_path, monkeypatch)
    reader = args["reader"]

    def changed(current, pin):
        if len(current.calls) != 3:
            return
        if mutation == "deadline":
            current.deadline += 1
        elif mutation == "ledger":
            current.consumed = 0
        elif mutation.startswith("late"):
            key, index = (
                ("deadline_wall_ns", 1)
                if mutation == "late_wall"
                else ("deadline_monotonic_ns", 0)
            )
            clock[index] = args["current_request_limits"]["limits"][key]
        else:
            raise OSError(fixtures.SECRET)

    reader.hook = changed
    with pytest.raises(m.BeforeRefusal) as exc:
        m.read_observed_before_artifacts(**args)
    assert fixtures.SECRET not in str(exc.value)


def test_final_rereads_share_original_ledger_and_can_exhaust_it(tmp_path, monkeypatch):
    args, _, _, _ = reader_fixture(tmp_path, monkeypatch)
    reader = args["reader"]
    reader.consumed = m.requests.METADATA_BUDGET - 2_000_000
    with pytest.raises(m.BeforeRefusal, match="metadata-budget"):
        m.read_observed_before_artifacts(**args)
    assert reader.consumed == m.requests.METADATA_BUDGET


def test_production_transport_rejects_real_private_symlink_without_payload(tmp_path):
    actual = tmp_path / "actual"
    actual.write_bytes(b"data")
    actual.chmod(0o444)
    alias = tmp_path / "alias"
    alias.symlink_to(actual)
    now = [time.monotonic_ns(), time.time_ns()]
    raw_reader = m.requests.PinnedReader(deadline=now[0] / 1e9 + 4, owner_uid=0)
    bounded = m._Read(raw_reader, current_window(now))
    pin = {"path": str(alias), "sha256": m.completion.sha(b"data"), "bytes": 4}
    with pytest.raises(m.requests.RequestRefusal):
        bounded.read(pin)
    assert raw_reader.consumed == 0


@pytest.mark.parametrize("size", [True, 0, 64 * 2**20 + 1])
def test_interpreter_metadata_cannot_hide_malformed_size_alias(
    tmp_path, monkeypatch, size
):
    args, files, put, _ = reader_fixture(tmp_path, monkeypatch)

    def changed(doc):
        doc["outer"]["python"]["bytes"] = size
        doc["outer"]["python"]["metadata"]["size"] = int(size)

    mutate_intent(args, files, put, changed)
    with pytest.raises(m.BeforeRefusal, match="before-python-hash"):
        m.read_observed_before_artifacts(**args)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 4)])
def test_one_path_cannot_admit_two_pins_before_any_second_operation(
    monkeypatch, field, value
):
    clock = [100 * 10**9, 1800000000100000000]
    reader = transport(
        monkeypatch, MemoryReader({"/input/proof.json": b"abc"}, clock, 104), clock
    )
    bounded = m._Read(reader, current_window(clock))
    pin = {"path": "/input/proof.json", "sha256": m.completion.sha(b"abc"), "bytes": 3}
    assert bounded.read(pin) == b"abc"
    seen = []
    monkeypatch.setattr(m.time, "monotonic_ns", lambda: seen.append(True) or clock[0])
    changed = {**pin, field: value}
    with pytest.raises(m.BeforeRefusal, match="before-pin-version-conflict"):
        bounded.read(changed, fresh=True)
    assert not seen and reader.calls == [pin] and reader.consumed == 3


def test_same_pin_cache_and_mandatory_fresh_read_remain_distinct(monkeypatch):
    clock = [100 * 10**9, 1800000000100000000]
    reader = transport(
        monkeypatch, MemoryReader({"/input/proof.json": b"abc"}, clock, 104), clock
    )
    bounded = m._Read(reader, current_window(clock))
    pin = {"path": "/input/proof.json", "sha256": m.completion.sha(b"abc"), "bytes": 3}
    assert bounded.read(pin) == bounded.read(deepcopy(pin)) == b"abc"
    assert reader.calls == [pin] and reader.consumed == 3
    assert bounded.read(pin, fresh=True) == b"abc"
    assert reader.calls == [pin, pin] and reader.consumed == 6
