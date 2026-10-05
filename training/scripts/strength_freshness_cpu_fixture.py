"""Closed, bounded target-interpreter fixture runner; never live DR qualification."""

from __future__ import annotations

from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from typing import Any

from scripts import strength_freshness_cpu_qualification as q

CASES = {
    "proof-freeze-and-retention": "tests/test_strength_freshness_linux_proof.py",
    "execution-gate-refusal": "tests/test_strength_freshness_linux.py::test_unqualified_or_drifted_plan_never_reaches_mutations",
}


def run_bound_fixtures(context: q.AuthorizedContext) -> dict[str, Any]:
    q.require(
        context.authorization["role"] == "dispatcher"
        and context.io._purpose == "dispatcher",
        "fixture-dispatcher-only",
    )
    p = context.plan.value
    argv = [
        p["python"]["path"],
        "-s",
        str(
            Path(p["control_root"]) / "scripts/strength_freshness_cpu_qualification.py"
        ),
        "--authorization",
        str(
            Path(p["input_root"])
            / (context.authorization["unit"] + ".authorization.json")
        ),
        "--fixture-child",
    ]
    end = min(
        context.anchor.effective_deadline("work", context.clock()),
        context.io.now() + 30,
    )
    environment = {**q.environment(p), "INVOCATION_ID": context.unit.invocation_id}
    context.log.record(
        "fixture-child-intent",
        {"argv": argv, "deadline": end, "scope": "isolated source fixtures only"},
    )
    result = context.io._backend.child(argv, environment, end)
    context.log.record("fixture-child-raw", result)
    q.require(
        set(result) == set(CASES)
        and all(
            v["status"] == "passed"
            and v["failures"] == v["errors"] == v["skipped"] == 0
            and v["tests"] > 0
            for v in result.values()
        ),
        "fixture-child-result",
    )
    return result


def run_worker(context: q.AuthorizedContext) -> dict[str, Any]:
    p = context.plan.value
    q.require(context.authorization["role"] == "dispatcher", "fixture-owner-role")
    root = Path(p["control_root"])
    output = Path(p["scratch_root"]) / "fixtures"
    q.require(
        not output.exists() and output.parent.resolve() == output.parent,
        "fixture-output-no-clobber",
    )
    output.mkdir(mode=0o700)
    result = {}
    pins = {x["path"]: x for x in p["source_pins"]}
    for case, node in CASES.items():
        end = min(
            context.anchor.effective_deadline("work", context.clock()),
            context.io.now() + 12,
        )
        source = root / node.split("::")[0]
        config = root / "pyproject.toml"
        for path in (source, config):
            q.require(str(path) in pins, "fixture-input-unpinned")
            context.io._backend.pin(pins[str(path)], end)
        xml = output / (case + ".xml")
        stdout = output / (case + ".log")
        argv = [
            p["python"]["path"],
            "-s",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(config),
            str(root / node),
            "--junitxml=" + str(xml),
            "--basetemp=" + str(output / (case + "-tmp")),
        ]
        environment = {
            **q.environment(p),
            "PYTHONPATH": str(root) + ":" + str(root / "tests"),
        }
        context.log.record(
            "fixture-command", {"case": case, "argv": argv, "deadline": end}
        )
        with stdout.open("xb") as stream:
            completed = subprocess.run(
                argv,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=max(0.001, end - time.monotonic()),
                check=False,
            )
        # Save all raw output pins before testing any success assertion.
        raw = stdout.read_bytes()
        q.require(len(raw) <= 2**20, "fixture-log-size")
        log_pin = {"path": str(stdout), "sha256": q.sha(raw), "bytes": len(raw)}
        context.log.record(
            "fixture-command-raw",
            {"case": case, "exit_code": completed.returncode, "log": log_pin},
        )
        q.require(
            completed.returncode == 0
            and context.io.now() < end
            and xml.is_file()
            and xml.stat().st_size <= 2**20,
            "fixture-command-failed",
        )
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        q.require(len(suites) == 1, "fixture-junit-schema")
        counts = {
            k: int(suites[0].attrib[k])
            for k in ("tests", "failures", "errors", "skipped")
        }
        q.require(
            counts["tests"] > 0
            and counts["failures"] == counts["errors"] == counts["skipped"] == 0,
            "fixture-junit-failed",
        )
        result[case] = {
            "status": "passed",
            **counts,
            "log": log_pin,
            "junit": {
                "path": str(xml),
                "sha256": q.sha(xml.read_bytes()),
                "bytes": xml.stat().st_size,
            },
            "execution_qualified": False,
            "scope": "isolated synthetic fixtures, not production boundary",
        }
    return result
