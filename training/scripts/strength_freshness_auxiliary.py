"""Classify the observed persistent Python auxiliaries without certifying work.

Callers must independently establish current cgroup membership, parent ownership,
and the pinned interpreter/package/import environment. This pure matcher neither
loads process state nor verifies executable files. Its policy is part of the
qualified host manifest; observed command lines alone cannot qualify a runtime.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import PurePosixPath
import re
from typing import Any


IMPORT_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONSAFEPATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
)


def validate_policy(policy: Mapping[str, Any]) -> None:
    expected = {
        "python_argv0",
        "executable",
        "cwd",
        "compile_worker_script",
        "torch_key",
        "compile_parent_roles",
    }
    if set(policy) not in (expected, expected | {"compile_environment"}):
        raise ValueError("auxiliary policy fields differ")
    for key in ("python_argv0", "executable", "cwd", "compile_worker_script"):
        value = policy[key]
        if not isinstance(value, str):
            raise ValueError("auxiliary policy path is not text")
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or ".." in path.parts
            or "\0" in value
        ):
            raise ValueError("auxiliary policy path is not canonical")
    key = policy["torch_key"]
    try:
        valid_key = (
            isinstance(key, str) and len(base64.b64decode(key, validate=True)) == 32
        )
    except (ValueError, UnicodeError):
        valid_key = False
    if not valid_key:
        raise ValueError("auxiliary policy torch key is invalid")
    roles = policy["compile_parent_roles"]
    if (
        not isinstance(roles, list)
        or not 1 <= len(roles) <= 32
        or any(
            not isinstance(r, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", r)
            for r in roles
        )
        or len(roles) != len(set(roles))
    ):
        raise ValueError("auxiliary policy parent roles differ")
    if "compile_environment" in policy:
        environment = policy["compile_environment"]
        if (
            not isinstance(environment, dict)
            or set(environment) != {"PYTHONPATH", "LD_LIBRARY_PATH"}
            or not all(isinstance(v, str) for v in environment.values())
        ):
            raise ValueError("auxiliary compiler environment fields differ")
        # Torch may materialize sys.path and an absent library path. These are
        # explicit qualified values, never wildcard prefixes or an implicit
        # equivalence between an unset and empty environment variable.
        for name, value in environment.items():
            if name == "LD_LIBRARY_PATH" and value == "":
                continue
            paths = value.split(":")
            if (name == "PYTHONPATH" and paths[0] != policy["cwd"]) or any(
                not p
                or not PurePosixPath(p).is_absolute()
                or PurePosixPath(p).as_posix() != p
                or ".." in PurePosixPath(p).parts
                or "\0" in p
                for p in paths
            ):
                raise ValueError("auxiliary compiler search path is not canonical")


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def _fds(*values: str) -> bool:
    return all(len(v) <= 10 and int(v) <= 2**31 - 1 for v in values)


def classify_auxiliary(
    policy: Mapping[str, Any],
    process: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    parent_role: str,
) -> str | None:
    """Return a known auxiliary class, or None for unclassified observations."""
    validate_policy(policy)
    if (
        not all(
            _positive(p.get(k))
            for p in (process, parent)
            for k in ("pid", "start_ticks")
        )
        or not _positive(process.get("ppid"))
        or process["pid"] == parent["pid"]
        or process.get("ppid") != parent["pid"]
        or process["start_ticks"] < parent["start_ticks"]
        or any(
            p.get("exe") != policy["executable"] or p.get("cwd") != policy["cwd"]
            for p in (process, parent)
        )
    ):
        return None
    child_env, parent_env = process.get("environment"), parent.get("environment")
    if not all(
        isinstance(env, Mapping)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
        for env in (child_env, parent_env)
    ):
        return None
    assert isinstance(child_env, Mapping) and isinstance(parent_env, Mapping)

    def environment_matches(*, compiler: bool = False) -> bool:
        expected = {key: parent_env.get(key) for key in IMPORT_ENVIRONMENT}
        if compiler:
            expected.update(policy.get("compile_environment", {}))
        return all(child_env.get(key) == value for key, value in expected.items())

    args = process.get("argv")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return None
    if not args or args[0] != policy["python_argv0"]:
        return None
    if len(args) == 9 and parent_role in policy["compile_parent_roles"]:
        fixed = [
            policy["compile_worker_script"],
            "--pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler",
            "--kind=fork",
            "--workers=2",
            f"--parent={parent['pid']}",
        ]
        read = re.fullmatch(r"--read-fd=([0-9]+)", args[6])
        write = re.fullmatch(r"--write-fd=([0-9]+)", args[7])
        if (
            args[1:6] == fixed
            and args[8] == "--torch-key=" + policy["torch_key"]
            and read
            and write
            and _fds(read[1], write[1])
            and environment_matches(compiler=True)
        ):
            return "torch-inductor-pool"
    if parent_role != "learner" or args[1:3] != ["-B", "-c"]:
        return None
    if len(args) == 4:
        match = re.fullmatch(
            r"from multiprocessing\.resource_tracker import main;main\(([0-9]+)\)",
            args[3],
        )
        if match and _fds(match[1]) and environment_matches():
            return "multiprocessing-resource-tracker"
    if len(args) == 5 and args[4] == "--multiprocessing-fork":
        match = re.fullmatch(
            r"from multiprocessing\.spawn import spawn_main; spawn_main\(tracker_fd=([0-9]+), pipe_handle=([0-9]+)\)",
            args[3],
        )
        if match and _fds(match[1], match[2]) and environment_matches():
            return "multiprocessing-spawn"
    return None
