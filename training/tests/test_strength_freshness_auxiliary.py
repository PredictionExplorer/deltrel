from __future__ import annotations

import copy

import pytest

from scripts.strength_freshness_auxiliary import classify_auxiliary, validate_policy


@pytest.fixture
def observed():
    policy = {
        "python_argv0": "/qualified/training/.venv/bin/python",
        "executable": "/qualified/python3.11",
        "cwd": "/qualified/training",
        "compile_worker_script": "/qualified/training/.venv/torch/compile_worker/__main__.py",
        "torch_key": "cTo03RUSqfmcYCgUyy2MUjPCGJXm6IgGWiWGw//uJng=",
        "compile_parent_roles": [
            "learner",
            "arena-promotion",
            *[f"actor-gpu-{i}" for i in range(1, 7)],
        ],
    }
    parent = {
        "pid": 100,
        "start_ticks": 1000,
        "exe": policy["executable"],
        "cwd": policy["cwd"],
        "environment": {
            "PYTHONPATH": "/qualified/training",
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    }
    child = {
        **copy.deepcopy(parent),
        "pid": 101,
        "ppid": 100,
        "start_ticks": 1001,
        "argv": [
            policy["python_argv0"],
            policy["compile_worker_script"],
            "--pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler",
            "--kind=fork",
            "--workers=2",
            "--parent=100",
            "--read-fd=65",
            "--write-fd=68",
            "--torch-key=" + policy["torch_key"],
        ],
    }
    return policy, parent, child


def classify(observed, role="learner"):
    policy, parent, child = observed
    return classify_auxiliary(policy, child, parent=parent, parent_role=role)


@pytest.mark.parametrize(
    "role", ["learner", "arena-promotion", *[f"actor-gpu-{i}" for i in range(1, 7)]]
)
def test_recorded_compile_pool_parent_roles(observed, role):
    assert classify(observed, role) == "torch-inductor-pool"


@pytest.mark.parametrize(
    "code,tail,kind",
    [
        (
            "from multiprocessing.resource_tracker import main;main(70)",
            [],
            "multiprocessing-resource-tracker",
        ),
        *[
            (
                f"from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=71, pipe_handle={fd})",
                ["--multiprocessing-fork"],
                "multiprocessing-spawn",
            )
            for fd in (75, 79, 83, 87)
        ],
    ],
)
def test_recorded_tracker_and_four_loader_forms(observed, code, tail, kind):
    policy, _, child = observed
    child["argv"] = [policy["python_argv0"], "-B", "-c", code, *tail]
    assert classify(observed) == kind
    assert classify(observed, "arena-promotion") is None


@pytest.mark.parametrize(
    "index,value",
    [
        (0, "/unqualified/python"),
        (1, "/unqualified/compile.py"),
        (2, "--pickler=arbitrary.Loader"),
        (3, "--kind=spawn"),
        (4, "--workers=99"),
        (5, "--parent=99"),
        (6, "--read-fd=-1"),
        (7, "--write-fd=2147483648"),
        (8, "--torch-key=other"),
        (6, "--read-fd=" + "9" * 100),
    ],
)
def test_compile_arguments_are_closed(observed, index, value):
    observed[2]["argv"][index] = value
    assert classify(observed) is None


def test_extra_argument_and_uncertified_role_are_pending(observed):
    assert classify(observed, "coordinator") is None
    observed[2]["argv"].append("--extra")
    assert classify(observed) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 100),
        ("ppid", 99),
        ("ppid", True),
        ("start_ticks", 999),
        ("start_ticks", True),
        ("exe", "/unqualified/python"),
        ("cwd", "/other"),
    ],
)
def test_child_identity_birth_and_origin_are_required(observed, field, value):
    observed[2][field] = value
    assert classify(observed) is None


@pytest.mark.parametrize(
    "field,value",
    [("pid", True), ("start_ticks", 1002), ("exe", "/other"), ("cwd", "/other")],
)
def test_parent_identity_is_not_inferred_from_argv(observed, field, value):
    observed[1][field] = value
    assert classify(observed) is None


@pytest.mark.parametrize(
    "key",
    [
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONNOUSERSITE",
    ],
)
def test_import_context_must_match_validated_parent(observed, key):
    observed[2]["environment"][key] = "unexpected"
    assert classify(observed) is None


def test_missing_environment_is_not_assumed_to_match(observed):
    del observed[2]["environment"]
    assert classify(observed) is None


@pytest.mark.parametrize(
    "code,tail",
    [
        (
            "from multiprocessing.resource_tracker import main;main(70);print('extra')",
            [],
        ),
        ("from multiprocessing.resource_tracker import main;main(-1)", []),
        (
            "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=71, pipe_handle=75)",
            [],
        ),
        (
            "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=71, pipe_handle=2147483648)",
            ["--multiprocessing-fork"],
        ),
        (
            "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=71, pipe_handle=75);pass",
            ["--multiprocessing-fork"],
        ),
    ],
)
def test_inline_python_cannot_be_extended(observed, code, tail):
    observed[2]["argv"] = [observed[0]["python_argv0"], "-B", "-c", code, *tail]
    assert classify(observed) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("cwd", "relative"),
        ("python_argv0", "/a/../b"),
        ("torch_key", "invalid"),
        ("compile_parent_roles", ["learner", "learner"]),
        ("compile_parent_roles", [".*"]),
        ("compile_parent_roles", []),
    ],
)
def test_malformed_policy_is_a_registration_error(observed, field, value):
    observed[0][field] = value
    with pytest.raises(ValueError):
        validate_policy(observed[0])


def test_inputs_are_never_modified(observed):
    before = copy.deepcopy(observed)
    assert classify(observed) == "torch-inductor-pool"
    assert observed == before
