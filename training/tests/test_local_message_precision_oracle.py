from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts import diagnose_local_message_precision as oracle
from startrain.contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.features import encode_batch
from test_local_message_benchmark import pipeline_profile
from test_local_message_inference import model
from test_model import position


def arguments(*extra):
    return oracle.parser().parse_args(
        ["--config", "profile.yaml", "--checkpoint", "manifest.json", *extra]
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--primary-batch-sizes", "128"),
        ("--small-batch-size", "33"),
        ("--max-memory-gib", "49"),
        ("--timeout-seconds", "601"),
        ("--repeats", "1"),
    ],
)
def test_resource_bounds(flag, value):
    with pytest.raises(ValueError):
        oracle.validate(arguments(flag, value))


def test_exact_failure_fixture_and_all_ring_variant_coverage():
    args = arguments()
    oracle.validate(args)
    cases = oracle.cases(args)
    assert cases[0] == {
        "ring": 10,
        "rows": 64,
        "variant": ["classic", 1, False],
        "seed": 971,
    }
    assert [c["rows"] for c in cases[:3]] == [64, 128, 256]
    assert len(cases) == 27
    assert len({(c["ring"], tuple(c["variant"])) for c in cases[3:]}) == 24
    assert {c["ring"] for c in cases} == {4, 6, 8, 10}
    assert all(c["rows"] == 16 for c in cases[3:])


def test_plan_pins_production_actor_graphs_and_diagnostic_counts(tmp_path):
    raw = tmp_path / "profile.yaml"
    raw.write_text("frozen input")
    args = arguments("--config", str(raw), "--measure-timing", "--compiled-bf16")
    config = pipeline_profile()
    assert not config.orchestration.model_refresh.inference.cuda_graphs
    manifest = SimpleNamespace(
        model_identity="identity",
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    with (
        patch("startrain.config.load_config", return_value=config),
        patch("startrain.checkpoint.load_model_manifest", return_value=manifest),
    ):
        plan = oracle.plan(args)
    assert plan["runtime"]["cuda_graphs"]
    assert plan["effective_adapter_config"]["cuda_graphs"]
    assert (
        plan["effective_adapter_config"]["cuda_graph_max_bytes"]
        == plan["runtime"]["per_model_cuda_graph_max_bytes"]
    )
    assert plan["timing_counts"] == {"warmups": 3, "repeats": 4, "iterations": 2}
    assert plan["criteria"] == oracle.CRITERIA
    assert (
        plan["compile"]["dynamic"]
        == config.orchestration.model_refresh.inference_compile_dynamic
    )


def test_error_metrics_match_independent_math_and_reject_nonfinite():
    actual, expected = torch.tensor([4.0, 3.0]), torch.tensor([3.0, 4.0])
    result = oracle.metrics(actual, expected, oracle.CRITERIA["fp32"])
    assert result["rmse"] == 1.0
    assert result["max_absolute"] == 1.0
    assert result["reference_rms"] == pytest.approx((25 / 2) ** 0.5)
    assert result["relative_l2"] == pytest.approx(2**0.5 / 5)
    assert not result["passed"]
    assert oracle.metrics(expected, expected, oracle.CRITERIA["fp32"])["passed"]
    assert not oracle.metrics(torch.tensor([float("nan")]), torch.tensor([1.0]))[
        "passed"
    ]


def test_utility_is_production_additive_margin_not_convex_combination():
    outcome = torch.tensor([[0.0, 0.0, -100.0], [-100.0, 100.0, -100.0]])
    margin = torch.full((2, SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1), -100.0)
    margin[0, 0] = 100.0
    margin[1, -1] = 100.0
    torch.testing.assert_close(
        oracle.utility(outcome, margin, 0), torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        oracle.utility(outcome, margin, 0.5), torch.tensor([-0.5, 1.0])
    )


def test_envelope_uses_both_head_errors_and_policy_errors():
    baseline = {
        "heads": {
            "utility": {
                "finite": True,
                "rmse": 0.01,
                "max_absolute": 0.02,
                "reference_rms": 0.5,
            }
        },
        "policy": {"mean_kl": 0.001, "max_probability_difference": 0.01},
    }
    assert oracle.envelope(baseline, baseline)["passed"]
    for section, key, value in (
        ("head", "rmse", 0.01101),
        ("head", "max_absolute", 0.02501),
        ("policy", "mean_kl", 0.001101),
        ("policy", "max_probability_difference", 0.01252),
    ):
        candidate = deepcopy(baseline)
        destination = (
            candidate["heads"]["utility"] if section == "head" else candidate["policy"]
        )
        destination[key] = value
        assert not oracle.envelope(candidate, baseline)["passed"]
    candidate = deepcopy(baseline)
    candidate["heads"]["utility"]["finite"] = False
    assert not oracle.envelope(candidate, baseline)["passed"]


def test_near_zero_envelope_has_predeclared_small_absolute_floor():
    baseline = {
        "heads": {
            "utility": {
                "finite": True,
                "rmse": 0.0,
                "max_absolute": 0.0,
                "reference_rms": 0.0,
            }
        },
        "policy": {"mean_kl": 0.0, "max_probability_difference": 0.0},
    }
    candidate = deepcopy(baseline)
    candidate["heads"]["utility"].update(rmse=0.9e-6, max_absolute=0.9e-6)
    assert oracle.envelope(candidate, baseline)["passed"]
    candidate["heads"]["utility"]["rmse"] = 1.1e-6
    assert not oracle.envelope(candidate, baseline)["passed"]


def synthetic_capture(logits):
    return {
        "heads": {
            "policy_logits": torch.tensor(logits),
            "utility": torch.tensor([0.25]),
        },
        "policy": SimpleNamespace(policy_offsets=[0, 2], policy_logits=logits),
    }


def test_oracle_envelope_never_rewrites_original_failed_gate():
    reference = synthetic_capture([0.0, 2.0])
    baseline = synthetic_capture([0.0, 2.04])
    candidate = synthetic_capture([0.0, 1.96])
    pairwise = oracle.compare_capture(candidate, baseline, fp32=False)
    assert not pairwise["original_pairwise_gate"]
    assert oracle.envelope(
        oracle.compare_capture(candidate, reference, fp32=False),
        oracle.compare_capture(baseline, reference, fp32=False),
    )["passed"]
    assert not pairwise["original_pairwise_gate"]


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_cpu_full_model_capture_checks_all_heads_and_utility(ring):
    network = model()
    batch = encode_batch([position(ring), position(ring)])
    reference = oracle.capture(network, batch, ring, "fp32", 0.1)
    assert set(reference["heads"]) == {
        "policy_logits",
        "soft_policy_logits",
        "outcome_logits",
        "score_margin_logits",
        "ownership_logits",
        "alive_logits",
        "utility",
    }
    assert len(reference["heads"]["policy_logits"]) == int(
        batch.legal_action_mask.sum()
    )
    for mode in ("project-first", "source-class"):
        network.set_local_message_execution(mode)
        result = oracle.capture(network, batch, ring, "fp32", 0.1)
        comparison = oracle.compare_capture(result, reference, fp32=True)
        assert comparison["fp32_equivalent"]
        assert comparison["original_pairwise_gate"]


def test_fp32_oracle_math_is_stricter_than_production_bf16_math():
    from startrain.device import enable_fast_math

    old = torch.get_float32_matmul_precision()
    old_cuda, old_cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        assert oracle.configure_math(True, "cuda:2") == {
            "float32_matmul_precision": "highest",
            "cuda_tf32": False,
            "cudnn_tf32": False,
        }
        with patch(
            "startrain.device.enable_fast_math", wraps=enable_fast_math
        ) as enable:
            actual = oracle.configure_math(False, "cuda:2")
        enable.assert_called_once_with("cuda:2")
        assert actual["float32_matmul_precision"] == "high"
        assert actual["cuda_tf32"]
    finally:
        torch.set_float32_matmul_precision(old)
        torch.backends.cuda.matmul.allow_tf32 = old_cuda
        torch.backends.cudnn.allow_tf32 = old_cudnn
