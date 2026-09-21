"""Optional auxiliary predictions retain the existing model and loss contracts."""

from dataclasses import replace
import math

import pytest
import torch

from deltreltrain.features import encode_batch
from deltreltrain.losses import LossWeights, compute_losses
from deltreltrain.model import GraphResTNet, ModelConfig, model_parameter_count
from deltreltrain.symmetry import D5Transform, permute_nodes, transform_position
from deltreltrain.topology import SUPPORTED_RINGS, get_topology
from test_losses import outputs, targets
from test_model import position


AUX_NAMES = (
    "opponent_reply",
    "second_stone",
    "final_shores",
    "final_networks",
    "final_capes",
)


def config(enabled=True):
    return ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
        auxiliary_predictions=enabled,
    )


def auxiliary_outputs():
    return outputs()._replace(
        opponent_reply_logits=torch.zeros(2, 4, requires_grad=True),
        second_stone_logits=torch.zeros(2, 3, requires_grad=True),
        final_shores_logits=torch.zeros(2, 2, 51, requires_grad=True),
        final_networks_logits=torch.zeros(2, 2, 26, requires_grad=True),
        final_capes_logits=torch.zeros(2, 2, 6, requires_grad=True),
    )


def auxiliary_targets():
    available = torch.tensor([True, False])
    return replace(
        targets(),
        # The unavailable row intentionally contains invalid values.
        opponent_reply=torch.tensor([[0.0, 0.0, 0.0, 1.0], [float("nan")] * 4]),
        second_stone=torch.tensor([[0.0, 1.0, 0.0], [float("nan")] * 3]),
        final_shores=torch.tensor([[4, 7], [-999, -999]]),
        final_networks=torch.tensor([[2, 1], [-999, -999]]),
        final_capes=torch.tensor([[2, 3], [-999, -999]]),
        **{f"{name}_mask": available.clone() for name in AUX_NAMES},
    )


def loss_options():
    return dict(
        legal_action_mask=torch.tensor([[True, True, False], [False, False, False]]),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
        weights=LossWeights(**{name: 0.2 for name in AUX_NAMES}),
    )


def test_disabled_model_preserves_parameters_and_primary_initialization():
    torch.manual_seed(47)
    old = GraphResTNet(config(False)).eval()
    torch.manual_seed(47)
    new = GraphResTNet(config(True)).eval()
    assert old.parameter_count() == model_parameter_count(config(False))
    assert new.parameter_count() == model_parameter_count(config(True))
    assert new.parameter_count() - old.parameter_count() == (16 + 1) * 169
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    batch = encode_batch([position(4)])
    with torch.no_grad():
        old_output = old(*batch.model_args())
        new_output = new(*batch.model_args())
        skipped_output = new(*batch.model_args(), include_auxiliary=False)
    assert old_output[6:] == (None,) * 5
    assert skipped_output[6:] == (None,) * 5
    for actual, expected, skipped in zip(
        new_output[:6], old_output[:6], skipped_output[:6], strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(skipped, expected, rtol=0, atol=0)


def test_auxiliary_shapes_board_limits_and_action_masks():
    model = GraphResTNet(config()).eval()
    batch = encode_batch([position(ring) for ring in SUPPORTED_RINGS])
    with torch.no_grad():
        out = model(*batch.model_args())
    assert out.opponent_reply_logits.shape == (4, batch.max_nodes + 1)
    assert out.second_stone_logits.shape == (4, batch.max_nodes)
    assert out.final_shores_logits.shape == (4, 2, 51)
    assert out.final_networks_logits.shape == (4, 2, 26)
    assert out.final_capes_logits.shape == (4, 2, 6)
    minimum = torch.finfo(out.final_shores_logits.dtype).min
    for row, ring in enumerate(SUPPORTED_RINGS):
        assert (out.final_shores_logits[row, :, 5 * ring + 1 :] == minimum).all()
        assert (out.final_networks_logits[row, :, (5 * ring) // 2 + 1 :] == minimum).all()
        assert (out.final_shores_logits[row, :, : 5 * ring + 1] > minimum).all()
        assert out.opponent_reply_logits[row, -1] > minimum
    illegal = ~batch.legal_action_mask
    assert (out.opponent_reply_logits[:, :-1][illegal] == minimum).all()
    assert (out.second_stone_logits[illegal] == minimum).all()


def test_auxiliary_model_compiles_with_bf16_and_omits_heads_for_leaf_inference():
    model = GraphResTNet(config()).eval()
    batch = encode_batch([position(4)])
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(*batch.model_args())
        actual = compiled(*batch.model_args())
        leaf = compiled(*batch.model_args(), include_auxiliary=False)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(leaf[:6], expected[:6], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert leaf[6:] == (None,) * 5


def test_auxiliary_d5_node_equivariance_and_count_invariance():
    model = GraphResTNet(config()).eval()
    source = position(4)
    with torch.no_grad():
        baseline = model(*encode_batch([source]).model_args())
        for index in range(10):
            transform = D5Transform.from_index(index)
            transformed = model(
                *encode_batch([transform_position(source, transform)]).model_args()
            )
            permutation = get_topology(4).d5_permutation(
                transform.rotation, transform.reflected
            )
            torch.testing.assert_close(
                transformed.opponent_reply_logits[0, :-1],
                permute_nodes(baseline.opponent_reply_logits[0, :-1], permutation),
                atol=3e-5,
                rtol=3e-5,
            )
            torch.testing.assert_close(
                transformed.second_stone_logits[0],
                permute_nodes(baseline.second_stone_logits[0], permutation),
                atol=3e-5,
                rtol=3e-5,
            )
            for name in AUX_NAMES[2:]:
                torch.testing.assert_close(
                    getattr(transformed, f"{name}_logits"),
                    getattr(baseline, f"{name}_logits"),
                    atol=3e-5,
                    rtol=3e-5,
                )
            torch.testing.assert_close(
                transformed.opponent_reply_logits[:, -1],
                baseline.opponent_reply_logits[:, -1],
                atol=3e-5,
                rtol=3e-5,
            )


@pytest.mark.parametrize("validate", [True, False])
def test_auxiliary_cross_entropies_and_gradients_mask_unavailable_rows(validate):
    output = auxiliary_outputs()
    loss = compute_losses(
        output, auxiliary_targets(), validate_targets=validate, **loss_options()
    )
    expected = dict(
        zip(
            AUX_NAMES,
            (math.log(3), math.log(2), math.log(51), math.log(26), math.log(6)),
            strict=True,
        )
    )
    for name, value in expected.items():
        assert loss[name].item() == pytest.approx(value, abs=1e-6)
    assert loss["total"].item() == pytest.approx(0.2 * sum(expected.values()), abs=1e-6)
    loss["total"].backward()
    for name in AUX_NAMES:
        grad = getattr(output, f"{name}_logits").grad
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[0]) > 0
        assert torch.count_nonzero(grad[1]) == 0
    assert output.opponent_reply_logits.grad[0, 2] == 0
    assert output.second_stone_logits.grad[0, 2] == 0
    # Both players share the component head weight equally.
    expected_gradient = torch.full((2, 6), 0.1 / 6)
    expected_gradient[0, 2] -= 0.1
    expected_gradient[1, 3] -= 0.1
    torch.testing.assert_close(output.final_capes_logits.grad[0], expected_gradient)


def test_missing_targets_and_zero_auxiliary_weights_have_finite_zero_gradients():
    for target, weights in (
        (targets(), loss_options()["weights"]),
        (auxiliary_targets(), LossWeights()),
    ):
        output = auxiliary_outputs()
        # Even a completely masked head must produce zero without sum overflow.
        with torch.no_grad():
            output.second_stone_logits.fill_(torch.finfo(torch.float32).min)
        loss = compute_losses(output, target, **{**loss_options(), "weights": weights})
        assert loss["total"].item() == 0
        loss["total"].backward()
        for name in AUX_NAMES:
            grad = getattr(output, f"{name}_logits").grad
            assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


def test_auxiliary_supervision_diagnostics_distinguish_counts_from_future_labels():
    target = replace(
        auxiliary_targets(),
        opponent_reply_mask=torch.zeros(2, dtype=torch.bool),
        second_stone_mask=torch.zeros(2, dtype=torch.bool),
    )
    output = auxiliary_outputs()
    plain = compute_losses(output, target, **loss_options())
    diagnostic = compute_losses(
        output, target, include_diagnostics=True, **loss_options()
    )
    torch.testing.assert_close(diagnostic["total"], plain["total"], rtol=0, atol=0)
    assert not any(f"{name}_available" in plain for name in AUX_NAMES)
    for name in AUX_NAMES:
        count = diagnostic[f"{name}_available"]
        assert count.item() == (0 if name in AUX_NAMES[:2] else 1)
        assert count.device == output.policy_logits.device
        assert count.dtype == torch.int64 and not count.requires_grad
    missing = compute_losses(
        output, targets(), include_diagnostics=True, **loss_options()
    )
    assert all(missing[f"{name}_available"].item() == 0 for name in AUX_NAMES)


@pytest.mark.parametrize("disabled", ["sample_weight", "loss_weight"])
def test_auxiliary_supervision_diagnostics_count_only_positive_weight_rows(disabled):
    target = auxiliary_targets()
    options = loss_options()
    if disabled == "sample_weight":
        target = replace(target, sample_weight=torch.tensor([0.0, 1.0]))
    else:
        options["weights"] = LossWeights()
    diagnostic = compute_losses(
        auxiliary_outputs(), target, include_diagnostics=True, **options
    )
    assert all(diagnostic[f"{name}_available"].item() == 0 for name in AUX_NAMES)


def test_legacy_model_has_no_auxiliary_supervision_diagnostics():
    diagnostic = compute_losses(
        outputs(),
        targets(),
        include_diagnostics=True,
        legal_action_mask=torch.ones(2, 3, dtype=torch.bool),
        node_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert not any(f"{name}_available" in diagnostic for name in AUX_NAMES)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"opponent_reply_mask": None}, "occur together"),
        ({"second_stone_mask": torch.ones(2)}, "boolean"),
        ({"final_capes": torch.ones(2, 2)}, "integer"),
        ({"final_capes": torch.tensor([[6, 0], [0, 0]])}, "supported range"),
        (
            {"second_stone": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])},
            "empty nodes",
        ),
        ({"opponent_reply": torch.zeros(2, 4)}, "sum to one"),
    ],
)
def test_auxiliary_contract_rejects_invalid_available_labels(change, match):
    with pytest.raises(ValueError, match=match):
        compute_losses(
            auxiliary_outputs(),
            replace(auxiliary_targets(), **change),
            **loss_options(),
        )


def test_count_loss_rejects_impossible_board_class_and_transfers_optional_targets():
    output = auxiliary_outputs()
    with torch.no_grad():
        output.final_networks_logits[:, :, 11:] = torch.finfo(torch.float32).min
    target = replace(auxiliary_targets(), final_networks=torch.tensor([[11, 1], [-1, -1]]))
    with pytest.raises(ValueError, match="impossible for the board size"):
        compute_losses(output, target, **loss_options())
    transferred = target.to("cpu")
    for name in AUX_NAMES:
        torch.testing.assert_close(
            getattr(transferred, name), getattr(target, name), equal_nan=True
        )
        torch.testing.assert_close(
            getattr(transferred, f"{name}_mask"), getattr(target, f"{name}_mask")
        )


def test_auxiliary_only_supervision_reaches_trunk_and_every_head():
    model = GraphResTNet(config())
    batch = encode_batch([position(4)])
    out = model(*batch.model_args())
    n = batch.max_nodes
    policy = torch.zeros(1, n)
    policy[0, 1] = 1
    target = replace(
        targets(1, n, n),
        opponent_reply=torch.cat((policy, torch.zeros(1, 1)), dim=-1),
        second_stone=policy,
        final_shores=torch.tensor([[3, 4]]),
        final_networks=torch.tensor([[1, 2]]),
        final_capes=torch.tensor([[2, 3]]),
        **{f"{name}_mask": torch.ones(1, dtype=torch.bool) for name in AUX_NAMES},
    )
    weights = LossWeights(
        policy=0,
        outcome=0,
        score_margin=0,
        ownership=0,
        alive=0,
        soft_policy=0,
        **{name: 0.1 for name in AUX_NAMES},
    )
    losses = compute_losses(
        out,
        target,
        legal_action_mask=batch.legal_action_mask,
        node_mask=batch.node_mask,
        weights=weights,
    )
    losses["total"].backward()
    assert torch.count_nonzero(model.node_projection.weight.grad) > 0
    for name, parameter in model.named_parameters():
        if any(
            head in name
            for head in (
                "opponent_reply",
                "second_stone",
                "final_shores",
                "final_networks",
                "final_capes",
            )
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
