"""Keep archived arena evidence bit-identical while avoiding traced inner loops."""

import math
import random

import pytest

from deltreltrain import arena


def scalar_reference(observations, *, null_mean, direction):
    denominator = null_mean if direction == "greater" else 1.0 - null_mean
    transformed = (
        tuple(observations)
        if direction == "greater"
        else tuple(1.0 - observation for observation in observations)
    )
    wealths = [
        sum(
            math.log(1.0 - fraction + fraction * observation / denominator)
            for observation in transformed
        )
        for fraction in arena._PAIR_BETTING_FRACTIONS
    ]
    maximum = max(wealths)
    return (
        maximum
        + math.log(sum(math.exp(value - maximum) for value in wealths))
        - math.log(len(wealths))
    )


@pytest.mark.parametrize("direction", ["greater", "less"])
@pytest.mark.parametrize("null_mean", [1e-12, 0.17, 0.5, 0.83, 1.0 - 1e-12])
def test_log_evidence_preserves_scalar_bits(direction, null_mean):
    rng = random.Random(9183)
    for observations in (
        [],
        [0.0],
        [1.0],
        [0.0, 0.5, 1.0] * 833,
        [rng.random() for _ in range(257)],
    ):
        expected = scalar_reference(
            observations, null_mean=null_mean, direction=direction
        )
        actual = arena._bounded_log_e_value(
            observations, null_mean=null_mean, direction=direction
        )
        assert actual.hex() == expected.hex()


def test_confidence_bounds_preserve_every_bisection_decision(monkeypatch):
    observations = [0.0] * 27 + [0.5] * 84 + [1.0] * 192
    actual = arena.bounded_confidence_sequence(observations, error_probability=0.025)
    monkeypatch.setattr(arena, "_bounded_log_e_value", scalar_reference)
    expected = arena.bounded_confidence_sequence(observations, error_probability=0.025)
    assert tuple(value.hex() for value in actual) == tuple(
        value.hex() for value in expected
    )
