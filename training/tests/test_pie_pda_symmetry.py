"""Pie swap shortcuts require equal physical-seat search strength."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from startrain.features import encode_position
from startrain.inference import InferenceResponse
from startrain.native import positions_from_native
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, VariantMixtureConfig


class UniformEvaluator:
    model_version = model_identity = "probe"
    model_step = 0

    def evaluate(self, requests):
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[0.0] * len(requests),
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class Sink:
    def __init__(self):
        self.samples = []

    def append(self, samples, **_kwargs):
        self.samples.extend(samples)
        return SimpleNamespace(sample_count=len(samples))


def actor(
    *,
    mode="double",
    pie=False,
    handicap=1,
    contract="cohort-v1",
    fraction=1.0,
    native=None,
    sink=None,
):
    config = replace(
        SelfPlayConfig.cpu_smoke(seed=17),
        mode=mode,
        pie=pie,
        handicap=handicap,
        seed_contract=contract,
        variants=VariantMixtureConfig(enabled=True, asymmetric_pda_fraction=fraction),
    )
    return SelfPlayActor(native or object(), UniformEvaluator(), sink or Sink(), config)


@pytest.mark.parametrize("mode", ["classic", "double"])
@pytest.mark.parametrize("contract", ["cohort-v1", "game-v1"])
@pytest.mark.parametrize("fraction", [0.0, 0.2, 1.0])
def test_pie_never_draws_asymmetric_seat_advantages(
    mode, contract, fraction, monkeypatch
):
    subject = actor(mode=mode, pie=True, contract=contract, fraction=fraction)
    monkeypatch.setattr(
        subject, "_seed", lambda *_: pytest.fail("pie PDA must not consume a seed")
    )
    assert (
        subject._draw_pda_seats(3, 1000, game_ids=[f"game-{i}" for i in range(1000)])
        == [(0, 0)] * 1000
    )


@pytest.mark.parametrize("mode", ["classic", "double"])
@pytest.mark.parametrize(
    "contract,expected",
    [
        (
            "cohort-v1",
            [
                (1, -1),
                (-2, 2),
                (1, -1),
                (2, -2),
                (2, -2),
                (2, -2),
                (2, -2),
                (-2, 2),
                (1, -1),
                (1, -1),
                (2, -2),
                (1, -1),
            ],
        ),
        (
            "game-v1",
            [
                (-2, 2),
                (-1, 1),
                (1, -1),
                (-2, 2),
                (-2, 2),
                (-1, 1),
                (-2, 2),
                (2, -2),
                (2, -2),
                (1, -1),
                (2, -2),
                (-2, 2),
            ],
        ),
    ],
)
def test_nonpie_retains_exact_legacy_pda_seed_streams(mode, contract, expected):
    subject = actor(mode=mode, contract=contract)
    ids = [f"game-{i}" for i in range(12)]
    before = [subject._seed("search-game-v1", game, 7) for game in ids]
    assert subject._draw_pda_seats(3, 12, game_ids=ids) == expected
    assert [subject._seed("search-game-v1", game, 7) for game in ids] == before


@pytest.mark.parametrize("mode", ["classic", "double"])
@pytest.mark.parametrize("handicap", [2, 4, 6, 9])
def test_handicap_compensation_is_unchanged(mode, handicap):
    subject = actor(mode=mode, handicap=handicap)
    advantage = subject.config.variants.pda_for_handicap(handicap)
    assert subject._draw_pda_seats(2, 12) == [(-advantage, advantage)] * 12


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_swap_relabeling_preserves_inputs_only_with_symmetric_pda(mode):
    native = pytest.importorskip("star_native")
    states = native.StateBatch(4, 1, mode=mode, pie=True)
    states.apply_many([0], [0])
    before = positions_from_native(states.data(), pda=[2])[0]
    states.apply_many([0], [states.node_count])
    after = positions_from_native(states.data(), pda=[-2])[0]
    # This comparison-only keep state removes the option; production replay
    # remains the true state. Relative stones/history match after recoloring.
    keep = encode_position(replace(before, swap_available=False))
    swap = encode_position(after)
    assert torch.equal(keep.node_features, swap.node_features)
    assert torch.nonzero(
        keep.global_features != swap.global_features
    ).flatten().tolist() == [24]
    symmetric_keep = encode_position(replace(before, swap_available=False, pda=0))
    symmetric_swap = encode_position(replace(after, pda=0))
    assert torch.equal(symmetric_keep.global_features, symmetric_swap.global_features)
    # Value is conditional on PDA. -abs(K(+d)) incorrectly assumes K(+d)=K(-d).
    keep_value, swapped_next_player_value = -0.1, -0.7
    actual_opening_value = -max(keep_value, -swapped_next_player_value)
    assert actual_opening_value == -0.7
    assert -abs(keep_value) == -0.1
    # With both alternatives positive, even the keep/swap decision can be wrong.
    assert 0.1 > 0 and -swapped_next_player_value > 0.1


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_real_pie_selfplay_records_symmetric_policy_and_zero_pda(mode):
    native = pytest.importorskip("star_native")
    sink = Sink()
    subject = actor(mode=mode, pie=True, native=native, sink=sink)
    summaries = subject.run()
    assert len(summaries) == 1
    assert (summaries[0].pda_seat0, summaries[0].pda_seat1) == (0, 0)
    assert sink.samples
    assert all(sample.pda == 0 for sample in sink.samples)
    assert all(
        "pie_pda=symmetric-seats-v1" in sample.search_provenance.split(":")
        for sample in sink.samples
    )
    assert subject.metrics_snapshot().asymmetric_games == 0
