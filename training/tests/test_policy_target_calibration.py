import pytest

from scripts.analyze_policy_targets import compare_target


def test_unvisited_q_is_not_evidence_of_zero_value():
    row = {
        "actions": [0, 1],
        "policy_target": [0.2, 0.8],
        "priors": [0.5, 0.5],
        "selected_action": 1,
        "q_values": [0.7, 0],
        "visits": [10, 0],
    }
    result = compare_target(row, row, scale=1, max_kl=None)
    assert result["reference_visited_target_mass"] == pytest.approx(0.2)
    assert result["reference_expected_regret"] is None


def test_full_reference_coverage_allows_expected_regret():
    row = {
        "actions": [0, 1],
        "policy_target": [0.2, 0.8],
        "priors": [0.5, 0.5],
        "selected_action": 1,
        "q_values": [0.7, -0.3],
        "visits": [10, 10],
    }
    result = compare_target(row, row, scale=1, max_kl=None)
    assert result["reference_expected_regret"] == pytest.approx(0.8)
    softened = compare_target(row, row, scale=0.25, max_kl=None)
    assert softened["selected_action_unchanged"] == 1
    assert softened["entropy_nats"] > result["entropy_nats"]
    assert softened["kl_to_prior_nats"] < result["kl_to_prior_nats"]
