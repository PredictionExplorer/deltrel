import pytest

from scripts.benchmark_replay_eligibility import benchmark


def test_benchmark_preserves_counts_and_measures_one_scan_per_step():
    report = benchmark(shards=180, checks_per_step=3, repeats=2)
    assert report["metadata_shards"] == 180
    assert len(report["scenarios"]) == 2
    for scenario in report["scenarios"]:
        assert scenario["identical_counts"] is True
        assert scenario["aggregate_scans"] == {"cached": 2, "uncached": 6}
        assert all(value > 0 for value in scenario["seconds_per_step_median"].values())


@pytest.mark.parametrize("field", ["shards", "checks_per_step", "repeats"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_benchmark_rejects_invalid_workloads(field, value):
    options = dict(shards=180, checks_per_step=3, repeats=2)
    options[field] = value
    with pytest.raises(ValueError, match="positive integers"):
        benchmark(**options)
