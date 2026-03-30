from pathlib import Path

from repro_harness import compare_metric_maps, extract_metric_means, sha256_file


def test_sha256_file_returns_none_for_missing_path(tmp_path):
    missing = tmp_path / "missing.bin"
    assert sha256_file(missing) is None


def test_extract_metric_means_prefers_suite_aggregated_values():
    payload = {
        "stats": {
            "adg_pnl": {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0},
        },
        "suite_metrics": {
            "metrics": {
                "adg_pnl": {
                    "aggregated": 2.0,
                    "stats": {"mean": 3.0, "min": 1.0, "max": 5.0, "std": 2.0},
                },
                "backtest_completion_ratio": {
                    "stats": {"mean": 0.9, "min": 0.8, "max": 1.0, "std": 0.1},
                },
            }
        },
    }

    metrics = extract_metric_means(payload)

    assert metrics["adg_pnl"] == 2.0
    assert metrics["backtest_completion_ratio"] == 0.9


def test_extract_metric_means_reads_top_level_metrics_block():
    payload = {
        "metrics": {
            "stats": {
                "adg_pnl": {"mean": 0.5, "min": 0.5, "max": 0.5, "std": 0.0},
            }
        }
    }

    metrics = extract_metric_means(payload)

    assert metrics["adg_pnl"] == 0.5


def test_compare_metric_maps_reports_only_differences():
    left = {"adg_pnl": 1.0, "mdg_pnl_w": 2.0}
    right = {"adg_pnl": 1.0 + 1e-13, "mdg_pnl_w": 4.0, "extra": 5.0}

    diff = compare_metric_maps(left, right, tolerance=1e-12)

    assert "adg_pnl" not in diff
    assert diff["mdg_pnl_w"]["abs_diff"] == 2.0
    assert diff["extra"]["right"] == 5.0
