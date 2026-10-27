from pathlib import Path

from repro_harness import compare_metric_maps, extract_metric_means, sha256_file


def test_sha256_file_returns_none_for_missing_path(tmp_path):
    missing = tmp_path / "missing.bin"
    assert sha256_file(missing) is None


def test_extract_metric_means_prefers_suite_aggregated_values():
    payload = {
        "definition_version": 1,
        "stats": {
            "cagr_fraction_strategy_equity": {
                "mean": 1.0,
                "min": 1.0,
                "max": 1.0,
                "std": 0.0,
            },
        },
        "suite_metrics": {
            "definition_version": 1,
            "metrics": {
                "cagr_fraction_strategy_equity": {
                    "aggregated": 2.0,
                    "stats": {"mean": 3.0, "min": 1.0, "max": 5.0, "std": 2.0},
                },
                "backtest_completion_fraction": {
                    "stats": {"mean": 0.9, "min": 0.8, "max": 1.0, "std": 0.1},
                },
            },
        },
    }

    metrics = extract_metric_means(payload)

    assert metrics["cagr_fraction_strategy_equity"] == 2.0
    assert metrics["backtest_completion_fraction"] == 0.9


def test_extract_metric_means_reads_top_level_metrics_block():
    payload = {
        "metrics": {
            "definition_version": 1,
            "stats": {
                "cagr_fraction_strategy_equity": {
                    "mean": 0.5,
                    "min": 0.5,
                    "max": 0.5,
                    "std": 0.0,
                },
            },
        }
    }

    metrics = extract_metric_means(payload)

    assert metrics["cagr_fraction_strategy_equity"] == 0.5


def test_compare_metric_maps_reports_only_differences():
    left = {
        "cagr_fraction_strategy_equity": 1.0,
        "ulcer_index_fraction_strategy_equity": 2.0,
    }
    right = {
        "cagr_fraction_strategy_equity": 1.0 + 1e-13,
        "ulcer_index_fraction_strategy_equity": 4.0,
        "omega_ratio_strategy_equity": 5.0,
    }

    diff = compare_metric_maps(left, right, tolerance=1e-12)

    assert "cagr_fraction_strategy_equity" not in diff
    assert diff["ulcer_index_fraction_strategy_equity"]["abs_diff"] == 2.0
    assert diff["omega_ratio_strategy_equity"]["right"] == 5.0


def test_repro_readers_preserve_infinite_and_unavailable_facts():
    import math
    import json
    from metrics_schema import encode_metric_values

    source = {
        "metric_metadata": {"definition_version": 1, "observation_count": 12},
        "liquidated": False,
        **encode_metric_values({"omega_ratio_strategy_equity": math.inf}),
    }
    source["sortino_ratio_strategy_equity"] = None
    source["metric_status"]["sortino_ratio_strategy_equity"] = {
        "status": "undefined",
        "reason": "flat sample",
    }
    values = extract_metric_means(source)
    assert values["omega_ratio_strategy_equity"] is None
    assert (
        values["metric_status"]["omega_ratio_strategy_equity"]["status"]
        == "positive_infinity"
    )
    assert "liquidated" not in values and "observation_count" not in values
    assert (
        compare_metric_maps(values, values)["sortino_ratio_strategy_equity"][
            "comparison"
        ]
        == "unavailable"
    )
    assert "omega_ratio_strategy_equity" not in compare_metric_maps(values, values)
    difference = compare_metric_maps(
        encode_metric_values({"omega_ratio_strategy_equity": -math.inf}), values
    )
    json.dumps(difference, allow_nan=False)
    assert (
        difference["omega_ratio_strategy_equity"]["metric_status"]["abs_diff"]["status"]
        == "positive_infinity"
    )


def test_repro_reader_rejects_stale_metric_definitions():
    import pytest

    with pytest.raises(ValueError, match="definition"):
        extract_metric_means(
            {"stats": {"cagr_fraction_strategy_equity": {"mean": 0.2}}}
        )


def test_suite_precedence_clears_replaced_unavailable_status():
    metrics = extract_metric_means(
        {
            "definition_version": 1,
            "stats": {
                "cagr_fraction_strategy_equity": {
                    "mean": None,
                    "metric_status": {"mean": {"status": "invalid"}},
                }
            },
            "suite_metrics": {
                "definition_version": 1,
                "metrics": {"cagr_fraction_strategy_equity": {"aggregated": 0.5}},
            },
        }
    )
    assert metrics == {"cagr_fraction_strategy_equity": 0.5}


def test_optimizer_replay_reads_current_evaluation_packet(monkeypatch):
    import asyncio
    import numpy as np
    import backtest
    import optimize
    import repro_harness
    from optimization.evaluation_payload import build_evaluation_payload

    async def noop(*args, **kwargs):
        return None

    async def prepare(*args, **kwargs):
        return ["BTC"], np.ones((3, 1, 4)), {"BTC": {}}, "unused", "unused", np.ones(3), np.arange(3)

    class Evaluator:
        def __init__(self, **kwargs):
            self.bounds = []
            self.sig_digits = []
            self.shared_hlcvs_np = {}
            self.shared_btc_np = {}

        def evaluate(self, individual, overrides):
            return build_evaluation_payload([-.2], 0, {"definition_version": 1,
                "stats": {"cagr_fraction_strategy_equity": {"mean": .2}}}, individual)

    monkeypatch.setattr(repro_harness, "format_approved_ignored_coins", noop)
    monkeypatch.setattr(repro_harness, "extract_suite_config", lambda *_: {"enabled": False})
    monkeypatch.setattr(backtest, "prepare_hlcvs_mss", prepare)
    monkeypatch.setattr(optimize, "_maybe_aggregate_backtest_data", lambda candles, timestamps, btc, *_: (candles, timestamps, btc))
    monkeypatch.setattr(optimize, "Evaluator", Evaluator)
    monkeypatch.setattr(optimize, "config_to_individual", lambda *_: [])
    result = asyncio.run(repro_harness.run_optimizer_replay({"backtest": {"exchanges": ["fake"]}, "optimize": {}}))
    assert result["objectives"] == [-.2]
    assert result["constraint_violation"] == 0
    assert result["metrics"] == {"cagr_fraction_strategy_equity": .2}
    assert result["selected_coins"] == {"fake": ["BTC"]}
