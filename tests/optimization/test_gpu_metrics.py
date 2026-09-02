import math
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.metrics import (
    ENTRY_INTERVAL_METRICS,
    GPU_EXACT_ONLY_METRICS,
    SUPPORTED_METRICS,
    _GAP_HIST_UPPER_STEPS,
    _btc_account_metrics,
    _daily_peak_recovery_ms,
    _equity_shape_metrics,
    _directional_pnl_metrics,
    _fill_activity_metrics,
    _entry_interval_metrics,
    _fill_gap_metrics,
    _hard_stop_lifecycle_metrics,
    _hard_stop_panic_loss_metrics,
    _strategy_eq_recovery_distribution_metrics,
    _loss_profit_ratio,
    _masked_median,
    _mean_worst_one_pct_abs,
    _omega_ratio,
    _pct_change,
    _sharpe_sortino,
    _smoothed_adg,
    _smoothed_gain_adg,
    _weighted_adg,
    _weighted_daily_series_metrics,
    _weighted_subset_context,
    _weighted_subsets,
    _weighted_strategy_eq_metrics,
    compute_objectives,
    validate_gpu_metric_names,
)


def test_exact_only_metrics_are_disjoint_and_fail_closed_for_proxy_scoring():
    assert not GPU_EXACT_ONLY_METRICS & set(SUPPORTED_METRICS)
    assert "strategy_eq_recovery_days_max" in SUPPORTED_METRICS
    assert "peak_recovery_days_strategy_eq" in GPU_EXACT_ONLY_METRICS

    for name in GPU_EXACT_ONLY_METRICS:
        with pytest.raises(ValueError, match="exact Rust backtests and analysis"):
            validate_gpu_metric_names([name])


def test_proxy_metric_validation_canonicalizes_retained_cpu_shorthand():
    assert validate_gpu_metric_names(["adg", "drawdown_worst"]) == {
        "adg_usd",
        "drawdown_worst_usd",
    }


def _assert_proxy_surface_partition(metric_names):
    names = set(metric_names)
    assert names - GPU_EXACT_ONLY_METRICS <= set(SUPPORTED_METRICS)
    assert not set(SUPPORTED_METRICS) & GPU_EXACT_ONLY_METRICS


def test_btc_daily_peak_recovery_matches_non_strict_rust_contract():
    day_end = torch.tensor(
        [
            [10.0, 9.0, 10.0, 8.0],
            [10.0, 9.0, 8.0, 7.0],
        ],
        dtype=torch.float64,
    )
    active = torch.ones_like(day_end, dtype=torch.bool)

    recovery = _daily_peak_recovery_ms(day_end, active) / 86_400_000.0

    assert recovery.tolist() == [2.0, 0.0]


def test_fill_activity_ratio_recovers_integer_steps_at_whole_day_boundary():
    first_step = 7_509
    last_step = first_step + 1_440
    first_ts = np.float32(first_step * 60_000)
    last_ts = np.float32(last_step * 60_000)
    assert (float(last_ts) - float(first_ts)) / 86_400_000.0 > 1.0

    metrics = _fill_activity_metrics(
        {
            "fill_count": torch.tensor([1.0]),
            "fill_count_entry": torch.tensor([1.0]),
            "fill_count_long": torch.tensor([1.0]),
            "fills_active_days_count": torch.tensor([1.0]),
            "first_eq_ts": torch.tensor([first_ts]),
            "last_eq_ts": torch.tensor([last_ts]),
        },
        SimpleNamespace(interval_ms=60_000),
        {
            "fills_active_days_ratio",
            "fills_active_symbols_count",
            "fills_analysis_duration_days",
            "fills_top_symbol_share",
        },
    )

    assert metrics["fills_analysis_duration_days"].item() == 1.0
    assert metrics["fills_active_days_ratio"].item() == 1.0
    assert metrics["fills_active_symbols_count"].item() == 1.0
    assert metrics["fills_top_symbol_share"].item() == 1.0


def test_loss_profit_ratio_matches_rust_cap_and_neutral_contract():
    actual = _loss_profit_ratio(
        torch.tensor([25.0, 5.0, 0.0, 2_000.0]),
        torch.tensor([100.0, 0.0, 0.0, 1.0]),
    )

    assert actual.tolist() == [0.25, 1_000.0, 1.0, 1_000.0]
    assert "loss_profit_ratio" in SUPPORTED_METRICS


def test_directional_pnl_metrics_match_rust_neutral_and_alias_contracts():
    metrics = _directional_pnl_metrics(
        {
            "profit_sum_long": torch.tensor([100.0, 0.0, 0.0]),
            "loss_sum_long": torch.tensor([25.0, 5.0, 0.0]),
            "profit_sum_short": torch.tensor([50.0, 0.0, 0.0]),
            "loss_sum_short": torch.tensor([75.0, 0.0, 0.0]),
        }
    )

    assert metrics["loss_profit_ratio_long"].tolist() == [0.25, 1_000.0, 1.0]
    assert metrics["loss_profit_ratio_short"].tolist() == [1.5, 1.0, 1.0]
    assert metrics["pnl_ratio_long_short"].tolist() == [1.5, 1.0, 0.5]
    assert set(metrics) <= set(SUPPORTED_METRICS)


def test_equity_balance_diff_and_paper_loss_metrics_match_rust_contract():
    day_end = torch.tensor([[100.0, 110.0, 121.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones_like(day_end, dtype=torch.bool),
        "fill_count": torch.tensor([2.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([2 * 86_400_000.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([2 * 86_400_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([2 * 86_400_000.0]),
        "liq_step": torch.tensor([-1]),
        "equity_balance_diff_pos_max": torch.tensor([0.20]),
        "equity_balance_diff_pos_mean": torch.tensor([0.10]),
        "equity_balance_diff_neg_max": torch.tensor([0.30]),
        "equity_balance_diff_neg_mean": torch.tensor([0.15]),
        "equity_balance_diff_pos_max_btc": torch.tensor([0.25]),
        "equity_balance_diff_pos_mean_btc": torch.tensor([0.125]),
        "equity_balance_diff_neg_max_btc": torch.tensor([0.35]),
        "equity_balance_diff_neg_mean_btc": torch.tensor([0.175]),
    }
    requested = {
        f"equity_balance_diff_{sign}_{stat}_{currency}"
        for sign in ("neg", "pos")
        for stat in ("max", "mean")
        for currency in ("btc", "usd")
    } | {
        f"paper_loss{middle}_ratio_{currency}"
        for middle in ("", "_mean")
        for currency in ("btc", "usd")
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=86_400_000,
        ),
        {
            "ts0": 0.0,
            "n": 3,
            "btc_day_end_price": np.ones(3),
            "btc_prices": np.ones(3),
        },
        needed=requested,
    )

    _gain, adg = _smoothed_gain_adg(
        day_end, torch.ones_like(day_end, dtype=torch.bool)
    )
    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    expected_differences = {
        "usd": (0.20, 0.10, 0.30, 0.15),
        "btc": (0.25, 0.125, 0.35, 0.175),
    }
    for currency, (pos_max, pos_mean, neg_max, neg_mean) in expected_differences.items():
        assert metrics[f"equity_balance_diff_pos_max_{currency}"].item() == pytest.approx(pos_max)
        assert metrics[f"equity_balance_diff_pos_mean_{currency}"].item() == pytest.approx(pos_mean)
        assert metrics[f"equity_balance_diff_neg_max_{currency}"].item() == pytest.approx(neg_max)
        assert metrics[f"equity_balance_diff_neg_mean_{currency}"].item() == pytest.approx(neg_mean)
        assert metrics[f"paper_loss_ratio_{currency}"].item() == pytest.approx(
            adg.item() / neg_max
        )
        assert metrics[f"paper_loss_mean_ratio_{currency}"].item() == pytest.approx(
            adg.item() / neg_mean
        )


def test_equity_balance_diff_metrics_fail_closed_without_metal_output():
    day_end = torch.tensor([[100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "fill_count": torch.zeros(1),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.full((1,), float("nan")),
        "last_fill_ts": torch.full((1,), float("nan")),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([0.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([0.0]),
        "liq_step": torch.tensor([-1]),
    }

    with pytest.raises(RuntimeError, match="equity-balance-diff output is missing"):
        compute_objectives(
            out,
            SimpleNamespace(
                requested_start_ts_ms=0,
                guard_ts_ms=0,
                interval_ms=60_000,
            ),
            {"ts0": 0.0, "n": 1},
            needed={"paper_loss_ratio_usd"},
        )


def test_equity_balance_diff_metrics_use_rust_defaults_without_fills():
    day_end = torch.tensor([[100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "fill_count": torch.zeros(1),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.full((1,), float("nan")),
        "last_fill_ts": torch.full((1,), float("nan")),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([0.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([0.0]),
        "liq_step": torch.tensor([-1]),
        "equity_balance_diff_pos_max": torch.zeros(1),
        "equity_balance_diff_pos_mean": torch.zeros(1),
        "equity_balance_diff_neg_max": torch.zeros(1),
        "equity_balance_diff_neg_mean": torch.zeros(1),
        "equity_balance_diff_pos_max_btc": torch.zeros(1),
        "equity_balance_diff_pos_mean_btc": torch.zeros(1),
        "equity_balance_diff_neg_max_btc": torch.zeros(1),
        "equity_balance_diff_neg_mean_btc": torch.zeros(1),
    }
    requested = {
        f"equity_balance_diff_{sign}_{stat}_{currency}"
        for sign in ("neg", "pos")
        for stat in ("max", "mean")
        for currency in ("btc", "usd")
    } | {
        f"paper_loss{middle}_ratio_{currency}"
        for middle in ("", "_mean")
        for currency in ("btc", "usd")
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=60_000,
        ),
        {
            "ts0": 0.0,
            "n": 1,
            "btc_day_end_price": np.ones(1),
            "btc_prices": np.ones(1),
        },
        needed=requested,
    )

    for name in requested:
        expected = 0.0 if name.startswith("paper_loss") else 1.0
        assert metrics[name].item() == expected


def test_equity_shape_metrics_match_rust_daily_series_contract():
    day_eq = torch.tensor(
        [
            [100.0, 110.0, 90.0, 120.0],
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 110.0, float("inf"), float("inf")],
            [100.0, 0.0, 110.0, float("inf")],
            [100.0, float("inf"), float("inf"), float("inf")],
            [100.0, float("inf"), 110.0, 90.0],
        ],
        dtype=torch.float64,
    )
    active = torch.isfinite(day_eq)

    metrics = _equity_shape_metrics(day_eq, active)

    assert metrics["equity_choppiness_usd"][0].item() == pytest.approx(3.0)
    assert metrics["equity_jerkiness_usd"][0].item() == pytest.approx(
        (0.3 + 0.46875) / 2.0
    )
    log_equity = np.log(np.array([100.0, 110.0, 90.0, 120.0]))
    x = np.arange(4, dtype=float)
    slope, intercept = np.polyfit(x, log_equity, 1)
    expected_fit_error = np.mean((slope * x + intercept - log_equity) ** 2)
    assert metrics["exponential_fit_error_usd"][0].item() == pytest.approx(
        expected_fit_error
    )
    assert math.isinf(metrics["equity_choppiness_usd"][1].item())
    assert metrics["equity_jerkiness_usd"][2].item() == 0.0
    assert metrics["exponential_fit_error_usd"][2].item() == pytest.approx(0.0)
    assert math.isinf(metrics["exponential_fit_error_usd"][3].item())
    assert metrics["equity_choppiness_usd"][4].item() == 0.0
    assert metrics["equity_jerkiness_usd"][4].item() == 0.0
    assert math.isinf(metrics["exponential_fit_error_usd"][4].item())
    assert metrics["equity_choppiness_usd"][5].item() == pytest.approx(3.0)
    assert metrics["equity_jerkiness_usd"][5].item() == pytest.approx(0.3)


def test_equity_curve_metrics_use_rust_defaults_without_fills():
    day_eq = torch.tensor([[100.0, 110.0, 90.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_eq,
        "day_min_eq": day_eq.clone(),
        "day_max_dd": torch.zeros_like(day_eq),
        "day_volume": torch.zeros_like(day_eq),
        "day_has_fill": torch.zeros_like(day_eq, dtype=torch.bool),
        "fill_count": torch.zeros(1),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.full((1,), float("nan")),
        "last_fill_ts": torch.full((1,), float("nan")),
        "recovery_max_ms": torch.zeros(1),
        "account_recovery_max_ms": torch.tensor([60_000.0]),
        "last_high_ts": torch.tensor([2 * 86_400_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([2 * 86_400_000.0]),
        "liq_step": torch.tensor([-1]),
    }
    requested = {
        "equity_choppiness_usd",
        "equity_jerkiness_usd",
        "exponential_fit_error_usd",
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=86_400_000,
        ),
        {"ts0": 0.0, "n": 3},
        needed=requested,
    )

    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    assert {name: value.item() for name, value in metrics.items()} == {
        "equity_choppiness_usd": 1.0,
        "equity_jerkiness_usd": 1.0,
        "exponential_fit_error_usd": 1.0,
        "peak_recovery_days_equity_usd": 0.0,
        "peak_recovery_hours_equity_usd": 0.0,
    }


def test_fill_activity_metrics_match_rust_full_timestamp_span_contract():
    day_end = torch.full((2, 3), 100.0, dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.tensor(
            [[True, True, False], [False, False, False]]
        ),
        "day_fill_count": torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        ),
        "fill_count": torch.tensor([5.0, 0.0]),
        "fill_count_entry": torch.tensor([3.0, 0.0]),
        "fill_count_long": torch.tensor([4.0, 0.0]),
        "fills_active_days_count": torch.tensor([2.0, 0.0]),
        "coin_fill_counts": torch.tensor([[4.0, 1.0], [0.0, 0.0]]),
        "position_slots_long": torch.tensor([2.0, 1.0]),
        "position_slots_short": torch.tensor([1.0, 0.0]),
        "max_dd": torch.zeros(2),
        "held_max_ms": torch.zeros(2),
        "position_unchanged_max_ms": torch.zeros(2),
        "gap_hist": torch.zeros((2, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(2),
        "first_fill_ts": torch.tensor([0.0, float("nan")]),
        "last_fill_ts": torch.tensor([3_600_000.0, float("nan")]),
        "recovery_max_ms": torch.zeros(2),
        "last_high_ts": torch.tensor([216_000_000.0, 14_400_000.0]),
        "first_eq_ts": torch.tensor([0.0, 0.0]),
        "last_eq_ts": torch.tensor([216_000_000.0, 14_400_000.0]),
        "liq_step": torch.tensor([-1.0, -1.0]),
    }
    requested = {
        "fills_active_days_count",
        "fills_active_days_ratio",
        "fills_active_symbols_count",
        "fills_analysis_duration_days",
        "fills_count",
        "fills_count_close",
        "fills_count_entry",
        "fills_count_long",
        "fills_count_short",
        "fills_entry_per_close",
        "fills_per_day",
        "fills_per_day_close",
        "fills_per_day_entry",
        "fills_per_day_long",
        "fills_per_day_per_position_slot",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
        "fills_per_day_short",
        "fills_top_symbol_share",
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
        ),
        {"ts0": 0.0, "n": 241},
        needed=requested,
    )

    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    assert metrics["fills_analysis_duration_days"].tolist() == pytest.approx(
        [2.5, 4.0 / 24.0]
    )
    assert metrics["fills_active_days_count"].tolist() == [2.0, 0.0]
    assert metrics["fills_active_days_ratio"].tolist() == pytest.approx(
        [2.0 / 3.0, 0.0]
    )
    assert metrics["fills_active_symbols_count"].tolist() == [2.0, 0.0]
    assert metrics["fills_top_symbol_share"].tolist() == pytest.approx([0.8, 0.0])
    assert metrics["fills_count"].tolist() == [5.0, 0.0]
    assert metrics["fills_count_entry"].tolist() == [3.0, 0.0]
    assert metrics["fills_count_close"].tolist() == [2.0, 0.0]
    assert metrics["fills_count_long"].tolist() == [4.0, 0.0]
    assert metrics["fills_count_short"].tolist() == [1.0, 0.0]
    assert metrics["fills_entry_per_close"].tolist() == [1.5, 0.0]
    assert metrics["fills_per_day"].tolist() == pytest.approx([2.0, 0.0])
    assert metrics["fills_per_day_entry"].tolist() == pytest.approx([1.2, 0.0])
    assert metrics["fills_per_day_close"].tolist() == pytest.approx([0.8, 0.0])
    assert metrics["fills_per_day_long"].tolist() == pytest.approx([1.6, 0.0])
    assert metrics["fills_per_day_short"].tolist() == pytest.approx([0.4, 0.0])
    assert metrics["fills_per_day_per_position_slot_long"].tolist() == pytest.approx(
        [0.8, 0.0]
    )
    assert metrics["fills_per_day_per_position_slot_short"].tolist() == pytest.approx(
        [0.4, 0.0]
    )
    assert metrics["fills_per_day_per_position_slot"].tolist() == pytest.approx(
        [0.6, 0.0]
    )


def test_fill_activity_metrics_ignore_inactive_daily_slots_and_zero_single_sample_span():
    day_min = torch.tensor([[100.0, float("inf")]], dtype=torch.float64)
    out = {
        "day_end_eq": torch.tensor([[100.0, 0.0]], dtype=torch.float64),
        "day_min_eq": day_min,
        "day_max_dd": torch.zeros_like(day_min),
        "day_volume": torch.zeros_like(day_min),
        "day_has_fill": torch.tensor([[True, False]]),
        "day_fill_count": torch.tensor([[2.0, 99.0]], dtype=torch.float64),
        "fill_count": torch.tensor([2.0]),
        "fill_count_entry": torch.tensor([1.0]),
        "fill_count_long": torch.tensor([2.0]),
        "fills_active_days_count": torch.tensor([0.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "position_unchanged_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([0.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([0.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([0.0]),
        "liq_step": torch.tensor([-1.0]),
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
        ),
        {"ts0": 0.0, "n": 1},
        needed={
            "fills_analysis_duration_days",
            "fills_count",
            "fills_per_day",
        },
    )

    assert metrics["fills_count"].item() == 2.0
    assert metrics["fills_analysis_duration_days"].item() == 0.0
    assert metrics["fills_per_day"].item() == 0.0


def test_duration_alias_metrics_match_rust_unit_contracts():
    day_end = torch.tensor([[100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "fill_count": torch.ones(1, dtype=torch.float64),
        "max_dd": torch.zeros(1, dtype=torch.float64),
        "held_max_ms": torch.tensor([36 * 3_600_000.0], dtype=torch.float64),
        "held_sum_ms": torch.tensor([48 * 3_600_000.0], dtype=torch.float64),
        "held_count": torch.tensor([2.0], dtype=torch.float64),
        "position_unchanged_max_ms": torch.tensor(
            [18 * 3_600_000.0], dtype=torch.float64
        ),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.tensor([30 * 3_600_000.0], dtype=torch.float64),
        "account_recovery_max_ms": torch.tensor(
            [12 * 3_600_000.0], dtype=torch.float64
        ),
        "pnl_recovery_max_ms": torch.tensor(
            [18 * 3_600_000.0], dtype=torch.float64
        ),
        "last_high_ts": torch.tensor([30 * 3_600_000.0], dtype=torch.float64),
        "first_eq_ts": torch.tensor([0.0], dtype=torch.float64),
        "last_eq_ts": torch.tensor([36 * 3_600_000.0], dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
        "entry_initial_balance_pct_long": torch.tensor([0.125]),
        "entry_initial_balance_pct_short": torch.tensor([0.075]),
        "total_wallet_exposure_max": torch.tensor([1.25]),
        "total_wallet_exposure_mean": torch.tensor([0.625]),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )
    requested = {
        "position_held_days_mean",
        "position_held_days_max",
        "position_held_hours_mean",
        "position_held_hours_max",
        "positions_held_per_day",
        "position_unchanged_days_max",
        "position_unchanged_hours_max",
        "strategy_eq_recovery_days_max",
        "peak_recovery_days_strategy_eq",
        "peak_recovery_hours_strategy_eq",
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
        "peak_recovery_days_pnl",
        "peak_recovery_hours_pnl",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "adg_strategy_eq",
        "exposure_ratio_usd",
        "exposure_mean_ratio_usd",
    }

    metrics = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 36 * 60 + 1},
        needed=requested,
    )

    assert set(metrics) == requested
    assert metrics["position_held_days_mean"].item() == pytest.approx(1.0)
    assert metrics["position_held_days_max"].item() == pytest.approx(1.5)
    assert metrics["position_held_hours_mean"].item() == pytest.approx(24.0)
    assert metrics["position_held_hours_max"].item() == pytest.approx(36.0)
    assert metrics["positions_held_per_day"].item() == pytest.approx(4.0 / 3.0)
    assert metrics["position_unchanged_hours_max"].item() == pytest.approx(18.0)
    assert metrics["position_unchanged_days_max"].item() == pytest.approx(0.75)
    assert metrics["strategy_eq_recovery_days_max"].item() == pytest.approx(1.25)
    assert metrics["peak_recovery_days_strategy_eq"].item() == pytest.approx(1.25)
    assert metrics["peak_recovery_hours_strategy_eq"].item() == pytest.approx(30.0)
    assert metrics["peak_recovery_days_equity_usd"].item() == pytest.approx(0.5)
    assert metrics["peak_recovery_hours_equity_usd"].item() == pytest.approx(12.0)
    assert metrics["peak_recovery_days_pnl"].item() == pytest.approx(0.75)
    assert metrics["peak_recovery_hours_pnl"].item() == pytest.approx(18.0)
    assert metrics["entry_initial_balance_pct_long"].item() == pytest.approx(0.125)
    assert metrics["entry_initial_balance_pct_short"].item() == pytest.approx(0.075)
    assert metrics["total_wallet_exposure_max"].item() == pytest.approx(1.25)
    assert metrics["total_wallet_exposure_mean"].item() == pytest.approx(0.625)
    adg = metrics["adg_strategy_eq"].item()
    assert metrics["exposure_ratio_usd"].item() == pytest.approx(adg / 1.25)
    assert metrics["exposure_mean_ratio_usd"].item() == pytest.approx(adg / 0.625)
    _assert_proxy_surface_partition(requested)


def test_zero_variance_sharpe_and_sortino_match_rust_zero_contract():
    changes = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[True, True]])
    adg = torch.tensor([0.1], dtype=torch.float64)

    sharpe, sortino = _sharpe_sortino(changes, mask, adg)

    assert sharpe.item() == 0.0
    assert sortino.item() == 0.0


def test_daily_pnl_metrics_match_rust_fill_day_contract():
    day_end = torch.full((1, 4), 100.0, dtype=torch.float64)
    day_has_fill = torch.tensor([[True, True, False, True]])
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": day_has_fill,
        "day_net_pnl": torch.tensor([[10.0, -5.0, 999.0, 0.0]]),
        "day_last_fill_balance": torch.tensor([[110.0, 105.0, 1.0, 105.0]]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "position_unchanged_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([180_000.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([180_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([180_000.0]),
        "liq_step": torch.tensor([-1]),
    }
    requested = {
        "adg_pnl",
        "mdg_pnl",
        "sharpe_ratio_pnl",
        "sortino_ratio_pnl",
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
        ),
        {"ts0": 0.0, "n": 4},
        needed=requested,
    )

    ratios = torch.tensor([10.0 / 110.0, -5.0 / 105.0, 0.0])
    mean = ratios.mean()
    std = torch.sqrt(((ratios - mean) ** 2).mean())
    downside = torch.sqrt((ratios[ratios < 0.0] ** 2).mean())
    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    assert metrics["adg_pnl"].item() == pytest.approx(mean.item())
    assert metrics["mdg_pnl"].item() == 0.0
    assert metrics["sharpe_ratio_pnl"].item() == pytest.approx(
        (mean / std).item()
    )
    assert metrics["sortino_ratio_pnl"].item() == pytest.approx(
        (mean / downside).item()
    )


def test_weighted_daily_pnl_metrics_match_rust_suffix_contract():
    day_ms = 86_400_000
    ratios = torch.tensor(
        [0.10, -0.05, 0.02, -0.01, 0.03, 0.04, -0.02, 0.01, 0.05, -0.03],
        dtype=torch.float64,
    )
    balances = torch.full((1, 10), 100.0, dtype=torch.float64)
    day_end = torch.full((1, 10), 100.0, dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones((1, 10), dtype=torch.bool),
        "day_net_pnl": ratios.unsqueeze(0) * balances,
        "day_last_fill_balance": balances,
        "day_fill_count": torch.ones((1, 10), dtype=torch.float64),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "position_unchanged_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([9.0 * day_ms]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([9.0 * day_ms]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([9.0 * day_ms]),
        "liq_step": torch.tensor([-1]),
    }
    requested = {
        "adg_pnl_w",
        "mdg_pnl_w",
        "sharpe_ratio_pnl_w",
        "sortino_ratio_pnl_w",
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=day_ms),
        {"ts0": 0.0, "n": 10},
        needed=requested,
    )

    suffixes = [
        ratios,
        ratios[5:],
        ratios[7:],
        ratios[8:],
        ratios[8:],
        ratios[8:],
        ratios[9:],
        ratios[9:],
        ratios[9:],
        ratios[9:],
    ]
    expected = {name: 0.0 for name in requested}
    for suffix in suffixes:
        mean = suffix.mean()
        median = suffix.median() if len(suffix) % 2 else suffix.sort().values[
            len(suffix) // 2 - 1 : len(suffix) // 2 + 1
        ].mean()
        std = torch.sqrt(((suffix - mean) ** 2).mean())
        downside_values = suffix[suffix < 0.0]
        downside = (
            torch.sqrt((downside_values**2).mean())
            if len(downside_values)
            else torch.tensor(0.0)
        )
        expected["adg_pnl_w"] += mean.item() / 10.0
        expected["mdg_pnl_w"] += median.item() / 10.0
        expected["sharpe_ratio_pnl_w"] += (
            0.0 if std.item() == 0.0 else (mean / std).item() / 10.0
        )
        expected["sortino_ratio_pnl_w"] += (
            0.0 if downside.item() == 0.0 else (mean / downside).item() / 10.0
        )
    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    for name, value in expected.items():
        assert metrics[name].item() == pytest.approx(value)


def test_weighted_daily_series_metrics_match_rust_suffix_contract():
    day_ms = 86_400_000
    day_count = 100
    day_index = torch.arange(day_count, dtype=torch.float64)
    day_end = (100.0 * torch.exp(day_index * 0.001)).unsqueeze(0)
    day_volume = (day_index + 1.0).unsqueeze(0)
    active = torch.ones_like(day_end, dtype=torch.bool)
    day_has_fill = torch.ones_like(active)
    requested = {
        "equity_choppiness_w_usd",
        "equity_jerkiness_w_usd",
        "exponential_fit_error_w_usd",
        "volume_pct_per_day_avg_w",
    }

    metrics = _weighted_daily_series_metrics(
        day_end,
        day_volume,
        day_has_fill,
        active,
        torch.tensor([float(day_count)]),
        torch.tensor([(day_count - 1) * day_ms], dtype=torch.float64),
        torch.tensor([0.0]),
        torch.tensor([(day_count - 1) * day_ms], dtype=torch.float64),
        0.0,
        day_ms,
        requested,
    )

    starts = [0, 50, 67, 75, 80, 83, 86, 88, 89, 90]

    def reference_shape(values):
        values = np.asarray(values, dtype=np.float64)
        choppiness = np.abs(np.diff(values)).sum() / abs(values[-1] - values[0])
        jerkiness = np.mean(
            np.abs(values[2:] - 2.0 * values[1:-1] + values[:-2])
            / np.abs((values[:-2] + values[1:-1] + values[2:]) / 3.0)
        )
        x = np.arange(len(values), dtype=np.float64)
        y = np.log(values)
        slope = (
            len(values) * np.sum(x * y) - np.sum(x) * np.sum(y)
        ) / (len(values) * np.sum(x * x) - np.sum(x) ** 2)
        intercept = (np.sum(y) - slope * np.sum(x)) / len(values)
        fit_error = np.mean((slope * x + intercept - y) ** 2)
        return choppiness, jerkiness, fit_error

    expected_shape = np.asarray(
        [reference_shape(day_end[0, start:].numpy()) for start in starts]
    ).mean(axis=0)
    expected_volume = np.mean(
        [day_volume[0, start:].mean().item() for start in starts]
    )

    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    assert metrics["equity_choppiness_w_usd"].item() == pytest.approx(
        expected_shape[0]
    )
    assert metrics["equity_jerkiness_w_usd"].item() == pytest.approx(
        expected_shape[1]
    )
    assert metrics["exponential_fit_error_w_usd"].item() == pytest.approx(
        expected_shape[2], abs=1e-24
    )
    assert metrics["volume_pct_per_day_avg_w"].item() == pytest.approx(
        expected_volume
    )


def test_weighted_daily_series_metrics_preserve_sparse_fill_rust_defaults():
    day_end = torch.tensor([[100.0, 101.0]], dtype=torch.float64)
    requested = set(
        [
            "equity_choppiness_w_usd",
            "equity_jerkiness_w_usd",
            "exponential_fit_error_w_usd",
            "volume_pct_per_day_avg_w",
        ]
    )

    metrics = _weighted_daily_series_metrics(
        day_end,
        torch.tensor([[0.5, 0.0]], dtype=torch.float64),
        torch.tensor([[True, False]]),
        torch.ones_like(day_end, dtype=torch.bool),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([86_400_000.0]),
        0.0,
        86_400_000,
        requested,
    )

    assert set(metrics) == requested
    assert metrics["volume_pct_per_day_avg_w"].item() == 0.0
    for name in requested - {"volume_pct_per_day_avg_w"}:
        assert metrics[name].item() == 1.0


def test_weighted_daily_series_metrics_keep_one_sample_full_run_tenth():
    requested = {
        "equity_choppiness_w_usd",
        "equity_jerkiness_w_usd",
        "exponential_fit_error_w_usd",
        "volume_pct_per_day_avg_w",
    }

    metrics = _weighted_daily_series_metrics(
        torch.tensor([[100.0]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        0.0,
        60_000,
        requested,
    )

    assert metrics["equity_choppiness_w_usd"].item() == 0.0
    assert metrics["equity_jerkiness_w_usd"].item() == 0.0
    assert math.isinf(metrics["exponential_fit_error_w_usd"].item())
    assert metrics["volume_pct_per_day_avg_w"].item() == pytest.approx(0.05)


def test_weighted_volume_excludes_ambiguous_intraday_cutoff_day():
    metrics = _weighted_daily_series_metrics(
        torch.tensor([[100.0]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
        torch.tensor([2.0]),
        torch.tensor([3_600_000.0]),
        torch.tensor([0.0]),
        torch.tensor([3_600_000.0]),
        0.0,
        60_000,
        {"volume_pct_per_day_avg_w"},
    )

    # Every trailing cutoff is inside the same UTC day. The daily aggregate
    # contains pre-cutoff volume, so only the unambiguous full-run tenth is
    # admitted; exact Rust validation owns partial-day volume after a cutoff.
    assert metrics["volume_pct_per_day_avg_w"].item() == pytest.approx(0.05)


def test_weighted_suffix_admission_uses_integer_candle_steps():
    interval_ms = 60_000
    sample_count = 17_898
    first_step = 1
    last_step = first_step + sample_count - 1
    active = torch.ones((1, 13), dtype=torch.bool)

    _, _, subset_start_steps, _, _ = _weighted_subset_context(
        active,
        torch.tensor([np.float32(first_step * interval_ms)]),
        torch.tensor([np.float32(last_step * interval_ms)]),
        0.0,
        interval_ms,
    )
    last_fill_ts = torch.tensor(
        [np.float32((first_step + sample_count // 2) * interval_ms)]
    )
    last_fill_step = torch.floor(
        last_fill_ts / float(interval_ms) + 0.5
    ).to(torch.long)

    # The equivalent float32 timestamp expressions differ by 64 ms at this
    # realistic span. Integer candle-step admission still includes Rust's
    # exact half-window boundary fill.
    assert np.float32(first_step * interval_ms) + np.float32(
        (sample_count // 2) * interval_ms
    ) != last_fill_ts.item()
    assert subset_start_steps[1].item() == last_fill_step.item()

    day_end = torch.arange(100.0, 113.0).unsqueeze(0)
    day_has_fill = torch.zeros_like(day_end, dtype=torch.bool)
    day_has_fill[:, 6] = True
    metrics = _weighted_daily_series_metrics(
        day_end,
        torch.zeros_like(day_end),
        day_has_fill,
        active,
        torch.tensor([2.0]),
        last_fill_ts,
        torch.tensor([np.float32(first_step * interval_ms)]),
        torch.tensor([np.float32(last_step * interval_ms)]),
        0.0,
        interval_ms,
        {"equity_choppiness_w_usd"},
    )

    # Linear full and half-window daily series both have choppiness 1.0;
    # Rust admits both and stops at the next suffix, for a weighted 0.2.
    assert metrics["equity_choppiness_w_usd"].item() == pytest.approx(0.2)


def test_weighted_subsets_normalize_relative_timestamps_to_unix_origin():
    day_ms = 86_400_000
    origin = 1_704_067_200_000.0
    active = torch.ones((1, 10), dtype=torch.bool)

    relative_eligible, relative_subsets = _weighted_subsets(
        active,
        torch.tensor([0.0]),
        torch.tensor([9.0 * day_ms]),
        origin,
        day_ms,
    )
    absolute_eligible, absolute_subsets = _weighted_subsets(
        active,
        torch.tensor([origin]),
        torch.tensor([origin + 9.0 * day_ms]),
        origin,
        day_ms,
    )

    assert relative_eligible.item()
    assert torch.equal(relative_eligible, absolute_eligible)
    for relative, absolute in zip(relative_subsets, absolute_subsets):
        assert torch.equal(relative, absolute)
    assert relative_subsets[1].tolist() == [[False] * 5 + [True] * 5]


def test_weighted_pnl_uses_fill_count_not_fill_day_count_for_eligibility():
    day_ms = 86_400_000
    day_end = torch.full((1, 10), 100.0, dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.tensor([[True] + [False] * 9]),
        "day_net_pnl": torch.tensor([[10.0] + [0.0] * 9]),
        "day_last_fill_balance": torch.tensor([[110.0] + [100.0] * 9]),
        "day_fill_count": torch.tensor([[2.0] + [0.0] * 9]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "position_unchanged_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([0.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([9.0 * day_ms]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([9.0 * day_ms]),
        "liq_step": torch.tensor([-1]),
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=day_ms),
        {"ts0": 0.0, "n": 10},
        needed={"adg_pnl_w"},
    )

    assert metrics["adg_pnl_w"].item() == pytest.approx((10.0 / 110.0) / 10.0)


def test_empty_median_return_series_matches_rust_zero_contract():
    values = torch.empty((1, 0), dtype=torch.float64)
    mask = torch.empty((1, 0), dtype=torch.bool)

    assert _masked_median(values, mask).item() == 0.0


def test_even_median_averages_middle_values_like_rust():
    values = torch.tensor([[0.1, 0.5, 99.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])

    assert _masked_median(values, mask).item() == pytest.approx(0.3)


def test_weighted_adg_keeps_short_active_subsets_nonempty():
    day_eq = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    active = torch.tensor([[True, True]])

    assert _weighted_adg(
        day_eq,
        active,
        torch.tensor([0.0]),
        torch.tensor([172_740_000.0]),
        0,
        60_000,
    ).item() == pytest.approx(0.0)


def test_weighted_adg_slices_minutes_before_daily_reduction():
    day_eq = torch.tensor([[100.0, 100.0, 100.0, 121.0]], dtype=torch.float64)
    active = torch.tensor([[True, True, True, True]])
    last_ts = float((4 * 1440 - 1) * 60_000)

    actual = _weighted_adg(
        day_eq,
        active,
        torch.tensor([0.0]),
        torch.tensor([last_ts]),
        0,
        60_000,
    )

    full = _smoothed_adg(day_eq, active)
    last_two = _smoothed_adg(day_eq, torch.tensor([[False, False, True, True]]))
    assert actual.item() == pytest.approx(((full + 2.0 * last_two) / 10.0).item())


def test_fill_gap_summary_metric_surface_is_supported():
    metrics = {
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p95_hours",
        "fills_gap_p99_hours",
        "fills_gap_time_weighted_mean_hours",
    }
    _assert_proxy_surface_partition(metrics)
    assert metrics & set(SUPPORTED_METRICS) == {
        "fills_gap_p95_hours",
        "fills_gap_time_weighted_mean_hours",
    }


def test_fill_gap_summary_without_fills_uses_whole_active_span():
    out = {
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "first_fill_ts": torch.tensor([float("nan")]),
        "last_fill_ts": torch.tensor([float("nan")]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([7_200_000.0]),
    }

    metrics = _fill_gap_metrics(out, SimpleNamespace(interval_ms=60_000))

    assert all(value.item() == pytest.approx(2.0) for value in metrics.values())


def test_fill_gap_histogram_is_conservative_for_interpolated_percentiles():
    gap_hist = torch.zeros((1, 128), dtype=torch.int32)
    gap_minutes = 120
    bin_index = int(math.log(gap_minutes + 1.0) * 127.0 / math.log(4_000_001.0))
    gap_hist[0, bin_index] = 1
    out = {
        "gap_hist": gap_hist,
        "first_fill_ts": torch.tensor([3_600_000.0]),
        "last_fill_ts": torch.tensor([10_800_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([14_400_000.0]),
    }

    metrics = _fill_gap_metrics(out, SimpleNamespace(interval_ms=60_000))

    # Exact distinct-candle gaps are [1h, 2h, 1h]. Log-bin upper edges may
    # overstate the inter-fill gap but must never make a minimizing proxy
    # metric more optimistic than exact Rust.
    assert metrics["fills_gap_mean_hours"].item() >= 4.0 / 3.0
    assert metrics["fills_gap_median_hours"].item() >= 1.0
    assert metrics["fills_gap_p95_hours"].item() >= 1.9
    assert metrics["fills_gap_p99_hours"].item() >= 1.98
    assert metrics["fills_gap_time_weighted_mean_hours"].item() >= 1.5


def test_fill_gap_float32_bin_decode_never_understates_boundary_samples():
    samples = {0, 1, 4_000_000}
    log_max = math.log(4_000_001.0)
    for index in range(127):
        edge = math.exp((index + 1) * log_max / 127.0) - 1.0
        center = math.floor(edge)
        samples.update(max(0, center + delta) for delta in range(-2, 3))

    for gap in samples:
        encoded = int(
            np.float32(
                np.log(np.float32(gap) + np.float32(1.0))
                * np.float32(127.0)
                / np.log(np.float32(4_000_001.0))
            )
        )
        encoded = min(max(encoded, 0), 127)
        assert _GAP_HIST_UPPER_STEPS[encoded] >= gap


def test_fill_gap_boundary_decode_recovers_large_float32_candle_offsets():
    interval_ms = 60_000
    first_eq_step = 3_900_000
    first_fill_step = first_eq_step + 1
    last_fill_step = first_eq_step + 3
    last_eq_step = first_eq_step + 4
    out = {
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "first_fill_ts": torch.tensor(
            [first_fill_step * interval_ms], dtype=torch.float32
        ),
        "last_fill_ts": torch.tensor(
            [last_fill_step * interval_ms], dtype=torch.float32
        ),
        "first_eq_ts": torch.tensor(
            [first_eq_step * interval_ms], dtype=torch.float32
        ),
        "last_eq_ts": torch.tensor(
            [last_eq_step * interval_ms], dtype=torch.float32
        ),
    }

    metrics = _fill_gap_metrics(out, SimpleNamespace(interval_ms=interval_ms))

    assert metrics["fills_gap_mean_hours"].item() == pytest.approx(1.0 / 60.0)
    assert metrics["fills_gap_median_hours"].item() == pytest.approx(1.0 / 60.0)
    assert metrics["fills_gap_p95_hours"].item() == pytest.approx(1.0 / 60.0)
    assert metrics["fills_gap_p99_hours"].item() == pytest.approx(1.0 / 60.0)


def test_fill_gap_time_weighted_mean_uses_exact_boundary_gaps():
    out = {
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "first_fill_ts": torch.tensor([3_600_000.0]),
        "last_fill_ts": torch.tensor([3_600_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([7_200_000.0]),
    }

    metrics = _fill_gap_metrics(out, SimpleNamespace(interval_ms=60_000))

    assert metrics["fills_gap_time_weighted_mean_hours"].item() == pytest.approx(1.0)


def _entry_interval_output(gaps):
    histogram = torch.zeros((1, 128), dtype=torch.float32)
    for gap in gaps:
        bin_index = int(
            np.float32(
                np.log(np.float32(gap) + np.float32(1.0))
                * np.float32(127.0)
                / np.log(np.float32(4_000_001.0))
            )
        )
        histogram[0, min(max(bin_index, 0), 127)] += 1.0
    return {
        "fill_count": torch.tensor([4.0]),
        "entry_interval_sum_steps": torch.tensor([sum(gaps)], dtype=torch.float32),
        "entry_interval_count": torch.tensor([len(gaps)], dtype=torch.float32),
        "entry_interval_max_steps": torch.tensor([max(gaps, default=0)], dtype=torch.float32),
        "entry_interval_hist": histogram,
    }


def test_entry_interval_metrics_keep_exact_mean_and_max_with_conservative_percentiles():
    metrics = _entry_interval_metrics(
        _entry_interval_output([60, 180]),
        SimpleNamespace(interval_ms=60_000),
        "trailing_martingale",
    )

    assert ENTRY_INTERVAL_METRICS <= set(SUPPORTED_METRICS)
    assert metrics["entry_interval_hours_mean"].item() == pytest.approx(2.0)
    assert metrics["entry_interval_hours_max"].item() == pytest.approx(3.0)
    assert metrics["entry_interval_hours_median"].item() >= 2.0
    assert metrics["entry_interval_hours_p95"].item() >= 2.9
    assert metrics["entry_interval_hours_p99"].item() >= 2.98


def test_entry_interval_metrics_keep_ema_anchor_canonical_zero_without_kernel_output():
    metrics = _entry_interval_metrics(
        {"fill_count": torch.tensor([3.0])},
        SimpleNamespace(interval_ms=60_000),
        "ema_anchor",
    )

    assert set(metrics) == set(ENTRY_INTERVAL_METRICS)
    assert all(value.item() == 0.0 for value in metrics.values())


def test_entry_interval_metrics_fail_closed_for_missing_or_inconsistent_output():
    with pytest.raises(RuntimeError, match="recognized strategy kind"):
        _entry_interval_metrics(
            {"fill_count": torch.tensor([1.0])},
            SimpleNamespace(interval_ms=60_000),
            "",
        )

    with pytest.raises(RuntimeError, match="entry-interval output is missing"):
        _entry_interval_metrics(
            {"fill_count": torch.tensor([1.0])},
            SimpleNamespace(interval_ms=60_000),
            "trailing_martingale",
        )

    output = _entry_interval_output([60])
    output["entry_interval_count"] = torch.tensor([1.25])
    with pytest.raises(RuntimeError, match="fractional counts"):
        _entry_interval_metrics(
            output,
            SimpleNamespace(interval_ms=60_000),
            "trailing_martingale",
        )

    output = _entry_interval_output([60])
    output["entry_interval_count"] = torch.tensor([2.0])
    with pytest.raises(RuntimeError, match="histogram count disagrees"):
        _entry_interval_metrics(
            output,
            SimpleNamespace(interval_ms=60_000),
            "trailing_martingale",
        )


def test_strategy_equity_summary_metric_surface_is_supported():
    strategy_metrics = {
        "gain_strategy_eq",
        "omega_ratio_strategy_eq",
        "expected_shortfall_1pct_strategy_eq",
        "calmar_ratio_strategy_eq",
        "sterling_ratio_strategy_eq",
        "strategy_eq_underwater_pct_median",
    }
    _assert_proxy_surface_partition(strategy_metrics)
    usd_metrics = {
        "adg_usd",
        "calmar_ratio_usd",
        "drawdown_worst_mean_1pct_usd",
        "drawdown_worst_usd",
        "expected_shortfall_1pct_usd",
        "gain_usd",
        "mdg_usd",
        "omega_ratio_usd",
        "sharpe_ratio_usd",
        "sortino_ratio_usd",
        "sterling_ratio_usd",
        "exposure_ratio_usd",
        "exposure_mean_ratio_usd",
    }
    _assert_proxy_surface_partition(usd_metrics)


def test_hard_stop_lifecycle_metric_surface_is_supported():
    lifecycle_metrics = {
        "hard_stop_triggers",
        "hard_stop_triggers_per_year",
        "hard_stop_triggers_long",
        "hard_stop_triggers_short",
        "hard_stop_restarts",
        "hard_stop_restarts_per_year",
        "hard_stop_restarts_per_year_long",
        "hard_stop_restarts_per_year_short",
        "hard_stop_restarts_long",
        "hard_stop_restarts_short",
        "hard_stop_time_in_yellow_pct",
        "hard_stop_time_in_orange_pct",
        "hard_stop_time_in_red_pct",
        "hard_stop_duration_minutes_mean",
        "hard_stop_duration_minutes_max",
        "hard_stop_trigger_drawdown_mean",
        "hard_stop_flatten_time_minutes_mean",
        "hard_stop_post_restart_retrigger_pct",
    }
    _assert_proxy_surface_partition(lifecycle_metrics)
    assert {
        "drawdown_worst_ema_strategy_eq",
        "drawdown_worst_ema_strategy_eq_long",
        "drawdown_worst_ema_strategy_eq_short",
        "drawdown_worst_mean_1pct_ema_strategy_eq",
        "drawdown_worst_mean_1pct_ema_strategy_eq_long",
        "drawdown_worst_mean_1pct_ema_strategy_eq_short",
        "peak_recovery_hours_strategy_eq_long",
        "peak_recovery_hours_strategy_eq_short",
        "peak_recovery_days_strategy_eq_long",
        "peak_recovery_days_strategy_eq_short",
    } <= set(SUPPORTED_METRICS)


def test_hard_stop_ema_drawdown_reduction_matches_rust_side_max_contract():
    from optimization.gpu.metrics import _hard_stop_ema_drawdown_metrics

    metrics = _hard_stop_ema_drawdown_metrics(
        {
            "hsl_drawdown_ema_max_long": torch.tensor([0.12, 0.0]),
            "hsl_drawdown_ema_max_short": torch.tensor([0.08, 0.2]),
        }
    )

    assert metrics["drawdown_worst_ema_strategy_eq"].tolist() == pytest.approx(
        [0.12, 0.2]
    )
    assert metrics["drawdown_worst_ema_strategy_eq_long"].tolist() == pytest.approx(
        [0.12, 0.0]
    )
    assert metrics["drawdown_worst_ema_strategy_eq_short"].tolist() == pytest.approx(
        [0.08, 0.2]
    )


def test_hard_stop_ema_drawdown_metrics_fail_closed_without_directional_outputs():
    from optimization.gpu.metrics import _hard_stop_ema_drawdown_metrics

    with pytest.raises(RuntimeError, match="drawdown-EMA outputs are missing"):
        _hard_stop_ema_drawdown_metrics({})

    with pytest.raises(RuntimeError, match="hsl_drawdown_ema_max_short"):
        _hard_stop_ema_drawdown_metrics(
            {"hsl_drawdown_ema_max_long": torch.tensor([0.1])}
        )


def test_hard_stop_raw_drawdown_metrics_map_each_side():
    from optimization.gpu.metrics import _hard_stop_raw_drawdown_metrics

    metrics = _hard_stop_raw_drawdown_metrics(
        {
            "hsl_drawdown_raw_max_long": torch.tensor([0.17, 0.0]),
            "hsl_drawdown_raw_max_short": torch.tensor([0.08, 0.23]),
            "hsl_drawdown_raw_mean_worst_1pct_long": torch.tensor([0.12, 0.0]),
            "hsl_drawdown_raw_mean_worst_1pct_short": torch.tensor([0.06, 0.19]),
        }
    )

    assert metrics["drawdown_worst_strategy_eq_long"].tolist() == pytest.approx(
        [0.17, 0.0]
    )
    assert metrics["drawdown_worst_strategy_eq_short"].tolist() == pytest.approx(
        [0.08, 0.23]
    )
    assert metrics[
        "drawdown_worst_mean_1pct_strategy_eq_long"
    ].tolist() == pytest.approx([0.12, 0.0])
    assert metrics[
        "drawdown_worst_mean_1pct_strategy_eq_short"
    ].tolist() == pytest.approx([0.06, 0.19])


def test_hard_stop_raw_drawdown_metrics_fail_closed_without_outputs():
    from optimization.gpu.metrics import _hard_stop_raw_drawdown_metrics

    with pytest.raises(RuntimeError, match="raw-drawdown outputs are missing"):
        _hard_stop_raw_drawdown_metrics({})

    with pytest.raises(RuntimeError, match="hsl_drawdown_raw_max_short"):
        _hard_stop_raw_drawdown_metrics(
            {
                "hsl_drawdown_raw_max_long": torch.tensor([0.1]),
                "hsl_drawdown_raw_mean_worst_1pct_long": torch.tensor([0.08]),
                "hsl_drawdown_raw_mean_worst_1pct_short": torch.tensor([0.05]),
            }
        )


def test_hard_stop_ema_tail_reduction_matches_rust_side_max_contract():
    from optimization.gpu.metrics import _hard_stop_ema_tail_metrics

    metrics = _hard_stop_ema_tail_metrics(
        {
            "hsl_drawdown_ema_mean_worst_1pct_long": torch.tensor([0.14, 0.0]),
            "hsl_drawdown_ema_mean_worst_1pct_short": torch.tensor([0.09, 0.21]),
        }
    )

    assert metrics["drawdown_worst_mean_1pct_ema_strategy_eq"].tolist() == (
        pytest.approx([0.14, 0.21])
    )
    assert metrics[
        "drawdown_worst_mean_1pct_ema_strategy_eq_long"
    ].tolist() == pytest.approx([0.14, 0.0])
    assert metrics[
        "drawdown_worst_mean_1pct_ema_strategy_eq_short"
    ].tolist() == pytest.approx([0.09, 0.21])


def test_hard_stop_ema_tail_metrics_fail_closed_without_directional_outputs():
    from optimization.gpu.metrics import _hard_stop_ema_tail_metrics

    with pytest.raises(RuntimeError, match="drawdown-EMA tail outputs are missing"):
        _hard_stop_ema_tail_metrics({})

    with pytest.raises(
        RuntimeError, match="hsl_drawdown_ema_mean_worst_1pct_short"
    ):
        _hard_stop_ema_tail_metrics(
            {
                "hsl_drawdown_ema_mean_worst_1pct_long": torch.tensor([0.1])
            }
        )


def test_hard_stop_strategy_equity_recovery_matches_rust_units():
    from optimization.gpu.metrics import _hard_stop_strategy_eq_recovery_metrics

    metrics = _hard_stop_strategy_eq_recovery_metrics(
        {
            "hsl_strategy_eq_recovery_max_ms_long": torch.tensor(
                [3_600_000.0, 86_400_000.0]
            ),
            "hsl_strategy_eq_recovery_max_ms_short": torch.tensor(
                [7_200_000.0, 172_800_000.0]
            ),
        }
    )

    assert metrics["peak_recovery_hours_strategy_eq_long"].tolist() == pytest.approx(
        [1.0, 24.0]
    )
    assert metrics["peak_recovery_hours_strategy_eq_short"].tolist() == pytest.approx(
        [2.0, 48.0]
    )
    assert metrics["peak_recovery_days_strategy_eq_long"].tolist() == pytest.approx(
        [1.0 / 24.0, 1.0]
    )
    assert metrics["peak_recovery_days_strategy_eq_short"].tolist() == pytest.approx(
        [2.0 / 24.0, 2.0]
    )


def test_hard_stop_strategy_equity_recovery_fails_closed_without_raw_outputs():
    from optimization.gpu.metrics import _hard_stop_strategy_eq_recovery_metrics

    with pytest.raises(RuntimeError, match="strategy-equity recovery outputs are missing"):
        _hard_stop_strategy_eq_recovery_metrics({})

    with pytest.raises(RuntimeError, match="hsl_strategy_eq_recovery_max_ms_short"):
        _hard_stop_strategy_eq_recovery_metrics(
            {
                "hsl_strategy_eq_recovery_max_ms_long": torch.tensor([60_000.0]),
            }
        )


def test_strategy_eq_recovery_distribution_maps_mps_columns():
    expected = torch.tensor(
        [[1.25, 1.0, 3.5, 4.0, 4.5, 5.0, 5.5]], dtype=torch.float32
    )

    metrics = _strategy_eq_recovery_distribution_metrics(
        {"strategy_eq_recovery_distribution": expected}
    )

    assert {name: value.item() for name, value in metrics.items()} == {
        "strategy_eq_recovery_days_mean": 1.25,
        "strategy_eq_recovery_days_median": 1.0,
        "strategy_eq_recovery_days_p95": 3.5,
        "strategy_eq_recovery_days_p99": 4.0,
        "strategy_eq_recovery_days_mean_worst_5pct": 4.5,
        "strategy_eq_recovery_days_mean_worst_1pct": 5.0,
    }
    assert set(metrics) <= set(SUPPORTED_METRICS)


def test_strategy_eq_recovery_distribution_fails_closed_without_mps_output():
    with pytest.raises(RuntimeError, match="recovery-distribution output is missing"):
        _strategy_eq_recovery_distribution_metrics({})


def test_hard_stop_lifecycle_reduction_matches_rust_formulas():
    out = {
        "day_end_eq": torch.zeros((2, 1), dtype=torch.float32),
        "max_dd": torch.zeros(2, dtype=torch.float32),
        "first_eq_ts": torch.tensor([0.0, 0.0]),
        "last_eq_ts": torch.tensor([86_400_000.0, 0.0]),
        "hsl_triggers_long": torch.tensor([2.0, 0.0]),
        "hsl_triggers_short": torch.tensor([1.0, 0.0]),
        "hsl_restarts_long": torch.tensor([1.0, 0.0]),
        "hsl_restarts_short": torch.tensor([1.0, 0.0]),
        "hsl_tier_samples_total": torch.tensor([1441.0, 1.0]),
        "hsl_tier_samples_yellow": torch.tensor([144.0, 0.0]),
        "hsl_tier_samples_orange": torch.tensor([288.0, 0.0]),
        "hsl_tier_samples_red": torch.tensor([720.0, 0.0]),
        "hsl_duration_sum_steps": torch.tensor([45.0, 0.0]),
        "hsl_duration_max_steps": torch.tensor([30.0, 0.0]),
        "hsl_duration_count": torch.tensor([3.0, 0.0]),
        "hsl_trigger_drawdown_sum": torch.tensor([0.9, 0.0]),
        "hsl_trigger_drawdown_count": torch.tensor([3.0, 0.0]),
        "hsl_flatten_time_sum_steps": torch.tensor([9.0, 0.0]),
        "hsl_flatten_time_count": torch.tensor([3.0, 0.0]),
        "hsl_restart_retrigger_count": torch.tensor([1.0, 0.0]),
    }

    metrics = _hard_stop_lifecycle_metrics(
        out, SimpleNamespace(interval_ms=60_000)
    )

    assert metrics["hard_stop_triggers"].tolist() == [3.0, 0.0]
    assert metrics["hard_stop_triggers_per_year"].tolist() == pytest.approx(
        [3.0 * 365.25, 0.0]
    )
    assert metrics["hard_stop_restarts"].tolist() == [2.0, 0.0]
    assert metrics["hard_stop_restarts_per_year_long"].tolist() == pytest.approx(
        [365.25, 0.0]
    )
    assert metrics["hard_stop_restarts_per_year_short"].tolist() == pytest.approx(
        [365.25, 0.0]
    )
    assert metrics["hard_stop_time_in_yellow_pct"][0].item() == pytest.approx(
        144.0 / 1441.0
    )
    assert metrics["hard_stop_time_in_orange_pct"][0].item() == pytest.approx(
        288.0 / 1441.0
    )
    assert metrics["hard_stop_time_in_red_pct"][0].item() == pytest.approx(
        720.0 / 1441.0
    )
    assert metrics["hard_stop_duration_minutes_mean"].tolist() == [15.0, 0.0]
    assert metrics["hard_stop_duration_minutes_max"].tolist() == [30.0, 0.0]
    assert metrics["hard_stop_trigger_drawdown_mean"].tolist() == pytest.approx(
        [0.3, 0.0]
    )
    assert metrics["hard_stop_flatten_time_minutes_mean"].tolist() == [3.0, 0.0]
    assert metrics["hard_stop_post_restart_retrigger_pct"].tolist() == [0.5, 0.0]


def test_hard_stop_lifecycle_metrics_fail_closed_without_directional_outputs():
    with pytest.raises(RuntimeError, match="lifecycle outputs are missing"):
        _hard_stop_lifecycle_metrics(
            {"max_dd": torch.tensor([0.1, 0.2])},
            SimpleNamespace(interval_ms=60_000),
        )


def test_hard_stop_panic_loss_reduction_matches_rust_formulas():
    out = {
        "max_dd": torch.zeros(2, dtype=torch.float32),
        "hsl_halt_to_restart_equity_loss": torch.tensor([25.0, 0.0]),
        "hsl_panic_close_loss_sum": torch.tensor([45.0, 0.0]),
        "hsl_panic_close_loss_max": torch.tensor([30.0, 0.0]),
        "hsl_panic_loss_drawdown_min": torch.tensor([0.01, 0.0]),
        "hsl_panic_loss_drawdown_sum": torch.tensor([0.06, 0.0]),
        "hsl_panic_loss_drawdown_max": torch.tensor([0.05, 0.0]),
        "hsl_panic_loss_drawdown_count": torch.tensor([2.0, 0.0]),
    }

    metrics = _hard_stop_panic_loss_metrics(
        out, SimpleNamespace(starting_balance=1_000.0)
    )

    assert metrics["hard_stop_halt_to_restart_equity_loss_pct"].tolist() == [
        0.025,
        0.0,
    ]
    assert metrics["hard_stop_panic_close_loss_sum"].tolist() == [45.0, 0.0]
    assert metrics["hard_stop_panic_close_loss_max"].tolist() == [30.0, 0.0]
    assert metrics[
        "hard_stop_panic_close_loss_drawdown_pct_min"
    ].tolist() == pytest.approx([0.01, 0.0])
    assert metrics[
        "hard_stop_panic_close_loss_drawdown_pct_mean"
    ].tolist() == pytest.approx([0.03, 0.0])
    assert metrics[
        "hard_stop_panic_close_loss_drawdown_pct_max"
    ].tolist() == pytest.approx([0.05, 0.0])


def test_hard_stop_panic_loss_metrics_fail_closed_without_directional_outputs():
    with pytest.raises(RuntimeError, match="panic-loss outputs are missing"):
        _hard_stop_panic_loss_metrics(
            {"max_dd": torch.tensor([0.1, 0.2])},
            SimpleNamespace(starting_balance=1_000.0),
        )


def test_weighted_strategy_equity_metric_surface_is_supported():
    assert {
        "mdg_strategy_eq_w",
        "sharpe_ratio_strategy_eq_w",
        "sortino_ratio_strategy_eq_w",
        "omega_ratio_strategy_eq_w",
        "calmar_ratio_strategy_eq_w",
        "sterling_ratio_strategy_eq_w",
    } <= set(SUPPORTED_METRICS)
    assert {
        "adg_w_usd",
        "calmar_ratio_w_usd",
        "mdg_w_usd",
        "omega_ratio_w_usd",
        "sharpe_ratio_w_usd",
        "sortino_ratio_w_usd",
        "sterling_ratio_w_usd",
    } <= set(SUPPORTED_METRICS)


def test_pnl_recovery_metrics_fail_closed_without_kernel_output():
    day_end = torch.tensor([[100.0, 101.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.full((1,), float("nan")),
        "last_fill_ts": torch.full((1,), float("nan")),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([60_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([60_000.0]),
        "liq_step": torch.tensor([-1]),
    }

    with pytest.raises(RuntimeError, match="recovery output is missing"):
        compute_objectives(
            out,
            SimpleNamespace(
                requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
            ),
            {"ts0": 0.0, "n": 2},
            needed={"peak_recovery_hours_pnl"},
        )


def test_smoothed_gain_and_adg_match_rust_terminal_contract():
    gain, adg = _smoothed_gain_adg(
        torch.tensor([[100.0, 110.0, 120.0, 130.0]], dtype=torch.float64),
        torch.tensor([[True, True, True, True]]),
    )

    expected_gain = (110.0 + 120.0 + 130.0) / 3.0 / 100.0
    assert gain.item() == pytest.approx(expected_gain)
    assert adg.item() == pytest.approx(expected_gain ** (1.0 / 4.0) - 1.0)


def test_omega_ratio_matches_rust_zero_and_cap_contracts():
    capped = _omega_ratio(
        torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        torch.tensor([[True, True]]),
    )
    flat = _omega_ratio(
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[True, True]]),
    )

    assert capped.item() == 1_000.0
    assert flat.item() == 0.0


def test_expected_shortfall_uses_worst_one_percent_daily_min_return():
    values = torch.tensor([[-0.4, -0.1, 0.2, -0.3]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, True]])

    assert _mean_worst_one_pct_abs(values, mask).item() == pytest.approx(0.4)


def test_weighted_strategy_metrics_use_rust_minute_sliced_subset_average():
    day_end = torch.tensor([[100.0, 90.0, 100.0, 90.0]], dtype=torch.float64)
    day_min = day_end.clone()
    day_dd = torch.tensor([[0.0, 0.1, 0.05, 0.1]], dtype=torch.float64)
    active = torch.ones_like(day_end, dtype=torch.bool)
    last_ts = float((4 * 1440 - 1) * 60_000)
    requested = {
        "mdg_strategy_eq_w",
        "sharpe_ratio_strategy_eq_w",
        "sortino_ratio_strategy_eq_w",
        "omega_ratio_strategy_eq_w",
        "calmar_ratio_strategy_eq_w",
        "sterling_ratio_strategy_eq_w",
    }

    metrics = _weighted_strategy_eq_metrics(
        day_end,
        day_min,
        day_dd,
        active,
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([last_ts], dtype=torch.float64),
        0,
        60_000,
        requested,
    )

    full_returns, full_mask = _pct_change(day_end, active)
    last_two = torch.tensor([[False, False, True, True]])
    tail_returns, tail_mask = _pct_change(day_end, last_two)
    expected_mdg = (
        _masked_median(full_returns, full_mask)
        + 2.0 * _masked_median(tail_returns, tail_mask)
    ) / 10.0
    expected_omega = (
        _omega_ratio(full_returns, full_mask)
        + 2.0 * _omega_ratio(tail_returns, tail_mask)
    ) / 10.0

    assert set(metrics) == requested
    assert metrics["mdg_strategy_eq_w"].item() == pytest.approx(
        expected_mdg.item()
    )
    assert metrics["omega_ratio_strategy_eq_w"].item() == pytest.approx(
        expected_omega.item()
    )
    assert all(torch.isfinite(value).all().item() for value in metrics.values())


def test_new_strategy_equity_metrics_reduce_existing_compact_surface():
    day_end = torch.tensor([[100.0, 110.0, 90.0, 120.0]], dtype=torch.float64)
    day_min = torch.tensor([[100.0, 105.0, 80.0, 100.0]], dtype=torch.float64)
    day_dd = torch.tensor([[0.0, 0.05, 0.30, 0.20]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_min,
        "day_max_dd": day_dd,
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "max_dd": torch.tensor([0.30], dtype=torch.float64),
        "held_max_ms": torch.zeros(1, dtype=torch.float64),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.zeros(1, dtype=torch.float64),
        "last_high_ts": torch.tensor([180_000.0], dtype=torch.float64),
        "first_eq_ts": torch.tensor([0.0], dtype=torch.float64),
        "last_eq_ts": torch.tensor([180_000.0], dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
        "candidate_total_wallet_exposure_limit_long": torch.tensor(
            [1.25], dtype=torch.float64
        ),
        "candidate_total_wallet_exposure_limit_short": torch.tensor(
            [0.5], dtype=torch.float64
        ),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )
    requested = {
        "adg_strategy_eq",
        "drawdown_worst_mean_1pct_strategy_eq",
        "drawdown_worst_strategy_eq",
        "gain_strategy_eq",
        "mdg_strategy_eq",
        "omega_ratio_strategy_eq",
        "sharpe_ratio_strategy_eq",
        "sortino_ratio_strategy_eq",
        "expected_shortfall_1pct_strategy_eq",
        "calmar_ratio_strategy_eq",
        "sterling_ratio_strategy_eq",
        "strategy_eq_underwater_pct_median",
    }
    alias_sources = {
        "adg_usd": "adg_strategy_eq",
        "adg_w_usd": "adg_strategy_eq_w",
        "calmar_ratio_usd": "calmar_ratio_strategy_eq",
        "calmar_ratio_w_usd": "calmar_ratio_strategy_eq_w",
        "drawdown_worst_mean_1pct_usd": "drawdown_worst_mean_1pct_strategy_eq",
        "drawdown_worst_usd": "drawdown_worst_strategy_eq",
        "expected_shortfall_1pct_usd": "expected_shortfall_1pct_strategy_eq",
        "gain_usd": "gain_strategy_eq",
        "mdg_usd": "mdg_strategy_eq",
        "mdg_w_usd": "mdg_strategy_eq_w",
        "omega_ratio_usd": "omega_ratio_strategy_eq",
        "omega_ratio_w_usd": "omega_ratio_strategy_eq_w",
        "sharpe_ratio_usd": "sharpe_ratio_strategy_eq",
        "sharpe_ratio_w_usd": "sharpe_ratio_strategy_eq_w",
        "sortino_ratio_usd": "sortino_ratio_strategy_eq",
        "sortino_ratio_w_usd": "sortino_ratio_strategy_eq_w",
        "sterling_ratio_usd": "sterling_ratio_strategy_eq",
        "sterling_ratio_w_usd": "sterling_ratio_strategy_eq_w",
    }
    requested.update(alias_sources)
    requested.update(alias_sources.values())
    per_exposure_sources = {
        "adg_per_exposure_long_usd": ("adg_strategy_eq", 1.25),
        "adg_per_exposure_short_usd": ("adg_strategy_eq", 0.5),
        "adg_w_per_exposure_long_usd": ("adg_strategy_eq_w", 1.25),
        "adg_w_per_exposure_short_usd": ("adg_strategy_eq_w", 0.5),
        "gain_per_exposure_long_usd": ("gain_strategy_eq", 1.25),
        "gain_per_exposure_short_usd": ("gain_strategy_eq", 0.5),
        "mdg_per_exposure_long_usd": ("mdg_strategy_eq", 1.25),
        "mdg_per_exposure_short_usd": ("mdg_strategy_eq", 0.5),
        "mdg_w_per_exposure_long_usd": ("mdg_strategy_eq_w", 1.25),
        "mdg_w_per_exposure_short_usd": ("mdg_strategy_eq_w", 0.5),
    }
    requested.update(per_exposure_sources)
    _assert_proxy_surface_partition(per_exposure_sources)

    metrics = compute_objectives(out, run, {"ts0": 0.0, "n": 4}, needed=requested)

    gain, adg = _smoothed_gain_adg(day_end, torch.ones_like(day_end, dtype=torch.bool))
    assert set(metrics) == requested
    assert metrics["gain_strategy_eq"].item() == pytest.approx(gain.item())
    assert metrics["calmar_ratio_strategy_eq"].item() == pytest.approx(
        adg.item() / 0.30
    )
    assert metrics["sterling_ratio_strategy_eq"].item() == pytest.approx(
        adg.item() / 0.30
    )
    assert metrics["strategy_eq_underwater_pct_median"].item() == pytest.approx(0.125)
    assert metrics["expected_shortfall_1pct_strategy_eq"].item() == pytest.approx(
        abs((80.0 - 105.0) / 105.0)
    )
    expected_omega = ((110.0 - 100.0) / 100.0 + (120.0 - 90.0) / 90.0) / abs(
        (90.0 - 110.0) / 110.0
    )
    assert metrics["omega_ratio_strategy_eq"].item() == pytest.approx(expected_omega)
    for alias, source in alias_sources.items():
        assert metrics[alias].item() == pytest.approx(metrics[source].item())
    for name, (source, denominator) in per_exposure_sources.items():
        assert metrics[name].item() == pytest.approx(
            metrics[source].item() / denominator
        )

    out["candidate_total_wallet_exposure_limit_short"] = torch.zeros(
        1, dtype=torch.float64
    )
    zero_exposure = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 4},
        needed={"gain_per_exposure_short_usd"},
    )
    assert zero_exposure["gain_per_exposure_short_usd"].item() == 0.0


def test_btc_account_metrics_use_prepared_daily_price_context():
    day_end = torch.tensor([[100.0, 110.0, 90.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones_like(day_end, dtype=torch.bool),
        "fill_count": torch.tensor([3.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([2 * 86_400_000.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([86_400_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([2 * 86_400_000.0]),
        "liq_step": torch.tensor([-1]),
        "total_wallet_exposure_max": torch.tensor([2.0]),
        "total_wallet_exposure_mean": torch.tensor([1.0]),
        "candidate_total_wallet_exposure_limit_long": torch.tensor([1.25]),
        "candidate_total_wallet_exposure_limit_short": torch.tensor([0.5]),
    }
    requested = {
        "adg_btc",
        "gain_btc",
        "gain_per_exposure_long_btc",
        "peak_recovery_days_equity_btc",
    }
    btc_day_end = np.array([10.0, 10.0, 20.0])

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=86_400_000,
        ),
        {
            "ts0": 0.0,
            "n": 3,
            "btc_day_end_price": btc_day_end,
            "btc_prices": btc_day_end,
        },
        needed=requested,
    )

    btc_equity = torch.tensor([[10.0, 11.0, 4.5]], dtype=torch.float64)
    expected_gain, expected_adg = _smoothed_gain_adg(
        btc_equity, torch.ones_like(btc_equity, dtype=torch.bool)
    )
    assert set(metrics) == requested
    _assert_proxy_surface_partition(requested)
    assert metrics["gain_btc"].item() == pytest.approx(expected_gain.item())
    assert metrics["adg_btc"].item() == pytest.approx(expected_adg.item())
    assert metrics["gain_per_exposure_long_btc"].item() == pytest.approx(
        expected_gain.item() / 1.25
    )
    assert metrics["peak_recovery_days_equity_btc"].item() == 1.0


def test_btc_account_metrics_fail_closed_without_price_context():
    day_end = torch.tensor([[100.0, 101.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones_like(day_end, dtype=torch.bool),
        "fill_count": torch.tensor([1.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([86_400_000.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([86_400_000.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([86_400_000.0]),
        "liq_step": torch.tensor([-1]),
    }

    with pytest.raises(RuntimeError, match="BTC account metric context"):
        compute_objectives(
            out,
            SimpleNamespace(
                requested_start_ts_ms=0,
                guard_ts_ms=0,
                interval_ms=86_400_000,
            ),
            {"ts0": 0.0, "n": 2},
            needed={"adg_btc"},
        )


def test_btc_risk_metrics_use_synchronized_intraday_surface():
    day_end_usd = torch.tensor(
        [[1_000.0, 1_100.0, 900.0, 1_200.0]], dtype=torch.float64
    )
    day_end_btc = torch.tensor([[10.0, 8.0, 12.0, 9.0]], dtype=torch.float64)
    day_min_btc = torch.tensor([[9.0, 7.0, 10.0, 6.0]], dtype=torch.float64)
    day_max_dd_btc = torch.tensor(
        [[0.10, 0.30, 0.20, 0.50]], dtype=torch.float64
    )
    out = {
        "day_end_eq": day_end_usd,
        # Deliberately incompatible USD minima prove the reducer does not
        # derive BTC risk from independently sampled USD and BTC surfaces.
        "day_min_eq": torch.tensor(
            [[999.0, 998.0, 997.0, 996.0]], dtype=torch.float64
        ),
        "day_max_dd": torch.zeros_like(day_end_usd),
        "day_volume": torch.zeros_like(day_end_usd),
        "day_has_fill": torch.ones_like(day_end_usd, dtype=torch.bool),
        "btc_day_end_eq": day_end_btc,
        "btc_day_min_eq": day_min_btc,
        "btc_day_max_dd": day_max_dd_btc,
        "fill_count": torch.tensor([4.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([3 * 86_400_000.0]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([0.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([3 * 86_400_000.0]),
        "liq_step": torch.tensor([-1]),
    }
    requested = {
        "calmar_ratio_btc",
        "drawdown_worst_btc",
        "drawdown_worst_mean_1pct_btc",
        "expected_shortfall_1pct_btc",
        "gain_btc",
        "sharpe_ratio_btc",
        "sortino_ratio_btc",
        "sterling_ratio_btc",
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=86_400_000,
        ),
        {
            "ts0": 0.0,
            "n": 4,
            "btc_day_end_price": np.full(4, 100.0),
            "btc_prices": np.full(4, 100.0),
        },
        needed=requested,
    )

    active = torch.ones_like(day_end_btc, dtype=torch.bool)
    _gain, adg = _smoothed_gain_adg(day_end_btc, active)
    min_changes, min_change_mask = _pct_change(day_min_btc, active)
    sharpe, sortino = _sharpe_sortino(min_changes, min_change_mask, adg)
    expected_shortfall = _mean_worst_one_pct_abs(
        min_changes, min_change_mask
    )
    assert set(metrics) == requested
    safe_day_end_btc = day_end_usd / 100.0
    safe_gain, _safe_adg = _smoothed_gain_adg(safe_day_end_btc, active)
    assert metrics["gain_btc"].item() == pytest.approx(safe_gain.item())
    assert metrics["drawdown_worst_btc"].item() == pytest.approx(0.50)
    assert metrics["drawdown_worst_mean_1pct_btc"].item() == pytest.approx(
        0.50
    )
    assert metrics["expected_shortfall_1pct_btc"].item() == pytest.approx(
        expected_shortfall.item()
    )
    assert metrics["sharpe_ratio_btc"].item() == pytest.approx(sharpe.item())
    assert metrics["sortino_ratio_btc"].item() == pytest.approx(sortino.item())
    assert metrics["calmar_ratio_btc"].item() == pytest.approx(
        adg.item() / 0.50
    )
    assert metrics["sterling_ratio_btc"].item() == pytest.approx(
        adg.item() / 0.50
    )
def test_btc_risk_metrics_fail_closed_without_synchronized_surface():
    day_end = torch.tensor([[100.0, 101.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones_like(day_end, dtype=torch.bool),
        "fill_count": torch.tensor([2.0]),
        "max_dd": torch.zeros(1),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([86_400_000.0]),
    }

    with pytest.raises(RuntimeError, match="synchronized BTC-risk output"):
        _btc_account_metrics(
            out,
            SimpleNamespace(interval_ms=86_400_000),
            {"ts0": 0.0, "n": 2},
            {"sharpe_ratio_btc"},
        )


def test_btc_risk_metrics_match_no_fill_defaults():
    day_end = torch.tensor([[10.0, 8.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "btc_day_end_eq": day_end,
        "btc_day_min_eq": torch.tensor([[9.0, 7.0]], dtype=torch.float64),
        "btc_day_max_dd": torch.tensor([[0.1, 0.3]], dtype=torch.float64),
        "fill_count": torch.tensor([0.0]),
        "max_dd": torch.zeros(1),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([86_400_000.0]),
    }
    requested = {
        "calmar_ratio_btc",
        "drawdown_worst_btc",
        "drawdown_worst_mean_1pct_btc",
        "expected_shortfall_1pct_btc",
        "sharpe_ratio_btc",
        "sortino_ratio_btc",
        "sterling_ratio_btc",
    }

    metrics = _btc_account_metrics(
        out,
        SimpleNamespace(interval_ms=86_400_000),
        {
            "ts0": 0.0,
            "n": 2,
            "btc_day_end_price": np.ones(2),
            "btc_prices": np.ones(2),
        },
        requested,
    )

    assert metrics["drawdown_worst_btc"].item() == 1.0
    assert metrics["drawdown_worst_mean_1pct_btc"].item() == 1.0
    for metric in requested - {
        "drawdown_worst_btc",
        "drawdown_worst_mean_1pct_btc",
    }:
        assert metrics[metric].item() == 0.0


def test_btc_account_metrics_use_candidate_liquidation_endpoint_price():
    day_ms = 86_400_000
    minute_ms = 60_000
    btc_prices = np.full(1442, 10.0)
    btc_prices[1440] = 20.0
    btc_prices[1441] = 40.0
    day_end = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.ones_like(day_end, dtype=torch.bool),
        "fill_count": torch.tensor([2.0]),
        "max_dd": torch.zeros(1),
        "held_max_ms": torch.zeros(1),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1),
        "first_fill_ts": torch.tensor([0.0]),
        "last_fill_ts": torch.tensor([float(day_ms)]),
        "recovery_max_ms": torch.zeros(1),
        "last_high_ts": torch.tensor([0.0]),
        "first_eq_ts": torch.tensor([0.0]),
        "last_eq_ts": torch.tensor([float(day_ms)]),
        "liq_step": torch.tensor([1440]),
    }

    metrics = compute_objectives(
        out,
        SimpleNamespace(
            requested_start_ts_ms=0,
            guard_ts_ms=0,
            interval_ms=minute_ms,
        ),
        {
            "ts0": 0,
            "n": len(btc_prices),
            "btc_day_end_price": np.array([10.0, 40.0]),
            "btc_prices": btc_prices,
        },
        needed={"gain_btc"},
    )

    assert metrics["gain_btc"].item() == pytest.approx(0.75)


def test_objectives_include_final_active_calendar_day():
    day_end = torch.tensor([[100.0, 100.0, 121.0]], dtype=torch.float64)
    active = torch.tensor([[True, True, True]])
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end),
        "max_dd": torch.zeros(1, dtype=torch.float64),
        "held_max_ms": torch.zeros(1, dtype=torch.float64),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.zeros(1, dtype=torch.float64),
        "last_high_ts": torch.tensor([120_000.0], dtype=torch.float64),
        "first_eq_ts": torch.tensor([0.0], dtype=torch.float64),
        "last_eq_ts": torch.tensor([120_000.0], dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )

    metrics = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 3},
        needed={"adg_strategy_eq"},
    )

    expected = _smoothed_adg(day_end, active).item()
    assert metrics["adg_strategy_eq"].item() == pytest.approx(expected)
    assert metrics["adg_strategy_eq"].item() > 0.0


def test_completion_uses_rust_exclusive_requested_end():
    day_end = torch.tensor([[100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end),
        "max_dd": torch.zeros(1, dtype=torch.float64),
        "held_max_ms": torch.zeros(1, dtype=torch.float64),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.zeros(1, dtype=torch.float64),
        "last_high_ts": torch.tensor([60_000.0], dtype=torch.float64),
        "first_eq_ts": torch.tensor([0.0], dtype=torch.float64),
        "last_eq_ts": torch.tensor([60_000.0], dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )

    metrics = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 3},
        needed={"backtest_completion_ratio"},
    )

    assert metrics["backtest_completion_ratio"].item() == pytest.approx(2.0 / 3.0)


def test_completion_is_zero_when_no_equity_sample_exists():
    day_end = torch.tensor([[0.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": torch.full_like(day_end, float("inf")),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end, dtype=torch.bool),
        "max_dd": torch.zeros(1, dtype=torch.float64),
        "held_max_ms": torch.zeros(1, dtype=torch.float64),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.zeros(1, dtype=torch.float64),
        "last_high_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "first_eq_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_eq_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )

    metrics = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 3},
        needed={
            "adg_strategy_eq",
            "adg_strategy_eq_w",
            "backtest_completion_ratio",
            "fills_gap_longest_days",
            "fills_gap_mean_hours",
            "fills_gap_median_hours",
            "fills_gap_p95_hours",
            "fills_gap_p99_hours",
            "fills_gap_time_weighted_mean_hours",
        },
    )

    assert metrics["backtest_completion_ratio"].item() == 0.0
    assert metrics["adg_strategy_eq"].item() == 0.0
    assert metrics["adg_strategy_eq_w"].item() == 0.0
    assert metrics["fills_gap_longest_days"].item() == 0.0
    for name in (
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p95_hours",
        "fills_gap_p99_hours",
        "fills_gap_time_weighted_mean_hours",
    ):
        assert metrics[name].item() == 0.0


def test_completion_uses_raw_requested_start_before_available_history():
    day_end = torch.tensor([[100.0]], dtype=torch.float64)
    out = {
        "day_end_eq": day_end,
        "day_min_eq": day_end.clone(),
        "day_max_dd": torch.zeros_like(day_end),
        "day_volume": torch.zeros_like(day_end),
        "day_has_fill": torch.zeros_like(day_end),
        "max_dd": torch.zeros(1, dtype=torch.float64),
        "held_max_ms": torch.zeros(1, dtype=torch.float64),
        "gap_hist": torch.zeros((1, 128), dtype=torch.int32),
        "gap_max_ms": torch.zeros(1, dtype=torch.float64),
        "first_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "last_fill_ts": torch.full((1,), float("nan"), dtype=torch.float64),
        "recovery_max_ms": torch.zeros(1, dtype=torch.float64),
        "last_high_ts": torch.tensor([60_000.0], dtype=torch.float64),
        "first_eq_ts": torch.tensor([0.0], dtype=torch.float64),
        "last_eq_ts": torch.tensor([60_000.0], dtype=torch.float64),
        "liq_step": torch.tensor([-1.0], dtype=torch.float64),
    }
    run = SimpleNamespace(
        requested_start_ts_ms=-86_400_000,
        guard_ts_ms=0,
        interval_ms=60_000,
    )

    metrics = compute_objectives(
        out,
        run,
        {"ts0": 0.0, "n": 3},
        needed={"backtest_completion_ratio"},
    )

    assert metrics["backtest_completion_ratio"].item() == pytest.approx(1442.0 / 1443.0)
