import math
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.metrics import (
    SUPPORTED_METRICS,
    _GAP_HIST_UPPER_STEPS,
    _fill_gap_metrics,
    _hard_stop_lifecycle_metrics,
    _hard_stop_panic_loss_metrics,
    _masked_median,
    _mean_worst_one_pct_abs,
    _omega_ratio,
    _pct_change,
    _sharpe_sortino,
    _smoothed_adg,
    _smoothed_gain_adg,
    _weighted_adg,
    _weighted_strategy_eq_metrics,
    compute_objectives,
)


def test_zero_variance_sharpe_and_sortino_match_rust_zero_contract():
    changes = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[True, True]])
    adg = torch.tensor([0.1], dtype=torch.float64)

    sharpe, sortino = _sharpe_sortino(changes, mask, adg)

    assert sharpe.item() == 0.0
    assert sortino.item() == 0.0


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
    assert {
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p95_hours",
        "fills_gap_p99_hours",
    } <= set(SUPPORTED_METRICS)


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


def test_strategy_equity_summary_metric_surface_is_supported():
    assert {
        "gain_strategy_eq",
        "omega_ratio_strategy_eq",
        "expected_shortfall_1pct_strategy_eq",
        "calmar_ratio_strategy_eq",
        "sterling_ratio_strategy_eq",
        "strategy_eq_underwater_pct_median",
    } <= set(SUPPORTED_METRICS)


def test_hard_stop_lifecycle_metric_surface_is_supported():
    assert {
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
    } <= set(SUPPORTED_METRICS)
    assert "drawdown_worst_ema_strategy_eq" not in SUPPORTED_METRICS


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


def test_hard_stop_lifecycle_metrics_are_zero_without_directional_hsl_outputs():
    metrics = _hard_stop_lifecycle_metrics(
        {"max_dd": torch.tensor([0.1, 0.2])},
        SimpleNamespace(interval_ms=60_000),
    )

    assert set(metrics) <= set(SUPPORTED_METRICS)
    assert len(metrics) == 18
    assert all(metric.tolist() == [0.0, 0.0] for metric in metrics.values())


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


def test_hard_stop_panic_loss_metrics_are_zero_without_directional_outputs():
    metrics = _hard_stop_panic_loss_metrics(
        {"max_dd": torch.tensor([0.1, 0.2])},
        SimpleNamespace(starting_balance=1_000.0),
    )

    assert len(metrics) == 6
    assert all(metric.tolist() == [0.0, 0.0] for metric in metrics.values())


def test_weighted_strategy_equity_metric_surface_is_supported():
    assert {
        "mdg_strategy_eq_w",
        "sharpe_ratio_strategy_eq_w",
        "sortino_ratio_strategy_eq_w",
        "omega_ratio_strategy_eq_w",
        "calmar_ratio_strategy_eq_w",
        "sterling_ratio_strategy_eq_w",
    } <= set(SUPPORTED_METRICS)


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
    }
    run = SimpleNamespace(
        requested_start_ts_ms=0, guard_ts_ms=0, interval_ms=60_000
    )
    requested = {
        "gain_strategy_eq",
        "omega_ratio_strategy_eq",
        "expected_shortfall_1pct_strategy_eq",
        "calmar_ratio_strategy_eq",
        "sterling_ratio_strategy_eq",
        "strategy_eq_underwater_pct_median",
    }

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
