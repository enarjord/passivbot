from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.metrics import (
    SUPPORTED_METRICS,
    _masked_median,
    _mean_worst_one_pct_abs,
    _omega_ratio,
    _sharpe_sortino,
    _smoothed_adg,
    _smoothed_gain_adg,
    _weighted_adg,
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


def test_interpolated_fill_gap_percentile_fails_closed_for_gpu_proxy():
    assert "fills_gap_p95_hours" not in SUPPORTED_METRICS


def test_strategy_equity_summary_metric_surface_is_supported():
    assert {
        "gain_strategy_eq",
        "omega_ratio_strategy_eq",
        "expected_shortfall_1pct_strategy_eq",
        "calmar_ratio_strategy_eq",
        "sterling_ratio_strategy_eq",
        "strategy_eq_underwater_pct_median",
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
        },
    )

    assert metrics["backtest_completion_ratio"].item() == 0.0
    assert metrics["adg_strategy_eq"].item() == 0.0
    assert metrics["adg_strategy_eq_w"].item() == 0.0
    assert metrics["fills_gap_longest_days"].item() == 0.0


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
