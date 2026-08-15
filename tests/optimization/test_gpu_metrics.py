from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.metrics import (
    SUPPORTED_METRICS,
    _masked_median,
    _sharpe_sortino,
    _smoothed_adg,
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
