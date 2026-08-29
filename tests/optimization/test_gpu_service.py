import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from config.shared_bot import flatten_shared_bot_side
from config.schema import get_template_config
from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_PARAM_KEYS,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_PARAM_KEYS,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
    single_coin_shader_topology,
    validate_hsl_signal_topology,
    validate_single_coin_hsl_signal_topology,
)
from optimization.gpu.service import (
    CORE_OUTPUT_KEYS,
    DIRECTIONAL_HSL_OUTPUT_KEYS,
    MPS_MAX_DISPATCH_CANDIDATE_BARS,
    MpsEmaAnchorProxy,
    MpsSingleCoinProxy,
    MpsMulticoinEmaProxy,
    _build_multicoin_ema_coin_overrides,
    _build_multicoin_tm_coin_overrides,
    _btc_daily_price_context,
    _candidate_wallet_exposure_limit_outputs,
    _candidate_position_slot_outputs,
    _combine_hedged_multicoin_hsl_outputs,
    _combine_hedged_multicoin_outputs,
    _directional_coin_hsl_lookback_bars,
    _directional_entry_initial_metrics,
    _directional_gross_pnl_outputs,
    _gpu_proxy_execution_checkpoint_contract,
    _gpu_profile_unattributed_seconds,
    _add_gpu_terminal_profile,
    _hsl_params,
    _hsl_diagnostics_needed,
    _mps_dispatch_batch_size,
    _mps_strategy_eq_recovery_distribution,
    _new_gpu_dispatch_progress,
    _new_gpu_proxy_profile,
    _update_gpu_dispatch_progress,
    _add_gpu_runner_profile,
    _multicoin_exposure_eligible_coins,
    _position_exposure_enforcer_params,
    _prepared_single_coin_side_enabled,
    _require_exact_safe_proxy_candles,
    _require_multicoin_metric_topology,
    _require_no_internal_invalid_hsl_candles,
    _require_no_internal_invalid_multicoin_hsl_candles,
    _require_no_unsafe_single_coin_candles,
    _require_supported_multicoin_valid_tails,
    _refresh_hedged_multicoin_hsl_at_portfolio_cutoff,
    _single_coin_candle_interval_minutes,
    _single_coin_exposure_params,
    _total_exposure_enforcer_params,
    _unstuck_params,
)


@pytest.mark.parametrize("proxy_cls", [MpsSingleCoinProxy, MpsMulticoinEmaProxy])
def test_proxy_constructors_reject_exact_only_metrics_before_setup(
    monkeypatch, proxy_cls
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    with pytest.raises(ValueError, match="exact Rust backtests and analysis"):
        proxy_cls(
            config={},
            hlcvs=None,
            mss={},
            btc=None,
            timestamps=None,
            exchange="bybit",
            batch_size=1,
            needed_metrics={"fills_count"},
        )


@pytest.mark.parametrize("raw_interval", [5, 5.0, "5"])
def test_single_coin_candle_interval_accepts_positive_integers(raw_interval):
    assert (
        _single_coin_candle_interval_minutes(
            {"candle_interval_minutes": raw_interval}
        )
        == 5
    )


@pytest.mark.parametrize("raw_interval", [0, -1, 1.5, float("nan"), None, "bad"])
def test_single_coin_candle_interval_rejects_invalid_values(raw_interval):
    with pytest.raises(ValueError, match="integer >= 1"):
        _single_coin_candle_interval_minutes(
            {"candle_interval_minutes": raw_interval}
        )


def test_gpu_proxy_execution_checkpoint_contract_tracks_effective_inputs():
    backtest_params = {
        "coins": ["BTC"],
        "starting_balance": 10_000.0,
        "candle_interval_minutes": 1,
        "requested_start_timestamp_ms": 1_000,
        "first_timestamp_ms": 1_000,
        "first_valid_indices": [0],
        "last_valid_indices": [2],
        "trade_start_indices": [1],
        "global_warmup_bars": 1,
        "liquidation_threshold": 0.05,
        "filter_by_min_effective_cost": False,
        "dynamic_wel_by_tradability": True,
        "hedge_mode": False,
        "max_realized_loss_pct": 1.0,
        "pnls_max_lookback_days": 30.0,
        "market_order_slippage_pct": 0.0005,
        "market_orders_allowed": False,
        "market_order_near_touch_threshold": 0.001,
        "forager_score_hysteresis_pct": 0.02,
    }
    market = {
        "qty_step": 0.001,
        "price_step": 0.1,
        "min_qty": 0.001,
        "min_cost": 5.0,
        "c_mult": 1.0,
        "maker_fee": 0.0004,
        "taker_fee": 0.00055,
    }
    original = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    with_hsl_ring = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        directional_hsl_rolling_capacity=8_192,
    )
    changed_hsl_ring = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        directional_hsl_rolling_capacity=4_096,
    )
    changed_fee = dict(market, maker_fee=0.0005)
    changed = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[changed_fee],
        base_params={"long": {"offset": 0.01}},
    )
    changed_timestamps = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_100, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    changed_hlcvs_values = np.arange(12, dtype=np.float64).reshape(3, 1, 4)
    changed_hlcvs_values[1, 0, 2] += 0.5
    changed_hlcvs = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=changed_hlcvs_values,
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
    )
    changed_base_params = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.02}},
    )
    btc_prices = np.array([50_000.0, 51_000.0, 49_000.0])
    with_btc = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        btc_prices=btc_prices,
    )
    changed_btc = _gpu_proxy_execution_checkpoint_contract(
        strategy_kind="ema_anchor",
        exchange="bybit",
        enabled_sides=["long"],
        hlcvs=np.arange(12, dtype=np.float64).reshape(3, 1, 4),
        timestamps=np.array([1_000, 2_000, 3_000], dtype=np.int64),
        backtest_params=backtest_params,
        exchange_params=[market],
        base_params={"long": {"offset": 0.01}},
        btc_prices=btc_prices + np.array([0.0, 1.0, 0.0]),
    )

    assert changed != original
    assert changed_timestamps != original
    assert changed_hlcvs != original
    assert changed_base_params != original
    assert "directional_hsl_rolling_capacity" not in original
    assert with_hsl_ring["directional_hsl_rolling_capacity"] == 8_192
    assert changed_hsl_ring != with_hsl_ring
    assert "btc_analysis" not in original
    assert changed_btc != with_btc
    assert original["timestamps"]["count"] == 3


def test_btc_daily_price_context_matches_proxy_utc_day_grid():
    day_ms = 86_400_000
    context = _btc_daily_price_context(
        [10.0, 12.0, 11.0, 20.0],
        [0, day_ms - 1, day_ms, 2 * day_ms - 1],
        expected_count=4,
        expected_days=2,
    )

    assert context["btc_day_end_price"].tolist() == [12.0, 20.0]


@pytest.mark.parametrize(
    "btc_prices",
    ([10.0, float("nan")], [10.0, 0.0], [10.0]),
)
def test_btc_daily_price_context_rejects_unsafe_series(btc_prices):
    with pytest.raises(ValueError, match="MPS BTC analysis"):
        _btc_daily_price_context(
            btc_prices,
            [0, 60_000],
            expected_count=2,
            expected_days=1,
        )


def test_btc_daily_price_context_rejects_skipped_utc_days():
    day_ms = 86_400_000
    with pytest.raises(ValueError, match="missing prepared prices"):
        _btc_daily_price_context(
            [10.0, 20.0],
            [0, 2 * day_ms],
            expected_count=2,
            expected_days=3,
        )


def test_gpu_proxy_execution_checkpoint_contract_rejects_timestamp_shape_mismatch():
    with pytest.raises(ValueError, match="timestamp identity disagrees"):
        _gpu_proxy_execution_checkpoint_contract(
            strategy_kind="ema_anchor",
            exchange="bybit",
            enabled_sides=["long"],
            hlcvs=np.zeros((3, 1, 4), dtype=np.float64),
            timestamps=np.array([1_000, 2_000], dtype=np.int64),
            backtest_params={"coins": ["BTC"]},
            exchange_params=[],
            base_params={},
        )


@pytest.mark.parametrize(
    ("lookback_days", "signal_mode", "enabled", "expected"),
    [
        (30.0, "coin", True, 43_200),
        (0.0, "coin", True, 1),
        (-1.0, "coin", True, 0),
        (30.0, "pside", True, 0),
        (30.0, "coin", False, 0),
    ],
)
def test_directional_coin_hsl_lookback_bar_contract(
    lookback_days, signal_mode, enabled, expected
):
    assert (
        _directional_coin_hsl_lookback_bars(
            {
                "pnls_max_lookback_days": lookback_days,
                "candle_interval_minutes": 1,
            },
            signal_mode=signal_mode,
            hsl_enabled=enabled,
        )
        == expected
    )


def test_recovery_distribution_postprocessor_is_opt_in_and_fail_closed(monkeypatch):
    needed = {"strategy_eq_recovery_days_p99"}

    assert _mps_strategy_eq_recovery_distribution({}, {"adg_strategy_eq"}) is None
    with pytest.raises(RuntimeError, match="recovery sampling output is missing"):
        _mps_strategy_eq_recovery_distribution({}, needed)

    samples = object()
    expected = SimpleNamespace()
    expected.cpu = lambda: (_ for _ in ()).throw(
        AssertionError("recovery distribution left MPS during reduction")
    )
    called = {}

    def fake_postprocessor(values, *, sample_interval_days):
        called["values"] = values
        called["sample_interval_days"] = sample_interval_days
        return expected

    fake_mps_kernel = ModuleType("optimization.gpu.mps_kernel")
    fake_mps_kernel.strategy_eq_recovery_distribution_from_samples = (
        fake_postprocessor
    )
    monkeypatch.setitem(
        sys.modules, "optimization.gpu.mps_kernel", fake_mps_kernel
    )
    actual = _mps_strategy_eq_recovery_distribution(
        {
            "strategy_eq_recovery_samples": samples,
            "strategy_eq_recovery_sample_interval_days": 1.0 / 24.0,
        },
        needed,
    )

    assert called == {
        "values": samples,
        "sample_interval_days": 1.0 / 24.0,
    }
    assert actual is expected


@pytest.mark.parametrize(
    (
        "long_enabled",
        "short_enabled",
        "hsl_enabled",
        "hsl_one_side_enabled",
        "expected",
    ),
    [
        (True, False, False, False, "long_no_hsl"),
        (False, True, False, False, "short_no_hsl"),
        (True, True, False, False, "generic"),
        (True, False, True, False, "generic"),
        (False, True, True, False, "generic"),
        (True, False, True, True, "long_hsl"),
        (False, True, True, True, "short_hsl"),
        (True, True, True, True, "generic"),
        (False, False, True, True, "generic"),
    ],
)
def test_single_coin_shader_topology_is_fail_closed(
    long_enabled,
    short_enabled,
    hsl_enabled,
    hsl_one_side_enabled,
    expected,
):
    assert (
        single_coin_shader_topology(
            long_enabled=long_enabled,
            short_enabled=short_enabled,
            hsl_enabled=hsl_enabled,
            hsl_one_side_enabled=hsl_one_side_enabled,
        )
        == expected
    )


def test_mps_dispatch_batch_size_bounds_single_and_multicoin_work():
    n_bars = 1_923_175

    assert _mps_dispatch_batch_size(8192, n_bars=n_bars) == 519
    assert _mps_dispatch_batch_size(8192, n_bars=n_bars, n_coins=4) == 129
    assert (
        _mps_dispatch_batch_size(8192, n_bars=n_bars, n_coins=4, n_sides=2)
        == 64
    )
    assert (
        _mps_dispatch_batch_size(
            8192,
            n_bars=n_bars,
            max_candidate_bars=500_000_000,
        )
        == 259
    )
    assert _mps_dispatch_batch_size(32, n_bars=1000, n_coins=4) == 32
    with pytest.raises(ValueError, match="one GPU candidate exceeds"):
        _mps_dispatch_batch_size(8192, n_bars=3 * 10**9)
    with pytest.raises(ValueError, match="max_candidate_bars"):
        _mps_dispatch_batch_size(32, n_bars=1000, max_candidate_bars=0)
    assert MPS_MAX_DISPATCH_CANDIDATE_BARS == 1_000_000_000


def _minimal_single_coin_proxy(*, interrupt_check=lambda: None):
    torch = pytest.importorskip("torch")
    proxy = MpsSingleCoinProxy.__new__(MpsSingleCoinProxy)
    proxy.batch_size = 8
    proxy.dispatch_batch_size = 2
    proxy.max_dispatch_candidate_bars = MPS_MAX_DISPATCH_CANDIDATE_BARS
    proxy.interrupt_check = interrupt_check
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.needed_metrics = {"score"}
    proxy._parameter_matrix = lambda candidates: np.asarray(
        [[candidate["value"]] for candidate in candidates], dtype=np.float64
    )
    calls = []

    def run(parameters, **kwargs):
        calls.append(parameters[:, 0].tolist())
        batch = len(parameters)
        output = {
            key: torch.zeros(batch)
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            )
        }
        output["max_dd"] = torch.as_tensor(parameters[:, 0])
        output["alive"] = torch.ones(batch, dtype=torch.bool)
        return output

    proxy.runner = SimpleNamespace(run=run)
    proxy._compute_objectives = lambda output, *args, **kwargs: {
        "score": output["max_dd"]
    }
    return proxy, calls


def test_single_coin_proxy_preserves_order_across_bounded_dispatches():
    proxy, calls = _minimal_single_coin_proxy()
    candidates = [{"value": float(index)} for index in range(5)]

    assert proxy.evaluate(candidates) == [
        {"score": float(index)} for index in range(5)
    ]
    assert calls == [[0.0, 1.0], [2.0, 3.0], [4.0]]


def test_single_coin_proxy_recent_history_expands_safe_dispatch_and_routes_window():
    proxy, calls = _minimal_single_coin_proxy()
    proxy.batch_size = 4096
    proxy.dispatch_batch_size = 258
    proxy.strategy_kind = "trailing_martingale"
    proxy.runner.n = 1_931_815
    proxy.run = ProxyRun(
        1_000.0,
        100,
        10,
        0,
        0,
        0,
        60_000,
        0.05,
        0,
        1_931_814,
    )
    proxy.history_warmup_bars = 100
    original_run = proxy.runner.run
    routed_windows = []
    metric_runs = []

    def routed_run(parameters, **kwargs):
        routed_windows.append(dict(kwargs))
        return original_run(parameters, **kwargs)

    proxy.runner.run = routed_run
    original_compute = proxy._compute_objectives

    def capture_compute(output, run, *args, **kwargs):
        metric_runs.append(run)
        return original_compute(output, run, *args, **kwargs)

    proxy._compute_objectives = capture_compute

    history_start, trade_start = proxy.recent_window_for_history_fraction(0.25)
    results = proxy.evaluate(
        [{"value": float(index)} for index in range(1024)],
        history_start_step=history_start,
        trade_start_step=trade_start,
    )

    assert (history_start, trade_start) == (1_448_762, 1_448_863)
    assert len(results) == 1024
    assert [len(call) for call in calls] == [1024]
    assert routed_windows == [
        {
            "profile": False,
            "end_step": 1_931_815,
            "history_start_step": 1_448_762,
            "trade_start_step": 1_448_863,
        }
    ]
    assert len(metric_runs) == 1
    assert metric_runs[0].trade_start_idx == 1_448_863
    assert metric_runs[0].requested_start_ts_ms == 1_448_863 * 60_000

    proxy.runner.n = 1_250
    proxy.run = ProxyRun(
        1_000.0,
        100,
        1_000,
        0,
        0,
        0,
        60_000,
        0.05,
        900,
        1_249,
    )
    assert proxy.recent_window_for_history_fraction(1.0) == (900, 1_000)
    assert proxy.recent_window_for_history_fraction(0.25) == (1_086, 1_187)


def test_single_coin_proxy_profile_records_dispatch_shape_and_timings(monkeypatch):
    torch = pytest.importorskip("torch")
    proxy, calls = _minimal_single_coin_proxy()
    proxy.profile_enabled = True
    proxy.strategy_kind = "ema_anchor"
    proxy.runner.n = 100
    proxy.runner.long_enabled = True
    proxy.runner.short_enabled = True
    proxy.run = SimpleNamespace(interval_ms=60_000)
    original_run = proxy.runner.run

    def profiled_run(parameters, **kwargs):
        output = original_run(parameters, **kwargs)
        candidate_ids = parameters[:, 0].astype(int)
        output["alive"] = torch.as_tensor(candidate_ids % 2 == 0)
        output["last_eq_ts"] = torch.as_tensor(
            np.where(candidate_ids % 2 == 0, 99, candidate_ids + 9) * 60_000.0
        )
        proxy.runner.last_profile = {
            "cpu_pack_seconds": 0.001,
            "upload_and_zero_seconds": 0.002,
            "compile_seconds": 0.003,
            "pre_dispatch_sync_seconds": 0.004,
            "kernel_seconds": 0.005,
            "metric_decode_seconds": 0.006,
            "batch_size": len(parameters),
            "dispatch_count": 1,
            "cold": len(calls) == 1,
            "dispatch_specialization": {
                "trailing_entry_only": True,
                "trailing_close_only": True,
                "reducers_disabled": True,
                "market_orders_disabled": True,
                "loss_gate_disabled": True,
                "volatility_disabled": True,
            },
        }
        return output

    proxy.runner.run = profiled_run
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)

    proxy.evaluate([{"value": float(index)} for index in range(5)])

    profile = proxy.last_profile
    assert profile["candidate_count"] == 5
    assert profile["max_dispatch_candidate_bars"] == 1_000_000_000
    assert profile["dispatch_batch_size"] == 2
    assert profile["dispatch_chunk_count"] == 3
    assert profile["actual_dispatch_batch_sizes"] == [2, 2, 1]
    assert profile["dispatch_specializations"] == [
        {
            "trailing_entry_only": True,
            "trailing_close_only": True,
            "reducers_disabled": True,
            "market_orders_disabled": True,
            "loss_gate_disabled": True,
            "volatility_disabled": True,
        }
    ] * 3
    assert profile["dispatch_count"] == 3
    assert profile["cold_dispatch_count"] == 1
    assert profile["warm_dispatch_count"] == 2
    assert len(profile["dispatch_chunk_wall_seconds"]) == 3
    assert all(value >= 0.0 for value in profile["dispatch_chunk_wall_seconds"])
    assert profile["candidate_bars"] == 1_000
    assert profile["kernel_candidate_bars"] == 1_000
    assert profile["terminal_candidate_count"] == 2
    assert profile["terminal_without_equity_count"] == 0
    assert profile["estimated_post_terminal_candidate_bars"] == 348
    assert profile["estimated_post_terminal_candidate_bar_fraction"] == pytest.approx(
        0.348
    )
    assert profile["terminal_step_fraction_p50"] == pytest.approx(11.0 / 98.0)
    assert profile["terminal_step_fraction_p90"] == pytest.approx(11.8 / 98.0)
    assert profile["coin_count"] == 1
    assert profile["side_count"] == 2
    assert profile["timings_seconds"]["kernel_execution"] == pytest.approx(
        0.015
    )
    assert profile["timings_seconds"]["cold_compilation"] == pytest.approx(
        0.003
    )
    assert profile["timings_seconds"]["warm_library_lookup"] == pytest.approx(
        0.006
    )
    assert profile["wall_seconds"] >= 0.0


def test_gpu_terminal_profile_rebases_recent_window_steps():
    torch = pytest.importorskip("torch")
    profile = {
        "terminal_candidate_count": 0,
        "terminal_without_equity_count": 0,
        "estimated_post_terminal_candidate_bars": 0,
        "_terminal_step_fractions": [],
        "side_count": 1,
    }

    _add_gpu_terminal_profile(
        profile,
        {
            "alive": torch.as_tensor([False]),
            "last_eq_ts": torch.as_tensor([75 * 60_000.0]),
        },
        interval_ms=60_000,
        effective_start_step=50,
        effective_end_step=100,
    )

    assert profile["terminal_candidate_count"] == 1
    assert profile["estimated_post_terminal_candidate_bars"] == 23
    assert profile["_terminal_step_fractions"] == [pytest.approx(25.0 / 48.0)]


def test_gpu_dispatch_progress_is_rate_limited_and_reports_eta(
    monkeypatch, caplog
):
    readings = iter((100.0, 120.0, 131.0, 162.0))
    monkeypatch.setattr(
        "optimization.gpu.service.time.monotonic", lambda: next(readings)
    )
    progress = _new_gpu_dispatch_progress(8, 2)

    _update_gpu_dispatch_progress(
        progress, completed_candidates=2, strategy="trailing_martingale"
    )
    assert not caplog.records

    with caplog.at_level("INFO"):
        _update_gpu_dispatch_progress(
            progress, completed_candidates=4, strategy="trailing_martingale"
        )
        _update_gpu_dispatch_progress(
            progress, completed_candidates=8, strategy="trailing_martingale"
        )

    assert "chunks=2/4" in caplog.records[0].message
    assert "candidates=4/8" in caplog.records[0].message
    assert "eta=" in caplog.records[0].message
    assert "chunks=4/4" in caplog.records[1].message


def test_single_coin_proxy_profile_is_empty_when_disabled(monkeypatch):
    proxy, _calls = _minimal_single_coin_proxy()
    monkeypatch.setattr(
        "optimization.gpu.service.time.perf_counter",
        lambda: (_ for _ in ()).throw(
            AssertionError("disabled proxy profiling read the clock")
        ),
    )

    proxy.evaluate([{"value": 1.0}])

    assert proxy.last_profile == {}


def test_gpu_profile_candidate_bars_include_coin_and_side_topology():
    proxy = SimpleNamespace(
        batch_size=4,
        dispatch_batch_size=4,
        strategy_kind="ema_anchor",
    )
    runner = SimpleNamespace(
        n=100,
        n_coins=8,
        last_profile={
            "batch_size": 1,
            "dispatch_count": 1,
            "cold": False,
        },
    )
    profile = _new_gpu_proxy_profile(
        proxy, [{}], (runner,), coin_count=8, side_count=2
    )

    _add_gpu_runner_profile(profile, runner, side_count=2)

    assert profile["candidate_bars"] == 1_600
    assert profile["kernel_candidate_bars"] == 1_600


def test_gpu_profile_candidate_bars_use_truncated_effective_steps():
    proxy = SimpleNamespace(
        batch_size=4,
        dispatch_batch_size=4,
        strategy_kind="ema_anchor",
    )
    runner = SimpleNamespace(
        n=100,
        n_coins=8,
        last_profile={
            "batch_size": 2,
            "dispatch_count": 1,
            "cold": False,
        },
    )
    profile = _new_gpu_proxy_profile(
        proxy, [{}, {}], (runner,), coin_count=8, side_count=2
    )

    _add_gpu_runner_profile(
        profile,
        runner,
        side_count=2,
        effective_candidate_steps=np.asarray([10, 20]),
    )

    assert profile["candidate_bars"] == 3_200
    assert profile["kernel_candidate_bars"] == 480


def test_gpu_profile_unattributed_seconds_excludes_nested_runner_and_transfer():
    timings = {
        "device_to_host": 3.0,
        "candidate_packing": 1.0,
        "upload_and_buffer_clear": 1.0,
        "cold_compilation": 1.0,
        "warm_library_lookup": 0.0,
        "kernel_execution": 2.0,
        "metric_reduction": 2.0,
    }

    assert _gpu_profile_unattributed_seconds(
        timings,
        12.0,
        device_to_host_before=1.0,
        runner_seconds_before=2.0,
    ) == pytest.approx(5.0)


def test_single_coin_proxy_honors_interrupt_between_mps_dispatches():
    checks = 0

    def interrupt_check():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt

    proxy, calls = _minimal_single_coin_proxy(interrupt_check=interrupt_check)

    with pytest.raises(KeyboardInterrupt):
        proxy.evaluate([{"value": float(index)} for index in range(5)])

    assert calls == [[0.0, 1.0]]


def test_hsl_params_preserve_grouped_tier_ratios_after_flattening():
    bot = flatten_shared_bot_side(
        {
            "hsl": {
                "tier_ratios": {"yellow": 0.31, "orange": 0.82},
            }
        }
    )

    packed = _hsl_params(bot, signal_mode="coin")

    assert packed["hsl_tier_ratio_yellow"] == pytest.approx(0.31)
    assert packed["hsl_tier_ratio_orange"] == pytest.approx(0.82)


def test_core_output_contract_retains_gross_pnl_aggregates():
    assert {
        "profit_sum",
        "loss_sum",
        "position_unchanged_max_ms",
        "entry_initial_balance_pct",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "day_fill_count",
        "fill_count",
        "fill_count_entry",
        "fill_count_long",
        "fills_active_days_count",
        "coin_fill_counts",
        "pnl_recovery_max_ms",
        "account_recovery_max_ms",
    } <= CORE_OUTPUT_KEYS


@pytest.mark.parametrize(
    "metric",
    [
        "adg_pnl",
        "adg_pnl_w",
        "fills_analysis_duration_days",
        "fills_active_days_count",
        "fills_active_days_ratio",
        "fills_active_symbols_count",
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
        "long_short_profit_ratio",
        "loss_profit_ratio_long",
        "loss_profit_ratio_short",
        "mdg_pnl",
        "mdg_pnl_w",
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
        "peak_recovery_days_pnl",
        "peak_recovery_hours_pnl",
        "pnl_ratio_long_short",
        "sharpe_ratio_pnl",
        "sharpe_ratio_pnl_w",
        "sortino_ratio_pnl",
        "sortino_ratio_pnl_w",
        "volume_pct_per_day_avg_w",
    ],
)
def test_dual_side_multicoin_intraday_cutoff_metrics_fail_closed(metric):
    with pytest.raises(ValueError, match="shared-liquidation cutoff"):
        _require_multicoin_metric_topology(["long", "short"], {metric})

    _require_multicoin_metric_topology(["long"], {metric})
    _require_multicoin_metric_topology(["long", "short"], {"adg_strategy_eq"})
    _require_multicoin_metric_topology(
        ["long", "short"],
        {metric},
        shared_account_controller=True,
    )


def test_multicoin_exposure_eligibility_unions_fused_sides():
    long = np.zeros((3, 2), dtype=np.float32)
    short = np.zeros((3, 2), dtype=np.float32)
    short[1, 1] = 0.5
    short[2, 1] = np.nan

    eligible = _multicoin_exposure_eligible_coins(
        {"long": long, "short": short},
        ["long", "short"],
        1,
    )

    assert eligible.tolist() == [False, True, True]


@pytest.mark.parametrize("side", ["long", "short"])
def test_directional_entry_initial_metrics_preserve_candidate_batch_shape(side):
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.1, 0.2, 0.3])

    metrics = _directional_entry_initial_metrics(side, values)

    assert metrics[f"entry_initial_balance_pct_{side}"].tolist() == values.tolist()
    other = "short" if side == "long" else "long"
    assert metrics[f"entry_initial_balance_pct_{other}"].shape == values.shape
    assert metrics[f"entry_initial_balance_pct_{other}"].tolist() == [0.0, 0.0, 0.0]


def test_candidate_wallet_exposure_limits_preserve_sides_and_base_fallback():
    torch = pytest.importorskip("torch")

    outputs = _candidate_wallet_exposure_limit_outputs(
        [
            {"long_total_wallet_exposure_limit": 1.25},
            {
                "long_total_wallet_exposure_limit": 1.5,
                "short_total_wallet_exposure_limit": 0.75,
            },
        ],
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["candidate_total_wallet_exposure_limit_long"].tolist() == [
        1.25,
        1.5,
    ]
    assert outputs["candidate_total_wallet_exposure_limit_short"].tolist() == [
        0.5,
        0.75,
    ]


@pytest.mark.parametrize("side", ["long", "short"])
def test_directional_gross_pnl_outputs_zero_the_inactive_side(side):
    torch = pytest.importorskip("torch")
    profit = torch.tensor([7.0])
    loss = torch.tensor([2.0])

    outputs = _directional_gross_pnl_outputs(side, profit, loss)
    other_side = "short" if side == "long" else "long"

    assert outputs[f"profit_sum_{side}"] is profit
    assert outputs[f"loss_sum_{side}"] is loss
    assert outputs[f"profit_sum_{other_side}"].item() == 0.0
    assert outputs[f"loss_sum_{other_side}"].item() == 0.0


def test_candidate_position_slots_follow_candidate_positions_and_enabledness():
    torch = pytest.importorskip("torch")

    outputs = _candidate_position_slot_outputs(
        [
            {"long_n_positions": 3.0, "short_total_wallet_exposure_limit": 0.0},
            {"short_n_positions": 2.0},
        ],
        {"long": 2.0, "short": 1.0},
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["position_slots_long"].tolist() == [3.0, 2.0]
    assert outputs["position_slots_short"].tolist() == [0.0, 2.0]


def test_single_coin_side_eligibility_uses_prepared_coin_payload():
    config = {
        "bot": {
            "long": {"risk": {"total_wallet_exposure_limit": 1.0, "n_positions": 1}}
        },
        "live": {"approved_coins": {"long": ["BTC"]}},
    }

    assert _prepared_single_coin_side_enabled(
        config, "long", {"entry_eligible": True}
    )
    assert not _prepared_single_coin_side_enabled(
        config, "long", {"entry_eligible": False}
    )


def test_single_coin_side_eligibility_requires_canonical_payload_flag():
    with pytest.raises(ValueError, match="entry_eligible"):
        _prepared_single_coin_side_enabled({}, "short", {})


def test_directional_hsl_output_contract_retains_lifecycle_and_panic_scalars():
    assert {
        "hsl_triggers_long",
        "hsl_duration_sum_steps",
        "hsl_restart_retrigger_count",
        "hsl_halt_to_restart_equity_loss",
        "hsl_panic_close_loss_sum",
        "hsl_panic_loss_drawdown_count",
        "hsl_strategy_eq_recovery_max_ms_long",
        "hsl_strategy_eq_recovery_max_ms_short",
        "hsl_drawdown_ema_mean_worst_1pct_long",
        "hsl_drawdown_ema_mean_worst_1pct_short",
        "hsl_drawdown_raw_mean_worst_1pct_long",
        "hsl_drawdown_raw_mean_worst_1pct_short",
    } <= DIRECTIONAL_HSL_OUTPUT_KEYS


def test_hsl_diagnostics_are_requested_only_for_hsl_metric_families():
    assert not _hsl_diagnostics_needed(
        {
            "adg_strategy_eq",
            "drawdown_worst_strategy_eq",
            "fills_per_day",
        }
    )
    assert _hsl_diagnostics_needed({"hard_stop_triggers_per_year"})
    assert _hsl_diagnostics_needed({"drawdown_worst_ema_strategy_eq_long"})
    assert _hsl_diagnostics_needed({"peak_recovery_days_strategy_eq_long"})


def test_combine_hedged_multicoin_hsl_outputs_reduces_pside_episodes():
    torch = pytest.importorskip("torch")
    from optimization.gpu.metrics import (
        _hard_stop_lifecycle_metrics,
        _hard_stop_panic_loss_metrics,
    )

    def side_output(*, side, triggers, restarts, minimum, drawdown_count):
        zeros = torch.zeros(1)
        output = {key: zeros.clone() for key in DIRECTIONAL_HSL_OUTPUT_KEYS}
        output[f"hsl_{side}_enabled"] = torch.ones(1, dtype=torch.bool)
        output[f"hsl_triggers_{side}"] = torch.tensor([triggers])
        output[f"hsl_restarts_{side}"] = torch.tensor([restarts])
        output["hsl_tier_samples_total"] = torch.tensor([10.0])
        output["hsl_tier_samples_yellow"] = torch.tensor([2.0])
        output["hsl_tier_samples_orange"] = torch.tensor([3.0])
        output["hsl_tier_samples_red"] = torch.tensor([4.0])
        output["hsl_duration_sum_steps"] = torch.tensor([7.0])
        output["hsl_duration_max_steps"] = torch.tensor([5.0])
        output["hsl_duration_count"] = torch.tensor([2.0])
        output["hsl_trigger_drawdown_sum"] = torch.tensor([0.3])
        output["hsl_trigger_drawdown_count"] = torch.tensor([1.0])
        output["hsl_flatten_time_sum_steps"] = torch.tensor([4.0])
        output["hsl_flatten_time_count"] = torch.tensor([1.0])
        output["hsl_restart_retrigger_count"] = torch.tensor([1.0])
        output["hsl_halt_to_restart_equity_loss"] = torch.tensor([12.0])
        output["hsl_panic_close_loss_sum"] = torch.tensor([8.0])
        output["hsl_panic_close_loss_max"] = torch.tensor([6.0])
        output["hsl_panic_loss_drawdown_min"] = torch.tensor([minimum])
        output["hsl_panic_loss_drawdown_sum"] = torch.tensor([0.4])
        output["hsl_panic_loss_drawdown_max"] = torch.tensor([0.3])
        output["hsl_panic_loss_drawdown_count"] = torch.tensor(
            [drawdown_count]
        )
        output[f"hsl_drawdown_ema_max_{side}"] = torch.tensor(
            [0.2 if side == "long" else 0.1]
        )
        output[f"hsl_strategy_eq_recovery_max_ms_{side}"] = torch.tensor(
            [7_200_000.0 if side == "long" else 3_600_000.0]
        )
        output[f"hsl_drawdown_ema_mean_worst_1pct_{side}"] = torch.tensor(
            [0.18 if side == "long" else 0.09]
        )
        output[f"hsl_drawdown_raw_mean_worst_1pct_{side}"] = torch.tensor(
            [0.16 if side == "long" else 0.07]
        )
        return output

    combined = _combine_hedged_multicoin_hsl_outputs(
        side_output(
            side="long", triggers=2.0, restarts=1.0, minimum=0.2, drawdown_count=2.0
        ),
        side_output(
            side="short", triggers=3.0, restarts=2.0, minimum=0.1, drawdown_count=1.0
        ),
    )

    assert combined["hsl_long_enabled"].item()
    assert combined["hsl_short_enabled"].item()
    assert combined["hsl_triggers_long"].item() == 2.0
    assert combined["hsl_triggers_short"].item() == 3.0
    assert combined["hsl_restarts_long"].item() == 1.0
    assert combined["hsl_restarts_short"].item() == 2.0
    assert combined["hsl_duration_sum_steps"].item() == 14.0
    assert combined["hsl_duration_max_steps"].item() == 5.0
    assert combined["hsl_duration_count"].item() == 4.0
    assert combined["hsl_panic_close_loss_sum"].item() == 16.0
    assert combined["hsl_panic_close_loss_max"].item() == 6.0
    assert combined["hsl_panic_loss_drawdown_min"].item() == pytest.approx(0.1)
    assert combined["hsl_panic_loss_drawdown_count"].item() == 3.0
    assert combined["hsl_drawdown_ema_max_long"].item() == pytest.approx(0.2)
    assert combined["hsl_drawdown_ema_max_short"].item() == pytest.approx(0.1)
    assert combined["hsl_strategy_eq_recovery_max_ms_long"].item() == 7_200_000.0
    assert combined["hsl_strategy_eq_recovery_max_ms_short"].item() == 3_600_000.0
    assert combined["hsl_drawdown_ema_mean_worst_1pct_long"].item() == pytest.approx(
        0.18
    )
    assert combined["hsl_drawdown_ema_mean_worst_1pct_short"].item() == pytest.approx(
        0.09
    )
    assert combined["hsl_drawdown_raw_mean_worst_1pct_long"].item() == pytest.approx(
        0.16
    )
    assert combined["hsl_drawdown_raw_mean_worst_1pct_short"].item() == pytest.approx(
        0.07
    )
    assert combined["hsl_tier_samples_total"].item() == 10.0
    assert combined["hsl_tier_samples_red"].item() == 8.0
    assert combined["hsl_tier_samples_orange"].item() == 2.0
    assert combined["hsl_tier_samples_yellow"].item() == 0.0

    combined.update(
        {
            "max_dd": torch.tensor([0.2]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([86_400_000.0]),
        }
    )
    run = SimpleNamespace(interval_ms=60_000, starting_balance=1_000.0)
    lifecycle = _hard_stop_lifecycle_metrics(combined, run)
    panic = _hard_stop_panic_loss_metrics(combined, run)
    assert lifecycle["hard_stop_triggers"].item() == 5.0
    assert lifecycle["hard_stop_restarts"].item() == 3.0
    assert lifecycle["hard_stop_duration_minutes_mean"].item() == 3.5
    assert panic["hard_stop_panic_close_loss_sum"].item() == 16.0
    assert panic["hard_stop_panic_close_loss_drawdown_pct_mean"].item() == (
        pytest.approx(0.8 / 3.0)
    )
    assert len(DIRECTIONAL_HSL_OUTPUT_KEYS) == 35


def test_refresh_hedged_multicoin_hsl_replays_only_cutoff_candidates():
    torch = pytest.importorskip("torch")

    class FakeRunner:
        def __init__(self, replacement):
            self.replacement = replacement
            self.calls = []

        def run(self, params, *, end_steps):
            self.calls.append((params.copy(), end_steps.copy()))
            return {
                key: torch.full(
                    (len(params),),
                    bool(self.replacement)
                    if key.endswith("_enabled")
                    else self.replacement,
                    dtype=(
                        torch.bool if key.endswith("_enabled") else torch.float32
                    ),
                )
                for key in DIRECTIONAL_HSL_OUTPUT_KEYS
            }

    side_outputs = {
        side: {
            key: torch.tensor(
                [True, True]
                if key.endswith("_enabled")
                else [10.0, 20.0],
                dtype=torch.bool if key.endswith("_enabled") else torch.float32,
            )
            for key in DIRECTIONAL_HSL_OUTPUT_KEYS
        }
        for side in ("long", "short")
    }
    runners = {"long": FakeRunner(3.0), "short": FakeRunner(4.0)}
    matrices = {
        "long": np.asarray([[1.0], [2.0]]),
        "short": np.asarray([[3.0], [4.0]]),
    }

    profiled_steps = []
    refreshed = _refresh_hedged_multicoin_hsl_at_portfolio_cutoff(
        side_outputs=side_outputs,
        runners=runners,
        parameter_matrices=matrices,
        combined_output={"liq_step": torch.tensor([-1.0, 2.0])},
        start_minute_of_day=60,
        runner_profile_callback=(
            lambda _runner, *, effective_candidate_steps: profiled_steps.append(
                effective_candidate_steps.copy()
            )
        ),
    )

    assert refreshed
    assert runners["long"].calls[0][0].tolist() == [[2.0]]
    assert runners["short"].calls[0][0].tolist() == [[4.0]]
    assert runners["long"].calls[0][1].tolist() == [2820]
    assert runners["short"].calls[0][1].tolist() == [2820]
    assert [steps.tolist() for steps in profiled_steps] == [[2820], [2820]]
    assert side_outputs["long"]["hsl_triggers_long"].tolist() == [10.0, 3.0]
    assert side_outputs["short"]["hsl_triggers_short"].tolist() == [10.0, 4.0]


def test_hedged_multicoin_hsl_cutoff_replay_honors_interrupt_between_sides():
    torch = pytest.importorskip("torch")
    calls = []
    checks = 0

    class FakeRunner:
        def run(self, params, *, end_steps):
            calls.append(params.tolist())
            return {
                key: torch.zeros(
                    len(params),
                    dtype=torch.bool if key.endswith("_enabled") else torch.float32,
                )
                for key in DIRECTIONAL_HSL_OUTPUT_KEYS
            }

    def interrupt_check():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt

    side_outputs = {
        side: {
            key: torch.zeros(
                1,
                dtype=torch.bool if key.endswith("_enabled") else torch.float32,
            )
            for key in DIRECTIONAL_HSL_OUTPUT_KEYS
        }
        for side in ("long", "short")
    }

    with pytest.raises(KeyboardInterrupt):
        _refresh_hedged_multicoin_hsl_at_portfolio_cutoff(
            side_outputs=side_outputs,
            runners={"long": FakeRunner(), "short": FakeRunner()},
            parameter_matrices={
                "long": np.asarray([[1.0]]),
                "short": np.asarray([[2.0]]),
            },
            combined_output={"liq_step": torch.tensor([1.0])},
            start_minute_of_day=0,
            interrupt_check=interrupt_check,
        )

    assert calls == [[[1.0]]]


def test_single_coin_proxy_preserves_directional_hsl_outputs_for_reduction():
    torch = pytest.importorskip("torch")
    proxy = MpsSingleCoinProxy.__new__(MpsSingleCoinProxy)
    proxy.batch_size = 1
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.needed_metrics = {"hard_stop_panic_close_loss_sum"}
    proxy._parameter_matrix = lambda candidates: np.zeros((len(candidates), 0))
    raw = {
        key: torch.zeros(1)
        for key in (
            "first_fill_ts",
            "last_fill_ts",
            "last_high_ts",
            "first_eq_ts",
            "last_eq_ts",
        )
    }
    raw["hsl_triggers_long"] = torch.tensor([2.0])
    raw["hsl_panic_close_loss_sum"] = torch.tensor([37.5])
    proxy.runner = SimpleNamespace(run=lambda *args, **kwargs: raw)

    def reduce(output, *args, **kwargs):
        assert output["hsl_triggers_long"].item() == 2.0
        assert output["hsl_panic_close_loss_sum"].item() == 37.5
        return {
            "hard_stop_panic_close_loss_sum": output[
                "hsl_panic_close_loss_sum"
            ]
        }

    proxy._compute_objectives = reduce

    assert proxy.evaluate([{}]) == [{"hard_stop_panic_close_loss_sum": 37.5}]


def test_single_coin_proxy_preserves_entry_interval_outputs_for_reduction():
    torch = pytest.importorskip("torch")
    proxy = MpsSingleCoinProxy.__new__(MpsSingleCoinProxy)
    proxy.batch_size = 1
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.needed_metrics = {"entry_interval_hours_mean"}
    proxy._parameter_matrix = lambda candidates: np.zeros((len(candidates), 0))
    raw = {
        key: torch.zeros(1)
        for key in (
            "first_fill_ts",
            "last_fill_ts",
            "last_high_ts",
            "first_eq_ts",
            "last_eq_ts",
        )
    }
    raw.update(
        {
            "entry_interval_sum_steps": torch.tensor([30.0]),
            "entry_interval_count": torch.tensor([2.0]),
            "entry_interval_max_steps": torch.tensor([20.0]),
            "entry_interval_hist": torch.ones((1, 128)),
        }
    )
    proxy.runner = SimpleNamespace(run=lambda *args, **kwargs: raw)

    def reduce(output, *args, **kwargs):
        assert output["entry_interval_sum_steps"].item() == 30.0
        assert output["entry_interval_count"].item() == 2.0
        assert output["entry_interval_max_steps"].item() == 20.0
        assert output["entry_interval_hist"].shape == (1, 128)
        return {"entry_interval_hours_mean": torch.tensor([0.25])}

    proxy._compute_objectives = reduce

    assert proxy.evaluate([{}]) == [{"entry_interval_hours_mean": 0.25}]


def test_multicoin_proxy_preserves_directional_hsl_outputs_for_reduction():
    torch = pytest.importorskip("torch")
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.batch_size = 1
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.sides = ["long"]
    proxy.needed_metrics = {"hard_stop_panic_close_loss_sum"}
    proxy._parameter_matrix = lambda candidates, side=None: np.zeros(
        (len(candidates), 0)
    )
    raw = {
        key: torch.zeros(1)
        for key in (
            "first_fill_ts",
            "last_fill_ts",
            "last_high_ts",
            "first_eq_ts",
            "last_eq_ts",
            "profit_sum",
            "loss_sum",
            "entry_initial_balance_pct",
        )
    }
    raw["hsl_triggers_long"] = torch.tensor([2.0])
    raw["hsl_panic_close_loss_sum"] = torch.tensor([37.5])
    proxy.runners = {"long": SimpleNamespace(run=lambda *args, **kwargs: raw)}

    def reduce(output, *args, **kwargs):
        assert output["hsl_triggers_long"].item() == 2.0
        assert output["hsl_panic_close_loss_sum"].item() == 37.5
        return {
            "hard_stop_panic_close_loss_sum": output[
                "hsl_panic_close_loss_sum"
            ]
        }

    proxy._compute_objectives = reduce

    assert proxy.evaluate([{}]) == [{"hard_stop_panic_close_loss_sum": 37.5}]


@pytest.mark.parametrize(
    ("param_keys", "candidate_key"),
    [
        (EMA_ANCHOR_MULTICOIN_PARAM_KEYS, "offset"),
        (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
            "entry_initial_qty_pct",
        ),
    ],
)
def test_multicoin_proxy_routes_dual_side_batch_through_fused_runner(
    param_keys, candidate_key
):
    torch = pytest.importorskip("torch")
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.batch_size = 2
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 1_000.0}
    proxy.run = SimpleNamespace()
    proxy.sides = ["long", "short"]
    proxy.needed_metrics = {"hard_stop_triggers"}
    proxy.param_keys = param_keys
    proxy.base_params = {
        side: {
            key: float(index + (100 if side == "short" else 0))
            for index, key in enumerate(param_keys)
        }
        for side in ("long", "short")
    }
    seen = {}

    def run(params, **kwargs):
        seen["params"] = params.copy()
        seen["kwargs"] = kwargs
        batch = len(params)
        raw = {
            key: torch.zeros(batch)
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            )
        }
        raw["hsl_triggers_long"] = torch.tensor([1.0, 2.0])
        raw["hsl_triggers_short"] = torch.tensor([3.0, 4.0])
        return raw

    proxy.fused_runner = SimpleNamespace(run=run)
    proxy.runners = {}

    def reduce(output, *args, **kwargs):
        return {
            "hard_stop_triggers": output["hsl_triggers_long"]
            + output["hsl_triggers_short"]
        }

    proxy._compute_objectives = reduce
    candidates = [
        {f"long_{candidate_key}": 0.25, f"short_{candidate_key}": 0.5},
        {f"long_{candidate_key}": 0.75, f"short_{candidate_key}": 1.0},
    ]

    assert proxy.evaluate(candidates) == [
        {"hard_stop_triggers": 4.0},
        {"hard_stop_triggers": 6.0},
    ]
    assert seen["params"].shape == (
        2,
        len(param_keys) * 2,
    )
    parameter_index = param_keys.index(candidate_key)
    side_width = len(param_keys)
    assert seen["params"][:, parameter_index].tolist() == [0.25, 0.75]
    assert seen["params"][:, side_width + parameter_index].tolist() == [0.5, 1.0]
    assert seen["kwargs"] == {"profile": False}


@pytest.mark.parametrize(
    (
        "strategy_kind",
        "runner_name",
        "override_cols",
        "proxy_mode",
        "interval_minutes",
    ),
    [
        (
            "ema_anchor",
            "MpsEmaAnchorMulticoinFusedRunner",
            EMA_ANCHOR_COIN_OVERRIDE_COLS,
            "shared-account-fused-ema-v1",
            5,
        ),
        (
            "trailing_martingale",
            "MpsTrailingMartingaleMulticoinFusedRunner",
            TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
            "shared-account-fused-tm-v1",
            1,
        ),
        (
            "trailing_martingale",
            "MpsTrailingMartingaleMulticoinFusedRunner",
            TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
            "shared-account-fused-tm-v1",
            5,
        ),
    ],
)
@pytest.mark.parametrize(
    (
        "needed_metrics",
        "tail_enabled",
        "raw_drawdown_enabled",
        "raw_tail_enabled",
        "recovery_distribution_enabled",
    ),
    [
        ({"hard_stop_time_in_red_pct"}, False, False, False, False),
        ({"drawdown_worst_mean_1pct_ema_strategy_eq"}, True, False, False, False),
        ({"drawdown_worst_strategy_eq_long"}, False, True, False, False),
        ({"drawdown_worst_mean_1pct_strategy_eq_long"}, False, True, True, False),
        ({"strategy_eq_recovery_days_p99"}, False, False, False, True),
        ({"entry_interval_hours_p95"}, False, False, False, False),
    ],
)
@pytest.mark.parametrize("dynamic_wel_by_tradability", [True, False])
def test_multicoin_proxy_constructs_fused_shared_account_runner(
    monkeypatch,
    strategy_kind,
    runner_name,
    override_cols,
    proxy_mode,
    interval_minutes,
    needed_metrics,
    tail_enabled,
    raw_drawdown_enabled,
    raw_tail_enabled,
    recovery_distribution_enabled,
    dynamic_wel_by_tradability,
):
    torch = pytest.importorskip("torch")

    import optimization.gpu.mps_kernel as mps_kernel
    import optimization.gpu.service as gpu_service

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    backtest = ModuleType("backtest")
    backtest._get_backtest_coin_override = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "backtest", backtest)
    config = get_template_config()
    config["live"].update(
        {
            "strategy_kind": strategy_kind,
            "approved_coins": {
                "long": ["BTC"],
                "short": ["ETH"],
            },
            "ignored_coins": {"long": ["ETH"], "short": ["BTC"]},
            "hedge_mode": False,
            "hsl_signal_mode": "coin",
        }
    )
    config["backtest"]["dynamic_wel_by_tradability"] = (
        dynamic_wel_by_tradability
    )
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(
            {
                "n_positions": 2,
                "total_wallet_exposure_limit": 1.0,
                "position_exposure_enforcer_enabled": (
                    strategy_kind == "trailing_martingale"
                ),
                "position_exposure_enforcer_threshold": 0.7,
                "total_exposure_enforcer_enabled": True,
                "total_exposure_enforcer_policy": "reduce_portfolio",
                "total_exposure_enforcer_threshold": 0.8,
            }
        )
        # Construction must retain the fused shared-account path when both
        # directional surfaces participate in global auto-unstuck selection.
        config["bot"][side]["unstuck"]["enabled"] = True
        config["bot"][side]["hsl"]["enabled"] = True
        flat = flatten_shared_bot_side(config["bot"][side])
        config["bot"][side].update(
            {key: value for key, value in flat.items() if key.startswith("unstuck_")}
        )

    def bot_payload(side, coin):
        flat = flatten_shared_bot_side(config["bot"][side])
        entry_eligible = (side, coin) in {("long", "BTC"), ("short", "ETH")}
        return {
            "entry_eligible": entry_eligible,
            "wallet_exposure_limit": -1.0 if entry_eligible else 0.0,
            "entry_cooldown_minutes": flat["risk_entry_cooldown_minutes"],
            "filter_volume_ema_span_1m": flat[
                "forager_volume_ema_span_1m"
            ],
            "filter_volatility_ema_span_1m": flat[
                "forager_volatility_ema_span_1m"
            ],
            "filter_volume_drop_pct": flat["forager_volume_drop_pct"],
            "forager_score_weights": flat["forager_score_weights"],
            "n_positions": flat["n_positions"],
            "total_wallet_exposure_limit": flat[
                "total_wallet_exposure_limit"
            ],
            "risk_twel_entry_gate_enabled": flat[
                "risk_twel_entry_gate_enabled"
            ],
            "risk_twel_enforcer_enabled": flat["risk_twel_enforcer_enabled"],
            "risk_twel_enforcer_policy": flat["risk_twel_enforcer_policy"],
            "risk_twel_enforcer_threshold": flat[
                "risk_twel_enforcer_threshold"
            ],
            "risk_wel_enforcer_enabled": flat[
                "risk_wel_enforcer_enabled"
            ],
            "risk_wel_enforcer_threshold": flat[
                "risk_wel_enforcer_threshold"
            ],
            "unstuck_enabled": flat["unstuck_enabled"],
            "hsl_enabled": flat["hsl_enabled"],
        }

    payload = SimpleNamespace(
        bot_params_list=[
            {side: bot_payload(side, coin) for side in ("long", "short")}
            for coin in ("BTC", "ETH")
        ],
        strategy_params_list=[
            {
                side: dict(config["bot"][side]["strategy"][strategy_kind])
                for side in ("long", "short")
            }
            for _ in range(2)
        ],
        exchange_params=[
            {
                "qty_step": 0.001,
                "price_step": 0.01,
                "min_qty": 0.001,
                "min_cost": 5.0,
                "c_mult": 1.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.0006,
            }
            for _ in range(2)
        ],
        backtest_params={
            "candle_interval_minutes": interval_minutes,
            "dynamic_wel_by_tradability": dynamic_wel_by_tradability,
            "forager_score_hysteresis_pct": 0.0,
            "last_valid_indices": [2, 3],
            "first_valid_indices": [0, 0],
            "equity_hard_stop_loss": {"signal_mode": "coin"},
            "coins": ["BTC", "ETH"],
            "starting_balance": 1_000.0,
            "global_warmup_bars": 1,
            "trade_start_indices": [1, 1],
            "requested_start_timestamp_ms": 0,
            "first_timestamp_ms": 0,
            "liquidation_threshold": 0.05,
            "filter_by_min_effective_cost": True,
            "max_realized_loss_pct": 1.0,
            "market_order_slippage_pct": 0.0,
            "hedge_mode": config["live"]["hedge_mode"],
        },
    )
    backtest.build_backtest_payload = lambda *args, **kwargs: payload
    built_data_kwargs = {}

    def fake_build_mps_multicoin_data(*args, **kwargs):
        built_data_kwargs.update(kwargs)
        return {
            "n": 4,
            "n_coins": 2,
            "n_days": 1,
            "ts0": 0,
        }

    monkeypatch.setattr(
        gpu_service,
        "build_mps_multicoin_data",
        fake_build_mps_multicoin_data,
    )
    constructed = {}

    class FakeFusedRunner:
        def __init__(self, run, data, **kwargs):
            constructed.update({"run": run, "data": data, "kwargs": kwargs})

    monkeypatch.setattr(mps_kernel, runner_name, FakeFusedRunner)
    values = np.ones((4, 2, 4), dtype=np.float64)
    timestamps = np.arange(4, dtype=np.int64) * interval_minutes * 60_000

    proxy = MpsMulticoinEmaProxy(
        config=config,
        hlcvs=values,
        mss={"BTC": {}, "ETH": {}},
        btc=np.ones(4, dtype=np.float64),
        timestamps=timestamps,
        exchange="bybit",
        batch_size=8,
        needed_metrics=needed_metrics,
    )

    assert isinstance(proxy.fused_runner, FakeFusedRunner)
    assert built_data_kwargs["include_hourly_ranges"] is True
    assert constructed["run"].interval_ms == interval_minutes * 60_000
    assert proxy.runners == {}
    assert proxy.coin_override_contract["proxy_mode"] == proxy_mode
    assert proxy.coin_override_contract["hsl_proxy_mode"] == proxy_mode
    assert constructed["kwargs"]["long_coin_overrides"].shape == (
        2,
        override_cols,
    )
    assert constructed["kwargs"]["short_coin_overrides"].shape == (
        2,
        override_cols,
    )
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    long_overrides = constructed["kwargs"]["long_coin_overrides"]
    short_overrides = constructed["kwargs"]["short_coin_overrides"]
    assert np.isnan(long_overrides[0, wallet_exposure_column])
    assert long_overrides[1, wallet_exposure_column] == 0.0
    assert short_overrides[0, wallet_exposure_column] == 0.0
    assert np.isnan(short_overrides[1, wallet_exposure_column])
    assert constructed["kwargs"]["hsl_ema_tail_enabled"] is tail_enabled
    assert (
        constructed["kwargs"]["hsl_raw_drawdown_enabled"]
        is raw_drawdown_enabled
    )
    assert constructed["kwargs"]["hsl_raw_tail_enabled"] is raw_tail_enabled
    assert (
        constructed["kwargs"]["recovery_distribution_enabled"]
        is recovery_distribution_enabled
    )
    assert "high_exposure_enabled" not in constructed["kwargs"]
    assert constructed["kwargs"]["hedge_mode"] is False
    assert constructed["kwargs"]["filter_by_min_effective_cost"] is True
    assert (
        constructed["kwargs"]["dynamic_wel_by_tradability"]
        is dynamic_wel_by_tradability
    )
    assert constructed["kwargs"]["entry_interval_enabled"] is (
        strategy_kind == "trailing_martingale"
        and "entry_interval_hours_p95" in needed_metrics
    )


def test_gpu_multicoin_proxy_accepts_staggered_valid_gaps_and_ended_tails():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [99, 98])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 50], [49, 99])
    _require_supported_multicoin_valid_tails(hlcvs, [100, 0], [99, 99])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [98, 97])
    _require_supported_multicoin_valid_tails(hlcvs, [0, 60], [39, 99])
    _require_supported_multicoin_valid_tails(
        np.ones((1401, 2, 4), dtype=np.float64),
        [0, 0],
        [1400, 0],
    )
    with pytest.raises(ValueError, match="at least one prepared coin"):
        _require_supported_multicoin_valid_tails(
            np.ones((100, 0, 4), dtype=np.float64), [], []
        )
    with pytest.raises(ValueError, match="matching first/last"):
        _require_supported_multicoin_valid_tails(hlcvs, [0], [99, 98])
    with pytest.raises(ValueError, match="non-empty prepared valid range"):
        _require_supported_multicoin_valid_tails(
            hlcvs, [100, 100], [99, 99]
        )
    with pytest.raises(ValueError, match="first_valid_idx within"):
        _require_supported_multicoin_valid_tails(hlcvs, [99, 0], [98, 99])


def test_gpu_multicoin_proxy_accepts_all_nan_candle_in_windows():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[40, :, :3] = np.nan

    _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [59, 99])


def test_gpu_multicoin_proxy_rejects_all_nan_first_valid_candle():
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[10, 1, :3] = np.nan

    with pytest.raises(ValueError, match="first-valid candle"):
        _require_supported_multicoin_valid_tails(hlcvs, [0, 10], [99, 99])


@pytest.mark.parametrize(
    "unsupported_value",
    [
        0.0,
        -1.0,
        np.inf,
        float(np.nextafter(0.0, 1.0)),
        float(np.finfo(np.float32).max) * 2.0,
    ],
)
def test_gpu_multicoin_proxy_rejects_finite_unmodeled_all_invalid_candle(
    unsupported_value,
):
    hlcvs = np.ones((100, 2, 4), dtype=np.float64)
    hlcvs[40, :, :3] = unsupported_value

    with pytest.raises(ValueError, match=r"finite but.*candle_index=40"):
        _require_supported_multicoin_valid_tails(hlcvs, [0, 0], [59, 99])


def test_gpu_multicoin_valid_tail_gate_rejects_out_of_range_last_index():
    with pytest.raises(ValueError, match="prepared valid range"):
        _require_supported_multicoin_valid_tails(
            np.ones((100, 2, 4), dtype=np.float64),
            [0, 0],
            [100, 99],
        )


@pytest.mark.parametrize(
    "unrepresentable_close",
    [np.nextafter(0.0, 1.0), np.finfo(np.float64).max],
)
def test_gpu_multicoin_forced_delist_requires_representable_final_candle(
    unrepresentable_close,
):
    hlcvs = np.ones((1401, 2, 4), dtype=np.float64)
    hlcvs[0, 0, 2] = unrepresentable_close

    with pytest.raises(ValueError, match="forced-delist final candle"):
        _require_supported_multicoin_valid_tails(
            hlcvs,
            [0, 0],
            [0, 1400],
        )


def test_gpu_hsl_requires_contiguous_valid_candles():
    high = np.array([100.0, np.nan, 100.0])
    low = np.array([99.0, np.nan, 99.0])
    close = np.array([99.5, np.nan, 99.5])

    with pytest.raises(ValueError, match="invalid candle at 1"):
        _require_no_internal_invalid_hsl_candles(
            high, low, close, first_valid_idx=0, last_valid_idx=2
        )
    _require_no_internal_invalid_hsl_candles(
        high, low, close, first_valid_idx=2, last_valid_idx=2
    )


def test_gpu_multicoin_hsl_requires_each_coin_to_have_contiguous_valid_candles():
    hlcvs = np.ones((3, 2, 4), dtype=np.float64)
    hlcvs[1, 1, :3] = np.nan

    with pytest.raises(ValueError, match="invalid candle at 1"):
        _require_no_internal_invalid_multicoin_hsl_candles(
            hlcvs,
            hsl_enabled_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[2, 2],
        )
    _require_no_internal_invalid_multicoin_hsl_candles(
        hlcvs,
        hsl_enabled_coins=[True, False],
        first_valid_indices=[0, 0],
        last_valid_indices=[2, 2],
    )


def test_gpu_proxy_accepts_only_exact_balance_only_internal_gaps():
    hlcvs = np.ones((4, 2, 4), dtype=np.float64)
    hlcvs[2, 1, :3] = np.nan

    _require_exact_safe_proxy_candles(
        hlcvs,
        exposure_eligible_coins=[True, True],
        first_valid_indices=[0, 0],
        last_valid_indices=[3, 3],
        require_positive_high_low=True,
    )

    hlcvs[2, 1, :3] = 0.0
    with pytest.raises(ValueError, match="coin index 1, invalid candle at 2"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )

    hlcvs[2, 1, :3] = [np.nan, np.nan, 1.0]
    with pytest.raises(ValueError, match="partially invalid"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )

    hlcvs[2, 1, :3] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _require_exact_safe_proxy_candles(
            hlcvs,
            exposure_eligible_coins=[True, True],
            first_valid_indices=[0, 0],
            last_valid_indices=[3, 3],
            require_positive_high_low=True,
        )


def test_gpu_single_coin_accepts_nan_gap_but_rejects_zero_for_all_metrics():
    hlcvs = np.ones((4, 1, 4), dtype=np.float64)
    hlcvs[2, 0, :3] = np.nan
    kwargs = {
        "first_valid_idx": 0,
        "last_valid_idx": 3,
    }

    _require_no_unsafe_single_coin_candles(hlcvs, **kwargs)

    hlcvs[2, 0, :3] = 0.0
    with pytest.raises(ValueError, match="invalid candle at 2"):
        _require_no_unsafe_single_coin_candles(hlcvs, **kwargs)

    hlcvs[2, 0, :3] = [0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="invalid candle at 2"):
        _require_no_unsafe_single_coin_candles(hlcvs, **kwargs)


def test_gpu_single_coin_rejects_nan_first_valid_candle():
    hlcvs = np.ones((4, 1, 4), dtype=np.float64)
    hlcvs[1, 0, :3] = np.nan

    with pytest.raises(ValueError, match="first-valid candle"):
        _require_no_unsafe_single_coin_candles(
            hlcvs,
            first_valid_idx=1,
            last_valid_idx=3,
        )


def test_gpu_single_coin_rejects_nan_forced_delist_endpoint():
    hlcvs = np.ones((1402, 1, 4), dtype=np.float64)
    hlcvs[1, 0, :3] = np.nan

    with pytest.raises(ValueError, match="forced-delist final candle"):
        _require_no_unsafe_single_coin_candles(
            hlcvs,
            first_valid_idx=0,
            last_valid_idx=1,
        )


@pytest.mark.parametrize(
    ("mode", "legacy_raw"), [("bounded", 0.0), ("legacy_raw", 1.0)]
)
def test_single_coin_exposure_policy_packs_rust_inputs(mode, legacy_raw):
    packed = _single_coin_exposure_params(
        {
            "we_excess_allowance_pct": 0.25,
            "we_excess_allowance_mode": mode,
            "total_exposure_entry_gate_enabled": False,
            "total_exposure_enforcer_threshold": 0.8,
        },
        side="long",
    )

    assert packed == {
        "we_excess_allowance_pct": 0.25,
        "we_excess_allowance_legacy_raw": legacy_raw,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.8,
    }


def test_single_coin_exposure_policy_rejects_unknown_allowance_mode():
    with pytest.raises(ValueError, match="we_excess_allowance_mode"):
        _single_coin_exposure_params(
            {"we_excess_allowance_mode": "raw"}, side="short"
        )


def test_tm_position_exposure_repair_packs_exact_rust_inputs():
    assert _position_exposure_enforcer_params(
        {
            "position_exposure_enforcer_enabled": True,
            "position_exposure_enforcer_threshold": 0.8,
        },
        side="short",
    ) == {
        "wel_enforcer_enabled": 1.0,
        "wel_enforcer_threshold": 0.8,
    }

    with pytest.raises(ValueError, match="finite positive"):
        _position_exposure_enforcer_params(
            {
                "position_exposure_enforcer_enabled": True,
                "position_exposure_enforcer_threshold": 0.0,
            },
            side="long",
        )


@pytest.mark.parametrize(
    ("policy", "reduce_portfolio"),
    [("reduce_overweight", 0.0), ("reduce_portfolio", 1.0)],
)
def test_tm_total_exposure_repair_packs_exact_rust_inputs(
    policy, reduce_portfolio
):
    assert _total_exposure_enforcer_params(
        {
            "total_exposure_enforcer_enabled": True,
            "total_exposure_enforcer_policy": policy,
        },
        side="long",
    ) == {
        "twel_enforcer_enabled": 1.0,
        "twel_enforcer_reduce_portfolio": reduce_portfolio,
    }

    with pytest.raises(ValueError, match="total_exposure_enforcer_policy"):
        _total_exposure_enforcer_params(
            {"total_exposure_enforcer_policy": "largest_loss"},
            side="short",
        )


def test_single_coin_unstuck_packs_exact_rust_inputs():
    assert _unstuck_params(
        {
            "unstuck_enabled": True,
            "unstuck_ema_gating_enabled": False,
            "unstuck_close_pct": 0.125,
            "unstuck_ema_dist": -0.01,
            "unstuck_loss_allowance_pct": 0.02,
            "unstuck_threshold": 0.85,
        }
    ) == {
        "unstuck_enabled": 1.0,
        "unstuck_ema_gating_enabled": 0.0,
        "unstuck_close_pct": 0.125,
        "unstuck_ema_dist": -0.01,
        "unstuck_loss_allowance_pct": 0.02,
        "unstuck_threshold": 0.85,
    }


def test_single_coin_hsl_packs_state_machine_inputs():
    packed = _hsl_params(
        {
            "hsl_enabled": True,
            "hsl_red_threshold": 0.2,
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 120.0,
            "hsl_no_restart_drawdown_threshold": 0.8,
            "hsl_restart_after_red_policy": "threshold",
            "hsl_tier_ratio_yellow": 0.5,
            "hsl_tier_ratio_orange": 0.75,
            "hsl_orange_tier_mode": "graceful_stop",
            "n_positions": 4,
        },
        signal_mode="coin",
    )

    assert packed == {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.2,
        "hsl_ema_span_minutes": 60.0,
        "hsl_cooldown_minutes_after_red": 120.0,
        "hsl_no_restart_drawdown_threshold": 0.8,
        "hsl_restart_policy": 1.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 1.0,
        "hsl_signal_mode": 2.0,
        "hsl_slot_count": 1.0,
    }

    with pytest.raises(ValueError, match="cannot represent"):
        _hsl_params(
            {
                "hsl_enabled": True,
                "hsl_no_restart_drawdown_threshold": 0.99999999,
            },
            signal_mode="coin",
        )


@pytest.mark.parametrize(
    ("bot_patch", "match"),
    [
        ({"hsl_red_threshold": -0.2}, "red_threshold must satisfy"),
        ({"hsl_ema_span_minutes": 0.0}, "ema_span_minutes must be >= 1"),
        (
            {
                "hsl_tier_ratio_yellow": 0.9,
                "hsl_tier_ratio_orange": 0.2,
            },
            "tier_ratios must satisfy",
        ),
    ],
)
def test_hsl_params_reject_invalid_effective_settings(bot_patch, match):
    bot = {"hsl_enabled": True, **bot_patch}

    with pytest.raises(ValueError, match=match):
        _hsl_params(bot, signal_mode="coin")


def test_hsl_params_reject_tier_ratios_that_collapse_in_float32():
    with pytest.raises(ValueError, match="remain strictly ordered.*float32"):
        _hsl_params(
            {
                "hsl_enabled": True,
                "hsl_tier_ratio_yellow": 0.50000001,
                "hsl_tier_ratio_orange": 0.50000002,
            },
            signal_mode="coin",
        )


@pytest.mark.parametrize(
    ("signal_mode", "expected_id"),
    [("unified", 0.0), ("pside", 1.0), ("coin", 2.0)],
)
def test_single_coin_hsl_packs_explicit_signal_mode_ids(signal_mode, expected_id):
    packed = _hsl_params({"hsl_enabled": False}, signal_mode=signal_mode)

    assert packed["hsl_signal_mode"] == expected_id


@pytest.mark.parametrize("signal_mode", ["unified", "coin", "pside"])
def test_dual_side_single_coin_hsl_accepts_all_signal_modes(signal_mode):
    validate_single_coin_hsl_signal_topology(signal_mode, enabled_side_count=2)


def test_one_sided_single_coin_hsl_accepts_unified_signal_mode():
    validate_single_coin_hsl_signal_topology("unified", enabled_side_count=1)


def test_single_coin_hsl_rejects_unknown_signal_mode():
    with pytest.raises(ValueError, match="coin, pside, or unified"):
        validate_single_coin_hsl_signal_topology("portfolio", enabled_side_count=1)


@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_one_sided_multicoin_hsl_accepts_all_signal_modes(signal_mode):
    validate_hsl_signal_topology(
        signal_mode, coin_count=3, enabled_side_count=1
    )


def test_dual_side_multicoin_hsl_accepts_decomposable_pside_mode():
    validate_hsl_signal_topology(
        "pside", coin_count=3, enabled_side_count=2
    )


@pytest.mark.parametrize("signal_mode", ["unified", "pside", "coin"])
def test_dual_side_multicoin_hsl_accepts_shared_account_controller(signal_mode):
    validate_hsl_signal_topology(
        signal_mode,
        coin_count=3,
        enabled_side_count=2,
        shared_account_controller=True,
    )


@pytest.mark.parametrize("signal_mode", ["coin", "unified"])
def test_dual_side_multicoin_hsl_rejects_joint_account_modes(signal_mode):
    with pytest.raises(ValueError, match="supports only pside"):
        validate_hsl_signal_topology(
            signal_mode, coin_count=3, enabled_side_count=2
        )


def test_directional_parameter_matrix_keeps_side_values_separate():
    proxy = MpsEmaAnchorProxy.__new__(MpsEmaAnchorProxy)
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_PARAM_KEYS},
    }
    proxy.param_keys = EMA_ANCHOR_PARAM_KEYS

    matrix = proxy._parameter_matrix(
        [{"long_offset": 0.125, "short_offset": 0.25}]
    )

    assert matrix.shape == (1, 2 * len(EMA_ANCHOR_PARAM_KEYS))
    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[0, len(EMA_ANCHOR_PARAM_KEYS) + offset_index] == 0.25


def test_single_coin_static_overrides_shadow_candidate_values_exact_last():
    proxy = MpsEmaAnchorProxy.__new__(MpsEmaAnchorProxy)
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_PARAM_KEYS},
    }
    proxy.param_keys = EMA_ANCHOR_PARAM_KEYS
    proxy.static_coin_override_params = {
        "long": {"offset": 0.125, "entry_cooldown_minutes": 37.0},
        "short": {},
    }

    matrix = proxy._parameter_matrix(
        [
            {
                "long_offset": 0.5,
                "long_entry_cooldown_minutes": 5.0,
                "short_offset": 0.25,
            }
        ]
    )

    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    cooldown_index = EMA_ANCHOR_PARAM_KEYS.index("entry_cooldown_minutes")
    assert matrix[0, offset_index] == pytest.approx(0.125)
    assert matrix[0, cooldown_index] == pytest.approx(37.0)
    assert matrix[0, len(EMA_ANCHOR_PARAM_KEYS) + offset_index] == pytest.approx(
        0.25
    )


@pytest.mark.parametrize(("side", "base"), [("long", 1.0), ("short", 2.0)])
def test_multicoin_parameter_matrix_uses_only_enabled_side(side, base):
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = [side]
    proxy.base_params = {
        side: {key: base for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS}
    }

    other_side = "short" if side == "long" else "long"
    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_offset": 0.125,
                f"{side}_we_excess_allowance_pct": 0.25,
                f"{side}_twel_enforcer_threshold": 0.8,
                f"{other_side}_offset": 9.0,
            }
        ]
    )

    assert matrix.shape == (1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS))
    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("we_excess_allowance_pct")
    ] == pytest.approx(0.25)
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] == pytest.approx(0.8)


def test_multicoin_parameter_matrix_keeps_dual_side_values_separate():
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = ["long", "short"]
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
    }
    candidate = {"long_offset": 0.125, "short_offset": 0.25}

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert long_matrix[0, offset_index] == 0.125
    assert short_matrix[0, offset_index] == 0.25


@pytest.mark.parametrize("side", ["long", "short"])
def test_multicoin_tm_parameter_matrix_keeps_forager_and_strategy_values(side):
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = [side]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        side: {key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS}
    }

    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_entry_threshold_base_pct": 0.125,
                f"{side}_forager_volume_drop_pct": 0.25,
                f"{side}_we_excess_allowance_pct": 0.4,
                f"{side}_twel_enforcer_threshold": 0.75,
            }
        ]
    )

    assert matrix.shape == (1, len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS))
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_threshold_base_pct"
        ),
    ] == 0.125
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "forager_volume_drop_pct"
        ),
    ] == 0.25
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "we_excess_allowance_pct"
        ),
    ] == pytest.approx(0.4)
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        ),
    ] == pytest.approx(0.75)


def test_multicoin_tm_parameter_matrix_keeps_dual_side_values_separate():
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = ["long", "short"]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        "long": {
            key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        },
        "short": {
            key: 2.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        },
    }
    candidate = {
        "long_entry_threshold_base_pct": 0.125,
        "short_entry_threshold_base_pct": 0.25,
    }

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "entry_threshold_base_pct"
    )
    assert long_matrix[0, index] == 0.125
    assert short_matrix[0, index] == 0.25


def test_combine_hedged_multicoin_outputs_uses_conservative_surface():
    torch = pytest.importorskip("torch")

    def side_output(*, end, minimum, fill, first_fill, last_fill, liq):
        return {
            "day_end_eq": torch.tensor([end], dtype=torch.float64),
            "day_min_eq": torch.tensor([minimum], dtype=torch.float64),
            "day_max_dd": torch.tensor([[0.10, 0.20]]),
            "day_volume": torch.tensor([[0.4, 0.5]]),
            "day_has_fill": torch.tensor([fill]),
            "day_min_balance": torch.tensor([[1_000.0, 1_000.0]]),
            "day_net_pnl": torch.tensor(
                [[end[0] - 1_000.0, end[1] - end[0]]]
            ),
            "day_last_fill_balance": torch.tensor([end]),
            "day_fill_count": torch.tensor(
                [[float(fill[0]), float(fill[1])]]
            ),
            "max_dd": torch.tensor([0.20]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([300.0]),
            "held_count": torch.tensor([2.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([first_fill]),
            "last_fill_ts": torch.tensor([last_fill]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([900.0]),
            "first_eq_ts": torch.tensor([100.0]),
            "last_eq_ts": torch.tensor([1_000.0]),
            "liq_step": torch.tensor([liq]),
            "profit_sum": torch.tensor([20.0]),
            "loss_sum": torch.tensor([5.0]),
            "fill_count": torch.tensor([float(sum(fill))]),
            "fill_count_entry": torch.tensor([float(sum(fill))]),
            "fill_count_long": torch.tensor([float(sum(fill))]),
            "fills_active_days_count": torch.tensor([float(any(fill))]),
        }

    long = side_output(
        end=[1_100.0, 1_200.0],
        minimum=[1_050.0, 1_100.0],
        fill=[True, False],
        first_fill=float("nan"),
        last_fill=700.0,
        liq=-1,
    )
    short = side_output(
        end=[950.0, 900.0],
        minimum=[925.0, 850.0],
        fill=[False, True],
        first_fill=300.0,
        last_fill=float("nan"),
        liq=-1,
    )
    short["day_max_dd"] = torch.tensor([[0.05, 0.30]])
    short["day_volume"] = torch.tensor([[0.1, 0.2]])
    short["max_dd"] = torch.tensor([0.30])
    short["held_max_ms"] = torch.tensor([200.0])
    short["held_sum_ms"] = torch.tensor([500.0])
    short["held_count"] = torch.tensor([3.0])
    short["position_unchanged_max_ms"] = torch.tensor([250.0])
    short["gap_max_ms"] = torch.tensor([250.0])
    short["recovery_max_ms"] = torch.tensor([500.0])
    short["pnl_recovery_max_ms"] = torch.tensor([450.0])
    short["last_high_ts"] = torch.tensor([800.0])

    combined = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )

    assert combined["day_end_eq"].tolist() == [[1_050.0, 1_100.0]]
    assert combined["day_min_eq"].tolist() == [[975.0, 950.0]]
    np.testing.assert_allclose(combined["day_max_dd"].numpy(), [[0.15, 0.50]])
    assert combined["max_dd"].item() == pytest.approx(0.50)
    np.testing.assert_allclose(combined["day_volume"].numpy(), [[0.5, 0.7]])
    assert combined["day_has_fill"].tolist() == [[True, True]]
    assert combined["day_net_pnl"].tolist() == [[50.0, 50.0]]
    assert combined["day_last_fill_balance"].tolist() == [[1_050.0, 1_100.0]]
    assert combined["day_fill_count"].tolist() == [[1.0, 1.0]]
    assert combined["first_fill_ts"].item() == 300.0
    assert combined["last_fill_ts"].item() == 700.0
    assert combined["last_high_ts"].item() == 800.0
    assert combined["liq_step"].item() == -1
    assert combined["profit_sum"].item() == 40.0
    assert combined["loss_sum"].item() == 10.0
    assert combined["profit_sum_long"].item() == 20.0
    assert combined["loss_sum_long"].item() == 5.0
    assert combined["profit_sum_short"].item() == 20.0
    assert combined["loss_sum_short"].item() == 5.0
    assert combined["fill_count"].item() == 2.0
    assert combined["fill_count_entry"].item() == 2.0
    assert combined["fill_count_long"].item() == 2.0
    assert combined["fills_active_days_count"].item() == 1.0
    assert combined["held_sum_ms"].item() == 800.0
    assert combined["held_count"].item() == 5.0
    assert combined["position_unchanged_max_ms"].item() == 250.0
    assert combined["pnl_recovery_max_ms"].item() == 450.0

    short["day_min_eq"][0, 1] = float("inf")
    short["last_eq_ts"] = torch.tensor([800.0])
    truncated = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )
    assert truncated["day_end_eq"][0, 1].item() == 0.0
    assert torch.isinf(truncated["day_min_eq"][0, 1])
    assert truncated["day_volume"][0, 1].item() == 0.0
    assert not truncated["day_has_fill"][0, 1].item()
    assert truncated["last_eq_ts"].item() == 800.0

    short["day_min_eq"][0, 1] = 850.0
    long["liq_step"] = torch.tensor([1])
    liquidated = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )
    assert torch.isfinite(liquidated["day_min_eq"][0, 0])
    assert torch.isinf(liquidated["day_min_eq"][0, 1])
    assert liquidated["day_end_eq"][0, 1].item() == 0.0
    assert liquidated["liq_step"].item() == 1


def test_combine_hedged_multicoin_outputs_detects_shared_equity_liquidation():
    torch = pytest.importorskip("torch")

    def side_output():
        return {
            "day_end_eq": torch.tensor([[1_000.0, 520.0, 900.0]]),
            "day_min_eq": torch.tensor([[900.0, 520.0, 800.0]]),
            "day_max_dd": torch.tensor([[0.10, 0.48, 0.20]]),
            "day_volume": torch.tensor([[0.1, 0.1, 0.1]]),
            "day_has_fill": torch.tensor([[True, True, True]]),
            "day_min_balance": torch.tensor([[900.0, 900.0, 900.0]]),
            "day_net_pnl": torch.zeros((1, 3)),
            "day_last_fill_balance": torch.full((1, 3), 1_000.0),
            "day_fill_count": torch.ones((1, 3)),
            "max_dd": torch.tensor([0.48]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([100.0]),
            "held_count": torch.tensor([1.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
            "profit_sum": torch.tensor([0.0]),
            "loss_sum": torch.tensor([0.0]),
            "fill_count": torch.tensor([3.0]),
            "fill_count_entry": torch.tensor([2.0]),
            "fill_count_long": torch.tensor([2.0]),
            "fills_active_days_count": torch.tensor([1.0]),
        }

    combined = _combine_hedged_multicoin_outputs(
        side_output(), side_output(), 1_000.0, 0.05, 0, 60_000
    )

    assert combined["liq_step"].item() == 1
    assert torch.isfinite(combined["day_min_eq"][0, 0])
    assert torch.isinf(combined["day_min_eq"][0, 1])
    assert torch.isinf(combined["day_min_eq"][0, 2])
    assert combined["day_end_eq"][0, 1].item() == 0.0
    assert combined["last_eq_ts"].item() == 86_340_000.0


def test_combine_hedged_multicoin_outputs_detects_shared_balance_depletion():
    torch = pytest.importorskip("torch")

    def side_output():
        return {
            "day_end_eq": torch.tensor([[900.0, 600.0, 800.0]]),
            "day_min_eq": torch.tensor([[800.0, 600.0, 700.0]]),
            "day_max_dd": torch.tensor([[0.10, 0.40, 0.30]]),
            "day_volume": torch.tensor([[0.1, 0.1, 0.1]]),
            "day_has_fill": torch.tensor([[True, True, True]]),
            "day_min_balance": torch.tensor([[900.0, 450.0, 700.0]]),
            "day_net_pnl": torch.zeros((1, 3)),
            "day_last_fill_balance": torch.full((1, 3), 1_000.0),
            "day_fill_count": torch.ones((1, 3)),
            "max_dd": torch.tensor([0.40]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([100.0]),
            "held_count": torch.tensor([1.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
            "profit_sum": torch.tensor([0.0]),
            "loss_sum": torch.tensor([0.0]),
            "fill_count": torch.tensor([3.0]),
            "fill_count_entry": torch.tensor([2.0]),
            "fill_count_long": torch.tensor([2.0]),
            "fills_active_days_count": torch.tensor([1.0]),
        }

    combined = _combine_hedged_multicoin_outputs(
        side_output(), side_output(), 1_000.0, 0.05, 0, 60_000
    )

    # Combined equity remains 200, above the 50 floor, while conservative
    # combined realized balance is -100 and must terminate the screen.
    assert combined["liq_step"].item() == 1
    assert torch.isfinite(combined["day_min_eq"][0, 0])
    assert torch.isinf(combined["day_min_eq"][0, 1])
    assert combined["last_eq_ts"].item() == 86_340_000.0


def test_multicoin_coin_overrides_pack_only_explicit_exact_values():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    strategy_override = dict(strategy_base, offset=0.25, ema_span_0=90.0)
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 15.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "ema_anchor": {"offset": 0.25, "ema_span_0": 90.0}
                        },
                        "risk": {
                            "entry_cooldown_minutes": 15.0,
                            "we_excess_allowance_pct": 0.25,
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_dist": -0.01,
                            "loss_allowance_pct": 0.02,
                            "threshold": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, EMA_ANCHOR_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("offset")] == pytest.approx(0.25)
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("ema_span_0")] == pytest.approx(90.0)
    assert matrix[1, 10] == pytest.approx(15.0)
    assert matrix[1, 11] == pytest.approx(0.4)
    assert matrix[1, 12] == pytest.approx(0.25)
    assert matrix[1, 13:19].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    assert np.isnan(matrix[1, 19:]).all()
    assert contract["coins"] == ["BTC", "ETH"]
    assert contract["values"][0] == [None] * EMA_ANCHOR_COIN_OVERRIDE_COLS
    assert contract["exact_overrides"] == [
        {},
        config["coin_overrides"]["ETH"],
    ]


def test_coin_override_contract_keeps_exact_values_beyond_float32_precision():
    first = 0.4
    second = float(np.nextafter(first, 1.0))

    def build(value):
        config = {
            "coin_overrides": {
                "ETH": {
                    "bot": {
                        "long": {
                            "strategy": {"ema_anchor": {"offset": value}}
                        }
                    }
                }
            }
        }
        payload = SimpleNamespace(
            strategy_params_list=[{"long": {"offset": value}}],
            bot_params_list=[
                {
                    "long": {
                        "entry_eligible": True,
                        "wallet_exposure_limit": -1.0,
                    }
                }
            ],
        )
        return _build_multicoin_ema_coin_overrides(
            config=config,
            mss={"ETH": {}},
            exchange="bybit",
            coins=["ETH"],
            payload=payload,
            side="long",
            resolve_override=lambda config, _mss, _exchange, coin: config[
                "coin_overrides"
            ].get(coin, {}),
        )

    first_matrix, first_contract = build(first)
    second_matrix, second_contract = build(second)

    assert np.array_equal(first_matrix, second_matrix, equal_nan=True)
    assert first_contract["values"] == second_contract["values"]
    assert first_contract["exact_overrides"] != second_contract["exact_overrides"]
    first_exact = first_contract["exact_overrides"][0]["bot"]["long"]
    second_exact = second_contract["exact_overrides"][0]["bot"]["long"]
    assert first_exact["strategy"]["ema_anchor"]["offset"] == first
    assert second_exact["strategy"]["ema_anchor"]["offset"] == second


def test_coin_override_contract_keeps_backtest_inert_live_values():
    exact_patch = {
        "live": {
            "forced_mode_long": "graceful_stop",
            "forced_mode_short": "panic",
            "leverage": 3,
        }
    }
    config = {"coin_overrides": {"ETH": exact_patch}}
    payload = SimpleNamespace(
        strategy_params_list=[{"long": {}}],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": False,
                    "wallet_exposure_limit": -1.0,
                }
            }
        ],
    )

    matrix, contract = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"ETH": {}},
        exchange="bybit",
        coins=["ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert np.isnan(matrix).all()
    assert contract["values"] == [[None] * EMA_ANCHOR_COIN_OVERRIDE_COLS]
    assert contract["exact_overrides"] == [exact_patch]


@pytest.mark.parametrize(
    ("builder", "strategy", "forced_column"),
    [
        (
            _build_multicoin_ema_coin_overrides,
            {},
            EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
        ),
        (
            _build_multicoin_tm_coin_overrides,
            {"entry": {}, "close": {}},
            TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
        ),
    ],
)
def test_multicoin_coin_overrides_pack_forced_normal_active_slot(
    builder, strategy, forced_column
):
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": False,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "entry_eligible": True,
                    "is_forced_active": True,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {"live": {"forced_mode_long": "normal"}}
        }
    }

    matrix, contract = builder(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert np.isnan(matrix[0, forced_column])
    assert matrix[1, forced_column] == 1.0
    assert contract["values"][0][forced_column] is None
    assert contract["values"][1][forced_column] == 1.0


def test_multicoin_coin_overrides_pack_dual_sides_independently():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base, "short": strategy_base},
            {
                "long": dict(strategy_base, offset=0.25),
                "short": dict(strategy_base, offset=0.5),
            },
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                },
                "short": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": -1.0,
                },
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "wallet_exposure_limit": 0.4,
                },
                "short": {
                    "risk_entry_cooldown_minutes": 30.0,
                    "wallet_exposure_limit": -1.0,
                },
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {"ema_anchor": {"offset": 0.25}},
                        "wallet_exposure_limit": 0.4,
                    },
                    "short": {
                        "strategy": {"ema_anchor": {"offset": 0.5}},
                        "risk": {"entry_cooldown_minutes": 30.0},
                    },
                }
            }
        }
    }
    def resolver(config, _mss, _exchange, coin):
        return config["coin_overrides"].get(coin, {})

    long_matrix, _ = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=resolver,
    )
    short_matrix, _ = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="short",
        resolve_override=resolver,
    )

    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert long_matrix[1, offset_index] == pytest.approx(0.25)
    assert long_matrix[1, 11] == pytest.approx(0.4)
    assert np.isnan(long_matrix[1, 10])
    assert short_matrix[1, offset_index] == pytest.approx(0.5)
    assert short_matrix[1, 10] == pytest.approx(30.0)
    assert np.isnan(short_matrix[1, 11])


@pytest.mark.parametrize(
    ("builder", "wallet_exposure_column"),
    [
        (_build_multicoin_ema_coin_overrides, 11),
        (_build_multicoin_tm_coin_overrides, 24),
    ],
)
def test_multicoin_coin_overrides_preserve_side_entry_eligibility(
    builder, wallet_exposure_column
):
    strategy = {
        "entry": {},
        "close": {},
    }
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy},
            {"long": strategy},
        ],
        bot_params_list=[
            {
                "long": {
                    "entry_eligible": False,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.0,
                }
            },
            {
                "long": {
                    "entry_eligible": True,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
        ],
    )

    matrix, contract = builder(
        config={"coin_overrides": {}},
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda *_args: {},
    )

    assert matrix[0, wallet_exposure_column] == 0.0
    assert np.isnan(matrix[1, wallet_exposure_column])
    assert contract["values"][0][wallet_exposure_column] == 0.0
    assert contract["values"][1][wallet_exposure_column] is None


def test_multicoin_coin_overrides_pack_complete_hsl_group():
    strategy = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    effective_hsl = {
        "hsl_enabled": False,
        "hsl_red_threshold": 0.2,
        "hsl_ema_span_minutes": 5.5,
        "hsl_cooldown_minutes_after_red": 12.5,
        "hsl_no_restart_drawdown_threshold": 0.8,
        "hsl_restart_after_red_policy": "always",
        "hsl_tier_ratio_yellow": 0.4,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_tier_mode": "graceful_stop",
        "hsl_panic_close_order_type": "market",
    }
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {"long": {}},
            {"long": effective_hsl},
        ],
    )
    hsl_patch = {
        "enabled": False,
        "red_threshold": 0.2,
        "ema_span_minutes": 5.5,
        "cooldown_minutes_after_red": 12.5,
        "no_restart_drawdown_threshold": 0.8,
        "restart_after_red_policy": "always",
        "tier_ratios": {"yellow": 0.4, "orange": 0.75},
        "orange_tier_mode": "graceful_stop",
        "panic_close_order_type": "market",
    }
    config = {
        "coin_overrides": {"ETH": {"bot": {"long": {"hsl": hsl_patch}}}}
    }

    matrix, contract = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, EMA_ANCHOR_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[
        1, 19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
    ].tolist() == pytest.approx(
        [0.0, 0.2, 5.5, 12.5, 0.8, 0.0, 0.4, 0.75, 1.0, 1.0]
    )
    assert contract["values"][1][
        19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
    ] == pytest.approx(
        matrix[1, 19:EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN].tolist()
    )
    assert np.isnan(matrix[1, EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN])


def test_multicoin_coin_overrides_reject_invalid_hsl_panic_order_type():
    strategy = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    payload = SimpleNamespace(
        strategy_params_list=[{"long": strategy}, {"long": strategy}],
        bot_params_list=[
            {"long": {}},
            {
                "long": {
                    "hsl_enabled": True,
                    "hsl_panic_close_order_type": "makret",
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {"hsl": {"panic_close_order_type": "makret"}}
                }
            }
        }
    }

    with pytest.raises(ValueError, match="to be limit or market"):
        _build_multicoin_ema_coin_overrides(
            config=config,
            mss={"BTC": {}, "ETH": {}},
            exchange="bybit",
            coins=["BTC", "ETH"],
            payload=payload,
            side="long",
            resolve_override=lambda config, _mss, _exchange, coin: config[
                "coin_overrides"
            ].get(coin, {}),
        )


def test_multicoin_tm_coin_overrides_pack_only_explicit_exact_values():
    assert tuple(
        key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
    ) == TRAILING_MARTINGALE_PARAM_KEYS[:23]
    strategy_base = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_gate_mode": "all",
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    strategy_override = {
        **strategy_base,
        "entry": {
            **strategy_base["entry"],
            "ema_gate_mode": "reentry",
            "threshold_base_pct": 0.25,
            "retracement_base_pct": 1.0e-50,
        },
        "close": {
            **strategy_base["close"],
            "qty_pct": 0.5,
            "retracement_base_pct": 1.0e-50,
        },
    }
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {
                "long": {
                    "risk_entry_cooldown_minutes": 0.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "risk_entry_cooldown_minutes": 15.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "risk_wel_enforcer_enabled": True,
                    "risk_wel_enforcer_threshold": 0.8,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {
                                    "ema_gate_mode": "reentry",
                                    "threshold_base_pct": 0.25,
                                    "retracement_base_pct": 1.0e-50,
                                },
                                "close": {
                                    "qty_pct": 0.5,
                                    "retracement_base_pct": 1.0e-50,
                                },
                            }
                        },
                        "risk": {
                            "entry_cooldown_minutes": 15.0,
                            "we_excess_allowance_pct": 0.25,
                            "position_exposure_enforcer_enabled": True,
                            "position_exposure_enforcer_threshold": 0.8,
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_dist": -0.01,
                            "loss_allowance_pct": 0.02,
                            "threshold": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = _build_multicoin_tm_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, TRAILING_MARTINGALE_COIN_OVERRIDE_COLS)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, 7] == pytest.approx(0.25)
    assert matrix[1, 11] == np.finfo(np.float32).tiny
    assert matrix[1, 15] == pytest.approx(0.5)
    assert matrix[1, 20] == np.finfo(np.float32).tiny
    assert (
        matrix[
            1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN
        ]
        == 0.0
    )
    assert (
        matrix[
            1, TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN
        ]
        == 1.0
    )
    assert matrix[1, 23] == pytest.approx(15.0)
    assert matrix[1, 24] == pytest.approx(0.4)
    assert matrix[1, 25] == pytest.approx(0.25)
    assert matrix[1, 26] == pytest.approx(1.0)
    assert matrix[1, 27] == pytest.approx(0.8)
    assert matrix[1, 28:34].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    assert np.isnan(
        matrix[
            1,
            34:TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
        ]
    ).all()
    assert contract["values"][0] == [
        None
    ] * TRAILING_MARTINGALE_COIN_OVERRIDE_COLS


def test_trailing_parameter_matrix_keeps_nested_flattened_sides_separate():
    proxy = MpsEmaAnchorProxy.__new__(MpsEmaAnchorProxy)
    proxy.param_keys = TRAILING_MARTINGALE_PARAM_KEYS
    proxy.base_params = {
        "long": {key: 1.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
        "short": {key: 2.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
    }

    matrix = proxy._parameter_matrix(
        [
            {
                "long_entry_threshold_base_pct": 0.125,
                "short_close_qty_pct": 0.25,
            }
        ]
    )

    assert matrix.shape == (1, 2 * len(TRAILING_MARTINGALE_PARAM_KEYS))
    entry_index = TRAILING_MARTINGALE_PARAM_KEYS.index(
        "entry_threshold_base_pct"
    )
    close_index = TRAILING_MARTINGALE_PARAM_KEYS.index("close_qty_pct")
    assert matrix[0, entry_index] == 0.125
    assert (
        matrix[0, len(TRAILING_MARTINGALE_PARAM_KEYS) + close_index] == 0.25
    )


def test_gpu_side_enablement_uses_config_risk_not_per_coin_sentinel():
    config = {
        "bot": {
            "long": {
                "risk": {"total_wallet_exposure_limit": 1.0, "n_positions": 1}
            },
            "short": {
                "risk": {"total_wallet_exposure_limit": 0.0, "n_positions": 0}
            },
        },
        "live": {"approved_coins": {"long": ["BTC"], "short": ["BTC"]}},
    }

    assert gpu_side_enabled(config, "long")
    assert not gpu_side_enabled(config, "short")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("disabled", (0.0, 0.0)),
        ("all", (1.0, 1.0)),
        ("initial", (1.0, 0.0)),
        ("reentry", (0.0, 1.0)),
    ],
)
def test_trailing_martingale_flattening_preserves_nested_params_and_gates(
    mode, expected
):
    strategy = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_gate_mode": mode,
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    flattened = flatten_trailing_martingale_params(
        strategy,
        {"entry_cooldown_minutes": 7.0, "total_wallet_exposure_limit": 1.5},
    )

    assert tuple(flattened) == TRAILING_MARTINGALE_PARAM_KEYS
    assert flattened["entry_double_down_factor"] == 1.1
    assert flattened["close_qty_pct"] == 0.2
    assert (flattened["gate_initial"], flattened["gate_reentry"]) == expected


def test_trailing_martingale_flattening_rejects_unknown_gate_mode():
    with pytest.raises(ValueError, match="ema_gate_mode"):
        flatten_trailing_martingale_params(
            {
                "entry": {"ema_gate_mode": "mystery"},
                "close": {},
            },
            {"total_wallet_exposure_limit": 1.0},
        )


def test_trailing_martingale_flattening_reads_canonical_payload_cooldown():
    strategy = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {"ema_gate_mode": "all"},
        "close": {},
    }
    for key in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS:
        _name, path = key
        target = strategy
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target.setdefault(path[-1], 0.1)

    flattened = flatten_trailing_martingale_params(
        strategy,
        {
            "risk_entry_cooldown_minutes": 37.0,
            "total_wallet_exposure_limit": 1.5,
        },
    )

    assert flattened["entry_cooldown_minutes"] == 37.0
