import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
    _build_hourly_log_range,
    _directional_touch_ticks,
    _maximum_effective_min_cost,
    _minimum_entry_qty_encoding,
    _strict_fill_tick_boundaries,
    build_mps_data,
    build_mps_multicoin_data,
)
from optimization.gpu.mps_kernel import _encode_max_realized_loss_pct
from optimization.gpu.metrics import _fill_gap_metrics


@pytest.mark.parametrize(
    "value", [0.0, 0.05, 0.999999999, float(np.nextafter(1.0, 0.0))]
)
def test_mps_realized_loss_limit_float32_encoding_never_loosens(value):
    encoded = _encode_max_realized_loss_pct(value)

    assert encoded <= value
    assert encoded < 1.0


@pytest.mark.parametrize("value", [1.0, 2.0, 1.0e100])
def test_mps_disabled_realized_loss_limit_has_finite_float32_encoding(value):
    assert _encode_max_realized_loss_pct(value) == 1.0


def _single_coin_exposure_fields(
    *, allowance_pct=0.0, legacy_raw=False, entry_gate=True, threshold=1.0
):
    return [
        allowance_pct,
        float(legacy_raw),
        float(entry_gate),
        threshold,
    ]


def _tm_wel_enforcer_fields(*, enabled=False, threshold=1.0):
    return [float(enabled), threshold]


def _tm_twel_enforcer_fields(
    *,
    enabled=False,
    unstuck_enabled=False,
    unstuck_ema_gating_enabled=True,
    unstuck_close_pct=0.1,
    unstuck_ema_dist=0.0,
    unstuck_loss_allowance_pct=0.1,
    unstuck_threshold=0.5,
):
    return [
        float(enabled),
        float(unstuck_enabled),
        float(unstuck_ema_gating_enabled),
        unstuck_close_pct,
        unstuck_ema_dist,
        unstuck_loss_allowance_pct,
        unstuck_threshold,
    ] + list(_HSL_DISABLED_VALUES.values())


_UNSTUCK_DISABLED_VALUES = {
    "unstuck_enabled": 0.0,
    "unstuck_ema_gating_enabled": 1.0,
    "unstuck_close_pct": 0.1,
    "unstuck_ema_dist": 0.0,
    "unstuck_loss_allowance_pct": 0.1,
    "unstuck_threshold": 0.5,
}

_HSL_DISABLED_VALUES = {
    "hsl_enabled": 0.0,
    "hsl_red_threshold": 0.2,
    "hsl_ema_span_minutes": 60.0,
    "hsl_cooldown_minutes_after_red": 0.0,
    "hsl_no_restart_drawdown_threshold": 1.0,
    "hsl_restart_policy": 1.0,
    "hsl_tier_ratio_yellow": 0.5,
    "hsl_tier_ratio_orange": 0.75,
    "hsl_orange_graceful_stop": 0.0,
    "hsl_signal_coin": 0.0,
    "hsl_slot_count": 1.0,
}


def _single_coin_param_row(values, keys):
    merged = {**_UNSTUCK_DISABLED_VALUES, **_HSL_DISABLED_VALUES, **values}
    return [merged[key] for key in keys]


def _multicoin_exposure_fixture(
    strategy_kind,
    side,
    coin_overrides=None,
    *,
    count=64,
    markets=None,
    closes=(100.0, 120.0),
    highs=None,
    lows=None,
    max_realized_loss_pct=1.0,
    first_valid_indices=(0, 0),
    liquidation_threshold=0.05,
):
    coin_count = 2
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    close_matrix = np.asarray(closes, dtype=np.float64)
    for coin in range(coin_count):
        close = (
            np.full(count, close_matrix[coin])
            if close_matrix.ndim == 1
            else close_matrix[:, coin]
        )
        hlcvs[:, coin, 0] = (
            close * 1.01 if highs is None else np.asarray(highs)[:, coin]
        )
        hlcvs[:, coin, 1] = (
            close * 0.99 if lows is None else np.asarray(lows)[:, coin]
        )
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    if markets is None:
        markets = [
            ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
            for _ in range(coin_count)
        ]
    runs = [
        ProxyRun(
            1_000.0,
            1,
            max(1, first_valid_indices[coin]),
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            liquidation_threshold,
            first_valid_indices[coin],
            count - 1,
        )
        for coin in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    if strategy_kind == "ema_anchor":
        values = {
            "base_qty_pct": 1.0,
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "entry_double_down_factor": 0.0,
            "offset": 0.0,
            "offset_psize_weight": 0.0,
            "offset_volatility_1h_weight": 0.0,
            "offset_volatility_1m_weight": 0.0,
            "offset_volatility_ema_span_1h": 2.0,
            "offset_volatility_ema_span_1m": 2.0,
            "entry_cooldown_minutes": 0.0,
            "total_wallet_exposure_limit": 1.0,
            "forager_volume_ema_span_1m": 2.0,
            "forager_volatility_ema_span_1m": 2.0,
            "forager_volume_drop_pct": 0.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "we_excess_allowance_legacy_raw": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
        }
        row = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
        runner = MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            side=side,
            coin_overrides=coin_overrides,
            max_realized_loss_pct=max_realized_loss_pct,
        )
    else:
        values = {
            "ema_span_0": 2.0,
            "ema_span_1": 3.0,
            "volatility_ema_span_1h": 2.0,
            "volatility_ema_span_1m": 2.0,
            "entry_double_down_factor": 0.0,
            "entry_initial_ema_dist": 0.0,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_threshold_we_weight": 0.0,
            "entry_threshold_volatility_1h_weight": 0.0,
            "entry_threshold_volatility_1m_weight": 0.0,
            "entry_retracement_base_pct": 0.0,
            "entry_retracement_we_weight": 0.0,
            "entry_retracement_volatility_1h_weight": 0.0,
            "entry_retracement_volatility_1m_weight": 0.0,
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 10.0,
            "close_threshold_we_weight": 0.0,
            "close_threshold_volatility_1h_weight": 0.0,
            "close_threshold_volatility_1m_weight": 0.0,
            "close_retracement_base_pct": 0.0,
            "close_retracement_volatility_1h_weight": 0.0,
            "close_retracement_volatility_1m_weight": 0.0,
            "entry_cooldown_minutes": 0.0,
            "total_wallet_exposure_limit": 1.0,
            "gate_initial": 0.0,
            "gate_reentry": 0.0,
            "forager_volume_ema_span_1m": 2.0,
            "forager_volatility_ema_span_1m": 2.0,
            "forager_volume_drop_pct": 0.0,
            "forager_score_weights_volume": 1.0,
            "forager_score_weights_ema_readiness": 0.0,
            "forager_score_weights_volatility": 0.0,
            "n_positions": 2.0,
            "we_excess_allowance_pct": 0.0,
            "we_excess_allowance_legacy_raw": 0.0,
            "twel_entry_gate_enabled": 1.0,
            "twel_enforcer_threshold": 1.0,
            "wel_enforcer_enabled": 0.0,
            "wel_enforcer_threshold": 1.0,
            "twel_enforcer_enabled": 0.0,
            "twel_enforcer_reduce_portfolio": 0.0,
            **_UNSTUCK_DISABLED_VALUES,
        }
        row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]
        runner = MpsTrailingMartingaleMulticoinRunner(
            runs[0],
            data,
            side=side,
            coin_overrides=coin_overrides,
            max_realized_loss_pct=max_realized_loss_pct,
        )
    return runner, row


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_tracks_position_unchanged_max(strategy_kind, side):
    runner, row = _multicoin_exposure_fixture(strategy_kind, side, count=16)

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    unchanged_ms = output["position_unchanged_max_ms"].item()
    assert unchanged_ms > 0.0
    assert unchanged_ms <= output["held_max_ms"].item()
    assert output["entry_initial_balance_pct"].item() == pytest.approx(0.5)
    assert output["total_wallet_exposure_max"].item() > 0.0
    assert (
        output["total_wallet_exposure_max"].item()
        >= output["total_wallet_exposure_mean"].item()
    )
    fill_days = output["day_has_fill"].bool()
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_uses_first_coin_override(
    strategy_kind, side
):
    override_cols = 19 if strategy_kind == "ema_anchor" else 34
    initial_qty_column = 0 if strategy_kind == "ema_anchor" else 6
    allowance_column = 12 if strategy_kind == "ema_anchor" else 25
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, initial_qty_column] = 0.25
    overrides[0, allowance_column] = 0.5
    runner, row = _multicoin_exposure_fixture(
        strategy_kind, side, coin_overrides=overrides, count=16
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    # TWEL 1.0 / two effective positions, with a bounded 50% allowance.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(0.1875)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_freezes_denominator_at_liquidation(
    strategy_kind, side
):
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=20,
        first_valid_indices=(0, 12),
        liquidation_threshold=1.0,
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert not output["alive"].item()
    # The liquidation floor terminates before the second coin becomes
    # tradable, so exact Rust retains a one-position divisor.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(1.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_initial_entry_pct_freezes_before_post_fill_balance_depletion(
    strategy_kind, side
):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 2.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 2.0),
    ]
    runner, row = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=12,
        markets=markets,
        first_valid_indices=(0, 3),
    )

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    assert not output["alive"].item()
    # Coin one fills and its fee depletes balance on the same candle that coin
    # two first becomes tradable. Exact Rust liquidates before growing the
    # effective-position denominator.
    assert output["entry_initial_balance_pct"].item() == pytest.approx(1.0)


def test_directional_touch_ticks_preserve_alignment_and_round_non_aligned_prices():
    down, up, nearest = _directional_touch_ticks(
        np.array([100.0, 100.006, 0.1 + 0.2]), 0.01
    )

    assert down.tolist() == [10_000, 10_000, 30]
    assert up.tolist() == [10_000, 10_001, 30]
    assert nearest.tolist() == [10_000, 10_001, 30]


def test_minimum_entry_qty_encoding_uses_original_float64_touch_price():
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)

    bits, relation = _minimum_entry_qty_encoding(
        np.array([100.004, 100.0]), market
    )
    rounded = np.ascontiguousarray(bits).view(np.float32)
    assert rounded.tolist() == pytest.approx([0.1, 0.101])
    assert relation.tolist() == [-1, -1]


def test_minimum_entry_qty_encoding_preserves_just_above_aligned_minimum():
    market = ProxyMarket(1.0, 0.001, 0.0, 11.0, 1.0, 0.0)

    bits, relation = _minimum_entry_qty_encoding(np.array([0.088]), market)
    rounded = np.ascontiguousarray(bits).view(np.float32)
    assert rounded.tolist() == [125.0]
    assert relation.tolist() == [1]


def test_maximum_effective_min_cost_preserves_float64_executable_threshold():
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)

    threshold = _maximum_effective_min_cost(
        np.array([100.004, 100.0]), market
    )

    assert threshold == pytest.approx(0.101 * 100.0)


from optimization.gpu.mps_kernel import (
    MpsEmaAnchorMulticoinRunner,
    MpsEmaAnchorMulticoinLongRunner,
    MpsEmaAnchorRunner,
    MpsTrailingMartingaleMulticoinRunner,
    MpsTrailingMartingaleRunner,
)


def test_strict_fill_tick_boundaries_preserve_float32_candle_crossing():
    step = 0.01
    order_tick = 371_177
    exact_order_price = order_tick * step
    represented_above = float(np.float32(exact_order_price))
    assert represented_above > exact_order_price

    high_fill_max, low_nonfill_max = _strict_fill_tick_boundaries(
        np.array([represented_above, exact_order_price]),
        np.array([exact_order_price, represented_above]),
        step,
    )

    assert high_fill_max.tolist() == [order_tick, order_tick - 1]
    assert low_nonfill_max.tolist() == [order_tick, order_tick]


def test_strict_fill_tick_boundaries_preserve_rust_step_decimal_rounding():
    step = 0.1
    order_tick = 371_177
    rust_order_price = 37_117.7
    assert order_tick * step > rust_order_price

    high_fill_max, low_nonfill_max = _strict_fill_tick_boundaries(
        np.array([rust_order_price]),
        np.array([rust_order_price]),
        step,
    )

    assert high_fill_max.tolist() == [order_tick - 1]
    assert low_nonfill_max.tolist() == [order_tick]


def test_initial_single_candle_hour_bucket_matches_rust_skip_contract():
    timestamps = 3_540_000 + np.arange(62, dtype=np.int64) * 60_000
    high = np.full(62, 105.0)
    low = np.full(62, 95.0)
    high[0] = 110.0
    low[0] = 90.0
    high[30] = 106.0
    low[30] = 94.0
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=1,
        trade_start_idx=1,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=len(timestamps) - 1,
    )

    hour_log_range, hour_valid = _build_hourly_log_range(
        high, low, timestamps, run
    )

    assert not hour_valid[1]
    assert hour_valid[61]
    assert hour_log_range[61] == pytest.approx(np.log(106.0 / 94.0))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_shader_smoke():
    import passivbot_rust

    source = passivbot_rust.mps_ema_anchor_source_py()
    assert "kernel void passivbot_ema_anchor" in source
    assert "constant int SIDE_PARAMS = 34" in source
    assert "total_exposure_reducer_qty" in source
    assert "secondary_close_qty" in source
    assert "realized_loss_gate_allows" in source
    assert "float32_floor_nonnegative" in source
    assert "record_realized_net" in source
    assert "const float max_realized_loss_pct = settings[14]" in source
    assert "const float taker_fee = settings[15]" in source
    assert "const float market_order_slippage_pct = fmax(settings[16], 0.0f)" in source
    assert "const bool long_hsl_panic_market = settings[17] > 0.5f" in source
    assert "const bool short_hsl_panic_market = settings[18] > 0.5f" in source
    assert "market_panic ? taker_fee : maker_fee" in source
    assert "constant int SCALAR_COLS = 50" in source
    assert "record_gross_pnl" in source
    assert "hsl_tier_samples_total" in source
    assert "h.restart_retrigger_count" in source
    assert "record_hsl_panic_fill(" in source
    assert "side.psize * price_now * c_mult / balance" in source
    assert "short_side.psize * c_mult * (short_side.pprice - close)" in source
    assert "long_close_fill || long_entry_fill" in source
    assert "short_close_fill || short_entry_fill" in source
    assert "fabs(adj) * cp * c_mult /" not in source
    assert "fabs(eq) * ep * c_mult /" not in source
    assert "fabs(adj) * cp / balance" in source
    assert "fabs(eq) * ep / balance" in source
    assert "floor(value / step + 0.5f) * step" in source
    assert "rint(value / step)" not in source
    assert source.count("touch_down_ticks") >= 3
    assert source.count("touch_up_ticks") >= 3
    assert "nearest_ticks(price_now, price_step)" not in source
    assert "nextafter(" not in source
    assert "(run_peak - eqf) / fmax(fabs(run_peak)" in source
    assert "fabs(raw_steps - nearest_count) <= 1.0e-8f" in source
    assert "if (current_cost_we >= cap" in source
    assert "const bool filter_by_min_effective_cost" in source
    assert "passes_min_effective_cost" in source
    assert "projected_cost_lower" in source
    assert "float guaranteed_balance_lower" in source
    assert "accumulate_min_cost_balance_error" not in source
    assert source.index("float eqf = liq ? liq_floor : equity") < source.index(
        "(run_peak - eqf) / fmax(fabs(run_peak)"
    )

    count = 512
    phase = np.linspace(0.0, 10.0 * np.pi, count)
    close = 100.0 + np.sin(phase) * 5.0
    high = close * 1.01
    low = close * 0.99
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
    )
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=10,
        trade_start_idx=10,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [
        0.1,
        10.0,
        30.0,
        1.5,
        0.01,
        0.0,
        0.0,
        0.0,
        60.0,
        60.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    parameters = np.array([row + row, row + row], dtype=np.float64)

    output = MpsEmaAnchorRunner(market, run, data).run(parameters)
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
    fill_days = output["day_has_fill"]
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )
    assert (output["total_wallet_exposure_max"] > 0.0).all()
    assert (
        output["total_wallet_exposure_max"]
        >= output["total_wallet_exposure_mean"]
    ).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_anchor_multicoin_directional_shader_smoke(side):
    import passivbot_rust

    source = passivbot_rust.mps_ema_anchor_multicoin_source_py()
    assert "kernel void passivbot_ema_anchor_multicoin" in source
    assert "kernel void passivbot_ema_anchor_multicoin_long" in source
    assert "constant int PARAM_COLS = 31" in source
    assert "constant int OVERRIDE_COLS = 19" in source
    assert "allowed_wallet_exposure_limit" in source
    assert "twel_entry_gate_enabled" in source
    assert "twel_enforcer_enabled" in source
    assert "twel_enforcer_reduce_portfolio" in source
    assert "clamped_market_price" in source
    assert "secondary_close_qty" in source
    assert "realized_loss_proxy_allows_close" in source
    assert "const bool loss_gate_enabled = run_settings[5] < 1.0f" in source
    assert "constant int DAILY_COLS = 9" in source
    assert "day_min_balance" in source
    assert "coin_override_or" in source
    assert "const float score_hysteresis = fmax(run_settings[4], 0.0f)" in source
    assert "incumbent[c] = selected[c] && psize[c] <= 0.0f" in source
    assert "if (!selected[c] || incumbent[c] || !survivor[c]) continue" in source
    assert "score[challenger] - score[incumbent_coin]" in source
    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (4.0 + coin)
        hlcvs[:, coin, 0] = close * 1.01
        hlcvs[:, coin, 1] = close * 0.99
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    row = [
        0.1,
        10.0,
        30.0,
        1.5,
        0.01,
        0.0,
        0.0,
        0.0,
        60.0,
        60.0,
        0.0,
        1.0,
        60.0,
        60.0,
        0.0,
        1.0,
        0.0,
        0.0,
        2.0,
    ]
    row += _single_coin_exposure_fields() + [0.0, 0.0]
    row += list(_UNSTUCK_DISABLED_VALUES.values())

    runner = MpsEmaAnchorMulticoinRunner(
        runs[0],
        data,
        side=side,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=0.1,
    )
    assert runner.settings.cpu()[4].item() == pytest.approx(0.02)
    assert runner.settings.cpu()[5].item() <= 0.1
    output = runner.run(np.array([row, row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_min_balance"].shape == output["day_min_eq"].shape
    assert torch.isfinite(
        output["day_min_balance"][output["day_min_eq"].isfinite()]
    ).all()
    assert output["day_has_fill"].sum().item() > 0
    assert (output["open_positions"] <= 2.0).all()

    with pytest.raises(ValueError, match="finite and non-negative"):
        MpsEmaAnchorMulticoinRunner(
            runs[0],
            data,
            side=side,
            forager_score_hysteresis_pct=-0.01,
        )
    with pytest.raises(ValueError, match="max_realized_loss_pct"):
        MpsEmaAnchorMulticoinRunner(
            runs[0], data, side=side, max_realized_loss_pct=float("nan")
        )

    legacy_long = MpsEmaAnchorMulticoinLongRunner(runs[0], data)
    assert legacy_long.side == "long"

    disabled = np.full((coin_count, 19), np.nan, dtype=np.float32)
    disabled[:, 11] = 0.0
    disabled_output = MpsEmaAnchorMulticoinRunner(
        runs[0], data, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()
    assert disabled_output["day_has_fill"].sum().item() == 0
    assert disabled_output["open_positions"].item() == 0.0

    exact_last = np.full((coin_count, 19), np.nan, dtype=np.float32)
    exact_last[:, :11] = np.asarray(row[:11], dtype=np.float32)
    changed_candidate = list(row)
    changed_candidate[:11] = [
        0.01,
        200.0,
        400.0,
        0.0,
        0.05,
        2.0,
        5.0,
        5.0,
        10.0,
        10.0,
        30.0,
    ]
    exact_last_output = MpsEmaAnchorMulticoinRunner(
        runs[0], data, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    torch.mps.synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0], exact_last_output["day_has_fill"][1]
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0], exact_last_output["day_end_eq"][1]
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_directional_shader_smoke(side):
    import passivbot_rust

    source = passivbot_rust.mps_trailing_martingale_multicoin_source_py()
    assert "kernel void passivbot_trailing_martingale_multicoin" in source
    assert "constant int PARAM_COLS = 48" in source
    assert "effective_n_positions" in source
    assert "min_since_open" in source
    assert "entry_retracement_base" in source
    assert "close_retracement_base" in source
    assert "as_type<float>(touch_min_qty_bits[k * C + c])" in source
    assert "constant int OVERRIDE_COLS = 34" in source
    assert "allowed_wallet_exposure_limit" in source
    assert "twel_entry_gate_enabled" in source
    assert "coin_override_or" in source
    assert "merge_reducer" not in source
    assert "finalized_reducer_qty" in source
    assert "realized_loss_proxy_allows_close" in source
    assert "const bool loss_gate_enabled = run_settings[5] < 1.0f" in source

    count = 512
    coin_count = 3
    phase = np.linspace(0.0, 12.0 * np.pi, count)
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    for coin in range(coin_count):
        close = 100.0 + coin * 20.0 + np.sin(phase + coin) * (6.0 + coin)
        hlcvs[:, coin, 0] = close * 1.015
        hlcvs[:, coin, 1] = close * 0.985
        hlcvs[:, coin, 2] = close
        hlcvs[:, coin, 3] = 100.0 * (coin + 1)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
        for _ in range(coin_count)
    ]
    runs = [
        ProxyRun(
            1_000.0,
            10,
            10,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    assert data["touch_nearest_ticks"].shape == (count, coin_count)
    assert data["touch_nearest_ticks"].dtype == torch.int32
    assert data["touch_min_qty_bits"].shape == (count, coin_count)
    assert data["touch_min_qty_relation"].shape == (count, coin_count)
    values = {
        "ema_span_0": 10.0,
        "ema_span_1": 30.0,
        "volatility_ema_span_1h": 60.0,
        "volatility_ema_span_1m": 60.0,
        "entry_double_down_factor": 1.5,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.1,
        "entry_threshold_base_pct": 0.02,
        "entry_threshold_we_weight": 0.0,
        "entry_threshold_volatility_1h_weight": 0.0,
        "entry_threshold_volatility_1m_weight": 0.0,
        "entry_retracement_base_pct": 0.0,
        "entry_retracement_we_weight": 0.0,
        "entry_retracement_volatility_1h_weight": 0.0,
        "entry_retracement_volatility_1m_weight": 0.0,
        "close_qty_pct": 0.2,
        "close_threshold_base_pct": 0.01,
        "close_threshold_we_weight": 0.0,
        "close_threshold_volatility_1h_weight": 0.0,
        "close_threshold_volatility_1m_weight": 0.0,
        "close_retracement_base_pct": 0.0,
        "close_retracement_volatility_1h_weight": 0.0,
        "close_retracement_volatility_1m_weight": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_1m": 60.0,
        "forager_volatility_ema_span_1m": 60.0,
        "forager_volume_drop_pct": 0.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 2.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        **_UNSTUCK_DISABLED_VALUES,
    }
    row = [values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS]

    runner = MpsTrailingMartingaleMulticoinRunner(
        runs[0],
        data,
        side=side,
        forager_score_hysteresis_pct=0.02,
        max_realized_loss_pct=0.1,
    )
    assert runner.settings.cpu()[5].item() <= 0.1
    output = runner.run(np.array([row, row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
    assert (output["open_positions"] <= 2.0).all()

    disabled = np.full((coin_count, 34), np.nan, dtype=np.float32)
    disabled[:, 24] = 0.0
    disabled_output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, side=side, coin_overrides=disabled
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()
    assert disabled_output["day_has_fill"].sum().item() == 0
    assert disabled_output["open_positions"].item() == 0.0

    exact_last = np.full((coin_count, 34), np.nan, dtype=np.float32)
    exact_last[:, :25] = np.asarray(row[:25], dtype=np.float32)
    changed_candidate = list(row)
    changed_candidate[:25] = [
        200.0,
        400.0,
        240.0,
        180.0,
        0.0,
        0.05,
        0.01,
        0.1,
        2.0,
        5.0,
        5.0,
        0.02,
        3.0,
        5.0,
        5.0,
        0.8,
        0.05,
        2.0,
        5.0,
        5.0,
        0.02,
        5.0,
        5.0,
        120.0,
        1.0,
    ]
    exact_last_output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, side=side, coin_overrides=exact_last
    ).run(np.array([row, changed_candidate], dtype=np.float64))
    torch.mps.synchronize()
    assert torch.equal(
        exact_last_output["day_has_fill"][0],
        exact_last_output["day_has_fill"][1],
    )
    assert torch.equal(
        exact_last_output["day_end_eq"][0],
        exact_last_output["day_end_eq"][1],
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_multicoin_legacy_raw_allowance_with_gate_disabled_expands_volume(
    strategy_kind, side
):
    runner, baseline = _multicoin_exposure_fixture(strategy_kind, side)
    expanded = list(baseline)
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    expanded[keys.index("we_excess_allowance_pct")] = 0.5
    expanded[keys.index("we_excess_allowance_legacy_raw")] = 1.0
    expanded[keys.index("twel_entry_gate_enabled")] = 0.0

    output = runner.run(np.asarray([baseline, expanded], dtype=np.float64))
    torch.mps.synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert volume[1] > volume[0] * 1.1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_coin_override_allowance_expands_one_symbol(strategy_kind):
    override_cols = 19 if strategy_kind == "ema_anchor" else 34
    allowance_column = 12 if strategy_kind == "ema_anchor" else 25
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, allowance_column] = 1.0
    baseline_runner, baseline = _multicoin_exposure_fixture(strategy_kind, "long")
    override_runner, overridden = _multicoin_exposure_fixture(
        strategy_kind, "long", overrides
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    baseline[keys.index("twel_entry_gate_enabled")] = 0.0
    overridden[keys.index("twel_entry_gate_enabled")] = 0.0

    baseline_output = baseline_runner.run(
        np.asarray([baseline], dtype=np.float64)
    )
    override_output = override_runner.run(
        np.asarray([overridden], dtype=np.float64)
    )
    torch.mps.synchronize()

    baseline_volume = baseline_output["day_volume"].sum().item()
    override_volume = override_output["day_volume"].sum().item()
    assert baseline_volume > 0.0
    assert override_volume > baseline_volume * 1.1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_twel_threshold_reduces_entry_volume(strategy_kind):
    runner, full_cap = _multicoin_exposure_fixture(strategy_kind, "long")
    reduced_cap = list(full_cap)
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    reduced_cap[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(np.asarray([full_cap, reduced_cap], dtype=np.float64))
    torch.mps.synchronize()
    volume = output["day_volume"].sum(dim=1).cpu().numpy()

    assert volume[0] > 0.0
    assert 0.0 < volume[1] < volume[0] * 0.75


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_multicoin_equal_distance_twel_tie_keeps_higher_coin_index(
    strategy_kind,
):
    override_cols = 19 if strategy_kind == "ema_anchor" else 34
    wallet_exposure_column = 11 if strategy_kind == "ema_anchor" else 24
    overrides = np.full((2, override_cols), np.nan, dtype=np.float32)
    overrides[0, wallet_exposure_column] = 0.4
    overrides[1, wallet_exposure_column] = 0.5
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 300.0, 1.0, 0.0)
        for _ in range(2)
    ]
    runner, candidate = _multicoin_exposure_fixture(
        strategy_kind,
        "long",
        overrides,
        count=6,
        markets=markets,
        closes=(100.0, 100.0),
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    candidate[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()
    volume = output["day_volume"].sum().item()

    # Exact Rust removes equal-distance entries by ascending symbol index.
    # That leaves the higher-index coin's 0.5-WEL order. Keeping index zero
    # instead would admit only its 0.4-WEL order because the residual is dust.
    # EMA Anchor also closes the selected flat-price order in this short fixture,
    # so it reports the matching entry and exit volume.
    expected_volume = 1.0 if strategy_kind == "ema_anchor" else 0.5
    assert expected_volume - 0.05 < volume < expected_volume + 0.05


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_preserves_tick_aligned_computed_target():
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # A 100.00 bid must fill when the following candle trades one tick below.
    # Unconditionally nudging the aligned target down to 99.99 misses this fill.
    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_directionally_rounds_non_aligned_candle_touch():
    count = 5
    close = np.full(count, 100.006)
    high = np.full(count, 100.006)
    low = np.array([100.006, 100.006, 100.006, 100.005, 100.006])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, -0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust rounds a bid clamped to 100.006 down to 100.00. Nearest snapping
    # would place 100.01 and incorrectly fill on the 100.005 low.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_volume_uses_raw_non_positive_post_fill_balance():
    count = 64
    close = np.full(count, 100.0)
    high = np.full(count, 101.0)
    low = np.full(count, 99.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=2.0,
    )
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=1,
        trade_start_idx=1,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [1.0, 2.0, 3.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    parameters = np.array(
        [row + row],
        dtype=np.float64,
    )

    output = MpsEmaAnchorRunner(market, run, data).run(parameters)
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert output["day_volume"].sum().item() < 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_mps_dual_side_respects_one_way_initial_arbitration(hedge_mode):
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
    )
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=1,
        trade_start_idx=1,
        requested_start_ts_ms=int(timestamps[0]),
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=hedge_mode,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0
    assert (output["short_psize"].item() > 0.0) is hedge_mode


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_short_only_opens_short_position():
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.full(count, 100.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [0.1, 2.0, 3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() > 0.0


def _tm_row(*, initial_ema_dist=0.01, gate_initial=1.0, gate_reentry=1.0):
    return [
        2.0,  # ema_span_0
        3.0,  # ema_span_1
        2.0,  # volatility_ema_span_1h
        2.0,  # volatility_ema_span_1m
        1.0,  # entry_double_down_factor
        initial_ema_dist,
        0.1,  # entry_initial_qty_pct
        0.01,  # entry_threshold_base_pct
        0.0,  # entry_threshold_we_weight
        0.0,  # entry_threshold_volatility_1h_weight
        0.0,  # entry_threshold_volatility_1m_weight
        0.001,  # entry_retracement_base_pct
        0.0,  # entry_retracement_we_weight
        0.0,  # entry_retracement_volatility_1h_weight
        0.0,  # entry_retracement_volatility_1m_weight
        1.0,  # close_qty_pct
        0.01,  # close_threshold_base_pct
        0.0,  # close_threshold_we_weight
        0.0,  # close_threshold_volatility_1h_weight
        0.0,  # close_threshold_volatility_1m_weight
        0.001,  # close_retracement_base_pct
        0.0,  # close_retracement_volatility_1h_weight
        0.0,  # close_retracement_volatility_1m_weight
        0.0,  # entry_cooldown_minutes
        1.0,  # total_wallet_exposure_limit
        gate_initial,
        gate_reentry,
    ]


def _tm_single_row(
    *,
    initial_ema_dist=0.01,
    gate_initial=1.0,
    gate_reentry=1.0,
    allowance_pct=0.0,
    legacy_raw=False,
    entry_gate=True,
    threshold=1.0,
    wel_enforcer_enabled=False,
    wel_enforcer_threshold=1.0,
    twel_enforcer_enabled=False,
    unstuck_enabled=False,
    unstuck_ema_gating_enabled=True,
    unstuck_close_pct=0.1,
    unstuck_ema_dist=0.0,
    unstuck_loss_allowance_pct=0.1,
    unstuck_threshold=0.5,
):
    return _tm_row(
        initial_ema_dist=initial_ema_dist,
        gate_initial=gate_initial,
        gate_reentry=gate_reentry,
    ) + _single_coin_exposure_fields(
        allowance_pct=allowance_pct,
        legacy_raw=legacy_raw,
        entry_gate=entry_gate,
        threshold=threshold,
    ) + _tm_wel_enforcer_fields(
        enabled=wel_enforcer_enabled,
        threshold=wel_enforcer_threshold,
    ) + _tm_twel_enforcer_fields(
        enabled=twel_enforcer_enabled,
        unstuck_enabled=unstuck_enabled,
        unstuck_ema_gating_enabled=unstuck_ema_gating_enabled,
        unstuck_close_pct=unstuck_close_pct,
        unstuck_ema_dist=unstuck_ema_dist,
        unstuck_loss_allowance_pct=unstuck_loss_allowance_pct,
        unstuck_threshold=unstuck_threshold,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_position_unchanged_includes_open_tail(strategy_kind, side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[3] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(initial_ema_dist=0.01)
        runner_cls = MpsTrailingMartingaleRunner
    else:
        row = _single_coin_param_row(
            {
                "base_qty_pct": 0.1,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 0.0,
                "offset": 0.01,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        runner_cls = MpsEmaAnchorRunner

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position_size = output["psize" if side == "long" else "short_psize"]
    assert position_size.item() > 0.0
    expected_open_tail_ms = output["last_eq_ts"] - output["last_fill_ts"]
    assert expected_open_tail_ms.item() > 0.0
    assert output["position_unchanged_max_ms"].item() == pytest.approx(
        expected_open_tail_ms.item()
    )
    expected_long = 0.1 if side == "long" else 0.0
    expected_short = 0.1 if side == "short" else 0.0
    assert output["entry_initial_balance_pct_long"].item() == pytest.approx(
        expected_long
    )
    assert output["entry_initial_balance_pct_short"].item() == pytest.approx(
        expected_short
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_hsl_panics_and_permanently_halts(strategy_kind, side):
    count = 30
    close = np.full(count, 100.0)
    close[8:] = 70.0 if side == "long" else 130.0
    high = close * 1.02
    low = close * 0.98
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        baseline = _tm_single_row(initial_ema_dist=0.0)
        baseline[6] = 0.5
        baseline[7] = 0.5
        baseline[16] = 0.5
        keys = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsTrailingMartingaleRunner
    else:
        baseline = _single_coin_param_row(
            {
                "base_qty_pct": 0.5,
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "entry_double_down_factor": 1.0,
                "offset": 0.0,
                "offset_psize_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_ema_span_1m": 2.0,
                "entry_cooldown_minutes": 0.0,
                "total_wallet_exposure_limit": 1.0,
                "we_excess_allowance_pct": 0.0,
                "we_excess_allowance_legacy_raw": 0.0,
                "twel_entry_gate_enabled": 1.0,
                "twel_enforcer_threshold": 1.0,
                "twel_enforcer_enabled": 0.0,
            },
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
        )
        keys = EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
        runner_cls = MpsEmaAnchorRunner
    hsl = list(baseline)
    for key, value in {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.01,
        "hsl_ema_span_minutes": 1.0,
        "hsl_cooldown_minutes_after_red": 0.0,
        "hsl_no_restart_drawdown_threshold": 1.0,
        "hsl_restart_policy": 2.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 0.0,
        "hsl_signal_coin": 1.0,
        "hsl_slot_count": 1.0,
    }.items():
        hsl[keys.index(key)] = value
    restarting_hsl = list(hsl)
    restarting_hsl[keys.index("hsl_restart_policy")] = 0.0
    restarting_hsl[keys.index("hsl_cooldown_minutes_after_red")] = 2.0
    zero_cooldown_hsl = list(hsl)
    zero_cooldown_hsl[keys.index("hsl_restart_policy")] = 0.0
    unscaled_coin_hsl = list(hsl)
    unscaled_coin_hsl[keys.index("hsl_red_threshold")] = 0.6
    scaled_coin_hsl = list(unscaled_coin_hsl)
    scaled_coin_hsl[keys.index("hsl_slot_count")] = 4.0
    capped_coin_hsl = list(hsl)
    capped_coin_hsl[keys.index("hsl_restart_policy")] = 1.0
    capped_coin_hsl[keys.index("hsl_cooldown_minutes_after_red")] = 2.0
    capped_coin_hsl[keys.index("hsl_slot_count")] = 4.0
    tiny_threshold_hsl = list(hsl)
    tiny_threshold_hsl[keys.index("hsl_red_threshold")] = 1.0e-8
    negative_span_hsl = list(hsl)
    negative_span_hsl[keys.index("hsl_ema_span_minutes")] = -2.0
    recursive_close_hsl = list(hsl)
    if strategy_kind == "trailing_martingale":
        recursive_close_hsl[keys.index("close_retracement_base_pct")] = 0.0
    inactive = list(baseline)
    rows = (
        [
            baseline + inactive,
            hsl + inactive,
            restarting_hsl + inactive,
            zero_cooldown_hsl + inactive,
            unscaled_coin_hsl + inactive,
            scaled_coin_hsl + inactive,
            capped_coin_hsl + inactive,
            tiny_threshold_hsl + inactive,
            negative_span_hsl + inactive,
            recursive_close_hsl + inactive,
        ]
        if side == "long"
        else [
            inactive + baseline,
            inactive + hsl,
            inactive + restarting_hsl,
            inactive + zero_cooldown_hsl,
            inactive + unscaled_coin_hsl,
            inactive + scaled_coin_hsl,
            inactive + capped_coin_hsl,
            inactive + tiny_threshold_hsl,
            inactive + negative_span_hsl,
            inactive + recursive_close_hsl,
        ]
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray(rows, dtype=np.float64))
    market_runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        taker_fee=0.01,
        market_order_slippage_pct=0.02,
        hsl_panic_market_long=side == "long",
        hsl_panic_market_short=side == "short",
    )
    market_output = market_runner.run(
        np.asarray([rows[1]], dtype=np.float64)
    )
    gated_output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.0,
    ).run(np.asarray([rows[9]], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() > 0.0
    assert output[size_key][1].item() == 0.0
    assert output[size_key][2].item() > 0.0
    assert output[size_key][3].item() == 0.0
    assert output[size_key][4].item() > 0.0
    assert output[size_key][5].item() == 0.0
    assert output[size_key][6].item() > 0.0
    assert output[size_key][7].item() == 0.0
    assert output["balance"][7].item() < 990.0
    assert output[size_key][8].item() == 0.0
    assert output["balance"][8].item() == pytest.approx(
        output["balance"][1].item(), abs=1.0e-4
    )
    assert output["day_volume"][1].sum().item() > 1.0
    trigger_key = f"hsl_triggers_{side}"
    other_trigger_key = (
        "hsl_triggers_short" if side == "long" else "hsl_triggers_long"
    )
    restart_key = f"hsl_restarts_{side}"
    assert output[trigger_key][0].item() == 0.0
    assert output[trigger_key][1].item() == 1.0
    assert output[other_trigger_key][1].item() == 0.0
    assert output[restart_key][1].item() == 0.0
    assert output[restart_key][2].item() >= 1.0
    assert output["hsl_tier_samples_total"][1].item() > 0.0
    assert output["hsl_tier_samples_red"][1].item() > 0.0
    assert output["hsl_duration_count"][1].item() == 1.0
    assert output["hsl_duration_max_steps"][1].item() > 0.0
    assert output["hsl_trigger_drawdown_sum"][1].item() > 0.0
    assert output["hsl_trigger_drawdown_count"][1].item() == 1.0
    assert output["hsl_flatten_time_count"][1].item() == 1.0
    assert output["hsl_panic_close_loss_sum"][1].item() > 0.0
    assert output["hsl_panic_close_loss_max"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_count"][1].item() == 1.0
    assert output["hsl_panic_loss_drawdown_min"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_sum"][1].item() > 0.0
    assert output["hsl_panic_loss_drawdown_max"][1].item() > 0.0
    assert output[size_key][9].item() == 0.0
    assert output["hsl_panic_close_loss_sum"][9].item() > 0.0
    assert output["hsl_panic_loss_drawdown_count"][9].item() == 1.0
    assert gated_output[size_key].item() == 0.0
    assert gated_output["hsl_panic_close_loss_sum"].item() > 0.0
    assert market_runner.settings[15].item() == pytest.approx(0.01)
    assert market_runner.settings[16].item() == pytest.approx(0.02)
    assert market_runner.settings[17].item() == float(side == "long")
    assert market_runner.settings[18].item() == float(side == "short")
    assert market_output[size_key].item() == 0.0
    assert market_output["balance"].item() < output["balance"][1].item()
    assert (
        market_output["hsl_panic_close_loss_sum"].item()
        > output["hsl_panic_close_loss_sum"][1].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_auto_unstuck_reduces_eligible_position(
    strategy_kind, side
):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.01)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[:] = 100.0
        high[3] = 102.0
        low[:] = 99.99
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)

    def candidate(unstuck_enabled, *, ema_gating=False, ema_dist=0.0):
        if strategy_kind == "trailing_martingale":
            row = _tm_single_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=0.0,
                entry_gate=True,
                unstuck_enabled=unstuck_enabled,
                unstuck_ema_gating_enabled=ema_gating,
                unstuck_close_pct=0.1,
                unstuck_ema_dist=ema_dist,
                unstuck_loss_allowance_pct=0.2,
                unstuck_threshold=0.5,
            )
            row[6] = 1.0
            row[7] = 10.0
            row[11] = 0.0
            row[16] = 0.5
            row[20] = 0.0
            row[23] = 100.0
            return row + row
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=unstuck_enabled,
            unstuck_ema_gating_enabled=ema_gating,
            unstuck_close_pct=0.1,
            unstuck_ema_dist=ema_dist,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )
        return row + row

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
    ).run(
        np.asarray(
            [
                candidate(False),
                candidate(True),
                candidate(True, ema_gating=True, ema_dist=0.1),
            ],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()

    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    initial_size = (
        9.0 if strategy_kind == "ema_anchor" and side == "short" else 10.0
    )
    assert remaining[0] == pytest.approx(initial_size)
    assert remaining[1] == pytest.approx(initial_size - 1.0)
    assert remaining[2] == pytest.approx(initial_size)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_auto_unstuck_scales_loss_to_own_allowance(
    strategy_kind, side
):
    count = 6
    close = np.full(count, 100.0)
    if side == "long":
        close[3:] = 98.0
        high = close + 0.01
        low = close.copy()
    else:
        close[3:] = 102.0
        high = close.copy()
        low = close - 0.01
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)

    def candidate(*, unstuck_enabled, loss_allowance_pct):
        if strategy_kind == "trailing_martingale":
            row = _tm_single_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=0.0,
                entry_gate=True,
                unstuck_enabled=unstuck_enabled,
                unstuck_ema_gating_enabled=False,
                unstuck_close_pct=1.0,
                unstuck_loss_allowance_pct=loss_allowance_pct,
                unstuck_threshold=0.5,
            )
            row[6] = 1.0
            row[7] = 10.0
            row[11] = 0.0
            row[16] = 0.5
            row[20] = 0.0
            row[23] = 100.0
            return row + row
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=unstuck_enabled,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=1.0,
            unstuck_loss_allowance_pct=loss_allowance_pct,
            unstuck_threshold=0.5,
        )
        return row + row

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
    ).run(
        np.asarray(
            [
                candidate(unstuck_enabled=False, loss_allowance_pct=0.0005),
                candidate(unstuck_enabled=True, loss_allowance_pct=0.0005),
                candidate(unstuck_enabled=True, loss_allowance_pct=0.02),
            ],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()

    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    initial_size = (
        9.0 if strategy_kind == "ema_anchor" and side == "short" else 10.0
    )
    assert remaining[0] == pytest.approx(initial_size)
    assert remaining[1] == pytest.approx(initial_size - 1.0)
    generous_remainder = (
        1.0 if strategy_kind == "trailing_martingale" and side == "short" else 0.0
    )
    assert remaining[2] == pytest.approx(generous_remainder)

    strict_output = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.0005,
    ).run(
        np.asarray(
            [candidate(unstuck_enabled=True, loss_allowance_pct=0.02)],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()
    assert strict_output[key].item() == pytest.approx(initial_size)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_side_multicoin_auto_unstuck_selects_only_one_coin(
    strategy_kind, side
):
    count = 6
    closes = np.full((count, 2), 100.0)
    if side == "long":
        closes[3:, 0] = 97.5
        closes[3:, 1] = 98.5
        highs = closes + 0.01
        lows = closes.copy()
    else:
        closes[3:, 0] = 102.5
        closes[3:, 1] = 101.5
        highs = closes.copy()
        lows = closes - 0.01
    markets = [
        ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0),
        ProxyMarket(1.0, 0.01, 2.0, 0.0, 1.0, 0.0),
    ]
    runner, disabled = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.05,
    )
    keys = (
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        if strategy_kind == "ema_anchor"
        else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    )
    disabled[keys.index("entry_cooldown_minutes")] = 100.0
    disabled[keys.index("forager_volume_drop_pct")] = 1.0
    disabled[keys.index("twel_entry_gate_enabled")] = 0.0
    if strategy_kind == "ema_anchor":
        disabled[keys.index("offset")] = 0.01
    else:
        disabled[keys.index("entry_initial_ema_dist")] = 0.01
        disabled[keys.index("entry_initial_qty_pct")] = 1.0
        disabled[keys.index("entry_threshold_base_pct")] = 10.0
        disabled[keys.index("entry_retracement_base_pct")] = 0.0
        disabled[keys.index("close_threshold_base_pct")] = 0.5
        disabled[keys.index("close_retracement_base_pct")] = 0.0
        disabled[keys.index("gate_initial")] = 1.0
    enabled = list(disabled)
    enabled[keys.index("unstuck_enabled")] = 1.0
    enabled[keys.index("unstuck_ema_gating_enabled")] = 0.0
    enabled[keys.index("unstuck_close_pct")] = 0.1
    enabled[keys.index("unstuck_loss_allowance_pct")] = 0.2
    enabled[keys.index("unstuck_threshold")] = 0.5
    gated = list(enabled)
    gated[keys.index("unstuck_ema_gating_enabled")] = 1.0
    gated[keys.index("unstuck_ema_dist")] = 0.1
    allowance_disabled = list(disabled)
    allowance_disabled[keys.index("we_excess_allowance_pct")] = 1.0
    allowance_disabled[keys.index("twel_entry_gate_enabled")] = 1.0
    allowance_disabled[keys.index("twel_enforcer_threshold")] = 0.5
    allowance_enabled = list(enabled)
    allowance_enabled[keys.index("we_excess_allowance_pct")] = 1.0
    allowance_enabled[keys.index("twel_entry_gate_enabled")] = 1.0
    allowance_enabled[keys.index("twel_enforcer_threshold")] = 0.5

    output = runner.run(
        np.asarray(
            [
                disabled,
                enabled,
                gated,
                allowance_disabled,
                allowance_enabled,
            ],
            dtype=np.float64,
        )
    )
    torch.mps.synchronize()
    key = "psize" if side == "long" else "short_psize"
    remaining = output[key].cpu().numpy()
    open_positions = output["open_positions"].cpu().tolist()
    assert open_positions[:3] == [2.0] * 3
    assert remaining[0] > 1.0
    # Coin one is less stuck and has a two-unit minimum. Selecting coin zero,
    # or selecting both coins, would produce a different aggregate remainder.
    assert remaining[1] == pytest.approx(remaining[0] - 2.0)
    assert remaining[2] == pytest.approx(remaining[0])
    # The bounded allowance doubles each 0.5 per-coin WEL to 1.0, while the
    # side-wide entry gate caps exposure at 0.5. Exact Rust evaluates unstuck's
    # threshold against the allowed limit, so the position does not exceed it.
    assert open_positions[4] == open_positions[3]
    assert remaining[4] == pytest.approx(remaining[3])

    strict_runner, _ = _multicoin_exposure_fixture(
        strategy_kind,
        side,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.0005,
    )
    strict_output = strict_runner.run(
        np.asarray([enabled], dtype=np.float64)
    )
    torch.mps.synchronize()
    assert strict_output[key].item() == pytest.approx(remaining[0])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_single_coin_auto_unstuck_selects_one_least_stuck_side(strategy_kind):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.01)
    low = np.full(count, 99.99)
    high[3] = 102.0
    low[3] = 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)

    if strategy_kind == "trailing_martingale":
        row = _tm_single_row(
            initial_ema_dist=0.01,
            gate_initial=1.0,
            gate_reentry=0.0,
            entry_gate=True,
            unstuck_enabled=True,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=0.1,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )
        row[6] = 1.0
        row[7] = 10.0
        row[11] = 0.0
        row[16] = 0.5
        row[20] = 0.0
        row[23] = 100.0
    else:
        row = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            100.0,
            1.0,
        ]
        row += _single_coin_exposure_fields(entry_gate=True)
        row += _tm_twel_enforcer_fields(
            unstuck_enabled=True,
            unstuck_ema_gating_enabled=False,
            unstuck_close_pct=0.1,
            unstuck_loss_allowance_pct=0.2,
            unstuck_threshold=0.5,
        )

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == pytest.approx(9.0)
    expected_short = 9.0 if strategy_kind == "ema_anchor" else 10.0
    assert output["short_psize"].item() == pytest.approx(expected_short)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_equal_unstuck_twel_reducers_keep_nearer_twel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
    else:
        high[3] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=True,
        twel_enforcer_enabled=True,
        threshold=0.5,
        unstuck_enabled=True,
        unstuck_ema_gating_enabled=False,
        unstuck_close_pct=0.5,
        unstuck_loss_allowance_pct=0.2,
        unstuck_threshold=0.5,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.05,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The equal-size TWEL reducer is one price band more executable than the
    # auto-unstuck reducer. The next candle crosses TWEL but only touches
    # unstuck, proving selection follows Rust's reachability tie-break.
    assert output[size_key].item() == pytest.approx(5.0)
    assert output["total_wallet_exposure_max"].item() > 0.0
    fill_days = output["day_has_fill"]
    assert torch.isfinite(output["day_net_pnl"][fill_days]).all()
    assert (output["day_last_fill_balance"][fill_days] > 0.0).all()
    assert (output["day_fill_count"][fill_days] >= 1.0).all()
    assert torch.equal(
        output["day_fill_count"], output["day_fill_count"].round()
    )
    assert (
        output["total_wallet_exposure_max"].item()
        >= output["total_wallet_exposure_mean"].item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_unstuck_finalization_keeps_valid_ordinary_close_remainder():
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3] = 98.0
    high[4] = 201.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.5, 0.01, 0.5, 500.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.01,
        gate_initial=1.0,
        gate_reentry=0.0,
        entry_gate=True,
        threshold=1.0,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.2,
        unstuck_enabled=True,
        unstuck_ema_gating_enabled=False,
        unstuck_close_pct=0.7,
        unstuck_loss_allowance_pct=0.2,
        unstuck_threshold=0.5,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 1.0
    row[20] = 0.01
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=0.05,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # At the 100 touch, the requested unstuck close is 7 and its minimum is 5.
    # The ordinary close near 198 has a 3-unit quantity above its ~2.5 minimum,
    # so exact finalization keeps the unstuck selection key at 7 instead of
    # promoting it to the full 10. The genuinely larger WEL reducer therefore
    # wins and leaves less than the unstuck request would have left.
    assert output["psize"].item() < 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_unstuck_finalization_keeps_ordinary_close_remainder():
    count = 7
    closes = np.full((count, 2), 100.0)
    highs = np.full((count, 2), 100.0)
    lows = np.full((count, 2), 100.0)
    lows[3, 0] = 98.0
    highs[4, 0] = 201.0
    markets = [
        ProxyMarket(0.5, 0.01, 0.5, 500.0, 1.0, 0.0),
        ProxyMarket(0.5, 0.01, 0.5, 0.0, 1.0, 0.0),
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        "long",
        coin_overrides=overrides,
        count=count,
        markets=markets,
        closes=closes,
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.05,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "close_qty_pct": 0.3,
            "close_threshold_base_pct": 1.0,
            "close_retracement_base_pct": 0.01,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.2,
            "unstuck_enabled": 1.0,
            "unstuck_ema_gating_enabled": 0.0,
            "unstuck_close_pct": 0.7,
            "unstuck_loss_allowance_pct": 0.2,
            "unstuck_threshold": 0.5,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    # As in the directional kernel regression above, the valid ordinary close
    # keeps unstuck's finalized selection key below the larger WEL reducer.
    assert output["psize"].item() < 3.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_exposure_headroom_and_entry_gate(
    strategy_kind, side
):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)

    def row(*, legacy_raw, entry_gate):
        exposure = _single_coin_exposure_fields(
            allowance_pct=0.5,
            legacy_raw=legacy_raw,
            entry_gate=entry_gate,
            threshold=0.5,
        )
        if strategy_kind == "trailing_martingale":
            values = _tm_row(
                initial_ema_dist=0.01,
                gate_initial=1.0,
                gate_reentry=1.0,
            )
            values[6] = 1.0
            return (
                values
                + exposure
                + _tm_wel_enforcer_fields()
                + _tm_twel_enforcer_fields()
            )
        values = [
            1.0,
            2.0,
            3.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.0,
            1.0,
        ]
        return values + exposure + _tm_twel_enforcer_fields()

    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    runner = runner_cls(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    )
    rows = [
        row(legacy_raw=False, entry_gate=False),
        row(legacy_raw=True, entry_gate=False),
        row(legacy_raw=True, entry_gate=True),
    ]
    output = runner.run(
        np.asarray([values + values for values in rows], dtype=np.float64)
    )
    torch.mps.synchronize()
    sizes = (
        output["psize"] if side == "long" else output["short_psize"]
    ).cpu().numpy()

    assert sizes[0] > 0.0
    assert sizes[1] > sizes[0] * 1.4
    assert sizes[2] < sizes[0] * 0.6


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_position_exposure_repair_reduces_strictly_below_target(side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    baseline = _tm_single_row()
    baseline[6] = 1.0
    baseline[7] = 10.0
    baseline[11] = 0.0
    baseline[16] = 10.0
    baseline[20] = 0.0
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_enabled"
        )
    ] = 1.0
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_threshold"
        )
    ] = 0.5
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    repaired_we = (
        sizes[1]
        * output["pprice" if side == "long" else "short_pprice"][1].item()
        / output["balance"][1].item()
    )
    assert 0.0 < sizes[1] < sizes[0]
    assert 0.45 < repaired_we < 0.5
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_single_coin_total_exposure_repair(side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    # A coarse quantity step makes the semantic distinction observable:
    # TWEL repair rounds up to the target, while WEL repair intentionally
    # takes one additional step to finish strictly below its target.
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    baseline = _tm_single_row(entry_gate=False, threshold=0.5)
    baseline[6] = 1.0
    baseline[7] = 10.0
    baseline[11] = 0.0
    baseline[16] = 10.0
    baseline[20] = 0.0
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    assert sizes[1] == pytest.approx(5.0)
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_single_coin_total_exposure_repair(side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    if side == "long":
        close[2:] = 99.9
        high[2:] = 99.9
        low[3] = 98.0
        low[4:] = 99.9
    else:
        close[2:] = 100.1
        high[3] = 102.0
        high[4:] = 100.1
        low[2:] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 1.0,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 0.0,
    }
    baseline = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    repaired_values = dict(values, twel_enforcer_enabled=1.0)
    repaired = _single_coin_param_row(
        repaired_values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS
    )
    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(
        np.asarray(
            [baseline + baseline, repaired + repaired], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 8.0
    pprice_key = "pprice" if side == "long" else "short_pprice"
    repaired_we = (
        sizes[1]
        * output[pprice_key][1].item()
        / output["balance"][1].item()
    )
    assert 0.0 < sizes[1] < sizes[0]
    assert 0.35 < repaired_we <= 0.51
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_realized_loss_gate_blocks_lossy_total_exposure_repair(side):
    count = 10
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    if side == "long":
        close[2:] = 99.9
        high[2:] = 99.9
        low[3] = 98.0
        low[4:] = 99.9
    else:
        close[2:] = 100.1
        high[3] = 102.0
        high[4:] = 100.1
        low[2:] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 1.0,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
    }

    ungated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=0.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() < gated[size_key][0].item()
    assert gated[size_key][0].item() > 8.0
    assert gated["balance"][0].item() >= ungated["balance"][0].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_realized_loss_gate_shares_budget_between_sides():
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 1.0,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
        max_realized_loss_pct=0.0031,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    sizes = sorted([output["psize"][0].item(), output["short_psize"][0].item()])
    assert sizes[0] < 8.0
    assert sizes[1] > 8.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_realized_loss_gate_reserves_unfilled_batch_loss():
    count = 6
    close = np.array([100.0, 100.0, 100.0, 100.0, 99.9, 99.9])
    high = np.array([100.0, 100.0, 100.0, 100.01, 99.9, 99.9])
    low = np.array([100.0, 100.0, 100.0, 99.99, 99.9, 99.9])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 1.0,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.5,
        "twel_enforcer_enabled": 1.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    common = {
        "long_enabled": True,
        "short_enabled": True,
        "hedge_mode": True,
    }

    ungated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=1.0, **common
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market, run, data, max_realized_loss_pct=0.0031, **common
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    sizes = {
        "ungated_long": ungated["psize"][0].item(),
        "ungated_short": ungated["short_psize"][0].item(),
        "gated_long": gated["psize"][0].item(),
        "gated_short": gated["short_psize"][0].item(),
    }
    assert sizes["ungated_long"] > 8.0, sizes
    assert sizes["ungated_short"] < 8.0, sizes
    assert sizes["gated_long"] > 8.0, sizes
    assert sizes["gated_short"] > 8.0, sizes


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_zero_loss_budget_blocks_loss_below_balance_ulp():
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 100.01, 100.01, 100.0])
    low = np.array([100.0, 100.0, 100.0, 99.99, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.000001, 0.01, 0.000001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        100_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    values = {
        "base_qty_pct": 0.000001,
        "ema_span_0": 2.0,
        "ema_span_1": 3.0,
        "entry_double_down_factor": 0.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 2.0,
        "offset_volatility_ema_span_1m": 2.0,
        "entry_cooldown_minutes": 100.0,
        "total_wallet_exposure_limit": 1.0,
        "we_excess_allowance_pct": 0.0,
        "we_excess_allowance_legacy_raw": 0.0,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
    }
    row = _single_coin_param_row(values, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)

    ungated = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=1.0,
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
        max_realized_loss_pct=0.0,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert ungated["psize"][0].item() == pytest.approx(0.0)
    assert gated["psize"][0].item() > 0.0009


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_realized_loss_gate_blocks_lossy_total_exposure_repair(side):
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    high[3], low[3] = 105.0, 95.0
    if side == "long":
        close[4:] = 98.0
        high[4:] = 98.0
        low[4:] = 97.9
    else:
        close[4:] = 102.0
        high[4:] = 102.1
        low[4:] = 102.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
    }

    ungated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=0.1, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() < gated[size_key][0].item()
    assert gated[size_key][0].item() > 9.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_loss_gate_rebuilds_profitable_ordinary_close_after_reducer(side):
    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    high[3], low[3] = 105.0, 95.0
    if side == "long":
        close[4] = high[4] = 98.0
        low[4] = 97.9
        high[5] = 100.1
        low[5] = 98.0
    else:
        close[4] = low[4] = 102.0
        high[4] = 102.1
        high[5] = 102.0
        low[5] = 99.9
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 0.01
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        max_realized_loss_pct=0.1,
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_realized_loss_gate_blocks_fee_only_ordinary_close(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[2] = 99.99
        close[3:] = 99.0
        high[3] = 99.0
        low[3] = 98.9
        high[4] = 99.6
        low[4] = 99.0
    else:
        high[2] = 100.01
        close[3:] = 101.0
        high[3] = 101.1
        low[3] = 101.0
        high[4] = 101.0
        low[4] = 100.4
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        initial_ema_dist=0.0,
        gate_initial=False,
        gate_reentry=False,
        entry_gate=False,
    )
    row[6] = 0.1
    row[7] = 10.0
    row[11] = 0.0
    row[16] = -0.005
    row[17] = 0.0
    row[20] = 0.0
    kwargs = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
    }

    ungated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=1.0, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    gated = MpsTrailingMartingaleRunner(
        market, run, data, max_realized_loss_pct=0.1, **kwargs
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key][0].item() == pytest.approx(0.0)
    assert gated[size_key][0].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_equal_wel_twel_reducers_keep_nearer_wel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The WEL reducer rests at 100.00 and is not strictly touched by the next
    # candle.  Choosing the farther TWEL reducer would close half the position.
    assert output[size_key][0].item() == pytest.approx(10.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_finalized_reducer_tie_keeps_nearer_wel(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.05,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.1,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    twel_only = list(row)
    twel_only[
        TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(
            "wel_enforcer_enabled"
        )
    ] = 0.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(
        np.asarray(
            [row + row, twel_only + twel_only], dtype=np.float64
        )
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Raw TWEL is larger, but both candidates absorb their sub-minimum tail and
    # finalize to the full position. Exact Rust then selects the nearer WEL,
    # which is only touched (not strictly crossed) by the next candle.
    assert output[size_key][0].item() > 9.0
    assert output[size_key][1].item() == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_offsets_raw_close_before_tick_quantization(side):
    count = 6
    close = np.full(count, 100.019)
    high = np.full(count, 100.019)
    low = np.full(count, 100.019)
    high[3] = 105.0
    low[3] = 95.0
    if side == "long":
        close[4] = 99.95
        high[4] = 99.955
        low[4] = 99.90
    else:
        close[4] = 100.08
        high[4] = 100.10
        low[4] = 100.075
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[16] = 10.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Exact raw-first quantization emits 99.96 long / 100.07 short, neither of
    # which is strictly touched.  Rounding the close first moves the order one
    # tick farther and incorrectly fills it.
    assert output[size_key][0].item() == pytest.approx(10.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_repair_preserves_triggered_trailing_close(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.2
    row[16] = 0.0
    row[20] = 0.001
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # TWEL repair alone leaves five contracts.  The independently triggered
    # trailing close must therefore reduce the position further in the same
    # candle without exceeding the original position.
    assert 0.0 < output[size_key][0].item() < 5.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_wel_repair_suppresses_triggered_trailing_close(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    trailing = _tm_single_row(
        entry_gate=False,
        wel_enforcer_enabled=True,
        wel_enforcer_threshold=0.5,
    )
    trailing[6] = 1.0
    trailing[7] = 10.0
    trailing[11] = 0.0
    trailing[15] = 0.2
    trailing[16] = 0.0
    trailing[20] = 0.001
    trailing[23] = 100.0
    nontrailing = list(trailing)
    nontrailing[16] = 10.0
    nontrailing[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([trailing + trailing, nontrailing + nontrailing]))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Strategy WEL takes precedence inside calc_closes_*; unlike an appended
    # TWEL reducer, it must not retain a triggered trailing close.
    assert output[size_key][0].item() == pytest.approx(
        output[size_key][1].item()
    )
    assert output[size_key][0].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_and_trailing_close_absorb_dust_in_signed_order(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 99.9, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 98.0, 95.0, 100.0])
    if side == "long":
        close[4] = 99.0
        threshold = 0.4
        entry_price = 99.0
    else:
        close[4] = 101.0
        high[4] = 102.0
        low[4] = 100.1
        threshold = 0.5
        entry_price = 101.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 2.0, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=threshold,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 0.3
    row[16] = 0.0
    row[20] = 0.001
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key][0].item() == pytest.approx(0.0)

    # Both orders are generated on the retracement candle (99 long / 101
    # short), not on the following fill candle.
    reducer_price = 98.95 if side == "long" else 101.06
    ordinary_price = 99.0 if side == "long" else 101.0
    balance = 1_000.0 - 10.0 * entry_price * market.maker_fee
    expected_volume = 10.0 * entry_price / balance
    reducer_pnl = 6.0 * (
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
    )
    balance += reducer_pnl - 6.0 * reducer_price * market.maker_fee
    expected_volume += 6.0 * reducer_price / balance
    ordinary_pnl = 4.0 * (
        ordinary_price - entry_price
        if side == "long"
        else entry_price - ordinary_price
    )
    balance += ordinary_pnl - 4.0 * ordinary_price * market.maker_fee
    expected_volume += 4.0 * ordinary_price / balance
    # The TWEL order is marketable and therefore has a negative signed
    # distance; exact canonical ordering executes it before the touch close.
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_twel_grid_uses_pre_repair_strategy_state(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 105.0, 105.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 95.0, 95.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 1.0, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(
        entry_gate=False,
        threshold=0.5,
        twel_enforcer_enabled=True,
    )
    row[6] = 1.0
    row[7] = 10.0
    row[11] = 0.0
    row[15] = 1.0
    row[16] = 0.0
    row[17] = 0.08
    row[20] = 0.0
    row[23] = 100.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([row + row], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The original-position grid is outside this candle while the TWEL order
    # fills. Rebuilding from the post-TWEL position moves the grid inward and
    # incorrectly closes the remainder.
    assert output[size_key][0].item() > 3.5


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_position_reducer_reachability_survives_grid_pruning(side):
    count = 6
    close = np.full(count, 100.0)
    high = np.full(count, 102.0)
    low = np.full(count, 98.0)
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.10
    candidate[17] = 0.10
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    pprice_key = "pprice" if side == "long" else "short_pprice"
    size = output[size_key].item()
    repaired_we = size * output[pprice_key].item() / output["balance"].item()
    assert 0.0 < repaired_we < 0.5
    assert output["day_volume"].sum().item() > 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_grid_can_fill_without_off_tick_reducer(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Long: the 100.4 touch makes a reducer at 101 and rebuilt grid at 100;
    # high=100.5 reaches only the grid.  Short is the mirror image: reducer
    # 100, grid 101, low=100.5.  The residual grid quantity therefore closes
    # while the reducer-sized half remains open.
    assert output[size_key].item() == pytest.approx(5.05, abs=0.1)
    assert 1.0 < output["day_volume"].sum().item() < 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_off_tick_grid_precedes_reducer_for_volume(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_price = 100.0 if side == "long" else 101.0
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_price = 101.0 if side == "long" else 100.0
    reducer_qty = entry_qty - grid_qty
    balance = 1_000.0 - entry_qty * entry_price * market.maker_fee
    expected_volume = entry_qty * entry_price / balance
    grid_pnl = grid_qty * (
        grid_price - entry_price if side == "long" else entry_price - grid_price
    )
    balance += grid_pnl - grid_qty * grid_price * market.maker_fee
    expected_volume += grid_qty * grid_price / balance
    reducer_pnl = reducer_qty * (
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
    )
    balance += reducer_pnl - reducer_qty * reducer_price * market.maker_fee
    expected_volume += reducer_qty * reducer_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["profit_sum"].item() == pytest.approx(
        grid_pnl + reducer_pnl, abs=3.0e-4
    )
    assert output["loss_sum"].item() == 0.0
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_reducer_consumes_one_recursive_close_slot(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=0.5
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[15] = 0.0001
    candidate[16] = 0.0
    candidate[17] = 1.0e-6
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    # One reducer plus 499 ordinary closes exhausts Rust's shared 500-order
    # generation loop.  A fresh 500-rung proxy loop would remove one extra
    # quantity step here.
    expected_size = 4.052 if side == "long" else 4.451
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(expected_size, abs=2.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_small_excess_reducer_crosses_strict_target(side):
    count = 6
    entry_price = 99.0 if side == "long" else 101.0
    close = np.array([100.0, 100.0, 100.0, entry_price, entry_price, entry_price])
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 102.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 98.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0e-8, 1.0, 1.0e-8, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    target = 0.99999
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        wel_enforcer_enabled=True, wel_enforcer_threshold=target
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[20] = 0.001
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    pprice_key = "pprice" if side == "long" else "short_pprice"
    exposure = (
        output[size_key].item()
        * output[pprice_key].item()
        / output["balance"].item()
    )
    assert output[size_key].item() > 0.0
    assert exposure < target


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_same_tick_twel_and_grid_use_separate_volume_denominators(side):
    count = 6
    generation_touch = 100.4 if side == "long" else 100.6
    close = np.array([100.0, 100.0, 100.0, generation_touch, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 101.5, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 99.5, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.001)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    candidate = _tm_single_row(
        entry_gate=False, threshold=0.5, twel_enforcer_enabled=True
    )
    candidate[6] = 1.0
    candidate[7] = 10.0
    candidate[11] = 0.0
    candidate[16] = 0.0
    candidate[17] = 0.0
    candidate[20] = 0.0
    candidate[23] = 100.0
    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
    ).run(np.asarray([candidate + candidate], dtype=np.float64))
    torch.mps.synchronize()

    # The coarse price step sends both the raw-touch grid and the offset TWEL
    # reducer to the same tick. Exact Rust nevertheless keeps them as separate
    # orders, executing the grid first by order-type ID.
    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_qty = 5.045
    reducer_qty = entry_qty - grid_qty
    close_price = 100.0 if side == "long" else 101.0
    balance_after_entry = 1_000.0 - entry_qty * entry_price * 0.001
    grid_pnl = grid_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_grid = (
        balance_after_entry + grid_pnl - grid_qty * close_price * 0.001
    )
    reducer_pnl = reducer_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_close = (
        balance_after_grid
        + reducer_pnl
        - reducer_qty * close_price * 0.001
    )
    expected_volume = entry_qty * entry_price / balance_after_entry
    expected_volume += grid_qty * close_price / balance_after_grid
    expected_volume += reducer_qty * close_price / balance_after_close
    assert output["balance"].item() == pytest.approx(
        balance_after_close, abs=2.0e-4
    )
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


def _tm_multicoin_off_tick_reducer_case(
    side, *, maker_fee, close_qty_pct=1.0, close_threshold_we=0.0
):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    generation_touch = 100.4 if side == "long" else 100.6
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.0, 101.5, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 98.0, 99.5, 100.0]
    hlcvs[:, 0, 2] = [100.0, 100.0, 100.0, generation_touch, 100.0, 100.0]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, maker_fee)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "close_qty_pct": close_qty_pct,
            "close_threshold_base_pct": 0.0,
            "close_threshold_we_weight": close_threshold_we,
            "close_retracement_base_pct": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    runner = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    )
    return runner, candidate, market


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_unstuck_reserves_passive_grid_like_equal_wel():
    runner, wel_candidate, _ = _tm_multicoin_off_tick_reducer_case(
        "long", maker_fee=0.0
    )
    values = dict(
        zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, wel_candidate)
    )
    values.update(
        {
            "wel_enforcer_enabled": 0.0,
            "unstuck_enabled": 1.0,
            "unstuck_ema_gating_enabled": 0.0,
            "unstuck_close_pct": 0.5102,
            "unstuck_loss_allowance_pct": 0.2,
            "unstuck_threshold": 0.5,
        }
    )
    unstuck_candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(
        np.asarray([wel_candidate, unstuck_candidate], dtype=np.float64)
    )
    torch.mps.synchronize()

    # These candidates request the same touch and finalized reducer quantity.
    # The ordinary passive grid must therefore be generated from the same
    # reducer-reserved position size for WEL and unstuck.
    assert output["psize"][1].item() == pytest.approx(
        output["psize"][0].item(), abs=2.0e-4
    )
    assert output["balance"][1].item() == pytest.approx(
        output["balance"][0].item(), abs=2.0e-4
    )
    assert output["day_volume"][1].sum().item() == pytest.approx(
        output["day_volume"][0].sum().item(), rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_off_tick_grid_precedes_reducer_for_volume(side):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side, maker_fee=0.001
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    # Multicoin's shared total-entry headroom floors the short by one step.
    entry_qty = 10.101 if side == "long" else 9.9
    grid_price = 100.0 if side == "long" else 101.0
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_price = 101.0 if side == "long" else 100.0
    reducer_qty = entry_qty - grid_qty
    balance = 1_000.0 - entry_qty * entry_price * market.maker_fee
    expected_volume = entry_qty * entry_price / balance
    grid_pnl = grid_qty * (
        grid_price - entry_price if side == "long" else entry_price - grid_price
    )
    balance += grid_pnl - grid_qty * grid_price * market.maker_fee
    expected_volume += grid_qty * grid_price / balance
    reducer_pnl = reducer_qty * (
        reducer_price - entry_price
        if side == "long"
        else entry_price - reducer_price
    )
    balance += reducer_pnl - reducer_qty * reducer_price * market.maker_fee
    expected_volume += reducer_qty * reducer_price / balance

    assert output["balance"].item() == pytest.approx(balance, abs=3.0e-4)
    assert output["profit_sum"].item() == pytest.approx(
        grid_pnl + reducer_pnl, abs=3.0e-4
    )
    assert output["loss_sum"].item() == 0.0
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_consumes_one_recursive_close_slot(side):
    runner, candidate, _ = _tm_multicoin_off_tick_reducer_case(
        side,
        maker_fee=0.0,
        close_qty_pct=0.0001,
        close_threshold_we=1.0e-6,
    )
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    expected_size = 4.052 if side == "long" else 4.451
    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(expected_size, abs=2.0e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_global_position_exposure_repair(side):
    runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=10
    )
    repaired = list(baseline)
    repaired[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")
    ] = 1.0
    repaired[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert 0.0 < sizes[1] < sizes[0]
    assert (
        output["day_volume"][1].sum().item()
        > output["day_volume"][0].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_total_exposure_repair_policy(side):
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[0, 24] = 0.1
    runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        overrides,
        count=10,
        closes=(100.0, 100.0),
    )
    overweight = list(baseline)
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_entry_gate_enabled"
        )
    ] = 0.0
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        )
    ] = 0.5
    overweight[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    portfolio = list(overweight)
    portfolio[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    torch.mps.synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_blocks_lossy_total_exposure_repair(side):
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[0, 24] = 0.1
    runner_kwargs = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "coin_overrides": overrides,
        "count": 10,
        "closes": (100.0, 100.0),
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.1, **runner_kwargs
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    assert gated["open_positions"].item() == pytest.approx(2.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_blocks_fee_only_ordinary_close(side):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
        for _ in range(2)
    ]
    runner_kwargs = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "count": 10,
        "closes": (100.0, 100.0),
        "markets": markets,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.1, **runner_kwargs
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 0.0,
            "close_threshold_we_weight": 0.0,
            "entry_cooldown_minutes": 100.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated[size_key].item() > 0.0
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_preserves_profitable_close_after_repair(side):
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[0, 24] = 0.1
    count = 10
    blocked_highs = np.full((count, 2), 100.5)
    blocked_lows = np.full((count, 2), 99.5)
    recovery_highs = blocked_highs.copy()
    recovery_lows = blocked_lows.copy()
    if side == "long":
        recovery_highs[7:, :] = 102.0
    else:
        recovery_lows[7:, :] = 98.0
    common = {
        "strategy_kind": "trailing_martingale",
        "side": side,
        "coin_overrides": overrides,
        "count": count,
        "closes": (100.0, 100.0),
        "max_realized_loss_pct": 0.1,
    }
    blocked_runner, candidate = _multicoin_exposure_fixture(
        highs=blocked_highs, lows=blocked_lows, **common
    )
    recovery_runner, _ = _multicoin_exposure_fixture(
        highs=recovery_highs, lows=recovery_lows, **common
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 1.0,
            "close_threshold_base_pct": 0.01,
            "close_threshold_we_weight": 0.0,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    blocked = blocked_runner.run(np.asarray([candidate], dtype=np.float64))
    recovered = recovery_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert blocked[size_key].item() > 0.0
    assert recovered[size_key].item() == pytest.approx(0.0)
    assert recovered["balance"].item() > blocked["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_scans_past_blocked_recursive_rung(side):
    count = 10
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        highs[5:, 0] = 102.0
    else:
        lows[5:, 0] = 98.0
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=(100.0, 100.0),
        highs=highs,
        lows=lows,
        max_realized_loss_pct=0.1,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "close_qty_pct": 0.25,
            "close_threshold_base_pct": 0.015,
            "close_threshold_we_weight": -0.025,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # The first immutable recursive rungs are below break-even, but later
    # rungs become profitable as their simulated exposure falls. Exact Rust
    # filters the early rungs independently and retains those later closes.
    assert output[size_key].item() < 5.0
    assert output["balance"].item() > 1_000.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_loss_gate_does_not_reallocate_blocked_twel(side):
    count = 10
    closes = np.full((count, 2), 100.0)
    if side == "long":
        closes[5:, 1] = 102.0
    else:
        closes[5:, 1] = 98.0
    highs = closes * 1.01
    lows = closes * 0.99
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
        for _ in range(2)
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[0, 24] = 0.6
    overrides[1, 24] = 0.4
    runner, candidate = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        coin_overrides=overrides,
        count=count,
        closes=closes,
        highs=highs,
        lows=lows,
        markets=markets,
        max_realized_loss_pct=0.1,
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    baseline = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    repaired_values = dict(values, twel_enforcer_enabled=1.0)
    repaired = [
        repaired_values[key]
        for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]

    output = runner.run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    # Coin zero wins TWEL's equal-adverse tie and alone reaches the target, but
    # its fee-only reducer is loss-gated. Exact Rust does not then reallocate
    # that action to profitable coin one.
    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 2.0])
    assert output[size_key][1].item() == pytest.approx(
        output[size_key][0].item(), abs=2.0e-4
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_total_exposure_repair_policy(side):
    overrides = np.full((2, 19), np.nan, dtype=np.float32)
    overrides[0, 11] = 0.1
    count = 70
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        lows[62, :] = 98.0
    else:
        highs[62, :] = 102.0
    runner, baseline = _multicoin_exposure_fixture(
        "ema_anchor",
        side,
        overrides,
        count=count,
        closes=(100.0, 100.0),
        highs=highs,
        lows=lows,
    )
    overweight = list(baseline)
    overweight[EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")] = 0.01
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_entry_gate_enabled")
    ] = 0.0
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] = 0.5
    overweight[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 1.0
    portfolio = list(overweight)
    portfolio[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([overweight, portfolio], dtype=np.float64))
    torch.mps.synchronize()

    assert output["open_positions"].cpu().tolist() == pytest.approx([2.0, 1.0])
    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = (output[pprice_key] / output["balance"]).cpu()
    assert (total_twe < 0.5).all()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_lossy_total_exposure_repair(side):
    overrides = np.full((2, 19), np.nan, dtype=np.float32)
    overrides[0, 11] = 0.1
    count = 70
    closes = np.full((count, 2), 100.0)
    highs = np.full((count, 2), 100.5)
    lows = np.full((count, 2), 99.5)
    if side == "long":
        lows[62, :] = 98.0
        closes[63:, :] = 98.0
        highs[63:, :] = 98.0
        lows[63:, :] = 97.9
    else:
        highs[62, :] = 102.0
        closes[63:, :] = 102.0
        highs[63:, :] = 102.1
        lows[63:, :] = 102.0

    runner_kwargs = {
        "strategy_kind": "ema_anchor",
        "side": side,
        "coin_overrides": overrides,
        "count": count,
        "closes": closes,
        "highs": highs,
        "lows": lows,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.1, **runner_kwargs
    )
    values = dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "offset": 0.01,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    candidate = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() < gated[size_key].item()
    assert gated["open_positions"].item() == pytest.approx(2.0)
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_loss_gate_blocks_fee_only_ordinary_close(side):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.001)
        for _ in range(2)
    ]
    runner_kwargs = {
        "strategy_kind": "ema_anchor",
        "side": side,
        "count": 70,
        "closes": (100.0, 100.0),
        "markets": markets,
    }
    ungated_runner, candidate = _multicoin_exposure_fixture(
        max_realized_loss_pct=1.0, **runner_kwargs
    )
    gated_runner, _ = _multicoin_exposure_fixture(
        max_realized_loss_pct=0.1, **runner_kwargs
    )
    candidate[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0

    ungated = ungated_runner.run(np.asarray([candidate], dtype=np.float64))
    gated = gated_runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert ungated[size_key].item() == pytest.approx(0.0)
    assert gated[size_key].item() > 0.0
    assert gated["balance"].item() >= ungated["balance"].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_twel_ranking_clamps_delisted_market_price(side):
    count = 7
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # Coin zero delists after this close. Exact orchestration continues to use
    # that clamped market price for open-position repair ranking.
    hlcvs[3, 0, :3] = [111.0, 109.0, 110.0]
    hlcvs[4:, 0, :] = 0.0
    # The still-tradable coin is adverse for long and favorable for short.
    hlcvs[4:, 1, :3] = [91.0, 89.0, 90.0]
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            last_valid,
        )
        for last_valid in (3, count - 1)
    ]
    data = build_mps_multicoin_data(
        hlcvs, timestamps, runs, [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.75,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    repaired = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    baseline = list(repaired)
    baseline[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        runs[1], data, side=side
    ).run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    if side == "long":
        # The favorable delisted long ranks first; its reducer cannot fill on
        # invalid bars, so the live coin remains untouched.
        assert sizes[1] == pytest.approx(sizes[0])
    else:
        # The adverse delisted short ranks behind the favorable live short,
        # whose reducer remains reachable and therefore reduces total size.
        assert sizes[1] < sizes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_ema_multicoin_twel_ranking_clamps_delisted_market_price(side):
    count = 70
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, :, 0] = 101.0
    hlcvs[:, :, 1] = 99.0
    hlcvs[:, :, 2] = 100.0
    hlcvs[:, :, 3] = 100.0
    # Fill the wide-offset entries, then make coin zero favorable for long
    # and adverse for short before it delists. Coin one has the inverse rank.
    hlcvs[62, 0, :3] = [140.0, 70.0, 130.0]
    hlcvs[62, 1, :3] = [130.0, 70.0, 90.0]
    hlcvs[63:, 0, :] = 0.0
    hlcvs[63:, 1, :3] = [101.0, 89.0, 90.0]
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            last_valid,
        )
        for last_valid in (62, count - 1)
    ]
    data = build_mps_multicoin_data(
        hlcvs, timestamps, runs, [market, market]
    )
    _, row = _multicoin_exposure_fixture("ema_anchor", side, count=count)
    values = dict(zip(EMA_ANCHOR_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "offset": 0.2,
            "entry_cooldown_minutes": 100.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.75,
            "twel_enforcer_enabled": 1.0,
            "twel_enforcer_reduce_portfolio": 1.0,
        }
    )
    repaired = [values[key] for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
    baseline = list(repaired)
    baseline[
        EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_enabled")
    ] = 0.0
    output = MpsEmaAnchorMulticoinRunner(
        runs[1], data, side=side
    ).run(np.asarray([baseline, repaired], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    if side == "long":
        # The favorable delisted long ranks first; its reducer cannot fill on
        # invalid bars, so the live coin remains untouched.
        assert sizes[1] == pytest.approx(sizes[0])
    else:
        # The favorable live short ranks before the adverse delisted short.
        assert sizes[1] < sizes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_twel_repair_preserves_trailing_closes(side):
    highs = np.full((7, 2), 100.0)
    lows = np.full((7, 2), 100.0)
    highs[3, :] = 105.0
    lows[3, :] = 95.0
    highs[5, :] = 105.0
    lows[5, :] = 95.0
    if side == "long":
        highs[4, :] = 99.9
        lows[4, :] = 98.0
        closes = np.full((7, 2), 100.0)
        closes[4, :] = 99.0
    else:
        highs[4, :] = 102.0
        lows[4, :] = 100.1
        closes = np.full((7, 2), 100.0)
        closes[4, :] = 101.0
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=7,
        closes=closes,
        highs=highs,
        lows=lows,
    )
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_qty_pct")
    ] = 0.2
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_threshold_base_pct"
        )
    ] = 0.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_retracement_base_pct"
        )
    ] = 0.001
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_cooldown_minutes"
        )
    ] = 100.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_entry_gate_enabled"
        )
    ] = 0.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        )
    ] = 0.8
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_enabled"
        )
    ] = 1.0
    row[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_reduce_portfolio"
        )
    ] = 1.0

    output = runner.run(np.asarray([row], dtype=np.float64))
    torch.mps.synchronize()

    pprice_key = "pprice" if side == "long" else "short_pprice"
    total_twe = output[pprice_key][0].item() / output["balance"][0].item()
    # Two ordinary trailing closes remove 0.2 TWE in total and TWEL repair
    # independently removes another 0.2.  Suppressing one trailing close
    # leaves approximately 0.7 instead of 0.6.
    assert total_twe < 0.65


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_reducer_retains_reachable_post_repair_grid(side):
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=6
    )
    enabled = list(row)
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_enabled")
    ] = 1.0
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("wel_enforcer_threshold")
    ] = 0.5
    enabled[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("entry_cooldown_minutes")
    ] = 100.0
    reducer_only = list(enabled)
    reducer_only[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "close_retracement_base_pct"
        )
    ] = 0.001
    repaired_grid = list(enabled)
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_base_pct")
    ] = 0.0
    repaired_grid[
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index("close_threshold_we_weight")
    ] = 0.0
    output = runner.run(
        np.asarray([reducer_only, repaired_grid], dtype=np.float64)
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    volumes = output["day_volume"].sum(dim=1).cpu().numpy()
    assert sizes[0] > 0.0
    assert sizes[1] == pytest.approx(0.0)
    assert volumes[1] > volumes[0]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_grid_can_fill_without_off_tick_reducer(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    generation_touch = 100.4 if side == "long" else 100.6
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.0, 100.5, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 98.0, 100.5, 100.0]
    hlcvs[:, 0, 2] = [100.0, 100.0, 100.0, generation_touch, 100.0, 100.0]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, 1.0, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_ema_dist": 0.01,
            "entry_cooldown_minutes": 100.0,
            "gate_initial": 1.0,
            "n_positions": 1.0,
            "close_threshold_base_pct": 0.0,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert output[size_key].item() == pytest.approx(5.0, abs=0.1)
    assert 1.0 < output["day_volume"].sum().item() < 2.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_small_excess_reducer_crosses_strict_target(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 101.0, 101.0, 101.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 99.0, 99.0, 99.0]
    hlcvs[:, 0, 2] = 100.0
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(1.0e-8, 1.0, 1.0e-8, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
    target = 0.99999
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "n_positions": 1.0,
            "close_retracement_base_pct": 0.001,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": target,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    exposure = output[size_key].item() * 100.0 / output["balance"].item()
    assert output[size_key].item() > 0.0
    assert exposure < target


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_tm_multicoin_short_post_repair_grid_uses_negative_target_tick():
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.zeros((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 102.5, 103.0, 103.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 99.0, 101.5, 101.5]
    hlcvs[:, 0, 2] = [100.0, 100.0, 100.0, 102.0, 102.0, 102.0]
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    market = ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], [market, market]
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", "short", count=count
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "n_positions": 1.0,
            "close_threshold_base_pct": -0.01,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side="short", coin_overrides=overrides
    ).run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    # The reducer at the 102 bid fills, while Rust's 101 target grid order
    # remains below the next candle's 101.5 low and must not fill.
    assert output["short_psize"].item() == pytest.approx(4.999, abs=0.002)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_post_repair_grid_volume_uses_per_fill_balance(side):
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 0.0, 1.0, 0.0),
    ]
    runner, row = _multicoin_exposure_fixture(
        "trailing_martingale",
        side,
        count=6,
        markets=markets,
        closes=(100.0, 100.0),
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_cooldown_minutes": 100.0,
            "close_threshold_base_pct": 0.009,
            "close_threshold_we_weight": 0.0,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.5,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    first_reducer_qty = 2.501
    second_reducer_qty = 2.5
    grid_qty = 2.499
    reducer_price = 100.0
    grid_price = 100.9 if side == "long" else 99.1
    grid_pnl = grid_qty * abs(grid_price - 100.0)
    balance_0 = 1_000.0
    balance_1 = balance_0 + grid_pnl
    balance_2 = balance_1 + grid_pnl
    # Total-entry headroom clips the second exact-0.5-WEL entry by one step.
    expected_volume = 0.5 + 0.4999
    expected_volume += first_reducer_qty * reducer_price / balance_0
    expected_volume += grid_qty * grid_price / balance_1
    expected_volume += second_reducer_qty * reducer_price / balance_1
    expected_volume += grid_qty * grid_price / balance_2

    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_same_tick_twel_and_grid_use_separate_denominators(side):
    runner, candidate, market = _tm_multicoin_off_tick_reducer_case(
        side, maker_fee=0.001
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, candidate))
    values.update(
        {
            "wel_enforcer_enabled": 0.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.5,
            "twel_enforcer_enabled": 1.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    output = runner.run(np.asarray([candidate], dtype=np.float64))
    torch.mps.synchronize()

    entry_price = 99.0 if side == "long" else 101.0
    entry_qty = 10.101 if side == "long" else 9.901
    grid_qty = 5.045 if side == "long" else 4.945
    reducer_qty = entry_qty - grid_qty
    close_price = 100.0 if side == "long" else 101.0
    balance_after_entry = (
        1_000.0 - entry_qty * entry_price * market.maker_fee
    )
    expected_volume = entry_qty * entry_price / balance_after_entry
    grid_pnl = grid_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_grid = (
        balance_after_entry
        + grid_pnl
        - grid_qty * close_price * market.maker_fee
    )
    expected_volume += grid_qty * close_price / balance_after_grid
    reducer_pnl = reducer_qty * (
        close_price - entry_price
        if side == "long"
        else entry_price - close_price
    )
    balance_after_close = (
        balance_after_grid
        + reducer_pnl
        - reducer_qty * close_price * market.maker_fee
    )
    expected_volume += reducer_qty * close_price / balance_after_close
    assert output["balance"].item() == pytest.approx(
        balance_after_close, abs=3.0e-4
    )
    assert output["day_volume"].sum().item() == pytest.approx(
        expected_volume, rel=2.0e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_finalized_reducer_tie_keeps_nearer_wel(side):
    count = 6
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    hlcvs = np.empty((count, 2, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = [100.0, 100.0, 100.0, 105.0, 100.0, 100.0]
    hlcvs[:, 0, 1] = [100.0, 100.0, 100.0, 95.0, 100.0, 100.0]
    hlcvs[:, 0, 2] = 100.0
    hlcvs[:, 0, 3] = 200.0
    hlcvs[:, 1, 0] = 121.0
    hlcvs[:, 1, 1] = 119.0
    hlcvs[:, 1, 2] = 120.0
    hlcvs[:, 1, 3] = 100.0
    markets = [
        ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 9.0, 0.0, 1.0, 0.0),
    ]
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_multicoin_data(
        hlcvs, timestamps, [run, run], markets
    )
    _, row = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=count
    )
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(
        {
            "entry_initial_qty_pct": 1.0,
            "entry_threshold_base_pct": 10.0,
            "entry_retracement_base_pct": 0.0,
            "entry_cooldown_minutes": 100.0,
            "close_threshold_base_pct": 10.0,
            "close_retracement_base_pct": 0.0,
            "n_positions": 1.0,
            "twel_entry_gate_enabled": 0.0,
            "twel_enforcer_threshold": 0.05,
            "wel_enforcer_enabled": 1.0,
            "wel_enforcer_threshold": 0.1,
            "twel_enforcer_enabled": 1.0,
        }
    )
    candidate = [
        values[key] for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    twel_only_values = dict(values)
    twel_only_values["wel_enforcer_enabled"] = 0.0
    twel_only = [
        twel_only_values[key]
        for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    ]
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[1, 24] = 0.0
    output = MpsTrailingMartingaleMulticoinRunner(
        run, data, side=side, coin_overrides=overrides
    ).run(np.asarray([candidate, twel_only], dtype=np.float64))
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    sizes = output[size_key].cpu().numpy()
    assert sizes[0] > 9.0
    assert sizes[1] == pytest.approx(0.0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_tm_multicoin_static_override_repairs_only_selected_symbol(side):
    overrides = np.full((2, 34), np.nan, dtype=np.float32)
    overrides[0, 26] = 1.0
    overrides[0, 27] = 0.5
    baseline_runner, baseline = _multicoin_exposure_fixture(
        "trailing_martingale", side, count=10
    )
    override_runner, overridden = _multicoin_exposure_fixture(
        "trailing_martingale", side, overrides, count=10
    )
    baseline_output = baseline_runner.run(
        np.asarray([baseline], dtype=np.float64)
    )
    override_output = override_runner.run(
        np.asarray([overridden], dtype=np.float64)
    )
    torch.mps.synchronize()

    size_key = "psize" if side == "long" else "short_psize"
    assert (
        0.0
        < override_output[size_key].item()
        < baseline_output[size_key].item()
    )
    assert (
        override_output["day_volume"].sum().item()
        > baseline_output["day_volume"].sum().item()
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_single_coin_min_effective_cost_filter_blocks_only_flat_entries(
    strategy_kind, side
):
    count = 8
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 97.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 200.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    ema_row = [
        0.1,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    ema_row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    row = _tm_single_row() if strategy_kind == "trailing_martingale" else ema_row
    runner_cls = (
        MpsTrailingMartingaleRunner
        if strategy_kind == "trailing_martingale"
        else MpsEmaAnchorRunner
    )
    common = {
        "long_enabled": side == "long",
        "short_enabled": side == "short",
        "hedge_mode": True,
    }

    promoted = runner_cls(market, run, data, **common).run(
        np.array([row + row], dtype=np.float64)
    )
    filtered_runner = runner_cls(
        market,
        run,
        data,
        filter_by_min_effective_cost=True,
        **common,
    )
    filtered = filtered_runner.run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert filtered_runner.settings.cpu()[12].item() == 1.0
    assert promoted["day_has_fill"].sum().item() > 0
    assert filtered["day_has_fill"].sum().item() == 0
    assert filtered["psize"].item() == 0.0
    assert filtered["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_min_effective_cost_uses_downward_projected_cost_bound():
    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0e-9, 0.01, 1.0e-9, 10.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    base_qty_pct = 0.199999999
    row = [
        base_qty_pct,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
    guaranteed_balance_lower = run.starting_balance * run.liquidation_threshold
    rounded_projection = np.float32(guaranteed_balance_lower) * np.float32(
        base_qty_pct
    )

    assert guaranteed_balance_lower * base_qty_pct < 10.0
    assert rounded_projection >= np.float32(data["max_effective_min_cost"])

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        filter_by_min_effective_cost=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() == 0
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_mps_one_way_min_cost_eligibility_precedes_distance_arbitration(
    strategy_kind,
):
    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 97.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 4.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    if strategy_kind == "trailing_martingale":
        long_row = _tm_single_row(initial_ema_dist=0.02)
        short_row = _tm_single_row(initial_ema_dist=0.01)
        short_row[6] = 0.01
        runner_cls = MpsTrailingMartingaleRunner
    else:
        long_row = [
            0.1,
            2.0,
            3.0,
            0.0,
            0.02,
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            0.0,
            1.0,
        ]
        long_row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()
        short_row = list(long_row)
        short_row[0] = 0.01
        short_row[4] = 0.01
        runner_cls = MpsEmaAnchorRunner

    output = runner_cls(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=False,
        filter_by_min_effective_cost=True,
    ).run(np.array([long_row + short_row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert output["psize"].item() > 0.0
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_min_effective_cost_filter_keeps_managing_an_open_position(side):
    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if side == "long":
        low[3] = 98.0
        high[4] = 102.0
    else:
        high[3] = 102.0
        low[4] = 98.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    # Leave enough headroom above the liquidation-floor projection to open the
    # position; this test targets management after opening, not equality.
    market = ProxyMarket(0.001, 0.01, 0.001, 4.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = [
        0.1,
        2.0,
        3.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]
    row += _single_coin_exposure_fields() + _tm_twel_enforcer_fields()

    output = MpsEmaAnchorRunner(
        market,
        run,
        data,
        long_enabled=side == "long",
        short_enabled=side == "short",
        hedge_mode=True,
        filter_by_min_effective_cost=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert output["gap_hist"].sum().item() >= 1
    assert output["psize"].item() == 0.0
    assert output["short_psize"].item() == 0.0
    gap_metrics = _fill_gap_metrics(
        {
            key: output[key].cpu()
            for key in (
                "gap_hist",
                "first_fill_ts",
                "last_fill_ts",
                "first_eq_ts",
                "last_eq_ts",
            )
        },
        run,
    )
    assert all(
        torch.isfinite(value).all().item()
        for value in gap_metrics.values()
    )
    assert gap_metrics["fills_gap_p99_hours"].item() >= gap_metrics[
        "fills_gap_p95_hours"
    ].item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_mps_trailing_martingale_multicoin_sizes_raw_touch_close_before_price_finalization(
    side,
):
    count = 6
    coin_count = 2
    hlcvs = np.empty((count, coin_count, 4), dtype=np.float64)
    hlcvs[:, 0, 0] = 100.004
    hlcvs[:, 0, 1] = 100.004
    if side == "long":
        hlcvs[3, 0, 1] = 99.0
        hlcvs[4, 0, 0] = 101.0
    else:
        hlcvs[3, 0, 0] = 101.0
        hlcvs[4, 0, 1] = 99.0
    hlcvs[:, 0, 2] = 100.004
    hlcvs[:, 0, 3] = 100.0
    hlcvs[:, 1, 0:3] = 200.0
    hlcvs[:, 1, 3] = 1.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = [
        ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0),
        ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0),
    ]
    runs = [
        ProxyRun(
            1_000.0,
            1,
            1,
            int(timestamps[0]),
            int(timestamps[0]),
            int(timestamps[0]),
            60_000,
            0.05,
            0,
            count - 1,
        )
        for _ in range(coin_count)
    ]
    data = build_mps_multicoin_data(hlcvs, timestamps, runs, markets)
    row = _tm_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.01
    row[16] = 0.199
    row[17] = -2.0
    row[20] = 0.0
    row.extend(
        [
            60.0,  # forager volume span
            60.0,  # forager volatility span
            0.0,  # volume drop
            1.0,  # volume score weight
            0.0,  # EMA readiness score weight
            0.0,  # volatility score weight
            1.0,  # n_positions
        ]
    )
    row.extend(_single_coin_exposure_fields())
    row.extend(_tm_wel_enforcer_fields())
    row.append(0.0)  # TWEL enforcer disabled
    row.append(0.0)  # TWEL reduce_overweight policy
    row.extend(_UNSTUCK_DISABLED_VALUES.values())

    output = MpsTrailingMartingaleMulticoinRunner(
        runs[0], data, side=side
    ).run(np.array([row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust sizes against the raw 100.004 touch before finalizing the
    # executable close to 100.00. The raw minimum is 0.100; recomputing it
    # after finalization would yield 0.101 and leave the wrong remainder.
    position_key = "psize" if side == "long" else "short_psize"
    assert output[position_key].item() == pytest.approx(0.9, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize(
    ("long_enabled", "short_enabled"),
    [(True, False), (False, True), (True, True)],
)
def test_mps_trailing_martingale_shader_contract_and_directional_smoke(
    long_enabled, short_enabled
):
    import passivbot_rust
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    source = passivbot_rust.mps_trailing_martingale_source_py()
    assert "kernel void passivbot_trailing_martingale" in source
    assert "constant int SIDE_PARAMS = 51" in source
    assert "s.allowed_wel" in source
    assert "s.entry_cap" in source
    assert "min_since_open" in source
    assert "max_since_min" in source
    assert "max_since_open" in source
    assert "min_since_max" in source
    assert "s.entry_retracement_base > 0.0f" in source
    assert "s.close_retracement_base > 0.0f" in source
    assert "entry_gen_balance" in source
    assert "close_gen_balance" in source
    assert "merge_reducer" not in source
    assert "finalized_reducer_qty" in source
    assert "finalized_reducer_qty_with_ordinary" in source
    assert "reducer_candidate_preferred" in source
    assert "strict distance to the executable" in source
    assert "for (int rung = 0; rung < 500; ++rung)" in source
    assert "ladder_side, ladder_balance" in source
    assert "recursive_close_groups" in source
    assert "realized_loss_proxy_allows_close" in source
    assert "const float max_realized_loss_pct = settings[14]" in source
    assert "const bool loss_gate_enabled = max_realized_loss_pct < 1.0f" in source
    assert "const float taker_fee = settings[15]" in source
    assert "const float market_order_slippage_pct = fmax(settings[16], 0.0f)" in source
    assert "const bool long_hsl_panic_market = settings[17] > 0.5f" in source
    assert "const bool short_hsl_panic_market = settings[18] > 0.5f" in source
    assert "market_panic ? taker_fee : maker_fee" in source
    assert "the proxy uses a zero-loss envelope" in source
    assert "const bool filter_by_min_effective_cost" in source
    assert "passes_min_effective_cost" in source
    assert "projected_cost_lower" in source
    assert "float guaranteed_balance_lower" in source
    assert "accumulate_min_cost_balance_error" not in source
    assert "cooldown_min != 0.0f" in source
    assert "int entry_touch = is_long ? touch_down_ticks : touch_up_ticks" in source
    assert "int raw_reentry_ticks = reentry_target_is_touch" in source
    assert "int cticks = touch_controls ? touch_nearest_ticks : target_ticks" in source
    assert "entry_raw_touch" not in source
    assert "close_raw_touch" not in source
    assert "nextafter(" not in source

    count = 8
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 102.0, 103.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 98.0, 97.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row()

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=long_enabled,
        short_enabled=short_enabled,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["day_has_fill"].sum().item() > 0
    assert (output["psize"].item() > 0.0) is long_enabled
    assert (output["short_psize"].item() > 0.0) is short_enabled


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("is_long", [True, False])
def test_mps_trailing_martingale_fills_recursive_entry_ladder(is_long):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if is_long:
        low[2] = 99.99  # Fill the initial entry generated on the prior bar.
        low[3] = 90.0  # Strictly cross several pre-generated recursive rungs.
    else:
        high[2] = 100.01
        high[3] = 110.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[4] = 1.5  # entry double-down factor
    row[6] = 0.05  # initial entry uses 5% of the exposure budget
    row[11] = 0.0  # recursive entry mode
    row[16] = 1.0  # keep trailing closes unreachable in this fixture

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=is_long,
        short_enabled=not is_long,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position = output["psize" if is_long else "short_psize"].item()
    # Initial qty is 0.5 and the first reentry is 0.75. Crossing more than
    # those two orders proves the recursive ladder, not merely one reentry,
    # was filled in the same candle.
    assert position > 1.25


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_fills_recursive_entry_ladders_in_hedge_mode():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[2], high[2] = 99.99, 100.01
    low[3], high[3] = 90.0, 110.0
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[4], row[6], row[11], row[16] = 1.5, 0.05, 0.0, 1.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 1.25
    assert output["short_psize"].item() > 1.25


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
@pytest.mark.parametrize("is_long", [True, False])
@pytest.mark.parametrize("close_threshold_we", [-0.005, 0.005])
def test_mps_trailing_martingale_fills_sorted_recursive_close_grid(
    is_long, close_threshold_we
):
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 7
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    if is_long:
        low[3] = 99.0  # Fill the initial entry generated on the prior bar.
        high[4] = 100.85 if close_threshold_we > 0.0 else 100.6
    else:
        high[3] = 101.0
        low[4] = 99.15 if close_threshold_we > 0.0 else 99.4
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[6] = 0.1  # Open a 1-unit position.
    row[7] = 0.01  # Exposure cap prevents reentries in this fixture.
    row[15] = 0.05  # Two recursive close rungs at the standard lower bound.
    row[16] = 0.005 if close_threshold_we > 0.0 else 0.01
    row[17] = close_threshold_we * 10.0
    row[20] = 0.0  # Recursive close mode.

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=is_long,
        short_enabled=not is_long,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    position = output["psize" if is_long else "short_psize"].item()
    # The candle strictly crosses one of two sorted 0.5-unit closes. With
    # positive WE weight, the generated grid must be reversed before filling;
    # its first generated (farthest) order is deliberately not crossed.
    assert position == pytest.approx(0.5, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_fills_recursive_close_grids_in_hedge_mode():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.full(count, 100.0)
    low[3], high[3] = 99.0, 101.0
    low[4], high[4] = 99.15, 100.85
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[6], row[7], row[15] = 1.0, 1.0, 0.2
    row[16], row[17], row[20] = 0.005, 0.005, 0.0

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=True,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() == pytest.approx(4.0, abs=1e-6)
    assert output["short_psize"].item() == pytest.approx(4.0, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_preserves_tick_aligned_computed_target():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.99, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_quantizes_non_aligned_entry_down():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.006)
    high = np.full(count, 100.006)
    low = np.array([100.006, 100.006, 100.006, 100.005, 100.006])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust selects the raw 100.006 bid, then finalize_next_entry floors
    # the executable long order to 100.00. The 100.005 low must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_quantizes_non_aligned_short_entry_up():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.006, dtype=np.float64)
    high = np.array([100.006, 100.006, 100.006, 100.007, 100.006])
    low = close.copy()
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # finalize_next_entry ceils the selected 100.006 short touch to 100.01.
    # A 100.007 high remains below the executable order and must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_entry_quantizes_before_strict_fill():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.000003, dtype=np.float64)
    high = close.copy()
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[2]) == np.float32(low[3])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust first chooses the raw touch, then floors the executable long
    # entry to 100.00. A low above 100.00 must not fill it.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_initial_gate_chooses_tick_before_float32_collapse():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.000003, dtype=np.float64)
    high = close.copy()
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[2]) == np.float32(low[3])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=1.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust's min(raw bid, EMA target) chooses the 100.00 tick. The
    # following 100.000001 low must not fill it. A float32 pre-comparison
    # incorrectly retained the raw 100.000003 bid and filled.
    assert not output["day_has_fill"].any().item()
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_quantizes_selected_raw_close_to_nearest_tick():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.000003, dtype=np.float64)
    high = np.array(
        [100.000003, 100.000003, 100.000003, 100.000003, 100.000002, 100.000003],
        dtype=np.float64,
    )
    low = np.array(
        [100.000003, 100.000003, 100.000003, 100.000001, 100.000003, 100.000003],
        dtype=np.float64,
    )
    assert np.float32(close[3]) == np.float32(high[4])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[16] = 0.0  # tick-aligned close target at the float32 position price
    row[20] = 0.0  # disable trailing-close touch override

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust chooses the raw 100.000003 ask over the 100.00 target, then
    # calc_closes_long quantizes it to the nearest tick, 100.00. The following
    # 100.000002 high must therefore close it.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_sizes_raw_touch_close_before_price_finalization():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.004, dtype=np.float64)
    high = np.array([100.004, 100.004, 100.004, 100.004, 101.0, 100.004])
    low = np.array([100.004, 100.004, 100.004, 99.0, 100.004, 100.004])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 10.0004, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.01
    row[16] = 0.199
    row[17] = -2.0  # keep later grid closes above this fixture's high
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # The raw 100.004 ask implies a 0.100 minimum close. Only after sizing does
    # Rust finalize the executable close to 100.00, whose minimum would be
    # 0.101 if it were incorrectly used for sizing. The 1.000 position must
    # therefore retain 0.900 after the close.
    assert output["psize"].item() == pytest.approx(0.9, abs=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_preserves_just_above_aligned_raw_touch_minimum():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 0.088, dtype=np.float64)
    high = np.array([0.088, 0.088, 0.088, 0.088, 0.1, 0.088])
    low = np.array([0.088, 0.088, 0.088, 0.07, 0.088, 0.088])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(1.0, 0.01, 0.0, 11.0, 1.0, 0.0)
    run = ProxyRun(
        200.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[15] = 0.05
    row[16] = 0.299
    row[17] = -3.0  # keep a possible second grid close above the high
    row[20] = 0.0

    rounded_only_data = dict(data)
    rounded_only_data["touch_min_qty_relation"] = torch.zeros_like(
        data["touch_min_qty_relation"]
    )
    rounded_only = MpsTrailingMartingaleRunner(
        market,
        run,
        rounded_only_data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()
    assert rounded_only["psize"].item() == 125.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust's raw-touch minimum is 125.00000000000001. A nominal 125-unit
    # remainder is therefore below the effective minimum, so the 250-unit
    # position must be closed in full rather than leaving half open.
    assert output["psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_quantizes_selected_short_close_to_nearest_tick():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 6
    close = np.full(count, 100.004, dtype=np.float64)
    high = np.array([100.004, 100.004, 100.004, 101.0, 100.004, 100.004])
    low = np.array([100.004, 100.004, 100.004, 100.004, 100.001, 100.004])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(gate_initial=0.0, gate_reentry=0.0)
    row[16] = 0.0
    row[20] = 0.0

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=False, short_enabled=True
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # The short opens at the ceiled 100.01 entry. Rust then selects the raw
    # 100.004 bid for its close and rounds it to 100.00. A 100.001 low is not
    # strictly below that executable close, so the position must remain open.
    assert output["short_psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_one_way_arbitrates_initial_entry():
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 5
    close = np.full(count, 100.0)
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 98.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0002)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row()

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=True,
        hedge_mode=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0
    assert output["short_psize"].item() == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_entry_cap_uses_rust_nearest_step_rounding():
    """Guard the one-quantity divergence that can split a later trailing path."""

    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 8
    close = np.full(count, 100.0)
    high = np.full(count, 100.0)
    low = np.array([100.0, 100.0, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0])
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[6] = 2.0  # Oversize the initial order so the exposure cap crops it.
    row[7] = 0.5  # Keep any subsequent reentry far below the fixture market.
    row[16] = 0.5  # Keep the close above the fixture market.
    row[20] = 0.0
    row[24] = 0.24036  # Exact cap quantity is 2.4036 at price 100.

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Rust finalization rounds the cap to the nearest quantity step. Flooring
    # this to 2.403 was enough to change a later close/reentry decision.
    assert output["psize"].item() == pytest.approx(2.404, abs=1.0e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_touch_close_preserves_raw_price():
    """Match Rust finalization when an off-tick market touch controls a close."""

    from optimization.gpu.mps_kernel import MpsTrailingMartingaleRunner

    count = 9
    close = np.array(
        [
            100.0,
            100.0,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
            100.004,
        ]
    )
    high = np.array(
        [
            100.0,
            100.0,
            100.004,
            100.005,
            100.005,
            100.005,
            100.005,
            100.005,
            100.004,
        ]
    )
    low = np.array(
        [
            100.0,
            100.0,
            99.0,
            100.003,
            100.003,
            100.003,
            100.004,
            100.004,
            100.004,
        ]
    )
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(0.001, 0.01, 0.001, 5.0, 1.0, 0.0)
    run = ProxyRun(
        1_000.0,
        1,
        1,
        int(timestamps[0]),
        int(timestamps[0]),
        int(timestamps[0]),
        60_000,
        0.05,
        0,
        count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    row = _tm_single_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
    row[7] = 0.5  # Prevent another entry in this fixture.
    row[16] = 0.0  # Use the current touch after the close trail retraces.
    row[20] = 0.000001

    output = MpsTrailingMartingaleRunner(
        market,
        run,
        data,
        long_enabled=True,
        short_enabled=False,
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust keeps the raw 100.004 ask, which fills at the next 100.005
    # high. Rounding the touch up to 100.01 would leave the position open.
    assert output["psize"].item() == 0.0
