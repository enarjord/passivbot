import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.model import (
    ProxyMarket,
    ProxyRun,
    _build_hourly_log_range,
    _directional_touch_ticks,
    _positive_f64_words,
    _strict_fill_tick_boundaries,
    build_mps_data,
)


def test_directional_touch_ticks_preserve_alignment_and_round_non_aligned_prices():
    down, up = _directional_touch_ticks(
        np.array([100.0, 100.006, 0.1 + 0.2]), 0.01
    )

    assert down.tolist() == [10_000, 10_000, 30]
    assert up.tolist() == [10_000, 10_001, 30]


def test_positive_f64_words_preserve_strict_price_ordering():
    values = np.array([100.0, 100.000001, 100.000003], dtype=np.float64)
    high, low = _positive_f64_words(values)
    words = (high.view(np.uint32).astype(np.uint64) << np.uint64(32)) | low.view(
        np.uint32
    ).astype(np.uint64)

    assert np.all(words[:-1] < words[1:])
from optimization.gpu.mps_kernel import MpsEmaAnchorRunner


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
    assert "constant int SCALAR_COLS = 18" in source
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
    parameters = np.array([row + row, row + row], dtype=np.float64)

    output = MpsEmaAnchorRunner(market, run, data).run(parameters)
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (2,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0


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
    assert "constant int SIDE_PARAMS = 27" in source
    assert "min_since_open" in source
    assert "max_since_min" in source
    assert "max_since_open" in source
    assert "min_since_max" in source
    assert "s.entry_retracement_base > 0.0f" in source
    assert "s.close_retracement_base > 0.0f" in source
    assert "int entry_touch = is_long ? touch_down_ticks : touch_up_ticks" in source
    assert "int raw_reentry_ticks = reentry_target_is_touch" in source
    assert "long_side.entry_raw_touch" in source
    assert "short_side.close_raw_touch" in source
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
    row = _tm_row()

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
    row = _tm_row(initial_ema_dist=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_martingale_preserves_non_aligned_raw_touch():
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
    row = _tm_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust keeps the raw 100.006 bid. The following 100.005 low must
    # fill it; rounding the touch down to 100.00 would miss the fill.
    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_raw_touch_keeps_float64_strict_ordering():
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
    row = _tm_row(gate_initial=0.0, gate_reentry=0.0)

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Both prices collapse to 100.0 in float32, but exact Rust sees the low
    # strictly below the raw bid and fills. Metal compares the original f64
    # bit words for this decision.
    assert output["psize"].item() > 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_trailing_chooses_raw_close_before_float32_collapse():
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
    row = _tm_row(gate_initial=0.0, gate_reentry=0.0)
    row[16] = 0.0  # tick-aligned close target at the float32 position price
    row[20] = 0.0  # disable trailing-close touch override

    output = MpsTrailingMartingaleRunner(
        market, run, data, long_enabled=True, short_enabled=False
    ).run(np.array([row + row], dtype=np.float64))
    torch.mps.synchronize()

    # Exact Rust chooses the raw 100.000003 ask over the 100.00 target. The
    # following 100.000002 high must not close it. A float32 pre-comparison
    # incorrectly chose 100.00 and filled.
    assert output["psize"].item() > 0.0


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
    row = _tm_row()

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
    row = _tm_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
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
    row = _tm_row(initial_ema_dist=0.0, gate_initial=0.0, gate_reentry=0.0)
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
