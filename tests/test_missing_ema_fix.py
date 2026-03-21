"""Tests for MissingEma fix: EMA paths and error handling."""

import math
import time
import json
import pytest
import numpy as np

from candlestick_manager import (
    CandlestickManager,
    CANDLE_DTYPE,
    ONE_MIN_MS,
    _floor_minute,
)

ONE_HOUR_MS = 3_600_000


def _make_cm(tmp_path):
    return CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))


# ---------------------------------------------------------------------------
# get_latest_ema_close with leading gaps → finite result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_ema_close_with_leading_gaps(tmp_path, monkeypatch):
    """When leading candles are missing, EMA should still return a finite value."""
    cm = _make_cm(tmp_path)
    fixed_now_ms = 1_725_590_400_000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    span = 10
    # Create only the last 3 candles out of the 10 expected
    base = fixed_now_ms - span * ONE_MIN_MS
    arr = []
    for i in range(span - 3, span):
        ts = base + i * ONE_MIN_MS
        arr.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1.0))
    arr = np.array(arr, dtype=CANDLE_DTYPE)
    symbol = "SPARSE/USDT"
    cm._cache[symbol] = arr

    ema = await cm.get_latest_ema_close(symbol, float(span))
    assert math.isfinite(ema), f"EMA should be finite, got {ema}"
    assert ema > 0


@pytest.mark.asyncio
async def test_get_latest_ema_close_no_candles_returns_nan(tmp_path, monkeypatch):
    """When there are zero candles, EMA should return NaN (not crash)."""
    cm = _make_cm(tmp_path)
    fixed_now_ms = 1_725_590_400_000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    symbol = "EMPTY/USDT"
    cm._cache[symbol] = np.empty((0,), dtype=CANDLE_DTYPE)

    ema = await cm.get_latest_ema_close(symbol, 5.0)
    assert math.isnan(ema)


# ---------------------------------------------------------------------------
# MissingEma graceful handling in passivbot.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_ema_raises_from_snapshot(monkeypatch):
    """MissingEma in calc_ideal_orders_orchestrator_from_snapshot re-raises."""
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    class FakeBot:
        positions = {}
        balance = 1000.0
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = pb_mod.Passivbot._build_monitor_runtime_market_hints
        _build_monitor_runtime_unstuck_hints = pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def get_raw_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

        def get_hysteresis_snapped_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
    }

    def fake_compute(json_str):
        raise Exception("MissingEma { symbol_idx: 0 }")

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    with pytest.raises(Exception, match="MissingEma"):
        await method(bot, snapshot, return_snapshot=False)


@pytest.mark.asyncio
async def test_missing_ema_raises_from_snapshot_with_return(monkeypatch):
    """MissingEma with return_snapshot=True also re-raises."""
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    class FakeBot:
        positions = {}
        balance = 1000.0
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = pb_mod.Passivbot._build_monitor_runtime_market_hints
        _build_monitor_runtime_unstuck_hints = pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def get_raw_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

        def get_hysteresis_snapped_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
    }

    def fake_compute(json_str):
        raise Exception("MissingEma { symbol_idx: 0 }")

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    with pytest.raises(Exception, match="MissingEma"):
        await method(bot, snapshot, return_snapshot=True)


def _rust_bot_params(**overrides):
    params = {
        "close_grid_markup_end": 0.01,
        "close_grid_markup_start": 0.01,
        "close_grid_qty_pct": 1.0,
        "close_trailing_retracement_pct": 0.0,
        "close_trailing_grid_ratio": 0.0,
        "close_trailing_qty_pct": 0.0,
        "close_trailing_threshold_pct": 0.0,
        "entry_grid_double_down_factor": 1.0,
        "entry_grid_spacing_volatility_weight": 0.0,
        "entry_grid_spacing_we_weight": 0.0,
        "entry_grid_spacing_pct": 0.02,
        "entry_volatility_ema_span_hours": 0.0,
        "entry_initial_ema_dist": 0.0,
        "entry_initial_qty_pct": 0.1,
        "entry_trailing_double_down_factor": 0.0,
        "entry_trailing_retracement_pct": 0.0,
        "entry_trailing_retracement_we_weight": 0.0,
        "entry_trailing_retracement_volatility_weight": 0.0,
        "entry_trailing_grid_ratio": 0.0,
        "entry_trailing_threshold_pct": 0.0,
        "entry_trailing_threshold_we_weight": 0.0,
        "entry_trailing_threshold_volatility_weight": 0.0,
        "filter_volatility_ema_span": 10.0,
        "filter_volatility_drop_pct": 0.0,
        "filter_volume_ema_span": 10.0,
        "filter_volume_drop_pct": 0.0,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_threshold": 0.0,
        "risk_twel_enforcer_threshold": 0.0,
        "risk_we_excess_allowance_pct": 0.0,
        "unstuck_close_pct": 0.0,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.0,
        "unstuck_threshold": 0.0,
    }
    params.update(overrides)
    return params


def _make_orchestrator_payload(symbol, m1_close_pairs, m1_volume_pairs, m1_lr_pairs):
    trailing = {
        "min_since_open": 0.0,
        "max_since_min": 0.0,
        "max_since_open": 0.0,
        "min_since_max": 0.0,
    }
    return {
        "balance": 1000.0,
        "global": {
            "filter_by_min_effective_cost": False,
            "unstuck_allowance_long": 0.0,
            "unstuck_allowance_short": 0.0,
            "max_realized_loss_pct": 1.0,
            "realized_pnl_cumsum_max": 0.0,
            "realized_pnl_cumsum_last": 0.0,
            "sort_global": True,
            "global_bot_params": {
                "long": _rust_bot_params(),
                "short": _rust_bot_params(n_positions=0, total_wallet_exposure_limit=0.0),
            },
            "hedge_mode": True,
        },
        "symbols": [
            {
                "symbol_idx": 0,
                "order_book": {"bid": 30.0, "ask": 30.0},
                "exchange": {
                    "qty_step": 0.01,
                    "price_step": 0.01,
                    "min_qty": 0.0,
                    "min_cost": 0.0,
                    "c_mult": 1.0,
                },
                "tradable": True,
                "next_candle": None,
                "effective_min_cost": 1.0,
                "emas": {
                    "m1": {
                        "close": m1_close_pairs,
                        "log_range": m1_lr_pairs,
                        "volume": m1_volume_pairs,
                    },
                    "h1": {"close": [], "log_range": [], "volume": []},
                },
                "long": {
                    "mode": "normal",
                    "position": {"size": 0.0, "price": 0.0},
                    "trailing": trailing,
                    "bot_params": _rust_bot_params(),
                },
                "short": {
                    "mode": "manual",
                    "position": {"size": 0.0, "price": 0.0},
                    "trailing": trailing,
                    "bot_params": _rust_bot_params(
                        n_positions=0, total_wallet_exposure_limit=0.0
                    ),
                },
            }
        ],
        "peek_hints": None,
    }


class _BundleReproBot:
    def __init__(
        self,
        symbol,
        close_mode,
        close_value=100.0,
        h1_mode="value",
        h1_log_range_value=0.0015,
        entry_h1_span_hours=0.0,
        prev_close_ema=None,
        prev_age_ms=5_000,
    ):
        self.symbol = symbol
        self.PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
        self.positions = {
            symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0, "price": 0.0}}
        }
        self.close_mode = close_mode
        self.close_value = float(close_value)
        self.h1_mode = h1_mode
        self.h1_log_range_value = float(h1_log_range_value)
        self.entry_h1_span_hours = float(entry_h1_span_hours)
        self._orchestrator_close_ema_fallback_counts = {}
        now_ms = int(time.time() * 1000)
        if prev_close_ema is None:
            self._orchestrator_prev_close_ema = {}
        else:
            self._orchestrator_prev_close_ema = {
                symbol: {
                    float(span): (float(val), now_ms - int(prev_age_ms))
                    for span, val in prev_close_ema.items()
                }
            }

        class _CM:
            def __init__(self, outer):
                self.outer = outer

            async def get_latest_ema_close(self, symbol, span, max_age_ms=30_000):
                if self.outer.close_mode == "timeout":
                    raise TimeoutError("kucoinfutures GET ... RequestTimeout")
                if self.outer.close_mode == "nan":
                    return float("nan")
                return float(self.outer.close_value)

            async def get_latest_ema_quote_volume(self, symbol, span, max_age_ms=60_000):
                return 250000.0

            async def get_latest_ema_log_range(self, symbol, span, tf=None, max_age_ms=60_000):
                if tf == "1h":
                    if self.outer.h1_mode == "timeout":
                        raise TimeoutError("kucoinfutures GET ... RequestTimeout")
                    if self.outer.h1_mode == "nan":
                        return float("nan")
                    return float(self.outer.h1_log_range_value)
                return 0.0015

        self.cm = _CM(self)

    def _pb_mode_to_orchestrator_mode(self, mode: str) -> str:
        return (mode or "manual").strip().lower()

    def has_position(self, pside=None, symbol=None):
        if pside is None:
            return False
        return bool(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0))

    def bp(self, pside, key, symbol=None):
        if key == "ema_span_0":
            return 10.0
        if key == "ema_span_1":
            return 20.0
        if key == "entry_volatility_ema_span_hours":
            return self.entry_h1_span_hours
        return 0.0

    def bot_value(self, pside, key):
        if key == "filter_volume_ema_span":
            return 10.0
        if key == "filter_volatility_ema_span":
            return 10.0
        return 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["timeout", "nan"])
async def test_kucoin_avax_bundle_drop_reproduces_missing_ema_symbol_idx_0(close_mode):
    try:
        import passivbot as pb_mod
        import passivbot_rust as pbr
    except ImportError:
        pytest.skip("passivbot or passivbot_rust module not importable in test environment")

    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires real passivbot_rust extension")

    symbol = "AVAX/USDT:USDT"
    bot = _BundleReproBot(symbol, close_mode=close_mode)
    with pytest.raises(
        RuntimeError, match=r"missing required close EMA for AVAX/USDT:USDT"
    ):
        await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)

    payload = _make_orchestrator_payload(
        symbol,
        m1_close_pairs=[],
        m1_volume_pairs=[[10.0, 250000.0]],
        m1_lr_pairs=[[10.0, 0.0015]],
    )

    with pytest.raises(ValueError, match=r"orchestrator compute_ideal_orders failed: MissingEma \{ symbol_idx: 0 \}"):
        pbr.compute_ideal_orders_json(json.dumps(payload))


@pytest.mark.asyncio
async def test_kucoin_avax_close_ema_fallback_uses_previous_ema_not_price():
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    symbol = "AVAX/USDT:USDT"
    span0 = 10.0
    span1 = 20.0
    span2 = (span0 * span1) ** 0.5
    prev = {span0: 100.04, span1: 100.03, span2: 100.02}
    bot = _BundleReproBot(symbol, close_mode="timeout", close_value=110.37, prev_close_ema=prev)

    (
        m1_close_emas,
        _m1_volume_emas,
        _m1_log_range_emas,
        _h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)

    got = m1_close_emas[symbol]
    assert got[span0] == pytest.approx(prev[span0])
    assert got[span1] == pytest.approx(prev[span1])
    assert got[span2] == pytest.approx(prev[span2])
    assert got[span0] != pytest.approx(bot.close_value)
    assert got[span1] != pytest.approx(bot.close_value)
    assert got[span2] != pytest.approx(bot.close_value)
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span0)] == 1
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span1)] == 1
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span2)] == 1


@pytest.mark.asyncio
async def test_kucoin_avax_close_ema_fallback_count_resets_on_recovery():
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    symbol = "AVAX/USDT:USDT"
    span0 = 10.0
    span1 = 20.0
    span2 = (span0 * span1) ** 0.5
    prev = {span0: 90.0, span1: 91.0, span2: 92.0}
    bot = _BundleReproBot(symbol, close_mode="timeout", prev_close_ema=prev)

    await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span0)] == 1
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span1)] == 1
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span2)] == 1

    bot.close_mode = "value"
    bot.close_value = 105.5
    (
        m1_close_emas,
        _m1_volume_emas,
        _m1_log_range_emas,
        _h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)

    got = m1_close_emas[symbol]
    assert got[span0] == pytest.approx(105.5)
    assert got[span1] == pytest.approx(105.5)
    assert got[span2] == pytest.approx(105.5)
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span0)] == 0
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span1)] == 0
    assert bot._orchestrator_close_ema_fallback_counts[(symbol, span2)] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("h1_mode", ["timeout", "nan"])
async def test_required_h1_log_range_ema_raises_when_missing(h1_mode):
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    symbol = "AVAX/USDT:USDT"
    bot = _BundleReproBot(
        symbol,
        close_mode="value",
        h1_mode=h1_mode,
        entry_h1_span_hours=4.0,
    )
    with pytest.raises(RuntimeError, match=r"missing required h1_log_range EMA for AVAX/USDT:USDT"):
        await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)


@pytest.mark.asyncio
async def test_required_h1_log_range_ema_present_in_bundle():
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    symbol = "AVAX/USDT:USDT"
    bot = _BundleReproBot(
        symbol,
        close_mode="value",
        h1_mode="value",
        h1_log_range_value=0.0042,
        entry_h1_span_hours=4.0,
    )
    (
        _m1_close_emas,
        _m1_volume_emas,
        _m1_log_range_emas,
        h1_log_range_emas,
        _volumes_long,
        _log_ranges_long,
    ) = await pb_mod.Passivbot._load_orchestrator_ema_bundle(bot, [symbol], bot.PB_modes)

    assert h1_log_range_emas[symbol][4.0] == pytest.approx(0.0042)
