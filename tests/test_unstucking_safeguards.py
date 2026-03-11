import json
import sys
import types

import pytest


def _make_mock_pbr():
    module = types.ModuleType("passivbot_rust")

    class _EquityHardStopRollingPeak:
        def __init__(self):
            self._peaks = []
            self._last_ts = None

        def reset(self):
            self._peaks = []
            self._last_ts = None

        def len(self):
            return len(self._peaks)

        def update(self, timestamp_ms, equity, lookback_ms):
            if self._last_ts is not None and timestamp_ms < self._last_ts:
                raise ValueError("timestamp_ms must be non-decreasing")
            self._last_ts = timestamp_ms
            while self._peaks and timestamp_ms - self._peaks[0][0] > lookback_ms:
                self._peaks.pop(0)
            while self._peaks and self._peaks[-1][1] <= equity:
                self._peaks.pop()
            self._peaks.append((timestamp_ms, equity))
            return float(self._peaks[0][1])

    module.EquityHardStopRollingPeak = _EquityHardStopRollingPeak

    class _EquityHardStopRuntime:
        def __init__(self):
            self._state_reset()

        def _state_reset(self):
            self._initialized = False
            self._red_latched = False
            self._tier = "green"
            self._drawdown_ema = 0.0
            self._peak_strategy_equity = 0.0
            self._rolling_peak = _EquityHardStopRollingPeak()
            self._last_rolling_peak = 0.0

        def reset(self):
            self._state_reset()

        def reset_state_keep_peak(self):
            self._initialized = False
            self._red_latched = False
            self._tier = "green"
            self._drawdown_ema = 0.0
            self._peak_strategy_equity = 0.0

        def initialized(self):
            return bool(self._initialized)

        def red_latched(self):
            return bool(self._red_latched)

        def tier(self):
            return str(self._tier)

        def drawdown_ema(self):
            return float(self._drawdown_ema)

        def peak_strategy_equity(self):
            return float(self._peak_strategy_equity)

        def rolling_peak_strategy_equity(self):
            return float(self._last_rolling_peak)

        def apply_sample(
            self,
            *,
            timestamp_ms,
            equity,
            peak_strategy_equity,
            sample_minutes,
            red_threshold,
            ema_span_minutes,
            tier_ratio_yellow,
            tier_ratio_orange,
        ):
            self._last_rolling_peak = float(peak_strategy_equity)
            span_samples = ema_span_minutes / sample_minutes
            alpha = 2.0 / (span_samples + 1.0)
            prev_tier = self._tier

            if not self._initialized:
                self._initialized = True
                self._peak_strategy_equity = float(peak_strategy_equity)
                self._drawdown_ema = 0.0
                self._tier = "red" if self._red_latched else "green"
                return {
                    "initialized": True,
                    "red_latched": bool(self._red_latched),
                    "peak_strategy_equity": float(self._peak_strategy_equity),
                    "rolling_peak_strategy_equity": float(self._last_rolling_peak),
                    "drawdown_ema": float(self._drawdown_ema),
                    "tier": self._tier,
                    "drawdown_raw": 0.0,
                    "drawdown_score": 0.0,
                    "changed": self._tier != prev_tier,
                    "span_samples": float(span_samples),
                    "alpha": float(alpha),
                }

            self._peak_strategy_equity = float(peak_strategy_equity)
            drawdown_raw = max(0.0, 1.0 - equity / max(self._peak_strategy_equity, 1e-12))
            self._drawdown_ema = alpha * drawdown_raw + (1.0 - alpha) * self._drawdown_ema
            drawdown_score = min(drawdown_raw, self._drawdown_ema)
            if self._red_latched or drawdown_score >= red_threshold:
                self._tier = "red"
                self._red_latched = True
            elif drawdown_score >= red_threshold * tier_ratio_orange:
                self._tier = "orange"
            elif drawdown_score >= red_threshold * tier_ratio_yellow:
                self._tier = "yellow"
            else:
                self._tier = "green"
            return {
                "initialized": True,
                "red_latched": bool(self._red_latched),
                "peak_strategy_equity": float(self._peak_strategy_equity),
                "rolling_peak_strategy_equity": float(self._last_rolling_peak),
                "drawdown_ema": float(self._drawdown_ema),
                "tier": self._tier,
                "drawdown_raw": float(drawdown_raw),
                "drawdown_score": float(drawdown_score),
                "changed": self._tier != prev_tier,
                "span_samples": float(span_samples),
                "alpha": float(alpha),
            }

    module.EquityHardStopRuntime = _EquityHardStopRuntime
    module.calc_auto_unstuck_allowance = (
        lambda balance, allowance_pct, max_pnl, last_pnl: allowance_pct * balance
    )
    module.calc_wallet_exposure = (
        lambda c_mult, balance, size, price: abs(size) * price / max(balance, 1e-6)
    )
    module.calc_min_entry_qty = lambda *args, **kwargs: 0.0
    module.calc_min_entry_qty_py = lambda *args, **kwargs: 0.0
    module.cost_to_qty = lambda *args, **kwargs: 0.0
    module.round_dn = lambda value, step: value
    module.round_up = lambda value, step: value
    module.round_dynamic = lambda value, sig: value
    module.calc_pnl_long = (
        lambda entry_price, close_price, qty, c_mult: (close_price - entry_price) * qty
    )
    module.calc_pnl_short = (
        lambda entry_price, close_price, qty, c_mult: (entry_price - close_price) * qty
    )
    module.calc_pprice_diff_int = lambda *args, **kwargs: 0.0
    module.calc_diff = lambda price, reference: price - reference
    module.calc_order_price_diff = lambda side, price, market: (
        (0.0 if not market else (1 - price / market))
        if str(side).lower() in ("buy", "long")
        else (0.0 if not market else (price / market - 1))
    )
    module.round_ = lambda value, step: value
    module.compute_ideal_orders_json = lambda *_args, **_kwargs: "{}"
    def _equity_hard_stop_step_py(
        *,
        initialized,
        red_latched,
        drawdown_ema,
        tier,
        red_threshold,
        ema_span_minutes,
        tier_ratio_yellow,
        tier_ratio_orange,
        equity,
        peak_strategy_equity,
        sample_minutes,
    ):
        span_samples = ema_span_minutes / sample_minutes
        alpha = 2.0 / (span_samples + 1.0)
        if not initialized:
            out_tier = "red" if red_latched else "green"
            return {
                "initialized": True,
                "red_latched": bool(red_latched),
                "peak_strategy_equity": float(peak_strategy_equity),
                "drawdown_ema": 0.0,
                "tier": out_tier,
                "drawdown_raw": 0.0,
                "drawdown_score": 0.0,
                "changed": out_tier != tier,
                "span_samples": float(span_samples),
                "alpha": float(alpha),
            }
        drawdown_raw = max(0.0, 1.0 - equity / max(peak_strategy_equity, 1e-12))
        drawdown_ema_next = alpha * drawdown_raw + (1.0 - alpha) * drawdown_ema
        drawdown_score = min(drawdown_raw, drawdown_ema_next)
        if red_latched or drawdown_score >= red_threshold:
            out_tier = "red"
            red_latched = True
        elif drawdown_score >= red_threshold * tier_ratio_orange:
            out_tier = "orange"
        elif drawdown_score >= red_threshold * tier_ratio_yellow:
            out_tier = "yellow"
        else:
            out_tier = "green"
        return {
            "initialized": True,
            "red_latched": bool(red_latched),
            "peak_strategy_equity": float(peak_strategy_equity),
            "drawdown_ema": float(drawdown_ema_next),
            "tier": out_tier,
            "drawdown_raw": float(drawdown_raw),
            "drawdown_score": float(drawdown_score),
            "changed": out_tier != tier,
            "span_samples": float(span_samples),
            "alpha": float(alpha),
        }
    module.equity_hard_stop_step_py = _equity_hard_stop_step_py

    def _get_order_id_type_from_string(name: str) -> int:
        mapping = {
            "close_unstuck_long": 0x1234,
            "close_unstuck_short": 0x1235,
            "close_panic_long": 0x1240,
            "close_panic_short": 0x1241,
            "empty": 0x0000,
        }
        return mapping.get(name, 0x9999)

    def _order_type_id_to_snake(type_id: int) -> str:
        mapping = {
            0x1234: "close_unstuck_long",
            0x1235: "close_unstuck_short",
            0x1240: "close_panic_long",
            0x1241: "close_panic_short",
            0x0000: "empty",
        }
        return mapping.get(type_id, "other")

    module.get_order_id_type_from_string = _get_order_id_type_from_string
    module.order_type_id_to_snake = _order_type_id_to_snake
    return module


@pytest.fixture(autouse=True)
def mock_pbr(monkeypatch):
    stub_module = _make_mock_pbr()
    monkeypatch.setitem(sys.modules, "passivbot_rust", stub_module)

    class _DummyLockException(Exception):
        pass

    class _DummyLock:
        def __init__(self, *args, **kwargs):
            self._locked = False
            self.fail_when_locked = kwargs.get("fail_when_locked", False)

        def acquire(self, timeout=None, fail_when_locked=False):
            if self._locked and (self.fail_when_locked or fail_when_locked):
                raise _DummyLockException("lock already acquired")
            self._locked = True
            return True

        def release(self):
            self._locked = False

        def __enter__(self):
            if self._locked and self.fail_when_locked:
                raise _DummyLockException("lock already acquired")
            self._locked = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self._locked = False
            return False

    portalocker_stub = types.SimpleNamespace(
        Lock=_DummyLock, exceptions=types.SimpleNamespace(LockException=_DummyLockException)
    )
    monkeypatch.setitem(sys.modules, "portalocker", portalocker_stub)

    import passivbot

    monkeypatch.setattr(passivbot, "pbr", stub_module, raising=False)


def _dummy_config():
    from config_utils import get_template_config, format_config

    cfg = get_template_config()
    cfg["live"]["user"] = "test_user"
    cfg["live"]["minimum_coin_age_days"] = 0
    for side in ("long", "short"):
        cfg["bot"][side]["unstuck_loss_allowance_pct"] = 0.01
        cfg["bot"][side]["wallet_exposure_limit"] = 1.0
        cfg["bot"][side]["unstuck_threshold"] = 0.5
        cfg["bot"][side]["unstuck_close_pct"] = 0.1
        cfg["bot"][side]["unstuck_ema_dist"] = 0.0
        cfg["bot"][side]["ema_span_0"] = 1
        cfg["bot"][side]["ema_span_1"] = 1
        cfg["bot"][side]["entry_grid_spacing_pct"] = 0.01
        cfg["bot"][side]["close_grid_markup_start"] = 0.01
        cfg["bot"][side]["close_grid_markup_end"] = 0.01
    cfg = format_config(cfg, live_only=True, verbose=False)
    return cfg


def _make_dummy_bot(config, *, last_price=100.0):
    from passivbot import Passivbot
    import passivbot_rust as pbr

    class DummyBot(Passivbot):
        def __init__(self, cfg):
            # minimal init without hitting live APIs
            self.config = cfg
            self.user = cfg["live"]["user"]
            self.user_info = {"exchange": "test_exchange"}
            self.exchange = self.user_info["exchange"]
            self.utc_offset = 0
            self.broker_code = ""
            self.custom_id_max_length = 36
            self.sym_padding = 17
            self.stop_signal_received = False
            self.balance = 1000.0
            self.hedge_mode = True
            self._config_hedge_mode = True
            self.inverse = False
            self.active_symbols = []
            self.open_orders = {}
            self.positions = {}
            self.symbol_ids = {}
            self.min_costs = {}
            self.min_qtys = {}
            self.qty_steps = {}
            self.price_steps = {}
            self.c_mults = {}
            self.max_leverage = {}
            self.pside_int_map = {"long": 0, "short": 1}
            self._pnls_manager = None
            self.pnls_cache_filepath = ""
            self.state_change_detected_by_symbol = set()
            self.recent_order_executions = []
            self.recent_order_cancellations = []
            self.eligible_symbols = set()
            self.ineligible_symbols = {}
            self.approved_coins_minus_ignored_coins = {"long": [], "short": []}
            self.PB_modes = {"long": {}, "short": {}}
            self._runtime_forced_modes = {"long": {}, "short": {}}
            self.inactive_coin_candle_ttl_ms = 60_000
            self.trailing_prices = {}
            self.markets_dict = {}
            self.effective_min_cost = {}
            self.equity_hard_stop_loss = {
                "enabled": False,
                "red_threshold": 0.25,
                "ema_span_minutes": 60.0,
                "cooldown_minutes_after_red": 0.0,
                "no_restart_drawdown_threshold": 1.0,
                "tier_ratios": {"yellow": 0.5, "orange": 0.75},
                "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "panic_close_order_type": "market",
            }
            self._equity_hard_stop_runtime = pbr.EquityHardStopRuntime()
            self._equity_hard_stop_strategy_pnl_peak = pbr.EquityHardStopRollingPeak()
            self._bp_defaults = {
                "ema_span_0": 1.0,
                "ema_span_1": 2.0,
                "entry_volatility_ema_span_hours": 0.0,
                "entry_grid_spacing_pct": 0.0,
                "entry_initial_qty_pct": 0.0,
                "entry_grid_double_down_factor": 1.0,
                "entry_grid_spacing_volatility_weight": 0.0,
                "entry_grid_spacing_we_weight": 0.0,
                "entry_initial_ema_dist": 0.0,
                "entry_trailing_double_down_factor": 0.0,
                "entry_trailing_grid_ratio": 0.0,
                "entry_trailing_retracement_pct": 0.0,
                "entry_trailing_retracement_we_weight": 0.0,
                "entry_trailing_retracement_volatility_weight": 0.0,
                "entry_trailing_threshold_pct": 0.0,
                "entry_trailing_threshold_we_weight": 0.0,
                "entry_trailing_threshold_volatility_weight": 0.0,
                "wallet_exposure_limit": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "close_grid_markup_end": 0.0,
                "close_grid_markup_start": 0.0,
                "close_grid_qty_pct": 0.0,
                "close_trailing_grid_ratio": 0.0,
                "close_trailing_qty_pct": 0.0,
                "close_trailing_retracement_pct": 0.0,
                "close_trailing_threshold_pct": 0.0,
                "risk_wel_enforcer_threshold": 0.0,
                "unstuck_threshold": 0.0,
                "unstuck_close_pct": 0.0,
                "unstuck_ema_dist": 0.0,
            }
            self._bot_value_defaults = {
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
                "risk_twel_enforcer_threshold": 0.0,
                "filter_volume_ema_span": 0.0,
                "filter_volatility_ema_span": 0.0,
                "filter_volume_drop_pct": 0.0,
                "filter_volatility_drop_pct": 0.0,
                "unstuck_loss_allowance_pct": 0.0,
            }
            self._live_values = {
                "filter_by_min_effective_cost": False,
                "price_distance_threshold": 1.0,
                "market_orders_allowed": False,
                "order_match_tolerance_pct": 0.0,
            }

            async def _get_last_prices(symbols, max_age_ms=None):
                return {s: last_price for s in symbols}

            self.cm = types.SimpleNamespace(
                get_last_prices=_get_last_prices,
                get_ema_bounds=lambda symbol, span0, span1, max_age_ms=None: (
                    last_price - 10,
                    last_price + 10,
                ),
                get_current_close=lambda symbol, max_age_ms=None: last_price,
                get_latest_ema_log_range_many=lambda items, **kwargs: {sym: 0.0 for sym, _ in items},
                get_ema_bounds_many=lambda items, **kwargs: {
                    sym: (last_price - 10, last_price + 10) for sym, _, _ in items
                },
            )

        def bp(self, pside: str, key: str, symbol: str | None = None):
            return self._bp_defaults.get(key, 0.0)

        def bot_value(self, pside: str, key: str):
            return self._bot_value_defaults.get(key, 0.0)

        def live_value(self, key: str):
            return self._live_values.get(key, 0.0)

    return DummyBot(config)


def _set_basic_state(bot, symbol="TEST/USDT"):
    bot.active_symbols = [symbol]
    bot.open_orders = {symbol: []}
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.qty_steps[symbol] = 0.01
    bot.price_steps[symbol] = 0.01
    bot.min_qtys[symbol] = 0.0
    bot.min_costs[symbol] = 0.0
    bot.c_mults[symbol] = 1.0
    bot.PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
    bot.approved_coins_minus_ignored_coins = {"long": [symbol], "short": []}
    bot.pnls = [{"pnl": 5.0, "timestamp": 0, "id": 1}]
    return symbol


def _make_unstuck_order(symbol="TEST/USDT", type_id=0x1234, price=100.0):
    return {
        "symbol": symbol,
        "side": "sell",
        "position_side": "long",
        "qty": 0.1,
        "price": price,
        "reduce_only": True,
        "custom_id": f"0x{type_id:04x}dummy",
    }


def _make_order(
    symbol="TEST/USDT",
    *,
    side="buy",
    position_side="long",
    qty=0.1,
    price=100.0,
    reduce_only=False,
    custom_hex=0x0001,
):
    suffix = "ro" if reduce_only else "entry"
    return {
        "symbol": symbol,
        "side": side,
        "position_side": position_side,
        "qty": qty,
        "price": price,
        "reduce_only": reduce_only,
        "id": f"{position_side}_{side}_{price}",
        "custom_id": f"0x{custom_hex:04x}{suffix}",
    }


@pytest.mark.asyncio
async def test_existing_unstuck_blocks_new(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)
    import passivbot_rust as pbr

    # Pretend an unstuck order is already live on the exchange
    bot.open_orders[symbol] = [
        {
            "symbol": symbol,
            "side": "sell",
            "position_side": "long",
            "qty": 0.1,
            "price": 100.0,
            "reduce_only": True,
            "custom_id": "0x1234existing",
        }
    ]

    bot.markets_dict = {symbol: {"active": True}}
    bot.effective_min_cost = {symbol: 1.0}
    bot.trailing_prices = {
        symbol: {
            "long": {
                "min_since_open": 0.0,
                "max_since_min": 0.0,
                "max_since_open": 0.0,
                "min_since_max": 0.0,
            },
            "short": {
                "min_since_open": 0.0,
                "max_since_min": 0.0,
                "max_since_open": 0.0,
                "min_since_max": 0.0,
            },
        }
    }

    captured = {}

    async def fake_load_bundle(self, symbols, modes):
        m1_close = {symbol: {1.0: 100.0, 2.0: 100.0, (1.0 * 2.0) ** 0.5: 100.0}}
        m1_volume = {symbol: {}}
        m1_log_range = {symbol: {}}
        h1_log_range = {symbol: {}}
        return m1_close, m1_volume, m1_log_range, h1_log_range, {}, {}

    def fake_compute(input_json: str) -> str:
        captured["input"] = input_json
        return '{"orders": [], "diagnostics": {"warnings": []}}'

    monkeypatch.setattr(bot, "_load_orchestrator_ema_bundle", types.MethodType(fake_load_bundle, bot))
    monkeypatch.setattr(pbr, "compute_ideal_orders_json", fake_compute)

    await bot.calc_ideal_orders_orchestrator()

    payload = json.loads(captured["input"])
    assert payload["global"]["unstuck_allowance_long"] == 0.0
    assert payload["global"]["unstuck_allowance_short"] == 0.0


@pytest.mark.asyncio
async def test_manual_mode_skips_side_orders(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.PB_modes["long"][symbol] = "manual"
    bot.open_orders[symbol] = [
        _make_order(symbol, side="buy", position_side="long", price=101.0, custom_hex=0x0002)
    ]
    bot.active_symbols = [symbol]

    async def fake_calc(self):
        return {}

    bot.calc_ideal_orders = types.MethodType(fake_calc, bot)

    async def fake_mprice(sym, **kwargs):
        return 100.0

    bot.cm.get_current_close = fake_mprice

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert to_cancel == []
    assert to_create == []


@pytest.mark.asyncio
async def test_tp_only_filters_entry_orders(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.PB_modes["long"][symbol] = "tp_only"
    bot.open_orders[symbol] = []
    bot.active_symbols = [symbol]

    async def fake_calc(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    side="buy",
                    position_side="long",
                    price=101.0,
                    reduce_only=False,
                    custom_hex=0x0003,
                ),
                _make_order(
                    symbol,
                    side="sell",
                    position_side="long",
                    price=102.0,
                    reduce_only=True,
                    custom_hex=0x0004,
                ),
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc, bot)

    async def fake_mprice(sym, **kwargs):
        return 100.0

    bot.cm.get_current_close = fake_mprice

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert to_cancel == []
    assert len(to_create) == 1
    assert to_create[0]["reduce_only"] is True
    assert to_create[0]["price"] == 102.0


@pytest.mark.asyncio
async def test_tp_only_with_active_entry_cancellation_filters_entries_but_keeps_entry_cancels(
    monkeypatch,
):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.PB_modes["long"][symbol] = "tp_only_with_active_entry_cancellation"
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            side="buy",
            position_side="long",
            price=101.0,
            reduce_only=False,
            custom_hex=0x0101,
        )
    ]
    bot.active_symbols = [symbol]

    async def fake_calc(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    side="buy",
                    position_side="long",
                    price=102.0,
                    reduce_only=False,
                    custom_hex=0x0102,
                ),
                _make_order(
                    symbol,
                    side="sell",
                    position_side="long",
                    price=103.0,
                    reduce_only=True,
                    custom_hex=0x0103,
                ),
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc, bot)

    async def fake_mprice(sym, **kwargs):
        return 100.0

    bot.cm.get_current_close = fake_mprice

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert len(to_cancel) == 1
    assert to_cancel[0]["reduce_only"] is False
    assert to_cancel[0]["price"] == 101.0
    assert len(to_create) == 1
    assert to_create[0]["reduce_only"] is True
    assert to_create[0]["price"] == 103.0


def test_runtime_forced_mode_takes_precedence_over_config():
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.config["live"]["forced_mode_long"] = "manual"
    bot._runtime_forced_modes["long"][symbol] = "panic"

    assert bot.get_forced_PB_mode("long", symbol) == "panic"


def test_orange_overlay_graceful_stop_preserves_restrictive_modes():
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.equity_hard_stop_loss["enabled"] = True
    bot.equity_hard_stop_loss["orange_tier_mode"] = "graceful_stop"
    bot._equity_hard_stop_runtime._initialized = True
    bot._equity_hard_stop_runtime._tier = "orange"
    bot._equity_hard_stop_runtime._red_latched = False
    bot.PB_modes["long"][symbol] = "normal"
    bot.PB_modes["short"][symbol] = "manual"

    bot._apply_equity_hard_stop_orange_overlay()

    assert bot.PB_modes["long"][symbol] == "graceful_stop"
    assert bot.PB_modes["short"][symbol] == "manual"


def test_orange_overlay_tp_only_with_active_entry_cancellation_only_for_open_positions():
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.equity_hard_stop_loss["enabled"] = True
    bot.equity_hard_stop_loss["orange_tier_mode"] = "tp_only_with_active_entry_cancellation"
    bot._equity_hard_stop_runtime._initialized = True
    bot._equity_hard_stop_runtime._tier = "orange"
    bot._equity_hard_stop_runtime._red_latched = False
    bot.PB_modes["long"][symbol] = "normal"
    bot.PB_modes["short"][symbol] = "normal"
    bot.positions[symbol]["long"]["size"] = 1.0
    bot.positions[symbol]["short"]["size"] = 0.0

    bot._apply_equity_hard_stop_orange_overlay()

    assert bot.PB_modes["long"][symbol] == "tp_only_with_active_entry_cancellation"
    assert bot.PB_modes["short"][symbol] == "normal"


def test_panic_close_order_type_pref_market_overrides_limit_path():
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)
    import passivbot_rust as pbr

    panic_id = pbr.get_order_id_type_from_string("close_panic_long")
    ideal_orders = {symbol: [(0.1, 100.0, "close_panic_long", panic_id)]}
    last_prices = {symbol: 100.0}

    bot.equity_hard_stop_loss["panic_close_order_type"] = "market"
    orders, _ = bot._to_executable_orders(ideal_orders, last_prices)
    assert orders[symbol][0]["type"] == "market"

    bot.equity_hard_stop_loss["panic_close_order_type"] = "limit"
    orders, _ = bot._to_executable_orders(ideal_orders, last_prices)
    assert orders[symbol][0]["type"] == "limit"


def test_hard_stop_apply_sample_delegates_to_rust(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)

    bot._equity_hard_stop_runtime._initialized = True
    bot._equity_hard_stop_runtime._tier = "yellow"
    bot._equity_hard_stop_runtime._drawdown_ema = 0.08
    bot._equity_hard_stop_runtime._red_latched = False

    captured = {}

    def fake_apply_sample(**kwargs):
        captured.update(kwargs)
        return {
            "initialized": True,
            "red_latched": False,
            "peak_strategy_equity": 1200.0,
            "rolling_peak_strategy_equity": 1210.0,
            "drawdown_ema": 0.13,
            "tier": "orange",
            "drawdown_raw": 0.19,
            "drawdown_score": 0.13,
            "changed": True,
            "span_samples": 1800.0,
            "alpha": 0.001110493,
        }

    monkeypatch.setattr(bot._equity_hard_stop_runtime, "apply_sample", fake_apply_sample)

    metrics = bot._equity_hard_stop_apply_sample(1_700_000_000_000, 900.0, 25.0, 50.0, 1.0)

    assert captured["sample_minutes"] == 1.0
    assert captured["equity"] == 950.0
    assert captured["peak_strategy_equity"] == pytest.approx(950.0)
    assert captured["timestamp_ms"] == 1_700_000_000_000
    assert metrics["tier"] == "orange"
    assert metrics["span_samples"] == pytest.approx(1800.0)


def test_hard_stop_apply_sample_rolling_peak_prunes_by_lookback():
    cfg = _dummy_config()
    cfg["live"]["pnls_max_lookback_days"] = 0.0
    bot = _make_dummy_bot(cfg)

    m0 = bot._equity_hard_stop_apply_sample(1_000, 100.0, 0.0, 0.0, 1.0)
    m1 = bot._equity_hard_stop_apply_sample(2_000, 95.0, 0.0, 0.0, 1.0)

    assert m0["peak_strategy_equity"] == pytest.approx(100.0)
    assert m1["peak_strategy_equity"] == pytest.approx(95.0)
    assert m1["rolling_peak_strategy_equity"] == pytest.approx(95.0)
    assert m1["drawdown_raw"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_hard_stop_finalize_red_stop_terminal_latches_and_stops(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 5.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.1

    async def fake_compute(_ts):
        return {
            "stop_event_timestamp_ms": 1_700_000_000_000,
            "equity": 98.0,
            "peak_strategy_equity": 110.0,
            "trigger_peak_strategy_equity": 101.0,
            "drawdown_raw": 0.109090909,
            "drawdown_ema": 0.105,
            "drawdown_score": 0.105,
        }

    captured = {}

    def fake_write(payload):
        captured["payload"] = payload
        return "/tmp/hs_latch_terminal.json"

    async def fake_wait(_until):
        raise AssertionError("cooldown wait should not run for terminal no-restart latch")

    monkeypatch.setattr(bot, "_equity_hard_stop_compute_stop_event", fake_compute)
    monkeypatch.setattr(bot, "_equity_hard_stop_write_latch", fake_write)
    monkeypatch.setattr(bot, "_equity_hard_stop_wait_for_cooldown", fake_wait)

    await bot._equity_hard_stop_finalize_red_stop()

    assert bot.stop_signal_received is True
    assert captured["payload"]["no_restart_latched"] is True
    assert captured["payload"]["cooldown_until_ms"] is None


@pytest.mark.asyncio
async def test_hard_stop_finalize_red_stop_equal_threshold_latches_terminal(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 5.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.1

    async def fake_compute(_ts):
        return {
            "stop_event_timestamp_ms": 1_700_000_000_000,
            "equity": 98.0,
            "peak_strategy_equity": 110.0,
            "trigger_peak_strategy_equity": 101.0,
            "drawdown_raw": 0.1,
            "drawdown_ema": 0.095,
            "drawdown_score": 0.095,
        }

    captured = {}

    def fake_write(payload):
        captured["payload"] = payload
        return "/tmp/hs_latch_terminal_equal.json"

    async def fake_wait(_until):
        raise AssertionError("cooldown wait should not run for equality no-restart latch")

    monkeypatch.setattr(bot, "_equity_hard_stop_compute_stop_event", fake_compute)
    monkeypatch.setattr(bot, "_equity_hard_stop_write_latch", fake_write)
    monkeypatch.setattr(bot, "_equity_hard_stop_wait_for_cooldown", fake_wait)

    await bot._equity_hard_stop_finalize_red_stop()

    assert bot.stop_signal_received is True
    assert captured["payload"]["no_restart_latched"] is True
    assert captured["payload"]["cooldown_until_ms"] is None


@pytest.mark.asyncio
async def test_hard_stop_finalize_red_stop_autorestarts_after_cooldown(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 1.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.2
    bot._equity_hard_stop_runtime._initialized = True
    bot._equity_hard_stop_runtime._red_latched = True
    bot._equity_hard_stop_runtime._tier = "red"
    bot._equity_hard_stop_runtime._drawdown_ema = 0.12

    async def fake_compute(_ts):
        return {
            "stop_event_timestamp_ms": 1_700_000_000_000,
            "equity": 104.0,
            "peak_strategy_equity": 110.0,
            "trigger_peak_strategy_equity": 106.0,
            "drawdown_raw": 0.05454545,
            "drawdown_ema": 0.08,
            "drawdown_score": 0.08,
        }

    captured = {}
    waited = {}
    cleared = {}

    def fake_write(payload):
        captured["payload"] = payload
        return "/tmp/hs_latch_auto.json"

    async def fake_wait(until_ms):
        waited["until_ms"] = until_ms

    def fake_remove():
        cleared["done"] = True

    monkeypatch.setattr(bot, "_equity_hard_stop_compute_stop_event", fake_compute)
    monkeypatch.setattr(bot, "_equity_hard_stop_write_latch", fake_write)
    monkeypatch.setattr(bot, "_equity_hard_stop_wait_for_cooldown", fake_wait)
    monkeypatch.setattr(bot, "_equity_hard_stop_remove_latch_file", fake_remove)

    await bot._equity_hard_stop_finalize_red_stop()

    assert bot.stop_signal_received is False
    assert captured["payload"]["no_restart_latched"] is False
    assert captured["payload"]["cooldown_until_ms"] is not None
    assert waited["until_ms"] == captured["payload"]["cooldown_until_ms"]
    assert cleared["done"] is True
    assert bot._equity_hard_stop_runtime.red_latched() is False


@pytest.mark.asyncio
async def test_hard_stop_initialize_from_history_terminal_stop_sets_latch(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["enabled"] = True
    bot.equity_hard_stop_loss["red_threshold"] = 0.05
    bot.equity_hard_stop_loss["ema_span_minutes"] = 1.0
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 5.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.1
    bot.balance = 80.0

    async def fake_history(*, current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 1_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 61_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": -20.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 121_000,
                    "balance": 80.0,
                    "realized_pnl": -20.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": True,
                },
            ]
        }

    captured = {}

    def fake_write(payload):
        captured["payload"] = payload
        return "/tmp/hs_replay_terminal.json"

    monkeypatch.setattr(bot, "get_balance_equity_history", fake_history)
    monkeypatch.setattr(bot, "_equity_hard_stop_write_latch", fake_write)

    await bot._equity_hard_stop_initialize_from_history()

    assert bot.stop_signal_received is True
    assert captured["payload"]["no_restart_latched"] is True
    assert captured["payload"]["cooldown_until_ms"] is None
    assert captured["payload"]["stop_event_timestamp_ms"] == 121_000


@pytest.mark.asyncio
async def test_hard_stop_initialize_from_history_reconstructs_active_cooldown_without_latch(
    monkeypatch,
):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["enabled"] = True
    bot.equity_hard_stop_loss["red_threshold"] = 0.05
    bot.equity_hard_stop_loss["ema_span_minutes"] = 1.0
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 1.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.2
    bot.balance = 104.0
    bot._live_values["execution_delay_seconds"] = 60.0

    async def fake_history(*, current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 1_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 61_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 10.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 121_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 4.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 181_000,
                    "balance": 104.0,
                    "realized_pnl": 4.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": True,
                },
            ]
        }

    current_time = {"ts": 200_000}
    waited = {}

    async def fake_wait(until_ms):
        waited["until_ms"] = until_ms
        current_time["ts"] = until_ms

    async def fake_upnl():
        return 0.0

    monkeypatch.setattr(bot, "get_balance_equity_history", fake_history)
    monkeypatch.setattr(bot, "_equity_hard_stop_wait_for_cooldown", fake_wait)
    monkeypatch.setattr(bot, "get_exchange_time", lambda: current_time["ts"])
    monkeypatch.setattr(bot, "_calc_upnl_sum_strict", fake_upnl)

    await bot._equity_hard_stop_initialize_from_history()

    assert waited["until_ms"] == 241_000
    assert bot.stop_signal_received is False
    assert bot._equity_hard_stop_runtime.red_latched() is False
    assert bot._equity_hard_stop_pending_red_since_ms is None


@pytest.mark.asyncio
async def test_hard_stop_initialize_from_history_replay_cooldown_resets_cycle(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    bot.equity_hard_stop_loss["enabled"] = True
    bot.equity_hard_stop_loss["red_threshold"] = 0.05
    bot.equity_hard_stop_loss["ema_span_minutes"] = 1.0
    bot.equity_hard_stop_loss["cooldown_minutes_after_red"] = 1.0
    bot.equity_hard_stop_loss["no_restart_drawdown_threshold"] = 0.2
    bot.balance = 104.0
    bot._live_values["execution_delay_seconds"] = 60.0

    async def fake_history(*, current_balance=None):
        return {
            "timeline": [
                {
                    "timestamp": 1_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 61_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 10.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 121_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 4.0,
                    "is_flat": False,
                },
                {
                    "timestamp": 181_000,
                    "balance": 104.0,
                    "realized_pnl": 4.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": True,
                },
                {
                    "timestamp": 241_000,
                    "balance": 104.0,
                    "realized_pnl": 4.0,
                    "unrealized_pnl": 0.0,
                    "is_flat": False,
                },
            ]
        }

    reset_calls = {"count": 0}

    def wrapped_reset():
        reset_calls["count"] += 1
        bot._equity_hard_stop_runtime._initialized = False
        bot._equity_hard_stop_runtime._red_latched = False
        bot._equity_hard_stop_runtime._tier = "green"
        bot._equity_hard_stop_runtime._drawdown_ema = 0.0
        bot._equity_hard_stop_runtime._peak_strategy_equity = 0.0
        bot._equity_hard_stop_strategy_pnl_peak.reset()

    async def fake_upnl():
        return 0.0

    monkeypatch.setattr(bot, "get_balance_equity_history", fake_history)
    monkeypatch.setattr(bot._equity_hard_stop_runtime, "reset", wrapped_reset)
    monkeypatch.setattr(bot, "_calc_upnl_sum_strict", fake_upnl)
    monkeypatch.setattr(bot, "get_exchange_time", lambda: 301_000)

    await bot._equity_hard_stop_initialize_from_history()

    assert bot.stop_signal_received is False
    assert reset_calls["count"] == 2
    assert bot._equity_hard_stop_runtime.red_latched() is False
    assert bot._equity_hard_stop_runtime.tier() != "red"
    assert bot._equity_hard_stop_pending_red_since_ms is None


@pytest.mark.asyncio
async def test_orders_sorted_by_market_diff(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    bot.PB_modes["long"][symbol] = "normal"
    bot.open_orders[symbol] = [
        _make_order(symbol, side="buy", position_side="long", price=102.0, custom_hex=0x0005),
        _make_order(symbol, side="buy", position_side="long", price=97.0, custom_hex=0x0006),
    ]
    bot.active_symbols = [symbol]

    async def fake_calc(self):
        return {
            symbol: [
                _make_order(symbol, side="buy", position_side="long", price=101.0, custom_hex=0x0007),
                _make_order(symbol, side="buy", position_side="long", price=95.0, custom_hex=0x0008),
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc, bot)

    async def fake_mprice(sym, **kwargs):
        return 100.0

    bot.cm.get_current_close = fake_mprice

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [order["price"] for order in to_cancel] == [102.0, 97.0]
    assert [order["price"] for order in to_create] == [101.0, 95.0]
