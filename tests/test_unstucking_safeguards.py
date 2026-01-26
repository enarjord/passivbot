import json
import sys
import types

import pytest


def _make_mock_pbr():
    module = types.ModuleType("passivbot_rust")
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

    def _get_order_id_type_from_string(name: str) -> int:
        mapping = {
            "close_unstuck_long": 0x1234,
            "close_unstuck_short": 0x1235,
            "empty": 0x0000,
        }
        return mapping.get(name, 0x9999)

    def _order_type_id_to_snake(type_id: int) -> str:
        mapping = {
            0x1234: "close_unstuck_long",
            0x1235: "close_unstuck_short",
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

    class DummyBot(Passivbot):
        def __init__(self, cfg):
            # minimal init without hitting live APIs
            self.config = cfg
            self.user = cfg["live"]["user"]
            self.user_info = {"exchange": "test_exchange"}
            self.exchange = self.user_info["exchange"]
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
            self.pnls_cache_filepath = ""
            self.state_change_detected_by_symbol = set()
            self.recent_order_executions = []
            self.recent_order_cancellations = []
            self.eligible_symbols = set()
            self.ineligible_symbols = {}
            self.approved_coins_minus_ignored_coins = {"long": [], "short": []}
            self.PB_modes = {"long": {}, "short": {}}
            self.inactive_coin_candle_ttl_ms = 60_000
            self.trailing_prices = {}
            self.markets_dict = {}
            self.effective_min_cost = {}
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
