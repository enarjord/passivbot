import sys
import types

import pytest


def _make_mock_pbr():
    return types.SimpleNamespace(
        calc_auto_unstuck_allowance=lambda balance, allowance_pct, max_pnl, last_pnl: allowance_pct
        * balance,
        calc_wallet_exposure=lambda c_mult, balance, size, price: abs(size)
        * price
        / max(balance, 1e-6),
        calc_min_entry_qty=lambda *args, **kwargs: 0.0,
        calc_min_entry_qty_py=lambda *args, **kwargs: 0.0,
        cost_to_qty=lambda *args, **kwargs: 0.0,
        round_dn=lambda value, step: value,
        round_up=lambda value, step: value,
        round_dynamic=lambda value, sig: value,
        calc_pnl_long=lambda entry_price, close_price, qty, c_mult: (close_price - entry_price) * qty,
        calc_pnl_short=lambda entry_price, close_price, qty, c_mult: (entry_price - close_price)
        * qty,
        calc_pprice_diff_int=lambda *args, **kwargs: 0.0,
        get_order_id_type_from_string=lambda s: 0x1234,
        order_type_id_to_snake=lambda type_id: {
            0x1234: "close_unstuck_long",
            0x1235: "close_unstuck_short",
        }.get(type_id, "other"),
        calc_diff=lambda price, reference: price - reference,
        round_=lambda value, step: value,
    )


@pytest.fixture(autouse=True)
def mock_pbr(monkeypatch):
    monkeypatch.setitem(sys.modules, "passivbot_rust", _make_mock_pbr())


def _dummy_config():
    from config_utils import get_template_config, format_config

    cfg = get_template_config("v7")
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

            self.cm = types.SimpleNamespace(
                get_last_prices=lambda symbols, max_age_ms=None: {s: last_price for s in symbols},
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


@pytest.mark.asyncio
async def test_calc_orders_emits_single_unstuck(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

    import types as _types

    async def fake_calc(self, allow_unstuck=True):
        assert allow_unstuck is True
        return {
            symbol: [
                _make_unstuck_order(symbol, 0x1234, 101.0),
                _make_unstuck_order(symbol, 0x1235, 102.0),
            ]
        }

    bot.calc_ideal_orders = _types.MethodType(fake_calc, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    unstuck_created = [
        order
        for order in to_create
        if order["custom_id"].startswith("0x1234") or order["custom_id"].startswith("0x1235")
    ]
    assert len(unstuck_created) == 1


@pytest.mark.asyncio
async def test_existing_unstuck_blocks_new(monkeypatch):
    cfg = _dummy_config()
    bot = _make_dummy_bot(cfg)
    symbol = _set_basic_state(bot)

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

    import types as _types

    flags = {}

    async def fake_calc(self, allow_unstuck=True):
        flags["allow_unstuck"] = allow_unstuck
        if not allow_unstuck:
            return {symbol: []}
        return {symbol: [_make_unstuck_order(symbol, 0x1234, 101.0)]}

    bot.calc_ideal_orders = _types.MethodType(fake_calc, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert flags.get("allow_unstuck") is False
    assert all(
        not order["custom_id"].startswith("0x1234") for order in to_create
    ), "No new unstuck orders should be created when one is live"
