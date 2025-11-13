from typing import Dict, Optional, Tuple

import pytest

import passivbot_rust as pbr
from passivbot import Passivbot


class DummyCM:
    def __init__(
        self,
        last_prices: Dict[str, float],
        ema_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        log_ranges: Optional[Dict[str, float]] = None,
        current_closes: Optional[Dict[str, float]] = None,
    ):
        self.last_prices = last_prices
        self.ema_bounds = ema_bounds or {}
        self.log_ranges = log_ranges or {}
        self.current_closes = current_closes or last_prices

    async def get_last_prices(self, symbols, max_age_ms=None):
        return {s: self.last_prices[s] for s in symbols}

    async def get_ema_bounds_many(self, items, max_age_ms=None):
        result = {}
        for symbol, _span0, _span1 in items:
            result[symbol] = self.ema_bounds.get(
                symbol,
                (
                    self.last_prices.get(symbol, float("nan")),
                    self.last_prices.get(symbol, float("nan")),
                ),
            )
        return result

    async def get_latest_ema_log_range_many(self, items, tf=None, max_age_ms=None):
        return {symbol: self.log_ranges.get(symbol, 0.0) for symbol, _ in items}

    async def get_current_close(self, symbol, max_age_ms=None):
        return self.current_closes.get(symbol, self.last_prices.get(symbol, float("nan")))


class DummyBot(Passivbot):
    _bp_defaults = {
        "entry_grid_double_down_factor": 1.0,
        "entry_grid_spacing_volatility_weight": 0.0,
        "entry_grid_spacing_we_weight": 0.0,
        "entry_grid_spacing_pct": 0.0,
        "entry_initial_ema_dist": 0.0,
        "entry_initial_qty_pct": 0.0,
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
        "ema_span_0": 1.0,
        "ema_span_1": 1.0,
        "risk_wel_enforcer_threshold": 0.0,
        "entry_volatility_ema_span_hours": 0.0,
    }

    _bot_value_defaults = {
        "risk_twel_enforcer_threshold": 0.0,
        "total_wallet_exposure_limit": 0.0,
        "n_positions": 0.0,
        "unstuck_loss_allowance_pct": 0.0,
    }

    def __init__(
        self,
        *,
        modes: Dict[str, Dict[str, str]],
        positions: Dict[str, Dict[str, Dict[str, float]]],
        last_prices: Dict[str, float],
        bp_overrides: Optional[Dict[Tuple[str, str, str], float]] = None,
        bot_value_overrides: Optional[Dict[Tuple[str, str], float]] = None,
        live_values: Optional[Dict[str, float]] = None,
    ):
        # Skip Passivbot.__init__
        self.PB_modes = modes
        self.positions = positions
        self.balance = 1_000.0
        self.hedge_mode = True
        self.active_symbols = list(positions.keys())
        self.qty_steps = {symbol: 0.01 for symbol in positions}
        self.price_steps = {symbol: 0.01 for symbol in positions}
        self.min_qtys = {symbol: 0.0 for symbol in positions}
        self.min_costs = {symbol: 0.0 for symbol in positions}
        self.c_mults = {symbol: 1.0 for symbol in positions}
        self.trailing_prices = {
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
            for symbol in positions
        }
        ema_bounds = {symbol: (price * 0.9, price * 1.1) for symbol, price in last_prices.items()}
        self.cm = DummyCM(last_prices, ema_bounds=ema_bounds)
        self._bp_overrides = bp_overrides or {}
        self._bot_value_overrides = bot_value_overrides or {}
        self._live_values = {"price_distance_threshold": 1.0, "market_orders_allowed": False}
        if live_values:
            self._live_values.update(live_values)
        self.pnls = []
        self.custom_id_max_length = 36

    def bp(self, pside, key, symbol=None):
        return self._bp_overrides.get((pside, symbol, key), self._bp_defaults.get(key, 0.0))

    def bot_value(self, pside, key):
        return self._bot_value_overrides.get((pside, key), self._bot_value_defaults.get(key, 0.0))

    def live_value(self, key: str):
        return self._live_values.get(key, 0.0)

    async def calc_unstucking_close(self, allow_new_unstuck: bool = True):
        return "", (0.0, 0.0, "", 0)

    def format_custom_id_single(self, order_type_id: int) -> str:
        return f"cid-{order_type_id}"


@pytest.mark.asyncio
async def test_panic_mode_emits_close_order(monkeypatch):
    symbol = "TEST/USDT"
    modes = {"long": {symbol: "panic"}, "short": {symbol: "manual"}}
    positions = {
        symbol: {
            "long": {"size": 1.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot = DummyBot(modes=modes, positions=positions, last_prices={symbol: 95.0})

    orders = await bot.calc_ideal_orders()
    assert symbol in orders
    assert orders[symbol], f"expected orders for {symbol}, got {orders[symbol]}"
    assert len(orders[symbol]) == 1
    order = orders[symbol][0]
    assert order["position_side"] == "long"
    assert order["side"] == "sell"
    assert order["reduce_only"] is True
    assert order["qty"] == pytest.approx(1.5)
    assert order["price"] == pytest.approx(95.0)
    assert order["type"] == "limit"  # market_orders_allowed defaults to False
    assert order["custom_id"].startswith("cid-")


@pytest.mark.asyncio
async def test_close_orders_respect_price_threshold(monkeypatch):
    symbol = "TEST/USDT"
    modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
    positions = {
        symbol: {
            "long": {"size": 0.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot = DummyBot(
        modes=modes,
        positions=positions,
        last_prices={symbol: 100.0},
        bot_value_overrides={
            ("long", "total_wallet_exposure_limit"): 1.0,
            ("long", "n_positions"): 1.0,
        },
        live_values={"price_distance_threshold": 0.015, "market_orders_allowed": False},
    )

    close_type_id = pbr.order_type_snake_to_id("close_grid_long")

    monkeypatch.setattr(pbr, "calc_entries_long_py", lambda *_args, **_kwargs: [])

    def fake_calc_closes_long_py(*_args, **_kwargs):
        return [
            (0.5, 101.0, close_type_id),
            (0.5, 104.0, close_type_id),
        ]

    monkeypatch.setattr(pbr, "calc_closes_long_py", fake_calc_closes_long_py)
    monkeypatch.setattr(pbr, "gate_entries_by_twel_py", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pbr, "calc_twel_enforcer_orders_py", lambda *_args, **_kwargs: [])

    orders = await bot.calc_ideal_orders()
    assert symbol in orders
    close_orders = [order for order in orders[symbol] if order["reduce_only"]]
    assert len(close_orders) == 1, orders[symbol]
    close_order = close_orders[0]
    assert close_order["price"] == pytest.approx(101.0)
    assert close_order["qty"] == pytest.approx(0.0)
    assert close_order["type"] == "limit"
