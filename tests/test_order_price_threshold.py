import pytest

import passivbot_rust as pbr
from passivbot import Passivbot


class PriceThresholdBot(Passivbot):
    def __init__(self, symbol: str, last_price: float):
        self.balance = 1_000.0
        self.active_symbols = [symbol]
        self.open_orders = {symbol: []}
        self.positions = {
            symbol: {
                "long": {"size": 0.0, "price": last_price},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        self.PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
        self.qty_steps = {symbol: 0.01}
        self.price_steps = {symbol: 0.01}
        self.min_qtys = {symbol: 0.0}
        self.min_costs = {symbol: 0.0}
        self.c_mults = {symbol: 1.0}
        self.custom_id_max_length = 36
        self._live_values = {"price_distance_threshold": 0.015, "market_orders_allowed": False}

    def live_value(self, key: str):
        return self._live_values.get(key, 0.0)

    def format_custom_id_single(self, order_type_id: int) -> str:
        return f"cid-{order_type_id}"


def test_close_orders_respect_price_threshold():
    symbol = "TEST/USDT"
    bot = PriceThresholdBot(symbol, last_price=100.0)
    close_type_id = pbr.order_type_snake_to_id("close_grid_long")
    ideal_orders = {
        symbol: [
            (0.5, 101.0, "close_grid_long", close_type_id),
            (0.5, 104.0, "close_grid_long", close_type_id),
        ]
    }

    orders_by_symbol, _ = bot._to_executable_orders(ideal_orders, {symbol: 100.0})

    close_orders = [order for order in orders_by_symbol[symbol] if order["reduce_only"]]
    assert len(close_orders) == 1
    close_order = close_orders[0]
    assert close_order["price"] == pytest.approx(101.0)
    assert close_order["qty"] == pytest.approx(0.0)
    assert close_order["type"] == "limit"
