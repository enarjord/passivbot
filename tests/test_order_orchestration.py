import types

import pytest
import passivbot_rust as pbr
from passivbot import Passivbot


class OrchestrationBot(Passivbot):
    """Minimal bot wrapper exposing high-level orchestration helpers for testing."""

    def __init__(self, market_prices: dict[str, float]):
        self.balance = 1_000.0
        self.active_symbols: list[str] = []
        self.open_orders: dict[str, list[dict]] = {}
        self.positions: dict[str, dict[str, dict[str, float]]] = {}
        self.PB_modes = {"long": {}, "short": {}}
        self._market_prices = market_prices
        self._live_values = {"price_distance_threshold": 1.0, "market_orders_allowed": False}
        self.qty_steps = {}
        self.price_steps = {}
        self.min_qtys = {}
        self.min_costs = {}
        self.c_mults = {}

    def register_symbol(self, symbol: str) -> None:
        self.active_symbols.append(symbol)
        self.positions.setdefault(
            symbol,
            {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            },
        )
        self.PB_modes["long"].setdefault(symbol, "normal")
        self.PB_modes["short"].setdefault(symbol, "normal")
        self.open_orders.setdefault(symbol, [])
        self.qty_steps.setdefault(symbol, 0.01)
        self.price_steps.setdefault(symbol, 0.01)
        self.min_qtys.setdefault(symbol, 0.0)
        self.min_costs.setdefault(symbol, 0.0)
        self.c_mults.setdefault(symbol, 1.0)

    def live_value(self, key: str):
        return self._live_values.get(key, 0.0)

    async def _fetch_market_prices(self, symbols: set[str]) -> dict[str, float | None]:
        return {symbol: self._market_prices.get(symbol) for symbol in symbols}

    async def calc_unstucking_close(self, allow_new_unstuck: bool = True):
        return "", (0.0, 0.0, "", 0)

    def has_open_unstuck_order(self) -> bool:
        return False

    def format_custom_id_single(self, order_type_id: int) -> str:
        return f"order-0x{order_type_id:04x}"


def _make_order(
    symbol: str,
    side: str,
    position_side: str,
    qty: float,
    price: float,
    order_type: str,
    *,
    reduce_only: bool = False,
    order_kind: str = "limit",
) -> dict:
    order_type_id = pbr.order_type_snake_to_id(order_type)
    return {
        "symbol": symbol,
        "side": side,
        "position_side": position_side,
        "qty": qty,
        "price": price,
        "reduce_only": reduce_only,
        "custom_id": f"order-0x{order_type_id:04x}",
        "type": order_kind,
    }


@pytest.mark.asyncio
async def test_calc_orders_to_cancel_and_create_reconciles_orders(monkeypatch):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)

    matching_type = "entry_grid_normal_long"
    new_type = "entry_grid_cropped_long"

    bot.open_orders[symbol] = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 100.0,
            "custom_id": f"order-0x{pbr.order_type_snake_to_id(matching_type):04x}",
        },
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 0.5,
            "price": 98.0,
            "custom_id": "order-0xdead",
        },
    ]

    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                100.0,
                matching_type,
            ),
            _make_order(
                symbol,
                "buy",
                "long",
                0.5,
                102.0,
                new_type,
            ),
        ]
    }

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [order["custom_id"] for order in to_cancel] == ["order-0xdead"]
    assert [order["custom_id"] for order in to_create] == [
        f"order-0x{pbr.order_type_snake_to_id(new_type):04x}"
    ]


@pytest.mark.asyncio
async def test_order_hysteresis_skips_near_identical_cancel_create(monkeypatch):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["order_match_tolerance_pct"] = 0.001  # 0.1%

    bot.open_orders[symbol] = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 100.0,
            "custom_id": f"order-0x{pbr.order_type_snake_to_id('entry_grid_normal_long'):04x}",
        }
    ]

    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                100.05,  # within 0.1% tolerance, so should not churn
                "entry_grid_normal_long",
            )
        ]
    }

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)
    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
    assert not to_cancel
    assert not to_create


@pytest.mark.asyncio
async def test_to_create_orders_sorted_by_market_diff(monkeypatch):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.open_orders[symbol] = []

    fast_fill = _make_order(symbol, "buy", "long", 0.5, 101.0, "entry_grid_normal_long")
    slow_fill = _make_order(symbol, "buy", "long", 0.5, 95.0, "entry_grid_cropped_long")

    ideal = {symbol: [slow_fill, fast_fill]}  # Deliberately unsorted

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert not to_cancel
    assert [order["price"] for order in to_create] == [101.0, 95.0]
