import logging
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
        self._live_values = {"market_orders_allowed": False}
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


def test_to_executable_orders_respects_rust_market_execution_hint():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["market_orders_allowed"] = False

    order_type = "close_unstuck_long"
    order_type_id = pbr.order_type_snake_to_id(order_type)
    ideal = {symbol: [(-0.5, 100.0, order_type, order_type_id, "market")]}

    orders, _ = bot._to_executable_orders(ideal, {symbol: 100.0})

    assert orders[symbol][0]["type"] == "market"


def test_to_executable_orders_respects_rust_limit_execution_hint():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["market_orders_allowed"] = True

    order_type = "close_unstuck_long"
    order_type_id = pbr.order_type_snake_to_id(order_type)
    ideal = {symbol: [(-0.5, 100.0, order_type, order_type_id, "limit")]}

    orders, _ = bot._to_executable_orders(ideal, {symbol: 100.0})

    assert orders[symbol][0]["type"] == "limit"


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


@pytest.mark.asyncio
async def test_order_sort_preserves_original_order_when_market_price_missing(caplog):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({})
    bot.register_symbol(symbol)
    bot.open_orders[symbol] = []

    first = _make_order(symbol, "buy", "long", 0.5, 95.0, "entry_grid_cropped_long")
    second = _make_order(symbol, "buy", "long", 0.5, 101.0, "entry_grid_normal_long")
    ideal = {symbol: [first, second]}

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    with caplog.at_level(logging.WARNING):
        to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert not to_cancel
    assert [order["price"] for order in to_create] == [95.0, 101.0]
    assert any("preserving to_create order" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_initial_entry_distance_gate_blocks_far_create_and_throttles_logs(
    caplog,
):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["initial_entry_exec_max_market_dist_pct"] = 0.005
    bot._live_values["order_match_tolerance_pct"] = 0.0002
    prices = iter([99.0, 99.01, 98.9])

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "buy",
                    "long",
                    1.0,
                    next(prices),
                    "entry_initial_normal_long",
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    with caplog.at_level(logging.INFO):
        for _ in range(3):
            to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
            assert not to_cancel
            assert not to_create

    messages = [
        record.message
        for record in caplog.records
        if "initial entry staged but not placed" in record.message
    ]
    assert len(messages) == 1
    assert "price=99" in messages[0]


def test_initial_entry_distance_gate_info_heartbeat_is_hourly(monkeypatch, caplog):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["order_match_tolerance_pct"] = 0.0002
    order = _make_order(
        symbol,
        "buy",
        "long",
        1.0,
        99.0,
        "entry_initial_normal_long",
    )
    now = {"ms": 0}
    monkeypatch.setattr("passivbot.utc_ms", lambda: now["ms"])

    with caplog.at_level(logging.INFO):
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100.0,
            signed_dist=0.01,
            threshold=0.005,
        )
        now["ms"] = 59 * 60 * 1000
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100.0,
            signed_dist=0.01,
            threshold=0.005,
        )
        now["ms"] = 60 * 60 * 1000
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100.0,
            signed_dist=0.01,
            threshold=0.005,
        )

    messages = [
        record.message
        for record in caplog.records
        if "initial entry staged but not placed" in record.message
    ]
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_initial_entry_distance_gate_keeps_existing_within_tolerance():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["initial_entry_exec_max_market_dist_pct"] = 0.005
    bot._live_values["order_match_tolerance_pct"] = 0.0002
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_initial_normal_long",
        )
    ]
    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                99.01,
                "entry_initial_normal_long",
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
async def test_initial_entry_distance_gate_cancels_far_drift_without_recreate():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["initial_entry_exec_max_market_dist_pct"] = 0.005
    bot._live_values["order_match_tolerance_pct"] = 0.0002
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_initial_normal_long",
        )
    ]
    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                98.9,
                "entry_initial_normal_long",
            )
        ]
    }

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)
    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
    assert [order["price"] for order in to_cancel] == [99.0]
    assert not to_create


@pytest.mark.asyncio
async def test_initial_entry_distance_gate_allows_near_market_initial():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["initial_entry_exec_max_market_dist_pct"] = 0.005
    bot._live_values["order_match_tolerance_pct"] = 0.0002
    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                99.8,
                "entry_initial_normal_long",
            )
        ]
    }

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)
    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
    assert not to_cancel
    assert [order["price"] for order in to_create] == [99.8]


@pytest.mark.asyncio
async def test_initial_entry_distance_gate_ignores_market_initial():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["initial_entry_exec_max_market_dist_pct"] = 0.005
    ideal = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                90.0,
                "entry_initial_normal_long",
                order_kind="market",
            )
        ]
    }

    async def fake_calc_ideal_orders(self):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)
    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
    assert not to_cancel
    assert [order["type"] for order in to_create] == ["market"]
