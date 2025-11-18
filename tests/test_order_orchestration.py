import types

import pytest
import asyncio

import passivbot_rust as pbr
from passivbot import Passivbot, BaseOrderPlan, try_decode_type_id_from_custom_id, snake_of
from collections import defaultdict


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


class IdealOrdersBot(Passivbot):
    """Test double providing controlled behavior for calc_ideal_orders."""

    def __init__(
        self,
        symbol: str,
        market_price: float,
        *,
        long_position: tuple[float, float] = (0.0, 0.0),
        short_position: tuple[float, float] = (0.0, 0.0),
    ):
        self.balance = 1_000.0
        self.symbol = symbol
        self.active_symbols = [symbol]
        self.open_orders: dict[str, list[dict]] = {symbol: []}
        self.positions = {
            symbol: {
                "long": {"size": long_position[0], "price": long_position[1]},
                "short": {"size": short_position[0], "price": short_position[1]},
            }
        }
        self.PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "normal"}}
        self.qty_steps = {symbol: 0.1}
        self.price_steps = {symbol: 0.1}
        self.min_qtys = {symbol: 0.0}
        self.min_costs = {symbol: 0.0}
        self.c_mults = {symbol: 1.0}
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
        }
        self._market_price = market_price
        self._load_context_result = (
            {symbol: market_price},
            {"long": {symbol: market_price}, "short": {symbol: market_price}},
            {"long": {symbol: 0.0}, "short": {symbol: 0.0}},
        )
        self._bp_defaults = {
            "entry_grid_double_down_factor": 0.0,
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
            "risk_wel_enforcer_threshold": 0.0,
            "entry_volatility_ema_span_hours": 0.0,
        }
        self._bp_overrides: dict[tuple[str, str, str], float] = {}
        self._bot_value_overrides: dict[tuple[str, str], float] = {}
        self._live_values = {"price_distance_threshold": 1.0, "market_orders_allowed": False}

    def set_bp_override(self, pside: str, key: str, value: float):
        self._bp_overrides[(pside, self.symbol, key)] = value

    def set_bot_value_override(self, pside: str, key: str, value: float):
        self._bot_value_overrides[(pside, key)] = value

    def set_live_value(self, key: str, value: float):
        self._live_values[key] = value

    def live_value(self, key: str):
        return self._live_values.get(key, 0.0)

    def bp(self, pside: str, key: str, symbol: str | None = None):
        symbol = symbol or self.symbol
        return self._bp_overrides.get((pside, symbol, key), self._bp_defaults.get(key, 0.0))

    def bot_value(self, pside: str, key: str):
        return self._bot_value_overrides.get((pside, key), 0.0)

    async def _load_price_context(self):
        return self._load_context_result

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

    async def fake_calc_ideal_orders(self, allow_unstuck=True):
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

    async def fake_calc_ideal_orders(self, allow_unstuck=True):
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

    async def fake_calc_ideal_orders(self, allow_unstuck=True):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert not to_cancel
    assert [order["price"] for order in to_create] == [101.0, 95.0]


@pytest.mark.asyncio
async def test_only_one_unstuck_order_survives(monkeypatch):
    symbols = ["BTC/USDT", "ETH/USDT"]
    market_prices = {symbol: 100.0 for symbol in symbols}
    bot = OrchestrationBot(market_prices)
    for symbol in symbols:
        bot.register_symbol(symbol)
        bot.open_orders[symbol] = []

    unstuck_type = "close_unstuck_long"
    ideal = {
        symbols[0]: [
            _make_order(symbols[0], "sell", "long", 1.0, 101.0, unstuck_type, reduce_only=True)
        ],
        symbols[1]: [
            _make_order(symbols[1], "sell", "long", 1.0, 101.0, unstuck_type, reduce_only=True)
        ],
    }

    async def fake_calc_ideal_orders(self, allow_unstuck=True):
        return ideal

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert not to_cancel
    assert len(to_create) == 1
    remaining = to_create[0]
    assert remaining["symbol"] == symbols[0]
    assert remaining["custom_id"].endswith(f"0x{pbr.order_type_snake_to_id(unstuck_type):04x}")


@pytest.mark.asyncio
async def test_calc_ideal_orders_includes_closes_and_entries(monkeypatch):
    symbol = "BTC/USDT"
    bot = IdealOrdersBot(symbol, market_price=100.0, long_position=(1.0, 100.0))
    bot.set_bp_override("long", "entry_initial_qty_pct", 0.1)
    bot.set_bot_value_override("long", "total_wallet_exposure_limit", 1.0)
    bot.set_bot_value_override("long", "n_positions", 1.0)

    entry_type = "entry_grid_normal_long"
    close_type = "close_grid_long"

    monkeypatch.setattr(
        pbr,
        "calc_entries_long_py",
        lambda *args, **kwargs: [
            (1.0, 95.0, pbr.order_type_snake_to_id(entry_type)),
        ],
    )
    monkeypatch.setattr(pbr, "calc_entries_short_py", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        pbr,
        "calc_closes_long_py",
        lambda *args, **kwargs: [
            (-1.0, 105.0, pbr.order_type_snake_to_id(close_type)),
        ],
    )
    monkeypatch.setattr(pbr, "calc_closes_short_py", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        pbr,
        "gate_entries_by_twel_py",
        lambda *args, **kwargs: [(0, 1.0, 95.0, pbr.order_type_snake_to_id(entry_type))],
    )
    monkeypatch.setattr(pbr, "calc_twel_enforcer_orders_py", lambda *args, **kwargs: [])

    orders = await bot.calc_ideal_orders()
    assert symbol in orders
    snake_types = {
        snake_of(try_decode_type_id_from_custom_id(order["custom_id"])) for order in orders[symbol]
    }
    assert close_type in snake_types
    assert entry_type in snake_types
    entry_orders = [
        order
        for order in orders[symbol]
        if snake_of(try_decode_type_id_from_custom_id(order["custom_id"])) == entry_type
    ]
    assert entry_orders and entry_orders[0]["reduce_only"] is False


@pytest.mark.asyncio
async def test_twel_entry_gating_appends_gated_entries(monkeypatch):
    symbol = "BTC/USDT"
    bot = IdealOrdersBot(symbol, market_price=100.0)
    bot.set_bot_value_override("long", "total_wallet_exposure_limit", 1.0)
    bot.set_bot_value_override("long", "n_positions", 1.0)

    def fake_build_base_orders(self, *_args, **_kwargs):
        return BaseOrderPlan(
            close_candidates=defaultdict(list, {symbol: []}),
            entry_candidates={
                "long": [
                    {
                        "idx": 0,
                        "qty": 0.5,
                        "price": 99.0,
                        "qty_step": 0.1,
                        "min_qty": 0.0,
                        "min_cost": 0.0,
                        "c_mult": 1.0,
                        "market_price": 100.0,
                        "order_type_id": pbr.order_type_snake_to_id("entry_grid_normal_long"),
                    }
                ],
                "short": [],
            },
            entry_index_map={"long": {0: symbol}, "short": {}},
        )

    bot._build_base_orders = types.MethodType(fake_build_base_orders, bot)
    bot._load_context_result = (
        {symbol: 100.0},
        {"long": {symbol: 100.0}, "short": {symbol: 100.0}},
        {"long": {symbol: 0.0}, "short": {symbol: 0.0}},
    )

    gated_entry_type = "entry_grid_cropped_long"
    monkeypatch.setattr(
        pbr,
        "gate_entries_by_twel_py",
        lambda *args, **kwargs: [(0, 0.5, 99.0, pbr.order_type_snake_to_id(gated_entry_type))],
    )
    monkeypatch.setattr(pbr, "calc_twel_enforcer_orders_py", lambda *args, **kwargs: [])

    orders = await bot.calc_ideal_orders()
    assert symbol in orders
    entry_orders = [
        order
        for order in orders[symbol]
        if snake_of(try_decode_type_id_from_custom_id(order["custom_id"])) == gated_entry_type
    ]
    assert entry_orders
    assert entry_orders[0]["price"] == pytest.approx(99.0)


@pytest.mark.asyncio
async def test_twel_enforcer_injects_reduce_only(monkeypatch):
    symbol = "BTC/USDT"
    bot = IdealOrdersBot(symbol, market_price=100.0, long_position=(2.0, 100.0))
    bot.set_bot_value_override("long", "risk_twel_enforcer_threshold", 1.0)
    bot.set_bot_value_override("long", "total_wallet_exposure_limit", 0.5)
    bot.set_bot_value_override("long", "n_positions", 1.0)

    def fake_build_base_orders(self, *_args, **_kwargs):
        return BaseOrderPlan(
            close_candidates=defaultdict(list, {symbol: []}),
            entry_candidates={"long": [], "short": []},
            entry_index_map={"long": {}, "short": {}},
        )

    bot._build_base_orders = types.MethodType(fake_build_base_orders, bot)
    bot._load_context_result = (
        {symbol: 100.0},
        {"long": {symbol: 100.0}, "short": {symbol: 100.0}},
        {"long": {symbol: 0.0}, "short": {symbol: 0.0}},
    )

    reduce_type = "close_auto_reduce_twel_long"

    def fake_enforcer(side, threshold, total_wel, n_positions, balance, payload, skip_idx):
        return [
            (0, -3.0, 99.0, pbr.order_type_snake_to_id(reduce_type)),
        ]

    monkeypatch.setattr(pbr, "gate_entries_by_twel_py", lambda *args, **kwargs: [])
    monkeypatch.setattr(pbr, "calc_twel_enforcer_orders_py", fake_enforcer)

    orders = await bot.calc_ideal_orders()
    assert symbol in orders
    reduce_orders = [
        order
        for order in orders[symbol]
        if snake_of(try_decode_type_id_from_custom_id(order["custom_id"])) == reduce_type
    ]
    assert reduce_orders
    reduce_order = reduce_orders[0]
    assert reduce_order["reduce_only"] is True
    assert reduce_order["qty"] == pytest.approx(2.0)
