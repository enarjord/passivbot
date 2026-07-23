import logging
import types

import pytest
import passivbot_rust as pbr
from passivbot import Passivbot
from exchanges.ccxt_bot import CCXTBot
from runtime_identity import RuntimeIdentity


TEST_RUNTIME_IDENTITY = RuntimeIdentity(
    schema_version=1,
    run_id="a" * 32,
    started_at_ms=1_700_000_000_000,
    passivbot_version="test",
    python_git_commit="b" * 40,
    python_git_dirty=False,
    config_sha256="c" * 64,
    rust_crate_version="test",
    rust_source_sha256="d" * 64,
    rust_artifact_sha256="e" * 64,
)


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
        self._last_action_summary = {}
        self.action_str_max_len = len("posting order")

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
    exchange_id: str | None = None,
) -> dict:
    order_type_id = pbr.order_type_snake_to_id(order_type)
    order = {
        "symbol": symbol,
        "side": side,
        "position_side": position_side,
        "qty": qty,
        "price": price,
        "reduce_only": reduce_only,
        "custom_id": f"order-0x{order_type_id:04x}",
        "type": order_kind,
        "pb_order_type": order_type,
    }
    if exchange_id is not None:
        order["id"] = exchange_id
    return order


def test_finalize_reduce_only_orders_keeps_compatible_unstuck_and_trailing_close():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.positions[symbol]["long"] = {"size": 95.60, "price": 100.0}
    orders = {
        symbol: [
            _make_order(
                symbol,
                "sell",
                "long",
                1.95,
                99.0,
                "close_unstuck_long",
                reduce_only=True,
            ),
            _make_order(
                symbol,
                "sell",
                "long",
                26.24,
                101.0,
                "close_trailing_long",
                reduce_only=True,
            ),
        ]
    }

    finalized = bot._finalize_reduce_only_orders(orders, {symbol: 100.0})

    assert [(order["pb_order_type"], order["qty"]) for order in finalized[symbol]] == [
        ("close_unstuck_long", 1.95),
        ("close_trailing_long", 26.24),
    ]


def test_finalize_reduce_only_orders_trims_ordinary_before_protective_reducer():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.positions[symbol]["long"] = {"size": 20.0, "price": 100.0}
    orders = {
        symbol: [
            _make_order(
                symbol,
                "sell",
                "long",
                1.95,
                99.0,
                "close_unstuck_long",
                reduce_only=True,
            ),
            _make_order(
                symbol,
                "sell",
                "long",
                26.24,
                101.0,
                "close_trailing_long",
                reduce_only=True,
            ),
        ]
    }

    finalized = bot._finalize_reduce_only_orders(orders, {symbol: 100.0})
    by_type = {order["pb_order_type"]: order["qty"] for order in finalized[symbol]}

    assert by_type["close_unstuck_long"] == 1.95
    assert by_type["close_trailing_long"] == 18.05
    assert sum(by_type.values()) == 20.0


def test_coin_hsl_pending_replay_mode_override_is_pair_scoped():
    bot = Passivbot.__new__(Passivbot)
    bot.hsl = {
        "long": {"orange_tier_mode": "graceful_stop"},
        "short": {"orange_tier_mode": "graceful_stop"},
    }
    bot._runtime_forced_modes = {
        "long": {"BTC/USDT:USDT": "panic", "ETH/USDT:USDT": "panic"},
        "short": {},
    }
    bot._equity_hard_stop_coin_replay_pending_pairs = {
        ("long", "BTC/USDT:USDT"),
        ("long", "MANUAL/USDT:USDT"),
    }
    bot._equity_hard_stop_enabled = lambda pside=None: True
    bot._equity_hard_stop_signal_mode = lambda: "coin"
    bot._hsl_state = lambda pside: {"halted": False}
    bot._equity_hard_stop_runtime_red_latched = lambda pside: False
    bot._equity_hard_stop_runtime_tier = lambda pside: "green"
    bot.config_get = lambda path, symbol=None: (
        "manual" if symbol == "MANUAL/USDT:USDT" else None
    )
    bot.markets_dict = {
        "BTC/USDT:USDT": {"active": True},
        "ETH/USDT:USDT": {"active": True},
        "MANUAL/USDT:USDT": {"active": True},
    }
    bot.ineligible_symbols = {}
    bot._apply_ignored_coin_mode = lambda pside, symbol, mode=None: mode

    assert (
        bot._orchestrator_mode_override("long", "BTC/USDT:USDT")
        == "graceful_stop"
    )
    assert bot.get_forced_PB_mode("long", "BTC/USDT:USDT") == "graceful_stop"
    assert bot._orchestrator_mode_override("short", "BTC/USDT:USDT") is None
    assert bot._orchestrator_mode_override("long", "ETH/USDT:USDT") == "panic"
    assert bot.get_forced_PB_mode("long", "ETH/USDT:USDT") == "panic"
    assert bot._orchestrator_mode_override("long", "MANUAL/USDT:USDT") == "manual"
    assert bot.get_forced_PB_mode("long", "MANUAL/USDT:USDT") == "manual"


@pytest.mark.parametrize(
    ("order_type", "qty", "position_side", "position_size"),
    [
        ("entry_ema_anchor_long", 0.1, "long", 0.0),
        ("close_ema_anchor_long", -0.1, "long", 1.0),
        ("entry_ema_anchor_short", -0.1, "short", 0.0),
        ("close_ema_anchor_short", 0.1, "short", -1.0),
    ],
)
def test_ema_anchor_limit_orders_route_to_ccxt_post_only_params(
    order_type, qty, position_side, position_size
):
    symbol = "TEST/USDT:USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.positions[symbol][position_side]["size"] = position_size
    order_type_id = pbr.order_type_snake_to_id(order_type)
    ideal_orders = {
        symbol: [(qty, 100.0, order_type, order_type_id, "limit", "ordinary")]
    }

    orders_by_symbol, _ = bot._to_executable_orders(ideal_orders, {symbol: 100.0})
    [order] = orders_by_symbol[symbol]

    assert order["pb_order_type"] == order_type
    assert order["type"] == "limit"
    assert order["position_side"] == position_side

    ccxt_bot = CCXTBot.__new__(CCXTBot)
    ccxt_bot.config = {"live": {"time_in_force": "post_only"}}

    params = ccxt_bot._build_order_params(order)

    assert params["postOnly"] is True


def test_startup_banner_warns_when_market_orders_allowed(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.runtime_identity = TEST_RUNTIME_IDENTITY
    bot.user = "hyperliquid_pf1"
    bot.exchange = "hyperliquid"
    live_values = {
        "market_orders_allowed": True,
        "market_order_near_touch_threshold": 0.001,
    }
    bot.live_value = lambda key: live_values.get(key)
    bot.bot_value = lambda pside, key: {
        ("long", "total_wallet_exposure_limit"): 1.5,
        ("short", "total_wallet_exposure_limit"): 0.0,
        ("long", "n_positions"): 3,
        ("short", "n_positions"): 0,
    }.get((pside, key), 0.0)

    with caplog.at_level(logging.WARNING):
        Passivbot._log_startup_banner(bot)

    assert any(
        "live market order execution is enabled" in rec.message
        and "market_order_near_touch_threshold=0.001" in rec.message
        for rec in caplog.records
    )


def test_order_summary_marks_market_execution(caplog):
    bot = OrchestrationBot({})
    order = _make_order(
        "HYPE/USDC:USDC",
        "sell",
        "long",
        496.4,
        58.383,
        "close_grid_long",
        reduce_only=True,
        order_kind="market",
    )

    with caplog.at_level(logging.INFO):
        bot._log_order_action_summary({"HYPE/USDC:USDC": [order]}, "post")

    assert any(
        "[order]" in rec.message
        and "close_grid_long" in rec.message
        and "exec=market" in rec.message
        for rec in caplog.records
    )


def test_market_execution_notice_is_not_suppressed(caplog):
    bot = OrchestrationBot({})
    order = _make_order(
        "HYPE/USDC:USDC",
        "sell",
        "long",
        496.4,
        58.383,
        "close_grid_long",
        reduce_only=True,
        order_kind="market",
    )

    with caplog.at_level(logging.INFO):
        bot._log_market_execution_notice(order, context="plan_sync")
        bot._log_market_execution_notice(order, context="plan_sync")

    records = [rec for rec in caplog.records if "MARKET order submission" in rec.message]
    assert len(records) == 2
    assert all("pb_type=close_grid_long" in rec.message for rec in records)


def test_base_did_create_order_rejects_terminal_statuses():
    bot = Passivbot.__new__(Passivbot)

    assert bot.did_create_order({"id": "open-1", "status": "open"})
    assert not bot.did_create_order({"id": "", "status": "open"})
    assert not bot.did_create_order({"id": "reject-1", "status": "rejected"})
    assert not bot.did_create_order({"id": "cancel-1", "info": {"status": "canceled"}})
    assert not bot.did_create_order({"id": "expire-1", "info": {"ordStatus": "EXPIRED"}})


@pytest.mark.asyncio
async def test_calc_orders_to_cancel_and_create_reconciles_orders(monkeypatch):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)

    matching_type = "entry_grid_normal_long"
    new_type = "entry_grid_cropped_long"

    bot.open_orders[symbol] = [
        {
            "id": "matching-order",
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "remaining": 1.0,
            "price": 100.0,
            "reduceOnly": False,
            "type": "limit",
            "custom_id": f"order-0x{pbr.order_type_snake_to_id(matching_type):04x}",
        },
        {
            "id": "stale-order",
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 0.5,
            "remaining": 0.5,
            "price": 98.0,
            "reduceOnly": False,
            "type": "limit",
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
async def test_calc_orders_preserves_orders_when_trailing_anchor_unavailable():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}

    bot.open_orders[symbol] = [
        {
            "symbol": symbol,
            "side": "sell",
            "position_side": "long",
            "qty": 1.0,
            "price": 101.0,
            "custom_id": "order-0x0004",
        }
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    103.0,
                    "close_grid_long",
                    reduce_only=True,
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert to_cancel == []
    assert to_create == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unavailable_reason",
    [
        "missing_trailing_candles",
        "missing_position_change_anchor",
        "candle_fetch_failed",
    ],
)
async def test_calc_orders_allows_panic_close_when_trailing_unavailable(
    unavailable_reason,
):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: [unavailable_reason]
    }

    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_grid_normal_long",
            exchange_id="entry-long",
        ),
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            101.0,
            "close_grid_long",
            reduce_only=True,
            exchange_id="close-long",
        ),
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    100.0,
                    "close_panic_long",
                    reduce_only=True,
                    order_kind="market",
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [(order["side"], order["position_side"], order["price"]) for order in to_cancel] == [
        ("buy", "long", 99.0),
        ("sell", "long", 101.0),
    ]
    assert [order["pb_order_type"] for order in to_create] == ["close_panic_long"]


def test_no_silent_execution_type_defaults_in_rust_order_conversion():
    # Codex P1 regression on the exec-type fail-loud contract: the Rust JSON
    # conversion sites must not default a missing execution_type (a silent
    # default upstream would defeat the reconciler's fail-loud guard and
    # could downgrade a panic market close to a limit order). Any .get with a
    # default on the execution_type key in the live modules is an offender;
    # defaultless .get (classification of exchange-side orders) is allowed.
    import ast
    from pathlib import Path

    offenders = []
    for rel in ("src/passivbot.py", "src/live/reconciler.py"):
        tree = ast.parse(Path(rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "execution_type"
                and len(node.args) > 1
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], offenders


@pytest.mark.asyncio
async def test_calc_protective_panic_reconciles_when_active_symbols_stale():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.active_symbols = []
    bot.positions[symbol]["long"]["size"] = 1.0

    async def fake_protective_ideal(self):
        self._protective_panic_reconcile_symbols = [symbol]
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    100.0,
                    "close_panic_long",
                    reduce_only=True,
                    order_kind="market",
                )
            ]
        }

    bot.calc_protective_panic_ideal_orders_orchestrator = types.MethodType(
        fake_protective_ideal, bot
    )

    to_cancel, to_create = await bot.calc_protective_panic_orders_to_cancel_and_create()

    assert to_cancel == []
    assert [order["pb_order_type"] for order in to_create] == ["close_panic_long"]


@pytest.mark.asyncio
async def test_protective_panic_reconciliation_preserves_healthy_pside_orders():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.active_symbols = []
    bot.positions[symbol]["long"]["size"] = 1.0
    bot.positions[symbol]["short"]["size"] = -1.0
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_grid_normal_long",
            exchange_id="entry-long",
        ),
        _make_order(
            symbol,
            "sell",
            "short",
            1.0,
            101.0,
            "entry_grid_normal_short",
            exchange_id="entry-short",
        ),
        _make_order(
            symbol,
            "buy",
            "short",
            1.0,
            98.0,
            "close_grid_short",
            reduce_only=True,
            exchange_id="close-short",
        ),
    ]

    async def fake_protective_ideal(self):
        self._protective_panic_reconcile_symbols = [symbol]
        self._protective_panic_reconcile_psides_by_symbol = {symbol: {"long"}}
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    100.0,
                    "close_panic_long",
                    reduce_only=True,
                    order_kind="market",
                )
            ]
        }

    bot.calc_protective_panic_ideal_orders_orchestrator = types.MethodType(
        fake_protective_ideal, bot
    )

    to_cancel, to_create = await bot.calc_protective_panic_orders_to_cancel_and_create()

    assert [(order["position_side"], order["side"], order["price"]) for order in to_cancel] == [
        ("long", "buy", 99.0)
    ]
    assert [order["pb_order_type"] for order in to_create] == ["close_panic_long"]


@pytest.mark.asyncio
async def test_protective_panic_reconciliation_ignores_stale_normal_mode_filter():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot.active_symbols = []
    bot.PB_modes["long"][symbol] = "manual"
    bot.positions[symbol]["long"]["size"] = 1.0
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_grid_normal_long",
            exchange_id="entry-long",
        )
    ]

    async def fake_protective_ideal(self):
        self._protective_panic_reconcile_symbols = [symbol]
        self._protective_panic_reconcile_psides_by_symbol = {symbol: {"long"}}
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    100.0,
                    "close_panic_long",
                    reduce_only=True,
                    order_kind="market",
                )
            ]
        }

    bot.calc_protective_panic_ideal_orders_orchestrator = types.MethodType(
        fake_protective_ideal, bot
    )

    to_cancel, to_create = await bot.calc_protective_panic_orders_to_cancel_and_create()

    assert [(order["position_side"], order["side"], order["price"]) for order in to_cancel] == [
        ("long", "buy", 99.0)
    ]
    assert [order["pb_order_type"] for order in to_create] == ["close_panic_long"]


@pytest.mark.asyncio
async def test_protective_panic_ideal_does_not_fetch_ticker_for_cancel_only_symbol():
    symbol = "DOGE/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {
        symbol: {
            "long": {"size": 0.0, "price": 0.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.open_orders = {
        symbol: [
            _make_order(
                symbol,
                "buy",
                "long",
                1.0,
                0.1,
                "entry_grid_normal_long",
                exchange_id="entry-long",
            )
        ]
    }

    async def fail_market_snapshot_fetch(symbols):
        raise AssertionError(f"cancel-only symbols must not require market snapshots: {symbols}")

    bot._get_orchestrator_market_snapshots = fail_market_snapshot_fetch
    bot._orchestrator_mode_override = lambda pside, sym: "panic" if pside == "long" else None

    ideal = await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot)

    assert ideal == {}
    assert bot._protective_panic_reconcile_symbols == [symbol]


@pytest.mark.asyncio
async def test_red_supervisor_uses_protective_refresh_and_order_plan():
    calls = []

    class FakeBot:
        _equity_hard_stop_supervisor_running = False
        stop_signal_received = False

        def __init__(self):
            self.state = {
                "red_flat_confirmations": 0,
                "last_red_progress": None,
                "halted": False,
            }

        def _hsl_psides(self):
            return ("long",)

        def _hsl_state(self, pside):
            assert pside == "long"
            return self.state

        def _equity_hard_stop_enabled(self, pside=None):
            return True

        def _equity_hard_stop_runtime_red_latched(self, pside):
            return True

        async def refresh_protective_authoritative_state(self):
            calls.append("protective_refresh")
            return True

        async def update_pos_oos_pnls_ohlcvs(self):
            raise AssertionError("RED supervisor must not require normal market refresh")

        def _equity_hard_stop_count_open_positions(self, pside):
            return 1

        def _equity_hard_stop_count_blocking_open_orders(self, pside):
            return 1, 0

        def _equity_hard_stop_log_red_progress(self, *args):
            calls.append("log_progress")

        def _equity_hard_stop_set_red_runtime_forced_modes(self, pside):
            calls.append("force_panic")

        def _equity_hard_stop_refresh_halted_runtime_forced_modes(self):
            calls.append("refresh_halted")

        async def calc_protective_panic_orders_to_cancel_and_create(self):
            calls.append("protective_plan")
            return [{"symbol": "BTC/USDT:USDT"}], [{"symbol": "BTC/USDT:USDT"}]

        async def execute_order_plan_to_exchange(
            self,
            to_cancel,
            to_create,
            *,
            configure_creations=True,
        ):
            calls.append(
                (
                    "execute_plan",
                    list(to_cancel),
                    list(to_create),
                    configure_creations,
                )
            )
            self.stop_signal_received = True

        async def execute_to_exchange(self, *args, **kwargs):
            raise AssertionError("RED supervisor must not use normal execution cycle")

        def live_value(self, key):
            assert key == "execution_delay_seconds"
            return 0.0

    bot = FakeBot()
    await Passivbot._equity_hard_stop_run_red_supervisor(bot)

    assert calls[:4] == [
        "protective_refresh",
        "log_progress",
        "force_panic",
        "refresh_halted",
    ]
    assert calls[4] == "protective_plan"
    assert calls[5][0] == "execute_plan"
    assert calls[5][3] is False
    assert bot._equity_hard_stop_supervisor_running is False


@pytest.mark.asyncio
async def test_red_supervisor_refreshes_late_flatten_fill_and_exits():
    events = [
        {"timestamp": 90_000, "pside": "long", "symbol": "OLD"},
    ]

    class FakeBot:
        _equity_hard_stop_supervisor_running = False
        _equity_hard_stop_cooldown_log_interval_ms = 60_000
        stop_signal_received = False
        _equity_hard_stop_latest_flatten_fill_timestamp_optional_ms = (
            Passivbot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms
        )
        _equity_hard_stop_defer_missing_flatten_fill = (
            Passivbot._equity_hard_stop_defer_missing_flatten_fill
        )
        _equity_hard_stop_flatten_fill_timestamp_with_refresh = (
            Passivbot._equity_hard_stop_flatten_fill_timestamp_with_refresh
        )

        def __init__(self):
            self.state = {
                "red_flat_confirmations": 0,
                "last_red_progress": None,
                "halted": False,
                "pending_red_since_ms": 120_000,
                "pending_stop_event": None,
                "last_missing_flatten_fill_log_ms": 0,
                "last_missing_flatten_fill_refresh_ms": 0,
            }
            self._pnls_manager = types.SimpleNamespace(get_events=lambda: events)
            self.refresh_sources = []

        def _hsl_psides(self):
            return ("long",)

        def _hsl_state(self, pside):
            return self.state

        def _equity_hard_stop_enabled(self, pside=None):
            return True

        def _equity_hard_stop_runtime_red_latched(self, pside):
            return True

        async def refresh_protective_authoritative_state(self):
            return True

        async def update_pnls(self, *, source, since_ms=None):
            self.refresh_sources.append(source)
            assert since_ms == 120_000
            events.append(
                {"timestamp": 170_000, "pside": "long", "symbol": "BTC/USDT:USDT"}
            )
            return True

        def get_exchange_time(self):
            return 180_000

        def _equity_hard_stop_count_open_positions(self, pside):
            return 0

        def _equity_hard_stop_count_blocking_open_orders(self, pside):
            return 0, 0

        async def _equity_hard_stop_compute_stop_event(self, pside, stop_ts_ms):
            return {"stop_event_timestamp_ms": stop_ts_ms}

        async def _equity_hard_stop_finalize_red_stop(self, pside, stop_event, **kwargs):
            assert stop_event["stop_event_timestamp_ms"] == 170_000
            self.state["halted"] = True

        def _equity_hard_stop_log_red_progress(self, *args):
            pass

        def _equity_hard_stop_signal_mode(self):
            return "pside"

        async def _calc_upnl_sum_strict(self, pside=None):
            return 0.0

        def get_raw_balance(self):
            return 100.0

        def _equity_hard_stop_realized_pnl_now(self, pside=None):
            return 0.0

        def _equity_hard_stop_apply_sample(self, *args, **kwargs):
            return {"red_active_now": True}

        def _equity_hard_stop_set_red_paused_runtime_forced_modes(self, pside):
            pass

        def _equity_hard_stop_set_red_runtime_forced_modes(self, pside):
            pass

        def _equity_hard_stop_refresh_halted_runtime_forced_modes(self):
            pass

        async def calc_protective_panic_orders_to_cancel_and_create(self):
            return [], []

        async def execute_order_plan_to_exchange(self, *args, **kwargs):
            pass

        def live_value(self, key):
            return 0.0

    bot = FakeBot()

    await Passivbot._equity_hard_stop_run_red_supervisor(bot)

    assert bot.state["halted"] is True
    assert bot.state["red_flat_confirmations"] == 2
    assert bot.refresh_sources == ["hsl_flatten_confirmation"]
    assert bot._equity_hard_stop_supervisor_running is False


@pytest.mark.asyncio
async def test_coin_red_supervisor_refreshes_late_cooldown_repanic_fill():
    symbol = "BTC/USDT:USDT"
    events = [
        {"timestamp": 90_000, "pside": "long", "symbol": symbol},
        {
            "timestamp": 150_000,
            "pside": "long",
            "symbol": symbol,
            "action": "increase",
            "qty": 1.0,
        },
        {
            "timestamp": 165_000,
            "pside": "long",
            "symbol": symbol,
            "action": "increase",
            "qty": 1.0,
        },
    ]

    class FakeBot:
        _equity_hard_stop_supervisor_running = False
        _equity_hard_stop_cooldown_log_interval_ms = 60_000
        stop_signal_received = False
        _equity_hard_stop_latest_flatten_fill_timestamp_optional_ms = (
            Passivbot._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms
        )
        _equity_hard_stop_defer_missing_flatten_fill = (
            Passivbot._equity_hard_stop_defer_missing_flatten_fill
        )
        _equity_hard_stop_flatten_fill_timestamp_with_refresh = (
            Passivbot._equity_hard_stop_flatten_fill_timestamp_with_refresh
        )

        def __init__(self):
            self.state = {
                "red_flat_confirmations": 0,
                "halted": True,
                "pending_red_since_ms": None,
                "pending_stop_event": None,
                "last_stop_event": {"stop_event_timestamp_ms": 120_000},
                "cooldown_repanic_reset_pending": True,
                "cooldown_repanic_since_ms": 160_000,
                "cooldown_repanic_start_sizes": {symbol: 1.0},
                "last_missing_flatten_fill_log_ms": 0,
                "last_missing_flatten_fill_refresh_ms": 0,
            }
            self._equity_hard_stop_coin = {"long": {symbol: self.state}}
            self._pnls_manager = types.SimpleNamespace(get_events=lambda: events)
            self.refresh_sources = []

        def _hsl_psides(self):
            return ("long",)

        def _hsl_coin_state(self, pside, requested_symbol):
            assert requested_symbol == symbol
            return self.state

        def _equity_hard_stop_coin_needs_panic_supervision(
            self, pside, requested_symbol, state
        ):
            return bool(state["cooldown_repanic_reset_pending"])

        async def refresh_protective_authoritative_state(self):
            return True

        async def update_pnls(self, *, source, since_ms=None):
            self.refresh_sources.append((source, since_ms))
            events.append(
                {
                    "timestamp": 170_000,
                    "pside": "long",
                    "symbol": symbol,
                    "action": "decrease",
                    "qty": 2.0,
                }
            )
            return True

        def get_exchange_time(self):
            return 180_000

        def _equity_hard_stop_has_open_position_symbol(self, pside, requested_symbol):
            return False

        def _equity_hard_stop_count_blocking_open_orders_symbol(
            self, pside, requested_symbol
        ):
            return 0, 0

        async def _equity_hard_stop_refresh_coin_cooldown_after_repanic(
            self, pside, requested_symbol, now_ms
        ):
            assert (
                self._equity_hard_stop_latest_flatten_fill_timestamp_optional_ms(
                    pside,
                    symbol=requested_symbol,
                    since_ms=160_000,
                    replay_start_sizes={symbol: 1.0},
                )
                == 170_000
            )
            self.state["cooldown_repanic_reset_pending"] = False
            return True

        def _equity_hard_stop_apply_coin_sample(self, *args, **kwargs):
            return {"red_active_now": True}

        async def _calc_upnl_sum_strict(self, *args):
            return 0.0

        def get_raw_balance(self):
            return 100.0

        def _equity_hard_stop_set_coin_runtime_forced_mode(self, *args):
            pass

        async def calc_protective_panic_orders_to_cancel_and_create(self):
            return [], []

        async def execute_order_plan_to_exchange(self, *args, **kwargs):
            pass

        def live_value(self, key):
            return 0.0

    bot = FakeBot()

    await Passivbot._equity_hard_stop_run_coin_red_supervisor(bot)

    assert bot.state["cooldown_repanic_reset_pending"] is False
    assert bot.refresh_sources == [("hsl_flatten_confirmation", 160_000)]
    assert bot._equity_hard_stop_supervisor_running is False


@pytest.mark.asyncio
async def test_calc_orders_blocks_entry_creates_when_trailing_candles_pending():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["missing_trailing_candles"]
    }

    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "long",
            1.0,
            99.0,
            "entry_grid_normal_long",
            exchange_id="pending-entry-long",
        ),
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            101.0,
            "close_grid_long",
            reduce_only=True,
            exchange_id="pending-close-long",
        ),
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "buy",
                    "long",
                    1.0,
                    98.0,
                    "entry_grid_normal_long",
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [(order["side"], order["position_side"], order["price"]) for order in to_cancel] == [
        ("buy", "long", 99.0)
    ]
    assert to_create == []


@pytest.mark.asyncio
async def test_calc_orders_retires_stale_trailing_close_when_trailing_candles_pending():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["missing_trailing_candles"]
    }

    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            101.0,
            "close_trailing_long",
            reduce_only=True,
            exchange_id="stale-trailing-long",
        )
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    102.0,
                    "close_grid_long",
                    reduce_only=True,
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [bot._resolve_pb_order_type(order) for order in to_cancel] == [
        "close_trailing_long"
    ]
    assert [order["pb_order_type"] for order in to_create] == ["close_grid_long"]


@pytest.mark.asyncio
async def test_calc_orders_blocks_new_trailing_close_when_trailing_candles_pending():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["missing_trailing_candles"]
    }
    bot._orchestrator_trailing_unavailable_psides = {symbol: ["long"]}

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    99.0,
                    "close_trailing_long",
                    reduce_only=True,
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert to_cancel == []
    assert to_create == []


@pytest.mark.asyncio
async def test_calc_orders_trailing_unavailable_is_position_side_scoped():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["missing_trailing_candles"]
    }
    bot._orchestrator_trailing_unavailable_psides = {symbol: ["long"]}

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    99.0,
                    "close_trailing_long",
                    reduce_only=True,
                ),
                _make_order(
                    symbol,
                    "buy",
                    "short",
                    1.0,
                    101.0,
                    "close_trailing_short",
                    reduce_only=True,
                ),
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert to_cancel == []
    assert [order["pb_order_type"] for order in to_create] == ["close_trailing_short"]


@pytest.mark.asyncio
async def test_calc_orders_retires_trailing_close_during_fetch_failure():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["candle_fetch_failed"]
    }
    bot._orchestrator_trailing_unavailable_psides = {symbol: ["long"]}
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            101.0,
            "close_trailing_long",
            reduce_only=True,
            exchange_id="failed-trailing-long",
        )
    ]

    async def fake_calc_ideal_orders(self):
        return {symbol: []}

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [bot._resolve_pb_order_type(order) for order in to_cancel] == [
        "close_trailing_long"
    ]
    assert to_create == []


@pytest.mark.asyncio
async def test_calc_orders_hard_trailing_failure_preserves_unaffected_side_replace():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["bundle_compute_failed"]
    }
    bot._orchestrator_trailing_unavailable_psides = {symbol: ["long"]}
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "buy",
            "short",
            1.0,
            101.0,
            "close_trailing_short",
            reduce_only=True,
            exchange_id="unaffected-trailing-short",
        )
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "buy",
                    "short",
                    1.0,
                    102.0,
                    "close_trailing_short",
                    reduce_only=True,
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [(order["position_side"], order["price"]) for order in to_cancel] == [
        ("short", 101.0)
    ]
    assert [(order["position_side"], order["price"]) for order in to_create] == [
        ("short", 102.0)
    ]


@pytest.mark.asyncio
async def test_calc_orders_unavailable_long_panic_preserves_short_replace():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["position_fill_confirmation_pending"]
    }
    bot._orchestrator_trailing_unavailable_psides = {symbol: ["long"]}
    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            99.0,
            "close_grid_long",
            reduce_only=True,
            exchange_id="panic-source-long",
        ),
        _make_order(
            symbol,
            "buy",
            "short",
            1.0,
            101.0,
            "close_trailing_short",
            reduce_only=True,
            exchange_id="replace-source-short",
        ),
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    98.0,
                    "close_panic_long",
                    reduce_only=True,
                ),
                _make_order(
                    symbol,
                    "buy",
                    "short",
                    1.0,
                    102.0,
                    "close_trailing_short",
                    reduce_only=True,
                ),
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert {(order["position_side"], order["price"]) for order in to_cancel} == {
        ("long", 99.0),
        ("short", 101.0),
    }
    assert {(order["position_side"], order["price"]) for order in to_create} == {
        ("long", 98.0),
        ("short", 102.0),
    }


@pytest.mark.asyncio
async def test_calc_orders_allows_same_family_reduce_only_replace_when_trailing_candles_pending():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._orchestrator_trailing_unavailable_symbols = {symbol}
    bot._orchestrator_trailing_unavailable_reasons = {
        symbol: ["missing_trailing_candles"]
    }

    bot.open_orders[symbol] = [
        _make_order(
            symbol,
            "sell",
            "long",
            1.0,
            101.0,
            "close_grid_long",
            reduce_only=True,
            exchange_id="same-family-close-long",
        )
    ]

    async def fake_calc_ideal_orders(self):
        return {
            symbol: [
                _make_order(
                    symbol,
                    "sell",
                    "long",
                    1.0,
                    102.0,
                    "close_grid_long",
                    reduce_only=True,
                )
            ]
        }

    bot.calc_ideal_orders = types.MethodType(fake_calc_ideal_orders, bot)

    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()

    assert [(order["side"], order["position_side"], order["price"]) for order in to_cancel] == [
        ("sell", "long", 101.0)
    ]
    assert [order["pb_order_type"] for order in to_create] == ["close_grid_long"]


def test_to_executable_orders_respects_rust_market_execution_hint():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["market_orders_allowed"] = False

    order_type = "close_unstuck_long"
    order_type_id = pbr.order_type_snake_to_id(order_type)
    ideal = {
        symbol: [
            (-0.5, 100.0, order_type, order_type_id, "market", "risk_critical")
        ]
    }

    orders, _ = bot._to_executable_orders(ideal, {symbol: 100.0})

    assert orders[symbol][0]["type"] == "market"
    assert orders[symbol][0]["execution_priority"] == "risk_critical"


def test_to_executable_orders_respects_rust_limit_execution_hint():
    symbol = "BTC/USDT"
    bot = OrchestrationBot({symbol: 100.0})
    bot.register_symbol(symbol)
    bot._live_values["market_orders_allowed"] = True

    order_type = "close_unstuck_long"
    order_type_id = pbr.order_type_snake_to_id(order_type)
    ideal = {
        symbol: [
            (-0.5, 100.0, order_type, order_type_id, "limit", "risk_critical")
        ]
    }

    orders, _ = bot._to_executable_orders(ideal, {symbol: 100.0})

    assert orders[symbol][0]["type"] == "limit"
    assert orders[symbol][0]["execution_priority"] == "risk_critical"


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
async def test_order_sort_fetch_failure_redacts_diagnostic_and_preserves_original_order(
    caplog, capsys
):
    symbol = "BTC/USDT"
    bot = OrchestrationBot({})
    bot.register_symbol(symbol)
    first = _make_order(symbol, "buy", "long", 0.5, 95.0, "entry_grid_cropped_long")
    second = _make_order(symbol, "buy", "long", 0.5, 101.0, "entry_grid_normal_long")
    orders = [first, second]
    fetch_calls = []
    hostile_detail = (
        "hostile-message https://order-sort.invalid/price?api_key=operator-secret&token=token-123"
    )
    hostile_error = type("ApiTokenCredentialsError", (RuntimeError,), {})(hostile_detail)

    async def fail_get_live_last_prices(symbols, **kwargs):
        fetch_calls.append((symbols, kwargs))
        raise hostile_error

    bot._fetch_market_prices = types.MethodType(Passivbot._fetch_market_prices, bot)
    bot._get_live_last_prices = fail_get_live_last_prices

    with caplog.at_level(logging.WARNING):
        result = await bot._sort_orders_by_market_diff(orders, log_label="to_create")

    diagnostic = next(
        record.getMessage()
        for record in caplog.records
        if "market price lookup failed for order sorting" in record.getMessage()
    )
    captured = capsys.readouterr()
    output = "\n".join((caplog.text, captured.out, captured.err))

    assert result == orders
    assert result is not orders
    assert all(returned is original for returned, original in zip(result, orders))
    assert fetch_calls == [
        (
            {symbol},
            {
                "max_age_ms": 10_000,
                "context": "order_sort",
                "allow_completed_candle_fallback": True,
            },
        )
    ]
    assert diagnostic == (
        "[order] market price lookup failed for order sorting | "
        "symbols=BTC error_type=RuntimeError action=preserve_original_order"
    )
    assert captured.out == ""
    assert captured.err == ""
    for unsafe_value in (
        "hostile-message",
        "ApiTokenCredentialsError",
        "https://order-sort.invalid",
        "api_key",
        "operator-secret",
        "token-123",
        "Traceback",
    ):
        assert unsafe_value not in output
