import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import RateLimitExceeded


class FreshPlanningSnapshot:
    symbols = (
        "BTC/USDT:USDT",
        "CANCEL/USDT:USDT",
        "ETH/USDT:USDT",
        "PENDING/USDT:USDT",
        "READY/USDT:USDT",
    )

    def invalid_details(self, now_ms=None):
        return []


class FreshMarketSnapshotProvider:
    async def get_snapshots(self, symbols, max_age_ms=10_000):
        from market_snapshot import MarketSnapshot
        from utils import utc_ms

        return {
            symbol: MarketSnapshot(
                symbol=symbol,
                bid=1.0,
                ask=1.0,
                last=1.0,
                fetched_ms=utc_ms(),
                source="test",
            )
            for symbol in symbols
        }


class DummyTask:
    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        return self._coro.__await__()


class DummyKucoinCCA:
    def __init__(self):
        self.leverage_calls = []

    async def set_margin_mode(self, **_params):
        return {}

    async def set_leverage(self, **params):
        self.leverage_calls.append(params)
        return params


def make_kucoin_config_bot():
    from exchanges.kucoin import KucoinBot

    bot = KucoinBot.__new__(KucoinBot)
    bot.cca = DummyKucoinCCA()
    bot.hedge_mode = True
    bot.max_leverage = {}
    return bot


@pytest.mark.asyncio
async def test_update_exchange_configs_marks_only_successful_symbols(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        exchange = "bybit"
        active_symbols = ["A", "B"]
        already_updated_exchange_config_symbols = set()
        _health_rate_limits = 0

        def __init__(self):
            self.calls = []
            self._exchange_config_retry_attempts = {}
            self._exchange_config_retry_after_ms = {}

        async def update_exchange_config_by_symbols(self, symbols):
            symbol = symbols[0]
            self.calls.append(symbol)
            if symbol == "A":
                raise Exception("boom")

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = lambda self: False

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    bot = FakeBot()
    await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A", "B"]
    assert bot.already_updated_exchange_config_symbols == {"B"}
    assert bot._exchange_config_retry_attempts["A"] == 1
    assert bot._exchange_config_retry_after_ms["A"] > 0
    assert bot._health_rate_limits == 0


@pytest.mark.asyncio
async def test_update_exchange_configs_does_not_mark_invalid_kucoin_leverage_cap(
    monkeypatch,
):
    import passivbot as pb_mod

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    bot = make_kucoin_config_bot()
    bot.exchange = "kucoin"
    bot.active_symbols = ["BTC/USDT:USDT"]
    bot.already_updated_exchange_config_symbols = set()
    bot._exchange_config_retry_attempts = {}
    bot._exchange_config_retry_after_ms = {}
    bot._health_rate_limits = 0
    bot.max_leverage = {"BTC/USDT:USDT": 0}
    bot.config_get = lambda path, *, symbol=None: 5
    bot._is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception.__get__(
        bot
    )
    bot._exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds.__get__(
        bot
    )
    bot._exchange_config_success_pause_seconds = (
        pb_mod.Passivbot._exchange_config_success_pause_seconds.__get__(bot)
    )
    bot._shutdown_requested = lambda: False

    configured = await pb_mod.Passivbot.update_exchange_configs(bot)

    assert configured == set()
    assert bot.already_updated_exchange_config_symbols == set()
    assert bot.cca.leverage_calls == []
    assert bot._exchange_config_retry_attempts["BTC/USDT:USDT"] == 1
    assert bot._exchange_config_retry_after_ms["BTC/USDT:USDT"] > 0


@pytest.mark.asyncio
async def test_update_exchange_configs_rate_limit_breaks_and_defers_remaining(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        exchange = "bybit"
        active_symbols = ["A", "B"]
        already_updated_exchange_config_symbols = set()
        _health_rate_limits = 0

        def __init__(self):
            self.calls = []
            self._exchange_config_retry_attempts = {}
            self._exchange_config_retry_after_ms = {}

        async def update_exchange_config_by_symbols(self, symbols):
            symbol = symbols[0]
            self.calls.append(symbol)
            if symbol == "A":
                raise RateLimitExceeded("bybit retCode 10006 rate limit")

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = lambda self: False

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    bot = FakeBot()
    await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A"]
    assert bot.already_updated_exchange_config_symbols == set()
    assert bot._exchange_config_retry_attempts["A"] == 1
    assert bot._exchange_config_retry_after_ms["A"] > 0
    assert bot._health_rate_limits == 1


@pytest.mark.asyncio
async def test_update_exchange_configs_retries_failed_symbol_after_backoff(monkeypatch):
    import passivbot as pb_mod

    now_ms = 1_000_000

    class FakeBot:
        exchange = "bybit"
        active_symbols = ["A"]
        already_updated_exchange_config_symbols = set()
        _health_rate_limits = 0

        def __init__(self):
            self.calls = []
            self._exchange_config_retry_attempts = {}
            self._exchange_config_retry_after_ms = {}

        async def update_exchange_config_by_symbols(self, symbols):
            symbol = symbols[0]
            self.calls.append(symbol)
            if len(self.calls) == 1:
                raise Exception("boom")

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = lambda self: False

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)
    monkeypatch.setattr(pb_mod.random, "uniform", lambda a, b: 0.0)

    bot = FakeBot()
    await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A"]
    assert bot.already_updated_exchange_config_symbols == set()
    assert bot._exchange_config_retry_attempts["A"] == 1
    assert bot._exchange_config_retry_after_ms["A"] == now_ms + 5000

    await pb_mod.Passivbot.update_exchange_configs(bot)
    assert bot.calls == ["A"]

    now_ms += 5001
    await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A", "A"]
    assert bot.already_updated_exchange_config_symbols == {"A"}
    assert bot._exchange_config_retry_attempts == {}
    assert bot._exchange_config_retry_after_ms == {}


@pytest.mark.asyncio
async def test_update_exchange_configs_accepts_symbol_subset(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        exchange = "okx"
        active_symbols = ["A", "B", "C"]
        already_updated_exchange_config_symbols = {"B"}
        _health_rate_limits = 0

        def __init__(self):
            self.calls = []
            self._exchange_config_retry_attempts = {}
            self._exchange_config_retry_after_ms = {}

        async def update_exchange_config_by_symbols(self, symbols):
            self.calls.append(symbols[0])

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = lambda self: False

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    bot = FakeBot()
    configured = await pb_mod.Passivbot.update_exchange_configs(bot, ["C", "B", "C"])

    assert bot.calls == ["C"]
    assert bot.already_updated_exchange_config_symbols == {"B", "C"}
    assert configured == {"B", "C"}


@pytest.mark.asyncio
async def test_update_exchange_configs_stops_after_shutdown_signal(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        exchange = "bybit"
        active_symbols = ["A", "B"]
        already_updated_exchange_config_symbols = set()
        _health_rate_limits = 0

        def __init__(self):
            self.calls = []
            self.stop_signal_received = False
            self._exchange_config_retry_attempts = {}
            self._exchange_config_retry_after_ms = {}

        async def update_exchange_config_by_symbols(self, symbols):
            self.calls.append(symbols[0])
            self.stop_signal_received = True

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    async def fake_sleep(_seconds):
        raise AssertionError("shutdown should skip post-success exchange-config sleep")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    bot = FakeBot()
    await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A"]
    assert bot.already_updated_exchange_config_symbols == {"A"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("exchanges.binance", "BinanceBot"),
        ("exchanges.bitget", "BitgetBot"),
        ("exchanges.kucoin", "KucoinBot"),
    ],
)
async def test_exchange_update_config_reraises_hedge_mode_failures(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    bot_cls = getattr(module, class_name)
    bot = bot_cls.__new__(bot_cls)
    bot.cca = SimpleNamespace(set_position_mode=AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await bot.update_exchange_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name", "response"),
    [
        ("exchanges.bitget", "BitgetBot", {"code": "00000", "data": {"posMode": "hedge_mode"}}),
        ("exchanges.kucoin", "KucoinBot", {"code": "200000", "data": {"positionMode": 1}}),
    ],
)
async def test_exchange_update_config_accepts_live_same_mode_success(
    module_name, class_name, response
):
    module = __import__(module_name, fromlist=[class_name])
    bot_cls = getattr(module, class_name)
    bot = bot_cls.__new__(bot_cls)
    bot.cca = SimpleNamespace(set_position_mode=AsyncMock(return_value=response))

    await bot.update_exchange_config()

    bot.cca.set_position_mode.assert_awaited_once_with(True)


@pytest.mark.asyncio
async def test_binance_update_config_accepts_already_hedged_response():
    from exchanges.binance import BinanceBot

    bot = BinanceBot.__new__(BinanceBot)
    bot.cca = SimpleNamespace(
        set_position_mode=AsyncMock(side_effect=Exception('{"code":-4059,"msg":"No need"}'))
    )

    await bot.update_exchange_config()


@pytest.mark.asyncio
async def test_okx_detect_account_config_reraises_unknown_failure():
    from exchanges.okx import OKXBot

    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True
    bot.hedge_mode = True
    bot.cca = SimpleNamespace(
        private_get_account_config=AsyncMock(side_effect=RuntimeError("cfg boom"))
    )

    with pytest.raises(RuntimeError, match="Unable to detect OKX account configuration"):
        await bot._detect_account_config()


@pytest.mark.asyncio
async def test_okx_update_config_reraises_unknown_hedge_mode_failure():
    from exchanges.okx import OKXBot

    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True
    bot.hedge_mode = True
    bot.cca = SimpleNamespace(
        private_get_account_config=AsyncMock(
            return_value={"data": [{"posMode": "long_short_mode"}]}
        ),
        set_position_mode=AsyncMock(side_effect=RuntimeError("hedge boom")),
    )

    with pytest.raises(RuntimeError, match="hedge boom"):
        await bot.update_exchange_config()


@pytest.mark.asyncio
async def test_okx_update_config_verified_net_mode_skips_hedge_switch():
    from exchanges.okx import OKXBot

    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True
    bot.hedge_mode = True
    bot.cca = SimpleNamespace(
        private_get_account_config=AsyncMock(
            return_value={"data": [{"posMode": "net_mode"}]}
        ),
        set_position_mode=AsyncMock(),
    )

    await bot.update_exchange_config()

    assert bot.okx_dual_side is False
    assert bot.hedge_mode is False
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_to_exchange_stops_after_exchange_config_shutdown():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False

        def __init__(self):
            self.stop_signal_received = False
            self.calc_called = False
            self.cancel_called = False
            self.create_called = False
            self.config_symbols = None
            self.debug_mode = False
            self.balance_threshold = 0.0
            self.quote = "USDT"
            self.state_change_detected_by_symbol = set()

        async def execution_cycle(self):
            return None

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = symbols
            self.stop_signal_received = True
            return set()

        async def calc_orders_to_cancel_and_create(self):
            self.calc_called = True
            return [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                }
            ], [
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                }
            ]

        async def execute_cancellations_parent(self, orders):
            self.cancel_called = True
            return []

        def get_raw_balance(self):
            return 1.0

        def order_was_recently_updated(self, order):
            return 0

        async def execute_orders_parent(self, orders):
            self.create_called = True
            return []

        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    result = await pb_mod.Passivbot.execute_to_exchange(bot)

    assert result is None
    assert bot.calc_called
    assert bot.cancel_called
    assert bot.config_symbols == ["ETH/USDT:USDT"]
    assert not bot.create_called


@pytest.mark.asyncio
async def test_execute_to_exchange_allows_cancellations_when_balance_too_low(
    caplog,
):
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False
        balance_threshold = 1.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()

        def __init__(self):
            self.cancel_called = False
            self.create_called = False
            self.config_called = False
            self.execution_scheduled = False

        async def execution_cycle(self):
            return None

        async def calc_orders_to_cancel_and_create(self):
            return [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                }
            ], [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 0.9,
                    "qty": 1.0,
                }
            ]

        def get_raw_balance(self):
            return 0.0

        async def execute_cancellations_parent(self, orders):
            self.cancel_called = True
            return []

        async def update_exchange_configs(self, symbols=None):
            self.config_called = True
            return set(symbols or [])

        async def execute_orders_parent(self, orders):
            self.create_called = True
            return []

        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    with caplog.at_level(logging.INFO):
        await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot.cancel_called
    assert not bot.config_called
    assert not bot.create_called
    assert bot.execution_scheduled is True
    assert any(
        "skipped 1 exposure-increasing order creates" in record.message
        for record in caplog.records
    )
    assert any("allowing 1 cancellations and 0 protective creates" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_execute_to_exchange_allows_reduce_only_create_when_balance_too_low():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False
        balance_threshold = 1.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self.cancel_called = False
            self.config_symbols = None
            self.created_orders = None
            self.execution_scheduled = False
            self._current_planning_snapshot = FreshPlanningSnapshot()
            self.market_snapshot_provider = FreshMarketSnapshotProvider()

        async def execution_cycle(self):
            return None

        async def calc_orders_to_cancel_and_create(self):
            return [], [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "sell",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                    "reduce_only": True,
                    "pb_order_type": "close_panic_long",
                    "type": "market",
                }
            ]

        def get_raw_balance(self):
            return 0.0

        async def execute_cancellations_parent(self, orders):
            self.cancel_called = True
            return []

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = symbols
            return set(symbols or [])

        def order_was_recently_updated(self, order):
            return 0

        async def execute_orders_parent(self, orders):
            self.created_orders = list(orders)
            return []

        def _current_planning_snapshot_invalid_for_creations(self, symbols):
            return []

        async def _get_live_market_snapshots(
            self,
            symbols,
            *,
            max_age_ms=10_000,
            context="live",
            allow_completed_candle_fallback=False,
        ):
            return await self.market_snapshot_provider.get_snapshots(
                symbols, max_age_ms=max_age_ms
            )

        def _live_market_snapshot_max_age_ms(self):
            return 10_000

        def _record_market_snapshot_surface(self, symbols, snapshots):
            return None

        def _market_snapshot_signature_invalid(self, symbols):
            return []

        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot.cancel_called
    assert bot.config_symbols == ["BTC/USDT:USDT"]
    assert [order["pb_order_type"] for order in bot.created_orders] == [
        "close_panic_long"
    ]
    assert bot.execution_scheduled is True


@pytest.mark.asyncio
async def test_execute_to_exchange_configures_only_symbols_with_creations():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self.stop_signal_received = False
            self.config_symbols = None
            self.created_orders = None
            self._current_planning_snapshot = FreshPlanningSnapshot()
            self.market_snapshot_provider = FreshMarketSnapshotProvider()

        async def execution_cycle(self):
            return None

        async def calc_orders_to_cancel_and_create(self):
            return [
                {
                    "symbol": "CANCEL/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
            ], [
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 2.0,
                    "qty": 1.0,
                },
            ]

        async def execute_cancellations_parent(self, orders):
            return []

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = symbols
            return set(symbols or [])

        def get_raw_balance(self):
            return 1.0

        def order_was_recently_updated(self, order):
            return 0

        async def execute_orders_parent(self, orders):
            self.created_orders = orders
            return []

        async def _refresh_forager_candidate_candles(self):
            return None

        def _current_planning_snapshot_invalid_for_creations(self, symbols):
            return []

        async def _get_live_market_snapshots(
            self,
            symbols,
            *,
            max_age_ms=10_000,
            context="live",
            allow_completed_candle_fallback=False,
        ):
            return await self.market_snapshot_provider.get_snapshots(
                symbols, max_age_ms=max_age_ms
            )

        def _live_market_snapshot_max_age_ms(self):
            return 10_000

        def _record_market_snapshot_surface(self, symbols, snapshots):
            return None

        def _market_snapshot_signature_invalid(self, symbols):
            return []

        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot.config_symbols == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert len(bot.created_orders) == 3


@pytest.mark.asyncio
async def test_execute_order_plan_posts_replacement_matching_cancel_same_cycle():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self.cancelled_orders = None
            self.config_symbols = None
            self.created_orders = None
            self.execution_scheduled = False
            self._current_planning_snapshot = FreshPlanningSnapshot()
            self.market_snapshot_provider = FreshMarketSnapshotProvider()

        def get_raw_balance(self):
            return 100.0

        async def execute_cancellations_parent(self, orders):
            self.cancelled_orders = list(orders)
            return list(orders)

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = list(symbols or [])
            return set(symbols or [])

        def order_was_recently_updated(self, order):
            return 0

        async def execute_orders_parent(self, orders):
            self.created_orders = list(orders)
            return list(orders)

        def _current_planning_snapshot_invalid_for_creations(self, symbols):
            return []

        async def _get_live_market_snapshots(
            self,
            symbols,
            *,
            max_age_ms=10_000,
            context="live",
            allow_completed_candle_fallback=False,
        ):
            return await self.market_snapshot_provider.get_snapshots(
                symbols, max_age_ms=max_age_ms
            )

        def _live_market_snapshot_max_age_ms(self):
            return 10_000

        def _record_market_snapshot_surface(self, symbols, snapshots):
            return None

        def _market_snapshot_signature_invalid(self, symbols):
            return []

        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    symbol = "BTC/USDT:USDT"
    to_cancel = [
        {
            "symbol": symbol,
            "side": "sell",
            "position_side": "long",
            "price": 100.0,
            "qty": 1.0,
            "reduce_only": True,
        }
    ]
    to_create = [
        {
            "symbol": symbol,
            "side": "sell",
            "position_side": "long",
            "price": 100.0,
            "qty": 1.0,
            "reduce_only": True,
            "pb_order_type": "close_grid_long",
        }
    ]

    bot = FakeBot()
    await pb_mod.Passivbot.execute_order_plan_to_exchange(bot, to_cancel, to_create)

    assert bot.cancelled_orders == to_cancel
    assert bot.config_symbols == [symbol]
    assert bot.created_orders == to_create
    assert bot.execution_scheduled is True


@pytest.mark.asyncio
async def test_execute_to_exchange_skips_creations_pending_exchange_config():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self.stop_signal_received = False
            self.config_symbols = None
            self.created_orders = None
            self._current_planning_snapshot = FreshPlanningSnapshot()
            self.market_snapshot_provider = FreshMarketSnapshotProvider()

        async def execution_cycle(self):
            return None

        async def calc_orders_to_cancel_and_create(self):
            return [], [
                {
                    "symbol": "READY/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
                {
                    "symbol": "PENDING/USDT:USDT",
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
            ]

        async def execute_cancellations_parent(self, orders):
            return []

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = symbols
            return {"READY/USDT:USDT"}

        def get_raw_balance(self):
            return 1.0

        def order_was_recently_updated(self, order):
            return 0

        async def execute_orders_parent(self, orders):
            self.created_orders = orders
            return []

        async def _refresh_forager_candidate_candles(self):
            return None

        def _current_planning_snapshot_invalid_for_creations(self, symbols):
            return []

        async def _get_live_market_snapshots(
            self,
            symbols,
            *,
            max_age_ms=10_000,
            context="live",
            allow_completed_candle_fallback=False,
        ):
            return await self.market_snapshot_provider.get_snapshots(
                symbols, max_age_ms=max_age_ms
            )

        def _live_market_snapshot_max_age_ms(self):
            return 10_000

        def _record_market_snapshot_surface(self, symbols, snapshots):
            return None

        def _market_snapshot_signature_invalid(self, symbols):
            return []

        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot.config_symbols == ["PENDING/USDT:USDT", "READY/USDT:USDT"]
    assert [order["symbol"] for order in bot.created_orders] == ["READY/USDT:USDT"]
