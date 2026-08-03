import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import BadRequest, MarginModeAlreadySet, RateLimitExceeded
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes


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
async def test_binance_already_set_margin_mode_is_successful_noop(caplog):
    from exchanges.binance import BinanceBot

    symbol = "BTC/USDT:USDT"
    bot = BinanceBot.__new__(BinanceBot)
    bot.cca = SimpleNamespace(
        set_margin_mode=AsyncMock(
            side_effect=MarginModeAlreadySet("No need to change margin type")
        ),
        set_leverage=AsyncMock(return_value={"leverage": 5}),
    )
    bot._get_margin_mode_for_symbol = lambda _symbol: "cross"
    bot._calc_leverage_for_symbol = lambda _symbol: 5

    with caplog.at_level(logging.DEBUG):
        await bot.update_exchange_config_by_symbols([symbol])

    bot.cca.set_margin_mode.assert_awaited_once_with("cross", symbol=symbol)
    bot.cca.set_leverage.assert_awaited_once_with(5, symbol=symbol)
    assert "margin mode unchanged" in caplog.text
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR and "margin" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_update_exchange_configs_marks_only_successful_symbols(monkeypatch, caplog):
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
                raise RuntimeError("SECRET")

        _is_rate_limit_like_exception = pb_mod.Passivbot._is_rate_limit_like_exception
        _exchange_config_backoff_seconds = pb_mod.Passivbot._exchange_config_backoff_seconds
        _exchange_config_success_pause_seconds = pb_mod.Passivbot._exchange_config_success_pause_seconds
        _shutdown_requested = lambda self: False

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    bot = FakeBot()
    with caplog.at_level(logging.WARNING):
        await pb_mod.Passivbot.update_exchange_configs(bot)

    assert bot.calls == ["A", "B"]
    assert bot.already_updated_exchange_config_symbols == {"B"}
    assert bot._exchange_config_retry_attempts["A"] == 1
    assert bot._exchange_config_retry_after_ms["A"] > 0
    assert bot._health_rate_limits == 0
    assert "error_type=RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text


def test_format_exchange_config_error_is_bounded_and_value_safe():
    import passivbot as pb_mod

    assert (
        pb_mod.Passivbot._format_exchange_config_error(RuntimeError("SECRET"))
        == "error_type=RuntimeError"
    )
    unsafe_type = type("SECRET" * 20, (Exception,), {})
    assert (
        pb_mod.Passivbot._format_exchange_config_error(unsafe_type("SECRET"))
        == "error_type=Exception"
    )
    structured = RuntimeError(
        'gate {"label":"RISK_LIMIT_EXCEEDED","message":"position risk limit is zero"}'
    )
    assert pb_mod.Passivbot._format_exchange_config_error(structured) == (
        "error_type=RuntimeError error_label=RISK_LIMIT_EXCEEDED "
        "error_reason=position risk limit is zero"
    )


def make_defx_config_bot(cca):
    from exchanges.defx import DefxBot

    bot = DefxBot.__new__(DefxBot)
    bot.cca = cca
    bot.max_leverage = {"BTC/USDC:USDC": 10}
    bot.config_get = lambda path, *, symbol=None: 5
    bot.get_wallet_exposure_limit = lambda pside, symbol: 1.0
    return bot


@pytest.mark.asyncio
async def test_defx_exchange_config_response_is_value_safe(caplog):
    class SuccessfulCCA:
        async def set_leverage(self, **params):
            return {"leverage": params["leverage"], "apiKey": "SECRET"}

    bot = make_defx_config_bot(SuccessfulCCA())

    with caplog.at_level(logging.INFO):
        await bot.update_exchange_config_by_symbols(["BTC/USDC:USDC"])

    assert "set_leverage leverage=2x" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("SECRET"), "error_type=RuntimeError"),
        (
            RuntimeError('{"code":"59107","msg":"SECRET"}'),
            "set_leverage ok (unchanged) code=59107",
        ),
    ],
)
async def test_defx_exchange_config_failure_is_value_safe(caplog, error, expected):
    class FailingCCA:
        async def set_leverage(self, **_params):
            raise error

    bot = make_defx_config_bot(FailingCCA())

    with caplog.at_level(logging.INFO):
        await bot.update_exchange_config_by_symbols(["BTC/USDC:USDC"])

    assert expected in caplog.text
    assert "SECRET" not in caplog.text


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
async def test_update_exchange_configs_records_only_failed_attempts_not_backoff_skips(
    monkeypatch,
):
    import passivbot as pb_mod

    now_ms = 10_000

    class FakeBot:
        already_updated_exchange_config_symbols = set()
        _exchange_config_retry_attempts = {}
        _exchange_config_retry_after_ms = {"SKIP": now_ms + 1_000}
        active_symbols = ["FAIL", "SKIP"]
        exchange = "weex"
        _health_rate_limits = 0

        def _shutdown_requested(self):
            return False

        async def update_exchange_config_by_symbols(self, symbols):
            if symbols == ["FAIL"]:
                raise RuntimeError("failed write")
            raise AssertionError(f"unexpected configuration attempt: {symbols}")

        def _exchange_config_backoff_seconds(self, _attempt):
            return 2.0

        def _activate_exchange_symbol_unavailable_cooldown(self, *_args, **_kwargs):
            return False

        def _is_rate_limit_like_exception(self, _exc):
            return False

        def _exchange_config_success_pause_seconds(self):
            return 0.0

        _format_exchange_config_error = staticmethod(
            pb_mod.Passivbot._format_exchange_config_error
        )

    bot = FakeBot()
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: now_ms)

    configured = await pb_mod.Passivbot.update_exchange_configs(
        bot,
        ["FAIL", "SKIP"],
        eligibility_now_ms=now_ms,
    )

    assert configured == set()
    assert bot._last_exchange_config_failed_attempt_symbols == {"FAIL"}

    configured = await pb_mod.Passivbot.update_exchange_configs(
        bot,
        ["FAIL", "SKIP"],
        eligibility_now_ms=now_ms + 1,
    )

    assert configured == set()
    assert bot._last_exchange_config_failed_attempt_symbols == set()


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
async def test_exchange_update_config_reraises_hedge_mode_failures(
    module_name, class_name, caplog
):
    module = __import__(module_name, fromlist=[class_name])
    bot_cls = getattr(module, class_name)
    bot = bot_cls.__new__(bot_cls)
    bot.cca = SimpleNamespace(set_position_mode=AsyncMock(side_effect=RuntimeError("SECRET")))
    if class_name == "BitgetBot":
        # Bitget probes the UTA account mode before setting hedge mode.
        bot.cca.private_uta_get_v3_account_assets = AsyncMock(return_value={"code": "00000"})

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="SECRET"):
            await bot.update_exchange_config()

    assert "error_type=RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text


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
    if class_name == "BitgetBot":
        # Bitget probes the UTA account mode before setting hedge mode.
        bot.cca.private_uta_get_v3_account_assets = AsyncMock(return_value={"code": "00000"})

    await bot.update_exchange_config()

    bot.cca.set_position_mode.assert_awaited_once_with(True)


@pytest.mark.asyncio
async def test_binance_update_config_accepts_already_hedged_response(caplog):
    from exchanges.binance import BinanceBot

    bot = BinanceBot.__new__(BinanceBot)
    bot.cca = SimpleNamespace(
        set_position_mode=AsyncMock(
            side_effect=Exception('{"code":-4059,"msg":"No need SECRET"}')
        )
    )

    with caplog.at_level(logging.DEBUG):
        await bot.update_exchange_config()

    assert "hedge mode unchanged | code=-4059" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_bybit_update_config_accepts_already_hedged_bad_request():
    from exchanges.bybit import BybitBot

    bot = BybitBot.__new__(BybitBot)
    bot.cca = SimpleNamespace(
        set_position_mode=AsyncMock(
            side_effect=BadRequest(
                'bybit {"retCode":110025,"retMsg":"position mode is not modified"}'
            )
        )
    )

    await bot.update_exchange_config()

    bot.cca.set_position_mode.assert_awaited_once_with(True)


@pytest.mark.asyncio
async def test_bybit_update_config_reraises_unknown_bad_request():
    from exchanges.bybit import BybitBot

    bot = BybitBot.__new__(BybitBot)
    bot.cca = SimpleNamespace(set_position_mode=AsyncMock(side_effect=BadRequest("boom")))

    with pytest.raises(BadRequest, match="boom"):
        await bot.update_exchange_config()


@pytest.mark.asyncio
async def test_bitget_fetch_pnls_legacy_path_is_disabled():
    from exchanges.bitget import BitgetBot

    bot = BitgetBot.__new__(BitgetBot)

    with pytest.raises(NotImplementedError, match="fetch_fill_events"):
        await bot.fetch_pnls()


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
async def test_okx_update_config_bounds_known_skip_diagnostic(caplog):
    from exchanges.okx import OKXBot

    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True
    bot.hedge_mode = True
    bot.cca = SimpleNamespace(
        private_get_account_config=AsyncMock(
            return_value={"data": [{"posMode": "long_short_mode"}]}
        ),
        set_position_mode=AsyncMock(
            side_effect=RuntimeError('{"code":"59000","msg":"SECRET"}')
        ),
    )

    with caplog.at_level(logging.INFO):
        await bot.update_exchange_config()

    assert "hedge mode update skipped | code=59000" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_okx_update_config_verified_net_mode_fails_loudly():
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

    with pytest.raises(RuntimeError, match="requires dual-side/hedge mode"):
        await bot.update_exchange_config()

    assert bot.okx_dual_side is False
    assert bot.hedge_mode is False
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
async def test_okx_already_gone_cancel_does_not_log_raw_exception(caplog, capsys):
    from exchanges.okx import OKXBot

    markers = []

    async def cancel_order(*_args, **_kwargs):
        markers.append("connector")
        raise Exception('{"sCode":"51400","apiKey":"RAW_OKX_CANCEL_SECRET"}')

    bot = OKXBot.__new__(OKXBot)
    bot.cca = SimpleNamespace(cancel_order=cancel_order)
    bot._emit_execution_connector_call_started_event = lambda **kwargs: markers.append(
        ("event", kwargs)
    )
    order = {
        "id": "order-1",
        "symbol": "BTC/USDT:USDT",
        "raw_payload": "RAW_OKX_ORDER_SECRET",
    }

    with caplog.at_level(logging.INFO):
        result = await bot.execute_cancellation(order)

    assert result["status"] == "success"
    assert result["_passivbot_cancel_requires_full_authoritative_confirmation"] is True
    captured = capsys.readouterr()
    rendered = captured.out + captured.err + caplog.text
    assert "RAW_OKX_CANCEL_SECRET" not in rendered
    assert "RAW_OKX_ORDER_SECRET" not in rendered
    assert "sCode" not in rendered
    assert "cancel skipped: BTC" in caplog.text
    assert "error_type=Exception" in caplog.text
    assert markers[0][0] == "event"
    assert markers[0][1]["action"] == "cancel"
    assert markers[0][1]["connector_route"] == "okx"
    assert markers[1] == "connector"


@pytest.mark.asyncio
async def test_execute_to_exchange_stops_after_exchange_config_shutdown(monkeypatch):
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
            return [], [
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

    async def keep_creations(_bot, orders):
        return orders

    monkeypatch.setattr(
        pb_mod.Passivbot,
        "_filter_fresh_market_snapshot_creations",
        keep_creations,
    )
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
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _live_event_console_available = pb_mod.Passivbot._live_event_console_available
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

        debug_mode = False
        balance_threshold = 1.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()

        def __init__(self):
            self._live_event_current_cycle_id = "cy_low_balance"
            self._live_event_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink],
                monitor_sinks=[],
            )
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
                    "side": "sell",
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

    bot = FakeBot()
    with caplog.at_level(logging.INFO):
        await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot.cancel_called
    assert not bot.config_called
    assert not bot.create_called
    assert bot.execution_scheduled is True
    skipped_events = [
        event
        for event in bot._live_event_sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(skipped_events) == 1
    assert skipped_events[0].cycle_id == "cy_low_balance"
    assert skipped_events[0].status == "skipped"
    assert skipped_events[0].reason_code == ReasonCodes.LOW_BALANCE
    assert skipped_events[0].data["order_count"] == 1
    assert skipped_events[0].data["symbols"] == ["BTC/USDT:USDT"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    assert any(
        "skipped 1 exposure-increasing order creates" in record.message
        for record in caplog.records
    )
    assert any("allowing 1 cancellations and 0 protective creates" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_low_balance_create_skip_uses_event_console_when_available(caplog):
    import passivbot as pb_mod

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger
        _live_event_console_available = pb_mod.Passivbot._live_event_console_available
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

        debug_mode = False
        balance_threshold = 1.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()

        def __init__(self):
            self.live_event_console_enabled = True
            self._live_event_current_cycle_id = "cy_low_balance"
            self._live_event_sink = ListEventSink()
            self._live_event_console_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink],
                monitor_sinks=[],
                console_sink=self._live_event_console_sink,
            )
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
                    "side": "sell",
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

    bot = FakeBot()
    with caplog.at_level(logging.INFO):
        await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot.cancel_called
    assert not bot.create_called
    assert not any("[balance] too low" in record.message for record in caplog.records)
    assert [event.event_type for event in bot._live_event_console_sink.events] == [
        EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    console_event = bot._live_event_console_sink.events[0]
    assert console_event.reason_code == ReasonCodes.LOW_BALANCE
    assert console_event.data["raw_balance"] == pytest.approx(0.0)
    assert console_event.data["balance_threshold"] == pytest.approx(1.0)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


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
            return [], [
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
async def test_execute_to_exchange_emits_recent_execution_deferred_event():
    import passivbot as pb_mod

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self._live_event_current_cycle_id = "cy_recent_execution"
            self._live_event_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink],
                monitor_sinks=[],
            )
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
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
                {
                    "symbol": "ETH/USDT:USDT",
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
            return set(symbols or [])

        def get_raw_balance(self):
            return 1.0

        def order_was_recently_updated(self, order):
            if order["symbol"] == "BTC/USDT:USDT":
                return 5_000
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

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot.config_symbols == ["ETH/USDT:USDT"]
    assert [order["symbol"] for order in bot.created_orders] == ["ETH/USDT:USDT"]
    deferred_events = [
        event
        for event in bot._live_event_sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_DEFERRED
    ]
    assert len(deferred_events) == 1
    assert deferred_events[0].cycle_id == "cy_recent_execution"
    assert deferred_events[0].status == "deferred"
    assert deferred_events[0].reason_code == ReasonCodes.RECENT_EXECUTION
    assert deferred_events[0].data["order_count"] == 1
    assert deferred_events[0].data["symbols"] == ["BTC/USDT:USDT"]
    assert deferred_events[0].data["max_delay_ms"] == 5_000
    assert bot.execution_scheduled is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execute_to_exchange_emits_state_change_skipped_event():
    import passivbot as pb_mod

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = {"BTC/USDT:USDT"}
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self._live_event_current_cycle_id = "cy_state_change"
            self._live_event_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink],
                monitor_sinks=[],
            )
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
                    "side": "buy",
                    "position_side": "long",
                    "price": 1.0,
                    "qty": 1.0,
                },
                {
                    "symbol": "ETH/USDT:USDT",
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
            return set(symbols or [])

        def get_raw_balance(self):
            return 1.0

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

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot.config_symbols == ["ETH/USDT:USDT"]
    assert [order["symbol"] for order in bot.created_orders] == ["ETH/USDT:USDT"]
    skipped_events = [
        event
        for event in bot._live_event_sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(skipped_events) == 1
    assert skipped_events[0].cycle_id == "cy_state_change"
    assert skipped_events[0].status == "skipped"
    assert skipped_events[0].reason_code == ReasonCodes.STATE_CHANGE_DETECTED
    assert skipped_events[0].data["order_count"] == 1
    assert skipped_events[0].data["symbols"] == ["BTC/USDT:USDT"]
    assert skipped_events[0].data["blocked_symbols_count"] == 1
    assert bot.execution_scheduled is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execute_order_plan_defers_replacement_until_cancel_confirmation():
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
            "price": 1.0,
            "qty": 1.0,
            "reduce_only": True,
        }
    ]
    to_create = [
        {
            "symbol": symbol,
            "side": "sell",
            "position_side": "long",
            "price": 1.0,
            "qty": 1.0,
            "reduce_only": True,
            "pb_order_type": "close_grid_long",
        }
    ]

    bot = FakeBot()
    await pb_mod.Passivbot.execute_order_plan_to_exchange(bot, to_cancel, to_create)

    assert bot.cancelled_orders == to_cancel
    assert bot.config_symbols is None
    assert bot.created_orders is None
    assert set(bot._authoritative_pending_confirmations) == {
        "balance",
        "positions",
        "open_orders",
        "fills",
    }
    assert bot.execution_scheduled is True


@pytest.mark.asyncio
async def test_execute_order_plan_blocks_pending_hsl_entry_and_defers_close_after_cancel():
    import passivbot as pb_mod

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested
        _ensure_freshness_ledger = pb_mod.Passivbot._ensure_freshness_ledger

        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        stop_signal_received = False
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self._live_event_current_cycle_id = "cy_hsl_replay_pending"
            self._live_event_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink], monitor_sinks=[]
            )
            self._equity_hard_stop_coin_replay_pending_pairs = {
                ("long", "BTC/USDT:USDT")
            }
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

    symbol = "BTC/USDT:USDT"
    to_cancel = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "price": 0.9,
            "qty": 1.0,
            "reduce_only": False,
        }
    ]
    entry = {
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "price": 0.9,
        "qty": 1.0,
        "reduce_only": False,
        "pb_order_type": "entry_grid_long",
    }
    close = {
        "symbol": symbol,
        "side": "sell",
        "position_side": "long",
        "price": 1.1,
        "qty": 1.0,
        "reduce_only": True,
        "pb_order_type": "close_grid_long",
    }

    bot = FakeBot()
    await pb_mod.Passivbot.execute_order_plan_to_exchange(
        bot, to_cancel, [entry, close]
    )

    assert bot.cancelled_orders == to_cancel
    assert bot.config_symbols is None
    assert bot.created_orders is None
    assert bot.execution_scheduled is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    skipped = [
        event
        for event in bot._live_event_sink.events
        if event.reason_code == ReasonCodes.HSL_REPLAY_PENDING
    ]
    assert len(skipped) == 1
    assert skipped[0].data["order_count"] == 1
    assert skipped[0].data["pending_pairs_count"] == 1
    barriers = [
        event
        for event in bot._live_event_sink.events
        if event.reason_code == ReasonCodes.ACCOUNT_CANCEL_FIRST_BARRIER
    ]
    assert len(barriers) == 1
    assert barriers[0].data["order_count"] == 1
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execute_to_exchange_skips_creations_pending_exchange_config():
    import passivbot as pb_mod

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_create_filter_event = (
            pb_mod.Passivbot._emit_execution_create_filter_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

        debug_mode = False
        balance_threshold = 0.0
        quote = "USDT"
        state_change_detected_by_symbol = set()
        config = {"live": {}, "_raw_effective": {"live": {}}}

        def __init__(self):
            self.stop_signal_received = False
            self._live_event_current_cycle_id = "cy_pending_config"
            self._live_event_sink = ListEventSink()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[self._live_event_sink],
                monitor_sinks=[],
            )
            self.config_symbols = None
            self.created_orders = None
            self.restart_budget_calls = 0
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
                {
                    "symbol": "PENDING/USDT:USDT",
                    "side": "sell",
                    "position_side": "long",
                    "price": 1.1,
                    "qty": 1.0,
                    "reduce_only": True,
                    "pb_order_type": "close_grid_long",
                },
            ]

        async def execute_cancellations_parent(self, orders):
            return []

        async def update_exchange_configs(self, symbols=None):
            self.config_symbols = symbols
            return {"READY/USDT:USDT"}

        def _order_requires_exchange_config_before_create(self, order):
            return order.get("reduce_only") is not True

        def _pending_exchange_config_consumes_error_budget(self, blocked_orders):
            return bool(blocked_orders)

        async def restart_bot_on_too_many_errors(self):
            self.restart_budget_calls += 1

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

    bot = FakeBot()
    await pb_mod.Passivbot.execute_to_exchange(bot)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot.config_symbols == ["PENDING/USDT:USDT", "READY/USDT:USDT"]
    assert [order["symbol"] for order in bot.created_orders] == [
        "READY/USDT:USDT",
        "PENDING/USDT:USDT",
    ]
    assert bot.created_orders[1]["reduce_only"] is True
    assert bot.restart_budget_calls == 1
    skipped_events = [
        event
        for event in bot._live_event_sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(skipped_events) == 1
    assert skipped_events[0].cycle_id == "cy_pending_config"
    assert skipped_events[0].status == "skipped"
    assert skipped_events[0].reason_code == ReasonCodes.PENDING_EXCHANGE_CONFIG
    assert skipped_events[0].data["order_count"] == 1
    assert skipped_events[0].data["symbols"] == ["PENDING/USDT:USDT"]
    assert skipped_events[0].data["protective_allowed_count"] == 1
    assert skipped_events[0].data["pending_symbols_count"] == 1
    assert bot._live_event_pipeline.close(timeout=2.0) is True
