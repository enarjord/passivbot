import asyncio

import pytest
from ccxt.base.errors import RateLimitExceeded


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
async def test_execute_to_exchange_stops_after_exchange_config_shutdown():
    import passivbot as pb_mod

    class FakeBot:
        debug_mode = False

        def __init__(self):
            self.stop_signal_received = False
            self.calc_called = False

        async def execution_cycle(self):
            return None

        async def update_exchange_configs(self):
            self.stop_signal_received = True

        async def calc_orders_to_cancel_and_create(self):
            self.calc_called = True
            return [], []

        _shutdown_requested = pb_mod.Passivbot._shutdown_requested

    bot = FakeBot()
    result = await pb_mod.Passivbot.execute_to_exchange(bot)

    assert result is None
    assert not bot.calc_called
