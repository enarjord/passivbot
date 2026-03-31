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
