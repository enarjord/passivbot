import pytest


class _FakeBot:
    exchange = "bybit"
    quote = "USDT"
    cca = object()
    sym_padding = 0

    def __init__(self, update_exchange_config_impl):
        self._update_exchange_config_impl = update_exchange_config_impl
        self.update_exchange_config_calls = 0
        self.determine_utc_offset_calls = 0
        self.market_specific_settings_calls = 0
        self.positions_balance_calls = 0
        self.open_orders_calls = 0
        self.min_cost_calls = 0
        self.abstraction_refresh_calls = 0

    async def update_exchange_config(self):
        self.update_exchange_config_calls += 1
        return await self._update_exchange_config_impl(self.update_exchange_config_calls)

    async def determine_utc_offset(self, verbose=True):
        self.determine_utc_offset_calls += 1

    async def refresh_and_log_user_abstraction_state(self):
        self.abstraction_refresh_calls += 1

    def set_market_specific_settings(self):
        self.market_specific_settings_calls += 1

    def init_coin_overrides(self):
        return None

    def refresh_approved_ignored_coins_lists(self):
        return None

    def set_wallet_exposure_limits(self):
        return None

    def _authoritative_refresh_mode(self):
        return "legacy"

    async def update_positions_and_balance(self):
        self.positions_balance_calls += 1

    async def update_open_orders(self):
        self.open_orders_calls += 1

    def _assert_supported_live_state(self):
        return None

    async def update_effective_min_cost(self):
        self.min_cost_calls += 1

    def is_forager_mode(self):
        return False


@pytest.mark.asyncio
async def test_init_markets_retries_request_timeout_then_succeeds(monkeypatch):
    import passivbot as pb_mod

    sleeps = []

    async def _nosleep(seconds):
        sleeps.append(seconds)

    async def _load_markets(*_args, **_kwargs):
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    async def _update_exchange_config(attempt):
        if attempt < 3:
            raise pb_mod.RequestTimeout("timed out")

    monkeypatch.setattr(pb_mod.asyncio, "sleep", _nosleep)
    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {"DOGE/USDT:USDT": "bad"}),
    )

    bot = _FakeBot(_update_exchange_config)

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.update_exchange_config_calls == 3
    assert sleeps == [5, 10]
    assert bot.markets_dict == {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
    assert bot.eligible_symbols == {"BTC/USDT:USDT"}
    assert bot.ineligible_symbols == {"DOGE/USDT:USDT": "bad"}
    assert bot.market_specific_settings_calls == 1
    assert bot.abstraction_refresh_calls == 1
    assert bot.positions_balance_calls == 1
    assert bot.open_orders_calls == 1
    assert bot.min_cost_calls == 1


@pytest.mark.asyncio
async def test_init_markets_reraises_non_retryable_update_exchange_config_error(monkeypatch):
    import passivbot as pb_mod

    sleeps = []

    async def _nosleep(seconds):
        sleeps.append(seconds)

    async def _load_markets(*_args, **_kwargs):
        raise AssertionError("load_markets should not run after a fatal config error")

    async def _update_exchange_config(_attempt):
        raise ValueError("boom")

    monkeypatch.setattr(pb_mod.asyncio, "sleep", _nosleep)
    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)

    bot = _FakeBot(_update_exchange_config)

    with pytest.raises(ValueError, match="boom"):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.update_exchange_config_calls == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_init_markets_reraises_after_max_network_retries(monkeypatch):
    import passivbot as pb_mod

    sleeps = []

    async def _nosleep(seconds):
        sleeps.append(seconds)

    async def _load_markets(*_args, **_kwargs):
        raise AssertionError("load_markets should not run after retry exhaustion")

    async def _update_exchange_config(_attempt):
        raise pb_mod.NetworkError("network down")

    monkeypatch.setattr(pb_mod.asyncio, "sleep", _nosleep)
    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)

    bot = _FakeBot(_update_exchange_config)

    with pytest.raises(pb_mod.NetworkError, match="network down"):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.update_exchange_config_calls == 3
    assert sleeps == [5, 10]
