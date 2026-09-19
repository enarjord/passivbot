import pytest
from passivbot_exceptions import FatalBotException
from exchanges.ccxt_bot import CCXTBot
from passivbot import Passivbot


class _FakeBot:
    _load_market_metadata = Passivbot._load_market_metadata
    exchange = "bybit"
    quote = "USDT"
    cca = object()
    sym_padding = 0

    def __init__(
        self,
        update_exchange_config_impl,
        assert_supported_live_state_impl=None,
        exchange_config_ready_impl=None,
    ):
        self._update_exchange_config_impl = update_exchange_config_impl
        self._assert_supported_live_state_impl = assert_supported_live_state_impl
        self._exchange_config_ready_impl = exchange_config_ready_impl
        self.update_exchange_config_calls = 0
        self.exchange_config_ready_calls = 0
        self.determine_utc_offset_calls = 0
        self.market_specific_settings_calls = 0
        self.positions_balance_calls = 0
        self.open_orders_calls = 0
        self.refresh_authoritative_state_calls = 0
        self.min_cost_calls = 0
        self.abstraction_refresh_calls = 0
        self.assert_supported_live_state_calls = 0
        self.stop_signal_received = False
        self.shutdown_sleeps = []

    async def update_exchange_config(self):
        self.update_exchange_config_calls += 1
        return await self._update_exchange_config_impl(self.update_exchange_config_calls)

    async def _exchange_config_write_ready(self):
        self.exchange_config_ready_calls += 1
        if self._exchange_config_ready_impl is None:
            return True
        return await self._exchange_config_ready_impl(
            self.exchange_config_ready_calls
        )

    async def _sleep_unless_shutdown(self, seconds, *, stage):
        self.shutdown_sleeps.append((seconds, stage))

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

    async def refresh_authoritative_state(self):
        self.refresh_authoritative_state_calls += 1

    async def update_positions_and_balance(self):
        self.positions_balance_calls += 1

    async def update_open_orders(self):
        self.open_orders_calls += 1

    def _assert_supported_live_state(self):
        self.assert_supported_live_state_calls += 1
        if self._assert_supported_live_state_impl is not None:
            self._assert_supported_live_state_impl(self.assert_supported_live_state_calls)
        return None

    async def update_effective_min_cost(self):
        self.min_cost_calls += 1

    def is_forager_mode(self):
        return False


class _InitMarketsSizingBot(CCXTBot):
    exchange = "bybit"
    quote = "USDT"
    cca = object()
    sym_padding = 0

    def __init__(self, *, coin_overrides=None, positions_after_refresh=None):
        self.coin_overrides_to_set = coin_overrides or {}
        self.positions_after_refresh = positions_after_refresh or {}
        self.update_exchange_config_calls = 0
        self.market_specific_settings_calls = 0
        self.refresh_authoritative_state_calls = 0
        self.assert_supported_live_state_calls = 0
        self.min_cost_calls = 0
        self.abstraction_refresh_calls = 0
        self.symbol_ids = {}
        self.min_costs = {}
        self.min_qtys = {}
        self.qty_steps = {}
        self.price_steps = {}
        self.c_mults = {}
        self.active_symbols = []
        self.coin_overrides = {}
        self.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
        self.positions = {}
        self.open_orders = {}
        self.stop_signal_received = False

    async def _exchange_config_write_ready(self):
        return True

    async def update_exchange_config(self):
        self.update_exchange_config_calls += 1

    async def refresh_and_log_user_abstraction_state(self):
        self.abstraction_refresh_calls += 1

    def set_market_specific_settings(self):
        self.market_specific_settings_calls += 1
        super().set_market_specific_settings()

    def init_coin_overrides(self):
        self.coin_overrides = dict(self.coin_overrides_to_set)

    def refresh_approved_ignored_coins_lists(self):
        self.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}

    def set_wallet_exposure_limits(self):
        return None

    async def refresh_authoritative_state(self):
        self.refresh_authoritative_state_calls += 1
        self.positions = self.positions_after_refresh

    def _assert_supported_live_state(self):
        self.assert_supported_live_state_calls += 1

    async def update_effective_min_cost(self):
        self.min_cost_calls += 1

    def is_forager_mode(self):
        return False


def _market(symbol_id, *, min_qty, qty_step):
    return {
        "id": symbol_id,
        "limits": {
            "cost": {"min": 0.1},
            "amount": {"min": min_qty},
        },
        "precision": {
            "amount": qty_step,
            "price": 0.01,
        },
        "contractSize": 1.0,
    }


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
    assert bot.market_specific_settings_calls == 2
    assert bot.abstraction_refresh_calls == 1
    assert bot.refresh_authoritative_state_calls == 1
    assert bot.positions_balance_calls == 0
    assert bot.open_orders_calls == 0
    assert bot.min_cost_calls == 1


@pytest.mark.asyncio
async def test_init_markets_gates_exchange_config_write_on_balance_readiness(monkeypatch):
    import passivbot as pb_mod

    events = []

    async def _load_markets(*_args, **_kwargs):
        events.append("load_markets")
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    async def _exchange_config_ready(attempt):
        events.append(f"readiness:{attempt}")
        return attempt >= 3

    async def _update_exchange_config(_attempt):
        events.append("update_exchange_config")

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {}),
    )

    bot = _FakeBot(
        _update_exchange_config,
        exchange_config_ready_impl=_exchange_config_ready,
    )

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert events[:4] == [
        "readiness:1",
        "readiness:2",
        "readiness:3",
        "update_exchange_config",
    ]
    assert bot.shutdown_sleeps == [
        (5.0, "exchange_config_balance_readiness"),
        (5.0, "exchange_config_balance_readiness"),
    ]


@pytest.mark.asyncio
async def test_init_markets_stops_after_balance_readiness_check(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        raise AssertionError("load_markets should not run after shutdown")

    async def _update_exchange_config(_attempt):
        raise AssertionError("exchange config should not be written after shutdown")

    bot = _FakeBot(_update_exchange_config)

    async def _exchange_config_ready(_attempt):
        bot.stop_signal_received = True
        return True

    bot._exchange_config_ready_impl = _exchange_config_ready
    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.exchange_config_ready_calls == 1
    assert bot.update_exchange_config_calls == 0


@pytest.mark.asyncio
async def test_init_markets_retries_transient_balance_readiness_errors(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    async def _exchange_config_ready(attempt):
        if attempt < 3:
            raise pb_mod.RequestTimeout("timed out")
        return True

    async def _update_exchange_config(_attempt):
        return None

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {}),
    )
    bot = _FakeBot(
        _update_exchange_config,
        exchange_config_ready_impl=_exchange_config_ready,
    )

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.exchange_config_ready_calls == 3
    assert bot.update_exchange_config_calls == 1
    assert bot.shutdown_sleeps == [
        (5, "exchange_config_balance_readiness_retry"),
        (10, "exchange_config_balance_readiness_retry"),
    ]


@pytest.mark.asyncio
async def test_init_markets_reraises_after_max_balance_readiness_retries(monkeypatch):
    import passivbot as pb_mod

    async def _exchange_config_ready(_attempt):
        raise pb_mod.NetworkError("unavailable")

    async def _update_exchange_config(_attempt):
        raise AssertionError("exchange config should not run without readiness")

    bot = _FakeBot(
        _update_exchange_config,
        exchange_config_ready_impl=_exchange_config_ready,
    )

    with pytest.raises(pb_mod.NetworkError, match="unavailable"):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.exchange_config_ready_calls == 3
    assert bot.update_exchange_config_calls == 0
    assert bot.shutdown_sleeps == [
        (5, "exchange_config_balance_readiness_retry"),
        (10, "exchange_config_balance_readiness_retry"),
    ]


@pytest.mark.asyncio
async def test_init_markets_validates_coin_overrides_before_sizing(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {
            "BTC/USDT:USDT": _market("BTCUSDT", min_qty=0.001, qty_step=0.001),
            "BAD/USDT:USDT": _market("BADUSDT", min_qty=None, qty_step=None),
        }

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (
            ["BTC/USDT:USDT"],
            [],
            {"BAD/USDT:USDT": "wrong quote"},
        ),
    )

    bot = _InitMarketsSizingBot(coin_overrides={"BAD/USDT:USDT": {}})

    with pytest.raises(
        ValueError,
        match="BAD/USDT:USDT: missing min qty and qty step",
    ):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.refresh_authoritative_state_calls == 0
    assert bot.market_specific_settings_calls == 1


@pytest.mark.asyncio
async def test_init_markets_revalidates_sizing_after_account_state(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {
            "BTC/USDT:USDT": _market("BTCUSDT", min_qty=0.001, qty_step=0.001),
            "BAD/USDT:USDT": _market("BADUSDT", min_qty=None, qty_step=None),
        }

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (
            ["BTC/USDT:USDT"],
            [],
            {"BAD/USDT:USDT": "wrong quote"},
        ),
    )

    bot = _InitMarketsSizingBot(
        positions_after_refresh={
            "BAD/USDT:USDT": {
                "long": {"size": 1.0},
                "short": {"size": 0.0},
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="BAD/USDT:USDT: missing min qty and qty step",
    ):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.refresh_authoritative_state_calls == 1
    assert bot.market_specific_settings_calls == 2


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


@pytest.mark.asyncio
async def test_init_markets_fails_before_refresh_when_supported_state_invalid(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {}),
    )

    async def _update_exchange_config(_attempt):
        return None

    def _assert_supported_live_state(call_number):
        if call_number == 1:
            raise FatalBotException("unsupported config state")

    bot = _FakeBot(_update_exchange_config, _assert_supported_live_state)

    with pytest.raises(FatalBotException, match="unsupported config state"):
        await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.abstraction_refresh_calls == 1
    assert bot.assert_supported_live_state_calls == 1
    assert bot.positions_balance_calls == 0
    assert bot.open_orders_calls == 0
    assert bot.min_cost_calls == 0


@pytest.mark.asyncio
async def test_init_markets_uses_staged_refresh_for_bybit(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {}),
    )

    async def _update_exchange_config(_attempt):
        return None

    bot = _FakeBot(_update_exchange_config)

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.refresh_authoritative_state_calls == 1
    assert bot.positions_balance_calls == 0
    assert bot.open_orders_calls == 0
    assert bot.min_cost_calls == 1


@pytest.mark.asyncio
async def test_init_markets_waits_for_initial_balance_consistency(monkeypatch):
    import passivbot as pb_mod

    async def _load_markets(*_args, **_kwargs):
        return {"BTC/USDT:USDT": {"id": "BTCUSDT"}}

    monkeypatch.setattr(pb_mod, "load_markets", _load_markets)
    monkeypatch.setattr(
        pb_mod,
        "filter_markets",
        lambda *_args, **_kwargs: (["BTC/USDT:USDT"], [], {}),
    )

    async def _update_exchange_config(_attempt):
        return None

    bot = _FakeBot(_update_exchange_config)
    bot.exchange = "bitunix"
    bot.stop_signal_received = False
    bot._last_authoritative_block_reason = "balance_consistency_check"
    refresh_results = iter([False, False, True])
    sleeps = []

    async def _refresh_authoritative_state():
        bot.refresh_authoritative_state_calls += 1
        return next(refresh_results)

    async def _sleep_unless_shutdown(seconds, *, stage):
        sleeps.append((seconds, stage))

    bot.refresh_authoritative_state = _refresh_authoritative_state
    bot._sleep_unless_shutdown = _sleep_unless_shutdown

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.refresh_authoritative_state_calls == 3
    assert sleeps == [
        (5.0, "initial_balance_consistency_check"),
        (5.0, "initial_balance_consistency_check"),
    ]
    assert bot.min_cost_calls == 1


@pytest.mark.asyncio
async def test_init_markets_with_unavailable_overrides_completes_mode_lookup(monkeypatch):
    import passivbot as pb_mod

    class Bot(_FakeBot, pb_mod.Passivbot):
        init_coin_overrides = pb_mod.Passivbot.init_coin_overrides

        async def update_effective_min_cost(self):
            # Exercise the real symbol/mode lookup that sizing consumes at startup.
            self.sizing_symbols = self.get_symbols_approved_or_has_pos()
            self.min_cost_calls += 1

    async def load_markets(*args, **kwargs):
        return {"AAA/USDT:USDT": {"id": "AAAUSDT", "active": True}}

    async def update_exchange_config(attempt):
        return None

    monkeypatch.setattr(pb_mod, "load_markets", load_markets)
    monkeypatch.setattr(pb_mod, "filter_markets", lambda *a, **kw: ({"AAA/USDT:USDT"}, {}, {}))
    bot = Bot(update_exchange_config)
    bot.config = {
        "live": {"forced_mode_long": "", "forced_mode_short": ""},
        "coin_overrides": {"AAA": {}, "UNLISTED": {}},
    }
    bot.positions = {}
    bot.approved_coins_minus_ignored_coins = {"long": {"AAA/USDT:USDT"}, "short": set()}
    bot.coin_to_symbol = lambda coin, verbose=True: f"{coin}/USDT:USDT"
    bot._equity_hard_stop_enabled = lambda pside: False

    await pb_mod.Passivbot.init_markets(bot, verbose=False)

    assert bot.refresh_authoritative_state_calls == 1
    assert bot.min_cost_calls == 1
    assert bot.sizing_symbols == {"AAA/USDT:USDT"}
    assert bot.coin_overrides == {"AAA/USDT:USDT": {}}


@pytest.mark.asyncio
async def test_hourly_market_refresh_does_not_drain_runtime_commitments(monkeypatch):
    from unittest.mock import AsyncMock
    import passivbot as pb
    from live.hsl_protection import ProtectionHealth, Health, Scope
    bot = _FakeBot(AsyncMock())
    bot._bot_ready = True
    health = ProtectionHealth()
    health.scopes[Scope('coin', 'long', 'A')] = Health(exit_committed=True, exit_started_ms=100)
    bot._hsl_protection_health = health
    monkeypatch.setattr(pb, 'load_markets', AsyncMock(return_value={'A': {'id': 'A'}}))
    monkeypatch.setattr(pb, 'filter_markets', lambda *a, **kw: ({'A'}, {}, {}))
    drain = AsyncMock(side_effect=AssertionError('runtime refresh must not execute orders'))
    monkeypatch.setattr(pb.risk_input_recovery, 'drain_startup_commitments', drain)
    await Passivbot.init_markets(bot)
    drain.assert_not_awaited()
    assert health.pending_exits()
    assert bot.refresh_authoritative_state_calls == 1


@pytest.mark.asyncio
async def test_protective_startup_initializes_account_mode_before_drain(monkeypatch):
    from unittest.mock import AsyncMock
    import passivbot as pb
    from live.hsl_protection import ProtectionHealth, Health, Scope
    calls = []
    async def mode(attempt):
        calls.append('mode')
    bot = _FakeBot(mode)
    bot._equity_hard_stop_signal_mode = lambda: 'coin'
    bot._equity_hard_stop_enabled = lambda *a, **kw: True
    bot.bot_value = lambda *a: 1.0
    bot._prepare_protective_account = AsyncMock(side_effect=lambda: calls.append('mode'))
    health = ProtectionHealth()
    health.scopes[Scope('coin', 'long', 'A')] = Health(exit_committed=True, exit_started_ms=100)
    bot._hsl_protection_health = health
    monkeypatch.setattr(pb, 'load_markets', AsyncMock(return_value={'A': {'id': 'A'}}))
    monkeypatch.setattr(pb, 'filter_markets', lambda *a, **kw: ({'A'}, {}, {}))
    async def drain(owner):
        assert calls == ['mode']
        owner.stop_signal_received = True
    monkeypatch.setattr(pb.risk_input_recovery, 'drain_startup_commitments', drain)
    await Passivbot.init_markets(bot)
    assert bot.refresh_authoritative_state_calls == 0
    assert bot.exchange_config_ready_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('uta', [True, False])
@pytest.mark.parametrize('position_mode', ['hedge_mode', 'one_way_mode', None, 'unknown'])
@pytest.mark.parametrize('contracts', [1.0, 0.0])
async def test_bitget_protective_snapshot_checks_mode_after_routing(uta, position_mode, contracts):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import ccxt.async_support as ccxt
    from exchanges.bitget import BitgetBot
    bot = BitgetBot.__new__(BitgetBot)
    bot.exchange = 'bitget'
    calls = []
    async def detect():
        calls.append('detect')
        if not uta:
            raise RuntimeError('{"code":"40084"}')
        return {}
    async def positions():
        assert bot.is_uta is uta
        assert bot.cca.options['uta'] is uta and bot.ccp.options['uta'] is uta
        calls.append('positions')
        # Exercise the installed parser for both native contracts, without I/O.
        client = ccxt.bitget()
        row = {'symbol': 'BTCUSDT', 'total': str(contracts),
               'holdMode' if uta else 'posMode': position_mode}
        market = {'id': 'BTCUSDT', 'symbol': 'BTC/USDT:USDT', 'contractSize': 1, 'contract': True}
        return [client.parse_position(row, market)]
    bot.cca = SimpleNamespace(options={}, private_uta_get_v3_account_assets=detect,
                              set_position_mode=AsyncMock(), fetch_balance=AsyncMock(),
                              fetch_positions=AsyncMock(side_effect=positions))
    bot.ccp = SimpleNamespace(options={})
    await bot._prepare_protective_account()
    _, snapshot = await bot.capture_positions_snapshot()
    if contracts and position_mode != 'hedge_mode':
        with pytest.raises(FatalBotException, match='requires existing hedge position mode'):
            bot._validate_protective_position_snapshot(snapshot)
    else:
        bot._validate_protective_position_snapshot(snapshot)
    assert calls == ['detect', 'positions']
    bot.cca.fetch_balance.assert_not_awaited()
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('hedged', [True, False, None])
@pytest.mark.parametrize('exchange', ['bitunix', 'binance', 'kucoin'])
async def test_protective_preflight_is_read_only_and_requires_existing_mode(hedged, exchange):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from exchanges.bitunix import BitunixBot
    from exchanges.binance import BinanceBot
    from exchanges.kucoin import KucoinBot
    cls = {'bitunix': BitunixBot, 'binance': BinanceBot, 'kucoin': KucoinBot}[exchange]
    bot = cls.__new__(cls)
    bot.cca = SimpleNamespace(fetch_position_mode=AsyncMock(return_value={'hedged': hedged}),
                              fetch_balance=AsyncMock(), set_position_mode=AsyncMock())
    if hedged is True:
        await bot._prepare_protective_account()
    else:
        with pytest.raises(RuntimeError, match='existing hedge'):
            await bot._prepare_protective_account()
    bot.cca.fetch_balance.assert_not_awaited()
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('position_idx', [0, 1, 2])
async def test_bybit_protective_snapshot_uses_held_position_mode(position_idx):
    from unittest.mock import AsyncMock
    from exchanges.bybit import BybitBot
    bot = BybitBot.__new__(BybitBot)
    positions = [{'size': 1.0, 'info': {'positionIdx': position_idx}}]
    if position_idx == 0:
        with pytest.raises(FatalBotException, match='existing hedge'):
            bot._validate_protective_position_snapshot(positions)
    else:
        bot._validate_protective_position_snapshot(positions)


@pytest.mark.asyncio
@pytest.mark.parametrize('pos_mode', ['long_short_mode', 'net_mode'])
async def test_okx_protective_preflight_discovers_mode_without_writes(pos_mode):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from exchanges.okx import OKXBot
    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True
    bot.cca = SimpleNamespace(private_get_account_config=AsyncMock(return_value={
        'data': [{'posMode': pos_mode, 'acctLv': '2'}]}), set_position_mode=AsyncMock())
    if pos_mode == 'net_mode':
        with pytest.raises(RuntimeError, match='requires'):
            await bot._prepare_protective_account()
    else:
        await bot._prepare_protective_account()
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['disabled', 'mode'])
async def test_obsolete_startup_commitment_retires_before_account_preflight(monkeypatch, tmp_path, change):
    from unittest.mock import AsyncMock
    import passivbot as pb
    from live.hsl_protection import ProtectionHealth, Health, Scope

    bot = _FakeBot(AsyncMock())
    bot._equity_hard_stop_signal_mode = lambda: 'pside' if change == 'mode' else 'coin'
    bot._equity_hard_stop_enabled = lambda *a, **kw: change != 'disabled'
    bot._prepare_protective_account = AsyncMock(side_effect=RuntimeError('existing one-way mode'))
    health = ProtectionHealth(tmp_path / 'protection.json')
    health.scopes[Scope('coin', 'long', 'A')] = Health(exit_committed=True, exit_started_ms=100)
    health.save()
    bot._hsl_protection_health = health
    monkeypatch.setattr(pb, 'load_markets', AsyncMock(return_value={'A': {'id': 'A'}}))
    monkeypatch.setattr(pb, 'filter_markets', lambda *a, **kw: ({'A'}, {}, {}))
    drain = AsyncMock(side_effect=AssertionError('obsolete scope must not execute'))
    monkeypatch.setattr(pb.risk_input_recovery, 'drain_startup_commitments', drain)

    await Passivbot.init_markets(bot)

    bot._prepare_protective_account.assert_not_awaited()
    drain.assert_not_awaited()
    assert bot.update_exchange_config_calls == 1
    assert bot.refresh_authoritative_state_calls == 1
    assert not health.scopes
    assert not ProtectionHealth(health.path).scopes


@pytest.mark.asyncio
@pytest.mark.parametrize('data', [[], [{}], [{'posMode': None}], [{'posMode': 'unknown'}],
                                  [{'posMode': 'long_short_mode'}, {'posMode': 'net_mode'}]])
async def test_okx_protective_preflight_requires_explicit_unambiguous_mode(data):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from exchanges.okx import OKXBot
    bot = OKXBot.__new__(OKXBot)
    bot.okx_dual_side = True  # Prior/default state cannot authorize this read.
    bot.cca = SimpleNamespace(private_get_account_config=AsyncMock(return_value={'data': data}),
                              set_position_mode=AsyncMock())
    with pytest.raises(RuntimeError, match='Unable to detect'):
        await bot._prepare_protective_account()
    assert bot.okx_dual_side is False
    bot.cca.set_position_mode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('exchange', ['bitget', 'weex', 'bybit'])
async def test_protective_refresh_rechecks_newly_held_position_before_applying_orders(exchange):
    from unittest.mock import AsyncMock
    from exchanges.bitget import BitgetBot
    from exchanges.weex import WeexBot
    from exchanges.bybit import BybitBot
    from live.state_refresh import refresh_protective_authoritative_state
    cls = {'bitget': BitgetBot, 'weex': WeexBot, 'bybit': BybitBot}[exchange]
    bot = cls.__new__(cls)
    bot.stop_signal_received = False
    bot._begin_authoritative_refresh_epoch = lambda: None
    held = {'symbol': 'BTC/USDT:USDT', 'position_side': 'long', 'size': 1.0,
            'price': 100.0, 'hedged': False, 'info': {'separatedMode': 'SEPARATED', 'positionIdx': 0}}
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(side_effect=[
        {'positions': [], 'open_orders': []}, {'positions': [held], 'open_orders': []}])
    # Stop the accepted flat refresh after validation; no unrelated machinery needed.
    bot._apply_open_orders_snapshot = AsyncMock(return_value=False)
    assert not await refresh_protective_authoritative_state(bot, require_balance=False)
    with pytest.raises(FatalBotException, match='protective execution requires'):
        await refresh_protective_authoritative_state(bot, require_balance=False)
    assert bot._apply_open_orders_snapshot.await_count == 1
