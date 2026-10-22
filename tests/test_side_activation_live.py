"""Activation blocks increases without changing source approval or order ownership."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.schema import get_template_config
from live import reconciler
from passivbot import Passivbot
from test_side_activation_native import activation_snapshot
from test_orchestrator_json_api import compute


@pytest.mark.parametrize('kind', ['trailing_martingale', 'ema_anchor'])
@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('disapproved_mode', ['graceful_stop', 'manual'])
@pytest.mark.parametrize('forced_mode', [None, 'manual', 'tp_only'])
@pytest.mark.parametrize('activation,capacity', [(False, 1), (False, 0), (True, 0)])
def test_refresh_and_repeated_plans_keep_approved_held_policy_when_disabled(
    kind, side, disapproved_mode, forced_mode, activation, capacity, caplog,
):
    import passivbot_rust as native
    symbol = 'BTC/USDT:USDT'
    config = get_template_config()
    config['bot']['portfolio']['strategy_kind'] = kind
    config['bot']['global']['universe']['approved_coins'] = ['BTC']
    for direction in ('long', 'short'):
        config['bot'][direction]['enabled'] = direction == side
        config['bot'][direction]['universe']['approved_coins'] = None
    bot = Passivbot.__new__(Passivbot)
    bot.config = prepare_config(config, live_only=True, verbose=False)
    bot.config['bot'][side]['enabled'] = activation
    bot.exchange = 'binance'
    bot.eligible_symbols = {symbol}
    bot.markets_dict = {symbol: {'active': True}}
    bot.positions = {symbol: {direction: {'size': (1 if direction == 'long' else -1) if direction == side else 0}
                              for direction in ('long', 'short')}}
    bot.coin_overrides = {}
    bot.open_orders = {}
    bot._disabled_psides_logged = set()
    bot._unsupported_coin_warnings = set()
    bot._runtime_forced_modes = {'long': {}, 'short': {}}
    bot.PB_mode_stop = dict.fromkeys(('long', 'short'), disapproved_mode)
    bot.coin_to_symbol = lambda coin, verbose=True: symbol
    bot.bot_value = lambda direction, key: capacity
    bot.config_get = lambda path, symbol=None: forced_mode if path == ['live', 'modes', side] else None
    bot._log_coin_symbol_fallback_summary = lambda: None
    for _ in range(2):
        bot.refresh_approved_ignored_coins_lists()
        assert symbol in bot.approved_coins[side]
        assert symbol in bot.approved_coins_minus_ignored_coins[side]
        mode = bot._orchestrator_mode_override(side, symbol)
        assert mode == forced_mode
        value = activation_snapshot(kind, side, held=True, mode=mode or 'normal')
        value['global']['side_activation'][side] = activation
        if not capacity:
            value['global']['global_bot_params'][side]['n_positions'] = 0
            value['symbols'][0][side]['bot_params']['n_positions'] = 0
        out = compute(native, value)
        reconciler.validate_rust_orchestrator_output(out, {0: symbol}, value)
        assert not any(order['order_type'].startswith('entry_') for order in out['orders'])
        assert bool(out['orders']) is (forced_mode != 'manual')
    assert 'refresh failed' not in caplog.text
    # Actual removal of approval still uses the configured disapproved mode.
    bot.config['bot']['global']['universe']['approved_coins'] = []
    bot.config_get = lambda path, symbol=None: None
    bot.refresh_approved_ignored_coins_lists()
    assert bot._orchestrator_mode_override(side, symbol) == disapproved_mode


@pytest.mark.parametrize('mode,expect_cancel', [('normal', True), ('graceful_stop', True),
    ('manual', False), ('tp_only', False), ('tp_only_with_active_entry_cancellation', True)])
def test_disabled_entry_reconciliation_preserves_explicit_mode_ownership(mode, expect_cancel):
    symbol = 'BTC/USDT:USDT'
    bot = SimpleNamespace(PB_modes={'long': {symbol: mode}, 'short': {symbol: 'normal'}})
    entry = {'symbol': symbol, 'position_side': 'long', 'reduce_only': False}
    close = {'symbol': symbol, 'position_side': 'long', 'reduce_only': True}
    original = deepcopy([entry, close])
    cancelled, created = reconciler.apply_mode_filters(bot, symbol, [entry, close], [close])
    assert (entry in cancelled) is expect_cancel
    assert created == ([] if mode == 'manual' else [close])
    assert [entry, close] == original
