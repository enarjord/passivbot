"""Offline integration of the staged revised owner with the real bot and Rust."""
import argparse
import json

import hjson
import pytest

from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
from config_utils import load_config
from config import prepare_config
from config.hsl_revised import generated_template
from live import hsl_revised_live
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('path', ['wave', 'loop', 'cancel_first', 'recovered_before_write', 'malformed_before_write'])
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_revised_protective_wave_uses_actual_executor_without_history(tmp_path, monkeypatch, mode, path, side):
    import config.hsl_revised as config_hsl
    import passivbot_rust as pbr
    assert not getattr(pbr, '__is_stub__', False)
    # Only this offline test bypasses the public activation gate.
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    user = f'fake_revised_live_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    if side == 'short':
        from copy import deepcopy
        cfg['bot']['short'] = deepcopy(cfg['bot']['long'])
        cfg['bot']['long']['hsl']['enabled'] = False
        cfg['bot']['long']['risk'].update(n_positions=0, total_wallet_exposure_limit=0.)
    blocks = [cfg['bot']['hsl']] if mode == 'unified' else [cfg['bot'][side]['hsl']]
    for block in blocks:
        block.update(enabled=True, red_threshold=.06, ema_span_minutes=1000., panic_close_order_type='market')
    cfg['live']['pnls_max_lookback_days'] = 1.
    cfg = {key: value for key, value in cfg.items() if not key.startswith('_')}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account']['fills'] = []
    scenario['account']['positions'][0]['position_side'] = side
    scenario['run_initial_cycle'] = True
    scenario_path = tmp_path / 'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        symbol = 'BTC/USDT:USDT'
        bot.cca.now_ms += 60_000
        bot.cca.get_current_step()['prices'][symbol] = 90. if side == 'long' else 110.
        if path == 'cancel_first':
            bot.cca._load_boot_order(dict(id='pending-entry', symbol=symbol, position_side=side,
                side='buy' if side == 'long' else 'sell', amount=1., price=80. if side == 'long' else 120.))
        bot.market_snapshot_provider._cache.clear()
        await bot.refresh_protective_authoritative_state(require_balance=True)
        instance = hsl_revised_live.owner(bot)
        instance.poll_inputs()
        instance.sources.clear()  # Exercise total historical loss after startup.
        bot._pnls_manager = None
        if path in {'recovered_before_write', 'malformed_before_write'}:
            from live import market_data
            from live.hsl_revised_runtime import InvalidHslOutput
            original_filter = market_data.filter_fresh_market_snapshot_creations
            native = pbr.hsl_revised_evaluate
            boundary_reached = []
            async def change_at_boundary(current_bot, orders, **kwargs):
                boundary_reached.append(True)
                if path == 'recovered_before_write':
                    bot.cca.get_current_step()['prices'][symbol] = 100.
                    bot.market_snapshot_provider._cache.clear()
                result = await original_filter(current_bot, orders, **kwargs)
                if path == 'malformed_before_write':
                    monkeypatch.setattr(pbr, 'hsl_revised_evaluate', lambda value: '{"decision": null}')
                return result
            monkeypatch.setattr(market_data, 'filter_fresh_market_snapshot_creations', change_at_boundary)
            try:
                if path == 'malformed_before_write':
                    with pytest.raises(InvalidHslOutput):
                        await instance.protect()
                else:
                    await instance.protect()
            finally:
                monkeypatch.setattr(pbr, 'hsl_revised_evaluate', native)
                monkeypatch.setattr(market_data, 'filter_fresh_market_snapshot_creations', original_filter)
            assert boundary_reached
            assert not any(c['method'] == 'create_order' for c in bot.cca.export_request_log())
        elif path == 'wave':
            assert await instance.protect()
        elif path == 'cancel_first':
            # Discovering an external order and later confirming its cancellation
            # invalidate account cohorts. Bounded subsequent polls must finish
            # protection without awaiting history or ordinary planning.
            for _ in range(4):
                await bot.refresh_protective_authoritative_state(require_balance=True)
                await instance.protect()
                if bot.cca.positions[symbol, side]['size'] == 0.:
                    break
            assert any(c['method'] == 'cancel_order' for c in bot.cca.export_request_log())
        else:
            import asyncio
            entered = asyncio.Event()
            released = asyncio.Event()
            async def stalled_history(**kwargs):
                entered.set()
                await released.wait()
                return False
            async def stalled_preparation():
                await released.wait()
            bot.update_pnls = stalled_history
            bot.prepare_planning_universe = stalled_preparation
            async def stalled_candles(*args, **kwargs):
                await released.wait()
                return []
            bot.cm.get_candles = stalled_candles
            # Start GREEN. The loss arrives only after repair has stalled.
            bot.cca.get_current_step()['prices'][symbol] = 100.
            bot.market_snapshot_provider._cache.clear()
            cycles = []
            async def step(*args, **kwargs):
                if kwargs.get('stage') != 'revised_execution_delay':
                    await asyncio.sleep(.01)
                    return
                cycles.append(True)
                await asyncio.sleep(.01)
                if len(cycles) == 1:
                    assert entered.is_set()
                    bot.cca.now_ms += 60_000
                    bot.cca.get_current_step()['prices'][symbol] = 90. if side == 'long' else 110.
                    bot.market_snapshot_provider._cache.clear()
                elif bot.positions[symbol][side]['size'] == 0.:
                    bot.stop_signal_received = True
                assert len(cycles) < 10
            original_sleep = bot._sleep_unless_shutdown
            bot._sleep_unless_shutdown = step
            try:
                await asyncio.wait_for(bot.run_execution_loop(), 10.)
                assert entered.is_set() and not released.is_set()
            finally:
                released.set()
                bot._sleep_unless_shutdown = original_sleep
            bot.stop_signal_received = False
        await bot.refresh_protective_authoritative_state()
        if path.endswith('before_write'):
            assert abs(bot.positions[symbol][side]['size']) == 5.
            assert bot.cca.fills == []
        else:
            assert bot.positions[symbol][side]['size'] == 0.
            assert any(f.get('reduceOnly') for f in bot.cca.fills)
        assert not hasattr(bot, '_hsl_protection_health')
        completed.append(True)
        return {'revised_closed': True}

    monkeypatch.setattr(runner, '_run_fake_cycle', exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
                                  max_steps=1, output_dir=str(tmp_path), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.parametrize('change', ['balance', 'position', 'quote', 'stale_quote', 'policy', 'disabled', 'generation', 'restart'])
def test_revised_wave_is_not_execution_authority_after_inputs_change(monkeypatch, change):
    from dataclasses import replace
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW-200)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    order = {'symbol': SYMBOL, 'position_side': 'long'}
    instance.bind(wave, (), (order,))
    assert instance.admit(order)
    if change == 'balance':
        bot.get_raw_balance = lambda: 999.
    elif change == 'position':
        bot.positions[SYMBOL]['long']['size'] = 9.
    elif change == 'quote':
        instance.quotes[SYMBOL] = replace(quotes()[SYMBOL], last=110., bid=110., ask=110.)
    elif change == 'stale_quote':
        instance.quotes[SYMBOL] = replace(quotes()[SYMBOL], fetched_ms=NOW-10_001)
    elif change == 'policy':
        bot.config['bot']['long']['hsl']['red_threshold'] = .99
    elif change == 'disabled':
        bot.config['bot']['long']['hsl']['enabled'] = False
    elif change == 'generation':
        bot._account_invalidation_generation = 1
    else:
        instance = hsl_revised_live.Owner(bot)
        instance.quotes.update(quotes())
    assert not instance.admit(order)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_revised_green_can_plan_entries_without_hsl_history(tmp_path, monkeypatch, mode):
    import asyncio
    import config.hsl_revised as config_hsl
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    user = f'fake_revised_green_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    for side in ('long', 'short'):
        cfg['bot'][side]['unstuck']['enabled'] = False
        cfg['bot'][side]['risk']['entry_cooldown_minutes'] = 0.
    cfg['live']['max_realized_loss_pct'] = 1.
    cfg['live']['approved_coins']['long'] = ['BTC']
    cfg['live']['pnls_max_lookback_days'] = 1.
    policy = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot']['long']['hsl']
    policy.update(enabled=True, ema_span_minutes=1., red_threshold=.5)
    cfg = {key: value for key, value in cfg.items() if not key.startswith('_')}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account'].update(positions=[], fills=[])
    scenario['run_initial_cycle'] = True
    scenario_path = tmp_path / 'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        instance = hsl_revised_live.owner(bot)
        instance.poll_inputs()
        instance.sources.clear()
        bot._pnls_manager = None
        bot._hsl_revised_fill_capture_interval = None
        ledger = bot._ensure_freshness_ledger()
        ledger.surfaces['fills'].epoch = -1
        assert 'fills' not in bot._staged_planner_required_surfaces()
        await bot.refresh_protective_authoritative_state()
        plan = await instance._ordinary_plan()
        assert plan is not None
        assert bot.active_symbols, (bot.approved_coins, bot.approved_coins_minus_ignored_coins, bot.config['live']['approved_coins'])
        cancels, creates, snapshot = plan
        assert any(not order['reduce_only'] for order in creates)
        bot._current_planning_snapshot = snapshot
        await bot.execute_order_plan_to_exchange(cancels, creates)
        assert any(call['method'] == 'create_order' for call in bot.cca.export_request_log())
        completed.append(True)
        return {'green_entries': True}

    monkeypatch.setattr(runner, '_run_fake_cycle', exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
                                  max_steps=1, output_dir=str(tmp_path), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.parametrize('obstacle', ['unfilled_limit', 'missing_quote'])
async def test_revised_other_coin_closes_while_first_coin_is_pending(tmp_path, monkeypatch, obstacle):
    from copy import deepcopy
    import config.hsl_revised as config_hsl
    from live.market_snapshot import MarketSnapshotUnavailable
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    user = f'fake_revised_fair_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, 'coin')
    cfg['bot']['long']['hsl'].update(enabled=True, ema_span_minutes=1., red_threshold=.06,
                                    panic_close_order_type='limit')
    cfg['live']['pnls_max_lookback_days'] = 1.
    cfg = {key: value for key, value in cfg.items() if not key.startswith('_')}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account']['fills'] = []
    scenario['run_initial_cycle'] = True
    first, second = 'BTC/USDT:USDT', 'ETH/USDT:USDT'
    scenario['symbols'][second] = deepcopy(scenario['symbols'][first])
    scenario['replay']['symbols'][second] = deepcopy(scenario['replay']['symbols'][first])
    scenario['account']['positions'].append(dict(symbol=second, position_side='long', qty=5., price=100.))
    scenario_path = tmp_path / 'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        instance = hsl_revised_live.owner(bot)
        instance.poll_inputs()
        instance.sources.clear()
        bot._pnls_manager = None
        bot.coin_overrides[second] = {'bot': {'long': {'hsl': {'panic_close_order_type': 'market'}}}}
        crossed = bot.cca._limit_crossed_now
        # Model a venue leaving the aggressive limit unfilled. The fake client
        # still owns acknowledgements, open orders, and subsequent fills.
        bot.cca._limit_crossed_now = lambda order: False if order['symbol'] == first else crossed(order)
        bot.cca.now_ms += 60_000
        bot.cca.get_current_step()['prices'].update({first: 90., second: 100.})
        bot.market_snapshot_provider._cache.clear()
        await bot.refresh_protective_authoritative_state()
        assert await instance.protect()
        await bot.refresh_protective_authoritative_state()
        assert bot.positions[first]['long']['size'] == 5.
        assert bot.open_orders[first]
        if obstacle == 'missing_quote':
            original_quotes = bot._get_orchestrator_market_snapshots
            async def quotes(symbols):
                if first in symbols:
                    raise MarketSnapshotUnavailable('offline quote outage')
                return await original_quotes(symbols)
            bot._get_orchestrator_market_snapshots = quotes
        bot.cca.get_current_step()['prices'][second] = 80.
        bot.market_snapshot_provider._cache.clear()
        assert await instance.protect()
        await bot.refresh_protective_authoritative_state()
        assert bot.positions[second]['long']['size'] == 0.
        assert bot.positions[first]['long']['size'] == 5.
        completed.append(True)
        return {'independent_scope_protected': True}

    monkeypatch.setattr(runner, '_run_fake_cycle', exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
                                  max_steps=1, output_dir=str(tmp_path), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
async def test_resistant_quote_read_is_bounded_and_does_not_suspend_peer():
    import asyncio
    from collections import Counter
    from types import SimpleNamespace
    from test_hsl_revised_runtime import quotes, SYMBOL
    release = asyncio.Event()
    calls = Counter()
    async def read(symbols):
        symbol, = symbols
        calls[symbol] += 1
        if symbol == 'blocked':
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return {}
        return quotes()
    instance = hsl_revised_live.Owner(SimpleNamespace(_get_orchestrator_market_snapshots=read))
    try:
        for _ in range(2):
            result = await asyncio.wait_for(instance.acquire_quotes({'blocked', SYMBOL}), 1.)
            assert SYMBOL in result
        assert calls['blocked'] == 1
        assert len(instance._quote_tasks) == 1
    finally:
        release.set()
        await asyncio.gather(*instance._quote_tasks.values())


@pytest.mark.asyncio
async def test_quote_read_can_finish_after_the_wave_time_slice():
    import asyncio
    from types import SimpleNamespace
    from test_hsl_revised_runtime import quotes, SYMBOL
    release = asyncio.Event()
    started = 0
    async def read(symbols):
        nonlocal started
        started += 1
        await release.wait()
        return quotes()
    instance = hsl_revised_live.Owner(SimpleNamespace(_get_orchestrator_market_snapshots=read))
    assert await instance.acquire_quotes({SYMBOL}) == {}
    task = instance._quote_tasks[SYMBOL]
    assert not task.done() and not task.cancelling()
    release.set()
    result = await instance.acquire_quotes({SYMBOL})
    assert SYMBOL in result and started == 1


def test_planner_flat_padding_does_not_invalidate_an_unchanged_protective_position(monkeypatch):
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW-200)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    order = {'symbol': SYMBOL, 'position_side': 'long'}
    instance.bind(wave, (), (order,))
    # Ordinary universe preparation materializes flat account-response absences.
    bot.positions['CANDIDATE/USDT:USDT'] = {side: dict(size=0., price=0.) for side in ('long', 'short')}
    assert instance.admit(order)


def test_pending_history_request_does_not_erase_an_unchanged_remote_receipt():
    from test_hsl_revised_runtime import bot as make_bot, NOW
    from test_hsl_revised_inputs import event
    from live.hsl_revised_runtime import observe_fills, observed_fill_interval, capture_fills
    events = []
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot._pnls_manager.get_events = lambda **kwargs: events
    bot._hsl_revised_fill_observation = observe_fills(bot, (NOW-150, NOW-50))
    def interval():
        return observed_fill_interval(bot, capture_fills(events, bot.c_mults), NOW-86_400_000, NOW)
    assert interval() == (NOW-150, NOW-50)
    bot._hsl_revised_fill_capture_interval = None  # Next remote request begins.
    assert interval() == (NOW-150, NOW-50)
    events.append(event(timestamp=NOW-60_000, c_mult=1.))
    assert interval() is None
    bot._hsl_revised_fill_capture_interval = (NOW-150, NOW-50)
    assert interval() is None  # Old metadata cannot certify the correction.
    bot._hsl_revised_fill_observation = observe_fills(bot, (NOW-30, NOW-10))
    assert interval() == (NOW-30, NOW-10)
    bot._pnls_manager = None
    assert interval() is None


@pytest.mark.parametrize('change', ['correction', 'replacement', 'duplicate', 'expiry', 'expired_correction', 'order'])
def test_fill_receipt_is_atomic_and_clipped_before_first_consumer(change):
    from dataclasses import replace
    from types import SimpleNamespace
    from live.hsl_revised_runtime import observe_fills, observed_fill_interval, capture_fills
    from test_hsl_revised_runtime import bot as make_bot, NOW
    from test_hsl_revised_inputs import event
    start = NOW - 86_400_000
    events = [event(timestamp=start, c_mult=1.), event(id='later', timestamp=NOW-60_000, c_mult=1.)]
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot._pnls_manager.get_events = lambda **kwargs: events
    bot._hsl_revised_fill_observation = observe_fills(bot, (NOW-150, NOW-50))
    if change == 'correction':
        events[-1] = replace(events[-1], pnl=-100.)
    elif change == 'replacement':
        bot._pnls_manager = SimpleNamespace(get_events=lambda **kwargs: events)
    elif change == 'duplicate':
        events.append(events[-1])
    elif change == 'expiry':
        events.pop(0)
        start += 1
    elif change == 'expired_correction':
        events[0] = replace(events[0], pnl=-100.)
        start += 1
    else:
        events.reverse()
    actual = observed_fill_interval(bot, capture_fills(events, bot.c_mults), start, NOW)
    assert actual == ((NOW-150, NOW-50) if change in {'expiry', 'expired_correction', 'order'} else None)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('restart', ['always', 'never'])
@pytest.mark.parametrize('partial', [False, True])
async def test_revised_real_close_reconstructs_halt_on_fresh_bot_then_expires(tmp_path, monkeypatch, mode, restart, partial):
    import asyncio
    from copy import deepcopy
    import config.hsl_revised as config_hsl
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    symbol = 'BTC/USDT:USDT'
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    cfg['live']['pnls_max_lookback_days'] = 1.
    cfg['live']['max_realized_loss_pct'] = 1.
    for side in ('long', 'short'):
        cfg['bot'][side]['unstuck']['enabled'] = False
        cfg['bot'][side]['risk']['entry_cooldown_minutes'] = 0.
    block = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot']['long']['hsl']
    block.update(enabled=True, red_threshold=.06, ema_span_minutes=1.,
                 panic_close_order_type='market', restart_after_red_policy=restart,
                 cooldown_minutes_after_red=2.)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account']['fills'] = [dict(id='1', order='1', timestamp='2026-01-01T00:00:00Z',
        symbol=symbol, position_side='long', side='buy', amount=5., price=100., pnl=0.)]
    scenario['run_initial_cycle'] = True
    state = None
    completed = []

    async def settled_wave(bot):
        instance = hsl_revised_live.owner(bot)
        # The fake replay clock otherwise freezes on the fill's timestamp even
        # while UTC advances. Move exchange time past the fill before reading
        # account/fill snapshots, as an exchange would on a subsequent poll.
        bot.cca.now_ms += 1_000
        # Real account + remote fill reads. The first fill change may invalidate
        # the account cohort; reacquire it instead of forging receipt timestamps.
        for _ in range(2):
            await bot.refresh_protective_authoritative_state(require_balance=True)
            instance.remember_position()
            await asyncio.sleep(.002)
            await bot.update_pnls(source='hsl_revised')
        await bot.refresh_protective_authoritative_state(require_balance=True)
        marks = await instance.acquire_quotes({symbol})
        return instance.capture(marks)

    async def exercise_first(bot):
        nonlocal state
        bot.cca.now_ms += 60_000
        bot.cca.get_current_step()['prices'][symbol] = 90.
        bot.market_snapshot_provider._cache.clear()
        await bot.refresh_protective_authoritative_state(require_balance=True)
        instance = hsl_revised_live.owner(bot)
        fill = bot.cca._fill_order
        if partial:
            def partial_ioc(order, *, fill_price, liquidity):
                requested = order['amount']
                order['amount'] = requested / 2.
                fill(order, fill_price=fill_price, liquidity=liquidity)
                order.update(amount=requested, remaining=requested / 2., status='canceled')
            bot.cca._fill_order = partial_ioc
        try:
            assert await instance.protect()
        finally:
            bot.cca._fill_order = fill
        wave = await settled_wave(bot)
        assert not wave.unavailable
        assert wave.permission(symbol, 'long')[0] == ('panic' if partial else 'halted'), wave
        state = bot.cca.export_state()
        assert state['balance_total'] == (175. if partial else 150.)
        assert (state['positions'][0]['size'] if partial else len(state['positions'])) == (2.5 if partial else 0)
        completed.append('closed')
        return {'closed': True}

    async def exercise_restart(bot):
        assert not hasattr(bot, '_hsl_protection_health')
        if partial and bot.positions.get(symbol, {}).get('long', {}).get('size', 0.) != 0.:
            wave = await settled_wave(bot)
            assert wave.permission(symbol, 'long')[0] == 'panic', wave
            assert await hsl_revised_live.owner(bot).protect()
        wave = await settled_wave(bot)
        assert not wave.unavailable
        assert wave.permission(symbol, 'long')[0] == 'halted', wave
        decision, = [d for d in wave.decisions if hsl_revised_live.matches(d.scope, symbol, 'long')]
        flat_at = bot.cca.fills[-1]['timestamp']
        assert json.loads(decision.payload)['decision']['flat_at'] == flat_at
        if partial:
            # Fresh process, recovered price, no journal: remaining quantity is
            # still closed from exchange history, never a retained panic flag.
            creates = [c for c in bot.cca.export_request_log() if c['method'] == 'create_order']
            assert creates and all(c['amount'] == 2.5 and c['reduce_only'] for c in creates)
            assert bot.cca.balance_total == 175.
        # The same running bot forgets expired policy evidence without a local
        # flag. A normal cooldown needs minutes; never needs the configured day.
        bot.cca.now_ms = flat_at + (86_400_001 if restart == 'never' else 120_001)
        bot.market_snapshot_provider._cache.clear()
        wave = await settled_wave(bot)
        assert not wave.unavailable
        assert wave.permission(symbol, 'long')[0] in (None, 'normal'), wave
        completed.append('reconstructed_and_expired')
        return {'reconstructed_and_expired': True}

    for phase, callback in enumerate((exercise_first, exercise_restart)):
        user = f'fake_revised_restart_{tmp_path.name}_{phase}'
        _cleanup_fake_user_state(user)
        if phase:
            scenario = deepcopy(scenario)
            positions = [dict(symbol=p['symbol'], position_side=p['position_side'], qty=p['size'], price=p['entry_price'])
                         for p in state['positions']]
            scenario['account'] = dict(balance=state['balance_total'], positions=positions, fills=state['fills'], open_orders=[])
            scenario['boot_index'] = 3  # One minute after the actual close.
            if partial:
                for candle in scenario['replay']['symbols'][symbol]['candles'][3:]:
                    candle[1:5] = [100.] * 4
        scenario_path = tmp_path / f'scenario-{phase}.hjson'
        scenario_path.write_text(hjson.dumps(scenario))
        monkeypatch.setattr(runner, '_run_fake_cycle', callback)
        try:
            args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
                max_steps=1, output_dir=str(tmp_path / f'run-{phase}'), log_level=1, snapshot_each_step=False)
            assert await runner._async_main(args) == 0
        finally:
            _cleanup_fake_user_state(user)
    assert completed == ['closed', 'reconstructed_and_expired']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_revised_startup_services_new_red_during_stalled_warmup(tmp_path, monkeypatch, mode):
    import asyncio
    import config.hsl_revised as config_hsl
    from passivbot import Passivbot
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    user = f'fake_revised_startup_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    cfg['live']['pnls_max_lookback_days'] = 1.
    block = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot']['long']['hsl']
    block.update(enabled=True, red_threshold=.06, ema_span_minutes=1., panic_close_order_type='market')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account']['fills'] = []
    scenario['run_initial_cycle'] = True
    scenario_path = tmp_path / 'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    symbol = 'BTC/USDT:USDT'
    warmup_entered = []

    async def stalled_warmup(bot):
        assert bot.cca.positions[symbol, 'long']['size'] == 5.
        warmup_entered.append(True)
        # The generic harness suppresses writes during startup. This regression
        # explicitly exercises production startup writes against its fake client.
        bot.debug_mode = False
        bot.cca.now_ms += 60_000
        bot.cca.get_current_step()['prices'][symbol] = 90.
        bot.market_snapshot_provider._cache.clear()
        async def observe_close():
            while bot.cca.positions[symbol, 'long']['size'] != 0.:
                await asyncio.sleep(.01)
        # Warmup cannot finish until protection closes the position. A startup
        # implementation which awaits all preparation first fails this deadline.
        try:
            await asyncio.wait_for(observe_close(), 5.)
        finally:
            # Retain the harness's start_bot return boundary; otherwise normal
            # startup would enter the perpetual production execution loop.
            bot.debug_mode = True
        # The dependency is now released. Its candle warmup behavior is covered
        # separately; this test's fake dependency only controls completion time.

    async def verify(bot):
        assert warmup_entered
        assert bot.cca.positions[symbol, 'long']['size'] == 0.
        assert any(f['reduceOnly'] for f in bot.cca.fills)
        return {'protected_during_startup': True}

    monkeypatch.setattr(Passivbot, 'warmup_trading_ready_candles', stalled_warmup)
    monkeypatch.setattr(runner, '_run_fake_cycle', verify)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
    finally:
        _cleanup_fake_user_state(user)


def test_fresh_owner_cannot_recycle_an_old_orders_permission(monkeypatch):
    from dataclasses import replace
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW-200)
    old_owner = hsl_revised_live.owner(bot)
    old_wave = old_owner.capture(quotes())
    old_order = {'symbol': SYMBOL, 'position_side': 'long'}
    old_owner.bind(old_wave, (), (old_order,))
    assert old_owner.admit(old_order)
    fresh_owner = hsl_revised_live.Owner(bot)
    fresh_wave = fresh_owner.capture({SYMBOL: replace(quotes()[SYMBOL], last=100., bid=100., ask=100.)})
    fresh_order = {'symbol': SYMBOL, 'position_side': 'long'}
    fresh_owner.bind(fresh_wave, (), (fresh_order,))
    assert fresh_owner.admit(fresh_order)
    assert not fresh_owner.admit(old_order)


@pytest.mark.parametrize('expires', ['account', 'mark'])
def test_synchronous_reconstruction_must_not_outlive_current_input_freshness(monkeypatch, expires):
    from dataclasses import replace
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    from live import hsl_revised_runtime
    import utils
    clock = [NOW]
    monkeypatch.setattr(utils, 'utc_ms', lambda: clock[0])
    bot = make_bot()
    bot.get_exchange_time = lambda: clock[0]
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW-200)
    instance = hsl_revised_live.owner(bot)
    marks = quotes()
    if expires == 'mark':
        marks = {SYMBOL: replace(marks[SYMBOL], fetched_ms=NOW-9_500)}
    wave = instance.capture(marks)
    order = {'symbol': SYMBOL, 'position_side': 'long'}
    instance.bind(wave, (), (order,))
    evaluate = hsl_revised_runtime.evaluate
    def slow_evaluation(requests):
        result = evaluate(requests)
        clock[0] += 10_000 if expires == 'account' else 1_000
        return result
    monkeypatch.setattr(hsl_revised_runtime, 'evaluate', slow_evaluation)
    assert not instance.admit(order)
