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
@pytest.mark.parametrize('path', ['wave', 'loop', 'cancel_first', 'recovered_before_write', 'malformed_before_write', 'slow_projection', 'slow_sink'])
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_revised_protective_wave_uses_actual_executor_without_history(tmp_path, monkeypatch, mode, path, side):
    import passivbot_rust as pbr
    assert not getattr(pbr, '__is_stub__', False)
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
            native = pbr.hsl_revised_evaluate_grids
            boundary_reached = []
            async def change_at_boundary(current_bot, orders, **kwargs):
                boundary_reached.append(True)
                if path == 'recovered_before_write':
                    bot.cca.get_current_step()['prices'][symbol] = 100.
                    bot.market_snapshot_provider._cache.clear()
                result = await original_filter(current_bot, orders, **kwargs)
                if path == 'malformed_before_write':
                    monkeypatch.setattr(pbr, 'hsl_revised_evaluate_grids', lambda value, grids: '{"decision": null}')
                return result
            monkeypatch.setattr(market_data, 'filter_fresh_market_snapshot_creations', change_at_boundary)
            try:
                if path == 'malformed_before_write':
                    with pytest.raises(InvalidHslOutput):
                        await instance.protect()
                else:
                    await instance.protect()
            finally:
                monkeypatch.setattr(pbr, 'hsl_revised_evaluate_grids', native)
                monkeypatch.setattr(market_data, 'filter_fresh_market_snapshot_creations', original_filter)
            assert boundary_reached
            assert not any(c['method'] == 'create_order' for c in bot.cca.export_request_log())
        elif path in {'wave', 'slow_projection', 'slow_sink'}:
            report_calls = []
            clock_offset = [0]
            if path.startswith('slow_'):
                import utils
                from live import hsl_revised_diagnostics as reporting
                original_clock = utils.utc_ms
                monkeypatch.setattr(utils, 'utc_ms', lambda: original_clock() + clock_offset[0])
                original_report = reporting._row if path == 'slow_projection' else bot._emit_live_event
                def slow_report(*args, **kwargs):
                    from live.event_bus import EventTypes
                    if path == 'slow_sink' and args[0] != EventTypes.HSL_STATUS:
                        return original_report(*args, **kwargs)
                    report_calls.append(any(c['method'] == 'create_order' for c in bot.cca.export_request_log()))
                    clock_offset[0] += 20_000
                    return original_report(*args, **kwargs)
                if path == 'slow_projection':
                    monkeypatch.setattr(reporting, '_row', slow_report)
                else:
                    bot._emit_live_event = slow_report
                bot._hsl_revised_diagnostic_event = None
            assert await instance.protect()
            if path.startswith('slow_'):
                assert report_calls and all(report_calls)
                clock_offset[0] = 0
            from live.hsl_revised_diagnostics import snapshot
            from utils import utc_ms
            diagnostics = snapshot(bot, now_ms=int(utc_ms()))
            assert diagnostics['engine'] == 'revised'
            assert diagnostics['signal_mode'] == mode
            assert diagnostics['counts']['red'] >= 1
            if mode == 'unified':
                assert diagnostics['scope_count'] == 1
                assert diagnostics['scopes'][0]['pside'] is None
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
        if path == 'wave':
            # A just-observed position delta can require one newer confirmation.
            await bot.refresh_protective_authoritative_state()
            monitor = await bot._build_monitor_snapshot()
            expected_equity = bot.get_raw_balance() + sum(
                position['upnl'] for sides in monitor['positions'].values()
                for position in sides.values() if position['size'] != 0.)
            assert monitor['account']['equity'] == pytest.approx(expected_equity)
            assert monitor['health']['equity'] == pytest.approx(expected_equity)
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


@pytest.mark.parametrize('change', ['balance', 'position', 'quote', 'stale_quote', 'policy', 'disabled', 'generation', 'restart', 'open_orders'])
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
    if change == 'open_orders':
        bot.open_orders[SYMBOL] = [dict(id='new-order', qty=1., price=100.)]
    elif change == 'balance':
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
@pytest.mark.parametrize('raw_balance_drift', [False, True])
@pytest.mark.parametrize('execution', ['direct', 'owner_refresh_drift'])
async def test_revised_green_can_plan_entries_without_hsl_history(tmp_path, monkeypatch, mode, raw_balance_drift, execution):
    import asyncio
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
        cancels, creates, snapshot, wave = plan
        assert any(not order['reduce_only'] for order in creates)
        if raw_balance_drift:
            raw, strategy = bot.get_raw_balance(), bot.get_hysteresis_snapped_balance()
            bot.cca.balance_total += .001
            assert await bot.refresh_protective_authoritative_state(require_balance=True)
            assert bot.get_raw_balance() != raw
            assert bot.get_hysteresis_snapped_balance() == strategy
        bot._current_planning_snapshot = snapshot
        if execution == 'direct':
            await bot.execute_order_plan_to_exchange(cancels, creates)
        else:
            # Exercise the production owner, with every subsequent confirming
            # account read changing raw balance inside the same sizing band.
            async def ready():
                return plan
            instance._ordinary = asyncio.create_task(ready())
            await instance._ordinary
            refresh = bot.refresh_protective_authoritative_state
            async def drifting_refresh(**kwargs):
                bot.cca.balance_total += .001
                return await refresh(**kwargs)
            monkeypatch.setattr(bot, 'refresh_protective_authoritative_state', drifting_refresh)
            result = await instance.cycle()
            assert result['ordinary_completed']
        assert any(call['method'] == 'create_order' for call in bot.cca.export_request_log()) == (not raw_balance_drift)
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
    from live.market_snapshot import MarketSnapshotUnavailable
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
@pytest.mark.parametrize('stage', ['warmup_trading_ready_candles', '_exchange_config_write_ready', 'update_exchange_config', 'warmup_transient_failure'])
async def test_revised_startup_services_new_red_during_stalled_warmup(tmp_path, monkeypatch, mode, stage):
    import asyncio
    from passivbot import Passivbot
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
        # A transient failure after protection must still let startup continue.
        if stage == 'warmup_transient_failure':
            from ccxt.base.errors import NetworkError
            raise NetworkError('offline simulated candle outage')
        # The dependency is now released; ordinary warmup/configuration behavior
        # has separate coverage. This dependency only controls completion time.
        return True

    async def verify(bot):
        assert warmup_entered
        assert bot.cca.positions[symbol, 'long']['size'] == 0.
        assert any(f['reduceOnly'] for f in bot.cca.fills)
        return {'protected_during_startup': True}

    from exchanges.fake import FakeBot
    monkeypatch.setattr(FakeBot, 'warmup_trading_ready_candles' if stage == 'warmup_transient_failure' else stage,
                        stalled_warmup)
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


@pytest.mark.parametrize('change', ['undated', 'unattributed', 'quality_only', 'new_pair_undated'])
def test_fill_receipt_rejects_completeness_changes(change):
    from dataclasses import replace
    from test_hsl_revised_runtime import bot as make_bot, NOW
    from test_hsl_revised_inputs import event
    from live.hsl_revised_runtime import observe_fills, observed_fill_interval, capture_fills
    rows = [event(timestamp=NOW-60_000, c_mult=1.)]
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot._pnls_manager.get_events = lambda **kwargs: rows
    bot._hsl_revised_fill_observation = observe_fills(bot, (NOW-150, NOW-50))
    if change == 'quality_only':
        rows[0] = replace(rows[0], fee_quality='estimated')
    elif change == 'unattributed':
        rows.append(replace(rows[0], symbol=''))
    else:
        rows.append(replace(rows[0], timestamp=0,
                            symbol='OTHER/USDT:USDT' if change == 'new_pair_undated' else rows[0].symbol))
    assert observed_fill_interval(bot, capture_fills(rows, bot.c_mults), NOW-86_400_000, NOW) is None
    bot._hsl_revised_fill_observation = observe_fills(bot, (NOW-30, NOW-10))
    assert observed_fill_interval(bot, capture_fills(rows, bot.c_mults), NOW-86_400_000, NOW) == (NOW-30, NOW-10)


@pytest.mark.asyncio
async def test_hourly_preparation_does_not_start_another_protective_writer():
    import asyncio
    from types import SimpleNamespace
    instance = hsl_revised_live.Owner(SimpleNamespace())
    instance._running = True
    entered, release = asyncio.Event(), asyncio.Event()
    async def metadata():
        entered.set()
        await release.wait()
        return 'updated'
    task = asyncio.create_task(instance.during_preparation(metadata()))
    await asyncio.wait_for(entered.wait(), 1.)
    assert not task.done()
    release.set()
    # A second protection loop would access nonexistent account methods above.
    assert await task == 'updated'


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [ValueError, hsl_revised_live.runtime.InvalidHslOutput])
async def test_completed_planner_failure_surfaces_while_account_refresh_is_blocked(error_type):
    import asyncio
    from types import SimpleNamespace
    count = 0
    async def refresh(**kwargs):
        nonlocal count
        count += 1
        assert count < 3, 'fatal planner error was hidden behind unavailable account inputs'
        return count == 1
    async def fail():
        raise error_type('invalid producer')
    async def noop(*args, **kwargs):
        await asyncio.sleep(0)
    bot = SimpleNamespace(stop_signal_received=False,
        _begin_live_event_cycle=lambda **kwargs: None,
        refresh_protective_authoritative_state=refresh,
        _sleep_unless_shutdown=noop, _maybe_log_health_summary=lambda: None,
        live_value=lambda key: .05)
    instance = hsl_revised_live.Owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    instance.protect = noop
    instance._ordinary_plan = fail
    with pytest.raises(error_type, match='invalid producer'):
        await instance.run()
    assert count == 1
    assert not instance._running


@pytest.mark.asyncio
@pytest.mark.parametrize('batch', ['base', 'ccxt'])
@pytest.mark.parametrize('action', ['create', 'cancel'])
@pytest.mark.parametrize('outcome', ['ok', 'ambiguous'])
async def test_revised_batch_waits_for_write_then_requires_new_account(monkeypatch, batch, action, outcome):
    import asyncio
    from types import SimpleNamespace
    from passivbot import Passivbot
    from exchanges.ccxt_bot import CCXTBot
    from live import executor
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot('unified')
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    ledger = bot._ensure_freshness_ledger()
    ledger.stamp('open_orders', now_ms=NOW-200)
    bot._request_authoritative_confirmation = Passivbot._request_authoritative_confirmation.__get__(bot)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    orders = [dict(id=str(i), symbol=SYMBOL, position_side='long', side='sell', qty=1., price=90., type='market')
              for i in range(2)]
    instance.bind(wave, orders if action == 'cancel' else (), orders if action == 'create' else ())
    entered, release = asyncio.Event(), asyncio.Event()
    calls, records = [], []
    async def connector(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        if outcome == 'ambiguous':
            raise OSError('offline transport failure')
        return {'id': 'ack'}
    bot.cca = SimpleNamespace(create_order=connector, cancel_order=connector)
    bot._build_order_params = lambda order: {}
    bot._emit_execution_connector_call_started_event = lambda **kwargs: None
    monkeypatch.setattr(executor, 'record_create_connector_admission', lambda bot, order: records.append(order['id']))
    monkeypatch.setattr(executor, 'record_cancel_connector_admission', lambda bot, order: records.append(order['id']))
    bot.execute_order = Passivbot.execute_order.__get__(bot)
    bot.execute_cancellation = Passivbot.execute_cancellation.__get__(bot)
    bot.execute_multiple = Passivbot.execute_multiple.__get__(bot)
    async def handle(failures):
        assert len(failures) == int(outcome == 'ambiguous')
    bot._handle_order_write_failures = handle
    cls = Passivbot if batch == 'base' else CCXTBot
    method = cls.execute_orders if action == 'create' else cls.execute_cancellations
    pending = asyncio.create_task(method(bot, orders))
    await asyncio.wait_for(entered.wait(), 1.)
    # Both batch tasks have been scheduled, but only one has entered the connector.
    await asyncio.sleep(0)
    assert len(calls) == 1
    release.set()
    results = await pending
    deferred = executor.DeferredOrderCreation if action == 'create' else executor.DeferredOrderCancellation
    assert isinstance(results[1], deferred)
    assert records == ['0']
    assert set(bot._authoritative_pending_confirmations) == {'balance', 'positions', 'open_orders'}
    # The next confirmed cohort may admit a new plan; no permanent write latch.
    ledger.begin_epoch(now_ms=NOW)
    for surface in ('balance', 'positions', 'open_orders'):
        ledger.stamp(surface, now_ms=NOW)
    assert instance.admit(orders[1])


@pytest.mark.asyncio
@pytest.mark.parametrize('adapter', ['base', 'okx', 'hyperliquid'])
async def test_deferred_cancellation_has_no_submission_provenance(monkeypatch, adapter):
    from types import SimpleNamespace
    from passivbot import Passivbot
    from exchanges.okx import OKXBot
    from exchanges.hyperliquid import HyperliquidBot
    from live import executor
    from live.event_bus import EventTypes
    from test_hsl_revised_runtime import bot as make_bot, NOW, SYMBOL
    events, calls, recorded = [], [], []
    bot = make_bot()
    # No receipt: a stale or unplanned cancellation must be deferred, including
    # when passed through the full parent bookkeeping and adapter override.
    bot.live_value = lambda key: 10
    bot.add_to_recent_order_cancellations = lambda order: recorded.append(order)
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args: None
    bot.state_change_detected_by_symbol = set()
    bot.get_exchange_time = lambda: NOW
    async def cancel(*args, **kwargs):
        calls.append(args)
        return {'id': 'old'}
    bot.cca = SimpleNamespace(cancel_order=cancel)
    cls = {'base': Passivbot, 'okx': OKXBot, 'hyperliquid': HyperliquidBot}[adapter]
    bot.execute_cancellation = cls.execute_cancellation.__get__(bot)
    bot.execute_multiple = Passivbot.execute_multiple.__get__(bot)
    bot.execute_cancellations = Passivbot.execute_cancellations.__get__(bot)
    async def handle(failures):
        assert not failures
    bot._handle_order_write_failures = handle
    monkeypatch.setattr(Passivbot, '_emit_execution_order_event',
                        lambda *args, **kwargs: events.append(kwargs['event_type']))
    order = dict(id='old', symbol=SYMBOL, position_side='long', side='sell', qty=1., price=100.)
    assert await executor.execute_cancellations_parent(bot, [order]) == []
    assert not calls and not recorded
    assert EventTypes.EXECUTION_CANCEL_SENT not in events
    assert not bot.state_change_detected_by_symbol


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['universe', 'market', 'reconcile', 'after_rust'])
@pytest.mark.parametrize('change', ['position', 'balance', 'strategy_balance', 'order', 'fill_pnl', 'fill_fee', 'unchanged_confirmation'])
async def test_pending_planner_requires_unchanged_account_facts(stage, change):
    import asyncio
    from test_hsl_revised_runtime import bot as make_bot, NOW, SYMBOL
    bot = make_bot()
    bot._staged_planner_required_surfaces = lambda **kwargs: {'fills'}
    instance = hsl_revised_live.owner(bot)
    entered, release = asyncio.Event(), asyncio.Event()
    reached = []
    async def step(name):
        reached.append(name)
        if stage == name:
            entered.set()
            await release.wait()
        return True
    bot.prepare_planning_universe = lambda: step('universe')
    bot.refresh_market_state_if_needed = lambda: step('market')
    bot._staged_execution_ready_state = lambda **kwargs: (True, {})
    bot._current_planning_snapshot = object()
    async def reconcile():
        await step('reconcile')
        from types import SimpleNamespace
        bot._hsl_revised_planning_wave = hsl_revised_live.Wave(NOW, NOW, (), (), (), "", (), bot.get_raw_balance(), 0)
        await step('after_rust')
        return [], []
    bot.calc_orders_to_cancel_and_create = reconcile
    task = asyncio.create_task(instance._ordinary_plan())
    await asyncio.wait_for(entered.wait(), 1.)
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch(now_ms=NOW+1)
    ledger.stamp('positions', now_ms=NOW+1)
    if change == 'position':
        bot.positions['NEW/USDT:USDT'] = {'long': dict(size=1., price=100.)}
    elif change == 'balance':
        bot.get_raw_balance = lambda: 900.
    elif change == 'strategy_balance':
        bot.get_hysteresis_snapped_balance = lambda: 900.
    elif change == 'order':
        bot.open_orders[SYMBOL] = [dict(id='expected-own-order', qty=1., price=100.)]
    elif change.startswith('fill_'):
        ledger.stamp('fills', signature=((change, -10.),), now_ms=NOW+1)
    else:
        # Harmless confirming reads and planner flat padding must not starve a
        # slow, otherwise coherent plan; only changed account facts invalidate it.
        bot.positions['FLAT/USDT:USDT'] = {'long': dict(size=0., price=0.)}
        bot.open_orders['FLAT/USDT:USDT'] = []
    release.set()
    result = await task
    allowed = change == 'unchanged_confirmation' or (change == 'balance' and stage != 'after_rust')
    assert (result is not None) == allowed
    if not allowed and stage in {'universe', 'market'}:
        assert 'reconcile' not in reached


@pytest.mark.asyncio
async def test_next_planner_cannot_run_while_previous_plan_is_writing():
    import asyncio
    from types import SimpleNamespace
    plans = []
    async def noop(*args, **kwargs):
        await asyncio.sleep(0)
        return True
    bot = SimpleNamespace(stop_signal_received=False,
        _begin_live_event_cycle=lambda **kwargs: None,
        refresh_protective_authoritative_state=noop,
        _sleep_unless_shutdown=noop, _maybe_log_health_summary=lambda: None,
        live_value=lambda key: .05)
    instance = hsl_revised_live.Owner(bot)
    instance._account_matches = lambda *args: True
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    instance.protect = noop
    async def plan():
        plans.append(True)
        return [], [dict(symbol='TEST/USDT:USDT')], object(), object()
    async def write(*args):
        assert len(plans) == 1
        await asyncio.sleep(0)
        # The old scheduling order started a second planner during this await.
        assert len(plans) == 1
        bot.stop_signal_received = True
    instance._ordinary_plan = plan
    bot.execute_order_plan_to_exchange = write
    await asyncio.wait_for(instance.run(), 1.)


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_stage', ['fetch', 'commit'])
async def test_revised_account_refresh_serializes_startup_and_protection(blocked_stage):
    import asyncio
    from types import SimpleNamespace
    from live import state_refresh
    entered, release = asyncio.Event(), asyncio.Event()
    calls, commits = [], []
    async def fetch(plan):
        calls.append(len(calls) + 1)
        number = calls[-1]
        if number == 1 and blocked_stage == 'fetch':
            entered.set()
            await release.wait()
        return dict(balance=1000., positions=[number], open_orders=[number])
    async def apply_orders(rows, **kwargs):
        if rows == [1] and blocked_stage == 'commit':
            entered.set()
            await release.wait()
        return True
    bot = SimpleNamespace(config={'live': {'hsl_engine': 'revised'}}, stop_signal_received=False,
        positions={}, _begin_authoritative_refresh_epoch=lambda: None,
        _fetch_authoritative_state_staged_snapshot=fetch,
        _prepare_balance_snapshot=lambda balance: {'balance': balance},
        _apply_open_orders_snapshot=apply_orders,
        _apply_positions_snapshot=lambda rows: (commits.append(rows[0]), rows),
        _commit_balance_snapshot=lambda snapshot: None,
        _record_authoritative_surface=lambda *args: None,
        get_hysteresis_snapped_balance=lambda: 1000., _positions_signature=tuple,
        _update_entry_cooldown_position_delta_guard=lambda *args, **kwargs: None,
        get_exchange_time=lambda: 1000, _finalize_authoritative_refresh_consistency=lambda plan: None)
    first = asyncio.create_task(state_refresh.refresh_authoritative_state(bot))
    await asyncio.wait_for(entered.wait(), 1.)
    second = asyncio.create_task(state_refresh.refresh_protective_authoritative_state(bot))
    await asyncio.sleep(0)
    assert calls == [1]
    # A writer must not admit against a partly committed or still-pending cohort.
    assert not hsl_revised_live.owner(bot)._account_matches(None, 1000)
    release.set()
    assert await first and await second
    assert calls == commits == [1, 2]
    assert not hsl_revised_live.owner(bot)._refresh_lock.locked()


@pytest.mark.parametrize('ordinary,required', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('change', ['pnl', 'fee', 'confirmation', 'pending'])
def test_fill_enrichment_rechecks_only_required_ordinary_receipts(monkeypatch, ordinary, required, change):
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._staged_planner_required_surfaces = lambda **kwargs: {'fills'} if required else set()
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    ledger.stamp('open_orders', now_ms=NOW-200)
    original = (('execution', -1., -.1),)
    ledger.stamp('fills', signature=original, now_ms=NOW-200)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    order = dict(symbol=SYMBOL, position_side='long')
    instance.bind(wave, (), (order,), ordinary=ordinary)
    assert instance.admit(order)
    if change == 'pending':
        bot._authoritative_pending_confirmations = {'fills': ledger.epoch+1}
    else:
        signature = original if change == 'confirmation' else (('execution', -2. if change == 'pnl' else -1., -.2),)
        ledger.begin_epoch()
        ledger.stamp('fills', signature=signature, now_ms=NOW)
    assert instance.admit(order) == (change == 'confirmation' or not (ordinary and required))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['network', 'cache', 'candle', 'programmer'])
async def test_revised_startup_warmup_failure_policy(failure, caplog):
    from types import SimpleNamespace
    from ccxt.base.errors import NetworkError
    from candlestick_manager import OhlcvFetchError
    errors = dict(network=NetworkError, cache=OSError, candle=OhlcvFetchError, programmer=ValueError)
    async def warmup():
        raise errors[failure]('test failure')
    instance = hsl_revised_live.Owner(SimpleNamespace(warmup_trading_ready_candles=warmup))
    if failure == 'programmer':
        with pytest.raises(ValueError):
            await instance.warmup()
    else:
        import logging
        with caplog.at_level(logging.INFO):
            await instance.warmup()
        assert 'trading-ready candle warmup skipped' in caplog.text


@pytest.mark.asyncio
async def test_stalled_startup_retries_completed_sources_with_bounded_cadence():
    import asyncio
    from types import SimpleNamespace
    release = asyncio.Event()
    async def refresh(**kwargs):
        return True
    bot = SimpleNamespace(stop_signal_received=False, refresh_protective_authoritative_state=refresh, cm=object())
    instance = hsl_revised_live.Owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = lambda: None
    reads = []
    async def read_sources():
        reads.append(True)
        # A transient source failure is a completed factual result, not an
        # exception: the real reader records it and the next acquisition repairs.
        instance.sources = {'test': 'unavailable' if len(reads) == 1 else 'recovered'}
    instance._read_sources = read_sources
    waves = []
    async def protect():
        waves.append(True)
        if len(waves) == 2:
            assert reads == [True]
            assert instance.sources == {'test': 'unavailable'}
            # The second startup pass respected the retry deadline. Advance only
            # that deadline, preserving the real event loop clock and task ownership.
            instance._next_sources = 0.
        elif len(waves) == 4:
            assert len(reads) == 2
            assert instance.sources == {'test': 'recovered'}
            release.set()
    instance.protect = protect
    await asyncio.wait_for(instance.during_preparation(release.wait()), 6.)
    instance.cancel_inputs()


@pytest.mark.asyncio
async def test_source_retry_does_not_replace_resistant_task_or_hide_late_failure():
    import asyncio
    from types import SimpleNamespace
    instance = hsl_revised_live.Owner(SimpleNamespace(cm=object()))
    entered, release = asyncio.Event(), asyncio.Event()
    reads = []
    async def read_sources():
        reads.append(True)
        entered.set()
        await release.wait()
        raise ValueError('malformed producer')
    instance._read_sources = read_sources
    instance.schedule_sources()
    original = instance._source_task
    await entered.wait()
    instance._next_sources = 0.
    instance.schedule_sources()
    assert instance._source_task is original and reads == [True]
    release.set()
    await asyncio.wait((original,))
    # Even before the next retry deadline, a completed programming failure must
    # be retrieved and propagated; neither retry nor the clock may hide it.
    instance._next_sources = float('inf')
    with pytest.raises(ValueError, match='malformed producer'):
        instance.schedule_sources()
    assert instance._source_task is original


@pytest.mark.asyncio
async def test_failed_account_refresh_releases_revised_transaction(monkeypatch):
    from types import SimpleNamespace
    from live import state_refresh
    from ccxt.base.errors import NetworkError
    bot = SimpleNamespace(config={'live': {'hsl_engine': 'revised'}})
    calls = []
    async def refresh(bot, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise NetworkError('offline simulated account outage')
        return True
    monkeypatch.setattr(state_refresh, '_refresh_protective_authoritative_state', refresh)
    with pytest.raises(NetworkError):
        await state_refresh.refresh_protective_authoritative_state(bot)
    assert not hsl_revised_live.owner(bot)._refresh_lock.locked()
    assert await state_refresh.refresh_protective_authoritative_state(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize('required', [True, False])
@pytest.mark.parametrize('failure', ['incomplete', 'network', 'cache', 'fill_data'])
async def test_failed_fill_refresh_revokes_only_required_consumers_until_repaired(monkeypatch, required, failure):
    from types import MethodType
    from ccxt.base.errors import NetworkError
    from passivbot import Passivbot
    from passivbot_exceptions import FillEventDataError
    from live.planning_gates import staged_planner_precondition_state
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    surfaces = {'balance', 'positions', 'open_orders'} | ({'fills'} if required else set())
    bot._staged_planner_required_surfaces = lambda **kwargs: surfaces
    bot._staged_planner_surface_min_epochs = MethodType(Passivbot._staged_planner_surface_min_epochs, bot)
    bot._request_authoritative_confirmation = MethodType(Passivbot._request_authoritative_confirmation, bot)
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    for surface in ('balance', 'positions', 'open_orders', 'fills'):
        ledger.stamp(surface, signature=('old tape',) if surface == 'fills' else None, now_ms=NOW-200)
    calls = []
    async def refresh(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            if failure == 'incomplete':
                return False
            raise {'network': NetworkError, 'cache': OSError, 'fill_data': FillEventDataError}[failure]('offline failure')
        ledger.stamp('fills', signature=('repaired tape',), now_ms=NOW)
        return True
    bot.update_pnls = refresh
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    ordinary = dict(symbol=SYMBOL, position_side='long')
    protective = dict(ordinary)
    instance.bind(wave, (), (ordinary,), ordinary=True)
    instance.bind(wave, (), (protective,))
    assert staged_planner_precondition_state(bot, include_market_snapshot=False)[0]
    assert await instance._refresh_history({}) is False
    ready, details = staged_planner_precondition_state(bot, include_market_snapshot=False)
    assert ready == (not required)
    assert details['missing'] == (['fills'] if required else [])
    assert instance.admit(ordinary) == (not required)
    assert instance.admit(protective)
    # The next owner account pass advances the shared epoch; only an actual
    # successful fill refresh can stamp the required fill confirmation there.
    ledger.begin_epoch()
    assert await instance._refresh_history({}) is True
    assert staged_planner_precondition_state(bot, include_market_snapshot=False)[0]
    fresh = dict(symbol=SYMBOL, position_side='long')
    instance.bind(instance.capture(), (), (fresh,), ordinary=True)
    assert instance.admit(fresh)
    assert instance.admit(protective)


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_write_admission_evaluates_only_its_authorizing_scope(monkeypatch, mode):
    import utils
    import passivbot_rust as pbr
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    from dataclasses import replace
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot(mode)
    other = 'PEER/USDT:USDT'
    bot.positions[other] = {'long': dict(size=2., price=100.)}
    bot.c_mults[other], bot.qty_steps[other] = 1., .1
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL, other}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW-200)
    owner = hsl_revised_live.owner(bot)
    wave = owner.capture({**quotes(), other: replace(quotes()[SYMBOL], symbol=other)})
    order = {'symbol': SYMBOL, 'position_side': 'long'}
    owner.bind(wave, (), (order,))
    diagnostic_before = object()
    bot._hsl_revised_diagnostic_observation = diagnostic_before
    native = pbr.hsl_revised_evaluate_grids
    calls = []
    def evaluate(metadata, grids):
        calls.append(json.loads(metadata)['snapshot'])
        return native(metadata, grids)
    monkeypatch.setattr(pbr, 'hsl_revised_evaluate_grids', evaluate)
    assert owner.admit(order)
    assert len(calls) == 1
    assert bot._hsl_revised_diagnostic_observation is diagnostic_before
    assert {pair['symbol'] for pair in calls[0]['pairs']} == ({SYMBOL} if mode == 'coin' else {SYMBOL, other})
    # Scoping risk does not waive complete account-cohort confirmation.
    bot.positions[other]['long']['size'] += 1
    assert not owner.admit(order)
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('change', ['raw_only', 'strategy', 'risk_action', 'stale_balance', 'pending_balance'])
def test_ordinary_admission_rejects_changed_risk_inputs(monkeypatch, mode, side, change):
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot(mode, side=side)
    block = bot.config['bot']['hsl'] if mode == 'unified' else bot.config['bot'][side]['hsl']
    block['red_threshold'] = .2
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {s: {SYMBOL} if s == side else set() for s in ('long', 'short')}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._staged_planner_required_surfaces = lambda **kwargs: set()
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    ledger.stamp('open_orders', now_ms=NOW-200)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes(side))
    assert wave.permission(SYMBOL, side)[0] == 'normal'
    order = dict(symbol=SYMBOL, position_side=side, reduce_only=True)
    instance.bind(wave, (), (order,), ordinary=True)
    assert instance.admit(order)
    bot.get_raw_balance = lambda: 999.
    if change == 'strategy':
        bot.get_hysteresis_snapped_balance = lambda: 990.
    elif change == 'risk_action':
        # Even with unchanged strategy sizing, fresh raw balance changes the
        # Rust HSL action. An old ordinary close must not replace panic intent.
        bot.get_raw_balance = lambda: 100.
        assert instance.capture().permission(SYMBOL, side)[0] == 'panic'
    elif change == 'stale_balance':
        ledger.surfaces['balance'].updated_ms = NOW-10_001
    elif change == 'pending_balance':
        bot._authoritative_pending_confirmations = {'balance': ledger.epoch+1}
    assert not instance.admit(order)


@pytest.mark.asyncio
@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('change', ['account_confirmation', 'required_fills'])
@pytest.mark.parametrize('entrypoint', ['cycle', 'fake_runner'])
async def test_completed_plan_invalidated_by_fill_confirmation_is_replanned(monkeypatch, empty, change, entrypoint):
    import asyncio
    import utils
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._staged_planner_required_surfaces = lambda **kwargs: set()
    bot._begin_live_event_cycle = lambda **kwargs: None
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    ledger.stamp('open_orders', now_ms=NOW)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    order = dict(symbol=SYMBOL, position_side='long', reduce_only=True)
    from dataclasses import replace
    bot._staged_planner_required_surfaces = lambda **kwargs: {'fills'}
    wave = replace(wave, required_fills=instance.required_fill_facts())
    instance.bind(wave, (), (order,), ordinary=True)
    assert instance._account_matches(wave, NOW)
    async def prepared():
        return [], [] if empty else [order], object(), replace(wave, required_fills=instance.required_fill_facts())
    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    calls = []
    invalidated = False
    async def protect(**kwargs):
        nonlocal invalidated
        if invalidated:
            return False
        invalidated = True
        # A delayed fill observation requires another account confirmation.
        if change == 'account_confirmation':
            bot._authoritative_pending_confirmations = {'balance': ledger.epoch+1}
        else:
            ledger.stamp('fills', signature=('changed-required-fills',), now_ms=NOW)
        return False
    async def refresh(**kwargs):
        calls.append('refresh')
        if entrypoint == 'cycle':
            return False
        ledger.begin_epoch(now_ms=NOW)
        for surface in ('balance', 'positions', 'open_orders'):
            ledger.stamp(surface, now_ms=NOW)
        return True
    async def execute(*args):
        calls.append('write')
    instance.protect = protect
    instance._ordinary_plan = prepared
    instance.schedule_history = instance.schedule_sources = lambda: None
    bot.refresh_protective_authoritative_state = refresh
    bot.execute_order_plan_to_exchange = execute
    try:
        if entrypoint == 'cycle':
            result = await instance.cycle()
            assert result['ordinary_completed'] and not result['ordinary_executed']
            assert instance._ordinary is None
            assert calls == ['refresh']
        else:
            result = await runner._run_fake_cycle_ready(bot)
            assert result['passes'] == 2 and result['ordinary_executed']
            assert calls == ['refresh', 'write', 'refresh']
    finally:
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['before', 'quotes', 'reconcile'])
@pytest.mark.parametrize('flag', ['stop_signal_received', '_shutdown_in_progress'])
async def test_shutdown_during_ready_protection_prevents_further_service(monkeypatch, stage, flag):
    import asyncio
    import utils
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._begin_live_event_cycle = lambda **kwargs: None
    bot._record_market_snapshot_surface = lambda *args: None
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    assert wave.permission(SYMBOL, 'long')[0] == 'panic'
    calls = []
    async def acquire(symbols):
        calls.append('quotes')
        await asyncio.sleep(0)
        if stage == 'quotes': setattr(bot, flag, True)
        return quotes()
    async def reconcile(**kwargs):
        calls.append('reconcile')
        await asyncio.sleep(0)
        if stage == 'reconcile': setattr(bot, flag, True)
        return [], [dict(symbol=SYMBOL, position_side='long')]
    async def execute(*args, **kwargs):
        calls.append('write')
    async def refresh(**kwargs):
        calls.append('refresh')
        return True
    async def prepared():
        return [], [], object(), wave
    instance.acquire_quotes = acquire
    bot.calc_protective_panic_orders_to_cancel_and_create = reconcile
    bot.execute_order_plan_to_exchange = execute
    bot.refresh_protective_authoritative_state = refresh
    instance.report = lambda wave: None
    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    if stage == 'before': setattr(bot, flag, True)
    result = await instance.cycle()
    assert not result['ordinary_executed'] and not result['updated']
    assert 'write' not in calls and 'refresh' not in calls
    assert calls == {'before': [], 'quotes': ['quotes'], 'reconcile': ['quotes', 'reconcile']}[stage]
    assert instance._ordinary is None


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['create', 'cancel'])
@pytest.mark.parametrize('flag', ['stop_signal_received', '_shutdown_in_progress'])
async def test_shutdown_rejects_connector_work_queued_on_write_lock(monkeypatch, action, flag):
    import asyncio
    import utils
    from live import executor
    from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp('open_orders', now_ms=NOW)
    instance = hsl_revised_live.owner(bot)
    wave = instance.capture(quotes())
    order = dict(symbol=SYMBOL, position_side='long')
    instance.bind(wave, (), (order,))
    assert instance.admit(order)
    calls = []
    @hsl_revised_live.connector_write(action)
    async def connector(bot, order):
        calls.append(order)
    await instance._write_lock.acquire()
    task = asyncio.create_task(connector(bot, order))
    await asyncio.sleep(0)
    assert not task.done()
    setattr(bot, flag, True)
    instance._write_lock.release()
    result = await task
    expected = executor.DeferredOrderCreation if action == 'create' else executor.DeferredOrderCancellation
    assert isinstance(result, expected) and not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', ['stop_signal_received', '_shutdown_in_progress'])
async def test_revised_run_exits_on_each_canonical_shutdown_flag(flag):
    from types import SimpleNamespace
    bot = SimpleNamespace(stop_signal_received=False)
    instance = hsl_revised_live.Owner(bot)
    calls = []
    async def cycle():
        calls.append('cycle')
        setattr(bot, flag, True)
        return {'updated': True}
    async def sleep(*args, **kwargs):
        assert calls == ['cycle']
    instance.cycle = cycle
    bot._maybe_log_health_summary = lambda: None
    bot._sleep_unless_shutdown = sleep
    bot.live_value = lambda key: .05
    await instance.run()
    assert calls == ['cycle'] and not instance._running
