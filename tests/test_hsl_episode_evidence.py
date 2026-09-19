"""Startup/live transitions must consume the same proven fill episode."""
import pytest

import passivbot_hsl as hsl
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager


@pytest.mark.asyncio
async def test_startup_proof_survives_clipped_history_with_leading_close():
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=1.0)
    bot.hsl['short'].update(enabled=True, restart_after_red_policy='always', cooldown_minutes_after_red=60.0)
    now = 10 * 86_400_000
    bot.get_exchange_time = lambda: now
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    events = [
        dict(timestamp=now-7_200_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
        dict(timestamp=now-3_000_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-2.0),
        dict(timestamp=now-90_123, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0, fee_paid=-0.01),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    async def history(**kwargs):
        assert kwargs['hsl_replay_start_ms'] == now-3_600_000
        return {'timeline': [{
            'timestamp': now-60_000, 'balance': 100.0, 'realized_pnl': -2.01,
            'realized_pnl_by_coin_pside': {'A': {'long': -2.01, 'short': 0.0}},
            'unrealized_pnl_by_coin_pside': {'A': {'long': 0.0, 'short': 0.0}},
        }], 'fill_events': events[1:], 'panic_flatten_events': []}
    bot.get_balance_equity_history = history
    await bot._equity_hard_stop_initialize_coin_from_history()
    await bot._equity_hard_stop_check_coin()
    metrics = bot._hsl_coin_state('long', 'A')['last_metrics']
    assert metrics['realized_pnl'] == pytest.approx(-0.01)
    assert bot._hsl_coin_state('long', 'A')['pnl_reset_timestamp_ms'] == events[-1]['timestamp']


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['position', 'late_fill', 'pnl', 'fee', 'policy'])
async def test_history_io_invalidates_changed_observation_before_replacing_protection(change):
    bot = make_coin_bot()
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    events = [dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0)]
    bot._pnls_manager = make_fake_pnls_manager(events)
    old_state = bot._hsl_coin_state('long', 'A')
    old_state['halted'] = True
    old_state['no_restart_latched'] = True
    async def history(**kwargs):
        if change == 'position':
            bot.positions['A']['long']['size'] = 2.0
        elif change == 'late_fill':
            events.append(dict(events[0], timestamp=120_000))
        elif change == 'pnl':
            events[0]['pnl'] = -10.0
        elif change == 'fee':
            events[0]['fee_paid'] = -0.5
        else:
            bot.hsl['long']['restart_after_red_policy'] = 'never'
        return {'timeline': [], 'fill_events': events, 'panic_flatten_events': []}
    bot.get_balance_equity_history = history
    with pytest.raises(hsl.EpisodeEvidenceUnavailable, match='observation_changed_during_replay'):
        await bot._equity_hard_stop_initialize_coin_from_history()
    assert bot._hsl_coin_state('long', 'A') is old_state
    assert old_state['halted'] and old_state['no_restart_latched']


def test_episode_evidence_preserves_exact_boundaries_prefixes_and_cooldown_chain():
    from live.hsl_episode import EpisodeEvidence
    rows = [(10, 'increase', 1.0, -0.1), (20, 'decrease', 1.0, -2.0),
            (20, 'increase', 2.0, -0.2), (30, 'decrease', 2.0, 1.0),
            (100, 'increase', 1.0, -0.1)]
    evidence = EpisodeEvidence.reconstruct(rows)
    assert evidence.flatten_indices == (1, 3)
    assert evidence.sizes == (0.0, 1.0, 0.0, 2.0, 0.0, 1.0)
    assert evidence.realized_prefix[-1] == pytest.approx(-1.4)
    assert evidence.required_start(1.0, 70) == 100
    assert evidence.required_start(1.0, 71) == 10
    assert evidence.required_start(2.0, 71) is None
    truncated = EpisodeEvidence.reconstruct(rows[1:])
    assert truncated.unavailable == 'missing_opening_fill'
    assert truncated.required_start(1.0, 0) is None


def test_episode_evidence_is_equal_for_reordered_normalized_fill_tape():
    events = [dict(timestamp=t, symbol='A', pside='long', action=action, qty=1.0, pnl=0.0)
              for t, action in [(100, 'increase'), (200, 'decrease'), (300, 'increase')]]
    assert hsl._equity_hard_stop_coin_episode_evidence(events, 'long', 'A') == hsl._equity_hard_stop_coin_episode_evidence(list(reversed(events)), 'long', 'A')


def test_projected_episode_retains_opening_quantity_and_pnl_baseline():
    from live.hsl_episode import EpisodeEvidence
    full = EpisodeEvidence.reconstruct([(10, 'increase', 2.0, -0.1),
        (20, 'decrease', 2.0, -2.0), (100, 'increase', 1.0, -0.2)])
    window = full.window(15, 100)
    assert window.sizes == (2.0, 0.0, 1.0)
    assert window.flatten_indices == (0,)
    assert window.realized_prefix == pytest.approx((0.0, -2.0, -2.2))
    assert window.unavailable is None


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['pnl', 'fee', 'late_round_trip'])
async def test_revised_sampled_evidence_requests_replay_and_preserves_runtime(monkeypatch, change):
    from unittest.mock import AsyncMock
    bot = make_coin_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    events = [dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0)]
    bot._pnls_manager = make_fake_pnls_manager(events)
    await bot._equity_hard_stop_check_coin()
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state('long', 'A')
    original = state['last_metrics']
    # Initial check had no state yet; second check recorded the proven tape.
    if change == 'late_round_trip':
        events.extend([
            dict(events[0], timestamp=90_000, action='decrease'),
            dict(events[0], timestamp=120_000),
        ])
    else:
        events[0]['pnl' if change == 'pnl' else 'fee_paid'] = -1.0
    replay = AsyncMock(return_value=False)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    with pytest.raises(hsl.EpisodeEvidenceUnavailable, match='revised_episode_replay_unavailable'):
        await bot._equity_hard_stop_check_coin()
    assert state['last_metrics'] is original
    replay.assert_awaited_once_with(bot, 'long', 'A')
    replay.return_value = True
    with pytest.raises(hsl.EpisodeEvidenceUnavailable, match='episode_observation_not_stable'):
        await bot._equity_hard_stop_check_coin()
    assert replay.await_count == 3


def test_recent_proven_flat_episode_retains_its_own_cooldown_history():
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=5.0)
    bot.positions = {'A': {'long': {'size': 0.0}, 'short': {'size': 0.0}}}
    now = 10_000_000
    events = [dict(timestamp=t, symbol='A', pside='long', action=a, qty=1.0, pnl=0.0)
              for t, a in [(1_000, 'increase'), (2_000, 'decrease'),
                           (now-400_000, 'increase'), (now-100_000, 'decrease')]]
    bot._pnls_manager = make_fake_pnls_manager(events)
    required, start, pairs = hsl._equity_hard_stop_required_fill_history_scope(bot, now, pnl_start_ms=0)
    assert required and start == now-400_000
    assert pairs == {('long', 'A'): now-400_000}
    # A cooldown-connected earlier episode remains necessary, even while flat.
    events[1]['timestamp'] = now-500_000
    required, start, pairs = hsl._equity_hard_stop_required_fill_history_scope(bot, now, pnl_start_ms=0)
    assert start == 1_000
    # An unexplained flat observation never supplies its own flatten timestamp.
    del events[-1]
    events[-1]['timestamp'] = now-200_000
    assert hsl._equity_hard_stop_required_fill_history_scope(bot, now, pnl_start_ms=0)[2] is None


@pytest.mark.asyncio
async def test_held_to_flat_keeps_discarded_symbols_out_and_matches_restart():
    now = [10 * 86_400_000]
    events = [dict(timestamp=t, symbol=symbol, pside='long', action=a, qty=1.0, pnl=pnl)
              for t, symbol, a, pnl in [(now[0]-86_400_000, 'OLD', 'increase', 0.0),
                (now[0]-86_340_000, 'OLD', 'decrease', -20.0),
                (now[0]-90_123, 'A', 'increase', 0.0)]]
    def make():
        bot = make_coin_bot()
        bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=1.0)
        bot.hsl['short'].update(enabled=True, restart_after_red_policy='always', cooldown_minutes_after_red=60.0)
        bot.get_exchange_time = lambda: now[0]
        bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
        bot._pnls_manager = make_fake_pnls_manager(events)
        async def history(**kwargs):
            retained = [e for e in events if e['timestamp'] >= kwargs['hsl_replay_start_ms']]
            assert all(e['symbol'] == 'A' for e in retained)
            return {'timeline': [{'timestamp': now[0]//60_000*60_000, 'balance': 100.0,
                'realized_pnl': 0.0, 'realized_pnl_by_coin_pside': {'A': {'long': 0.0, 'short': 0.0}},
                'unrealized_pnl_by_coin_pside': {'A': {'long': 0.0, 'short': 0.0}}}],
                'fill_events': retained, 'panic_flatten_events': []}
        bot.get_balance_equity_history = history
        return bot
    bot = make()
    await bot._equity_hard_stop_initialize_coin_from_history()
    await bot._equity_hard_stop_check_coin()
    now[0] += 60_123
    events.append(dict(events[-1], timestamp=now[0]-1, action='decrease'))
    bot.positions['A']['long']['size'] = 0.0
    await bot._equity_hard_stop_check_coin()
    assert 'OLD' not in bot._equity_hard_stop_coin_symbols()
    cold = make()
    cold.positions['A']['long']['size'] = 0.0
    await cold._equity_hard_stop_initialize_coin_from_history()
    await cold._equity_hard_stop_check_coin()
    for side in ('long', 'short'):
        live = bot._hsl_coin_state(side, 'A')
        fresh = cold._hsl_coin_state(side, 'A')
        assert live['halted'] == fresh['halted']
        for key in ('realized_pnl', 'tier', 'drawdown_raw'):
            assert live['last_metrics'][key] == fresh['last_metrics'][key]


@pytest.mark.asyncio
async def test_replay_completion_refreshes_live_sample_time(monkeypatch):
    bot = make_coin_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    now = [180_000]
    bot.get_exchange_time = lambda: now[0]
    bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', now[0], 100.0, 0.0, 0.0, 0.0)
    calls = []
    async def boundary(bot, timestamp, balance):
        calls.append(timestamp)
        if len(calls) == 1:
            now[0] = 300_000
            bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', now[0], 100.0, 0.0, 0.0, 0.0)
            return True
        return False
    monkeypatch.setattr(hsl, '_equity_hard_stop_refresh_live_coin_episode_boundaries', boundary)
    await bot._equity_hard_stop_check_coin()
    assert calls == [180_000, 300_000]
    assert bot._hsl_coin_state('long', 'A')['last_metrics']['timestamp_ms'] == 300_000


@pytest.mark.asyncio
async def test_corrected_closed_episode_inside_cooldown_requires_replay(monkeypatch):
    from unittest.mock import AsyncMock
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=10.0)
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_coin_boundary_start_ms = 60_000
    bot.positions = {'A': {'long': {'size': 0.0}, 'short': {'size': 0.0}}}
    events = [dict(timestamp=t, symbol='A', pside='long', action=a, qty=1.0, pnl=0.0)
              for t, a in [(60_000, 'increase'), (120_000, 'decrease')]]
    bot._pnls_manager = make_fake_pnls_manager(events)
    state = bot._hsl_coin_state('long', 'A')
    state['pnl_reset_timestamp_ms'] = 120_001
    bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', 180_000, 100.0, 0.0, 0.0, 0.0)
    await bot._equity_hard_stop_check_coin()
    events[-1]['pnl'] = -30.0
    replay = AsyncMock(return_value=False)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    with pytest.raises(hsl.EpisodeEvidenceUnavailable, match='revised_episode_replay_unavailable'):
        await bot._equity_hard_stop_check_coin()
    replay.assert_awaited_once_with(bot, 'long', 'A')


@pytest.mark.asyncio
async def test_consumed_historical_boundary_does_not_replay_forever(monkeypatch):
    from unittest.mock import AsyncMock
    bot = make_coin_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot.get_exchange_time = lambda: 600_000
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    events = [
        dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=0.0),
        dict(timestamp=180_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', 540_000, 100.0, 0.0, 0.0, 0.0)
    state = bot._hsl_coin_state('long', 'A')
    state['episode_evidence'] = hsl._equity_hard_stop_coin_episode_evidence(events, 'long', 'A')
    # The input tape was fully consumed by replay, but no reset watermark was
    # materialized for an old boundary outside the reconstructed price rows.
    assert state['pnl_reset_timestamp_ms'] is None
    replay = AsyncMock(return_value=True)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    await bot._equity_hard_stop_check_coin()
    await bot._equity_hard_stop_check_coin()
    replay.assert_not_awaited()
    assert state['last_metrics']['timestamp_ms'] == 600_000
    # A correction still invalidates consumed evidence and requests replay.
    events[1]['pnl'] = -2.0
    replay.return_value = False
    with pytest.raises(hsl.EpisodeEvidenceUnavailable):
        await bot._equity_hard_stop_check_coin()
    replay.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_boundary_batch_does_not_mark_later_failed_replay_consumed(monkeypatch):
    from unittest.mock import AsyncMock
    from live.hsl_episode import EpisodeEvidence
    bot = make_coin_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    rows = [(60_000, 'increase', 1.0, 0.0), (120_000, 'decrease', 1.0, 0.0),
            (180_000, 'increase', 1.0, 0.0), (240_000, 'decrease', 1.0, 0.0),
            (240_000, 'increase', 1.0, 0.0)]
    evidence = EpisodeEvidence.reconstruct(rows)
    bot._pnls_manager = make_fake_pnls_manager([])
    bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', 60_000, 100.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(hsl, '_equity_hard_stop_live_coin_episode_evidence', lambda *a: evidence)
    monkeypatch.setattr(hsl, '_equity_hard_stop_coin_episode_evidence', lambda *a, **k: evidence)
    replay = AsyncMock(return_value=False)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    for attempt in (1, 2):
        with pytest.raises(hsl.AuthoritativeSurfaceUnavailable):
            await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, 300_000, 100.0)
        consumed = bot._hsl_coin_state('long', 'A')['episode_evidence']
        assert [consumed.rows[i][0] for i in consumed.flatten_indices] == [120_000]
        assert replay.await_count == attempt
        assert replay.await_args.kwargs['replay_flatten_timestamp_ms'] == 240_000


@pytest.mark.parametrize('reverse', [False, True])
def test_unordered_cohort_that_cannot_flatten_keeps_episode_with_conservative_pnl(reverse):
    events = [dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=5.0, pnl=0.0)]
    cohort = [
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-10.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=2.0),
    ]
    events.extend(reversed(cohort) if reverse else cohort)
    evidence = hsl._equity_hard_stop_coin_episode_evidence(events, 'long', 'A')
    assert evidence.unavailable is None
    assert evidence.degraded_reason == 'unordered_nonflattening_fill_cohort'
    assert evidence.flatten_indices == ()
    assert evidence.ending_size == 5.0
    assert evidence.realized_prefix == (0.0, 0.0, 2.0, -8.0)
    assert evidence.window(60_000, 180_000).degraded_reason == evidence.degraded_reason


@pytest.mark.parametrize('opening', [0.0, 1.0])
def test_potential_flatten_in_unordered_cohort_stays_unavailable(opening):
    events = [dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=opening, pnl=0.0)] if opening else []
    events.extend([
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-10.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
    ])
    assert hsl._equity_hard_stop_coin_episode_evidence(events, 'long', 'A').unavailable


@pytest.mark.asyncio
async def test_degraded_nonflattening_cohort_keeps_existing_ema_and_reports_quality():
    from live.hsl_protection import ProtectionHealth, Scope
    bot = make_coin_bot()
    bot._hsl_protection_health = ProtectionHealth()
    bot._equity_hard_stop_coin_initialized = True
    bot.positions = {'A': {'long': {'size': 5.0}, 'short': {'size': 0.0}}}
    bot.bot_value = lambda side, key: 1
    bot.get_exchange_time = lambda: 180_000
    events = [dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=5.0, pnl=0.0)]
    bot._pnls_manager = make_fake_pnls_manager(events)
    bot._equity_hard_stop_apply_coin_metrics_sample('long', 'A', 60_000, 1000.0, 0.0, 0.0, 0.0)
    state = bot._hsl_coin_state('long', 'A')
    runtime = state['runtime']
    state['episode_evidence'] = hsl._equity_hard_stop_coin_episode_evidence(events, 'long', 'A')
    events.extend([
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-10.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
    ])
    await bot._equity_hard_stop_check_coin()
    assert state['runtime'] is runtime
    assert state['last_metrics']['elapsed_minutes'] == 2
    health = bot._hsl_protection_health.scopes[Scope('coin', 'long', 'A')]
    assert health.status == 'degraded'
    assert health.reason == 'unordered_nonflattening_fill_cohort'
    assert health.unavailable_since_ms is None


@pytest.mark.parametrize('condition,expected', [('coherent', 30.0), ('new_episode', 0.0), ('new_episode_uninitialized', 0.0), ('pending', None), ('gap', None), ('position_mismatch', None)])
def test_optional_emergency_realized_loss_requires_coherent_current_episode(condition, expected):
    bot = make_coin_bot()
    events = [
        dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=5.0, pnl=0.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-30.0),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    bot.positions = {'A': {'long': {'size': 4.0}, 'short': {'size': 0.0}}}
    bot._fill_history_coverage_status = lambda **kwargs: {'ready': condition != 'gap'}
    if condition == 'pending':
        events[-1]['pnl_source'] = 'pending'
    elif condition == 'position_mismatch':
        bot.positions['A']['long']['size'] = 3.0
    elif condition in {'new_episode', 'new_episode_uninitialized'}:
        events.extend([
            dict(timestamp=150_000, symbol='A', pside='long', action='decrease', qty=4.0, pnl=-5.0),
            dict(timestamp=151_000, symbol='A', pside='long', action='increase', qty=2.0, pnl=0.0),
        ])
        bot.positions['A']['long']['size'] = 2.0
        if condition == 'new_episode':
            bot._hsl_coin_state('long', 'A')['pnl_reset_timestamp_ms'] = 150_001
    assert hsl._equity_hard_stop_emergency_realized_loss(bot, 'long', 'A', 180_000) == expected


def test_realized_loss_sample_ignores_ambiguous_future_fill_cohort():
    bot = make_coin_bot()
    events = [
        dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=-0.1),
        dict(timestamp=240_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-20.0),
        dict(timestamp=240_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    assert bot._equity_hard_stop_coin_realized_pnl_peak_last('long', 'A', 180_000) == (0.0, -0.1)


def test_optional_realized_loss_unavailable_reset_does_not_block_raw_fallback():
    bot = make_coin_bot()
    bot._pnls_manager = make_fake_pnls_manager([
        dict(timestamp=60_000, symbol='A', pside='long', action='increase', qty=1.0, pnl=0.0),
    ])
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    bot._fill_history_coverage_status = lambda **kwargs: {'ready': True}
    def unavailable(*args, **kwargs):
        raise hsl.EpisodeEvidenceUnavailable('ambiguous_fill_order_or_values', pside='long', symbol='A')
    bot._equity_hard_stop_coin_realized_pnl_peak_last = unavailable
    assert hsl._equity_hard_stop_emergency_realized_loss(bot, 'long', 'A', 180_000) is None
