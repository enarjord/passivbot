"""Uncertain history must not poison independently reconstructable HSL scopes."""
from itertools import permutations

import pytest

import passivbot_hsl as hsl
from live.hsl_episode import EpisodeEvidence
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager, _make_aggregate_episode_bot


def fill(ts, symbol, action, qty=1.0, pnl=0.0, pside='long'):
    return dict(timestamp=ts, symbol=symbol, pside=pside, action=action, qty=qty, pnl=pnl)


@pytest.mark.parametrize('mode', ['unified', 'pside'])
@pytest.mark.parametrize('other_side', ['long', 'short'])
def test_cross_pair_tie_retains_scope_until_proven_flat(mode, other_side):
    bot = _make_aggregate_episode_bot(mode)
    bot.positions = {}
    opening = fill(60_000, 'A', 'increase')
    cohort = [fill(120_000, 'A', 'decrease', pnl=-10.0),
              fill(120_000, 'B', 'increase', pside=other_side)]
    closing = fill(180_000, 'B', 'decrease', pnl=-5.0, pside=other_side)
    results = []
    for permutation in permutations(cohort):
        samples, quality = hsl._equity_hard_stop_scope_flatten_samples(
            bot, [opening, *permutation, closing], 'long', mode,
            240_000, 985.0, -15.0,
            {'long': -15.0 if other_side == 'long' else -10.0,
             'short': -5.0 if other_side == 'short' else 0.0},
            include_entry_seeds=True, include_quality=True)
        results.append(samples)
        flats = [s for s in samples if s.get('_hsl_scope_flatten_fill')]
        cross_scope = mode == 'pside' and other_side == 'short'
        assert [s['timestamp'] for s in flats] == [120_000 if cross_scope else 180_000]
        assert quality == (() if cross_scope else (120_000,))
        assert len([s for s in samples if s.get('_hsl_scope_entry_seed')]) == 1
        if not cross_scope:
            assert flats[0]['balance'] == 985.0
    assert results[0] == results[1]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['unified', 'pside'])
async def test_cross_pair_startup_and_live_keep_loss_without_fabricated_reset(mode):
    bot = _make_aggregate_episode_bot(mode, closing_loss=20.0)
    events = bot._pnls_manager.get_events()
    events[-1].update(timestamp=events[1]['timestamp'], symbol='B')
    bot.positions['A']['long']['size'] = 0.0
    bot.positions['B'] = {'long': {'size': 1.0}, 'short': {'size': 0.0}}
    bot.hsl['long']['red_threshold'] = 0.5
    await bot._equity_hard_stop_initialize_from_history()
    state = bot._hsl_state('long')
    assert state['last_metrics'] is not None
    assert state['pnl_reset_timestamp_ms'] is None
    assert state['last_metrics']['drawdown_raw'] > 0
    assert hsl._equity_hard_stop_scope_evidence_quality(state) == 'unordered_cross_pair_fill_cohort'
    for _ in range(3):
        assert not await hsl._equity_hard_stop_refresh_live_scope_episode_boundaries(bot, 300_000, 979.0)
    assert state['pnl_reset_timestamp_ms'] is None
    events.append(fill(300_001, 'B', 'decrease'))
    bot.positions['B']['long']['size'] = 0.0
    assert not await hsl._equity_hard_stop_refresh_live_scope_episode_boundaries(bot, 300_002, 979.0)
    assert state['pnl_reset_timestamp_ms'] == 300_002
    assert hsl._equity_hard_stop_scope_evidence_quality(state) == ''


@pytest.mark.parametrize('bad', ['overclose', 'mismatch', 'ambiguous'])
def test_cross_pair_ordering_does_not_hide_bad_pair_history(bad):
    bot = _make_aggregate_episode_bot('unified')
    bot.positions = {}
    events = [fill(1, 'A', 'increase'), fill(2, 'A', 'decrease'),
              fill(2, 'B', 'increase'), fill(3, 'B', 'decrease')]
    if bad == 'overclose':
        events[1]['qty'] = 2.0
    elif bad == 'mismatch':
        bot.positions = {'B': {'long': {'size': 1.0}}}
    else:
        events += [fill(2, 'B', 'decrease'), fill(2, 'B', 'increase')]
    assert hsl._equity_hard_stop_scope_flatten_samples(
        bot, events, 'long', 'unified', 4, 100., 0., {'long': 0., 'short': 0.}) is None


def test_incomplete_prefix_recovers_only_after_independent_flat_gap():
    evidence = EpisodeEvidence.reconstruct([
        (10, 'increase', 3.0, 0.0), (20, 'decrease', 5.0, -20.0),
        (100, 'increase', 2.0, -0.1), (110, 'decrease', 1.0, -2.0)])
    recovered = evidence.recover_closed_prefix(1.0, 30)
    assert recovered.unavailable is None
    assert recovered.required_start(1.0, 30) == 100
    assert recovered.sizes == (0., 2., 1.)
    assert recovered.realized_prefix == pytest.approx((0., -0.1, -2.1))
    assert recovered.degraded_reason == 'position_anchored_episode_suffix'
    assert evidence.recover_closed_prefix(1.0, 90) is evidence
    assert evidence.recover_closed_prefix(2.0, 0) is evidence
    assert evidence.recover_closed_prefix(0.5, 0) is evidence


def test_recovery_retains_cooldown_connected_suffix_episodes():
    evidence = EpisodeEvidence.reconstruct([
        (10, 'decrease', 5., -20.), (100, 'increase', 2., 0.),
        (110, 'decrease', 2., -4.), (120, 'increase', 1., 0.)])
    recovered = evidence.recover_closed_prefix(1., 30)
    assert recovered.required_start(1., 30) == 100
    assert recovered.rows[0][0] == 100
    assert recovered.realized_prefix[-1] == -4.
    assert evidence.recover_closed_prefix(1., 100) is evidence


@pytest.mark.parametrize('reason', ['ambiguous_fill_order_or_values', None])
def test_suffix_recovery_does_not_replace_other_evidence(reason):
    evidence = EpisodeEvidence.reconstruct([(1, 'increase', 1., 0.)], ambiguous=bool(reason))
    assert evidence.recover_closed_prefix(2., 0) is evidence


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['always', 'never', 'threshold'])
async def test_coin_prefix_recovery_coverage_replay_and_live(policy):
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy=policy, cooldown_minutes_after_red=1.0)
    now = 10 * 86_400_000
    bot.get_exchange_time = lambda: now
    bot.positions = {'A': {'long': {'size': 2.0}, 'short': {'size': 0.0}}}
    events = [fill(now-7_200_000, 'A', 'increase', 3.),
              fill(now-3_000_000, 'A', 'decrease', 5., -20.),
              fill(now-90_123, 'A', 'increase', 2., -0.1)]
    bot._pnls_manager = make_fake_pnls_manager(events)
    evidence = hsl._equity_hard_stop_coin_observed_evidence(bot, events, 'long', 'A')
    if policy != 'always':
        assert evidence.unavailable == 'missing_opening_fill'
        assert hsl._equity_hard_stop_required_fill_history_scope(bot, now, pnl_start_ms=0)[2] is None
        return
    assert evidence.required_start(2., 60_000) == events[-1]['timestamp']
    async def history(**kwargs):
        assert kwargs['hsl_replay_start_ms'] == events[-1]['timestamp']
        return {'timeline': [{
            'timestamp': now-60_000, 'balance': 100., 'realized_pnl': -0.1,
            'realized_pnl_by_coin_pside': {'A': {'long': -0.1, 'short': 0.}},
            'unrealized_pnl_by_coin_pside': {'A': {'long': 0., 'short': 0.}},
        }], 'fill_events': events[-1:], 'panic_flatten_events': []}
    bot.get_balance_equity_history = history
    await bot._equity_hard_stop_initialize_coin_from_history()
    for _ in range(3):
        await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state('long', 'A')
    assert state['last_metrics']['realized_pnl'] == pytest.approx(-0.1)
    assert state['pnl_reset_timestamp_ms'] == events[-1]['timestamp']
    assert state['episode_evidence'].unavailable is None


def test_coin_suffix_requires_coverage_of_the_flat_gap():
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=0.)
    bot.positions = {'A': {'long': {'size': 2.}}}
    events = [fill(10, 'A', 'increase', 3.), fill(20, 'A', 'decrease', 5.),
              fill(100, 'A', 'increase', 2.)]
    checked = []
    def coverage(**kwargs):
        checked.append(kwargs['start_ms'])
        return {'ready': False}
    bot._fill_history_coverage_status = coverage
    assert hsl._equity_hard_stop_coin_observed_evidence(bot, events, 'long', 'A').unavailable == 'missing_opening_fill'
    assert checked == [20]  # New-episode-only coverage at 100 is insufficient.
    bot._fill_history_coverage_status = lambda **kwargs: {'ready': True}
    assert hsl._equity_hard_stop_coin_observed_evidence(bot, events, 'long', 'A').unavailable is None


def test_recovered_closed_episode_retains_its_cooldown_after_restart():
    bot = make_coin_bot()
    bot.hsl['long'].update(restart_after_red_policy='always', cooldown_minutes_after_red=1.)
    now = 1_000_000
    bot.get_exchange_time = lambda: now
    bot.positions = {'A': {'long': {'size': 0.}}}
    events = [fill(10, 'A', 'increase', 3.), fill(20, 'A', 'decrease', 5.),
              fill(now-120_000, 'A', 'increase', 2.),
              fill(now-30_000, 'A', 'decrease', 2., -10.)]
    bot._pnls_manager = make_fake_pnls_manager(events)
    required, start, pairs = hsl._equity_hard_stop_required_fill_history_scope(bot, now, pnl_start_ms=0)
    assert required and start == now-120_000
    assert pairs == {('long', 'A'): now-120_000}
