"""Live coin boundaries retain the same proven suffix as canonical HSL replay."""

from unittest.mock import AsyncMock

import pytest

import passivbot_hsl as hsl
from test_hsl_coin_mode import make_coin_bot, make_fake_pnls_manager
from test_hsl_episode_evidence import _mark_emergency_tail_fresh


def _make_live_prefix_bot():
    """Provide an incomplete old episode followed by a provable held suffix."""
    bot = make_coin_bot()
    _mark_emergency_tail_fresh(bot)
    bot.hsl['long'].update(
        restart_after_red_policy='always', cooldown_minutes_after_red=1.0
    )
    bot.positions = {'A': {'long': {'size': 1.0}, 'short': {'size': 0.0}}}
    events = [
        dict(timestamp=10_000, symbol='A', pside='long', action='increase', qty=3.0, pnl=0.0),
        dict(timestamp=20_000, symbol='A', pside='long', action='decrease', qty=5.0, pnl=-20.0),
        dict(timestamp=100_000, symbol='A', pside='long', action='increase', qty=3.0, pnl=-0.1),
        dict(timestamp=110_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-2.0),
        dict(timestamp=120_000, symbol='A', pside='long', action='decrease', qty=1.0, pnl=-1.0),
    ]
    bot._pnls_manager = make_fake_pnls_manager(events)
    return bot, events


@pytest.mark.parametrize('start_ms', [None, 0, 110_000])
def test_live_coin_evidence_recovers_before_projecting_the_window(start_ms):
    """A clipped reduction keeps its proven starting quantity and PnL baseline."""
    bot, events = _make_live_prefix_bot()
    evidence = hsl._equity_hard_stop_live_coin_episode_evidence(
        bot, events, 'long', 'A', {'pnl_reset_timestamp_ms': None}, start_ms
    )
    evidence.require_position(1.0, pside='long', symbol='A')
    assert evidence.recovered_from_flat_ms == 20_000
    if start_ms == 110_000:
        assert evidence.rows[0][0] == 110_000
        assert evidence.sizes == (3.0, 2.0, 1.0)
        assert evidence.realized_prefix == pytest.approx((0.0, -2.0, -3.0))
    else:
        assert evidence.rows[0][0] == 100_000
        assert evidence.realized_prefix[-1] == pytest.approx(-3.1)
        assert evidence.degraded_reason == 'position_anchored_episode_suffix'


@pytest.mark.asyncio
async def test_live_boundary_poll_does_not_resurrect_consumed_incomplete_prefix(monkeypatch):
    """Repeated live polls accept unchanged canonical suffix evidence without replay."""
    bot, events = _make_live_prefix_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_apply_coin_metrics_sample(
        'long', 'A', 120_000, 100.0, 0.0, 0.0, 0.0
    )
    state = bot._hsl_coin_state('long', 'A')
    state['episode_evidence'] = hsl._equity_hard_stop_coin_observed_evidence(
        bot, events, 'long', 'A'
    )
    assert state['pnl_reset_timestamp_ms'] is None
    monkeypatch.setattr(hsl, '_equity_hard_stop_live_coin_history_start_ms', lambda *args: None)
    replay = AsyncMock(return_value=True)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    for _ in range(3):
        assert not await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(
            bot, 180_000, 100.0
        )
    replay.assert_not_awaited()

    assert state['episode_evidence'].unavailable is None
    assert state['episode_evidence'].ending_size == 1.0


@pytest.mark.asyncio
async def test_initializer_and_live_share_proof_across_a_clipped_opening():
    """Canonical replay projects full-tape proof when lookback begins mid-episode."""
    bot, events = _make_live_prefix_bot()
    bot.config['live']['pnls_max_lookback_days'] = 75_000 / 86_400_000
    cutoff = 105_000

    async def history(**kwargs):
        """Return factual samples and only the fills inside configured lookback."""
        assert kwargs['hsl_replay_start_ms'] == cutoff
        return {
            'timeline': [{
                'timestamp': ts, 'balance': 100.0, 'realized_pnl': -3.0,
                'realized_pnl_by_coin_pside': {'A': {'long': -3.0, 'short': 0.0}},
                'unrealized_pnl_by_coin_pside': {'A': {'long': 0.0, 'short': 0.0}},
            } for ts in (120_000, 180_000)],
            'fill_events': events[-2:], 'panic_flatten_events': [],
        }

    bot.get_balance_equity_history = history
    await bot._equity_hard_stop_initialize_coin_from_history()
    state = bot._hsl_coin_state('long', 'A')
    assert state['episode_evidence'].unavailable is None
    assert state['episode_evidence'].sizes == (3.0, 2.0, 1.0)
    for _ in range(3):
        await bot._equity_hard_stop_check_coin()
    assert bot._hsl_coin_state('long', 'A') is state
    assert state['episode_evidence'].unavailable is None
    assert state['episode_evidence'].ending_size == 1.0


@pytest.mark.parametrize('blocker', [
    'threshold', 'never', 'coverage_gap', 'stale_tail', 'pending_confirmation',
    'short_gap', 'position_mismatch', 'ambiguous_retained', 'overclose_retained',
])
def test_live_prefix_recovery_preserves_evidence_guards(blocker):
    """Incomplete evidence never gains permission from the live consumer alone."""
    bot, events = _make_live_prefix_bot()
    if blocker in ('threshold', 'never'):
        bot.hsl['long']['restart_after_red_policy'] = blocker
    elif blocker == 'coverage_gap':
        bot._fill_history_coverage_status = lambda **kwargs: {'ready': False}
    elif blocker == 'stale_tail':
        bot._hsl_fill_tail_observation = None
    elif blocker == 'pending_confirmation':
        bot._authoritative_pending_confirmations = {
            'positions': bot.freshness_ledger.epoch + 1
        }
    elif blocker == 'short_gap':
        bot.hsl['long']['cooldown_minutes_after_red'] = 2.0
    elif blocker == 'position_mismatch':
        bot.positions['A']['long']['size'] = 2.0
    elif blocker == 'ambiguous_retained':
        events[-1]['timestamp'] = events[-2]['timestamp']
        events[-1]['action'] = 'increase'
    else:
        events[-1]['qty'] = 4.0
    evidence = hsl._equity_hard_stop_live_coin_episode_evidence(
        bot, events, 'long', 'A', {'pnl_reset_timestamp_ms': None}, None
    )
    with pytest.raises(hsl.EpisodeEvidenceUnavailable):
        evidence.require_position(bot.positions['A']['long']['size'], pside='long', symbol='A')


def test_live_evidence_preserves_proven_reset_tail():
    """A reset watermark excludes old fills without losing current entry costs."""
    bot, events = _make_live_prefix_bot()
    evidence = hsl._equity_hard_stop_live_coin_episode_evidence(
        bot, events, 'long', 'A', {'pnl_reset_timestamp_ms': 100_000}, None
    )
    evidence.require_position(1.0, pside='long', symbol='A')
    assert evidence.rows[0][0] == 100_000
    assert evidence.realized_prefix[-1] == pytest.approx(-3.1)
    assert evidence.recovered_from_flat_ms is None


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['fresh', 'timeout', 'new_positions', 'pending_confirmation'])
async def test_live_boundary_refresh_orders_candidate_tail_before_acceptance(monkeypatch, outcome):
    """Live recovery refreshes stale tail proof without waiving failed confirmations."""
    bot, events = _make_live_prefix_bot()
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_apply_coin_metrics_sample(
        'long', 'A', 120_000, 100.0, 0.0, 0.0, 0.0
    )
    state = bot._hsl_coin_state('long', 'A')
    state['episode_evidence'] = hsl._equity_hard_stop_coin_observed_evidence(
        bot, events, 'long', 'A'
    )
    bot._hsl_fill_tail_observation = None
    attempts = []

    async def update():
        """Simulate the owned fill refresh and its position revision certificate."""
        attempts.append(True)
        if outcome == 'timeout':
            raise TimeoutError()
        ledger = bot.freshness_ledger
        bot._hsl_fill_tail_observation = (ledger.epoch, ledger.surfaces['positions'].revision)
        if outcome == 'new_positions':
            ledger.stamp('positions', now_ms=180_001)
        elif outcome == 'pending_confirmation':
            bot._authoritative_pending_confirmations = {'positions': ledger.epoch + 1}

    bot.update_pnls = update
    monkeypatch.setattr(hsl, '_equity_hard_stop_live_coin_history_start_ms', lambda *args: None)
    replay = AsyncMock(return_value=True)
    monkeypatch.setattr(hsl, '_equity_hard_stop_replay_live_restart', replay)
    if outcome == 'fresh':
        assert not await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, 180_000, 100.0)
        assert state['episode_evidence'].unavailable is None
    else:
        with pytest.raises(hsl.EpisodeEvidenceUnavailable):
            await hsl._equity_hard_stop_refresh_live_coin_episode_boundaries(bot, 180_000, 100.0)
    assert attempts == [True]
    replay.assert_not_awaited()
