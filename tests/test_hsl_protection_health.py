import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from live.hsl_protection import Health, ProtectionHealth, Scope, affected_scopes, evaluate_emergency


@pytest.fixture
def real_rust():
    import passivbot_rust as pbr
    assert not getattr(pbr, '__is_stub__', False)
    return pbr


def test_continuous_outage_is_not_renewed_by_reason_changes_or_restart(tmp_path):
    path = tmp_path / 'protection.json'
    scope = Scope('coin', 'short', 'A')
    health = ProtectionHealth(path)
    health.unavailable(scope, now_ms=1_000_000, reason='timeout', grace_ms=120_000)
    health.unavailable(scope, now_ms=1_060_000, reason='position_mismatch', grace_ms=120_000)
    restored = ProtectionHealth(path)
    assert restored.scopes[scope].unavailable_since_ms == 1_000_000
    assert restored.payload(1_100_000, 120_000)[0]['grace_remaining_seconds'] == 20.0
    restored.evaluated_successfully(scope, now_ms=1_110_000)
    assert ProtectionHealth(path).scopes == {}
    restored.unavailable(scope, now_ms=1_200_000, reason='timeout', grace_ms=120_000)
    assert restored.scopes[scope].unavailable_since_ms == 1_200_000


def test_exit_commitment_survives_normal_signal_recovery_until_flat(tmp_path):
    path = tmp_path / 'protection.json'
    scope = Scope('coin', 'long', 'A')
    health = ProtectionHealth(path)
    state = health.unavailable(scope, now_ms=1_000_000, reason='no_history', grace_ms=120_000)
    state.exit_committed = True
    state.exit_started_ms = 1_000_000
    health.save()
    restored = ProtectionHealth(path)
    restored.evaluated_successfully(scope, now_ms=1_200_000)
    assert restored.pending_exits() == {scope}
    restored.confirm_flat(scope, now_ms=1_200_000)
    assert not restored.pending_exits()
    assert restored.scopes[scope].exit_confirmed_flat
    restored.evaluated_successfully(scope, now_ms=1_210_000)
    assert restored.scopes[scope].exit_confirmed_flat
    restored.release_flat_hold(scope)
    assert not restored.scopes[scope].exit_confirmed_flat


@pytest.mark.parametrize('content', ['{', '{}', '{"version":1,"scopes":[{}]}'])
def test_corrupt_journal_does_not_grant_a_fresh_grace(tmp_path, content):
    path = tmp_path / 'protection.json'
    path.write_text(content)
    health = ProtectionHealth(path)
    state = health.unavailable(Scope('pside', 'long'), now_ms=1_000_000,
                               reason='history_unavailable', grace_ms=120_000)
    assert state.unavailable_since_ms == 880_000


def test_degraded_but_usable_evaluation_clears_only_its_scope():
    health = ProtectionHealth()
    a, b = Scope('coin', 'long', 'A'), Scope('coin', 'long', 'B')
    for scope in (a, b):
        health.unavailable(scope, now_ms=1_000_000, reason='timeout', grace_ms=120_000)
    health.evaluated_successfully(a, now_ms=1_100_000, degraded_reason='coherent_prefix')
    assert health.scopes[a].status == 'degraded'
    assert health.scopes[a].unavailable_since_ms is None
    assert health.scopes[b].unavailable_since_ms == 1_000_000


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_failure_attribution_is_scoped(mode):
    bot = SimpleNamespace(_equity_hard_stop_signal_mode=lambda: mode,
                          _equity_hard_stop_enabled=lambda *_, **kw: True,
                          bot_value=lambda *args: 1)
    scopes = affected_scopes(bot, {'A': {'long'}, 'B': {'short'}},
                             {'pside': 'long', 'symbol': 'A'})
    expected = ({Scope('coin', 'long', 'A')} if mode == 'coin' else
                {Scope('pside', 'long')} if mode == 'pside' else
                {Scope('unified', 'long'), Scope('unified', 'short')})
    assert scopes == expected


@pytest.mark.parametrize('elapsed,trigger', [(0, False), (119_999, False), (120_000, True), (120_001, True)])
def test_real_rust_raw_loss_only_triggers_after_grace(real_rust, elapsed, trigger):
    budget, dd, expired, panic = real_rust.hsl_emergency_signal(
        True, 10_000.0, 1, -1_000.0, 0.1, elapsed, 120_000, False)
    assert budget == 10_000.0
    assert dd == 0.1
    assert expired == (elapsed >= 120_000)
    assert panic is trigger


def test_real_rust_coin_budget_and_known_realized_loss(real_rust):
    budget, dd, _, panic = real_rust.hsl_emergency_signal(
        True, 10_000.0, 10, -50.0, 0.1, 120_000, 120_000, False, 60.0)
    assert budget == 1_000.0
    assert dd == 0.11
    assert panic
    assert not real_rust.hsl_emergency_signal(
        True, 10_000.0, 10, -50.0, 0.1, 120_000, 120_000, False)[3]


def test_missing_execution_history_exits_after_grace_even_when_profitable(real_rust):
    assert not real_rust.hsl_emergency_signal(True, 1_000.0, 1, 100.0, 0.1, 119_999, 120_000, True)[3]
    assert real_rust.hsl_emergency_signal(True, 1_000.0, 1, 100.0, 0.1, 120_000, 120_000, True)[3]
    assert not real_rust.hsl_emergency_signal(False, 1_000.0, 1, -1000.0, 0.1, 120_000, 120_000, True)[3]


def test_emergency_stop_provenance_survives_recovery_but_is_bounded_to_close_window(tmp_path):
    from live.hsl_protection import emergency_stop_applies
    path = tmp_path / 'protection.json'
    scope = Scope('coin', 'long', 'A')
    health = ProtectionHealth(path)
    state = health.unavailable(scope, now_ms=1000, reason='history_timeout', grace_ms=0)
    state.exit_started_ms = 2000
    state.exit_committed = True
    health.confirm_flat(scope, now_ms=3000)
    health.evaluated_successfully(scope, now_ms=4000)
    health.release_flat_hold(scope)
    bot = SimpleNamespace(_hsl_protection_health=ProtectionHealth(path),
                          _equity_hard_stop_signal_mode=lambda: 'coin')
    assert emergency_stop_applies(bot, 'long', 'A', 2000)
    assert emergency_stop_applies(bot, 'long', 'A', 3000)
    assert not emergency_stop_applies(bot, 'long', 'A', 1999)
    assert not emergency_stop_applies(bot, 'long', 'A', 3001)
    assert not emergency_stop_applies(bot, 'short', 'A', 2500)
    assert not emergency_stop_applies(bot, 'long', 'B', 2500)


def test_clock_rollback_is_persisted_without_renewing_grace(tmp_path):
    path = tmp_path / 'protection.json'
    scope = Scope('coin', 'long', 'A')
    health = ProtectionHealth(path)
    health.unavailable(scope, now_ms=1_000_000, reason='timeout', grace_ms=120_000)
    health.unavailable(scope, now_ms=990_000, reason='timeout', grace_ms=120_000)
    assert ProtectionHealth(path).scopes[scope].unavailable_since_ms == 870_000


def test_deliberate_mode_or_enablement_change_retires_old_scope():
    from live.hsl_protection import reconcile_config
    health = ProtectionHealth()
    health.scopes = {Scope('coin', 'long', 'A'): Health(exit_committed=True),
                     Scope('pside', 'short'): Health(), Scope('pside', 'long'): Health()}
    bot = SimpleNamespace(_hsl_protection_health=health,
                          _equity_hard_stop_signal_mode=lambda: 'pside',
                          _equity_hard_stop_enabled=lambda side, **kwargs: side == 'long')
    reconcile_config(bot)
    assert set(health.scopes) == {Scope('pside', 'long')}


def test_corrupt_journal_recovery_restores_grace_only_for_evaluated_scope(tmp_path):
    path = tmp_path / "protection.json"
    path.write_text("{")
    health = ProtectionHealth(path)
    a, b = Scope("coin", "long", "A"), Scope("coin", "long", "B")
    health.unavailable(a, now_ms=1_000_000, reason="timeout", grace_ms=120_000)
    health.evaluated_successfully(a, now_ms=1_100_000)
    health.unavailable(a, now_ms=1_200_000, reason="new_timeout", grace_ms=120_000)
    health.unavailable(b, now_ms=1_200_000, reason="timeout", grace_ms=120_000)
    assert health.scopes[a].unavailable_since_ms == 1_200_000
    assert health.scopes[b].unavailable_since_ms == 1_080_000


def test_completed_normal_evaluation_clears_old_quote_blockage():
    health = ProtectionHealth()
    scope = Scope('coin', 'long', 'A')
    state = health.unavailable(scope, now_ms=1000, reason='timeout', grace_ms=120_000)
    state.execution_blocked = 'MarketSnapshotUnavailable'
    health.evaluated_successfully(scope, now_ms=2000)
    assert state.status == 'usable'
    assert state.execution_blocked == ''


@pytest.mark.asyncio
async def test_recommit_after_flat_confirmation_survives_restart(tmp_path, real_rust):
    path = tmp_path / 'protection.json'
    scope = Scope('coin', 'long', 'A')
    health = ProtectionHealth(path)
    state = health.unavailable(scope, now_ms=1000, reason='timeout', grace_ms=0)
    state.exit_committed = True
    state.exit_started_ms = 1000
    health.confirm_flat(scope, now_ms=2000)
    bot = SimpleNamespace(
        _hsl_protection_health=health,
        config={'live': {'hsl_unavailable_grace_seconds': 0.0}},
        positions={'A': {'long': {'size': 1.0}}}, open_orders={},
        _equity_hard_stop_signal_mode=lambda: 'coin',
        _equity_hard_stop_enabled=lambda *a, **k: True,
        _equity_hard_stop_config=lambda *a: {'red_threshold': 0.1},
        _calc_upnl_sum_strict=AsyncMock(return_value=-100.0),
        get_exchange_time=lambda: 3000, get_raw_balance=lambda: 100.0,
        bot_value=lambda *a: 1,
        _pnls_manager=SimpleNamespace(get_events=lambda: [object()]),
    )
    await evaluate_emergency(bot, {'A': {'long'}})
    restored = ProtectionHealth(path)
    assert not restored.journal_invalid
    assert restored.pending_exits() == {scope}
    assert not restored.scopes[scope].exit_confirmed_flat
    assert restored.scopes[scope].exit_started_ms == 3000


@pytest.mark.parametrize('fields', [
    {'exit_committed': True}, {'exit_confirmed_flat': True},
    {'exit_confirmed_flat': True, 'exit_started_ms': 1000},
    {'exit_confirmed_flat': True, 'exit_started_ms': 1000, 'exit_flat_ms': 999},
    {'exit_committed': True, 'exit_started_ms': 1000, 'exit_flat_ms': 2000},
    {'exit_flat_ms': 2000}, {'exit_started_ms': 1000},
])
def test_semantically_incomplete_emergency_journal_is_rejected(tmp_path, fields):
    from dataclasses import asdict
    path = tmp_path / 'protection.json'
    path.write_text(json.dumps({'version': 1, 'scopes': [
        {'scope': asdict(Scope('coin', 'long', 'A')), 'health': asdict(Health(**fields))},
    ]}))
    restored = ProtectionHealth(path)
    assert restored.journal_invalid
    assert not restored.scopes
    state = restored.unavailable(Scope('coin', 'long', 'A'), now_ms=1_000_000,
                                  reason='history', grace_ms=120_000)
    assert state.unavailable_since_ms == 880_000


def test_unified_commitment_targets_and_holds_both_enabled_candidate_sides():
    from live.hsl_protection import targets_for_scopes, holds_after_emergency_exit, emergency_stop_applies
    health = ProtectionHealth()
    scope = Scope('unified', 'long')
    health.scopes[scope] = Health(exit_committed=True, exit_started_ms=1000)
    bot = SimpleNamespace(_hsl_protection_health=health, _equity_hard_stop_signal_mode=lambda: 'unified')
    candidates = {'A': {'long', 'short'}, 'B': {'short'}}
    assert targets_for_scopes(bot, {scope}, candidates) == candidates
    assert holds_after_emergency_exit(bot, 'short', 'B')
    assert emergency_stop_applies(bot, 'short', None, 1500)
    health.confirm_flat(scope, now_ms=2000)
    assert holds_after_emergency_exit(bot, 'short', 'B')
    assert not emergency_stop_applies(bot, 'short', None, 2001)


def test_corrupt_journal_repairs_evaluated_scope_durably_without_granting_unknown_scopes_grace(tmp_path):
    path = tmp_path / 'protection.json'
    path.write_text('{')
    health = ProtectionHealth(path)
    a, b = Scope('coin', 'long', 'A'), Scope('coin', 'long', 'B')
    health.evaluated_successfully(a, now_ms=1_000_000)
    restored = ProtectionHealth(path)
    assert restored.durable
    restored.unavailable(a, now_ms=1_200_000, reason='new_outage', grace_ms=120_000)
    restored.unavailable(b, now_ms=1_200_000, reason='unknown_scope', grace_ms=120_000)
    assert restored.scopes[a].unavailable_since_ms == 1_200_000
    assert restored.scopes[b].unavailable_since_ms == 1_080_000


def test_degraded_evaluation_count_is_visible_and_resets_on_recovery():
    health = ProtectionHealth()
    scope = Scope('coin', 'long', 'A')
    for count in range(1, 4):
        health.evaluated_successfully(scope, now_ms=count * 1000, degraded_reason='unordered_nonflattening_fill_cohort')
        assert health.payload(count * 1000, 120_000)[0]['degraded_evaluations'] == count
    health.evaluated_successfully(scope, now_ms=4000)
    assert health.scopes[scope].degraded_evaluations == 0
    health.evaluated_successfully(scope, now_ms=5000, degraded_reason='unordered_nonflattening_fill_cohort')
    health.unavailable(scope, now_ms=6000, reason='history', grace_ms=120_000)
    assert health.scopes[scope].degraded_evaluations == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "invalid_fill", "cache", "fatal", "unexpected_value"])
async def test_optional_tail_timeout_is_bounded_and_cannot_delay_raw_red(monkeypatch, real_rust, failure):
    import asyncio
    import live.hsl_protection as protection
    monkeypatch.setattr(protection, '_EMERGENCY_FILL_REFRESH_TIMEOUT_SECONDS', 0.01)
    health = ProtectionHealth()
    scope = Scope('coin', 'long', 'A')
    health.unavailable(scope, now_ms=1000, reason='history', grace_ms=0)
    cancelled = asyncio.Event()
    calls = []
    async def stalled(**kwargs):
        calls.append(kwargs)
        from fill_events_manager import FillEventCacheContractError
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from passivbot_exceptions import FatalBotException
        errors = {
            'invalid_fill': AuthoritativeSurfaceUnavailable('fills', 'unusable fetched fill'),
            'cache': FillEventCacheContractError('invalid fill-cache contract'),
            'fatal': FatalBotException('fatal producer failure'),
            'unexpected_value': ValueError('invalid configuration'),
        }
        if failure in errors:
            raise errors[failure]
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    bot = SimpleNamespace(
        _hsl_protection_health=health, config={'live': {'hsl_unavailable_grace_seconds': 0.0}},
        positions={'A': {'long': {'size': 1.0}}}, open_orders={},
        _equity_hard_stop_signal_mode=lambda: 'coin', _equity_hard_stop_enabled=lambda *a, **k: True,
        _equity_hard_stop_config=lambda *a: {'red_threshold': 0.1},
        _calc_upnl_sum_strict=AsyncMock(return_value=-1.0), get_exchange_time=lambda: 1000,
        get_raw_balance=lambda: 100.0, bot_value=lambda *a: 1,
        _pnls_manager=SimpleNamespace(get_events=lambda: [object()]), update_pnls=stalled,
    )
    if failure in {'fatal', 'unexpected_value'}:
        from passivbot_exceptions import FatalBotException
        with pytest.raises(FatalBotException if failure == 'fatal' else ValueError):
            await evaluate_emergency(bot, {'A': {'long'}})
        return
    await evaluate_emergency(bot, {'A': {'long'}})
    assert cancelled.is_set() is (failure == 'timeout')
    assert not health.pending_exits()
    assert health.scopes[scope].emergency_active
    assert health.scopes[scope].unavailable_since_ms == 1000
    # Enrichment's retry backoff cannot postpone a fresh raw-loss threshold.
    bot._calc_upnl_sum_strict.return_value = -20.0
    await evaluate_emergency(bot, {'A': {'long'}})
    assert health.pending_exits() == {scope}
    assert calls == [{'source': 'hsl_emergency'}]


@pytest.mark.asyncio
async def test_new_fill_confirmation_defers_enriched_recursion(real_rust):
    health = ProtectionHealth()
    scope = Scope('coin', 'long', 'A')
    health.unavailable(scope, now_ms=1000, reason='history', grace_ms=0)
    bot = SimpleNamespace(
        _hsl_protection_health=health, config={'live': {'hsl_unavailable_grace_seconds': 0.0}},
        positions={'A': {'long': {'size': 1.0}}}, open_orders={},
        _equity_hard_stop_signal_mode=lambda: 'coin', _equity_hard_stop_enabled=lambda *a, **k: True,
        _equity_hard_stop_config=lambda *a: {'red_threshold': 0.1},
        _calc_upnl_sum_strict=AsyncMock(return_value=-1.0), get_exchange_time=lambda: 1000,
        get_raw_balance=lambda: 100.0, bot_value=lambda *a: 1,
        _pnls_manager=SimpleNamespace(get_events=lambda: [object()]),
    )
    async def discover_same_size_round_trip(**kwargs):
        bot._authoritative_pending_confirmations = {'positions': 2, 'balance': 2}
    bot.update_pnls = AsyncMock(side_effect=discover_same_size_round_trip)
    await evaluate_emergency(bot, {'A': {'long'}})
    bot.update_pnls.assert_awaited_once()
    bot._calc_upnl_sum_strict.assert_awaited_once()
    assert not health.pending_exits()
    assert health.scopes[scope].emergency_active
    assert health.scopes[scope].unavailable_since_ms == 1000
