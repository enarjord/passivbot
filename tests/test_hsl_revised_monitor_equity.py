"""Current account reporting must not depend on legacy balance callbacks."""
from types import SimpleNamespace

import pytest

from live.freshness import FreshnessLedger
from live.market_snapshot import MarketSnapshot
from passivbot_monitor import _build_health_summary_payload

NOW = 1_800_000_000_000


def bot():
    ledger = FreshnessLedger(now_ms=NOW)
    for surface in ('balance', 'positions'):
        ledger.stamp(surface, now_ms=NOW)
    return SimpleNamespace(
        config={'live': {'hsl_engine': 'revised'}},
        positions={'A': {'long': {'size': 2., 'price': 100.},
                         'short': {'size': -3., 'price': 110.}}},
        c_mults={'A': 2.},
        market_snapshot_provider=SimpleNamespace(_cache={
            'A': MarketSnapshot('A', 89., 91., 90., NOW, 'fixture')}),
        freshness_ledger=ledger, _authoritative_pending_confirmations={},
        _live_market_snapshot_max_age_ms=lambda: 10_000,
        get_raw_balance=lambda: 100., get_hysteresis_snapped_balance=lambda: 100.,
        _monitor_last_equity=1e-12, _health_start_ms=NOW,
        _health_orders_placed=0, _health_orders_cancelled=0, _health_fills=0,
        _health_pnl=0., _health_ws_reconnects=0, _health_rate_limits=0,
    )


def test_health_uses_current_account_and_both_sides_without_legacy_callback():
    b = bot()
    # Long -40, short +120. The startup sentinel is never an equity observation.
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == 180.
    b.get_raw_balance = lambda: 80.
    b.positions['A']['long']['size'] = 0.
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == 200.
    b.positions['A']['short']['size'] = 0.
    b.market_snapshot_provider._cache.clear()
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == 80.
    assert b._monitor_last_equity == 1e-12  # Passive reporting, no trading state mutation.


@pytest.mark.parametrize('damage', ['balance_stale', 'positions_pending', 'quote_missing',
    'quote_stale', 'quote_future', 'bad_basis', 'bad_size', 'bad_multiplier', 'native_failure'])
def test_unavailable_equity_is_not_a_partial_portfolio_or_stale_placeholder(monkeypatch, damage):
    b = bot()
    if damage == 'balance_stale':
        b.freshness_ledger.stamp('balance', now_ms=NOW-10_001)
    elif damage == 'positions_pending':
        b._authoritative_pending_confirmations['positions'] = 1
    elif damage == 'quote_missing':
        b.market_snapshot_provider._cache.clear()
    elif damage.startswith('quote_'):
        from dataclasses import replace
        b.market_snapshot_provider._cache['A'] = replace(b.market_snapshot_provider._cache['A'],
            fetched_ms=NOW-10_001 if damage == 'quote_stale' else NOW+1)
    elif damage == 'bad_basis': b.positions['A']['long']['price'] = float('nan')
    elif damage == 'bad_size': b.positions['A']['long']['size'] = float('nan')
    elif damage == 'bad_multiplier': b.c_mults['A'] = float('inf')
    else:
        import passivbot_monitor
        def broken(*args): raise RuntimeError('diagnostic calculation failure')
        monkeypatch.setattr(passivbot_monitor, '_calc_monitor_pnl', broken)
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] is None


def test_zero_and_negative_equity_are_reported_without_balance_substitution():
    b = bot()
    b.positions['A']['short']['size'] = 0.
    b.get_raw_balance = lambda: 40.
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == 0.
    b.get_raw_balance = lambda: 30.
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == -10.


def test_legacy_reporting_is_unchanged():
    b = bot()
    b.config['live']['hsl_engine'] = 'legacy'
    b._monitor_last_equity = 123.
    assert _build_health_summary_payload(b, now_ms=NOW)['equity'] == 123.
