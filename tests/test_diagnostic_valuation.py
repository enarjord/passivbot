from types import SimpleNamespace

import pytest
import passivbot_rust as pbr

from live.diagnostic_valuation import balance_equity_observation, remember_position_quotes
from live.market_snapshot import MarketSnapshot
from passivbot_monitor import _monitor_equity

NOW = 1_700_000_000_000
SYMBOL = "TEST/USDT:USDT"


@pytest.fixture(autouse=True)
def require_real_passivbot_rust_module():
    if getattr(pbr, "__is_stub__", False):
        pytest.skip("Diagnostic PnL assertions require the real Rust extension")


def quote(price=12.0, age=0, source="fetch_tickers"):
    return MarketSnapshot(SYMBOL, price, price, price, NOW - age, source)


def bot_with_position():
    return SimpleNamespace(
        positions={SYMBOL: {"long": {"size": 2.0, "price": 10.0}}},
        c_mults={SYMBOL: 1.0},
        freshness_ledger=SimpleNamespace(surfaces={
            name: SimpleNamespace(updated_ms=NOW, epoch=1)
            for name in ("balance", "positions")
        }),
        _live_market_snapshot_max_age_ms=lambda: 10_000,
        market_snapshot_provider=SimpleNamespace(_cache={SYMBOL: quote()}),
    )


@pytest.mark.parametrize("age,estimated", [(0, False), (10_000, False), (10_001, True), (60_000, True)])
def test_complete_valuation_and_strict_monitor_is_unchanged(age, estimated):
    bot = bot_with_position()
    bot.market_snapshot_provider._cache[SYMBOL] = quote(age=age)
    result = balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)
    assert result["equity"] == 104.0
    assert result["equity_estimated"] is estimated
    assert result["equity_observation_age_ms"] == age
    assert _monitor_equity(bot, balance_raw=100.0, now_ms=NOW) == (None if estimated else 104.0)
    assert bot._live_market_snapshot_max_age_ms() == 10_000


def test_candle_quotes_are_reused_only_by_diagnostics_and_expire():
    bot = bot_with_position()
    bot.market_snapshot_provider._cache.clear()
    snapshots = {SYMBOL: quote(source="completed_candle_fallback")}
    rows = [{"symbol": SYMBOL, "size": 2.0}]
    remember_position_quotes(bot, snapshots, rows, now_ms=NOW)
    result = balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)
    assert result == {"equity": 104.0, "equity_estimated": True,
                      "equity_valuation_source": "completed_candle", "equity_observation_age_ms": 0}
    assert bot.market_snapshot_provider._cache == {}
    assert _monitor_equity(bot, balance_raw=100.0, now_ms=NOW) is None
    # Keep account facts fresh to isolate price expiry.
    for state in bot.freshness_ledger.surfaces.values():
        state.updated_ms = NOW + 60_001
    assert balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW + 60_001) == {
        "equity": None, "equity_unavailable_reason": "price_missing_or_stale"}
    remember_position_quotes(bot, {}, [], now_ms=NOW + 60_001)
    assert bot._diagnostic_position_quotes == {}


def test_latest_quote_wins_and_never_reuses_old_position_pnl():
    bot = bot_with_position()
    bot._diagnostic_position_quotes = {SYMBOL: quote(price=11.0, age=5_000)}
    bot.positions[SYMBOL]["long"] = {"size": 4.0, "price": 11.0}
    assert balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)["equity"] == 104.0
    bot.market_snapshot_provider._cache[SYMBOL] = quote(price=10.0, age=20_000)
    assert balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)["equity"] == 100.0


def test_signed_short_and_multiplier_match_rust_pnl():
    bot = bot_with_position()
    bot.positions[SYMBOL]["short"] = {"size": -3.0, "price": 15.0}
    bot.c_mults[SYMBOL] = 2.0
    # Long: 2*(12-10)*2=8; short: 3*(15-12)*2=18.
    assert balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)["equity"] == 126.0


@pytest.mark.parametrize("failure,reason", [
    ("partial", "price_missing_or_stale"), ("stale_quote", "price_missing_or_stale"),
    ("future_quote", "price_missing_or_stale"), ("nan_quote", "price_missing_or_stale"),
    ("invalid_position", "invalid_position"), ("invalid_balance", "invalid_balance"),
    ("pending", "account_confirmation_pending"), ("stale_account", "account_state_stale"),
    ("future_account", "account_state_stale"), ("missing_multiplier", "valuation_error"),
])
def test_incomplete_or_invalid_input_never_publishes_partial_equity(failure, reason):
    bot = bot_with_position()
    raw = 100.0
    if failure == "partial":
        bot.positions["OTHER/USDT:USDT"] = {"long": {"size": 1.0, "price": 10.0}}
        bot.c_mults["OTHER/USDT:USDT"] = 1.0
    elif failure == "stale_quote":
        bot.market_snapshot_provider._cache[SYMBOL] = quote(age=60_001)
    elif failure == "future_quote":
        bot.market_snapshot_provider._cache[SYMBOL] = quote(age=-1)
    elif failure == "nan_quote":
        bot.market_snapshot_provider._cache[SYMBOL] = quote(price=float("nan"))
    elif failure == "invalid_position":
        bot.positions[SYMBOL]["long"]["size"] = float("nan")
    elif failure == "invalid_balance":
        raw = float("inf")
    elif failure == "pending":
        bot._authoritative_pending_confirmations = {"positions": 2}
    elif failure == "stale_account":
        bot.freshness_ledger.surfaces["balance"].updated_ms = NOW - 60_001
    elif failure == "future_account":
        bot.freshness_ledger.surfaces["positions"].updated_ms = NOW + 1
    elif failure == "missing_multiplier":
        bot.c_mults.clear()
    assert balance_equity_observation(bot, balance_raw=raw, now_ms=NOW) == {
        "equity": None, "equity_unavailable_reason": reason}


def test_flat_account_and_old_account_estimate():
    bot = bot_with_position()
    bot.positions.clear()
    bot.freshness_ledger.surfaces["positions"].updated_ms = NOW - 20_000
    result = balance_equity_observation(bot, balance_raw=0.0, now_ms=NOW)
    assert result == {"equity": 0.0, "equity_estimated": True,
                      "equity_valuation_source": "flat", "equity_observation_age_ms": 20_000}


def test_calculation_failure_does_not_leak_exception_or_partial_pnl(monkeypatch):
    bot = bot_with_position()
    def fail(*args):
        raise RuntimeError("private-connector-secret")
    monkeypatch.setattr(pbr, "calc_pnl_long", fail)
    result = balance_equity_observation(bot, balance_raw=100.0, now_ms=NOW)
    assert result == {"equity": None, "equity_unavailable_reason": "valuation_error"}
