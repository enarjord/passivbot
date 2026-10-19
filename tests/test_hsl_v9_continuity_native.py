"""Offline capture -> retained factual owner -> real Rust continuity fold."""

import json

from live.hsl_candles import Sources
from live.hsl_inputs import capture_candles
from live.hsl_runtime import capture, evaluate
from live.market_snapshot import MarketSnapshot
from test_hsl_runtime import NOW, SYMBOL, bot


def query(value, now, mark, sources=None):
    value.get_exchange_time = lambda: now
    ledger = value._ensure_freshness_ledger()
    for surface in ("balance", "positions", "open_orders"):
        ledger.stamp(surface, now_ms=now)
    requests, unavailable = capture(
        value,
        {SYMBOL: MarketSnapshot(SYMBOL, mark, mark, mark, now - 1, "fake")},
        sources or {},
        symbols={"long": [SYMBOL], "short": []},
        now_ms=now,
        utc_now_ms=now,
        max_current_age_ms=10_000,
    )
    assert not unavailable
    (decision,) = evaluate(requests)
    return decision, requests[0].replay


def test_current_quote_replacement_does_not_reseed_the_committed_initializer():
    value = bot()
    value.config["bot"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] = 1_000_000.0
    initial, replay = query(value, NOW, 90.0)
    advances = replay.cache_info()["advanced_events"]
    recovered, same = query(value, NOW + 1, 100.0)
    returned, _ = query(value, NOW + 2, 90.0)
    assert same is replay
    assert initial.action == returned.action == "panic"
    assert recovered.action == "normal"
    assert replay.cache_info()["advanced_events"] == advances
    assert json.loads(initial.payload)["risk"]["drawdown"] == 100.0
    assert json.loads(returned.payload)["risk"]["drawdown"] == 100.0
    assert replay.anchor.observed_ms == NOW


def test_extra_quote_queries_cannot_change_a_later_completed_candle_fold():
    direct, queried = bot(), bot()
    query(direct, NOW, 90.0)
    query(queried, NOW, 90.0)
    query(queried, NOW + 1, 1.0)
    query(queried, NOW + 2, 100.0)
    end = NOW + 60_001
    source = Sources(
        (capture_candles([dict(ts=NOW, o=100.0, h=100.0, l=80.0, c=80.0)],
                        minutes=1, observed_at=end),), (), 0
    )
    first, _ = query(direct, end, 80.0, {SYMBOL: source})
    second, _ = query(queried, end, 80.0, {SYMBOL: source})
    assert json.loads(first.payload)["risk"] == json.loads(second.payload)["risk"]


def test_discarding_only_the_derived_native_fold_preserves_the_same_result():
    value = bot()
    before, replay = query(value, NOW, 90.0)
    anchor = replay.anchor
    replay.clear_derived_cache()
    after, same = query(value, NOW, 90.0)
    assert same is replay and replay.anchor is anchor
    assert json.loads(before.payload) == json.loads(after.payload)
    assert replay.cache_info()["rebuilds"] == 2


def test_policy_change_reuses_the_raw_initializer_with_the_current_policy():
    value = bot()
    initial, replay = query(value, NOW, 90.0)
    anchor = replay.anchor
    value.config["bot"]["long"]["hsl"]["red_threshold"] = 0.99
    current, same = query(value, NOW + 1, 90.0)
    assert initial.action == "panic" and current.action == "normal"
    assert same is replay and replay.anchor is anchor
    assert replay.cache_info()["rebuilds"] == 2


def test_same_timestamp_account_correction_is_committed_before_preview():
    value = bot()
    _, replay = query(value, NOW, 90.0)
    anchor = replay.anchor
    advances = replay.cache_info()["advanced_events"]
    query(value, NOW, 90.0)
    assert replay.cache_info()["advanced_events"] == advances
    value.positions[SYMBOL]["long"]["size"] = 20.0
    changed, same = query(value, NOW, 90.0)
    assert same is replay
    assert json.loads(changed.payload)["risk"]["drawdown"] == 200.0
    value.positions[SYMBOL]["long"].update(size=0.0, price=0.0)
    flat, _ = query(value, NOW, 90.0)
    assert not json.loads(flat.payload)["risk"]["exposed"]
    before = json.loads(flat.payload)
    replay.clear_derived_cache()
    rebuilt, _ = query(value, NOW, 90.0)
    assert json.loads(rebuilt.payload) == before
    assert replay.anchor is anchor
