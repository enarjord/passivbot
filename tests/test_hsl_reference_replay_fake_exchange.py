"""Offline fake-live exchange supplies fills/positions for boundary reconstruction."""

from dataclasses import replace

import pytest

from hsl_reference import Fill, LifecycleEvidence, Observation, Position, dec, permission, signal
from hsl_reference_replay import capture, capture_pair, scope_boundaries


@pytest.mark.fake_live
@pytest.mark.parametrize("pside", ["long", "short"])
def test_fake_exchange_partial_final_delayed_fill_and_cache_free_replay(pside):
    from exchanges.fake import FakeCCXTClient

    symbol = "TEST/USDT:USDT"
    direction = 1 if pside == "long" else -1
    entry, close = ("buy", "sell") if direction == 1 else ("sell", "buy")
    prices = [100, 100, 80, 70, 70] if direction == 1 else [100, 100, 120, 130, 130]

    def action(side, qty):
        return {"type": "manual_fill", "symbol": symbol, "position_side": pside,
                "side": side, "qty": qty, "reduce_only": side == close}

    client = FakeCCXTClient({
        "name": "hsl_reference_boundary", "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60, "account": {"balance": 1000},
        "symbols": {symbol: {"qty_step": .1, "price_step": .1, "min_qty": .1,
                             "min_cost": 1, "taker": .001}},
        "timeline": [{"t": i, "prices": {symbol: price},
                      "actions": [action(entry, 2)] if i == 1 else
                                 [action(close, 1)] if i in (2, 3) else []}
                     for i, price in enumerate(prices)],
    })
    start = client.now_ms

    def snapshot(events=None):
        state = client.positions[(symbol, pside)]
        if events is None:
            events = client.get_fill_events(start, client.now_ms)
        fills = [Fill(e["id"], e["timestamp"], e["qty"] * (1 if e["side"] == "buy" else -1),
                      e["price"], e["pnl"], -e["fees"]["cost"], sequence=int(e["id"]))
                 for e in events]
        p = capture_pair(symbol, Position(direction * state["size"], state["entry_price"],
                                          prices[client.current_index], pside=pside),
                         client.now_ms, client.now_ms, fills, {},
                         fills_started_at=client.now_ms, fills_at=client.now_ms,
                         prices_at=client.now_ms, fills_after_position=True)
        return capture(client.now_ms, start, client.balance_total, client.now_ms, [p])

    assert client.advance_time()  # open
    assert not scope_boundaries(snapshot(), "unified").boundaries
    assert client.advance_time()  # partial
    assert not scope_boundaries(snapshot(), "unified").boundaries
    assert client.advance_time()  # actual flat
    events = client.get_fill_events(start, client.now_ms)
    estimated = scope_boundaries(snapshot(events[:-1]), "unified")
    assert [b.timestamp for b in estimated.boundaries] == [events[-2]["timestamp"]]
    assert "current_flat_timestamp_estimate" in estimated.reasons
    clean = snapshot()
    trace = scope_boundaries(clean, "unified")
    boundary, = trace.boundaries
    assert boundary.timestamp == client.now_ms
    assert float(boundary.observation.pnl) == pytest.approx(client.realized_pnl - client.realized_fees)
    # Current flatness is authoritative even when the simulator clock ties the
    # fill and position observations; timing uncertainty remains diagnostic.
    assert "current_flat_timestamp_estimate" not in trace.reasons
    assert client.advance_time()
    clean = snapshot()
    trace = scope_boundaries(clean, "unified")
    boundary, = trace.boundaries
    risk = signal([Observation(start, dec(0), dec(0)), boundary.observation],
                  client.balance_total, 1, ".02")
    assert risk.panic[-1]
    evidence = LifecycleEvidence(boundary.timestamp, boundary.timestamp)
    assert permission(client.now_ms, 86_400_000, 120_000, "always", "panic", evidence,
                      exposed=False, red_now=False) == "halted"
    # Repeated observation does not refresh cooldown.
    repeated = scope_boundaries(snapshot(), "unified")
    assert repeated.boundaries == trace.boundaries
    fresh = replace(clean, pairs=tuple(replace(p, fills=tuple(replace(f) for f in p.fills))
                                      for p in clean.pairs))
    assert scope_boundaries(fresh, "unified") == trace
