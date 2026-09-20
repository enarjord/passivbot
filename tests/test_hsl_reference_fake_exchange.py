"""Use the offline fake-live exchange's actual ledger as oracle input.

The revised HSL is not installed into a bot. These tests validate reconstruction
against independently maintained fake account positions, realized PnL and fees.
"""

from dataclasses import replace

import pytest

from hsl_reference import Fill, Position, dec, reconstruct, signal


@pytest.mark.fake_live
@pytest.mark.parametrize("pside", ["long", "short"])
def test_fake_ledger_damage_repair_and_fresh_reconstruction(pside):
    from exchanges.fake import FakeCCXTClient

    symbol = "TEST/USDT:USDT"
    sign = 1 if pside == "long" else -1
    prices = [100, 100, 80, 70, 60] if sign == 1 else [100, 100, 120, 130, 140]
    entry_side, close_side = ("buy", "sell") if sign == 1 else ("sell", "buy")

    def action(side, qty):
        return {"type": "manual_fill", "symbol": symbol, "position_side": pside,
                "side": side, "qty": qty, "reduce_only": side == close_side}

    client = FakeCCXTClient({
        "name": "hsl_reference_damage", "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60, "account": {"balance": 1000},
        "symbols": {symbol: {"qty_step": .1, "price_step": .1, "min_qty": .1,
                             "min_cost": 1, "taker": .001}},
        "timeline": [{"t": i, "prices": {symbol: price},
                      "actions": [action(entry_side, 1)] if i in (1, 2) else
                                 [action(close_side, .5)] if i == 3 else []}
                     for i, price in enumerate(prices)],
    })
    start = client.now_ms
    grid = {start: prices[0]}
    while client.advance_time():
        grid[client.now_ms] = prices[client.current_index]
    state = client.positions[(symbol, pside)]
    position = Position(sign * state["size"], state["entry_price"], prices[-1], pside=pside)
    events = client.get_fill_events(start, client.now_ms)
    fills = [Fill(e["id"], e["timestamp"], e["qty"] * (1 if e["side"] == "buy" else -1),
                  e["price"], e["pnl"], -e["fees"]["cost"], sequence=int(e["id"]))
             for e in events]
    clean = reconstruct(position, fills, grid, start, client.now_ms)
    assert clean.sizes[-1] == dec(sign * state["size"])
    assert clean.bases[-1] == dec(state["entry_price"])
    assert float(clean.rows[-1].pnl) == pytest.approx(client.realized_pnl - client.realized_fees)
    assert float(clean.rows[-1].pnl) == pytest.approx(client.balance_total - 1000)
    assert clean.rows[-1].upnl == -45
    expected = signal(clean.rows, client.balance_total, 1, ".05")
    assert expected.panic[-1]
    for index in range(len(fills)):
        delayed = fills[:index] + fills[index + 1:]
        damaged = reconstruct(position, delayed, grid, start, client.now_ms)
        assert damaged.rows[-1].upnl == clean.rows[-1].upnl
        assert all(x.is_finite() for x in signal(damaged.rows, client.balance_total, 1, ".05").raw)
        # Delivery restores the entire expected series, not just a ready flag.
        repaired = reconstruct(position, [*delayed, fills[index]], grid, start, client.now_ms)
        assert signal(repaired.rows, client.balance_total, 1, ".05") == expected
    fresh = reconstruct(replace(position), [replace(f) for f in fills], dict(grid), start, client.now_ms)
    assert fresh == clean
