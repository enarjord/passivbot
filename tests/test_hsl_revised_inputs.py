"""Factual manager transport: no I/O, policy, ledger writes or hidden fill proof."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from fill_events_manager import FillEvent, PNL_CONTRACT_CURRENT
from live.hsl_revised_inputs import capture_candles, capture_fills

SYMBOL = "TEST/USDT:USDT"


def event(**changes):
    base = FillEvent(
        id="900", timestamp=60_000, datetime="", symbol=SYMBOL,
        side="buy", qty=2.0, price=100.0, pnl=0.0, fee_paid=-.2,
        pnl_status="complete", fees=None, pb_order_type="entry_grid_normal",
        position_side="long", client_order_id="", c_mult=10.0,
        fee_quality="exact", pnl_contract=PNL_CONTRACT_CURRENT,
    )
    return replace(base, **changes)


def pair(events, multiplier=10.0):
    return capture_fills(events, {SYMBOL: multiplier}).pairs[0]


def history(fills, *, size=2.0, basis=100.0, mark=90.0, pside="long"):
    import passivbot_rust as pbr
    request = dict(start=0, end=120_000,
                   position=dict(size=size, basis=basis, mark=mark, multiplier=10.0,
                                 inverse=False, pside=pside),
                   fills=fills, prices={"60000": 100.0})
    return json.loads(pbr.hsl_revised_history(json.dumps(request, allow_nan=False)))


def test_canonical_fill_contract_units_signed_fees_and_no_derived_positions():
    source = event(psize=999, pprice=-99, raw=[{"tradeId": "1", "sequence": 2}])
    captured = pair([source])
    assert captured.reasons == ()
    fill, = captured.payload()
    assert fill == dict(identity="900", timestamp=60_000, delta=2., price=100.,
                        realized=0., fee=-.2, sequence=None, revision=0)
    result = history(captured.payload())
    assert result["samples"][-1]["upnl"] == -200.
    assert result["samples"][-1]["pnl"] == -.2
    assert result["samples"][-1]["size"] == 2.
    assert source.psize == 999
    assert source.raw == [{"tradeId": "1", "sequence": 2}]
    with pytest.raises(FrozenInstanceError):
        captured.fills[0].price = 50.
    fill["price"] = 1.
    assert captured.payload()[0]["price"] == 100.


@pytest.mark.parametrize("side,qty,pside", [("buy", 2., "long"), ("sell", -2., "short"),
                                           ("sell", -1., "long"), ("buy", 1., "short")])
def test_direction_is_order_side_not_position_side(side, qty, pside):
    row = pair([event(side=side, qty=qty, position_side=pside)]).payload()[0]
    assert row["delta"] == qty


@pytest.mark.parametrize("quantity,side", [(2., "sell"), (-2., "buy"), (float("nan"), "buy"),
                                          (True, "buy"), (2., "unknown")])
def test_bad_quantity_does_not_erase_cashflows(quantity, side):
    captured = pair([event(qty=quantity, side=side, pnl=-30., fee_paid=2.)])
    row = captured.payload()[0]
    assert row["delta"] is None
    assert (row["realized"], row["fee"]) == (-30., 2.)
    assert "invalid_fill_quantity" in captured.reasons
    assert history(captured.payload())["samples"][-1]["pnl"] == -28.


@pytest.mark.parametrize("field,value", [("c_mult", 1.), ("c_mult", 0.), ("c_mult", None)])
def test_unknown_contract_quantity_is_not_silently_converted(field, value):
    captured = pair([event(**{field: value}, pnl=-30.)])
    assert captured.payload()[0]["delta"] is None
    assert captured.payload()[0]["realized"] == -30.
    assert "fill_contract_units_unavailable" in captured.reasons


def test_missing_current_metadata_does_not_certify_contract_quantity():
    captured = capture_fills([event()], {}).pairs[0]
    assert captured.payload()[0]["delta"] is None


@pytest.mark.parametrize("changes", [{"pnl_status": "pending"}, {"pnl_source": "pending"}])
def test_pending_placeholder_is_not_authoritative_zero(changes):
    captured = pair([event(**changes)])
    row = captured.payload()[0]
    assert row["realized"] is None and row["fee"] == -.2
    assert "pending_realized_pnl" in captured.reasons


def test_usable_estimated_pnl_retained_with_diagnostic():
    captured = pair([event(pnl=-30., pnl_source="synthetic_fill_reconstruction_degraded")])
    assert captured.payload()[0]["realized"] == -30.
    assert "estimated_realized_pnl" in captured.reasons
    assert history(captured.payload())["samples"][-1]["pnl"] == -30.2


@pytest.mark.parametrize("contract", ["", None, "gross_pnl_signed_fee_paid_v1"])
def test_incompatible_accounting_contract_is_not_guessed(contract):
    captured = pair([event(pnl_contract=contract, pnl=-30.)])
    row = captured.payload()[0]
    assert row["delta"] == 2.
    assert row["realized"] is row["fee"] is None
    assert "fill_accounting_contract_unavailable" in captured.reasons


@pytest.mark.parametrize("fee,quality,expected", [(0., "exact", False), (1., "exact", False),
                                                  (-.1, "converted", False),
                                                  (-.2, "fallback", True)])
def test_fee_sign_and_quality_preserved(fee, quality, expected):
    captured = pair([event(fee_paid=fee, fee_quality=quality)])
    assert captured.payload()[0]["fee"] == fee
    assert ("estimated_fill_fee" in captured.reasons) is expected


def test_correction_replaces_batch_without_inheriting_old_snapshot():
    source = [event()]
    old = capture_fills(source, {SYMBOL: 10.})
    source[0] = event(price=105., pnl=-10.)
    new = capture_fills(source, {SYMBOL: 10.})
    assert old.pairs[0].payload()[0]["price"] == 100.
    assert new.pairs[0].payload()[0]["price"] == 105.
    assert len(new.pairs[0].fills) == 1
    # Conflicting or duplicate rows remain for Rust, not Python risk policy.
    conflict = pair([event(), event(price=105.)])
    assert len(conflict.fills) == 2
    assert "conflicting_identity" in history(conflict.payload())["reasons"]


def test_same_time_cross_pair_and_ids_do_not_create_ordering_proof():
    events = [event(id="9"), event(id="10", symbol="OTHER/USDT:USDT", side="sell",
                                  qty=-2., position_side="short"), event(id="11")]
    tape = capture_fills(events, {SYMBOL: 10., "OTHER/USDT:USDT": 10.})
    assert len(tape.pairs) == 2
    assert all(fill.sequence is None for p in tape.pairs for fill in p.fills)


@pytest.mark.parametrize("changes", [{"timestamp": 1.5}, {"timestamp": float("inf")},
                                      {"timestamp": -1}, {"timestamp": 2**63}, {"id": ""}])
def test_undated_rows_disclosed_without_poisoning_other_history(changes):
    captured = pair([event(**changes), event(id="valid", timestamp=120_000)])
    assert len(captured.fills) == 1
    assert "unidentified_or_undated_fill" in captured.reasons


def test_unknown_pair_disclosed_without_assigning_it_to_a_position():
    tape = capture_fills([event(position_side="both"), event(symbol=""), event()], {SYMBOL: 10.})
    assert tape.reasons == ("unattributed_fill",)
    assert len(tape.pairs) == 1 and len(tape.pairs[0].fills) == 1


def test_candles_copy_manager_array_without_projection_or_mutation():
    from candlestick_manager import CANDLE_DTYPE
    rows = np.zeros(1, dtype=CANDLE_DTYPE)
    for key, value in dict(ts=0, o=100., h=120., l=80., c=90.).items():
        rows[key] = value
    tape = capture_candles(rows, minutes=15, observed_at=900_000)
    rows["c"] = 999.
    assert tape.payload()[0] == dict(start=0, minutes=15, open=100., high=120., low=80.,
                                     close=90., available_at=900_000)
    import passivbot_rust as pbr
    result = json.loads(pbr.hsl_revised_prices(json.dumps(dict(start=0, end=900_000,
                                                              candles=tape.payload()))))
    assert len(result["rows"]) == 16
    assert result["rows"][-1]["close"] == 90.
    assert "coarse_candle" in result["reasons"]


def test_bad_historical_candle_component_is_null_and_rust_keeps_usable_close():
    rows = [dict(ts=0, o=None, h=float("nan"), l=float("inf"), c=90.)]
    tape = capture_candles(rows, minutes=1, observed_at=60_000)
    assert tape.reasons == ("missing_candle_component",)
    import passivbot_rust as pbr
    result = json.loads(pbr.hsl_revised_prices(json.dumps(dict(start=0, end=60_000,
                                                              candles=tape.payload()), allow_nan=False)))
    assert result["rows"][-1]["close"] == 90.


@pytest.mark.parametrize("minutes,observed_at", [(2, 0), (True, 0), (1, -1), (1, 1.2)])
def test_capture_metadata_is_not_historical_approximation(minutes, observed_at):
    with pytest.raises(ValueError):
        capture_candles([], minutes=minutes, observed_at=observed_at)


@pytest.mark.fake_live
@pytest.mark.asyncio
@pytest.mark.parametrize("pside", ["long", "short"])
async def test_real_manager_fake_exchange_to_rust_and_cache_free_rebuild(tmp_path, pside):
    from exchanges.fake import FakeCCXTClient
    from fill_events_manager import FillEventsManager, FakeFetcher
    direction = 1 if pside == "long" else -1
    entry_side, close_side = ("buy", "sell") if direction == 1 else ("sell", "buy")
    prices = [100., 100., 80., 70.] if direction == 1 else [100., 100., 120., 130.]
    def action(side, quantity):
        return dict(type="manual_fill", symbol=SYMBOL, position_side=pside,
                    side=side, qty=quantity, reduce_only=side == close_side)
    client = FakeCCXTClient(dict(
        name="revised_input_transport", start_time="2026-01-01T00:00:00Z",
        tick_interval_seconds=60, account=dict(balance=1000.),
        symbols={SYMBOL: dict(qty_step=.1, price_step=.1, min_qty=.1, min_cost=1,
                              taker=.001, contract_size=1.)},
        timeline=[dict(t=i, prices={SYMBOL: price},
                       actions=[action(entry_side, 2)] if i == 1 else
                               [action(close_side, 1)] if i == 2 else [])
                  for i, price in enumerate(prices)],
    ))
    start = client.now_ms
    for _ in range(3):
        assert client.advance_time()
    async def rebuild(folder):
        manager = FillEventsManager(exchange="fake", user="test", fetcher=FakeFetcher(client),
                                    cache_path=tmp_path / folder)
        await manager.refresh(start_ms=start, end_ms=client.now_ms, mark_refreshed=False)
        source = manager.get_events()
        before = [fill.to_dict() for fill in source]
        captured = capture_fills(source, {SYMBOL: client.markets[SYMBOL]["contractSize"]})
        assert [fill.to_dict() for fill in source] == before
        return captured
    continuous = await rebuild("first")
    fresh = await rebuild("fresh")
    assert fresh == continuous
    captured, = continuous.pairs
    assert [fill.delta for fill in captured.fills] == [direction * 2, -direction]
    assert all(fill.sequence is None for fill in captured.fills)
    import passivbot_rust as pbr
    position = client.positions[(SYMBOL, pside)]
    request = dict(start=start, end=client.now_ms,
                   position=dict(size=position["size"] * direction, basis=position["entry_price"],
                                 mark=prices[-1], multiplier=client.markets[SYMBOL]["contractSize"],
                                 inverse=False, pside=pside),
                   fills=captured.payload(), prices={str(start+60_000):100.})
    result = json.loads(pbr.hsl_revised_history(json.dumps(request, allow_nan=False)))
    endpoint = result["samples"][-1]
    assert endpoint["size"] == direction
    assert endpoint["upnl"] == -30.
    assert endpoint["pnl"] == pytest.approx(client.realized_pnl - client.realized_fees)
