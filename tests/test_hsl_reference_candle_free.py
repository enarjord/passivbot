"""Hand-computed scope cashflow cases with no historical candle dependency."""

from dataclasses import replace
from decimal import Decimal
from itertools import permutations

import pytest

from hsl_reference import Fill, Position, dec, minimal_signal
from hsl_reference_candle_free import estimate_candle_free
from hsl_reference_replay import Settings, capture, capture_pair, estimate_pair


REFERENCE_ESTIMATE = estimate_candle_free


@pytest.fixture(params=["reference", "rust"], autouse=True)
def estimator_backend(request, monkeypatch):
    if request.param == "reference":
        return
    import json
    import sys
    import passivbot_rust as pbr
    from test_hsl_revised_snapshot import payload
    assert hasattr(pbr, "hsl_revised_candle_free"), "rebuild the source-matched extension"

    def compare(snapshot, mode, **kwargs):
        encoded = json.dumps(dict(snapshot=payload(snapshot, mode, **kwargs),
                                  slots=snapshot.settings.slots, span=float(snapshot.settings.ema_span),
                                  threshold=float(snapshot.settings.threshold)), allow_nan=False)
        try:
            expected = REFERENCE_ESTIMATE(snapshot, mode, **kwargs)
        except ValueError:
            with pytest.raises(ValueError):
                pbr.hsl_revised_candle_free(encoded)
            raise
        actual = json.loads(pbr.hsl_revised_candle_free(encoded))
        for field in ("realized", "realized_peak", "upnl"):
            assert actual[field] == pytest.approx(float(getattr(expected, field)))
        assert expected.reasons <= set(actual["reasons"])
        if expected.signal is None:
            assert actual["signal"] is None
        else:
            for field in ("equity", "peaks", "raw", "ema"):
                assert actual["signal"][field] == pytest.approx([float(x) for x in getattr(expected.signal, field)])
            assert actual["signal"]["panic"] == list(expected.signal.panic)
            assert len(actual["signal"]["ema"]) == 1
        return expected
    monkeypatch.setattr(sys.modules[__name__], "estimate_candle_free", compare)


def pair(symbol="TEST", side="long", fills=(), size=1, basis=100, mark=80, **kwargs):
    return capture_pair(symbol, Position(size if side == "long" else -size, basis, mark,
                                         pside=side, **kwargs),
                        300_000, 300_000, fills, {}, fills_started_at=300_000,
                        fills_at=300_000, prices_at=300_000, fills_after_position=True)


def snap(*pairs, span=1, slots=1, start=0):
    return capture(300_000, start, 1000, 300_000, pairs,
                   Settings(ema_span=span, threshold=".05", slots=slots))


def coin(snapshot):
    return estimate_candle_free(snapshot, "coin", pside="long", symbol="TEST")


@pytest.mark.parametrize("span", [1, "2.5", 1_000_000])
@pytest.mark.parametrize("mark", [80, 100, 120])
def test_empty_fills_match_minimal_formula_exactly(span, mark):
    result = coin(snap(pair(mark=mark), span=span))
    assert result.signal == minimal_signal(mark - 100, 1000, span, ".05", 300_000)
    assert result.signal.ema == result.signal.raw


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_known_loss_and_fee_survive_with_no_candles(mode):
    fills = [Fill("open", 60_000, 2, 100, 0, -1),
             Fill("partial", 120_000, -1, 80, -20, -1)]
    snapshot = snap(pair(fills=fills), span=1_000_000)
    args = {} if mode == "unified" else {"pside": "long"}
    if mode == "coin":
        args["symbol"] = "TEST"
    result = estimate_candle_free(snapshot, mode, **args)
    assert (result.realized, result.realized_peak, result.upnl) == (-22, 0, -20)
    assert float(result.signal.raw[-1]) == pytest.approx(42 / 1042)
    assert result.signal.ema == result.signal.raw  # no fictional zero seed


def test_realized_profit_peak_is_not_discarded_or_added_twice():
    fills = [Fill("open", 60_000, 3, 100, 0),
             Fill("profit", 120_000, -1, 200, 100),
             Fill("loss", 180_000, -1, 20, -80)]
    result = coin(snap(pair(fills=fills)))
    assert (result.realized, result.realized_peak, result.upnl) == (20, 100, -20)
    assert float(result.signal.raw[-1]) == pytest.approx(100 / 1100)
    assert result.signal.panic[-1]


def test_current_profit_offsets_known_losses_in_currency():
    p = pair(fills=[Fill("loss", 60_000, -1, 80, -20, -1)], mark=150)
    result = coin(snap(p))
    assert (result.realized, result.upnl) == (-21, 50)
    assert result.signal.raw == (0,)


@pytest.mark.parametrize("delta,price,realized,fee,reason", [
    ("bad", "bad", -75, -2, "invalid_quantity"),
    ("NaN", 100, -75, -2, "invalid_quantity"),
    (-1, "bad", -75, -2, "estimated_fill_price"),
    (-1, 80, None, -2, "estimated_realized_pnl"),
    (-1, 80, -75, "bad", "unknown_fee"),
])
def test_historical_damage_still_evaluates(delta, price, realized, fee, reason):
    p = pair(fills=[Fill("open", 60_000, 2, 100, 0),
                   Fill("close", 120_000, delta, price, realized, fee)])
    result = coin(snap(p))
    assert reason in result.reasons
    expected_realized = (-20 if realized is None else realized) + (fee if isinstance(fee, int) else 0)
    assert result.realized == expected_realized
    assert float(result.signal.raw[-1]) == pytest.approx((20 - expected_realized) / (1020 - expected_realized))


def test_same_symbol_hedged_upnl_and_cross_pair_cashflow_cancel_before_peak():
    a = pair("A", fills=[Fill("profit", 60_000, -1, 100, 100)], mark=80)
    b = pair("A", "short", fills=[Fill("loss", 60_000, 1, 100, -100)], mark=80)
    for order in permutations([a, b]):
        result = estimate_candle_free(snap(*order), "unified")
        assert (result.realized, result.realized_peak, result.upnl) == (0, 0, 0)
        assert result.signal.raw == (0,)
        assert "cohort_cashflow_peak" in result.reasons
    assert estimate_candle_free(snap(a, b), "pside", pside="long").signal.raw[-1] > 0


def test_per_pair_sequences_do_not_manufacture_cross_pair_peak():
    a = pair("A", fills=[Fill("gain", 60_000, -1, 100, 100, sequence=1)], mark=100)
    b = pair("B", fills=[Fill("loss", 60_000, -1, 100, -100, sequence=2)], mark=100)
    assert estimate_candle_free(snap(a, b), "unified").realized_peak == 0


@pytest.mark.parametrize("sequences,peak", [((1, 2), 100), ((None, None), 0), ((1, 1), 0)])
def test_actual_within_pair_sequence_preserves_peak(sequences, peak):
    fills = [Fill("gain", 60_000, -1, 100, 100, sequence=sequences[0]),
             Fill("loss", 60_000, -1, 100, -100, sequence=sequences[1])]
    result = coin(snap(pair(fills=fills, mark=100)))
    assert result.realized_peak == peak
    assert bool(result.signal.raw[-1]) == bool(peak)


def test_clipping_correction_duplicates_and_cache_free_repeat():
    old = Fill("loss", 60_000, -1, 100, -100)
    fixed = replace(old, realized=-10, revision=1)
    result = coin(snap(pair(fills=[old, fixed, fixed])))
    assert result.realized == -10
    assert coin(snap(pair(fills=[fixed]))) == result
    assert coin(snap(pair(fills=[old, fixed]), start=60_001)).realized == 0
    assert coin(snap(pair(fills=[old, fixed]), start=60_000)).realized == -10
    conflicting = replace(fixed, realized=-20)
    conflict = coin(snap(pair(fills=[fixed, conflicting])))
    assert conflict.realized == 0
    assert "conflicting_identity" in conflict.reasons


def test_post_position_and_impossible_future_correction_are_isolated():
    old = Fill("loss", 60_000, -1, 100, -100)
    impossible = replace(old, timestamp=400_000, realized=100, revision=1)
    assert coin(snap(pair(fills=[old, impossible]))).realized == -100
    after_anchor = Fill("future_to_position", 290_000, -1, 100, -50)
    p = replace(pair(fills=[old, after_anchor]), position_at=280_000)
    result = coin(snap(p))
    assert result.realized == -100
    assert "post_position_fill" in result.reasons


def test_coin_slots_and_inactive_scope_and_empty_portfolio():
    p = pair(mark=0.01)
    assert coin(snap(p, slots=0)).signal is None
    assert coin(snap(p, slots=2)).signal.raw[-1] > coin(snap(p)).signal.raw[-1]
    assert estimate_candle_free(snap(p, slots=0), "unified").signal.panic[-1]
    assert estimate_candle_free(snap(), "unified").signal.raw == (0,)


@pytest.mark.parametrize("side,size,basis,mark,multiplier,inverse,upnl", [
    ("long", 2, 100, 90, 3, False, -60),
    ("short", 2, 100, 110, 3, False, -60),
    ("long", 10, 100, 50, 100, True, -10),
    ("short", 10, 50, 100, 100, True, -10),
])
def test_current_contract_units(side, size, basis, mark, multiplier, inverse, upnl):
    p = pair(side=side, size=size, basis=basis, mark=mark, multiplier=multiplier, inverse=inverse)
    result = estimate_candle_free(snap(p), "coin", pside=side, symbol="TEST")
    assert result.upnl == upnl
    assert float(result.signal.raw[-1]) == pytest.approx(-upnl / (1000 - upnl))


def test_available_history_cannot_be_discarded_for_singleton():
    p = replace(pair(mark=0.01), prices=((60_000, dec(100)), (120_000, dec(100))))
    snapshot = snap(p, span=1_000_000)
    with pytest.raises(ValueError, match="cannot discard"):
        coin(snapshot)
    # Retained observations keep their EMA even if a new history request failed.
    rich = estimate_pair(snapshot, p.key)
    assert not rich.signal.panic[-1]
    assert coin(snap(replace(p, prices=()), span=1_000_000)).signal.panic[-1]


def test_unrelated_pair_history_does_not_affect_coin_domain():
    p = pair()
    other = replace(pair("OTHER"), prices=((60_000, Decimal(100)),))
    assert coin(snap(p, other)) == coin(snap(p))


@pytest.mark.parametrize("field", ["balance_at", "position_at", "mark_at"])
def test_fresh_selected_observation_skew_is_visible(field):
    p = pair()
    snapshot = (replace(snap(p), balance_at=299_000) if field == "balance_at" else
                snap(replace(p, **{field: 299_000})))
    result = coin(snapshot)
    assert "snapshot_skew" in result.reasons
    assert result.signal == coin(snap(p)).signal


def test_skew_is_scope_local_and_includes_all_selected_pairs():
    p, other = pair(), replace(pair("OTHER", "short"), mark_at=299_000)
    snapshot = snap(p, other)
    assert "snapshot_skew" not in coin(snapshot).reasons
    assert "snapshot_skew" not in estimate_candle_free(snapshot, "pside", pside="long").reasons
    assert "snapshot_skew" in estimate_candle_free(snapshot, "pside", pside="short").reasons
    assert "snapshot_skew" in estimate_candle_free(snapshot, "unified").reasons


@pytest.mark.fake_live
@pytest.mark.parametrize("pside", ["long", "short"])
def test_fake_exchange_cashflows_without_candles(pside):
    from exchanges.fake import FakeCCXTClient

    symbol = "TEST/USDT:USDT"
    direction = 1 if pside == "long" else -1
    actions = lambda side, qty: [{"type": "manual_fill", "symbol": symbol,
                                  "position_side": pside, "side": side, "qty": qty}]
    entry, close = ("buy", "sell") if direction == 1 else ("sell", "buy")
    client = FakeCCXTClient({
        "name": "hsl_candle_free", "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60, "account": {"balance": 1000},
        "symbols": {symbol: {"qty_step": .1, "price_step": .1, "min_qty": .1,
                             "min_cost": 1, "taker": .001}},
        "timeline": [{"t": 0, "prices": {symbol: 100}},
                     {"t": 1, "prices": {symbol: 100}, "actions": actions(entry, 2)},
                     {"t": 2, "prices": {symbol: 80 if direction == 1 else 120},
                      "actions": actions(close, 1)}],
    })
    start = client.now_ms
    assert client.advance_time() and client.advance_time()
    fills = [Fill(e["id"], e["timestamp"], e["qty"] * (1 if e["side"] == "buy" else -1),
                  e["price"], e["pnl"], -e["fees"]["cost"])
             for e in client.get_fill_events(start, client.now_ms)]
    state = client.positions[(symbol, pside)]
    p = capture_pair(symbol, Position(direction * state["size"], state["entry_price"],
                                      80 if direction == 1 else 120, pside=pside),
                     client.now_ms, client.now_ms, fills, {}, prices_at=client.now_ms)
    snapshot = capture(client.now_ms, start, client.balance_total, client.now_ms, [p])
    result = estimate_candle_free(snapshot, "unified")
    net = client.realized_pnl - client.realized_fees
    assert float(result.realized) == pytest.approx(net)
    assert result.upnl == -20
    assert float(result.signal.raw[-1]) == pytest.approx((20 - net) / (client.balance_total + 20 - net))
    assert estimate_candle_free(replace(snapshot, pairs=(replace(p),)), "unified") == result
