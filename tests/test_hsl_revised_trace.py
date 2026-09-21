"""Scoped numerical histories compose into the independent lifecycle controller."""
from dataclasses import replace
import json
from itertools import permutations

import pytest

from hsl_reference import Fill, Observation, dec, reconstruct
from hsl_reference_controller import Episode, Point, replay
from hsl_reference_replay import scope_boundaries, selected_pairs, _steps
import test_hsl_reference_replay as cases
from test_hsl_revised_snapshot import payload

M = cases.M


def rust_trace(request):
    import passivbot_rust as pbr
    return json.loads(pbr.hsl_revised_trace(json.dumps(request, allow_nan=False)))


def oracle(snapshot, mode, **selectors):
    selected = selected_pairs(snapshot, mode, **selectors)
    histories = [reconstruct(p.position, p.fills, dict(p.prices), snapshot.start, snapshot.now)
                 for p in selected]
    if not histories:
        return [Episode((Point(Observation(snapshot.now, dec(0), dec(0)), False),))]
    current_pnl = sum((h.rows[-1].pnl for h in histories), dec(0))
    boundaries = scope_boundaries(snapshot, mode, **selectors)
    flat_rows = [Point(Observation(b.timestamp, b.observation.pnl-current_pnl, dec(0)), False, True)
                 for b in boundaries.boundaries if b.lifecycle_eligible]
    sample_rows = []
    for index in range(len(histories[0].rows)):
        rows = [h.rows[index] for h in histories]
        assert len({r.timestamp for r in rows}) == 1
        sample_rows.append(Point(Observation(rows[0].timestamp,
                                            sum((r.pnl for r in rows), dec(0))-current_pnl,
                                            sum((r.upnl for r in rows), dec(0))),
                                 any(h.sizes[index] != 0 for h in histories)))
    # Stable order preserves multiple independently supported flats in one timestamp.
    timeline = sorted(flat_rows+sample_rows,
                      key=lambda p: (p.observation.timestamp, not p.flatten))
    episodes, points = [], []
    for point in timeline:
        points.append(point)
        if point.flatten:
            episodes.append(Episode(tuple(points)))
            points = [Point(point.observation, False)]
    if points:
        episodes.append(Episode(tuple(points)))
    # Derive lifecycle metadata independently from the exact fill-prefix sets
    # at consecutive supported flats, not from sampled position sizes.
    flats = [b for b in boundaries.boundaries if b.lifecycle_eligible]
    for index in range(1, len(episodes)):
        preceding = dict(flats[index-1].consumed)
        following = dict(flats[index].consumed) if index < len(flats) else None
        openings = []
        for pair in selected:
            steps, _, _ = _steps(pair, snapshot.start, snapshot.now)
            for step in steps:
                if step.fill in preceding[pair.key]:
                    continue
                if following is not None and step.fill not in following[pair.key]:
                    continue
                if step.clean_tail and step.before == 0 and step.after > 0:
                    openings.append(step.fill.timestamp)
        episodes[index] = replace(episodes[index], opened_at=min(openings, default=None))
    if any("estimated_opening_basis" in h.reasons for h in histories):
        reference_delta = -sum((h.rows[-1].pnl + h.rows[-1].upnl for h in histories), dec(0))
        episodes[0] = replace(episodes[0], entry_reference_delta=reference_delta)
    return episodes


def compare(snapshot, mode="unified", **selectors):
    actual = rust_trace(payload(snapshot, mode, quantity_step=.1, **selectors))
    expected = oracle(snapshot, mode, **selectors)
    assert len(actual["episodes"]) == len(expected)
    for a, e in zip(actual["episodes"], expected):
        assert a["entry_reference"] is None
        if e.entry_reference_delta is None:
            assert a["entry_reference_delta"] is None
        else:
            assert a["entry_reference_delta"] == pytest.approx(float(e.entry_reference_delta))
        assert a["opened_at"] == e.opened_at
        assert len(a["points"]) == len(e.points)
        for point, ref in zip(a["points"], e.points):
            assert point["timestamp"] == ref.observation.timestamp
            assert point["exposed"] == ref.exposed
            assert point["flatten"] == ref.flatten
            assert point["pnl"] == pytest.approx(float(ref.observation.pnl))
            assert point["upnl"] == pytest.approx(float(ref.observation.upnl))
    return actual, expected


def decisions(snapshot, mode="unified", *, intervention="panic", restart="always", threshold=.1, **selectors):
    actual, expected = compare(snapshot, mode, **selectors)
    settings = dict(now=snapshot.now, start=snapshot.start, budget=float(snapshot.balance),
                    span=1.0, threshold=threshold, cooldown=2*M, restart=restart, intervention=intervention)
    reference = replay(expected, **settings)
    settings["cooldown_ms"] = settings.pop("cooldown")
    import passivbot_rust as pbr
    results = json.loads(pbr.hsl_revised_controller(json.dumps(dict(episodes=actual["episodes"], **settings))))
    assert len(results) == len(reference)
    for result, ref in zip(results, reference):
        for key in ("timestamp", "action", "flat_at", "red_at", "reason"):
            assert result[key] == getattr(ref, key)
        assert result["raw"] == pytest.approx(float(ref.raw))
        assert result["ema"] == pytest.approx(float(ref.ema))
    return results


def closed(*, pside="long", reopened=False):
    direction = 1 if pside == "long" else -1
    price = 80 if direction == 1 else 120
    fills = [Fill("open", M, direction*2, 100, 0, -1),
             Fill("partial", 2*M, -direction, price, -20, -1),
             Fill("flat", 3*M, -direction, price, -20, -1, sequence=10)]
    if reopened:
        fills.append(Fill("reopen", 3*M, direction, price, 0, -1, sequence=11))
    return cases.pair(pside=pside, size=direction if reopened else 0,
                      basis=price if reopened else 0, mark=price, fills=fills,
                      prices={0:100, M:100, 2*M:price, 3*M:price})


@pytest.mark.parametrize("mode,selectors", [("unified", {}), ("pside", {"pside":"long"}),
                                            ("coin", {"pside":"long", "symbol":"A"})])
def test_scope_flatten_is_final_risk_sample_then_cooldown(mode, selectors):
    results = decisions(cases.frame(closed()), mode, **selectors)
    assert results[-1]["action"] == "halted"
    assert results[-1]["flat_at"] == 3*M


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize("intervention", ["panic", "normal"])
@pytest.mark.parametrize("restart", ["always", "never"])
def test_same_timestamp_reopen_is_after_supported_flat(pside, intervention, restart):
    snapshot = cases.frame(closed(pside=pside, reopened=True))
    results = decisions(snapshot, intervention=intervention, restart=restart)
    assert results[-1]["action"] == ("panic" if intervention == "panic" else "normal")
    at_flat = [r for r in results if r["timestamp"] == 3*M]
    assert at_flat[0]["reason"] == "stop_flattened"
    assert any(r["reason"] == f"{intervention}_intervention" for r in at_flat[1:])


def test_multi_flat_same_timestamp_preserves_cashflow_prefixes():
    fills = [Fill("open", M, 1, 100, 0),
             Fill("flat1", 2*M, -1, 80, -20, sequence=10),
             Fill("open2", 2*M, 1, 80, 0, -1, sequence=11),
             Fill("flat2", 2*M, -1, 70, -10, sequence=12)]
    p = cases.pair(fills=fills, prices={0:100, M:100, 2*M:70, 3*M:70})
    trace, _ = compare(cases.frame(p))
    assert len(trace["episodes"]) == 3
    flats = [e["points"][-1] for e in trace["episodes"][:-1]]
    assert [p["pnl"] for p in flats] == [11., 0.]


def test_unordered_round_trip_does_not_invent_episode_reset():
    p = closed(reopened=True)
    p = replace(p, fills=tuple(replace(f, sequence=None) for f in p.fills))
    # Rebind the causal fixture anchor to the same unchanged current position.
    trace, _ = compare(cases.frame(p))
    assert len(trace["episodes"]) == 1
    assert not any(point["flatten"] for point in trace["episodes"][0]["points"])


def test_opposite_positions_do_not_net_to_flat_and_currency_is_summed_first():
    prices = {0:100, M:100, 2*M:80, 3*M:70}
    a = cases.pair("A", size=1, basis=100, mark=70,
                   fills=[Fill("open", M, 1, 100, 0)], prices=prices)
    b = cases.pair("B", pside="short", size=-1, basis=100, mark=70,
                   fills=[Fill("open", M, -1, 100, 0)], prices=prices)
    trace, _ = compare(cases.frame(a,b))
    assert len(trace["episodes"]) == 1
    endpoint = trace["episodes"][-1]["points"][-1]
    assert endpoint["exposed"] and endpoint["upnl"] == 0
    assert decisions(cases.frame(a,b))[-1]["action"] == "normal"


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_scope_pair_order_does_not_change_trace(order):
    pairs = [closed(), replace(closed(pside="short"), symbol="B"),
             cases.pair("C", prices={0:100, M:100, 2*M:100, 3*M:100})]
    trace, _ = compare(cases.frame(*(pairs[i] for i in order)))
    assert trace == compare(cases.frame(*pairs))[0]


def test_missing_flat_fill_keeps_risk_but_no_cooldown_certificate():
    p = closed()
    p = replace(p, fills=p.fills[:-1])
    trace, _ = compare(cases.frame(p))
    assert not any(x["flatten"] for e in trace["episodes"] for x in e["points"])


def test_prices_must_be_normalized_without_silently_discarding_other_pairs_history():
    a = closed()
    b = cases.pair("B")
    with pytest.raises(ValueError, match="aligned prepared price grids"):
        rust_trace(payload(cases.frame(a,b)))


def test_empty_portfolio_has_one_current_flat_point():
    trace, _ = compare(cases.frame())
    assert trace["episodes"][0]["points"] == [dict(timestamp=4*M,pnl=0.,upnl=0.,exposed=False,flatten=False)]


def test_currency_centering_retains_small_fee_after_large_common_realized_prefix():
    p = cases.pair(fills=[Fill("open", M, 2, 100, 0),
                         Fill("profit", 2*M, -1, 100, 1e16),
                         Fill("fee", 3*M, -1, 100, 0, -1)],
                   prices={0:100,M:100,2*M:100,3*M:100})
    trace, _ = compare(cases.frame(p))
    before_fee = next(x for e in trace["episodes"] for x in e["points"] if x["timestamp"]==2*M)
    assert before_fee["pnl"] == 1.
    assert trace["episodes"][-1]["points"][-1]["pnl"] == 0.


def test_normal_intervention_still_panics_on_large_reopening_fee():
    p = closed(reopened=True)
    p = replace(p, fills=(*p.fills[:-1], replace(p.fills[-1], fee=dec(-25))))
    results = decisions(cases.frame(p), intervention="normal")
    assert results[-1]["action"] == "panic"
    assert results[-1]["red_at"] == 3*M
    assert results[-1]["raw"] == pytest.approx(.2)


@pytest.mark.parametrize("seed", range(30))
def test_generated_clean_partial_close_and_reopen_traces(seed):
    import random
    rng = random.Random(seed)
    side = "long" if seed % 2 else "short"
    d = 1 if side == "long" else -1
    now = 12*M
    prices = {i*M: rng.randint(65,135) for i in range(12)}
    prices[M] = 100
    fills = [Fill("open", M, 3*d, 100, 0, dec("-.1"))]
    for i in (3,5,7):
        fills.append(Fill(f"close{i}", i*M, -d, prices[i*M], d*(prices[i*M]-100), dec("-.1")))
    fills.append(Fill("reopen", 9*M, d, prices[9*M], 0, dec("-.1")))
    p = cases.pair(size=d, basis=prices[9*M], mark=prices[11*M], pside=side,
                   fills=fills, prices=prices, now=now)
    for intervention in ("panic", "normal"):
        decisions(cases.frame(p, now=now), intervention=intervention)


@pytest.mark.parametrize("damage", ["missing_open", "missing_partial", "unknown_quantity", "missing_cashflow"])
def test_historical_damage_stays_numerical_and_repair_rebuilds_trace(damage):
    clean = closed(reopened=True)
    fills = list(clean.fills)
    if damage == "missing_open":
        del fills[0]
    elif damage == "missing_partial":
        del fills[1]
    elif damage == "unknown_quantity":
        fills[1] = replace(fills[1], delta=None)
    else:
        fills[1] = replace(fills[1], realized=None)
    damaged = replace(clean, fills=tuple(fills))
    trace, _ = compare(cases.frame(damaged))
    assert trace["reasons"]
    assert trace["episodes"][-1]["points"][-1]["timestamp"] == 4*M
    # No prior decision argument: delivering the missing facts rebuilds cleanly.
    assert compare(cases.frame(clean))[0] == rust_trace(payload(cases.frame(clean), quantity_step=.1))


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize("intervention", ["normal", "panic"])
@pytest.mark.parametrize("restart", ["always", "never"])
@pytest.mark.parametrize("tied", [False, True])
def test_round_trip_between_candles_retains_exchange_intervention(pside, intervention, restart, tied):
    base = closed(pside=pside)
    direction = 1 if pside == "long" else -1
    price = 80 if direction == 1 else 120
    opened = 3*M if tied else 3*M+10
    flattened = 3*M if tied else 3*M+20
    fills = (*base.fills,
             Fill("reopen", opened, direction, price, 0, 0, sequence=11),
             Fill("reflat", flattened, -direction, price, 0, 0, sequence=12))
    p = replace(base, fills=fills)
    snapshot = cases.frame(p)
    actual, expected = compare(snapshot)
    assert actual["episodes"][1]["opened_at"] == opened
    assert not any(p["exposed"] for p in actual["episodes"][1]["points"])
    results = decisions(snapshot, intervention=intervention, restart=restart)
    assert results[-1]["action"] == ("normal" if intervention == "normal" else "halted")
    assert results[-1]["red_at"] == (None if intervention == "normal" else opened)
    assert results[-1]["flat_at"] == (None if intervention == "normal" else flattened)


@pytest.mark.parametrize("opening", [3*M+10, 5*M, 5*M+10])
@pytest.mark.parametrize("intervention", ["normal", "panic"])
def test_reopen_time_precedes_later_sample_cooldown_expiry(opening, intervention):
    base = closed()
    close_at = 5*M+20
    p = cases.pair(fills=(*base.fills,
                         Fill("reopen", opening, 1, 80, 0),
                         Fill("reflat", close_at, -1, 80, 0)),
                   prices={0:100, M:100, 2*M:80, 3*M:80}, now=6*M)
    results = decisions(cases.frame(p, now=6*M), intervention=intervention)
    panicked = intervention == "panic" and opening < 5*M
    assert results[-1]["action"] == ("halted" if panicked else "normal")
    assert results[-1]["red_at"] == (opening if panicked else None)


@pytest.mark.parametrize("side,quantity,entry", [("long", 1, 80), ("short", -1, 120)])
def test_explicit_simulator_phase_prevents_pre_entry_profit(side, quantity, entry):
    p = cases.pair(size=quantity, basis=entry, mark=entry, pside=side,
                   fills=[Fill("entry", M, quantity, entry, 0, 0, sequence=0)],
                   prices={M: 100, 2*M: entry, 3*M: entry})
    request = payload(cases.frame(p), "coin", pside=side, symbol="A")
    request["fills_before_same_time_price"] = False
    actual = rust_trace(request)
    points = actual["episodes"][0]["points"]
    assert points[0]["timestamp"] == M
    assert points[0]["exposed"] is False
    assert all(p["upnl"] == 0 for p in points)
    # Independent hand calculation: position first appears at its entry price,
    # there are no fees or subsequent price changes, hence no loss or RED.
    import passivbot_rust as pbr
    result = json.loads(pbr.hsl_revised_controller(json.dumps(dict(
        now=4*M, start=0, budget=100, span=1, threshold=.01,
        cooldown_ms=M, restart="always", intervention="panic",
        episodes=actual["episodes"]))))
    assert all(r["action"] == "normal" and r["raw"] == 0 for r in result)


def test_global_order_is_explicit_and_preserves_aggregate_flat_prefix():
    a = cases.pair(size=1, basis=90, mark=90,
        fills=[Fill("a0", M, 1, 100, 0, 0, sequence=0),
               Fill("a1", 2*M, -1, 90, -10, 0, sequence=2),
               Fill("a2", 2*M, 1, 90, 0, 0, sequence=4)], prices={M:100, 2*M:90, 3*M:90})
    b = cases.pair(symbol="B", fills=[
        Fill("b0", M, 1, 100, 0, 0, sequence=1),
        Fill("b1", 2*M, -1, 80, -20, 0, sequence=3)], prices={M:100, 2*M:80, 3*M:80})
    request = payload(cases.frame(a,b))
    assert len(rust_trace(request)["episodes"]) == 1
    request["global_fill_sequence"] = True
    actual = rust_trace(request)
    assert len(actual["episodes"]) == 2
    end = actual["episodes"][0]["points"][-1]
    assert end["flatten"] and end["timestamp"] == 2*M
    assert actual["episodes"][1]["opened_at"] == 2*M
    # Duplicate cross-pair sequence values cannot establish total order.
    request["pairs"][1]["fills"][1]["sequence"] = 2
    assert len(rust_trace(request)["episodes"]) == 1


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize("mode,selectors", [("coin", {"symbol": "A"}), ("pside", {}), ("unified", {})])
def test_missing_opening_with_flat_candles_seeds_entry_peak(pside, mode, selectors):
    direction = 1 if pside == "long" else -1
    price = 90 if direction == 1 else 110
    p = cases.pair(size=direction*10, basis=100, mark=price, pside=pside,
                   prices={t: price for t in range(0, 5*M, M)})
    selectors = dict(selectors, **({"pside": pside} if mode != "unified" else {}))
    snapshot = cases.frame(p, balance=1000)
    actual, expected = compare(snapshot, mode, **selectors)
    assert actual["episodes"][0]["entry_reference_delta"] == 100
    assert "estimated_entry_peak" in actual["reasons"]
    assert len(actual["episodes"][0]["points"]) == 5  # no synthetic EMA row
    settings = dict(now=snapshot.now, start=0, budget=1000, span=10_000.5,
                    threshold=.08, cooldown=0, restart="always", intervention="panic")
    reference = replay(expected, **settings)
    settings["cooldown_ms"] = settings.pop("cooldown")
    import passivbot_rust as pbr
    result = json.loads(pbr.hsl_revised_controller(json.dumps(dict(episodes=actual["episodes"], **settings))))
    assert len(result) == len(reference)
    for a, e in zip(result, reference):
        assert a["raw"] == pytest.approx(100/1100)
        assert a["ema"] == pytest.approx(float(e.ema))
        assert a["action"] == e.action == "panic"


def test_opening_evidence_replaces_estimate_instead_of_latching_red():
    price = 90
    p = cases.pair(size=10, basis=100, mark=price, prices={t: price for t in range(0, 5*M, M)})
    missing = rust_trace(payload(cases.frame(p, balance=1000), "unified"))
    assert missing["episodes"][0]["entry_reference_delta"] == 100
    # A late opening establishes an initially flat scope; ordinary candles/fees
    # now define the peak. The old approximation has no persistent authority.
    complete = cases.after_position(replace(p, fills=(Fill("open", M, 10, 100, 0, 0),)))
    observed, _ = compare(cases.frame(complete, balance=1000))
    assert observed["episodes"][0]["entry_reference_delta"] is None
    assert "estimated_entry_peak" not in observed["reasons"]


def test_aggregate_entry_reference_nets_positions_before_peak():
    losing = cases.pair("A", 10, 100, 90, prices={0:90, M:90, 2*M:90, 3*M:90})
    winning = cases.pair("B", 10, 100, 110, prices={0:110, M:110, 2*M:110, 3*M:110})
    snapshot = cases.frame(losing, winning, balance=1000)
    trace, _ = compare(snapshot)
    assert trace["episodes"][0]["entry_reference_delta"] == 0
    assert all(d["raw"] == 0 and d["action"] == "normal" for d in decisions(snapshot))


def test_partial_close_cashflow_is_counted_once_and_reference_resets_at_flat():
    p = cases.pair(size=0, basis=0, mark=90,
                   fills=[Fill("partial", M, -5, 90, -50, -1),
                          Fill("flat", 2*M, -5, 90, -50, -1)],
                   prices={0:90, M:90, 2*M:90, 3*M:90})
    snapshot = cases.frame(p, balance=1000)
    trace, _ = compare(snapshot)
    assert trace["episodes"][0]["entry_reference_delta"] == 102
    assert all(e["entry_reference_delta"] is None for e in trace["episodes"][1:])
    # Drawdown includes realized losses plus fees; their earlier basis estimate
    # is uncertain but neither cashflow is dropped or counted a second time.
    results = decisions(snapshot, restart="never", threshold=.08)
    at_flat = next(r for r in results if r["reason"] == "stop_flattened")
    assert at_flat["flat_at"] == 2*M
    assert results[-1]["action"] == "halted"
    assert max(r["raw"] for r in results) == pytest.approx(102/1102)
    assert results[-1]["raw"] == 0


def test_window_clipping_drops_old_cashflows_before_new_entry_reference():
    p = cases.pair(size=10, basis=100, mark=90,
                   fills=[Fill("expired-close", -M, -5, 80, -500, -2)],
                   prices={0:90, M:90, 2*M:90, 3*M:90})
    trace, _ = compare(cases.frame(p, balance=1000))
    assert trace["episodes"][0]["entry_reference_delta"] == 100


@pytest.mark.fake_live
@pytest.mark.parametrize("pside", ["long", "short"])
def test_fake_exchange_missing_opening_with_candles_rebuilds_without_cache(pside):
    from exchanges.fake import FakeCCXTClient
    symbol = "TEST/USDT:USDT"
    direction = 1 if pside == "long" else -1
    price = 90 if direction == 1 else 110
    client = FakeCCXTClient({
        "name": "hsl_entry_reference", "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60, "account": {"balance": 1000},
        "symbols": {symbol: {"qty_step": .1, "price_step": .1, "min_qty": .1,
                             "min_cost": 1, "taker": .001}},
        "timeline": [
            {"t": 0, "prices": {symbol: 100}},
            {"t": 1, "prices": {symbol: 100}, "actions": [{"type": "manual_fill",
             "symbol": symbol, "position_side": pside, "side": "buy" if direction == 1 else "sell",
             "qty": 10, "reduce_only": False}]},
            {"t": 2, "prices": {symbol: price}},
            {"t": 3, "prices": {symbol: price}},
        ],
    })
    start = client.now_ms
    assert client.advance_time()
    assert client.advance_time()
    assert client.advance_time()
    state = client.positions[(symbol, pside)]
    # Simulate a lost/missing opening fill and a retained late candle backfilled
    # across the window. Neither fault alters current exchange position truth.
    p = cases.pair(symbol, direction*state["size"], state["entry_price"], price,
                   pside=pside, now=client.now_ms,
                   prices={t:price for t in range(start, client.now_ms+1, M)})
    frame = cases.frame(p, start=start, now=client.now_ms, balance=client.balance_total)
    result = decisions(frame, "coin", pside=pside, symbol=symbol, threshold=.08)
    assert result[-1]["action"] == "panic"
    assert result[-1]["raw"] == pytest.approx(100/(client.balance_total+100))
    copied = replace(frame, pairs=tuple(replace(pair) for pair in frame.pairs))
    assert decisions(copied, "coin", pside=pside, symbol=symbol, threshold=.08) == result
