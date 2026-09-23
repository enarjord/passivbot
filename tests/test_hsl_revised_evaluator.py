"""Full snapshot-to-decision composition; deterministic and offline."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hsl_reference import Fill, dec
from hsl_reference_candle_free import estimate_candle_free
from hsl_reference_controller import replay
from test_hsl_revised_snapshot import payload
from test_hsl_revised_trace import oracle
import test_hsl_reference_replay as cases

M = cases.M


def evaluate(snapshot, mode="unified", *, slots=1, span=1, threshold=.05,
             cooldown_ms=2*M, restart="always", **selectors):
    import passivbot_rust as pbr
    request = dict(snapshot=payload(snapshot, mode, quantity_step=.1, **selectors),
                   slots=slots, span=span, threshold=threshold, cooldown_ms=cooldown_ms,
                   restart=restart)
    return json.loads(pbr.hsl_revised_evaluate(json.dumps(request, allow_nan=False)))


@pytest.mark.parametrize("mode,selectors", [("coin", {"pside":"long", "symbol":"A"}),
                                            ("pside", {"pside":"long"}), ("unified", {})])
@pytest.mark.parametrize("span", [1, 2.5, 1000000])
def test_all_candle_absent_matches_independent_cashflow_estimate(mode, selectors, span):
    p = cases.pair(size=1, basis=100, mark=80, fills=[
        Fill("open", M, 3, 100, 0, -1),
        Fill("profit", 2*M, -1, 200, 100, -1),
        Fill("loss", 3*M, -1, 20, -80, -1)])
    snapshot = cases.frame(p, balance=1000)
    snapshot = replace(snapshot, settings=replace(snapshot.settings, ema_span=dec(span), threshold=dec(.05)))
    ref = estimate_candle_free(snapshot, mode, **selectors)
    actual = evaluate(snapshot, mode, span=span, **selectors)
    assert actual["observations"] == 1
    assert actual["decision"]["raw"] == pytest.approx(float(ref.signal.raw[-1]))
    assert actual["decision"]["ema"] == pytest.approx(float(ref.signal.ema[-1]))
    assert actual["decision"]["action"] == "panic"


def test_candle_free_known_flat_resets_realized_peak_and_retains_cooldown():
    p = cases.pair(size=1, basis=100, mark=100, fills=[
        Fill("open", M, 2, 100, 0),
        Fill("flat", 2*M, -2, 50, -100, sequence=1),
        Fill("reopen", 3*M, 1, 100, 0, sequence=2)])
    snapshot = cases.frame(p, balance=1000)
    panic = evaluate(snapshot, restart="never")
    assert panic["episodes"] == 2
    assert panic["decision"]["action"] == "normal"
    assert panic["decision"]["red_at"] is None
    normal = evaluate(snapshot, restart="never")
    assert normal["decision"]["action"] == "normal"
    assert normal["decision"]["raw"] == 0  # prior episode loss not retained


def test_candle_free_current_profit_can_offset_realized_loss():
    p = cases.pair(size=1, basis=100, mark=200,
                   fills=[Fill("partial", M, -1, 80, -20, -1)])
    result = evaluate(cases.frame(p, balance=1000), span=1e6)
    assert result["observations"] == 1
    assert result["decision"]["raw"] == 0
    assert result["decision"]["action"] == "normal"


def test_cross_pair_cashflow_cohort_does_not_invent_global_profit_peak():
    a = cases.pair("A", size=1, basis=100, mark=100, fills=[Fill("gain", M, -1, 100, 100, sequence=1)])
    b = cases.pair("B", size=1, basis=100, mark=100, fills=[Fill("loss", M, -1, 100, -100, sequence=2)])
    for pairs in [(a,b), (b,a)]:
        result = evaluate(cases.frame(*pairs, balance=1000))
        assert result["decision"]["raw"] == 0
        assert "cohort_cashflow_peak" in result["reasons"]


@pytest.mark.parametrize("missing", [True, False])
def test_mixed_price_grid_keeps_other_pair_history_and_matches_oracle(missing):
    a = cases.pair("A", size=10, basis=100, mark=100,
                   fills=[Fill("entry", 0, 10, 100, 0)], prices={0:100, M:120, 2*M:80, 3*M:100})
    b = cases.pair("B", size=1, basis=100, mark=90,
                   prices={} if missing else {2*M:90})
    snapshot = cases.frame(a,b, balance=1000)
    # Independent oracle uses the explicit agreed approximation: source history
    # is ffilled/bfilled, wholly absent pair history uses its current mark.
    aligned = replace(snapshot, pairs=(
        replace(a, prices=tuple(sorted({**dict(a.prices),4*M:100}.items()))),
        replace(b, prices=tuple((t, dec(90)) for t in range(0,5*M,M)))))
    ref = replay(oracle(aligned, "unified"), now=snapshot.now, start=0,
                 budget=1000, span=2.5, threshold=.05, cooldown=2*M)
    result = evaluate(snapshot, span=2.5)
    assert result["decision"]["raw"] == pytest.approx(float(ref[-1].raw))
    assert result["decision"]["ema"] == pytest.approx(float(ref[-1].ema))
    assert result["decision"]["action"] == ref[-1].action == "panic"
    assert result["observations"] == 5
    assert ("current_mark_history_estimate" in result["reasons"]) == missing


def test_missing_all_fills_and_candles_is_one_sample_for_any_span():
    snapshot = cases.frame(cases.pair(size=10,basis=100,mark=90),balance=1000)
    for span in [1,1e6]:
        result = evaluate(snapshot,span=span)
        assert result["observations"] == 1
        assert result["decision"]["raw"] == pytest.approx(100/1000)
        assert result["decision"]["ema"] == result["decision"]["raw"]
        assert result == evaluate(deepcopy(snapshot),span=span)


def test_inactive_coin_validates_current_inputs_without_dividing_by_zero():
    snapshot = cases.frame(cases.pair(size=1,basis=100,mark=90),balance=1000)
    result = evaluate(snapshot,"coin",pside="long",symbol="A",slots=0)
    assert result["decision"] is None
    with pytest.raises(ValueError,match="minimum"):
        evaluate(replace(snapshot,balance=dec(0)),"coin",pside="long",symbol="A",slots=0)


def test_wrong_scope_current_inputs_do_not_block_selected_scope():
    a = cases.pair("A",size=10,basis=100,mark=90)
    b = cases.pair("B",size=-1,basis=100,mark=100,pside="short")
    snapshot=cases.frame(a,b,balance=1000)
    snapshot=replace(snapshot,pairs=(a,replace(b,mark_at=0)))
    assert evaluate(snapshot,"coin",pside="long",symbol="A")["decision"]["action"] == "panic"
    with pytest.raises(ValueError,match="position/mark"):
        evaluate(snapshot)


def test_expired_candle_free_stop_is_not_retained_locally():
    p = cases.pair(size=0,basis=0,mark=50,fills=[
        Fill("open",M,2,100,0),Fill("flat",2*M,-2,50,-100)])
    snapshot=cases.frame(p,balance=1000)
    assert evaluate(snapshot,restart="never")["decision"]["action"] == "halted"
    expired=replace(snapshot,start=2*M+1)
    result=evaluate(expired,restart="never")
    assert result["decision"]["action"] == "normal"
    assert result["decision"]["red_at"] is None
    assert result["observations"] == 1


def test_coin_budget_uses_slots_but_aggregate_uses_raw_balance():
    snapshot=cases.frame(cases.pair(size=10,basis=100,mark=90),balance=1000)
    coin=evaluate(snapshot,"coin",pside="long",symbol="A",slots=10)
    assert coin["decision"]["raw"] == 1.0
    for mode,args in [("unified",{}),("pside",{"pside":"long"})]:
        result=evaluate(snapshot,mode,slots=10,**args)
        assert result["decision"]["raw"] == pytest.approx(100/1000)


def test_global_sequence_cashflow_peak_requires_explicit_producer_contract():
    import passivbot_rust as pbr
    a=cases.pair("A",size=1,basis=100,mark=100,fills=[Fill("gain",M,-1,100,100,sequence=1)])
    b=cases.pair("B",size=1,basis=100,mark=100,fills=[Fill("loss",M,-1,100,-100,sequence=2)])
    request=dict(snapshot=payload(cases.frame(a,b,balance=1000)),slots=1,span=10000,
                 threshold=.05,cooldown_ms=0,restart="always")
    assert json.loads(pbr.hsl_revised_evaluate(json.dumps(request)))["decision"]["raw"] == 0
    request["snapshot"]["global_fill_sequence"]=True
    result=json.loads(pbr.hsl_revised_evaluate(json.dumps(request)))
    assert result["decision"]["raw"] == pytest.approx(100/1100)
    assert result["decision"]["ema"] == result["decision"]["raw"]


@pytest.mark.parametrize("seed",range(12))
def test_damaged_mixed_history_evaluates_and_is_permutation_invariant(seed):
    import random
    rng=random.Random(seed)
    pairs=[]
    for index in range(3):
        side="short" if index==1 else "long"
        sign=-1 if side=="short" else 1
        fills=[Fill(f"f{index}",M,sign*rng.choice([1,2]),rng.choice([None,100]),
                    rng.choice([None,-10,0]),rng.choice([None,-1]))]
        if rng.choice([True,False]):
            fills += [Fill(f"c{index}",2*M,-sign,rng.choice([None,80,120]),-20,-1)]
        prices={} if rng.choice([True,False]) else {rng.choice([0,M,2*M]):rng.choice([80,100,120])}
        pairs.append(cases.pair(str(index),sign,100,90 if sign==1 else 110,
                                fills=fills,prices=prices,pside=side))
    snapshot=cases.frame(*pairs,balance=1000)
    actual=evaluate(snapshot,span=2.5)
    assert actual["decision"]["timestamp"] == snapshot.now
    assert actual["decision"]["action"] in ("normal","panic","halted")
    shuffled=replace(snapshot,pairs=tuple(reversed(snapshot.pairs)))
    assert evaluate(shuffled,span=2.5) == actual


def test_invalid_interval_rejected_even_for_inactive_scope():
    snapshot=cases.frame(cases.pair(size=1,basis=100,mark=90),balance=1000)
    with pytest.raises(ValueError,match="interval"):
        evaluate(replace(snapshot,start=snapshot.now-91*86400000),"coin",pside="long",symbol="A",slots=0)


@pytest.mark.fake_live
@pytest.mark.parametrize("pside",["long","short"])
def test_fake_exchange_reconstruction_reaches_shared_evaluator(pside,monkeypatch):
    import test_hsl_revised_trace as trace_cases
    original=trace_cases.decisions

    def compare(snapshot,mode="unified",**kwargs):
        reference=original(snapshot,mode,**kwargs)
        actual=evaluate(snapshot,mode,**kwargs)["decision"]
        for key in ("timestamp","action","red_at","flat_at"):
            assert actual[key] == reference[-1][key]
        for key in ("raw","ema"):
            assert actual[key] == pytest.approx(reference[-1][key])
        return reference

    monkeypatch.setattr(trace_cases,"decisions",compare)
    trace_cases.test_fake_exchange_missing_opening_with_candles_rebuilds_without_cache(pside)


@pytest.mark.parametrize("mode,selectors", [("coin", {"pside": "long", "symbol": "A"}),
                                            ("pside", {"pside": "long"}), ("unified", {})])
def test_diagnostic_events_keep_zero_cooldown_flat_stop_and_expire_with_window(mode, selectors):
    pair = cases.pair(size=0, basis=0, mark=50, fills=[
        Fill("open", M, 2, 100, 0), Fill("flat", 2*M, -2, 50, -100)])
    snapshot = cases.frame(pair, balance=1000)
    result = evaluate(snapshot, mode, cooldown_ms=0, **selectors)
    assert result["decision"]["action"] == "normal"
    assert [(e["kind"], e["timestamp"]) for e in result["events"]] == [
        ("red", 2*M), ("flat", 2*M), ("restart", 2*M)]
    assert result["events"][0]["raw"] > .05
    assert result["events"][-1]["raw"] is None
    # A diagnostic report is reconstructed, never a persisted stop commitment.
    expired = evaluate(replace(snapshot, start=2*M+1), mode, cooldown_ms=0, **selectors)
    assert expired["events"] == []
    assert expired["decision"]["action"] == "normal"


def test_reopening_diagnostic_uses_actual_open_time_without_inventing_a_risk_sample():
    pair = cases.pair(size=1, basis=100, mark=100, fills=[
        Fill("open", M, 2, 100, 0), Fill("flat", 2*M, -2, 50, -100),
        Fill("reopen", 3*M, 1, 100, 0)])
    result = evaluate(cases.frame(pair, balance=1000), restart="never")
    event = result["events"][-1]
    assert (event["timestamp"], event["kind"], event["reason"]) == (3*M, "restart", "exposure_resumed")
    assert event["raw"] is None and event["ema"] is None
