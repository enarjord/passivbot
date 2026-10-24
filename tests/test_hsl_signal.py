"""Source-verified v9 runtime arithmetic against an independent rational oracle."""

from copy import deepcopy
import json

import pytest

from rust_utils import verify_loaded_runtime_extension
from v9_risk_reference import sample


@pytest.fixture(scope="module")
def native():
    import passivbot_rust

    verify_loaded_runtime_extension()
    return passivbot_rust


def request(*, mode="coin", side="long", slow=3.0, fast=3.0,
            net=-0.1, final_budget=999.9, now=120_001):
    direction = 1 if side == "long" else -1
    mark = 80.0 if side == "long" else 120.0
    return dict(
        snapshot=dict(
            now=now, start=0, balance=final_budget, balance_at=now, config_at=0,
            max_current_age_ms=120_000, mode=mode,
            pside=side if mode != "portfolio" else None,
            symbol="A" if mode == "coin" else None,
            pairs=[dict(
                symbol="A", position=dict(size=direction, basis=100.0,
                    mark=mark, multiplier=1.0, inverse=False, pside=side,
                    quantity_step=0.1),
                position_at=now, mark_at=now, fills_started_at=now, fills_at=now,
                prices_at=now, revisions=[0,0,0,0],
                fills=[dict(identity="entry", timestamp=60_000, delta=direction,
                    price=100.0, realized=0.0, fee=net, sequence=0, revision=0)],
                prices={"0":100.0,"60000":100.0,"120000":mark},
            )],
        ),
        slots=1, scale_budget_with_excess_allowance=False, exposure_budget=None,
        memory_ema_span_minutes=slow, drawdown_ratio_ema_span_minutes=fast, threshold=0.01,
        cooldown_ms=120_000,
        budget_observations=[
            dict(timestamp=0,order=-1,balance=1000.0,slots=1,
                 scale_budget_with_excess_allowance=False,exposure_budget=None,
                 estimated=False),
            dict(timestamp=60_000,order=0,balance=1000.0+net,slots=1,
                 scale_budget_with_excess_allowance=False,exposure_budget=None,
                 estimated=False),
            dict(timestamp=120_000,order=-1,balance=final_budget,slots=1,
                 scale_budget_with_excess_allowance=False,exposure_budget=None,
                 estimated=False),
        ],
    )


def evaluate(native, value):
    return json.loads(native.hsl_v9_evaluate(json.dumps(value,allow_nan=False)))


@pytest.mark.parametrize("mode",["coin","pside","portfolio"])
@pytest.mark.parametrize("side",["long","short"])
@pytest.mark.parametrize("slow,fast",[(3.0,1.0),(43200.0,360.0),(7.5,2.5)])
@pytest.mark.parametrize("net",[-0.1,0.025])
@pytest.mark.parametrize("final_budget",[999.9,2000.0])
def test_actual_native_risk_uses_ordered_f32_graph(
    native, mode, side, slow, fast, net, final_budget
):
    value=request(mode=mode,side=side,slow=slow,fast=fast,net=net,
                  final_budget=final_budget)
    entry=sample(previous_d=0,previous_m=0,previous_upnl=0,upnl=0,net=net,
                 budget=1000.0+net,slow_span=slow,fast_span=fast)
    marked=sample(previous_d=entry["drawdown"],previous_m=entry["ema"],
                  previous_upnl=0,upnl=-20,net=0,budget=final_budget,
                  slow_span=slow,fast_span=fast)
    actual=evaluate(native,value)["risk"]
    for key in ["drawdown","raw","ema","nonpositive_equity"]:
        assert actual[key] == marked[key],(key,actual,marked)


def test_retired_risk_exports_are_absent(native):
    for name in ["hsl_signal","hsl_controller","hsl_evaluate",
                 "hsl_evaluate_grids","hsl_trace","hsl_candle_free"]:
        assert not hasattr(native,name)


@pytest.mark.parametrize("key,value",[("drawdown_ratio_ema_span_minutes",0.5),("memory_ema_span_minutes",1e12),
                                      ("threshold",-1),("cooldown_ms",-1)])
def test_malformed_policy_is_a_producer_error(native,key,value):
    payload=request();payload[key]=value
    with pytest.raises(ValueError):
        evaluate(native,payload)


@pytest.mark.parametrize("alias", ["span", "lookback_span_minutes", "ema_span_minutes"])
def test_retired_policy_alias_is_explicitly_rejected(native, alias):
    payload = request()
    payload[alias] = 3.0
    with pytest.raises(ValueError, match="unknown field"):
        evaluate(native, payload)


def test_disposable_current_preview_does_not_change_the_factual_request(native):
    value=request();saved=deepcopy(value)
    first=evaluate(native,value)
    assert evaluate(native,value) == first
    assert value == saved
    repaired=deepcopy(value)
    repaired["snapshot"]["pairs"][0]["fills"][0]["fee"]=-1.0
    assert evaluate(native,repaired)["risk"]["drawdown"] > first["risk"]["drawdown"]
    assert evaluate(native,value) == first


@pytest.mark.parametrize("values",[
    [-1e16,-1.0,1e16],[1e16,-1e16,-1.0],[-1.0,1e16,-1e16],
])
def test_scope_upnl_is_summed_before_the_binary32_risk_boundary(native,values):
    value=request(mode="portfolio")
    pairs=[]
    for i,pnl in enumerate(values):
        pair=deepcopy(value["snapshot"]["pairs"][0])
        pair.update(symbol=str(i),fills=[],prices={})
        pair["position"].update(size=abs(pnl),basis=2.0,
                                 mark=1.0 if pnl<0 else 3.0)
        pairs.append(pair)
    value["snapshot"].update(pairs=pairs,balance=1000.0)
    value["budget_observations"]=[]
    output=evaluate(native,value)
    assert output["risk"]["drawdown"] == 1.0
    assert output["risk"]["raw"] == pytest.approx(0.001)
