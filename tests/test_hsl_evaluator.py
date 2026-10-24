"""Offline v9 factual reconstruction, lifecycle and correction regressions."""

from copy import deepcopy
import json

import pytest

from test_hsl_signal import evaluate, native, request

def evaluate_snapshot(snapshot, mode="portfolio", *, slots=1, span=1.0,
                      threshold=0.05, cooldown_ms=120_000, slow_span=43200.0, **selectors):
    """Compose factual dataclass fixtures through the sole v9 runtime API."""
    from test_hsl_snapshot import payload
    import passivbot_rust
    value=dict(snapshot=payload(snapshot,mode,**selectors),slots=slots,drawdown_ratio_ema_span_minutes=span,
               threshold=threshold,cooldown_ms=cooldown_ms,
               memory_ema_span_minutes=slow_span,scale_budget_with_excess_allowance=False,
               exposure_budget=None,budget_observations=[])
    return evaluate(passivbot_rust,value)



def closed_request(*, mode="coin", side="long", now=120_000, duration=120_000,
                   current_balance=91.0):
    value=request(mode=mode,side=side,net=0,final_budget=current_balance,now=now)
    direction=1 if side=="long" else -1
    price=90.0 if side=="long" else 110.0
    pair=value["snapshot"]["pairs"][0]
    pair["position"].update(size=0.0,basis=0.0,mark=price)
    pair["prices"]={"0":100.0,"60000":100.0}
    pair["fills"].append(dict(identity="manual-close",timestamp=120_000,
        delta=-direction,price=price,realized=-10.0,fee=1.0,sequence=1,revision=0))
    value.update(drawdown_ratio_ema_span_minutes=1.0,threshold=0.095,cooldown_ms=duration)
    for fact in value["budget_observations"]:
        fact["balance"]=100.0
    value["budget_observations"][-1].update(order=0,balance=91.0)
    value["budget_observations"].append(dict(timestamp=now,order=2**63-1,
        balance=current_balance,slots=1,scale_budget_with_excess_allowance=False,
        exposure_budget=None,estimated=False))
    return value


@pytest.mark.parametrize("mode",["coin","pside","portfolio"])
@pytest.mark.parametrize("side",["long","short"])
def test_pre_red_post_green_manual_flatten_starts_finite_deadline(native,mode,side):
    value=closed_request(mode=mode,side=side)
    actual=evaluate(native,value)
    assert actual["risk"]["raw"] < value["threshold"]
    assert actual["risk"]["ema"] < value["threshold"]
    assert actual["decision"]["action"] == "halted"
    assert actual["risk"]["cooldown_until"] == 240_000
    assert actual["decision"]["flat_at"] == 120_000
    assert any(e["kind"]=="flat" for e in actual["events"])
    for now,balance in [(180_000,100_000.0),(239_999,1.0),(240_000,91.0)]:
        later=evaluate(native,closed_request(mode=mode,side=side,now=now,
                                            current_balance=balance))
        assert later["decision"]["action"] == ("halted" if now<240_000 else "normal")
        assert later["risk"]["cooldown_until"] == 240_000


def test_zero_duration_and_genuine_reentry_clear_old_terminal_authority(native):
    zero=evaluate(native,closed_request(duration=0))
    assert zero["decision"]["action"] == "normal"
    value=closed_request(now=180_000)
    pair=value["snapshot"]["pairs"][0]
    pair["fills"].append(dict(identity="reentry",timestamp=180_000,
        delta=1,price=90.0,realized=0,fee=0,sequence=2,revision=0))
    pair["position"].update(size=1,basis=90.0,mark=90.0)
    result=evaluate(native,value)
    assert result["risk"]["exposed"]
    assert result["risk"]["cooldown_until"] is None
    assert result["risk"]["drawdown"] == 0
    assert result["decision"]["action"] == "normal"


@pytest.mark.parametrize("side",["long","short"])
def test_unknown_exposed_prefix_has_no_new_fast_warmup(native,side):
    value=request(side=side,slow=43200.0,fast=1_000_000.0,now=120_001)
    value["snapshot"]["start"]=120_000
    pair=value["snapshot"]["pairs"][0]
    pair.update(fills=[],prices={})
    result=evaluate(native,value)
    assert result["decision"]["action"] == "panic"
    assert "unknown_prefix_current_basis_seed" in result["reasons"]
    assert result["risk"]["drawdown"] == 20.0
    assert result["risk"]["ema"] == result["risk"]["raw"]


def test_corrected_facts_rebuild_without_previous_permission_input(native):
    original=closed_request()
    first=evaluate(native,original)
    corrected=deepcopy(original)
    pair=corrected["snapshot"]["pairs"][0]
    pair["fills"][-1].update(price=100.0,realized=0.0,fee=0.0,revision=1)
    corrected["budget_observations"][-2]["balance"]=100.0
    corrected["budget_observations"][-1]["balance"]=100.0
    corrected["snapshot"]["balance"]=100.0
    assert evaluate(native,corrected)["decision"]["action"] == "normal"
    assert evaluate(native,original) == first
    assert first["decision"]["action"] == "halted"


@pytest.mark.parametrize("mode",["coin","pside","portfolio"])
def test_fresh_empty_scope_requires_no_invented_price(native,mode):
    value=request(mode=mode)
    value["snapshot"]["pairs"]=[]
    if mode=="coin":
        value["snapshot"]["flat_coin"]=dict(symbol="A",pside="long",
            position_at=value["snapshot"]["now"],fills_at=value["snapshot"]["now"],
            history_start=0)
    result=evaluate(native,value)
    assert result["decision"]["action"] == "normal"
    assert result["risk"]["exposed"] is False


def test_only_current_mode_and_model_fields_are_runtime_inputs(native):
    for path in ["unified","restart"]:
        value=request()
        if path=="unified":value["snapshot"]["mode"]="unified"
        else:value["restart"]="never"
        with pytest.raises(ValueError):
            evaluate(native,value)
