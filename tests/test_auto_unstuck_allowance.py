"""The public v9 allowance boundary follows the independent binary32 oracle."""

import pytest

import passivbot_rust as pbr
from v9_risk_reference import allowance as reference_allowance, f32, mul

pytestmark = pytest.mark.skipif(
    getattr(pbr, "__is_stub__", False), reason="requires the native v9 risk API"
)


def test_rolling_allowance_export_is_retired():
    assert not hasattr(pbr, "calc_auto_unstuck_allowance")


def test_allowance_zero_when_no_drawdown_and_zero_pct():
    assert pbr.calc_auto_unstuck_allowance_v9(
        balance=1000.0, loss_pct=0.0, twel=1.0, drawdown=0.0
    ) == 0.0


def test_allowance_matches_configured_pct_without_remembered_loss():
    assert pbr.calc_auto_unstuck_allowance_v9(
        balance=1000.0, loss_pct=0.05, twel=1.0, drawdown=0.0
    ) == 50.0


def test_allowance_debits_loss_and_replenishes_with_decaying_memory():
    before_loss = pbr.calc_auto_unstuck_allowance_v9(1000.0, 0.05, 1.0, 0.0)
    after_loss = pbr.calc_auto_unstuck_allowance_v9(990.0, 0.05, 1.0, 10.0)
    after_decay = pbr.calc_auto_unstuck_allowance_v9(990.0, 0.05, 1.0, 5.0)
    assert (before_loss, after_loss, after_decay) == (50.0, 40.0, 44.75)
    assert after_loss < after_decay < before_loss


def test_allowance_clamped_to_zero_for_large_remembered_loss():
    assert pbr.calc_auto_unstuck_allowance_v9(
        balance=100.0, loss_pct=0.01, twel=1.0, drawdown=700.0
    ) == 0.0


@pytest.mark.parametrize("balance,loss_pct,twel,drawdown", [
    (12345.67, 0.01234567, 1.23456789, 45.6789),
    (1000.0, 0.05, 0.5, 10.0),
    (1000.0, 0.05, 2.0, 10.0),
    (1000.0, 0.05, 0.0, 10.0),
])
def test_allowance_uses_separately_rounded_percentage_and_twel(
    balance, loss_pct, twel, drawdown
):
    effective_pct = mul(f32(loss_pct), f32(twel))
    expected = reference_allowance(balance, effective_pct, f32(drawdown))
    assert pbr.calc_auto_unstuck_allowance_v9(balance, loss_pct, twel, drawdown) == expected


@pytest.mark.parametrize("field,value", [
    ("balance", 0.0), ("balance", -1.0), ("balance", float("nan")),
    ("balance", float("inf")), ("balance", 1e40),
    ("loss_pct", -0.1), ("loss_pct", float("nan")), ("loss_pct", 1e40),
    ("twel", -1.0), ("twel", float("inf")), ("twel", 1e40),
    ("drawdown", -1.0), ("drawdown", float("nan")), ("drawdown", 1e40),
])
def test_allowance_rejects_malformed_or_out_of_domain_inputs(field, value):
    inputs = dict(balance=1000.0, loss_pct=0.05, twel=1.0, drawdown=10.0)
    inputs[field] = value
    with pytest.raises(ValueError):
        pbr.calc_auto_unstuck_allowance_v9(**inputs)
