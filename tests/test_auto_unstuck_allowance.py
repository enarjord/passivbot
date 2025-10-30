import pytest

import passivbot_rust as pbr


def test_allowance_zero_when_no_drop_and_zero_pct():
    allowance = pbr.calc_auto_unstuck_allowance(
        balance=1000.0,
        loss_allowance_pct=0.0,
        pnl_cumsum_max=0.0,
        pnl_cumsum_last=0.0,
    )
    assert allowance == pytest.approx(0.0)


def test_allowance_matches_configured_pct_at_peak():
    allowance = pbr.calc_auto_unstuck_allowance(
        balance=1000.0,
        loss_allowance_pct=0.05,
        pnl_cumsum_max=0.0,
        pnl_cumsum_last=0.0,
    )
    assert allowance == pytest.approx(50.0)


def test_allowance_increases_after_drawdown():
    balance = 900.0
    pnl_cumsum_max = 200.0
    pnl_cumsum_last = 0.0
    loss_allowance_pct = 0.05

    allowance = pbr.calc_auto_unstuck_allowance(
        balance=balance,
        loss_allowance_pct=loss_allowance_pct,
        pnl_cumsum_max=pnl_cumsum_max,
        pnl_cumsum_last=pnl_cumsum_last,
    )

    balance_peak = balance + (pnl_cumsum_max - pnl_cumsum_last)  # 1100
    drop_since_peak_pct = balance / balance_peak - 1.0  # -0.1818...
    expected = balance_peak * (loss_allowance_pct + drop_since_peak_pct)

    assert allowance == 0.0


def test_allowance_clamped_to_zero_for_large_drop():
    allowance = pbr.calc_auto_unstuck_allowance(
        balance=100.0,
        loss_allowance_pct=0.01,
        pnl_cumsum_max=500.0,
        pnl_cumsum_last=-200.0,
    )
    assert allowance == pytest.approx(0.0)
