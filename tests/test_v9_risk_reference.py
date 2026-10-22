"""Hand-calculated public fixtures for the finalized risk contract."""

from fractions import Fraction
import pytest

from v9_risk_reference import allowance, coefficients, f32, fma, sample


def test_binary32_oracle_rounds_ties_to_even_and_fuses_once():
    assert f32(Fraction(1) + Fraction(1, 2**24)) == 1.0
    assert f32(Fraction(1) + Fraction(3, 2**24)) == 1.0 + 2**-22
    assert fma(1 + 2**-23, 1 - 2**-23, -1) == -2**-46
    assert coefficients(3.0) == (0.5, 0.5)
    assert coefficients(1.0) == (1.0, 0.0)


def test_entry_fee_is_consumed_after_zero_initialization():
    row = sample(previous_d=0, previous_m=0, previous_upnl=0,
                 upnl=0, net=-2, budget=98, slow_span=3, fast_span=3)
    assert row["drawdown"] == 2
    assert row["raw"] == f32(Fraction(2, 100))
    assert row["ema"] == f32(Fraction(1, 100))


def test_ratio_smoothing_keeps_previous_contemporaneous_normalization():
    entry_m = f32(Fraction(1, 100))
    loss = sample(previous_d=2, previous_m=entry_m, previous_upnl=0,
                  upnl=-10, net=0, budget=98, slow_span=3, fast_span=3)
    assert loss["drawdown"] == 11  # 0.5 * 2 + 10
    assert loss["raw"] == f32(Fraction(11, 99))
    changed = sample(previous_d=11, previous_m=entry_m, previous_upnl=-10,
                     upnl=-10, net=0, budget=196, slow_span=3, fast_span=3,
                     first_in_minute=False)
    assert changed["drawdown"] == 11  # budget change is no trading cashflow
    assert changed["raw"] == f32(Fraction(11, 197))
    assert changed["ema"] == fma(0.5, entry_m, f32(changed["raw"] / 2))
    assert changed["ema"] != f32(loss["ema"] * 99 / 197)


def test_final_fill_can_be_green_after_causal_pre_fill_red():
    before = sample(previous_d=0, previous_m=0, previous_upnl=0,
                    upnl=-10, net=0, budget=100, slow_span=3, fast_span=1)
    after = sample(previous_d=10, previous_m=0, previous_upnl=-10,
                   upnl=0, net=-9, budget=91, slow_span=3, fast_span=1,
                   first_in_minute=False)
    threshold = f32(0.095)
    assert min(before["raw"], before["ema"]) > threshold
    assert min(after["raw"], after["ema"]) < threshold
    # This fixture requires a finite cooldown despite final GREEN. Runtime
    # controller tests must assert deadline/expiry against these same economics.
    assert after["drawdown"] == 9


def test_ordered_clips_preserve_intraminute_profit_then_loss_peak():
    profit = sample(previous_d=0, previous_m=0, previous_upnl=0,
                    upnl=0, net=10, budget=110, slow_span=3, fast_span=1,
                    first_in_minute=False)
    loss = sample(previous_d=profit["drawdown"], previous_m=0, previous_upnl=0,
                  upnl=0, net=-15, budget=95, slow_span=3, fast_span=1,
                  first_in_minute=False)
    netted = sample(previous_d=0, previous_m=0, previous_upnl=0,
                    upnl=0, net=-5, budget=95, slow_span=3, fast_span=1,
                    first_in_minute=False)
    assert loss["drawdown"] == 15
    assert netted["drawdown"] == 5


@pytest.mark.parametrize("budget,upnl", [(100, -100), (100, -110), (0, 0), (-1, 0)])
def test_nonpositive_equity_is_explicit_loss_branch(budget, upnl):
    row = sample(previous_d=0, previous_m=0, previous_upnl=0,
                 upnl=upnl, net=0, budget=budget, slow_span=3, fast_span=360)
    assert row["nonpositive_equity"]
    assert row["raw"] == 1


def test_realized_allowance_debits_same_loss_from_balance_and_memory():
    # B falls from 1000 to 990 and D rises by 10. Therefore B+D remains 1000.
    assert allowance(1000, 0.05, 0) == 50
    assert allowance(990, 0.05, 10) == 40
    # Decay replenishes memory, but does not restore actual balance.
    assert allowance(990, 0.05, 5) == 44.75
