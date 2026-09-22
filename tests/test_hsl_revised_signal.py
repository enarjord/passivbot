"""Real-extension parity for the isolated revised numerical kernel."""

import json
import math
import random

import pytest

from hsl_reference import Observation, dec, minimal_signal, signal


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    assert not getattr(pbr, "__is_stub__", False)
    assert hasattr(pbr, "hsl_revised_signal"), "rebuild the source-matched Rust extension"
    return pbr


def compare(pbr, rows, budget, span, threshold, entry_reference=None):
    oracle = signal([Observation(t, dec(p), dec(u)) for t, p, u in rows],
                    budget, span, threshold, entry_reference=entry_reference)
    actual = json.loads(pbr.hsl_revised_signal(rows, budget, span, threshold, entry_reference))
    for name in ("equity", "peaks", "raw", "ema"):
        assert actual[name] == pytest.approx([float(x) for x in getattr(oracle, name)],
                                            rel=2e-12, abs=2e-13)
    assert actual["panic"] == list(oracle.panic)
    assert not actual["numeric_range_approximation"]
    return actual


@pytest.mark.parametrize("span", [1.0, 2.5, 3.0, 1e6])
def test_clean_recovery_boundaries_and_fractional_ema(require_real_passivbot_rust_module, span):
    rows = [(0, 0, 0), (60_000, 0, -100), (60_001, -100, -100),
            (120_000, -200, 0), (120_001, -200, 50), (180_000, -200, 200)]
    compare(require_real_passivbot_rust_module, rows, 1000, span, .12)


@pytest.mark.parametrize("threshold,expected", [(.124999, True), (.125, False), (.125001, False)])
def test_strict_threshold_boundary(require_real_passivbot_rust_module, threshold, expected):
    actual = compare(require_real_passivbot_rust_module,
                     [(0, 0, 0), (60_000, 0, -100), (120_000, 0, -200)], 1000, 3, threshold)
    assert actual["panic"][-1] is expected


@pytest.mark.parametrize("upnl", [-100, 0, 100])
@pytest.mark.parametrize("span", [1, 2.5, 1e6])
def test_singleton_entry_reference(require_real_passivbot_rust_module, upnl, span):
    expected = minimal_signal(upnl, 1000, span, .09)
    actual = compare(require_real_passivbot_rust_module, [(0, 0, upnl)],
                     1000, span, .09, 1000)
    assert actual["raw"] == actual["ema"]
    assert actual["panic"] == list(expected.panic)


def test_nonpositive_peak_and_common_currency_offset(require_real_passivbot_rust_module):
    compare(require_real_passivbot_rust_module,
            [(0, 0, -200), (60_000, 0, -150), (120_000, 0, 0)], 100, 3, .1)
    compare(require_real_passivbot_rust_module,
            [(0, 1e60, 0), (60_000, 1e60, -10)], 100, 1, .05)


@pytest.mark.parametrize("seed", range(40))
def test_deterministic_currency_paths_match_decimal_oracle(require_real_passivbot_rust_module, seed):
    rng = random.Random(seed)
    rows, timestamp, realized = [], -120_000, 0
    for _ in range(100):
        timestamp += rng.choice([0, 1, 20_000, 60_000, 180_000])
        realized += rng.randrange(-50, 51)
        rows.append((timestamp, realized, rng.randrange(-500, 501)))
    compare(require_real_passivbot_rust_module, rows, rng.choice([250, 1000, 10000]),
            rng.choice([1, 2.5, 30.5, 10000]), .075)


@pytest.mark.parametrize("rows,budget,span,threshold,reference", [
    ([], 1, 1, .1, None),
    ([(0, math.nan, 0)], 1, 1, .1, None),
    ([(0, 0, math.inf)], 1, 1, .1, None),
    ([(0, 0, 0)], 0, 1, .1, None),
    ([(0, 0, 0)], math.inf, 1, .1, None),
    ([(0, 0, 0)], 1, .5, .1, None),
    ([(0, 0, 0)], 1, 1, -1, None),
    ([(1, 0, 0), (0, 0, 0)], 1, 1, .1, None),
    ([(0, 0, 0)], 1, 1, .1, math.nan),
])
def test_invalid_inputs_are_errors(require_real_passivbot_rust_module, rows, budget, span, threshold, reference):
    with pytest.raises(ValueError, match="invalid revised HSL"):
        require_real_passivbot_rust_module.hsl_revised_signal(rows, budget, span, threshold, reference)


def test_extreme_finite_inputs_are_observably_bounded(require_real_passivbot_rust_module):
    big = float.fromhex("0x1.fffffffffffffp+1023")
    result = json.loads(require_real_passivbot_rust_module.hsl_revised_signal(
        [(0, big, 0), (60_000, -big, 0), (120_000, -big / 2, 0)], 1, 2.5, .1))
    assert result["numeric_range_approximation"]
    assert all(math.isfinite(v) for key in ("equity", "peaks", "raw", "ema") for v in result[key])
    assert result["equity"][-1] == 1
    assert result["panic"][-1]


def test_batch_poll_repeat_and_window_reset_have_no_hidden_state(require_real_passivbot_rust_module):
    pbr = require_real_passivbot_rust_module
    rows = [(0, 0, 200), (60_000, 0, -100), (120_000, 0, 0)]
    first = pbr.hsl_revised_signal(rows, 1000, 1, .1)
    compare(pbr, rows[1:], 1000, 1, .1)
    assert pbr.hsl_revised_signal(rows, 1000, 1, .1) == first


@pytest.mark.parametrize("variant", ["both_deltas_overflow", "one_delta_overflows"])
def test_cancelling_oversized_deltas_do_not_erase_representable_loss(require_real_passivbot_rust_module, variant):
    big = float.fromhex("0x1.fffffffffffffp+1023")
    rows, expected = ([(0, big, -big / 2), (60_000, -big, big)], big) if variant == "both_deltas_overflow" else (
        [(0, big, -big), (60_000, -big, 0)], big)
    result = json.loads(require_real_passivbot_rust_module.hsl_revised_signal(
        rows, 1, 1, .1))
    assert result["numeric_range_approximation"]
    assert result["equity"][0] == pytest.approx(expected)
    assert result["panic"][-1]


@pytest.mark.parametrize("budget,loss", [(1e16, 1.), (1e300, 1e280), (1e-200, 1e-220)])
@pytest.mark.parametrize("span", [1., 2.5, 10000.])
def test_multi_point_signal_retains_losses_below_budget_precision(budget, loss, span):
    from decimal import Decimal, localcontext
    import json
    import passivbot_rust as pbr
    rows = [(i*60_000, -i*loss, 0.) for i in range(4)]
    result = json.loads(pbr.hsl_revised_signal(rows, budget, span, 0.0))
    with localcontext() as context:
        context.prec = 100
        peak = Decimal.from_float(budget) + Decimal.from_float(3*loss)
        alpha = Decimal(2) / (Decimal.from_float(span)+1)
        expected_ema = Decimal(0)
        for i, (raw, ema) in enumerate(zip(result["raw"], result["ema"])):
            expected_raw = Decimal.from_float(i*loss) / peak
            expected_ema = alpha*expected_raw+(1-alpha)*expected_ema
            assert raw == pytest.approx(float(expected_raw), rel=1e-14, abs=0)
            assert ema == pytest.approx(float(expected_ema), rel=1e-14, abs=0)
    assert result["panic"] == [False, True, True, True]
