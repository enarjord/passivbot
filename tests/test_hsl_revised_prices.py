"""Independent price-projection parity and degradation cases for revised HSL."""

from dataclasses import asdict, replace
from itertools import permutations
import json

import pytest

from hsl_reference import Candle, MINUTE as M, minute_prices


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    assert not getattr(pbr, "__is_stub__", False)
    assert hasattr(pbr, "hsl_revised_prices"), "rebuild the source-matched Rust extension"
    return pbr


def run(pbr, candles, start, end):
    return json.loads(pbr.hsl_revised_prices(json.dumps(
        {"candles": [asdict(c) for c in candles], "start": start, "end": end})))


def compare(pbr, candles, start, end):
    expected = minute_prices(candles, start, end)
    actual = run(pbr, candles, start, end)
    assert [r["timestamp"] for r in actual["rows"]] == list(expected)
    assert [r["close"] for r in actual["rows"]] == pytest.approx([float(v) for v in expected.values()])
    return actual


@pytest.mark.parametrize("minutes", [5, 15, 60])
@pytest.mark.parametrize("open,close", [(100, 110), (110, 100), (100, 100)])
def test_coarse_zigzag_and_provenance(require_real_passivbot_rust_module, minutes, open, close):
    c = Candle(0, minutes, open, 120, 90, close)
    result = compare(require_real_passivbot_rust_module, [c], 0, minutes * M)
    assert result["rows"][0]["close"] == open
    assert result["rows"][-1]["close"] == close
    assert all(r["source_end"] == minutes * M for r in result["rows"])
    assert all(r["resolution_minutes"] == minutes for r in result["rows"])
    assert "coarse_candle" in result["reasons"]


def test_finest_source_internal_gaps_leading_bfill_and_trailing_ffill(require_real_passivbot_rust_module):
    candles = [Candle(0, 15, 100, 120, 80, 110),
               Candle(5 * M, 5, 100, 110, 90, 105),
               Candle(6 * M, 1, None, None, None, 123),
               Candle(20 * M, 1, None, None, None, 115)]
    for order in (candles, list(reversed(candles))):
        actual = compare(require_real_passivbot_rust_module, order, 0, 25 * M)
        by_time = {r["timestamp"]: r for r in actual["rows"]}
        assert by_time[7 * M]["close"] == 123
        assert by_time[7 * M]["resolution_minutes"] == 1
        assert by_time[0]["carried"]
        assert by_time[25 * M]["close"] == 115
        assert {"backfilled_price", "forward_filled_price"} <= set(actual["reasons"])


def test_complete_coarse_and_source_availability_are_causal(require_real_passivbot_rust_module):
    c = Candle(0, 15, 100, 120, 80, 110)
    for end in (0, 14 * M, 15 * M):
        compare(require_real_passivbot_rust_module, [c], 0, end)
    assert not run(require_real_passivbot_rust_module, [c], 0, 14 * M)["rows"]
    assert not run(require_real_passivbot_rust_module, [replace(c, available_at=16 * M)], 0, 15 * M)["rows"]
    # No out-of-window seed; a real 1m close itself can be at the left edge.
    assert not run(require_real_passivbot_rust_module, [c], M, 15 * M)["rows"]
    compare(require_real_passivbot_rust_module, [Candle(0, 1, None, None, None, 100)], M, 2 * M)


@pytest.mark.parametrize("candle", [
    Candle(0, 1, None, None, None, None),
    Candle(0, 1, 100, 100, 100, 0),
    Candle(0, 5, 100, 90, 80, 110),
    Candle(0, 5, None, 120, 80, 110),
])
def test_bad_historical_rows_do_not_prevent_minimal_evaluation(require_real_passivbot_rust_module, candle):
    result = compare(require_real_passivbot_rust_module, [candle], 0, 10 * M)
    assert not result["rows"]
    assert "unusable_historical_candle" in result["reasons"]
    assert "no_historical_candles" in result["reasons"]


def test_conflicting_finest_rows_use_uncontested_coarser_source(require_real_passivbot_rust_module):
    coarse = Candle(0, 5, 100, 120, 80, 110)
    a, b = Candle(M, 1, None, None, None, 99), Candle(M, 1, None, None, None, 101)
    expected = minute_prices([coarse], 0, 5 * M)
    for order in permutations([coarse, a, b]):
        result = run(require_real_passivbot_rust_module, list(order) + [a], 0, 5 * M)
        assert [r["close"] for r in result["rows"]] == pytest.approx([float(x) for x in expected.values()])
        assert "candle_conflict" in result["reasons"]
        assert all(r["resolution_minutes"] == 5 for r in result["rows"])
    # With no uncontested source, return the explicit no-candle domain, not an error.
    assert not run(require_real_passivbot_rust_module, [a, b], 0, 5 * M)["rows"]


def test_duplicate_rows_and_finer_overrides_are_not_conflicts(require_real_passivbot_rust_module):
    coarse = Candle(0, 5, 100, 120, 80, 110)
    fine = Candle(M, 1, None, None, None, 999)
    result = compare(require_real_passivbot_rust_module, [coarse, fine, fine], 0, 5 * M)
    assert "candle_conflict" not in result["reasons"]


@pytest.mark.parametrize("start,end", [(0, 90 * 24 * 60 * M + 1), (1, 0), (-(2**63), 2**63 - 1)])
def test_interval_allocation_bounds(require_real_passivbot_rust_module, start, end):
    with pytest.raises(ValueError, match="interval"):
        run(require_real_passivbot_rust_module, [], start, end)


def test_small_intervals_and_maximum_endpoint(require_real_passivbot_rust_module):
    assert not run(require_real_passivbot_rust_module, [], 2**63 - 2, 2**63 - 1)["rows"]
    compare(require_real_passivbot_rust_module, [Candle(0, 1, None, None, None, 100)], 1, 2 * M - 1)
