"""Hand-computed signal pins for the single HSL controller in every scope."""

from dataclasses import replace

import pytest

from hsl_reference import Fill
from test_hsl_reference_replay import M, frame, pair
from test_hsl_evaluator import evaluate


@pytest.mark.parametrize(
    "mode,selectors",
    [
        ("coin", {"pside": "long", "symbol": "A"}),
        ("pside", {"pside": "long"}),
        ("unified", {}),
    ],
)
@pytest.mark.parametrize("mark,expected", [(100, 0), (90, 0.1), (80, 0.2)])
def test_current_equity_anchor_and_strict_red_threshold(
    mode, selectors, mark, expected
):
    snapshot = frame(pair(size=1, basis=100, mark=mark))
    result = evaluate(snapshot, mode, threshold=0.1, **selectors)["decision"]
    assert result["raw"] == pytest.approx(expected)
    assert result["ema"] == pytest.approx(expected)
    assert result["action"] == ("panic" if expected > 0.1 else "normal")


def test_coin_budget_uses_slots_and_aggregate_uses_whole_balance():
    snapshot = frame(pair(size=1, basis=100, mark=90), balance=200)
    coin = evaluate(snapshot, "coin", slots=2, pside="long", symbol="A")["decision"]
    aggregate = evaluate(snapshot, "unified")["decision"]
    assert coin["raw"] == pytest.approx(0.1)
    assert aggregate["raw"] == pytest.approx(0.05)


@pytest.mark.parametrize("expired_at", [M - 1, M + 9999])
def test_exact_lookback_excludes_old_cashflow_even_in_same_minute(expired_at):
    start = M + 10000
    current = pair(
        size=1,
        basis=100,
        mark=90,
        fills=[Fill("entry", 2 * M, 1, 100, 0)],
        prices={2 * M: 100, 3 * M: 90, 4 * M: 90},
    )
    clean = frame(current, start=start)
    contaminated = replace(
        clean,
        pairs=(
            replace(
                current,
                fills=(Fill("expired", expired_at, -1, 100, -10000), *current.fills),
            ),
        ),
    )
    assert evaluate(clean) == evaluate(contaminated)
