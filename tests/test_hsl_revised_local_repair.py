import json
import pytest
from hsl_reference import Fill, Position
from test_hsl_revised_history import run
from test_hsl_revised_snapshot import payload
import test_hsl_reference_replay as cases


@pytest.mark.parametrize("side,d", [("long", 1), ("short", -1)])
def test_new_position_never_rewrites_completed_history(side, d):
    import passivbot_rust as r

    fills = [
        Fill("open", 60000, d * 10, 100, 0, 0),
        Fill("close", 120000, -d * 10, 80 if d == 1 else 120, -200, 0),
    ]
    for qty in (0.1, 1, 100):
        p = Position(d * qty, 100, 100, pside=side)
        out = run(r, p, fills, {60000: 100, 120000: 80 if d == 1 else 120})
        assert out["opening_size"] == 0
        assert [(e["before"], e["after"]) for e in out["events"]] == [(0, 10), (10, 0)]
        assert out["reconciliation"]["before"] == 0
        assert out["reconciliation"]["after"] == d * qty


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("side,d", [("long", 1), ("short", -1)])
def test_missing_open_is_green_at_basis_but_current_loss_still_panics(mode, side, d):
    import passivbot_rust as r

    for mark, expected in [(100, "normal"), (10 if d == 1 else 190, "panic")]:
        pair = cases.pair(
            pside=side,
            size=d * 2,
            basis=100,
            mark=mark,
            fills=[
                Fill("open", cases.M, d * 10, 100, 0, 0),
                Fill("close", 2 * cases.M, -d * 10, 80 if d == 1 else 120, -200, 0),
            ],
        )
        selectors = {} if mode == "unified" else {"pside": side}
        if mode == "coin":
            selectors["symbol"] = "A"
        snapshot = payload(cases.frame(pair, balance=1000), mode, **selectors)
        out = json.loads(
            r.hsl_revised_evaluate(
                json.dumps(
                    dict(
                        snapshot=snapshot,
                        slots=1,
                        span=356,
                        threshold=0.06,
                        cooldown_ms=116 * 60000,
                        restart="always",
                    )
                )
            )
        )
        assert out["decision"]["action"] == expected


def test_late_oversized_close_cannot_rewrite_earlier_episode():
    import passivbot_rust as r

    fills = [
        Fill("a", 1, 1, 100, 0, 0),
        Fill("b", 2, -1, 90, -10, 0),
        Fill("c", 3, 1, 100, 0, 0),
        Fill("d", 4, -3, 90, -30, 0),
    ]
    out = run(r, Position(0, 0, 90), fills, {1: 100, 2: 90, 3: 100, 4: 90})
    assert [(e["before"], e["after"]) for e in out["events"]] == [
        (0, 1),
        (1, 0),
        (0, 1),
        (3, 0),
    ]
    assert "local_quantity_reconciliation" in out["events"][-1]["reasons"]
    assert out["events"][-1]["gross_realized"] == -30
