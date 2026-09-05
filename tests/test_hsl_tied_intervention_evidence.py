import itertools

import pytest

import passivbot_hsl as hsl


def _fill(event_id, timestamp, before, after, *, pnl=0.0, fee=0.0, symbol="A", pside="long"):
    increase = after > before
    return dict(
        id=event_id,
        timestamp=timestamp,
        symbol=symbol,
        pside=pside,
        action="increase" if increase else "decrease",
        qty=abs(after - before),
        side=(
            ("buy" if increase else "sell") if pside == "long" else ("sell" if increase else "buy")
        ),
        pnl=pnl,
        fee_paid=fee,
        raw=[
            {
                "data": {
                    "amount": abs(after - before),
                    "price": 1.0,
                    "side": (
                        ("buy" if increase else "sell")
                        if pside == "long"
                        else ("sell" if increase else "buy")
                    ),
                    "info": {"startPosition": str(before if pside == "long" else -before)},
                }
            }
        ],
    )


def _evidence(events, **kwargs):
    return hsl._equity_hard_stop_intervention_entry_evidence(
        events,
        psides={"long"},
        stop_ms=60_000,
        cooldown_until_ms=120_000,
        symbol="A",
        **kwargs,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pside", ["long", "short"])
def test_tied_intervention_uses_exchange_chain_and_prefix_before_entry(reverse, pside):
    opening = _fill("open", 1_000, 0, 3, fee=-1.0, pside=pside)
    stop = _fill("stop", 60_000, 3, 0, pnl=-300.0, fee=-2.0, pside=pside)
    entry = _fill("entry", 60_000, 0, 2, fee=-5.0, pside=pside)
    events = [opening, *([entry, stop] if reverse else [stop, entry])]
    result = hsl._equity_hard_stop_intervention_entry_evidence(
        events,
        psides={pside},
        stop_ms=60_000,
        cooldown_until_ms=120_000,
        symbol="A",
    )
    assert result["entry_timestamp_ms"] == 60_000
    assert result["entry_event"] is entry
    assert result["stop_event"] is stop
    assert result["entry_index"] == 2
    assert result["stop_index"] == 1
    assert result["realized_before_entry"] == -303.0
    assert (
        hsl._equity_hard_stop_intervention_entry_timestamp(
            events,
            psides={pside},
            stop_ms=60_000,
            cooldown_until_ms=120_000,
            symbol="A",
        )
        == 60_000
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_multiple_tied_episodes_not_proven_by_existing_chain_validator_stay_closed(order):
    opening = _fill("open", 1_000, 0, 3, fee=-1.0)
    cohort = [
        _fill("stop1", 60_000, 3, 0, pnl=-300.0),
        _fill("entry1", 60_000, 0, 2, fee=-2.0),
        _fill("stop2", 60_000, 2, 0, pnl=-9.0),
        _fill("entry2", 60_000, 0, 1, fee=-4.0),
    ]
    events = [opening, *(cohort[index] for index in order)]
    # Existing unique-successor evidence cannot choose at the repeated zero
    # node. An explicit index must not bypass that ordering contract.
    assert _evidence(events) is None
    assert _evidence(events, stop_index=1) is None
    assert _evidence(events, stop_index=3) is None


@pytest.mark.parametrize(
    "case", ["missing_chain", "entry_before_stop", "partial_close", "cooldown_end"]
)
def test_tied_entry_requires_after_flatten_proof_and_strict_cooldown(case):
    events = [_fill("open", 1_000, 0, 3), _fill("stop", 60_000, 3, 0), _fill("entry", 60_000, 0, 2)]
    if case == "missing_chain":
        events[1].pop("raw")
    elif case == "entry_before_stop":
        events = [_fill("entry", 60_000, 0, 3), _fill("stop", 60_000, 3, 0)]
    elif case == "partial_close":
        events[1:] = [_fill("partial", 60_000, 3, 1), _fill("entry", 60_000, 1, 2)]
    else:
        events[-1]["timestamp"] = 120_000
    assert _evidence(events) is None
    assert (
        hsl._equity_hard_stop_intervention_entry_timestamp(
            events,
            psides={"long"},
            stop_ms=60_000,
            cooldown_until_ms=120_000,
            symbol="A",
        )
        is None
    )


def test_scope_must_be_flat_not_only_the_entry_coin():
    events = [
        _fill("openA", 1_000, 0, 3),
        _fill("openB", 2_000, 0, 1, symbol="B"),
        _fill("stop", 60_000, 3, 0),
        _fill("entry", 60_000, 0, 2),
    ]
    assert _evidence(events) is not None
    assert (
        hsl._equity_hard_stop_intervention_entry_evidence(
            events,
            psides={"long"},
            stop_ms=60_000,
            cooldown_until_ms=120_000,
        )
        is None
    )


def test_separate_timestamp_inference_does_not_require_missing_stop_tape():
    events = [dict(timestamp=60_001, symbol="A", pside="long", action="increase")]
    assert (
        hsl._equity_hard_stop_intervention_entry_timestamp(
            events,
            psides={"long"},
            stop_ms=60_000,
            cooldown_until_ms=120_000,
        )
        == 60_001
    )


@pytest.mark.parametrize("stop_ms, stop_index, prefix", [(60_000, 1, -301.0), (61_000, 3, -312.0)])
def test_explicit_stop_index_selects_its_own_cumulative_prefix(stop_ms, stop_index, prefix):
    events = [
        _fill("open", 1_000, 0, 3, fee=-1.0),
        _fill("stop1", 60_000, 3, 0, pnl=-300.0),
        _fill("entry1", 60_000, 0, 2, fee=-2.0),
        _fill("stop2", 61_000, 2, 0, pnl=-9.0),
        _fill("entry2", 61_000, 0, 1, fee=-4.0),
    ]
    result = hsl._equity_hard_stop_intervention_entry_evidence(
        reversed(events),
        psides={"long"},
        stop_ms=stop_ms,
        cooldown_until_ms=120_000,
        symbol="A",
        stop_index=stop_index,
    )
    assert result["entry_event"] is events[stop_index + 1]
    assert result["realized_before_entry"] == prefix
    assert result["realized_at_stop"] == prefix
    assert (
        hsl._equity_hard_stop_intervention_entry_evidence(
            events,
            psides={"long"},
            stop_ms=stop_ms,
            cooldown_until_ms=120_000,
            symbol="A",
            stop_index=stop_index + 1,
        )
        is None
    )


def test_tied_flatten_proof_tolerates_only_existing_hsl_quantity_epsilon():
    partial = _fill("partial", 2_000, 0.3, 0.2)
    partial["qty"] = 0.1
    partial["raw"][0]["data"]["amount"] = 0.1
    events = [
        _fill("open", 1_000, 0.0, 0.3), partial,
        _fill("stop", 60_000, 0.2, 0.0), _fill("entry", 60_000, 0.0, 0.4),
    ]
    ordered, ambiguous = hsl._equity_hard_stop_order_fill_cohorts(events)
    assert not ambiguous
    assert [event["id"] for event in ordered] == ["open", "partial", "stop", "entry"]
    result = _evidence(events)
    assert result["entry_event"] is events[-1]
    assert result["realized_before_entry"] == 0.0
    # A material canonical quantity mismatch still cannot prove a flatten.
    partial["qty"] = 0.1001
    assert _evidence(events) is None
