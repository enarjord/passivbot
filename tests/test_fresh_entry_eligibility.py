from copy import deepcopy

import pytest

from live.fresh_entry_eligibility import FreshEntryEligibilityTrace


def _initial(symbol="BTC/USDT:USDT", pside="long", **overrides):
    order = {
        "symbol": symbol,
        "position_side": pside,
        "pb_order_type": f"entry_initial_normal_{pside}",
    }
    order.update(overrides)
    return order


@pytest.mark.parametrize(
    ("configure", "expected"),
    [
        (lambda trace, order: trace.record_eligible_orders([order]), "eligible"),
        (
            lambda trace, order: trace.record_blocked_orders([order], "distance_gate"),
            "blocked_candidate",
        ),
        (
            lambda trace, order: trace.record_satisfied_orders([order], "open_order"),
            "already_satisfied",
        ),
        (
            lambda trace, order: trace.record_protective_orders([order]),
            "protective_only",
        ),
        (lambda trace, order: None, "no_candidate"),
    ],
)
def test_all_outcomes(configure, expected):
    trace = FreshEntryEligibilityTrace()
    order = _initial()
    if expected == "protective_only":
        order["reduce_only"] = True
        order["pb_order_type"] = "close_grid_long"
    trace.record_evaluated(order["symbol"], order["position_side"])
    configure(trace, order)

    event = trace.to_event_data()

    assert event["records"][0]["outcome"] == expected
    assert event["outcome_counts"][expected] == 1
    assert set(event["records"][0]) == {
        "symbol",
        "pside",
        "outcome",
        "evaluated_count",
        "ideal_count",
        "satisfied_count",
        "blocked_count",
        "protective_count",
        "eligible_count",
        "reason_counts",
    }


def test_outcome_precedence_prefers_eligible_then_blocked():
    trace = FreshEntryEligibilityTrace()
    eligible = _initial("ADA/USDT:USDT", "long")
    blocked = _initial("BTC/USDT:USDT", "short")

    trace.record_ideal_orders([eligible, blocked])
    trace.record_eligible_orders([eligible])
    trace.record_blocked_orders([eligible, blocked], "risk_gate")
    trace.record_satisfied_orders([blocked], "open_order")

    records = {(item["symbol"], item["pside"]): item for item in trace.to_event_data()["records"]}
    assert records[("ADA/USDT:USDT", "long")]["outcome"] == "eligible"
    assert records[("BTC/USDT:USDT", "short")]["outcome"] == "blocked_candidate"


def test_unaccounted_ideal_candidate_is_blocked_with_synthetic_reason():
    trace = FreshEntryEligibilityTrace()
    trace.record_ideal_orders([_initial()])

    event = trace.to_event_data()
    record = event["records"][0]
    assert record["outcome"] == "blocked_candidate"
    assert record["reason_counts"] == {"unclassified_candidate": 1}
    assert event["reason_counts"] == {"unclassified_candidate": 1}


def test_candidate_free_outcomes_have_stable_default_reasons():
    trace = FreshEntryEligibilityTrace()
    trace.record_evaluated("BTC/USDT:USDT", "long")
    trace.record_evaluated("ETH/USDT:USDT", "short")
    trace.record_protective_orders(
        [
            {
                "symbol": "ETH/USDT:USDT",
                "position_side": "short",
                "pb_order_type": "close_panic_short",
                "reduce_only": True,
            }
        ]
    )

    records = {
        (item["symbol"], item["pside"]): item
        for item in trace.to_event_data()["records"]
    }

    assert records[("BTC/USDT:USDT", "long")]["reason_counts"] == {
        "rust_no_initial_candidate": 1
    }
    assert records[("ETH/USDT:USDT", "short")]["reason_counts"] == {
        "protective_actions_only": 1
    }


def test_normalized_type_and_custom_id_classification_and_protective_filtering():
    trace = FreshEntryEligibilityTrace()
    from_type = _initial("ADA/USDT:USDT", pb_order_type=" ENTRY-INITIAL-NORMAL-LONG ")
    from_custom_id = _initial(
        "BTC/USDT:USDT", pb_order_type="unknown", custom_id="entry initial normal long"
    )
    panic = _initial("ETH/USDT:USDT", pb_order_type="close_panic_long")
    reduce_only_initial = _initial("SOL/USDT:USDT", reduceOnly=True)

    trace.record_ideal_orders([from_type, from_custom_id, panic, reduce_only_initial])
    trace.record_protective_orders([from_type, from_custom_id, panic, reduce_only_initial])
    records = {(item["symbol"], item["pside"]): item for item in trace.to_event_data()["records"]}

    assert records[("ADA/USDT:USDT", "long")]["ideal_count"] == 1
    assert records[("BTC/USDT:USDT", "long")]["ideal_count"] == 1
    assert records[("ETH/USDT:USDT", "long")]["protective_count"] == 1
    assert records[("SOL/USDT:USDT", "long")]["protective_count"] == 1
    assert ("ETH/USDT:USDT", "long") not in {
        key for key, value in records.items() if value["ideal_count"]
    }


def test_records_sort_deterministically_by_symbol_then_pside():
    trace = FreshEntryEligibilityTrace()
    trace.record_evaluated("BTC/USDT:USDT", "short")
    trace.record_evaluated("ADA/USDT:USDT", "short")
    trace.record_evaluated("BTC/USDT:USDT", "long")

    assert [(item["symbol"], item["pside"]) for item in trace.to_event_data()["records"]] == [
        ("ADA/USDT:USDT", "short"),
        ("BTC/USDT:USDT", "long"),
        ("BTC/USDT:USDT", "short"),
    ]


def test_default_32_record_truncation_preserves_full_aggregate_counts():
    trace = FreshEntryEligibilityTrace()
    for index in range(33):
        symbol = f"S{index:03d}/USDT:USDT"
        trace.record_evaluated(symbol, "long")
        trace.record_count(symbol, "long", "blocked", reason="distance_gate")

    event = trace.to_event_data()

    assert event["records_total"] == 33
    assert len(event["records"]) == 32
    assert event["records_truncated"] is True
    assert event["evaluated_count"] == 33
    assert event["outcome_counts"]["blocked_candidate"] == 33
    assert event["reason_counts"] == {"distance_gate": 33}


def test_unsafe_labels_and_invalid_pside_are_rejected_without_leaking_them():
    trace = FreshEntryEligibilityTrace()
    with pytest.raises(ValueError, match="safe query label"):
        trace.record_evaluated("BTC/USDT unsafe", "long")
    with pytest.raises(ValueError, match="safe query label"):
        trace.record_count("BTC/USDT:USDT", "long", "blocked", reason="secret=value")
    with pytest.raises(ValueError, match="safe query label"):
        trace.record_count("BTC/USDT:USDT", "long", "blocked", reason="r" * 121)
    with pytest.raises(ValueError, match="safe query label"):
        trace.record_evaluated("B" * 121, "long")
    with pytest.raises(ValueError, match="pside"):
        trace.record_evaluated("BTC/USDT:USDT", "both")

    assert trace.to_event_data()["records_total"] == 0


def test_malformed_orders_are_ignored_and_input_orders_are_not_mutated():
    trace = FreshEntryEligibilityTrace()
    valid = _initial()
    malformed = [
        None,
        "entry_initial_normal_long",
        {},
        _initial(symbol="bad symbol"),
        _initial(pside="both"),
    ]
    orders = [valid, *malformed]
    before = deepcopy(orders)

    trace.record_ideal_orders(orders)
    trace.record_satisfied_orders(orders, "open_order")
    trace.record_blocked_orders(orders, "distance_gate")
    trace.record_eligible_orders(orders)
    trace.record_protective_orders([_initial(reduce_only=True), *orders])

    assert orders == before
    event = trace.to_event_data()
    assert event["records_total"] == 1
    record = event["records"][0]
    assert record["ideal_count"] == record["satisfied_count"] == record["blocked_count"] == 1
    assert record["eligible_count"] == 1
    assert record["outcome"] == "eligible"


@pytest.mark.parametrize("count", [-1, 1.0, "1", True])
def test_record_count_requires_nonnegative_integer_count(count):
    trace = FreshEntryEligibilityTrace()
    with pytest.raises(ValueError, match="nonnegative integer"):
        trace.record_count("BTC/USDT:USDT", "long", "ideal", count=count)
    with pytest.raises(ValueError, match="fact"):
        trace.record_count("BTC/USDT:USDT", "long", "unknown")

    trace.record_count("BTC/USDT:USDT", "long", "ideal", count=0)
    assert trace.to_event_data()["records_total"] == 0
