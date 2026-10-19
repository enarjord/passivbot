"""Real native retention checks; run only against a source-verified extension."""

import json
import struct

import pytest

import passivbot_rust as pbr


@pytest.fixture(scope="module", autouse=True)
def verified_native_runtime():
    if getattr(pbr, "__is_stub__", False):
        pytest.skip("requires the source-verified Rust extension")
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()


MINUTE = 60_000
HORIZON = 3 * MINUTE


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def facts(minute, *, start=None, slots=1, size=1.0, mark=90.0):
    now = minute * MINUTE + 1
    if start is None:
        start = max(0, now - HORIZON)
    return {
        "snapshot": {
            "now": now, "start": start, "balance": 1000.0, "balance_at": now,
            "config_at": now, "max_current_age_ms": MINUTE, "mode": "coin",
            "pside": "long", "symbol": "A",
            "pairs": [{
                "symbol": "A", "position": {
                    "size": size, "basis": 100.0 if size else 0.0, "mark": mark,
                    "multiplier": 1.0, "inverse": False, "pside": "long",
                },
                "position_at": now, "mark_at": now, "fills_started_at": None,
                "fills_at": None, "prices_at": now, "prices": {
                    str(at * MINUTE): mark for at in range(minute + 1)
                    if at * MINUTE >= start
                },
                "fills": [], "revisions": [0, 0, 0, 0], "fills_position_anchor": None,
            }],
        },
        "slots": slots, "scale_budget_with_excess_allowance": False,
        "exposure_budget": None, "memory_ema_span_minutes": 7.0,
        "drawdown_ratio_ema_span_minutes": 1.0, "threshold": 0.001,
        "cooldown_ms": HORIZON,
        "budget_observations": [{
            "timestamp": 0, "order": -1, "balance": 1000.0, "slots": slots,
            "scale_budget_with_excess_allowance": False,
            "exposure_budget": None, "estimated": True,
        }],
    }


def evaluate(replay, value):
    return json.loads(replay.evaluate(json.dumps(value, allow_nan=False)))


def test_long_active_episode_matches_independent_f32_decay_and_bounded_counts():
    replay = pbr.HslV9Replay()
    expected = f32(10.0)
    # Establish the actually observed exposed seed in minute zero. Historical
    # marks before a first account observation do not prove earlier exposure.
    initial = evaluate(replay, facts(0))
    assert initial["risk"]["drawdown"] == expected
    for minute in range(1, 81):
        value = facts(minute)
        expected = f32(expected * f32(0.75))
        output = evaluate(replay, value)
        assert output["risk"]["drawdown"] == expected
        assert output["continuity"]["bootstrap_origin_ms"] == 0
        assert output["continuity"]["reconstruction"] == "bounded_bootstrap"
        before = replay.cache_info()["advanced_events"]
        assert evaluate(replay, value) == output
        assert replay.cache_info()["advanced_events"] == before
        if minute % 7 == 0:
            replay.clear_derived_cache()
            assert evaluate(replay, value) == output
        counts = replay.cache_info()
        assert counts["retained_prices"] <= 8
        assert counts["retained_positions"] <= 8
        assert counts["committed_events"] <= 20
        assert counts["retained_events"] <= 4
        assert counts["retained_budgets"] <= 1
    assert output["continuity"]["boundary_at_ms"] is not None


def test_discarded_prefix_reconstruction_is_visible_and_current_loss_has_no_warmup():
    replay = pbr.HslV9Replay()
    for minute in range(1, 18):
        evaluate(replay, facts(minute))
    old = facts(17)
    boundary = evaluate(replay, old)["continuity"]["boundary_at_ms"]
    old["snapshot"]["start"] = boundary - MINUTE
    old["snapshot"]["pairs"][0]["prices"] = {}
    old["drawdown_ratio_ema_span_minutes"] = 1_000_000.0
    output = evaluate(replay, old)
    assert output["continuity"]["reconstruction"] == "policy_changed_beyond_retained_support"
    assert output["continuity"]["boundary_at_ms"] is None
    assert output["continuity"]["bootstrap_origin_ms"] == boundary - MINUTE
    assert output["risk"]["drawdown"] > 0
    assert output["risk"]["ema"] >= output["risk"]["raw"]
    assert output["risk"]["action"] == "panic"
    assert "unknown_prefix_current_basis_seed" in output["reasons"]


def test_compacted_fill_below_request_start_is_clipped_but_requested_correction_rebuilds():
    replay = pbr.HslV9Replay()
    fill_at = 3 * MINUTE - 1
    fill = {
        "identity": "boundary-close", "timestamp": fill_at, "delta": -0.25,
        "price": 100.0, "realized": -7.0, "fee": 0.0, "sequence": 0, "revision": 0,
    }
    for minute in range(1, 8):
        value = facts(minute, size=0.75 if minute >= 3 else 1.0)
        if value["snapshot"]["start"] <= fill_at <= value["snapshot"]["now"]:
            value["snapshot"]["pairs"][0]["fills"] = [fill]
        output = evaluate(replay, value)
    assert output["continuity"]["boundary_at_ms"] == fill_at
    assert value["snapshot"]["start"] > fill_at
    before = replay.cache_info()["advanced_events"]

    # The native API clips transport rows outside the requested cohort. An
    # execution already summarized at the boundary must not debit risk again.
    value["snapshot"]["pairs"][0]["fills"] = [fill]
    assert evaluate(replay, value) == output
    assert replay.cache_info()["advanced_events"] == before
    replay.clear_derived_cache()
    assert evaluate(replay, value) == output

    value["snapshot"]["start"] = fill_at
    status = evaluate(replay, value)["continuity"]
    assert status["reconstruction"] == "request_before_retained_support"
    assert status["bootstrap_origin_ms"] == fill_at
    assert status["boundary_at_ms"] is None


def test_native_initializer_status_distinguishes_inactive_then_ready():
    replay = pbr.HslV9Replay()
    output = evaluate(replay, facts(1, slots=0, size=0.0))
    assert not output["continuity"]["initialized"]
    assert replay.cache_info()["initialized"] == 0
    ready = facts(2, start=MINUTE)
    output = evaluate(replay, ready)
    assert output["continuity"]["initialized"]
    assert output["continuity"]["bootstrap_origin_ms"] == MINUTE
    assert output["continuity"]["bootstrap_observed_ms"] == 2 * MINUTE + 1


def test_failed_reconstruction_does_not_replace_retained_initializer_or_boundary():
    replay = pbr.HslV9Replay()
    for minute in range(1, 18):
        evaluate(replay, facts(minute))
    current = facts(17)
    before = evaluate(replay, current)
    counts = replay.cache_info()
    malformed = facts(17, start=0)
    malformed["snapshot"]["pairs"][0]["position"]["basis"] = -1.0
    with pytest.raises(ValueError):
        evaluate(replay, malformed)
    assert replay.cache_info() == counts
    assert evaluate(replay, current) == before
