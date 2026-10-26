"""Current HSL quantity names retain units, zeroes and legacy tape boundaries."""

from copy import deepcopy
from pathlib import Path

import pytest

from live import smoke_report
from live.event_bus import EventTypes
from test_hsl_status_consumers import groups


@pytest.mark.parametrize("ema,gap,percent,ema_gap_percent", [
    (.03, .03, 50.0, 50.0), (.06, 0.0, 100.0, 0.0), (.08, -.02, 133.333, None),
])
def test_current_threshold_gap_is_signed_fraction_and_proximity_stays_scale_100(
    ema, gap, percent, ema_gap_percent,
):
    captured = groups(action="normal" if ema <= .06 else "panic")
    data = next(iter(captured.values()))["latest_data"]
    data["scopes"][0]["signal_ema_fraction"] = ema
    status = smoke_report._summarize_hsl_status(captured)
    row = status["closest_to_red"][0]
    assert row["red_threshold_gap_fraction"] == pytest.approx(gap)
    assert row["red_threshold_proximity_percent"] == percent
    assert row["drawdown_threshold_fraction"] == .06
    assert not {"dist_to_red", "red_proximity_pct", "red_threshold"}.intersection(row)
    if ema <= .06:
        pending = smoke_report._summarize_hsl_raw_red_pending(captured)["pending"][0]
        assert pending["ema_red_threshold_gap_percent"] == ema_gap_percent


def test_legacy_top_level_tape_is_read_without_mutating_or_aliasing_current_scopes():
    data = {"signal_mode": "coin", "tier": "green", "dist_to_red": .02,
            "red_threshold": .05, "drawdown_score": .03}
    original = deepcopy(data)
    group = smoke_report._risk_event_group(
        bot_key="fixture", row={"ts": 3000, "seq": 1},
        live_event={"event_type": EventTypes.HSL_STATUS, "data": data},
        path=Path("events.jsonl"), line_no=1,
    )
    row = smoke_report._summarize_hsl_status({("fixture",): group})["closest_to_red"][0]
    assert row["red_threshold_gap_fraction"] == .02
    assert row["red_threshold_proximity_percent"] == 60.0
    assert data == original and group["latest_data"]["dist_to_red"] == .02
    assert "dist_to_red" not in row

    current = groups()
    scope = next(iter(current.values()))["latest_data"]["scopes"][0]
    for old, name in (("raw", "signal_raw_fraction"), ("ema", "signal_ema_fraction"),
                      ("threshold", "drawdown_threshold_fraction")):
        scope[old] = scope.pop(name)
    assert smoke_report._summarize_hsl_status(current)["closest_to_red"] == []
    assert smoke_report._summarize_hsl_raw_red_pending(current)["total"] == 0


def test_native_risk_and_intent_expose_fraction_and_currency_fields_without_aliases():
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension
    from test_hsl_signal import evaluate, request

    verify_loaded_runtime_extension()
    result = evaluate(native, request())
    for name in ("decision", "risk"):
        assert {"signal_raw_fraction", "signal_ema_fraction"} <= result[name].keys()
        assert {"raw", "ema"}.isdisjoint(result[name])
    assert result["risk"]["drawdown_memory_usd"] > 1.0
    assert 0.0 < result["risk"]["signal_raw_fraction"] < 1.0
    assert "drawdown" not in result["risk"]
