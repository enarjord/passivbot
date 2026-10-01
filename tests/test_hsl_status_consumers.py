"""Current bounded HSL status observations reach operator reports."""

import json
from pathlib import Path
import shutil
import subprocess
import pytest
from live import smoke_report as report
from live.event_bus import EventTypes


def groups(status="current", action="normal", omitted=2):
    scopes = [
        dict(
            symbol="BTC/USDT:USDT",
            pside="long",
            signal_mode="coin",
            tier="green" if action == "normal" else "red",
            action=action,
            availability="available",
            raw=0.1,
            ema=0.03,
            threshold=0.06,
            red_at=1000 if action == "halted" else None,
            flat_at=2000 if action == "halted" else None,
        )
    ]
    data = dict(
        schema_version=1,
        observation_status=status,
        signal_mode="coin",
        tier="green" if action == "normal" else "red",
        counts={"green": 3, "red": 0, "estimated": 1},
        scope_count=3,
        omitted_scopes=omitted,
        scopes=scopes,
    )
    group = report._risk_event_group(
        bot_key="fake",
        row={"ts": 3000, "seq": 1},
        live_event={"event_type": EventTypes.HSL_STATUS, "data": data},
        path=Path("events.jsonl"),
        line_no=1,
    )
    return {("fake",): group}


def test_current_scope_proximity_and_raw_pending_survive_reporting():
    captured = groups()
    status = report._summarize_hsl_status(captured)
    assert status["total"] == 1
    assert status["closest_to_red"][0]["red_proximity_pct"] == 50.0
    assert status["closest_to_red"][0]["symbol"] == "BTC/USDT:USDT"
    assert status["observations"][0]["omitted_scopes"] == 2
    assert status["observations"][0]["scope_count"] == 3
    pending = report._summarize_hsl_raw_red_pending(captured)
    assert pending["total"] == 1
    assert pending["pending"][0]["ema_gap_to_red_pct"] == 50.0
    shareable = report._shareable_hsl_status(status)
    assert "red_threshold" not in shareable["closest_to_red"][0]
    assert shareable["observations"][0]["omitted_scopes"] == 2


@pytest.mark.parametrize("action", ["normal", "panic", "halted"])
@pytest.mark.parametrize("status", ["stale", "diagnostic_unavailable", "not_evaluated"])
def test_stale_scope_not_reported_as_current_proximity_pending_or_cooldown(
    action, status
):
    captured = groups(status, action)
    assert report._summarize_hsl_status(captured)["closest_to_red"] == []
    assert "cooldown_active" not in report._summarize_hsl_status(captured)
    assert report._summarize_hsl_raw_red_pending(captured)["total"] == 0
    assert (
        report._summarize_hsl_status(captured)["observations"][0]["observation_status"]
        == status
    )


def test_current_halted_scope_reports_terminal_timestamps():
    summary = report._summarize_hsl_status(groups(action="halted"))
    item = summary["cooldown_active"][0]
    assert (item["action"], item["red_at"], item["flat_at"]) == ("halted", 1000, 2000)
    assert (
        report._shareable_hsl_status(summary)["cooldown_active"][0]["flat_at"] == 2000
    )


def test_multi_scope_report_preserves_identity_without_inventing_portfolio_score():
    captured = groups(omitted=0)
    data = next(iter(captured.values()))["latest_data"]
    data["scopes"].append(
        {**data["scopes"][0], "symbol": "ETH/USDT:USDT", "raw": 0.02, "ema": 0.01}
    )
    summary = report._summarize_hsl_status(captured)
    assert summary["total"] == 1
    assert len(summary["closest_to_red"]) == 2
    assert "drawdown_score" not in summary
    assert report._summarize_hsl_raw_red_pending(captured)["total"] == 1


def test_dashboard_event_summary_reads_current_scope_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node required")
    source = (
        Path(__file__).resolve().parents[1]
        / "src/monitor_dashboard_static/dashboard.js"
    ).read_text()
    start = source.index("  function summarizeEvent(")
    end = source.index("\n  function ", start + 1)
    function = source[start:end]
    status = next(iter(groups().values()))["latest_data"]
    script = (
        "const compactEntries = (xs) => xs.filter((x) => x[1] != null); const fmtCompact = (v) => String(v);\n"
        + function
        + "\nconst text = summarizeEvent({kind:'hsl.status',payload:"
        + json.dumps(status)
        + "}); if (!text.includes('BTC/USDT:USDT/long normal raw 0.1 ema 0.03') || !text.includes('omitted 2')) throw new Error(text);"
    )
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("enabled", [True, False])
def test_smoke_unified_policy_ignores_inactive_sides(enabled):
    config = {
        "live": {"hsl_signal_mode": "unified"},
        "bot": {
            "hsl": {"enabled": enabled},
            "long": {"hsl": {"enabled": not enabled}},
            "short": {"hsl": {"enabled": not enabled}},
        },
    }
    assert report._smoke_hsl_enabled_scopes(config) == (
        ["portfolio"] if enabled else []
    )


def test_raw_pending_current_status_gets_risk_attention():
    group = next(iter(groups().values()))
    assert report._risk_attention_rank(group) == 40
    group["latest_data"]["observation_status"] = "stale"
    assert report._risk_attention_rank(group) == 0
