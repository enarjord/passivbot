from __future__ import annotations

import io
import json
from pathlib import Path

import live.performance_report as performance_report_module
from live.performance_report import build_live_performance_report
from tools import live_performance_report


def _monitor_row(
    *,
    event_type: str,
    seq: int,
    ts: int,
    exchange: str = "binance",
    user: str = "binance_01",
    component: str = "test",
    status: str = "succeeded",
    level: str = "debug",
    reason_code: str = "test",
    symbol: str | None = None,
    pside: str | None = None,
    ids: dict | None = None,
    data: dict | None = None,
) -> dict:
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": level,
        "source": "live",
        "component": component,
        "exchange": exchange,
        "user": user,
        "symbol": symbol,
        "pside": pside,
        "status": status,
        "reason_code": reason_code,
        "data": dict(data or {}),
        "ids": dict(ids or {}),
    }
    row = {
        "exchange": exchange,
        "user": user,
        "kind": event_type,
        "tags": ["test"],
        "payload": {"_live_event": live_event},
        "seq": seq,
        "ts": ts,
    }
    if symbol is not None:
        row["symbol"] = symbol
    if pside is not None:
        row["pside"] = pside
    return row


def _write_ndjson(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _groups_by_operation(report):
    return {
        group["operation"]: group
        for group in report["performance"]["groups"]
    }


def test_live_performance_report_aggregates_cycle_state_remote_and_hsl_timings(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                component="execution_loop",
                ids={"cycle_id": "cy_1"},
                data={
                    "elapsed_ms": 1200,
                    "timings_ms": {
                        "authoritative": 300,
                        "market_state": 200,
                        "execute": 500,
                        "monitor_flush": 100,
                    },
                },
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                component="execution_loop",
                ids={"cycle_id": "cy_2"},
                data={
                    "elapsed_ms": 1800,
                    "timings_ms": {
                        "authoritative": 700,
                        "market_state": 250,
                        "execute": 600,
                        "monitor_flush": 150,
                    },
                },
            ),
            _monitor_row(
                event_type="state.refresh_timing",
                seq=3,
                ts=3000,
                component="state.refresh",
                data={
                    "wall_ms": 900,
                    "surface_max_ms": 800,
                    "surface_sum_ms": 1700,
                    "residual_ms": 100,
                    "timings_ms": {"balance": 400, "open_orders": 800},
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=4,
                ts=4000,
                component="state.authoritative_fetch",
                reason_code="authoritative_open_orders",
                data={"surface": "open_orders", "elapsed_ms": 850},
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=5,
                ts=5000,
                component="candles.remote_fetch",
                reason_code="ccxt_fetch_ohlcv",
                symbol="BTC/USDT:USDT",
                data={"kind": "ccxt_fetch_ohlcv", "elapsed_ms": 120},
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=6,
                ts=6000,
                component="risk.hsl",
                symbol="BTC/USDT:USDT",
                pside="long",
                data={"stage": "pair_replay", "elapsed_s": 12.5},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")

    assert report["ok"] is True
    assert report["files_scanned"] == 1
    assert report["live_events"] == 6
    groups = _groups_by_operation(report)
    assert groups["cycle.elapsed"]["count"] == 2
    assert groups["cycle.elapsed"]["min_ms"] == 1200
    assert groups["cycle.elapsed"]["max_ms"] == 1800
    assert groups["cycle.elapsed"]["mean_ms"] == 1500
    assert groups["cycle.phase.execute"]["max_ms"] == 600
    assert groups["cycle.phase.monitor_flush"]["trading_impact"] == "diagnostics_only"
    assert groups["state_refresh.wall"]["max_ms"] == 900
    assert groups["state_refresh.surface.open_orders"]["max_ms"] == 800
    assert groups["remote_call.authoritative.open_orders"]["trading_impact"] == (
        "blocks_exchange_actions"
    )
    assert groups["remote_call.candle.ccxt_fetch_ohlcv"]["trading_impact"] == (
        "blocks_indicator_readiness"
    )
    assert groups["hsl_replay.pair_replay.elapsed"]["max_ms"] == 12500
    assert groups["hsl_replay.pair_replay.elapsed"]["timing_kind"] == "cumulative"


def test_live_performance_report_time_window_and_group_limit(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                data={"elapsed_ms": 2000},
            ),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        since_ms=1500,
        group_limit=0,
    )

    assert report["event_window"]["events_considered"] == 1
    assert report["event_window"]["events_skipped_before"] == 1
    assert report["performance"]["total_groups"] == 1
    assert report["performance"]["groups_truncated"] is True
    assert report["performance"]["groups"] == []


def test_live_performance_report_cli_outputs_json(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                data={"elapsed_ms": 1000},
            )
        ],
    )

    rc = live_performance_report.main([str(tmp_path / "monitor"), "--compact"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["performance"]["groups"][0]["operation"] == "cycle.elapsed"


def test_live_performance_report_redacts_missing_root_paths():
    report = build_live_performance_report("/Users/operator/passivbot/missing-monitor")

    rendered = json.dumps(report, sort_keys=True)
    assert report["root"] == "~/passivbot/missing-monitor"
    assert report["issues"][0]["path"] == "~/passivbot/missing-monitor"
    assert "/Users/operator" not in rendered


def test_live_performance_report_redacts_file_paths_and_oserror_messages(monkeypatch):
    event_path = Path("/root/passivbot/monitor/binance/binance_01/events/current.ndjson")

    monkeypatch.setattr(
        performance_report_module,
        "discover_event_files",
        lambda root, *, include_rotated=False: [event_path],
    )

    def fail_open(path):
        raise OSError(13, "Permission denied", str(path))

    monkeypatch.setattr(performance_report_module, "_open_text", fail_open)

    report = build_live_performance_report("/root/passivbot/monitor")

    rendered = json.dumps(report, sort_keys=True)
    assert report["root"] == "~/passivbot/monitor"
    assert report["files"] == ["~/passivbot/monitor/binance/binance_01/events/current.ndjson"]
    assert report["issues"][0]["path"] == (
        "~/passivbot/monitor/binance/binance_01/events/current.ndjson"
    )
    assert report["issues"][0]["message"] == "Permission denied"
    assert "/root/passivbot" not in rendered


def test_live_performance_report_redacts_file_paths_for_valid_events(monkeypatch):
    event_path = Path("/Users/operator/passivbot/monitor/binance/binance_01/events/current.ndjson")
    row = _monitor_row(
        event_type="cycle.completed",
        seq=1,
        ts=1000,
        data={"elapsed_ms": 1000},
    )

    monkeypatch.setattr(
        performance_report_module,
        "discover_event_files",
        lambda root, *, include_rotated=False: [event_path],
    )
    monkeypatch.setattr(
        performance_report_module,
        "_open_text",
        lambda path: io.StringIO(json.dumps(row) + "\n"),
    )

    report = build_live_performance_report("/Users/operator/passivbot/monitor")

    rendered = json.dumps(report, sort_keys=True)
    assert report["ok"] is True
    assert report["root"] == "~/passivbot/monitor"
    assert report["files"] == [
        "~/passivbot/monitor/binance/binance_01/events/current.ndjson"
    ]
    assert "/Users/operator" not in rendered
