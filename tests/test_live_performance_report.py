from __future__ import annotations

import io
import json
from pathlib import Path

import live.performance_report as performance_report_module
from live.performance_report import build_live_performance_report, summarize_live_performance_report
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


def test_live_performance_report_filters_by_bot_exchange_and_user(tmp_path):
    events_dir = tmp_path / "monitor" / "mixed" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                exchange="binance",
                user="binance_01",
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={"elapsed_ms": 2000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=3000,
                exchange="gateio",
                user="gateio_01",
                data={"elapsed_ms": 3000},
            ),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        bot_filters=["okx/okx_faisal"],
    )

    assert report["live_events"] == 1
    assert report["filters"] == {
        "enabled": True,
        "bots": ["okx/okx_faisal"],
        "exchanges": [],
        "users": [],
        "events_skipped": 2,
    }
    assert report["bots"] == [{"bot": "okx/okx_faisal", "events": 1}]
    groups = _groups_by_operation(report)
    assert groups["cycle.elapsed"]["max_ms"] == 2000

    exchange_user_report = build_live_performance_report(
        tmp_path / "monitor",
        exchange_filters=["gateio"],
        user_filters=["gateio_01"],
    )

    assert exchange_user_report["live_events"] == 1
    assert exchange_user_report["filters"]["events_skipped"] == 2
    assert exchange_user_report["bots"] == [{"bot": "gateio/gateio_01", "events": 1}]


def test_live_performance_report_summary_projection_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                data={
                    "elapsed_ms": 1000,
                    "timings_ms": {"execute": 600, "monitor_flush": 50},
                },
            )
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        group_limit=10,
        user_filters=["binance_01"],
    )
    summary = summarize_live_performance_report(report, group_limit=1)

    assert summary["ok"] is True
    assert summary["files_scanned"] == 1
    assert summary["filters"]["users"] == ["binance_01"]
    assert summary["performance"]["total_groups"] == 3
    assert len(summary["performance"]["groups"]) == 1
    assert "files" not in summary
    assert "event_types" not in summary


def test_live_performance_report_decision_boundary_lag(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                seq=1,
                ts=61_000,
                component="execution_loop",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=2,
                ts=62_500,
                component="rust_orchestrator",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.returned",
                seq=3,
                ts=62_700,
                component="rust_orchestrator",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="action.planned",
                seq=4,
                ts=62_800,
                component="action_planner",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="execution.create_sent",
                seq=5,
                ts=63_500,
                component="executor",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="execution.confirmation_satisfied",
                seq=6,
                ts=64_000,
                component="executor",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=7,
                ts=64_200,
                component="execution_loop",
                ids={"cycle_id": "cy_1"},
                data={"elapsed_ms": 3200},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    lag_groups = {
        group["operation"]: group
        for group in report["decision_boundary_lag"]["groups"]
    }

    assert report["decision_boundary_lag"]["cycles"] == 1
    assert report["decision_boundary_lag"]["cycles_with_write"] == 1
    assert lag_groups["decision_boundary.cycle_started"]["max_ms"] == 1000
    assert lag_groups["decision_boundary.rust_called"]["max_ms"] == 2500
    assert lag_groups["decision_boundary.action_planned"]["max_ms"] == 2800
    assert lag_groups["decision_boundary.first_write_sent"]["max_ms"] == 3500
    assert lag_groups["decision_boundary.confirmation_satisfied"]["max_ms"] == 4000
    assert lag_groups["decision_boundary.cycle_completed"]["max_ms"] == 4200
    assert lag_groups["decision_boundary.first_write_sent"]["trading_impact"] == (
        "blocks_exchange_actions"
    )


def test_live_performance_report_decision_boundary_handles_cycle_id_reuse(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_1"},
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(event_type="bot.started", seq=3, ts=10_000),
            _monitor_row(
                event_type="cycle.started",
                seq=4,
                ts=61_000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=5,
                ts=62_000,
                ids={"cycle_id": "cy_1"},
                data={"elapsed_ms": 1000},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")

    assert report["decision_boundary_lag"]["cycles"] == 2
    lag_groups = {
        group["operation"]: group
        for group in report["decision_boundary_lag"]["groups"]
    }
    assert lag_groups["decision_boundary.cycle_started"]["count"] == 2
    assert lag_groups["decision_boundary.cycle_started"]["max_ms"] == 1000


def test_live_performance_report_summary_includes_bounded_decision_lag(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                seq=1,
                ts=61_000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=2,
                ts=62_500,
                ids={"cycle_id": "cy_1"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert summary["decision_boundary_lag"]["cycles"] == 1
    assert summary["decision_boundary_lag"]["total_groups"] == 2
    assert len(summary["decision_boundary_lag"]["groups"]) == 1


def test_live_performance_report_input_staleness(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="data_packet.updated",
                seq=1,
                ts=1000,
                data={
                    "kind": "balance",
                    "revision": 7,
                    "response_received_ts_ms": 1000,
                },
            ),
            _monitor_row(
                event_type="data_packet.updated",
                seq=2,
                ts=1200,
                data={
                    "kind": "open_orders",
                    "revision": 8,
                    "response_received_ts_ms": 1200,
                },
            ),
            _monitor_row(
                event_type="data_packet.updated",
                seq=3,
                ts=1400,
                data={
                    "kind": "positions",
                    "revision": 9,
                    "response_received_ts_ms": 1400,
                },
            ),
            _monitor_row(
                event_type="snapshot.built",
                seq=4,
                ts=2000,
                data={
                    "cycle_id": 1,
                    "data_packets": [
                        {"kind": "balance", "revision": 7},
                        {"kind": "open_orders", "revision": 8},
                        {"kind": "positions", "revision": 9},
                    ],
                },
            ),
            _monitor_row(
                event_type="ema.bundle.completed",
                seq=5,
                ts=2500,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=6,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    staleness_groups = {
        group["operation"]: group
        for group in report["input_staleness"]["groups"]
    }

    assert report["input_staleness"]["snapshots_seen"] == 1
    assert report["input_staleness"]["rust_calls_seen"] == 1
    assert report["input_staleness"]["packet_refs_missing"] == 0
    assert staleness_groups["input_staleness.data_packet.balance"]["max_ms"] == 1000
    assert staleness_groups["input_staleness.data_packet.open_orders"]["max_ms"] == 800
    assert staleness_groups["input_staleness.data_packet.positions"]["max_ms"] == 600
    assert staleness_groups["input_staleness.snapshot_to_rust"]["max_ms"] == 1000
    assert staleness_groups["input_staleness.ema_bundle_to_rust"]["max_ms"] == 500
    assert staleness_groups["input_staleness.ema_bundle_to_rust"]["trading_impact"] == (
        "blocks_indicator_readiness"
    )


def test_live_performance_report_input_staleness_handles_cycle_id_reuse(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=2000,
                data={"cycle_id": 1, "data_packets": []},
            ),
            _monitor_row(event_type="bot.started", seq=2, ts=10_000),
            _monitor_row(
                event_type="cycle.started",
                seq=3,
                ts=11_000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="ema.bundle.completed",
                seq=4,
                ts=11_500,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=5,
                ts=12_000,
                ids={"cycle_id": "cy_1"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    staleness_groups = {
        group["operation"]: group
        for group in report["input_staleness"]["groups"]
    }

    assert report["input_staleness"]["rust_calls_missing_snapshot"] == 1
    assert "input_staleness.snapshot_to_rust" not in staleness_groups
    assert staleness_groups["input_staleness.ema_bundle_to_rust"]["max_ms"] == 500


def test_live_performance_report_summary_includes_bounded_input_staleness(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=2000,
                data={"cycle_id": 1, "data_packets": []},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=2,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert summary["input_staleness"]["snapshots_seen"] == 1
    assert summary["input_staleness"]["rust_calls_seen"] == 1
    assert summary["input_staleness"]["rust_calls_missing_ema"] == 1
    assert summary["input_staleness"]["total_groups"] == 1
    assert len(summary["input_staleness"]["groups"]) == 1


def test_live_performance_report_startup_readiness_summary(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=2000,
                component="bot.startup",
                data={"stage": "account", "elapsed_ms": 900},
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=3,
                ts=3000,
                component="bot.startup",
                data={"stage": "hsl", "elapsed_s": 2.5},
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=4,
                ts=4000,
                component="risk.hsl",
                status="started",
                data={
                    "signal_mode": "coin",
                    "stage": "pair_replay",
                    "pairs": 26,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 20,
                    "timeline_rows": 43201,
                    "applied_rows": 3000,
                    "total_applied_rows": 4425,
                    "rows_per_second": 289.4,
                    "elapsed_s": 15.2,
                },
            ),
            _monitor_row(event_type="bot.ready", seq=5, ts=5000),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]

    assert startup["bot_count"] == 1
    assert startup["ready_count"] == 1
    assert startup["hsl_replay_active_count"] == 1
    bot = startup["bots"][0]
    assert bot["bot"] == "binance/binance_01"
    assert bot["lifecycle_status"] == "ready"
    assert bot["bot_started_ts"] == 1000
    assert bot["bot_ready_ts"] == 5000
    assert bot["startup_phases_ms"] == {"account": 900, "hsl": 2500}
    assert bot["hsl_replay"]["stage"] == "pair_replay"
    assert bot["hsl_replay"]["pairs"] == 26
    assert bot["hsl_replay"]["held_pairs"] == 1
    assert bot["hsl_replay"]["rows_per_second"] == 289.4


def test_live_performance_report_startup_readiness_completed_hsl_not_active(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=2,
                ts=2000,
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "pairs": 2,
                    "skipped_pairs": 1,
                    "full_elapsed_s": 12.3,
                    "startup_blocking_elapsed_s": 12.3,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]

    assert startup["ready_count"] == 0
    assert startup["hsl_replay_active_count"] == 0
    assert startup["bots"][0]["hsl_replay"]["status"] == "succeeded"
    assert startup["bots"][0]["hsl_replay"]["skipped_pairs"] == 1


def test_live_performance_report_startup_readiness_resets_on_restart(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=1100,
                data={"stage": "account", "elapsed_ms": 300},
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=3,
                ts=1200,
                status="succeeded",
                data={"stage": "full_replay", "pairs": 2, "full_elapsed_s": 1.5},
            ),
            _monitor_row(event_type="bot.ready", seq=4, ts=1300),
            _monitor_row(event_type="bot.started", seq=5, ts=2000),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]

    assert startup["ready_count"] == 0
    assert startup["hsl_replay_active_count"] == 0
    bot = startup["bots"][0]
    assert bot["lifecycle_status"] == "started"
    assert bot["bot_started_ts"] == 2000
    assert "bot_ready_ts" not in bot
    assert "startup_phases_ms" not in bot
    assert "hsl_replay" not in bot


def test_live_performance_report_startup_readiness_is_bounded(tmp_path):
    for index in range(3):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="bot.started",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                ),
                _monitor_row(
                    event_type="bot.startup_timing",
                    seq=index + 10,
                    ts=1100 + index,
                    user=f"user_{index}",
                    data={"stage": "account", "elapsed_ms": 100},
                ),
                _monitor_row(
                    event_type="bot.startup_timing",
                    seq=index + 20,
                    ts=1200 + index,
                    user=f"user_{index}",
                    data={"stage": "hsl", "elapsed_ms": 200},
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor", group_limit=1)
    startup = report["startup_readiness"]

    assert startup["bot_count"] == 3
    assert startup["bots_truncated"] is True
    assert len(startup["bots"]) == 1
    assert startup["bots"][0]["startup_phases_truncated"] is True
    assert list(startup["bots"][0]["startup_phases_ms"]) == ["account"]


def test_live_performance_report_startup_readiness_hsl_whitelist(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2000,
                status="started",
                data={
                    "stage": "pair_replay",
                    "pairs": 1,
                    "equity": 12345.0,
                    "balance": 6789.0,
                    "raw_payload": {"secret": "nope"},
                    "price": 42.0,
                    "drawdown": 0.12,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    hsl_replay = report["startup_readiness"]["bots"][0]["hsl_replay"]

    assert hsl_replay == {
        "latest_ts": 2000,
        "event_type": "hsl.replay.progress",
        "status": "started",
        "reason_code": "test",
        "stage": "pair_replay",
        "pairs": 1,
    }


def test_live_performance_report_summary_includes_startup_readiness(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(event_type="bot.ready", seq=2, ts=2000),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report)

    assert summary["startup_readiness"]["bot_count"] == 1
    assert summary["startup_readiness"]["ready_count"] == 1


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


def test_live_performance_report_cli_summary_and_filters(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                exchange="binance",
                user="binance_01",
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={"elapsed_ms": 2000},
            ),
        ],
    )

    rc = live_performance_report.main(
        [
            str(tmp_path / "monitor"),
            "--summary",
            "--compact",
            "--exchange",
            "okx",
            "--group-limit",
            "1",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["live_events"] == 1
    assert out["filters"]["exchanges"] == ["okx"]
    assert out["filters"]["events_skipped"] == 1
    assert len(out["performance"]["groups"]) == 1
    assert "files" not in out


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
