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
    side: str | None = None,
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
        "side": side,
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
    if side is not None:
        row["side"] = side
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


def _operation_duration_groups_by_operation(report):
    return {
        group["operation"]: group
        for group in report["operation_durations"]["groups"]
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

    operation_durations = report["operation_durations"]
    duration_groups = _operation_duration_groups_by_operation(report)
    assert operation_durations["total_groups"] >= len(groups)
    assert operation_durations["operation_category_counts"]["cycle"] >= 2
    assert operation_durations["operation_category_counts"]["hsl_replay"] == 1
    assert operation_durations["blocking_scope_counts"]["delays_protective_readiness"] == 1
    assert duration_groups["hsl_replay.pair_replay.elapsed"]["source_section"] == "performance"
    assert duration_groups["hsl_replay.pair_replay.elapsed"]["operation_category"] == (
        "hsl_replay"
    )
    assert duration_groups["remote_call.authoritative.open_orders"]["blocking_scope"] == (
        "delays_exchange_actions"
    )


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
    assert summary["operation_durations"]["total_groups"] == 3
    assert len(summary["operation_durations"]["groups"]) == 1
    assert "files" not in summary
    assert "event_types" not in summary


def test_live_performance_report_slowest_blockers(tmp_path):
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
                event_type="cycle.completed",
                seq=2,
                ts=63_000,
                ids={"cycle_id": "cy_1"},
                data={
                    "elapsed_ms": 2000,
                    "timings_ms": {"execute": 1100, "monitor_flush": 9000},
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=3,
                ts=64_000,
                component="risk.hsl",
                data={"stage": "pair_replay", "elapsed_s": 12.0},
            ),
            _monitor_row(
                event_type="data_packet.updated",
                seq=4,
                ts=65_000,
                data={
                    "kind": "balance",
                    "revision": 1,
                    "response_received_ts_ms": 60_000,
                },
            ),
            _monitor_row(
                event_type="snapshot.built",
                seq=5,
                ts=66_000,
                data={"cycle_id": 2, "data_packets": [{"kind": "balance", "revision": 1}]},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    blockers = report["slowest_blockers"]
    operations = [group["operation"] for group in blockers["groups"]]

    assert blockers["total_groups"] >= 4
    assert operations[0] == "hsl_replay.pair_replay.elapsed"
    assert "cycle.phase.monitor_flush" not in operations
    hsl_group = blockers["groups"][0]
    assert hsl_group["source_section"] == "performance"
    assert hsl_group["blocking_scope"] == "delays_protective_readiness"
    assert hsl_group["p95_ms"] == 12000


def test_live_performance_report_slowest_blockers_summary_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                data={"elapsed_ms": 1000, "timings_ms": {"execute": 900}},
            ),
            _monitor_row(
                event_type="state.refresh_timing",
                seq=2,
                ts=2000,
                data={"wall_ms": 2000, "surface_max_ms": 1500},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert summary["slowest_blockers"]["total_groups"] == 4
    assert len(summary["slowest_blockers"]["groups"]) == 1
    assert summary["slowest_blockers"]["groups"][0]["operation"] == "state_refresh.wall"


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
                    "surface_ages": [
                        {
                            "name": "balance",
                            "age_ms": 900,
                            "epoch": 3,
                            "min_epoch": 3,
                        },
                        {
                            "name": "completed_candles",
                            "age_ms": 1500,
                            "epoch": 2,
                            "min_epoch": 2,
                        },
                    ],
                    "market_snapshot_summary": {
                        "count": 2,
                        "symbol_count": 2,
                        "missing_count": 0,
                        "max_age_ms": 700,
                        "mean_age_ms": 450,
                        "configured_max_age_ms": 600,
                        "sources": ["ticker"],
                        "sources_count": 1,
                    },
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
    assert report["input_staleness"]["snapshot_surface_age_rows"] == 2
    assert report["input_staleness"]["snapshot_market_summaries_seen"] == 1
    assert report["input_staleness"]["snapshot_market_stale_count"] == 1
    assert report["input_staleness"]["rust_calls_seen"] == 1
    assert report["input_staleness"]["packet_refs_missing"] == 0
    assert report["input_staleness"]["snapshot_to_rust_exact_matches"] == 0
    assert report["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 1
    assert staleness_groups["input_staleness.surface.balance"]["max_ms"] == 900
    assert staleness_groups["input_staleness.surface.completed_candles"]["max_ms"] == 1500
    assert staleness_groups["input_staleness.surface.completed_candles"][
        "trading_impact"
    ] == "blocks_indicator_readiness"
    assert staleness_groups["input_staleness.market_snapshot.max"]["max_ms"] == 700
    assert staleness_groups["input_staleness.market_snapshot.mean"]["max_ms"] == 450
    assert staleness_groups["input_staleness.market_snapshot.configured_excess"][
        "max_ms"
    ] == 100
    assert staleness_groups["input_staleness.market_snapshot.configured_excess"][
        "timing_kind"
    ] == "configured_age_excess"
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
                data={
                    "cycle_id": 1,
                    "surface_ages": [{"name": "balance", "age_ms": 500}],
                    "market_snapshot_summary": {"max_age_ms": 250, "mean_age_ms": 125},
                    "data_packets": [],
                },
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
    assert report["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 0
    assert "input_staleness.snapshot_to_rust" not in staleness_groups
    assert staleness_groups["input_staleness.ema_bundle_to_rust"]["max_ms"] == 500


def test_live_performance_report_input_staleness_prefers_envelope_cycle_id(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=2000,
                ids={"cycle_id": "cy_9"},
                data={
                    "cycle_id": 1,
                    "surface_ages": [],
                    "market_snapshot_summary": {},
                    "data_packets": [],
                },
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=2,
                ts=2600,
                ids={"cycle_id": "cy_9"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    staleness_groups = {
        group["operation"]: group
        for group in report["input_staleness"]["groups"]
    }

    assert report["input_staleness"]["snapshot_to_rust_exact_matches"] == 1
    assert report["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 0
    assert report["input_staleness"]["rust_calls_missing_snapshot"] == 0
    assert staleness_groups["input_staleness.snapshot_to_rust"]["max_ms"] == 600


def test_live_performance_report_input_staleness_uses_latest_legacy_snapshot(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=1000,
                data={
                    "cycle_id": 1,
                    "surface_ages": [],
                    "market_snapshot_summary": {},
                    "data_packets": [],
                },
            ),
            _monitor_row(
                event_type="snapshot.built",
                seq=2,
                ts=2400,
                data={
                    "cycle_id": 2,
                    "surface_ages": [],
                    "market_snapshot_summary": {},
                    "data_packets": [],
                },
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=3,
                ts=2700,
                ids={"cycle_id": "cy_99"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    staleness_groups = {
        group["operation"]: group
        for group in report["input_staleness"]["groups"]
    }

    assert report["input_staleness"]["snapshot_to_rust_exact_matches"] == 0
    assert report["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 1
    assert report["input_staleness"]["rust_calls_missing_snapshot"] == 0
    assert staleness_groups["input_staleness.snapshot_to_rust"]["max_ms"] == 300


def test_live_performance_report_summary_includes_bounded_input_staleness(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=2000,
                data={
                    "cycle_id": 1,
                    "surface_ages": [{"name": "balance", "age_ms": 500}],
                    "market_snapshot_summary": {
                        "max_age_ms": 250,
                        "mean_age_ms": 125,
                        "configured_max_age_ms": 200,
                    },
                    "data_packets": [],
                },
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
    assert summary["input_staleness"]["snapshot_surface_age_rows"] == 1
    assert summary["input_staleness"]["snapshot_market_summaries_seen"] == 1
    assert summary["input_staleness"]["snapshot_market_stale_count"] == 1
    assert summary["input_staleness"]["rust_calls_seen"] == 1
    assert summary["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 1
    assert summary["input_staleness"]["rust_calls_missing_ema"] == 1
    assert summary["input_staleness"]["total_groups"] == 5
    assert len(summary["input_staleness"]["groups"]) == 1


def test_live_performance_report_startup_readiness_summary(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.started",
                seq=1,
                ts=1000,
                data={
                    "live_event_debug_profiles": [
                        "rust",
                        "ema",
                        "api_key=should_not_render",
                    ]
                },
            ),
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
    assert bot["debug_profiles"] == ["ema", "rust"]
    assert startup["debug_profile_counts"] == {"ema": 1, "rust": 1}
    assert "api_key=should_not_render" not in json.dumps(startup, sort_keys=True)
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
            _monitor_row(
                event_type="bot.started",
                seq=5,
                ts=2000,
                data={"live_event_debug_profiles": ["state"]},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]

    assert startup["ready_count"] == 0
    assert startup["hsl_replay_active_count"] == 0
    bot = startup["bots"][0]
    assert bot["lifecycle_status"] == "started"
    assert bot["bot_started_ts"] == 2000
    assert bot["debug_profiles"] == ["state"]
    assert startup["debug_profile_counts"] == {"state": 1}
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


def test_live_performance_report_hsl_replay_profile(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=1,
                ts=1000,
                component="risk.hsl",
                status="started",
                reason_code="coin_history_replay",
                data={"signal_mode": "coin", "lookback_days": 30},
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2000,
                component="risk.hsl",
                status="started",
                reason_code="history_loaded",
                data={
                    "signal_mode": "coin",
                    "stage": "loaded",
                    "symbols": 5,
                    "pairs": 4,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 3,
                    "timeline_rows": 10,
                    "fill_events": 7,
                    "panic_events": 2,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=3,
                ts=3000,
                component="risk.hsl",
                status="started",
                reason_code="pair_replay_progress",
                symbol="XLM/USDT:USDT",
                pside="long",
                data={
                    "signal_mode": "coin",
                    "stage": "pair_replay",
                    "pair_idx": 2,
                    "pairs": 4,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 3,
                    "timeline_rows": 10,
                    "applied_rows": 8,
                    "total_applied_rows": 12,
                    "rows_per_second": 123.4567,
                    "is_held_pair": True,
                    "is_cooldown_pair": False,
                    "elapsed_s": 2.5,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=4,
                ts=4000,
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "pairs": 4,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 3,
                    "timeline_rows": 10,
                    "rows": 30,
                    "applied_rows": 30,
                    "skipped_pairs": 1,
                    "rows_per_second": 40,
                    "full_elapsed_s": 7.5,
                    "startup_blocking_elapsed_s": 7.5,
                    "elapsed_s": 7.5,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.failed",
                seq=5,
                ts=5000,
                component="risk.hsl",
                status="failed",
                reason_code="coin_history_replay_failed",
                data={
                    "signal_mode": "coin",
                    "elapsed_s": 8.0,
                    "error_type": "RuntimeError",
                    "secret": "must-not-render",
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    profile = report["hsl_replay_profile"]
    group = profile["groups"][0]

    assert profile["total_events"] == 5
    assert profile["bot_count"] == 1
    assert profile["event_types"] == {
        "hsl.replay.started": 1,
        "hsl.replay.progress": 2,
        "hsl.replay.completed": 1,
        "hsl.replay.failed": 1,
    }
    assert group["bot"] == "binance/binance_01"
    assert group["event_types"]["hsl.replay.progress"] == 2
    assert group["loaded"]["data"]["symbols"] == 5
    assert group["progress"]["data"]["is_held_pair"] is True
    assert group["progress"]["derived"]["observed_work_pct"] == 30
    assert group["progress"]["derived"]["estimated_dense_pair_row_work"] == 40
    assert group["progress"]["derived"]["estimated_held_pair_row_work"] == 10
    assert group["progress"]["derived"]["estimated_required_pair_row_work"] == 30
    assert group["progress"]["derived"]["latest_elapsed_ms"] == 2500
    assert group["completed"]["derived"]["startup_blocking_elapsed_ms"] == 7500
    assert group["completed"]["derived"]["startup_blocking"] is True
    assert group["failed"]["event_type"] == "hsl.replay.failed"
    assert group["failed"]["status"] == "failed"
    assert group["failed"]["derived"]["latest_elapsed_ms"] == 8000
    assert "must-not-render" not in json.dumps(group, sort_keys=True)
    assert group["completed"]["derived"]["observed_work_pct"] == 75


def test_live_performance_report_hsl_replay_profile_whitelists_values(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=1,
                ts=1000,
                component="risk.hsl",
                data={
                    "stage": "price_history_symbol_fetch_completed",
                    "timeframe": "1m",
                    "error_type": "TimeoutError",
                    "events": 2,
                    "price_replay_symbols": 3,
                    "history_minutes": 10,
                    "rows": 9,
                    "elapsed_s": 4.5,
                    "balance": 1000.0,
                    "equity": 999.0,
                    "raw_payload": {"leak_marker": "raw"},
                    "api_key": "secret",
                    "drawdown_raw": 0.1,
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    rendered = json.dumps(report["hsl_replay_profile"], sort_keys=True)

    assert report["hsl_replay_profile"]["groups"][0]["progress"]["data"] == {
        "elapsed_s": 4.5,
        "error_type": "TimeoutError",
        "events": 2,
        "history_minutes": 10,
        "price_replay_symbols": 3,
        "rows": 9,
        "stage": "price_history_symbol_fetch_completed",
        "timeframe": "1m",
    }
    assert "balance" not in rendered
    assert "equity" not in rendered
    assert "raw_payload" not in rendered
    assert "leak_marker" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert "drawdown_raw" not in rendered


def test_live_performance_report_hsl_replay_profile_summary_is_bounded(tmp_path):
    for index in range(2):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="hsl.replay.progress",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                    component="risk.hsl",
                    data={"stage": "loaded", "pairs": 1, "timeline_rows": 10},
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["hsl_replay_profile"]["bot_count"] == 2
    assert len(summary["hsl_replay_profile"]["groups"]) == 1
    assert summary["hsl_replay_profile"]["groups_truncated"] is True


def test_live_performance_report_cache_warmup_from_existing_events(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cache.warmup_decision",
                seq=1,
                ts=1000,
                component="cache.warmup",
                reason_code="warmup_cache_decision",
                data={
                    "context": "startup_trading_ready",
                    "timeframe": "1m",
                    "symbol_count": 5,
                    "reused_count": 3,
                    "cold_count": 2,
                    "cold_path_required": True,
                    "reason_counts": {"warm_cache_accepted": 3, "missing_coverage": 2},
                    "elapsed_ms": 123,
                    "concurrency": 2,
                    "ttl_ms": 300000,
                },
            ),
            _monitor_row(
                event_type="cache.load.completed",
                seq=2,
                ts=2000,
                component="cache.candles",
                reason_code="candle_disk_load_completed",
                symbol="XLM/USDT:USDT",
                data={
                    "timeframe": "1m",
                    "loaded_rows": 120,
                    "loaded_start_ts": 100,
                    "loaded_end_ts": 200,
                    "days": 2,
                    "source_days": {"primary": 1, "legacy": 1},
                    "elapsed_ms": 45,
                },
            ),
            _monitor_row(
                event_type="cache.flush.completed",
                seq=3,
                ts=3000,
                component="cache.candles",
                reason_code="candle_disk_flush_completed",
                symbol="XLM/USDT:USDT",
                data={
                    "timeframe": "1m",
                    "persisted_rows": 80,
                    "persisted_start_ts": 150,
                    "persisted_end_ts": 250,
                    "suppressed_count": 1,
                    "suppressed_rows": 4,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    cache = report["cache_warmup"]
    group = cache["groups"][0]
    operations = _groups_by_operation(report)

    assert cache["total_events"] == 3
    assert cache["bot_count"] == 1
    assert cache["event_types"] == {
        "cache.warmup_decision": 1,
        "cache.load.completed": 1,
        "cache.flush.completed": 1,
    }
    assert group["bot"] == "binance/binance_01"
    assert group["warmup"]["contexts"] == {"startup_trading_ready": 1}
    assert group["warmup"]["reason_counts"] == {
        "warm_cache_accepted": 3,
        "missing_coverage": 2,
    }
    assert group["warmup"]["symbol_count"] == 5
    assert group["warmup"]["reused_count"] == 3
    assert group["warmup"]["cold_count"] == 2
    assert group["warmup"]["cold_path_decisions"] == 1
    assert group["warmup"]["elapsed_ms"]["max"] == 123
    assert group["load"]["loaded_rows"] == 120
    assert group["load"]["source_days"] == {"primary": 1, "legacy": 1}
    assert group["load"]["elapsed_ms"]["max"] == 45
    assert group["flush"]["persisted_rows"] == 80
    assert group["flush"]["suppressed_events"] == 1
    assert group["flush"]["suppressed_rows"] == 4
    assert (
        operations["cache_warmup_decision"]["trading_impact"]
        == "blocks_indicator_readiness"
    )
    assert (
        operations["cache_load_completed"]["trading_impact"]
        == "blocks_indicator_readiness"
    )


def test_live_performance_report_cache_warmup_whitelists_values(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cache.load.completed",
                seq=1,
                ts=1000,
                component="cache.candles",
                data={
                    "timeframe": "1m",
                    "loaded_rows": 10,
                    "path": "/home/operator/secret/cache.json",
                    "cache_path": "/root/passivbot/caches/private",
                    "raw_payload": {"leak_marker": "raw"},
                    "api_key": "secret",
                    "balance": 1000.0,
                    "equity": 999.0,
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    rendered = json.dumps(report["cache_warmup"], sort_keys=True)

    assert report["cache_warmup"]["groups"][0]["load"]["latest"]["data"] == {
        "loaded_rows": 10,
        "timeframe": "1m",
    }
    assert "/home/operator" not in rendered
    assert "/root/passivbot" not in rendered
    assert "secret" not in rendered


def test_live_performance_report_forager_ema_readiness_from_existing_events(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="forager.selection",
                seq=1,
                ts=1000,
                component="forager.selection",
                pside="long",
                reason_code="selected",
                data={
                    "candidate_count": 40,
                    "eligible_count": 38,
                    "selected_count": 3,
                    "selected_symbols": ["XLM/USDT:USDT", "ZEC/USDT:USDT"],
                    "incumbent_count": 1,
                    "incumbent_symbols": ["XRP/USDT:USDT"],
                    "slots_open": True,
                    "max_n_positions": 3,
                    "slots_to_fill": 1,
                    "max_age_ms": 300000,
                    "fetch_budget": 4,
                    "source": "python_filter",
                    "feature_unavailable_count": 7,
                    "volatility_dropped_count": 2,
                    "hysteresis_event_count": 1,
                },
            ),
            _monitor_row(
                event_type="forager.feature_unavailable",
                seq=2,
                ts=2000,
                component="forager.selection",
                pside="long",
                status="skipped",
                reason_code="ranking_features_unavailable",
                data={
                    "candidate_count": 40,
                    "unavailable": ["NEAR/USDT:USDT", "ATOM/USDT:USDT"],
                    "volume_count": 35,
                    "log_range_count": 34,
                    "max_age_ms": 300000,
                    "fetch_budget": 4,
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=3,
                ts=3000,
                component="ema.bundle",
                status="degraded",
                reason_code="required_ema_unavailable",
                data={
                    "optional_drop_count": 2,
                    "optional_drop_groups": [
                        {
                            "ema_type": "m1_volume",
                            "reason": "optional stale",
                            "symbols": ["DOGE/USDT:USDT"],
                            "spans": [100],
                        }
                    ],
                    "candidate_unavailable": ["NEAR/USDT:USDT"],
                    "candidate_unavailable_groups": [
                        {
                            "reason": "missing feature basis",
                            "symbols": ["NEAR/USDT:USDT"],
                            "error_types": ["ValueError"],
                            "example_error": "raw detail omitted",
                        }
                    ],
                    "unavailable": ["ATOM/USDT:USDT"],
                    "unavailable_reasons": [
                        {"reason": "missing required m1_close EMA", "symbols": ["ATOM/USDT:USDT"]}
                    ],
                },
            ),
            _monitor_row(
                event_type="ema.fallback_used",
                seq=4,
                ts=4000,
                component="ema.bundle",
                status="recovered",
                reason_code="ema_fallback_used",
                data={
                    "close_recovered_count": 3,
                    "close_recovered_symbols": ["XLM/USDT:USDT"],
                    "close_fallback_count": 1,
                    "close_fallback_symbols": ["ZEC/USDT:USDT"],
                    "forager_cached_fallback_count": 2,
                    "forager_cached_fallback_symbols": ["NEAR/USDT:USDT"],
                    "examples": {"raw": "not copied"},
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    readiness = report["forager_ema_readiness"]
    group = readiness["groups"][0]

    assert readiness["total_events"] == 4
    assert readiness["event_types"] == {
        "ema.fallback_used": 1,
        "ema.unavailable": 1,
        "forager.feature_unavailable": 1,
        "forager.selection": 1,
    }
    assert group["bot"] == "binance/binance_01"
    assert group["psides"] == {"long": 2}
    assert group["forager_selection"]["candidate_count"]["max"] == 40
    assert group["forager_selection"]["selected_count"]["max"] == 3
    assert group["forager_selection"]["feature_unavailable_count"]["max"] == 7
    assert group["forager_selection"]["selected_symbols"] == {
        "count": 2,
        "sample": ["XLM/USDT:USDT", "ZEC/USDT:USDT"],
    }
    assert group["forager_feature_unavailable"]["unavailable_symbols"] == {
        "count": 2,
        "sample": ["NEAR/USDT:USDT", "ATOM/USDT:USDT"],
    }
    assert group["forager_feature_unavailable"]["volume_count"]["max"] == 35
    assert group["forager_feature_unavailable"]["log_range_count"]["max"] == 34
    assert group["ema_unavailable"]["optional_drop_count"] == 2
    assert group["ema_unavailable"]["candidate_symbol_sample_count"] == 1
    assert group["ema_unavailable"]["unavailable_symbol_sample_count"] == 1
    assert group["ema_unavailable"]["candidate_reasons"] == {"missing feature basis": 1}
    assert group["ema_unavailable"]["unavailable_reasons"] == {
        "missing required m1_close EMA": 1
    }
    assert group["ema_unavailable"]["optional_drop_reasons"] == {"optional stale": 1}
    assert group["ema_unavailable"]["error_types"] == {"ValueError": 1}
    assert group["ema_fallback"]["close_recovered_count"] == 3
    assert group["ema_fallback"]["close_fallback_count"] == 1
    assert group["ema_fallback"]["forager_cached_fallback_count"] == 2
    assert group["latest"]["event_type"] == "ema.fallback_used"


def test_live_performance_report_forager_ema_readiness_whitelists_values(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="forager.selection",
                seq=1,
                ts=1000,
                component="forager.selection",
                data={
                    "candidate_count": 1,
                    "selected_symbols": ["XLM/USDT:USDT"],
                    "top_scores": [{"symbol": "XLM/USDT:USDT", "score": 0.9}],
                    "path": "/home/operator/private/cache.json",
                    "raw_payload": {"leak_marker": "raw"},
                    "api_key": "secret",
                    "balance": 1000,
                    "equity": 999,
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=2,
                ts=2000,
                component="ema.bundle",
                data={
                    "candidate_unavailable": ["XLM/USDT:USDT"],
                    "candidate_unavailable_groups": [
                        {
                            "reason": "missing feature basis",
                            "symbols": ["XLM/USDT:USDT"],
                            "error_types": ["RuntimeError"],
                            "example_error": "raw exception with /root/passivbot secret",
                        }
                    ],
                    "debug": {"raw": "not copied"},
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    rendered = json.dumps(report["forager_ema_readiness"], sort_keys=True)

    assert "top_scores" not in rendered
    assert "score" not in rendered
    assert "/home/operator" not in rendered
    assert "/root/passivbot" not in rendered
    assert "raw_payload" not in rendered
    assert "leak_marker" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert "balance" not in rendered
    assert "equity" not in rendered
    assert "example_error" not in rendered
    assert "raw exception" not in rendered


def test_live_performance_report_forager_ema_readiness_summary_is_bounded(tmp_path):
    for index in range(2):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="forager.selection",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                    component="forager.selection",
                    data={"candidate_count": 10, "selected_count": 1},
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["forager_ema_readiness"]["bot_count"] == 2
    assert len(summary["forager_ema_readiness"]["groups"]) == 1
    assert summary["forager_ema_readiness"]["groups_truncated"] is True


def test_live_performance_report_cache_warmup_summary_is_bounded(tmp_path):
    for index in range(2):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="cache.warmup_decision",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                    component="cache.warmup",
                    data={
                        "context": "startup_trading_ready",
                        "symbol_count": 1,
                        "reused_count": index,
                        "cold_count": 1 - index,
                    },
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["cache_warmup"]["bot_count"] == 2
    assert len(summary["cache_warmup"]["groups"]) == 1
    assert summary["cache_warmup"]["groups_truncated"] is True


def test_live_performance_report_execution_timing_pairs_order_events(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="order_wave.started",
                seq=1,
                ts=1000,
                status="started",
                component="order_wave",
                ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
                data={"planned_create": 1, "planned_cancel": 1},
            ),
            _monitor_row(
                event_type="execution.create_sent",
                seq=2,
                ts=1100,
                status="started",
                component="execution.order_write",
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "ow_1",
                    "action_id": "ow_1:create:0",
                },
                symbol="BTC/USDT:USDT",
                pside="long",
                data={"client_order_id_short": "cid-short"},
            ),
            _monitor_row(
                event_type="execution.create_succeeded",
                seq=3,
                ts=1450,
                status="succeeded",
                component="execution.order_write",
                reason_code="exchange_acknowledged",
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "ow_1",
                    "action_id": "ow_1:create:0",
                },
                symbol="BTC/USDT:USDT",
                pside="long",
                data={"raw_order_payload": {"secret": "must-not-leak"}},
            ),
            _monitor_row(
                event_type="execution.cancel_sent",
                seq=4,
                ts=1500,
                status="started",
                component="execution.order_write",
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "ow_1",
                    "action_id": "ow_1:cancel:0",
                },
                symbol="ETH/USDT:USDT",
                pside="short",
            ),
            _monitor_row(
                event_type="execution.cancel_ambiguous_terminal",
                seq=5,
                ts=2100,
                status="degraded",
                component="execution.order_write",
                reason_code="requires_full_authoritative_confirmation",
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "ow_1",
                    "action_id": "ow_1:cancel:0",
                },
                symbol="ETH/USDT:USDT",
                pside="short",
            ),
            _monitor_row(
                event_type="execution.confirmation_requested",
                seq=6,
                ts=2200,
                status="started",
                component="execution.confirmation",
                ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
            ),
            _monitor_row(
                event_type="execution.confirmation_satisfied",
                seq=7,
                ts=3400,
                status="succeeded",
                component="execution.confirmation",
                ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
                data={"confirm_ms": 975, "elapsed_ms": 1200},
            ),
            _monitor_row(
                event_type="order_wave.completed",
                seq=8,
                ts=3500,
                status="succeeded",
                component="order_wave",
                ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
                data={"elapsed_ms": 2450},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    execution = report["execution_timing"]
    groups = {
        group["operation"]: group
        for group in execution["groups"]
    }
    rendered = json.dumps(execution, sort_keys=True)

    assert execution["total_events"] == 8
    assert execution["timing_observations"] == {
        "execution.cancel_response": 1,
        "execution.confirmation": 1,
        "execution.create_response": 1,
        "order_wave.total": 1,
    }
    assert execution["terminal_outcome_counts"] == {
        "cancel.ambiguous_terminal": 1,
        "confirmation.satisfied": 1,
        "create.succeeded": 1,
    }
    assert execution["pending_start_counts"] == {}
    assert groups["execution.create_response"]["max_ms"] == 350
    assert groups["execution.cancel_response"]["max_ms"] == 600
    assert groups["execution.confirmation"]["max_ms"] == 975
    assert groups["order_wave.total"]["max_ms"] == 2450
    assert groups["execution.cancel_response"]["statuses"] == {"degraded": 1}
    assert groups["execution.create_response"]["symbols_sample"] == ["BTC/USDT:USDT"]
    assert "must-not-leak" not in rendered
    assert "raw_order_payload" not in rendered
    assert "ow_1:create:0" not in rendered
    assert "cid-short" not in rendered


def test_live_performance_report_execution_timing_counts_missing_and_unpaired_ids(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                seq=1,
                ts=1000,
                status="started",
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="execution.create_failed",
                seq=2,
                ts=1200,
                status="failed",
                ids={"cycle_id": "cy_1", "action_id": "missing_start"},
            ),
            _monitor_row(
                event_type="execution.cancel_sent",
                seq=3,
                ts=1300,
                status="started",
                ids={"cycle_id": "cy_1", "action_id": "pending_cancel"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    execution = report["execution_timing"]

    assert execution["total_events"] == 3
    assert execution["terminal_outcome_counts"] == {"create.failed": 1}
    assert execution["missing_id_counts"] == {"execution.create_response": 1}
    assert execution["unpaired_terminal_counts"] == {"execution.create_response": 1}
    assert execution["pending_start_counts"] == {"execution.write_response": 1}
    assert execution["total_groups"] == 0


def test_live_performance_report_execution_timing_summary_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="order_wave.completed",
                seq=1,
                ts=1000,
                component="order_wave",
                ids={"order_wave_id": "ow_1"},
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="execution.confirmation_satisfied",
                seq=2,
                ts=2000,
                component="execution.confirmation",
                ids={"order_wave_id": "ow_1"},
                data={"confirm_ms": 2000},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert summary["execution_timing"]["total_groups"] == 2
    assert summary["execution_timing"]["groups_truncated"] is True
    assert len(summary["execution_timing"]["groups"]) == 1
    assert summary["operation_durations"]["total_groups"] == 2
    assert summary["operation_durations"]["operation_category_counts"] == {"execution": 2}
    assert summary["operation_durations"]["blocking_scope_counts"] == {"exchange_io": 2}
    assert summary["slowest_blockers"]["total_groups"] >= 2


def test_live_performance_report_account_state_changes_are_value_safe(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fill.ingested",
                seq=1,
                ts=1000,
                component="fills.ingest",
                reason_code="new_fill",
                symbol="BTC/USDT:USDT",
                pside="long",
                side="buy",
                data={
                    "fill_id_hash": "hash-must-not-surface",
                    "client_order_id_short": "cid-must-not-surface",
                    "qty": 1.2345,
                    "price": 45678.9,
                    "pnl": -12.34,
                    "fee": 0.56,
                    "raw_payload": {"secret_marker": "raw-fill-secret"},
                },
            ),
            _monitor_row(
                event_type="position.changed",
                seq=2,
                ts=2000,
                component="account.position",
                reason_code="increased",
                symbol="BTC/USDT:USDT",
                pside="long",
                data={
                    "old_size": 0,
                    "new_size": 1.2345,
                    "new_price": 45678.9,
                    "upnl": -23.45,
                    "secret_marker": "position-secret",
                },
            ),
            _monitor_row(
                event_type="balance.changed",
                seq=3,
                ts=3000,
                component="account.balance",
                reason_code="balance_changed",
                data={
                    "balance_raw": 98765.43,
                    "balance_snapped": 98700,
                    "equity": 98600,
                    "secret_marker": "balance-secret",
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    changes = report["account_state_changes"]
    groups = {
        group["event_type"]: group
        for group in changes["groups"]
    }
    rendered = json.dumps(changes, sort_keys=True)

    assert changes["total_events"] == 3
    assert changes["event_types"] == {
        "balance.changed": 1,
        "fill.ingested": 1,
        "position.changed": 1,
    }
    assert changes["bot_count"] == 1
    assert groups["fill.ingested"]["symbols_sample"] == ["BTC/USDT:USDT"]
    assert groups["fill.ingested"]["psides"] == {"long": 1}
    assert groups["fill.ingested"]["sides"] == {"buy": 1}
    assert groups["balance.changed"]["components"] == {"account.balance": 1}
    assert "hash-must-not-surface" not in rendered
    assert "cid-must-not-surface" not in rendered
    assert "raw-fill-secret" not in rendered
    assert "position-secret" not in rendered
    assert "balance-secret" not in rendered
    assert "98765.43" not in rendered
    assert "45678.9" not in rendered


def test_live_performance_report_account_state_changes_summary_is_bounded(tmp_path):
    for index in range(2):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="position.changed",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                    component="account.position",
                    reason_code="position_changed",
                    symbol=f"COIN{index}/USDT:USDT",
                    pside="long",
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["account_state_changes"]["bot_count"] == 2
    assert len(summary["account_state_changes"]["groups"]) == 1
    assert len(summary["account_state_changes"]["bots"]) == 1
    assert summary["account_state_changes"]["groups_truncated"] is True
    assert summary["account_state_changes"]["bots_truncated"] is True


def test_live_performance_report_risk_activity_is_value_safe(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.status",
                seq=1,
                ts=1000,
                component="risk.hsl.status",
                status="degraded",
                reason_code="cooldown_active",
                symbol="BTC/USDT:USDT",
                pside="long",
                data={
                    "tier": "red",
                    "drawdown_score": 0.1234,
                    "strategy_equity": 98765.43,
                    "unrealized_pnl": -12.34,
                    "secret_marker": "hsl-secret",
                },
            ),
            _monitor_row(
                event_type="hsl.red_triggered",
                seq=2,
                ts=2000,
                component="risk.hsl",
                status="degraded",
                reason_code="red_threshold_crossed",
                symbol="BTC/USDT:USDT",
                pside="long",
                data={
                    "balance": 12345.67,
                    "peak_strategy_equity": 45678.9,
                    "raw_payload": {"secret_marker": "red-secret"},
                },
            ),
            _monitor_row(
                event_type="risk.mode_changed",
                seq=3,
                ts=3000,
                component="risk.hsl.mode",
                reason_code="hsl_halted",
                pside="long",
                data={
                    "mode": "halted",
                    "symbols": ["BTC/USDT:USDT"],
                    "secret_marker": "mode-secret",
                },
            ),
            _monitor_row(
                event_type="unstuck.selection",
                seq=4,
                ts=4000,
                component="risk.unstuck.selection",
                reason_code="unstuck_selection",
                symbol="ETH/USDT:USDT",
                pside="short",
                data={
                    "entry_price": 1111.22,
                    "current_price": 999.88,
                    "allowance": -0.123,
                    "secret_marker": "unstuck-secret",
                },
            ),
            _monitor_row(
                event_type="risk.realized_loss_gate_blocked",
                seq=5,
                ts=4500,
                component="risk.realized_loss_gate",
                status="deferred",
                reason_code="realized_loss_gate_blocked",
                symbol="SOL/USDT:USDT",
                pside="long",
                data={
                    "order_type": "close_auto_reduce_wel_long",
                    "projected_pnl": -200.0,
                    "projected_balance_after": 9800.0,
                    "balance_floor": 9900.0,
                    "secret_marker": "loss-gate-secret",
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=6,
                ts=5000,
                component="risk.hsl.replay",
                reason_code="replay_progress",
                symbol="XRP/USDT:USDT",
                pside="long",
                data={"rows": 100, "secret_marker": "replay-secret"},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    risk = report["risk_activity"]
    groups = {
        group["event_type"]: group
        for group in risk["groups"]
    }
    rendered = json.dumps(risk, sort_keys=True)

    assert risk["total_events"] == 5
    assert risk["event_types"] == {
        "hsl.red_triggered": 1,
        "hsl.status": 1,
        "risk.mode_changed": 1,
        "risk.realized_loss_gate_blocked": 1,
        "unstuck.selection": 1,
    }
    assert "hsl.replay.progress" not in risk["event_types"]
    assert groups["hsl.status"]["statuses"] == {"degraded": 1}
    assert groups["hsl.status"]["reason_codes"] == {"cooldown_active": 1}
    assert groups["hsl.status"]["symbols_sample"] == ["BTC/USDT:USDT"]
    assert groups["hsl.status"]["psides"] == {"long": 1}
    assert groups["unstuck.selection"]["symbols_sample"] == ["ETH/USDT:USDT"]
    assert groups["risk.realized_loss_gate_blocked"]["symbols_sample"] == ["SOL/USDT:USDT"]
    assert groups["risk.realized_loss_gate_blocked"]["statuses"] == {"deferred": 1}
    assert "98765.43" not in rendered
    assert "45678.9" not in rendered
    assert "1111.22" not in rendered
    assert "hsl-secret" not in rendered
    assert "red-secret" not in rendered
    assert "mode-secret" not in rendered
    assert "unstuck-secret" not in rendered
    assert "loss-gate-secret" not in rendered
    assert "9800" not in rendered
    assert "replay-secret" not in rendered


def test_live_performance_report_risk_activity_summary_is_bounded(tmp_path):
    for index in range(2):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="hsl.status",
                    seq=index + 1,
                    ts=1000 + index,
                    user=f"user_{index}",
                    component="risk.hsl.status",
                    status="degraded",
                    reason_code="cooldown_active",
                    symbol=f"COIN{index}/USDT:USDT",
                    pside="long",
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["risk_activity"]["bot_count"] == 2
    assert len(summary["risk_activity"]["groups"]) == 1
    assert len(summary["risk_activity"]["bots"]) == 1
    assert summary["risk_activity"]["groups_truncated"] is True
    assert summary["risk_activity"]["bots_truncated"] is True


def test_live_performance_report_resource_pressure_from_health_summary(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                component="monitor.health",
                data={
                    "rss_bytes": 1000,
                    "memory_percent": 5.5,
                    "open_fds": 11,
                    "loadavg_1m": 0.25,
                    "cpu_count": 1,
                    "event_queue_depth": 3,
                    "event_dropped_total": 1,
                    "event_sink_error_total": 0,
                    "event_degraded_count": 2,
                    "event_drop_counts": {"queue_full": 1},
                    "event_sink_error_counts": {"disk": 0},
                    "event_pipeline_stopping": False,
                    "event_pipeline_worker_alive": True,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                component="monitor.health",
                data={
                    "rss_bytes": 1500,
                    "memory_percent": 6.5,
                    "open_fds": 13,
                    "loadavg_1m": 0.75,
                    "cpu_count": 1,
                    "event_queue_depth": 5,
                    "event_dropped_total": 4,
                    "event_sink_error_total": 1,
                    "event_degraded_count": 3,
                    "event_drop_counts": {"queue_full": 4},
                    "event_sink_error_counts": {"disk": 1},
                    "event_pipeline_stopping": False,
                    "event_pipeline_worker_alive": True,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    pressure = report["resource_pressure"]
    group = pressure["groups"][0]

    assert pressure["total"] == 2
    assert pressure["bots"] == 1
    assert pressure["event_types"] == {"health.summary": 2}
    assert group["bot"] == "binance/binance_01"
    assert group["count"] == 2
    assert group["latest_ts"] == 2000
    assert group["fields"]["rss_bytes"] == {
        "latest": 1500,
        "count": 2,
        "min": 1000,
        "max": 1500,
        "mean": 1250,
        "median": 1250,
        "p95": 1475,
    }
    assert group["fields"]["memory_percent"] == {
        "latest": 6.5,
        "count": 2,
        "min": 5.5,
        "max": 6.5,
        "mean": 6,
        "median": 6,
        "p95": 6.45,
    }
    assert group["fields"]["event_queue_depth"]["max"] == 5
    assert group["fields"]["event_queue_depth"]["p95"] == 5
    assert group["fields"]["event_dropped_total"]["latest"] == 4
    assert group["fields"]["event_sink_error_total"]["latest"] == 1
    assert group["fields"]["event_degraded_count"]["latest"] == 3
    assert group["latest_event_drop_counts"] == {"queue_full": 4}
    assert group["latest_event_sink_error_counts"] == {"disk": 1}
    assert group["latest_event_pipeline_stopping"] is False
    assert group["latest_event_pipeline_worker_alive"] is True


def test_live_performance_report_resource_pressure_whitelists_health_fields(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                component="monitor.health",
                data={
                    "rss_bytes": 1000,
                    "balance_raw": {"leak_marker": "raw-balance"},
                    "balance_snapped": {"leak_marker": "snapped-balance"},
                    "equity": "leak-equity",
                    "pnl": "leak-pnl",
                    "raw_payload": {"leak_marker": "raw-payload"},
                    "event_drop_counts": {"queue_full": 1},
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    rendered = json.dumps(report["resource_pressure"], sort_keys=True)

    assert report["resource_pressure"]["groups"][0]["fields"]["rss_bytes"]["latest"] == 1000
    assert "balance_raw" not in rendered
    assert "balance_snapped" not in rendered
    assert "leak-equity" not in rendered
    assert "leak-pnl" not in rendered
    assert "raw-balance" not in rendered
    assert "snapped-balance" not in rendered
    assert "raw-payload" not in rendered


def test_live_performance_report_resource_pressure_summary_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "mixed" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                exchange="binance",
                user="binance_01",
                data={"rss_bytes": 1000, "event_queue_depth": 1},
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={"rss_bytes": 2000, "event_queue_depth": 2},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["resource_pressure"]["bots"] == 2
    assert len(summary["resource_pressure"]["groups"]) == 1
    assert summary["resource_pressure"]["groups_truncated"] is True
    assert summary["resource_pressure"]["groups"][0]["bot"] == "okx/okx_faisal"


def test_live_performance_report_shutdown_latency_from_lifecycle_events(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.stopping",
                seq=1,
                ts=1000,
                status="started",
                component="lifecycle",
                reason_code="shutdown_gracefully",
                data={"reason": "shutdown_gracefully"},
            ),
            _monitor_row(
                event_type="bot.shutdown.stage",
                seq=2,
                ts=2000,
                status="succeeded",
                component="shutdown",
                reason_code="maintainers_stopped",
                data={"stage": "maintainers_stopped", "elapsed_s": 2.5, "task_count": 4},
            ),
            _monitor_row(
                event_type="bot.shutdown.stage",
                seq=3,
                ts=3000,
                status="degraded",
                level="warning",
                component="shutdown",
                reason_code="execution_loop_timeout",
                data={
                    "stage": "execution_loop_timeout",
                    "elapsed_s": 6.25,
                    "timeout_s": 5.0,
                    "error": "api_key=AKIA123 Authorization: Bearer TOKEN123",
                },
            ),
            _monitor_row(
                event_type="bot.stopped",
                seq=4,
                ts=4000,
                status="succeeded",
                component="lifecycle",
                reason_code="shutdown_gracefully",
                data={"reason": "shutdown_gracefully", "elapsed_s": 7.5},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    shutdown = report["shutdown_latency"]
    groups = {group["operation"]: group for group in shutdown["groups"]}
    rendered = json.dumps(shutdown, sort_keys=True)

    assert shutdown["total_events"] == 4
    assert shutdown["shutdowns_started"] == 1
    assert shutdown["shutdowns_completed"] == 1
    assert shutdown["event_types"] == {
        "bot.stopping": 1,
        "bot.shutdown.stage": 2,
        "bot.stopped": 1,
    }
    assert shutdown["stage_counts"] == {
        "maintainers_stopped": 1,
        "execution_loop_timeout": 1,
    }
    assert groups["shutdown.stage.execution_loop_timeout"]["max_ms"] == 6250
    assert groups["shutdown.stage.execution_loop_timeout"]["statuses"] == {"degraded": 1}
    assert groups["shutdown.stage.execution_loop_timeout"]["timing_kind"] == "cumulative"
    assert groups["shutdown.total"]["max_ms"] == 7500
    assert groups["shutdown.total"]["timing_kind"] == "duration"
    assert "AKIA123" not in rendered
    assert "TOKEN123" not in rendered
    assert "api_key" not in rendered


def test_live_performance_report_shutdown_latency_summary_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.shutdown.stage",
                seq=1,
                ts=1000,
                component="shutdown",
                reason_code="requested",
                data={"stage": "requested", "elapsed_s": 0.1},
            ),
            _monitor_row(
                event_type="bot.stopped",
                seq=2,
                ts=2000,
                component="lifecycle",
                data={"elapsed_s": 2.0},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["shutdown_latency"]["total_groups"] == 2
    assert len(summary["shutdown_latency"]["groups"]) == 1
    assert summary["shutdown_latency"]["groups_truncated"] is True
    assert summary["shutdown_latency"]["groups"][0]["operation"] == "shutdown.total"
    assert summary["operation_durations"]["total_groups"] == 2
    assert summary["operation_durations"]["operation_category_counts"] == {"shutdown": 2}
    assert summary["operation_durations"]["blocking_scope_counts"] == {"observability": 2}
    assert summary["operation_durations"]["groups"][0]["operation"] == "shutdown.total"


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
