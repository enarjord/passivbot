from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import live.performance_report as performance_report_module
from live.performance_report import (
    build_live_performance_report,
    project_live_performance_report_sections,
    summarize_live_performance_report,
)
from tools import live_performance_report


def _fake_event_file_discovery(files):
    return SimpleNamespace(
        files=list(files),
        to_dict=lambda: {
            "bot_path_pruning_applied": False,
            "candidate_files": len(files),
            "event_segments": len(files),
            "opaque_bot_id_full_scan": False,
            "rotated_skipped": 0,
            "scope_pruned": 0,
        },
    )


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
    (events_dir / "20260629.ndjson.gz").write_bytes(b"")

    report = build_live_performance_report(tmp_path / "monitor")

    assert report["ok"] is True
    assert report["files_scanned"] == 1
    assert report["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 1,
        "scope_pruned": 0,
    }
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
                data={"elapsed_ms": 1000, "debug_profile": "state"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={"elapsed_ms": 2000, "debug_profile": "startup"},
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
        "debug_profiles": [],
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


def test_live_performance_report_timing_groups_include_latest_safe_ids(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1", "plan_id": "plan_1"},
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={
                    "bot_id": "binance_01",
                    "cycle_id": "cy_2",
                    "snapshot_id": "snapshot_2",
                    "plan_id": "plan_2",
                    "action_id": "action_2",
                    "order_wave_id": "wave_2",
                    "remote_call_id": "rc_2",
                    "remote_call_group_id": "rcg_2",
                },
                data={"elapsed_ms": 2000},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report)
    expected_ids = {
        "bot_id": "binance_01",
        "cycle_id": "cy_2",
        "snapshot_id": "snapshot_2",
        "plan_id": "plan_2",
        "order_wave_id": "wave_2",
        "remote_call_id": "rc_2",
        "remote_call_group_id": "rcg_2",
    }

    performance_group = next(
        group
        for group in report["performance"]["groups"]
        if group["operation"] == "cycle.elapsed"
    )
    duration_group = next(
        group
        for group in report["operation_durations"]["groups"]
        if group["operation"] == "cycle.elapsed"
    )
    blocker_group = next(
        group
        for group in report["slowest_blockers"]["groups"]
        if group["operation"] == "cycle.elapsed"
    )
    summary_duration_group = next(
        group
        for group in summary["operation_durations"]["groups"]
        if group["operation"] == "cycle.elapsed"
    )
    summary_blocker_group = next(
        group
        for group in summary["slowest_blockers"]["groups"]
        if group["operation"] == "cycle.elapsed"
    )

    assert performance_group["latest_ids"] == expected_ids
    assert duration_group["latest_ids"] == expected_ids
    assert blocker_group["latest_ids"] == expected_ids
    assert summary_duration_group["latest_ids"] == expected_ids
    assert summary_blocker_group["latest_ids"] == expected_ids
    rendered = json.dumps(report, sort_keys=True)
    assert "action_2" not in rendered


def test_live_performance_report_latest_ids_do_not_stick_to_newer_sample(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
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

    report = build_live_performance_report(tmp_path / "monitor")
    groups = [
        next(
            group
            for group in report[section]["groups"]
            if group["operation"] == "cycle.elapsed"
        )
        for section in ("performance", "operation_durations", "slowest_blockers")
    ]

    assert all(group["latest_ts"] == 2000 for group in groups)
    assert all("latest_ids" not in group for group in groups)


def test_live_performance_report_latest_ids_use_persistent_event_position(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=1000,
                ids={"cycle_id": "cy_2", "plan_id": "plan_2"},
                data={"elapsed_ms": 2000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1", "plan_id": "plan_1"},
                data={"elapsed_ms": 1000},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report)
    expected_ids = {"cycle_id": "cy_2", "plan_id": "plan_2"}
    groups = [
        next(
            group
            for group in section["groups"]
            if group["operation"] == "cycle.elapsed"
        )
        for section in (
            report["performance"],
            report["operation_durations"],
            report["slowest_blockers"],
            summary["operation_durations"],
            summary["slowest_blockers"],
        )
    ]

    assert all(group["latest_ts"] == 1000 for group in groups)
    assert all(group["latest_ids"] == expected_ids for group in groups)


def test_live_performance_report_latest_ids_normalize_legacy_snapshot_id(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=1000,
                data={
                    "snapshot_id": "snap_legacy",
                    "surface_ages": [{"name": "balance", "age_ms": 500}],
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report)
    expected_ids = {"snapshot_id": "snap_legacy"}
    groups = [
        next(
            group
            for group in section["groups"]
            if group["operation"] == "input_staleness.surface.balance"
        )
        for section in (
            report["input_staleness"],
            report["operation_durations"],
            report["slowest_blockers"],
            summary["input_staleness"],
            summary["operation_durations"],
            summary["slowest_blockers"],
        )
    ]

    assert all(group["latest_ids"] == expected_ids for group in groups)


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
    assert report["input_staleness"]["market_snapshot"] == {
        "observations": 1,
        "count": {"count": 1, "max": 2, "mean": 2, "median": 2, "min": 2, "p95": 2},
        "symbol_count": {
            "count": 1,
            "max": 2,
            "mean": 2,
            "median": 2,
            "min": 2,
            "p95": 2,
        },
        "missing_count": {
            "count": 1,
            "max": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "p95": 0,
        },
        "missing_symbols_total": 0,
        "missing_observation_count": 0,
        "max_age_ms": {
            "count": 1,
            "max": 700,
            "mean": 700,
            "median": 700,
            "min": 700,
            "p95": 700,
        },
        "mean_age_ms": {
            "count": 1,
            "max": 450,
            "mean": 450,
            "median": 450,
            "min": 450,
            "p95": 450,
        },
        "configured_max_age_ms": {
            "count": 1,
            "max": 600,
            "mean": 600,
            "median": 600,
            "min": 600,
            "p95": 600,
        },
        "configured_excess_ms": {
            "count": 1,
            "max": 100,
            "mean": 100,
            "median": 100,
            "min": 100,
            "p95": 100,
        },
        "sources": {"ticker": 1},
    }
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


def test_live_performance_report_market_snapshot_summary_counts_missing_sources(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                seq=1,
                ts=1000,
                data={
                    "market_snapshot_summary": {
                        "count": 5,
                        "symbol_count": 6,
                        "missing_count": 1,
                        "max_age_ms": 1500,
                        "mean_age_ms": 400,
                        "configured_max_age_ms": 1000,
                        "sources": ["ticker", "websocket"],
                    }
                },
            ),
            _monitor_row(
                event_type="snapshot.built",
                seq=2,
                ts=2000,
                data={
                    "market_snapshot_summary": {
                        "count": 6,
                        "symbol_count": 6,
                        "missing_count": 0,
                        "max_age_ms": 800,
                        "mean_age_ms": 300,
                        "configured_max_age_ms": 1000,
                        "sources": ["ticker"],
                    }
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    market = report["input_staleness"]["market_snapshot"]

    assert report["input_staleness"]["snapshot_market_summaries_seen"] == 2
    assert report["input_staleness"]["snapshot_market_stale_count"] == 1
    assert market["observations"] == 2
    assert market["count"]["max"] == 6
    assert market["symbol_count"]["min"] == 6
    assert market["missing_count"]["max"] == 1
    assert market["missing_symbols_total"] == 1
    assert market["missing_observation_count"] == 1
    assert market["max_age_ms"]["max"] == 1500
    assert market["mean_age_ms"]["mean"] == 350
    assert market["configured_max_age_ms"]["median"] == 1000
    assert market["configured_excess_ms"]["max"] == 500
    assert market["sources"] == {"ticker": 2, "websocket": 1}


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
    assert summary["input_staleness"]["market_snapshot"]["observations"] == 1
    assert summary["input_staleness"]["market_snapshot"]["configured_excess_ms"][
        "max"
    ] == 50
    assert summary["input_staleness"]["rust_calls_seen"] == 1
    assert summary["input_staleness"]["snapshot_to_rust_latest_snapshot_matches"] == 1
    assert summary["input_staleness"]["rust_calls_missing_ema"] == 1
    assert summary["input_staleness"]["total_groups"] == 5
    assert len(summary["input_staleness"]["groups"]) == 1


def test_live_performance_report_startup_readiness_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(performance_report_module, "utc_ms", lambda: 64000)
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
                data={
                    "stage": "account",
                    "elapsed_ms": 900,
                    "since_previous_ms": 900,
                    "readiness_scope": "account_critical",
                    "trading_impact": "protective_blocker",
                },
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=3,
                ts=3000,
                component="bot.startup",
                data={
                    "stage": "hsl",
                    "elapsed_s": 2.5,
                    "since_previous_ms": 1600,
                    "readiness_scope": "held_position_protective",
                    "trading_impact": "protective_blocker",
                },
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
    assert bot["readiness_sla"] == {
        "account_critical": {
            "phase": "account",
            "elapsed_ms": 900,
            "trading_impact": "protective_blocker",
        },
        "held_position_protective": {
            "phase": "hsl",
            "elapsed_ms": 2500,
            "trading_impact": "protective_blocker",
        },
    }
    assert startup["readiness_scope_counts"] == {
        "account_critical": 1,
        "held_position_protective": 1,
    }
    assert startup["readiness_scope_elapsed_ms"]["account_critical"]["max"] == 900
    assert (
        startup["readiness_scope_elapsed_ms"]["held_position_protective"]["max"]
        == 2500
    )
    assert startup["readiness_trading_impact_counts"] == {
        "protective_blocker": 2,
    }
    assert startup["startup_phase_counts"] == {"account": 1, "hsl": 1}
    assert startup["startup_phase_elapsed_ms"]["account"] == {
        "count": 1,
        "min": 900,
        "max": 900,
        "mean": 900,
        "median": 900,
        "p95": 900,
    }
    assert startup["startup_phase_elapsed_ms"]["hsl"]["max"] == 2500
    assert startup["startup_phase_since_previous_ms"]["account"]["max"] == 900
    assert startup["startup_phase_since_previous_ms"]["hsl"]["max"] == 1600
    assert bot["hsl_replay"]["stage"] == "pair_replay"
    assert bot["hsl_replay"]["pairs"] == 26
    assert bot["hsl_replay"]["held_pairs"] == 1
    assert bot["hsl_replay"]["rows_per_second"] == 289.4
    assert bot["hsl_replay"]["latest_event_age_ms"] == 60000


def test_live_performance_report_rejects_mismatched_startup_readiness_contract(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.startup_timing",
                seq=1,
                ts=1000,
                data={
                    "phase": "account",
                    "elapsed_ms": 900,
                    "readiness_scope": "held_position_protective",
                    "trading_impact": "diagnostics_only",
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]

    assert startup["bots"][0]["startup_phases_ms"] == {"account": 900}
    assert "readiness_sla" not in startup["bots"][0]
    assert startup["readiness_scope_counts"] == {}
    startup_groups = {
        group["operation"]: group for group in report["operation_durations"]["groups"]
    }
    assert startup_groups["startup.account"]["trading_impact"] == (
        "blocks_startup_readiness"
    )


def test_live_performance_report_startup_phase_uses_stage_only_as_legacy_fallback(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.startup_timing",
                seq=1,
                ts=1000,
                data={"stage": "hsl", "elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=2000,
                data={
                    "phase": "account",
                    "stage": "hsl",
                    "elapsed_ms": 2000,
                },
            ),
        ],
    )

    startup = build_live_performance_report(tmp_path / "monitor")["startup_readiness"]

    assert startup["startup_phase_counts"] == {"hsl": 1}
    assert startup["bots"][0]["startup_phases_ms"] == {"hsl": 1000}
    startup_groups = {
        group["operation"]: group
        for group in build_live_performance_report(tmp_path / "monitor")[
            "operation_durations"
        ]["groups"]
        if group["operation"].startswith("startup.")
    }
    assert set(startup_groups) == {"startup.hsl"}
    assert startup_groups["startup.hsl"]["count"] == 1


def test_live_performance_report_startup_readiness_completed_hsl_not_active(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.started",
                seq=1,
                ts=1000,
                data={"live_event_debug_profiles": ["rust"]},
            ),
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
    assert "latest_event_age_ms" not in startup["bots"][0]["hsl_replay"]
    assert startup["bots"][0]["hsl_replay"]["skipped_pairs"] == 1


def test_live_performance_report_startup_readiness_keeps_sparse_hsl_context(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2000,
                status="progress",
                data={
                    "signal_mode": "coin",
                    "stage": "pair_replay",
                    "pairs": 26,
                    "held_pairs": 1,
                    "rows_per_second": 300,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.failed",
                seq=3,
                ts=3000,
                status="failed",
                reason_code="replay_failed",
                data={"signal_mode": "coin", "elapsed_s": 5},
            ),
        ],
    )

    hsl_replay = build_live_performance_report(tmp_path / "monitor")[
        "startup_readiness"
    ]["bots"][0]["hsl_replay"]

    assert hsl_replay == {
        "latest_ts": 3000,
        "event_type": "hsl.replay.failed",
        "status": "failed",
        "reason_code": "replay_failed",
        "signal_mode": "coin",
        "stage": "pair_replay",
        "pairs": 26,
        "held_pairs": 1,
        "rows_per_second": 300,
        "elapsed_s": 5,
    }


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


@pytest.mark.parametrize("current_started_ts", [None, 1000])
def test_live_performance_report_startup_readiness_uses_event_order_with_file_cap(
    tmp_path, current_started_ts
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=1100,
                data={
                    "stage": "hsl",
                    "elapsed_ms": 9000,
                    "readiness_scope": "held_position_protective",
                    "trading_impact": "protective_blocker",
                },
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=3,
                ts=1200,
                status="succeeded",
                data={"stage": "full_replay", "pairs": 2},
            ),
            _monitor_row(event_type="bot.ready", seq=4, ts=1300),
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.started",
                seq=1,
                ts=current_started_ts,
                data={"live_event_debug_profiles": ["state"]},
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=2100,
                data={
                    "stage": "account",
                    "elapsed_ms": 300,
                    "readiness_scope": "account_critical",
                    "trading_impact": "protective_blocker",
                },
            ),
            _monitor_row(event_type="bot.ready", seq=3, ts=2200),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
    )
    startup = report["startup_readiness"]
    bot = startup["bots"][0]

    assert bot["lifecycle_status"] == "ready"
    if current_started_ts is None:
        assert "bot_started_ts" not in bot
    else:
        assert bot["bot_started_ts"] == current_started_ts
    assert bot["bot_ready_ts"] == 2200
    assert bot["debug_profiles"] == ["state"]
    assert bot["startup_phases_ms"] == {"account": 300}
    assert bot["readiness_sla"] == {
        "account_critical": {
            "phase": "account",
            "elapsed_ms": 300,
            "trading_impact": "protective_blocker",
        }
    }
    assert "hsl_replay" not in bot
    assert startup["startup_phase_counts"] == {"account": 1, "hsl": 1}
    assert startup["startup_phase_elapsed_ms"]["hsl"]["max"] == 9000
    assert startup["readiness_scope_counts"] == {
        "account_critical": 1,
        "held_position_protective": 1,
    }


def test_live_performance_report_startup_readiness_joins_complete_rotated_lifecycle(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=1100,
                data={
                    "phase": "account",
                    "elapsed_ms": 100,
                    "readiness_scope": "account_critical",
                    "trading_impact": "protective_blocker",
                },
            ),
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.startup_timing",
                seq=3,
                ts=2000,
                data={
                    "phase": "hsl",
                    "elapsed_ms": 1000,
                    "readiness_scope": "held_position_protective",
                    "trading_impact": "protective_blocker",
                },
            ),
            _monitor_row(event_type="bot.ready", seq=4, ts=2100),
        ],
    )

    startup = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
    )["startup_readiness"]
    bot = startup["bots"][0]

    assert bot["bot_started_ts"] == 1000
    assert bot["bot_ready_ts"] == 2100
    assert bot["lifecycle_status"] == "ready"
    assert bot["startup_phases_ms"] == {"account": 100, "hsl": 1000}
    assert set(bot["readiness_sla"]) == {
        "account_critical",
        "held_position_protective",
    }


def test_live_performance_report_startup_readiness_rejects_old_tail_anchor(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [_monitor_row(event_type="bot.started", seq=1, ts=1000)],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=2, ts=4000),
            _monitor_row(event_type="bot.ready", seq=3, ts=5000),
        ],
    )

    startup = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
        event_tail_lines=1,
    )["startup_readiness"]
    bot = startup["bots"][0]

    assert "bot_started_ts" not in bot
    assert "bot_ready_ts" not in bot
    assert "lifecycle_status" not in bot


@pytest.mark.parametrize("max_event_files_per_bot", [0, 2])
def test_live_performance_report_rejects_old_startup_data_after_unrelated_tail(
    tmp_path,
    max_event_files_per_bot,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=1500,
                data={"phase": "account", "elapsed_ms": 500},
            ),
            _monitor_row(event_type="execution.create_sent", seq=3, ts=2000),
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=4, ts=4000),
            _monitor_row(event_type="health.summary", seq=5, ts=5000),
            _monitor_row(event_type="health.summary", seq=6, ts=6000),
            _monitor_row(event_type="health.summary", seq=7, ts=7000),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=max_event_files_per_bot,
        event_tail_lines=3,
    )

    assert report["event_window"]["event_tail_limited_files"] == 2
    assert report["startup_readiness"]["bots"] == []
    assert report["startup_milestones"]["bots"] == []
    assert report["startup_milestones"]["observed_counts"] == {
        "first_cycle_started": 0,
        "first_rust_called": 0,
        "first_fresh_entry_eligible": 0,
        "first_exchange_write_submitted": 0,
    }


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


def test_live_performance_report_startup_milestones_current_lifecycle(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="cycle.started",
                seq=2,
                ts=1100,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                seq=3,
                ts=1300,
                ids={"cycle_id": "cy_1", "snapshot_id": "snap_1"},
            ),
            _monitor_row(
                event_type="execution.create_connector_call_started",
                seq=4,
                ts=1500,
                symbol="BTCUSDT",
                pside="long",
                side="buy",
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "ow_1",
                    "action_id": "ow_1:create:0",
                },
                data={
                    "action": "create",
                    "connector_method": "cca.create_order",
                    "connector_route": "base",
                },
            ),
            _monitor_row(
                event_type="execution.create_sent",
                seq=5,
                ts=1600,
                symbol="BTCUSDT",
                pside="long",
                side="buy",
                ids={
                    "cycle_id": "cy_1",
                    "action_id": "action_1",
                    "remote_call_id": "remote_1",
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_milestones"]

    assert startup["bot_count"] == 1
    assert startup["events_without_start"] == 0
    assert startup["observed_counts"] == {
        "first_cycle_started": 1,
        "first_rust_called": 1,
        "first_fresh_entry_eligible": 0,
        "first_exchange_write_submitted": 1,
    }
    assert startup["elapsed_ms"]["first_cycle_started"]["max"] == 100
    assert startup["elapsed_ms"]["first_rust_called"]["max"] == 300
    assert startup["elapsed_ms"]["first_exchange_write_submitted"]["max"] == 600
    bot = startup["bots"][0]
    assert bot["bot_started_ts"] == 1000
    assert bot["milestones"]["first_cycle_started"] == {
        "status": "observed",
        "event_type": "cycle.started",
        "trading_impact": "cycle_delay",
        "ts_ms": 1100,
        "elapsed_ms": 100,
        "event_id": "evt_2",
        "cycle_id": "cy_1",
    }
    write = bot["milestones"]["first_exchange_write_submitted"]
    assert write["event_type"] == "execution.create_sent"
    assert write["action"] == "create"
    assert write["symbol"] == "BTCUSDT"
    assert write["pside"] == "long"
    assert write["side"] == "buy"
    assert write["remote_call_id"] == "remote_1"
    assert "action_1" not in json.dumps(report["startup_milestones"], sort_keys=True)


def test_live_performance_report_startup_milestone_requires_eligible_initial_entry(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=2,
                ts=1200,
                ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
                data={"outcome_counts": {"blocked_candidate": 2, "eligible": 0}},
            ),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=3,
                ts=1300,
                data={"outcome_counts": {"eligible": "1"}},
            ),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=4,
                ts=1400,
                data={"outcome_counts": {"eligible": True}},
            ),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=5,
                ts=1500,
                data={"outcome_counts": {"eligible": 1.0}},
            ),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=6,
                ts=1600,
                ids={"cycle_id": "cy_2", "order_wave_id": "ow_2"},
                data={"outcome_counts": {"eligible": 1, "no_candidate": 37}},
            ),
            _monitor_row(
                event_type="entry.initial_eligibility",
                seq=7,
                ts=1700,
                data={"outcome_counts": {"eligible": 2}},
            ),
        ],
    )

    startup = build_live_performance_report(tmp_path / "monitor")[
        "startup_milestones"
    ]

    assert startup["observed_counts"]["first_fresh_entry_eligible"] == 1
    assert startup["elapsed_ms"]["first_fresh_entry_eligible"]["max"] == 600
    milestone = startup["bots"][0]["milestones"]["first_fresh_entry_eligible"]
    assert milestone == {
        "status": "observed",
        "event_type": "entry.initial_eligibility",
        "trading_impact": "entry_blocker",
        "ts_ms": 1600,
        "elapsed_ms": 600,
        "event_id": "evt_6",
        "cycle_id": "cy_2",
        "order_wave_id": "ow_2",
    }


def test_startup_milestone_accumulator_retains_one_candidate_per_milestone():
    accumulator = performance_report_module._StartupMilestoneAccumulator()
    started = _monitor_row(event_type="bot.started", seq=1, ts=1000)
    accumulator.add(row=started, live_event=started["payload"]["_live_event"])

    event_types = (
        "cycle.started",
        "rust_orchestrator.called",
        "execution.create_sent",
    )
    for index in range(1000):
        for offset, event_type in enumerate(event_types):
            row = _monitor_row(
                event_type=event_type,
                seq=2 + index * len(event_types) + offset,
                ts=1100 + index * len(event_types) + offset,
            )
            accumulator.add(row=row, live_event=row["payload"]["_live_event"])

    state = accumulator.bots["binance/binance_01"]
    assert state["milestone_events_seen"] == 3000
    assert len(state["milestones"]) == 3
    assert accumulator.to_dict()["bots"][0]["milestones"]["first_cycle_started"][
        "event_id"
    ] == "evt_2"


def test_live_performance_report_startup_milestones_cancel_only_and_unknown(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(event_type="execution.create_skipped", seq=2, ts=1100),
            _monitor_row(event_type="execution.create_deferred", seq=3, ts=1200),
            _monitor_row(
                event_type="execution.cancel_sent",
                seq=4,
                ts=1300,
                ids={"remote_call_id": "cancel_1"},
            ),
        ],
    )

    startup = build_live_performance_report(tmp_path / "monitor")[
        "startup_milestones"
    ]
    milestones = startup["bots"][0]["milestones"]

    assert milestones["first_cycle_started"] == {
        "status": "unknown",
        "reason": "not_observed_in_selected_events",
        "trading_impact": "cycle_delay",
    }
    assert milestones["first_rust_called"]["status"] == "unknown"
    assert milestones["first_exchange_write_submitted"]["action"] == "cancel"
    assert milestones["first_exchange_write_submitted"]["event_type"] == (
        "execution.cancel_sent"
    )


@pytest.mark.parametrize("current_started_ts", [None, 1000])
def test_live_performance_report_startup_milestones_use_latest_ordered_lifecycle(
    tmp_path, current_started_ts
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(event_type="cycle.started", seq=2, ts=1100),
            _monitor_row(event_type="rust_orchestrator.called", seq=3, ts=1200),
            _monitor_row(event_type="execution.create_sent", seq=4, ts=1300),
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=current_started_ts),
            _monitor_row(event_type="cycle.started", seq=2, ts=2100),
        ],
    )

    startup = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
    )["startup_milestones"]
    bot = startup["bots"][0]

    if current_started_ts is None:
        assert "bot_started_ts" not in bot
        assert "elapsed_ms" not in bot["milestones"]["first_cycle_started"]
    else:
        assert bot["bot_started_ts"] == current_started_ts
        assert bot["milestones"]["first_cycle_started"]["elapsed_ms"] == 1100
    assert bot["milestones"]["first_cycle_started"]["ts_ms"] == 2100
    assert bot["milestones"]["first_rust_called"]["status"] == "unknown"
    assert bot["milestones"]["first_exchange_write_submitted"]["status"] == (
        "unknown"
    )
    assert startup["observed_counts"]["first_rust_called"] == 0
    assert startup["observed_counts"]["first_exchange_write_submitted"] == 0


def test_live_performance_report_startup_milestones_reject_old_tail_anchor(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [_monitor_row(event_type="bot.started", seq=1, ts=1000)],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=4000),
            _monitor_row(event_type="execution.create_sent", seq=2, ts=5000),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
        event_tail_lines=1,
    )
    startup = report["startup_milestones"]
    bot = startup["bots"][0]

    assert report["event_window"]["event_tail_skipped_lines"] == 1
    assert "bot_started_ts" not in bot
    assert bot["milestones"]["first_exchange_write_submitted"] == {
        "status": "unknown",
        "reason": "bot_started_not_observed_in_selected_events",
        "trading_impact": "cycle_delay",
    }
    assert startup["events_without_start"] == 1


def test_live_performance_report_startup_milestones_join_complete_rotated_lifecycle(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [_monitor_row(event_type="bot.started", seq=1, ts=1000)],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="rust_orchestrator.called", seq=2, ts=2000),
            _monitor_row(event_type="execution.create_sent", seq=3, ts=3000),
        ],
    )

    startup = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
        event_tail_lines=10,
    )["startup_milestones"]
    bot = startup["bots"][0]

    assert bot["bot_started_ts"] == 1000
    assert bot["milestones"]["first_rust_called"]["elapsed_ms"] == 1000
    assert bot["milestones"]["first_exchange_write_submitted"]["elapsed_ms"] == 2000


def test_live_performance_report_startup_milestones_are_bounded_and_projectable(
    tmp_path,
):
    for index in range(3):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="bot.started",
                    seq=1,
                    ts=1000,
                    user=f"user_{index}",
                ),
                _monitor_row(
                    event_type="cycle.started",
                    seq=2,
                    ts=1100,
                    user=f"user_{index}",
                ),
            ],
        )
    orphan_dir = tmp_path / "monitor" / "okx" / "orphan" / "events"
    _write_ndjson(
        orphan_dir / "current.ndjson",
        [_monitor_row(event_type="cycle.started", seq=1, ts=1000, user="orphan")],
    )

    full_report = build_live_performance_report(tmp_path / "monitor", group_limit=10)
    report = build_live_performance_report(tmp_path / "monitor", group_limit=2)
    startup = report["startup_milestones"]
    summary = summarize_live_performance_report(report, group_limit=1)
    projected = project_live_performance_report_sections(
        report, ["startup_milestones"]
    )

    assert startup["bot_count"] == 4
    assert startup["bots_truncated"] is True
    assert len(startup["bots"]) == 2
    assert startup["events_without_start"] == 1
    orphan = next(
        item
        for item in full_report["startup_milestones"]["bots"]
        if item["bot"] == "binance/orphan"
    )
    assert orphan["milestones"]["first_cycle_started"]["reason"] == (
        "bot_started_not_observed_in_selected_events"
    )
    assert summary["startup_milestones"]["bot_count"] == 4
    assert len(summary["startup_milestones"]["bots"]) == 1
    assert "startup_milestones" in projected
    assert "startup_readiness" not in projected


def test_live_performance_report_startup_fill_cache_proof_uses_exact_post_start_proof(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=1100,
                status="succeeded",
                reason_code="fill_cache_ready",
                data={
                    "source": "startup",
                    "refresh_mode": "cache_load",
                    "history_scope": "window",
                    "event_count_after": 99,
                },
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=1200,
                status="deferred",
                reason_code="fill_history_coverage_unavailable",
                data={
                    "coverage_after": {
                        "ready": False,
                        "reason": "window_coverage_not_proven",
                        "history_scope": "window",
                        "covered_start_ms": 0,
                        "oldest_event_ts": 500,
                    }
                },
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=4,
                ts=1300,
                data={"phase": "hsl", "elapsed_ms": 300},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=5,
                ts=1400,
                status="succeeded",
                reason_code="fills_refresh_succeeded",
                data={
                    "coverage_after": {
                        "ready": True,
                        "reason": "window_covered",
                        "history_scope": "window",
                        "covered_start_ms": 400,
                        "oldest_event_ts": 300,
                    }
                },
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=6,
                ts=1500,
                data={
                    "coverage_after": {
                        "ready": True,
                        "reason": "full_history",
                        "history_scope": "all",
                    }
                },
            ),
        ],
    )

    section = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]
    proof = section["bots"][0]

    assert proof == {
        "bot": "binance/binance_01",
        "bot_started_ts": 1000,
        "lifecycle_source_complete": True,
        "status": "proven",
        "cache_load_relation": "before_proof",
        "cache_load": {
            "status": "succeeded",
            "reason": "fill_cache_ready",
            "history_scope": "window",
            "ts_ms": 1100,
            "elapsed_ms_from_start": 100,
        },
        "proof": {
            "ready": True,
            "reason": "window_covered",
            "history_scope": "window",
            "covered_start_ms": 400,
            "oldest_event_ts": 300,
        },
        "proof_elapsed_ms_from_start": 400,
        "startup_phase_relation": "after_hsl",
    }
    assert section["status_counts"] == {"proven": 1}
    assert section["proof_elapsed_ms"] == {
        "count": 1,
        "min": 400,
        "max": 400,
        "mean": 400,
        "median": 400,
        "p95": 400,
    }


def test_live_performance_report_startup_fill_cache_proof_keeps_cache_without_proof_unknown(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=1100,
                status="succeeded",
                reason_code="fill_cache_ready",
                data={
                    "source": "startup",
                    "refresh_mode": "cache_load",
                    "history_scope": "all",
                    "coverage_ready_after": True,
                },
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof == {
        "bot": "binance/binance_01",
        "bot_started_ts": 1000,
        "lifecycle_source_complete": True,
        "status": "unknown",
        "cache_load_relation": "not_observed",
        "cache_load": {
            "status": "succeeded",
            "reason": "fill_cache_ready",
            "history_scope": "all",
            "ts_ms": 1100,
            "elapsed_ms_from_start": 100,
        },
        "startup_phase_relation": "unknown",
    }


def test_live_performance_report_startup_fill_cache_proof_reports_reverse_cache_order(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=1200,
                data={"coverage_after": {"ready": True, "reason": "full_history"}},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=1300,
                status="succeeded",
                reason_code="fill_cache_ready",
                data={"source": "startup", "refresh_mode": "cache_load"},
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["status"] == "proven"
    assert proof["cache_load_relation"] == "after_proof"


def test_live_performance_report_startup_fill_cache_proof_uses_latest_unproven_reason(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=1200,
                data={
                    "coverage_after": {
                        "ready": False,
                        "reason": "window_coverage_not_proven",
                        "history_scope": "window",
                        "covered_start_ms": 100,
                    }
                },
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=1300,
                data={
                    "coverage_after": {
                        "ready": False,
                        "reason": "known_gap_overlaps_lookback",
                        "history_scope": "window",
                        "covered_start_ms": 100,
                        "oldest_event_ts": 90,
                        "gap_reason": "fetch_failed",
                        "gap_start_ts": 120,
                        "gap_end_ts": 180,
                    }
                },
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["status"] == "unproven"
    assert proof["cache_load_relation"] == "not_observed"
    assert proof["proof"] == {
        "ready": False,
        "reason": "known_gap_overlaps_lookback",
        "history_scope": "window",
        "covered_start_ms": 100,
        "oldest_event_ts": 90,
        "gap_reason": "fetch_failed",
        "gap_start_ts": 120,
        "gap_end_ts": 180,
    }
    assert proof["proof_elapsed_ms_from_start"] == 300


@pytest.mark.parametrize(
    "phase, phase_ts, proof_ts, expected_relation",
    [
        ("hsl", 1300, 1200, "before_hsl"),
        ("startup", 1100, 1200, "after_startup"),
    ],
)
def test_live_performance_report_startup_fill_cache_proof_uses_only_exact_phase_anchors(
    tmp_path, phase, phase_ts, proof_ts, expected_relation
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    rows = [_monitor_row(event_type="bot.started", seq=1, ts=1000)]
    if phase_ts < proof_ts:
        rows.append(
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=phase_ts,
                data={"phase": phase, "elapsed_ms": phase_ts - 1000},
            )
        )
    rows.append(
        _monitor_row(
            event_type="fills.refresh_summary",
            seq=3,
            ts=proof_ts,
            data={"coverage_after": {"ready": True, "reason": "full_history"}},
        )
    )
    if phase_ts > proof_ts:
        rows.append(
            _monitor_row(
                event_type="bot.startup_timing",
                seq=4,
                ts=phase_ts,
                data={"phase": phase, "elapsed_ms": phase_ts - 1000},
            )
        )
    _write_ndjson(events_dir / "current.ndjson", rows)

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["status"] == "proven"
    assert proof["startup_phase_relation"] == expected_relation


def test_live_performance_report_startup_fill_cache_proof_prefers_after_startup_relation(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=1100,
                data={"phase": "hsl", "elapsed_ms": 100},
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=3,
                ts=1200,
                data={"phase": "startup", "elapsed_ms": 200},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=4,
                ts=1300,
                data={"coverage_after": {"ready": True, "reason": "full_history"}},
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["startup_phase_relation"] == "after_startup"


def test_live_performance_report_startup_fill_cache_proof_resets_and_rejects_incomplete_sources(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "2026-07-10T00-00-00.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=1200,
                data={"coverage_after": {"ready": True, "reason": "full_history"}},
            ),
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=3, ts=2000),
            _monitor_row(event_type="health.summary", seq=4, ts=2100),
        ],
    )

    reset = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
    )["startup_fill_cache_proof"]["bots"][0]

    assert reset == {
        "bot": "binance/binance_01",
        "bot_started_ts": 2000,
        "lifecycle_source_complete": True,
        "status": "unknown",
        "cache_load_relation": "not_observed",
        "startup_phase_relation": "unknown",
    }

    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="health.summary", seq=4, ts=2100),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=5,
                ts=2200,
                data={"phase": "hsl", "elapsed_ms": 200},
            ),
        ],
    )
    stale = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
        event_tail_lines=1,
    )["startup_fill_cache_proof"]["bots"][0]

    assert stale == {
        "bot": "binance/binance_01",
        "lifecycle_source_complete": False,
        "status": "unknown",
        "cache_load_relation": "not_observed",
        "startup_phase_relation": "unknown",
    }


@pytest.mark.parametrize("started_ts, proof_ts", [(None, 1200), (1000, None), (1000, -1)])
def test_live_performance_report_startup_fill_cache_proof_rejects_invalid_timestamps(
    tmp_path, started_ts, proof_ts
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=started_ts),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=proof_ts,
                data={"coverage_after": {"ready": True, "reason": "full_history"}},
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["status"] == "unknown"
    assert "proof_elapsed_ms_from_start" not in proof
    assert proof["startup_phase_relation"] == "unknown"


def test_live_performance_report_startup_fill_cache_proof_uses_later_valid_evidence(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=None,
                status="succeeded",
                reason_code="malformed_cache_ts",
                data={"source": "startup", "refresh_mode": "cache_load"},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=1100,
                status="succeeded",
                reason_code="fill_cache_ready",
                data={"source": "startup", "refresh_mode": "cache_load"},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=4,
                ts=None,
                data={"coverage_after": {"ready": True, "reason": "malformed_ts"}},
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=5,
                ts=1200,
                data={"coverage_after": {"ready": True, "reason": "full_history"}},
            ),
        ],
    )

    proof = build_live_performance_report(tmp_path / "monitor")[
        "startup_fill_cache_proof"
    ]["bots"][0]

    assert proof["status"] == "proven"
    assert proof["cache_load_relation"] == "before_proof"
    assert proof["cache_load"] == {
        "status": "succeeded",
        "reason": "fill_cache_ready",
        "ts_ms": 1100,
        "elapsed_ms_from_start": 100,
    }
    assert proof["proof"] == {"ready": True, "reason": "full_history"}
    assert proof["proof_elapsed_ms_from_start"] == 200


def test_live_performance_report_startup_fill_cache_proof_is_bounded_whitelisted_and_projectable(
    tmp_path,
):
    for index in range(3):
        events_dir = tmp_path / "monitor" / "binance" / f"user_{index}" / "events"
        _write_ndjson(
            events_dir / "current.ndjson",
            [
                _monitor_row(
                    event_type="bot.started",
                    seq=1,
                    ts=1000 + index,
                    user=f"user_{index}",
                ),
                _monitor_row(
                    event_type="fills.refresh_summary",
                    seq=2,
                    ts=1100 + index,
                    user=f"user_{index}",
                    data={
                        "coverage_after": {
                            "ready": False,
                            "reason": (
                                "private_secret"
                                if index == 0
                                else "known_gap_overlaps_lookback"
                            ),
                            "history_scope": "private/secret" if index == 0 else "window",
                            "gap_reason": (
                                "/private/secret/api_key_abc"
                                if index == 0
                                else "fetch_failed"
                            ),
                            "gap_start_ts": 100,
                            "gap_end_ts": 200,
                            "raw_payload": "leak_marker",
                            "path": "/root/passivbot/private",
                        },
                        "error": "api_key=secret",
                        "debug": {"raw_payload": "leak_marker"},
                    },
                ),
            ],
        )

    report = build_live_performance_report(tmp_path / "monitor", group_limit=2)
    summary = summarize_live_performance_report(report, group_limit=1)
    projected = project_live_performance_report_sections(
        report, ["startup_fill_cache_proof"]
    )
    section = report["startup_fill_cache_proof"]
    rendered = json.dumps(section, sort_keys=True)

    assert section["bot_count"] == 3
    assert section["status_counts"] == {"unproven": 3}
    assert section["proof_elapsed_ms"]["count"] == 3
    assert section["bots_truncated"] is True
    assert len(section["bots"]) == 2
    assert set(section["bots"][0]) == {
        "bot",
        "bot_started_ts",
        "lifecycle_source_complete",
        "status",
        "cache_load_relation",
        "proof",
        "proof_elapsed_ms_from_start",
        "startup_phase_relation",
    }
    assert set(section["bots"][0]["proof"]) == {
        "ready",
        "reason",
        "history_scope",
        "gap_reason",
        "gap_start_ts",
        "gap_end_ts",
    }
    assert section["bots"][0]["proof"]["reason"] == "other"
    assert section["bots"][0]["proof"]["history_scope"] == "other"
    assert section["bots"][0]["proof"]["gap_reason"] == "other"
    assert "leak_marker" not in rendered
    assert "/root/passivbot" not in rendered
    assert "api_key" not in rendered
    assert len(summary["startup_fill_cache_proof"]["bots"]) == 1
    assert "startup_fill_cache_proof" in projected
    assert "startup_milestones" not in projected


def test_live_performance_report_startup_readiness_hsl_whitelist(tmp_path, monkeypatch):
    monkeypatch.setattr(performance_report_module, "utc_ms", lambda: 122000)
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
        "latest_event_age_ms": 120000,
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


def test_live_performance_report_startup_phase_labels_are_whitelisted(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="bot.started", seq=1, ts=1000),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=2000,
                data={
                    "stage": "api_key=should_not_render",
                    "elapsed_ms": 100,
                    "since_previous_ms": 50,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    startup = report["startup_readiness"]
    rendered = json.dumps(report, sort_keys=True)

    assert "api_key=should_not_render" not in rendered
    assert startup["startup_phase_counts"] == {"other": 1}
    assert startup["startup_phase_elapsed_ms"]["other"]["max"] == 100
    assert startup["startup_phase_since_previous_ms"]["other"]["max"] == 50
    assert startup["bots"][0]["startup_phases_ms"] == {"other": 100}
    assert report["operation_durations"]["groups"][0]["operation"] == "startup.other"


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
                    "candidate_rows": 30,
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
    assert profile["stage_counts"] == {
        "loaded": 1,
        "pair_replay": 1,
        "full_replay": 1,
    }
    assert profile["latest_status_counts"] == {"failed": 1}
    assert profile["latest_stage_counts"] == {}
    assert profile["active_stage_counts"] == {}
    assert profile["active_bot_count"] == 0
    assert profile["completed_bot_count"] == 0
    assert profile["failed_bot_count"] == 1
    assert group["bot"] == "binance/binance_01"
    assert group["event_types"]["hsl.replay.progress"] == 2
    assert group["loaded"]["data"]["symbols"] == 5
    assert group["progress"]["data"]["is_held_pair"] is True
    assert group["progress"]["derived"]["observed_work_pct"] == 30
    assert group["progress"]["derived"]["observed_required_work_pct"] == 40
    assert group["progress"]["derived"]["estimated_dense_pair_row_work"] == 40
    assert group["progress"]["derived"]["estimated_held_pair_row_work"] == 10
    assert group["progress"]["derived"]["estimated_required_pair_row_work"] == 30
    assert group["progress"]["derived"]["estimated_dense_remaining_rows"] == 28
    assert group["progress"]["derived"]["estimated_required_remaining_rows"] == 18
    assert group["progress"]["derived"]["estimated_remaining_rows"] == 28
    assert group["progress"]["derived"]["estimated_dense_remaining_ms"] == 227
    assert group["progress"]["derived"]["estimated_required_remaining_ms"] == 146
    assert group["progress"]["derived"]["estimated_remaining_ms"] == 227
    assert group["progress"]["derived"]["latest_elapsed_ms"] == 2500
    assert group["completed"]["derived"]["startup_blocking_elapsed_ms"] == 7500
    assert group["completed"]["derived"]["startup_blocking"] is True
    assert group["failed"]["event_type"] == "hsl.replay.failed"
    assert group["failed"]["status"] == "failed"
    assert group["failed"]["derived"]["latest_elapsed_ms"] == 8000
    assert "must-not-render" not in json.dumps(group, sort_keys=True)
    assert group["completed"]["derived"]["observed_work_pct"] == 75
    assert group["completed"]["derived"]["estimated_dense_remaining_rows"] == 10
    assert group["completed"]["derived"]["estimated_dense_remaining_ms"] == 250
    assert group["completed"]["derived"]["work_estimate_source"] == (
        "candidate_rows_terminal"
    )
    assert group["completed"]["derived"]["estimated_remaining_rows"] == 0
    assert group["completed"]["derived"]["estimated_remaining_ms"] == 0


@pytest.mark.parametrize("required_pairs", (0, 1))
def test_hsl_replay_profile_prefers_scanned_work_for_eta(required_pairs):
    data = {
        "stage": "pair_replay",
        "timeline_rows": 100,
        "pairs": 3,
        "required_pairs": required_pairs,
        "total_applied_rows": 10,
        "rows_per_second": 2.0,
        "scanned_rows": 75,
        "candidate_rows": 150,
        "total_scanned_rows": 150,
        "scanned_rows_per_second": 50.0,
        "pair_elapsed_s": 1.5,
        "secret": "must-not-render",
    }
    bounded = performance_report_module._bounded_hsl_replay_data(data)
    derived = performance_report_module._derive_hsl_replay_profile(bounded)

    assert bounded["scanned_rows"] == 75
    assert bounded["candidate_rows"] == 150
    assert bounded["pair_elapsed_s"] == 1.5
    assert "secret" not in bounded
    assert derived["throughput_source"] == "scanned_rows"
    assert derived["observed_applied_rows"] == 10
    assert derived["observed_scanned_rows"] == 150
    assert derived["observed_work_pct"] == 50.0
    assert derived["estimated_dense_remaining_rows"] == 150
    assert derived["estimated_dense_remaining_ms"] == 3000
    assert derived["estimated_required_remaining_rows"] == 0
    assert derived["estimated_required_remaining_ms"] == 0
    assert derived["work_estimate_source"] == "dense_rows_upper_bound"
    assert derived["estimated_remaining_rows"] == 150
    assert derived["estimated_remaining_ms"] == 3000

    terminal = performance_report_module._derive_hsl_replay_profile(
        {**bounded, "stage": "full_replay"}
    )
    assert terminal["work_estimate_source"] == "candidate_rows_terminal"
    assert terminal["estimated_candidate_pair_row_work"] == 150
    assert terminal["estimated_candidate_remaining_rows"] == 0
    assert terminal["estimated_remaining_rows"] == 0


def test_hsl_replay_profile_keeps_legacy_applied_work_fallback():
    data = {
        "timeline_rows": 100,
        "pairs": 3,
        "total_applied_rows": 10,
        "rows_per_second": 2.0,
    }
    derived = performance_report_module._derive_hsl_replay_profile(data)

    assert derived["throughput_source"] == "applied_rows_legacy"
    assert derived["observed_applied_rows"] == 10
    assert "observed_scanned_rows" not in derived
    assert derived["estimated_dense_remaining_rows"] == 290
    assert derived["estimated_dense_remaining_ms"] == 145000
    assert derived["work_estimate_source"] == "dense_rows_upper_bound"

    terminal = performance_report_module._derive_hsl_replay_profile(
        {**data, "stage": "full_replay"}
    )
    assert terminal["estimated_dense_remaining_rows"] == 290
    assert terminal["work_estimate_source"] == "legacy_terminal_no_candidate_rows"
    assert terminal["estimated_remaining_rows"] == 0
    assert terminal["estimated_remaining_ms"] == 0


def test_live_performance_report_hsl_replay_profile_stage_summary(tmp_path):
    rows = [
        _monitor_row(
            event_type="hsl.replay.progress",
            seq=1,
            ts=1000,
            exchange="binance",
            user="binance_01",
            component="risk.hsl",
            status="started",
            data={
                "stage": "price_history_fetch_started",
                "history_build_elapsed_s": 45.0,
            },
        ),
        _monitor_row(
            event_type="hsl.replay.progress",
            seq=2,
            ts=2000,
            exchange="gateio",
            user="gateio_01",
            component="risk.hsl",
            status="started",
            data={"stage": "pair_replay", "elapsed_s": 12.0},
        ),
        _monitor_row(
            event_type="hsl.replay.completed",
            seq=3,
            ts=3000,
            exchange="okx",
            user="okx_01",
            component="risk.hsl",
            status="succeeded",
            data={"stage": "full_replay", "full_elapsed_s": 18.0},
        ),
        _monitor_row(
            event_type="hsl.replay.failed",
            seq=4,
            ts=4000,
            exchange="kucoin",
            user="kucoin_01",
            component="risk.hsl",
            status="failed",
            data={"stage": "price_history_fetch_started", "elapsed_s": 3.0},
        ),
    ]
    for row in rows:
        exchange = row["exchange"]
        user = row["user"]
        _write_ndjson(
            tmp_path / "monitor" / exchange / user / "events" / "current.ndjson",
            [row],
        )

    report = build_live_performance_report(tmp_path / "monitor")
    profile = report["hsl_replay_profile"]

    assert profile["bot_count"] == 4
    assert profile["stage_counts"] == {
        "price_history_fetch_started": 2,
        "pair_replay": 1,
        "full_replay": 1,
    }
    assert profile["latest_status_counts"] == {
        "active": 2,
        "completed": 1,
        "failed": 1,
    }
    assert profile["latest_stage_counts"] == {
        "price_history_fetch_started": 2,
        "full_replay": 1,
        "pair_replay": 1,
    }
    assert profile["active_stage_counts"] == {
        "price_history_fetch_started": 1,
        "pair_replay": 1,
    }
    assert profile["active_bot_count"] == 2
    assert profile["completed_bot_count"] == 1
    assert profile["failed_bot_count"] == 1


def test_live_performance_report_hsl_replay_profile_exposes_protective_scorecard(
    tmp_path,
):
    fixtures = {
        ("binance", "binance_01"): [
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=1,
                ts=1000,
                exchange="binance",
                user="binance_01",
                component="risk.hsl",
                status="started",
                reason_code="history_loaded",
                data={
                    "signal_mode": "coin",
                    "stage": "loaded",
                    "history_format": "compact",
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2000,
                exchange="binance",
                user="binance_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="hsl_held_protective_ready",
                data={
                    "signal_mode": "coin",
                    "stage": "held_protective_ready",
                    "protective_elapsed_s": 12.3,
                    "startup_blocking_elapsed_s": 12.3,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=3,
                ts=3000,
                exchange="binance",
                user="binance_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "history_format": "compact",
                    "replay_strategy": "sparse_change_points",
                    "candidate_rows": 120,
                    "dense_equivalent_rows": 1200,
                    "candidate_reduction_pct": 90.0,
                    "dense_replay_pairs": 1,
                    "dense_fallback_pairs": 0,
                    "sparse_replay_pairs": 9,
                    "full_elapsed_s": 30.0,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=4,
                ts=1500,
                exchange="binance",
                user="binance_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="hsl_held_protective_ready",
                data={
                    "signal_mode": "coin",
                    "stage": "held_protective_ready",
                    "protective_elapsed_s": 99.0,
                    "startup_blocking_elapsed_s": 99.0,
                },
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=5,
                ts=2500,
                exchange="binance",
                user="binance_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "history_format": "timeline",
                    "full_elapsed_s": 99.0,
                },
            ),
        ],
        ("gateio", "gateio_01"): [
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=1,
                ts=1500,
                exchange="gateio",
                user="gateio_01",
                component="risk.hsl",
                status="started",
                reason_code="history_loaded",
                data={
                    "signal_mode": "coin",
                    "stage": "loaded",
                    "history_format": "timeline",
                    "replay_strategy": "dense_timeline",
                },
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2500,
                exchange="gateio",
                user="gateio_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="hsl_held_protective_ready",
                data={
                    "signal_mode": "coin",
                    "stage": "held_protective_ready",
                    "startup_blocking_elapsed_s": 20.0,
                },
            ),
        ],
        ("okx", "okx_01"): [
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=1,
                ts=3500,
                exchange="okx",
                user="okx_01",
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "history_format": "compact",
                    "replay_strategy": "sparse_change_points",
                    "candidate_rows": 100,
                    "dense_equivalent_rows": 1000,
                    "candidate_reduction_pct": 90.0,
                    "protective_elapsed_s": 5.0,
                    "startup_blocking_elapsed_s": 5.0,
                    "full_elapsed_s": 25.0,
                },
            ),
        ],
    }
    for (exchange, user), rows in fixtures.items():
        _write_ndjson(
            tmp_path / "monitor" / exchange / user / "events" / "current.ndjson",
            rows,
        )

    report = build_live_performance_report(tmp_path / "monitor")
    profile = report["hsl_replay_profile"]
    summary_profile = summarize_live_performance_report(report)["hsl_replay_profile"]
    groups = {group["bot"]: group for group in profile["groups"]}

    assert profile["history_format_counts"] == {"compact": 2, "timeline": 1}
    assert profile["replay_strategy_counts"] == {
        "sparse_change_points": 2,
        "dense_timeline": 1,
    }
    assert profile["protective_ready_bot_count"] == 3
    assert profile["protective_ready_elapsed_ms"]["count"] == 3
    assert profile["protective_ready_elapsed_ms"]["min"] == 5000
    assert profile["protective_ready_elapsed_ms"]["max"] == 20000
    assert profile["full_replay_elapsed_ms"]["count"] == 2
    assert profile["full_replay_elapsed_ms"]["max"] == 30000
    assert summary_profile["history_format_counts"] == {"compact": 2, "timeline": 1}
    assert summary_profile["replay_strategy_counts"] == {
        "sparse_change_points": 2,
        "dense_timeline": 1,
    }
    assert summary_profile["protective_ready_elapsed_ms"]["max"] == 20000
    assert summary_profile["full_replay_elapsed_ms"]["max"] == 30000
    assert groups["binance/binance_01"]["history_format"] == "compact"
    assert groups["binance/binance_01"]["replay_strategy"] == "sparse_change_points"
    assert groups["binance/binance_01"]["completed"]["data"]["candidate_rows"] == 120
    assert (
        groups["binance/binance_01"]["completed"]["data"]["dense_equivalent_rows"]
        == 1200
    )
    assert (
        groups["binance/binance_01"]["completed"]["data"]["candidate_reduction_pct"]
        == 90.0
    )
    assert groups["binance/binance_01"]["completed"]["data"][
        "dense_replay_pairs"
    ] == 1
    assert groups["binance/binance_01"]["completed"]["data"][
        "dense_fallback_pairs"
    ] == 0
    assert groups["binance/binance_01"]["completed"]["data"][
        "sparse_replay_pairs"
    ] == 9
    assert groups["binance/binance_01"]["protective_ready"]["derived"] == {
        "latest_elapsed_ms": 12300,
        "protective_elapsed_ms": 12300,
        "startup_blocking": True,
        "startup_blocking_elapsed_ms": 12300,
    }
    assert groups["gateio/gateio_01"]["history_format"] == "timeline"
    assert groups["gateio/gateio_01"]["replay_strategy"] == "dense_timeline"
    assert groups["gateio/gateio_01"]["protective_ready"]["derived"] == {
        "latest_elapsed_ms": 20000,
        "startup_blocking": True,
        "startup_blocking_elapsed_ms": 20000,
    }
    assert groups["okx/okx_01"]["history_format"] == "compact"
    assert "protective_ready" not in groups["okx/okx_01"]
    assert groups["okx/okx_01"]["completed"]["derived"][
        "protective_elapsed_ms"
    ] == 5000


def test_live_performance_report_completion_requires_explicit_protective_elapsed(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=1,
                ts=1000,
                component="risk.hsl",
                status="succeeded",
                reason_code="coin_history_replay_completed",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "startup_blocking_elapsed_s": 9.0,
                    "full_elapsed_s": 10.0,
                },
            )
        ],
    )

    profile = build_live_performance_report(tmp_path / "monitor")[
        "hsl_replay_profile"
    ]

    assert profile["protective_ready_bot_count"] == 0
    assert profile["protective_ready_elapsed_ms"] == {}
    assert profile["full_replay_elapsed_ms"]["max"] == 10000


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
                    "current_position_pairs": 1,
                    "price_replay_symbols": 3,
                    "skipped_price_symbols": 1,
                    "missing_price_symbols": 2,
                    "history_minutes": 10,
                    "start_ts": 1782492000000,
                    "end_ts": 1782492600000,
                    "record_start_ts": 1782492000000,
                    "rows": 9,
                    "elapsed_s": 4.5,
                    "history_build_elapsed_s": 17.25,
                    "price_history_fetch_elapsed_s": 16.5,
                    "timeline_replay_elapsed_s": 3.2,
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
        "current_position_pairs": 1,
        "end_ts": 1782492600000,
        "elapsed_s": 4.5,
        "error_type": "TimeoutError",
        "events": 2,
        "history_build_elapsed_s": 17.25,
        "history_minutes": 10,
        "missing_price_symbols": 2,
        "price_history_fetch_elapsed_s": 16.5,
        "price_replay_symbols": 3,
        "record_start_ts": 1782492000000,
        "rows": 9,
        "skipped_price_symbols": 1,
        "stage": "price_history_symbol_fetch_completed",
        "start_ts": 1782492000000,
        "timeframe": "1m",
        "timeline_replay_elapsed_s": 3.2,
    }
    assert report["hsl_replay_profile"]["groups"][0]["progress"]["derived"] == {
        "history_build_elapsed_ms": 17250,
        "latest_elapsed_ms": 4500,
        "observed_applied_rows": 9,
        "price_history_fetch_elapsed_ms": 16500,
        "throughput_source": "applied_rows_legacy",
        "timeline_replay_elapsed_ms": 3200,
    }
    assert "balance" not in rendered
    assert "equity" not in rendered
    assert "raw_payload" not in rendered
    assert "leak_marker" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert "drawdown_raw" not in rendered


def test_bounded_hsl_replay_data_bounds_string_values():
    bounded = performance_report_module._bounded_hsl_replay_data(
        {
            "stage": "x" * 100_000,
            "signal_mode": "coin",
        }
    )

    assert bounded == {
        "signal_mode": "coin",
        "stage": "x" * 120,
    }


def test_live_performance_report_hsl_history_elapsed_is_latest_when_replay_not_started(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(performance_report_module, "utc_ms", lambda: 121000)
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=1,
                ts=1000,
                exchange="gateio",
                user="gateio_01",
                component="risk.hsl",
                status="started",
                reason_code="hsl_price_history_fetch_started",
                data={
                    "stage": "price_history_fetch_started",
                    "history_build_elapsed_s": 91.25,
                    "history_minutes": 43201,
                    "price_replay_symbols": 24,
                    "replay_concurrency": 4,
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    group = report["hsl_replay_profile"]["groups"][0]

    assert group["bot"] == "gateio/gateio_01"
    assert group["latest"]["data"]["stage"] == "price_history_fetch_started"
    assert group["latest"]["derived"] == {
        "history_build_elapsed_ms": 91250,
        "latest_event_age_ms": 120000,
        "latest_elapsed_ms": 91250,
    }
    assert group["active_latest_event_age_ms"] == 120000


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


def test_live_performance_report_fill_refresh_from_existing_events(tmp_path):
    hyperliquid_dir = (
        tmp_path / "monitor" / "hyperliquid" / "hyperliquid_tradfi" / "events"
    )
    binance_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        hyperliquid_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=1,
                ts=1000,
                exchange="hyperliquid",
                user="hyperliquid_tradfi",
                component="fills.refresh",
                status="failed",
                level="error",
                reason_code="fill_refresh_failed",
                data={
                    "source": "exchange",
                    "refresh_mode": "periodic",
                    "elapsed_ms": 12244,
                    "history_scope": "window",
                    "event_count_after": 2704,
                    "coverage_ready_after": False,
                    "coverage_reason_after": "window_uncovered",
                    "retry_count": 1,
                    "next_retry_in_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=2000,
                exchange="hyperliquid",
                user="hyperliquid_tradfi",
                component="fills.refresh",
                status="succeeded",
                level="debug",
                reason_code="fill_cache_ready",
                data={
                    "source": "exchange",
                    "refresh_mode": "periodic",
                    "elapsed_ms": 420,
                    "history_scope": "all",
                    "event_count_after": 2705,
                    "new_count": 1,
                    "enriched_count": 1,
                    "pending_pnl_count": 0,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )
    _write_ndjson(
        binance_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=3000,
                exchange="binance",
                user="binance_01",
                component="fills.refresh",
                status="succeeded",
                level="debug",
                reason_code="fill_cache_ready",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "elapsed_ms": 40,
                    "history_scope": "all",
                    "event_count_after": 100,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)
    refresh = report["fill_refresh"]
    operations = _groups_by_operation(report)
    operation_durations = _operation_duration_groups_by_operation(report)

    assert refresh["total_events"] == 3
    assert refresh["bot_count"] == 2
    assert refresh["event_types"] == {"fills.refresh_summary": 3}
    assert refresh["statuses"] == {"succeeded": 2, "failed": 1}
    assert refresh["reason_codes"] == {
        "fill_cache_ready": 2,
        "fill_refresh_failed": 1,
    }
    assert refresh["error_types"] == {"RequestTimeout": 1}
    assert refresh["failed_groups"] == 1
    assert refresh["latest_failed_groups"] == 0
    assert refresh["recovered_groups"] == 1
    recovered = refresh["groups"][0]
    assert recovered["bot"] == "hyperliquid/hyperliquid_tradfi"
    assert recovered["source"] == "exchange"
    assert recovered["refresh_mode"] == "periodic"
    assert recovered["total_events"] == 2
    assert recovered["failed"] == 1
    assert recovered["failure_pct"] == 50
    assert recovered["recovered"] is True
    assert recovered["latest"]["status"] == "succeeded"
    assert recovered["latest"]["data"]["history_scope"] == "all"
    assert recovered["history_scopes"] == {"window": 1, "all": 1}
    assert recovered["coverage_ready_after_true"] == 1
    assert recovered["coverage_ready_after_false"] == 1
    assert recovered["coverage_reasons_after"] == {"window_uncovered": 1}
    assert recovered["elapsed_ms"]["max"] == 12244
    assert recovered["elapsed_ms"]["min"] == 420
    assert recovered["event_count_after"]["max"] == 2705
    assert recovered["new_count"]["max"] == 1
    assert recovered["error_types"] == {"RequestTimeout": 1}
    assert operations["fills_refresh.elapsed"]["trading_impact"] == (
        "blocks_or_delays_hsl_readiness"
    )
    assert operation_durations["fills_refresh.elapsed"]["operation_category"] == (
        "fill_refresh"
    )
    assert summary["fill_refresh"]["groups_truncated"] is True
    assert len(summary["fill_refresh"]["groups"]) == 1


def test_live_performance_report_fill_refresh_whitelists_values(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=1,
                ts=1000,
                component="fills.refresh",
                status="failed",
                reason_code="fill_refresh_failed",
                data={
                    "source": "exchange",
                    "refresh_mode": "periodic",
                    "elapsed_ms": 1200,
                    "error_type": "RequestTimeout",
                    "error": "GET https://example.test?api_key=secret leak_marker",
                    "coverage_after": {
                        "raw": "not copied",
                        "path": "/root/passivbot/private",
                    },
                    "debug": {"raw_payload": "not copied"},
                    "path": "/home/operator/private/cache.json",
                    "api_key": "secret",
                    "balance": 1000,
                    "equity": 999,
                    "fill_id": "raw_fill_id",
                    "client_order_id": "raw_client_order_id",
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    rendered = json.dumps(report["fill_refresh"], sort_keys=True)

    assert report["fill_refresh"]["groups"][0]["latest"]["data"] == {
        "elapsed_ms": 1200,
        "error_type": "RequestTimeout",
        "refresh_mode": "periodic",
        "source": "exchange",
    }
    assert "leak_marker" not in rendered
    assert "raw_payload" not in rendered
    assert "not copied" not in rendered
    assert "/home/operator" not in rendered
    assert "/root/passivbot" not in rendered
    assert "api_key" not in rendered
    assert "secret" not in rendered
    assert "balance" not in rendered
    assert "equity" not in rendered
    assert "raw_fill_id" not in rendered
    assert "raw_client_order_id" not in rendered


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
                event_type="hsl.red_finalized_without_order",
                seq=3,
                ts=2500,
                component="risk.hsl",
                status="succeeded",
                reason_code="hsl_red_finalized_without_exchange_order",
                symbol="BTC/USDT:USDT",
                pside="long",
                data={
                    "cooldown_until_ms": 999999,
                    "stop_event_timestamp_ms": 2400,
                    "balance": 12345.67,
                    "secret_marker": "flat-secret",
                },
            ),
            _monitor_row(
                event_type="risk.mode_changed",
                seq=4,
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
                seq=5,
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
                event_type="trailing.status",
                seq=6,
                ts=4300,
                component="risk.trailing.status",
                status="pending",
                reason_code="trailing_status",
                symbol="XLM/USDT:USDT",
                pside="long",
                data={
                    "threshold_pct": 0.031,
                    "threshold_price": 0.20123,
                    "retracement_pct": 0.012,
                    "retracement_price": 0.19876,
                    "projected_retracement_price": 0.19765,
                    "secret_marker": "trailing-secret",
                },
            ),
            _monitor_row(
                event_type="risk.realized_loss_gate_blocked",
                seq=7,
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
                seq=8,
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

    assert risk["total_events"] == 7
    assert risk["event_types"] == {
        "hsl.red_finalized_without_order": 1,
        "hsl.red_triggered": 1,
        "hsl.status": 1,
        "risk.mode_changed": 1,
        "risk.realized_loss_gate_blocked": 1,
        "trailing.status": 1,
        "unstuck.selection": 1,
    }
    assert "hsl.replay.progress" not in risk["event_types"]
    assert groups["hsl.status"]["statuses"] == {"degraded": 1}
    assert groups["hsl.status"]["reason_codes"] == {"cooldown_active": 1}
    assert groups["hsl.status"]["symbols_sample"] == ["BTC/USDT:USDT"]
    assert groups["hsl.status"]["psides"] == {"long": 1}
    assert groups["hsl.red_finalized_without_order"]["statuses"] == {
        "succeeded": 1
    }
    assert groups["hsl.red_finalized_without_order"]["reason_codes"] == {
        "hsl_red_finalized_without_exchange_order": 1
    }
    assert groups["hsl.red_finalized_without_order"]["symbols_sample"] == [
        "BTC/USDT:USDT"
    ]
    assert groups["unstuck.selection"]["symbols_sample"] == ["ETH/USDT:USDT"]
    assert groups["trailing.status"]["symbols_sample"] == ["XLM/USDT:USDT"]
    assert groups["trailing.status"]["statuses"] == {"pending": 1}
    assert groups["trailing.status"]["reason_codes"] == {"trailing_status": 1}
    assert groups["trailing.status"]["psides"] == {"long": 1}
    assert groups["risk.realized_loss_gate_blocked"]["symbols_sample"] == ["SOL/USDT:USDT"]
    assert groups["risk.realized_loss_gate_blocked"]["statuses"] == {"deferred": 1}
    assert "98765.43" not in rendered
    assert "45678.9" not in rendered
    assert "1111.22" not in rendered
    assert "hsl-secret" not in rendered
    assert "red-secret" not in rendered
    assert "flat-secret" not in rendered
    assert "mode-secret" not in rendered
    assert "unstuck-secret" not in rendered
    assert "trailing-secret" not in rendered
    assert "loss-gate-secret" not in rendered
    assert "0.20123" not in rendered
    assert "0.19876" not in rendered
    assert "0.19765" not in rendered
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


def test_live_performance_report_resource_pressure_from_health_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(performance_report_module, "utc_ms", lambda: 5000)
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
                    "cpu_percent": 12.0,
                    "system_memory_total_bytes": 16_000,
                    "system_memory_available_bytes": 9_000,
                    "system_memory_percent": 40.0,
                    "swap_total_bytes": 4_000,
                    "swap_used_bytes": 800,
                    "swap_percent": 20.0,
                    "open_fds": 11,
                    "loadavg_1m": 0.25,
                    "cpu_count": 1,
                    "health_summary_lag_ms": 0,
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
                    "cpu_percent": 24.0,
                    "system_memory_total_bytes": 16_000,
                    "system_memory_available_bytes": 8_000,
                    "system_memory_percent": 50.0,
                    "swap_total_bytes": 4_000,
                    "swap_used_bytes": 1_000,
                    "swap_percent": 25.0,
                    "open_fds": 13,
                    "loadavg_1m": 0.75,
                    "cpu_count": 1,
                    "health_summary_lag_ms": 2500,
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
    assert pressure["latest_event_age_ms_max"] == 3000
    assert pressure["latest_event_age_reporting_bots"] == 1
    assert pressure["latest_event_queue_depth_max"] == 5
    assert pressure["latest_event_dropped_total_sum"] == 4
    assert pressure["latest_event_sink_error_total_sum"] == 1
    assert pressure["latest_event_degraded_count_sum"] == 3
    assert pressure["event_pipeline_unhealthy_bots"] == 1
    assert group["bot"] == "binance/binance_01"
    assert group["count"] == 2
    assert group["latest_ts"] == 2000
    assert group["latest_event_age_ms"] == 3000
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
    assert group["fields"]["cpu_percent"] == {
        "latest": 24,
        "count": 2,
        "min": 12,
        "max": 24,
        "mean": 18,
        "median": 18,
        "p95": 23,
    }
    assert group["fields"]["system_memory_available_bytes"]["latest"] == 8_000
    assert group["fields"]["system_memory_percent"] == {
        "latest": 50,
        "count": 2,
        "min": 40,
        "max": 50,
        "mean": 45,
        "median": 45,
        "p95": 50,
    }
    assert group["fields"]["swap_used_bytes"]["latest"] == 1_000
    assert group["fields"]["swap_percent"]["max"] == 25
    assert group["fields"]["event_queue_depth"]["max"] == 5
    assert group["fields"]["event_queue_depth"]["p95"] == 5
    assert group["fields"]["health_summary_lag_ms"] == {
        "latest": 2500,
        "count": 2,
        "min": 0,
        "max": 2500,
        "mean": 1250,
        "median": 1250,
        "p95": 2375,
    }
    assert group["fields"]["event_dropped_total"]["latest"] == 4
    assert group["fields"]["event_sink_error_total"]["latest"] == 1
    assert group["fields"]["event_degraded_count"]["latest"] == 3
    assert group["latest_event_drop_counts"] == {"queue_full": 4}
    assert group["latest_event_sink_error_counts"] == {"disk": 1}
    assert group["latest_event_pipeline_stopping"] is False
    assert group["latest_event_pipeline_worker_alive"] is True


def test_live_performance_report_resource_pressure_projects_event_pipeline_timing(
    tmp_path,
):
    binance_events = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    okx_events = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    _write_ndjson(
        binance_events / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                data={
                    "event_pipeline_timing_window_ms": 60,
                    "event_pipeline_processed_count": 2,
                    "event_queue_wait_ms_total": 1.5,
                    "event_queue_wait_ms_max": 1.0,
                    "event_worker_service_ms_total": 2.0,
                    "event_worker_service_ms_max": 2.0,
                    "event_structured_sink_write_count": 2,
                    "event_structured_sink_service_ms_total": 1.2,
                    "event_structured_sink_service_ms_max": 0.8,
                    "event_monitor_sink_write_count": 2,
                    "event_monitor_sink_service_ms_total": 0.8,
                    "event_monitor_sink_service_ms_max": 0.5,
                    "event_monitor_publisher_retention_inventory_ms_total": 99.0,
                    "event_monitor_publisher_retention_inventory_ms_max": 98.0,
                    "event_monitor_publisher_retention_age_unlink_ms_total": 97.0,
                    "event_monitor_publisher_retention_age_unlink_ms_max": 96.0,
                    "event_monitor_publisher_retention_cap_unlink_ms_total": 95.0,
                    "event_monitor_publisher_retention_cap_unlink_ms_max": 94.0,
                    "event_monitor_publisher_retention_inventory_entries_visited": 93,
                    "event_monitor_publisher_retention_inventory_candidates": 92,
                    "event_monitor_publisher_retention_age_deleted": 91,
                    "event_monitor_publisher_retention_cap_deleted": 90,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                data={
                    "event_pipeline_timing_window_ms": 120,
                    "event_pipeline_processed_count": 5,
                    "event_queue_wait_ms_total": 4.5,
                    "event_queue_wait_ms_max": 3.0,
                    "event_worker_service_ms_total": 7.5,
                    "event_worker_service_ms_max": 4.0,
                    "event_structured_sink_write_count": 5,
                    "event_structured_sink_service_ms_total": 3.5,
                    "event_structured_sink_service_ms_max": 1.5,
                    "event_monitor_sink_write_count": 5,
                    "event_monitor_sink_service_ms_total": 2.5,
                    "event_monitor_sink_service_ms_max": 1.2,
                    "event_monitor_prepare_ms_total": 0.3,
                    "event_monitor_prepare_ms_max": 0.2,
                    "event_monitor_publisher_lock_wait_ms_total": 0.2,
                    "event_monitor_publisher_lock_wait_ms_max": 0.1,
                    "event_monitor_publisher_rotation_ms_total": 0.4,
                    "event_monitor_publisher_rotation_ms_max": 0.25,
                    "event_monitor_publisher_persist_ms_total": 1.0,
                    "event_monitor_publisher_persist_ms_max": 0.6,
                    "event_monitor_publisher_maintenance_ms_total": 0.5,
                    "event_monitor_publisher_maintenance_ms_max": 0.3,
                    "event_monitor_publisher_manifest_checkpoint_count": 2,
                    "event_monitor_publisher_manifest_checkpoint_ms_total": 0.1,
                    "event_monitor_publisher_manifest_checkpoint_ms_max": 0.08,
                    "event_monitor_publisher_retention_run_count": 3,
                    "event_monitor_publisher_retention_ms_total": 0.2,
                    "event_monitor_publisher_retention_ms_max": 0.15,
                    "event_monitor_publisher_retention_inventory_ms_total": 0.35,
                    "event_monitor_publisher_retention_inventory_ms_max": 0.22,
                    "event_monitor_publisher_retention_age_filter_ms_total": 0.28,
                    "event_monitor_publisher_retention_age_filter_ms_max": 0.19,
                    "event_monitor_publisher_retention_cap_prune_ms_total": 0.31,
                    "event_monitor_publisher_retention_cap_prune_ms_max": 0.2,
                    "event_monitor_publisher_retention_age_unlink_ms_total": 0.18,
                    "event_monitor_publisher_retention_age_unlink_ms_max": 0.11,
                    "event_monitor_publisher_retention_cap_unlink_ms_total": 0.27,
                    "event_monitor_publisher_retention_cap_unlink_ms_max": 0.16,
                    "event_monitor_publisher_retention_inventory_entries_visited": 14,
                    "event_monitor_publisher_retention_inventory_candidates": 5,
                    "event_monitor_publisher_retention_age_deleted": 2,
                    "event_monitor_publisher_retention_cap_deleted": 1,
                },
            ),
        ],
    )
    _write_ndjson(
        okx_events / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                exchange="okx",
                user="okx_01",
                seq=3,
                ts=1500,
                data={
                    "event_pipeline_timing_window_ms": 90,
                    "event_pipeline_processed_count": 3,
                    "event_queue_wait_ms_total": 2.25,
                    "event_queue_wait_ms_max": 2.25,
                    "event_worker_service_ms_total": 3.5,
                    "event_worker_service_ms_max": 3.5,
                    "event_structured_sink_write_count": 3,
                    "event_structured_sink_service_ms_total": 2.0,
                    "event_structured_sink_service_ms_max": 1.0,
                    "event_monitor_sink_write_count": 3,
                    "event_monitor_sink_service_ms_total": 1.5,
                    "event_monitor_sink_service_ms_max": 0.9,
                    "event_monitor_prepare_ms_total": 0.2,
                    "event_monitor_prepare_ms_max": 0.2,
                    "event_monitor_publisher_lock_wait_ms_total": 0.1,
                    "event_monitor_publisher_lock_wait_ms_max": 0.1,
                    "event_monitor_publisher_rotation_ms_total": 0.2,
                    "event_monitor_publisher_rotation_ms_max": 0.2,
                    "event_monitor_publisher_persist_ms_total": 0.5,
                    "event_monitor_publisher_persist_ms_max": 0.5,
                    "event_monitor_publisher_maintenance_ms_total": 0.3,
                    "event_monitor_publisher_maintenance_ms_max": 0.3,
                    "event_monitor_publisher_manifest_checkpoint_count": 1,
                    "event_monitor_publisher_manifest_checkpoint_ms_total": 0.1,
                    "event_monitor_publisher_manifest_checkpoint_ms_max": 0.1,
                    "event_monitor_publisher_retention_run_count": 1,
                    "event_monitor_publisher_retention_ms_total": 0.1,
                    "event_monitor_publisher_retention_ms_max": 0.1,
                    "event_monitor_publisher_retention_inventory_ms_total": 0.2,
                    "event_monitor_publisher_retention_inventory_ms_max": 0.14,
                    "event_monitor_publisher_retention_age_filter_ms_total": 0.17,
                    "event_monitor_publisher_retention_age_filter_ms_max": 0.12,
                    "event_monitor_publisher_retention_cap_prune_ms_total": 0.16,
                    "event_monitor_publisher_retention_cap_prune_ms_max": 0.13,
                    "event_monitor_publisher_retention_age_unlink_ms_total": 0.15,
                    "event_monitor_publisher_retention_age_unlink_ms_max": 0.1,
                    "event_monitor_publisher_retention_cap_unlink_ms_total": 0.11,
                    "event_monitor_publisher_retention_cap_unlink_ms_max": 0.08,
                    "event_monitor_publisher_retention_inventory_entries_visited": 7,
                    "event_monitor_publisher_retention_inventory_candidates": 3,
                    "event_monitor_publisher_retention_age_deleted": 1,
                    "event_monitor_publisher_retention_cap_deleted": 0,
                },
            )
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    assert report["ok"] is True
    pressure = report["resource_pressure"]
    groups = {group["bot"]: group for group in pressure["groups"]}
    binance_fields = groups["binance/binance_01"]["fields"]
    okx_fields = groups["okx/okx_01"]["fields"]
    assert binance_fields["event_pipeline_processed_count"] == {
        "latest": 5,
        "count": 2,
        "min": 2,
        "max": 5,
        "mean": 4,
        "median": 4,
        "p95": 5,
    }
    assert binance_fields["event_queue_wait_ms_total"] == {
        "latest": 4.5,
        "count": 2,
        "min": 1.5,
        "max": 4.5,
        "mean": 3,
        "median": 3,
        "p95": 4.35,
    }
    assert binance_fields["event_worker_service_ms_max"]["latest"] == 4
    assert binance_fields["event_structured_sink_service_ms_total"]["latest"] == 3.5
    assert binance_fields["event_monitor_sink_write_count"]["latest"] == 5
    assert binance_fields["event_monitor_publisher_persist_ms_total"] == {
        "latest": 1,
        "count": 1,
        "min": 1,
        "max": 1,
        "mean": 1,
        "median": 1,
        "p95": 1,
    }
    assert binance_fields["event_monitor_publisher_manifest_checkpoint_count"] == {
        "latest": 2,
        "count": 1,
        "min": 2,
        "max": 2,
        "mean": 2,
        "median": 2,
        "p95": 2,
    }
    assert binance_fields["event_monitor_publisher_retention_age_filter_ms_total"][
        "count"
    ] == 1
    assert binance_fields["event_monitor_publisher_retention_age_filter_ms_max"][
        "count"
    ] == 1
    assert binance_fields["event_monitor_publisher_retention_cap_prune_ms_total"][
        "count"
    ] == 1
    assert binance_fields["event_monitor_publisher_retention_cap_prune_ms_max"][
        "count"
    ] == 1
    retention_fields = (
        "event_monitor_publisher_retention_inventory_ms_total",
        "event_monitor_publisher_retention_inventory_ms_max",
        "event_monitor_publisher_retention_age_filter_ms_total",
        "event_monitor_publisher_retention_age_filter_ms_max",
        "event_monitor_publisher_retention_cap_prune_ms_total",
        "event_monitor_publisher_retention_cap_prune_ms_max",
        "event_monitor_publisher_retention_age_unlink_ms_total",
        "event_monitor_publisher_retention_age_unlink_ms_max",
        "event_monitor_publisher_retention_cap_unlink_ms_total",
        "event_monitor_publisher_retention_cap_unlink_ms_max",
        "event_monitor_publisher_retention_inventory_entries_visited",
        "event_monitor_publisher_retention_inventory_candidates",
        "event_monitor_publisher_retention_age_deleted",
        "event_monitor_publisher_retention_cap_deleted",
    )
    assert {
        key: binance_fields[key]["latest"] for key in retention_fields
    } == {
        "event_monitor_publisher_retention_inventory_ms_total": 0.35,
        "event_monitor_publisher_retention_inventory_ms_max": 0.22,
        "event_monitor_publisher_retention_age_filter_ms_total": 0.28,
        "event_monitor_publisher_retention_age_filter_ms_max": 0.19,
        "event_monitor_publisher_retention_cap_prune_ms_total": 0.31,
        "event_monitor_publisher_retention_cap_prune_ms_max": 0.2,
        "event_monitor_publisher_retention_age_unlink_ms_total": 0.18,
        "event_monitor_publisher_retention_age_unlink_ms_max": 0.11,
        "event_monitor_publisher_retention_cap_unlink_ms_total": 0.27,
        "event_monitor_publisher_retention_cap_unlink_ms_max": 0.16,
        "event_monitor_publisher_retention_inventory_entries_visited": 14,
        "event_monitor_publisher_retention_inventory_candidates": 5,
        "event_monitor_publisher_retention_age_deleted": 2,
        "event_monitor_publisher_retention_cap_deleted": 1,
    }
    assert {
        key: okx_fields[key]["latest"] for key in retention_fields
    } == {
        "event_monitor_publisher_retention_inventory_ms_total": 0.2,
        "event_monitor_publisher_retention_inventory_ms_max": 0.14,
        "event_monitor_publisher_retention_age_filter_ms_total": 0.17,
        "event_monitor_publisher_retention_age_filter_ms_max": 0.12,
        "event_monitor_publisher_retention_cap_prune_ms_total": 0.16,
        "event_monitor_publisher_retention_cap_prune_ms_max": 0.13,
        "event_monitor_publisher_retention_age_unlink_ms_total": 0.15,
        "event_monitor_publisher_retention_age_unlink_ms_max": 0.1,
        "event_monitor_publisher_retention_cap_unlink_ms_total": 0.11,
        "event_monitor_publisher_retention_cap_unlink_ms_max": 0.08,
        "event_monitor_publisher_retention_inventory_entries_visited": 7,
        "event_monitor_publisher_retention_inventory_candidates": 3,
        "event_monitor_publisher_retention_age_deleted": 1,
        "event_monitor_publisher_retention_cap_deleted": 0,
    }
    assert {
        key: pressure[key]
        for key in (
            "latest_event_pipeline_processed_total",
            "latest_event_pipeline_timing_window_ms_max",
            "latest_event_queue_wait_ms_total_sum",
            "latest_event_queue_wait_ms_max",
            "latest_event_worker_service_ms_total_sum",
            "latest_event_worker_service_ms_max",
            "latest_event_structured_sink_write_count_sum",
            "latest_event_structured_sink_service_ms_total_sum",
            "latest_event_structured_sink_service_ms_max",
            "latest_event_monitor_sink_write_count_sum",
            "latest_event_monitor_sink_service_ms_total_sum",
            "latest_event_monitor_sink_service_ms_max",
            "latest_event_monitor_prepare_ms_total_sum",
            "latest_event_monitor_prepare_ms_max",
            "latest_event_monitor_publisher_lock_wait_ms_total_sum",
            "latest_event_monitor_publisher_lock_wait_ms_max",
            "latest_event_monitor_publisher_rotation_ms_total_sum",
            "latest_event_monitor_publisher_rotation_ms_max",
            "latest_event_monitor_publisher_persist_ms_total_sum",
            "latest_event_monitor_publisher_persist_ms_max",
            "latest_event_monitor_publisher_maintenance_ms_total_sum",
            "latest_event_monitor_publisher_maintenance_ms_max",
            "latest_event_monitor_publisher_manifest_checkpoint_count_sum",
            "latest_event_monitor_publisher_manifest_checkpoint_ms_total_sum",
            "latest_event_monitor_publisher_manifest_checkpoint_ms_max",
            "latest_event_monitor_publisher_retention_run_count_sum",
            "latest_event_monitor_publisher_retention_ms_total_sum",
            "latest_event_monitor_publisher_retention_ms_max",
            "latest_event_monitor_publisher_retention_inventory_ms_total_sum",
            "latest_event_monitor_publisher_retention_inventory_ms_max",
            "latest_event_monitor_publisher_retention_age_filter_ms_total_sum",
            "latest_event_monitor_publisher_retention_age_filter_ms_max",
            "latest_event_monitor_publisher_retention_cap_prune_ms_total_sum",
            "latest_event_monitor_publisher_retention_cap_prune_ms_max",
            "latest_event_monitor_publisher_retention_age_unlink_ms_total_sum",
            "latest_event_monitor_publisher_retention_age_unlink_ms_max",
            "latest_event_monitor_publisher_retention_cap_unlink_ms_total_sum",
            "latest_event_monitor_publisher_retention_cap_unlink_ms_max",
            "latest_event_monitor_publisher_retention_inventory_entries_visited_sum",
            "latest_event_monitor_publisher_retention_inventory_candidates_sum",
            "latest_event_monitor_publisher_retention_age_deleted_sum",
            "latest_event_monitor_publisher_retention_cap_deleted_sum",
        )
    } == {
        "latest_event_pipeline_processed_total": 8,
        "latest_event_pipeline_timing_window_ms_max": 120,
        "latest_event_queue_wait_ms_total_sum": 6.75,
        "latest_event_queue_wait_ms_max": 3,
        "latest_event_worker_service_ms_total_sum": 11,
        "latest_event_worker_service_ms_max": 4,
        "latest_event_structured_sink_write_count_sum": 8,
        "latest_event_structured_sink_service_ms_total_sum": 5.5,
        "latest_event_structured_sink_service_ms_max": 1.5,
        "latest_event_monitor_sink_write_count_sum": 8,
        "latest_event_monitor_sink_service_ms_total_sum": 4,
        "latest_event_monitor_sink_service_ms_max": 1.2,
        "latest_event_monitor_prepare_ms_total_sum": 0.5,
        "latest_event_monitor_prepare_ms_max": 0.2,
        "latest_event_monitor_publisher_lock_wait_ms_total_sum": 0.3,
        "latest_event_monitor_publisher_lock_wait_ms_max": 0.1,
        "latest_event_monitor_publisher_rotation_ms_total_sum": 0.6,
        "latest_event_monitor_publisher_rotation_ms_max": 0.25,
        "latest_event_monitor_publisher_persist_ms_total_sum": 1.5,
        "latest_event_monitor_publisher_persist_ms_max": 0.6,
        "latest_event_monitor_publisher_maintenance_ms_total_sum": 0.8,
        "latest_event_monitor_publisher_maintenance_ms_max": 0.3,
        "latest_event_monitor_publisher_manifest_checkpoint_count_sum": 3,
        "latest_event_monitor_publisher_manifest_checkpoint_ms_total_sum": 0.2,
        "latest_event_monitor_publisher_manifest_checkpoint_ms_max": 0.1,
        "latest_event_monitor_publisher_retention_run_count_sum": 4,
        "latest_event_monitor_publisher_retention_ms_total_sum": 0.3,
        "latest_event_monitor_publisher_retention_ms_max": 0.15,
        "latest_event_monitor_publisher_retention_inventory_ms_total_sum": 0.55,
        "latest_event_monitor_publisher_retention_inventory_ms_max": 0.22,
        "latest_event_monitor_publisher_retention_age_filter_ms_total_sum": 0.45,
        "latest_event_monitor_publisher_retention_age_filter_ms_max": 0.19,
        "latest_event_monitor_publisher_retention_cap_prune_ms_total_sum": 0.47,
        "latest_event_monitor_publisher_retention_cap_prune_ms_max": 0.2,
        "latest_event_monitor_publisher_retention_age_unlink_ms_total_sum": 0.33,
        "latest_event_monitor_publisher_retention_age_unlink_ms_max": 0.11,
        "latest_event_monitor_publisher_retention_cap_unlink_ms_total_sum": 0.38,
        "latest_event_monitor_publisher_retention_cap_unlink_ms_max": 0.16,
        "latest_event_monitor_publisher_retention_inventory_entries_visited_sum": 21,
        "latest_event_monitor_publisher_retention_inventory_candidates_sum": 8,
        "latest_event_monitor_publisher_retention_age_deleted_sum": 3,
        "latest_event_monitor_publisher_retention_cap_deleted_sum": 1,
    }


def test_live_performance_report_resource_pressure_whitelists_health_fields(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=0,
                ts=1000,
                component="monitor.health",
                data={
                    "event_monitor_publisher_retention_inventory_ms_total": 9.0,
                    "event_monitor_publisher_retention_inventory_ms_max": 8.0,
                    "event_monitor_publisher_retention_age_filter_ms_total": 7.5,
                    "event_monitor_publisher_retention_age_filter_ms_max": 6.5,
                    "event_monitor_publisher_retention_cap_prune_ms_total": 5.5,
                    "event_monitor_publisher_retention_cap_prune_ms_max": 4.5,
                    "event_monitor_publisher_retention_age_unlink_ms_total": 7.0,
                    "event_monitor_publisher_retention_age_unlink_ms_max": 6.0,
                    "event_monitor_publisher_retention_cap_unlink_ms_total": 5.0,
                    "event_monitor_publisher_retention_cap_unlink_ms_max": 4.0,
                    "event_monitor_publisher_retention_inventory_entries_visited": 3,
                    "event_monitor_publisher_retention_inventory_candidates": 2,
                    "event_monitor_publisher_retention_age_deleted": 1,
                    "event_monitor_publisher_retention_cap_deleted": 1,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=2000,
                component="monitor.health",
                data={
                    "rss_bytes": 1200,
                    "system_memory_percent": 60.0,
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

    assert report["ok"] is True
    assert report["resource_pressure"]["groups"][0]["fields"]["rss_bytes"]["latest"] == 1200
    assert (
        report["resource_pressure"]["groups"][0]["fields"]["system_memory_percent"][
            "latest"
        ]
        == 60
    )
    assert not any(
        key.startswith("event_monitor_publisher_manifest_checkpoint")
        or key.startswith("event_monitor_publisher_retention")
        for key in report["resource_pressure"]["groups"][0]["fields"]
    )
    assert "balance_raw" not in rendered
    assert "balance_snapped" not in rendered
    assert "leak-equity" not in rendered
    assert "leak-pnl" not in rendered
    assert "raw-balance" not in rendered
    assert "snapped-balance" not in rendered
    assert "raw-payload" not in rendered


def test_live_performance_report_resource_pressure_omits_missing_latest_rotated_field(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    stale_rotated = events_dir / "2026-06-24T00-00-00.ndjson"
    rotated = events_dir / "2026-06-25T00-00-00.ndjson"
    current = events_dir / "current.ndjson"
    _write_ndjson(
        stale_rotated,
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=500,
                data={"elapsed_ms": 1000},
            )
        ],
    )
    _write_ndjson(
        rotated,
        [
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=1000,
                component="monitor.health",
                data={"rss_bytes": 1000},
            )
        ],
    )
    _write_ndjson(
        current,
        [
            _monitor_row(
                event_type="health.summary",
                seq=3,
                ts=2000,
                component="monitor.health",
                data={"rss_bytes": None, "event_queue_depth": 0},
            )
        ],
    )
    os.utime(stale_rotated, (1000, 1000))
    os.utime(rotated, (1500, 1500))
    os.utime(current, (2000, 2000))

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files=2,
    )
    group = report["resource_pressure"]["groups"][0]

    assert report["files_scanned"] == 2
    assert group["count"] == 2
    assert "rss_bytes" not in group["fields"]
    assert group["fields"]["event_queue_depth"]["latest"] == 0


def test_live_performance_report_resource_pressure_latest_snapshot_clears_invalid_fields(tmp_path):
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
                    "memory_percent": 6.5,
                    "cpu_percent": 12.0,
                    "event_queue_depth": 5,
                    "event_pipeline_worker_alive": True,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                component="monitor.health",
                data={
                    "rss_bytes": None,
                    "memory_percent": float("nan"),
                    "cpu_percent": -1,
                    "event_queue_depth": 0,
                    "event_pipeline_worker_alive": None,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    group = report["resource_pressure"]["groups"][0]

    assert group["count"] == 2
    assert "rss_bytes" not in group["fields"]
    assert "memory_percent" not in group["fields"]
    assert "cpu_percent" not in group["fields"]
    assert group["fields"]["event_queue_depth"]["latest"] == 0
    assert group["fields"]["event_queue_depth"]["count"] == 2
    assert group["fields"]["event_queue_depth"]["min"] == 0
    assert group["fields"]["event_queue_depth"]["max"] == 5
    assert "latest_event_pipeline_worker_alive" not in group


def test_live_performance_report_resource_pressure_unordered_row_keeps_ordered_latest(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                component="monitor.health",
                data={"rss_bytes": 1000},
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts="invalid",
                component="monitor.health",
                data={"rss_bytes": None},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    group = report["resource_pressure"]["groups"][0]

    assert group["count"] == 2
    assert group["latest_ts"] == 1000
    assert group["fields"]["rss_bytes"]["latest"] == 1000
    assert group["fields"]["rss_bytes"]["count"] == 1


def test_live_performance_report_resource_pressure_summary_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(performance_report_module, "utc_ms", lambda: 5000)
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
                data={
                    "rss_bytes": 1000,
                    "event_queue_depth": 1,
                    "event_dropped_total": 1,
                    "event_pipeline_worker_alive": True,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={
                    "rss_bytes": 2000,
                    "event_queue_depth": 2,
                    "event_dropped_total": 0,
                    "event_pipeline_worker_alive": False,
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["resource_pressure"]["bots"] == 2
    assert report["resource_pressure"]["latest_event_age_ms_max"] == 4000
    assert report["resource_pressure"]["latest_event_age_reporting_bots"] == 2
    assert report["resource_pressure"]["latest_event_queue_depth_max"] == 2
    assert report["resource_pressure"]["latest_event_dropped_total_sum"] == 1
    assert report["resource_pressure"]["latest_event_sink_error_total_sum"] == 0
    assert report["resource_pressure"]["latest_event_degraded_count_sum"] == 0
    assert report["resource_pressure"]["event_pipeline_unhealthy_bots"] == 2
    assert len(summary["resource_pressure"]["groups"]) == 1
    assert summary["resource_pressure"]["groups_truncated"] is True
    assert summary["resource_pressure"]["groups"][0]["bot"] == "binance/binance_01"
    assert "latest_event_age_ms" in summary["resource_pressure"]["groups"][0]
    assert summary["resource_pressure"]["latest_event_age_ms_max"] == 4000
    assert summary["resource_pressure"]["latest_event_age_reporting_bots"] == 2
    assert summary["resource_pressure"]["latest_event_queue_depth_max"] == 2
    assert summary["resource_pressure"]["latest_event_dropped_total_sum"] == 1
    assert summary["resource_pressure"]["event_pipeline_unhealthy_bots"] == 2


def test_live_performance_report_exchange_config_refresh_health(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=1,
                ts=1000,
                component="exchange.config_refresh",
                status="succeeded",
                reason_code="exchange_config_refresh",
                ids={"cycle_id": "cy_cfg_1"},
                data={
                    "context": "maintain_hourly_cycle",
                    "operation": "init_markets",
                    "started_ms": 900,
                    "elapsed_ms": 100,
                },
            ),
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=2,
                ts=2000,
                component="exchange.config_refresh",
                status="failed",
                level="warning",
                reason_code="exchange_config_refresh_failed",
                ids={"cycle_id": "cy_cfg_2"},
                data={
                    "context": "maintain_hourly_cycle",
                    "operation": "init_markets",
                    "started_ms": 1800,
                    "elapsed_ms": 200,
                    "error_type": "ExchangeError",
                    "error": "binanceusdm apiKey=supersecret code=-4084",
                },
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report)

    refresh = report["exchange_config_refresh"]
    assert refresh["total"] == 2
    assert refresh["succeeded"] == 1
    assert refresh["failed"] == 1
    assert refresh["failure_pct"] == 50.0
    assert refresh["bots"] == 1
    assert refresh["failed_bots"] == 1
    assert refresh["latest_failed_bots"] == 1
    assert refresh["recovered_bots"] == 0
    assert refresh["statuses"] == {"succeeded": 1, "failed": 1}
    assert refresh["latest_statuses"] == {"failed": 1}
    assert refresh["event_types"] == {"exchange.config_refresh": 2}
    assert refresh["groups"][0]["status"] == "failed"
    assert refresh["groups"][0]["latest_data"] == {
        "context": "maintain_hourly_cycle",
        "operation": "init_markets",
        "error_type": "ExchangeError",
        "elapsed_ms": 200,
        "started_ms": 1800,
    }
    rendered = json.dumps(refresh, sort_keys=True)
    assert "supersecret" not in rendered
    assert "apiKey" not in rendered
    assert "code=-4084" not in rendered

    groups = {
        group["operation"]: group
        for group in report["operation_durations"]["groups"]
        if group["operation"].startswith("exchange_config_refresh.")
    }
    assert groups["exchange_config_refresh.init_markets"]["trading_impact"] == "exchange_io"
    assert groups["exchange_config_refresh.init_markets"]["max_ms"] == 200
    assert summary["exchange_config_refresh"]["total"] == 2
    assert summary["exchange_config_refresh"]["statuses"] == {
        "succeeded": 1,
        "failed": 1,
    }
    assert summary["operation_durations"]["operation_category_counts"][
        "exchange_config_refresh"
    ] == 1


def test_live_performance_report_marks_recovered_exchange_config_refresh_bot(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=1,
                ts=1000,
                exchange="kucoin",
                user="kucoin_01",
                status="failed",
                reason_code="exchange_config_refresh_failed",
                data={
                    "operation": "init_markets",
                    "elapsed_ms": 33915,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=2,
                ts=2000,
                exchange="kucoin",
                user="kucoin_01",
                status="succeeded",
                reason_code="exchange_config_refresh",
                data={"operation": "init_markets", "elapsed_ms": 6331},
            ),
        ],
    )

    refresh = build_live_performance_report(tmp_path / "monitor")[
        "exchange_config_refresh"
    ]

    assert refresh["failed"] == 1
    assert refresh["failed_bots"] == 1
    assert refresh["latest_statuses"] == {"succeeded": 1}
    assert refresh["latest_failed_bots"] == 0
    assert refresh["recovered_bots"] == 1


def test_live_performance_report_exchange_config_refresh_summary_is_bounded(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=1,
                ts=1000,
                exchange="binance",
                user="binance_01",
                data={"operation": "init_markets", "elapsed_ms": 100},
            ),
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                status="failed",
                reason_code="exchange_config_refresh_failed",
                data={"operation": "set_position_mode", "elapsed_ms": 300},
            ),
        ],
    )

    report = build_live_performance_report(tmp_path / "monitor")
    summary = summarize_live_performance_report(report, group_limit=1)

    assert report["exchange_config_refresh"]["total"] == 2
    assert report["exchange_config_refresh"]["bots"] == 2
    assert len(summary["exchange_config_refresh"]["groups"]) == 1
    assert summary["exchange_config_refresh"]["groups_truncated"] is True
    assert summary["exchange_config_refresh"]["groups"][0]["bot"] == "okx/okx_faisal"
    assert summary["operation_durations"]["total_groups"] >= 2
    assert summary["operation_durations"]["operation_category_counts"][
        "exchange_config_refresh"
    ] == 2


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
                data={"elapsed_ms": 1000, "debug_profile": "state"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                exchange="okx",
                user="okx_faisal",
                data={"elapsed_ms": 2000, "debug_profile": "startup"},
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
            "--debug-profile",
            "startup",
            "--group-limit",
            "1",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["live_events"] == 1
    assert out["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 1,
        "event_segments": 1,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 0,
        "scope_pruned": 0,
    }
    assert out["filters"]["exchanges"] == ["okx"]
    assert out["filters"]["debug_profiles"] == ["startup"]
    assert out["filters"]["events_skipped"] == 1
    assert len(out["performance"]["groups"]) == 1
    assert "files" not in out


def test_live_performance_report_section_projection_keeps_common_metadata(tmp_path):
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
                event_type="fills.refresh_summary",
                seq=2,
                ts=2000,
                component="fills.refresh",
                status="succeeded",
                reason_code="fill_cache_ready",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "elapsed_ms": 40,
                    "history_scope": "all",
                    "event_count_after": 100,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )

    report = summarize_live_performance_report(
        build_live_performance_report(tmp_path / "monitor"),
        group_limit=1,
    )

    projected = project_live_performance_report_sections(report, ["fill_refresh"])

    assert projected["ok"] is True
    assert projected["records_total"] == 2
    assert projected["live_events"] == 2
    assert "fill_refresh" in projected
    assert projected["fill_refresh"]["total_events"] == 1
    assert "performance" not in projected
    assert "operation_durations" not in projected
    assert "files" not in projected


def test_live_performance_report_cli_section_filter(tmp_path, capsys):
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
                event_type="fills.refresh_summary",
                seq=2,
                ts=2000,
                component="fills.refresh",
                status="succeeded",
                reason_code="fill_cache_ready",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "elapsed_ms": 40,
                    "history_scope": "all",
                    "event_count_after": 100,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )

    rc = live_performance_report.main(
        [
            str(tmp_path / "monitor"),
            "--summary",
            "--section",
            "fill_refresh",
            "--compact",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["fill_refresh"]["total_events"] == 1
    assert "performance" not in out
    assert "operation_durations" not in out


def test_live_performance_report_cli_rejects_unknown_section(capsys):
    with pytest.raises(SystemExit):
        live_performance_report.main(["monitor", "--section", "not_a_section"])

    assert "unknown --section value" in capsys.readouterr().err


def test_live_performance_report_event_tail_lines_bounds_monitor_scan(tmp_path):
    events_path = (
        tmp_path / "monitor" / "binance" / "binance_01" / "events" / "current.ndjson"
    )
    _write_ndjson(
        events_path,
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_old"},
                data={"elapsed_ms": 1000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_old"},
                data={"elapsed_ms": 2000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=3000,
                ids={"cycle_id": "cy_old"},
                data={"elapsed_ms": 3000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=4,
                ts=4000,
                ids={"cycle_id": "cy_fresh"},
                data={"elapsed_ms": 4000},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=5,
                ts=5000,
                ids={"cycle_id": "cy_fresh"},
                data={"elapsed_ms": 5000},
            ),
        ],
    )

    report = build_live_performance_report(
        tmp_path / "monitor",
        since_ms=3500,
        event_tail_lines=2,
    )

    assert report["ok"] is True
    assert report["records_total"] == 2
    assert report["live_events"] == 2
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 3500,
        "until_ms": None,
        "events_considered": 2,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "event_tail_lines": 2,
        "event_tail_limited_files": 1,
        "event_tail_skipped_lines": 3,
        "event_tail_skipped_lines_exact": True,
        "event_tail_skipped_bytes": 0,
        "event_tail_line_numbers_exact": True,
        "event_tail_methods": {"seek_tail": 1},
    }
    assert report["bots"] == [{"bot": "binance/binance_01", "events": 2}]
    assert report["performance"]["groups"][0]["count"] == 2


def test_live_performance_report_cli_event_tail_lines(tmp_path, capsys):
    events_path = (
        tmp_path / "monitor" / "binance" / "binance_01" / "events" / "current.ndjson"
    )
    _write_ndjson(
        events_path,
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

    rc = live_performance_report.main(
        [
            str(tmp_path / "monitor"),
            "--summary",
            "--compact",
            "--event-tail-lines",
            "1",
            "--max-event-files",
            "1",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["records_total"] == 1
    assert out["event_window"]["event_tail_lines"] == 1
    assert out["event_window"]["event_tail_limited_files"] == 1
    assert out["event_window"]["event_tail_methods"] == {"seek_tail": 1}
    assert out["event_window"]["max_event_files"] == 1
    assert out["event_window"]["event_file_limit_scope"] == "global"
    assert out["event_window"]["event_files_skipped_by_limit"] == 0


def test_live_performance_report_cli_rejects_negative_event_tail_lines(capsys):
    try:
        live_performance_report.main(["monitor", "--event-tail-lines", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("negative --event-tail-lines must be rejected")

    assert "--event-tail-lines must be >= 0" in capsys.readouterr().err


def test_live_performance_report_max_event_files_prefers_current_then_recent(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    old_rotated = events_dir / "2026-06-25T00-00-00.ndjson"
    new_rotated = events_dir / "2026-06-26T00-00-00.ndjson"
    current = events_dir / "current.ndjson"
    _write_ndjson(
        old_rotated,
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_old"},
                data={"elapsed_ms": 1000},
            )
        ],
    )
    _write_ndjson(
        new_rotated,
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_new"},
                data={"elapsed_ms": 2000},
            )
        ],
    )
    _write_ndjson(
        current,
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=3000,
                ids={"cycle_id": "cy_current"},
                data={"elapsed_ms": 3000},
            )
        ],
    )
    os.utime(old_rotated, (1000, 1000))
    os.utime(new_rotated, (2000, 2000))
    os.utime(current, (1500, 1500))

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files=2,
    )

    assert report["ok"] is True
    assert report["files"] == [
        str(current),
        str(new_rotated),
    ]
    assert report["files_scanned"] == 2
    assert report["records_total"] == 2
    assert report["event_window"] == {
        "enabled": False,
        "since_ms": None,
        "until_ms": None,
        "events_considered": 0,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "max_event_files": 2,
        "event_file_limit_scope": "global",
        "event_files_before_limit": 3,
        "event_files_skipped_by_limit": 1,
        "event_file_limit_order": "current_then_recent_mtime",
    }
    assert report["file_discovery"]["event_segments"] == 3


def test_live_performance_report_cli_rejects_negative_max_event_files(capsys):
    try:
        live_performance_report.main(["monitor", "--max-event-files", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("negative --max-event-files must be rejected")

    assert "--max-event-files must be >= 0" in capsys.readouterr().err


def test_live_performance_report_max_event_files_per_bot_is_fair(tmp_path):
    paths = {
        "binance_old": tmp_path
        / "monitor"
        / "binance"
        / "binance_01"
        / "events"
        / "2026-06-25T00-00-00.ndjson",
        "binance_new": tmp_path
        / "monitor"
        / "binance"
        / "binance_01"
        / "events"
        / "2026-06-26T00-00-00.ndjson",
        "binance_current": tmp_path
        / "monitor"
        / "binance"
        / "binance_01"
        / "events"
        / "current.ndjson",
        "okx_old": tmp_path
        / "monitor"
        / "okx"
        / "okx_01"
        / "events"
        / "2026-06-25T00-00-00.ndjson",
        "okx_new": tmp_path
        / "monitor"
        / "okx"
        / "okx_01"
        / "events"
        / "2026-06-26T00-00-00.ndjson",
        "okx_current": tmp_path
        / "monitor"
        / "okx"
        / "okx_01"
        / "events"
        / "current.ndjson",
    }
    for idx, (name, path) in enumerate(paths.items(), start=1):
        exchange = "okx" if name.startswith("okx") else "binance"
        user = "okx_01" if name.startswith("okx") else "binance_01"
        _write_ndjson(
            path,
            [
                _monitor_row(
                    event_type="cycle.completed",
                    seq=idx,
                    ts=idx * 1000,
                    exchange=exchange,
                    user=user,
                    ids={"cycle_id": f"cy_{name}"},
                    data={"elapsed_ms": idx * 1000},
                )
            ],
        )
    os.utime(paths["binance_old"], (1000, 1000))
    os.utime(paths["binance_new"], (2000, 2000))
    os.utime(paths["binance_current"], (1500, 1500))
    os.utime(paths["okx_old"], (1100, 1100))
    os.utime(paths["okx_new"], (2100, 2100))
    os.utime(paths["okx_current"], (1600, 1600))

    report = build_live_performance_report(
        tmp_path / "monitor",
        include_rotated=True,
        max_event_files_per_bot=2,
    )

    assert report["ok"] is True
    assert report["files"] == [
        str(paths["okx_current"]),
        str(paths["binance_current"]),
        str(paths["okx_new"]),
        str(paths["binance_new"]),
    ]
    assert report["files_scanned"] == 4
    assert report["records_total"] == 4
    assert report["event_window"] == {
        "enabled": False,
        "since_ms": None,
        "until_ms": None,
        "events_considered": 0,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "max_event_files_per_bot": 2,
        "event_file_limit_scope": "per_bot",
        "event_file_limit_groups": 2,
        "event_files_before_limit": 6,
        "event_files_skipped_by_limit": 2,
        "event_file_limit_order": "current_then_recent_mtime",
    }
    assert report["bots"] == [
        {"bot": "binance/binance_01", "events": 2},
        {"bot": "okx/okx_01", "events": 2},
    ]


def test_live_performance_report_rejects_ambiguous_event_file_limits():
    try:
        build_live_performance_report(
            "monitor",
            max_event_files=1,
            max_event_files_per_bot=1,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("ambiguous event file limits must be rejected")


def test_live_performance_report_cli_rejects_negative_max_event_files_per_bot(capsys):
    try:
        live_performance_report.main(["monitor", "--max-event-files-per-bot", "-1"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("negative --max-event-files-per-bot must be rejected")

    assert "--max-event-files-per-bot must be >= 0" in capsys.readouterr().err


def test_live_performance_report_cli_rejects_ambiguous_event_file_limits(capsys):
    try:
        live_performance_report.main(
            ["monitor", "--max-event-files", "1", "--max-event-files-per-bot", "1"]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ambiguous event file limits must be rejected")

    assert "mutually exclusive" in capsys.readouterr().err


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
        "discover_event_files_with_metadata",
        lambda root, *, include_rotated=False: _fake_event_file_discovery([event_path]),
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
        "discover_event_files_with_metadata",
        lambda root, *, include_rotated=False: _fake_event_file_discovery([event_path]),
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
