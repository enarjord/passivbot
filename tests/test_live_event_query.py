from __future__ import annotations

import gzip
import json
import os

import pytest

from live.event_query import build_event_report, discover_event_files
from tools import live_event_query


def _monitor_row(
    *,
    event_type: str,
    cycle_id: str | None,
    seq: int,
    ts: int,
    symbol: str | None = None,
    pside: str | None = None,
    side: str | None = None,
    status: str = "succeeded",
    reason_code: str = "test",
    ids: dict | None = None,
    data: dict | None = None,
    level: str = "debug",
    source: str = "live",
    component: str = "test",
    tags: list[str] | None = None,
    order_id: str | None = None,
    client_order_id: str | None = None,
    exchange: str = "binance",
    user: str = "binance_01",
) -> dict:
    event_ids = dict(ids or {})
    if cycle_id is not None:
        event_ids["cycle_id"] = cycle_id
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": level,
        "source": source,
        "component": component,
        "exchange": exchange,
        "user": user,
        "symbol": symbol,
        "pside": pside,
        "side": side,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "status": status,
        "reason_code": reason_code,
        "data": dict(data or {"seq": seq}),
        "ids": event_ids,
    }
    payload = {"_live_event": live_event}
    payload.update(live_event["data"])
    row = {
        "exchange": exchange,
        "user": user,
        "kind": event_type,
        "tags": list(tags or ["test"]),
        "payload": payload,
        "seq": seq,
        "ts": ts,
    }
    if symbol:
        row["symbol"] = symbol
    if pside:
        row["pside"] = pside
    return row


def _write_ndjson(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_gz_ndjson(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def test_event_query_discovers_rotated_current_and_reconstructs_cycle(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    rotated = events_dir / "2026-06-25T00-00-00.ndjson.gz"
    current = events_dir / "current.ndjson"
    _write_gz_ndjson(
        rotated,
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
            )
        ],
    )
    _write_ndjson(
        current,
        [
            _monitor_row(
                event_type="action.planned",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="BTC/USDT:USDT",
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
            ),
            {
                "kind": "legacy.event",
                "payload": {"ok": True},
                "seq": 4,
                "ts": 1300,
            },
        ],
    )

    assert discover_event_files(
        tmp_path / "monitor" / "binance" / "binance_01",
        include_rotated=True,
    ) == [rotated, current]

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_1",
        include_data=True,
        include_rotated=True,
    )

    assert report["ok"] is True
    assert report["include_rotated"] is True
    assert report["files_scanned"] == 2
    assert report["records_total"] == 4
    assert report["live_events"] == 3
    assert report["legacy_events"] == 1
    assert report["event_types"]["action.planned"] == 1
    assert report["cycle"]["matched_events"] == 2
    assert report["cycle"]["events_truncated"] is False
    assert [event["event_type"] for event in report["cycle"]["events"]] == [
        "cycle.started",
        "action.planned",
    ]
    assert report["cycle"]["events"][1]["symbol"] == "BTC/USDT:USDT"
    assert report["cycle"]["events"][1]["data"] == {"seq": 2}


def test_event_query_reports_invalid_json(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "user01" / "events"
    events_dir.mkdir(parents=True)
    (events_dir / "current.ndjson").write_text("{not json}\n", encoding="utf-8")

    report = build_event_report(tmp_path / "monitor")

    assert report["ok"] is False
    assert report["error_count"] == 1
    assert report["issues"][0]["code"] == "invalid_json"


def test_event_query_current_only_skips_rotated_segments(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "user01" / "events"
    _write_gz_ndjson(
        events_dir / "2026-06-25T00-00-00.ndjson.gz",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_old",
                seq=1,
                ts=1000,
            )
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_current",
                seq=2,
                ts=2000,
            )
        ],
    )

    report = build_event_report(tmp_path / "monitor", include_rotated=False)

    assert report["ok"] is True
    assert report["include_rotated"] is False
    assert report["files"] == [str(events_dir / "current.ndjson")]
    assert report["files_scanned"] == 1
    assert "query" not in report
    assert report["cycle_ids_sample"] == [{"cycle_id": "cy_current", "events": 1}]


def test_event_query_filters_by_event_type_without_changing_summary_counts(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                symbol="BTC/USDT:USDT",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="ETH/USDT:USDT",
            ),
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
                symbol="SOL/USDT:USDT",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        event_type="candle.tail_projected",
        include_data=True,
    )

    assert report["ok"] is True
    assert report["event_types"] == {
        "candle.tail_projected": 2,
        "remote_call.failed": 1,
    }
    assert report["query"]["filters"] == {
        "event_types": ["candle.tail_projected"],
    }
    assert report["query"]["matched_events"] == 2
    assert report["query"]["events_truncated"] is False
    assert [event["symbol"] for event in report["query"]["events"]] == [
        "BTC/USDT:USDT",
        "SOL/USDT:USDT",
    ]
    assert report["query"]["events"][0]["data"] == {"seq": 1}


def test_event_query_filters_by_level(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                level="info",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                level="warning",
            ),
            _monitor_row(
                event_type="execution.create_failed",
                cycle_id="cy_1",
                seq=3,
                ts=1200,
                level="error",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        level=["warning,error"],
        timeline=True,
        trace_summary=True,
    )

    assert report["event_types"] == {
        "cycle.started": 1,
        "execution.create_failed": 1,
        "remote_call.failed": 1,
    }
    assert report["query"]["filters"] == {"levels": ["error", "warning"]}
    assert report["query"]["matched_events"] == 2
    assert [event["level"] for event in report["query"]["events"]] == [
        "warning",
        "error",
    ]
    assert report["query"]["trace_summary"]["levels"] == {"error": 1, "warning": 1}
    assert report["query"]["timeline"] == [
        (
            "1100 seq=2 remote_call.failed status=succeeded "
            "reason_code=test ids=cycle_id=cy_1"
        ),
        (
            "1200 seq=3 execution.create_failed status=succeeded "
            "reason_code=test ids=cycle_id=cy_1"
        ),
    ]


def test_event_query_combines_cycle_and_event_type_filters(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_7",
                seq=2,
                ts=1100,
            ),
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_8",
                seq=3,
                ts=1200,
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_7",
        event_type=["candle.tail_projected,remote_call.failed"],
    )

    assert report["query"]["filters"] == {
        "cycle_id": "cy_7",
        "event_types": ["candle.tail_projected", "remote_call.failed"],
    }
    assert report["query"]["matched_events"] == 2
    assert report["cycle"]["cycle_id"] == "cy_7"
    assert report["cycle"]["filters"] == {
        "cycle_id": "cy_7",
        "event_types": ["candle.tail_projected", "remote_call.failed"],
    }
    assert report["cycle"]["event_types"] == [
        "candle.tail_projected",
        "remote_call.failed",
    ]
    assert report["cycle"]["matched_events"] == 2
    assert [event["event_type"] for event in report["cycle"]["events"]] == [
        "candle.tail_projected",
        "remote_call.failed",
    ]


def test_event_query_filters_by_ids_symbol_pside_reason_and_status(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                symbol="BTC/USDT:USDT",
                pside="long",
                status="started",
                reason_code="submitted_to_exchange",
                ids={"order_wave_id": "ow_1"},
            ),
            _monitor_row(
                event_type="execution.create_failed",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="BTC/USDT:USDT",
                pside="long",
                status="failed",
                reason_code="exchange_exception",
                ids={"order_wave_id": "ow_1"},
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
                symbol="ETH/USDT:USDT",
                pside="short",
                status="failed",
                reason_code="request_timeout",
                ids={"remote_call_id": "rc_9"},
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        order_wave_id="ow_1",
        symbol="BTC/USDT:USDT",
        pside="long",
        reason_code="exchange_exception",
        status="failed",
    )

    assert report["ok"] is True
    assert report["query"]["filters"] == {
        "order_wave_ids": ["ow_1"],
        "psides": ["long"],
        "reason_codes": ["exchange_exception"],
        "statuses": ["failed"],
        "symbols": ["BTC/USDT:USDT"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["event_type"] == "execution.create_failed"
    assert report["query"]["events"][0]["ids"]["order_wave_id"] == "ow_1"

    remote_report = build_event_report(
        tmp_path / "monitor",
        remote_call_id="rc_9",
        status="failed",
    )

    assert remote_report["query"]["filters"] == {
        "remote_call_ids": ["rc_9"],
        "statuses": ["failed"],
    }
    assert remote_report["query"]["matched_events"] == 1
    assert remote_report["query"]["events"][0]["event_type"] == "remote_call.failed"


def test_event_query_filters_by_tags(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                tags=["execution", "order"],
            ),
            _monitor_row(
                event_type="risk.hsl_status",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                tags=["risk", "summary"],
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
                tags=["remote_call", "exchange"],
            ),
        ],
    )

    report = build_event_report(tmp_path / "monitor", tag="execution,order")

    assert report["ok"] is True
    assert report["query"]["filters"] == {"tags": ["execution", "order"]}
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["event_type"] == "execution.create_sent"

    risk_report = build_event_report(tmp_path / "monitor", cycle_id="cy_1", tag="risk")

    assert risk_report["cycle"]["filters"] == {
        "cycle_id": "cy_1",
        "tags": ["risk"],
    }
    assert risk_report["cycle"]["matched_events"] == 1
    assert risk_report["cycle"]["events"][0]["event_type"] == "risk.hsl_status"


def test_event_query_filters_by_source_and_component(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                source="executor",
                component="order_wave",
            ),
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                source="candles",
                component="tail_projection",
            ),
            _monitor_row(
                event_type="execution.cancel_sent",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
                source="executor",
                component="order_wave",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_1",
        source="executor",
        component="order_wave",
    )

    assert report["ok"] is True
    assert report["cycle"]["filters"] == {
        "components": ["order_wave"],
        "cycle_id": "cy_1",
        "sources": ["executor"],
    }
    assert report["cycle"]["matched_events"] == 1
    assert report["cycle"]["events"][0]["event_type"] == "execution.create_sent"
    assert report["cycle"]["events"][0]["source"] == "executor"
    assert report["cycle"]["events"][0]["component"] == "order_wave"


def test_event_query_filters_by_top_level_data_fields(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.red_finalized_without_order",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                pside="long",
                tags=["risk", "hsl"],
                data={
                    "stop_event_anchor_source": "current_time_fallback",
                    "stop_event_anchor_fallback_used": True,
                    "flat_confirmations": 2,
                },
            ),
            _monitor_row(
                event_type="hsl.red_finalized_without_order",
                cycle_id="cy_2",
                seq=2,
                ts=1100,
                symbol="XLM/USDT:USDT",
                pside="long",
                tags=["risk", "hsl"],
                data={
                    "stop_event_anchor_source": "panic_fill",
                    "stop_event_anchor_fallback_used": False,
                    "flat_confirmations": 2,
                },
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        event_type="hsl.red_finalized_without_order",
        tag="hsl",
        data_eq=[
            "stop_event_anchor_source=current_time_fallback",
            "stop_event_anchor_fallback_used=true",
        ],
        include_data=True,
    )

    assert report["ok"] is True
    assert report["query"]["filters"] == {
        "data_eq": {
            "stop_event_anchor_fallback_used": ["true"],
            "stop_event_anchor_source": ["current_time_fallback"],
        },
        "event_types": ["hsl.red_finalized_without_order"],
        "tags": ["hsl"],
    }
    assert report["query"]["matched_events"] == 1
    event = report["query"]["events"][0]
    assert event["symbol"] == "ZEC/USDT:USDT"
    assert event["data"]["stop_event_anchor_source"] == "current_time_fallback"


def test_event_query_rejects_malformed_data_eq_filter(tmp_path):
    with pytest.raises(ValueError, match="key=value"):
        build_event_report(tmp_path, data_eq="missing_equals")


def test_event_query_filters_by_remaining_event_ids(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="rust_orchestrator.called",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                ids={
                    "bot_id": "bot_1",
                    "snapshot_id": "snap_1",
                    "plan_id": "plan_1",
                },
            ),
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="BTC/USDT:USDT",
                ids={
                    "bot_id": "bot_1",
                    "plan_id": "plan_1",
                    "action_id": "ow_1:create:0",
                    "order_wave_id": "ow_1",
                },
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_1",
                seq=3,
                ts=1200,
                status="failed",
                ids={
                    "bot_id": "bot_1",
                    "remote_call_id": "rc_1",
                    "remote_call_group_id": "cy_1:candles",
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                cycle_id="cy_2",
                seq=4,
                ts=1300,
                ids={
                    "bot_id": "bot_2",
                    "snapshot_id": "snap_2",
                    "remote_call_id": "rc_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
            ),
        ],
    )

    bot_report = build_event_report(tmp_path / "monitor", bot_id="bot_1", timeline=True)

    assert bot_report["query"]["filters"] == {"bot_ids": ["bot_1"]}
    assert bot_report["query"]["matched_events"] == 3
    assert [event["event_type"] for event in bot_report["query"]["events"]] == [
        "rust_orchestrator.called",
        "execution.create_sent",
        "remote_call.failed",
    ]
    assert bot_report["query"]["timeline"][0] == (
        "1000 seq=1 rust_orchestrator.called status=succeeded "
        "reason_code=test ids=bot_id=bot_1,cycle_id=cy_1,"
        "snapshot_id=snap_1,plan_id=plan_1"
    )

    snapshot_report = build_event_report(
        tmp_path / "monitor",
        snapshot_id="snap_1",
    )

    assert snapshot_report["query"]["filters"] == {"snapshot_ids": ["snap_1"]}
    assert snapshot_report["query"]["matched_events"] == 1
    assert snapshot_report["query"]["events"][0]["ids"]["snapshot_id"] == "snap_1"

    plan_report = build_event_report(tmp_path / "monitor", plan_id="plan_1")

    assert plan_report["query"]["filters"] == {"plan_ids": ["plan_1"]}
    assert plan_report["query"]["matched_events"] == 2
    assert [event["event_type"] for event in plan_report["query"]["events"]] == [
        "rust_orchestrator.called",
        "execution.create_sent",
    ]

    action_report = build_event_report(
        tmp_path / "monitor",
        action_id="ow_1:create:0",
    )

    assert action_report["query"]["filters"] == {"action_ids": ["ow_1:create:0"]}
    assert action_report["query"]["matched_events"] == 1
    assert action_report["query"]["events"][0]["ids"]["action_id"] == "ow_1:create:0"

    remote_group_report = build_event_report(
        tmp_path / "monitor",
        remote_call_group_id="cy_1:candles",
        status="failed",
    )

    assert remote_group_report["query"]["filters"] == {
        "remote_call_group_ids": ["cy_1:candles"],
        "statuses": ["failed"],
    }
    assert remote_group_report["query"]["matched_events"] == 1
    assert remote_group_report["query"]["events"][0]["ids"] == {
        "bot_id": "bot_1",
        "cycle_id": "cy_1",
        "remote_call_id": "rc_1",
        "remote_call_group_id": "cy_1:candles",
    }


def test_event_query_bot_filter_falls_back_to_monitor_path(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                pside="long",
            ),
            _monitor_row(
                event_type="hsl.cooldown_started",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="ZEC/USDT:USDT",
                pside="long",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        bot_id="gateio/gateio_01",
        event_type=["hsl.status", "hsl.cooldown_started"],
        symbol="ZEC/USDT:USDT",
    )

    assert report["ok"] is True
    assert report["query"]["filters"] == {
        "bot_ids": ["gateio/gateio_01"],
        "event_types": ["hsl.cooldown_started", "hsl.status"],
        "symbols": ["ZEC/USDT:USDT"],
    }
    assert report["query"]["matched_events"] == 2
    assert [event["event_type"] for event in report["query"]["events"]] == [
        "hsl.status",
        "hsl.cooldown_started",
    ]

    wrong_bot_report = build_event_report(
        tmp_path / "monitor",
        bot_id="binance/binance_01",
        event_type=["hsl.status", "hsl.cooldown_started"],
    )

    assert wrong_bot_report["query"]["matched_events"] == 0


def test_discover_event_files_filters_monitor_paths_by_exchange_and_user(tmp_path):
    gateio_events = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    binance_events = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    kucoin_events = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(gateio_events / "current.ndjson", [])
    _write_gz_ndjson(gateio_events / "20260629.ndjson.gz", [])
    _write_ndjson(binance_events / "current.ndjson", [])
    _write_ndjson(kucoin_events / "current.ndjson", [])

    gateio_files = discover_event_files(
        tmp_path / "monitor",
        include_rotated=True,
        exchange="gateio",
        user="gateio_01",
    )

    assert [path.name for path in gateio_files] == [
        "20260629.ndjson.gz",
        "current.ndjson",
    ]
    assert {path.parent.parent.parent.name for path in gateio_files} == {"gateio"}
    assert {path.parent.parent.name for path in gateio_files} == {"gateio_01"}

    missing_files = discover_event_files(
        tmp_path / "monitor",
        exchange="gateio",
        user="binance_01",
    )

    assert missing_files == []


def test_event_query_filters_exchange_user_and_prunes_monitor_paths(tmp_path):
    gateio_events = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    binance_events = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        gateio_events / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.red_triggered",
                cycle_id="cy_gateio",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                pside="long",
                exchange="gateio",
                user="gateio_01",
            ),
        ],
    )
    _write_ndjson(
        binance_events / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.red_triggered",
                cycle_id="cy_binance",
                seq=2,
                ts=1100,
                symbol="ZEC/USDT:USDT",
                pside="long",
                exchange="binance",
                user="binance_01",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        exchange="gateio",
        user="gateio_01",
        event_type="hsl.red_triggered",
        symbol="ZEC/USDT:USDT",
    )

    assert report["ok"] is True
    assert report["files_scanned"] == 1
    assert report["query"]["filters"] == {
        "event_types": ["hsl.red_triggered"],
        "exchanges": ["gateio"],
        "symbols": ["ZEC/USDT:USDT"],
        "users": ["gateio_01"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["exchange"] == "gateio"
    assert report["query"]["events"][0]["user"] == "gateio_01"
    assert report["query"]["events"][0]["ids"]["cycle_id"] == "cy_gateio"


def test_event_query_exchange_user_filter_keeps_direct_events_dir(tmp_path):
    events_dir = tmp_path / "ad_hoc" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.red_triggered",
                cycle_id="cy_gateio",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                exchange="gateio",
                user="gateio_01",
            )
        ],
    )

    report = build_event_report(
        events_dir,
        exchange="gateio",
        user="gateio_01",
        event_type="hsl.red_triggered",
    )

    assert report["ok"] is True
    assert report["files_scanned"] == 1
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["exchange"] == "gateio"
    assert report["query"]["events"][0]["user"] == "gateio_01"


def test_event_query_path_scope_labels_legacy_rows_in_output(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    row = _monitor_row(
        event_type="hsl.status",
        cycle_id="cy_gateio",
        seq=1,
        ts=1000,
        symbol="ZEC/USDT:USDT",
        exchange="gateio",
        user="gateio_01",
    )
    live_event = row["payload"]["_live_event"]
    live_event.pop("exchange")
    live_event.pop("user")
    row.pop("exchange")
    row.pop("user")
    _write_ndjson(events_dir / "current.ndjson", [row])

    report = build_event_report(
        tmp_path / "monitor",
        exchange="gateio",
        user="gateio_01",
        event_type="hsl.status",
        trace_summary=True,
        cycle_trace=True,
    )

    assert report["ok"] is True
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["exchange"] == "gateio"
    assert report["query"]["events"][0]["user"] == "gateio_01"
    assert report["query"]["trace_summary"]["exchanges"] == {"gateio": 1}
    assert report["query"]["trace_summary"]["users"] == {"gateio_01": 1}
    cycle = report["query"]["cycle_trace"]["cycles"][0]
    assert cycle["timeline"][0]["exchange"] == "gateio"
    assert cycle["timeline"][0]["user"] == "gateio_01"
    assert cycle["trace_summary"]["exchanges"] == {"gateio": 1}
    assert cycle["trace_summary"]["users"] == {"gateio_01": 1}


def test_event_query_filters_legacy_snapshot_id_from_event_data(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="snapshot.built",
                cycle_id=None,
                seq=1,
                ts=1000,
                data={
                    "snapshot_id": "snap_legacy",
                    "cycle_id": 42,
                    "ready_symbols": 3,
                },
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                ids={"snapshot_id": "snap_2", "plan_id": "plan_2"},
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        snapshot_id="snap_legacy",
        include_data=True,
        trace_summary=True,
    )

    assert report["query"]["filters"] == {"snapshot_ids": ["snap_legacy"]}
    assert report["missing_cycle_id"] == 1
    assert report["query"]["matched_events"] == 1
    event = report["query"]["events"][0]
    assert event["event_type"] == "snapshot.built"
    assert event["ids"] == {"snapshot_id": "snap_legacy"}
    assert event["data"]["cycle_id"] == 42
    assert report["query"]["trace_summary"]["ids"]["snapshot_id"] == [
        {"id": "snap_legacy", "events": 1}
    ]


def test_event_query_filters_by_inclusive_time_window(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    invalid_ts = _monitor_row(
        event_type="cycle.completed",
        cycle_id="cy_bad",
        seq=6,
        ts=2600,
    )
    invalid_ts["ts"] = "bad"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_1",
                seq=1,
                ts=900,
            ),
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_2",
                seq=2,
                ts=1000,
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_2",
                seq=3,
                ts=1500,
                reason_code="request_timeout",
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_3",
                seq=4,
                ts=2000,
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_4",
                seq=5,
                ts=2500,
            ),
            invalid_ts,
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        since_ms=1000,
        until_ms=2000,
        timeline=True,
    )

    assert report["ok"] is True
    assert report["records_total"] == 6
    assert report["live_events"] == 6
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 1000,
        "until_ms": 2000,
        "events_considered": 3,
        "events_skipped_before": 1,
        "events_skipped_after": 1,
        "invalid_window_ts": 1,
        "files_skipped_before_window": 0,
    }
    assert report["event_types"] == {
        "cycle.completed": 1,
        "cycle.started": 1,
        "remote_call.failed": 1,
    }
    assert report["query"]["filters"] == {
        "since_ms": 1000,
        "until_ms": 2000,
    }
    assert report["query"]["matched_events"] == 3
    assert report["query"]["timeline"] == [
        "1000 seq=2 cycle.started status=succeeded reason_code=test ids=cycle_id=cy_2",
        (
            "1500 seq=3 remote_call.failed status=succeeded "
            "reason_code=request_timeout ids=cycle_id=cy_2"
        ),
        "2000 seq=4 cycle.completed status=succeeded reason_code=test ids=cycle_id=cy_3",
    ]


def test_event_query_prunes_rotated_files_before_time_window_by_mtime(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    old_segment = events_dir / "2026-06-29T18-00-00.ndjson.gz"
    recent_segment = events_dir / "2026-06-29T22-00-00.ndjson.gz"
    current_segment = events_dir / "current.ndjson"
    _write_gz_ndjson(
        old_segment,
        [
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_old",
                seq=1,
                ts=1_000,
                exchange="gateio",
                user="gateio_01",
            )
        ],
    )
    _write_gz_ndjson(
        recent_segment,
        [
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_recent",
                seq=2,
                ts=5_000,
                exchange="gateio",
                user="gateio_01",
            )
        ],
    )
    _write_ndjson(
        current_segment,
        [
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_current",
                seq=3,
                ts=6_000,
                exchange="gateio",
                user="gateio_01",
            )
        ],
    )
    os.utime(old_segment, (1.0, 1.0))
    os.utime(recent_segment, (5.0, 5.0))
    os.utime(current_segment, (6.0, 6.0))

    report = build_event_report(
        tmp_path / "monitor",
        exchange="gateio",
        user="gateio_01",
        event_type="hsl.status",
        since_ms=4_000,
        include_rotated=True,
        timeline=True,
    )

    assert report["ok"] is True
    assert report["files_scanned"] == 2
    assert report["event_window"]["files_skipped_before_window"] == 1
    assert report["event_window"]["events_considered"] == 2
    assert report["query"]["matched_events"] == 2
    assert [event["ids"]["cycle_id"] for event in report["query"]["events"]] == [
        "cy_recent",
        "cy_current",
    ]


def test_event_query_event_tail_lines_bounds_window_scan(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                exchange="gateio",
                user="gateio_01",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_2",
                seq=2,
                ts=2000,
                exchange="gateio",
                user="gateio_01",
            ),
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_3",
                seq=3,
                ts=3000,
                exchange="gateio",
                user="gateio_01",
            ),
            _monitor_row(
                event_type="hsl.status",
                cycle_id="cy_4",
                seq=4,
                ts=4000,
                exchange="gateio",
                user="gateio_01",
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_5",
                seq=5,
                ts=5000,
                exchange="gateio",
                user="gateio_01",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        exchange="gateio",
        user="gateio_01",
        since_ms=2500,
        event_tail_lines=2,
        timeline=True,
    )

    assert report["ok"] is True
    assert report["records_total"] == 2
    assert report["live_events"] == 2
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 2500,
        "until_ms": None,
        "events_considered": 2,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "files_skipped_before_window": 0,
        "event_tail_lines": 2,
        "event_tail_limited_files": 1,
        "event_tail_skipped_lines": 3,
    }
    assert report["query"]["matched_events"] == 2
    assert [event["ids"]["cycle_id"] for event in report["query"]["events"]] == [
        "cy_4",
        "cy_5",
    ]


def test_event_query_timeline_renders_cycle_and_query_matches(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                status="started",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_1",
                seq=2,
                ts=1100,
                symbol="ZEC/USDT:USDT",
                status="failed",
                reason_code="request_timeout",
                ids={"remote_call_id": "rc_1"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_2",
                seq=3,
                ts=1200,
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_1",
        status="failed",
        timeline=True,
    )

    assert report["query"]["matched_events"] == 1
    assert report["cycle"]["filters"] == {
        "cycle_id": "cy_1",
        "statuses": ["failed"],
    }
    assert report["query"]["timeline"] == [
        (
            "1100 seq=2 remote_call.failed status=failed "
            "reason_code=request_timeout symbol=ZEC/USDT:USDT "
            "ids=cycle_id=cy_1,remote_call_id=rc_1"
        )
    ]
    assert report["cycle"]["matched_events"] == 1
    assert report["cycle"]["timeline"] == report["query"]["timeline"]

    all_report = build_event_report(tmp_path / "monitor", limit=2, timeline=True)

    assert all_report["query"]["filters"] == {}
    assert all_report["query"]["matched_events"] == 3
    assert all_report["query"]["events_truncated"] is True
    assert all_report["query"]["timeline"] == [
        "1000 seq=1 cycle.started status=started reason_code=test ids=cycle_id=cy_1",
        (
            "1100 seq=2 remote_call.failed status=failed "
            "reason_code=request_timeout symbol=ZEC/USDT:USDT "
            "ids=cycle_id=cy_1,remote_call_id=rc_1"
        ),
    ]


def test_live_event_query_cli_outputs_json_and_status(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="order_wave.completed",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
            )
        ],
    )

    assert (
        live_event_query.main([str(tmp_path / "monitor"), "--cycle-id", "cy_7"])
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["cycle"]["cycle_id"] == "cy_7"
    assert report["cycle"]["matched_events"] == 1


def test_live_event_query_cli_accepts_event_type_and_kind_alias(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="candle.tail_projected",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_8",
                seq=2,
                ts=1100,
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--event-type",
                "candle.tail_projected",
                "--kind",
                "remote_call.failed",
                "--limit",
                "1",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"]["event_types"] == [
        "candle.tail_projected",
        "remote_call.failed",
    ]
    assert report["query"]["matched_events"] == 2
    assert report["query"]["events_truncated"] is True
    assert len(report["query"]["events"]) == 1


def test_live_event_query_cli_accepts_scope_filters_and_timeline(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.cancel_failed",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                symbol="SOL/USDT:USDT",
                pside="short",
                status="failed",
                reason_code="exchange_exception",
                source="executor",
                component="order_wave",
                ids={"order_wave_id": "ow_9"},
            ),
            _monitor_row(
                event_type="execution.cancel_succeeded",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                symbol="SOL/USDT:USDT",
                pside="short",
                status="succeeded",
                reason_code="exchange_acknowledged",
                source="executor",
                component="order_wave",
                ids={"order_wave_id": "ow_9"},
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--order-wave-id",
                "ow_9",
                "--symbol",
                "SOL/USDT:USDT",
                "--pside",
                "short",
                "--status",
                "failed",
                "--reason-code",
                "exchange_exception",
                "--source",
                "executor",
                "--component",
                "order_wave",
                "--timeline",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {
        "order_wave_ids": ["ow_9"],
        "components": ["order_wave"],
        "psides": ["short"],
        "reason_codes": ["exchange_exception"],
        "sources": ["executor"],
        "statuses": ["failed"],
        "symbols": ["SOL/USDT:USDT"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["timeline"] == [
        (
            "1000 seq=1 execution.cancel_failed status=failed "
            "reason_code=exchange_exception symbol=SOL/USDT:USDT pside=short "
            "ids=cycle_id=cy_9,order_wave_id=ow_9"
        )
    ]


def test_live_event_query_cli_accepts_exchange_and_user_filters(tmp_path, capsys):
    gateio_events = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    okx_events = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        gateio_events / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.cooldown_started",
                cycle_id="cy_gateio",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                exchange="gateio",
                user="gateio_01",
            )
        ],
    )
    _write_ndjson(
        okx_events / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.cooldown_started",
                cycle_id="cy_okx",
                seq=2,
                ts=1100,
                symbol="ZEC/USDT:USDT",
                exchange="okx",
                user="okx_faisal",
            )
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--exchange",
                "gateio",
                "--user",
                "gateio_01",
                "--event-type",
                "hsl.cooldown_started",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["files_scanned"] == 1
    assert report["query"]["filters"] == {
        "event_types": ["hsl.cooldown_started"],
        "exchanges": ["gateio"],
        "users": ["gateio_01"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["exchange"] == "gateio"
    assert report["query"]["events"][0]["ids"]["cycle_id"] == "cy_gateio"


def test_live_event_query_cli_accepts_time_window(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.cancel_failed",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                reason_code="exchange_exception",
            ),
            _monitor_row(
                event_type="execution.cancel_succeeded",
                cycle_id="cy_9",
                seq=2,
                ts=1200,
                reason_code="exchange_acknowledged",
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--since-ms",
                "1100",
                "--until-ms",
                "1300",
                "--timeline",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["event_window"]["events_considered"] == 1
    assert report["event_window"]["events_skipped_before"] == 1
    assert report["query"]["filters"] == {"since_ms": 1100, "until_ms": 1300}
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["seq"] == 2


def test_live_event_query_cli_accepts_level_filter(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.succeeded",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                level="info",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                level="error",
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--level",
                "error",
                "--include-data",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {"levels": ["error"]}
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["event_type"] == "remote_call.failed"
    assert report["query"]["events"][0]["level"] == "error"


def test_live_event_query_cli_accepts_data_eq_filter(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.red_finalized_without_order",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
                data={"stop_event_anchor_source": "current_time_fallback"},
            ),
            _monitor_row(
                event_type="hsl.red_finalized_without_order",
                cycle_id="cy_2",
                seq=2,
                ts=1100,
                data={"stop_event_anchor_source": "panic_fill"},
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--event-type",
                "hsl.red_finalized_without_order",
                "--data-eq",
                "stop_event_anchor_source=current_time_fallback",
                "--include-data",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {
        "data_eq": {"stop_event_anchor_source": ["current_time_fallback"]},
        "event_types": ["hsl.red_finalized_without_order"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["data"]["stop_event_anchor_source"] == (
        "current_time_fallback"
    )


def test_live_event_query_cli_rejects_malformed_data_eq_filter(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_event_query.main([str(tmp_path), "--data-eq", "missing_equals"])

    assert exc_info.value.code == 2
    assert "key=value" in capsys.readouterr().err


def test_live_event_query_cli_accepts_recent_minutes(tmp_path, capsys, monkeypatch):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    now_ms = 1_700_000_000_000
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.cancel_failed",
                cycle_id="cy_9",
                seq=1,
                ts=now_ms - 60_001,
            ),
            _monitor_row(
                event_type="execution.cancel_succeeded",
                cycle_id="cy_9",
                seq=2,
                ts=now_ms - 60_000,
            ),
        ],
    )
    monkeypatch.setattr(live_event_query.time, "time", lambda: now_ms / 1000)

    assert (
        live_event_query.main(
            [str(tmp_path / "monitor"), "--recent-minutes", "1", "--timeline"]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {"since_ms": now_ms - 60_000}
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["seq"] == 2


def test_live_event_query_cli_projects_event_tail_metadata(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_1",
                seq=1,
                ts=1000,
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_2",
                seq=2,
                ts=2000,
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--since-ms",
                "1",
                "--event-tail-lines",
                "1",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["records_total"] == 1
    assert report["event_window"]["event_tail_lines"] == 1
    assert report["event_window"]["event_tail_limited_files"] == 1
    assert report["event_window"]["event_tail_skipped_lines"] == 1
    assert report["query"]["events"][0]["ids"]["cycle_id"] == "cy_2"


def test_live_event_query_cli_rejects_invalid_time_window(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_event_query.main(
            [str(tmp_path), "--since-ms", "20", "--until-ms", "10"]
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "must be <= --until-ms" in err


def test_live_event_query_cli_rejects_negative_event_tail_lines(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_event_query.main([str(tmp_path), "--event-tail-lines", "-1"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--event-tail-lines must be non-negative" in err


def test_live_event_query_cli_accepts_additional_id_scopes(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                ids={
                    "bot_id": "bot_9",
                    "snapshot_id": "snap_9",
                    "plan_id": "plan_9",
                    "action_id": "ow_9:create:0",
                    "remote_call_group_id": "cy_9:orders",
                },
            ),
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                ids={
                    "bot_id": "bot_9",
                    "snapshot_id": "snap_10",
                    "plan_id": "plan_9",
                    "action_id": "ow_9:create:1",
                    "remote_call_group_id": "cy_9:orders",
                },
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--bot-id",
                "bot_9",
                "--snapshot-id",
                "snap_9",
                "--plan-id",
                "plan_9",
                "--action-id",
                "ow_9:create:0",
                "--remote-call-group-id",
                "cy_9:orders",
                "--timeline",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {
        "action_ids": ["ow_9:create:0"],
        "bot_ids": ["bot_9"],
        "plan_ids": ["plan_9"],
        "remote_call_group_ids": ["cy_9:orders"],
        "snapshot_ids": ["snap_9"],
    }
    assert report["query"]["matched_events"] == 1
    assert report["query"]["events"][0]["ids"]["action_id"] == "ow_9:create:0"
    assert report["query"]["timeline"] == [
        (
            "1000 seq=1 execution.create_sent status=succeeded "
            "reason_code=test ids=bot_id=bot_9,cycle_id=cy_9,"
            "snapshot_id=snap_9,plan_id=plan_9,action_id=ow_9:create:0,"
            "remote_call_group_id=cy_9:orders"
        )
    ]


def test_event_query_trace_summary_counts_all_matches_beyond_limit(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.cancel_sent",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                symbol="SOL/USDT:USDT",
                pside="short",
                side="sell",
                status="started",
                reason_code="test_cancel",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:cancel:0"},
            ),
            _monitor_row(
                event_type="execution.cancel_succeeded",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                symbol="SOL/USDT:USDT",
                pside="short",
                side="sell",
                status="succeeded",
                reason_code="exchange_acknowledged",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:cancel:0"},
            ),
            _monitor_row(
                event_type="execution.create_failed",
                cycle_id="cy_9",
                seq=3,
                ts=1200,
                symbol="ETH/USDT:USDT",
                pside="long",
                side="buy",
                status="failed",
                reason_code="exchange_exception",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:create:0"},
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                cycle_id="cy_9",
                seq=4,
                ts=1300,
                status="succeeded",
                reason_code="authoritative_refresh",
                ids={"remote_call_group_id": "cy_9:auth"},
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_9",
        limit=1,
        trace_summary=True,
    )

    assert report["cycle"]["matched_events"] == 4
    assert report["cycle"]["events_truncated"] is True
    assert len(report["cycle"]["events"]) == 1
    summary = report["cycle"]["trace_summary"]
    assert summary["matched_events"] == 4
    assert summary["first_ts"] == 1000
    assert summary["last_ts"] == 1300
    assert summary["event_types"] == {
        "execution.cancel_sent": 1,
        "execution.cancel_succeeded": 1,
        "execution.create_failed": 1,
        "remote_call.succeeded": 1,
    }
    assert summary["statuses"] == {
        "failed": 1,
        "started": 1,
        "succeeded": 2,
    }
    assert summary["reason_codes"] == {
        "authoritative_refresh": 1,
        "exchange_acknowledged": 1,
        "exchange_exception": 1,
        "test_cancel": 1,
    }
    assert summary["symbols"] == ["ETH/USDT:USDT", "SOL/USDT:USDT"]
    assert summary["psides"] == ["long", "short"]
    assert summary["sides"] == ["buy", "sell"]
    assert summary["ids"]["cycle_id"] == [{"id": "cy_9", "events": 4}]
    assert summary["ids"]["order_wave_id"] == [{"id": "ow_9", "events": 3}]
    assert summary["ids"]["remote_call_group_id"] == [
        {"id": "cy_9:auth", "events": 1}
    ]
    assert summary["order_waves"]["ow_9"] == {
        "events": 3,
        "event_types": {
            "execution.cancel_sent": 1,
            "execution.cancel_succeeded": 1,
            "execution.create_failed": 1,
        },
        "statuses": {"failed": 1, "started": 1, "succeeded": 1},
        "reason_codes": {
            "exchange_acknowledged": 1,
            "exchange_exception": 1,
            "test_cancel": 1,
        },
        "symbols": ["ETH/USDT:USDT", "SOL/USDT:USDT"],
        "psides": ["long", "short"],
        "action_ids": ["ow_9:cancel:0", "ow_9:create:0"],
    }


def test_event_query_order_trace_reconstructs_wave_actions_beyond_limit(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="order_wave.started",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                status="started",
                ids={"order_wave_id": "ow_9"},
                data={"planned_create": 1, "planned_cancel": 1},
            ),
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                symbol="ETH/USDT:USDT",
                pside="long",
                side="buy",
                status="started",
                reason_code="submitted_to_exchange",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:create:0"},
                data={
                    "index": 0,
                    "qty": 0.12,
                    "price": 2500.0,
                    "pb_order_type": "entry_grid_normal_long",
                    "client_order_id_short": "cid-entry-0",
                },
                client_order_id="cid-entry-0-full-value",
            ),
            _monitor_row(
                event_type="execution.create_succeeded",
                cycle_id="cy_9",
                seq=3,
                ts=1200,
                symbol="ETH/USDT:USDT",
                pside="long",
                side="buy",
                status="succeeded",
                reason_code="exchange_acknowledged",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:create:0"},
                data={
                    "index": 0,
                    "result_order_id_short": "abc123",
                    "result_client_order_id_short": "cid-entry-0",
                },
                order_id="abc123-full-value",
            ),
            _monitor_row(
                event_type="execution.cancel_ambiguous_terminal",
                cycle_id="cy_9",
                seq=4,
                ts=1300,
                symbol="SOL/USDT:USDT",
                pside="short",
                side="sell",
                status="degraded",
                reason_code="requires_full_authoritative_confirmation",
                ids={"order_wave_id": "ow_9", "action_id": "ow_9:cancel:0"},
                data={"index": 0, "order_id_short": "cancel123"},
            ),
            _monitor_row(
                event_type="execution.confirmation_requested",
                cycle_id="cy_9",
                seq=5,
                ts=1400,
                status="started",
                reason_code="authoritative_confirmation",
                ids={"order_wave_id": "ow_9"},
                data={"target_epoch": 9, "current_epoch": 8},
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        order_wave_id="ow_9",
        order_trace=True,
        limit=1,
    )

    assert report["query"]["matched_events"] == 5
    trace = report["query"]["order_trace"]
    assert trace["matched_order_events"] == 5
    assert trace["order_wave_count"] == 1
    assert trace["unscoped_event_count"] == 0
    wave = trace["order_waves"][0]
    assert wave["order_wave_id"] == "ow_9"
    assert wave["event_count"] == 5
    assert wave["events_truncated"] is True
    assert len(wave["timeline"]) == 1
    assert wave["event_types"] == {
        "execution.cancel_ambiguous_terminal": 1,
        "execution.confirmation_requested": 1,
        "execution.create_sent": 1,
        "execution.create_succeeded": 1,
        "order_wave.started": 1,
    }
    assert wave["statuses"] == {"degraded": 1, "started": 3, "succeeded": 1}
    assert wave["reason_codes"] == {
        "authoritative_confirmation": 1,
        "exchange_acknowledged": 1,
        "requires_full_authoritative_confirmation": 1,
        "submitted_to_exchange": 1,
        "test": 1,
    }
    assert wave["symbols"] == ["ETH/USDT:USDT", "SOL/USDT:USDT"]
    assert wave["action_count"] == 2
    assert wave["confirmation_count"] == 1
    assert wave["confirmations"][0]["event_type"] == "execution.confirmation_requested"

    actions = {item["action_id"]: item for item in wave["actions"]}
    create = actions["ow_9:create:0"]
    assert create["event_count"] == 2
    assert create["events_truncated"] is True
    assert create["latest_event_type"] == "execution.create_succeeded"
    assert create["latest_status"] == "succeeded"
    assert create["latest_reason_code"] == "exchange_acknowledged"
    assert create["order_ids_short"] == ["abc123"]
    assert create["client_order_ids_short"] == ["cid-entry-0"]
    assert create["events"][0]["price"] == 2500.0
    assert create["events"][0]["qty"] == 0.12

    cancel = actions["ow_9:cancel:0"]
    assert cancel["event_count"] == 1
    assert cancel["latest_event_type"] == "execution.cancel_ambiguous_terminal"
    assert cancel["latest_status"] == "degraded"
    assert cancel["order_ids_short"] == ["cancel123"]


def test_event_query_cycle_trace_reconstructs_cycle_with_nested_order_trace(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_9",
                seq=1,
                ts=1000,
                status="started",
            ),
            _monitor_row(
                event_type="rust_orchestrator.called",
                cycle_id="cy_9",
                seq=2,
                ts=1100,
                status="started",
                ids={"snapshot_id": "snap_9", "plan_id": "plan_9"},
            ),
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_9",
                seq=3,
                ts=1200,
                symbol="ETH/USDT:USDT",
                pside="long",
                side="buy",
                status="started",
                reason_code="submitted_to_exchange",
                ids={
                    "order_wave_id": "ow_9",
                    "action_id": "ow_9:create:0",
                    "plan_id": "plan_9",
                },
                data={"qty": 0.12, "price": 2500.0},
            ),
            _monitor_row(
                event_type="execution.create_failed",
                cycle_id="cy_9",
                seq=4,
                ts=1300,
                symbol="ETH/USDT:USDT",
                pside="long",
                side="buy",
                status="failed",
                reason_code="exchange_exception",
                ids={
                    "order_wave_id": "ow_9",
                    "action_id": "ow_9:create:0",
                    "plan_id": "plan_9",
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                cycle_id="cy_9",
                seq=5,
                ts=1400,
                status="succeeded",
                reason_code="authoritative_refresh",
                ids={
                    "remote_call_id": "rc_9",
                    "remote_call_group_id": "cy_9:authoritative",
                },
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_9",
                seq=6,
                ts=1500,
                status="succeeded",
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_10",
                seq=7,
                ts=1600,
                status="succeeded",
            ),
        ],
    )

    report = build_event_report(
        tmp_path / "monitor",
        cycle_id="cy_9",
        cycle_trace=True,
        include_data=True,
        limit=2,
    )

    assert report["cycle"]["matched_events"] == 6
    trace = report["cycle"]["cycle_trace"]
    assert trace["matched_cycle_events"] == 6
    assert trace["cycle_count"] == 1
    assert trace["missing_cycle_id_event_count"] == 0
    cycle = trace["cycles"][0]
    assert cycle["cycle_id"] == "cy_9"
    assert cycle["event_count"] == 6
    assert cycle["events_truncated"] is True
    assert cycle["first_ts"] == 1000
    assert cycle["last_ts"] == 1500
    assert [event["event_type"] for event in cycle["timeline"]] == [
        "cycle.started",
        "rust_orchestrator.called",
    ]
    assert cycle["timeline"][1]["data"] == {"seq": 2}

    summary = cycle["trace_summary"]
    assert summary["matched_events"] == 6
    assert summary["statuses"] == {"failed": 1, "started": 3, "succeeded": 2}
    assert summary["reason_codes"] == {
        "authoritative_refresh": 1,
        "exchange_exception": 1,
        "submitted_to_exchange": 1,
        "test": 3,
    }
    assert summary["ids"]["cycle_id"] == [{"id": "cy_9", "events": 6}]
    assert summary["ids"]["order_wave_id"] == [{"id": "ow_9", "events": 2}]
    assert summary["ids"]["plan_id"] == [{"id": "plan_9", "events": 3}]
    assert summary["ids"]["remote_call_group_id"] == [
        {"id": "cy_9:authoritative", "events": 1}
    ]

    order_trace = cycle["order_trace"]
    assert order_trace["matched_order_events"] == 2
    assert order_trace["order_wave_count"] == 1
    wave = order_trace["order_waves"][0]
    assert wave["order_wave_id"] == "ow_9"
    assert wave["event_count"] == 2
    assert wave["actions"][0]["action_id"] == "ow_9:create:0"
    assert wave["actions"][0]["latest_status"] == "failed"


def test_live_event_query_cli_accepts_trace_summary(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="order_wave.started",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
                status="started",
                source="executor",
                component="order_wave",
                tags=["execution", "order", "wave"],
                ids={"order_wave_id": "ow_7"},
            ),
            _monitor_row(
                event_type="order_wave.completed",
                cycle_id="cy_7",
                seq=2,
                ts=1100,
                status="succeeded",
                source="executor",
                component="order_wave",
                tags=["execution", "order", "summary"],
                ids={"order_wave_id": "ow_7"},
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--cycle-id",
                "cy_7",
                "--tag",
                "order",
                "--trace-summary",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["cycle"]["filters"] == {"cycle_id": "cy_7", "tags": ["order"]}
    summary = report["cycle"]["trace_summary"]
    assert summary["matched_events"] == 2
    assert summary["sources"] == {"executor": 2}
    assert summary["components"] == {"order_wave": 2}
    assert summary["tags"] == {
        "execution": 2,
        "order": 2,
        "summary": 1,
        "wave": 1,
    }
    assert summary["exchanges"] == {"binance": 2}
    assert summary["users"] == {"binance_01": 2}
    assert summary["order_waves"]["ow_7"]["event_types"] == {
        "order_wave.completed": 1,
        "order_wave.started": 1,
    }


def test_live_event_query_cli_accepts_order_trace(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
                status="started",
                ids={"order_wave_id": "ow_7", "action_id": "ow_7:create:0"},
            ),
            _monitor_row(
                event_type="execution.create_failed",
                cycle_id="cy_7",
                seq=2,
                ts=1100,
                status="failed",
                reason_code="exchange_exception",
                ids={"order_wave_id": "ow_7", "action_id": "ow_7:create:0"},
            ),
        ],
    )

    assert (
        live_event_query.main(
            [
                str(tmp_path / "monitor"),
                "--order-wave-id",
                "ow_7",
                "--order-trace",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    trace = report["query"]["order_trace"]
    assert trace["matched_order_events"] == 2
    assert trace["order_waves"][0]["actions"][0]["latest_status"] == "failed"


def test_live_event_query_cli_accepts_cycle_trace(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_7",
                seq=1,
                ts=1000,
                status="started",
            ),
            _monitor_row(
                event_type="execution.create_sent",
                cycle_id="cy_7",
                seq=2,
                ts=1100,
                status="started",
                ids={"order_wave_id": "ow_7", "action_id": "ow_7:create:0"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_8",
                seq=3,
                ts=1200,
            ),
            _monitor_row(
                event_type="remote_call.failed",
                cycle_id=None,
                seq=4,
                ts=1300,
                status="failed",
                ids={"remote_call_id": "rc_unscoped"},
            ),
        ],
    )

    assert (
        live_event_query.main(
            [str(tmp_path / "monitor"), "--cycle-trace", "--limit", "1"]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    trace = report["query"]["cycle_trace"]
    assert trace["matched_cycle_events"] == 3
    assert trace["cycle_count"] == 2
    assert trace["missing_cycle_id_event_count"] == 1
    assert trace["missing_cycle_id_events"][0]["event_type"] == "remote_call.failed"
    cycles = {item["cycle_id"]: item for item in trace["cycles"]}
    assert cycles["cy_7"]["event_count"] == 2
    assert cycles["cy_7"]["events_truncated"] is True
    assert cycles["cy_7"]["order_trace"]["matched_order_events"] == 1
    assert cycles["cy_8"]["event_count"] == 1


def test_live_event_query_cli_defaults_to_current_only_for_directory_scans(
    tmp_path, capsys
):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "2026-06-25T00-00-00.ndjson",
        [
            _monitor_row(
                event_type="cycle.started",
                cycle_id="cy_old",
                seq=1,
                ts=1000,
            )
        ],
    )
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                cycle_id="cy_current",
                seq=2,
                ts=2000,
            )
        ],
    )

    assert live_event_query.main([str(tmp_path / "monitor")]) == 0
    current_only_report = json.loads(capsys.readouterr().out)
    assert current_only_report["include_rotated"] is False
    assert current_only_report["files"] == [str(events_dir / "current.ndjson")]
    assert current_only_report["cycle_ids_sample"] == [
        {"cycle_id": "cy_current", "events": 1}
    ]

    assert live_event_query.main([str(tmp_path / "monitor"), "--include-rotated"]) == 0
    full_report = json.loads(capsys.readouterr().out)
    assert full_report["include_rotated"] is True
    assert full_report["files"] == [
        str(events_dir / "2026-06-25T00-00-00.ndjson"),
        str(events_dir / "current.ndjson"),
    ]
    assert full_report["cycle_ids_sample"] == [
        {"cycle_id": "cy_old", "events": 1},
        {"cycle_id": "cy_current", "events": 1},
    ]
