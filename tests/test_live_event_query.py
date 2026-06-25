from __future__ import annotations

import gzip
import json

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
) -> dict:
    event_ids = dict(ids or {})
    if cycle_id is not None:
        event_ids["cycle_id"] = cycle_id
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": "debug",
        "source": "live",
        "component": "test",
        "exchange": "binance",
        "user": "binance_01",
        "symbol": symbol,
        "pside": pside,
        "side": side,
        "status": status,
        "reason_code": reason_code,
        "data": {"seq": seq},
        "ids": event_ids,
    }
    payload = {"_live_event": live_event}
    payload.update(live_event["data"])
    row = {
        "exchange": "binance",
        "user": "binance_01",
        "kind": event_type,
        "tags": ["test"],
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
                "--timeline",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["query"]["filters"] == {
        "order_wave_ids": ["ow_9"],
        "psides": ["short"],
        "reason_codes": ["exchange_exception"],
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
