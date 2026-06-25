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
) -> dict:
    ids = {"cycle_id": cycle_id} if cycle_id is not None else {}
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
        "status": "succeeded",
        "reason_code": "test",
        "data": {"seq": seq},
        "ids": ids,
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

    assert discover_event_files(tmp_path / "monitor" / "binance" / "binance_01") == [
        rotated,
        current,
    ]

    report = build_event_report(tmp_path / "monitor", cycle_id="cy_1", include_data=True)

    assert report["ok"] is True
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

    assert live_event_query.main([str(tmp_path / "monitor"), "--cycle-id", "cy_7"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["cycle"]["cycle_id"] == "cy_7"
    assert report["cycle"]["matched_events"] == 1


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
