import json

from monitor_publisher import MonitorPublisher


def _make_publisher(tmp_path, **overrides):
    params = {
        "exchange": "bybit",
        "user": "user01",
        "root_dir": str(tmp_path),
        "snapshot_interval_seconds": 1.0,
        "checkpoint_interval_minutes": 10.0,
        "event_rotation_mb": 128.0,
        "event_rotation_minutes": 60.0,
        "retain_days": 7.0,
        "max_total_bytes": 1_000_000,
        "compress_rotated_segments": False,
        "retain_price_ticks": True,
        "retain_candles": True,
        "retain_fills": True,
        "price_tick_min_interval_ms": 500,
        "emit_completed_candles": True,
        "include_raw_fill_payloads": False,
    }
    params.update(overrides)
    return MonitorPublisher(**params)


def test_monitor_publisher_writes_manifest_events_and_snapshot(tmp_path):
    publisher = _make_publisher(tmp_path)

    event = publisher.record_event("bot.start", ("bot", "lifecycle"), {"status": "starting"}, ts=1000)
    assert event["seq"] == 1
    publisher.record_error("error.bot", RuntimeError("boom"), payload={"source": "test"}, ts=1001)

    root = tmp_path / "bybit" / "user01"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["last_seq"] == 2
    assert manifest["capabilities"]["history"] is True
    assert manifest["capabilities"]["history_streams"]["fills"] is True

    events = [json.loads(line) for line in (root / "events" / "current.ndjson").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["bot.start", "error.bot"]
    assert events[1]["payload"]["error_type"] == "RuntimeError"
    assert events[1]["payload"]["source"] == "test"

    written = publisher.write_snapshot(
        {"meta": {"custom": "value"}, "account": {"balance_raw": 123.0}},
        ts=2000,
        force=True,
    )
    assert written is True
    snapshot = json.loads((root / "state.latest.json").read_text())
    assert snapshot["meta"]["custom"] == "value"
    assert snapshot["meta"]["seq"] == 2
    assert snapshot["account"]["balance_raw"] == 123.0


def test_monitor_publisher_records_fills_ticks_and_completed_candles(tmp_path):
    publisher = _make_publisher(tmp_path, price_tick_min_interval_ms=500)

    publisher.record_fill(
        {"id": "fill-1", "qty": 0.01, "price": 100000.0},
        ts=1000,
        symbol="BTC/USDT:USDT",
        pside="long",
    )
    assert publisher.record_price_tick("BTC/USDT:USDT", 100000.0, ts=2000) is not None
    assert publisher.record_price_tick("BTC/USDT:USDT", 100100.0, ts=2200) is None
    assert publisher.record_price_tick("BTC/USDT:USDT", 100200.0, ts=2600) is not None

    emitted_first = publisher.record_completed_candles(
        "BTC/USDT:USDT",
        "1m",
        [
            {"ts": 60_000, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "bv": 10.0},
            {"ts": 120_000, "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "bv": 11.0},
            {"ts": 180_000, "o": 2.0, "h": 3.0, "l": 1.5, "c": 2.5, "bv": 12.0},
        ],
    )
    emitted_second = publisher.record_completed_candles(
        "BTC/USDT:USDT",
        "1m",
        [
            {"ts": 120_000, "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "bv": 11.0},
            {"ts": 180_000, "o": 2.0, "h": 3.0, "l": 1.5, "c": 2.5, "bv": 12.0},
            {"ts": 240_000, "o": 2.5, "h": 3.5, "l": 2.0, "c": 3.0, "bv": 13.0},
            {"ts": 300_000, "o": 3.0, "h": 4.0, "l": 2.5, "c": 3.5, "bv": 14.0},
        ],
    )

    assert len(emitted_first) == 1
    assert emitted_first[0]["payload"]["ts"] == 180_000
    assert len(emitted_second) == 2
    assert [entry["payload"]["ts"] for entry in emitted_second] == [240_000, 300_000]

    root = tmp_path / "bybit" / "user01" / "history"
    fills = [json.loads(line) for line in (root / "fills.current.ndjson").read_text().splitlines()]
    ticks = [json.loads(line) for line in (root / "price_ticks.current.ndjson").read_text().splitlines()]
    candles = [json.loads(line) for line in (root / "candles_1m.current.ndjson").read_text().splitlines()]

    assert len(fills) == 1
    assert fills[0]["kind"] == "fill"
    assert fills[0]["symbol"] == "BTC/USDT:USDT"
    assert len(ticks) == 2
    assert [entry["payload"]["last"] for entry in ticks] == [100000.0, 100200.0]
    assert [entry["payload"]["ts"] for entry in candles] == [180_000, 240_000, 300_000]


def test_monitor_publisher_includes_raw_fill_payload_when_enabled(tmp_path):
    publisher = _make_publisher(tmp_path, include_raw_fill_payloads=True)

    publisher.record_fill(
        {"id": "fill-1", "qty": 0.01, "price": 100000.0},
        ts=1000,
        symbol="BTC/USDT:USDT",
        pside="long",
        raw_payload={"exchange_id": "abc"},
    )

    root = tmp_path / "bybit" / "user01" / "history"
    fills = [json.loads(line) for line in (root / "fills.current.ndjson").read_text().splitlines()]
    assert fills[0]["payload"]["raw"] == {"exchange_id": "abc"}


def test_monitor_publisher_rotates_event_segments(tmp_path):
    publisher = _make_publisher(
        tmp_path,
        event_rotation_mb=0.00001,
        event_rotation_minutes=60.0,
    )

    publisher.record_event("one", ("test",), {"blob": "x" * 4000}, ts=1000)
    publisher.record_event("two", ("test",), {"blob": "y" * 4000}, ts=2000)

    events_dir = tmp_path / "bybit" / "user01" / "events"
    rotated = [path for path in events_dir.iterdir() if path.name != "current.ndjson"]
    assert rotated, "expected at least one rotated event file"
    rotated_lines = rotated[0].read_text().splitlines()
    current_lines = (events_dir / "current.ndjson").read_text().splitlines()
    assert len(rotated_lines) == 1
    assert json.loads(rotated_lines[0])["kind"] == "one"
    assert len(current_lines) == 1
    assert json.loads(current_lines[0])["kind"] == "two"
