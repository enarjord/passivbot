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

