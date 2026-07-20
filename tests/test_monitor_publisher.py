import errno
import json
import logging
import os
import stat
import threading
from pathlib import Path

from monitor_publisher import MonitorPublisher
import monitor_publisher as monitor_publisher_module


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
    runtime = {"schema_version": 1, "run_id": "run-123"}
    publisher = _make_publisher(tmp_path, runtime_identity=runtime)

    event = publisher.record_event("bot.start", ("bot", "lifecycle"), {"status": "starting"}, ts=1000)
    assert event["seq"] == 1
    publisher.record_error(
        "error.bot", RuntimeError("boom"), payload={"source": "start_bot"}, ts=1001
    )
    publisher.close()

    root = tmp_path / "bybit" / "user01"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["last_seq"] == 2
    assert manifest["capabilities"]["history"] is True
    assert manifest["capabilities"]["history_streams"]["fills"] is True
    assert manifest["runtime"] == runtime

    events = [json.loads(line) for line in (root / "events" / "current.ndjson").read_text().splitlines()]
    assert [event["kind"] for event in events] == ["bot.start", "error.bot"]
    assert events[1]["payload"]["error_type"] == "RuntimeError"
    assert events[1]["payload"]["source"] == "start_bot"

    written = publisher.write_snapshot(
        {"meta": {"custom": "value"}, "account": {"balance_raw": 123.0}},
        ts=2000,
        force=True,
    )
    assert written is True
    snapshot = json.loads((root / "state.latest.json").read_text())
    assert snapshot["meta"]["custom"] == "value"
    assert snapshot["meta"]["seq"] == 2
    assert snapshot["meta"]["runtime"] == runtime
    assert snapshot["account"]["balance_raw"] == 123.0


def test_monitor_publisher_record_error_redacts_diagnostic_fields_and_keeps_safe_context(
    tmp_path,
):
    publisher = _make_publisher(tmp_path)
    secret = "https://api.example.test/private?api_key=top-secret Authorization: Bearer top-secret"
    schemeless_secret = "api.example.test/private/api_key/top-secret"
    schemeless_url = "api.kucoin.com/api/v1/accounts"
    opaque_token = (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dGhpc19pc19vcGFxdWVfYnl0ZXM"
    )

    class Unrenderable:
        def __str__(self):
            raise RuntimeError(secret)

    event = publisher.record_error(
        "error.bot",
        RuntimeError(secret),
        tags=("error", "bot"),
        payload={
            "source": "start_bot",
            "operation": "load_account",
            "attempt": 2,
            "status": schemeless_url,
            "count": 1 << 80,
            "stage": "init_markets",
            "endpoint": schemeless_secret,
            "code": "api_key_top-secret",
            "action": "sk_live_123456",
            "cycle_id": opaque_token,
            "detail": Unrenderable(),
            "message": secret,
            "error": secret,
            "error_repr": secret,
            "exception": secret,
            "traceback": secret,
            "url": secret,
            "raw_response": secret,
            "authorization": secret,
            "context": {
                "operation": "load_account",
                "response_body": secret,
                "nested": {"request_headers": secret, "safe": "kept"},
            },
        },
        ts=1_234,
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    assert event["ts"] == 1_234
    assert event["seq"] == 1
    assert event["kind"] == "error.bot"
    assert event["tags"] == ["error", "bot"]
    assert event["symbol"] == "BTC/USDT:USDT"
    assert event["pside"] == "long"
    assert event["payload"] == {
        "source": "start_bot",
        "stage": "init_markets",
        "attempt": 2,
        "error_type": "RuntimeError",
    }

    opaque_event = publisher.record_error(
        "error.bot",
        RuntimeError("safe"),
        payload={
            "source": opaque_token,
            "stage": opaque_token,
            "status": opaque_token,
            "attempt": 3,
        },
        ts=1_235,
    )
    assert opaque_event["payload"] == {
        "attempt": 3,
        "error_type": "RuntimeError",
    }

    serialized = publisher.current_events_path.read_text()
    assert secret not in serialized
    assert schemeless_secret not in serialized
    assert schemeless_url not in serialized
    assert opaque_token not in serialized
    assert publisher.record_event("bot.stop", ("bot",), ts=1_236)["seq"] == 3
    assert len(publisher.current_events_path.read_text().splitlines()) == 3


def test_monitor_publisher_record_error_bounds_pathological_exception_type(tmp_path):
    publisher = _make_publisher(tmp_path)
    pathological_error = type("Error" + "X" * 512, (Exception,), {})

    event = publisher.record_error("error.bot", pathological_error("safe"))

    assert event["payload"]["error_type"] == "unknown"
    assert len(event["payload"]["error_type"]) <= 80


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


def test_monitor_publisher_concurrent_event_writes_keep_seq_and_manifest(tmp_path):
    publisher = _make_publisher(
        tmp_path,
        event_rotation_mb=0.00001,
        event_rotation_minutes=60.0,
    )
    n_threads = 4
    n_per_thread = 50
    results = []
    errors = []
    results_lock = threading.Lock()

    def write_events(worker_no):
        try:
            for event_no in range(n_per_thread):
                event = publisher.record_event(
                    "stress.event",
                    ("test",),
                    {"worker": worker_no, "event": event_no, "blob": "x" * 200},
                )
                if event_no % 10 == 0:
                    assert publisher.write_snapshot({"worker": worker_no}, force=True)
                with results_lock:
                    results.append(event)
        except Exception as exc:
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=write_events, args=(idx,)) for idx in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert errors == []
    assert len(results) == n_threads * n_per_thread
    assert all(event is not None for event in results)
    seqs = sorted(event["seq"] for event in results)
    assert seqs == list(range(1, n_threads * n_per_thread + 1))

    publisher.close()
    root = tmp_path / "bybit" / "user01"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["last_seq"] == n_threads * n_per_thread
    event_lines = []
    for path in sorted((root / "events").glob("*.ndjson")):
        event_lines.extend(path.read_text().splitlines())
    assert len(event_lines) == n_threads * n_per_thread
    assert not list(root.rglob("*.tmp"))


def test_monitor_disk_full_errors_are_coalesced(tmp_path, monkeypatch, caplog):
    publisher = _make_publisher(tmp_path)

    def fail_write(path, payload):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", fail_write)

    with caplog.at_level(logging.ERROR):
        publisher._write_manifest()
        publisher._write_manifest()
        publisher.write_snapshot({}, force=True)

    messages = [record.message for record in caplog.records]
    disk_messages = [msg for msg in messages if "disk full" in msg]
    assert len(disk_messages) == 1
    assert "suppressing repeat disk-full monitor errors for 60s" in disk_messages[0]


def test_monitor_manifest_coalesces_event_and_history_writes_within_cadence(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=1.0)
    original_atomic_write = monitor_publisher_module._atomic_write_json
    manifest_writes = []

    def count_manifest_writes(path, payload):
        if path == publisher.manifest_path:
            manifest_writes.append(payload)
        return original_atomic_write(path, payload)

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", count_manifest_writes)
    monkeypatch.setattr(publisher, "_monotonic_ms", lambda: 1_001)
    publisher._last_manifest_write_monotonic_ms = 1_000
    for offset in range(1, 6):
        assert publisher.record_event("test.event", ("test",), ts=1_000 + offset)
        assert publisher.record_history_entry(
            "fills", "fill", {"id": offset}, ts=1_000 + offset
        )

    assert manifest_writes == []
    assert publisher._manifest_dirty is True
    root = tmp_path / "bybit" / "user01"
    assert len((root / "events" / "current.ndjson").read_text().splitlines()) == 5
    assert len((root / "history" / "fills.current.ndjson").read_text().splitlines()) == 5


def test_monitor_manifest_checkpoints_latest_seq_after_cadence_expires(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=1.0)
    original_atomic_write = monitor_publisher_module._atomic_write_json
    manifest_writes = []

    def count_manifest_writes(path, payload):
        if path == publisher.manifest_path:
            manifest_writes.append(payload)
        return original_atomic_write(path, payload)

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", count_manifest_writes)
    monkeypatch.setattr(publisher, "_monotonic_ms", lambda: 2_000)
    publisher._last_manifest_write_monotonic_ms = 1_000
    event = publisher.record_event("test.event", ("test",), ts=2_000)

    assert event["seq"] == 1
    assert len(manifest_writes) == 1
    assert manifest_writes[0]["last_seq"] == 1
    assert publisher._manifest_dirty is False


def test_monitor_manifest_cadence_ignores_old_and_decreasing_record_timestamps(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=1.0)
    original_atomic_write = monitor_publisher_module._atomic_write_json
    manifest_writes = []

    def count_manifest_writes(path, payload):
        if path == publisher.manifest_path:
            manifest_writes.append(payload)
        return original_atomic_write(path, payload)

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", count_manifest_writes)
    monkeypatch.setattr(publisher, "_monotonic_ms", lambda: 1_001)
    publisher._last_manifest_write_monotonic_ms = 1_000

    assert publisher.record_history_entry("fills", "fill", {"id": 1}, ts=10_000)
    assert publisher.record_history_entry("fills", "fill", {"id": 2}, ts=5_000)
    assert publisher.record_history_entry("fills", "fill", {"id": 3}, ts=1)

    assert manifest_writes == []
    assert publisher._manifest_dirty is True


def test_monitor_failed_due_manifest_checkpoint_retries_on_next_write(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=1.0)
    original_atomic_write = monitor_publisher_module._atomic_write_json
    manifest_attempts = []

    def fail_first_manifest_write(path, payload):
        if path == publisher.manifest_path:
            manifest_attempts.append(payload)
            if len(manifest_attempts) == 1:
                raise OSError(errno.ENOSPC, "No space left on device")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(
        monitor_publisher_module, "_atomic_write_json", fail_first_manifest_write
    )
    monkeypatch.setattr(publisher, "_monotonic_ms", lambda: 2_000)
    publisher._last_manifest_write_monotonic_ms = 1_000

    first = publisher.record_event("test.event", ("test",), ts=2_000)
    assert first["seq"] == 1
    assert publisher._manifest_dirty is True
    assert publisher._manifest_retry_needed is True

    second = publisher.record_event("test.event", ("test",), ts=2_001)
    assert second["seq"] == 2
    assert len(manifest_attempts) == 2
    assert publisher._manifest_dirty is False
    assert publisher._manifest_retry_needed is False
    manifest = json.loads(publisher.manifest_path.read_text())
    assert manifest["last_seq"] == 2


def test_monitor_recovers_seq_from_stale_manifest_after_unclean_restart(tmp_path):
    publisher = _make_publisher(tmp_path)
    manifest = json.loads(publisher.manifest_path.read_text())
    manifest["last_seq"] = 3
    publisher.manifest_path.write_text(json.dumps(manifest) + "\n")
    publisher.current_events_path.write_text(
        publisher._serialize_event_line({"seq": 8, "kind": "persisted"}) + "\n"
    )

    restarted = _make_publisher(tmp_path)
    assert restarted.seq == 8
    event = restarted.record_event("test.event", ("test",), ts=1_000)
    assert event["seq"] == 9


def test_monitor_recovery_skips_blank_malformed_and_invalid_utf8_trailing_rows(tmp_path):
    publisher = _make_publisher(tmp_path)
    manifest = json.loads(publisher.manifest_path.read_text())
    manifest["last_seq"] = 3
    publisher.manifest_path.write_text(json.dumps(manifest) + "\n")
    publisher.current_events_path.write_bytes(
        publisher._serialize_event_line({"seq": 8, "kind": "persisted"}).encode("utf-8")
        + b"\n\n{not-json}\n\xff\xfe\n"
    )

    restarted = _make_publisher(tmp_path)
    assert restarted.seq == 8


def test_monitor_recovery_reads_oversized_final_row_in_bounded_chunks(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path)
    oversized_row = publisher._serialize_event_line(
        {
            "seq": 42,
            "kind": "persisted",
            "payload": {
                "blob": "x"
                * (monitor_publisher_module._CURRENT_EVENTS_REVERSE_READ_BYTES * 2)
            },
        },
    ).encode("utf-8")
    publisher.current_events_path.write_bytes(oversized_row)
    original_open = open
    read_sizes = []

    class TrackingFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.file.__exit__(exc_type, exc_value, traceback)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.file.read(size)

    def tracking_open(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        if path == publisher.current_events_path and args and args[0] == "rb":
            return TrackingFile(file)
        return file

    monkeypatch.setattr(monitor_publisher_module, "open", tracking_open, raising=False)

    assert publisher._max_recoverable_event_seq_in_current_segment() == 42
    assert len(read_sizes) >= 3
    assert max(read_sizes) <= monitor_publisher_module._CURRENT_EVENTS_REVERSE_READ_BYTES


def test_monitor_recovery_accepts_checksummed_crlf_row(tmp_path):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    manifest = json.loads(publisher.manifest_path.read_text())
    manifest["last_seq"] = 3
    publisher.manifest_path.write_text(json.dumps(manifest) + "\n")
    publisher.current_events_path.write_bytes(
        publisher._serialize_event_line({"seq": 8, "kind": "persisted"}).encode("utf-8")
        + b"\r\n"
    )

    restarted = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    event = restarted.record_event("test.after_restart", ("test",), {}, ts=2_000)

    assert restarted.seq == 9
    assert event["seq"] == 9


def test_monitor_restart_does_not_reuse_seq_after_oversized_uncheckpointed_event(tmp_path):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    first = publisher.record_event(
        "test.oversized",
        ("test",),
        {
            "blob": "x"
            * (monitor_publisher_module._CURRENT_EVENTS_REVERSE_READ_BYTES + 1024)
        },
        ts=1_000,
    )
    assert first["seq"] == 1
    assert json.loads(publisher.manifest_path.read_text())["last_seq"] == 0

    restarted = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    second = restarted.record_event("test.after_restart", ("test",), {}, ts=2_000)

    assert restarted.seq == 2
    assert second["seq"] == 2
    rows = [
        json.loads(line) for line in restarted.current_events_path.read_text().splitlines()
    ]
    assert [row["seq"] for row in rows] == [1, 2]


def test_monitor_recovery_ignores_lower_malformed_marker_after_newer_valid_rows(tmp_path):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    manifest = json.loads(publisher.manifest_path.read_text())
    manifest["last_seq"] = 3
    publisher.manifest_path.write_text(json.dumps(manifest) + "\n")
    corrupted_trailer = publisher._serialize_event_line(
        {"seq": 999_999, "kind": "persisted"}
    ).replace('"kind":"persisted"', '"kind":"tampered"')
    tampered_trailer_seq = publisher._serialize_event_line(
        {"seq": 42, "kind": "persisted"}
    ).replace(',"seq":42}}', ',"seq":9007199254740993}}')
    publisher.current_events_path.write_text(
        "".join(
            publisher._serialize_event_line({"seq": seq, "kind": "persisted"}) + "\n"
            for seq in range(4, 9)
        )
        + '{"seq":3,garbage}\n'
        + '{"seq":9007199254740993,garbage}\n'
        + '{"seq":3,"seq":9007199254740993}\n'
        + corrupted_trailer
        + "\n"
        + tampered_trailer_seq
        + "\n"
    )

    restarted = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    event = restarted.record_event("test.after_restart", ("test",), {}, ts=2_000)

    assert restarted.seq == 9
    assert event["seq"] == 9


def test_monitor_recovery_ignores_nested_seq_in_torn_event_payload(tmp_path):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    manifest = json.loads(publisher.manifest_path.read_text())
    manifest["last_seq"] = 3
    publisher.manifest_path.write_text(json.dumps(manifest) + "\n")
    publisher.current_events_path.write_text(
        "".join(
            publisher._serialize_event_line({"seq": seq, "kind": "persisted"}) + "\n"
            for seq in range(4, 9)
        )
        + '{"exchange":"x","kind":"test","payload":{"seq":9007199254740993}'
    )

    restarted = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    event = restarted.record_event("test.after_restart", ("test",), {}, ts=2_000)

    assert restarted.seq == 9
    assert event["seq"] == 9


def test_monitor_recovery_uses_envelope_seq_after_nested_payload_seq(tmp_path):
    publisher = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    event = publisher.record_event(
        "test.nested_seq",
        ("test",),
        {"seq": 999_999},
        ts=1_000,
    )
    assert event["seq"] == 1

    restarted = _make_publisher(tmp_path, snapshot_interval_seconds=3600.0)
    next_event = restarted.record_event("test.after_restart", ("test",), {}, ts=2_000)

    assert restarted.seq == 2
    assert next_event["seq"] == 2


def test_monitor_event_rotation_forces_pre_and_post_manifest_checkpoints(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, event_rotation_mb=0.00001)
    publisher._last_manifest_write_monotonic_ms = publisher._monotonic_ms()
    assert publisher.record_event("one", ("test",), ts=1_001)
    original_write_manifest = publisher._write_manifest
    checkpoints = []

    def capture_checkpoint(**kwargs):
        checkpoints.append((publisher.seq, publisher.current_segment_started_ms))
        return original_write_manifest(**kwargs)

    monkeypatch.setattr(publisher, "_write_manifest", capture_checkpoint)
    assert publisher.record_event("two", ("test",), ts=1_002)

    assert [seq for seq, _started_ms in checkpoints] == [1, 1]
    assert checkpoints[0][1] != 1_002
    assert checkpoints[1][1] == 1_002
    manifest = json.loads(publisher.manifest_path.read_text())
    assert manifest["last_seq"] == 1
    assert manifest["current_segment_started_ms"] == 1_002


def test_monitor_event_rotation_waits_for_successful_pre_rotation_checkpoint(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, event_rotation_mb=0.00001)
    publisher._last_manifest_write_monotonic_ms = publisher._monotonic_ms()
    assert publisher.record_event("one", ("test",), ts=1_001)
    current_before = publisher.current_events_path.read_text()
    original_write_manifest = publisher._write_manifest
    attempts = []

    def fail_first_checkpoint(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            publisher._manifest_dirty = True
            publisher._manifest_retry_needed = True
            return False
        return original_write_manifest(**kwargs)

    monkeypatch.setattr(publisher, "_write_manifest", fail_first_checkpoint)

    event = publisher.record_event("two", ("test",), ts=1_002)

    assert event["seq"] == 2
    assert current_before in publisher.current_events_path.read_text()
    assert list(publisher.events_dir.glob("*.ndjson")) == [publisher.current_events_path]
    assert publisher.current_segment_started_ms != 1_002
    assert len(attempts) == 2
    assert publisher._manifest_dirty is False
    assert publisher._manifest_retry_needed is False
    assert json.loads(publisher.manifest_path.read_text())["last_seq"] == 2

    third = publisher.record_event("three", ("test",), ts=1_003)

    assert third["seq"] == 3
    rotated = [
        path
        for path in publisher.events_dir.glob("*.ndjson")
        if path != publisher.current_events_path
    ]
    assert len(rotated) == 1
    assert publisher.current_segment_started_ms == 1_003


def test_monitor_history_rotation_snapshot_and_close_force_manifest_checkpoints(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, event_rotation_mb=0.00001)
    publisher._last_manifest_write_monotonic_ms = publisher._monotonic_ms()
    original_write_manifest = publisher._write_manifest
    checkpoints = []

    def capture_checkpoint(**kwargs):
        checkpoints.append((publisher.seq, dict(publisher.history_segment_started_ms)))
        return original_write_manifest(**kwargs)

    monkeypatch.setattr(publisher, "_write_manifest", capture_checkpoint)
    assert publisher.record_history_entry("fills", "fill", {"id": 1}, ts=1_001)
    assert publisher.record_history_entry("fills", "fill", {"id": 2}, ts=1_002)
    assert len(checkpoints) == 1
    assert checkpoints[0][1]["fills"] == 1_002

    assert publisher.write_snapshot({"account": {}}, ts=1_003, force=True)
    publisher.close()
    assert len(checkpoints) == 3


def test_monitor_publisher_event_phase_timing_uses_fixed_boundaries(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path)
    clock = {"ns": 1_000_000_000}
    thread_cpu_clock = {"ns": 500_000_000}
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "monotonic_ns",
        lambda: clock["ns"],
    )
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: thread_cpu_clock["ns"],
    )

    class TimedLock:
        def __enter__(self):
            clock["ns"] += 1_000_000

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    original_dumps = monitor_publisher_module.json.dumps
    original_open = open

    def timed_dumps(*args, **kwargs):
        clock["ns"] += 3_000_000
        return original_dumps(*args, **kwargs)

    class TimedFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self.file.__exit__(exc_type, exc_value, traceback)

        def write(self, value):
            clock["ns"] += 5_000_000
            return self.file.write(value)

    def timed_open(*args, **kwargs):
        return TimedFile(original_open(*args, **kwargs))

    publisher._lock = TimedLock()
    monkeypatch.setattr(
        publisher,
        "_rotate_events_if_needed",
        lambda **_kwargs: clock.__setitem__("ns", clock["ns"] + 2_000_000),
    )
    monkeypatch.setattr(
        publisher,
        "_write_manifest_if_due",
        lambda **kwargs: (
            kwargs["on_attempt"](),
            clock.__setitem__("ns", clock["ns"] + 7_000_000),
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_prune_retention",
        lambda **kwargs: (
            kwargs["on_run"](),
            thread_cpu_clock.__setitem__("ns", thread_cpu_clock["ns"] + 4_000_000),
            clock.__setitem__("ns", clock["ns"] + 11_000_000),
        ),
    )
    monkeypatch.setattr(monitor_publisher_module.json, "dumps", timed_dumps)
    monkeypatch.setattr(monitor_publisher_module, "open", timed_open, raising=False)
    publisher._last_manifest_write_monotonic_ms = None

    event, timing = publisher._record_event_timed("test.event", ("test",), {"value": 1})

    assert event is not None
    assert timing == {
        "lock_wait_ns": 1_000_000,
        "rotation_ns": 2_000_000,
        "persist_ns": 8_000_000,
        "maintenance_ns": 18_000_000,
        "manifest_checkpoint_count": 1,
        "manifest_checkpoint_ns_total": 7_000_000,
        "manifest_checkpoint_ns_max": 7_000_000,
        "retention_run_count": 1,
        "retention_ns_total": 11_000_000,
        "retention_ns_max": 11_000_000,
        "retention_thread_cpu_ns_total": 4_000_000,
        "retention_thread_cpu_ns_max": 4_000_000,
        "retention_non_cpu_ns_total": 7_000_000,
        "retention_non_cpu_ns_max": 7_000_000,
        "retention_inventory_ns_total": 0,
        "retention_inventory_ns_max": 0,
        "retention_age_filter_ns_total": 0,
        "retention_age_filter_ns_max": 0,
        "retention_cap_prune_ns_total": 0,
        "retention_cap_prune_ns_max": 0,
        "retention_age_unlink_ns_total": 0,
        "retention_age_unlink_ns_max": 0,
        "retention_cap_unlink_ns_total": 0,
        "retention_cap_unlink_ns_max": 0,
        "retention_inventory_entries_visited": 0,
        "retention_inventory_candidates": 0,
        "retention_age_deleted": 0,
        "retention_cap_deleted": 0,
    }


def test_monitor_publisher_event_failure_retains_reached_phase_timing(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path)
    clock = {"ns": 1_000_000_000}
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "monotonic_ns",
        lambda: clock["ns"],
    )

    def fail_dumps(*_args, **_kwargs):
        clock["ns"] += 3_000_000
        raise TypeError("not serializable")

    monkeypatch.setattr(
        publisher,
        "_rotate_events_if_needed",
        lambda **_kwargs: clock.__setitem__("ns", clock["ns"] + 2_000_000),
    )
    monkeypatch.setattr(monitor_publisher_module.json, "dumps", fail_dumps)

    event, timing = publisher._record_event_timed("test.event", ("test",), {"value": 1})

    assert event is None
    assert timing == {
        "lock_wait_ns": 0,
        "rotation_ns": 2_000_000,
        "persist_ns": 3_000_000,
        "maintenance_ns": 0,
        "manifest_checkpoint_count": 0,
        "manifest_checkpoint_ns_total": 0,
        "manifest_checkpoint_ns_max": 0,
        "retention_run_count": 0,
        "retention_ns_total": 0,
        "retention_ns_max": 0,
        "retention_thread_cpu_ns_total": 0,
        "retention_thread_cpu_ns_max": 0,
        "retention_non_cpu_ns_total": 0,
        "retention_non_cpu_ns_max": 0,
        "retention_inventory_ns_total": 0,
        "retention_inventory_ns_max": 0,
        "retention_age_filter_ns_total": 0,
        "retention_age_filter_ns_max": 0,
        "retention_cap_prune_ns_total": 0,
        "retention_cap_prune_ns_max": 0,
        "retention_age_unlink_ns_total": 0,
        "retention_age_unlink_ns_max": 0,
        "retention_cap_unlink_ns_total": 0,
        "retention_cap_unlink_ns_max": 0,
        "retention_inventory_entries_visited": 0,
        "retention_inventory_candidates": 0,
        "retention_age_deleted": 0,
        "retention_cap_deleted": 0,
    }


def test_monitor_publisher_event_timing_counts_due_maintenance_only(tmp_path):
    publisher = _make_publisher(tmp_path)
    publisher._last_manifest_write_monotonic_ms = None

    _first_event, first_timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )
    _second_event, second_timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 2}, ts=100_001
    )

    assert first_timing["manifest_checkpoint_count"] == 1
    assert first_timing["retention_run_count"] == 1
    assert first_timing["manifest_checkpoint_ns_total"] >= 0
    assert first_timing["retention_ns_total"] >= 0
    assert second_timing["manifest_checkpoint_count"] == 0
    assert second_timing["retention_run_count"] == 0
    assert second_timing["manifest_checkpoint_ns_total"] == 0
    assert second_timing["retention_ns_total"] == 0


def test_monitor_publisher_event_timing_counts_failed_due_attempts(tmp_path, monkeypatch, caplog):
    publisher = _make_publisher(tmp_path)
    publisher._last_manifest_write_monotonic_ms = None
    inventory_attempts = []

    def fail_atomic_write(*_args, **_kwargs):
        raise OSError("manifest unavailable")

    def fail_retention_inventory(**_kwargs):
        inventory_attempts.append(None)
        raise OSError("retention unavailable")

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", fail_atomic_write)
    monkeypatch.setattr(publisher, "_retention_inventory", fail_retention_inventory)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert timing["manifest_checkpoint_count"] == 1
    assert timing["retention_run_count"] == 1
    assert timing["manifest_checkpoint_ns_total"] >= timing["manifest_checkpoint_ns_max"]
    assert timing["retention_ns_total"] >= timing["retention_ns_max"]
    assert "writing manifest" in caplog.text
    assert "retention pruning failed" in caplog.text
    assert len(inventory_attempts) == 1

    publisher._prune_retention(now_ms=100_001)
    assert len(inventory_attempts) == 1
    publisher._prune_retention(now_ms=160_000)
    assert len(inventory_attempts) == 2


def test_monitor_publisher_event_timing_records_retention_phases_for_due_run(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=0.0, max_total_bytes=0)
    old_candidate = publisher.events_dir / "old.ndjson"
    cap_candidate = publisher.history_dir / "current.ndjson"
    _write_retention_file(old_candidate, size=10, mtime_ms=1)
    _write_retention_file(cap_candidate, size=10, mtime_ms=100_000)
    expected_entries_visited = len(list(publisher.root.rglob("*")))
    clock = {"ns": 0}
    original_scandir = monitor_publisher_module.os.scandir
    original_unlink = Path.unlink
    inventory_scans = []

    def timed_scandir(path):
        inventory_scans.append(Path(path))
        clock["ns"] += 1_000
        return original_scandir(path)

    def timed_unlink(path, *args, **kwargs):
        if path == old_candidate:
            clock["ns"] += 2_000
        elif path == cap_candidate:
            clock["ns"] += 3_000
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(monitor_publisher_module.time, "monotonic_ns", lambda: clock["ns"])
    monkeypatch.setattr(monitor_publisher_module.os, "scandir", timed_scandir)
    monkeypatch.setattr(Path, "unlink", timed_unlink)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert timing["retention_run_count"] == 1
    expected_inventory_ns = len(inventory_scans) * 1_000
    assert timing["retention_inventory_ns_total"] == expected_inventory_ns
    assert timing["retention_inventory_ns_max"] == expected_inventory_ns
    assert timing["retention_age_filter_ns_total"] == 2_000
    assert timing["retention_age_filter_ns_max"] == 2_000
    assert timing["retention_cap_prune_ns_total"] == 3_000
    assert timing["retention_cap_prune_ns_max"] == 3_000
    assert timing["retention_age_unlink_ns_total"] == 2_000
    assert timing["retention_age_unlink_ns_max"] == 2_000
    assert timing["retention_cap_unlink_ns_total"] == 3_000
    assert timing["retention_cap_unlink_ns_max"] == 3_000
    assert timing["retention_inventory_entries_visited"] == expected_entries_visited
    assert timing["retention_inventory_candidates"] == 2
    assert timing["retention_age_deleted"] == 1
    assert timing["retention_cap_deleted"] == 1


def test_monitor_publisher_event_timing_zeroes_retention_phases_when_not_due(tmp_path):
    publisher = _make_publisher(tmp_path)
    publisher.last_retention_ms = 100_000

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_001
    )

    assert event is not None
    assert timing["retention_run_count"] == 0
    for key in (
        "retention_thread_cpu_ns_total",
        "retention_thread_cpu_ns_max",
        "retention_non_cpu_ns_total",
        "retention_non_cpu_ns_max",
        "retention_inventory_ns_total",
        "retention_inventory_ns_max",
        "retention_age_filter_ns_total",
        "retention_age_filter_ns_max",
        "retention_cap_prune_ns_total",
        "retention_cap_prune_ns_max",
        "retention_age_unlink_ns_total",
        "retention_age_unlink_ns_max",
        "retention_cap_unlink_ns_total",
        "retention_cap_unlink_ns_max",
        "retention_inventory_entries_visited",
        "retention_inventory_candidates",
        "retention_age_deleted",
        "retention_cap_deleted",
    ):
        assert timing[key] == 0


def test_monitor_publisher_event_timing_records_retention_cpu_attribution_for_due_run(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path)
    clocks = {"wall_ns": 10_000, "thread_cpu_ns": 1_000}

    monkeypatch.setattr(
        monitor_publisher_module.time, "monotonic_ns", lambda: clocks["wall_ns"]
    )
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: clocks["thread_cpu_ns"],
    )

    def run_retention(**kwargs):
        kwargs["on_run"]()
        clocks["wall_ns"] += 80
        clocks["thread_cpu_ns"] += 100

    monkeypatch.setattr(publisher, "_prune_retention", run_retention)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert timing["retention_run_count"] == 1
    assert timing["retention_ns_total"] == 80
    assert timing["retention_thread_cpu_ns_total"] == 100
    assert timing["retention_thread_cpu_ns_max"] == 100
    assert timing["retention_non_cpu_ns_total"] == 0
    assert timing["retention_non_cpu_ns_max"] == 0


def test_monitor_publisher_event_timing_omits_cpu_clock_when_retention_not_due(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path)
    publisher.last_retention_ms = 100_000
    thread_cpu_calls = []

    monkeypatch.setattr(monitor_publisher_module.time, "monotonic_ns", lambda: 10_000)
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: thread_cpu_calls.append(None),
    )

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_001
    )

    assert event is not None
    assert thread_cpu_calls == []
    assert timing["retention_run_count"] == 0
    assert timing["retention_thread_cpu_ns_total"] == 0
    assert timing["retention_thread_cpu_ns_max"] == 0
    assert timing["retention_non_cpu_ns_total"] == 0
    assert timing["retention_non_cpu_ns_max"] == 0


def test_monitor_publisher_event_timing_retains_cpu_attribution_after_retention_error(
    tmp_path, monkeypatch, caplog
):
    publisher = _make_publisher(tmp_path)
    clocks = {"wall_ns": 10_000, "thread_cpu_ns": 1_000}

    monkeypatch.setattr(
        monitor_publisher_module.time, "monotonic_ns", lambda: clocks["wall_ns"]
    )
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: clocks["thread_cpu_ns"],
    )

    def fail_retention_inventory(**_kwargs):
        clocks["wall_ns"] += 80
        clocks["thread_cpu_ns"] += 30
        raise OSError("retention unavailable")

    monkeypatch.setattr(publisher, "_retention_inventory", fail_retention_inventory)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert "retention pruning failed: retention unavailable" in caplog.text
    assert timing["retention_run_count"] == 1
    assert timing["retention_ns_total"] == 80
    assert timing["retention_thread_cpu_ns_total"] == 30
    assert timing["retention_thread_cpu_ns_max"] == 30
    assert timing["retention_non_cpu_ns_total"] == 50
    assert timing["retention_non_cpu_ns_max"] == 50


def test_monitor_publisher_event_timing_includes_due_rotation_retention(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, event_rotation_mb=0.000001)
    publisher.current_events_path.write_text("rotated event\n")
    clocks = {"wall_ns": 10_000, "thread_cpu_ns": 1_000}

    monkeypatch.setattr(
        monitor_publisher_module.time, "monotonic_ns", lambda: clocks["wall_ns"]
    )
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: clocks["thread_cpu_ns"],
    )

    def timed_inventory(**_kwargs):
        clocks["wall_ns"] += 80
        clocks["thread_cpu_ns"] += 30
        return 0, []

    monkeypatch.setattr(publisher, "_retention_inventory", timed_inventory)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert timing["retention_run_count"] == 1
    assert timing["retention_ns_total"] == 80
    assert timing["retention_thread_cpu_ns_total"] == 30
    assert timing["retention_non_cpu_ns_total"] == 50


def test_monitor_publisher_event_timing_drains_external_retention(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path)
    clocks = {"wall_ns": 10_000, "thread_cpu_ns": 1_000}

    monkeypatch.setattr(
        monitor_publisher_module.time, "monotonic_ns", lambda: clocks["wall_ns"]
    )
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "thread_time_ns",
        lambda: clocks["thread_cpu_ns"],
    )

    def timed_inventory(**_kwargs):
        clocks["wall_ns"] += 70
        clocks["thread_cpu_ns"] += 20
        return 0, []

    monkeypatch.setattr(publisher, "_retention_inventory", timed_inventory)

    assert publisher.write_snapshot({"account": {}}, ts=100_000, force=True)
    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_001
    )

    assert event is not None
    assert timing["retention_run_count"] == 1
    assert timing["retention_ns_total"] == 70
    assert timing["retention_thread_cpu_ns_total"] == 20
    assert timing["retention_non_cpu_ns_total"] == 50


def test_monitor_publisher_thread_cpu_clock_failure_does_not_block_retention(
    tmp_path, monkeypatch, caplog
):
    publisher = _make_publisher(tmp_path)
    clock = {"wall_ns": 10_000}
    inventory_attempts = []

    monkeypatch.setattr(
        monitor_publisher_module.time, "monotonic_ns", lambda: clock["wall_ns"]
    )

    def fail_thread_cpu_clock():
        raise RuntimeError("thread clock unavailable")

    def timed_inventory(**_kwargs):
        inventory_attempts.append(None)
        clock["wall_ns"] += 80
        return 0, []

    monkeypatch.setattr(
        monitor_publisher_module.time, "thread_time_ns", fail_thread_cpu_clock
    )
    monkeypatch.setattr(publisher, "_retention_inventory", timed_inventory)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert inventory_attempts == [None]
    assert timing["retention_run_count"] == 1
    assert timing["retention_ns_total"] == 80
    assert timing["retention_thread_cpu_ns_total"] == 0
    assert timing["retention_non_cpu_ns_total"] == 0
    assert caplog.text.count("thread CPU clock unavailable") == 1


def test_monitor_publisher_event_timing_excludes_tolerated_age_disappearance(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=0.0)
    disappeared = publisher.events_dir / "disappeared.ndjson"
    _write_retention_file(disappeared, size=10, mtime_ms=1)
    original_unlink = Path.unlink

    def disappear_then_unlink(path, *args, **kwargs):
        if path == disappeared:
            original_unlink(path, *args, **kwargs)
            raise FileNotFoundError(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", disappear_then_unlink)
    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert not disappeared.exists()
    assert timing["retention_inventory_candidates"] == 1
    assert timing["retention_age_deleted"] == 0
    assert timing["retention_cap_deleted"] == 0
    assert timing["retention_age_unlink_ns_total"] >= 0
    assert timing["retention_age_filter_ns_total"] >= timing["retention_age_unlink_ns_total"]


def test_monitor_publisher_event_timing_keeps_completed_phases_after_cap_unlink_error(
    tmp_path, monkeypatch, caplog
):
    publisher = _make_publisher(tmp_path, retain_days=-1.0, max_total_bytes=0)
    candidate = publisher.events_dir / "candidate.ndjson"
    _write_retention_file(candidate, size=10, mtime_ms=1)
    clock = {"ns": 0}
    original_scandir = monitor_publisher_module.os.scandir

    def timed_scandir(path):
        clock["ns"] += 1_000
        return original_scandir(path)

    def fail_candidate_unlink(path, *args, **_kwargs):
        if path == candidate:
            clock["ns"] += 4_000
            raise OSError("candidate unavailable")
        raise AssertionError(f"unexpected unlink: {path}")

    monkeypatch.setattr(monitor_publisher_module.time, "monotonic_ns", lambda: clock["ns"])
    monkeypatch.setattr(monitor_publisher_module.os, "scandir", timed_scandir)
    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert candidate.exists()
    assert "retention pruning failed: candidate unavailable" in caplog.text
    assert timing["retention_inventory_entries_visited"] > 0
    assert timing["retention_inventory_candidates"] == 1
    assert timing["retention_inventory_ns_total"] > 0
    assert timing["retention_age_filter_ns_total"] == 0
    assert timing["retention_cap_prune_ns_total"] == 4_000
    assert timing["retention_cap_prune_ns_max"] == 4_000
    assert timing["retention_cap_unlink_ns_total"] == 4_000
    assert timing["retention_cap_unlink_ns_max"] == 4_000
    assert timing["retention_cap_deleted"] == 0


def test_monitor_publisher_event_timing_keeps_age_filter_after_age_unlink_error(
    tmp_path, monkeypatch, caplog
):
    publisher = _make_publisher(
        tmp_path, retain_days=0.0, max_total_bytes=1_000_000
    )
    candidate = publisher.events_dir / "candidate.ndjson"
    _write_retention_file(candidate, size=10, mtime_ms=1)
    clock = {"ns": 0}
    original_scandir = monitor_publisher_module.os.scandir

    def timed_scandir(path):
        clock["ns"] += 1_000
        return original_scandir(path)

    def fail_candidate_unlink(path, *args, **_kwargs):
        if path == candidate:
            clock["ns"] += 4_000
            raise OSError("age candidate unavailable")
        raise AssertionError(f"unexpected unlink: {path}")

    monkeypatch.setattr(monitor_publisher_module.time, "monotonic_ns", lambda: clock["ns"])
    monkeypatch.setattr(monitor_publisher_module.os, "scandir", timed_scandir)
    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)

    event, timing = publisher._record_event_timed(
        "test.event", ("test",), {"value": 1}, ts=100_000
    )

    assert event is not None
    assert candidate.exists()
    assert "retention pruning failed: age candidate unavailable" in caplog.text
    assert timing["retention_inventory_candidates"] == 1
    assert timing["retention_age_filter_ns_total"] == 4_000
    assert timing["retention_age_filter_ns_max"] == 4_000
    assert timing["retention_age_unlink_ns_total"] == 4_000
    assert timing["retention_age_unlink_ns_max"] == 4_000
    assert timing["retention_age_deleted"] == 0
    assert timing["retention_cap_prune_ns_total"] == 0
    assert timing["retention_cap_prune_ns_max"] == 0
    assert timing["retention_cap_unlink_ns_total"] == 0
    assert timing["retention_cap_unlink_ns_max"] == 0
    assert timing["retention_cap_deleted"] == 0


def test_monitor_publisher_event_timing_records_cap_prune_on_under_cap_return(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(
        tmp_path, retain_days=-1.0, max_total_bytes=1_000_000
    )
    candidate = publisher.events_dir / "candidate.ndjson"
    _write_retention_file(candidate, size=10, mtime_ms=1)
    phases = []

    publisher._prune_retention(
        now_ms=100_000,
        on_phase=lambda phase, duration_ns: phases.append((phase, duration_ns)),
    )

    assert candidate.exists()
    assert [phase for phase, _duration_ns in phases] == [
        "inventory",
        "age_filter",
        "cap_prune",
    ]
    assert phases[-1][1] >= 0


def _write_retention_file(path, *, size, mtime_ms):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime_ms * 1_000_000, mtime_ms * 1_000_000))


def test_monitor_retention_age_cutoff_orders_direct_candidates_and_protects_current_paths(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=1.0)
    now_ms = 200_000_000
    cutoff_ms = now_ms - 86_400_000
    oldest = publisher.events_dir / "oldest.ndjson"
    newer = publisher.checkpoints_dir / "newer.json"
    at_cutoff = publisher.history_dir / "at-cutoff.ndjson"
    protected_history = publisher._history_current_path("fills")
    _write_retention_file(oldest, size=1, mtime_ms=cutoff_ms - 2)
    _write_retention_file(newer, size=1, mtime_ms=cutoff_ms - 1)
    _write_retention_file(at_cutoff, size=1, mtime_ms=cutoff_ms)
    _write_retention_file(publisher.current_events_path, size=1, mtime_ms=cutoff_ms - 3)
    _write_retention_file(protected_history, size=1, mtime_ms=cutoff_ms - 3)
    unlinked = []
    original_unlink = Path.unlink

    def record_unlink(path, *args, **kwargs):
        unlinked.append(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    publisher._prune_retention(now_ms=now_ms)

    assert unlinked == [oldest, newer]
    assert not oldest.exists()
    assert not newer.exists()
    assert at_cutoff.exists()
    assert publisher.current_events_path.exists()
    assert protected_history.exists()


def test_monitor_retention_counts_nested_unknown_files_but_never_deletes_them(tmp_path):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    direct_candidate = publisher.events_dir / "rotated.ndjson"
    nested_unknown = publisher.root / "unknown" / "nested" / "keep.bin"
    _write_retention_file(direct_candidate, size=10, mtime_ms=1)
    _write_retention_file(nested_unknown, size=40, mtime_ms=2)
    total_bytes = sum(path.stat().st_size for path in publisher.root.rglob("*") if path.is_file())
    publisher.max_total_bytes = 0

    publisher._prune_retention(now_ms=100_000)

    assert not direct_candidate.exists()
    assert nested_unknown.exists()


def test_monitor_retention_inventory_tolerates_entry_disappearing_before_stat(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    disappeared = publisher.events_dir / "disappeared.ndjson"
    _write_retention_file(disappeared, size=10, mtime_ms=1)
    original_scandir = monitor_publisher_module.os.scandir
    original_unlink = Path.unlink
    entries_visited = []

    class DisappearedEntry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def stat(self, *_args, **_kwargs):
            original_unlink(disappeared)
            raise FileNotFoundError(disappeared)

    class WrappedScandir:
        def __init__(self, iterator):
            self._iterator = iterator

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            for entry in self._iterator:
                if Path(entry.path) == disappeared:
                    yield DisappearedEntry(entry)
                else:
                    yield entry

    def wrapped_scandir(path):
        return WrappedScandir(original_scandir(path))

    monkeypatch.setattr(monitor_publisher_module.os, "scandir", wrapped_scandir)
    total_bytes, candidates = publisher._retention_inventory(
        on_entry=lambda: entries_visited.append(None)
    )

    assert not disappeared.exists()
    assert total_bytes >= 0
    assert all(path != disappeared for path, _size, _mtime in candidates)
    assert len(entries_visited) > 0


def test_monitor_retention_inventory_tolerates_directory_disappearing_before_scan(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    vanished_dir = publisher.root / "unknown" / "vanished"
    nested = vanished_dir / "nested.bin"
    _write_retention_file(nested, size=10, mtime_ms=1)
    original_scandir = monitor_publisher_module.os.scandir
    vanished_scan_attempts = []

    def disappearing_scandir(path):
        if Path(path) == vanished_dir:
            vanished_scan_attempts.append(None)
            if nested.exists():
                nested.unlink()
                vanished_dir.rmdir()
            raise FileNotFoundError(vanished_dir)
        return original_scandir(path)

    monkeypatch.setattr(monitor_publisher_module.os, "scandir", disappearing_scandir)
    total_bytes, candidates = publisher._retention_inventory()

    assert not vanished_dir.exists()
    assert len(vanished_scan_attempts) == 2
    assert total_bytes >= 0
    assert all(path != nested for path, _size, _mtime in candidates)


def test_monitor_retention_inventory_retries_one_transient_directory_scan_error(
    tmp_path, monkeypatch
):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    candidate = publisher.events_dir / "candidate.ndjson"
    nested_unknown = publisher.root / "unknown" / "nested" / "keep.bin"
    _write_retention_file(candidate, size=11, mtime_ms=1)
    _write_retention_file(nested_unknown, size=13, mtime_ms=2)
    original_scandir = monitor_publisher_module.os.scandir
    root_scan_attempts = []

    def transient_scandir(path):
        if Path(path) == publisher.root:
            root_scan_attempts.append(None)
            if len(root_scan_attempts) == 1:
                raise OSError("transient scan failure")
        return original_scandir(path)

    monkeypatch.setattr(monitor_publisher_module.os, "scandir", transient_scandir)
    total_bytes, candidates = publisher._retention_inventory()

    assert len(root_scan_attempts) == 2
    assert total_bytes >= candidate.stat().st_size + nested_unknown.stat().st_size
    assert [path for path, _size, _mtime in candidates] == [candidate]


def test_monitor_retention_age_disappearance_updates_total_before_cap(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, retain_days=0.0)
    disappeared = publisher.events_dir / "disappeared.ndjson"
    at_cutoff = publisher.history_dir / "at-cutoff.ndjson"
    _write_retention_file(disappeared, size=10, mtime_ms=1)
    _write_retention_file(at_cutoff, size=10, mtime_ms=100_000)
    total_bytes = sum(path.stat().st_size for path in publisher.root.rglob("*") if path.is_file())
    publisher.max_total_bytes = total_bytes - disappeared.stat().st_size
    original_unlink = Path.unlink

    def disappear_then_unlink(path, *args, **kwargs):
        if path == disappeared:
            original_unlink(path, *args, **kwargs)
            raise FileNotFoundError(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", disappear_then_unlink)
    publisher._prune_retention(now_ms=100_000)

    assert not disappeared.exists()
    assert at_cutoff.exists()


def test_monitor_retention_logs_cap_unlink_failure_and_keeps_due_interval(
    tmp_path, monkeypatch, caplog
):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    candidate = publisher.events_dir / "rotated.ndjson"
    _write_retention_file(candidate, size=10, mtime_ms=1)
    publisher.max_total_bytes = 0
    inventory_attempts = []
    original_inventory = publisher._retention_inventory
    original_unlink = Path.unlink

    def count_inventory(**kwargs):
        inventory_attempts.append(None)
        return original_inventory(**kwargs)

    def fail_candidate_unlink(path, *args, **kwargs):
        if path == candidate:
            raise OSError("candidate unavailable")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(publisher, "_retention_inventory", count_inventory)
    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)
    publisher._prune_retention(now_ms=100_000)

    assert candidate.exists()
    assert inventory_attempts == [None]
    assert "retention pruning failed: candidate unavailable" in caplog.text

    publisher._prune_retention(now_ms=100_001)
    assert inventory_attempts == [None]


def test_monitor_retention_counts_and_unlinks_direct_file_symlink_only(tmp_path):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")
    file_link = publisher.events_dir / "linked.bin"
    file_link.symlink_to(outside_file)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.bin").write_bytes(b"nested")
    directory_link = publisher.root / "linked-dir"
    directory_link.symlink_to(outside_dir, target_is_directory=True)

    total_bytes, candidates = publisher._retention_inventory()
    candidate_sizes = {path: size for path, size, _mtime in candidates}
    assert candidate_sizes[file_link] == outside_file.stat().st_size
    publisher.max_total_bytes = total_bytes - candidate_sizes[file_link]

    publisher._prune_retention(now_ms=100_000)

    assert not file_link.is_symlink()
    assert outside_file.exists()
    assert directory_link.exists()
    assert (outside_dir / "nested.bin").exists()


def test_monitor_retention_scandir_inventory_matches_pathlib_semantics(tmp_path):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    direct_events = publisher.events_dir / "events.ndjson"
    direct_history = publisher.history_dir / "history.ndjson"
    nested_unknown = publisher.root / "unknown" / "nested" / "keep.bin"
    _write_retention_file(direct_events, size=11, mtime_ms=10)
    _write_retention_file(direct_history, size=13, mtime_ms=10)
    _write_retention_file(nested_unknown, size=17, mtime_ms=10)
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")
    file_link = publisher.checkpoints_dir / "linked.bin"
    file_link.symlink_to(outside_file)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "not-managed.bin").write_bytes(b"outside-dir")
    (publisher.root / "linked-dir").symlink_to(outside_dir, target_is_directory=True)
    protected = {
        publisher.manifest_path,
        publisher.state_latest_path,
        publisher.current_events_path,
        *publisher._current_history_paths.values(),
    }
    candidate_dirs = {
        publisher.events_dir,
        publisher.history_dir,
        publisher.checkpoints_dir,
    }
    expected_total = 0
    expected_candidates = []
    for path in publisher.root.rglob("*"):
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        expected_total += file_stat.st_size
        if path.parent in candidate_dirs and path not in protected:
            expected_candidates.append((path, file_stat.st_size, file_stat.st_mtime))
    expected_candidates.sort(key=lambda candidate: candidate[2])

    total_bytes, candidates = publisher._retention_inventory()

    assert total_bytes == expected_total
    assert candidates == expected_candidates


def test_monitor_retention_byte_cap_deletes_surviving_candidates_oldest_first(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    oldest = publisher.history_dir / "oldest.ndjson"
    newer = publisher.events_dir / "newer.ndjson"
    _write_retention_file(oldest, size=10, mtime_ms=1)
    _write_retention_file(newer, size=10, mtime_ms=2)
    total_bytes = sum(path.stat().st_size for path in publisher.root.rglob("*") if path.is_file())
    publisher.max_total_bytes = total_bytes - oldest.stat().st_size - newer.stat().st_size
    unlinked = []
    original_unlink = Path.unlink

    def record_unlink(path, *args, **kwargs):
        unlinked.append(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    publisher._prune_retention(now_ms=100_000)

    assert unlinked == [oldest, newer]
    assert not oldest.exists()
    assert not newer.exists()


def test_monitor_retention_uses_one_recursive_traversal_when_over_byte_cap(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path, retain_days=-1.0)
    candidate = publisher.events_dir / "rotated.ndjson"
    _write_retention_file(candidate, size=10, mtime_ms=1)
    total_bytes = sum(path.stat().st_size for path in publisher.root.rglob("*") if path.is_file())
    publisher.max_total_bytes = total_bytes - candidate.stat().st_size
    visited_paths = set(publisher.root.rglob("*"))
    expected_directories = {
        publisher.root,
        *(path for path in visited_paths if path.is_dir() and not path.is_symlink()),
    }
    original_scandir = monitor_publisher_module.os.scandir
    scanned_directories = []
    entry_stat_counts = {path: 0 for path in visited_paths}

    class CountingEntry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def stat(self, *args, **kwargs):
            entry_stat_counts[Path(self._entry.path)] += 1
            return self._entry.stat(*args, **kwargs)

    class CountingScandir:
        def __init__(self, iterator):
            self._iterator = iterator

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, *args):
            return self._iterator.__exit__(*args)

        def __iter__(self):
            return (CountingEntry(entry) for entry in self._iterator)

    def count_scandir(path):
        scanned_directories.append(Path(path))
        return CountingScandir(original_scandir(path))

    def fail_legacy_traversal(*_args, **_kwargs):
        raise AssertionError("legacy Path traversal used")

    monkeypatch.setattr(monitor_publisher_module.os, "scandir", count_scandir)
    monkeypatch.setattr(Path, "rglob", fail_legacy_traversal)
    monkeypatch.setattr(Path, "iterdir", fail_legacy_traversal)
    publisher._prune_retention(now_ms=100_000)

    assert set(scanned_directories) == expected_directories
    assert len(scanned_directories) == len(expected_directories)
    assert set(entry_stat_counts.values()) == {1}
    assert not candidate.exists()
