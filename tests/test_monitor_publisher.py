import errno
import json
import logging
import threading

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
    publisher = _make_publisher(tmp_path)

    event = publisher.record_event("bot.start", ("bot", "lifecycle"), {"status": "starting"}, ts=1000)
    assert event["seq"] == 1
    publisher.record_error("error.bot", RuntimeError("boom"), payload={"source": "test"}, ts=1001)
    publisher.close()

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
    monkeypatch.setattr(
        monitor_publisher_module.time,
        "monotonic_ns",
        lambda: clock["ns"],
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

    def fail_atomic_write(*_args, **_kwargs):
        raise OSError("manifest unavailable")

    def fail_rotatable_files():
        raise OSError("retention unavailable")

    monkeypatch.setattr(monitor_publisher_module, "_atomic_write_json", fail_atomic_write)
    monkeypatch.setattr(publisher, "_rotatable_files", fail_rotatable_files)

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
