"""Strict monitor JSON must preserve the last usable data when a producer is invalid."""
import json
import logging
import sys

import numpy as np
import pytest

from test_monitor_publisher import _make_publisher


INVALID_NUMBERS = [
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(np.float32("inf"), id="numpy-scalar"),
    pytest.param(np.array([1.0, float("nan")]), id="numpy-array"),
]


def _strict_loads(text):
    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_nonfinite_snapshot_preserves_previous_snapshot_and_checkpoints(tmp_path, caplog, value):
    publisher = _make_publisher(tmp_path)
    assert publisher.write_snapshot({"nested": {"value": 1.0}}, ts=1000, force=True)
    previous = publisher.state_latest_path.read_bytes()
    manifest = publisher.manifest_path.read_bytes()
    checkpoints = {p.name: p.read_bytes() for p in publisher.checkpoints_dir.iterdir()}
    last_snapshot_ms = publisher.last_snapshot_ms
    last_checkpoint_ms = publisher.last_checkpoint_ms

    with caplog.at_level(logging.WARNING):
        assert not publisher.write_snapshot({"nested": {"value": value}}, ts=2000, force=True)

    assert publisher.state_latest_path.read_bytes() == previous
    assert publisher.manifest_path.read_bytes() == manifest
    assert {p.name: p.read_bytes() for p in publisher.checkpoints_dir.iterdir()} == checkpoints
    assert publisher.last_snapshot_ms == last_snapshot_ms
    assert publisher.last_checkpoint_ms == last_checkpoint_ms
    assert not list(tmp_path.rglob("*.tmp"))
    assert "writing snapshot" in caplog.text
    assert publisher.write_snapshot({"nested": {"value": 0.0}}, ts=3000, force=True)
    assert _strict_loads(publisher.state_latest_path.read_text())["nested"]["value"] == 0.0


@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_nonfinite_event_does_not_append_a_partial_row_and_next_event_recovers(tmp_path, caplog, value):
    publisher = _make_publisher(tmp_path)
    first = publisher.record_event("test.valid", ("test",), {"value": 1.0}, ts=1000)
    previous = publisher.current_events_path.read_bytes()
    seq = publisher.seq

    with caplog.at_level(logging.WARNING):
        assert publisher.record_event("test.invalid", ("test",), {"nested": [value]}, ts=1001) is None

    assert publisher.current_events_path.read_bytes() == previous
    assert publisher.seq == seq
    assert "recording event" in caplog.text
    next_event = publisher.record_event("test.recovered", ("test",), {"value": 0.0}, ts=1002)
    assert next_event["seq"] > first["seq"]
    publisher.close()
    rows = [_strict_loads(line) for line in publisher.current_events_path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["test.valid", "test.recovered"]
    restarted = _make_publisher(tmp_path)
    assert restarted.record_event("test.restart", ("test",), {}, ts=1003)["seq"] > next_event["seq"]


@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_nonfinite_history_does_not_append_a_partial_row_and_next_write_recovers(tmp_path, caplog, value):
    publisher = _make_publisher(tmp_path)
    assert publisher.record_fill({"price": 1.0}, ts=1000)
    path = publisher._history_current_path("fills")
    previous = path.read_bytes()

    with caplog.at_level(logging.WARNING):
        assert publisher.record_fill({"price": value}, ts=1001) is None

    assert path.read_bytes() == previous
    assert "recording history entry fills/fill" in caplog.text
    assert publisher.record_fill({"price": 2.0}, ts=1002)
    rows = [_strict_loads(line) for line in path.read_text().splitlines()]
    assert [r["payload"]["price"] for r in rows] == [1.0, 2.0]


def test_monitor_json_preserves_finite_numpy_values_and_null(tmp_path):
    publisher = _make_publisher(tmp_path)
    payload = {"scalar": np.float32(1.5), "array": np.array([0.0, -2.0]), "unavailable": None}
    assert publisher.write_snapshot(payload, ts=1000, force=True)
    assert publisher.record_event("test.finite", ("test",), payload, ts=1001)
    assert publisher.record_fill(payload, ts=1002)
    snapshot = _strict_loads(publisher.state_latest_path.read_text())
    assert snapshot["scalar"] == 1.5
    assert snapshot["array"] == [0.0, -2.0]
    assert snapshot["unavailable"] is None
    for path in (publisher.current_events_path, publisher._history_current_path("fills")):
        assert _strict_loads(path.read_text())["payload"] == {
            "scalar": 1.5, "array": [0.0, -2.0], "unavailable": None,
        }


@pytest.mark.parametrize("stream", ["events", "fills"])
def test_invalid_payload_does_not_rotate_an_existing_full_segment(tmp_path, stream):
    publisher = _make_publisher(tmp_path, event_rotation_mib=0.00001)
    if stream == "events":
        record = lambda payload: publisher.record_event("test.row", ("test",), payload, ts=1000)
    else:
        record = lambda payload: publisher.record_fill(payload, ts=1000)
    assert record({"blob": "x" * 4000})
    root = publisher.current_events_path.parent.parent
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert record({"value": float("inf")}) is None
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_uncertain_append_keeps_sequence_reserved(tmp_path, monkeypatch):
    publisher = _make_publisher(tmp_path)
    original_open = open

    class FailedClose:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self.file.__enter__()

        def __exit__(self, *args):
            self.file.__exit__(*args)
            raise OSError("synthetic close failure")

    def fail_append_close(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        return FailedClose(file) if path == publisher.current_events_path and mode == "a" else file

    with monkeypatch.context() as patch:
        patch.setattr("monitor_publisher.open", fail_append_close, raising=False)
        assert publisher.record_event("test.uncertain", ("test",), {}, ts=1000) is None
    persisted = _strict_loads(publisher.current_events_path.read_text())
    assert persisted["seq"] == publisher.seq
    recovered = publisher.record_event("test.recovered", ("test",), {}, ts=1001)
    assert recovered["seq"] > persisted["seq"]


def test_nonfinite_candle_does_not_advance_history_watermark_or_skip_later_rows(tmp_path):
    publisher = _make_publisher(tmp_path)
    first = {"ts": 60000, "c": 1.0}
    assert publisher.record_completed_candles("BTC/USDT:USDT", "1m", [first])
    bad = {"ts": 120000, "c": float("nan")}
    later = {"ts": 180000, "c": 3.0}
    assert publisher.record_completed_candles("BTC/USDT:USDT", "1m", [bad, later]) == []
    assert publisher._last_candle_ts_by_key[("BTC/USDT:USDT", "1m")] == 60000
    corrected = {"ts": 120000, "c": 2.0}
    recovered = publisher.record_completed_candles("BTC/USDT:USDT", "1m", [corrected, later])
    assert [r["payload"]["ts"] for r in recovered] == [120000, 180000]
    path = publisher._history_current_path("candles_1m")
    assert [_strict_loads(line)["payload"]["c"] for line in path.read_text().splitlines()] == [1.0, 2.0, 3.0]


def test_rejected_price_tick_does_not_throttle_corrected_same_timestamp(tmp_path):
    publisher = _make_publisher(tmp_path)
    assert publisher.record_price_tick("BTC/USDT:USDT", float("inf"), ts=1000) is None
    assert publisher.record_price_tick("BTC/USDT:USDT", 1.0, ts=1000)


@pytest.mark.parametrize("side", ["long", "short"])
def test_native_diagnostic_with_default_extrema_publishes_as_strict_json(tmp_path, side):
    import passivbot_rust
    from test_trailing_diagnostics import _sample_config, _sample_snapshot
    from trailing_diagnostics import build_trailing_diagnostic, build_trailing_inputs_from_snapshot

    assert not getattr(passivbot_rust, "__is_stub__", False)
    inputs = build_trailing_inputs_from_snapshot(
        _sample_config(), _sample_snapshot(), symbol="BTC/USDT:USDT", pside="long"
    )
    inputs.update(pside=side, position_size=abs(inputs["position_size"]) * (1 if side == "long" else -1),
                  min_since_open=sys.float_info.max, max_since_min=0.0,
                  max_since_open=0.0, min_since_max=sys.float_info.max)
    diagnostic = build_trailing_diagnostic(inputs)
    publisher = _make_publisher(tmp_path)
    assert publisher.write_snapshot({"trailing": {"BTC/USDT:USDT": {side: diagnostic}}}, ts=1000, force=True)
    snapshot = _strict_loads(publisher.state_latest_path.read_text())
    projected = snapshot["trailing"]["BTC/USDT:USDT"][side]
    assert projected["entry"]["retracement_price"] is None
    assert projected["entry"]["retracement_price_status"] == "unavailable"
    assert projected["entry"]["extrema"]["min_since_open"] == sys.float_info.max
