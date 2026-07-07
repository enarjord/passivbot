import asyncio
import logging
from types import MethodType, SimpleNamespace

import pytest

from passivbot import Passivbot

pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
hsl = pytest.importorskip("passivbot_hsl", reason="live HSL dependencies not available")

if bool(getattr(pbr, "__is_stub__", False)):
    pytest.skip("passivbot_rust extension not available", allow_module_level=True)


class FakeHslBot(SimpleNamespace):
    pass


class FakeRiskCache:
    def __init__(self, covered_start_ms=1, history_scope="all"):
        self.covered_start_ms = covered_start_ms
        self.history_scope = history_scope

    def get_known_gaps(self):
        return []

    def get_covered_start_ms(self):
        return self.covered_start_ms

    def get_history_scope(self):
        return self.history_scope

    def load_metadata(self):
        return {
            "known_gaps": [],
            "covered_start_ms": self.covered_start_ms,
            "history_scope": self.history_scope,
            "oldest_event_ts": self.covered_start_ms,
            "newest_event_ts": 0,
        }


def make_fake_pnls_manager(events, *, covered_start_ms=1, history_scope="all"):
    cache = FakeRiskCache(covered_start_ms=covered_start_ms, history_scope=history_scope)
    return SimpleNamespace(
        get_events=lambda: events,
        cache=cache,
        get_history_scope=cache.get_history_scope,
    )


def test_hsl_signal_mode_requires_normalized_live_config():
    bot = FakeHslBot(config={"live": {"hsl_signal_mode": "coin"}})

    assert hsl._equity_hard_stop_signal_mode(bot) == "coin"

    bot.config = {"live": {}}
    with pytest.raises(KeyError, match="live.hsl_signal_mode"):
        hsl._equity_hard_stop_signal_mode(bot)


def test_parse_hsl_config_warns_about_history_reinterpretation(caplog):
    bot = FakeHslBot(config={"live": {"hsl_signal_mode": "coin"}})
    values = {
        "hsl_enabled": True,
        "hsl_red_threshold": 0.05,
        "hsl_ema_span_minutes": 120.0,
        "hsl_cooldown_minutes_after_red": 60.0,
        "hsl_no_restart_drawdown_threshold": 0.08,
        "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
        "hsl_tier_ratios.yellow": 0.5,
        "hsl_tier_ratios.orange": 0.75,
        "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
        "hsl_panic_close_order_type": "limit",
        "hsl_restart_after_red_policy": "threshold",
    }
    bot._hsl_psides = lambda: ["long"]
    bot._equity_hard_stop_signal_mode = MethodType(
        hsl._equity_hard_stop_signal_mode,
        bot,
    )
    bot.bot_value = lambda pside, key: values[key]

    with caplog.at_level(logging.WARNING):
        parsed = hsl._parse_hsl_config(bot)

    assert parsed["long"]["enabled"] is True
    messages = [record.getMessage() for record in caplog.records]
    assert any("docs/equity_hard_stop_loss_risks.md" in message for message in messages)
    assert any("Deposits, withdrawals" in message for message in messages)
    assert any("HSL mode changes" in message for message in messages)


def test_hsl_replay_matrix_row_derives_upnl_from_rust_pnl_helpers():
    long_row = hsl._hsl_replay_matrix_row(
        pside="long",
        ts=60_000,
        price=90.0,
        psize=2.0,
        pprice=100.0,
        pnl=-3.0,
        c_mult=1.0,
    )
    short_row = hsl._hsl_replay_matrix_row(
        pside="short",
        ts=60_000,
        price=110.0,
        psize=-2.0,
        pprice=100.0,
        pnl=4.0,
        c_mult=1.0,
    )

    assert set(long_row) == set(hsl._HSL_REPLAY_MATRIX_RAW_FIELDS)
    assert long_row["upnl"] == pytest.approx(pbr.calc_pnl_long(100.0, 90.0, 2.0, 1.0))
    assert short_row["upnl"] == pytest.approx(pbr.calc_pnl_short(100.0, 110.0, 2.0, 1.0))
    assert long_row["pnl"] == pytest.approx(-3.0)
    assert short_row["pnl"] == pytest.approx(4.0)


def test_hsl_replay_matrix_derived_series_keeps_persisted_pnl_raw():
    rows = [
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=60_000,
            price=100.0,
            psize=1.0,
            pprice=100.0,
            pnl=5.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=120_000,
            price=90.0,
            psize=1.0,
            pprice=100.0,
            pnl=-2.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=180_000,
            price=105.0,
            psize=0.0,
            pprice=0.0,
            pnl=7.0,
            c_mult=1.0,
        ),
    ]

    derived = hsl._hsl_replay_matrix_derived_series(rows, base_equity=1_000.0)

    assert [row["pnl"] for row in rows] == [5.0, -2.0, 7.0]
    assert [row["pnl_cumsum"] for row in derived] == pytest.approx([5.0, 3.0, 10.0])
    assert [row["equity"] for row in derived] == pytest.approx([1005.0, 993.0, 1010.0])


def test_hsl_replay_matrix_derived_series_requires_contiguous_minutes():
    rows = [
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=60_000,
            price=100.0,
            psize=0.0,
            pprice=0.0,
            pnl=0.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=180_000,
            price=100.0,
            psize=0.0,
            pprice=0.0,
            pnl=0.0,
            c_mult=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="contiguous 1m samples"):
        hsl._hsl_replay_matrix_derived_series(rows, base_equity=1_000.0)


def _hsl_cache_metadata(**overrides):
    metadata = {
        "exchange": "binance",
        "market_type": "swap",
        "user": "test_user",
        "config_digest": "cfg_hash",
        "signal_mode": "coin",
        "pside": "long",
        "symbol": "BTC/USDT:USDT",
        "fill_covered_start_ms": 60_000,
        "fill_covered_end_ms": 180_000,
        "fill_history_scope": "window",
        "fill_coverage_proven": True,
        "candle_covered_start_ms": 60_000,
        "candle_covered_end_ms": 180_000,
    }
    metadata.update(overrides)
    return metadata


def _hsl_cache_rows():
    return [
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=60_000,
            price=100.0,
            psize=1.0,
            pprice=100.0,
            pnl=0.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=120_000,
            price=90.0,
            psize=1.0,
            pprice=100.0,
            pnl=-2.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=180_000,
            price=95.0,
            psize=0.0,
            pprice=0.0,
            pnl=3.0,
            c_mult=1.0,
        ),
    ]


def test_hsl_replay_matrix_cache_round_trips_with_manifest(tmp_path):
    metadata = _hsl_cache_metadata()

    manifest = hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)

    assert manifest["schema_version"] == hsl._HSL_REPLAY_CACHE_SCHEMA_VERSION
    assert manifest["matrix_file"] == hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    assert manifest["row_count"] == 3
    assert manifest["metadata"] == metadata
    assert hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    ) == []


def test_hsl_replay_matrix_derived_arrays_matches_row_series():
    import numpy as np

    rows = _hsl_cache_rows()
    arrays = hsl._hsl_replay_matrix_arrays(rows)
    derived_arrays = hsl._hsl_replay_matrix_derived_arrays(arrays, base_equity=1_000.0)
    derived_rows = hsl._hsl_replay_matrix_derived_series(rows, base_equity=1_000.0)

    np.testing.assert_array_equal(
        derived_arrays["ts"],
        np.array([row["ts"] for row in derived_rows], dtype=np.int64),
    )
    np.testing.assert_allclose(
        derived_arrays["pnl_cumsum"],
        np.array([row["pnl_cumsum"] for row in derived_rows]),
    )
    np.testing.assert_allclose(
        derived_arrays["upnl"],
        np.array([row["upnl"] for row in derived_rows]),
    )
    np.testing.assert_allclose(
        derived_arrays["equity"],
        np.array([row["equity"] for row in derived_rows]),
    )


def test_hsl_replay_matrix_derived_arrays_rejects_missing_field():
    arrays = hsl._hsl_replay_matrix_arrays(_hsl_cache_rows())
    arrays.pop("upnl")

    with pytest.raises(ValueError, match="missing=\\['upnl'\\]"):
        hsl._hsl_replay_matrix_derived_arrays(arrays, base_equity=1_000.0)


def test_hsl_replay_matrix_derived_arrays_rejects_timestamp_gap():
    arrays = hsl._hsl_replay_matrix_arrays(_hsl_cache_rows())
    arrays["ts"][1] += hsl._HSL_REPLAY_MATRIX_INTERVAL_MS

    with pytest.raises(ValueError, match="timestamp_not_contiguous"):
        hsl._hsl_replay_matrix_derived_arrays(arrays, base_equity=1_000.0)


def test_hsl_replay_matrix_cache_reports_metadata_mismatch(tmp_path):
    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=_hsl_cache_metadata(config_digest="other_cfg_hash"),
    )

    assert reasons == ["metadata_mismatch:config_digest"]


def test_hsl_replay_matrix_cache_load_returns_manifest_and_arrays(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata()
    written_manifest = hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)

    manifest, arrays = hsl._load_hsl_replay_matrix_cache(
        tmp_path,
        expected_metadata=metadata,
    )

    assert manifest == written_manifest
    assert set(arrays) == set(hsl._HSL_REPLAY_MATRIX_RAW_FIELDS)
    np.testing.assert_array_equal(arrays["ts"], np.array([60_000, 120_000, 180_000]))
    np.testing.assert_allclose(arrays["pnl"], np.array([0.0, -2.0, 3.0]))


def test_hsl_replay_matrix_cache_load_rejects_metadata_mismatch(tmp_path):
    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)

    with pytest.raises(ValueError, match="metadata_mismatch:config_digest"):
        hsl._load_hsl_replay_matrix_cache(
            tmp_path,
            expected_metadata=_hsl_cache_metadata(config_digest="other_cfg_hash"),
        )


def test_hsl_replay_matrix_cache_try_load_emits_hit_event(tmp_path):
    import numpy as np
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    metadata = _hsl_cache_metadata()
    written_manifest = hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    bot = FakeHslBot()
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_cache"
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    loaded = hsl._try_load_hsl_replay_matrix_cache(
        bot,
        tmp_path,
        expected_metadata=metadata,
        pside="long",
        symbol=metadata["symbol"],
    )

    assert loaded is not None
    manifest, arrays = loaded
    assert manifest == written_manifest
    np.testing.assert_array_equal(arrays["ts"], np.array([60_000, 120_000, 180_000]))
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_CACHE]
    assert len(events) == 1
    event = events[0]
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_HIT
    assert event.cycle_id == "cy_hsl_cache"
    assert event.pside == "long"
    assert event.symbol == metadata["symbol"]
    assert event.data["cache_status"] == "hit"
    assert event.data["row_count"] == 3
    assert event.data["start_ts_ms"] == 60_000
    assert event.data["end_ts_ms"] == 180_000
    assert event.data["reason_count"] == 0
    assert "cache_dir" not in event.data
    assert "path" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_matrix_cache_try_load_emits_miss_event(tmp_path):
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    metadata = _hsl_cache_metadata()
    bot = FakeHslBot()
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    loaded = hsl._try_load_hsl_replay_matrix_cache(
        bot,
        tmp_path,
        expected_metadata=metadata,
        pside="long",
        symbol=metadata["symbol"],
    )

    assert loaded is None
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_CACHE]
    assert len(events) == 1
    event = events[0]
    assert event.status == "skipped"
    assert event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_MISS
    assert event.data["cache_status"] == "miss"
    assert event.data["reasons"] == ["manifest_missing"]
    assert event.data["reason_count"] == 1
    assert "cache_dir" not in event.data
    assert "path" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_matrix_cache_try_load_emits_rejected_event(tmp_path):
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    bot = FakeHslBot()
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    loaded = hsl._try_load_hsl_replay_matrix_cache(
        bot,
        tmp_path,
        expected_metadata=_hsl_cache_metadata(config_digest="other_cfg_hash"),
        pside="long",
        symbol=metadata["symbol"],
    )

    assert loaded is None
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_CACHE]
    assert len(events) == 1
    event = events[0]
    assert event.status == "skipped"
    assert event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_REJECTED
    assert event.data["cache_status"] == "rejected"
    assert event.data["reasons"] == ["metadata_mismatch:config_digest"]
    assert event.data["reason_count"] == 1
    assert "other_cfg_hash" not in str(event.data)
    assert "cache_dir" not in event.data
    assert "path" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_cache_config_digest_is_stable_and_hsl_scoped():
    bot = make_coin_bot()
    digest = hsl._hsl_replay_cache_config_digest(bot, "long")

    bot.config["irrelevant"] = {"changes": "do_not_affect_hsl_replay_cache"}
    assert hsl._hsl_replay_cache_config_digest(bot, "long") == digest

    bot.hsl["long"]["red_threshold"] = 0.25
    assert hsl._hsl_replay_cache_config_digest(bot, "long") != digest

    bot = make_coin_bot()
    original_bot_value = bot.bot_value

    def bot_value(pside, key):
        if key == "n_positions":
            return 3
        return original_bot_value(pside, key)

    bot.bot_value = bot_value
    assert hsl._hsl_replay_cache_config_digest(bot, "long") != digest


def test_hsl_replay_cache_expected_metadata_uses_trust_boundary_fields():
    bot = make_coin_bot()
    bot.market_type = "swap"

    metadata = hsl._hsl_replay_cache_expected_metadata(
        bot,
        "long",
        "BTC/USDT:USDT",
        fill_covered_start_ms=60_000,
        fill_covered_end_ms=120_000,
        fill_history_scope="window",
        fill_coverage_proven=True,
        candle_covered_start_ms=60_000,
        candle_covered_end_ms=180_000,
    )

    assert set(metadata) == set(hsl._HSL_REPLAY_CACHE_REQUIRED_METADATA)
    assert metadata["exchange"] == "test_exchange"
    assert metadata["market_type"] == "swap"
    assert metadata["user"] == "test_user"
    assert metadata["signal_mode"] == "coin"
    assert metadata["pside"] == "long"
    assert metadata["symbol"] == "BTC/USDT:USDT"
    assert metadata["config_digest"] == hsl._hsl_replay_cache_config_digest(bot, "long")
    assert len(metadata["config_digest"]) == 64
    assert metadata["fill_covered_start_ms"] == 60_000
    assert metadata["fill_covered_end_ms"] == 120_000
    assert metadata["fill_history_scope"] == "window"
    assert metadata["fill_coverage_proven"] is True
    assert metadata["candle_covered_start_ms"] == 60_000
    assert metadata["candle_covered_end_ms"] == 180_000


def test_hsl_replay_cache_metadata_rejects_invalid_coverage_proof_fields():
    with pytest.raises(ValueError, match="fill_history_scope"):
        hsl._normalize_hsl_replay_cache_metadata(
            _hsl_cache_metadata(fill_history_scope="everything")
        )
    with pytest.raises(ValueError, match="fill_coverage_proven"):
        hsl._normalize_hsl_replay_cache_metadata(
            _hsl_cache_metadata(fill_coverage_proven=1)
        )
    # Unproven coverage is valid metadata; the future read slice gates on it.
    normalized = hsl._normalize_hsl_replay_cache_metadata(
        _hsl_cache_metadata(fill_coverage_proven=False)
    )
    assert normalized["fill_coverage_proven"] is False


def test_hsl_replay_cache_validation_flags_missing_coverage_proof_fields(tmp_path):
    import json

    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), _hsl_cache_metadata())
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["metadata"]["fill_history_scope"]
    del manifest["metadata"]["fill_coverage_proven"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reasons = hsl._hsl_replay_cache_validation_reasons(tmp_path)

    assert "metadata_missing_required:fill_history_scope" in reasons
    assert "metadata_missing_required:fill_coverage_proven" in reasons


def _hsl_account_series_rows():
    return [
        hsl._hsl_replay_account_series_row(ts=60_000, pnl=0.0),
        hsl._hsl_replay_account_series_row(ts=120_000, pnl=-2.5),
        hsl._hsl_replay_account_series_row(ts=180_000, pnl=1.0),
    ]


def test_hsl_replay_account_series_round_trip(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata(
        pside=hsl._HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
        symbol=hsl._HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
    )
    rows = _hsl_account_series_rows()
    manifest = hsl._write_hsl_replay_matrix_cache(
        tmp_path, rows, metadata, series_kind="account_pnl"
    )

    assert manifest["series_kind"] == "account_pnl"
    assert manifest["fields"] == ["ts", "pnl"]
    assert (
        hsl._hsl_replay_cache_validation_reasons(tmp_path, expected_metadata=metadata)
        == []
    )
    loaded_manifest, arrays = hsl._load_hsl_replay_matrix_cache(
        tmp_path, expected_metadata=metadata
    )
    assert set(arrays) == {"ts", "pnl"}
    np.testing.assert_array_equal(arrays["ts"], np.array([60_000, 120_000, 180_000]))
    np.testing.assert_allclose(arrays["pnl"], [0.0, -2.5, 1.0])

    with pytest.raises(ValueError, match="contiguous"):
        hsl._hsl_replay_account_series_arrays(
            [
                hsl._hsl_replay_account_series_row(ts=60_000, pnl=0.0),
                hsl._hsl_replay_account_series_row(ts=240_000, pnl=1.0),
            ]
        )


def test_hsl_replay_account_series_persists_panic_markers(tmp_path):
    import json

    metadata = _hsl_cache_metadata(
        pside=hsl._HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
        symbol=hsl._HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
    )
    markers = [
        {
            "timestamp": 121_500,
            "minute_timestamp": 120_000,
            "pside": "long",
            "symbol": "BTC/USDT:USDT",
        }
    ]
    manifest = hsl._write_hsl_replay_matrix_cache(
        tmp_path,
        _hsl_account_series_rows(),
        metadata,
        series_kind="account_pnl",
        panic_flatten_events=markers,
    )

    assert manifest["panic_flatten_events"] == markers
    assert (
        hsl._hsl_replay_cache_validation_reasons(tmp_path, expected_metadata=metadata)
        == []
    )
    loaded_manifest, _arrays = hsl._load_hsl_replay_matrix_cache(
        tmp_path, expected_metadata=metadata
    )
    assert loaded_manifest["panic_flatten_events"] == markers

    # Tampering detection on the persisted manifest.
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stripped = dict(stored)
    del stripped["panic_flatten_events"]
    manifest_path.write_text(json.dumps(stripped), encoding="utf-8")
    assert "panic_events_missing" in hsl._hsl_replay_cache_validation_reasons(tmp_path)

    corrupted = dict(stored, panic_flatten_events=[{"timestamp": 1}])
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    assert "panic_events_invalid" in hsl._hsl_replay_cache_validation_reasons(tmp_path)


def test_hsl_replay_cache_panic_markers_fail_loud_contracts(tmp_path):
    import json

    marker = {
        "timestamp": 121_500,
        "minute_timestamp": 120_000,
        "pside": "long",
        "symbol": "BTC/USDT:USDT",
    }
    account_metadata = _hsl_cache_metadata(
        pside=hsl._HSL_REPLAY_CACHE_ACCOUNT_PSIDE,
        symbol=hsl._HSL_REPLAY_CACHE_ACCOUNT_SYMBOL,
    )
    # Markers are account-scoped; pair manifests must reject them.
    with pytest.raises(ValueError, match="account-scoped"):
        hsl._write_hsl_replay_matrix_cache(
            tmp_path,
            _hsl_cache_rows(),
            _hsl_cache_metadata(),
            panic_flatten_events=[marker],
        )
    # Off-grid minute rejects.
    with pytest.raises(ValueError, match="not minute-aligned"):
        hsl._write_hsl_replay_matrix_cache(
            tmp_path,
            _hsl_account_series_rows(),
            account_metadata,
            series_kind="account_pnl",
            panic_flatten_events=[dict(marker, minute_timestamp=90_000)],
        )
    # Outside the series span rejects.
    with pytest.raises(ValueError, match="outside the series span"):
        hsl._write_hsl_replay_matrix_cache(
            tmp_path,
            _hsl_account_series_rows(),
            account_metadata,
            series_kind="account_pnl",
            panic_flatten_events=[dict(marker, minute_timestamp=600_000)],
        )
    # A pair manifest carrying markers is rejected by the validator.
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), _hsl_cache_metadata())
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored["panic_flatten_events"] = [marker]
    manifest_path.write_text(json.dumps(stored), encoding="utf-8")
    assert "panic_events_wrong_kind" in hsl._hsl_replay_cache_validation_reasons(tmp_path)


def test_hsl_replay_cache_rejects_series_kind_tampering(tmp_path):
    import json

    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), _hsl_cache_metadata())
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tampered = dict(manifest, series_kind="account_pnl")
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert "fields_mismatch" in hsl._hsl_replay_cache_validation_reasons(tmp_path)

    stripped = dict(manifest)
    del stripped["series_kind"]
    manifest_path.write_text(json.dumps(stripped), encoding="utf-8")
    assert "series_kind_invalid" in hsl._hsl_replay_cache_validation_reasons(tmp_path)


def test_hsl_replay_cache_account_digest_scoped_to_lookback():
    bot = make_coin_bot()
    digest = hsl._hsl_replay_cache_account_config_digest(bot)
    assert len(digest) == 64

    bot.config["irrelevant"] = {"changes": "ignored"}
    bot.hsl["long"]["red_threshold"] = 0.33
    assert hsl._hsl_replay_cache_account_config_digest(bot) == digest

    bot.config["live"]["pnls_max_lookback_days"] = 7.0
    assert hsl._hsl_replay_cache_account_config_digest(bot) != digest


def test_coin_panic_supervision_requires_red_active_now():
    # B2.1 red split: a latched red episode authorizes panic supervision only
    # while the CURRENT sample is in RED.
    bot = make_coin_bot()
    symbol = "A"
    state = bot._hsl_coin_state("long", symbol)
    state["runtime"].apply_sample(
        timestamp_ms=60_000, equity=100.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    state["runtime"].apply_sample(
        timestamp_ms=120_000, equity=70.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    assert state["runtime"].red_latched() is True

    # No metrics yet against the latched state: stay protective.
    state["last_metrics"] = None
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )
    # Current sample in RED: panic authorized.
    state["last_metrics"] = {"red_active_now": True}
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )
    # Current sample recovered: no new panic orders for this episode.
    state["last_metrics"] = {"red_active_now": False}
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is False
    )
    # Halted repanic-reset supervision is unaffected by the split.
    state["halted"] = True
    state["cooldown_repanic_reset_pending"] = True
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is True
    )


@pytest.mark.asyncio
async def test_recovered_red_episode_finalizes_from_check_path():
    # Codex blocker regression on the red split: RED latches, the sample
    # recovers while the position is open (panic pauses, tp-only holds), the
    # position later flattens normally, and the episode MUST still be
    # finalized (halt + cooldown) by the regular check path without RED ever
    # re-activating and without the panic supervisor running.
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }
    bot.open_orders = {}
    bot.active_symbols = [symbol]
    upnl_box = {"value": 0.0}

    async def calc_upnl(pside=None, sym=None):
        return upnl_box["value"]

    bot._calc_upnl_sum_strict = calc_upnl
    now_box = {"ts": 60_000}
    bot.get_exchange_time = lambda: now_box["ts"]

    # Warmup green sample, then crash through RED (slot budget 50, red 0.5).
    await bot._equity_hard_stop_check_coin()
    now_box["ts"] = 120_000
    upnl_box["value"] = -30.0
    out = await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["tier"] == "red"
    assert state["runtime"].red_latched() is True
    assert bot._runtime_forced_modes["long"][symbol] == "panic"

    # Recovery while the position is still open: panic pauses, tp-only holds,
    # and the panic supervisor is NOT needed.
    now_box["ts"] = 180_000
    upnl_box["value"] = 0.0
    out = await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["red_active_now"] is False
    assert (
        bot._runtime_forced_modes["long"][symbol]
        == "tp_only_with_active_entry_cancellation"
    )
    assert (
        bot._equity_hard_stop_coin_needs_panic_supervision("long", symbol, state)
        is False
    )
    assert state["halted"] is False

    # The position closes normally under tp-only; two check cycles confirm
    # flat and finalize the episode: halt + cooldown, no RED re-activation.
    bot.positions = {
        symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0}}
    }
    bot.active_symbols = []
    now_box["ts"] = 240_000
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert state["red_flat_confirmations"] == 1
    assert state["halted"] is False
    now_box["ts"] = 300_000
    await bot._equity_hard_stop_check_coin()
    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] is not None


def test_forced_mode_refresher_preserves_paused_red(monkeypatch):
    # Hermes finding on the red split: the centralized refresher used to
    # overwrite the paused tp-only modes back to panic for any latched
    # non-halted pside. It must derive panic vs paused from the latest
    # sample's red_active_now, staying protective when no sample exists.
    bot = make_coin_bot()
    bot.positions = {"A": {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.active_symbols = ["A"]
    state = bot._hsl_state("long")
    state["runtime"].apply_sample(
        timestamp_ms=60_000, equity=100.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    state["runtime"].apply_sample(
        timestamp_ms=120_000, equity=70.0, peak_strategy_equity=100.0,
        red_threshold=0.2, ema_span_minutes=1.0,
        tier_ratio_yellow=0.5, tier_ratio_orange=0.75, latch_red=True,
    )
    assert state["runtime"].red_latched() is True

    # No sample recorded: protective panic.
    state["last_metrics"] = None
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert bot._runtime_forced_modes["long"]["A"] == "panic"

    # Active sample: panic.
    state["last_metrics"] = {"red_active_now": True}
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert bot._runtime_forced_modes["long"]["A"] == "panic"

    # Recovered sample: the paused tp-only modes survive the refresher.
    state["last_metrics"] = {"red_active_now": False}
    bot._equity_hard_stop_set_red_paused_runtime_forced_modes("long")
    bot._equity_hard_stop_refresh_halted_runtime_forced_modes()
    assert (
        bot._runtime_forced_modes["long"]["A"]
        == "tp_only_with_active_entry_cancellation"
    )


def test_cooldown_anchor_uses_scope_flattening_fill():
    # B2.1: cooldown anchors at the fill that flattened the scope, by any
    # means. A manual close after the last panic fill must win; with no fills
    # in the window the panic-fill fallback and then the caller fallback hold.
    bot = make_coin_bot()
    events = [
        SimpleNamespace(
            position_side="long", symbol="A", timestamp=120_500,
            pb_order_type="close_panic_long",
        ),
        SimpleNamespace(
            position_side="long", symbol="A", timestamp=150_000,
            pb_order_type="close_manual_long",
        ),
    ]
    bot._pnls_manager = SimpleNamespace(get_events=lambda: events)
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_ms(
            "long", symbol="A", since_ms=60_000, fallback_ms=999_000
        )
        == 150_000
    )
    # Window that excludes both fills: caller fallback.
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_ms(
            "long", symbol="A", since_ms=200_000, fallback_ms=999_000
        )
        == 999_000
    )
    # Other-pair fills never leak into the anchor.
    assert (
        bot._equity_hard_stop_latest_flatten_fill_timestamp_ms(
            "long", symbol="B", since_ms=60_000, fallback_ms=999_000
        )
        == 999_000
    )


def test_red_paused_forced_modes_block_entries_without_panic():
    bot = make_coin_bot()
    bot.positions = {"A": {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.open_orders = {"B": []}
    bot.active_symbols = ["C"]

    bot._equity_hard_stop_set_red_paused_runtime_forced_modes("long")

    forced = bot._runtime_forced_modes["long"]
    assert set(forced) == {"A", "B", "C"}
    assert set(forced.values()) == {"tp_only_with_active_entry_cancellation"}


def test_hsl_replay_cache_dir_is_sanitized_and_digest_scoped():
    bot = make_coin_bot()
    bot.exchange = "binance/usdm"
    bot.user = "user:one"
    digest = "abcdef0123456789fedcba9876543210abcdef0123456789fedcba9876543210"

    path = hsl._hsl_replay_cache_dir(bot, "long", "BTC/USDT:USDT", config_digest=digest)

    assert "binance_usdm" in path
    assert "user_one" in path
    assert "BTC_USDT_USDT" in path
    assert "abcdef0123456789" in path
    assert "BTC/USDT:USDT" not in path
    assert "user:one" not in path


def _make_persist_bot(tmp_path, monkeypatch):
    from live.event_bus import ListEventSink, LiveEventPipeline

    def fake_get_filepath(rel):
        path = tmp_path / rel
        path.mkdir(parents=True, exist_ok=True)
        return f"{path}/"

    monkeypatch.setattr(hsl, "make_get_filepath", fake_get_filepath)
    bot = make_coin_bot()
    bot.market_type = "swap"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_cache_write"
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    return bot, sink


def test_hsl_replay_cache_persist_matrices_round_trip(tmp_path, monkeypatch):
    import numpy as np
    from live.event_bus import EventTypes, ReasonCodes

    bot, sink = _make_persist_bot(tmp_path, monkeypatch)
    symbol = "BTC/USDT:USDT"
    rows = _hsl_cache_rows()
    coverage = {
        "fill_covered_start_ms": 60_000,
        "fill_covered_end_ms": 200_000,
        "fill_history_scope": "window",
        "fill_coverage_proven": True,
        "candle_covered_start_ms": 60_000,
        "candle_covered_end_ms": 180_000,
    }
    account_rows = _hsl_account_series_rows()
    panic_markers = [
        {
            "timestamp": 121_500,
            "minute_timestamp": 120_000,
            "pside": "long",
            "symbol": symbol,
        }
    ]
    history = {
        "hsl_replay_matrices": {"long": {symbol: rows}},
        "hsl_replay_account_series": account_rows,
        "panic_flatten_events": panic_markers,
        "hsl_replay_matrix_coverage": coverage,
    }

    written = hsl._equity_hard_stop_persist_replay_matrices(bot, history)

    assert written == 2
    expected_metadata = hsl._hsl_replay_cache_expected_metadata(
        bot,
        "long",
        symbol,
        fill_covered_start_ms=coverage["fill_covered_start_ms"],
        fill_covered_end_ms=coverage["fill_covered_end_ms"],
        fill_history_scope=coverage["fill_history_scope"],
        fill_coverage_proven=coverage["fill_coverage_proven"],
        candle_covered_start_ms=coverage["candle_covered_start_ms"],
        candle_covered_end_ms=coverage["candle_covered_end_ms"],
    )
    cache_dir = hsl._hsl_replay_cache_dir(bot, "long", symbol)
    assert (
        hsl._hsl_replay_cache_validation_reasons(
            cache_dir, expected_metadata=expected_metadata
        )
        == []
    )
    loaded = hsl._try_load_hsl_replay_matrix_cache(
        bot,
        cache_dir,
        expected_metadata=expected_metadata,
        pside="long",
        symbol=symbol,
    )
    assert loaded is not None
    manifest, arrays = loaded
    assert manifest["row_count"] == len(rows)
    np.testing.assert_array_equal(
        arrays["ts"], np.array([row["ts"] for row in rows], dtype=np.int64)
    )
    np.testing.assert_allclose(arrays["pnl"], [row["pnl"] for row in rows])
    np.testing.assert_allclose(arrays["upnl"], [row["upnl"] for row in rows])

    account_expected = hsl._hsl_replay_cache_account_expected_metadata(
        bot,
        fill_covered_start_ms=coverage["fill_covered_start_ms"],
        fill_covered_end_ms=coverage["fill_covered_end_ms"],
        fill_history_scope=coverage["fill_history_scope"],
        fill_coverage_proven=coverage["fill_coverage_proven"],
        candle_covered_start_ms=coverage["candle_covered_start_ms"],
        candle_covered_end_ms=coverage["candle_covered_end_ms"],
    )
    account_dir = hsl._hsl_replay_cache_account_series_dir(bot)
    assert (
        hsl._hsl_replay_cache_validation_reasons(
            account_dir, expected_metadata=account_expected
        )
        == []
    )
    account_manifest, account_arrays = hsl._load_hsl_replay_matrix_cache(
        account_dir, expected_metadata=account_expected
    )
    assert account_manifest["series_kind"] == "account_pnl"
    assert account_manifest["panic_flatten_events"] == panic_markers
    np.testing.assert_allclose(
        account_arrays["pnl"], [row["pnl"] for row in account_rows]
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    cache_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_CACHE
    ]
    written_events = [
        event
        for event in cache_events
        if event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_WRITTEN
    ]
    assert len(written_events) == 2
    assert [(event.pside, event.symbol) for event in written_events] == [
        ("long", symbol),
        (hsl._HSL_REPLAY_CACHE_ACCOUNT_PSIDE, hsl._HSL_REPLAY_CACHE_ACCOUNT_SYMBOL),
    ]
    event = written_events[0]
    assert event.status == "succeeded"
    assert event.pside == "long"
    assert event.symbol == symbol
    assert event.data["cache_status"] == "written"
    assert event.data["row_count"] == len(rows)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_cache_persist_matrices_write_failure_is_nonfatal(
    tmp_path, monkeypatch, caplog
):
    import logging as logging_module

    from live.event_bus import EventTypes, ReasonCodes

    bot, sink = _make_persist_bot(tmp_path, monkeypatch)
    symbol = "BTC/USDT:USDT"
    rows = _hsl_cache_rows()
    # Break 1m continuity so the fail-loud writer rejects the rows.
    rows[-1] = dict(rows[-1], ts=rows[-1]["ts"] + 60_000)
    history = {
        "hsl_replay_matrices": {"long": {symbol: rows}},
        "hsl_replay_matrix_coverage": {
            "fill_covered_start_ms": 60_000,
            "fill_covered_end_ms": 200_000,
            "fill_history_scope": "window",
            "fill_coverage_proven": True,
            "candle_covered_start_ms": 60_000,
            "candle_covered_end_ms": 180_000,
        },
    }

    with caplog.at_level(logging_module.WARNING):
        written = hsl._equity_hard_stop_persist_replay_matrices(bot, history)

    assert written == 0
    assert "replay cache write failed" in caplog.text
    cache_dir = hsl._hsl_replay_cache_dir(bot, "long", symbol)
    assert hsl._hsl_replay_cache_validation_reasons(cache_dir) == ["manifest_missing"]

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    failed_events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_REPLAY_CACHE
        and event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_WRITE_FAILED
    ]
    assert len(failed_events) == 1
    event = failed_events[0]
    assert event.status == "failed"
    assert event.pside == "long"
    assert event.symbol == symbol
    assert event.data["cache_status"] == "write_failed"
    assert event.data["reasons"] == ["write_exception:ValueError"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_cache_writer_rejects_non_bool_coverage_proof(tmp_path, monkeypatch):
    from live.event_bus import EventTypes, ReasonCodes

    bot, sink = _make_persist_bot(tmp_path, monkeypatch)
    symbol = "BTC/USDT:USDT"

    with pytest.raises(ValueError, match="fill_coverage_proven"):
        hsl._hsl_replay_cache_expected_metadata(
            bot,
            "long",
            symbol,
            fill_covered_start_ms=60_000,
            fill_covered_end_ms=200_000,
            fill_history_scope="window",
            fill_coverage_proven=1,
            candle_covered_start_ms=60_000,
            candle_covered_end_ms=180_000,
        )

    history = {
        "hsl_replay_matrices": {"long": {symbol: _hsl_cache_rows()}},
        "hsl_replay_matrix_coverage": {
            "fill_covered_start_ms": 60_000,
            "fill_covered_end_ms": 200_000,
            "fill_history_scope": "window",
            "fill_coverage_proven": 1,
            "candle_covered_start_ms": 60_000,
            "candle_covered_end_ms": 180_000,
        },
    }

    written = hsl._equity_hard_stop_persist_replay_matrices(bot, history)

    assert written == 0
    cache_dir = hsl._hsl_replay_cache_dir(bot, "long", symbol)
    assert hsl._hsl_replay_cache_validation_reasons(cache_dir) == ["manifest_missing"]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    failed_events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_REPLAY_CACHE
        and event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_WRITE_FAILED
    ]
    assert len(failed_events) == 1
    assert failed_events[0].data["reasons"] == ["write_exception:ValueError"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_cache_persist_skips_account_series_without_pair_writes(
    tmp_path, monkeypatch
):
    bot, sink = _make_persist_bot(tmp_path, monkeypatch)
    history = {
        "hsl_replay_matrices": {},
        "hsl_replay_account_series": _hsl_account_series_rows(),
        "hsl_replay_matrix_coverage": {
            "fill_covered_start_ms": 60_000,
            "fill_covered_end_ms": 200_000,
            "fill_history_scope": "window",
            "fill_coverage_proven": True,
            "candle_covered_start_ms": 60_000,
            "candle_covered_end_ms": 180_000,
        },
    }

    written = hsl._equity_hard_stop_persist_replay_matrices(bot, history)

    assert written == 0
    account_dir = hsl._hsl_replay_cache_account_series_dir(bot)
    assert hsl._hsl_replay_cache_validation_reasons(account_dir) == ["manifest_missing"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_cache_persist_matrices_skips_missing_coverage(tmp_path, monkeypatch, caplog):
    import logging as logging_module

    bot, _sink = _make_persist_bot(tmp_path, monkeypatch)
    history = {"hsl_replay_matrices": {"long": {"BTC/USDT:USDT": _hsl_cache_rows()}}}

    with caplog.at_level(logging_module.WARNING):
        written = hsl._equity_hard_stop_persist_replay_matrices(bot, history)

    assert written == 0
    assert "missing matrix coverage metadata" in caplog.text
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_hsl_replay_matrix_cache_reports_array_hash_mismatch(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    arrays["pnl"][1] = -99.0
    np.savez(matrix_path, **arrays)

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    )

    assert "array_hash_mismatch:pnl" in reasons


def test_hsl_replay_matrix_cache_load_rejects_corrupt_matrix(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    arrays["pnl"][1] = -99.0
    np.savez(matrix_path, **arrays)

    with pytest.raises(ValueError, match="array_hash_mismatch:pnl"):
        hsl._load_hsl_replay_matrix_cache(tmp_path, expected_metadata=metadata)


def test_hsl_replay_matrix_cache_write_rejects_semantically_invalid_raw_values(tmp_path):
    rows = _hsl_cache_rows()
    rows[0] = dict(rows[0], price=float("nan"))

    with pytest.raises(ValueError, match="price"):
        hsl._write_hsl_replay_matrix_cache(tmp_path, rows, _hsl_cache_metadata())

    rows = _hsl_cache_rows()
    rows[0] = dict(rows[0], pprice=0.0)

    with pytest.raises(ValueError, match="pprice"):
        hsl._write_hsl_replay_matrix_cache(tmp_path, rows, _hsl_cache_metadata())


def test_hsl_replay_matrix_cache_validation_reports_semantically_invalid_raw_values(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    arrays["price"][0] = float("nan")
    arrays["pprice"][1] = 0.0
    np.savez(matrix_path, **arrays)

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    )

    assert "array_hash_mismatch:price" in reasons
    assert "array_nonfinite:price" in reasons
    assert "array_hash_mismatch:pprice" in reasons
    assert "nonflat_pprice_nonpositive" in reasons


def test_hsl_replay_matrix_cache_validation_reports_non_numeric_array_without_raising(tmp_path):
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    arrays["price"] = np.array(["bad", "bad", "bad"], dtype="<U3")
    np.savez(matrix_path, **arrays)

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    )

    assert "array_dtype_mismatch:price" in reasons
    assert "array_hash_mismatch:price" in reasons
    assert "array_value_invalid:price" in reasons


def test_hsl_replay_matrix_cache_validation_rejects_complex_arrays_with_matching_manifest(
    tmp_path,
):
    import json
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    arrays["ts"] = arrays["ts"].astype(np.complex128) + 1j
    arrays["price"] = arrays["price"].astype(np.complex128) + 1j
    np.savez(matrix_path, **arrays)
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"] = hsl._hsl_replay_cache_array_manifest(arrays)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    )

    assert "array_value_invalid:ts" in reasons
    assert "array_value_invalid:price" in reasons


@pytest.mark.parametrize("bad_ts_kind", ["strings", "scalar", "two_dim"])
def test_hsl_replay_matrix_cache_validation_reports_corrupt_ts_without_raising(
    tmp_path, bad_ts_kind
):
    import numpy as np

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    matrix_path = tmp_path / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME
    with np.load(matrix_path, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in hsl._HSL_REPLAY_MATRIX_RAW_FIELDS}
    if bad_ts_kind == "strings":
        arrays["ts"] = np.array(["bad", "bad", "bad"], dtype="<U3")
    elif bad_ts_kind == "scalar":
        arrays["ts"] = np.array(0)
    else:
        arrays["ts"] = np.array([[60_000], [120_000], [180_000]])
    np.savez(matrix_path, **arrays)

    reasons = hsl._hsl_replay_cache_validation_reasons(
        tmp_path,
        expected_metadata=metadata,
    )

    assert "array_value_invalid:ts" in reasons
    assert any(reason.startswith("array_hash_mismatch:ts") for reason in reasons)


def test_hsl_replay_matrix_cache_requires_trust_boundary_metadata(tmp_path):
    metadata = _hsl_cache_metadata()
    metadata.pop("config_digest")

    with pytest.raises(ValueError, match="metadata missing required fields"):
        hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)


def test_hsl_replay_matrix_cache_validation_requires_manifest_trust_boundary_metadata(tmp_path):
    import json

    metadata = _hsl_cache_metadata()
    hsl._write_hsl_replay_matrix_cache(tmp_path, _hsl_cache_rows(), metadata)
    manifest_path = tmp_path / hsl._HSL_REPLAY_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"].pop("config_digest")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reasons = hsl._hsl_replay_cache_validation_reasons(tmp_path)

    assert reasons == ["metadata_missing_required:config_digest"]


def bind_hsl_methods(bot):
    for name in (
        "_hsl_psides",
        "_hsl_state",
        "_equity_hard_stop_enabled",
        "_equity_hard_stop_runtime_red_latched",
        "_equity_hard_stop_runtime_tier",
        "_equity_hard_stop_signal_mode",
        "_equity_hard_stop_cooldown_position_policy",
        "_calc_upnl_sum_strict",
        "_equity_hard_stop_apply_coin_sample",
        "_equity_hard_stop_apply_coin_metrics_sample",
        "_equity_hard_stop_maybe_emit_raw_red_pending",
        "_equity_hard_stop_activate_coin_red_from_metrics",
        "_equity_hard_stop_coin_active_pside",
        "_equity_hard_stop_coin_realized_pnl_peak_last",
        "_equity_hard_stop_coin_needs_panic_supervision",
        "_equity_hard_stop_coin_red_active",
        "_equity_hard_stop_coin_symbols",
        "_equity_hard_stop_handle_coin_position_during_cooldown",
        "_equity_hard_stop_has_open_position_symbol",
        "_equity_hard_stop_count_blocking_open_orders_symbol",
        "_equity_hard_stop_history_coin_value",
        "_equity_hard_stop_initialize_coin_from_history",
        "_equity_hard_stop_infer_coin_replay_contract",
        "_equity_hard_stop_lookback_ms",
        "_equity_hard_stop_log_transition",
        "_hsl_replay_cache_config_digest",
        "_hsl_replay_cache_expected_metadata",
        "_hsl_replay_cache_dir",
        "_hsl_replay_cache_account_config_digest",
        "_hsl_replay_cache_account_series_dir",
        "_hsl_replay_cache_account_expected_metadata",
        "_equity_hard_stop_persist_replay_matrices",
        "_try_load_hsl_replay_matrix_cache",
        "_equity_hard_stop_try_reuse_replay_cache",
        "_equity_hard_stop_set_red_paused_runtime_forced_modes",
        "_equity_hard_stop_latest_flatten_fill_timestamp_ms",
        "_equity_hard_stop_refresh_halted_runtime_forced_modes",
        "_equity_hard_stop_set_red_runtime_forced_modes",
        "_equity_hard_stop_runtime_red_latched",
        "_equity_hard_stop_clear_runtime_forced_modes",
        "_equity_hard_stop_halted_mode",
        "_equity_hard_stop_build_latch_payload",
        "_equity_hard_stop_check_coin",
        "_equity_hard_stop_clear_coin_runtime_forced_mode",
        "_equity_hard_stop_compute_coin_stop_event",
        "_equity_hard_stop_finalize_coin_red_stop",
        "_equity_hard_stop_latest_panic_fill_timestamp_ms",
        "_equity_hard_stop_latest_panic_fill_timestamp_optional_ms",
        "_equity_hard_stop_log_coin_cooldown_status",
        "_equity_hard_stop_emit_coin_status",
        "_equity_hard_stop_make_state",
        "_equity_hard_stop_prime_coin_runtime_for_replay",
        "_equity_hard_stop_reset_coin_after_restart",
        "_equity_hard_stop_set_coin_runtime_forced_mode",
        "_equity_hard_stop_symbol_supported_for_coin_replay",
        "_hsl_coin_state",
    ):
        setattr(bot, name, MethodType(getattr(hsl, name), bot))
    for name in (
        "_assert_no_pending_pnl_events",
        "_pnl_history_coverage_status",
        "_pnl_blocking_known_gaps",
        "_pnl_gap_is_confirmed_legitimate",
        "_pnl_gap_overlaps",
        "_pnl_event_preview",
        "_assert_pnl_history_safe_for_risk",
    ):
        setattr(bot, name, MethodType(getattr(Passivbot, name), bot))


def make_coin_bot(policy="panic"):
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.user = "test_user"
    bot.exchange = "test_exchange"
    bot._equity_hard_stop = {
        "long": bot._equity_hard_stop_make_state(),
        "short": bot._equity_hard_stop_make_state(),
    }
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._runtime_forced_modes = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.positions = {}
    bot.open_orders = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.c_mults = {}
    bot.config = {
        "live": {
            "hsl_signal_mode": "coin",
            "hsl_position_during_cooldown_policy": policy,
            "pnls_max_lookback_days": 30.0,
        }
    }
    bot.hsl = {
        "long": {
            "enabled": True,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
        "short": {
            "enabled": False,
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
            "cooldown_minutes_after_red": 5.0,
            "no_restart_drawdown_threshold": 0.9,
            "restart_after_red_policy": "threshold",
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market",
        },
    }
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._equity_hard_stop_write_latch = lambda pside, payload, symbol=None: "/tmp/hsl_coin.json"
    bot._equity_hard_stop_remove_latch_file = lambda pside, symbol=None: None
    bot.get_raw_balance = lambda: 100.0
    bot.get_exchange_time = lambda: 180_000
    bot.live_value = lambda key: bot.config["live"][key]
    bot._equity_hard_stop_realized_pnl_now = lambda pside=None: 0.0

    def bot_value(pside, key):
        values = {
            "n_positions": 2,
            "total_wallet_exposure_limit": 2.0,
        }
        return values[key]

    bot.bot_value = bot_value

    async def calc_upnl(pside=None, symbol=None):
        return 0.0

    bot._calc_upnl_sum_strict = calc_upnl
    return bot


def test_passivbot_binds_coin_hsl_replay_support_helper():
    assert hasattr(Passivbot, "_equity_hard_stop_symbol_supported_for_coin_replay")


@pytest.mark.asyncio
async def test_coin_hsl_replay_cancels_when_shutdown_requested_after_history_load():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        bot.stop_signal_received = True
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    async def fail_calc_upnl(*_args, **_kwargs):
        raise AssertionError("shutdown should cancel before live upnl fetch")

    bot.get_balance_equity_history = fake_history
    bot._calc_upnl_sum_strict = fail_calc_upnl

    with pytest.raises(asyncio.CancelledError, match="hsl_coin_history_replay_history_loaded"):
        await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot.stop_signal_received is True
    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is False
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type.startswith("hsl.replay.")]
    assert [event.event_type for event in events] == [
        EventTypes.HSL_REPLAY_STARTED,
        EventTypes.HSL_REPLAY_FAILED,
    ]
    assert events[0].status == "started"
    assert events[0].reason_code == "coin_history_replay"
    assert events[1].status == "failed"
    assert events[1].reason_code == "shutdown_cancelled"
    assert events[1].data["elapsed_s"] is not None
    assert events[1].data["history_fetch_elapsed_s"] is not None
    assert events[1].data["pre_replay_elapsed_s"] is None
    assert events[1].data["replay_loop_elapsed_s"] is None
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_emits_lifecycle_events():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_hsl_replay"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    async def fake_history(current_balance=None, **kwargs):
        await asyncio.sleep(0.01)
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -1.0, "short": 0.0}
                    },
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 1.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 1.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -0.5, "short": 0.0}
                    },
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type.startswith("hsl.replay.")]
    assert [event.event_type for event in events] == [
        EventTypes.HSL_REPLAY_STARTED,
        EventTypes.HSL_REPLAY_PROGRESS,
        EventTypes.HSL_REPLAY_COMPLETED,
    ]
    assert {event.cycle_id for event in events} == {"cy_hsl_replay"}
    assert events[0].status == "started"
    assert events[0].reason_code == "coin_history_replay"
    assert events[1].reason_code == "history_loaded"
    assert events[1].data["symbols"] == 1
    assert events[1].data["pairs"] == 1
    assert events[1].data["held_pairs"] == 1
    assert events[1].data["cooldown_pairs"] == 0
    assert events[1].data["required_pairs"] == 1
    assert events[1].data["timeline_rows"] == 2
    assert events[1].data["history_fetch_elapsed_s"] is not None
    assert events[1].data["pre_replay_elapsed_s"] is not None
    assert events[1].data["elapsed_s"] is not None
    assert events[2].status == "succeeded"
    assert events[2].reason_code == "coin_history_replay_completed"
    assert events[2].data["rows"] == 2
    assert events[2].data["applied_rows"] == 2
    assert events[2].data["pairs"] == 1
    assert events[2].data["held_pairs"] == 1
    assert events[2].data["cooldown_pairs"] == 0
    assert events[2].data["required_pairs"] == 1
    assert events[2].data["skipped_pairs"] == 0
    assert events[2].data["timeline_rows"] == 2
    assert events[2].data["fill_events"] == 0
    assert events[2].data["panic_events"] == 0
    assert events[2].data["rows_per_second"] is not None
    assert events[2].data["history_fetch_elapsed_s"] is not None
    assert events[2].data["pre_replay_elapsed_s"] is not None
    assert events[2].data["replay_loop_elapsed_s"] is not None
    assert events[2].data["full_elapsed_s"] is not None
    assert events[2].data["startup_blocking_elapsed_s"] is not None
    assert events[2].data["elapsed_s"] is not None
    assert events[2].data["history_fetch_elapsed_s"] > 0.0
    phase_elapsed_s = (
        events[2].data["history_fetch_elapsed_s"]
        + events[2].data["pre_replay_elapsed_s"]
        + events[2].data["replay_loop_elapsed_s"]
    )
    assert phase_elapsed_s <= events[2].data["startup_blocking_elapsed_s"] + 0.006
    assert bot._live_event_pipeline.close(timeout=2.0) is True


async def _run_parity_history(monkeypatch):
    """Run the real coin-mode collection over a small two-close scenario."""
    import numpy as np
    from unittest.mock import AsyncMock

    import passivbot as passivbot_module
    from live.event_bus import ListEventSink, LiveEventPipeline

    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 120_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_parity"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 84.0, 100.0, 60.0, 60.0, 1.0),
                    (base_ts + 120_000, 60.0, 102.0, 60.0, 101.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 90_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 101.0,
            "pnl": 0.5,
        },
    ]
    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    return history, symbol


@pytest.mark.asyncio
async def test_hsl_cache_synthesized_rows_match_authoritative_timeline(monkeypatch):
    history, symbol = await _run_parity_history(monkeypatch)
    pair_arrays = hsl._hsl_replay_matrix_arrays(
        history["hsl_replay_matrices"]["long"][symbol]
    )
    account_arrays = hsl._hsl_replay_account_series_arrays(
        history["hsl_replay_account_series"]
    )

    synthesized = hsl._hsl_replay_timeline_rows_from_cache(
        {("long", symbol): pair_arrays},
        account_arrays,
        current_balance=100.0,
    )

    timeline_by_ts = {int(row["timestamp"]): row for row in history["timeline"]}
    assert len(synthesized) == len(history["timeline"])
    seen_pair_values = 0
    for row in synthesized:
        # The synthesized rows expose an explicit coin-replay row contract.
        assert set(row) == {
            "timestamp",
            "balance",
            "realized_pnl",
            "realized_pnl_by_coin_pside",
            "unrealized_pnl_by_coin_pside",
        }
        auth = timeline_by_ts[row["timestamp"]]
        assert row["balance"] == pytest.approx(auth["balance"], abs=1e-9)
        # Account-level realized pnl feeds stop/latch payload diagnostics and
        # must match the authoritative record-window value exactly.
        assert row["realized_pnl"] == pytest.approx(auth["realized_pnl"], abs=1e-9)
        synth_realized = row["realized_pnl_by_coin_pside"].get(symbol, {}).get("long")
        synth_upnl = row["unrealized_pnl_by_coin_pside"].get(symbol, {}).get("long")
        auth_realized = (
            auth.get("realized_pnl_by_coin_pside", {}).get(symbol, {}).get("long")
        )
        auth_upnl = (
            auth.get("unrealized_pnl_by_coin_pside", {}).get(symbol, {}).get("long")
        )
        if auth_realized is not None:
            seen_pair_values += 1
            assert synth_realized == pytest.approx(auth_realized, abs=1e-9)
            assert synth_upnl == pytest.approx(auth_upnl, abs=1e-9)
        elif synth_realized is not None:
            # Collection may start pair rows at the first candle, before the
            # pair's first fill; those extra rows must be exactly neutral.
            assert synth_realized == 0.0
            assert synth_upnl == 0.0
    # The scenario's fill minutes must actually have been compared.
    assert seen_pair_values >= 2


@pytest.mark.asyncio
async def test_hsl_cache_synthesized_replay_reaches_identical_coin_state(monkeypatch):
    history, symbol = await _run_parity_history(monkeypatch)
    pair_arrays = hsl._hsl_replay_matrix_arrays(
        history["hsl_replay_matrices"]["long"][symbol]
    )
    account_arrays = hsl._hsl_replay_account_series_arrays(
        history["hsl_replay_account_series"]
    )
    synthesized = hsl._hsl_replay_timeline_rows_from_cache(
        {("long", symbol): pair_arrays},
        account_arrays,
        current_balance=100.0,
    )
    end_ts = int(history["timeline"][-1]["timestamp"])

    async def run_replay(timeline):
        bot = make_coin_bot()
        bot.positions = {
            symbol: {
                "long": {"size": 0.5, "price": 100.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        bot.c_mults = {symbol: 1.0}
        bot.get_exchange_time = lambda: end_ts

        async def fake_history(current_balance=None, **kwargs):
            return {
                "timeline": timeline,
                "panic_flatten_events": [],
                "fill_events": history["fill_events"],
            }

        bot.get_balance_equity_history = fake_history
        samples = []
        original_apply = bot._equity_hard_stop_apply_coin_metrics_sample

        def recording_apply(*args, **kwargs):
            metrics = original_apply(*args, **kwargs)
            samples.append(
                (
                    metrics["timestamp_ms"],
                    metrics["tier"],
                    round(metrics["drawdown_raw"], 12),
                    round(metrics["drawdown_score"], 12),
                    round(metrics["balance"], 9),
                    round(metrics["unrealized_pnl"], 9),
                )
            )
            return metrics

        bot._equity_hard_stop_apply_coin_metrics_sample = recording_apply
        await bot._equity_hard_stop_initialize_coin_from_history()
        return bot, samples

    bot_auth, samples_auth = await run_replay(history["timeline"])
    bot_synth, samples_synth = await run_replay(synthesized)

    # The scenario dips through the orange tier mid-replay, so the sequences
    # prove parity across tier transitions rather than only green states.
    auth_tiers = {tier for _, tier, *_ in samples_auth}
    assert "orange" in auth_tiers
    # The synthesized replay may include extra leading neutral samples (pair
    # rows can start at the first candle, before the first fill); after
    # aligning on the authoritative sample timestamps the sequences must match
    # exactly.
    auth_ts = {sample[0] for sample in samples_auth}
    assert [s for s in samples_synth if s[0] in auth_ts] == samples_auth
    extra = [s for s in samples_synth if s[0] not in auth_ts]
    assert all(tier == "green" and dd == 0.0 for _, tier, dd, *_ in extra)

    state_auth = bot_auth._hsl_coin_state("long", symbol)
    state_synth = bot_synth._hsl_coin_state("long", symbol)
    metrics_auth = state_auth["last_metrics"]
    metrics_synth = state_synth["last_metrics"]
    assert metrics_auth is not None and metrics_synth is not None
    for key in ("tier", "timestamp_ms"):
        assert metrics_synth[key] == metrics_auth[key]
    for key in (
        "balance",
        "slot_budget",
        "drawdown_usd",
        "drawdown_raw",
        "drawdown_ema",
        "drawdown_score",
        "unrealized_pnl",
    ):
        assert metrics_synth[key] == pytest.approx(metrics_auth[key], abs=1e-9)
    assert state_synth["halted"] == state_auth["halted"]
    assert state_synth["cooldown_until_ms"] == state_auth["cooldown_until_ms"]


async def _run_extension_history(monkeypatch):
    """Five-minute scenario with a fill after the intended cache watermark."""
    import numpy as np
    from unittest.mock import AsyncMock

    import passivbot as passivbot_module
    from live.event_bus import ListEventSink, LiveEventPipeline

    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 240_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.9, "price": (0.5 * 100.0 + 0.4 * 99.0) / 0.9},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_extension"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 84.0, 100.0, 60.0, 60.0, 1.0),
                    (base_ts + 120_000, 60.0, 102.0, 60.0, 101.0, 1.0),
                    (base_ts + 180_000, 99.0, 101.0, 98.0, 99.0, 1.0),
                    (base_ts + 240_000, 99.0, 104.0, 99.0, 103.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 90_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 101.0,
            "pnl": 0.5,
        },
        {
            "timestamp": base_ts + 210_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 0.4,
            "price": 99.0,
            "pnl": 0.0,
        },
    ]
    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    return history, symbol, base_ts


@pytest.mark.asyncio
async def test_hsl_cache_extension_matches_full_rebuild(monkeypatch):
    import numpy as np

    history, symbol, base_ts = await _run_extension_history(monkeypatch)
    watermark_ts = base_ts + 120_000
    end_ts = base_ts + 240_000

    full_pair_rows = history["hsl_replay_matrices"]["long"][symbol]
    full_pair = hsl._hsl_replay_matrix_arrays(full_pair_rows)
    full_account = hsl._hsl_replay_account_series_arrays(
        history["hsl_replay_account_series"]
    )

    # Slice the full arrays at the watermark, exactly what a cache written at
    # that time would contain (collection is deterministic; see parity tests).
    pair_cut = int(np.searchsorted(full_pair["ts"], watermark_ts, side="right"))
    cached_pair = {field: arr[:pair_cut] for field, arr in full_pair.items()}
    account_cut = int(
        np.searchsorted(full_account["ts"], watermark_ts, side="right")
    )
    cached_account = {field: arr[:account_cut] for field, arr in full_account.items()}

    extension_fills = [
        evt
        for evt in history["fill_events"]
        if int(evt["timestamp"]) >= watermark_ts + 60_000
    ]
    assert len(extension_fills) == 1
    closes_by_minute = {base_ts + 180_000: 99.0, base_ts + 240_000: 103.0}

    new_pair_rows = hsl._hsl_replay_extend_pair_rows(
        cached_pair,
        pside="long",
        symbol=symbol,
        fills=extension_fills,
        closes_by_minute=closes_by_minute,
        end_ts=end_ts,
        c_mult=1.0,
    )
    rebuilt_pair = hsl._hsl_replay_matrix_arrays(
        hsl._hsl_replay_rows_from_arrays(cached_pair, series_kind="pair_matrix")
        + new_pair_rows
    )
    for field in full_pair:
        np.testing.assert_allclose(
            rebuilt_pair[field], full_pair[field], rtol=0.0, atol=1e-12
        )

    new_account_rows = hsl._hsl_replay_extend_account_rows(
        cached_account,
        fills=extension_fills,
        end_ts=end_ts,
    )
    rebuilt_account = hsl._hsl_replay_account_series_arrays(
        hsl._hsl_replay_rows_from_arrays(cached_account, series_kind="account_pnl")
        + new_account_rows
    )
    for field in full_account:
        np.testing.assert_allclose(
            rebuilt_account[field], full_account[field], rtol=0.0, atol=1e-12
        )


def test_hsl_cache_extension_fail_loud_contracts():
    pair = hsl._hsl_replay_matrix_arrays(
        [
            hsl._hsl_replay_matrix_row(
                pside="long", ts=60_000, price=100.0, psize=1.0,
                pprice=100.0, pnl=0.0, c_mult=1.0,
            ),
            hsl._hsl_replay_matrix_row(
                pside="long", ts=120_000, price=101.0, psize=1.0,
                pprice=100.0, pnl=0.0, c_mult=1.0,
            ),
        ]
    )

    def fill(ts, action="increase", qty=0.1, price=100.0, pnl=0.0):
        return {
            "timestamp": ts,
            "symbol": "A",
            "pside": "long",
            "qty": qty,
            "price": price,
            "action": action,
            "pnl": pnl,
        }

    # Fill inside the cached window must reject the extension.
    with pytest.raises(ValueError, match="inside the cached window"):
        hsl._hsl_replay_extend_pair_rows(
            pair, pside="long", symbol="A",
            fills=[fill(150_000)], closes_by_minute={}, end_ts=240_000, c_mult=1.0,
        )
    # Fill beyond the extension end must reject.
    with pytest.raises(ValueError, match="beyond the extension end"):
        hsl._hsl_replay_extend_pair_rows(
            pair, pside="long", symbol="A",
            fills=[fill(400_000)], closes_by_minute={}, end_ts=240_000, c_mult=1.0,
        )
    # End before the watermark must reject.
    with pytest.raises(ValueError, match="precedes watermark"):
        hsl._hsl_replay_extend_pair_rows(
            pair, pside="long", symbol="A",
            fills=[], closes_by_minute={}, end_ts=60_000, c_mult=1.0,
        )
    # Fill from another pair must reject.
    wrong = fill(200_000)
    wrong["symbol"] = "B"
    with pytest.raises(ValueError, match="does not belong to pair"):
        hsl._hsl_replay_extend_pair_rows(
            pair, pside="long", symbol="A",
            fills=[wrong], closes_by_minute={}, end_ts=240_000, c_mult=1.0,
        )
    # Fills with stripped pair identity must reject, never default to the
    # target pair.
    for missing_keys in (("symbol",), ("pside",), ("symbol", "pside")):
        stripped = fill(200_000)
        for key in missing_keys:
            del stripped[key]
        with pytest.raises(ValueError, match="missing pair identity"):
            hsl._hsl_replay_extend_pair_rows(
                pair, pside="long", symbol="A",
                fills=[stripped], closes_by_minute={}, end_ts=240_000, c_mult=1.0,
            )
    # Zero-length extension is a no-op.
    assert (
        hsl._hsl_replay_extend_pair_rows(
            pair, pside="long", symbol="A",
            fills=[], closes_by_minute={}, end_ts=120_000, c_mult=1.0,
        )
        == []
    )
    # Missing closes carry the last known price forward.
    rows = hsl._hsl_replay_extend_pair_rows(
        pair, pside="long", symbol="A",
        fills=[], closes_by_minute={}, end_ts=240_000, c_mult=1.0,
    )
    assert [row["price"] for row in rows] == [101.0, 101.0]
    assert [row["psize"] for row in rows] == [1.0, 1.0]


def _bind_reuse_support(bot, tmp_path, monkeypatch, *, fills, covered_start_ms=1):
    """Equip a fake coin bot with everything the cache-reuse path consumes."""
    from unittest.mock import AsyncMock

    def fake_get_filepath(rel):
        path = tmp_path / rel
        path.mkdir(parents=True, exist_ok=True)
        return f"{path}/"

    monkeypatch.setattr(hsl, "make_get_filepath", fake_get_filepath)
    bot.market_type = "swap"
    bot.qty_steps = {}
    bot.init_pnls = AsyncMock()
    bot._pnl_history_coverage_status = MethodType(
        Passivbot._pnl_history_coverage_status, bot
    )
    bot._pnl_blocking_known_gaps = MethodType(Passivbot._pnl_blocking_known_gaps, bot)
    bot._pnl_gap_is_confirmed_legitimate = Passivbot._pnl_gap_is_confirmed_legitimate
    bot._pnl_gap_overlaps = Passivbot._pnl_gap_overlaps
    bot._hsl_extract_fill_events = MethodType(Passivbot._hsl_extract_fill_events, bot)
    bot._hsl_normalize_fill_symbol = MethodType(
        Passivbot._hsl_normalize_fill_symbol, bot
    )
    bot.get_symbol_id_inv = lambda sym: sym

    class _StubEvent:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return dict(self._payload)

    class _StubCache:
        def load_metadata(self):
            return {"covered_start_ms": covered_start_ms, "oldest_event_ts": 1}

        def get_covered_start_ms(self):
            return covered_start_ms

        def get_history_scope(self):
            return "all"

    class _StubPnlsManager:
        cache = _StubCache()

        def get_events(self):
            return [_StubEvent(payload) for payload in fills]

        def get_history_scope(self):
            return "all"

    bot._pnls_manager = _StubPnlsManager()
    return bot


_REUSE_BASE_TS = 1_800_000_000_000
_REUSE_SYMBOL = "BTC/USDT:USDT"


def _reuse_fill(offset_ms, *, side, qty, price, pnl=0.0, pb_order_type=""):
    return {
        "timestamp": _REUSE_BASE_TS + offset_ms,
        "symbol": _REUSE_SYMBOL,
        "position_side": "long",
        "side": side,
        "qty": qty,
        "price": price,
        "pnl": pnl,
        "pb_order_type": pb_order_type,
    }


_REUSE_PREFIX_FILLS = [
    _reuse_fill(0, side="buy", qty=1.0, price=100.0),
    _reuse_fill(90_000, side="sell", qty=0.5, price=101.0, pnl=0.5),
]
_REUSE_GAP_FILL = _reuse_fill(210_000, side="buy", qty=0.4, price=99.0)
_REUSE_CANDLES = [
    (_REUSE_BASE_TS, 99.0, 101.0, 98.0, 100.0, 1.0),
    (_REUSE_BASE_TS + 60_000, 84.0, 100.0, 60.0, 60.0, 1.0),
    (_REUSE_BASE_TS + 120_000, 60.0, 102.0, 60.0, 101.0, 1.0),
    (_REUSE_BASE_TS + 180_000, 99.0, 101.0, 98.0, 99.0, 1.0),
    (_REUSE_BASE_TS + 240_000, 99.0, 104.0, 99.0, 103.0, 1.0),
]


async def _reuse_collection_history(monkeypatch, *, n_minutes, fills, positions):
    """Run the real manager-backed collection over the first n scenario minutes."""
    import numpy as np
    from unittest.mock import AsyncMock

    import passivbot as passivbot_module
    from live.event_bus import ListEventSink, LiveEventPipeline

    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "test_exchange"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    ts_now = _REUSE_BASE_TS + (n_minutes - 1) * 60_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    bot.positions = positions
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()], monitor_sinks=[]
    )
    bot._live_event_current_cycle_id = "cy_reuse_collect"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {_REUSE_SYMBOL: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                _REUSE_CANDLES[:n_minutes], dtype=passivbot_module.CANDLE_DTYPE
            )

    bot.cm = _CM()

    class _StubEvent:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return dict(self._payload)

    class _StubCache:
        def load_metadata(self):
            return {"covered_start_ms": 1, "oldest_event_ts": 1}

        def get_covered_start_ms(self):
            return 1

        def get_history_scope(self):
            return "all"

    class _StubPnlsManager:
        cache = _StubCache()

        def __init__(self, payloads):
            self._payloads = payloads

        def get_events(self):
            return [_StubEvent(p) for p in self._payloads]

        def get_history_scope(self):
            return "all"

    bot._pnls_manager = _StubPnlsManager(fills)
    history = await bot.get_balance_equity_history(
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    return history


def _make_reuse_bot(tmp_path, monkeypatch, *, fills, exchange_now, positions):
    import numpy as np

    import passivbot as passivbot_module

    bot = make_coin_bot()
    bot.exchange = "test_exchange"
    bot.positions = positions
    bot.c_mults = {_REUSE_SYMBOL: 1.0}
    bot.get_exchange_time = lambda: exchange_now
    _bind_reuse_support(bot, tmp_path, monkeypatch, fills=fills)

    class _GapCM:
        async def get_candles(self, sym, *, start_ts, end_ts, **kwargs):
            rows = [
                row
                for row in _REUSE_CANDLES
                if start_ts <= row[0] <= end_ts
            ]
            return np.array(rows, dtype=passivbot_module.CANDLE_DTYPE)

    bot.cm = _GapCM()

    async def calc_upnl(pside=None, symbol=None):
        # Final close 103 vs pprice (0.5*100 + 0.4*99)/0.9.
        pprice = (0.5 * 100.0 + 0.4 * 99.0) / 0.9
        return (103.0 - pprice) * 0.9

    bot._calc_upnl_sum_strict = calc_upnl

    def fail_full_replay(*args, **kwargs):
        raise AssertionError("cache reuse must not fall back in this scenario")

    bot._fail_full_replay = fail_full_replay
    return bot


@pytest.mark.asyncio
async def test_hsl_cache_reuse_reaches_state_identical_to_full_replay(
    tmp_path, monkeypatch
):
    prefix_positions = {
        _REUSE_SYMBOL: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    final_positions = {
        _REUSE_SYMBOL: {
            "long": {"size": 0.9, "price": (0.5 * 100.0 + 0.4 * 99.0) / 0.9},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    all_fills = _REUSE_PREFIX_FILLS + [_REUSE_GAP_FILL]
    exchange_now = _REUSE_BASE_TS + 240_000

    # Boot 1: full replay over the prefix window writes the cache.
    prefix_history = await _reuse_collection_history(
        monkeypatch, n_minutes=3, fills=_REUSE_PREFIX_FILLS, positions=prefix_positions
    )
    writer_bot = make_coin_bot()
    writer_bot.exchange = "test_exchange"
    _bind_reuse_support(writer_bot, tmp_path, monkeypatch, fills=_REUSE_PREFIX_FILLS)
    assert writer_bot._equity_hard_stop_persist_replay_matrices(prefix_history) == 2

    # Boot 2: cache-fed replay over the full window; full fetch must not run.
    reuse_bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    reuse_bot.get_balance_equity_history = reuse_bot._fail_full_replay
    await reuse_bot._equity_hard_stop_initialize_coin_from_history()
    assert getattr(reuse_bot, "_equity_hard_stop_coin_initialized", False) is True

    # Boot 3 (control): authoritative full replay over the full window.
    full_history = await _reuse_collection_history(
        monkeypatch, n_minutes=5, fills=all_fills, positions=final_positions
    )
    control_bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )

    async def full_fetch(current_balance=None, **kwargs):
        return full_history

    control_bot.get_balance_equity_history = full_fetch
    await control_bot._equity_hard_stop_initialize_coin_from_history()

    state_reuse = reuse_bot._hsl_coin_state("long", _REUSE_SYMBOL)
    state_control = control_bot._hsl_coin_state("long", _REUSE_SYMBOL)
    metrics_reuse = state_reuse["last_metrics"]
    metrics_control = state_control["last_metrics"]
    assert metrics_reuse is not None and metrics_control is not None
    assert metrics_reuse["tier"] == metrics_control["tier"]
    for key in (
        "balance",
        "slot_budget",
        "drawdown_usd",
        "drawdown_raw",
        "drawdown_ema",
        "drawdown_score",
        "unrealized_pnl",
    ):
        assert metrics_reuse[key] == pytest.approx(metrics_control[key], abs=1e-9)
    assert state_reuse["halted"] == state_control["halted"]
    assert state_reuse["cooldown_until_ms"] == state_control["cooldown_until_ms"]


@pytest.mark.asyncio
async def test_hsl_cache_reuse_falls_back_on_missing_or_incompatible_cache(
    tmp_path, monkeypatch
):
    final_positions = {
        _REUSE_SYMBOL: {
            "long": {"size": 0.9, "price": (0.5 * 100.0 + 0.4 * 99.0) / 0.9},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    all_fills = _REUSE_PREFIX_FILLS + [_REUSE_GAP_FILL]
    exchange_now = _REUSE_BASE_TS + 240_000

    # No cache on disk: reuse returns None (miss) and the caller falls back.
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )

    # Write a valid cache, then break each reuse gate in turn.
    prefix_positions = {
        _REUSE_SYMBOL: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    prefix_history = await _reuse_collection_history(
        monkeypatch, n_minutes=3, fills=_REUSE_PREFIX_FILLS, positions=prefix_positions
    )
    writer_bot = make_coin_bot()
    writer_bot.exchange = "test_exchange"
    _bind_reuse_support(writer_bot, tmp_path, monkeypatch, fills=_REUSE_PREFIX_FILLS)
    assert writer_bot._equity_hard_stop_persist_replay_matrices(prefix_history) == 2

    # Sanity: the intact cache is reusable.
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is not None
    )

    # HSL config change rotates the digest: rejected.
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    bot.hsl["long"]["red_threshold"] = 0.31
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )

    # Positions that do not reconcile with the extended series: rejected.
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions={
            _REUSE_SYMBOL: {
                "long": {"size": 5.0, "price": 100.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        },
    )
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )

    # A panic fill inside the gap: skipped (full replay owns marker derivation).
    panic_fills = _REUSE_PREFIX_FILLS + [
        dict(_REUSE_GAP_FILL, pb_order_type="close_panic_long")
    ]
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=panic_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )

    # A cached panic marker for a supported flat coin whose pair matrix is
    # not loaded must reject reuse (full replay owns flat-pair cooldown
    # reconstruction) instead of aborting startup mid-replay.
    flat_symbol = "ETH/USDT:USDT"
    marker_history = dict(prefix_history)
    marker_history["panic_flatten_events"] = [
        {
            "timestamp": _REUSE_BASE_TS + 500,
            "minute_timestamp": _REUSE_BASE_TS,
            "pside": "long",
            "symbol": flat_symbol,
        }
    ]
    writer_bot = make_coin_bot()
    writer_bot.exchange = "test_exchange"
    _bind_reuse_support(writer_bot, tmp_path, monkeypatch, fills=_REUSE_PREFIX_FILLS)
    assert writer_bot._equity_hard_stop_persist_replay_matrices(marker_history) == 2
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    # The flat marker symbol is supported for coin replay but not held.
    bot.c_mults[flat_symbol] = 1.0
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )
    # With the marker symbol unsupported, the marker is ignored by the full
    # replay too, so reuse remains allowed.
    del bot.c_mults[flat_symbol]
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is not None
    )
    # Restore the marker-free cache for the remaining gate checks.
    writer_bot = make_coin_bot()
    writer_bot.exchange = "test_exchange"
    _bind_reuse_support(writer_bot, tmp_path, monkeypatch, fills=_REUSE_PREFIX_FILLS)
    assert writer_bot._equity_hard_stop_persist_replay_matrices(prefix_history) == 2

    # Unproven load-time coverage: skipped.
    bot = _make_reuse_bot(
        tmp_path,
        monkeypatch,
        fills=all_fills,
        exchange_now=exchange_now,
        positions=final_positions,
    )
    bot._pnls_manager.cache.get_history_scope = lambda: "window"
    bot._pnls_manager.get_history_scope = lambda: "window"
    bot._pnls_manager.cache.get_covered_start_ms = lambda: 0
    bot._pnls_manager.cache.load_metadata = lambda: {
        "covered_start_ms": 0,
        "oldest_event_ts": 1,
    }
    assert (
        await bot._equity_hard_stop_try_reuse_replay_cache(exchange_now) is None
    )


def test_hsl_cache_synthesized_rows_fail_loud_on_bad_inputs():
    account = hsl._hsl_replay_account_series_arrays(
        [
            hsl._hsl_replay_account_series_row(ts=60_000, pnl=0.0),
            hsl._hsl_replay_account_series_row(ts=120_000, pnl=0.5),
        ]
    )
    pair = hsl._hsl_replay_matrix_arrays(
        [
            hsl._hsl_replay_matrix_row(
                pside="long", ts=60_000, price=100.0, psize=1.0,
                pprice=100.0, pnl=0.0, c_mult=1.0,
            ),
            hsl._hsl_replay_matrix_row(
                pside="long", ts=120_000, price=101.0, psize=1.0,
                pprice=100.0, pnl=0.5, c_mult=1.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="current_balance"):
        hsl._hsl_replay_timeline_rows_from_cache(
            {("long", "A"): pair}, account, current_balance=0.0
        )

    out_of_span = hsl._hsl_replay_matrix_arrays(
        [
            hsl._hsl_replay_matrix_row(
                pside="long", ts=180_000, price=100.0, psize=1.0,
                pprice=100.0, pnl=0.0, c_mult=1.0,
            ),
        ]
    )
    with pytest.raises(ValueError, match="outside account span"):
        hsl._hsl_replay_timeline_rows_from_cache(
            {("long", "A"): out_of_span}, account, current_balance=100.0
        )

    # A synthesized balance dipping to <= 0 anywhere must be rejected because
    # the coin metrics layer requires balance > 0.
    deep_loss_account = hsl._hsl_replay_account_series_arrays(
        [
            hsl._hsl_replay_account_series_row(ts=60_000, pnl=0.0),
            hsl._hsl_replay_account_series_row(ts=120_000, pnl=250.0),
        ]
    )
    with pytest.raises(ValueError, match="finite and > 0"):
        hsl._hsl_replay_timeline_rows_from_cache(
            {}, deep_loss_account, current_balance=100.0
        )


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_persists_replay_matrices(tmp_path, monkeypatch):
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    def fake_get_filepath(rel):
        path = tmp_path / rel
        path.mkdir(parents=True, exist_ok=True)
        return f"{path}/"

    monkeypatch.setattr(hsl, "make_get_filepath", fake_get_filepath)
    bot = make_coin_bot()
    bot.market_type = "swap"
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_hsl_replay"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    matrix_rows = [
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=60_000,
            price=100.0,
            psize=1.0,
            pprice=101.0,
            pnl=0.0,
            c_mult=1.0,
        ),
        hsl._hsl_replay_matrix_row(
            pside="long",
            ts=120_000,
            price=99.5,
            psize=1.0,
            pprice=101.0,
            pnl=1.0,
            c_mult=1.0,
        ),
    ]
    coverage = {
        "fill_covered_start_ms": 0,
        "fill_covered_end_ms": 180_000,
        "fill_history_scope": "all",
        "fill_coverage_proven": True,
        "candle_covered_start_ms": 60_000,
        "candle_covered_end_ms": 120_000,
    }

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 0.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -1.0, "short": 0.0}
                    },
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 1.0,
                    "realized_pnl_by_coin_pside": {
                        symbol: {"long": 1.0, "short": 0.0}
                    },
                    "unrealized_pnl_by_coin_pside": {
                        symbol: {"long": -0.5, "short": 0.0}
                    },
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [],
            "hsl_replay_matrices": {"long": {symbol: matrix_rows}},
            "hsl_replay_account_series": [
                hsl._hsl_replay_account_series_row(ts=60_000, pnl=0.0),
                hsl._hsl_replay_account_series_row(ts=120_000, pnl=1.0),
            ],
            "hsl_replay_matrix_coverage": coverage,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert getattr(bot, "_equity_hard_stop_coin_initialized", False) is True
    expected_metadata = hsl._hsl_replay_cache_expected_metadata(
        bot,
        "long",
        symbol,
        fill_covered_start_ms=coverage["fill_covered_start_ms"],
        fill_covered_end_ms=coverage["fill_covered_end_ms"],
        fill_history_scope=coverage["fill_history_scope"],
        fill_coverage_proven=coverage["fill_coverage_proven"],
        candle_covered_start_ms=coverage["candle_covered_start_ms"],
        candle_covered_end_ms=coverage["candle_covered_end_ms"],
    )
    cache_dir = hsl._hsl_replay_cache_dir(bot, "long", symbol)
    assert (
        hsl._hsl_replay_cache_validation_reasons(
            cache_dir, expected_metadata=expected_metadata
        )
        == []
    )
    account_expected = hsl._hsl_replay_cache_account_expected_metadata(
        bot,
        fill_covered_start_ms=coverage["fill_covered_start_ms"],
        fill_covered_end_ms=coverage["fill_covered_end_ms"],
        fill_history_scope=coverage["fill_history_scope"],
        fill_coverage_proven=coverage["fill_coverage_proven"],
        candle_covered_start_ms=coverage["candle_covered_start_ms"],
        candle_covered_end_ms=coverage["candle_covered_end_ms"],
    )
    account_dir = hsl._hsl_replay_cache_account_series_dir(bot)
    assert (
        hsl._hsl_replay_cache_validation_reasons(
            account_dir, expected_metadata=account_expected
        )
        == []
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    replay_events = [
        event for event in sink.events if event.event_type.startswith("hsl.replay.")
    ]
    completed_idx = [
        idx
        for idx, event in enumerate(replay_events)
        if event.event_type == EventTypes.HSL_REPLAY_COMPLETED
    ]
    written_idx = [
        idx
        for idx, event in enumerate(replay_events)
        if event.event_type == EventTypes.HSL_REPLAY_CACHE
        and event.reason_code == ReasonCodes.HSL_REPLAY_CACHE_WRITTEN
    ]
    assert len(completed_idx) == 1
    assert len(written_idx) == 2
    assert all(idx > completed_idx[0] for idx in written_idx)
    written_event = replay_events[written_idx[0]]
    assert written_event.status == "succeeded"
    assert written_event.pside == "long"
    assert written_event.symbol == symbol
    assert written_event.data["row_count"] == 2
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_calc_upnl_sum_strict_preserves_symbol_filter():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot.fetched_positions = [
        {"symbol": "A", "position_side": "long", "price": 100.0, "size": 1.0},
        {"symbol": "B", "position_side": "long", "price": 100.0, "size": 2.0},
    ]
    bot.c_mults = {"A": 1.0, "B": 1.0}

    async def get_live_last_prices(symbols, max_age_ms, context):
        return {"A": 90.0, "B": 80.0}

    bot._get_live_last_prices = get_live_last_prices

    assert asyncio.run(bot._calc_upnl_sum_strict("long")) == pytest.approx(-50.0)
    assert asyncio.run(bot._calc_upnl_sum_strict("long", "A")) == pytest.approx(-10.0)


def test_coin_hsl_slot_budget_rejects_zero_n_positions():
    bot = FakeHslBot()
    bind_hsl_methods(bot)
    bot._equity_hard_stop_coin = {"long": {}, "short": {}}
    bot._pnls_manager = None
    bot.config = {"live": {"pnls_max_lookback_days": 30.0}}
    bot.hsl = {
        "long": {
            "red_threshold": 0.5,
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "ema_span_minutes": 1.0,
        }
    }

    def bot_value(pside, key):
        values = {
            "n_positions": 0,
            "total_wallet_exposure_limit": 1.0,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match="n_positions"):
        bot._equity_hard_stop_apply_coin_sample("long", "A", 60_000, 100.0, -1.0)


def test_hsl_transition_falls_back_to_monitor_when_pipeline_absent():
    bot = make_coin_bot()
    captured = []
    bot._live_event_pipeline = None
    bot._live_event_current_cycle_id = "cy_absent_pipeline"
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    def record_event(kind, tags, payload, *, pside=None, symbol=None, ts=None):
        captured.append(
            {
                "kind": kind,
                "tags": tuple(tags),
                "payload": payload,
                "pside": pside,
                "symbol": symbol,
                "ts": ts,
            }
        )

    bot._monitor_record_event = record_event
    metrics = {
        "pside": "long",
        "signal_mode": "pside",
        "timestamp_ms": 180_000,
        "balance": 100.0,
        "strategy_equity": 98.0,
        "peak_strategy_equity": 100.0,
        "rolling_peak_strategy_equity": 100.0,
        "drawdown_raw": 0.02,
        "drawdown_ema": 0.01,
        "drawdown_score": 0.01,
        "strategy_pnl": -2.0,
        "peak_strategy_pnl": 0.0,
        "red_threshold": 0.5,
        "tier": "yellow",
        "changed": True,
    }

    bot._equity_hard_stop_log_transition("long", metrics, "green")

    assert len(captured) == 1
    event = captured[0]
    assert event["kind"] == "hsl.transition"
    assert event["tags"] == ("hsl", "risk", "transition")
    assert event["pside"] == "long"
    assert event["ts"] == 180_000
    assert event["payload"]["previous_tier"] == "green"
    assert event["payload"]["tier"] == "yellow"
    assert event["payload"]["metrics"]["tier"] == "yellow"
    assert event["payload"]["metrics"]["changed"] is True


@pytest.mark.asyncio
async def test_coin_hsl_check_skips_enabled_side_with_zero_budget():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_hsl"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    out = await bot._equity_hard_stop_check_coin()

    assert set(out) == {f"long:{symbol}"}
    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_STATUS]
    assert len(events) == 1
    assert events[0].cycle_id == "cy_coin_hsl"
    assert events[0].symbol == symbol
    assert events[0].pside == "long"
    assert events[0].data["signal_mode"] == "coin"
    assert events[0].data["dist_to_red"] == pytest.approx(0.5)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_check_emits_raw_red_pending_event_with_bounded_payload():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_raw_red_pending"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    bot._equity_hard_stop_status_log_interval_ms = 15 * 60 * 1000
    bot.get_exchange_time = lambda: 1_000_000

    def pending_metrics(_pside, _symbol, timestamp_ms, _balance, _current_upnl):
        return {
            "pside": "long",
            "symbol": symbol,
            "signal_mode": "coin",
            "timestamp_ms": int(timestamp_ms),
            "drawdown_raw": 0.20,
            "drawdown_ema": 0.05,
            "drawdown_score": 0.05,
            "red_threshold": 0.10,
            "tier": "orange",
            "changed": False,
            "elapsed_minutes": 1,
            "slot_budget": 100.0,
            "realized_pnl": 0.0,
            "peak_realized_pnl": 20.0,
            "unrealized_pnl": 0.0,
        }

    bot._equity_hard_stop_apply_coin_sample = pending_metrics

    await bot._equity_hard_stop_check_coin()
    await bot._equity_hard_stop_check_coin()

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_RAW_RED_PENDING
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "warning"
    assert event.status == "degraded"
    assert event.reason_code == ReasonCodes.HSL_RAW_RED_PENDING_EMA_CONFIRMATION
    assert event.cycle_id == "cy_coin_raw_red_pending"
    assert event.symbol == symbol
    assert event.pside == "long"
    assert event.data["signal_mode"] == "coin"
    assert event.data["drawdown_raw"] == pytest.approx(0.20)
    assert event.data["drawdown_ema"] == pytest.approx(0.05)
    assert event.data["dist_to_red"] == pytest.approx(0.05)
    assert event.data["raw_excess"] == pytest.approx(0.10)
    assert event.data["balance_override_active"] is False
    assert "balance" not in event.data
    assert "slot_budget" not in event.data
    assert "realized_pnl" not in event.data
    assert "peak_realized_pnl" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_coin_hsl_runtime_forced_mode_changes_emit_risk_events():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_risk_mode"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._current_live_event_cycle_id = MethodType(Passivbot._current_live_event_cycle_id, bot)
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    bot._emit_risk_mode_changed_event = MethodType(
        Passivbot._emit_risk_mode_changed_event,
        bot,
    )

    bot._equity_hard_stop_set_coin_runtime_forced_mode("long", symbol, "panic")
    bot._equity_hard_stop_set_coin_runtime_forced_mode("long", symbol, "panic")
    bot._equity_hard_stop_clear_coin_runtime_forced_mode("long", symbol)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.RISK_MODE_CHANGED]
    assert len(events) == 2
    assert events[0].cycle_id == "cy_risk_mode"
    assert events[0].pside == "long"
    assert events[0].symbol == symbol
    assert events[0].reason_code == "hsl_runtime_forced_mode_set"
    assert events[0].data["action"] == "set"
    assert events[0].data["mode"] == "panic"
    assert "previous_mode" not in events[0].data
    assert events[1].reason_code == "hsl_runtime_forced_mode_clear"
    assert events[1].data["action"] == "clear"
    assert events[1].data["previous_mode"] == "panic"
    assert "mode" not in events[1].data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_skips_enabled_side_with_zero_budget():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["short"]["enabled"] = True
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    def bot_value(pside, key):
        values = {
            "long": {"n_positions": 2, "total_wallet_exposure_limit": 2.0},
            "short": {"n_positions": 3, "total_wallet_exposure_limit": 0.0},
        }
        return values[pside][key]

    bot.bot_value = bot_value

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert symbol in bot._equity_hard_stop_coin["long"]
    assert bot._equity_hard_stop_coin["short"] == {}


@pytest.mark.parametrize(
    "n_positions,total_wallet_exposure_limit,match",
    [
        (-1, 1.0, "n_positions"),
        (float("nan"), 1.0, "n_positions"),
        (0.4, 1.0, "round to > 0"),
        (1, -1.0, "total_wallet_exposure_limit"),
        (1, float("inf"), "total_wallet_exposure_limit"),
    ],
)
def test_coin_hsl_active_side_rejects_invalid_budget_config(
    n_positions, total_wallet_exposure_limit, match
):
    bot = make_coin_bot()

    def bot_value(pside, key):
        values = {
            "n_positions": n_positions,
            "total_wallet_exposure_limit": total_wallet_exposure_limit,
        }
        return values[key]

    bot.bot_value = bot_value

    with pytest.raises(ValueError, match=match):
        bot._equity_hard_stop_coin_active_pside("long")


@pytest.mark.parametrize("total_wallet_exposure_limit", [0.5, 1.0, 5.0])
def test_coin_hsl_live_slot_budget_ignores_twel(total_wallet_exposure_limit):
    bot = make_coin_bot()
    symbol = "A"

    def bot_value(pside, key):
        values = {
            "n_positions": 2,
            "total_wallet_exposure_limit": total_wallet_exposure_limit,
        }
        return values[key]

    bot.bot_value = bot_value

    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        0,
        100.0,
        0.0,
        0.0,
        0.0,
    )
    metrics = bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        60_000,
        100.0,
        0.0,
        -25.0,
        0.0,
    )

    assert metrics["slot_budget"] == pytest.approx(50.0)
    assert metrics["drawdown_usd"] == pytest.approx(25.0)
    assert metrics["drawdown_raw"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_does_not_latch_recovered_red_without_panic_marker():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is False
    assert state["runtime"].tier() == "green"
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"
    assert state["pending_red_since_ms"] is None
    assert state["pending_stop_event"] is None
    assert symbol not in bot._runtime_forced_modes["long"]
    assert bot._equity_hard_stop_coin_red_active() is False


@pytest.mark.asyncio
async def test_coin_hsl_check_defers_stop_event_until_flat_confirmation():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def calc_upnl(pside=None, symbol=None):
        return -80.0

    async def fail_compute(*_args, **_kwargs):
        raise AssertionError("coin HSL must not snapshot stop event at RED trigger time")

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_compute_coin_stop_event = fail_compute
    bot._equity_hard_stop_apply_coin_metrics_sample(
        "long",
        symbol,
        60_000,
        100.0,
        0.0,
        0.0,
        0.0,
    )

    out = await bot._equity_hard_stop_check_coin()

    state = bot._hsl_coin_state("long", symbol)
    assert out[f"long:{symbol}"]["tier"] == "red"
    assert state["pending_red_since_ms"] == 180_000
    assert state["pending_stop_event"] is None
    assert bot._runtime_forced_modes["long"][symbol] == "panic"


@pytest.mark.asyncio
async def test_coin_hsl_finalize_uses_latest_panic_fill_for_reset_boundary():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_finalize"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_panic_long",
                "pnl": -12.0,
                "fee_paid": -0.1,
            }
        ]
    )

    await bot._equity_hard_stop_finalize_coin_red_stop("long", symbol)

    assert state["last_stop_event"]["stop_event_timestamp_ms"] == 170_000
    assert state["pnl_reset_timestamp_ms"] == 170_001
    assert state["cooldown_until_ms"] == 470_000
    assert state["red_trigger_event_emitted"] is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED]
    assert len(events) == 1
    assert events[0].cycle_id == "cy_coin_finalize"
    assert events[0].pside == "long"
    assert events[0].symbol == symbol
    assert events[0].reason_code == "coin_red_stop_finalized"
    assert events[0].data["stop_event_timestamp_ms"] == 170_000
    assert events[0].data["cooldown_until_ms"] == 470_000
    assert [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ] == []
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_emits_flat_without_order_event():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_flat_finalize"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000

    await bot._equity_hard_stop_finalize_coin_red_stop(
        "long",
        symbol,
        finalized_without_order=True,
        flat_confirmations=2,
        entry_orders=0,
        nonpanic_close_orders=0,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_coin_flat_finalize"
    assert event.pside == "long"
    assert event.symbol == symbol
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.HSL_RED_FINALIZED_WITHOUT_EXCHANGE_ORDER
    assert event.data["no_exchange_close_needed"] is True
    assert event.data["exchange_close_order_submitted"] is False
    assert event.data["panic_order_submitted_count"] == 0
    assert event.data["symbol_position_open"] is False
    assert event.data["position_count"] == 0
    assert event.data["entry_orders"] == 0
    assert event.data["nonpanic_close_orders"] == 0
    assert event.data["flat_confirmations"] == 2
    assert event.data["stop_event_timestamp_ms"] == 180_000
    assert event.data["stop_event_anchor_source"] == "current_time_fallback"
    assert event.data["stop_event_anchor_timestamp_ms"] == 180_000
    assert event.data["stop_event_anchor_fallback_used"] is True
    assert event.data["cooldown_until_ms"] == 480_000
    assert event.data["drawdown_raw"] == 0.0
    red_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED
    ]
    assert len(red_events) == 1
    red_event = red_events[0]
    assert red_event.level == "info"
    assert red_event.status == "succeeded"
    assert red_event.reason_code == "coin_red_stop_finalized"
    assert red_event.data["no_exchange_close_needed"] is True
    assert red_event.data["exchange_close_order_submitted"] is False
    assert red_event.data["panic_order_submitted_count"] == 0
    assert red_event.data["symbol_position_open"] is False
    assert red_event.data["entry_orders"] == 0
    assert red_event.data["nonpanic_close_orders"] == 0
    assert red_event.data["flat_confirmations"] == 2
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_flat_without_order_event_records_panic_fill_anchor():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_flat_fill_anchor"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    bot._pnls_manager = make_fake_pnls_manager(
        [
            {
                "timestamp": 170_000,
                "symbol": symbol,
                "pside": "long",
                "pb_order_type": "close_panic_long",
                "pnl": -12.0,
                "fee_paid": -0.1,
            }
        ]
    )

    await bot._equity_hard_stop_finalize_coin_red_stop(
        "long",
        symbol,
        finalized_without_order=True,
        flat_confirmations=2,
        entry_orders=0,
        nonpanic_close_orders=0,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_coin_flat_fill_anchor"
    assert event.data["stop_event_timestamp_ms"] == 170_000
    assert event.data["stop_event_anchor_source"] == "panic_fill"
    assert event.data["stop_event_anchor_timestamp_ms"] == 170_000
    assert event.data["stop_event_anchor_fallback_used"] is False
    assert event.data["cooldown_until_ms"] == 470_000
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_finalize_does_not_duplicate_prior_red_trigger_event():
    from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

    bot = make_coin_bot()
    symbol = "A"
    bot.get_exchange_time = lambda: 180_000
    sink = ListEventSink()
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_coin_finalize_duplicate"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)
    state = bot._hsl_coin_state("long", symbol)
    state["pending_red_since_ms"] = 120_000
    state["red_trigger_event_emitted"] = True

    await bot._equity_hard_stop_finalize_coin_red_stop("long", symbol)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    red_events = [event for event in sink.events if event.event_type == EventTypes.HSL_RED_TRIGGERED]
    cooldown_events = [
        event for event in sink.events if event.event_type == EventTypes.HSL_COOLDOWN_STARTED
    ]
    assert red_events == []
    assert len(cooldown_events) == 1
    assert cooldown_events[0].reason_code == "coin_red_stop_finalized"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_rebases_lookback_window_realized_points():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["red_threshold"] = 0.9
    bot.config["live"]["pnls_max_lookback_days"] = 1.0 / 1440.0
    bot.get_exchange_time = lambda: 240_000
    fill_events = [
        {"timestamp": 120_000, "symbol": symbol, "pside": "long", "pnl": -20.0},
        {"timestamp": 240_000, "symbol": symbol, "pside": "long", "pnl": -35.0},
    ]
    bot._pnls_manager = make_fake_pnls_manager(fill_events)

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 240_000,
                    "balance": 100.0,
                    "realized_pnl": -55.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -55.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["runtime"].red_latched() is False
    assert state["last_metrics"]["realized_pnl"] == pytest.approx(-35.0)
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(0.70)


@pytest.mark.parametrize(
    ("restart_after_red_policy", "expected_latched", "expected_cooldown_until_ms"),
    [
        ("threshold", True, None),
        ("always", False, 420_500),
        ("never", True, None),
    ],
)
@pytest.mark.asyncio
async def test_coin_hsl_history_replay_honors_restart_after_red_policy(
    restart_after_red_policy, expected_latched, expected_cooldown_until_ms
):
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["no_restart_drawdown_threshold"] = 0.7
    bot.hsl["long"]["restart_after_red_policy"] = restart_after_red_policy

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["no_restart_latched"] is expected_latched
    assert state["cooldown_until_ms"] == expected_cooldown_until_ms
    assert state["last_stop_event"]["drawdown_raw"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_ignores_panic_marker_without_reconstructed_red():
    bot = make_coin_bot()
    symbol = "A"
    writes = []
    bot._equity_hard_stop_write_latch = (
        lambda pside, payload, symbol=None: writes.append((pside, symbol, payload))
        or "/tmp/hsl_coin_ignored.json"
    )

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history
    bot.get_exchange_time = lambda: 180_000

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"] is None
    assert state["runtime"].red_latched() is False
    assert writes == []


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_ignores_raw_red_pending_panic_marker(caplog):
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["ema_span_minutes"] = 100.0
    writes = []
    bot._equity_hard_stop_write_latch = (
        lambda pside, payload, symbol=None: writes.append((pside, symbol, payload))
        or "/tmp/hsl_coin_raw_pending_ignored.json"
    )

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_500,
                    "minute_timestamp": 120_000,
                    "pside": "long",
                    "symbol": symbol,
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history
    bot.get_exchange_time = lambda: 180_000

    with caplog.at_level(logging.WARNING):
        await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert state["last_stop_event"] is None
    assert state["runtime"].red_latched() is False
    assert writes == []
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ignored historical coin panic marker without reconstructed RED" in message
        and "drawdown_raw=1.000000" in message
        and "drawdown_score=0.019802" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_coin_timeline_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="realized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_open_position_missing_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_open_position_empty_coin_history_uses_current_sample():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_leading_rows_before_first_fill():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_relevant_symbol_fields():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {},
                    "unrealized_pnl_by_coin_pside": {},
                }
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "pnl": 0.0,
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="missing required coin HSL symbol"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_startup_skips_flat_nonpanic_history_missing_upnl():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                },
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    assert bot._equity_hard_stop_coin_initialized is True
    assert bot._runtime_forced_modes == {"long": {}, "short": {}}


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_allows_flat_realized_only_rows():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 60_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "qty": 1.0,
                    "price": 100.0,
                    "pnl": 0.0,
                },
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["last_metrics"]["timestamp_ms"] == 180_000
    assert state["last_metrics"]["tier"] == "green"


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_resets_current_episode_after_nonpanic_flatten():
    bot = make_coin_bot()
    symbol = "A"
    bot.hsl["long"]["red_threshold"] = 0.2
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }
    bot.get_exchange_time = lambda: 300_000
    fill_events = [
        {
            "timestamp": 60_500,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": 180_500,
            "symbol": symbol,
            "pside": "long",
            "action": "decrease",
            "qty": 1.0,
            "price": 80.0,
            "pnl": -20.0,
            "pb_order_type": "close_grid_long",
        },
        {
            "timestamp": 240_500,
            "symbol": symbol,
            "pside": "long",
            "action": "increase",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
    ]

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -30.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 80.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
                {
                    "timestamp": 240_000,
                    "balance": 80.0,
                    "realized_pnl": -20.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -20.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": fill_events,
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["pnl_reset_timestamp_ms"] == 180_501
    assert state["halted"] is False
    assert state["last_stop_event"] is None
    assert state["last_metrics"]["timestamp_ms"] == 300_000
    assert state["last_metrics"]["tier"] == "green"
    assert state["last_metrics"]["drawdown_raw"] == pytest.approx(0.0)
    assert bot._runtime_forced_modes == {"long": {}, "short": {}}


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_upnl_for_carry_in_decrease():
    bot = make_coin_bot()
    symbol = "A"
    bot.positions = {
        symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}
    }

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_history_replay_requires_upnl_for_flat_ambiguous_decrease():
    bot = make_coin_bot()
    symbol = "A"

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 120_000,
                    "balance": 95.0,
                    "realized_pnl": -5.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": -5.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {},
                },
            ],
            "panic_flatten_events": [
                {
                    "timestamp": 120_000,
                    "minute_timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                }
            ],
            "fill_events": [
                {
                    "timestamp": 120_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "qty": 1.0,
                    "price": 95.0,
                    "pnl": -5.0,
                    "pb_order_type": "close_panic_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    with pytest.raises(ValueError, match="unrealized_pnl_by_coin_pside"):
        await bot._equity_hard_stop_initialize_coin_from_history()


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_unresolved_panic_residue_on_restart():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 200_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 121_500,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                }
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 421_500
    assert state["cooldown_unresolved_residue"] is True
    assert state["cooldown_intervention_active"] is False
    assert bot._runtime_forced_modes["long"][symbol] == "panic"

    changed = await bot._equity_hard_stop_handle_coin_position_during_cooldown(
        "long", symbol, 200_000
    )
    assert changed is False
    assert state["halted"] is True


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_manual_cooldown_intervention_on_restart():
    bot = make_coin_bot(policy="manual")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 120_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": -80.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is True
    assert state["cooldown_until_ms"] == 400_000
    assert state["cooldown_unresolved_residue"] is False
    assert state["cooldown_intervention_active"] is True
    assert bot._runtime_forced_modes["long"][symbol] == "manual"


@pytest.mark.asyncio
async def test_coin_hsl_reconstructs_normal_cooldown_intervention_as_override():
    bot = make_coin_bot(policy="normal")
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    bot.get_exchange_time = lambda: 180_000

    async def fake_history(current_balance=None, **kwargs):
        return {
            "timeline": [
                {
                    "timestamp": 60_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
                {
                    "timestamp": 180_000,
                    "balance": 100.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                    "unrealized_pnl_by_coin_pside": {symbol: {"long": 0.0, "short": 0.0}},
                },
            ],
            "panic_flatten_events": [],
            "fill_events": [
                {
                    "timestamp": 100_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "decrease",
                    "pb_order_type": "close_panic_long",
                },
                {
                    "timestamp": 130_000,
                    "symbol": symbol,
                    "pside": "long",
                    "action": "increase",
                    "pb_order_type": "entry_initial_normal_long",
                },
            ],
        }

    bot.get_balance_equity_history = fake_history

    await bot._equity_hard_stop_initialize_coin_from_history()

    state = bot._hsl_coin_state("long", symbol)
    assert state["halted"] is False
    assert state["cooldown_until_ms"] is None
    assert symbol not in bot._runtime_forced_modes["long"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected_mode",
    [
        ("manual", "manual"),
        ("tp_only", "tp_only_with_active_entry_cancellation"),
    ],
)
async def test_coin_hsl_check_preserves_cooldown_policy_forced_mode(policy, expected_mode):
    bot = make_coin_bot(policy=policy)
    symbol = "A"
    bot.positions = {symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}}}
    state = bot._hsl_coin_state("long", symbol)
    state["halted"] = True
    state["cooldown_until_ms"] = 300_000

    await bot._equity_hard_stop_check_coin()

    assert bot._runtime_forced_modes["long"][symbol] == expected_mode


@pytest.mark.asyncio
async def test_coin_hsl_check_tp_only_orange_blocks_flat_initial_entries():
    bot = make_coin_bot()
    open_symbol = "A"
    flat_symbol = "B"
    bot.positions = {
        open_symbol: {"long": {"size": 1.0, "price": 100.0}, "short": {"size": 0.0}},
        flat_symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0}},
    }

    async def calc_upnl(pside=None, symbol=None):
        return -20.0

    bot._calc_upnl_sum_strict = calc_upnl
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", open_symbol, 180_000)
    bot._equity_hard_stop_prime_coin_runtime_for_replay("long", flat_symbol, 180_000)

    out = await bot._equity_hard_stop_check_coin()

    assert out[f"long:{open_symbol}"]["tier"] == "orange"
    assert out[f"long:{flat_symbol}"]["tier"] == "orange"
    assert (
        bot._runtime_forced_modes["long"][open_symbol]
        == "tp_only_with_active_entry_cancellation"
    )
    assert (
        bot._runtime_forced_modes["long"][flat_symbol]
        == "tp_only_with_active_entry_cancellation"
    )
