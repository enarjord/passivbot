from copy import deepcopy
import math

import numpy as np
import pytest

import candlestick_manager as candles
from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS


@pytest.fixture
def cm(tmp_path, monkeypatch):
    # This suite exercises the independent Python arithmetic/cache contract.
    monkeypatch.setattr(candles, "_RUST_EMA_LAST", None)
    return CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))


def rows(values, *, period=ONE_MIN_MS, start=ONE_MIN_MS):
    return np.array([(start + i * period, value, value, value, value, 1.0)
                     for i, value in enumerate(values)], dtype=CANDLE_DTYPE)


def value(cm, arr, *, span=3.0, metric="close", period=ONE_MIN_MS, commit=True):
    return cm._ema_from_completed_window("COIN", (metric, span, str(period)), arr,
                                         span, int(arr[-1]["ts"]), 100, commit=commit)


def test_normalized_short_bootstrap_then_scalar_suffix_once(cm):
    first = rows([2.0, 4.0])
    assert value(cm, first) == pytest.approx(10.0 / 3.0)
    following = rows([2.0, 4.0, 8.0])
    assert value(cm, following) == pytest.approx(17.0 / 3.0)
    assert value(cm, following) == pytest.approx(17.0 / 3.0)
    assert cm._ema(following["c"], 3.0) == pytest.approx(6.0)


@pytest.mark.parametrize("span", [0.5, 1.0, 2.25, 3.0, 10.75])
def test_bootstrap_caps_history_and_preserves_fractional_span(cm, span):
    data = np.arange(1.0, 100.0)
    tail = data[-max(1, math.ceil(2.0 * span)):]
    decay = 1.0 - min(1.0, 2.0 / (span + 1.0))
    weights = decay ** np.arange(len(tail) - 1, -1, -1)
    assert cm._ema(data, span) == pytest.approx(float(np.sum(weights * tail) / weights.sum()))
    assert cm._ema(np.asarray([2.0]), span) == 2.0


def test_correction_removal_and_changed_span_rebootstrap(cm):
    original = rows([2.0, 4.0, 8.0])
    value(cm, original[:2])
    assert value(cm, original) == pytest.approx(17.0 / 3.0)
    corrected = rows([3.0, 4.0, 8.0])
    assert value(cm, corrected) == pytest.approx(cm._ema(corrected["c"], 3.0))
    removed = corrected[1:]
    assert value(cm, removed) == pytest.approx(cm._ema(removed["c"], 3.0))
    assert value(cm, original, span=2.25) == pytest.approx(cm._ema(original["c"], 2.25))


def test_preview_is_cloned_and_does_not_advance_committed_state(cm):
    value(cm, rows([2.0, 4.0]))
    saved = deepcopy(cm._ema_cache)
    source = cm._ema_inputs["COIN"][("close", 3.0, str(ONE_MIN_MS))].copy()
    assert value(cm, rows([2.0, 4.0, 8.0]), commit=False) == pytest.approx(17.0 / 3.0)
    assert cm._ema_cache == saved
    np.testing.assert_array_equal(cm._ema_inputs["COIN"][("close", 3.0, str(ONE_MIN_MS))], source)
    assert value(cm, rows([2.0, 4.0, 10.0])) == pytest.approx(20.0 / 3.0)


def test_restart_uses_declared_available_history_approximation(cm, tmp_path):
    data = rows([2.0, 4.0, 8.0])
    value(cm, data[:2])
    running = value(cm, data)
    fresh = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path / "restart"))
    restarted = value(fresh, data)
    assert running == pytest.approx(17.0 / 3.0)
    assert restarted == pytest.approx(6.0)
    assert value(fresh, data) == restarted


def test_persisted_append_retains_prior_but_old_correction_invalidates(cm, monkeypatch):
    data = rows([2.0, 4.0])
    cm._cache["COIN"] = data.copy()
    value(cm, data)
    monkeypatch.setattr(cm, "_save_range_incremental", lambda *args, **kwargs: None)
    appended = rows([2.0, 4.0, 8.0])
    cm._persist_batch("COIN", appended[-1:], merge_cache=True)
    assert "COIN" in cm._ema_cache
    assert value(cm, appended) == pytest.approx(17.0 / 3.0)
    cm._persist_batch("COIN", rows([3.0]), merge_cache=True)
    assert "COIN" not in cm._ema_cache
    assert "COIN" not in cm._ema_inputs


def test_shared_index_correction_invalidates_processed_history(cm, tmp_path):
    data = rows([2.0, 4.0])
    cm._persist_batch("COIN", data, merge_cache=True)
    value(cm, data)
    writer = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    writer._persist_batch("COIN", rows([3.0]), merge_cache=True)
    cm._ensure_symbol_index("COIN")
    assert "COIN" not in cm._ema_cache
    assert "COIN" not in cm._ema_inputs


def test_shared_metadata_refresh_preserves_committed_ema(cm, tmp_path):
    data = rows([2.0, 4.0])
    cm._persist_batch("COIN", data, merge_cache=True)
    value(cm, data)
    saved = deepcopy(cm._ema_cache)
    writer = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    index = writer._ensure_symbol_index("COIN")
    index["meta"]["last_refresh_ms"] = 123456
    writer._save_index("COIN")
    cm._ensure_symbol_index("COIN")
    assert cm._ema_cache == saved


@pytest.mark.asyncio
async def test_latest_reader_keeps_scalar_across_append_and_ttl_expiry(cm, monkeypatch):
    data = rows([2.0, 4.0, 8.0])
    now = [int(data[1]["ts"]) + ONE_MIN_MS]
    monkeypatch.setattr(cm, "_now_ms", lambda: now[0])
    async def read(symbol, *, start_ts, end_ts, **kwargs):
        return data[(data["ts"] >= start_ts) & (data["ts"] <= end_ts)].copy()
    monkeypatch.setattr(cm, "get_candles", read)
    assert await cm.get_latest_ema_close("COIN", 3.0, max_age_ms=1) == pytest.approx(10.0 / 3.0)
    now[0] += ONE_MIN_MS
    assert await cm.get_latest_ema_close("COIN", 3.0, max_age_ms=1) == pytest.approx(17.0 / 3.0)
    now[0] += 10
    assert await cm.get_latest_ema_close("COIN", 3.0, max_age_ms=1) == pytest.approx(17.0 / 3.0)


@pytest.mark.asyncio
async def test_hourly_reader_uses_completed_buckets_and_one_span_readiness(cm, monkeypatch):
    hour = 60 * ONE_MIN_MS
    data = rows([2.0, 4.0, 8.0, 100.0], period=hour, start=0)
    now = [3 * hour + 10 * ONE_MIN_MS]
    monkeypatch.setattr(cm, "_now_ms", lambda: now[0])
    async def read(symbol, *, start_ts, end_ts, **kwargs):
        return data[(data["ts"] >= start_ts) & (data["ts"] <= end_ts)].copy()
    monkeypatch.setattr(cm, "get_candles", read)
    assert await cm.get_latest_ema_volume("COIN", 3.0, timeframe="1h") == 1.0
    assert await cm.get_latest_ema_close("COIN", 3.0, timeframe="1h") == pytest.approx(6.0)
    now[0] = 4 * hour + 10 * ONE_MIN_MS
    assert await cm.get_latest_ema_close("COIN", 3.0, timeframe="1h") == pytest.approx(53.0)
