"""
Tests that pin `build_backtest_payload` as a pure function over its `mss`
argument (i.e. it must not mutate the caller's dict) and that the fix for
the re-entrance bug documented in
docs/plans/2026-04-11-build-backtest-payload-mss-mutation-design.md holds.

Synthetic fixtures, millisecond runtime. Deliberately do NOT call
`execute_backtest` — the Rust engine is invariant under the refactor, and
these tests need to run in a tight loop when the refactor lands.
"""

from copy import deepcopy

import numpy as np

from backtest import build_backtest_payload
from config_utils import get_template_config


def _base_config(candle_interval_minutes: int = 1) -> dict:
    cfg = get_template_config()
    cfg["backtest"]["exchanges"] = ["binance"]
    cfg["backtest"]["coins"] = {"binance": ["BTC"]}
    cfg["backtest"]["candle_interval_minutes"] = candle_interval_minutes
    cfg["backtest"]["filter_by_min_effective_cost"] = False
    cfg["backtest"]["start_date"] = "2021-01-01"
    cfg["backtest"]["end_date"] = "2021-01-02"
    cfg["backtest"]["maker_fee_override"] = None
    cfg["backtest"]["taker_fee_override"] = None
    cfg["live"]["warmup_ratio"] = 0.0
    cfg["live"]["max_warmup_minutes"] = 0
    cfg["live"]["hedge_mode"] = False
    return cfg


def _base_mss(start_ts: int) -> dict:
    return {
        "BTC": {
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "maker": 0.0002,
            "taker": 0.0005,
            "exchange": "binance",
        },
        "__meta__": {
            "requested_start_ts": int(start_ts),
            "requested_start_date": "2021-01-01",
            "warmup_minutes_requested": 0,
        },
    }


def _synthetic_1m_hlcvs(n_minutes: int, start_ts: int):
    """Return (hlcvs, btc_usd_prices, timestamps) for `n_minutes` of 1-min data."""
    timestamps = np.arange(
        start_ts, start_ts + n_minutes * 60_000, 60_000, dtype=np.int64
    )
    hlcvs = np.zeros((n_minutes, 1, 4), dtype=np.float64)
    for i in range(n_minutes):
        base = 100.0 + (i % 10) * 0.1
        hlcvs[i, 0, 0] = base + 0.5  # high
        hlcvs[i, 0, 1] = base - 0.5  # low
        hlcvs[i, 0, 2] = base        # close
        hlcvs[i, 0, 3] = 1.0         # volume
    btc_usd_prices = np.full(n_minutes, 20_000.0, dtype=np.float64)
    return hlcvs, btc_usd_prices, timestamps


def test_build_backtest_payload_does_not_mutate_mss():
    """Pin the function as a pure consumer of `mss`. Covers the root cause
    of the re-entrance bug: callers that reuse `mss` across calls must see
    an unchanged dict after each call."""
    start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC, aligned to 2m/5m/etc.
    n_minutes = 60
    config = _base_config(candle_interval_minutes=2)
    mss = _base_mss(start_ts)
    hlcvs, btc, timestamps = _synthetic_1m_hlcvs(n_minutes, start_ts)

    snapshot = deepcopy(mss)
    build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    assert mss == snapshot, (
        "build_backtest_payload mutated its `mss` argument. "
        "Expected no side effects on the caller's dict."
    )


def test_build_backtest_payload_is_reentrant_on_same_mss():
    """Regression test for docs/plans/2026-04-11-build-backtest-payload-mss-mutation-design.md.

    Before the fix: second call reads `data_interval_minutes=2` from the
    mutated `mss`, skips the aggregation branch, and returns `n_minutes`
    bars instead of `n_minutes / 2`. After the fix: both calls aggregate
    identically and produce the same shape.
    """
    start_ts = 1609459200000
    n_minutes = 60
    config = _base_config(candle_interval_minutes=2)
    mss = _base_mss(start_ts)
    hlcvs, btc, timestamps = _synthetic_1m_hlcvs(n_minutes, start_ts)

    payload1 = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)
    payload2 = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    # Both calls should aggregate 1-min → 2-min identically.
    assert payload1.bundle.hlcvs.shape[0] == n_minutes // 2
    assert payload2.bundle.hlcvs.shape[0] == n_minutes // 2
    # And the two bundles should be bar-identical.
    assert payload1.bundle.hlcvs.shape == payload2.bundle.hlcvs.shape
    np.testing.assert_array_equal(
        np.asarray(payload1.bundle.timestamps),
        np.asarray(payload2.bundle.timestamps),
    )


def test_build_backtest_payload_honors_explicit_warmup_provided_override():
    """Document the contract that a caller pre-setting `warmup_minutes_provided`
    in `mss["__meta__"]` is honored — not silently recomputed.

    Under the current (buggy) code, src/backtest.py:588-592 unconditionally
    overwrites any pre-set value with the locally-computed one. After the
    fix, the caller-supplied value survives into the bundle.
    """
    start_ts = 1609459200000
    n_minutes = 60
    config = _base_config(candle_interval_minutes=1)
    mss = _base_mss(start_ts)
    # Sentinel value. With requested_start_ts == timestamps[0], the
    # local recomputation at src/backtest.py:591 would yield 0, so the
    # 99 is only visible in the bundle if the override path is live.
    mss["__meta__"]["warmup_minutes_provided"] = 99
    hlcvs, btc, timestamps = _synthetic_1m_hlcvs(n_minutes, start_ts)

    payload = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    assert payload.bundle.meta["warmup_minutes_provided"] == 99


def test_build_backtest_payload_propagates_metrics_only_flag():
    start_ts = 1609459200000
    n_minutes = 60
    config = _base_config(candle_interval_minutes=1)
    mss = _base_mss(start_ts)
    hlcvs, btc, timestamps = _synthetic_1m_hlcvs(n_minutes, start_ts)

    payload = build_backtest_payload(
        hlcvs,
        mss,
        config,
        "binance",
        btc,
        timestamps,
        metrics_only=True,
    )

    assert payload.backtest_params["metrics_only"] is True


def test_build_backtest_payload_aggregation_recomputes_effective_start_ts_over_stale_mss():
    """Pin that `build_backtest_payload` recomputes `effective_start_timestamp_ms`
    from the post-aggregation timestamps even when the caller pre-set a stale
    1-minute value in `mss["__meta__"]["effective_start_ts"]`.

    Upstream callers in src/hlcv_preparation.py and src/downloader.py pre-set
    `effective_start_ts` from 1-minute timestamps. If the config uses
    `candle_interval_minutes > 1` and the data start is not aligned to the
    aggregation boundary, `align_and_aggregate_hlcvs` trims leading bars and
    the post-aggregation first timestamp differs from the caller's pre-set
    value. This test exercises that scenario by starting at 00:01:00 UTC
    (unaligned to a 2m boundary) so `offset_bars >= 1`.
    """
    # 2021-01-01 00:01:00 UTC, *not* aligned to a 2m boundary.
    start_ts = 1609459260000
    n_minutes = 60
    config = _base_config(candle_interval_minutes=2)
    mss = _base_mss(start_ts)
    # Caller pre-sets the stale, pre-aggregation 1m start. Upstream
    # hlcv_preparation.py / downloader.py do exactly this.
    mss["__meta__"]["effective_start_ts"] = start_ts
    hlcvs, btc, timestamps = _synthetic_1m_hlcvs(n_minutes, start_ts)

    payload = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    post_agg_first_ts = int(payload.bundle.timestamps[0])
    # Sanity: aggregation actually trimmed at least one bar — otherwise the
    # test isn't exercising the drift scenario.
    assert post_agg_first_ts > start_ts
    # The bundle must reflect the post-aggregation start, not the caller's
    # stale pre-set.
    assert payload.bundle.meta["effective_start_timestamp_ms"] == post_agg_first_ts
    assert payload.bundle.meta["effective_start_timestamp_ms"] != start_ts
