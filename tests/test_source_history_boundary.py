"""Listing evidence is independent of retained candle observations and transport errors."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hlcv_preparation as hp
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from simulation_data import OfflineDataError, has_current_offline_proof, simulation_data_policy

START = 1_774_915_200_000
FIRST = START + 630 * 60_000
END = FIRST + 9 * 60_000
SYMBOL = "NEW/USDT:USDT"


def manager(*, listing=FIRST, observed=FIRST, partial=False, source="binance", raw_symbol="NEWUSDT"):
    class Manager:
        exchange = source
        gap_tolerance_ohlcvs_minutes = 0
        cm = None
        markets = {SYMBOL: {"id": "NEWUSDT", "symbol": SYMBOL,
            "info": {"symbol": raw_symbol, "onboardDate": listing}}}
        def get_symbol(self, coin):
            return SYMBOL
        def load_first_timestamp(self, coin):
            return observed
        def update_timestamp_range(self, start_ts, end_ts):
            pass
        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            timestamps = np.arange(FIRST, END + 60_000, 60_000, dtype=np.int64)
            if partial:
                timestamps = timestamps[:-2]
            return pd.DataFrame({"timestamp": timestamps, "high": 101., "low": 99.,
                "close": 100., "volume": 1.})
    return Manager()


def store(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "catalog.sqlite")
    return catalog, OhlcvStore(tmp_path / "ohlcvs", catalog)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unknown", "older_metadata", "late_retention", "partial", "wrong_venue", "wrong_contract"])
async def test_late_response_cannot_alone_prove_inception(tmp_path, monkeypatch, case):
    catalog, raw = store(tmp_path)
    kwargs = {"listing": FIRST}
    if case == "unknown":
        kwargs["listing"] = None
    elif case == "older_metadata":
        kwargs["observed"] = START - 86_400_000
    elif case == "late_retention":
        kwargs["listing"] = START
    elif case == "partial":
        kwargs["partial"] = True
    elif case == "wrong_venue":
        kwargs["source"] = "bybit"
    elif case == "wrong_contract":
        kwargs["raw_symbol"] = "OTHERUSDT"
    om = manager(**kwargs)
    async def first_timestamps(coins, exchange=None):
        return {coin: kwargs.get("observed", FIRST) for coin in coins}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    result = await hp._fetch_coin_range_via_ccxt_into_v2_store(
        om=om, catalog=catalog, store=raw, exchange="binance", coin="NEW", symbol=SYMBOL,
        start_ts=START, end_ts=END, allow_pre_inception_prefix=True,
    )
    assert result.reason != "pre_inception_boundary"
    gaps = catalog.get_gaps("binance", "1m", SYMBOL, START, END)
    assert not any(hp._pre_inception_gap_confirms_discovered_boundary(gap) for gap in gaps)


@pytest.mark.asyncio
async def test_independent_launch_proof_is_reused_offline_without_metadata_queries(tmp_path, monkeypatch):
    catalog, raw = store(tmp_path)
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=END, reason="pre_inception", persistent=True,
        retry_count=3, next_retry_at=0, note="mirrored_from_candlestick_manager")
    om = manager(listing=FIRST + 12_345)
    calls = []
    async def first_timestamps(coins, exchange=None):
        calls.append(exchange)
        return {coin: FIRST for coin in coins}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(om=om, catalog=catalog, store=raw, exchange="binance", coin="NEW",
        symbol=SYMBOL, start_ts=START, end_ts=END, allow_remote_fetch=True,
        local_hit_log_label="hit", remote_fetch_log_label="fetch")
    verified = await hp._resolve_v2_store_range(**args)
    assert verified.timestamps[0] == FIRST
    assert verified.valid.all()
    assert calls
    assert any(g.note == "corroborated_source_history_boundary_v2" for g in
        catalog.get_persistent_gaps("binance", "1m", SYMBOL, START, FIRST - 60_000))
    attempts_before = catalog.list_fetch_attempts("binance", "1m", SYMBOL, START, END)
    async def deny_metadata(*args, **kwargs):
        pytest.fail("current source-bound proof should not query listing metadata again")
    monkeypatch.setattr(hp, "get_first_timestamps_unified", deny_metadata)
    with simulation_data_policy({"backtest": {"offline": True}}):
        again = await hp._resolve_v2_store_range(**args)
    np.testing.assert_array_equal(again.timestamps, verified.timestamps)
    assert catalog.list_fetch_attempts("binance", "1m", SYMBOL, START, END) == attempts_before


@pytest.mark.asyncio
@pytest.mark.parametrize("note", ["authoritative_first_candle_boundary",
    "discovered_first_candle_during_stale_repair", "mirrored_from_candlestick_manager"])
async def test_old_weak_proof_cannot_be_recertified_offline(tmp_path, monkeypatch, note):
    catalog, raw = store(tmp_path)
    timestamps = np.arange(FIRST, END + 60_000, 60_000, dtype=np.int64)
    raw.write_rows("binance", "1m", SYMBOL, timestamps, np.tile([101., 99., 100., 1.], (len(timestamps), 1)))
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=FIRST - 60_000, reason="pre_inception", persistent=True,
        retry_count=3, next_retry_at=0, note=note)
    async def no_listing(coins, exchange=None):
        return {}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", no_listing)
    before = catalog.get_gaps("binance", "1m", SYMBOL, START, END)
    with simulation_data_policy({"backtest": {"offline": True}}), pytest.raises(OfflineDataError):
        await hp._resolve_v2_store_range(om=manager(listing=None), catalog=catalog, store=raw,
            exchange="binance", coin="NEW", symbol=SYMBOL, start_ts=START, end_ts=END,
            allow_remote_fetch=True, local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert catalog.get_gaps("binance", "1m", SYMBOL, START, END) == before


@pytest.mark.parametrize("snapshot", [None, {}, {"version": 1},
    {"version": 2, "boundary_proof_version": True}, {"version": 2, "boundary_proof_version": 1}])
def test_legacy_prepared_proof_cannot_bypass_current_raw_contract(snapshot):
    assert not has_current_offline_proof(snapshot)


def test_current_offline_proof_requires_explicit_version():
    with simulation_data_policy({"backtest": {"offline": True}}):
        from simulation_data import data_manifest
        assert has_current_offline_proof(data_manifest())


@pytest.mark.parametrize("source,field", [("binance", "onboardDate"), ("bybit", "launchTime")])
def test_only_explicit_source_contract_metadata_is_independent(source, field):
    om = manager(source=source)
    om.markets[SYMBOL]["info"] = {"symbol": "NEWUSDT", field: str(FIRST)}
    assert hp._venue_listing_boundary(om, source, "NEW", SYMBOL) == FIRST
    evidence = {"cm_authoritative_start_ts": FIRST, "unified_exchange_first_timestamp": FIRST}
    assert hp._source_history_boundary(evidence) is None
    evidence["venue_listing_timestamp"] = FIRST
    assert hp._source_history_boundary(evidence) == FIRST


@pytest.mark.asyncio
@pytest.mark.parametrize("listing", [None, START, FIRST + 12_345])
async def test_direct_candle_fallback_does_not_hide_unknown_leading_history(monkeypatch, listing):
    candles = np.zeros(10, dtype=[("ts", "i8"), ("o", "f8"), ("h", "f8"),
        ("l", "f8"), ("c", "f8"), ("bv", "f8")])
    candles["ts"] = np.arange(FIRST, END + 60_000, 60_000)
    for field in ("o", "h", "l", "c"):
        candles[field] = 100.
    candles["bv"] = 1.
    async def fetch(*args, **kwargs):
        return candles
    async def first_timestamps(coins, exchange=None):
        return {coin: FIRST for coin in coins}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    om = hp.HLCVManager("binance", cc=SimpleNamespace())
    om.markets = manager(listing=listing).markets
    om.get_symbol = lambda coin: SYMBOL
    om.cm = SimpleNamespace(get_candles=fetch, standardize_gaps=lambda data, **kwargs: data)
    om.update_timestamp_range(START, END)
    result = await om.get_ohlcvs("NEW")
    if listing is not None and listing // 60_000 * 60_000 == FIRST:
        assert result.timestamp.iloc[0] == FIRST
        assert result.valid.all()
    else:
        assert result.empty


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "NaN", "Infinity", "123.5", {}])
def test_invalid_launch_metadata_is_not_a_boundary(value):
    assert hp._venue_listing_boundary(manager(listing=value), "binance", "NEW", SYMBOL) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("offline", [False, True])
async def test_older_genuine_rows_invalidate_a_later_catalog_boundary(tmp_path, monkeypatch, offline):
    catalog, raw = store(tmp_path)
    timestamps = np.arange(FIRST - 60_000, END + 60_000, 60_000, dtype=np.int64)
    raw.write_rows("binance", "1m", SYMBOL, timestamps,
        np.tile([101., 99., 100., 1.], (len(timestamps), 1)))
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=FIRST - 60_000, reason="pre_inception", persistent=True,
        retry_count=0, note="verified_source_history_boundary_v2")
    async def first_timestamps(coins, exchange=None):
        return {coin: FIRST for coin in coins}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(om=manager(), catalog=catalog, store=raw, exchange="binance", coin="NEW",
        symbol=SYMBOL, start_ts=START, end_ts=END, allow_remote_fetch=False,
        local_hit_log_label="hit", remote_fetch_log_label="fetch")
    with simulation_data_policy({"backtest": {"offline": offline}}):
        if offline:
            with pytest.raises(OfflineDataError):
                await hp._resolve_v2_store_range(**args)
        else:
            assert await hp._resolve_v2_store_range(**args) is None
    rng = raw.read_range("binance", "1m", SYMBOL, FIRST - 60_000, END)
    assert rng.valid.all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_kind", ["coin_only", "unified_observation"])
@pytest.mark.parametrize("observed", [FIRST, END + 86_400_000])
async def test_local_v2_request_is_not_cropped_by_unverified_first_observation(
    tmp_path, monkeypatch, cache_kind, observed,
):
    from config.schema import get_expanded_config_prototype
    from config.universe import set_side_universe_sources
    from utils import ts_to_date
    monkeypatch.chdir(tmp_path)
    config = get_expanded_config_prototype()
    config["bot"]["long"]["enabled"] = True
    requested_end = START + 86_400_000
    config["backtest"].update(start_date=ts_to_date(START), end_date=ts_to_date(requested_end))
    set_side_universe_sources(config, "approved_coins", {"long": ["NEW"], "short": []})
    set_side_universe_sources(config, "ignored_coins", {"long": [], "short": []})
    class LocalManager:
        cc = None
        async def load_markets(self):
            pass
        def has_coin(self, coin):
            return True
        def load_first_timestamp(self, coin):
            # A legacy coin-only guess must not even be consulted as authority.
            return observed if cache_kind == "coin_only" else None
        def get_symbol(self, coin):
            return SYMBOL
        async def aclose(self):
            pass
    async def first_timestamps(coins, exchange=None):
        return {coin: observed if cache_kind == "unified_observation" else 0 for coin in coins}
    seen = []
    async def resolve(**kwargs):
        seen.append((kwargs["start_ts"], kwargs["end_ts"]))
        return None
    monkeypatch.setattr(hp, "HLCVManager", lambda *args, **kwargs: LocalManager())
    monkeypatch.setattr(hp, "compute_backtest_warmup_minutes", lambda config: 2)
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    monkeypatch.setattr(hp, "_resolve_v2_store_range", resolve)
    # Older genuine rows already exist, including the authored warmup start.
    # Even a weak observation beyond END must not skip their validation.
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    raw = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    timestamps = np.arange(START - 120_000, requested_end + 60_000, 60_000, dtype=np.int64)
    raw.write_rows("binance", "1m", SYMBOL, timestamps,
        np.tile([101., 99., 100., 1.], (len(timestamps), 1)))
    assert raw.read_range("binance", "1m", SYMBOL, START - 120_000, requested_end).valid.all()
    assert await hp.try_prepare_hlcvs_v2_local(config, "binance") is None
    assert seen == [(START - 120_000, requested_end)]


@pytest.mark.asyncio
async def test_independent_future_listing_limits_transport_without_inventing_coverage(tmp_path):
    catalog, raw = store(tmp_path)
    om = manager(listing=END + 86_400_000)
    om.get_binance_archive_client = lambda: pytest.fail("known future listing needs no old archives")
    rng = raw.read_range("binance", "1m", SYMBOL, START, END)
    assert await hp._fetch_binance_archives_into_v2_store(
        om=om, catalog=catalog, store=raw, exchange="binance", coin="NEW", symbol=SYMBOL,
        timestamps=rng.timestamps, valid=rng.valid,
    ) == 0
    assert not raw.read_range("binance", "1m", SYMBOL, START, END).valid.any()
    assert catalog.get_gaps("binance", "1m", SYMBOL, START, END) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("proof", ["catalog", "independent", "unknown"])
async def test_verified_listing_prefix_preserves_later_internal_gap_window(tmp_path, monkeypatch, proof):
    catalog, raw = store(tmp_path)
    first, later, end = START + 120_000, START + 7 * 60_000, START + 12 * 60_000
    timestamps = START + np.array([2, 3, 7, 8, 9, 10, 11, 12], dtype=np.int64) * 60_000
    raw.write_rows("binance", "1m", SYMBOL, timestamps,
        np.tile([101., 99., 100., 1.], (len(timestamps), 1)))
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
        start_ts=START, end_ts=first - 60_000, reason="pre_inception", persistent=True,
        retry_count=0, next_retry_at=4_102_444_800_000,
        note="verified_source_history_boundary_v2" if proof == "catalog" else "old_observation")
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
        start_ts=START + 4 * 60_000, end_ts=later - 60_000, reason="internal_gap",
        persistent=True, retry_count=0, next_retry_at=4_102_444_800_000, note="unavailable")
    om = manager(listing=first if proof == "independent" else None)
    async def first_timestamps(coins, exchange=None):
        return {coin: first for coin in coins}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    result = await hp._resolve_v2_store_range(om=om, catalog=catalog, store=raw,
        exchange="binance", coin="NEW", symbol=SYMBOL, start_ts=START, end_ts=end,
        allow_remote_fetch=False, allow_partial_window=True,
        local_hit_log_label="hit", remote_fetch_log_label="fetch")
    if proof == "unknown":
        assert result is None
    else:
        assert result.timestamps[0] == later
        assert result.timestamps[-1] == end
        assert result.valid.all()
        gaps = catalog.get_persistent_gaps("binance", "1m", SYMBOL, START, later)
        assert any(gap.reason == "pre_inception" and gap.end_ts == first - 60_000
                   and hp._pre_inception_gap_confirms_discovered_boundary(gap) for gap in gaps)
        assert any(gap.reason == "internal_gap" for gap in gaps)
