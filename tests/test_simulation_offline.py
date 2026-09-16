"""Offline simulation must succeed with sockets denied and fail before any download."""
import asyncio
import json
import os
import socket
from types import SimpleNamespace

import numpy as np
import pytest

import hlcv_preparation as hp
import procedures
import utils
from config.schema import get_template_config
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from simulation_data import (
    OfflineDataError, data_manifest, is_offline, simulation_data_policy, simulation_data_scope,
)

OFFLINE = {"backtest": {"offline": True}}
START = 1704067200000
SYMBOL = "BTC/USDT:USDT"


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("offline preparation attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def metadata(root):
    cache = root / "caches"
    market = {
        "id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT",
        "settle": "USDT", "swap": True, "linear": True, "active": True,
        "type": "swap", "spot": False, "future": False, "contract": True,
        "maker": 0.0002, "taker": 0.0005, "contractSize": 1.0,
        "precision": {"amount": 0.001, "price": 0.1},
        "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
    }
    (cache / "binance").mkdir(parents=True)
    files = {
        "binance/markets.json": {SYMBOL: market},
        "binance/first_timestamps.json": {"BTC": START - 86400000},
        "first_ohlcv_timestamps_unified.json": {"BTC": START - 86400000},
        "first_ohlcv_timestamps_unified_exchange_specific.json": {"BTC": {"binanceusdm": START - 86400000}},
        "first_ohlcv_timestamps_unified_exchange_specific_symbols.json": {"BTC": {"binanceusdm": SYMBOL}},
    }
    for name, value in files.items():
        path = cache / name
        path.write_text(json.dumps(value))
        os.utime(path, (1, 1))
    (cache / "first_ohlcv_timestamps_unified.version").write_text(str(procedures.FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION))
    return cache


@pytest.mark.asyncio
async def test_old_metadata_is_used_and_fingerprinted(tmp_path, monkeypatch):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    with simulation_data_policy(OFFLINE):
        assert SYMBOL in await utils.load_markets("binanceusdm", max_age_ms=0)
        assert await procedures.get_first_timestamps_unified(["BTC"], exchange="binance") == {"BTC": START - 86400000}
        assert "caches/binance/markets.json" in data_manifest()["metadata"]
        assert len(data_manifest()["metadata"]["caches/binance/markets.json"]["sha256"]) == 64
    assert not is_offline()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "bad json", "{}", '{"x": null}'])
async def test_bad_markets_fail_without_remote_fallback(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "caches/binance/markets.json"
    path.parent.mkdir(parents=True)
    if content is not None:
        path.write_text(content)
    with simulation_data_policy(OFFLINE), pytest.raises(OfflineDataError, match="markets.json"):
        await utils.load_markets("binance")


@pytest.mark.asyncio
async def test_missing_or_obsolete_listing_cache_fails(tmp_path, monkeypatch):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    with simulation_data_policy(OFFLINE):
        await utils.load_markets("binance")
        path = tmp_path / "caches/first_ohlcv_timestamps_unified_exchange_specific.json"
        original = path.read_text()
        path.write_text("{}")
        with pytest.raises(OfflineDataError, match="listing timestamps"):
            await procedures.get_first_timestamps_unified(["BTC"], exchange="binance")
        path.write_text(original)
        (tmp_path / "caches/first_ohlcv_timestamps_unified.version").unlink()
        with pytest.raises(OfflineDataError, match="resolver version"):
            await procedures.get_first_timestamps_unified(["BTC"], exchange="binance")


@pytest.mark.asyncio
async def test_policy_is_task_local_and_resets_on_failure():
    entered = asyncio.Event()
    release = asyncio.Event()
    @simulation_data_scope
    async def offline_job(config):
        assert is_offline()
        entered.set()
        await release.wait()
        raise RuntimeError("test")
    task = asyncio.create_task(offline_job(OFFLINE))
    await entered.wait()
    assert not is_offline()
    release.set()
    with pytest.raises(RuntimeError):
        await task
    assert not is_offline()


@pytest.mark.asyncio
async def test_live_coin_resolution_ignores_backtest_offline(monkeypatch):
    seen = []
    async def format_impl(*args, **kwargs):
        seen.append(is_offline())
    monkeypatch.setattr(utils, "_format_approved_ignored_coins", format_impl)
    await utils.format_approved_ignored_coins(OFFLINE, ["binance"])
    await utils.format_approved_ignored_coins(OFFLINE, ["binance"], prefer_backtest_coin_source_keys=True)
    assert seen == [False, True]


def raw_store(tmp_path, mask):
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    ts = START + np.arange(10, dtype=np.int64) * 60000
    values = np.tile(np.array([101, 99, 100, 10], dtype=np.float32), (10, 1))
    store.write_rows("binance", "1m", SYMBOL, ts[mask], values[mask])
    return catalog, store, ts


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [None, 0, 9, 4])
async def test_raw_coverage_and_missing_warmup_tail_internal(tmp_path, monkeypatch, missing):
    monkeypatch.chdir(tmp_path)
    mask = np.ones(10, dtype=bool)
    if missing is not None:
        mask[missing] = False
    catalog, store, ts = raw_store(tmp_path, mask)
    om = SimpleNamespace(cm=None, gap_tolerance_ohlcvs_minutes=0, force_refetch_gaps=False)
    kwargs = dict(om=om, catalog=catalog, store=store, legacy_root=None, exchange="binance",
                  coin="BTC", symbol=SYMBOL, start_ts=int(ts[0]), end_ts=int(ts[-1]),
                  allow_remote_fetch=True, local_hit_log_label="test", remote_fetch_log_label="test")
    with simulation_data_policy(OFFLINE):
        if missing is None:
            rng = await hp._resolve_v2_store_range(**kwargs)
            assert rng.valid.all()
            assert data_manifest()["ranges"][0]["valid_rows"] == 10
        else:
            with pytest.raises(OfflineDataError, match="BTC.*including warmup"):
                await hp._resolve_v2_store_range(**kwargs)


@pytest.mark.asyncio
async def test_confirmed_listing_boundary_remains_usable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mask = np.arange(10) >= 2
    catalog, store, ts = raw_store(tmp_path, mask)
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
                     start_ts=int(ts[0]), end_ts=int(ts[1]), reason="pre_inception",
                     persistent=True, retry_count=0, note="discovered_first_candle_during_stale_repair")
    with simulation_data_policy(OFFLINE):
        rng = await hp._resolve_v2_store_range(
            om=SimpleNamespace(cm=None, gap_tolerance_ohlcvs_minutes=0), catalog=catalog,
            store=store, legacy_root=None, exchange="binance", coin="BTC", symbol=SYMBOL,
            start_ts=int(ts[0]), end_ts=int(ts[-1]), allow_remote_fetch=True,
            local_hit_log_label="test", remote_fetch_log_label="test")
        assert rng.timestamps[0] == ts[2]


@pytest.mark.asyncio
async def test_remote_choke_points_are_guarded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    om = hp.HLCVManager("binance", "2024-01-01", "2024-01-02")
    with simulation_data_policy(OFFLINE):
        with pytest.raises(OfflineDataError):
            om.get_binance_archive_client()
        with pytest.raises(OfflineDataError):
            await om.get_first_timestamp("BTC")
        with pytest.raises(OfflineDataError):
            await om.fetch_ohlcvs_for_v2_store("BTC", start_ts=START, end_ts=START+60000)


def test_schema_defaults_to_online():
    assert get_template_config()["backtest"]["offline"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "combined"])
async def test_prepare_dataset_and_reuse_verified_cache(tmp_path, monkeypatch, exchange):
    from backtest import prepare_hlcvs_mss
    from utils import ts_to_date
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = get_template_config()
    config["backtest"].update(offline=True, exchanges=["binance"],
                              start_date=ts_to_date(START), end_date=ts_to_date(START + 9*60000),
                              compress_cache=False, gap_tolerance_ohlcvs_minutes=0)
    config["live"].update(approved_coins={"long": ["BTC"], "short": []},
                          ignored_coins={"long": [], "short": []},
                          minimum_coin_age_days=0, max_warmup_minutes=1)
    catalog = OhlcvCatalog(tmp_path / "caches/ohlcvs/catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches/ohlcvs", catalog)
    ts = START + np.arange(-1, 10, dtype=np.int64)*60000
    store.write_rows("binance", "1m", SYMBOL, ts,
                     np.tile(np.array([101, 99, 100, 10], dtype=np.float32), (len(ts), 1)))
    if exchange == "combined":
        import shutil
        shutil.copytree(tmp_path / "caches/binance", tmp_path / "caches/bybit")
        config["backtest"]["exchanges"] = ["binance", "bybit"]
        for name in ("first_ohlcv_timestamps_unified_exchange_specific.json",
                     "first_ohlcv_timestamps_unified_exchange_specific_symbols.json"):
            path = tmp_path / "caches" / name
            obj = json.loads(path.read_text())
            obj["BTC"]["bybit"] = obj["BTC"]["binanceusdm"]
            path.write_text(json.dumps(obj))
        store.write_rows("bybit", "1m", SYMBOL, ts,
                         np.tile(np.array([101, 99, 100, 10], dtype=np.float32), (len(ts), 1)))
    result = await prepare_hlcvs_mss(config, exchange)
    assert result[0] == ["BTC"]
    assert result[2]["__meta__"]["offline_snapshot"]["ranges"]
    async def no_rebuild(*args, **kwargs):
        pytest.fail("verified cache should be reused")
    monkeypatch.setattr("backtest.try_prepare_hlcvs_v2_local", no_rebuild)
    monkeypatch.setattr("backtest.prepare_hlcvs_combined", no_rebuild)
    again = await prepare_hlcvs_mss(config, exchange)
    np.testing.assert_equal(result[1], again[1])
    config["backtest"]["hlcvs_data_dir"] = str(result[4])
    override = await prepare_hlcvs_mss(config, exchange)
    assert override[2]["__meta__"]["offline_snapshot"] == result[2]["__meta__"]["offline_snapshot"]
    np.testing.assert_equal(result[1], override[1])


@pytest.mark.parametrize("command", ["backtest", "optimize"])
@pytest.mark.parametrize("flag", ["--offline", "--backtest.offline", "--backtest_offline"])
def test_cli_offline_roundtrip(command, flag):
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli, update_config_with_args
    config = get_template_config()
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, project_template_config_for_cli(config, command), command=command)
    update_config_with_args(config, parser.parse_args([flag, "y"]), allowed_keys=keys)
    assert config["backtest"]["offline"] is True
    update_config_with_args(config, parser.parse_args([flag, "n"]), allowed_keys=keys)
    assert config["backtest"]["offline"] is False


def test_offline_not_exposed_on_live_cli():
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, project_template_config_for_cli(get_template_config(), "live"), command="live")
    with pytest.raises(SystemExit):
        parser.parse_args(["--offline", "y"])


@pytest.mark.asyncio
async def test_online_market_expiry_still_refreshes(tmp_path, monkeypatch):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    called = []
    async def remote_markets(reload):
        called.append(reload)
        return {SYMBOL: {"symbol": SYMBOL}}
    monkeypatch.setattr(utils, "create_coin_symbol_map_cache", lambda *a, **k: None)
    await utils.load_markets("binance", max_age_ms=0, cc=SimpleNamespace(load_markets=remote_markets))
    assert called == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("module,function,extra", [
    ("suite_runner", "run_backtest_suite_async", {"disable_plotting": True}),
    ("optimize_suite", "prepare_suite_contexts", {"shared_array_manager": None}),
])
async def test_suite_policy_precedes_metadata_preload(tmp_path, monkeypatch, module, function, extra):
    import importlib
    monkeypatch.chdir(tmp_path)
    config = get_template_config()
    config["backtest"].update(offline=True, exchanges=["binance"])
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    with pytest.raises(OfflineDataError, match="markets.json"):
        await getattr(importlib.import_module(module), function)(
            config, {"scenarios": [{"label": "one"}]}, **extra)
    assert not is_offline()


@pytest.mark.asyncio
async def test_combined_listing_does_not_require_timestamp_on_unlisted_venue(tmp_path, monkeypatch):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "caches/bybit"
    other.mkdir()
    market = json.loads((tmp_path / "caches/binance/markets.json").read_text())[SYMBOL]
    market.update(id="ETHUSDT", symbol="ETH/USDT:USDT", base="ETH")
    (other / "markets.json").write_text(json.dumps({"ETH/USDT:USDT": market}))
    with simulation_data_policy(OFFLINE):
        result = await procedures.get_first_timestamps_unified(["BTC"], exchanges=["binance", "bybit"])
        assert result == {"BTC": START - 86400000}


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_combined_offline_never_swallows_missing_coin(monkeypatch, empty):
    async def unavailable(**kwargs):
        if empty:
            return []
        raise OfflineDataError("missing BTC candles")
    monkeypatch.setattr(hp, "_load_combined_coin_candidates", unavailable)
    with simulation_data_policy(OFFLINE), pytest.raises(OfflineDataError, match="BTC"):
        await hp._resolve_combined_coin(
            coin="BTC", sem=asyncio.Semaphore(1), base_start_ts=START,
            end_ts=START + 600000, first_timestamps_unified={"BTC": START - 60000},
            minimum_coin_age_days=0, min_coin_age_ms=0, tradfi_for_stock_perps=False,
            forced_sources={}, market_settings_sources={}, exchanges_to_consider=["binance"],
            normalization_candidate_exchanges=[], om_dict={"binance": object()},
            per_coin_warmups={}, default_warm=0, force_refetch_gaps=False,
            catalog=None, store=None, legacy_root=None)


@pytest.mark.asyncio
async def test_known_tail_does_not_authorize_unknown_leading_gap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mask = np.ones(10, dtype=bool)
    mask[[0, 9]] = False
    catalog, store, ts = raw_store(tmp_path, mask)
    catalog.mark_gap(exchange="binance", timeframe="1m", symbol=SYMBOL,
                     start_ts=int(ts[-1]), end_ts=int(ts[-1]), reason="trailing_unavailable",
                     persistent=True, retry_count=0, note="confirmed_by_v2_fetch")
    with simulation_data_policy(OFFLINE), pytest.raises(OfflineDataError, match="Unverified offline candle boundary"):
        await hp._resolve_v2_store_range(
            om=SimpleNamespace(cm=None, gap_tolerance_ohlcvs_minutes=0), catalog=catalog,
            store=store, legacy_root=None, exchange="binance", coin="BTC", symbol=SYMBOL,
            start_ts=int(ts[0]), end_ts=int(ts[-1]), allow_remote_fetch=True,
            local_hit_log_label="test", remote_fetch_log_label="test")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("all_missing", [False, True])
async def test_btc_candidates_try_all_local_sources(monkeypatch, use_v2, all_missing):
    import pandas as pd
    attempts = []
    ts = START + np.arange(2, dtype=np.int64) * 60000
    class Manager:
        cc = None
        def __init__(self, exchange, *args, **kwargs):
            self.exchange = exchange
        def update_date_range(self, *args):
            pass
        async def load_markets(self):
            pass
        def has_coin(self, coin):
            return True
        def get_symbol(self, coin):
            return SYMBOL
        async def get_ohlcvs(self, *args, **kwargs):
            attempts.append(self.exchange)
            if self.exchange == "bybit" or all_missing:
                raise OfflineDataError("missing BTC fixture")
            return pd.DataFrame({"timestamp": ts, "close": [100., 101.]})
        async def aclose(self):
            pass
    async def resolve(**kwargs):
        frame = await kwargs["om"].get_ohlcvs("BTC")
        return SimpleNamespace(timestamps=ts, values=np.column_stack([frame.close]*4))
    monkeypatch.setattr(hp, "HLCVManager", Manager)
    monkeypatch.setattr(hp, "_resolve_v2_store_range", resolve)
    with simulation_data_policy(OFFLINE):
        call = hp._load_combined_btc_prices(
            exchanges_to_consider=["bybit"], timestamps=ts,
            effective_start_date="2024-01-01", end_date="2024-01-02",
            gap_tolerance_ohlcvs_minutes=0, force_refetch_gaps=False,
            catalog=None, store=None, legacy_root=None, use_v2_local=use_v2)
        if all_missing:
            with pytest.raises(OfflineDataError, match="BTC benchmark"):
                await call
        else:
            frame, source = await call
            assert source == "binanceusdm"
            assert len(frame) == 2
    assert attempts == ["bybit", "binanceusdm"]


@pytest.mark.asyncio
async def test_single_venue_uses_cached_binance_btc_fallback(tmp_path, monkeypatch):
    from backtest import prepare_hlcvs_mss
    from utils import ts_to_date
    cache = metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    bybit = cache / "bybit"
    bybit.mkdir()
    markets = json.loads((cache / "binance/markets.json").read_text())
    eth = dict(markets[SYMBOL], id="ETHUSDT", base="ETH", symbol="ETH/USDT:USDT")
    markets[eth["symbol"]] = eth
    (bybit / "markets.json").write_text(json.dumps(markets))
    for name, value in (
        ("first_ohlcv_timestamps_unified.json", START-86400000),
        ("first_ohlcv_timestamps_unified_exchange_specific.json", {"bybit": START-86400000}),
        ("first_ohlcv_timestamps_unified_exchange_specific_symbols.json", {"bybit": eth["symbol"]}),
    ):
        path = cache / name
        data = json.loads(path.read_text())
        data["ETH"] = value
        path.write_text(json.dumps(data))
    config = get_template_config()
    config["backtest"].update(offline=True, exchanges=["bybit"], start_date=ts_to_date(START),
                              end_date=ts_to_date(START+9*60000), compress_cache=False)
    config["live"].update(approved_coins={"long": ["ETH", "UNLISTED"], "short": []},
                          minimum_coin_age_days=0, max_warmup_minutes=1)
    catalog = OhlcvCatalog(cache / "ohlcvs/catalog.sqlite")
    store = OhlcvStore(cache / "ohlcvs", catalog)
    ts = START + np.arange(-1, 10, dtype=np.int64)*60000
    values = np.tile(np.array([101, 99, 100, 10], dtype=np.float32), (len(ts), 1))
    store.write_rows("bybit", "1m", eth["symbol"], ts, values)
    store.write_rows("binance", "1m", SYMBOL, ts, values)
    result = await prepare_hlcvs_mss(config, "bybit")
    assert result[2]["__meta__"]["btc_source_exchange"] == "binance"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_source_directory_prefix_requires_listing_provenance(tmp_path, monkeypatch, confirmed):
    from utils import ts_to_date
    cache = metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    if confirmed:
        for name, value in (
            ("first_ohlcv_timestamps_unified.json", START+120000),
            ("first_ohlcv_timestamps_unified_exchange_specific.json", {"binanceusdm": START+120000}),
        ):
            (cache / name).write_text(json.dumps({"BTC": value}))
    source = tmp_path / "legacy"
    day = source / "binance/1m/BTC/2024-01-01.npz"
    day.parent.mkdir(parents=True)
    rows = np.zeros(8, dtype=[("ts", "i8"), ("o", "f8"), ("h", "f8"),
                             ("l", "f8"), ("c", "f8"), ("bv", "f8")])
    rows["ts"] = START + np.arange(2, 10)*60000
    for field in ("o", "h", "l", "c"):
        rows[field] = 100.
    rows["bv"] = 1.
    np.savez(day, candles=rows)
    om = hp.HLCVManager("binance", ts_to_date(START), ts_to_date(START+9*60000),
                        ohlcv_source_dir=str(source))
    om.update_timestamp_range(START, START+9*60000)
    try:
        with simulation_data_policy(OFFLINE):
            if confirmed:
                frame = await om.get_ohlcvs("BTC", source_dir_only=True)
                assert frame.timestamp.iloc[0] == START+120000
                assert frame.valid.all()
            else:
                with pytest.raises(OfflineDataError, match="Unverified source directory prefix"):
                    await om.get_ohlcvs("BTC", source_dir_only=True)
    finally:
        await om.aclose()
        if om.cc:
            await om.cc.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cli_override", [False, True])
async def test_iterative_preload_uses_stale_markets_offline(tmp_path, monkeypatch, cli_override):
    from tools.iterative_backtester import IterativeBacktestSession
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = get_template_config()
    config["backtest"].update(exchanges=["binance"], offline=not cli_override)
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    session = IterativeBacktestSession(
        path, log_level="warning", auto_run=False,
        cli_overrides=["backtest.offline=true"] if cli_override else [])
    loaded = await session._load_config()
    assert loaded["backtest"]["offline"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"exchange": "binance"}, {"exchanges": ["binance"]}])
async def test_unlisted_coins_need_no_inception_metadata(tmp_path, monkeypatch, scope):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    with simulation_data_policy(OFFLINE):
        result = await procedures.get_first_timestamps_unified(["BTC", "UNLISTED"], **scope)
        assert result == {"BTC": START-86400000}
        assert await procedures.get_first_timestamps_unified(["UNLISTED"], **scope) == {}


@pytest.mark.asyncio
async def test_offline_listing_filter_does_not_hide_ambiguous_identity(tmp_path, monkeypatch):
    metadata(tmp_path)
    monkeypatch.chdir(tmp_path)
    def ambiguous(*args, **kwargs):
        raise utils.AmbiguousMarketIdentifier("ambiguous fixture")
    monkeypatch.setattr(procedures, "coin_to_symbol", ambiguous)
    with simulation_data_policy(OFFLINE), pytest.raises(utils.AmbiguousMarketIdentifier):
        await procedures.get_first_timestamps_unified(["BTC"], exchange="binance")
