"""Prepared dataset age facts share resolution and survive cache/optimizer boundaries."""

from copy import deepcopy
from types import SimpleNamespace
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from test_age_fact_recovery import BASE, DAY, _cache_config, _save_cache


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import backtest as bt

    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_args, **_kwargs: coin)
    monkeypatch.setattr(bt, "compute_per_coin_warmup_minutes", lambda _: {"__default__": 0})
    monkeypatch.setattr(bt, "compute_backtest_warmup_minutes", lambda _: 0)
    materialize = AsyncMock(side_effect=AssertionError("existing candles must be reused"))
    monkeypatch.setattr(bt, "try_prepare_hlcvs_v2_local", materialize)
    monkeypatch.setattr(bt, "prepare_hlcvs_combined", materialize)
    return bt, _cache_config(tmp_path, (60.0, 0.0)), materialize


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "combined"])
async def test_refresh_legacy_prepared_age_without_rebuilding_arrays(runtime, monkeypatch, exchange):
    from optimization.prepared_dataset_identity import materialized_dataset_identity

    bt, config, materialize = runtime
    if exchange == "combined":
        config["backtest"]["exchanges"] = ["binance", "bybit"]
        monkeypatch.setattr(bt, "_load_and_reconcile_combined_sources", AsyncMock(return_value=({}, {})))
    directory, arrays, old = _save_cache(bt, config, BASE - 62 * DAY, legacy=True, exchange=exchange)
    array_inodes = {path.name: path.stat().st_ino for path in directory.glob("*.npy")}
    known = BASE - 7 * DAY
    lookup = AsyncMock(return_value={"AAA": SimpleNamespace(
        timestamp_ms=known, source="test", asset_id="test:aaa", is_provisional=False,
    )})
    monkeypatch.setattr(bt, "resolve_asset_inception", lookup)

    result = await bt.prepare_hlcvs_mss(config, exchange)
    materialize.assert_not_awaited()
    lookup.assert_awaited_once_with({"AAA": [("binance", old["AAA"])]}, allow_remote=True)
    assert result[2]["AAA"]["inception_timestamp_ms"] == known
    assert result[2]["AAA"]["asset_inception_asset_id"] == "test:aaa"
    assert {path.name: path.stat().st_ino for path in directory.glob("*.npy")} == array_inodes
    cached = bt.load_coins_hlcvs_from_cache(config, exchange)
    assert cached[3]["AAA"]["inception_timestamp_ms"] == known
    for actual, expected in zip((cached[2], cached[5], cached[6]), arrays):
        np.testing.assert_array_equal(actual, expected)
    await bt.prepare_hlcvs_mss(config, exchange)
    assert lookup.await_count == 1

    revised = deepcopy(old)
    revised["AAA"]["asset_inception_policy_version"] = bt.ASSET_INCEPTION_POLICY_VERSION
    args = (["AAA"], *arrays)
    assert materialized_dataset_identity(*args, old) != materialized_dataset_identity(*args, revised)


@pytest.mark.asyncio
async def test_legacy_override_stays_immutable_and_fails_closed(runtime, monkeypatch):
    bt, config, materialize = runtime
    directory, _, _ = _save_cache(bt, config, BASE - 62 * DAY, legacy=True)
    config["backtest"].update(
        hlcvs_data_dir=str(directory), hlcvs_data_override_mode="dataset",
    )
    originals = {path.name: path.read_bytes() for path in directory.iterdir()}
    lookup = AsyncMock(side_effect=AssertionError("dataset override pins its age inputs"))
    monkeypatch.setattr(bt, "resolve_asset_inception", lookup)

    result = await bt.prepare_hlcvs_mss(config, "binance")
    assert result[2]["AAA"]["inception_timestamp_ms"] is None
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == originals
    materialize.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepared_refresh_retains_prior_earlier_observation(runtime, monkeypatch):
    bt, config, _ = runtime
    earlier = BASE - 30 * DAY
    prior = {"AAA": {
        "exchange": "binance", "symbol": "AAA/USDT:USDT",
        "inception_timestamp_ms": earlier,
        "asset_inception_policy_version": bt.ASSET_INCEPTION_POLICY_VERSION,
        "asset_inception_asset_id": "test:aaa",
        "asset_inception_source": "venue_candle:binance",
        "asset_inception_provisional": True,
    }}
    lookup = AsyncMock(return_value={"AAA": SimpleNamespace(
        timestamp_ms=earlier + DAY, source="test", asset_id="test:aaa", is_provisional=False,
    )})
    monkeypatch.setattr(bt, "resolve_asset_inception", lookup)
    resolved = await bt._resolve_prepared_inception(config, "binance", ["AAA"], prior)
    assert resolved == prior
    lookup.assert_awaited_once()
    lookup.return_value["AAA"].timestamp_ms = earlier - DAY
    resolved = await bt._resolve_prepared_inception(config, "binance", ["AAA"], prior)
    assert resolved["AAA"]["inception_timestamp_ms"] == earlier - DAY
    assert not resolved["AAA"]["asset_inception_provisional"]


def test_stale_prepared_metadata_writer_preserves_newer_generation(runtime):
    bt, config, _ = runtime
    directory, _, old = _save_cache(bt, config, BASE - 62 * DAY, legacy=True)
    manifest = bt.load_hlcvs_manifest(directory)
    entry = manifest["files"]["market_specific_settings"]
    previous_path = directory / entry["path"]
    entry["path"] = "metadata/settings.json"
    path = directory / entry["path"]
    path.parent.mkdir()
    previous_path.rename(path)
    # A staged rewrite must unlink its hardlink before writing, even for nested paths.
    link = directory.parent / "previous-settings.json"
    link.hardlink_to(path)
    bt.write_hlcvs_manifest(directory, manifest)
    winner = deepcopy(old)
    winner["AAA"].update(
        inception_timestamp_ms=BASE - 64 * DAY,
        asset_inception_policy_version=bt.ASSET_INCEPTION_POLICY_VERSION,
    )
    loser = deepcopy(winner)
    loser["AAA"]["inception_timestamp_ms"] = BASE - 61 * DAY
    bt._publish_refreshed_inception(directory, old, winner)
    bt._publish_refreshed_inception(directory, old, loser)
    assert bt.load_coins_hlcvs_from_cache(config, "binance")[3] == winner
    assert json.loads(link.read_text()) == old


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["listTime", "preMktSwTime"])
async def test_prepared_rebuild_discards_prior_age_after_lifecycle_change(
    runtime, monkeypatch, changed_field,
):
    bt, config, materialize = runtime
    config["backtest"]["exchanges"] = ["okx"]
    directory, arrays, old = _save_cache(bt, config, BASE - 62 * DAY, exchange="okx")
    old["AAA"].update(
        exchange="okx",
        asset_inception_provisional=True,
        asset_inception_source="venue_candle:okx",
        info={"listTime": BASE - 63 * DAY, "preMktSwTime": ""},
    )
    bt.save_coins_hlcvs_to_cache(
        config, ["AAA"], arrays[0], "okx", old, arrays[1], arrays[2], force_overwrite=True,
    )
    replacement = deepcopy(old)
    replacement["AAA"]["info"][changed_field] = BASE - DAY
    replacement["AAA"].pop("inception_timestamp_ms")
    replacement["AAA"].pop("asset_inception_policy_version")
    materialize.side_effect = None
    materialize.return_value = (replacement, arrays[2], arrays[0], arrays[1])
    lookup = AsyncMock(return_value={})
    monkeypatch.setattr(bt, "resolve_asset_inception", lookup)

    result = await bt.prepare_hlcvs_mss(config, "okx", force_refetch_gaps=True)

    lookup.assert_awaited_once_with({"AAA": [("okx", replacement["AAA"])]}, allow_remote=True)
    assert result[2]["AAA"]["inception_timestamp_ms"] is None
    persisted = json.loads((directory / "market_specific_settings.json").read_text())
    assert persisted["AAA"]["inception_timestamp_ms"] is None
