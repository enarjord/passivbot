"""Durable listing facts survive independent candle-cache recovery."""

from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from test_age_fact_recovery import BASE, DAY, _cache_config, _save_cache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    import backtest as bt
    monkeypatch.setattr(bt, "HLCVS_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(bt, "coin_to_symbol", lambda coin, *_a, **_kw: coin)
    return bt, _cache_config(tmp_path, (60.0, 0.0))


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["warmup", "force"])
@pytest.mark.parametrize("unknown", [None, 0.0])
async def test_rebuild_preserves_verified_fact_in_returned_and_saved_metadata(cache, monkeypatch, reason, unknown):
    bt, config = cache
    known = BASE - 61 * DAY
    directory, arrays, old = _save_cache(bt, config, known)
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = unknown
    original = deepcopy(fresh)
    materialize = AsyncMock(return_value=(fresh, arrays[2], arrays[0], arrays[1]))
    monkeypatch.setattr(bt, "try_prepare_hlcvs_v2_local", materialize)
    monkeypatch.setattr(bt, "compute_per_coin_warmup_minutes", lambda _: {"__default__": 0})
    monkeypatch.setattr(bt, "compute_backtest_warmup_minutes", lambda _: 8 if reason == "warmup" else 0)
    result = await bt.prepare_hlcvs_mss(config, "binance", force_refetch_gaps=reason == "force")
    materialize.assert_awaited_once()
    assert result[2]["AAA"]["inception_timestamp_ms"] == known
    assert fresh == original  # The acquisition result is not mutated in place.
    disk = json.loads((directory / "market_specific_settings.json").read_text())
    assert disk["AAA"]["inception_timestamp_ms"] == known


@pytest.mark.parametrize("change,expected", [
    ({"inception_timestamp_ms": None}, BASE - 61 * DAY),
    ({"inception_timestamp_ms": 0.0}, BASE - 61 * DAY),
    ({"inception_timestamp_ms": BASE - 62 * DAY}, BASE - 62 * DAY),
    ({"inception_timestamp_ms": None, "exchange": "bybit"}, None),
    ({"inception_timestamp_ms": None, "symbol": "AAA/USD:AAA"}, None),
    ({"inception_timestamp_ms": None, "ohlcv_source": "bybit"}, None),
])
def test_direct_save_preserves_only_matching_verified_facts_without_mutation(cache, change, expected):
    bt, config = cache
    directory, arrays, old = _save_cache(bt, config, BASE - 61 * DAY)
    fresh = deepcopy(old)
    fresh["AAA"].update(change)
    original = deepcopy(fresh)
    bt.save_coins_hlcvs_to_cache(config, ["AAA"], arrays[0], "binance", fresh, arrays[1], arrays[2], force_overwrite=True)
    disk = json.loads((directory / "market_specific_settings.json").read_text())
    assert disk["AAA"]["inception_timestamp_ms"] == expected
    assert fresh == original


def test_fresh_positive_correction_replaces_sufficient_cache_without_force(cache):
    bt, config = cache
    directory, arrays, old = _save_cache(bt, config, BASE - 61 * DAY)
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = BASE - 62 * DAY
    original = deepcopy(fresh)
    result = bt.save_coins_hlcvs_to_cache(
        config, ["AAA"], arrays[0], "binance", fresh, arrays[1], arrays[2], warmup_minutes=7,
    )
    assert result == directory
    assert bt.load_coins_hlcvs_from_cache(config, "binance")[3] == original
    assert json.loads((directory / "market_specific_settings.json").read_text()) == original
    assert fresh == original


@pytest.mark.parametrize("damage", ["hash", "identity", "venue", "path", "symlink", "missing", "invalid_fact", "invalid_shape"])
def test_untrusted_cached_fact_is_never_borrowed(cache, damage):
    bt, config = cache
    directory, arrays, old = _save_cache(bt, config, BASE - 61 * DAY)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mss_path = directory / "market_specific_settings.json"
    if damage == "hash":
        bad = deepcopy(old)
        bad["AAA"]["inception_timestamp_ms"] += 1
        mss_path.write_text(json.dumps(bad))
    elif damage == "identity":
        manifest["config_hash"] = "different-cache-identity"
    elif damage == "venue":
        manifest["requested"]["exchange"] = "bybit"
    elif damage == "path":
        manifest["files"]["market_specific_settings"]["path"] = "../outside.json"
    elif damage == "missing":
        mss_path.unlink()
    elif damage == "symlink":
        outside = directory.parent / "outside.json"
        outside.write_bytes(mss_path.read_bytes())
        mss_path.unlink()
        mss_path.symlink_to(outside)
    else:
        from hlcvs_manifest import hash_json_value
        bad = [] if damage == "invalid_shape" else deepcopy(old)
        if damage == "invalid_fact":
            bad["AAA"]["inception_timestamp_ms"] = True
        mss_path.write_text(json.dumps(bad))
        manifest["files"]["market_specific_settings"]["sha256"] = hash_json_value(bad)
    manifest_path.write_text(json.dumps(manifest))
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = None
    bt.save_coins_hlcvs_to_cache(config, ["AAA"], arrays[0], "binance", fresh, arrays[1], arrays[2], force_overwrite=True)
    assert json.loads(mss_path.read_text())["AAA"]["inception_timestamp_ms"] is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan")])
def test_malformed_new_fact_is_not_repaired_from_cached_fact(cache, bad):
    bt, config = cache
    _, arrays, old = _save_cache(bt, config, BASE)
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = bad
    with pytest.raises(ValueError, match="inception_timestamp_ms"):
        bt.save_coins_hlcvs_to_cache(config, ["AAA"], arrays[0], "binance", fresh, arrays[1], arrays[2], force_overwrite=True)


def test_metadata_reader_uses_the_verified_declared_artifacts(cache):
    bt, config = cache
    directory, _, old = _save_cache(bt, config, BASE)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact, replacement in [("coins", "labels.json"), ("market_specific_settings", "facts.json")]:
        entry = manifest["files"][artifact]
        (directory / entry["path"]).rename(directory / replacement)
        entry["path"] = replacement
    manifest_path.write_text(json.dumps(manifest))
    assert bt.load_coins_hlcvs_from_cache(config, "binance")[3] == old
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = None
    assert bt._preserve_cached_inception_facts(config, "binance", fresh) == old


def test_valid_metadata_fact_can_survive_independent_array_corruption(cache):
    bt, config = cache
    directory, arrays, old = _save_cache(bt, config, BASE)
    (directory / "hlcvs.npy").write_bytes(b"invalid old candle payload")
    fresh = deepcopy(old)
    fresh["AAA"]["inception_timestamp_ms"] = None
    bt.save_coins_hlcvs_to_cache(config, ["AAA"], arrays[0], "binance", fresh, arrays[1], arrays[2], force_overwrite=True)
    assert bt.load_coins_hlcvs_from_cache(config, "binance")[3]["AAA"]["inception_timestamp_ms"] == BASE


@pytest.mark.parametrize("reason", ["unknown", "warmup", "bad_metadata"])
def test_cache_rejection_does_not_materialize_arrays(cache, monkeypatch, reason):
    import hlcvs_manifest as hm
    bt, config = cache
    directory, _, _ = _save_cache(bt, config, None if reason == "unknown" else BASE)
    if reason == "bad_metadata":
        (directory / "coins.json").write_text('["different"]')
    def unexpected(*_a, **_kw):
        pytest.fail("rejected cache must not read/decompress array artifacts")
    monkeypatch.setattr(hm, "_verify_array_artifact", unexpected)
    if reason == "bad_metadata":
        with pytest.raises(hm.HlcvsManifestError, match="hash mismatch"):
            bt.load_coins_hlcvs_from_cache(config, "binance")
    else:
        assert bt.load_coins_hlcvs_from_cache(config, "binance", 8 if reason == "warmup" else 0) is None


def test_reusable_cache_still_verifies_every_array(cache, monkeypatch):
    import hlcvs_manifest as hm
    bt, config = cache
    directory, _, _ = _save_cache(bt, config, BASE)
    verified = []
    real = hm._verify_array_artifact
    def record(*args):
        verified.append(args[1])
        return real(*args)
    monkeypatch.setattr(hm, "_verify_array_artifact", record)
    assert bt.load_coins_hlcvs_from_cache(config, "binance") is not None
    assert verified == ["hlcvs", "btc_usd_prices", "timestamps"]
    (directory / "hlcvs.npy").write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError, hm.HlcvsManifestError)):
        bt.load_coins_hlcvs_from_cache(config, "binance")
