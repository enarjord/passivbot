from copy import deepcopy

import pytest

import backtest
from backtest import get_cache_hash
from backtest_universe import effective_backtest_data_coins, normalize_backtest_coin
from config import prepare_config
from config_utils import get_template_config


def _base_config() -> dict:
    cfg = get_template_config()
    cfg["backtest"]["exchanges"] = ["binance"]
    cfg["backtest"]["start_date"] = "2021-01-01"
    cfg["backtest"]["end_date"] = "2021-01-02"
    cfg["live"]["approved_coins"] = {
        "long": ["A", "B", "C"],
        "short": ["A"],
    }
    cfg["live"]["ignored_coins"] = {"long": [], "short": []}
    cfg["bot"]["long"]["total_wallet_exposure_limit"] = 0.0
    cfg["bot"]["long"]["n_positions"] = 3
    cfg["bot"]["short"]["total_wallet_exposure_limit"] = 1.0
    cfg["bot"]["short"]["n_positions"] = 1
    return cfg


def test_effective_backtest_data_coins_ignores_disabled_side():
    cfg = _base_config()

    assert effective_backtest_data_coins(cfg) == ["A"]


def test_backtest_universe_preserves_exchange_qualified_market_identity():
    assert normalize_backtest_coin("bitget::ABCUSDT") == "bitget::ABCUSDT"


def test_backtest_universe_preserves_exact_ccxt_market_identity():
    assert normalize_backtest_coin("1000ABC/USDT:USDT") == "1000ABC/USDT:USDT"


def test_backtest_universe_preserves_native_market_id():
    assert normalize_backtest_coin("1000ABCUSDT") == "1000ABCUSDT"


def test_backtest_universe_preserves_hyphenated_native_market_id():
    assert normalize_backtest_coin("BTC-USDT-SWAP") == "BTC-USDT-SWAP"


def test_backtest_universe_preserves_suffix_bearing_native_market_id():
    assert normalize_backtest_coin("HOTUSDTM") == "HOTUSDTM"
    assert normalize_backtest_coin("XUSDTM") == "XUSDTM"


@pytest.mark.parametrize("identifier", ["1000ABC"])
def test_backtest_universe_retains_legacy_unqualified_canonical_keys(identifier):
    assert normalize_backtest_coin(identifier) == "ABC"


def test_backtest_universe_does_not_use_global_symbol_label_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    (cache_dir / "symbol_to_coin_map.json").write_text(
        '{"ABC": "venue:ABC"}', encoding="utf-8"
    )

    assert normalize_backtest_coin("ABC") == "ABC"


def test_effective_backtest_data_coins_supports_canonical_grouped_bot_config():
    cfg = prepare_config(
        _base_config(),
        verbose=False,
        target="canonical",
        runtime=None,
    )

    assert "total_wallet_exposure_limit" not in cfg["bot"]["long"]
    assert "n_positions" not in cfg["bot"]["short"]
    assert cfg["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert cfg["bot"]["short"]["risk"]["n_positions"] == 1
    assert effective_backtest_data_coins(cfg) == ["A"]


def test_hlcvs_cache_hash_changes_when_disabled_side_becomes_enabled():
    disabled_long = _base_config()
    enabled_long = deepcopy(disabled_long)
    enabled_long["bot"]["long"]["total_wallet_exposure_limit"] = 1.0

    assert effective_backtest_data_coins(disabled_long) == ["A"]
    assert effective_backtest_data_coins(enabled_long) == ["A", "B", "C"]
    assert get_cache_hash(disabled_long, "binance") != get_cache_hash(enabled_long, "binance")


def test_single_exchange_cache_hash_tracks_preparation_algorithm(monkeypatch):
    cfg = _base_config()
    first = get_cache_hash(cfg, "binance")
    monkeypatch.setattr(
        backtest,
        "HLCV_PREPARATION_ALGORITHM_VERSION",
        backtest.HLCV_PREPARATION_ALGORITHM_VERSION + 1,
    )

    assert get_cache_hash(cfg, "binance") != first


def test_hlcvs_cache_hash_tracks_resolved_market_identity(monkeypatch):
    cfg = _base_config()
    resolved_symbol = {"value": "A/USDT:USDT"}

    monkeypatch.setattr(
        backtest,
        "coin_to_symbol",
        lambda coin, exchange, verbose=False: resolved_symbol["value"],
    )
    first = get_cache_hash(cfg, "binance")
    resolved_symbol["value"] = "1000A/USDT:USDT"

    assert get_cache_hash(cfg, "binance") != first


@pytest.mark.asyncio
async def test_combined_sources_refresh_before_hlcvs_cache_lookup(monkeypatch):
    cfg = _base_config()
    cfg["backtest"]["exchanges"] = ["binance", "bybit"]
    cfg["backtest"]["coin_sources"] = {"AUSDT": "bitget"}
    cfg["backtest"]["market_settings_sources"] = {"AUSDT": "kucoin"}
    refreshed = False

    async def refresh_sources(forced, settings, coins, configured):
        nonlocal refreshed
        assert forced == {"AUSDT": "bitget"}
        assert settings == {"AUSDT": "kucoinfutures"}
        assert coins == ["A"]
        assert configured == ["binanceusdm", "bybit"]
        refreshed = True
        return {"A": "bitget"}, {"A": "kucoinfutures"}

    def cache_lookup(config, exchange, warmup_minutes):
        assert refreshed
        assert config["backtest"]["coin_sources"] == {"A": "bitget"}
        assert config["backtest"]["market_settings_sources"] == {
            "A": "kucoinfutures"
        }
        return None

    class PreparationReached(Exception):
        pass

    async def stop_after_cache_gate(*args, **kwargs):
        raise PreparationReached

    monkeypatch.setattr(backtest, "_load_and_reconcile_combined_sources", refresh_sources)
    monkeypatch.setattr(backtest, "load_hlcvs_data_override", lambda *args: None)
    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", cache_lookup)
    monkeypatch.setattr(backtest, "prepare_hlcvs_combined", stop_after_cache_gate)

    with pytest.raises(PreparationReached):
        await backtest.prepare_hlcvs_mss(cfg, "combined")


def test_hlcvs_cache_hash_changes_for_canonical_grouped_side_enablement():
    disabled_long = prepare_config(
        _base_config(),
        verbose=False,
        target="canonical",
        runtime=None,
    )
    enabled_long = deepcopy(disabled_long)
    enabled_long["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0

    assert effective_backtest_data_coins(disabled_long) == ["A"]
    assert effective_backtest_data_coins(enabled_long) == ["A", "B", "C"]
    assert get_cache_hash(disabled_long, "binance") != get_cache_hash(enabled_long, "binance")


def test_effective_backtest_data_coins_rejects_missing_approved_side():
    cfg = _base_config()
    cfg["live"]["approved_coins"] = {"short": ["A"]}

    with pytest.raises(KeyError, match="live\\.approved_coins\\.long"):
        effective_backtest_data_coins(cfg)


def test_effective_backtest_data_coins_rejects_null_approved_side():
    cfg = _base_config()
    cfg["bot"]["long"]["total_wallet_exposure_limit"] = 1.0
    cfg["live"]["approved_coins"]["long"] = None

    with pytest.raises(TypeError, match="approved coin sides"):
        effective_backtest_data_coins(cfg)
