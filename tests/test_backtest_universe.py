from copy import deepcopy

import pytest

from backtest import get_cache_hash
from backtest_universe import effective_backtest_data_coins
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
