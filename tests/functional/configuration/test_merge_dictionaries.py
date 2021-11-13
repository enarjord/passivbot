from typing import Any
from typing import Dict

import pytest

from passivbot import config


@pytest.fixture
def complete_dictionary() -> Dict[str, Dict[str, Any]]:
    return {
        "api-keys": {
            "account-1": {
                "exchange": "binance",
                "key": "this is the account-1 key",
                "secret": "this is the account-1 secret",
            },
            "account-2": {
                "exchange": "binance",
                "key": "this is the account-2 key",
                "secret": "this is the account-2 secret",
            },
        },
        "configs": {
            "config-1": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
            "config-2": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
        },
        "symbols": {
            "BTCUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
            "ETHUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
        },
    }


def test_missing_top_level_keys(complete_dictionary):
    keys: Dict[str, Dict[str, Any]] = {
        "api-keys": {
            "account-1": {
                "exchange": "binance",
                "key": "this is the account-1 key",
                "secret": "this is the account-1 secret",
            },
            "account-2": {
                "exchange": "binance",
                "key": "this is the account-2 key",
                "secret": "this is the account-2 secret",
            },
        }
    }
    configs: Dict[str, Any] = {
        "configs": {
            "config-1": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
            "config-2": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
        }
    }

    target_dict: Dict[str, Dict[str, Any]] = {
        "symbols": {
            "BTCUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
            "ETHUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
        },
    }
    assert "api-keys" not in target_dict
    assert "configs" not in target_dict
    config.merge_dictionaries(target_dict, keys, configs)
    assert target_dict == complete_dictionary


def test_nested_dictionaries(complete_dictionary):
    keys: Dict[str, Dict[str, Any]] = {
        "api-keys": {
            "account-1": {
                "exchange": "binance",
                "key": "this is the account-1 key",
                "secret": "this is the account-1 secret",
            },
            "account-2": {
                "exchange": "binance",
                "key": "this is the account-2 key",
                "secret": "this is the account-2 secret",
            },
        }
    }
    configs: Dict[str, Dict[str, Any]] = {
        "configs": {
            "config-1": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
            "config-2": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
        }
    }

    symbols: Dict[str, Dict[str, Any]] = {
        "symbols": {
            "ETHUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
        },
    }

    target_dict: Dict[str, Dict[str, Any]] = {
        "api-keys": {
            "account-1": {
                "exchange": "binance",
                "key": "this is the account-1 key",
                "secret": "this is the account-1 secret",
            },
        },
        "configs": {
            "config-1": {
                "long": {
                    "enabled": True,
                    "eprice_exp_base": 1.3164933633605387,
                    "eprice_pprice_diff": 0.010396126108277413,
                    "grid_span": 0.19126847969076527,
                    "initial_qty_pct": 0.010806866720334485,
                    "markup_range": 0.00867933346187278,
                    "max_n_entry_orders": 10.0,
                    "min_markup": 0.006563436956524566,
                    "n_close_orders": 8.3966954756245,
                    "wallet_exposure_limit": 0.15,
                    "secondary_allocation": 0.5,
                    "secondary_pprice_diff": 0.25837415008453263,
                },
                "short": {
                    "enabled": False,
                    "eprice_exp_base": 1.618034,
                    "eprice_pprice_diff": 0.001,
                    "grid_span": 0.03,
                    "initial_qty_pct": 0.001,
                    "markup_range": 0.004,
                    "max_n_entry_orders": 10,
                    "min_markup": 0.0005,
                    "n_close_orders": 7,
                    "wallet_exposure_limit": 0.5,
                    "secondary_allocation": 0,
                    "secondary_pprice_diff": 0.21,
                },
            },
        },
        "symbols": {
            "BTCUSDT": {
                "config_name": "config-1",
                "key_name": "account-1",
            },
        },
    }
    assert "account-2" not in target_dict["api-keys"]
    assert "config-2" not in target_dict["configs"]
    assert "ETHUSDT" not in target_dict["symbols"]
    config.merge_dictionaries(target_dict, keys, configs, symbols)
    assert target_dict == complete_dictionary


def test_nested_key_override(complete_dictionary):
    symbols: Dict[str, Dict[str, Any]] = {
        "symbols": {
            "ETHUSDT": {
                "config_name": "config-2",
            },
        },
    }

    assert complete_dictionary["symbols"]["ETHUSDT"]["key_name"] == "account-1"
    assert complete_dictionary["symbols"]["ETHUSDT"]["config_name"] == "config-1"
    config.merge_dictionaries(complete_dictionary, symbols)
    assert complete_dictionary["symbols"]["ETHUSDT"]["key_name"] == "account-1"
    assert complete_dictionary["symbols"]["ETHUSDT"]["config_name"] == "config-2"
