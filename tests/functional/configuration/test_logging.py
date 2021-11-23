from __future__ import annotations

from typing import Any

import pytest

from passivbot import config


@pytest.fixture
def complete_config_dictionary() -> dict[str, dict[str, Any]]:
    return {
        "api_keys": {
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


def test_cli_log_level(complete_config_dictionary):
    logging_config_dict = {"cli": {"level": "error"}}
    complete_config_dictionary["logging"] = logging_config_dict

    loaded = config.BaseConfig.parse_obj(complete_config_dictionary)
    assert isinstance(loaded, config.BaseConfig)
    assert loaded.logging.cli.level == "error"
    loaded_dict = loaded.dict()
    expected_cli_logging_dict = config.LoggingCliConfig().dict()
    expected_cli_logging_dict["level"] = logging_config_dict["cli"]["level"]
    assert loaded_dict["logging"]["cli"] == expected_cli_logging_dict
