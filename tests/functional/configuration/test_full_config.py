import json

from passivbot import config


def test_single_config_file(tmp_path):
    config_dict = {
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
    config_file = tmp_path / "example-config.json"
    config_file.write_text(json.dumps(config_dict, indent=2))

    loaded = config.PassivBotConfig.parse_file(config_file)
    assert isinstance(loaded, config.PassivBotConfig)
    assert "account-1" in loaded.api_keys
    assert "account-2" in loaded.api_keys
    assert "config-1" in loaded.configs
    assert "config-2" in loaded.configs
    assert "BTCUSDT" in loaded.symbols
    assert "ETHUSDT" in loaded.symbols
