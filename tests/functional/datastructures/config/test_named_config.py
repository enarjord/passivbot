from __future__ import annotations

import copy

import pytest

from passivbot.datastructures import StopMode
from passivbot.datastructures.config import NamedConfig


@pytest.fixture
def named_config_dict(tmp_path):
    return {
        "exchange": "binance",
        "api_key_name": "binance_test_key",
        "symbol": "BTCUSDT",
        "live_config_path": tmp_path / "config.json",
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
            "wallet_exposure_limit": 0.05,
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
    }


@pytest.fixture
def complete_named_config_dict(named_config_dict):
    named_config_dict = copy.deepcopy(named_config_dict)
    # Let's update the dictionary with what it's supposed to look like once loaded
    for key in ("short", "long"):
        for skey in ("max_n_entry_orders", "n_close_orders"):
            named_config_dict[key][skey] = round(named_config_dict[key][skey])
    named_config_dict["stop_mode"] = StopMode.NORMAL
    named_config_dict["market_type"] = "futures"
    named_config_dict["max_leverage"] = 25
    named_config_dict["assigned_balance"] = None
    named_config_dict["cross_wallet_pct"] = 1.0
    named_config_dict["profit_trans_pct"] = 0.0
    named_config_dict["last_price_diff_limit"] = 0.3
    named_config_dict["short"]["secondary_allocation"] = float(
        named_config_dict["short"]["secondary_allocation"]
    )
    return named_config_dict


def test_simple_parse_obj(named_config_dict, complete_named_config_dict):
    nc = NamedConfig.parse_obj(named_config_dict)
    assert nc.dict() == complete_named_config_dict


@pytest.mark.parametrize("order_type", ("long", "short"))
@pytest.mark.parametrize("order_type_key", ("max_n_entry_orders", "n_close_orders"))
@pytest.mark.parametrize(
    ("original_value", "rounded_value"),
    (
        (1.4, 1),
        (1.5, 2),
    ),
)
def test_int_rounding(
    named_config_dict,
    complete_named_config_dict,
    order_type,
    order_type_key,
    original_value,
    rounded_value,
):
    named_config_dict[order_type][order_type_key] = original_value
    complete_named_config_dict[order_type][order_type_key] = rounded_value

    nc = NamedConfig.parse_obj(named_config_dict)
    assert nc.dict() == complete_named_config_dict


def test_pre_v6_config_migration(named_config_dict, complete_named_config_dict):
    replacements = (
        ("secondary_pprice_diff", "secondary_grid_spacing"),
        ("secondary_allocation", "secondary_pbr_allocation"),
        ("wallet_exposure_limit", "pbr_limit"),
    )
    for position in ("short", "long"):
        for orig, replacement in replacements:
            named_config_dict[position][replacement] = named_config_dict[position].pop(orig)
    named_config_dict["shrt"] = named_config_dict.pop("short")

    nc = NamedConfig.parse_obj(named_config_dict)
    assert nc.dict() == complete_named_config_dict
