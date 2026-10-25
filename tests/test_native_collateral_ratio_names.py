"""Native collateral names retain leverage, debt sentinels and requiredness."""

import numpy as np
import pytest

from test_limit_order_fill_buffer import _native_args


@pytest.fixture(autouse=True)
def require_real_native_extension():
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(native, "__is_stub__", False)
    verify_loaded_runtime_extension()


@pytest.mark.parametrize("retired,current,value", [
    ("btc_collateral_cap", "btc_collateral_target_ratio", 1.375),
    ("btc_collateral_ltv_cap", "max_usd_debt_to_wallet_value_ratio", .625),
    ("max_usd_debt_to_equity_ratio", "max_usd_debt_to_wallet_value_ratio", .625),
])
@pytest.mark.parametrize("presence", ["retired_only", "both", "null_retired"])
def test_retired_collateral_fields_cannot_disappear_into_current_defaults(
    retired, current, value, presence,
):
    import passivbot_rust as native

    args = _native_args()
    if presence == "retired_only":
        args[-1].pop(current)
    args[-1][retired] = None if presence == "null_retired" else value
    with pytest.raises(ValueError, match=f"unsupported retired native backtest field '{retired}'"):
        native.run_backtest(*args)


def test_current_collateral_target_remains_required():
    import passivbot_rust as native

    args = _native_args()
    args[-1].pop("btc_collateral_target_ratio")
    with pytest.raises(ValueError, match="btc_collateral_target_ratio"):
        native.run_backtest(*args)


def test_omitted_current_debt_ceiling_retains_the_explicit_null_replay():
    import passivbot_rust as native

    args = _native_args()
    args[-1].update(btc_collateral_target_ratio=1.375, max_usd_debt_to_wallet_value_ratio=None)
    explicit = native.run_backtest(*args)
    assert len(explicit[0]) > 0
    args[-1].pop("max_usd_debt_to_wallet_value_ratio")
    omitted = native.run_backtest(*args)
    np.testing.assert_array_equal(explicit[0], omitted[0])
    np.testing.assert_array_equal(explicit[1], omitted[1])
    assert explicit[2] == omitted[2]


@pytest.mark.parametrize("loader", ["prepare", "clean", "migrate"])
def test_unpublished_equity_spelling_is_not_a_config_alias(loader):
    from config import prepare_config
    from config.schema import get_expanded_config_prototype
    from tools.clean_config import cleanup_config
    from tools.migrate_config_v9 import migrate

    config = get_expanded_config_prototype()
    config["backtest"]["max_usd_debt_to_equity_ratio"] = config["backtest"].pop(
        "max_usd_debt_to_wallet_value_ratio"
    )
    load = {"prepare": prepare_config, "clean": cleanup_config, "migrate": migrate}[loader]
    with pytest.raises(ValueError, match="max_usd_debt_to_equity_ratio"):
        load(config)


def test_collateral_ratio_metadata_uses_the_same_marked_wallet_denominator():
    from config.field_registry import get_field_registry

    fields = get_field_registry()
    definitions = [fields.find(("backtest", name)) for name in (
        "btc_collateral_target_ratio", "max_usd_debt_to_wallet_value_ratio",
    )]
    assert definitions[0].denominator == definitions[1].denominator
    assert all(field.unit == "ratio" for field in definitions)
    assert "excluding unrealized trading PnL" in definitions[0].denominator


@pytest.mark.parametrize("text,value", [("none", None), ("-0.125", -.125), ("0.625", .625)])
def test_wallet_debt_cli_retains_nullable_and_numeric_values(text, value):
    from argparse import ArgumentParser
    from config.schema import get_expanded_config_prototype
    from config_utils import add_config_arguments

    parser = ArgumentParser(allow_abbrev=False)
    add_config_arguments(parser, get_expanded_config_prototype(), command="backtest", help_all=True)
    current = "backtest.max_usd_debt_to_wallet_value_ratio"
    parsed = parser.parse_args(["--" + current, text])
    assert getattr(parsed, current) == value
    with pytest.raises(SystemExit):
        parser.parse_args(["--backtest.max_usd_debt_to_equity_ratio", text])
