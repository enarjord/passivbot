"""Released-v8 setup quantities convert directly to final public names."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_setup_quantities_preserve_values_and_final_report_paths(kind):
    source = get_released_v8_defaults(strategy_kind=kind)
    source["optimize"]["scoring"] = [
        row for row in source["optimize"]["scoring"] if not row["metric"].endswith("_w") and not row["metric"].startswith("strategy_eq_underwater_pct_")
    ]
    source["backtest"].update(
        btc_collateral_cap=2.375, btc_collateral_ltv_cap=0.625,
        maker_fee_override="-0.000125", taker_fee_override=None,
        liquidation_threshold=0.07125,
    )
    source["live"]["warmup_ratio"] = 0.2
    source["live"]["max_warmup_minutes"] = 0.5
    source["backtest"]["scenarios"] = [{"label": "authored", "overrides": {
        "backtest.btc_collateral_cap": 1.625,
        "backtest.btc_collateral_ltv_cap": None,
        "backtest.maker_fee_override": "-0.000375",
        "backtest.liquidation_threshold": 0.03125,
        "live.warmup_ratio": 0.625,
    }}]
    source["optimize"]["fixed_runtime_overrides"] = {
        "backtest.taker_fee_override": "0.000275",
        "backtest.btc_collateral_ltv_cap": -0.125,
        "live.warmup_ratio": 0.375,
    }
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    target = result.config
    expected = {
        "btc_collateral_target_ratio": 2.375,
        "max_usd_debt_to_wallet_value_ratio": 0.625,
        "maker_fee_override_fraction": -0.000125,
        "taker_fee_override_fraction": None,
        "equity_floor_fraction": 0.07125,
    }
    assert {key: target["backtest"][key] for key in expected} == expected
    assert target["bot"]["portfolio"]["warmup"] == {
        "activation_span_fraction": 0.2, "max_minutes": 0.5,
    }
    assert target["backtest"]["scenarios"][0]["overrides"] == {
        "backtest.btc_collateral_target_ratio": 1.625,
        "backtest.max_usd_debt_to_wallet_value_ratio": None,
        "backtest.maker_fee_override_fraction": -0.000375,
        "backtest.equity_floor_fraction": 0.03125,
        "bot.portfolio.warmup.activation_span_fraction": 0.625,
    }
    assert target["optimize"]["fixed_runtime_overrides"] == {
        "backtest.taker_fee_override_fraction": 0.000275,
        "backtest.max_usd_debt_to_wallet_value_ratio": -0.125,
        "bot.portfolio.warmup.activation_span_fraction": 0.375,
    }
    report_targets = {
        key for row in result.changes if isinstance(row["new"], dict) for key in row["new"]
    }
    assert {f"backtest.{key}" for key in expected} <= report_targets
    assert "bot.portfolio.warmup.activation_span_fraction" in report_targets
    loaded = prepare_config(target, verbose=False)
    assert {key: loaded["backtest"][key] for key in expected} == expected
    assert migrate(target).config == target
    assert migrate(target).changes == []
