"""Released-v8 risk units project directly to final v9 names in every authored layer."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from config.optimize_bounds import flatten_optimize_bounds
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_risk_units_preserve_values_sources_and_final_report_paths(tmp_path, kind):
    source = get_released_v8_defaults(strategy_kind=kind)
    source["backtest"]["visible_metrics"] = [
        name for name in source["backtest"]["visible_metrics"]
        if not name.endswith("_w") and not name.startswith("strategy_eq_underwater_pct_")
    ]
    source["optimize"]["scoring"] = [
        row for row in source["optimize"]["scoring"]
        if not row["metric"].endswith("_w") and not row["metric"].startswith("strategy_eq_underwater_pct_")
    ]
    source["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 2.375
    source["bot"]["long"]["hsl"].update(red_threshold=0.23, ema_span_minutes=12.5)
    source["live"]["max_realized_loss_pct"] = 0.125
    source["optimize"]["bounds"]["long"]["risk"]["total_wallet_exposure_limit"] = [0.25, 3.5]
    source["optimize"]["bounds"]["long"]["hsl"].update(
        red_threshold=[0.01, 0.3], ema_span_minutes=[12.5, 22.5],
    )
    source["optimize"]["fixed_params"] = [
        "bot.long.total_wallet_exposure_limit", "bot.long.hsl.ema_span_minutes",
    ]
    source["optimize"]["fixed_runtime_overrides"] = {
        "bot.short.total_wallet_exposure_limit": 1.625,
    }
    source["backtest"]["scenarios"] = [{"label": "authored", "overrides": {
        "bot.long.hsl.red_threshold": 0.31,
        "live.max_realized_loss_pct": 0.0625,
    }}]
    referenced = {"bot": {"long": {
        "wallet_exposure_limit": 0.11,
        "hsl": {"enabled": False, "red_threshold": 0.19},
    }}}
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps(referenced))
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json", "bot": {
        "long": {"wallet_exposure_limit": 0.37},
    }}}
    original = deepcopy(source)
    file_bytes = coin_file.read_bytes()

    result = migrate(source, memory_span_minutes=123.5,
                     acknowledge_realized_loss_gate_memory_change=True,
                     base_config_path=str(tmp_path / "base.json"))
    converted = result.config
    assert source == original
    assert coin_file.read_bytes() == file_bytes
    assert converted["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 2.375
    assert converted["bot"]["long"]["hsl"]["drawdown_threshold_fraction"] == 0.23
    assert converted["bot"]["long"]["hsl"]["drawdown_fraction_ema_span_minutes"] == 12.5
    assert converted["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] == 0.125
    coin = converted["coin_overrides"]["AAA"]
    assert "override_config_path" not in coin
    assert coin["bot"]["long"]["wallet_exposure_limit_ratio"] == 0.37
    assert coin["bot"]["long"]["hsl"]["drawdown_threshold_fraction"] == 0.19
    assert converted["optimize"]["fixed_params"] == [
        "bot.long.risk.total_wallet_exposure_limit_ratio",
        "bot.long.hsl.drawdown_fraction_ema_span_minutes",
    ]
    assert converted["optimize"]["fixed_runtime_overrides"] == {
        "bot.short.risk.total_wallet_exposure_limit_ratio": 1.625,
    }
    assert converted["backtest"]["scenarios"][0]["overrides"] == {
        "bot.long.hsl.drawdown_threshold_fraction": 0.31,
        "bot.portfolio.loss_gate.max_drawdown_fraction": 0.0625,
    }
    native = flatten_optimize_bounds(converted["optimize"]["bounds"], strategy_kind=kind)
    assert native["long_total_wallet_exposure_limit"] == [0.25, 3.5]
    assert native["long_hsl_red_threshold"] == [0.01, 0.3]
    assert native["long_hsl_drawdown_ratio_ema_span_minutes"] == [12.5, 22.5]
    report_targets = {
        name for row in result.changes if isinstance(row["new"], dict)
        for name in row["new"]
    }
    assert {
        "bot.long.risk.total_wallet_exposure_limit_ratio",
        "bot.long.hsl.drawdown_threshold_fraction",
        "bot.long.hsl.drawdown_fraction_ema_span_minutes",
        "bot.portfolio.loss_gate.max_drawdown_fraction",
        "optimize.bounds.long.risk.total_wallet_exposure_limit_ratio",
        "optimize.bounds.long.hsl.drawdown_threshold_fraction",
        "optimize.bounds.long.hsl.drawdown_fraction_ema_span_minutes",
    } <= report_targets
    coin_file.unlink()
    loaded = prepare_config(converted, live_only=True, verbose=False,
                            base_config_path=str(tmp_path / "moved.json"))
    assert loaded["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 2.375
