from copy import deepcopy

from config_utils import format_config, get_template_config


def test_optimize_suite_is_ignored_and_removed(caplog):
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # customize backtest suite for visibility
    base["backtest"]["suite"]["enabled"] = True
    base["backtest"]["suite"]["base_label"] = "bt-base"
    base["backtest"]["suite"]["scenarios"] = [{"label": "s1", "start_date": "2022-01-01"}]
    base["optimize"]["suite"] = {"enabled": True, "aggregate": {"default": "median"}}

    formatted = format_config(deepcopy(base), verbose=False)

    assert "suite" not in formatted.get("optimize", {})
    assert formatted["backtest"]["suite"]["enabled"] is True
    assert formatted["backtest"]["suite"]["base_label"] == "bt-base"
    assert formatted["backtest"]["suite"]["scenarios"] == [{"label": "s1", "start_date": "2022-01-01"}]
    assert any("optimize.suite" in rec.message for rec in caplog.records)


def test_suite_aggregate_preserves_metric_keys():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["suite"]["aggregate"]["mdg_usd"] = "median"
    base["backtest"]["suite"]["aggregate"]["sharpe_ratio"] = "std"
    formatted = format_config(deepcopy(base), verbose=False)
    assert formatted["backtest"]["suite"]["aggregate"]["mdg_usd"] == "median"
    assert formatted["backtest"]["suite"]["aggregate"]["sharpe_ratio"] == "std"
