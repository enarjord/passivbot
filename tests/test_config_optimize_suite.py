from copy import deepcopy

from config_utils import format_config, get_template_config


def test_optimize_suite_inherits_when_missing():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # customize backtest suite for visibility
    base["backtest"]["suite"]["enabled"] = True
    base["backtest"]["suite"]["base_label"] = "bt-base"
    base["backtest"]["suite"]["scenarios"] = [{"label": "s1", "start_date": "2022-01-01"}]
    # remove optimize.suite entirely to trigger fallback
    base["optimize"].pop("suite")

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["suite"] == formatted["backtest"]["suite"]
    assert formatted["optimize"]["suite"] is not formatted["backtest"]["suite"]
