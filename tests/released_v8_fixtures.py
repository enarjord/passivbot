"""Public released-v8 authoring inputs, independent of the current v9 schema."""

from config.migrations.released_v8 import get_released_v8_defaults


def released_v8_policy():
    """Use the released trailing-martingale template with retained metric formulas.

    Most policy tests do not concern metric formula migration. Authoring only
    the released nonweighted metrics keeps those tests independent of the
    separate required choices for retired weighted and daily-worst underwater metrics.
    Full-template migration has its own unmodified-input regression coverage.
    """
    config = get_released_v8_defaults(strategy_kind="trailing_martingale")
    config["backtest"]["visible_metrics"] = [
        name for name in config["backtest"]["visible_metrics"] if not name.endswith("_w") and not name.startswith("strategy_eq_underwater_pct_")
    ]
    config["optimize"]["scoring"] = [
        objective for objective in config["optimize"]["scoring"]
        if not objective["metric"].endswith("_w") and not objective["metric"].startswith("strategy_eq_underwater_pct_")
    ]
    return config
