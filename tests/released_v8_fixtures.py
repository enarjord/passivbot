"""Public released-v8 authoring inputs, independent of the current v9 schema."""

from config.migrations.released_v8 import get_released_v8_defaults


def released_v8_policy():
    """Use the released trailing-martingale template with nonweighted metrics.

    Most policy tests do not concern metric formula migration. Authoring only
    the released nonweighted metrics keeps those tests independent of the
    separate required choices for the public template's retired `_w` metrics.
    Full-template migration has its own unmodified-input regression coverage.
    """
    config = get_released_v8_defaults(strategy_kind="trailing_martingale")
    config["backtest"]["visible_metrics"] = [
        name for name in config["backtest"]["visible_metrics"] if not name.endswith("_w")
    ]
    config["optimize"]["scoring"] = [
        objective for objective in config["optimize"]["scoring"]
        if not objective["metric"].endswith("_w")
    ]
    return config
