from copy import deepcopy


def detect_flavor(config: dict, template: dict) -> str:
    pb_keys = {
        "user",
        "pnls_max_lookback_days",
        "loss_allowance_pct",
        "stuck_threshold",
        "unstuck_close_pct",
        "TWE_long",
        "TWE_short",
        "universal_live_config",
    }
    if all(k in config for k in pb_keys):
        return "pb_multi"
    required_current = {"bot", "live", "backtest", "optimize"}
    if required_current.issubset(config):
        return "current"
    if (
        "config" in config
        and isinstance(config["config"], dict)
        and required_current.issubset(config["config"])
    ):
        return "nested_current"
    if "bot" in config and "live" in config:
        return "live_only"
    return "unknown"


def build_base_config_from_flavor(config: dict, template: dict, flavor: str, verbose: bool) -> dict:
    import config_utils as legacy

    if flavor == "pb_multi":
        return legacy._build_from_pb_multi(config, template)
    if flavor == "current":
        return deepcopy(config)
    if flavor == "nested_current":
        return deepcopy(config["config"])
    if flavor == "live_only":
        return legacy._build_from_live_only(config, template)
    raise Exception("failed to format config: unknown flavor")
