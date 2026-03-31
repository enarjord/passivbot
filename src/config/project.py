from copy import deepcopy

from .transform_log import record_transform


_TARGET_SECTION_MAP = {
    "canonical": ("backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"),
    "live": ("bot", "coin_overrides", "live", "logging", "monitor"),
    "backtest": ("backtest", "bot", "coin_overrides", "live", "logging"),
    "optimize": ("backtest", "bot", "coin_overrides", "live", "logging", "optimize"),
    "monitor": ("live", "logging", "monitor"),
}


def project_config(config: dict, target: str, *, record_step: bool = True) -> dict:
    normalized_target = str(target).strip().lower()
    if normalized_target not in _TARGET_SECTION_MAP:
        allowed = ", ".join(sorted(_TARGET_SECTION_MAP))
        raise ValueError(f"target must be one of {{{allowed}}}, got {target!r}")

    result = {}
    for section in _TARGET_SECTION_MAP[normalized_target]:
        if section in config:
            result[section] = deepcopy(config[section])
    for key in ("_coins_sources", "_raw", "_transform_log"):
        if key in config:
            result[key] = deepcopy(config[key])

    if record_step:
        record_transform(result, "project_config", {"target": normalized_target})
    return result
