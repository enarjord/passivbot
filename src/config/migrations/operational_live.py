"""Explicit released-v8 structural conversion of operational live settings."""

from copy import deepcopy

from config.field_registry import LIVE_OPERATIONAL_PATHS


# Released source names are an explicit conversion boundary, never runtime IDs or
# aliases accepted by authored v9 configuration.
_RELEASED_V8_SOURCE_NAMES = {
    "balance_hysteresis_snap_fraction": "balance_hysteresis_snap_pct",
    "fallback_fee_fraction": "fee_pct_fallback",
    "max_abs_fee_fraction": "fee_pct_sanity_abs_max",
    "limit_order_create_max_market_distance_fraction": "limit_order_create_max_market_dist_pct",
    "order_match_tolerance_fraction": "order_match_tolerance_pct",
    "order_replacement_churn_gate_market_distance_fraction": "order_replacement_churn_gate_market_dist_pct",
}
RELEASED_V8_LIVE_OPERATIONAL_PATHS = {
    _RELEASED_V8_SOURCE_NAMES.get(field_id, field_id): path
    for field_id, path in LIVE_OPERATIONAL_PATHS.items()
}


def operational_live_selector(path):
    """Retarget released full selectors, including nested coin override selectors."""
    for old, current in RELEASED_V8_LIVE_OPERATIONAL_PATHS.items():
        source = f"live.{old}"
        for prefix in ("", "."):
            marker = prefix + source
            start = 0 if not prefix else path.find(marker)
            if start < 0 or (not prefix and not path.startswith(source)):
                continue
            end = start + len(marker)
            if end == len(path) or path[end] == ".":
                return path[:start] + prefix + ".".join(current) + path[end:]
    return path


def move_operational_live(document, *, record, path=""):
    """Move only authored leaves; sparse coin documents never receive defaults."""
    if isinstance(document, list):
        for index, child in enumerate(document):
            move_operational_live(child, record=record, path=f"{path}[{index}]")
        return
    if not isinstance(document, dict):
        return
    live = document.get("live")
    if isinstance(live, dict):
        for old, (_, group, leaf) in RELEASED_V8_LIVE_OPERATIONAL_PATHS.items():
            if old not in live:
                continue
            value = live.pop(old)
            block = live.setdefault(group, {})
            source = f"{path}.live.{old}".lstrip(".")
            target = f"{path}.live.{group}.{leaf}".lstrip(".")
            if not isinstance(block, dict) or (leaf in block and block[leaf] != value):
                raise ValueError(f"{source} conflicts with {target}")
            block[leaf] = deepcopy(value)
            record(
                source,
                value,
                {target: value},
                "grouped released operational live field",
            )
    for key, child in document.items():
        if not str(key).startswith("_"):
            move_operational_live(
                child, record=record, path=f"{path}.{key}".lstrip(".")
            )
