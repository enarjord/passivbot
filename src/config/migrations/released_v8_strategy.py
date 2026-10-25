"""Frozen released-v8 strategy vocabulary for the explicit v9 converter only.

These paths and native optimizer IDs describe released v8.6, not current Rust metadata.
Source: public master 772abadc10a5bbf258a84aa8209a9084699fa1bd.
Ordinary v9 loading must not consult this module or accept these old spellings.
"""

from copy import deepcopy
from types import MappingProxyType


RELEASED_V8_STRATEGY_PATH_RENAMES = MappingProxyType({
    'trailing_martingale': MappingProxyType({
        ('entry', 'ema_span_0'): ('entry', 'ema_span_0_minutes'),
        ('entry', 'ema_span_1'): ('entry', 'ema_span_1_minutes'),
        ('volatility_ema_span_1m',): ('log_range_ema_span_minutes',),
        ('volatility_ema_span_1h',): ('log_range_ema_span_1h_candles',),
        ('entry', 'initial_qty_pct'): ('entry', 'initial_qty_fraction'),
        ('entry', 'initial_ema_dist'): ('entry', 'initial_ema_distance_fraction'),
        ('entry', 'double_down_factor'): ('entry', 'reentry_qty_multiplier'),
        ('entry', 'threshold_we_weight'): ('entry', 'threshold_exposure_ratio_coefficient'),
        ('entry', 'retracement_we_weight'): ('entry', 'retracement_exposure_ratio_coefficient'),
        ('entry', 'threshold_volatility_1m_weight'): ('entry', 'threshold_log_range_1m_coefficient'),
        ('entry', 'threshold_volatility_1h_weight'): ('entry', 'threshold_log_range_1h_coefficient'),
        ('entry', 'retracement_volatility_1m_weight'): ('entry', 'retracement_log_range_1m_coefficient'),
        ('entry', 'retracement_volatility_1h_weight'): ('entry', 'retracement_log_range_1h_coefficient'),
        ('entry', 'threshold_base_pct'): ('entry', 'threshold_base_fraction'),
        ('entry', 'retracement_base_pct'): ('entry', 'retracement_base_fraction'),
        ('close', 'qty_pct'): ('close', 'qty_fraction'),
        ('close', 'threshold_base_pct'): ('close', 'threshold_base_fraction'),
        ('close', 'threshold_we_weight'): ('close', 'threshold_exposure_ratio_coefficient'),
        ('close', 'threshold_volatility_1m_weight'): ('close', 'threshold_log_range_1m_coefficient'),
        ('close', 'threshold_volatility_1h_weight'): ('close', 'threshold_log_range_1h_coefficient'),
        ('close', 'retracement_volatility_1m_weight'): ('close', 'retracement_log_range_1m_coefficient'),
        ('close', 'retracement_volatility_1h_weight'): ('close', 'retracement_log_range_1h_coefficient'),
        ('close', 'retracement_base_pct'): ('close', 'retracement_base_fraction'),
    }),
    'ema_anchor': MappingProxyType({
        ('ema_span_0',): ('ema_span_0_minutes',),
        ('ema_span_1',): ('ema_span_1_minutes',),
        ('offset_volatility_ema_span_1m',): ('offset_log_range_ema_span_minutes',),
        ('offset_volatility_ema_span_1h',): ('offset_log_range_ema_span_1h_candles',),
        ('offset_volatility_1m_weight',): ('offset_log_range_1m_coefficient',),
        ('offset_volatility_1h_weight',): ('offset_log_range_1h_coefficient',),
        ('base_qty_pct',): ('base_qty_fraction',),
        ('offset',): ('offset_fraction',),
        ('entry_double_down_factor',): ('entry_qty_exposure_coefficient',),
        ('offset_psize_weight',): ('inventory_offset_exposure_coefficient',),
    }),
})


# Released NestedParamSeed.name differs from its nested path for these two
# TM parameters. All remaining renamed seeds use the underscore-joined path.
RELEASED_V8_STRATEGY_OPTIMIZER_PATHS = MappingProxyType({
    kind: MappingProxyType({
        (old[-1] if kind == "trailing_martingale" and old in {
            ("entry", "ema_span_0"), ("entry", "ema_span_1"),
        } else "_".join(old)): new
        for old, new in fields.items()
    })
    for kind, fields in RELEASED_V8_STRATEGY_PATH_RENAMES.items()
})


def released_strategy_kind(config, *, default=None):
    """Read authored context without consulting mutable native defaults."""
    value = (config.get("bot", {}).get("portfolio", {}).get("strategy_kind")
             or config.get("live", {}).get("strategy_kind") or default)
    return str(value).strip().lower() if value is not None else None


def released_strategy_selector(path, *, strategy_kind=None):
    """Resolve only the dotted strategy aliases accepted by released v8.

    A bare native optimizer ID was not a fixed-parameter selector. A concrete
    side followed by a native ID was; its meaning depends on the active kind.
    """
    parts = tuple(path)
    for offset, part in enumerate(parts):
        if part == "bot":
            prefix, selected = parts[:offset], parts[offset:]
            break
    else:
        prefix, selected = (), parts
    if selected[:1] and selected[0] in {"long", "short"}:
        selected = ("bot", *selected)
    if (len(selected) == 3 and selected[0] == "bot"
            and selected[1] in {"long", "short"}):
        local = RELEASED_V8_STRATEGY_OPTIMIZER_PATHS.get(strategy_kind, {}).get(selected[2])
        if local is not None:
            return (*prefix, *selected[:2], "strategy", strategy_kind, *local)
    if (len(selected) >= 4 and selected[:1] == ("bot",)
            and selected[1] in {"long", "short", "*"} and selected[2] == "strategy"
            and selected[3] not in {*RELEASED_V8_STRATEGY_PATH_RENAMES, "*"}
            and strategy_kind in RELEASED_V8_STRATEGY_PATH_RENAMES):
        selected = (*selected[:3], strategy_kind, *selected[3:])
    if (len(selected) >= 2 and selected[-1] in {"ema_span_0", "ema_span_1"}
            and (selected[-2] == "trailing_martingale" or (
                selected[-3:-1] == ("strategy", "*")
                and strategy_kind == "trailing_martingale"))):
        selected = (*selected[:-1], "entry", selected[-1])
    return (*prefix, *selected)


def released_strategy_bound_path(path, *, strategy_kind=None):
    """Project a released root flat optimizer ID to its final public path."""
    parts = tuple(path)
    if parts[-3:-1] != ("optimize", "bounds"):
        return parts
    side, separator, native_id = parts[-1].partition("_")
    local = RELEASED_V8_STRATEGY_OPTIMIZER_PATHS.get(strategy_kind, {}).get(native_id)
    if separator and side in {"long", "short"} and local is not None:
        return (*parts[:-1], side, "strategy", strategy_kind, *local)
    return parts


def rewrite_released_strategy_bounds(config, *, strategy_kind=None, record=None):
    """Move valid v8 root optimizer IDs before current metadata interprets bounds."""
    def visit(node, path=()):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))
            return
        if not isinstance(node, dict):
            return
        for key in list(node):
            old = (*path, key)
            new = released_strategy_bound_path(old, strategy_kind=strategy_kind)
            if old != new:
                target = node
                for part in new[len(path):-1]:
                    target = target.setdefault(part, {})
                    if not isinstance(target, dict):
                        raise ValueError(f"{'.'.join(old)} conflicts with {'.'.join(new)}")
                value = node[key]
                if new[-1] in target and target[new[-1]] != value:
                    raise ValueError(f"{'.'.join(old)} conflicts with {'.'.join(new)}")
                target[new[-1]] = deepcopy(value)
                del node[key]
                if record is not None:
                    record(".".join(old), deepcopy(value), {".".join(new): deepcopy(value)},
                           "explicit released-v8 strategy optimizer ID; value unchanged")
            else:
                visit(node[key], old)
    visit(config)
    return config
