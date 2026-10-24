"""Translate public strategy paths at the unchanged native protocol boundary.

Native parameter names and optimizer IDs remain Rust-owned. Only authored local
paths are projected; this module does not authorize alternate config spellings.
"""

from copy import deepcopy


STRATEGY_NATIVE_TO_PUBLIC_PATHS: dict[str, dict[tuple[str, ...], tuple[str, ...]]] = {
    "trailing_martingale": {
        ("entry", "ema_span_0"): ("entry", "ema_span_0_minutes"),
        ("entry", "ema_span_1"): ("entry", "ema_span_1_minutes"),
        ("volatility_ema_span_1m",): ("volatility_ema_span_minutes",),
        ("volatility_ema_span_1h",): ("volatility_ema_span_1h_candles",),
        ("entry", "initial_qty_pct"): ("entry", "initial_qty_fraction"),
        ("entry", "initial_ema_dist"): ("entry", "initial_ema_distance_fraction"),
        ("entry", "threshold_base_pct"): ("entry", "threshold_base_fraction"),
        ("entry", "retracement_base_pct"): ("entry", "retracement_base_fraction"),
        ("close", "qty_pct"): ("close", "qty_fraction"),
        ("close", "threshold_base_pct"): ("close", "threshold_base_fraction"),
        ("close", "retracement_base_pct"): ("close", "retracement_base_fraction"),
    },
    "ema_anchor": {
        ("ema_span_0",): ("ema_span_0_minutes",),
        ("ema_span_1",): ("ema_span_1_minutes",),
        ("offset_volatility_ema_span_1m",): ("offset_volatility_ema_span_minutes",),
        ("offset_volatility_ema_span_1h",): ("offset_volatility_ema_span_1h_candles",),
        ("base_qty_pct",): ("base_qty_fraction",),
        ("offset",): ("offset_fraction",),
    },
}


def native_strategy_path_to_public(strategy_kind, path):
    path = tuple(path)
    return STRATEGY_NATIVE_TO_PUBLIC_PATHS.get(strategy_kind, {}).get(path, path)


def public_strategy_path_to_native(strategy_kind, path):
    path = tuple(path)
    native_to_public = STRATEGY_NATIVE_TO_PUBLIC_PATHS.get(strategy_kind, {})
    inverse = {public: native for native, public in native_to_public.items()}
    if len(inverse) != len(native_to_public):
        raise ValueError(f"ambiguous public strategy paths for {strategy_kind}")
    return inverse.get(path, path)


def project_strategy_spec(strategy_kind, native_spec):
    """Copy native metadata, projecting only its authored config paths."""
    result = deepcopy(native_spec)
    paths = set()
    for parameter in [*result.get("parameters", []), *result.get("fixed_parameters", [])]:
        path = parameter.get("config_path", ())
        if len(path) >= 3 and path[0] == "strategy":
            parameter["config_path"] = [
                *path[:2], *native_strategy_path_to_public(strategy_kind, path[2:]),
            ]
            projected = tuple(parameter["config_path"])
            if projected in paths:
                raise ValueError(f"conflicting public strategy path {'.'.join(projected)}")
            paths.add(projected)
    return result


def _project_strategy_tree(strategy_kind, source, project_path):
    if not STRATEGY_NATIVE_TO_PUBLIC_PATHS.get(strategy_kind):
        return deepcopy(source)
    result = {}

    def insert(path, value):
        target = result
        for part in path[:-1]:
            if part not in target:
                target[part] = {}
            if not isinstance(target[part], dict):
                raise ValueError(f"conflicting strategy path {'.'.join(path)}")
            target = target[part]
        if path[-1] in target:
            raise ValueError(f"conflicting strategy path {'.'.join(path)}")
        target[path[-1]] = deepcopy(value)

    def visit(value, prefix=()):
        if isinstance(value, dict) and value:
            for name, child in value.items():
                visit(child, (*prefix, name))
        elif prefix:
            insert(project_path(strategy_kind, prefix), value)

    visit(source)
    return result


def project_native_strategy_side(strategy_kind, native_side):
    """Copy a native strategy tree into its sole authored public spelling."""
    return _project_strategy_tree(strategy_kind, native_side, native_strategy_path_to_public)


def lower_runtime_strategy_side(strategy_kind, public_side):
    """Copy a merged public strategy into the unchanged native parameter tree."""
    return _project_strategy_tree(strategy_kind, public_side, public_strategy_path_to_native)
