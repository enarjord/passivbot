"""Explicit released-config spelling conversion; never used by normal loading.

The values and native IDs are unchanged. This table is deliberately restricted
to public paths, so metric names and native optimizer keys are not aliases.
"""

from copy import deepcopy

from .strategy_paths import STRATEGY_NATIVE_TO_PUBLIC_PATHS


PUBLIC_PATH_RENAMES = {
    ("backtest", "market_order_slippage_pct"): ("backtest", "market_order_slippage_fraction"),
    ("backtest", "limit_order_fill_buffer_pct"): ("backtest", "limit_order_fill_buffer_fraction"),
    ("live", "account", "balance_hysteresis_snap_pct"): ("live", "account", "balance_hysteresis_snap_fraction"),
    ("live", "fills", "fee_pct_fallback"): ("live", "fills", "fallback_fee_fraction"),
    ("live", "fills", "fee_pct_sanity_abs_max"): ("live", "fills", "max_abs_fee_fraction"),
    ("live", "execution", "limit_order_create_max_market_dist_pct"): ("live", "execution", "limit_order_create_max_market_distance_fraction"),
    ("live", "execution", "order_match_tolerance_pct"): ("live", "execution", "order_match_tolerance_fraction"),
    ("live", "execution", "order_replacement_churn_gate_market_dist_pct"): ("live", "execution", "order_replacement_churn_gate_market_distance_fraction"),
    ("bot", "global", "forager", "score_hysteresis_pct"): ("bot", "global", "forager", "score_hysteresis_margin"),
}

SHARED_SIDE_PATH_RENAMES = {
    ("risk", "we_excess_allowance_pct"): ("risk", "we_excess_allowance_fraction"),
    ("unstuck", "close_pct"): ("unstuck", "close_fraction"),
    ("unstuck", "loss_allowance_pct"): ("unstuck", "loss_allowance_fraction"),
}


def _public_path_renames():
    mappings = dict(PUBLIC_PATH_RENAMES)
    for side in ("long", "short"):
        for root in (("bot", side), ("optimize", "bounds", side)):
            for old, new in SHARED_SIDE_PATH_RENAMES.items():
                mappings[(*root, *old)] = (*root, *new)
            for kind, fields in STRATEGY_NATIVE_TO_PUBLIC_PATHS.items():
                for old, new in fields.items():
                    mappings[(*root, "strategy", kind, *old)] = (*root, "strategy", kind, *new)
    return mappings


def released_fixed_selector(selector):
    """Retain the gene membership of released prefix/suffix fixed selectors."""
    from .param_paths import bound_path_matches_selector

    parts = tuple(selector.split("."))
    candidates = set()
    for old, new in _public_path_renames().items():
        if (parts[-1:] == old[-1:] and old[:1] == ("bot",)
                and bound_path_matches_selector(old, parts)):
            candidates.add(".".join((*parts[:-1], new[-1])))
    if len(candidates) > 1:
        raise ValueError(f"released fixed selector {selector!r} has ambiguous renamed authority")
    return next(iter(candidates), ".".join(released_public_path(parts)))


def released_public_path(path):
    """Project a public root/patch/bound selector, preserving unknown components."""
    path = tuple(path)
    # Prefixes may contain a coin identifier or a scenario's overrides wrapper.
    # Require a complete known public path; never replace arbitrary leaf names.
    mappings = _public_path_renames()
    for offset in range(len(path)):
        for old, new in mappings.items():
            candidate = path[offset:offset + len(old)]
            if len(candidate) == len(old) and all(
                expected == "*" or actual == expected
                for actual, expected in zip(candidate, old)
            ):
                projected = tuple(candidate[index] if part == "*" else part
                                  for index, part in enumerate(new))
                return (*path[:offset], *projected, *path[offset + len(old):])
    return path


def rewrite_released_public_names(config, *, path_prefix=(), record=None):
    """Rename in place after released structural conversion, retaining authority.

    ``record`` follows the migration callback ``(path, old, new, reason)``.
    Each referenced file and inline patch must be converted before merging.
    """
    def report(old_path, old, new_path, new):
        if record is not None:
            record(".".join((*path_prefix, *old_path)), deepcopy(old),
                   {".".join((*path_prefix, *new_path)): deepcopy(new)},
                   "explicit public unit spelling; value unchanged")

    def visit(node, path=()):
        if isinstance(node, list):
            for index, child in enumerate(node):
                if path[-1:] == ("fixed_params",) and isinstance(child, str):
                    renamed = released_fixed_selector(child)
                    if child != renamed:
                        node[index] = renamed
                        report((*path, str(index)), child, (*path, str(index)), renamed)
                else:
                    visit(child, (*path, str(index)))
            return
        if not isinstance(node, dict):
            return
        selectors = path[-1:] in (("overrides",), ("fixed_runtime_overrides",))
        for key in list(node):
            parts = tuple(key.split(".")) if selectors else (key,)
            old_path = (*path, *parts)
            new_path = released_public_path(old_path)
            renamed_parts = new_path[len(path):]
            renamed = ".".join(renamed_parts) if selectors else renamed_parts[-1]
            value = node[key]
            if renamed != key:
                if renamed in node:
                    raise ValueError(f"{'.'.join(old_path)} conflicts with {'.'.join(new_path)}")
                del node[key]
                node[renamed] = value
                report(old_path, value, new_path, value)
            visit(value, new_path)

    visit(config)
    return config
