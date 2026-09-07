"""Machine-independent fixed simulation inputs used to validate saved fitness."""

from copy import deepcopy

from config.overrides import parse_overrides
from config.param_paths import resolve_optimizer_key_path
from config_utils import clean_config
from optimization.config_adapter import _flatten_bounds_for_config
from optimization.fine_tune_anchors import get_anchor_plan
from optimization.evaluation_implementation import evaluation_implementation_identity
from optimization.warmup import _apply_config_overrides
from optimizer_overrides import optimizer_overrides

CONTRACT_KEY = "optimizer_evaluation_contract"
CONTRACT_CACHE_KEY = "_optimizer_evaluation_contract"

# Live-owned settings consumed by backtest preparation, execution, or risk.
# Account I/O, polling, logging, and machine-local settings are intentionally absent.
BACKTEST_LIVE_KEYS = frozenset(
    {
        "approved_coins",
        "ignored_coins",
        "strategy_kind",
        "hsl_signal_mode",
        "hedge_mode",
        "max_realized_loss_pct",
        "pnls_max_lookback_days",
        "market_orders_allowed",
        "market_order_near_touch_threshold",
        "forager_score_hysteresis_pct",
        "minimum_coin_age_days",
        "warmup_ratio",
        "max_warmup_minutes",
    }
)


def _remove_path(config, path):
    node = config
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict) and path:
        node.pop(path[-1], None)


def has_unresolved_override_files(config: dict) -> bool:
    """Legacy file references cannot establish the policy used for saved fitness."""

    def contains_file(value):
        if isinstance(value, dict):
            return any(
                (str(key).split(".")[-1] == "override_config_path" and bool(item))
                or contains_file(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_file(item) for item in value)
        return False

    return contains_file(config.get("coin_overrides", {})) or contains_file(
        config.get("backtest", {}).get("scenarios", [])
    )


def build_evaluation_contract(config: dict) -> dict:
    """Retain fixed policy while excluding values owned by the candidate vector."""
    effective = clean_config(config)
    _apply_config_overrides(
        effective, effective.get("optimize", {}).get("fixed_runtime_overrides", {})
    )
    overrides = effective.get("optimize", {}).get("enable_overrides", [])
    effective = optimizer_overrides(overrides, effective, None)
    for side in ("long", "short"):
        effective = optimizer_overrides(overrides, effective, side)
    effective = clean_config(effective)
    base_config_path = config.get("live", {}).get("base_config_path")
    if base_config_path is not None:
        effective["live"]["base_config_path"] = base_config_path
    effective = parse_overrides(effective, verbose=False)

    # Resolve before removing the base values required to validate coin patches.
    coin_overrides = deepcopy(effective.get("coin_overrides", {}))
    bounds = _flatten_bounds_for_config(effective, effective["optimize"]["bounds"])
    for key in bounds:
        path = resolve_optimizer_key_path(effective, key)
        if path is not None:
            _remove_path(effective, path)

    plan = get_anchor_plan(config)
    anchor_contract = None
    if plan is not None:
        anchors = []
        for anchor in plan["anchors"]:
            values = sorted(
                (
                    {"path": list(item["path"]), "value": deepcopy(item["value"])}
                    for item in anchor.get("fixed_values", [])
                ),
                key=lambda item: item["path"],
            )
            for item in values:
                _remove_path(effective, item["path"])
            anchors.append(values)
        # Anchor order is meaningful to the anchor gene; filenames and labels are not.
        anchor_contract = {
            "key_paths": [list(path) for path in plan.get("key_paths", [])],
            "anchors": anchors,
        }
    return {
        "version": 1,
        "implementation": deepcopy(evaluation_implementation_identity()),
        "prepared_data": deepcopy(config.get("_optimizer_prepared_dataset_identity")),
        "bot": effective.get("bot", {}),
        "live": {
            key: effective["live"][key]
            for key in sorted(BACKTEST_LIVE_KEYS)
            if key in effective.get("live", {})
        },
        "coin_overrides": coin_overrides,
        "anchors": anchor_contract,
    }


def recorded_evaluation_contract(template: dict) -> dict:
    contract = template.get(CONTRACT_CACHE_KEY)
    return deepcopy(
        contract if isinstance(contract, dict) else build_evaluation_contract(template)
    )
