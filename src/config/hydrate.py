from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

from utils import (
    format_end_date,
    looks_like_exact_market_identifier,
    normalize_coins_source,
    symbol_to_coin,
)
from optimization.random_seed import normalize_optional_seed

from .limits import normalize_limit_entries
from .optimize_bounds import sort_optimize_bounds_in_place
from .scoring import extract_objective_specs
from .schema import get_template_config
from .tree_ops import add_missing_keys_recursively, remove_unused_keys_recursively

Path = tuple[str, ...]

PARTIALLY_OPEN_CONFIG_PATHS: set[Path] = {
    ("backtest", "reducer"),
    ("live", "startup", "startup_phase_budgets"),
    ("optimize", "fixed_runtime_overrides"),
}
BACKTEST_INHERITED_LIVE_KEYS: tuple[str, ...] = (
    "fee_pct_fallback",
    "fee_pct_sanity_abs_max",
    "market_orders_allowed",
    "market_order_near_touch_threshold",
    "lookback_ema_span_minutes",
)

TEMPLATE_SYNC_PRESERVE_PATHS: tuple[Path, ...] = (
    ("coin_overrides",),
    ("backtest", "suite", "aggregate"),
    ("backtest", "suite", "scenarios"),
    ("backtest", "market_settings"),
    ("backtest", "market_settings_sources"),
    *tuple(PARTIALLY_OPEN_CONFIG_PATHS),
)


from config.access import get_optional_config_value
from config.universe import get_side_universe_sources, require_side_universe_sources

def reject_backtest_inherited_live_fields(result: dict) -> None:
    backtest = result.setdefault("backtest", {})
    invalid = [key for key in BACKTEST_INHERITED_LIVE_KEYS if key in backtest]
    if invalid:
        joined = ", ".join(f"backtest.{key}" for key in invalid)
        owners = {
            "fee_pct_fallback": "live.fills.fallback_fee_fraction",
            "fee_pct_sanity_abs_max": "live.fills.max_abs_fee_fraction",
            "market_orders_allowed": "bot.portfolio.execution.market_orders_allowed",
            "market_order_near_touch_threshold": "bot.portfolio.execution.market_order_near_touch_fraction",
        }
        owner_paths = ", ".join(owners.get(key, f"live.{key}") for key in invalid)
        raise ValueError(
            f"{joined} {'is' if len(invalid) == 1 else 'are'} not supported; "
            f"set {'the value' if len(invalid) == 1 else 'these values'} under {owner_paths} instead"
        )


def hydrate_missing_template_fields(
    template: dict,
    result: dict,
    *,
    verbose: bool = True,
    tracker=None,
) -> None:
    add_missing_keys_recursively(
        template,
        result,
        verbose=verbose,
        tracker=tracker,
        preserve=PARTIALLY_OPEN_CONFIG_PATHS,
    )


def seed_missing_default_sections(
    template: dict, result: dict, *, tracker=None
) -> None:
    for pside in ("long", "short"):
        if pside not in result["bot"]:
            seeded = deepcopy(template["bot"][pside])
            result["bot"][pside] = seeded
            result["bot"][pside]["universe"] = {
                "approved_coins": None, "ignored_coins": [],
                "filter_by_min_effective_cost": True, "minimum_coin_age_days": 60.0,
            }
            if tracker is not None:
                tracker.add(["bot", pside], seeded)
    for pside in ("long", "short"):
        universe = result["bot"][pside].setdefault("universe", {})
        for key in ("approved_coins", "ignored_coins"):
            if key not in universe:
                universe[key] = None if key == "approved_coins" else []
                if tracker is not None:
                    tracker.add(["bot", pside, "universe", key], universe[key])
    if "bounds" not in result["optimize"]:
        seeded_bounds = deepcopy(template["optimize"]["bounds"])
        result["optimize"]["bounds"] = seeded_bounds
        if tracker is not None:
            tracker.add(["optimize", "bounds"], seeded_bounds)


def sync_with_template(
    template: dict,
    result: dict,
    base_config_path: str,
    *,
    verbose: bool = True,
    tracker=None,
) -> None:
    existing_base = result["live"].get("base_config_path") if "live" in result else None
    had_key = "live" in result and "base_config_path" in result["live"]
    if base_config_path or "base_config_path" not in result["live"]:
        result["live"]["base_config_path"] = base_config_path
        if tracker is not None:
            if not had_key:
                tracker.add(["live", "base_config_path"], base_config_path)
            elif existing_base != base_config_path:
                tracker.update(
                    ["live", "base_config_path"], existing_base, base_config_path
                )
    template_with_extras = deepcopy(template)
    from .optimize_bounds import preserve_optional_adaptive_bounds, set_flat_optimize_bound

    # Canonicalize legacy leaves before template pruning; a mixed dictionary
    # may contain both these leaves and explicit nested adaptive dimensions.
    bounds = result["optimize"]["bounds"]
    for key in list(bounds):
        if isinstance(key, str) and key.startswith(("long_", "short_", "hsl_")):
            set_flat_optimize_bound(bounds, result['bot']["portfolio"]['strategy_kind'], key, bounds.pop(key))
    preserve_optional_adaptive_bounds(template_with_extras, result)
    # Keep only explicitly supplied HSL portfolio authority and its search bounds.
    for section in (("bot",), ("optimize", "bounds")):
        target, source = template_with_extras, result
        for key in section:
            target, source = target[key], source[key]
        if "portfolio" in source:
            target["portfolio"] = deepcopy(source["portfolio"])
    template_with_extras.setdefault("live", {})["base_config_path"] = ""
    preserved_live_optimize_bounds = [
        ("optimize", "bounds", key)
        for key in result.get("optimize", {}).get("bounds", {})
        if isinstance(key, str) and key.startswith("live_")
    ]
    remove_unused_keys_recursively(
        template_with_extras,
        result,
        verbose=verbose,
        preserve=TEMPLATE_SYNC_PRESERVE_PATHS + tuple(preserved_live_optimize_bounds),
        tracker=tracker,
    )
    remove_unused_keys_recursively(
        template_with_extras["bot"], result["bot"], verbose=verbose, tracker=tracker
    )
    remove_unused_keys_recursively(
        template_with_extras["optimize"]["bounds"],
        result["optimize"]["bounds"],
        verbose=verbose,
        tracker=tracker,
    )
    remove_unused_keys_recursively(
        template.get("optimize", {}).get("limits", []),
        result["optimize"].setdefault("limits", []),
        verbose=verbose,
        tracker=tracker,
    )


def normalize_backtest_coin_sources(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("backtest.coin_sources must be a mapping of coin -> exchange")
    normalized: Dict[str, str] = {}
    for coin, exchange in raw.items():
        if exchange is None:
            continue
        raw_coin = str(coin).strip()
        coin_key = (
            raw_coin
            if looks_like_exact_market_identifier(raw_coin)
            else symbol_to_coin(raw_coin, verbose=False)
        )
        if not coin_key:
            continue
        exchange_value = str(exchange)
        existing = normalized.get(coin_key)
        if existing is not None and existing != exchange_value:
            raise ValueError(
                f"backtest.coin_sources maps conflicting exchanges for {coin_key}: "
                f"{existing} and {exchange_value}"
            )
        normalized[coin_key] = exchange_value
    return normalized


def preserve_coin_sources(
    result: dict, *, live_sources_input: Optional[Dict[str, Any]] = None
) -> None:
    sources = result.setdefault("_coins_sources", {})
    for key in ("approved_coins", "ignored_coins"):
        if key in sources:
            continue
        if live_sources_input is not None and key in live_sources_input:
            sources[key] = deepcopy(live_sources_input[key])
            continue
        value = get_side_universe_sources(result, key)
        if value is not None:
            sources[key] = deepcopy(value)


def apply_non_live_adjustments(
    result: dict,
    *,
    verbose: bool = True,
    tracker=None,
    raw_optimize_limits: Any = None,
    raw_optimize_limits_present: Optional[bool] = None,
) -> None:
    from .universe import resolve_universe_sources, materialize_side_universe_sources
    from .universe import ResolvedCoinList
    for key in ("approved_coins", "ignored_coins"):
        current = require_side_universe_sources(result, key)
        # Numeric re-preparation retains trusted dataset membership. Explicit
        # source declarations invalidate that typed view before reaching here.
        values = {side: list(value) for side, value in current.items()
                  if type(value) is ResolvedCoinList}
        pending = tuple(side for side in ("long", "short") if side not in values)
        if pending:
            resolved, _ = resolve_universe_sources(result, key, position_sides=pending)
            values.update(resolved)
        materialize_side_universe_sources(result, key, values)
    for pside in require_side_universe_sources(result, 'approved_coins'):
        from .universe import ResolvedCoinList
        current = result['bot'][pside]['universe']['approved_coins']
        result['bot'][pside]['universe']['approved_coins'] = ResolvedCoinList(
            [coin for coin in current if coin not in result['bot'][pside]['universe']['ignored_coins']], current.source, current.shared_source, source_present=current.source_present
        )
    result["backtest"]["end_date"] = format_end_date(result["backtest"]["end_date"])
    result["backtest"]["coin_sources"] = normalize_backtest_coin_sources(
        result["backtest"].get("coin_sources", {})
    )

    normalize_optimizer_settings(
        result,
        verbose=verbose,
        tracker=tracker,
        raw_optimize_limits=raw_optimize_limits,
        raw_optimize_limits_present=raw_optimize_limits_present,
    )


def normalize_optimizer_settings(
    result: dict,
    *,
    verbose: bool = True,
    tracker=None,
    raw_optimize_limits: Any = None,
    raw_optimize_limits_present: Optional[bool] = None,
) -> None:
    """Normalize optimizer policy without resolving dates or coin-list files."""
    result["optimize"]["scoring"] = [
        spec.to_config() for spec in extract_objective_specs(result)
    ]
    backend = str(result["optimize"].get("backend", "pymoo") or "pymoo").strip().lower()
    if backend not in {"gpu", "pymoo"}:
        raise ValueError(
            "optimize.backend must be one of ['gpu', 'pymoo']; "
            f"got {result['optimize'].get('backend')!r}"
        )
    result["optimize"]["backend"] = backend
    population_size = result["optimize"].get("population_size")
    if isinstance(population_size, str):
        normalized_population_size = population_size.strip().lower()
        if normalized_population_size in {"", "none", "null", "auto"}:
            population_size = None
        else:
            population_size = int(population_size)
    elif population_size is not None:
        population_size = int(population_size)
    if population_size is not None and population_size <= 0:
        raise ValueError("optimize.population_size must be > 0 when set")
    result["optimize"]["population_size"] = population_size
    objective_scenario = result["optimize"].get("objective_scenario")
    if objective_scenario is not None:
        objective_scenario = str(objective_scenario).strip() or None
    result["optimize"]["objective_scenario"] = objective_scenario
    result["optimize"]["seed"] = normalize_optional_seed(result["optimize"].get("seed"))

    if raw_optimize_limits_present is None:
        raw_optimize_limits_present = "limits" in result.get("optimize", {})
        if raw_optimize_limits is None and raw_optimize_limits_present:
            raw_optimize_limits = deepcopy(result["optimize"].get("limits"))
    template_limits = deepcopy(get_template_config()["optimize"]["limits"])
    if raw_optimize_limits_present:
        resolved_limits = normalize_limit_entries(raw_optimize_limits)
    else:
        resolved_limits = template_limits
    result["optimize"]["limits"] = resolved_limits
    sort_optimize_bounds_in_place(
        result["optimize"]["bounds"],
        strategy_kind=get_optional_config_value(result, 'bot.portfolio.strategy_kind'),
        portfolio_hsl=result["bot"]["portfolio"]["hsl"]["scope"] == "portfolio",
    )
