import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

from utils import format_end_date, normalize_coins_source, symbol_to_coin

from .limits import _resolve_optimize_limits_for_load
from .log_output import log_config_message
from .optimize_bounds import sort_optimize_bounds_in_place
from .scoring import extract_objective_specs
from .schema import get_template_config
from .tree_ops import add_missing_keys_recursively, remove_unused_keys_recursively


Path = tuple[str, ...]

PARTIALLY_OPEN_CONFIG_PATHS: set[Path] = {
    ("backtest", "aggregate"),
}

TEMPLATE_SYNC_PRESERVE_PATHS: tuple[Path, ...] = (
    ("coin_overrides",),
    ("backtest", "suite", "aggregate"),
    ("backtest", "suite", "scenarios"),
    ("backtest", "market_settings_sources"),
    *tuple(PARTIALLY_OPEN_CONFIG_PATHS),
)

def hydrate_missing_template_fields(
    template: dict,
    result: dict,
    *,
    verbose: bool = True,
    tracker=None,
) -> None:
    add_missing_keys_recursively(template, result, verbose=verbose, tracker=tracker)


def seed_missing_compatibility_sections(template: dict, result: dict, *, tracker=None) -> None:
    for pside in ("long", "short"):
        if pside not in result["bot"]:
            seeded = deepcopy(template["bot"][pside])
            # A fully omitted side should stay disabled after hydration instead of inheriting
            # whichever exposure default happens to be in the schema.
            seeded["total_wallet_exposure_limit"] = 0.0
            result["bot"][pside] = seeded
            if tracker is not None:
                tracker.add(["bot", pside], seeded)
    for key in ("approved_coins", "ignored_coins"):
        if key not in result["live"]:
            seeded = {"long": [], "short": []}
            result["live"][key] = seeded
            if tracker is not None:
                tracker.add(["live", key], seeded)
            continue
        if isinstance(result["live"][key], dict) and set(result["live"][key]).issubset({"long", "short"}):
            for pside in ("long", "short"):
                if pside not in result["live"][key]:
                    result["live"][key][pside] = []
                    if tracker is not None:
                        tracker.add(["live", key, pside], [])
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
                tracker.update(["live", "base_config_path"], existing_base, base_config_path)
    template_with_extras = deepcopy(template)
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
    remove_unused_keys_recursively(template["bot"], result["bot"], verbose=verbose, tracker=tracker)
    remove_unused_keys_recursively(
        template["optimize"]["bounds"],
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


def _normalize_coin_sources(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("backtest.coin_sources must be a mapping of coin -> exchange")
    normalized: Dict[str, str] = {}
    for coin, exchange in raw.items():
        if exchange is None:
            continue
        coin_key = symbol_to_coin(str(coin), verbose=False)
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


def preserve_coin_sources(result: dict, *, live_sources_input: Optional[Dict[str, Any]] = None) -> None:
    sources = result.setdefault("_coins_sources", {})
    live = result.get("live", {})
    for key in ("approved_coins", "ignored_coins"):
        if key in sources:
            continue
        if live_sources_input is not None and key in live_sources_input:
            sources[key] = deepcopy(live_sources_input[key])
            continue
        if key in live:
            sources[key] = deepcopy(live[key])


def apply_non_live_adjustments(
    result: dict,
    *,
    verbose: bool = True,
    tracker=None,
    raw_optimize_limits: Any = None,
    raw_optimize_limits_present: Optional[bool] = None,
) -> None:
    for key in ("approved_coins", "ignored_coins"):
        result["live"][key] = normalize_coins_source(
            result["live"].get(key, ""),
            allow_all=(key == "approved_coins"),
        )
    for pside in result["live"]["approved_coins"]:
        result["live"]["approved_coins"][pside] = [
            coin
            for coin in result["live"]["approved_coins"][pside]
            if coin not in result["live"]["ignored_coins"][pside]
        ]
    result["backtest"]["end_date"] = format_end_date(result["backtest"]["end_date"])
    result["backtest"]["coin_sources"] = _normalize_coin_sources(
        result["backtest"].get("coin_sources", {})
    )
    if result["backtest"].get("filter_by_min_effective_cost") is None:
        result["backtest"]["filter_by_min_effective_cost"] = bool(
            result["live"].get("filter_by_min_effective_cost", False)
        )

    result["optimize"]["scoring"] = [spec.to_config() for spec in extract_objective_specs(result)]
    backend = str(result["optimize"].get("backend", "pymoo") or "pymoo").strip().lower()
    if backend not in {"deap", "pymoo"}:
        raise ValueError(
            f"optimize.backend must be one of ['deap', 'pymoo']; got {result['optimize'].get('backend')!r}"
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

    current_limits = deepcopy(result["optimize"].get("limits", []))
    limits_snapshot = deepcopy(current_limits)
    if raw_optimize_limits_present is None:
        raw_optimize_limits_present = "limits" in result.get("optimize", {})
        if raw_optimize_limits is None and raw_optimize_limits_present:
            raw_optimize_limits = deepcopy(result["optimize"].get("limits"))
    template_limits = deepcopy(get_template_config()["optimize"]["limits"])
    resolved_limits, resolution = _resolve_optimize_limits_for_load(
        raw_optimize_limits=raw_optimize_limits,
        raw_optimize_limits_present=raw_optimize_limits_present,
        template_limits=template_limits,
    )
    result["optimize"]["limits"] = resolved_limits
    if resolution == "normalized_legacy":
        log_config_message(
            verbose,
            logging.INFO,
            "normalized optimize.limits to canonical schema (%d entries)",
            len(resolved_limits),
        )
        if tracker is not None:
            tracker.update(["optimize", "limits"], limits_snapshot, resolved_limits)
    elif resolution == "fallback_template":
        log_config_message(
            verbose,
            logging.WARNING,
            "optimize.limits malformed or unsupported; falling back to template defaults (%d entries)",
            len(template_limits),
        )
        if tracker is not None:
            tracker.update(["optimize", "limits"], limits_snapshot, resolved_limits)
    sort_optimize_bounds_in_place(
        result["optimize"]["bounds"],
        strategy_kind=result.get("live", {}).get("strategy_kind"),
    )
