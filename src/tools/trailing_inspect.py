from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from copy import deepcopy
from typing import Any, Mapping, Sequence


STRATEGY_KIND = "trailing_martingale"

DEFAULT_VOLATILITY_SCENARIOS = (
    ("quiet", 0.001, 0.0005),
    ("normal", 0.005, 0.0025),
    ("high", 0.015, 0.0075),
)
DEFAULT_EXPOSURE_RATIOS = (0.0, 0.5, 0.9)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass

PARAMETER_FLAGS = {
    "entry_threshold_base_pct": ("entry", "threshold_base_pct"),
    "entry_threshold_we_weight": ("entry", "threshold_we_weight"),
    "entry_threshold_volatility_1h_weight": (
        "entry",
        "threshold_volatility_1h_weight",
    ),
    "entry_threshold_volatility_1m_weight": (
        "entry",
        "threshold_volatility_1m_weight",
    ),
    "entry_retracement_base_pct": ("entry", "retracement_base_pct"),
    "entry_retracement_we_weight": ("entry", "retracement_we_weight"),
    "entry_retracement_volatility_1h_weight": (
        "entry",
        "retracement_volatility_1h_weight",
    ),
    "entry_retracement_volatility_1m_weight": (
        "entry",
        "retracement_volatility_1m_weight",
    ),
    "close_threshold_base_pct": ("close", "threshold_base_pct"),
    "close_threshold_we_weight": ("close", "threshold_we_weight"),
    "close_threshold_volatility_1h_weight": (
        "close",
        "threshold_volatility_1h_weight",
    ),
    "close_threshold_volatility_1m_weight": (
        "close",
        "threshold_volatility_1m_weight",
    ),
    "close_retracement_base_pct": ("close", "retracement_base_pct"),
    "close_retracement_volatility_1h_weight": (
        "close",
        "retracement_volatility_1h_weight",
    ),
    "close_retracement_volatility_1m_weight": (
        "close",
        "retracement_volatility_1m_weight",
    ),
}


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid {path}")
    return value


def _extract_strategy_params(config: Mapping[str, Any], pside: str) -> dict[str, Any]:
    live = _require_mapping(config.get("live"), "live")
    strategy_kind = str(live.get("strategy_kind") or STRATEGY_KIND).strip().lower()
    if strategy_kind != STRATEGY_KIND:
        raise ValueError(
            f"config uses live.strategy_kind={strategy_kind!r}; this inspector supports "
            f"only {STRATEGY_KIND!r}"
        )
    bot = _require_mapping(config.get("bot"), "bot")
    side = _require_mapping(bot.get(pside), f"bot.{pside}")
    strategies = _require_mapping(side.get("strategy"), f"bot.{pside}.strategy")
    params = _require_mapping(
        strategies.get(STRATEGY_KIND),
        f"bot.{pside}.strategy.{STRATEGY_KIND}",
    )
    entry = _require_mapping(params.get("entry"), f"bot.{pside}.strategy.{STRATEGY_KIND}.entry")
    close = _require_mapping(params.get("close"), f"bot.{pside}.strategy.{STRATEGY_KIND}.close")
    return {"entry": deepcopy(dict(entry)), "close": deepcopy(dict(close))}


def load_parameter_source(config_path: str | None, pside: str) -> tuple[dict[str, Any], str]:
    if config_path:
        from config import load_prepared_config

        config = load_prepared_config(
            config_path,
            live_only=True,
            verbose=False,
            target="canonical",
            log_info=False,
        )
        return _extract_strategy_params(config, pside), f"config {config_path} ({pside})"

    from config.strategy_spec import get_strategy_defaults

    defaults = get_strategy_defaults(STRATEGY_KIND)
    return _extract_strategy_params(
        {
            "live": {"strategy_kind": STRATEGY_KIND},
            "bot": {
                pside: {
                    "strategy": {STRATEGY_KIND: defaults[pside]},
                }
            },
        },
        pside,
    ), f"Rust-owned {STRATEGY_KIND} defaults ({pside})"


def _extract_side_context(config: Mapping[str, Any], pside: str) -> dict[str, Any]:
    bot = _require_mapping(config.get("bot"), "bot")
    side = _require_mapping(bot.get(pside), f"bot.{pside}")
    risk = _require_mapping(side.get("risk"), f"bot.{pside}.risk")
    strategies = _require_mapping(side.get("strategy"), f"bot.{pside}.strategy")
    strategy = _require_mapping(
        strategies.get(STRATEGY_KIND),
        f"bot.{pside}.strategy.{STRATEGY_KIND}",
    )
    entry = _require_mapping(strategy.get("entry"), "strategy.entry")
    close = _require_mapping(strategy.get("close"), "strategy.close")
    total_wallet_exposure_limit = _finite_float(
        risk.get("total_wallet_exposure_limit"),
        f"bot.{pside}.risk.total_wallet_exposure_limit",
    )
    n_positions = int(_finite_float(risk.get("n_positions"), f"bot.{pside}.risk.n_positions"))
    return {
        "active": total_wallet_exposure_limit > 0.0 and n_positions > 0,
        "total_wallet_exposure_limit": total_wallet_exposure_limit,
        "n_positions": n_positions,
        "entry_cooldown_minutes": _finite_float(
            risk.get("entry_cooldown_minutes", 0.0),
            f"bot.{pside}.risk.entry_cooldown_minutes",
        ),
        "entry_ema_gate_mode": str(entry.get("ema_gate_mode", "unknown")),
        "entry_initial_ema_dist": _finite_float(
            entry.get("initial_ema_dist", 0.0), "entry.initial_ema_dist"
        ),
        "entry_initial_qty_pct": _finite_float(
            entry.get("initial_qty_pct", 0.0), "entry.initial_qty_pct"
        ),
        "entry_double_down_factor": _finite_float(
            entry.get("double_down_factor", 0.0), "entry.double_down_factor"
        ),
        "close_qty_pct": _finite_float(close.get("qty_pct", 0.0), "close.qty_pct"),
        "volatility_ema_span_1m": _finite_float(
            strategy.get("volatility_ema_span_1m", 0.0), "volatility_ema_span_1m"
        ),
        "volatility_ema_span_1h": _finite_float(
            strategy.get("volatility_ema_span_1h", 0.0), "volatility_ema_span_1h"
        ),
    }


def load_overview_sources(
    config_path: str | None, psides: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], str]:
    if config_path:
        from config import load_prepared_config

        config = load_prepared_config(
            config_path,
            live_only=True,
            verbose=False,
            target="canonical",
            log_info=False,
        )
        sources = {
            pside: {
                "params": _extract_strategy_params(config, pside),
                "context": _extract_side_context(config, pside),
            }
            for pside in psides
        }
        return sources, f"config {config_path}"

    sources = {}
    for pside in psides:
        params, _ = load_parameter_source(None, pside)
        sources[pside] = {"params": params, "context": None}
    return sources, f"Rust-owned {STRATEGY_KIND} defaults"


def apply_parameter_overrides(params: dict[str, Any], args: argparse.Namespace) -> list[str]:
    changed: list[str] = []
    for dest, (section, name) in PARAMETER_FLAGS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        params[section][name] = _finite_float(value, dest)
        changed.append(f"{section}.{name}")
    return changed


def _dynamic_multiplier(
    *,
    volatility_ema_1m: float,
    volatility_ema_1h: float,
    weight_volatility_1m: float,
    weight_volatility_1h: float,
    wallet_exposure_ratio: float | None,
    weight_wallet_exposure: float,
) -> dict[str, float]:
    volatility_1h_term = volatility_ema_1h * weight_volatility_1h
    volatility_1m_term = volatility_ema_1m * weight_volatility_1m
    wallet_exposure_term = (wallet_exposure_ratio or 0.0) * weight_wallet_exposure
    raw = 1.0 + volatility_1h_term + volatility_1m_term + wallet_exposure_term
    return {
        "base": 1.0,
        "volatility_1h_term": volatility_1h_term,
        "volatility_1m_term": volatility_1m_term,
        "wallet_exposure_term": wallet_exposure_term,
        "raw": raw,
        "effective": max(1.0, raw),
    }


def _geometry(
    *,
    kind: str,
    pside: str,
    position_price: float,
    threshold_pct: float,
    retracement_pct: float,
) -> dict[str, Any]:
    threshold_direction = -1.0 if (kind, pside) in {("entry", "long"), ("close", "short")} else 1.0
    retracement_direction = -threshold_direction
    threshold_gate_active = threshold_pct > 0.0
    passive_reference_price = position_price * (1.0 + threshold_direction * threshold_pct)
    threshold_price = (
        passive_reference_price if threshold_gate_active else None
    )
    nominal_confirmation_price = (
        threshold_price * (1.0 + retracement_direction * retracement_pct)
        if threshold_price is not None and retracement_pct > 0.0
        else None
    )
    order_reference_price = (
        position_price
        * (1.0 + threshold_direction * threshold_pct + retracement_direction * retracement_pct)
        if threshold_gate_active and retracement_pct > 0.0
        else None
    )
    return {
        "threshold_gate_active": threshold_gate_active,
        "threshold_direction": "below" if threshold_direction < 0.0 else "above",
        "threshold_price": threshold_price,
        "passive_reference_price": passive_reference_price,
        "retracement_direction": "above" if retracement_direction > 0.0 else "below",
        "nominal_confirmation_price": nominal_confirmation_price,
        "nominal_confirmation_pct_from_position": (
            nominal_confirmation_price / position_price - 1.0
            if nominal_confirmation_price is not None
            else None
        ),
        "order_reference_price": order_reference_price,
        "order_reference_pct_from_position": (
            order_reference_price / position_price - 1.0
            if order_reference_price is not None
            else None
        ),
    }


def inspect_trailing(
    *,
    symbol: str,
    pside: str,
    position_price: float,
    position_size: float | None,
    wallet_exposure: float,
    effective_wallet_exposure_limit: float,
    volatility_ema_1m: float,
    volatility_ema_1h: float,
    params: Mapping[str, Any],
    parameter_source: str,
    overridden_parameters: Sequence[str] = (),
) -> dict[str, Any]:
    if pside not in {"long", "short"}:
        raise ValueError("pside must be 'long' or 'short'")
    position_price = _finite_float(position_price, "position_price")
    if position_size is not None:
        position_size = _finite_float(position_size, "position_size")
    wallet_exposure = _finite_float(wallet_exposure, "wallet_exposure")
    effective_wallet_exposure_limit = _finite_float(
        effective_wallet_exposure_limit,
        "effective_wallet_exposure_limit",
    )
    volatility_ema_1m = _finite_float(volatility_ema_1m, "volatility_ema_1m")
    volatility_ema_1h = _finite_float(volatility_ema_1h, "volatility_ema_1h")
    if position_price <= 0.0:
        raise ValueError("position_price must be greater than zero")
    if wallet_exposure < 0.0:
        raise ValueError("wallet_exposure must not be negative")
    if effective_wallet_exposure_limit <= 0.0:
        raise ValueError("effective_wallet_exposure_limit must be greater than zero")
    if volatility_ema_1m < 0.0 or volatility_ema_1h < 0.0:
        raise ValueError("volatility EMAs must not be negative")

    entry = _require_mapping(params.get("entry"), "params.entry")
    close = _require_mapping(params.get("close"), "params.close")
    wallet_exposure_ratio = wallet_exposure / effective_wallet_exposure_limit

    entry_threshold_multiplier = _dynamic_multiplier(
        volatility_ema_1m=volatility_ema_1m,
        volatility_ema_1h=volatility_ema_1h,
        weight_volatility_1m=_finite_float(
            entry.get("threshold_volatility_1m_weight", 0.0),
            "entry.threshold_volatility_1m_weight",
        ),
        weight_volatility_1h=_finite_float(
            entry.get("threshold_volatility_1h_weight", 0.0),
            "entry.threshold_volatility_1h_weight",
        ),
        wallet_exposure_ratio=wallet_exposure_ratio,
        weight_wallet_exposure=_finite_float(
            entry.get("threshold_we_weight", 0.0),
            "entry.threshold_we_weight",
        ),
    )
    entry_retracement_multiplier = _dynamic_multiplier(
        volatility_ema_1m=volatility_ema_1m,
        volatility_ema_1h=volatility_ema_1h,
        weight_volatility_1m=_finite_float(
            entry.get("retracement_volatility_1m_weight", 0.0),
            "entry.retracement_volatility_1m_weight",
        ),
        weight_volatility_1h=_finite_float(
            entry.get("retracement_volatility_1h_weight", 0.0),
            "entry.retracement_volatility_1h_weight",
        ),
        wallet_exposure_ratio=wallet_exposure_ratio,
        weight_wallet_exposure=_finite_float(
            entry.get("retracement_we_weight", 0.0),
            "entry.retracement_we_weight",
        ),
    )
    entry_threshold_base_configured = _finite_float(
        entry.get("threshold_base_pct", 0.0),
        "entry.threshold_base_pct",
    )
    entry_retracement_base = _finite_float(
        entry.get("retracement_base_pct", 0.0),
        "entry.retracement_base_pct",
    )
    entry_trailing_enabled = entry_retracement_base > 0.0
    # Rust clamps the threshold only in calc_trailing_entry_{long,short}. Passive
    # calc_reentry_price_{bid,ask} uses the configured signed value directly.
    entry_threshold_base = (
        max(0.0, entry_threshold_base_configured)
        if entry_trailing_enabled
        else entry_threshold_base_configured
    )
    entry_threshold_pct = entry_threshold_base * entry_threshold_multiplier["effective"]
    entry_retracement_pct = max(0.0, entry_retracement_base) * entry_retracement_multiplier[
        "effective"
    ]

    close_threshold_base = _finite_float(
        close.get("threshold_base_pct", 0.0),
        "close.threshold_base_pct",
    )
    close_threshold_terms = {
        "base": close_threshold_base,
        "wallet_exposure_term": wallet_exposure_ratio
        * _finite_float(close.get("threshold_we_weight", 0.0), "close.threshold_we_weight"),
        "volatility_1h_term": volatility_ema_1h
        * _finite_float(
            close.get("threshold_volatility_1h_weight", 0.0),
            "close.threshold_volatility_1h_weight",
        ),
        "volatility_1m_term": volatility_ema_1m
        * _finite_float(
            close.get("threshold_volatility_1m_weight", 0.0),
            "close.threshold_volatility_1m_weight",
        ),
    }
    close_threshold_pct = sum(close_threshold_terms.values())
    close_retracement_multiplier = _dynamic_multiplier(
        volatility_ema_1m=volatility_ema_1m,
        volatility_ema_1h=volatility_ema_1h,
        weight_volatility_1m=_finite_float(
            close.get("retracement_volatility_1m_weight", 0.0),
            "close.retracement_volatility_1m_weight",
        ),
        weight_volatility_1h=_finite_float(
            close.get("retracement_volatility_1h_weight", 0.0),
            "close.retracement_volatility_1h_weight",
        ),
        wallet_exposure_ratio=None,
        weight_wallet_exposure=0.0,
    )
    close_retracement_base = _finite_float(
        close.get("retracement_base_pct", 0.0),
        "close.retracement_base_pct",
    )
    close_retracement_pct = max(0.0, close_retracement_base) * close_retracement_multiplier[
        "effective"
    ]

    return {
        "symbol": symbol,
        "pside": pside,
        "position": {"size": position_size, "price": position_price},
        "wallet_exposure": wallet_exposure,
        "effective_wallet_exposure_limit": effective_wallet_exposure_limit,
        "wallet_exposure_ratio": wallet_exposure_ratio,
        "volatility_ema_1m": volatility_ema_1m,
        "volatility_ema_1h": volatility_ema_1h,
        "parameter_source": parameter_source,
        "overridden_parameters": list(overridden_parameters),
        "entry": {
            "trailing_enabled": entry_trailing_enabled,
            "threshold_base_pct": entry_threshold_base,
            "threshold_multiplier": entry_threshold_multiplier,
            "threshold_pct": entry_threshold_pct,
            "retracement_base_pct": max(0.0, entry_retracement_base),
            "retracement_multiplier": entry_retracement_multiplier,
            "retracement_pct": entry_retracement_pct,
            "geometry": _geometry(
                kind="entry",
                pside=pside,
                position_price=position_price,
                threshold_pct=entry_threshold_pct,
                retracement_pct=entry_retracement_pct,
            ),
        },
        "close": {
            "trailing_enabled": close_retracement_base > 0.0,
            "threshold_terms": close_threshold_terms,
            "threshold_pct": close_threshold_pct,
            "retracement_base_pct": max(0.0, close_retracement_base),
            "retracement_multiplier": close_retracement_multiplier,
            "retracement_pct": close_retracement_pct,
            "geometry": _geometry(
                kind="close",
                pside=pside,
                position_price=position_price,
                threshold_pct=close_threshold_pct,
                retracement_pct=close_retracement_pct,
            ),
        },
    }


def _threshold_style(kind: str, threshold_pct: float) -> str:
    if threshold_pct <= 0.0:
        return "immediate (no threshold gate)"
    cutoffs = (
        ((0.005 if kind == "entry" else 0.003), "aggressive/near"),
        ((0.02 if kind == "entry" else 0.01), "moderate"),
        ((0.05 if kind == "entry" else 0.03), "patient/deep"),
    )
    for cutoff, label in cutoffs:
        if threshold_pct <= cutoff:
            return label
    return "very deep"


def _retracement_style(retracement_pct: float) -> str:
    if retracement_pct <= 0.0:
        return "disabled"
    if retracement_pct <= 0.001:
        return "tight"
    if retracement_pct <= 0.005:
        return "modest"
    if retracement_pct <= 0.015:
        return "selective"
    return "deep"


def _sensitivity_style(
    low_values: Sequence[float], high_values: Sequence[float]
) -> str:
    relative_change = max(
        abs(high_value - low_value) / max(abs(low_value), 0.001)
        for low_value, high_value in zip(low_values, high_values, strict=True)
    )
    if relative_change < 0.1:
        return "low effective volatility sensitivity"
    if relative_change < 0.35:
        return "light volatility sensitivity"
    if relative_change < 0.75:
        return "strong volatility sensitivity"
    return "very strong volatility sensitivity"


def _movement_description(
    delta: float,
    subject: str,
    low_exposure_ratio: float,
    high_exposure_ratio: float,
    volatility_label: str,
) -> str:
    if low_exposure_ratio == high_exposure_ratio:
        return f"Only one exposure ratio is shown, so {subject} exposure sensitivity is not compared."
    if abs(delta) < 0.00005:
        return f"Exposure has almost no effect on the {subject}."
    verb = "widens" if delta > 0.0 else "narrows"
    return (
        f"Moving from {low_exposure_ratio * 100.0:g}% to {high_exposure_ratio * 100.0:g}% "
        f"of the exposure limit {verb} the {subject} by "
        f"{abs(delta) * 100.0:.4f} percentage points in the "
        f"{volatility_label!r} volatility example."
    )


def _classify_side(
    *,
    pside: str,
    context: Mapping[str, Any] | None,
    representative: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normal = representative["normal"]
    quiet = representative["quiet"]
    high = representative["high"]
    exposure_low = representative["exposure_low"]
    exposure_high = representative["exposure_high"]
    entry = normal["entry"]
    close = normal["close"]
    entry_trailing = entry["trailing_enabled"]
    close_trailing = close["trailing_enabled"]

    volatility_style = (
        "one volatility scenario shown"
        if representative["volatility_scenario_count"] == 1
        else _sensitivity_style(
            (
                quiet["entry"]["threshold_pct"],
                quiet["entry"]["retracement_pct"],
            ),
            (
                high["entry"]["threshold_pct"],
                high["entry"]["retracement_pct"],
            ),
        )
    )
    close_volatility_style = (
        "one volatility scenario shown"
        if representative["volatility_scenario_count"] == 1
        else _sensitivity_style(
            (
                quiet["close"]["threshold_pct"],
                quiet["close"]["retracement_pct"],
            ),
            (
                high["close"]["threshold_pct"],
                high["close"]["retracement_pct"],
            ),
        )
    )
    entry_headline = (
        f"{volatility_style}; "
        f"{_threshold_style('entry', entry['threshold_pct'])} entry threshold with "
        f"{_retracement_style(entry['retracement_pct'])} retracement"
    )
    close_headline = (
        f"{close_volatility_style}; "
        f"{_threshold_style('close', close['threshold_pct'])} close threshold with "
        f"{_retracement_style(close['retracement_pct'])} retracement"
    )

    entry_comments: list[str] = []
    close_comments: list[str] = []
    cooldown = context["entry_cooldown_minutes"] if context else None
    if not entry_trailing:
        if cooldown == 0.0:
            entry_comments.append(
                "Trailing entries are disabled and entry cooldown is zero: Rust may expose the "
                "full recursive entry ladder simultaneously."
            )
        elif cooldown is not None:
            entry_comments.append(
                "Trailing entries are disabled, but the entry cooldown is positive: the bot does "
                "not wait for a reversal; it stages only the next rung and waits "
                f"{cooldown:g} minutes after an entry fill before another add."
            )
        else:
            entry_comments.append(
                "Trailing entries are disabled: re-entries use passive recursive limit prices."
            )
    else:
        entry_comments.append(
            "Only the next add is staged. Price must first cross the adverse threshold, then "
            "reverse by the retracement distance from the running extreme."
        )
        if cooldown and cooldown > 0.0:
            entry_comments.append(
                f"After an entry fill, the {cooldown:g}-minute cooldown blocks the next add even "
                "if its trailing conditions are already satisfied."
            )
    if entry["threshold_pct"] <= 0.0 and entry_trailing:
        entry_comments.append(
            "The threshold gate is immediate, so reversal tracking begins as soon as the position changes."
        )
    entry_comments.append(
        _movement_description(
            exposure_high["entry"]["threshold_pct"] - exposure_low["entry"]["threshold_pct"],
            "entry threshold",
            representative["exposure_low_ratio"],
            representative["exposure_high_ratio"],
            representative["normal_label"],
        )
    )
    entry_retracement_delta = (
        exposure_high["entry"]["retracement_pct"]
        - exposure_low["entry"]["retracement_pct"]
    )
    entry_comments.append(
        _movement_description(
            entry_retracement_delta,
            "entry retracement",
            representative["exposure_low_ratio"],
            representative["exposure_high_ratio"],
            representative["normal_label"],
        )
    )

    close_params = representative["params"]["close"]
    if not close_trailing:
        if _finite_float(close_params.get("threshold_we_weight", 0.0), "threshold_we_weight") == 0.0:
            close_comments.append(
                "Trailing closes are disabled and the threshold has no exposure weight: Rust emits "
                "one full-position passive close instead of duplicate same-price slices."
            )
        else:
            close_comments.append(
                "Trailing closes are disabled: Rust builds passive recursive close slices and "
                "recomputes the exposure-weighted threshold after each hypothetical slice."
            )
    else:
        close_comments.append(
            "The close arms after favorable movement reaches the threshold, then confirms only "
            "after price reverses from the running favorable extreme."
        )
    if close["threshold_pct"] <= 0.0 and close_trailing:
        close_comments.append(
            "At this scenario the close threshold is non-positive, so the close behaves like an "
            "immediately armed trailing stop rather than waiting for profit first."
        )
    close_comments.append(
        _movement_description(
            exposure_high["close"]["threshold_pct"] - exposure_low["close"]["threshold_pct"],
            "close threshold",
            representative["exposure_low_ratio"],
            representative["exposure_high_ratio"],
            representative["normal_label"],
        )
    )
    close_comments.append(
        "Close retracement has no exposure term in Rust; only volatility changes it."
    )

    overall: list[str] = []
    if context:
        if context["active"]:
            overall.append(
                f"{pside.capitalize()} is enabled: total exposure limit "
                f"{context['total_wallet_exposure_limit'] * 100.0:.2f}% across "
                f"{context['n_positions']} configured position slot(s)."
            )
        else:
            overall.append(
                f"{pside.capitalize()} is disabled by its zero exposure limit or position count. "
                "The tables below describe dormant parameters, not orders the current config will place."
            )
        ema_gate_mode = context["entry_ema_gate_mode"]
        if ema_gate_mode in {"all", "reentry"}:
            overall.append(
                f"Entry EMA gate mode is {ema_gate_mode!r}; trailing re-entry prices may be pushed "
                "farther from market by the EMA gate before exchange rounding."
            )
        elif ema_gate_mode == "initial":
            overall.append(
                "Entry EMA gate mode is 'initial': initial entries are EMA-gated, but trailing "
                "re-entry prices are not."
            )
        else:
            overall.append(
                f"Entry EMA gate mode is {ema_gate_mode!r}; strategy entry prices are not EMA-gated."
            )
        overall.append(
            "Each add starts from the greater of current absolute position size × "
            f"{context['entry_double_down_factor']:.4g} or initial-entry quantity, before "
            "minimum-quantity, rounding, and exposure-cropping rules."
        )
        overall.append(
            f"Each trailing close targets {context['close_qty_pct'] * 100.0:.2f}% before "
            "minimum-size and remaining-position rules."
        )
    return {
        "entry_headline": entry_headline,
        "entry_comments": entry_comments,
        "close_headline": close_headline,
        "close_comments": close_comments,
        "overall_comments": overall,
        "basis": (
            f"Headlines use {representative['normal_label']!r} volatility at "
            f"{representative['middle_ratio'] * 100.0:g}% WE/WEL."
        ),
    }


def build_overview(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    parameter_source: str,
    price_anchor: float,
    volatility_scenarios: Sequence[tuple[str, float, float]] = DEFAULT_VOLATILITY_SCENARIOS,
    exposure_ratios: Sequence[float] = DEFAULT_EXPOSURE_RATIOS,
    overridden_parameters: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    price_anchor = _finite_float(price_anchor, "price_anchor")
    if price_anchor <= 0.0:
        raise ValueError("price_anchor must be greater than zero")
    if not volatility_scenarios:
        raise ValueError("at least one volatility scenario is required")
    if not exposure_ratios:
        raise ValueError("at least one exposure ratio is required")
    checked_scenarios = []
    for label, volatility_ema_1m, volatility_ema_1h in volatility_scenarios:
        vol_1m = _finite_float(volatility_ema_1m, f"{label} volatility_ema_1m")
        vol_1h = _finite_float(volatility_ema_1h, f"{label} volatility_ema_1h")
        if vol_1m < 0.0 or vol_1h < 0.0:
            raise ValueError("volatility scenario values must not be negative")
        checked_scenarios.append((str(label), vol_1m, vol_1h))
    scenario_labels = [item[0] for item in checked_scenarios]
    duplicate_labels = sorted(
        label for label in set(scenario_labels) if scenario_labels.count(label) > 1
    )
    if duplicate_labels:
        raise ValueError(
            "duplicate volatility scenario label(s): " + ", ".join(duplicate_labels)
        )
    checked_ratios = [_finite_float(value, "exposure_ratio") for value in exposure_ratios]
    if any(value < 0.0 for value in checked_ratios):
        raise ValueError("exposure ratios must not be negative")

    sides: dict[str, Any] = {}
    for pside, source in sources.items():
        params = _require_mapping(source.get("params"), f"source.{pside}.params")
        context = source.get("context")
        rows = []
        scenario_results: dict[tuple[str, float], Mapping[str, Any]] = {}
        for label, volatility_ema_1m, volatility_ema_1h in checked_scenarios:
            for exposure_ratio in checked_ratios:
                result = inspect_trailing(
                    symbol="ANCHOR",
                    pside=pside,
                    position_price=price_anchor,
                    position_size=None,
                    wallet_exposure=exposure_ratio,
                    effective_wallet_exposure_limit=1.0,
                    volatility_ema_1m=volatility_ema_1m,
                    volatility_ema_1h=volatility_ema_1h,
                    params=params,
                    parameter_source=parameter_source,
                    overridden_parameters=(overridden_parameters or {}).get(pside, ()),
                )
                scenario_results[(label, exposure_ratio)] = result
                rows.append(
                    {
                        "volatility_label": label,
                        "volatility_ema_1m": volatility_ema_1m,
                        "volatility_ema_1h": volatility_ema_1h,
                        "exposure_ratio": exposure_ratio,
                        "entry": result["entry"],
                        "close": result["close"],
                    }
                )

        normal_label = min(
            checked_scenarios,
            key=lambda item: abs(item[1] - 0.005) + abs(item[2] - 0.0025),
        )[0]
        quiet_label = min(
            checked_scenarios,
            key=lambda item: (item[1] + item[2], item[1], item[2], item[0]),
        )[0]
        high_label = max(
            checked_scenarios,
            key=lambda item: (item[1] + item[2], item[1], item[2], item[0]),
        )[0]
        middle_ratio = min(checked_ratios, key=lambda value: abs(value - 0.5))
        low_ratio = min(checked_ratios)
        high_ratio = max(checked_ratios)
        representative = {
            "normal": scenario_results[(normal_label, middle_ratio)],
            "quiet": scenario_results[(quiet_label, middle_ratio)],
            "high": scenario_results[(high_label, middle_ratio)],
            "exposure_low": scenario_results[(normal_label, low_ratio)],
            "exposure_high": scenario_results[(normal_label, high_ratio)],
            "exposure_low_ratio": low_ratio,
            "exposure_high_ratio": high_ratio,
            "middle_ratio": middle_ratio,
            "normal_label": normal_label,
            "volatility_scenario_count": len(checked_scenarios),
            "params": params,
        }
        sides[pside] = {
            "context": context,
            "parameters": deepcopy(dict(params)),
            "overridden_parameters": list((overridden_parameters or {}).get(pside, ())),
            "classification": _classify_side(
                pside=pside,
                context=context,
                representative=representative,
            ),
            "scenarios": rows,
        }
    return {
        "mode": "overview",
        "parameter_source": parameter_source,
        "price_anchor": price_anchor,
        "volatility_scenarios": [
            {"label": label, "volatility_ema_1m": vol_1m, "volatility_ema_1h": vol_1h}
            for label, vol_1m, vol_1h in checked_scenarios
        ],
        "exposure_ratios": checked_ratios,
        "sides": sides,
    }


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.10g}"


def _fmt_pct(value: float, *, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value * 100.0:{sign}.4f}%"


def _format_multiplier(label: str, multiplier: Mapping[str, float]) -> list[str]:
    named_terms = [
        ("1h", multiplier["volatility_1h_term"]),
        ("1m", multiplier["volatility_1m_term"]),
    ]
    if multiplier["wallet_exposure_term"] != 0.0:
        named_terms.append(("WE", multiplier["wallet_exposure_term"]))
    expression = "1"
    for term_label, value in named_terms:
        operator = "+" if value >= 0.0 else "-"
        expression += f" {operator} {_fmt_pct(abs(value))} [{term_label}]"
    return [
        f"  {label} multiplier: {multiplier['effective']:.6f} "
        f"(max(1, {expression}))"
    ]


def _format_geometry(kind: str, payload: Mapping[str, Any]) -> list[str]:
    geometry = payload["geometry"]
    lines: list[str] = []
    if not payload["trailing_enabled"]:
        lines.append("  Mode: trailing disabled (retracement_base_pct <= 0); passive recursive orders")
        lines.append(
            "  Passive analytical reference from position price and signed threshold "
            f"{_fmt_pct(payload['threshold_pct'], signed=True)} -> "
            f"{_fmt_number(geometry['passive_reference_price'])}"
        )
        lines.append("  Rust then clamps this reference against the current bid/ask before rounding.")
    elif geometry["threshold_gate_active"]:
        lines.append(
            f"  Threshold: {_fmt_pct(payload['threshold_pct'])} {geometry['threshold_direction']} "
            f"position price -> {_fmt_number(geometry['threshold_price'])}"
        )
        lines.append(
            f"  Retracement: {_fmt_pct(payload['retracement_pct'])} "
            f"{geometry['retracement_direction']} the running extreme"
        )
        lines.append(
            "  If reversal starts exactly at the threshold: confirmation -> "
            f"{_fmt_number(geometry['nominal_confirmation_price'])} "
            f"({_fmt_pct(geometry['nominal_confirmation_pct_from_position'], signed=True)} vs position)"
        )
        lines.append(
            "  Emitted-order reference after both conditions: "
            f"{_fmt_number(geometry['order_reference_price'])} "
            f"({_fmt_pct(geometry['order_reference_pct_from_position'], signed=True)} vs position)"
        )
    else:
        lines.append(
            f"  Threshold: {_fmt_pct(payload['threshold_pct'], signed=True)}; gate is active immediately"
        )
        lines.append(
            f"  Retracement: {_fmt_pct(payload['retracement_pct'])} "
            f"{geometry['retracement_direction']} the running extreme; no fixed target price"
        )
    if payload["trailing_enabled"]:
        lines.append("  Actual confirmation follows the running low/high, not a permanently fixed threshold target.")
    return lines


def render_report(result: Mapping[str, Any]) -> str:
    position = result["position"]
    size_text = f"{_fmt_number(position['size'])} @ " if position["size"] is not None else ""
    lines = [
        f"Trailing inspection: {result['symbol']} {result['pside']}",
        f"Position: {size_text}{_fmt_number(position['price'])}",
        (
            f"Wallet exposure: {_fmt_pct(result['wallet_exposure'])} / effective limit "
            f"{_fmt_pct(result['effective_wallet_exposure_limit'])} "
            f"(ratio {_fmt_pct(result['wallet_exposure_ratio'])})"
        ),
        (
            f"Volatility EMA: 1m {_fmt_pct(result['volatility_ema_1m'])}, "
            f"1h {_fmt_pct(result['volatility_ema_1h'])}"
        ),
        f"Parameters: {result['parameter_source']}",
    ]
    if result["overridden_parameters"]:
        lines.extend(
            textwrap.wrap(
                ", ".join(result["overridden_parameters"]),
                width=100,
                initial_indent="Overrides: ",
                subsequent_indent="           ",
            )
        )

    entry = result["entry"]
    lines.extend(["", "ENTRY"])
    lines.extend(_format_multiplier("Threshold", entry["threshold_multiplier"]))
    lines.append(
        f"  Effective threshold: {_fmt_pct(entry['threshold_base_pct'])} × "
        f"{entry['threshold_multiplier']['effective']:.6f} = {_fmt_pct(entry['threshold_pct'])}"
    )
    lines.extend(_format_multiplier("Retracement", entry["retracement_multiplier"]))
    lines.append(
        f"  Effective retracement: {_fmt_pct(entry['retracement_base_pct'])} × "
        f"{entry['retracement_multiplier']['effective']:.6f} = {_fmt_pct(entry['retracement_pct'])}"
    )
    lines.extend(_format_geometry("entry", entry))

    close = result["close"]
    close_terms = close["threshold_terms"]
    lines.extend(["", "CLOSE"])
    lines.append(
        "  Threshold (additive): "
        f"base {_fmt_pct(close_terms['base'], signed=True)} "
        f"+ WE {_fmt_pct(close_terms['wallet_exposure_term'], signed=True)} "
        f"+ 1h {_fmt_pct(close_terms['volatility_1h_term'], signed=True)} "
        f"+ 1m {_fmt_pct(close_terms['volatility_1m_term'], signed=True)} "
        f"= {_fmt_pct(close['threshold_pct'], signed=True)}"
    )
    lines.extend(_format_multiplier("Retracement", close["retracement_multiplier"]))
    lines.append(
        f"  Effective retracement: {_fmt_pct(close['retracement_base_pct'])} × "
        f"{close['retracement_multiplier']['effective']:.6f} = {_fmt_pct(close['retracement_pct'])}"
    )
    lines.extend(_format_geometry("close", close))
    lines.extend(
        [
            "",
            "Percent inputs use config ratios: 0.01 = 1%.",
            "Prices are analytical trigger/reference prices before tick rounding, bid/ask limits, or EMA gating.",
        ]
    )
    return "\n".join(lines)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    rendered = [
        "  " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  " + "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        rendered.append(
            "  "
            + "  ".join(
                value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
                for index, value in enumerate(row)
            )
        )
    return rendered


def _scenario_price(value: float | None, *, fallback: str = "-") -> str:
    return fallback if value is None else f"{value:.4f}"


def _scenario_rows(side: Mapping[str, Any], kind: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for scenario in side["scenarios"]:
        payload = scenario[kind]
        geometry = payload["geometry"]
        if not payload["trailing_enabled"]:
            confirmation = "passive"
            threshold_price = "n/a"
            order_reference = _scenario_price(geometry["passive_reference_price"])
        elif not geometry["threshold_gate_active"]:
            confirmation = "extreme-based"
            threshold_price = "immediate"
            order_reference = "market"
        else:
            confirmation = _scenario_price(geometry["nominal_confirmation_price"])
            threshold_price = _scenario_price(geometry["threshold_price"])
            order_reference = _scenario_price(geometry["order_reference_price"])
        rows.append(
            [
                scenario["volatility_label"],
                f"{scenario['volatility_ema_1m'] * 100.0:.2f}/{scenario['volatility_ema_1h'] * 100.0:.2f}%",
                f"{scenario['exposure_ratio'] * 100.0:.0f}%",
                _fmt_pct(payload["threshold_pct"], signed=kind == "close"),
                threshold_price,
                _fmt_pct(payload["retracement_pct"]),
                confirmation,
                order_reference,
            ]
        )
    return rows


def _parameter_summary(side: Mapping[str, Any], kind: str) -> list[str]:
    params = side["parameters"][kind]
    threshold = (
        f"base {_fmt_pct(_finite_float(params.get('threshold_base_pct', 0.0), 'threshold base'), signed=kind == 'close')}, "
        f"weights WE {_finite_float(params.get('threshold_we_weight', 0.0), 'threshold WE'):g}, "
        f"1m {_finite_float(params.get('threshold_volatility_1m_weight', 0.0), 'threshold 1m'):g}, "
        f"1h {_finite_float(params.get('threshold_volatility_1h_weight', 0.0), 'threshold 1h'):g}"
    )
    retracement_parts = [
        f"base {_fmt_pct(_finite_float(params.get('retracement_base_pct', 0.0), 'retracement base'))}"
    ]
    if kind == "entry":
        retracement_parts.append(
            f"WE {_finite_float(params.get('retracement_we_weight', 0.0), 'retracement WE'):g}"
        )
    retracement_parts.extend(
        [
            f"1m {_finite_float(params.get('retracement_volatility_1m_weight', 0.0), 'retracement 1m'):g}",
            f"1h {_finite_float(params.get('retracement_volatility_1h_weight', 0.0), 'retracement 1h'):g}",
        ]
    )
    return [f"  Threshold params: {threshold}", f"  Retracement params: {', '.join(retracement_parts)}"]


def render_overview(result: Mapping[str, Any]) -> str:
    lines = [
        "Trailing behavior overview",
        f"Parameters: {result['parameter_source']}",
        f"Price anchor (average position price): {_fmt_number(result['price_anchor'])}",
        "Scenario cells use volatility EMA values as 1m/1h percentages and exposure as WE / effective WEL.",
        "Entry distance = base × max(1, 1 + 1m term + 1h term + exposure term).",
        "Close threshold = base + 1m term + 1h term + exposure term; close retracement has no exposure term.",
    ]
    headers = (
        "Vol",
        "1m/1h",
        "WE/WEL",
        "Threshold",
        "T price",
        "Retrace",
        "R confirm*",
        "Order ref*",
    )
    for pside, side in result["sides"].items():
        classification = side["classification"]
        context = side["context"]
        lines.extend(["", f"{pside.upper()} — BEHAVIOR"])
        lines.append(f"  {classification['basis']}")
        lines.extend(f"  {comment}" for comment in classification["overall_comments"])
        if context:
            lines.append(
                "  Volatility EMA spans: "
                f"1m {context['volatility_ema_span_1m']:g}, "
                f"1h {context['volatility_ema_span_1h']:g}."
            )
        if side["overridden_parameters"]:
            lines.append("  Overrides: " + ", ".join(side["overridden_parameters"]))

        lines.extend(["", f"  ENTRY — {classification['entry_headline']}"])
        lines.extend(_parameter_summary(side, "entry"))
        lines.extend(f"  • {comment}" for comment in classification["entry_comments"])
        lines.extend(_format_table(headers, _scenario_rows(side, "entry")))

        lines.extend(["", f"  CLOSE — {classification['close_headline']}"])
        lines.extend(_parameter_summary(side, "close"))
        lines.extend(f"  • {comment}" for comment in classification["close_comments"])
        lines.extend(_format_table(headers, _scenario_rows(side, "close")))

    lines.extend(
        [
            "",
            "* R confirm is the nominal price if reversal starts exactly at T price. In reality, "
            "confirmation follows the running low/high, so a farther extreme moves the trigger.",
            "* Order ref is Rust's analytical emitted-order reference after both conditions pass; "
            "the live price is also constrained by bid/ask, tick rounding, sizing, exposure caps, and EMA gating.",
            "* With close trailing enabled, a non-positive threshold is immediate: trailing is "
            "armed from position open. With close trailing disabled, the row is passive and no "
            "extrema or reversal confirmation participate.",
            "* Trailing extrema reset after every fill for the same coin and position side.",
            "* Categories are descriptive heuristics for intuition, not trading-quality judgments.",
            "Percent inputs use config ratios: 0.01 = 1%.",
        ]
    )
    return "\n".join(lines)


def _parse_exposure_ratios(value: str) -> tuple[float, ...]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated ratios, e.g. 0,0.5,0.9") from exc
    if not ratios or any(not math.isfinite(item) or item < 0.0 for item in ratios):
        raise argparse.ArgumentTypeError("exposure ratios must be finite and non-negative")
    return ratios


def _parse_volatility_scenarios(value: str) -> tuple[tuple[str, float, float], ...]:
    scenarios = []
    try:
        for raw_scenario in value.split(","):
            label, vol_1m, vol_1h = (item.strip() for item in raw_scenario.split(":"))
            if not label:
                raise ValueError
            values = float(vol_1m), float(vol_1h)
            if any(not math.isfinite(item) or item < 0.0 for item in values):
                raise ValueError
            scenarios.append((label, *values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected label:1m:1h scenarios, e.g. quiet:0.001:0.0005,high:0.015:0.0075"
        ) from exc
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one volatility scenario is required")
    return tuple(scenarios)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool trailing-inspect",
        description=(
            "Inspect trailing_martingale entry and close thresholds without starting a bot. "
            "Percent inputs use config ratios (0.01 = 1%)."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Example:\n"
            "  passivbot tool trailing-inspect path/to/config.json\n"
            "  passivbot tool trailing-inspect path/to/config.json --price-anchor 250\n\n"
            "Detailed single-scenario mode (backward compatible):\n"
            "  passivbot tool trailing-inspect --symbol COIN --side long "
            "--position-size 150 --position-price 20 --wallet-exposure 0.6 "
            "--effective-wallet-exposure-limit 0.9 --volatility-ema-1m 0.007 "
            "--volatility-ema-1h 0.0033\n\n"
            "A positional config path is preferred; --config remains available for compatibility. "
            "Any parameter flag below overrides the config/default value."
        ),
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Canonical config to inspect; omit to inspect Rust-owned defaults",
    )
    state = parser.add_argument_group("position and market state")
    state.add_argument("--symbol", default="COIN", help="Display label only")
    state.add_argument(
        "--side",
        choices=("long", "short", "both"),
        default=None,
        help="Overview defaults to both sides; detailed mode defaults to long",
    )
    state.add_argument("--position-size", type=float, default=None, help="Display-only position size")
    state.add_argument(
        "--price-anchor",
        "--position-price",
        dest="price_anchor",
        type=float,
        default=100.0,
        help="Average position price used for example trigger geometry",
    )
    state.add_argument("--wallet-exposure", type=float, default=None, help="Current WE ratio")
    state.add_argument(
        "--effective-wallet-exposure-limit",
        type=float,
        default=None,
        help="Effective per-position WEL used by the strategy",
    )
    state.add_argument("--volatility-ema-1m", type=float, default=None)
    state.add_argument("--volatility-ema-1h", type=float, default=None)
    state.add_argument(
        "--config",
        default=None,
        help="Legacy alternative to the positional config path",
    )
    state.add_argument(
        "--exposure-ratios",
        type=_parse_exposure_ratios,
        default=DEFAULT_EXPOSURE_RATIOS,
        help="Overview WE/WEL examples as comma-separated ratios",
    )
    state.add_argument(
        "--volatility-scenarios",
        type=_parse_volatility_scenarios,
        default=DEFAULT_VOLATILITY_SCENARIOS,
        help="Overview examples as comma-separated label:1m:1h triples",
    )
    state.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    entry_threshold = parser.add_argument_group("entry threshold overrides")
    entry_threshold.add_argument("--entry-threshold-base-pct", type=float)
    entry_threshold.add_argument("--entry-threshold-we-weight", type=float)
    entry_threshold.add_argument("--entry-threshold-volatility-1h-weight", type=float)
    entry_threshold.add_argument("--entry-threshold-volatility-1m-weight", type=float)
    entry_retracement = parser.add_argument_group("entry retracement overrides")
    entry_retracement.add_argument("--entry-retracement-base-pct", type=float)
    entry_retracement.add_argument("--entry-retracement-we-weight", type=float)
    entry_retracement.add_argument("--entry-retracement-volatility-1h-weight", type=float)
    entry_retracement.add_argument("--entry-retracement-volatility-1m-weight", type=float)
    close_threshold = parser.add_argument_group("close threshold overrides")
    close_threshold.add_argument("--close-threshold-base-pct", type=float)
    close_threshold.add_argument("--close-threshold-we-weight", type=float)
    close_threshold.add_argument("--close-threshold-volatility-1h-weight", type=float)
    close_threshold.add_argument("--close-threshold-volatility-1m-weight", type=float)
    close_retracement = parser.add_argument_group("close retracement overrides")
    close_retracement.add_argument("--close-retracement-base-pct", type=float)
    close_retracement.add_argument("--close-retracement-volatility-1h-weight", type=float)
    close_retracement.add_argument("--close-retracement-volatility-1m-weight", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.config_path and args.config and args.config_path != args.config:
            raise ValueError("positional config path and --config name different files")
        config_path = args.config_path or args.config
        detailed_mode = any(
            value is not None
            for value in (
                args.position_size,
                args.wallet_exposure,
                args.effective_wallet_exposure_limit,
                args.volatility_ema_1m,
                args.volatility_ema_1h,
            )
        )
        if detailed_mode:
            if args.side == "both":
                raise ValueError("detailed single-scenario mode requires --side long or --side short")
            if (
                args.wallet_exposure is None
                or args.effective_wallet_exposure_limit is None
            ):
                raise ValueError(
                    "detailed single-scenario mode requires both --wallet-exposure and "
                    "--effective-wallet-exposure-limit"
                )
            pside = args.side or "long"
            params, source = load_parameter_source(config_path, pside)
            overridden = apply_parameter_overrides(params, args)
            result = inspect_trailing(
                symbol=args.symbol,
                pside=pside,
                position_price=args.price_anchor,
                position_size=args.position_size,
                wallet_exposure=args.wallet_exposure,
                effective_wallet_exposure_limit=args.effective_wallet_exposure_limit,
                volatility_ema_1m=(
                    args.volatility_ema_1m if args.volatility_ema_1m is not None else 0.0
                ),
                volatility_ema_1h=(
                    args.volatility_ema_1h if args.volatility_ema_1h is not None else 0.0
                ),
                params=params,
                parameter_source=source,
                overridden_parameters=overridden,
            )
        else:
            psides = (args.side,) if args.side in {"long", "short"} else ("long", "short")
            sources, source = load_overview_sources(config_path, psides)
            overridden_by_side = {}
            for pside in psides:
                overridden_by_side[pside] = apply_parameter_overrides(
                    sources[pside]["params"], args
                )
            result = build_overview(
                sources=sources,
                parameter_source=source,
                price_anchor=args.price_anchor,
                volatility_scenarios=args.volatility_scenarios,
                exposure_ratios=args.exposure_ratios,
                overridden_parameters=overridden_by_side,
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result.get("mode") == "overview":
        print(render_overview(result))
    else:
        print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
