from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from typing import Any, Mapping, Optional

import passivbot_rust as pbr
from config.shared_bot import flatten_shared_bot_side
from risk_limits import (
    effective_we_excess_allowance_fraction as _effective_we_excess_allowance_fraction,
    wallet_exposure_limit_ratio_with_allowance as _wallet_exposure_limit_ratio_with_allowance,
)


ENTRY_CONFIG_KEYS = [
    "entry_grid_reentry_qty_multiplier",
    "entry_grid_threshold_base_fraction",
    "entry_initial_ema_distance_fraction",
    "entry_initial_qty_fraction",
    "entry_trailing_reentry_qty_multiplier",
    "entry_trailing_retracement_base_fraction",
    "entry_trailing_threshold_base_fraction",
    "entry_log_range_1h_coefficient",
    "entry_log_range_1m_coefficient",
    "entry_exposure_ratio_coefficient",
    "wallet_exposure_limit_ratio",
    "total_wallet_exposure_limit_ratio",
    "risk_we_excess_allowance_fraction",
]


CLOSE_CONFIG_KEYS = [
    "close_grid_qty_fraction",
    "close_trailing_qty_fraction",
    "close_trailing_retracement_base_fraction",
    "close_trailing_threshold_base_fraction",
    "close_log_range_1h_coefficient",
    "close_log_range_1m_coefficient",
    "wallet_exposure_limit_ratio",
    "total_wallet_exposure_limit_ratio",
    "risk_we_excess_allowance_fraction",
    "risk_wel_enforcer_threshold_ratio",
]


TRAILING_EXTREMA_KEYS = [
    "min_since_open",
    "max_since_min",
    "max_since_open",
    "min_since_max",
]


NUMERIC_INPUT_KEYS = [
    "balance_raw",
    "current_price",
    "position_size",
    "position_price",
    "qty_step",
    "price_step",
    "min_qty",
    "min_cost",
    "c_mult",
    "ema_lower",
    "ema_upper",
    "h1_log_range_ema",
    "m1_log_range_ema",
    *TRAILING_EXTREMA_KEYS,
    *sorted(set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS)),
]


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _effective_wallet_exposure_limit_ratio(
    config: Mapping[str, Any],
    *,
    pside: str,
) -> float:
    bot_cfg = config.get("bot", {})
    if not isinstance(bot_cfg, Mapping):
        return 0.0
    side_cfg = flatten_shared_bot_side(bot_cfg.get(pside, {}))
    if not isinstance(side_cfg, Mapping):
        return 0.0
    direct = side_cfg.get("wallet_exposure_limit_ratio")
    if direct is not None:
        direct_value = _float(direct, default=0.0)
        if direct_value > 0.0:
            return direct_value
    twel = _float(side_cfg.get("total_wallet_exposure_limit_ratio"), default=0.0)
    n_positions = int(round(_float(side_cfg.get("n_positions"), default=0.0)))
    if twel <= 0.0 or n_positions <= 0:
        return 0.0
    return round(twel / n_positions, 8)


def normalize_trailing_extrema(bundle: Mapping[str, Any]) -> dict[str, float]:
    return {
        "min_since_open": _float(bundle.get("min_since_open", 0.0)),
        "max_since_min": _float(bundle.get("max_since_min", 0.0)),
        "max_since_open": _float(bundle.get("max_since_open", 0.0)),
        "min_since_max": _float(bundle.get("min_since_max", 0.0)),
    }


def trailing_status(
    *,
    triggered: bool,
    threshold_met: bool,
    retracement_met: bool,
) -> str:
    if triggered:
        return "triggered"
    if not threshold_met:
        return "waiting_threshold"
    if not retracement_met:
        return "waiting_retracement"
    return "armed"


def _trailing_projection_fields(
    *,
    low_trigger: bool,
    position_price: float,
    current_price: float,
    threshold_fraction: Optional[float],
    retracement_fraction: Optional[float],
    extrema: Mapping[str, float],
) -> dict[str, Any]:
    """Project optional prices without interpreting native initialization as a price.

    Native calls keep the original bundle, including Rust's finite extrema sentinels.
    These classifications describe only the separate human-facing projections.
    """
    out: dict[str, Any] = {}

    def put(name, value, status="available", reason=None):
        out[name] = value
        out[f"{name}_status"] = status
        if reason is not None:
            out[f"{name}_reason"] = reason

    def extremum_reason(key):
        value = extrema[key]
        if not math.isfinite(value):
            return "trailing_extremum_non_finite"
        if (key.startswith("min_") and value == sys.float_info.max) or (
            key.startswith("max_") and value == 0.0
        ):
            return "trailing_extremum_uninitialized"
        return None

    trigger_key = "min_since_open" if low_trigger else "max_since_open"
    recovery_key = "max_since_min" if low_trigger else "min_since_max"
    for name, fraction, basis, basis_key, observed_key, direction in (
        ("threshold", threshold_fraction, position_price, None, trigger_key, -1 if low_trigger else 1),
        ("retracement", retracement_fraction, extrema[trigger_key], trigger_key, recovery_key, 1 if low_trigger else -1),
    ):
        price_key = f"{name}_price"
        met_key = f"{name}_met"
        change_key = f"current_price_change_from_{name}_fraction"
        if fraction is None or not math.isfinite(fraction):
            reason = "projection_fraction_unavailable" if fraction is None else "projection_fraction_non_finite"
            put(f"{name}_fraction", None, "unavailable", reason)
        else:
            put(f"{name}_fraction", fraction)
            reason = None
        if reason is None and fraction <= 0.0:
            put(price_key, None, "not_required")
            put(met_key, True, "not_required")
            put(change_key, None, "not_required")
            continue
        if reason is None:
            reason = extremum_reason(basis_key) if basis_key is not None else (
                None if math.isfinite(basis) else "position_price_non_finite"
            )
        projected = None if reason else basis * (1.0 + direction * fraction)
        if reason is None and not math.isfinite(projected):
            reason = "projected_price_overflow"
        if reason:
            put(price_key, None, "unavailable", reason)
            put(met_key, None, "unavailable", reason)
            put(change_key, None, "unavailable", reason)
            continue
        put(price_key, projected)
        observation_reason = extremum_reason(observed_key)
        if observation_reason:
            put(met_key, None, "unavailable", observation_reason)
        else:
            observed = extrema[observed_key]
            less_than = low_trigger if name == "threshold" else not low_trigger
            put(met_key, observed < projected if less_than else observed > projected)
        if projected <= 0.0:
            put(change_key, None, "not_required")
        elif not math.isfinite(current_price):
            put(change_key, None, "unavailable", "current_price_non_finite")
        else:
            change = current_price / projected - 1.0
            if math.isfinite(change):
                put(change_key, change)
            else:
                put(change_key, None, "unavailable", "projected_change_overflow")
    out["projection_status"] = (
        "unavailable" if any(
            value == "unavailable" for key, value in out.items() if key.endswith("_status")
        ) else "available"
    )
    return out


def _projected_trailing_status(triggered: bool, projection: Mapping[str, Any]) -> str:
    if triggered:
        return "triggered"
    if projection["projection_status"] == "unavailable":
        return "unavailable"
    return trailing_status(
        triggered=False,
        threshold_met=projection["threshold_met"],
        retracement_met=projection["retracement_met"],
    )


def selected_mode_from_order_type(order_type: str, *, has_order: bool) -> str:
    normalized = str(order_type or "").lower()
    if "trailing" in normalized:
        return "trailing"
    if "auto_reduce" in normalized:
        return "auto_reduce"
    if "unstuck" in normalized:
        return "unstuck"
    if "grid" in normalized:
        return "grid"
    if has_order:
        return "other"
    return "none"


def calculate_wallet_exposure(
    *,
    balance_raw: float,
    position_size: float,
    position_price: float,
    c_mult: float,
) -> float:
    if balance_raw <= 0.0 or position_size == 0.0 or position_price <= 0.0 or c_mult <= 0.0:
        return 0.0
    return float(pbr.qty_to_cost(abs(position_size), position_price, c_mult) / balance_raw)


def wallet_exposure_limit_ratio_with_allowance(
    *,
    wallet_exposure_limit_ratio: float,
    risk_we_excess_allowance_fraction: float,
    total_wallet_exposure_limit_ratio: float = 0.0,
) -> float:
    return _wallet_exposure_limit_ratio_with_allowance(
        wallet_exposure_limit_ratio=wallet_exposure_limit_ratio,
        risk_we_excess_allowance_fraction=risk_we_excess_allowance_fraction,
        total_wallet_exposure_limit_ratio=total_wallet_exposure_limit_ratio,
    )


def effective_we_excess_allowance_fraction(
    *,
    wallet_exposure_limit_ratio: float,
    risk_we_excess_allowance_fraction: float,
    total_wallet_exposure_limit_ratio: float = 0.0,
) -> float:
    return _effective_we_excess_allowance_fraction(
        wallet_exposure_limit_ratio=wallet_exposure_limit_ratio,
        risk_we_excess_allowance_fraction=risk_we_excess_allowance_fraction,
        total_wallet_exposure_limit_ratio=total_wallet_exposure_limit_ratio,
    )


def entry_trailing_limit_cap(
    *,
    wallet_exposure_limit_ratio: float,
    risk_we_excess_allowance_fraction: float,
    total_wallet_exposure_limit_ratio: float = 0.0,
    entry_trailing_retracement_base_fraction: float,
    wallet_exposure: float,
) -> tuple[Optional[float], Optional[str]]:
    allowed_limit = wallet_exposure_limit_ratio_with_allowance(
        wallet_exposure_limit_ratio=wallet_exposure_limit_ratio,
        risk_we_excess_allowance_fraction=risk_we_excess_allowance_fraction,
        total_wallet_exposure_limit_ratio=total_wallet_exposure_limit_ratio,
    )
    if allowed_limit <= 0.0:
        return None, None
    if float(entry_trailing_retracement_base_fraction) > 0.0:
        return allowed_limit, "trailing_only"
    return None, "grid_only"


def _entry_ema_reference(inputs: Mapping[str, Any]) -> float:
    pside = str(inputs.get("pside", "long"))
    return _float(inputs.get("ema_lower" if pside == "long" else "ema_upper"), default=0.0)


def build_trailing_entry_diagnostic(inputs: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    symbol = str(inputs.get("symbol", ""))
    pside = str(inputs.get("pside", "long"))
    balance_raw = _float(inputs.get("balance_raw"))
    current_price = _float(inputs.get("current_price"))
    position_size = _float(inputs.get("position_size"))
    position_price = _float(inputs.get("position_price"))
    c_mult = _float(inputs.get("c_mult"))
    wallet_exposure = calculate_wallet_exposure(
        balance_raw=balance_raw,
        position_size=position_size,
        position_price=position_price,
        c_mult=c_mult,
    )
    limit_cap, mode = entry_trailing_limit_cap(
        wallet_exposure_limit_ratio=_float(inputs.get("wallet_exposure_limit_ratio")),
        risk_we_excess_allowance_fraction=_float(inputs.get("risk_we_excess_allowance_fraction")),
        total_wallet_exposure_limit_ratio=_float(inputs.get("total_wallet_exposure_limit_ratio")),
        entry_trailing_retracement_base_fraction=_float(inputs.get("entry_trailing_retracement_base_fraction")),
        wallet_exposure=wallet_exposure,
    )
    if limit_cap is None:
        return None
    ema_reference = _entry_ema_reference(inputs)
    if ema_reference <= 0.0 or current_price <= 0.0:
        return None
    trailing_bundle = normalize_trailing_extrema(inputs)
    entry_volatility_1h = _float(inputs.get("h1_log_range_ema"))
    entry_volatility_1m = _float(inputs.get("m1_log_range_ema"))
    effective_allowance_fraction = effective_we_excess_allowance_fraction(
        wallet_exposure_limit_ratio=_float(inputs.get("wallet_exposure_limit_ratio")),
        risk_we_excess_allowance_fraction=_float(inputs.get("risk_we_excess_allowance_fraction")),
        total_wallet_exposure_limit_ratio=_float(inputs.get("total_wallet_exposure_limit_ratio")),
    )
    common_args = [
        _float(inputs.get("qty_step")),
        _float(inputs.get("price_step")),
        _float(inputs.get("min_qty")),
        _float(inputs.get("min_cost")),
        c_mult,
        _float(inputs.get("entry_grid_reentry_qty_multiplier")),
        _float(inputs.get("entry_grid_threshold_base_fraction")),
        _float(inputs.get("entry_initial_ema_distance_fraction")),
        _float(inputs.get("entry_initial_qty_fraction")),
        _float(inputs.get("entry_trailing_reentry_qty_multiplier")),
        _float(inputs.get("entry_trailing_retracement_base_fraction")),
        _float(inputs.get("entry_trailing_threshold_base_fraction")),
        _float(inputs.get("entry_log_range_1h_coefficient")),
        _float(inputs.get("entry_log_range_1m_coefficient")),
        _float(inputs.get("entry_exposure_ratio_coefficient")),
        _float(inputs.get("wallet_exposure_limit_ratio")),
        effective_allowance_fraction,
        balance_raw,
        position_size,
        position_price,
        trailing_bundle["min_since_open"],
        trailing_bundle["max_since_min"],
        trailing_bundle["max_since_open"],
        trailing_bundle["min_since_max"],
        ema_reference,
        entry_volatility_1h,
        entry_volatility_1m,
        current_price,
    ]
    if pside == "long":
        qty, price, order_type = pbr.calc_next_entry_long_py(*common_args)
    else:
        qty, price, order_type = pbr.calc_next_entry_short_py(*common_args)
    order_type = str(order_type)
    has_order = bool(abs(_float(qty)) > 0.0 and _float(price) > 0.0)
    is_trailing_order = "trailing" in order_type.lower()

    entry_multiplier = max(
        1.0,
        1.0
        + entry_volatility_1h * _float(inputs.get("entry_log_range_1h_coefficient"))
        + entry_volatility_1m * _float(inputs.get("entry_log_range_1m_coefficient"))
        + (
            (wallet_exposure / limit_cap) * _float(inputs.get("entry_exposure_ratio_coefficient"))
            if limit_cap > 0.0
            else 0.0
        ),
    )
    threshold_fraction = _float(inputs.get("entry_trailing_threshold_base_fraction")) * entry_multiplier
    retracement_fraction = _float(inputs.get("entry_trailing_retracement_base_fraction")) * entry_multiplier

    projection = _trailing_projection_fields(
        low_trigger=pside == "long", position_price=position_price, current_price=current_price,
        threshold_fraction=threshold_fraction, retracement_fraction=retracement_fraction,
        extrema=trailing_bundle,
    )
    triggered = bool(is_trailing_order and has_order)
    return {
        "kind": "entry",
        "symbol": symbol,
        "pside": pside,
        "mode": mode,
        "selected_mode": selected_mode_from_order_type(order_type, has_order=has_order),
        "order_type": order_type,
        "wallet_exposure": wallet_exposure,
        "limit_cap": float(limit_cap),
        "triggered": triggered,
        "status": _projected_trailing_status(triggered, projection),
        "qty": float(qty),
        "price": float(price),
        "current_price": float(current_price),
        **projection,
        "extrema": deepcopy(trailing_bundle),
    }


def build_trailing_close_diagnostic(inputs: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    symbol = str(inputs.get("symbol", ""))
    pside = str(inputs.get("pside", "long"))
    balance_raw = _float(inputs.get("balance_raw"))
    current_price = _float(inputs.get("current_price"))
    position_size = _float(inputs.get("position_size"))
    position_price = _float(inputs.get("position_price"))
    if position_size == 0.0 or position_price <= 0.0 or current_price <= 0.0:
        return None
    trailing_bundle = normalize_trailing_extrema(inputs)
    close_volatility_1h = _float(inputs.get("h1_log_range_ema"))
    close_volatility_1m = _float(inputs.get("m1_log_range_ema"))
    effective_allowance_fraction = effective_we_excess_allowance_fraction(
        wallet_exposure_limit_ratio=_float(inputs.get("wallet_exposure_limit_ratio")),
        risk_we_excess_allowance_fraction=_float(inputs.get("risk_we_excess_allowance_fraction")),
        total_wallet_exposure_limit_ratio=_float(inputs.get("total_wallet_exposure_limit_ratio")),
    )
    common_args = [
        _float(inputs.get("qty_step")),
        _float(inputs.get("price_step")),
        _float(inputs.get("min_qty")),
        _float(inputs.get("min_cost")),
        _float(inputs.get("c_mult")),
        _float(inputs.get("close_grid_qty_fraction")),
        _float(inputs.get("close_trailing_qty_fraction")),
        _float(inputs.get("close_trailing_retracement_base_fraction")),
        _float(inputs.get("close_trailing_threshold_base_fraction")),
        _float(inputs.get("wallet_exposure_limit_ratio")),
        effective_allowance_fraction,
        _float(inputs.get("risk_wel_enforcer_threshold_ratio")),
        balance_raw,
        position_size,
        position_price,
        trailing_bundle["min_since_open"],
        trailing_bundle["max_since_min"],
        trailing_bundle["max_since_open"],
        trailing_bundle["min_since_max"],
        current_price,
        0.0,
        close_volatility_1h,
        close_volatility_1m,
        _float(inputs.get("close_log_range_1h_coefficient")),
        _float(inputs.get("close_log_range_1m_coefficient")),
    ]
    if pside == "long":
        qty, price, order_type = pbr.calc_next_close_long_py(*common_args)
    else:
        qty, price, order_type = pbr.calc_next_close_short_py(*common_args)
    order_type = str(order_type)
    has_order = bool(abs(_float(qty)) > 0.0 and _float(price) > 0.0)
    is_trailing_order = "trailing" in order_type.lower()

    close_multiplier = max(
        1.0,
        1.0
        + close_volatility_1h * _float(inputs.get("close_log_range_1h_coefficient"))
        + close_volatility_1m * _float(inputs.get("close_log_range_1m_coefficient")),
    )
    threshold_fraction = _float(inputs.get("close_trailing_threshold_base_fraction")) * close_multiplier
    retracement_fraction = _float(inputs.get("close_trailing_retracement_base_fraction")) * close_multiplier
    projection = _trailing_projection_fields(
        low_trigger=pside == "short", position_price=position_price, current_price=current_price,
        threshold_fraction=threshold_fraction, retracement_fraction=retracement_fraction,
        extrema=trailing_bundle,
    )
    triggered = bool(is_trailing_order and has_order)
    return {
        "kind": "close",
        "symbol": symbol,
        "pside": pside,
        "selected_mode": selected_mode_from_order_type(order_type, has_order=has_order),
        "order_type": order_type,
        "triggered": triggered,
        "status": _projected_trailing_status(triggered, projection),
        "qty": float(qty),
        "price": float(price),
        "current_price": float(current_price),
        **projection,
        "extrema": deepcopy(trailing_bundle),
    }


def build_trailing_martingale_close_diagnostic(
    inputs: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    position = inputs.get("position")
    if not isinstance(position, Mapping):
        return None
    position_size = _float(position.get("size"))
    position_price = _float(position.get("price"))
    current_price = _float(inputs.get("current_price"))
    if position_size == 0.0 or position_price <= 0.0 or current_price <= 0.0:
        return None

    payload = json.loads(
        pbr.calc_trailing_martingale_close_diagnostic_py(json.dumps(dict(inputs)))
    )
    if not isinstance(payload, dict):
        raise TypeError("Rust trailing_martingale close diagnostic must return an object")
    pside = str(inputs.get("pside") or "long")
    # serde represents a non-finite native diagnostic fraction as null. It is
    # unavailable, not a zero coefficient that disables the condition.
    threshold_fraction = payload.get("threshold_fraction")
    retracement_fraction = payload.get("retracement_fraction")
    extrema = normalize_trailing_extrema(payload.get("extrema", {}))
    projection = _trailing_projection_fields(
        low_trigger=pside == "short", position_price=position_price, current_price=current_price,
        threshold_fraction=threshold_fraction, retracement_fraction=retracement_fraction,
        extrema=extrema,
    )
    payload.update(
        {
            "symbol": str(inputs.get("symbol") or ""),
            "current_price": current_price,
            "position_price": position_price,
            "position_size": position_size,
            **projection,
            "extrema": extrema,
        }
    )
    if projection["projection_status"] == "unavailable" and not payload.get("triggered"):
        payload["status"] = "unavailable"
    return payload




def build_trailing_diagnostic(inputs: Mapping[str, Any]) -> dict[str, Any]:
    normalized_inputs = {key: inputs.get(key) for key in inputs}
    trailing_bundle = normalize_trailing_extrema(inputs)
    wallet_exposure = calculate_wallet_exposure(
        balance_raw=_float(inputs.get("balance_raw")),
        position_size=_float(inputs.get("position_size")),
        position_price=_float(inputs.get("position_price")),
        c_mult=_float(inputs.get("c_mult")),
    )
    allowed_limit = wallet_exposure_limit_ratio_with_allowance(
        wallet_exposure_limit_ratio=_float(inputs.get("wallet_exposure_limit_ratio")),
        risk_we_excess_allowance_fraction=_float(inputs.get("risk_we_excess_allowance_fraction")),
        total_wallet_exposure_limit_ratio=_float(inputs.get("total_wallet_exposure_limit_ratio")),
    )
    entry_cap, entry_mode = entry_trailing_limit_cap(
        wallet_exposure_limit_ratio=_float(inputs.get("wallet_exposure_limit_ratio")),
        risk_we_excess_allowance_fraction=_float(inputs.get("risk_we_excess_allowance_fraction")),
        total_wallet_exposure_limit_ratio=_float(inputs.get("total_wallet_exposure_limit_ratio")),
        entry_trailing_retracement_base_fraction=_float(inputs.get("entry_trailing_retracement_base_fraction")),
        wallet_exposure=wallet_exposure,
    )
    return {
        "symbol": str(inputs.get("symbol", "")),
        "pside": str(inputs.get("pside", "long")),
        "inputs": normalized_inputs,
        "extrema": trailing_bundle,
        "wallet_exposure": wallet_exposure,
        "allowed_wallet_exposure_limit_ratio": allowed_limit,
        "entry_limit_cap": entry_cap,
        "entry_mode": entry_mode,
        "entry": build_trailing_entry_diagnostic(inputs),
        "close": build_trailing_close_diagnostic(inputs),
    }


def snapshot_payload(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = snapshot.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return snapshot


def log_range_input_required(params: Mapping[str, Any], timeframe: str, *, branch=None) -> Optional[bool]:
    """Resolve diagnostic consumption from complete canonical strategy coefficients."""
    suffix = {"m1": "1m", "h1": "1h"}[timeframe]
    try:
        offset = f"offset_log_range_{suffix}_coefficient"
        if offset in params:
            values = [params[offset]]
        else:
            values = []
            for name in ((branch,) if branch is not None else ("entry", "close")):
                section = params[name]
                values.extend(section[f"{kind}_log_range_{suffix}_coefficient"]
                              for kind in ("threshold", "retracement"))
        if any(isinstance(value, bool) for value in values):
            return None
        values = [float(value) for value in values]
        return any(value != 0.0 for value in values) if all(math.isfinite(v) for v in values) else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _snapshot_indicator(value: Any, field: str, *, positive: bool = False) -> float:
    """Snapshot imports cannot turn an unavailable observed input into a wizard default."""
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("unavailable")
        result = float(value)
        if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
            raise ValueError("unavailable")
        return result
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"snapshot indicator unavailable: {field}; obtain a ready snapshot or use wizard mode") from None


def _snapshot_logrange_unused(config, symbol, pside, timeframe):
    from config.strategy import build_runtime_strategy_side, get_active_strategy_side

    bot = config.get("bot", {})
    kind = bot.get("portfolio", {}).get("strategy_kind")
    side = bot.get(pside, {})
    if kind != "trailing_martingale" or not isinstance(side.get("strategy", {}).get(kind), Mapping):
        return False
    overrides = config.get("coin_overrides", {})
    if any(key != symbol for key in overrides):
        # The offline importer has no exchange market resolver. An authored coin
        # alias may refer to this symbol; never guess that an override is unrelated.
        return False
    params = build_runtime_strategy_side(
        get_active_strategy_side(side, strategy_kind=kind, pside=pside),
        strategy_kind=kind, pside=pside,
        override_side=overrides.get(symbol, {}).get("bot", {}).get(pside, {}),
    )
    return log_range_input_required(params, timeframe) is False


def build_trailing_inputs_from_snapshot(
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    symbol: str,
    pside: str,
) -> dict[str, Any]:
    snap = snapshot_payload(snapshot)
    market = snap.get("market", {})
    positions = snap.get("positions", {})
    account = snap.get("account", {})
    trailing = snap.get("trailing", {})
    if not isinstance(market, Mapping) or symbol not in market:
        raise KeyError(f"snapshot missing market entry for {symbol}")
    if not isinstance(positions, Mapping):
        raise KeyError("snapshot missing positions section")
    if not isinstance(account, Mapping):
        raise KeyError("snapshot missing account section")
    market_entry = market[symbol]
    if not isinstance(market_entry, Mapping):
        raise KeyError(f"snapshot market entry for {symbol} is invalid")
    pos_entry = positions.get(symbol, {})
    if not isinstance(pos_entry, Mapping):
        pos_entry = {}
    side_position = pos_entry.get(pside, {})
    if not isinstance(side_position, Mapping):
        side_position = {}
    side_market_bands = market_entry.get("ema_bands", {})
    if not isinstance(side_market_bands, Mapping) or not isinstance(
        side_market_bands.get(pside), Mapping
    ):
        raise KeyError(f"snapshot missing market.ema_bands.{pside} for {symbol}")
    side_band = side_market_bands[pside]
    h1_by_side = market_entry.get("entry_log_range_ema_1h", {})
    if not isinstance(h1_by_side, Mapping) or pside not in h1_by_side:
        raise KeyError(
            f"snapshot missing market.entry_log_range_ema_1h.{pside} for {symbol}; "
            "restart the bot with a newer monitor snapshot or use wizard mode"
        )
    if "c_mult" not in market_entry:
        raise KeyError(
            f"snapshot missing market.c_mult for {symbol}; restart the bot with a newer monitor snapshot or use wizard mode"
        )
    observed_inputs = market_entry.get("strategy_inputs", {})
    observed_side = observed_inputs.get(pside, {}) if isinstance(observed_inputs, Mapping) else {}
    observed_values = {}
    unused = []
    if isinstance(observed_side, Mapping):
        for metric in ("m1_log_range", "h1_log_range"):
            if metric not in observed_side:
                continue
            row = observed_side[metric]
            field = f"market.strategy_inputs.{pside}.{metric}"
            if (isinstance(row, Mapping) and row.get("status") == "not_required"
                    and row.get("required") is False
                    and _snapshot_logrange_unused(config, symbol, pside, metric.split("_")[0])):
                # Current config independently proves this dimension inert. This is
                # diagnostic transport, not an invented observed indicator value.
                observed_values[metric] = 0.0
                unused.append(metric)
                continue
            if not isinstance(row, Mapping) or row.get("status") != "available":
                raise ValueError(f"snapshot indicator unavailable: {field}; obtain a ready snapshot or use wizard mode")
            observed_values[metric] = _snapshot_indicator(row.get("value"), field)
    if "m1_log_range" not in observed_values:
        if not _snapshot_logrange_unused(config, symbol, pside, "m1"):
            raise ValueError(f"snapshot indicator unavailable: market.strategy_inputs.{pside}.m1_log_range; obtain a ready snapshot or use wizard mode")
        observed_values["m1_log_range"] = 0.0
        unused.append("m1_log_range")
    h1_log_range = (0.0 if h1_by_side[pside] is None and "h1_log_range" in unused else
                   _snapshot_indicator(h1_by_side[pside], f"market.entry_log_range_ema_1h.{pside}"))
    symbol_trailing = trailing.get(symbol, {}) if isinstance(trailing, Mapping) else {}
    side_trailing = symbol_trailing.get(pside, {}) if isinstance(symbol_trailing, Mapping) else {}
    if not isinstance(side_trailing, Mapping):
        side_trailing = {}
    extrema_source = side_trailing.get("extrema")
    if not isinstance(extrema_source, Mapping):
        raw_market_trailing = market_entry.get("trailing", {})
        if isinstance(raw_market_trailing, Mapping) and isinstance(raw_market_trailing.get(pside), Mapping):
            extrema_source = raw_market_trailing[pside]
        else:
            extrema_source = {}
    balance_raw = account.get("balance_raw", account.get("balance"))
    bot_cfg = config.get("bot", {})
    if not isinstance(bot_cfg, Mapping) or not isinstance(bot_cfg.get(pside), Mapping):
        raise KeyError(f"config missing bot.{pside} section")
    side_cfg = flatten_shared_bot_side(bot_cfg[pside])
    out: dict[str, Any] = {
        "symbol": symbol,
        "pside": pside,
        "balance_raw": _float(balance_raw),
        "current_price": _float(market_entry.get("last_price")),
        "position_size": _float(side_position.get("size")),
        "position_price": _float(side_position.get("price")),
        "qty_step": _float(market_entry.get("qty_step")),
        "price_step": _float(market_entry.get("price_step")),
        "min_qty": _float(market_entry.get("min_qty")),
        "min_cost": max(
            _float(market_entry.get("effective_min_cost")),
            _float(market_entry.get("min_cost")),
        ),
        "c_mult": _float(market_entry.get("c_mult")),
        "ema_lower": _snapshot_indicator(side_band.get("lower"), f"market.ema_bands.{pside}.lower", positive=True),
        "ema_upper": _snapshot_indicator(side_band.get("upper"), f"market.ema_bands.{pside}.upper", positive=True),
        "h1_log_range_ema": h1_log_range,
    }
    if "m1_log_range" in observed_values:
        out["m1_log_range_ema"] = observed_values["m1_log_range"]
    if unused:
        out["not_required_indicators"] = unused
    out.update(normalize_trailing_extrema(extrema_source))
    for key in sorted(set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS)):
        if key == "wallet_exposure_limit_ratio":
            out[key] = _effective_wallet_exposure_limit_ratio(config, pside=pside)
        else:
            out[key] = _float(side_cfg.get(key))
    return out
