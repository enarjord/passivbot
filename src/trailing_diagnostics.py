from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

import passivbot_rust as pbr
from config.shared_bot import flatten_shared_bot_side


ENTRY_CONFIG_KEYS = [
    "entry_grid_double_down_factor",
    "entry_grid_spacing_volatility_weight",
    "entry_grid_spacing_we_weight",
    "entry_grid_spacing_pct",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_trailing_double_down_factor",
    "entry_trailing_grid_ratio",
    "entry_trailing_retracement_pct",
    "entry_trailing_retracement_we_weight",
    "entry_trailing_retracement_volatility_weight",
    "entry_trailing_threshold_pct",
    "entry_trailing_threshold_we_weight",
    "entry_trailing_threshold_volatility_weight",
    "wallet_exposure_limit",
    "risk_we_excess_allowance_pct",
]


CLOSE_CONFIG_KEYS = [
    "close_grid_markup_end",
    "close_grid_markup_start",
    "close_grid_qty_pct",
    "close_trailing_grid_ratio",
    "close_trailing_qty_pct",
    "close_trailing_retracement_pct",
    "close_trailing_threshold_pct",
    "wallet_exposure_limit",
    "risk_we_excess_allowance_pct",
    "risk_wel_enforcer_threshold",
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
    *TRAILING_EXTREMA_KEYS,
    *sorted(set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS)),
]


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _effective_wallet_exposure_limit(
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
    direct = side_cfg.get("wallet_exposure_limit")
    if direct is not None:
        direct_value = _float(direct, default=0.0)
        if direct_value > 0.0:
            return direct_value
    twel = _float(side_cfg.get("total_wallet_exposure_limit"), default=0.0)
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


def wallet_exposure_limit_with_allowance(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
) -> float:
    return float(wallet_exposure_limit) * (1.0 + max(0.0, float(risk_we_excess_allowance_pct)))


def entry_trailing_limit_cap(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
    entry_trailing_grid_ratio: float,
    wallet_exposure: float,
) -> tuple[Optional[float], Optional[str]]:
    allowed_limit = wallet_exposure_limit_with_allowance(
        wallet_exposure_limit=wallet_exposure_limit,
        risk_we_excess_allowance_pct=risk_we_excess_allowance_pct,
    )
    if allowed_limit <= 0.0:
        return None, None
    trailing_ratio = float(entry_trailing_grid_ratio)
    if trailing_ratio >= 1.0 or trailing_ratio <= -1.0:
        return allowed_limit, "trailing_only"
    if trailing_ratio == 0.0:
        return None, "grid_only"
    wallet_exposure_ratio = wallet_exposure / allowed_limit if allowed_limit > 0.0 else 0.0
    if trailing_ratio > 0.0:
        if wallet_exposure_ratio < trailing_ratio:
            if wallet_exposure == 0.0:
                return allowed_limit, "trailing_first"
            return min(allowed_limit * trailing_ratio * 1.01, allowed_limit), "trailing_first"
        return None, "grid_first"
    if wallet_exposure_ratio < 1.0 + trailing_ratio:
        return None, "grid_first"
    return allowed_limit, "trailing_after_grid"


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
        wallet_exposure_limit=_float(inputs.get("wallet_exposure_limit")),
        risk_we_excess_allowance_pct=_float(inputs.get("risk_we_excess_allowance_pct")),
        entry_trailing_grid_ratio=_float(inputs.get("entry_trailing_grid_ratio")),
        wallet_exposure=wallet_exposure,
    )
    if limit_cap is None:
        return None
    ema_reference = _entry_ema_reference(inputs)
    if ema_reference <= 0.0 or current_price <= 0.0:
        return None
    trailing_bundle = normalize_trailing_extrema(inputs)
    entry_volatility_lr = _float(inputs.get("h1_log_range_ema"))
    common_args = [
        _float(inputs.get("qty_step")),
        _float(inputs.get("price_step")),
        _float(inputs.get("min_qty")),
        _float(inputs.get("min_cost")),
        c_mult,
        _float(inputs.get("entry_grid_double_down_factor")),
        _float(inputs.get("entry_grid_spacing_volatility_weight")),
        _float(inputs.get("entry_grid_spacing_we_weight")),
        _float(inputs.get("entry_grid_spacing_pct")),
        _float(inputs.get("entry_initial_ema_dist")),
        _float(inputs.get("entry_initial_qty_pct")),
        _float(inputs.get("entry_trailing_double_down_factor")),
        _float(inputs.get("entry_trailing_grid_ratio")),
        _float(inputs.get("entry_trailing_retracement_pct")),
        _float(inputs.get("entry_trailing_retracement_we_weight")),
        _float(inputs.get("entry_trailing_retracement_volatility_weight")),
        _float(inputs.get("entry_trailing_threshold_pct")),
        _float(inputs.get("entry_trailing_threshold_we_weight")),
        _float(inputs.get("entry_trailing_threshold_volatility_weight")),
        _float(inputs.get("wallet_exposure_limit")),
        _float(inputs.get("risk_we_excess_allowance_pct")),
        balance_raw,
        position_size,
        position_price,
        trailing_bundle["min_since_open"],
        trailing_bundle["max_since_min"],
        trailing_bundle["max_since_open"],
        trailing_bundle["min_since_max"],
        ema_reference,
        entry_volatility_lr,
        current_price,
    ]
    if pside == "long":
        qty, price, order_type = pbr.calc_next_entry_long_py(*common_args)
    else:
        qty, price, order_type = pbr.calc_next_entry_short_py(*common_args)
    order_type = str(order_type)
    if "trailing" not in order_type:
        return None

    threshold_multiplier = (
        (wallet_exposure / limit_cap) * _float(inputs.get("entry_trailing_threshold_we_weight"))
        if limit_cap > 0.0
        else 0.0
    )
    threshold_log_multiplier = entry_volatility_lr * _float(
        inputs.get("entry_trailing_threshold_volatility_weight")
    )
    threshold_pct = _float(inputs.get("entry_trailing_threshold_pct")) * max(
        0.0, 1.0 + threshold_multiplier + threshold_log_multiplier
    )

    retracement_multiplier = (
        (wallet_exposure / limit_cap) * _float(inputs.get("entry_trailing_retracement_we_weight"))
        if limit_cap > 0.0
        else 0.0
    )
    retracement_log_multiplier = entry_volatility_lr * _float(
        inputs.get("entry_trailing_retracement_volatility_weight")
    )
    retracement_pct = _float(inputs.get("entry_trailing_retracement_pct")) * max(
        0.0, 1.0 + retracement_multiplier + retracement_log_multiplier
    )

    if pside == "long":
        threshold_price = position_price * (1.0 - threshold_pct) if threshold_pct > 0.0 else None
        threshold_met = True if threshold_pct <= 0.0 else trailing_bundle["min_since_open"] < threshold_price
        retracement_price = (
            trailing_bundle["min_since_open"] * (1.0 + retracement_pct)
            if retracement_pct > 0.0
            else None
        )
        retracement_met = (
            True
            if retracement_pct <= 0.0
            else trailing_bundle["max_since_min"] > float(retracement_price)
        )
    else:
        threshold_price = position_price * (1.0 + threshold_pct) if threshold_pct > 0.0 else None
        threshold_met = True if threshold_pct <= 0.0 else trailing_bundle["max_since_open"] > threshold_price
        retracement_price = (
            trailing_bundle["max_since_open"] * (1.0 - retracement_pct)
            if retracement_pct > 0.0
            else None
        )
        retracement_met = (
            True
            if retracement_pct <= 0.0
            else trailing_bundle["min_since_max"] < float(retracement_price)
        )
    triggered = bool(abs(_float(qty)) > 0.0 and _float(price) > 0.0)
    return {
        "kind": "entry",
        "symbol": symbol,
        "pside": pside,
        "mode": mode,
        "order_type": order_type,
        "wallet_exposure": wallet_exposure,
        "limit_cap": float(limit_cap),
        "triggered": triggered,
        "status": trailing_status(
            triggered=triggered,
            threshold_met=bool(threshold_met),
            retracement_met=bool(retracement_met),
        ),
        "qty": float(qty),
        "price": float(price),
        "current_price": float(current_price),
        "threshold_pct": float(threshold_pct),
        "threshold_price": float(threshold_price) if threshold_price is not None else None,
        "threshold_met": bool(threshold_met),
        "retracement_pct": float(retracement_pct),
        "retracement_price": float(retracement_price) if retracement_price is not None else None,
        "retracement_met": bool(retracement_met),
        "current_vs_threshold_ratio": (
            float(current_price / threshold_price - 1.0)
            if threshold_price and threshold_price > 0.0
            else None
        ),
        "current_vs_retracement_ratio": (
            float(current_price / retracement_price - 1.0)
            if retracement_price and retracement_price > 0.0
            else None
        ),
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
    common_args = [
        _float(inputs.get("qty_step")),
        _float(inputs.get("price_step")),
        _float(inputs.get("min_qty")),
        _float(inputs.get("min_cost")),
        _float(inputs.get("c_mult")),
        _float(inputs.get("close_grid_markup_end")),
        _float(inputs.get("close_grid_markup_start")),
        _float(inputs.get("close_grid_qty_pct")),
        _float(inputs.get("close_trailing_grid_ratio")),
        _float(inputs.get("close_trailing_qty_pct")),
        _float(inputs.get("close_trailing_retracement_pct")),
        _float(inputs.get("close_trailing_threshold_pct")),
        _float(inputs.get("wallet_exposure_limit")),
        _float(inputs.get("risk_we_excess_allowance_pct")),
        _float(inputs.get("risk_wel_enforcer_threshold")),
        balance_raw,
        position_size,
        position_price,
        trailing_bundle["min_since_open"],
        trailing_bundle["max_since_min"],
        trailing_bundle["max_since_open"],
        trailing_bundle["min_since_max"],
        current_price,
    ]
    if pside == "long":
        qty, price, order_type = pbr.calc_next_close_long_py(*common_args)
    else:
        qty, price, order_type = pbr.calc_next_close_short_py(*common_args)
    order_type = str(order_type)
    if "trailing" not in order_type:
        return None

    threshold_pct = _float(inputs.get("close_trailing_threshold_pct"))
    retracement_pct = _float(inputs.get("close_trailing_retracement_pct"))
    if pside == "long":
        threshold_price = position_price * (1.0 + threshold_pct) if threshold_pct > 0.0 else None
        threshold_met = True if threshold_pct <= 0.0 else trailing_bundle["max_since_open"] > threshold_price
        retracement_price = (
            trailing_bundle["max_since_open"] * (1.0 - retracement_pct)
            if retracement_pct > 0.0
            else None
        )
        retracement_met = (
            True
            if retracement_pct <= 0.0
            else trailing_bundle["min_since_max"] < float(retracement_price)
        )
    else:
        threshold_price = position_price * (1.0 - threshold_pct) if threshold_pct > 0.0 else None
        threshold_met = True if threshold_pct <= 0.0 else trailing_bundle["min_since_open"] < threshold_price
        retracement_price = (
            trailing_bundle["min_since_open"] * (1.0 + retracement_pct)
            if retracement_pct > 0.0
            else None
        )
        retracement_met = (
            True
            if retracement_pct <= 0.0
            else trailing_bundle["max_since_min"] > float(retracement_price)
        )
    triggered = bool(abs(_float(qty)) > 0.0 and _float(price) > 0.0)
    return {
        "kind": "close",
        "symbol": symbol,
        "pside": pside,
        "order_type": order_type,
        "triggered": triggered,
        "status": trailing_status(
            triggered=triggered,
            threshold_met=bool(threshold_met),
            retracement_met=bool(retracement_met),
        ),
        "qty": float(qty),
        "price": float(price),
        "current_price": float(current_price),
        "threshold_pct": float(threshold_pct),
        "threshold_price": float(threshold_price) if threshold_price is not None else None,
        "threshold_met": bool(threshold_met),
        "retracement_pct": float(retracement_pct),
        "retracement_price": float(retracement_price) if retracement_price is not None else None,
        "retracement_met": bool(retracement_met),
        "current_vs_threshold_ratio": (
            float(current_price / threshold_price - 1.0)
            if threshold_price and threshold_price > 0.0
            else None
        ),
        "current_vs_retracement_ratio": (
            float(current_price / retracement_price - 1.0)
            if retracement_price and retracement_price > 0.0
            else None
        ),
        "extrema": deepcopy(trailing_bundle),
    }


def build_trailing_diagnostic(inputs: Mapping[str, Any]) -> dict[str, Any]:
    normalized_inputs = {key: inputs.get(key) for key in inputs}
    trailing_bundle = normalize_trailing_extrema(inputs)
    wallet_exposure = calculate_wallet_exposure(
        balance_raw=_float(inputs.get("balance_raw")),
        position_size=_float(inputs.get("position_size")),
        position_price=_float(inputs.get("position_price")),
        c_mult=_float(inputs.get("c_mult")),
    )
    allowed_limit = wallet_exposure_limit_with_allowance(
        wallet_exposure_limit=_float(inputs.get("wallet_exposure_limit")),
        risk_we_excess_allowance_pct=_float(inputs.get("risk_we_excess_allowance_pct")),
    )
    entry_cap, entry_mode = entry_trailing_limit_cap(
        wallet_exposure_limit=_float(inputs.get("wallet_exposure_limit")),
        risk_we_excess_allowance_pct=_float(inputs.get("risk_we_excess_allowance_pct")),
        entry_trailing_grid_ratio=_float(inputs.get("entry_trailing_grid_ratio")),
        wallet_exposure=wallet_exposure,
    )
    return {
        "symbol": str(inputs.get("symbol", "")),
        "pside": str(inputs.get("pside", "long")),
        "inputs": normalized_inputs,
        "extrema": trailing_bundle,
        "wallet_exposure": wallet_exposure,
        "allowed_wallet_exposure_limit": allowed_limit,
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
    h1_by_side = market_entry.get("entry_volatility_logrange_ema", {})
    if not isinstance(h1_by_side, Mapping) or pside not in h1_by_side:
        raise KeyError(
            f"snapshot missing market.entry_volatility_logrange_ema.{pside} for {symbol}; "
            "restart the bot with a newer monitor snapshot or use wizard mode"
        )
    if "c_mult" not in market_entry:
        raise KeyError(
            f"snapshot missing market.c_mult for {symbol}; restart the bot with a newer monitor snapshot or use wizard mode"
        )
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
    side_cfg = bot_cfg[pside]
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
        "ema_lower": _float(side_band.get("lower")),
        "ema_upper": _float(side_band.get("upper")),
        "h1_log_range_ema": _float(h1_by_side.get(pside)),
    }
    out.update(normalize_trailing_extrema(extrema_source))
    for key in sorted(set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS)):
        if key == "wallet_exposure_limit":
            out[key] = _effective_wallet_exposure_limit(config, pside=pside)
        else:
            out[key] = _float(side_cfg.get(key))
    return out
