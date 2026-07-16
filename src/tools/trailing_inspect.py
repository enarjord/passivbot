from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from copy import deepcopy
from typing import Any, Mapping, Sequence


STRATEGY_KIND = "trailing_martingale"

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
    threshold_price = (
        position_price * (1.0 + threshold_direction * threshold_pct)
        if threshold_gate_active
        else None
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
    entry_threshold_base = max(
        0.0,
        _finite_float(entry.get("threshold_base_pct", 0.0), "entry.threshold_base_pct"),
    )
    entry_retracement_base = _finite_float(
        entry.get("retracement_base_pct", 0.0),
        "entry.retracement_base_pct",
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
            "trailing_enabled": entry_retracement_base > 0.0,
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
        if geometry["threshold_gate_active"]:
            lines.append(
                f"  Passive threshold/reference: {_fmt_pct(payload['threshold_pct'])} "
                f"{geometry['threshold_direction']} position price -> "
                f"{_fmt_number(geometry['threshold_price'])}"
            )
        else:
            lines.append("  Effective threshold <= 0; passive reference is at the current market touch")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool trailing-inspect",
        description=(
            "Inspect trailing_martingale entry and close thresholds without starting a bot. "
            "Percent inputs use config ratios (0.01 = 1%)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  passivbot tool trailing-inspect --symbol COIN --side long "
            "--position-size 150 --position-price 20 --wallet-exposure 0.6 "
            "--effective-wallet-exposure-limit 0.9 --volatility-ema-1m 0.007 "
            "--volatility-ema-1h 0.0033\n\n"
            "Add --config path/to/config.json to use that side's active strategy parameters. "
            "Any parameter flag below overrides the config/default value."
        ),
    )
    state = parser.add_argument_group("position and market state")
    state.add_argument("--symbol", default="COIN", help="Display label only")
    state.add_argument("--side", choices=("long", "short"), default="long")
    state.add_argument("--position-size", type=float, default=None, help="Display-only position size")
    state.add_argument("--position-price", type=float, required=True)
    state.add_argument("--wallet-exposure", type=float, required=True, help="Current WE ratio")
    state.add_argument(
        "--effective-wallet-exposure-limit",
        type=float,
        required=True,
        help="Effective per-position WEL used by the strategy",
    )
    state.add_argument("--volatility-ema-1m", type=float, default=0.0)
    state.add_argument("--volatility-ema-1h", type=float, default=0.0)
    state.add_argument(
        "--config",
        default=None,
        help="Canonical config source; otherwise Rust-owned defaults are used",
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
        params, source = load_parameter_source(args.config, args.side)
        overridden = apply_parameter_overrides(params, args)
        result = inspect_trailing(
            symbol=args.symbol,
            pside=args.side,
            position_price=args.position_price,
            position_size=args.position_size,
            wallet_exposure=args.wallet_exposure,
            effective_wallet_exposure_limit=args.effective_wallet_exposure_limit,
            volatility_ema_1m=args.volatility_ema_1m,
            volatility_ema_1h=args.volatility_ema_1h,
            params=params,
            parameter_source=source,
            overridden_parameters=overridden,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
