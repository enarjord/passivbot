from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import load_prepared_config
from trailing_diagnostics import (
    CLOSE_CONFIG_KEYS,
    ENTRY_CONFIG_KEYS,
    NUMERIC_INPUT_KEYS,
    TRAILING_EXTREMA_KEYS,
    build_trailing_diagnostic,
    build_trailing_inputs_from_snapshot,
    snapshot_payload,
)


EDITABLE_INPUT_KEYS = [
    "balance_raw",
    "current_price",
    "position_size",
    "position_price",
    "ema_lower",
    "ema_upper",
    "h1_log_range_ema",
    "qty_step",
    "price_step",
    "min_qty",
    "min_cost",
    "c_mult",
    *TRAILING_EXTREMA_KEYS,
    *ENTRY_CONFIG_KEYS,
    *CLOSE_CONFIG_KEYS,
]

WIZARD_CORE_KEYS = [
    "balance_raw",
    "current_price",
    "position_size",
    "position_price",
    "ema_lower",
    "ema_upper",
    "h1_log_range_ema",
    "m1_log_range_ema",
    *TRAILING_EXTREMA_KEYS,
    "wallet_exposure_limit",
    "risk_we_excess_allowance_pct",
    "entry_trailing_grid_ratio",
    "entry_trailing_threshold_pct",
    "entry_trailing_retracement_pct",
    "entry_weight_volatility_1h",
    "entry_weight_volatility_1m",
    "entry_we_weight",
    "close_trailing_grid_ratio",
    "close_trailing_qty_pct",
    "close_trailing_threshold_pct",
    "close_trailing_retracement_pct",
]

WIZARD_ADVANCED_KEYS = [
    "qty_step",
    "price_step",
    "min_qty",
    "min_cost",
    "c_mult",
    "entry_grid_double_down_factor",
    "entry_grid_spacing_pct",
    "entry_weight_volatility_1h",
    "entry_weight_volatility_1m",
    "entry_we_weight",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_trailing_double_down_factor",
    "close_grid_markup_end",
    "close_grid_markup_start",
    "close_grid_qty_pct",
    "risk_wel_enforcer_threshold",
]


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _wrap_box(title: str, lines: list[str], width: int) -> list[str]:
    width = max(20, width)
    inner = max(1, width - 4)
    out = [
        "+" + "-" * (width - 2) + "+",
        f"| {_truncate(title, inner):<{inner}} |",
        "|" + "-" * (width - 2) + "|",
    ]
    for line in lines or ["-"]:
        out.append(f"| {_truncate(line, inner):<{inner}} |")
    out.append("+" + "-" * (width - 2) + "+")
    return out


def _pad_lines(lines: list[str], height: int, width: int) -> list[str]:
    padded = list(lines[:height])
    while len(padded) < height:
        padded.append(" " * width)
    return [line.ljust(width)[:width] for line in padded]


def _combine_columns(left: list[str], right: list[str], left_width: int, right_width: int) -> list[str]:
    height = max(len(left), len(right))
    left_padded = _pad_lines(left, height, left_width)
    right_padded = _pad_lines(right, height, right_width)
    return [f"{l} {r}" for l, r in zip(left_padded, right_padded)]


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_compact(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text


def _fmt_ratio_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100.0:+.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _symbol_aliases(symbol: str) -> set[str]:
    alias_set = {symbol.upper()}
    base = symbol.split(":")[0]
    alias_set.add(base.upper())
    alias_set.add(base.replace("/", "").upper())
    alias_set.add(base.split("/")[0].upper())
    return alias_set


def _resolve_symbol_alias(query: str, snapshot: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    cleaned = query.strip().upper()
    market = snapshot_payload(snapshot).get("market", {})
    if not isinstance(market, dict):
        return None, "snapshot has no market section"
    matches = [symbol for symbol in sorted(market) if cleaned in _symbol_aliases(symbol)]
    if not matches:
        return None, f"no symbol matched '{query}'"
    if len(matches) > 1:
        return None, f"ambiguous symbol '{query}': {', '.join(matches[:4])}"
    return matches[0], None


def _load_snapshot_from_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_snapshot_path(
    *,
    monitor_root: Optional[str],
    exchange: Optional[str],
    user: Optional[str],
    snapshot_path: Optional[str],
) -> Optional[Path]:
    if snapshot_path:
        return Path(snapshot_path).expanduser()
    if not monitor_root:
        return None
    root = Path(monitor_root).expanduser()
    if exchange and user:
        return root / exchange / user / "state.latest.json"
    matches = sorted(root.glob("*/*/state.latest.json"))
    return matches[-1] if matches else None


def _manual_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "symbol": "BTC/USDT:USDT",
        "pside": "long",
        "balance_raw": 1000.0,
        "current_price": 70000.0,
        "position_size": 0.0,
        "position_price": 70000.0,
        "qty_step": 0.001,
        "price_step": 0.1,
        "min_qty": 0.001,
        "min_cost": 5.0,
        "c_mult": 1.0,
        "ema_lower": 69500.0,
        "ema_upper": 70500.0,
        "h1_log_range_ema": 0.0,
        "min_since_open": 69500.0,
        "max_since_min": 70000.0,
        "max_since_open": 70500.0,
        "min_since_max": 70000.0,
    }
    for key in set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS):
        defaults[key] = 0.0
    defaults["wallet_exposure_limit"] = 1.7
    return defaults


def _snapshot_seed_defaults(snapshot: dict[str, Any], *, symbol: str, pside: str) -> dict[str, Any]:
    snap = snapshot_payload(snapshot)
    market = snap.get("market", {})
    positions = snap.get("positions", {})
    if not isinstance(market, dict) or symbol not in market:
        raise KeyError(f"snapshot missing market entry for {symbol}")
    market_entry = market[symbol]
    if not isinstance(market_entry, dict):
        raise KeyError(f"snapshot market entry for {symbol} is invalid")
    pos = positions.get(symbol, {}).get(pside, {}) if isinstance(positions, dict) else {}
    if not isinstance(pos, dict):
        pos = {}
    side_band = market_entry.get("ema_bands", {}).get(pside, {})
    trailing = snap.get("trailing", {}).get(symbol, {}).get(pside, {})
    extrema = trailing.get("extrema")
    if not isinstance(extrema, dict):
        extrema = market_entry.get("trailing", {}).get(pside, {})
    out = _manual_defaults()
    out.update(
        {
            "symbol": symbol,
            "pside": pside,
            "balance_raw": float(snap.get("account", {}).get("balance_raw", out["balance_raw"])),
            "current_price": float(market_entry.get("last_price", out["current_price"]) or out["current_price"]),
            "position_size": float(pos.get("size", out["position_size"]) or out["position_size"]),
            "position_price": float(pos.get("price", out["position_price"]) or out["position_price"]),
            "qty_step": float(market_entry.get("qty_step", out["qty_step"]) or out["qty_step"]),
            "price_step": float(market_entry.get("price_step", out["price_step"]) or out["price_step"]),
            "min_qty": float(market_entry.get("min_qty", out["min_qty"]) or out["min_qty"]),
            "min_cost": max(
                float(market_entry.get("effective_min_cost", 0.0) or 0.0),
                float(market_entry.get("min_cost", out["min_cost"]) or out["min_cost"]),
            ),
            "c_mult": float(market_entry.get("c_mult", out["c_mult"]) or out["c_mult"]),
            "ema_lower": float(side_band.get("lower", out["ema_lower"]) or out["ema_lower"]),
            "ema_upper": float(side_band.get("upper", out["ema_upper"]) or out["ema_upper"]),
            "h1_log_range_ema": float(
                market_entry.get("entry_volatility_logrange_ema", {}).get(pside, out["h1_log_range_ema"])
                or out["h1_log_range_ema"]
            ),
        }
    )
    if isinstance(extrema, dict):
        for key in TRAILING_EXTREMA_KEYS:
            out[key] = float(extrema.get(key, out[key]) or out[key])
    return out


def _prompt_text(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else (default or "")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def _prompt_bool(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter yes or no.")


def prompt_manual_wizard(base_inputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = deepcopy(base_inputs) if base_inputs is not None else _manual_defaults()
    defaults.setdefault("symbol", "BTC/USDT:USDT")
    defaults.setdefault("pside", "long")
    out = deepcopy(defaults)
    out["symbol"] = _prompt_text("Symbol", str(defaults.get("symbol", "BTC/USDT:USDT")))
    out["pside"] = _prompt_text("Position side", str(defaults.get("pside", "long"))).lower() or "long"
    for key in WIZARD_CORE_KEYS:
        out[key] = _prompt_float(key, float(defaults.get(key, 0.0) or 0.0))
    if _prompt_bool("Edit advanced sizing/grid parameters too?", default=False):
        for key in WIZARD_ADVANCED_KEYS:
            out[key] = _prompt_float(key, float(defaults.get(key, 0.0) or 0.0))
    return out


@dataclass
class TrailingDiagnosticsState:
    source_label: str
    config_path: Optional[str] = None
    snapshot_path: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    snapshot: Optional[dict[str, Any]] = None
    symbol: str = "BTC/USDT:USDT"
    pside: str = "long"
    baseline_inputs: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    status_lines: list[str] = field(default_factory=list)

    def diagnostic(self) -> dict[str, Any]:
        return build_trailing_diagnostic(self.inputs)

    def reset(self) -> None:
        self.inputs = deepcopy(self.baseline_inputs)
        self.symbol = str(self.inputs.get("symbol", self.symbol))
        self.pside = str(self.inputs.get("pside", self.pside))

    def changed_keys(self) -> list[str]:
        changed: list[str] = []
        for key in sorted(self.inputs):
            if self.inputs.get(key) != self.baseline_inputs.get(key):
                changed.append(key)
        return changed

    def can_reload_from_snapshot(self) -> bool:
        return self.config is not None and self.snapshot is not None

    def reload_symbol_pside(self, symbol: str, pside: str) -> None:
        if not self.can_reload_from_snapshot():
            raise RuntimeError("symbol/side reload requires snapshot + config source")
        fresh = build_trailing_inputs_from_snapshot(
            self.config,
            self.snapshot,
            symbol=symbol,
            pside=pside,
        )
        self.symbol = symbol
        self.pside = pside
        self.baseline_inputs = deepcopy(fresh)
        self.inputs = deepcopy(fresh)


def create_state_from_sources(
    *,
    config_path: Optional[str],
    monitor_root: Optional[str],
    exchange: Optional[str],
    user: Optional[str],
    snapshot_path: Optional[str],
    symbol: Optional[str],
    pside: str,
    wizard: bool,
) -> TrailingDiagnosticsState:
    resolved_snapshot = _resolve_snapshot_path(
        monitor_root=monitor_root,
        exchange=exchange,
        user=user,
        snapshot_path=snapshot_path,
    )
    config = (
        load_prepared_config(config_path, live_only=True, verbose=False, target="canonical")
        if config_path
        else None
    )
    snapshot = _load_snapshot_from_path(resolved_snapshot) if resolved_snapshot and resolved_snapshot.exists() else None
    if snapshot is not None:
        if symbol:
            resolved_symbol, error = _resolve_symbol_alias(symbol, snapshot)
            if error:
                raise ValueError(error)
            symbol = resolved_symbol
        else:
            snap = snapshot_payload(snapshot)
            trailing = snap.get("trailing", {})
            if isinstance(trailing, dict) and trailing:
                symbol = sorted(trailing)[0]
            else:
                market = snap.get("market", {})
                if isinstance(market, dict) and market:
                    symbol = sorted(market)[0]
    if wizard or snapshot is None or config is None:
        base_inputs = None
        source_label = "manual wizard"
        if snapshot is not None and symbol:
            base_inputs = _snapshot_seed_defaults(snapshot, symbol=symbol, pside=pside)
            source_label = f"wizard seeded from {resolved_snapshot}"
        elif config is not None:
            base_inputs = _manual_defaults()
            for key in sorted(set(ENTRY_CONFIG_KEYS + CLOSE_CONFIG_KEYS)):
                base_inputs[key] = float(config["bot"][pside].get(key, 0.0) or 0.0)
            base_inputs["pside"] = pside
        inputs = prompt_manual_wizard(base_inputs)
        return TrailingDiagnosticsState(
            source_label=source_label,
            config_path=config_path,
            snapshot_path=str(resolved_snapshot) if resolved_snapshot else None,
            config=config,
            snapshot=snapshot,
            symbol=str(inputs["symbol"]),
            pside=str(inputs["pside"]),
            baseline_inputs=deepcopy(inputs),
            inputs=deepcopy(inputs),
            status_lines=["Wizard inputs loaded."],
        )
    if symbol is None:
        raise ValueError("no symbol available from snapshot; specify --symbol or use --wizard")
    inputs = build_trailing_inputs_from_snapshot(config, snapshot, symbol=symbol, pside=pside)
    return TrailingDiagnosticsState(
        source_label=f"snapshot={resolved_snapshot} config={config_path}",
        config_path=config_path,
        snapshot_path=str(resolved_snapshot) if resolved_snapshot else None,
        config=config,
        snapshot=snapshot,
        symbol=symbol,
        pside=pside,
        baseline_inputs=deepcopy(inputs),
        inputs=deepcopy(inputs),
        status_lines=["Loaded trailing diagnostics from snapshot + config."],
    )


def _render_summary_box(state: TrailingDiagnosticsState, diagnostic: dict[str, Any], width: int) -> list[str]:
    changed = state.changed_keys()
    lines = [
        f"source={state.source_label}",
        f"symbol={state.symbol} pside={state.pside} snapshot={state.snapshot_path or '-'}",
        (
            f"balance={_fmt_compact(state.inputs.get('balance_raw'))} "
            f"current={_fmt_compact(state.inputs.get('current_price'))} "
            f"pos={_fmt_compact(state.inputs.get('position_size'))}@{_fmt_compact(state.inputs.get('position_price'))}"
        ),
        (
            f"WE={_fmt_compact(diagnostic.get('wallet_exposure'))} "
            f"limit={_fmt_compact(diagnostic.get('allowed_wallet_exposure_limit'))} "
            f"entry_cap={_fmt_compact(diagnostic.get('entry_limit_cap'))} "
            f"mode={diagnostic.get('entry_mode') or '-'}"
        ),
        f"changed={', '.join(changed[:8]) if changed else '-'}",
    ]
    return _wrap_box("Summary", lines, width)


def _render_market_box(state: TrailingDiagnosticsState, width: int) -> list[str]:
    i = state.inputs
    lines = [
        (
            f"price={_fmt_compact(i.get('current_price'))} "
            f"ema_lo={_fmt_compact(i.get('ema_lower'))} "
            f"ema_hi={_fmt_compact(i.get('ema_upper'))} "
            f"h1_lr={_fmt_compact(i.get('h1_log_range_ema'))}"
        ),
        (
            f"qty_step={_fmt_compact(i.get('qty_step'))} "
            f"price_step={_fmt_compact(i.get('price_step'))} "
            f"min_qty={_fmt_compact(i.get('min_qty'))} "
            f"min_cost={_fmt_compact(i.get('min_cost'))} "
            f"c_mult={_fmt_compact(i.get('c_mult'))}"
        ),
        (
            f"min_open={_fmt_compact(i.get('min_since_open'))} "
            f"max_min={_fmt_compact(i.get('max_since_min'))}"
        ),
        (
            f"max_open={_fmt_compact(i.get('max_since_open'))} "
            f"min_max={_fmt_compact(i.get('min_since_max'))}"
        ),
    ]
    return _wrap_box("Market / State", lines, width)


def _render_side_box(title: str, payload: Optional[dict[str, Any]], width: int) -> list[str]:
    if payload is None:
        return _wrap_box(title, ["No trailing diagnostic for current inputs."], width)
    lines = [
        (
            f"status={payload.get('status')} type={payload.get('order_type')} "
            f"qty={_fmt_compact(payload.get('qty'))} px={_fmt_compact(payload.get('price'))}"
        ),
        (
            f"cur={_fmt_compact(payload.get('current_price'))} "
            f"thr={_fmt_compact(payload.get('threshold_price'))} "
            f"met={payload.get('threshold_met')} "
            f"ret={_fmt_compact(payload.get('retracement_price'))} "
            f"met={payload.get('retracement_met')}"
        ),
        (
            f"thr_pct={_fmt_ratio_pct(payload.get('threshold_pct'))} "
            f"ret_pct={_fmt_ratio_pct(payload.get('retracement_pct'))}"
        ),
        (
            f"vs_thr={_fmt_ratio_pct(payload.get('current_vs_threshold_ratio'))} "
            f"vs_ret={_fmt_ratio_pct(payload.get('current_vs_retracement_ratio'))}"
        ),
    ]
    if payload.get("kind") == "entry":
        lines.append(f"mode={payload.get('mode', '-')}")
    extrema = payload.get("extrema", {})
    if isinstance(extrema, dict):
        lines.append(
            "extrema "
            f"min_open={_fmt_compact(extrema.get('min_since_open'))} "
            f"max_min={_fmt_compact(extrema.get('max_since_min'))} "
            f"max_open={_fmt_compact(extrema.get('max_since_open'))} "
            f"min_max={_fmt_compact(extrema.get('min_since_max'))}"
        )
    return _wrap_box(title, lines, width)


def _render_inputs_box(state: TrailingDiagnosticsState, width: int) -> list[str]:
    important = [
        "entry_trailing_grid_ratio",
        "entry_trailing_threshold_pct",
        "entry_trailing_retracement_pct",
        "entry_weight_volatility_1h",
        "entry_weight_volatility_1m",
        "entry_we_weight",
        "close_trailing_grid_ratio",
        "close_trailing_qty_pct",
        "close_trailing_threshold_pct",
        "close_trailing_retracement_pct",
        "wallet_exposure_limit",
        "risk_we_excess_allowance_pct",
        "risk_wel_enforcer_threshold",
    ]
    lines = [f"{key}={_fmt_compact(state.inputs.get(key))}" for key in important]
    return _wrap_box("Config Inputs", lines, width)


def _render_status_box(state: TrailingDiagnosticsState, width: int) -> list[str]:
    lines = state.status_lines[-6:] or [
        "Use 'set <key> <value>' to edit one parameter.",
        "Type 'help' for commands.",
    ]
    lines.append("> ")
    return _wrap_box("Command", lines, width)


def render_screen(state: TrailingDiagnosticsState, *, width: Optional[int] = None) -> str:
    term_width = width or max(100, shutil.get_terminal_size((120, 40)).columns - 1)
    diagnostic = state.diagnostic()
    if term_width >= 140:
        left_width = max(50, term_width // 2)
        right_width = term_width - left_width - 1
        left = []
        left.extend(_render_summary_box(state, diagnostic, left_width))
        left.extend(_render_market_box(state, left_width))
        left.extend(_render_inputs_box(state, left_width))
        left.extend(_render_status_box(state, left_width))
        right = []
        right.extend(_render_side_box("Entry", diagnostic.get("entry"), right_width))
        right.extend(_render_side_box("Close", diagnostic.get("close"), right_width))
        return "\n".join(_combine_columns(left, right, left_width, right_width))
    sections: list[str] = []
    for box in (
        _render_summary_box(state, diagnostic, term_width),
        _render_market_box(state, term_width),
        _render_side_box("Entry", diagnostic.get("entry"), term_width),
        _render_side_box("Close", diagnostic.get("close"), term_width),
        _render_inputs_box(state, term_width),
        _render_status_box(state, term_width),
    ):
        sections.extend(box)
    return "\n".join(sections)


def _parse_numeric_value(raw: str) -> float:
    return float(raw)


def _write_dump(state: TrailingDiagnosticsState) -> str:
    dump_dir = Path("tmp")
    dump_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dump_dir / f"trailing_diagnostics_dump_{ts}.json"
    payload = {
        "source": state.source_label,
        "symbol": state.symbol,
        "pside": state.pside,
        "inputs": state.inputs,
        "diagnostic": state.diagnostic(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def execute_command(state: TrailingDiagnosticsState, command: str) -> bool:
    raw = command.strip()
    if not raw:
        state.status_lines = ["Type 'help' for commands."]
        return False
    parts = raw.split()
    cmd = parts[0].lower()
    args = parts[1:]
    if cmd in {"quit", "exit"}:
        return True
    if cmd == "help":
        state.status_lines = [
            "Commands:",
            "  set <key> <value>     edit one numeric input or config field",
            "  edit <key> <value>    alias for set",
            "  symbol <coin|symbol>  switch symbol from loaded snapshot",
            "  side <long|short>     switch side from loaded snapshot",
            "  reset                 restore current symbol/side baseline",
            "  wizard                prompt for core trailing fields manually",
            "  dump                  write current inputs + diagnostic to tmp/",
            "  list                  show editable keys",
            "  quit                  exit the tool",
        ]
        return False
    if cmd == "list":
        state.status_lines = [
            "Editable keys:",
            ", ".join(EDITABLE_INPUT_KEYS[:9]),
            ", ".join(EDITABLE_INPUT_KEYS[9:18]),
            ", ".join(EDITABLE_INPUT_KEYS[18:27]),
            ", ".join(EDITABLE_INPUT_KEYS[27:36]),
            ", ".join(EDITABLE_INPUT_KEYS[36:]),
        ]
        return False
    if cmd == "reset":
        state.reset()
        state.status_lines = ["Restored baseline inputs for current symbol/side."]
        return False
    if cmd == "dump":
        path = _write_dump(state)
        state.status_lines = [f"Dumped diagnostic to {path}"]
        return False
    if cmd == "wizard":
        state.inputs = prompt_manual_wizard(state.inputs)
        state.symbol = str(state.inputs.get("symbol", state.symbol))
        state.pside = str(state.inputs.get("pside", state.pside))
        state.baseline_inputs = deepcopy(state.inputs)
        state.status_lines = ["Wizard values applied as new baseline."]
        return False
    if cmd == "symbol":
        if not args:
            state.status_lines = ["symbol requires a coin or full symbol"]
            return False
        if state.snapshot is None or state.config is None:
            state.status_lines = ["symbol switching requires snapshot + config source"]
            return False
        resolved, error = _resolve_symbol_alias(" ".join(args), state.snapshot)
        if error:
            state.status_lines = [error]
            return False
        state.reload_symbol_pside(str(resolved), state.pside)
        state.status_lines = [f"Loaded symbol {resolved} ({state.pside}) from snapshot + config."]
        return False
    if cmd == "side":
        if not args or args[0].lower() not in {"long", "short"}:
            state.status_lines = ["side requires 'long' or 'short'"]
            return False
        new_pside = args[0].lower()
        if state.snapshot is None or state.config is None:
            state.inputs["pside"] = new_pside
            state.pside = new_pside
            state.status_lines = [f"Updated side to {new_pside} in manual mode."]
            return False
        state.reload_symbol_pside(state.symbol, new_pside)
        state.status_lines = [f"Loaded side {new_pside} for {state.symbol} from snapshot + config."]
        return False
    if cmd in {"set", "edit"}:
        if len(args) < 2:
            state.status_lines = [f"{cmd} requires '<key> <value>'"]
            return False
        key = args[0]
        value_raw = " ".join(args[1:])
        if key not in EDITABLE_INPUT_KEYS and key not in NUMERIC_INPUT_KEYS:
            state.status_lines = [f"unknown key '{key}'"]
            return False
        try:
            state.inputs[key] = _parse_numeric_value(value_raw)
        except ValueError:
            state.status_lines = [f"invalid numeric value '{value_raw}'"]
            return False
        state.status_lines = [f"Set {key}={state.inputs[key]}"]
        return False
    state.status_lines = [f"unknown command '{cmd}'"]
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive trailing diagnostics explorer for Passivbot configs and monitor snapshots."
    )
    parser.add_argument("--config", type=str, default=None, help="Live config path (HJSON/JSON).")
    parser.add_argument("--monitor-root", type=str, default="monitor", help="Monitor root directory.")
    parser.add_argument("--exchange", type=str, default=None, help="Exchange name for snapshot lookup.")
    parser.add_argument("--user", type=str, default=None, help="User/account name for snapshot lookup.")
    parser.add_argument("--snapshot-path", type=str, default=None, help="Explicit state.latest.json path.")
    parser.add_argument("--symbol", type=str, default=None, help="Initial symbol or coin alias.")
    parser.add_argument("--pside", type=str, default="long", help="Initial position side (long/short).")
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Start in manual wizard mode instead of loading from monitor snapshot + config.",
    )
    parser.add_argument(
        "--print-once",
        action="store_true",
        help="Render once and exit instead of entering the interactive command loop.",
    )
    return parser


def run_interactive(args: argparse.Namespace) -> int:
    state = create_state_from_sources(
        config_path=args.config,
        monitor_root=args.monitor_root,
        exchange=args.exchange,
        user=args.user,
        snapshot_path=args.snapshot_path,
        symbol=args.symbol,
        pside=str(args.pside).lower(),
        wizard=bool(args.wizard),
    )
    while True:
        print("\x1b[2J\x1b[H", end="")
        print(render_screen(state))
        if args.print_once:
            return 0
        try:
            command = input("Cmd> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if execute_command(state, command):
            return 0
