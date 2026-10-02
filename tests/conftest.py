import math
import os
import sys
import types

# Ensure we can import modules from the src/ directory directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _install_passivbot_rust_stub():
    if "passivbot_rust" in sys.modules:
        return

    try:
        import importlib

        importlib.import_module("passivbot_rust")
        return
    except Exception:
        pass

    # If pytest is launched outside the venv, try the project venv site-packages
    # before falling back to the lightweight stub.
    try:
        import importlib

        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site = os.path.join(ROOT_DIR, "venv", "lib", pyver, "site-packages")
        if os.path.isdir(venv_site) and venv_site not in sys.path:
            sys.path.insert(0, venv_site)
        importlib.import_module("passivbot_rust")
        return
    except Exception:
        pass

    stub = types.ModuleType("passivbot_rust")
    stub.__is_stub__ = True

    def _identity(x, *_args, **_kwargs):
        return x

    def _round(value, step):
        if step == 0:
            return value
        return round(value / step) * step

    def _round_up(value, step):
        if step == 0:
            return value
        return math.ceil(value / step) * step

    def _round_dn(value, step):
        if step == 0:
            return value
        return math.floor(value / step) * step

    stub.calc_diff = lambda price, reference: price - reference
    stub.calc_order_price_diff = lambda side, price, market: (
        (0.0 if not market else (1 - price / market))
        if str(side).lower() in ("buy", "long")
        else (0.0 if not market else (price / market - 1))
    )
    stub.calc_min_entry_qty = lambda *args, **kwargs: 0.0
    stub.calc_min_entry_qty_py = stub.calc_min_entry_qty

    stub.round_ = _round
    stub.round_dn = _round_dn
    stub.round_up = _round_up
    stub.round_dynamic = _identity
    stub.round_dynamic_up = _identity
    stub.round_dynamic_dn = _identity
    stub.calc_pnl_long = (
        lambda entry_price, close_price, qty, c_mult=1.0: (close_price - entry_price)
        * qty
    )
    stub.calc_pnl_short = (
        lambda entry_price, close_price, qty, c_mult=1.0: (entry_price - close_price)
        * qty
    )
    stub.calc_pprice_diff_int = lambda *args, **kwargs: 0

    def _calc_auto_unstuck_allowance(
        balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last
    ):
        balance_peak = balance + (pnl_cumsum_max - pnl_cumsum_last)
        drop_since_peak_pct = balance / balance_peak - 1.0
        return max(0.0, balance_peak * (loss_allowance_pct + drop_since_peak_pct))

    stub.calc_auto_unstuck_allowance = _calc_auto_unstuck_allowance
    stub.calc_wallet_exposure = (
        lambda c_mult, balance, size, price: abs(size) * price / max(balance, 1e-12)
    )
    stub.cost_to_qty = lambda cost, price, c_mult=1.0: (
        0.0 if price == 0 else cost / (price * (c_mult if c_mult else 1.0))
    )
    stub.qty_to_cost = (
        lambda qty, price, c_mult=1.0: qty * price * (c_mult if c_mult else 1.0)
    )

    stub.hysteresis = _identity
    stub.calc_entries_long_py = lambda *args, **kwargs: []
    stub.calc_entries_short_py = lambda *args, **kwargs: []
    stub.calc_closes_long_py = lambda *args, **kwargs: []
    stub.calc_closes_short_py = lambda *args, **kwargs: []
    stub.calc_unstucking_close_py = lambda *args, **kwargs: None

    # Order type IDs must match passivbot_rust exactly
    _order_map = {
        "entry_initial_normal_long": 0,
        "entry_initial_partial_long": 1,
        "entry_trailing_normal_long": 2,
        "entry_trailing_cropped_long": 3,
        "entry_grid_normal_long": 4,
        "entry_grid_cropped_long": 5,
        "entry_grid_inflated_long": 6,
        "close_grid_long": 7,
        "close_trailing_long": 8,
        "close_unstuck_long": 9,
        "close_auto_reduce_twel_long": 10,
        "entry_initial_normal_short": 11,
        "entry_initial_partial_short": 12,
        "entry_trailing_normal_short": 13,
        "entry_trailing_cropped_short": 14,
        "entry_grid_normal_short": 15,
        "entry_grid_cropped_short": 16,
        "entry_grid_inflated_short": 17,
        "close_grid_short": 18,
        "close_trailing_short": 19,
        "close_unstuck_short": 20,
        "close_auto_reduce_twel_short": 21,
        "close_panic_long": 22,
        "close_panic_short": 23,
        "close_auto_reduce_wel_long": 24,
        "close_auto_reduce_wel_short": 25,
        "entry_ema_anchor_long": 26,
        "close_ema_anchor_long": 27,
        "entry_ema_anchor_short": 28,
        "close_ema_anchor_short": 29,
        "empty": 65535,
    }
    _order_id_map = {v: k for k, v in _order_map.items()}

    def _order_type_snake_to_id(name):
        if name not in _order_map:
            raise ValueError("unknown order type name")
        return _order_map[name]

    def _order_type_id_to_snake(type_id):
        if type_id not in _order_id_map:
            raise ValueError("unknown order type id")
        return _order_id_map[type_id]

    stub.get_order_id_type_from_string = _order_type_snake_to_id
    stub.order_type_id_to_snake = _order_type_id_to_snake
    stub.order_type_snake_to_id = _order_type_snake_to_id

    stub.run_backtest = lambda *args, **kwargs: {}
    stub.gate_entries_by_twel_py = lambda *args, **kwargs: []
    stub.calc_twel_enforcer_orders_py = lambda *args, **kwargs: []

    # Minimal stub for orchestrator JSON API
    def _compute_ideal_orders_json(input_json: str) -> str:
        """Stub orchestrator that returns empty orders."""
        import json

        payload = json.loads(input_json)
        global_bot_params = payload.get("global", {}).get("global_bot_params", {})
        symbols = payload.get("symbols", [])
        hedge_mode = payload.get("global", {}).get("hedge_mode", True)
        active_indices = {"long": set(), "short": set()}
        for pside in ("long", "short"):
            side_params = global_bot_params.get(pside, {})
            global_side_enabled = (
                float(side_params.get("total_wallet_exposure_limit", 0.0)) > 0.0
                and int(side_params.get("n_positions", 0)) > 0
            )
            eligible_indices = [
                symbol["symbol_idx"]
                for symbol in symbols
                if bool(symbol.get("tradable", False))
                and float(
                    symbol[pside]
                    .get("bot_params", {})
                    .get("wallet_exposure_limit", 0.0)
                )
                != 0.0
            ]
            held_indices = [
                symbol["symbol_idx"]
                for symbol in symbols
                if float(symbol[pside].get("position", {}).get("size", 0.0)) != 0.0
                and symbol[pside].get("mode") != "manual"
            ]
            forced_indices = [
                symbol["symbol_idx"]
                for symbol in symbols
                if symbol["symbol_idx"] in eligible_indices
                and symbol[pside].get("mode") == "normal"
            ]
            effective_n_positions = max(
                min(int(side_params.get("n_positions", 0)), len(eligible_indices)),
                len(forced_indices),
            )
            active_indices[pside].update(held_indices)
            if global_side_enabled:
                candidates = forced_indices + [
                    symbol_idx
                    for symbol_idx in eligible_indices
                    if symbol_idx not in forced_indices
                ]
                for symbol_idx in candidates:
                    if len(active_indices[pside]) >= effective_n_positions:
                        break
                    symbol = symbols[symbol_idx]
                    opposite = "short" if pside == "long" else "long"
                    if (
                        not hedge_mode
                        and float(symbol[opposite].get("position", {}).get("size", 0.0))
                        != 0.0
                    ):
                        continue
                    if symbol[pside].get("mode") in (None, "normal"):
                        active_indices[pside].add(symbol_idx)
        symbol_states = []
        for symbol in symbols:
            row = {"symbol_idx": symbol["symbol_idx"]}
            for pside in ("long", "short"):
                input_mode = symbol[pside].get("mode")
                position_size = float(
                    symbol[pside].get("position", {}).get("size", 0.0)
                )
                has_position = position_size != 0.0
                effective_mode = (
                    "normal"
                    if input_mode is None
                    or (input_mode == "graceful_stop" and has_position)
                    else input_mode
                )
                wallet_exposure_limit = float(
                    symbol[pside]
                    .get("bot_params", {})
                    .get("wallet_exposure_limit", 0.0)
                )
                side_params = global_bot_params.get(pside, {})
                global_side_enabled = (
                    float(side_params.get("total_wallet_exposure_limit", 0.0)) > 0.0
                    and int(side_params.get("n_positions", 0)) > 0
                )
                symbol_side_eligible = (
                    bool(symbol.get("tradable", False)) and wallet_exposure_limit != 0.0
                )
                active = (
                    symbol_side_eligible
                    and symbol["symbol_idx"] in active_indices[pside]
                )
                if not global_side_enabled:
                    effective_mode = "manual"
                opposite = "short" if pside == "long" else "long"
                one_way_blocked = not hedge_mode and (
                    float(symbol[opposite].get("position", {}).get("size", 0.0)) != 0.0
                    or (
                        not has_position
                        and pside == "short"
                        and symbol["symbol_idx"] in active_indices["long"]
                    )
                )
                row[pside] = {
                    "input_mode": input_mode,
                    "effective_mode": effective_mode,
                    "active": active,
                    "allow_initial": (
                        active
                        and global_side_enabled
                        and effective_mode == "normal"
                        and not one_way_blocked
                    ),
                }
            symbol_states.append(row)
        return json.dumps(
            {
                "orders": [],
                "diagnostics": {
                    "warnings": [],
                    "symbol_states": symbol_states,
                    "loss_gate_blocks": [],
                    "min_effective_cost_blocks": [],
                    "forager_selections": [],
                },
            }
        )

    stub.compute_ideal_orders_json = _compute_ideal_orders_json

    sys.modules["passivbot_rust"] = stub


_install_passivbot_rust_stub()
