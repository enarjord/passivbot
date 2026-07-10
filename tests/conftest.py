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

    def _hsl_no_restart_triggered(
        restart_after_red_policy, drawdown_raw, drawdown_ema, no_restart_drawdown_threshold
    ):
        # Mirrors ehsl::no_restart_triggered exactly (max(raw, ema) contract).
        if restart_after_red_policy == "always":
            return False
        if restart_after_red_policy == "threshold":
            return max(float(drawdown_raw), float(drawdown_ema)) >= float(
                no_restart_drawdown_threshold
            )
        if restart_after_red_policy == "never":
            return True
        raise ValueError(
            "hsl_restart_after_red_policy must be one of always, threshold, never; "
            f"got {restart_after_red_policy!r}"
        )

    stub.hsl_no_restart_triggered = _hsl_no_restart_triggered

    def _hsl_coin_drawdown_signal(
        *, balance, n_positions, peak_realized, last_realized, current_upnl
    ):
        balance = float(balance)
        n_positions = int(n_positions)
        peak_realized = float(peak_realized)
        last_realized = float(last_realized)
        current_upnl = float(current_upnl)
        if not math.isfinite(balance) or balance <= 0.0:
            raise ValueError("balance must be finite and > 0")
        if n_positions <= 0:
            raise ValueError("n_positions must be > 0")
        for name, value in (
            ("peak_realized", peak_realized),
            ("last_realized", last_realized),
            ("current_upnl", current_upnl),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        slot_budget = balance / n_positions
        drawdown_usd = max(0.0, peak_realized - (last_realized + current_upnl))
        return {
            "slot_budget": slot_budget,
            "drawdown_usd": drawdown_usd,
            "drawdown_raw": drawdown_usd / slot_budget,
        }

    stub.hsl_coin_drawdown_signal = _hsl_coin_drawdown_signal

    def _hsl_red_episode_finalization(
        *,
        restart_after_red_policy,
        stop_timestamp_ms,
        stop_equity,
        stop_peak_strategy_equity,
        previous_no_restart_peak_strategy_equity,
        drawdown_ema,
        red_threshold,
        no_restart_drawdown_threshold,
        cooldown_minutes_after_red,
    ):
        if not (0.0 < float(red_threshold) <= float(no_restart_drawdown_threshold) <= 1.0):
            raise ValueError(
                "no_restart_drawdown_threshold must satisfy red_threshold <= threshold <= 1"
            )
        peak = max(
            float(previous_no_restart_peak_strategy_equity),
            float(stop_peak_strategy_equity),
            float(stop_equity),
        )
        raw = max(0.0, 1.0 - float(stop_equity) / peak)
        no_restart = _hsl_no_restart_triggered(
            restart_after_red_policy,
            raw,
            drawdown_ema,
            no_restart_drawdown_threshold,
        )
        cooldown_until_ms = None
        if not no_restart and float(cooldown_minutes_after_red) > 0.0:
            cooldown_ms = max(1, round(float(cooldown_minutes_after_red) * 60_000.0))
            cooldown_until_ms = int(stop_timestamp_ms) + int(cooldown_ms)
        return {
            "no_restart_peak_strategy_equity": peak,
            "no_restart_drawdown_raw": raw,
            "no_restart_latched": no_restart,
            "cooldown_until_ms": cooldown_until_ms,
            "disposition": (
                "no_restart"
                if no_restart
                else "cooldown"
                if cooldown_until_ms is not None
                else "halted_no_cooldown"
            ),
        }

    stub.hsl_red_episode_finalization = _hsl_red_episode_finalization
    stub.round_ = _round
    stub.round_dn = _round_dn
    stub.round_up = _round_up
    stub.round_dynamic = _identity
    stub.round_dynamic_up = _identity
    stub.round_dynamic_dn = _identity
    stub.calc_pnl_long = (
        lambda entry_price, close_price, qty, c_mult=1.0: (close_price - entry_price) * qty
    )
    stub.calc_pnl_short = (
        lambda entry_price, close_price, qty, c_mult=1.0: (entry_price - close_price) * qty
    )
    stub.calc_pprice_diff_int = lambda *args, **kwargs: 0

    def _calc_auto_unstuck_allowance(balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last):
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
    stub.qty_to_cost = lambda qty, price, c_mult=1.0: qty * price * (c_mult if c_mult else 1.0)

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
    stub.get_order_id_type_from_string = lambda name: _order_map.get(name, 0)
    stub.order_type_id_to_snake = lambda type_id: {v: k for k, v in _order_map.items()}.get(
        type_id, "other"
    )
    stub.order_type_snake_to_id = lambda name: _order_map.get(name, 0)

    stub.run_backtest = lambda *args, **kwargs: {}
    stub.gate_entries_by_twel_py = lambda *args, **kwargs: []
    stub.calc_twel_enforcer_orders_py = lambda *args, **kwargs: []

    # Minimal stub for orchestrator JSON API
    def _compute_ideal_orders_json(input_json: str) -> str:
        """Stub orchestrator that returns empty orders."""
        import json

        return json.dumps({"orders": []})

    stub.compute_ideal_orders_json = _compute_ideal_orders_json

    sys.modules["passivbot_rust"] = stub


_install_passivbot_rust_stub()
