import math
import os
import sys
import types

# Ensure we can import modules from the src/ directory as "downloader"
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
        lambda entry_price, close_price, qty, c_mult=1.0: (close_price - entry_price) * qty
    )
    stub.calc_pnl_short = (
        lambda entry_price, close_price, qty, c_mult=1.0: (entry_price - close_price) * qty
    )
    stub.calc_pprice_diff_int = lambda *args, **kwargs: 0
    stub.calc_auto_unstuck_allowance = (
        lambda balance, allowance_pct, max_pnl, last_pnl: allowance_pct * balance
    )
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

    _order_map = {
        "close_unstuck_long": 0x1234,
        "close_unstuck_short": 0x1235,
        "entry_initial_normal_long": 0x2001,
        "entry_initial_normal_short": 0x2002,
        "entry_grid_normal_long": 0x2003,
        "entry_grid_normal_short": 0x2004,
        "close_grid_long": 0x3001,
        "close_grid_short": 0x3002,
        "close_auto_reduce_wel_long": 0x3003,
        "close_auto_reduce_wel_short": 0x3004,
        "close_panic_long": 0x3005,
        "close_panic_short": 0x3006,
    }
    stub.get_order_id_type_from_string = lambda name: _order_map.get(name, 0)
    stub.order_type_id_to_snake = lambda type_id: {v: k for k, v in _order_map.items()}.get(
        type_id, "other"
    )
    stub.order_type_snake_to_id = lambda name: _order_map.get(name, 0)

    stub.run_backtest = lambda *args, **kwargs: {}
    stub.gate_entries_by_twel_py = lambda *args, **kwargs: []
    stub.calc_twel_enforcer_orders_py = lambda *args, **kwargs: []

    sys.modules["passivbot_rust"] = stub


_install_passivbot_rust_stub()
