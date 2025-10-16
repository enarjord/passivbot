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

    stub = types.ModuleType("passivbot_rust")

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

    stub.hysteresis_rounding = _identity
    stub.calc_entries_long_py = lambda *args, **kwargs: []
    stub.calc_entries_short_py = lambda *args, **kwargs: []
    stub.calc_closes_long_py = lambda *args, **kwargs: []
    stub.calc_closes_short_py = lambda *args, **kwargs: []

    stub.get_order_id_type_from_string = lambda name: {
        "close_unstuck_long": 0x1234,
        "close_unstuck_short": 0x1235,
    }.get(name, 0)
    stub.order_type_id_to_snake = lambda type_id: {
        0x1234: "close_unstuck_long",
        0x1235: "close_unstuck_short",
    }.get(type_id, "other")
    stub.order_type_snake_to_id = lambda name: {
        "close_unstuck_long": 0x1234,
        "close_unstuck_short": 0x1235,
    }.get(name, 0)

    stub.run_backtest = lambda *args, **kwargs: {}

    sys.modules["passivbot_rust"] = stub


_install_passivbot_rust_stub()
