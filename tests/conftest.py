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

        mod = importlib.import_module("passivbot_rust")
        if hasattr(mod, "select_forager_candidates_py"):
            return
        sys.modules.pop("passivbot_rust", None)
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
        mod = importlib.import_module("passivbot_rust")
        if hasattr(mod, "select_forager_candidates_py"):
            return
        sys.modules.pop("passivbot_rust", None)
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

    def _calc_order_price_diff(side, price, market):
        s = str(side).strip().lower()
        if s in ("buy", "long"):
            return 0.0 if not market else (1 - price / market)
        elif s in ("sell", "short"):
            return 0.0 if not market else (price / market - 1)
        else:
            raise ValueError(f"invalid side: {side!r}")

    stub.calc_order_price_diff = _calc_order_price_diff
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

    def _calc_pprice_diff_int(pside, pprice, price):
        if not pprice or not math.isfinite(pprice) or pprice <= 0:
            return 0.0
        if pside == 0:  # LONG
            return (pprice - price) / pprice
        else:  # SHORT
            return (price - pprice) / pprice

    stub.calc_pprice_diff_int = _calc_pprice_diff_int
    stub.calc_pside_price_diff_int = _calc_pprice_diff_int
    stub.calc_price_diff_pside_int = _calc_pprice_diff_int

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

    def _hysteresis(value, previous, pct):
        if previous is None or abs(previous) < 1e-12:
            return value
        if abs(value - previous) / abs(previous) <= pct:
            return previous
        return value

    stub.hysteresis = _hysteresis
    stub.trailing_bundle_default_py = lambda: (0.0, 0.0, 0.0, 0.0)
    stub.update_trailing_bundle_py = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    stub.calc_next_entry_long_py = lambda *args, **kwargs: (0.0, 0.0, "entry_trailing_normal_long")
    stub.calc_next_entry_short_py = lambda *args, **kwargs: (0.0, 0.0, "entry_trailing_normal_short")
    stub.calc_next_close_long_py = lambda *args, **kwargs: (0.0, 0.0, "close_trailing_long")
    stub.calc_next_close_short_py = lambda *args, **kwargs: (0.0, 0.0, "close_trailing_short")
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

    def _normalize_higher(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return [1.0 for _ in values]
        lo = min(finite)
        hi = max(finite)
        if abs(hi - lo) <= 1e-12:
            return [1.0 if math.isfinite(float(v)) else 0.0 for v in values]
        return [((float(v) - lo) / (hi - lo)) if math.isfinite(float(v)) else 0.0 for v in values]

    def _normalize_lower(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return [1.0 for _ in values]
        lo = min(finite)
        hi = max(finite)
        if abs(hi - lo) <= 1e-12:
            return [1.0 if math.isfinite(float(v)) else 0.0 for v in values]
        return [((hi - float(v)) / (hi - lo)) if math.isfinite(float(v)) else 0.0 for v in values]

    def _canonicalize_weights(weights):
        normalized = {
            "volume": float(weights["volume"]),
            "ema_readiness": float(weights["ema_readiness"]),
            "volatility": float(weights["volatility"]),
        }
        if any((not math.isfinite(value)) or value < 0.0 for value in normalized.values()):
            raise ValueError("forager_score_weights must be finite and non-negative")
        total = sum(normalized.values())
        if total <= 0.0:
            return {"volume": 1.0, "ema_readiness": 0.0, "volatility": 0.0}
        return {key: value / total for key, value in normalized.items()}

    def _select_coin_indices_py(py_features, slots_to_fill, volume_drop_pct, weights, require_forager):
        weights = _canonicalize_weights(weights)
        enabled_positions = [i for i, feature in enumerate(py_features) if feature["enabled"]]
        if not enabled_positions:
            return []
        if not require_forager:
            return [py_features[i]["index"] for i in enabled_positions]

        keep = round(len(enabled_positions) * (1.0 - max(0.0, min(1.0, float(volume_drop_pct)))))
        keep = max(1, min(len(enabled_positions), max(int(slots_to_fill), int(keep))))
        enabled_positions = sorted(
            enabled_positions,
            key=lambda pos: (-float(py_features[pos]["volume_score"]), py_features[pos]["index"]),
        )[:keep]
        volume_scores = _normalize_higher([py_features[pos]["volume_score"] for pos in enabled_positions])
        ema_scores = _normalize_lower(
            [py_features[pos]["ema_readiness_score"] for pos in enabled_positions]
        )
        volatility_scores = _normalize_higher(
            [py_features[pos]["volatility_score"] for pos in enabled_positions]
        )
        scored = []
        for i, pos in enumerate(enabled_positions):
            score = (
                float(weights["volume"]) * volume_scores[i]
                + float(weights["ema_readiness"]) * ema_scores[i]
                + float(weights["volatility"]) * volatility_scores[i]
            )
            scored.append((py_features[pos]["index"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [index for index, _score in scored[: max(int(slots_to_fill), 0)]]

    def _select_forager_candidates_py(
        py_candidates, pside, slots_to_fill, volume_drop_pct, weights, require_forager
    ):
        features = []
        pside_l = str(pside).lower()
        if pside_l not in ("long", "short"):
            raise ValueError(f"invalid forager position side: {pside}")
        for candidate in py_candidates:
            entry_dist = float(candidate["entry_initial_ema_dist"])
            if pside_l == "long":
                threshold = float(candidate["ema_lower"]) * (1.0 - entry_dist)
                readiness = float(candidate["bid"]) / threshold - 1.0
            else:
                threshold = float(candidate["ema_upper"]) * (1.0 + entry_dist)
                readiness = 1.0 - float(candidate["ask"]) / threshold
            if not math.isfinite(readiness):
                raise ValueError(
                    f"invalid forager candidate input 'forager_ema_readiness' at index {candidate['index']}"
                )
            features.append(
                {
                    "index": candidate["index"],
                    "enabled": candidate["enabled"],
                    "volume_score": candidate["volume_score"],
                    "volatility_score": candidate["volatility_score"],
                    "ema_readiness_score": readiness,
                }
            )
        return _select_coin_indices_py(
            features, slots_to_fill, volume_drop_pct, weights, require_forager
        )

    stub.compute_ideal_orders_json = _compute_ideal_orders_json
    stub.select_coin_indices_py = _select_coin_indices_py
    stub.select_forager_candidates_py = _select_forager_candidates_py

    sys.modules["passivbot_rust"] = stub


_install_passivbot_rust_stub()
