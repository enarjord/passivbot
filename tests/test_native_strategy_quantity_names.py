"""Canonical quantity names reach the native strategy and packet boundaries."""

from copy import deepcopy
import inspect
import json

import passivbot_rust as pbr
import pytest

from config import get_template_config
from config.param_paths import resolve_optimizer_key_path
from config.strategy import build_runtime_strategy_side
from config.strategy_paths import STRATEGY_NATIVE_TO_PUBLIC_PATHS
from optimization.gpu.model import EMA_ANCHOR_PARAM_KEYS, TRAILING_MARTINGALE_PARAM_KEYS
from test_orchestrator_json_api import make_input, make_symbol


FIELDS = [
    ("trailing_martingale", ("entry", "initial_qty_pct"), ("entry", "initial_qty_fraction"), "entry_initial_qty_fraction"),
    ("trailing_martingale", ("entry", "initial_ema_dist"), ("entry", "initial_ema_distance_fraction"), "entry_initial_ema_distance_fraction"),
    ("trailing_martingale", ("entry", "double_down_factor"), ("entry", "reentry_qty_multiplier"), "entry_reentry_qty_multiplier"),
    ("trailing_martingale", ("entry", "threshold_base_pct"), ("entry", "threshold_base_fraction"), "entry_threshold_base_fraction"),
    ("trailing_martingale", ("entry", "threshold_we_weight"), ("entry", "threshold_exposure_ratio_coefficient"), "entry_threshold_exposure_ratio_coefficient"),
    ("trailing_martingale", ("entry", "retracement_base_pct"), ("entry", "retracement_base_fraction"), "entry_retracement_base_fraction"),
    ("trailing_martingale", ("entry", "retracement_we_weight"), ("entry", "retracement_exposure_ratio_coefficient"), "entry_retracement_exposure_ratio_coefficient"),
    ("trailing_martingale", ("close", "qty_pct"), ("close", "qty_fraction"), "close_qty_fraction"),
    ("trailing_martingale", ("close", "threshold_base_pct"), ("close", "threshold_base_fraction"), "close_threshold_base_fraction"),
    ("trailing_martingale", ("close", "threshold_we_weight"), ("close", "threshold_exposure_ratio_coefficient"), "close_threshold_exposure_ratio_coefficient"),
    ("trailing_martingale", ("close", "retracement_base_pct"), ("close", "retracement_base_fraction"), "close_retracement_base_fraction"),
    ("ema_anchor", ("base_qty_pct",), ("base_qty_fraction",), "base_qty_fraction"),
    ("ema_anchor", ("offset",), ("offset_fraction",), "offset_fraction"),
    ("ema_anchor", ("entry_double_down_factor",), ("entry_qty_exposure_coefficient",), "entry_qty_exposure_coefficient"),
    ("ema_anchor", ("offset_psize_weight",), ("inventory_offset_exposure_coefficient",), "inventory_offset_exposure_coefficient"),
]


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    assert not getattr(pbr, "__is_stub__", False), "requires the compiled Rust extension"
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()


@pytest.mark.parametrize("kind,old,path,native_id", FIELDS)
def test_quantity_metadata_and_optimizer_bindings_need_no_runtime_alias(kind, old, path, native_id):
    native = pbr.get_strategy_spec(kind)
    for side in ("long", "short"):
        record = next(row for row in native["parameters"]
                      if row["side"] == side and tuple(row["config_path"][2:]) == path)
        assert record["name"] == native_id
        assert record["optimize_key"] == f"{side}_{native_id}"
        config = get_template_config()
        config["bot"]["portfolio"]["strategy_kind"] = kind
        assert resolve_optimizer_key_path(config, record["optimize_key"]) == (
            "bot", side, "strategy", kind, *path,
        )
        runtime = build_runtime_strategy_side(config["bot"][side], strategy_kind=kind, pside=side)
        current = runtime
        for part in path:
            current = current[part]
        assert current == record["default"]
    assert old not in STRATEGY_NATIVE_TO_PUBLIC_PATHS[kind]
    assert not any(tuple(row["config_path"][2:]) == old for row in native["parameters"])


@pytest.mark.parametrize("kind,old,path,native_id", FIELDS)
def test_native_parser_rejects_retired_quantity_keys(kind, old, path, native_id):
    config = get_template_config()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    strategy = build_runtime_strategy_side(config["bot"]["long"], strategy_kind=kind, pside="long")
    if kind == "ema_anchor":
        strategy.update(ema_span_0=10.0, ema_span_1=20.0)
    else:
        strategy["entry"].update(ema_span_0=10.0, ema_span_1=20.0)
        strategy.update(volatility_ema_span_1m=10.0, volatility_ema_span_1h=0.0)
    symbol = make_symbol(0, bid=100.0, ask=100.1,
                         long_strategy=strategy, short_strategy=deepcopy(strategy))
    payload = make_input(balance=1000.0, strategy_kind=kind, symbols=[symbol])
    # Validate the current packet first, so the retired-key assertion cannot pass
    # merely because some other required trading input is absent.
    pbr.compute_ideal_orders_json(json.dumps(payload))
    changed = payload["symbols"][0]["long"]["strategy_params"]
    for part in path[:-1]:
        changed = changed[part]
    changed[old[-1]] = changed.pop(path[-1])
    with pytest.raises(ValueError, match=old[-1]):
        pbr.compute_ideal_orders_json(json.dumps(payload))


def test_quantity_parameter_columns_keep_the_existing_gpu_order():
    assert EMA_ANCHOR_PARAM_KEYS[:10] == (
        "base_qty_fraction", "ema_span_0", "ema_span_1",
        "entry_qty_exposure_coefficient", "offset_fraction",
        "inventory_offset_exposure_coefficient", "offset_volatility_1h_weight",
        "offset_volatility_1m_weight", "offset_volatility_ema_span_1h",
        "offset_volatility_ema_span_1m",
    )
    assert TRAILING_MARTINGALE_PARAM_KEYS[:23] == (
        "ema_span_0", "ema_span_1", "volatility_ema_span_1h", "volatility_ema_span_1m",
        "entry_reentry_qty_multiplier", "entry_initial_ema_distance_fraction",
        "entry_initial_qty_fraction", "entry_threshold_base_fraction",
        "entry_threshold_exposure_ratio_coefficient", "entry_threshold_volatility_1h_weight",
        "entry_threshold_volatility_1m_weight", "entry_retracement_base_fraction",
        "entry_retracement_exposure_ratio_coefficient", "entry_retracement_volatility_1h_weight",
        "entry_retracement_volatility_1m_weight", "close_qty_fraction",
        "close_threshold_base_fraction", "close_threshold_exposure_ratio_coefficient",
        "close_threshold_volatility_1h_weight", "close_threshold_volatility_1m_weight",
        "close_retracement_base_fraction", "close_retracement_volatility_1h_weight",
        "close_retracement_volatility_1m_weight",
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_direct_entry_keywords_keep_ignored_mirrors_in_their_original_positions(side):
    parameters = tuple(inspect.signature(getattr(pbr, f"calc_entries_{side}_py")).parameters)
    assert parameters[5:16] == (
        "entry_grid_reentry_qty_multiplier", "entry_grid_threshold_base_fraction",
        "entry_initial_ema_distance_fraction", "entry_initial_qty_fraction",
        "entry_trailing_reentry_qty_multiplier", "entry_trailing_retracement_base_fraction",
        "entry_trailing_threshold_base_fraction", "entry_weight_volatility_1h",
        "entry_weight_volatility_1m", "entry_exposure_ratio_coefficient", "wallet_exposure_limit_ratio",
    )
