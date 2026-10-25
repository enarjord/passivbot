"""Canonical native indicators preserve units, strict names and coupled sources."""

from copy import deepcopy
import json

import numpy as np
import passivbot_rust as pbr
import pytest

from config import get_template_config
from config.migrations.released_v8_strategy import RELEASED_V8_STRATEGY_PATH_RENAMES
from config.param_paths import resolve_optimizer_key_path
from config.strategy import build_runtime_strategy_side
from optimization.gpu.model import (
    COUPLED_UNSTUCK_EMA_SOURCE_KEYS, EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
)
from test_orchestrator_json_api import make_input, make_symbol


FIELDS = [(kind, old, new)
          for kind, mappings in RELEASED_V8_STRATEGY_PATH_RENAMES.items()
          for old, new in mappings.items()
          if "ema_span" in old[-1] or "volatility" in old[-1]]
assert len(FIELDS) == 18


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(pbr, "__is_stub__", False)
    verify_loaded_runtime_extension()


@pytest.mark.parametrize("kind,old,path", FIELDS)
def test_native_indicator_paths_and_gene_ids_match_the_authored_owner(kind, old, path):
    config = get_template_config()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    spec = pbr.get_strategy_spec(kind)
    for side in ("long", "short"):
        record = next(row for row in spec["parameters"]
                      if row["side"] == side and tuple(row["config_path"][2:]) == path)
        assert record["name"] == "_".join(path)
        assert record["optimize_key"] == side + "_" + "_".join(path)
        assert resolve_optimizer_key_path(config, record["optimize_key"]) == (
            "bot", side, "strategy", kind, *path,
        )
        assert not any(tuple(row["config_path"][2:]) == old for row in spec["parameters"])


@pytest.mark.parametrize("kind,old,path", FIELDS)
def test_native_parser_rejects_retired_indicator_keys(kind, old, path):
    config = get_template_config()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    strategy = build_runtime_strategy_side(config["bot"]["long"], strategy_kind=kind, pside="long")
    if kind == "ema_anchor":
        strategy.update(ema_span_0_minutes=10.0, ema_span_1_minutes=20.0)
    else:
        strategy["entry"].update(ema_span_0_minutes=10.0, ema_span_1_minutes=20.0)
        strategy.update(log_range_ema_span_minutes=10.0, log_range_ema_span_1h_candles=0.0)
    symbol = make_symbol(0, bid=100.0, ask=100.1,
                         long_strategy=strategy, short_strategy=deepcopy(strategy))
    packet = make_input(balance=1000.0, strategy_kind=kind, symbols=[symbol])
    pbr.compute_ideal_orders_json(json.dumps(packet))
    target = packet["symbols"][0]["long"]["strategy_params"]
    for part in path[:-1]:
        target = target[part]
    target[old[-1]] = target.pop(path[-1])
    with pytest.raises(ValueError, match=old[-1]):
        pbr.compute_ideal_orders_json(json.dumps(packet))


@pytest.mark.parametrize("kind,keys", [
    ("ema_anchor", EMA_ANCHOR_MULTICOIN_PARAM_KEYS),
    ("trailing_martingale", TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
])
@pytest.mark.parametrize("origin", ["base", "candidate", "fixed"])
def test_coupled_gpu_spans_use_each_strategy_owner_and_existing_precedence(kind, keys, origin):
    from optimization.gpu.service import _candidate_parameter_matrix

    sources = COUPLED_UNSTUCK_EMA_SOURCE_KEYS[kind]
    base = {key: 1.0 for key in keys}
    candidates = [{}]
    fixed = {"long": {}}
    for index, source in enumerate(sources.values()):
        base[source] = 12.5 + index * 25.25
        if origin in {"candidate", "fixed"}:
            candidates[0]["long_" + source] = 125.5 + index * 250.25
        if origin == "fixed":
            fixed["long"][source] = 7.5 + index * 15.25
    matrix = _candidate_parameter_matrix(
        candidates, keys, {"long": base}, static_overrides=fixed, couple_unstuck_emas=True,
    )
    for destination, source in sources.items():
        expected = fixed["long"].get(source, candidates[0].get("long_" + source, base[source]))
        assert matrix[0, keys.index(destination)] == expected
        assert matrix[0, keys.index(source)] == expected
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("kind,keys", [
    ("ema_anchor", EMA_ANCHOR_MULTICOIN_PARAM_KEYS),
    ("trailing_martingale", TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS),
])
def test_bootstrap_metadata_reads_each_strategy_price_columns_without_reordering(monkeypatch, kind, keys):
    from optimization.gpu import model
    from optimization.gpu.ema_bootstrap import BOOTSTRAP_COUNT_COLS, pack_bootstrap_counts

    strategy_keys = (model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS if kind == "ema_anchor"
                     else tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS))
    columns = (model.EMA_ANCHOR_COIN_OVERRIDE_COLS if kind == "ema_anchor"
               else model.TRAILING_MARTINGALE_COIN_OVERRIDE_COLS)
    sources = tuple(COUPLED_UNSTUCK_EMA_SOURCE_KEYS[kind].values())
    values = {key: 10.0 for key in keys}
    values[sources[0]], values[sources[1]] = 12.5, 37.75
    params = np.array([[values[key] for key in keys]], dtype=np.float64)
    pins = np.full((2, columns), np.nan, dtype=np.float64)
    pins[1, strategy_keys.index(sources[0])] = 71.5
    captured = []
    native_helper = pbr.gpu_ema_bootstrap_counts_py

    def record(values, interval, available):
        captured.append(np.array(values, copy=True))
        return native_helper(values, interval, available)

    monkeypatch.setattr(pbr, "gpu_ema_bootstrap_counts_py", record)
    counts = pack_bootstrap_counts(params, keys, (pins,), interval_minutes=5.0, available=6000)
    assert counts.shape == (1, 1, 2, BOOTSTRAP_COUNT_COLS)
    assert captured[0][:, :2].tolist() == [[12.5, 37.75], [71.5, 37.75]]
    assert counts.dtype == np.int32 and (counts > 0).all()


@pytest.mark.parametrize("kind,coefficient", [
    ("trailing_martingale", f"{leg}_{role}_log_range_1h_coefficient")
    for leg in ("entry", "close") for role in ("threshold", "retracement")
] + [("ema_anchor", "offset_log_range_1h_coefficient")])
def test_every_hourly_sensitivity_retains_its_required_input_admission(kind, coefficient):
    from optimization.gpu import model
    from optimization.gpu.ema_bootstrap import validate_hourly_bootstrap_inputs

    keys = (EMA_ANCHOR_MULTICOIN_PARAM_KEYS if kind == "ema_anchor"
            else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
    strategy_keys = (model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS if kind == "ema_anchor"
                     else tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS))
    columns = (model.EMA_ANCHOR_COIN_OVERRIDE_COLS if kind == "ema_anchor"
               else model.TRAILING_MARTINGALE_COIN_OVERRIDE_COLS)
    hour = ("offset_log_range_ema_span_1h_candles" if kind == "ema_anchor"
            else "log_range_ema_span_1h_candles")
    params = np.zeros((1, len(keys)), dtype=np.float64)
    for key, value in (("total_wallet_exposure_limit_ratio", 1.0), ("n_positions", 1.0),
                       (hour, 2.875), (coefficient, .375)):
        params[0, keys.index(key)] = value
    pins = np.full((1, columns), np.nan, dtype=np.float64)
    kwargs = dict(coin_ranges=[(0, 99, 0)], requested_start=0, warmup_bars=0, available=100)
    with pytest.raises(ValueError, match="unavailable hourly EMA"):
        validate_hourly_bootstrap_inputs(params, keys, (pins,), first_observed=np.array([3]), **kwargs)
    validate_hourly_bootstrap_inputs(params, keys, (pins,), first_observed=np.array([2]), **kwargs)
    pins[0, strategy_keys.index(coefficient)] = 0.0
    validate_hourly_bootstrap_inputs(params, keys, (pins,), first_observed=np.array([3]), **kwargs)


@pytest.mark.parametrize("flag,attribute", [
    ("ema-1m", "log_range_ema_1m"), ("ema-1h", "log_range_ema_1h"),
])
def test_trailing_inspector_exposes_one_log_range_input_spelling(flag, attribute):
    from tools.trailing_inspect import build_parser

    parser = build_parser()
    args = parser.parse_args(["--log-range-" + flag, "0.0123456789"])
    assert getattr(args, attribute) == 0.0123456789
    with pytest.raises(SystemExit):
        parser.parse_args(["--volatility-" + flag, "0.0123456789"])


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_price_span_bounds_keep_their_fractional_warmup_horizon(kind, side):
    from warmup_utils import compute_backtest_warmup_minutes
    from config.schema import get_expanded_config_prototype

    config = get_expanded_config_prototype()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    config["bot"]["portfolio"]["warmup"].update(activation_span_fraction=1.0, max_minutes=None)
    path = ("entry", "ema_span_0_minutes") if kind == "trailing_martingale" else ("ema_span_0_minutes",)
    bound = {}
    target = bound
    for part in path[:-1]:
        target = target.setdefault(part, {})
    target[path[-1]] = [600000.5, 600000.5]
    config["optimize"]["bounds"] = {side: {"strategy": {kind: bound}}}
    assert compute_backtest_warmup_minutes(config) == 600001


@pytest.mark.parametrize("leg,role,horizon", [
    (leg, role, horizon) for leg in ("entry", "close")
    for role in ("threshold", "retracement") for horizon in ("1m", "1h")
])
def test_log_range_coefficient_flags_reach_actual_parameters(leg, role, horizon):
    from tools.trailing_inspect import build_parser, load_parameter_source, apply_parameter_overrides

    parser = build_parser()
    args = parser.parse_args([f"--{leg}-{role}-log-range-{horizon}-coefficient", "0.625"])
    params, _ = load_parameter_source(None, "long")
    changed = apply_parameter_overrides(params, args)
    field = f"{role}_log_range_{horizon}_coefficient"
    assert params[leg][field] == 0.625
    assert len(changed) == 1 and field in changed[0]
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{leg}-{role}-volatility-{horizon}-weight", "0.625"])
