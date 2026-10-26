"""Fixed directional score margins stay independent through real native selection."""
from copy import deepcopy

import numpy as np
import pytest

from config.forager_hysteresis import (
    forager_score_hysteresis, validate_forager_score_hysteresis,
)


INVALID = [None, True, 0.0, {}, {"long": 0.0}, {"long": True, "short": 0.0},
           {"long": "0.1", "short": 0.0}, {"long": -0.1, "short": 0.0},
           {"long": float("nan"), "short": 0.0},
           {"long": 0.0, "short": float("inf")},
           {"long": 0.0, "short": 0.0, "extra": 0.0}]


@pytest.mark.parametrize("invalid", INVALID)
def test_fixed_pair_rejects_incomplete_non_numeric_or_invalid_values(invalid):
    with pytest.raises((TypeError, ValueError), match="forager_score_hysteresis"):
        validate_forager_score_hysteresis(invalid)


def test_resolved_pair_reads_each_side_without_inheritance_or_coercion():
    config = {"bot": {side: {"forager": {"score_hysteresis_margin": value}}
                      for side, value in (("long", 0), ("short", 0.125))}}
    assert forager_score_hysteresis(config) == {"long": 0.0, "short": 0.125}
    del config["bot"]["short"]["forager"]["score_hysteresis_margin"]
    with pytest.raises(KeyError):
        forager_score_hysteresis(config)


def selection_snapshot(kind, *, high_side="short", hedge_mode=True):
    from test_side_activation_native import activation_snapshot

    value = activation_snapshot(kind, "long")
    value["global"]["side_activation"] = {"long": True, "short": True}
    value["global"]["hedge_mode"] = hedge_mode
    value["global"]["forager_score_hysteresis"] = {
        side: 1.0 if side == high_side else 0.001 for side in ("long", "short")}
    prototype = value["symbols"][0]
    value["symbols"] = []
    for idx, volume in enumerate((1000.0, 900.0, 100.0)):
        symbol = deepcopy(prototype)
        symbol["symbol_idx"] = idx
        symbol["emas"]["m1"]["volume"] = [[10.0, volume]]
        for side in ("long", "short"):
            symbol[side]["mode"] = None
            for target in (symbol[side]["bot_params"], value["global"]["global_bot_params"][side]):
                target["forager_score_weights"] = {
                    "volume": 1.0, "ema_readiness": 0.0,
                    "volatility": 0.0, "unilateralness": 0.0}
        value["symbols"].append(symbol)
    value["forager_hysteresis"] = {"incumbent_long": [1], "incumbent_short": [1]}
    return value


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("high_side", ["long", "short"])
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_native_unequal_margin_keeps_one_incumbent_and_rotates_other(kind, high_side, hedge_mode):
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension
    from test_orchestrator_json_api import compute

    verify_loaded_runtime_extension()
    value = selection_snapshot(kind, high_side=high_side, hedge_mode=hedge_mode)
    before = deepcopy(value)
    result = compute(native, value)
    diagnostics = {row["pside"]: row for row in result["diagnostics"]["forager_selections"]}
    for side in ("long", "short"):
        row = diagnostics[side]
        assert row["selected_symbol_indices"] == ([1] if side == high_side else [0])
        assert row["score_hysteresis_pct"] == value["global"]["forager_score_hysteresis"][side]
        event, = row["hysteresis_events"]
        assert 0.001 < event["score_gap"] < 1.0
        assert event["kept_incumbent"] is (side == high_side)
    assert value == before


@pytest.mark.parametrize("invalid", INVALID)
def test_native_json_pair_is_required_complete_and_strict(invalid):
    import passivbot_rust as native
    from test_orchestrator_json_api import compute

    value = selection_snapshot("ema_anchor")
    value["global"]["forager_score_hysteresis"] = invalid
    with pytest.raises(ValueError):
        compute(native, value)


def test_native_json_pair_cannot_be_omitted():
    import passivbot_rust as native
    from test_orchestrator_json_api import compute

    value = selection_snapshot("ema_anchor")
    del value["global"]["forager_score_hysteresis"]
    with pytest.raises(ValueError, match="forager_score_hysteresis"):
        compute(native, value)

def replay_inputs(kind, topology, margins):
    from config import prepare_config
    from config.schema import get_expanded_config_prototype
    from config.universe import set_side_universe_sources

    config = get_expanded_config_prototype()
    config["optimize"]["bounds"] = {}
    config["optimize"]["fixed_runtime_overrides"] = {}
    config["optimize"]["enable_overrides"] = []
    config["bot"]["portfolio"]["strategy_kind"] = kind
    config["bot"]["portfolio"]["hedge_mode"] = topology != "oneway"
    config["bot"]["portfolio"]["warmup"].update(max_minutes=1.0, span_multiplier=1.0)
    config["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.0
    config["bot"]["portfolio"]["execution"]["market_orders_allowed"] = False
    coins = ["AAA", "BBB", "CCC"]
    enabled = (topology,) if topology in ("long", "short") else ("long", "short")
    set_side_universe_sources(config, "approved_coins", {
        side: coins if side in enabled else [] for side in ("long", "short")})
    set_side_universe_sources(config, "ignored_coins", {"long": [], "short": []})
    for side in ("long", "short"):
        bot = config["bot"][side]
        bot["enabled"] = side in enabled
        bot["universe"]["minimum_coin_age_days"] = 0.0
        bot["risk"].update(max_positions=1, total_wallet_exposure_limit_ratio=1.0,
                           we_excess_allowance_fraction=0.0)
        bot["risk"]["position_exposure_enforcer"]["enabled"] = False
        bot["risk"]["total_exposure_enforcer"]["enabled"] = False
        bot["hsl"]["enabled"] = False
        bot["unstuck"]["enabled"] = False
        bot["entry_cooldown"]["base_duration_minutes"] = 1000.0
        bot["forager"].update(volume_ema_span_minutes=1.0,
                              score_hysteresis_margin=margins[side])
        bot["forager"]["score_weights"] = {
            "volume": 1.0, "ema_readiness": 0.0, "log_range": 0.0, "unilateralness": 0.0}
        strategy = bot["strategy"][kind]
        band = strategy["entry"] if kind == "trailing_martingale" else strategy
        band.update(ema_span_0_minutes=2.0, ema_span_1_minutes=3.0)
        if kind == "ema_anchor":
            strategy.update(base_qty_fraction=0.01, offset_fraction=0.01,
                            inventory_offset_exposure_coefficient=0.0, offset_log_range_1m_coefficient=0.0,
                            offset_log_range_1h_coefficient=0.0, entry_qty_exposure_coefficient=0.0)
        else:
            strategy["entry"].update(initial_qty_fraction=0.01, initial_ema_distance_fraction=0.01,
                                      reentry_qty_multiplier=0.0, threshold_base_fraction=0.01,
                                      retracement_base_fraction=0.0)
            for leg in ("entry", "close"):
                for name in ("threshold_log_range_1h_coefficient", "retracement_log_range_1h_coefficient",
                             "threshold_log_range_1m_coefficient", "retracement_log_range_1m_coefficient"):
                    strategy[leg][name] = 0.0
    config["backtest"].update(exchanges=["binance"], starting_balance=1000.0,
                               btc_collateral_cap=0.0, filter_by_min_effective_cost=False,
                               dynamic_wel_by_tradability=False)
    config = prepare_config(config, verbose=False, target="canonical", runtime=None)
    config["backtest"]["coins"] = {"binance": coins}
    n = 64
    candles = np.empty((n, 3, 4), dtype=np.float64)
    candles[:] = [100.1, 99.9, 100.0, 100.0]
    candles[:20, :, 3] = [1000.0, 900.0, 100.0]
    candles[20:, :, 3] = [900.0, 1000.0, 100.0]
    candles[40:, 1, 0] = 102.0
    candles[40:, 1, 1] = 98.0
    timestamps = np.arange(n, dtype=np.int64) * 60_000 + 1_704_067_200_000
    markets = {coin: {"maker": 0.0, "taker": 0.0, "qty_step": 0.001,
                      "price_step": 0.01, "min_qty": 0.001, "min_cost": 1.0,
                      "c_mult": 1.0, "first_valid_index": 0, "last_valid_index": n - 1,
                      "warmup_minutes": 1.0, "inception_timestamp_ms": None}
               for coin in coins}
    return config, candles, markets, np.full(n, 50000.0), timestamps


@pytest.mark.parametrize("kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("high_side", ["long", "short"])
@pytest.mark.parametrize("topology", ["fused", "oneway"])
def test_actual_native_replay_applies_each_margin_before_later_fill(kind, high_side, topology):
    from backtest import build_backtest_payload, execute_backtest

    margins = {side: 0.5 if side == high_side else 0.001 for side in ("long", "short")}
    config, candles, markets, btc, timestamps = replay_inputs(kind, topology, margins)
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    assert payload.backtest_params["forager_score_hysteresis"] == margins
    fills, _, result = execute_backtest(payload, config)
    low_side = "short" if high_side == "long" else "long"
    entries = [row for row in fills if str(row[13]).startswith("entry_")]
    assert result["fills_count_" + high_side] == 0
    if topology == "oneway" and high_side == "long":
        # Existing one-way tie arbitration keeps the long pending order. A low
        # short margin does not override that independent entry permission.
        assert not entries and result["fills_count_short"] == 0
    else:
        assert entries and all(str(row[13]).endswith("_" + low_side) for row in entries)
        assert result["fills_count_" + low_side] > 0
        assert int(fills[0][1]) == int(timestamps[40])


@pytest.mark.parametrize("invalid", INVALID)
def test_backtest_native_pair_rejects_invalid_policy(invalid):
    from backtest import build_backtest_payload, execute_backtest

    config, candles, markets, btc, timestamps = replay_inputs(
        "ema_anchor", "long", {"long": 0.02, "short": 0.03})
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    payload.backtest_params["forager_score_hysteresis"] = invalid
    with pytest.raises((TypeError, ValueError)):
        execute_backtest(payload, config)


def test_cached_native_policy_requires_current_strict_pair():
    from backtest import build_backtest_payload, prep_backtest_args

    config, candles, markets, btc, timestamps = replay_inputs(
        "ema_anchor", "long", {"long": 0.02, "short": 0.03})
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    cached = deepcopy(payload.backtest_params)
    cached["forager_score_hysteresis"]["short"] = 0.04
    with pytest.raises(ValueError, match="cached Forager hysteresis"):
        prep_backtest_args(config, markets, "binance", backtest_params=cached)
    cached["forager_score_hysteresis"]["short"] = True
    with pytest.raises(TypeError, match="forager_score_hysteresis.short"):
        prep_backtest_args(config, markets, "binance", backtest_params=cached)
