"""Executed protective orders retain their cause through execution-mark recovery."""

from copy import deepcopy

import numpy as np
import pytest

from config.hsl import generated_template
from config.schema import get_expanded_config_prototype
from config.universe import set_side_universe_sources


HSL_METRICS = (
    "hsl_execution_episode_count",
    "hsl_panic_episode_loss_max_fraction",
    "hsl_panic_episode_loss_mean_fraction",
    "hsl_triggers_per_year",
    "hsl_red_time_fraction",
)


def recovered_panic_inputs(side, mode, strategy_kind, outcome="fee_red"):
    """Two hand-computable fills; no market data or private configuration."""
    config = generated_template(get_expanded_config_prototype(), mode)
    config["bot"]["portfolio"]["strategy_kind"] = strategy_kind
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = 1
    config["bot"]["portfolio"]["execution"]["market_orders_allowed"] = False
    set_side_universe_sources(config, "approved_coins", {"long": ["BTC"], "short": ["BTC"]})
    config["backtest"].update(
        coins={"bybit": ["BTC"]}, exchanges=["bybit"], starting_balance=1024.0,
        hsl_detailed_report=True, market_order_slippage_fraction=0.0,
        maker_fee_override_fraction=0.0,
        taker_fee_override_fraction=0.001 if outcome == "fee_red" else 0.0,
    )
    for direction in ("long", "short"):
        bot = config["bot"][direction]
        bot["enabled"] = direction == side
        bot["universe"].update(minimum_coin_age_days=0.0, filter_by_min_effective_cost=False)
        bot["entry_cooldown"]["base_duration_minutes"] = 10000.0
        bot["unstuck"]["enabled"] = False
        bot["risk"].update(
            max_positions=1 if direction == side else 0,
            total_wallet_exposure_limit_ratio=1.0 if direction == side else 0.0,
            we_excess_allowance_fraction=0.0,
        )
        bot["risk"]["position_exposure_enforcer"]["enabled"] = False
        bot["risk"]["total_exposure_enforcer"]["enabled"] = False
        bot["hsl"].update(
            enabled=direction == side and mode != "portfolio",
            drawdown_threshold_fraction=0.049, memory_ema_span_minutes=1440.0,
            drawdown_fraction_ema_span_minutes=1.0, cooldown_after_red_minutes=5.0,
            panic_close_order_type="market", scale_budget_with_excess_allowance=False,
        )
        bot["strategy"]["ema_anchor"].update(
            base_qty_fraction=0.5, ema_span_0_minutes=2.0, ema_span_1_minutes=3.0,
            offset_fraction=0.01, offset_log_range_1h_coefficient=0.0,
            inventory_offset_exposure_coefficient=0.0, entry_qty_exposure_coefficient=0.0,
        )
        tm = bot["strategy"]["trailing_martingale"]
        tm["entry"].update(
            initial_qty_fraction=0.5, ema_span_0_minutes=2.0, ema_span_1_minutes=3.0,
            ema_gate_mode="disabled", threshold_base_fraction=0.5,
            retracement_base_fraction=0.0,
        )
        tm["close"].update(threshold_base_fraction=0.5, retracement_base_fraction=0.0)
        for branch in ("entry", "close"):
            tm[branch].update(
                threshold_log_range_1h_coefficient=0.0,
                retracement_log_range_1h_coefficient=0.0,
            )
    if mode == "portfolio":
        config["bot"]["portfolio"]["hsl"].update(
            deepcopy(config["bot"][side]["hsl"]), enabled=True,
        )

    count = 16
    candles = np.full((count, 1, 4), 100.0, dtype=np.float64)
    candles[:, :, 3] = 1.0
    candles[3, :, 0], candles[3, :, 1] = 101.0, 99.0
    basis = 100.0 if strategy_kind == "trailing_martingale" else 99.0 if side == "long" else 101.0
    sign = 1.0 if side == "long" else -1.0
    adverse = basis - sign * 20.0
    recovery = {"fee_red": -10.0, "zero": 0.0, "profit": 10.0}[outcome]
    candles[5, :, :3] = adverse
    candles[5, :, 0], candles[5, :, 1] = max(100.0, adverse), min(100.0, adverse)
    candles[6:, :, :3] = basis + sign * recovery
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    markets = {
        "BTC": dict(
            qty_step=1.0, price_step=1.0, min_qty=1.0, min_cost=0.0,
            c_mult=1.0, maker=0.0, taker=0.0, exchange="bybit",
            first_valid_index=0, last_valid_index=count - 1, warmup_minutes=1,
        ),
        "__meta__": {"requested_start_ts": int(timestamps[0])},
    }
    return config, candles, markets, np.full(count, 50_000.0), timestamps


def _run(inputs, *, metrics_only=False):
    from backtest import build_backtest_payload, execute_backtest

    config, candles, markets, btc, timestamps = inputs
    payload = build_backtest_payload(
        candles, markets, deepcopy(config), "bybit", btc, timestamps,
        metrics_only=metrics_only, metric_requests=HSL_METRICS,
    )
    return execute_backtest(payload, config), payload


def recovered_batch_inputs(side, mode, strategy_kind):
    """One controller submits two closes before either execution mark recovers."""
    config, candles, markets, btc, timestamps = recovered_panic_inputs(
        side, mode, strategy_kind,
    )
    coins = ["BTC", "ETH"]
    set_side_universe_sources(config, "approved_coins", {"long": coins, "short": coins})
    config["backtest"]["coins"] = {"bybit": coins}
    config["bot"][side]["risk"].update(
        max_positions=2, total_wallet_exposure_limit_ratio=2.0,
    )
    policy = config["bot"]["portfolio"]["hsl"] if mode == "portfolio" else config["bot"][side]["hsl"]
    # Total loss is 200 at the trigger. The first recovered execution mark
    # reduces it to 150, its fee crosses RED again, and the second recovery
    # turns GREEN. Both submitted orders still have the original attribution.
    policy["drawdown_threshold_fraction"] = 0.1467
    markets["ETH"] = deepcopy(markets["BTC"])
    return config, np.repeat(candles, 2, axis=1), markets, btc, timestamps


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
def test_recovered_same_origin_batch_remains_one_execution_episode(side, mode, strategy_kind):
    inputs = recovered_batch_inputs(side, mode, strategy_kind)
    (fills, equities, analysis), payload = _run(inputs)
    closes = [fill for fill in fills if str(fill[13]).startswith("close_")]
    assert len(fills) == 4 and len(closes) == 2
    assert [fill[2] for fill in closes] == ["BTC", "ETH"]
    assert all(int(fill[0]) == 6 and float(fill[3]) == -50.0 for fill in closes)
    fee = -float(closes[0][4])
    assert float(closes[1][4]) == -fee
    loss = 2.0 * (50.0 + fee)
    final_balance = (1024.0 - (50.0 + fee)) - (50.0 + fee)
    np.testing.assert_array_equal(equities[:, 1], [1024.0] * 4 + [824.0] + [final_balance] * 9)
    report = payload.hard_stop_plot_data["hsl"]
    assert report["summary"]["triggers"] == 2
    assert report["summary"]["red_minutes"] == 0.0
    assert report["summary"]["panic_close_fills"] == 2
    assert analysis["hsl_execution_episode_count"] == 1
    assert analysis["hsl_panic_episode_loss_max_fraction"] == loss / 1024.0
    assert analysis["hsl_panic_episode_loss_mean_fraction"] == loss / 1024.0
    episode, = report["execution_episodes"]
    assert episode["fill_count"] == 2
    assert episode["account_balance_usd"] == 1024.0
    assert episode["loss_usd"] == loss
    assert episode["finished_timestamp_ms"] == int(inputs[-1][6])
    (_, _, compact), _ = _run(inputs, metrics_only=True)
    assert compact == analysis


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("outcome", ["fee_red", "zero", "profit"])
def test_recovered_hsl_order_counts_execution_without_changing_economics(
    side, mode, strategy_kind, outcome,
):
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    inputs = recovered_panic_inputs(side, mode, strategy_kind, outcome)
    (fills, equities, analysis), payload = _run(inputs)
    timestamps = inputs[-1]
    basis = 100.0 if strategy_kind == "trailing_martingale" else 99.0 if side == "long" else 101.0
    sign = 1.0 if side == "long" else -1.0
    recovery = {"fee_red": -10.0, "zero": 0.0, "profit": 10.0}[outcome]
    fill_price = basis + sign * recovery
    fee = 5.0 * fill_price * (0.001 if outcome == "fee_red" else 0.0)
    pnl = 5.0 * recovery
    final_balance = 1024.0 + pnl - fee
    entry_k = 3 if strategy_kind == "trailing_martingale" else 5
    entry_type = "entry_initial_normal" if strategy_kind == "trailing_martingale" else "entry_ema_anchor"
    exposure = sign * 5.0 * basis / 1024.0
    expected_fills = np.array([
        [entry_k, int(timestamps[entry_k]), "BTC", 0.0, -0.0, 1024.0, 0.0, 1024.0,
         50000.0, sign * 5.0, basis, sign * 5.0, basis, f"{entry_type}_{side}", "maker",
         exposure, exposure if side == "long" else 0.0,
         exposure if side == "short" else 0.0, exposure],
        [6, int(timestamps[6]), "BTC", pnl, -fee, final_balance, 0.0, final_balance,
         50000.0, -sign * 5.0, fill_price, 0.0, basis, f"close_panic_{side}", "taker",
         0.0, 0.0, 0.0, 0.0],
    ], dtype=object)
    np.testing.assert_array_equal(fills, expected_fills)
    expected_equity = np.array([1024.0] * 4 + [924.0] + [final_balance] * 9)
    np.testing.assert_array_equal(equities, np.column_stack([
        timestamps[2:], expected_equity, expected_equity / 50000.0, expected_equity,
    ]))

    report = payload.hard_stop_plot_data["hsl"]
    assessments = [s for s in report["samples"] if s["timestamp"] == int(timestamps[6])]
    assert [(s["phase"], s["action"]) for s in assessments] == [
        ("bar_close", "panic"), ("pre_fill", "normal"),
        ("post_fill", "halted" if outcome == "fee_red" else "normal"),
    ]
    assert report["summary"]["triggers"] == (2 if outcome == "fee_red" else 1)
    assert report["summary"]["red_minutes"] == (5.0 if outcome == "fee_red" else 0.0)
    assert report["summary"]["panic_close_fills"] == 1
    assert analysis["hsl_execution_episode_count"] == 1
    assert analysis["metric_metadata"]["hsl_execution_episode_count"] == 1
    expected_loss = max(-(pnl - fee), 0.0)
    for name in ("hsl_panic_episode_loss_max_fraction", "hsl_panic_episode_loss_mean_fraction"):
        assert analysis[name] == expected_loss / 1024.0
    episode, = report["execution_episodes"]
    assert episode["fill_count"] == 1
    assert episode["account_balance_usd"] == 1024.0
    assert episode["loss_usd"] == expected_loss
    assert episode["loss_fraction"] == expected_loss / 1024.0
    assert episode["trigger_timestamp_ms"] == int(timestamps[6])
    assert episode["first_fill_timestamp_ms"] == int(timestamps[6])
    assert episode["finished_timestamp_ms"] == int(timestamps[6])

    (compact_fills, compact_equities, compact_analysis), _ = _run(inputs, metrics_only=True)
    assert compact_fills is None and compact_equities is None
    assert compact_analysis == analysis
