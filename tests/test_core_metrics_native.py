"""Public synthetic replay checks through the real native metrics boundary."""

from copy import deepcopy
import json

import numpy as np
import pytest

from backtest import build_backtest_payload, execute_backtest
from rust_utils import verify_loaded_runtime_extension

DAY = 86_400_000
CAGR = "cagr_fraction_strategy_equity"
ULCER = "ulcer_index_fraction_strategy_equity"
LOSS = "loss_profit_ratio"
SHORT = "short_net_profit_contribution_fraction"
FILL_GAP = "fills_gap_time_weighted_mean_hours"


@pytest.fixture(scope="module")
def replay():
    import passivbot_rust
    from tools.gpu_parity import build_parser, fixture_inputs

    assert not getattr(passivbot_rust, "__is_stub__", False)
    artifact = verify_loaded_runtime_extension()
    assert artifact["runtime_compiled_source_stamp"] == artifact["expected_source_fingerprint"]
    args = build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", "both", "--bars", "8640",
    ])
    config, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    result = execute_backtest(payload, config)
    return config, candles, markets, btc, timestamps, result


def test_native_daily_and_minute_metrics_match_exported_equity(replay):
    config, candles, markets, btc, timestamps, (fills, equity, analysis) = replay
    clocks, values = equity[:, 0].astype(np.int64), equity[:, 3]
    assert np.all(np.diff(clocks) == 60_000)
    assert clocks[0] == timestamps[61]
    assert clocks[-1] == timestamps[-1]
    metadata = analysis["metric_metadata"]
    assert metadata["start_timestamp_ms"] == clocks[0]
    assert metadata["actual_end_timestamp_ms"] == clocks[-1]
    closes = values[clocks % DAY == 0]
    returns = closes[1:] / closes[:-1] - 1
    assert metadata["daily_return_count"] == len(returns) >= 4
    assert metadata["equity_observation_count"] == len(values)
    mean, std = returns.mean(), returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    losses = np.maximum(-returns, 0).sum()
    assert mean > 0
    expected = {
        "volatility_annual_fraction_strategy_equity": std * np.sqrt(365.25),
        "sharpe_ratio_strategy_equity": mean / std * np.sqrt(365.25),
        "sortino_ratio_strategy_equity": mean / downside * np.sqrt(365.25) if downside else np.inf,
        "omega_ratio_strategy_equity": np.maximum(returns, 0).sum() / losses if losses else np.inf,
    }
    for metric, value in expected.items():
        if np.isposinf(value):
            assert analysis[metric] is None
            assert analysis["metric_status"][metric]["status"] == "positive_infinity"
        else:
            assert analysis[metric] == pytest.approx(value, rel=1e-9, abs=1e-10)
    drawdown = 1 - values / np.maximum.accumulate(values)
    assert analysis["drawdown_max_fraction_strategy_equity"] == pytest.approx(drawdown.max())
    assert analysis[ULCER] == pytest.approx(np.sqrt(np.mean(drawdown ** 2)))
    assert metadata["peak_recovery_tail_censored"] is bool(drawdown[-1] > 0)
    assert analysis["pain_index_fraction_strategy_equity"] == pytest.approx(drawdown.mean())
    years = (clocks[-1] - clocks[0]) / DAY / 365.25
    assert analysis[CAGR] == pytest.approx(np.expm1(np.log(values[-1] / values[0]) / years))
    assert analysis["backtest_completion_fraction"] == 1.0
    np.testing.assert_array_equal(equity[:, 1], equity[:, 3])
    json.dumps(analysis, allow_nan=False)


def test_native_profit_ratios_reconcile_with_fill_cashflows_and_open_positions(replay):
    config, candles, markets, btc, timestamps, (fills, equity, analysis) = replay
    net = np.array([float(fill[3]) + float(fill[4]) for fill in fills])
    assert analysis[LOSS] == pytest.approx(np.maximum(-net, 0).sum() / np.maximum(net, 0).sum())
    side_profit = dict(long=0.0, short=0.0)
    positions = {}
    for fill, cashflow in zip(fills, net):
        side = "long" if "long" in str(fill[13]) else "short"
        assert side in str(fill[13])
        side_profit[side] += cashflow
        positions[(str(fill[2]), side)] = (float(fill[11]), float(fill[12]))
    coins = config["backtest"]["coins"]["binance"]
    for (coin, side), (size, price) in positions.items():
        mark = candles[-2, coins.index(coin), 2]
        signed_size = abs(size) if side == "long" else -abs(size)
        side_profit[side] += (mark - price) * signed_size * markets[coin]["c_mult"]
    total = sum(side_profit.values())
    assert total == pytest.approx(equity[-1, 3] - equity[0, 3], abs=1e-8)
    assert total > 0
    assert analysis[SHORT] == pytest.approx(side_profit["short"] / total)


def test_native_fill_gap_weighted_mean_matches_exported_unique_fill_clocks(replay):
    *_, (fills, equity, analysis) = replay
    clocks = np.unique([int(fill[1]) for fill in fills])
    boundaries = np.concatenate(([equity[0, 0]], clocks, [equity[-1, 0]]))
    gaps = np.diff(boundaries) / 3_600_000
    assert np.all(gaps >= 0)
    assert analysis[FILL_GAP] == pytest.approx(np.sum(gaps ** 2) / np.sum(gaps), rel=1e-12)


@pytest.mark.parametrize("requests", [[CAGR, ULCER], [LOSS, SHORT], [FILL_GAP], []])
def test_request_masks_preserve_economics_and_selected_values(replay, requests):
    config, candles, markets, btc, timestamps, full = replay
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps,
                                     metric_requests=requests)
    fills, equity, analysis = execute_backtest(payload, config)
    np.testing.assert_array_equal(np.asarray(fills, dtype=object), np.asarray(full[0], dtype=object))
    np.testing.assert_array_equal(equity, full[1])
    for metric in requests:
        assert analysis[metric] == full[2][metric]
    metadata = analysis["metric_metadata"]
    assert "daily_return_count" not in metadata
    assert "ew_half_life_days" not in metadata
    assert "peak_recovery_tail_censored" not in metadata
    compact = build_backtest_payload(candles, markets, config, "binance", btc, timestamps,
                                     metrics_only=True, metric_requests=requests)
    compact_fills, compact_equity, compact_analysis = execute_backtest(compact, config)
    assert compact_fills is compact_equity is None
    assert compact_analysis == analysis


def test_native_half_life_rejects_boolean_instead_of_treating_it_as_one_day(replay):
    import passivbot_rust

    config, candles, markets, btc, timestamps, _ = replay
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps)
    params = deepcopy(payload.backtest_params)
    params["ew_half_life_days"] = True
    with pytest.raises((ValueError, TypeError), match="half.life|boolean|bool"):
        passivbot_rust.run_backtest_bundle(payload.bundle, payload.bot_params_list,
            payload.strategy_params_list, payload.exchange_params, params)


@pytest.mark.parametrize("half_life", [True, False, 0.0, -1.0, float("nan"), float("inf")])
def test_gpu_native_finalizer_rejects_invalid_half_life_without_device(half_life):
    import passivbot_rust

    verify_loaded_runtime_extension()
    fields = passivbot_rust.get_core_gpu_observation_catalog()["fields"]
    packet = np.empty((0, len(fields)), dtype=np.float32)
    with pytest.raises((ValueError, TypeError), match="ew_half_life_days"):
        passivbot_rust.core_metrics_from_gpu_observations(
            packet, [CAGR], 0, 60000, 60000, half_life)


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_gpu_native_policy_metadata_requires_actual_booleans(flag):
    import passivbot_rust

    verify_loaded_runtime_extension()
    fields = passivbot_rust.get_core_gpu_observation_catalog()["fields"]
    packet = np.empty((0, len(fields)), dtype=np.float32)
    with pytest.raises(TypeError):
        passivbot_rust.core_metrics_from_gpu_observations(
            packet, [CAGR], 0, 60000, 60000, None, [flag])
