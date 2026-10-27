"""Bounded full native account extraction, using the source-verified CPU runner."""

import math

import numpy as np
import pytest

from tools.optimizer_benchmark.replay_account import (
    account_valuation_input_identity,
    cpu_confirmation_from_native,
)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("through_execute", [False, True])
def test_full_native_account_outputs_feed_version_neutral_confirmation(mode, through_execute):
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    # Verify before importing backtest, so an in-progress source edit produces
    # a clean failure instead of triggering its automatic compilation path.
    runtime = verify_loaded_runtime_extension()
    from test_hsl_backtest_config import inputs, payload, prepared_payload, run

    config, markets, _ = inputs(mode)
    args = payload(mode, cfg=config, mss=markets)
    assert args[-1]["equity_hard_stop_loss"]["mode"] == (
        "pside" if mode == "position_side" else mode
    )
    if through_execute:
        from backtest import execute_backtest

        result = execute_backtest(prepared_payload(args), config)
    else:
        result = run(args)
    assert type(result[2]["liquidated"]) is bool
    params = args[-1]
    # This fixture activates equity tracking at k>warmup_bars=1. The native
    # replay reserves the final input candle and evaluates through T-2.
    prepared = dict(hlcvs=args[0], candle_timestamps_ms=params["first_timestamp_ms"] +
                    np.arange(len(args[0]), dtype=np.int64) * 60000,
                    coins=params["coins"], market_settings=markets,
                    initial_capital=params["starting_balance"], first_bar_index=2,
                    last_bar_index=len(args[0]) - 2, candle_interval_ms=60000,
                    bootstrap_bar_indices=[])
    row = cpu_confirmation_from_native(
        result, model="v9", effective_config=config, input_sha256="a" * 64,
        runtime_sha256=runtime["runtime_compiled_sha256"], elapsed_seconds=lambda: 1.0,
        execution_completed=True, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**prepared), **prepared)
    metrics = row["metrics"]
    assert metrics["completed"] and not metrics["liquidated"]
    assert metrics["observed_end_ms"] == int(prepared["candle_timestamps_ms"][-2]) + 60000
    metadata = result[2]["metric_metadata"]
    assert metadata["definition_version"] == metrics["native_equity_definition_version"] == 1
    assert metrics["sample_count"] == len(result[1]) == metadata["equity_observation_count"]
    assert result[1][0, 0] == metadata["start_timestamp_ms"] == prepared["candle_timestamps_ms"][2]
    assert result[1][0, 1] == params["starting_balance"]
    assert metrics["final_equity"] == result[1][-1, 1]
    assert metrics["net_return_fraction"] == pytest.approx(result[1][-1, 1] / params["starting_balance"] - 1)
    assert metrics["net_realized_pnl"] == pytest.approx(math.fsum(float(fill[3]) + float(fill[4]) for fill in result[0]))
    assert metrics["fill_count"] == len(result[0]) > 0
    assert metrics["gross_exposure_max_ratio"] > 0
    assert metrics["exposure_denominator_value"] == params["starting_balance"]
    assert "adg_fraction_strategy_equity" not in metrics


    # Independent ordinary-account reconstruction uses native executed fills and
    # prepared causal close prices, without any strategy-equity/HSL signal curve.
    positions, balances, exposures, accounts = {}, params["starting_balance"], [0.0], [params["starting_balance"]]
    cursor = 0
    coin_index = {coin: index for index, coin in enumerate(params["coins"])}
    for timestamp, *_ in result[1][1:]:
        bar = int((timestamp - params["first_timestamp_ms"]) // 60000) - 1
        while cursor < len(result[0]) and int(result[0][cursor][0]) <= bar:
            fill = result[0][cursor]
            key = (fill[2], str(fill[13]).rsplit("_", 1)[-1])
            positions[key] = (float(fill[11]), float(fill[12]))
            balances += float(fill[3]) + float(fill[4])
            cursor += 1
        upnl, notional = [], []
        for (coin, _), (size, entry_price) in positions.items():
            mark = float(args[0][bar, coin_index[coin], 2])
            multiplier = markets[coin]["c_mult"]
            upnl.append(size * (mark - entry_price) * multiplier)
            notional.append(abs(size) * mark * multiplier)
        accounts.append(balances + math.fsum(upnl))
        exposures.append(math.fsum(notional) / params["starting_balance"])
    assert cursor == len(result[0])
    np.testing.assert_allclose(result[1][:, 1], accounts, rtol=1e-12, atol=1e-10)
    assert metrics["gross_exposure_max_ratio"] == pytest.approx(max(exposures))
    assert metrics["gross_exposure_terminal_ratio"] == pytest.approx(exposures[-1])
    assert metrics["gross_exposure_mean_ratio"] == pytest.approx(math.fsum(exposures[:-1]) / (len(exposures) - 1))
    assert metrics["exposed_time_fraction"] == pytest.approx(sum(value > 0 for value in exposures[:-1]) / (len(exposures) - 1))
