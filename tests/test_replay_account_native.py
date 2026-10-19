"""Bounded full native account extraction, using the source-verified CPU runner."""

import math

import numpy as np
import pytest

from tools.optimizer_benchmark.replay_account import (
    account_valuation_input_identity,
    cpu_confirmation_from_native,
)


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
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
    assert metrics["sample_count"] == len(result[1]) + 1
    assert metrics["final_equity"] == result[1][-1, 1]
    assert metrics["net_return"] == pytest.approx(result[1][-1, 1] / params["starting_balance"] - 1)
    assert metrics["net_realized_pnl"] == pytest.approx(math.fsum(float(fill[3]) + float(fill[4]) for fill in result[0]))
    assert metrics["fill_count"] == len(result[0]) > 0
    assert metrics["gross_exposure_ratio_max"] > 0
    assert metrics["exposure_denominator_value"] == params["starting_balance"]
    assert "adg_strategy_eq" not in metrics
