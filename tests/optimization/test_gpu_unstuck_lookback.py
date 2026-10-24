"""Continuous realized-loss memory through actual GPU and exact Rust replays."""

import numpy as np
import pytest

from test_gpu_entry_sizing_parity import _fixture
from test_gpu_hsl_multicoin import compare, raw

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def make_proxy(sides, memory_span=8.0, strategy="trailing_martingale", coins=2,
               hsl_mode="coin", interval_minutes=1):
    from optimization.gpu.service import SharedAccountReplay

    config, _, markets, _, timestamps = _fixture(sides[0], coins, "initial")
    config['bot']["portfolio"]['strategy_kind'] = strategy
    config["backtest"]["candle_interval_minutes"] = interval_minutes
    config['bot']["portfolio"]['hedge_mode'] = True
    if hsl_mode != "coin":
        from config.hsl import generated_template
        config = generated_template(config, hsl_mode)
        config["live"]["hsl_signal_mode"] = hsl_mode
        if hsl_mode == "portfolio":
            config["bot"]["hsl"]["enabled"] = False
    count = 50
    config["live"]["lookback_ema_span_minutes"] = memory_span
    for side in sides:
        bot = config["bot"][side]
        bot["risk"].update(
            n_positions=coins,
            total_wallet_exposure_limit=float(coins),
        )
        # The fixture template already carries the canonical cooldown leaf;
        # a legacy risk leaf cannot override it. Prevent reentry after closes.
        bot["entry_cooldown"]["base_duration_minutes"] = 1000.0
        bot["unstuck"].update(
            enabled=True,
            ema_gating_enabled=False,
            close_pct=0.1,
            loss_allowance_pct=0.001,
            threshold=0.3,
        )
        if strategy == "trailing_martingale":
            bot["strategy"][strategy]["entry"].update(
                initial_qty_pct=0.5, threshold_base_pct=10.0
            )
        else:
            bot["strategy"][strategy].update(
                base_qty_pct=0.5, ema_span_0=1000.0, ema_span_1=1000.0,
                offset=0.0, offset_psize_weight=0.0, entry_double_down_factor=0.0,
            )
    candles = np.full((count, coins, 4), 100.4)
    candles[:, :, 3] = 1.0
    candles[3, :, 0] = 102.0
    candles[3, :, 1] = 99.0
    candles[4:, :, :3] = 95.0 if sides[0] == "long" else 105.0
    candles[4:, :, 0] += 1.0
    candles[4:, :, 1] -= 1.0
    interval_ms = interval_minutes * 60_000
    first_ts = int(timestamps[0]) // interval_ms * interval_ms
    timestamps = first_ts + np.arange(count, dtype=np.int64) * interval_ms
    btc = np.full(count, 50_000.0)
    for coin in ("BTC", "ETH")[:coins]:
        markets[coin].update(
            last_valid_index=count * interval_minutes - 1,
            price_step=0.01,
            qty_step=0.1,
            min_qty=0.1,
            maker=0.0002,
        )
    markets["__meta__"].update(data_interval_minutes=interval_minutes, requested_start_ts=first_ts)
    inputs = candles, markets, config, "bybit", btc, timestamps
    proxy = SharedAccountReplay(
        config=config,
        hlcvs=candles,
        mss=markets,
        btc=btc,
        timestamps=timestamps,
        exchange="bybit",
        batch_size=3,
        needed_metrics={"adg_strategy_eq"},
    )
    return proxy, inputs


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_ema_unstuck_reuses_forward_memory_without_cpu_execution(monkeypatch, sides, coins):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA service required")
    import backtest
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    proxy, inputs = make_proxy(sides, strategy="ema_anchor", coins=coins)
    candidates = [{}, {f"{sides[0]}_unstuck_loss_allowance_pct": 0.002}, {}]
    expected = proxy.evaluate_results(candidates)

    def forbidden(*args, **kwargs):
        pytest.fail("native GPU replay must not execute a CPU backtest")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    candles, markets, config, exchange, btc, timestamps = inputs
    with _native_dataset((config, candles, markets, btc, timestamps), exchange,
                         {"adg_strategy_eq"}) as dataset:
        with CudaBacktestService(batch_size=2, tuning_mode="off") as service:
            service.register_dataset("finite-unstuck", dataset)
            pending = [service.submit(BacktestRequest(str(i), "finite-unstuck", values))
                       for i, values in enumerate(candidates)]
            actual = [future.result() for future in pending]
            # A subsequent request must start with a fresh continuous account state.
            repeated = service.submit(BacktestRequest("repeat", "finite-unstuck", {})).result()
    assert [result.metrics for result in actual] == [result.metrics for result in expected]
    assert repeated.metrics == expected[0].metrics


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_unstuck_forward_replay_reuse_and_bounded_batches(sides, strategy):
    proxy, _ = make_proxy(sides, strategy=strategy)
    candidates = [
        {},
        {f"{sides[0]}_unstuck_loss_allowance_pct": 0.002},
        {f"{sides[0]}_unstuck_loss_allowance_pct": 0.003},
    ]
    runner, expected = raw(proxy, candidates)
    if len(sides) == 1 and strategy == "trailing_martingale":
        runner.max_dispatch_candidate_bars = 24
    ends = np.array([runner.n, runner.n, runner.n], dtype=np.int32)
    _, chunked = raw(proxy, candidates, end_steps=ends)
    compare(expected, chunked)
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate() * 2
    _, split = raw(proxy, candidates, profile=True)
    compare(expected, split)
    assert runner.last_profile["candidate_batch_count"] == 2
    _, repeated = raw(proxy, candidates)
    compare(expected, repeated)
    _, reordered = raw(proxy, candidates[::-1])
    compare(
        {
            k: v.flip(0) if isinstance(v, torch.Tensor) else v
            for k, v in expected.items()
        },
        reordered,
    )
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate() - 1
    with pytest.raises(ValueError, match="history exceeds"):
        raw(proxy, [{}])


@pytest.mark.parametrize("side", ["long", "short"])
def test_interrupted_unstuck_replay_starts_next_run_fresh(side):
    proxy, _ = make_proxy((side,))
    runner, expected = raw(proxy, [{}])
    runner.max_dispatch_candidate_bars = 8
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise InterruptedError("test interruption after fills")

    runner.interrupt_check = interrupt
    with pytest.raises(InterruptedError, match="after fills"):
        raw(proxy, [{}])
    runner.interrupt_check = lambda: None
    _, restarted = raw(proxy, [{}])
    compare(expected, restarted)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_continuous_unstuck_memory_uses_elapsed_integer_minutes(strategy, side, interval_minutes):
    from backtest import run_backtest
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    proxy, inputs = make_proxy((side,), memory_span=3.0, strategy=strategy,
                              coins=2, interval_minutes=interval_minutes)
    runner, actual = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert len(fills) > 2
    assert any("close_unstuck" in fill[13] for fill in fills)
    assert actual["fill_count"].item() == len(fills)
    assert actual["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=.002)
    expected_size = abs(sum(float(fill[9]) for fill in fills if fill[13].endswith(side)))
    assert actual["short_psize" if side == "short" else "psize"].item() == pytest.approx(expected_size, abs=2e-5)
    assert not hasattr(runner, "_unstuck_pnl_buffers")
    from test_gpu_v9_native import _save_evidence
    candles, markets, config, _, btc, timestamps = inputs
    _save_evidence(f"realized-clock-{strategy}-{side}-{interval_minutes}m", {
        "fill_count": len(fills), "gpu_fill_count": actual["fill_count"].item(),
        "balance": float(fills[-1, 5]), "gpu_balance": actual["balance"].item(),
        "interval_minutes": interval_minutes, "slow_span_minutes": 3.0,
    }, (config, candles, markets, btc, timestamps))
