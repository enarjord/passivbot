"""Finite auto-unstuck PnL history through real GPU and exact Rust backtests."""

import numpy as np
import pytest

from test_gpu_entry_sizing_parity import _fixture
from test_gpu_revised_hsl_multicoin import compare, raw

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def make_proxy(sides, lookback=8 / 1440):
    from optimization.gpu.service import MpsMulticoinProxy

    config, _, markets, _, timestamps = _fixture(sides[0], 2, "initial")
    config["live"]["hedge_mode"] = True
    count = 50
    config["live"]["pnls_max_lookback_days"] = lookback
    for side in sides:
        bot = config["bot"][side]
        bot["risk"].update(
            n_positions=2, total_wallet_exposure_limit=2.0, entry_cooldown_minutes=1000.0
        )
        bot["unstuck"].update(
            enabled=True, ema_gating_enabled=False, close_pct=0.1,
            loss_allowance_pct=0.001, threshold=0.3,
        )
        bot["strategy"]["trailing_martingale"]["entry"].update(
            initial_qty_pct=0.5, threshold_base_pct=10.0
        )
    candles = np.full((count, 2, 4), 100.4)
    candles[:, :, 3] = 1.0
    candles[3, :, 0] = 102.0
    candles[3, :, 1] = 99.0
    candles[4:, :, :3] = 95.0 if sides[0] == "long" else 105.0
    candles[4:, :, 0] += 1.0
    candles[4:, :, 1] -= 1.0
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.0)
    for coin in ("BTC", "ETH"):
        markets[coin].update(
            last_valid_index=count - 1, price_step=0.01,
            qty_step=0.1, min_qty=0.1, maker=0.0002,
        )
    inputs = candles, markets, config, "bybit", btc, timestamps
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange="bybit", batch_size=3, needed_metrics={"adg_strategy_eq"},
    )
    return proxy, inputs


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("lookback", [8 / 1440, "all"])
def test_unstuck_expiring_loss_budget_matches_exact_rust(sides, lookback):
    from backtest import run_backtest

    proxy, inputs = make_proxy(sides, lookback)
    runner, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert output["fill_count"].item() == len(fills)
    assert any("close_unstuck" in fill[13] for fill in fills)
    # Expired losses replenish the configured allowance; all-history exhausts it.
    assert (len(fills) > 10) == (lookback != "all")
    for side in sides:
        expected = sum(float(f[9]) for f in fills if f[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)
    assert runner.unstuck_pnl_lookback_bars == (0 if lookback == "all" else 8)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_unstuck_history_replay_reuse_and_bounded_batches(sides):
    proxy, _ = make_proxy(sides)
    candidates = [{}, {f"{sides[0]}_unstuck_loss_allowance_pct": 0.002},
                  {f"{sides[0]}_unstuck_loss_allowance_pct": 0.003}]
    runner, expected = raw(proxy, candidates)
    if len(sides) == 1:
        runner.max_dispatch_candidate_bars = 24
    ends = np.array([runner.n, runner.n, runner.n], dtype=np.int32)
    _, chunked = raw(proxy, candidates, end_steps=ends)
    compare(expected, chunked)
    runner.revised_scratch_budget_bytes = runner._history_bytes_per_candidate() * 2
    _, split = raw(proxy, candidates, profile=True)
    compare(expected, split)
    assert runner.last_profile["candidate_batch_count"] == 2
    _, repeated = raw(proxy, candidates)
    compare(expected, repeated)
    _, reordered = raw(proxy, candidates[::-1])
    compare({k: v.flip(0) if isinstance(v, torch.Tensor) else v
             for k, v in expected.items()}, reordered)
    runner.revised_scratch_budget_bytes = runner._history_bytes_per_candidate() - 1
    with pytest.raises(ValueError, match="history exceeds"):
        raw(proxy, [{}])


@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_unstuck_history_overflow_fails_closed(sides):
    proxy, _ = make_proxy(sides)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    # Deliberately violate the capacity guarantee to test the error path.
    runner.unstuck_pnl_capacity = 1
    if len(sides) == 1:
        runner.max_dispatch_candidate_bars = 4
    with pytest.raises(RuntimeError, match="auto-unstuck PnL history overflow"):
        raw(proxy, [{}])


def test_shared_unstuck_window_preserves_intrabar_peak_and_expires_without_fills():
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = (
        "#define PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS 9\n"
        "#define PASSIVBOT_UNSTUCK_PNL_CAPACITY 10\n"
        + passivbot_rust.mps_trailing_martingale_multicoin_source_py()
        + r"""
kernel void unstuck_window_probe(
    device float2* values, device int2* indices, device float* out,
    uint b [[thread_position_in_grid]]
) {
    JointPortfolioAccount account = init_joint_portfolio_account(1000.0f);
    bind_unstuck_pnl_window(account, values, indices, int(b));
    account.unstuck_pnl_k = 1;
    record_joint_portfolio_fill(account, 100.0f, true);
    // Long and short fills share a portfolio, including the intrabar peak.
    account.unstuck_pnl_k = 2;
    record_joint_portfolio_fill(account, 20.0f, false);
    record_joint_portfolio_fill(account, -100.0f, true);
    account.unstuck_pnl_k = 3;
    record_joint_portfolio_fill(account, -1.0f, false); // entry fee
    record_joint_portfolio_fill(account, 20.0f, true);
    refresh_unstuck_pnl_window(account);
    out[0] = unstuck_pnl_drawdown(account);
    account.unstuck_pnl_k = 11; // candle 2 remains on the inclusive boundary
    refresh_unstuck_pnl_window(account);
    out[1] = unstuck_pnl_drawdown(account);
    account.unstuck_pnl_k = 12; // no fills, but old loss/peak must expire
    refresh_unstuck_pnl_window(account);
    out[2] = unstuck_pnl_drawdown(account);
    out[3] = account.realized_pnl_peak - account.realized_pnl_total;
    out[4] = account.balance;
}
"""
    )
    values = torch.empty((10, 2), dtype=torch.float32, device=gpu_device())
    indices = torch.empty((10, 2), dtype=torch.int32, device=gpu_device())
    output = torch.empty(5, dtype=torch.float32, device=gpu_device())
    compile_shader(source).unstuck_window_probe(values, indices, output, threads=1)
    assert output.cpu().tolist() == [81.0, 81.0, 0.0, 81.0, 1039.0]


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
