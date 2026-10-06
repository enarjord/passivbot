"""Finite auto-unstuck PnL history through real GPU and exact Rust backtests."""

import numpy as np
import pytest

from test_gpu_entry_sizing_parity import _fixture
from test_gpu_hsl_multicoin import compare, raw

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def make_proxy(sides, lookback=8 / 1440, strategy="trailing_martingale", coins=2,
               hsl_mode="coin"):
    from optimization.gpu.service import MpsMulticoinProxy

    config, _, markets, _, timestamps = _fixture(sides[0], coins, "initial")
    config["live"]["strategy_kind"] = strategy
    config["live"]["hedge_mode"] = True
    if hsl_mode != "coin":
        from config.hsl import generated_template
        config = generated_template(config, hsl_mode)
        config["live"]["hsl_signal_mode"] = hsl_mode
        if hsl_mode == "unified":
            config["bot"]["hsl"]["enabled"] = False
    count = 50
    config["live"]["pnls_max_lookback_days"] = lookback
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
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    btc = np.full(count, 50_000.0)
    for coin in ("BTC", "ETH")[:coins]:
        markets[coin].update(
            last_valid_index=count - 1,
            price_step=0.01,
            qty_step=0.1,
            min_qty=0.1,
            maker=0.0002,
        )
    inputs = candles, markets, config, "bybit", btc, timestamps
    proxy = MpsMulticoinProxy(
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
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize(
    "policy", ["disabled", "enabled", "coin_enabled", "coins_disabled"]
)
def test_unstuck_history_requires_an_effective_consumer(sides, policy, strategy):
    from optimization.gpu.service import MpsMulticoinProxy

    config, candles, markets, btc, timestamps = _fixture(sides[0], 2, "initial")
    config["live"]["strategy_kind"] = strategy
    config["live"]["pnls_max_lookback_days"] = 30.0
    for side in sides:
        config["bot"][side]["risk"].update(
            n_positions=2, total_wallet_exposure_limit=2.0
        )
        config["bot"][side]["unstuck"]["enabled"] = policy in {
            "enabled",
            "coins_disabled",
        }
        # These are optimizer genes: a zero base value must not elide history.
        config["bot"][side]["unstuck"]["loss_allowance_pct"] = 0.0
    if len(sides) == 1:
        inactive_side = "short" if sides[0] == "long" else "long"
        config["bot"][inactive_side]["unstuck"]["enabled"] = True
    if policy == "coin_enabled":
        config["coin_overrides"] = {
            "ETH": {"bot": {sides[-1]: {"unstuck": {"enabled": True}}}}
        }
    elif policy == "coins_disabled":
        config["coin_overrides"] = {
            coin: {"bot": {side: {"unstuck": {"enabled": False}} for side in sides}}
            for coin in ("BTC", "ETH")
        }
    proxy = MpsMulticoinProxy(
        config=config,
        hlcvs=candles,
        mss=markets,
        btc=btc,
        timestamps=timestamps,
        exchange="bybit",
        batch_size=2,
        needed_metrics={"adg_strategy_eq"},
    )
    runner, finite = raw(proxy, [{}, {}])
    expected = policy in {"enabled", "coin_enabled"}
    assert runner.unstuck_pnl_lookback_bars == (43_200 if expected else 0)
    assert runner.unstuck_pnl_capacity == (len(candles) if expected else 0)
    assert bool(runner._unstuck_pnl_buffers) == expected
    if not expected:
        # Disabled HSL retains bounded binding scratch, not a history tape.
        assert runner.hsl_capacity == 2
        assert runner._history_bytes_per_candidate() < 1024
        config["live"]["pnls_max_lookback_days"] = "all"
        all_history_proxy = MpsMulticoinProxy(
            config=config,
            hlcvs=candles,
            mss=markets,
            btc=btc,
            timestamps=timestamps,
            exchange="bybit",
            batch_size=2,
            needed_metrics={"adg_strategy_eq"},
        )
        _, all_history = raw(all_history_proxy, [{}, {}])
        for key in ("balance", "fill_count", "psize"):
            np.testing.assert_array_equal(finite[key].cpu(), all_history[key].cpu())


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("lookback", [8 / 1440, "all"])
@pytest.mark.parametrize("coins", [1, 2])
def test_unstuck_expiring_loss_budget_matches_exact_rust(sides, lookback, strategy, coins):
    from backtest import run_backtest

    proxy, inputs = make_proxy(sides, lookback, strategy, coins)
    runner, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert output["fill_count"].item() == len(fills)
    assert sum(str(fill[13]).startswith("entry_") for fill in fills) == coins * len(sides)
    assert any("close_unstuck" in fill[13] for fill in fills)
    # Expired losses replenish the configured allowance; all-history exhausts it.
    assert (len(fills) > 10) == (lookback != "all")
    for side in sides:
        expected = sum(float(f[9]) for f in fills if f[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)
    assert runner.unstuck_pnl_lookback_bars == (0 if lookback == "all" else 8)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_ema_unstuck_reuses_finite_history_without_cpu_execution(monkeypatch, sides, coins):
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
            # A subsequent request must start with a fresh rolling account window.
            repeated = service.submit(BacktestRequest("repeat", "finite-unstuck", {})).result()
    assert [result.metrics for result in actual] == [result.metrics for result in expected]
    assert repeated.metrics == expected[0].metrics


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_unstuck_history_replay_reuse_and_bounded_batches(sides, strategy):
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
@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_ema_finite_unstuck_survives_disabled_hsl_specialization(side, mode):
    proxy, _ = make_proxy((side,), strategy="ema_anchor", hsl_mode=mode)
    runner, compact = raw(proxy, [{}])
    assert runner.dispatch_hsl_disabled
    assert runner.unstuck_pnl_capacity == 9
    runner.hsl_disabled_specialization = False
    _, general = raw(proxy, [{}])
    assert not runner.dispatch_hsl_disabled
    compare(compact, general)


@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_unstuck_history_overflow_fails_closed(sides, strategy):
    proxy, _ = make_proxy(sides, strategy=strategy)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    # Deliberately violate the capacity guarantee to test the error path.
    runner.unstuck_pnl_capacity = 1
    if len(sides) == 1 and strategy == "trailing_martingale":
        runner.max_dispatch_candidate_bars = 4
    with pytest.raises(RuntimeError, match="fill-PnL history overflow"):
        raw(proxy, [{}])


def test_shared_unstuck_window_preserves_intrabar_peak_and_expires_without_fills():
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = (
        "#define PASSIVBOT_HSL_CAPACITY 1\n"
        "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
        "#define PASSIVBOT_HSL_LOOKBACK 0\n"
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
    out[0] = effective_realized_pnl_drawdown(account);
    account.unstuck_pnl_k = 11; // candle 2 remains on the inclusive boundary
    refresh_unstuck_pnl_window(account);
    out[1] = effective_realized_pnl_drawdown(account);
    account.unstuck_pnl_k = 12; // no fills, but old loss/peak must expire
    refresh_unstuck_pnl_window(account);
    out[2] = effective_realized_pnl_drawdown(account);
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
