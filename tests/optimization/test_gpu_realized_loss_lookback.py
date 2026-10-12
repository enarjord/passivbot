"""EMA loss admission uses the same finite fill history as exact Rust."""

from copy import deepcopy

import pytest

from test_gpu_hsl_multicoin import compare, raw
from test_gpu_unstuck_lookback import make_proxy

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def loss_gate_proxy(
    sides, coins=2, lookback=8 / 1440, *, unstuck=False, hsl_mode="coin", max_loss_pct=None
):
    from optimization.gpu.service import MpsMulticoinProxy

    original, inputs = make_proxy(
        sides, lookback, strategy="ema_anchor", coins=coins, hsl_mode=hsl_mode
    )
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    config = deepcopy(config)
    config["live"]["max_realized_loss_pct"] = 0.003 if unstuck else 0.0003
    if max_loss_pct is not None:
        config["live"]["max_realized_loss_pct"] = max_loss_pct
    if not unstuck:
        # Each flat-price close loses only its fee. Entry fees initially exhaust
        # the gate; after expiry, ordinary closes are admitted without unstuck.
        candles = candles.copy()
        candles[:, :, :3] = (101.0, 99.0, 100.4)
        for side in sides:
            config["bot"][side]["unstuck"]["enabled"] = False
    inputs = candles, markets, config, exchange, btc, timestamps
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, exchange=exchange,
        btc=btc, timestamps=timestamps, batch_size=3,
        needed_metrics={"adg_strategy_eq", "fills_per_day"},
    )
    return proxy, inputs


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.5, 1.0])
def test_ema_history_is_present_only_for_a_finite_loss_consumer(sides, max_loss_pct):
    proxy, _ = loss_gate_proxy(sides, max_loss_pct=max_loss_pct)
    runner, _ = raw(proxy, [{}])
    enabled = max_loss_pct < 1.0
    assert runner.unstuck_pnl_lookback_bars == (8 if enabled else 0)
    assert runner.unstuck_pnl_capacity == (9 if enabled else 0)
    assert bool(runner._unstuck_pnl_buffers) == enabled


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("lookback", [8 / 1440, "all"])
@pytest.mark.parametrize("unstuck", [False, True])
def test_ema_loss_gate_expiry_matches_rust(sides, coins, lookback, unstuck):
    from backtest import run_backtest

    proxy, inputs = loss_gate_proxy(sides, coins, lookback, unstuck=unstuck)
    runner, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)
    assert runner.unstuck_pnl_lookback_bars == (0 if lookback == "all" else 8)
    assert runner.unstuck_pnl_capacity == (0 if lookback == "all" else 9)
    if lookback != "all":
        assert len(fills) > coins * len(sides)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_ema_loss_gate_history_reuse_specialization_and_scratch(side, mode):
    proxy, _ = loss_gate_proxy((side,), hsl_mode=mode)
    candidates = [{}, {f"{side}_base_qty_pct": 0.3}, {}]
    runner, expected = raw(proxy, candidates)
    assert expected["balance"][0].item() != expected["balance"][1].item()
    assert runner.dispatch_hsl_disabled
    assert runner.unstuck_pnl_capacity == 9
    runner.hsl_disabled_specialization = False
    _, general = raw(proxy, candidates)
    compare(expected, general)
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate() * 2
    _, split = raw(proxy, candidates, profile=True)
    compare(expected, split)
    assert runner.last_profile["candidate_batch_count"] == 2
    _, repeated = raw(proxy, candidates)
    compare(expected, repeated)


@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_ema_loss_gate_history_overflow_fails_closed(sides):
    # Admit adjacent entry/close events so a one-slot tape really is too small.
    proxy, _ = loss_gate_proxy(sides, max_loss_pct=0.5)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    runner.unstuck_pnl_capacity = 1
    with pytest.raises(RuntimeError, match="fill-PnL history overflow"):
        raw(proxy, [{}])


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_loss_gate_reuses_history_without_cpu_backtests(monkeypatch, sides, coins):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA service required")
    import backtest
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    proxy, inputs = loss_gate_proxy(sides, coins)
    candidates = [{}, {f"{sides[0]}_base_qty_pct": 0.3}, {}]
    expected = proxy.evaluate_results(candidates)

    def forbidden(*args, **kwargs):
        pytest.fail("native loss-gate replay must not execute a CPU backtest")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    candles, markets, config, exchange, btc, timestamps = inputs
    with _native_dataset((config, candles, markets, btc, timestamps), exchange,
                         {"adg_strategy_eq", "fills_per_day"}) as dataset:
        with CudaBacktestService(batch_size=2, tuning_mode="off") as service:
            service.register_dataset("finite-loss", dataset)
            pending = [service.submit(BacktestRequest(str(i), "finite-loss", values))
                       for i, values in enumerate(candidates)]
            actual = [future.result() for future in pending]
            repeated = service.submit(BacktestRequest("repeat", "finite-loss", {})).result()
    assert [result.metrics for result in actual] == [result.metrics for result in expected]
    assert repeated.metrics == expected[0].metrics
