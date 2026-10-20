"""Loss admission and unstuck share one continuous realized-memory fold."""

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
    sides, coins=2, memory_span=8.0, *, unstuck=False, hsl_mode="coin", max_loss_pct=None,
    interval_minutes=1,
):
    from optimization.gpu.service import SharedAccountReplay

    original, inputs = make_proxy(
        sides, memory_span, strategy="ema_anchor", coins=coins, hsl_mode=hsl_mode,
        interval_minutes=interval_minutes,
    )
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    config = deepcopy(config)
    config["live"]["lookback_ema_span_minutes"] = memory_span
    config["live"]["max_realized_loss_pct"] = 0.003 if unstuck else 0.0003
    if max_loss_pct is not None:
        config["live"]["max_realized_loss_pct"] = max_loss_pct
    if not unstuck:
        # Each flat-price close loses only its fee. Entry fees initially exhaust
        # the gate; continuous decay later admits ordinary closes without unstuck.
        candles = candles.copy()
        candles[:, :, :3] = (101.0, 99.0, 100.4)
        for side in sides:
            config["bot"][side]["unstuck"]["enabled"] = False
    inputs = candles, markets, config, exchange, btc, timestamps
    proxy = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets, exchange=exchange,
        btc=btc, timestamps=timestamps, batch_size=3,
        needed_metrics={"adg_strategy_eq", "fills_per_day"},
    )
    return proxy, inputs


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides,fraction,span,interval,unstuck", [
    pytest.param(("long",), 0.0003, 3.0, 1, False, id="long-gate-only"),
    pytest.param(("short",), 0.0003, 3.0, 1, False, id="short-gate-only"),
    pytest.param(("long", "short"), 0.0003, 3.0, 1, False, id="both-gate-only"),
    pytest.param(("long", "short"), 0.0003, 8.0, 5, False, id="both-five-minute"),
    pytest.param(("long", "short"), 0.003, 8.0, 1, True, id="both-consumers"),
    pytest.param(("long",), 0.0, 8.0, 1, False, id="zero"),
    pytest.param(("long",), 0.99999999, 8.0, 1, False, id="active-rounds-one"),
    pytest.param(("long",), 1e308, 8.0, 1, False, id="disabled-huge-finite"),
])
def test_shared_gate_production_consumers_and_transport(
    strategy, sides, fraction, span, interval, unstuck
):
    import numpy as np
    from backtest import run_backtest
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_tm_loss_admission import fee_only_close_fixture

    if strategy == "ema_anchor":
        proxy, inputs = loss_gate_proxy(
            sides, memory_span=span, unstuck=unstuck, max_loss_pct=fraction,
            interval_minutes=interval,
        )
    elif not unstuck:
        proxy, inputs = fee_only_close_fixture(
            sides, 2, span, fraction, interval_minutes=interval,
        )
    else:
        original, inputs = make_proxy(sides, span, strategy=strategy)
        del original
        candles, markets, config, exchange, btc, timestamps = inputs
        config["live"]["max_realized_loss_pct"] = fraction
        proxy = SharedAccountReplay(
            config=config, hlcvs=candles, mss=markets, exchange=exchange,
            btc=btc, timestamps=timestamps, batch_size=2,
            needed_metrics={"adg_strategy_eq", "fills_per_day"},
        )
    runner, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert len(fills) > 0
    assert runner.loss_gate_enabled is (fraction < 1.0)
    assert runner.settings[-1].item() == float(fraction < 1.0)
    assert runner.settings[5].item() == float(np.float32(min(fraction, 1.0)))
    if not unstuck:
        assert runner.dispatch_unstuck_enabled is False
    assert not hasattr(runner, "_unstuck_pnl_buffers")
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)
    if 0.0 < fraction < 1.0 and not unstuck:
        assert any(str(fill[13]).startswith("close_") for fill in fills)
    # Retained memory must survive real dispatch boundaries, then start fresh
    # on the next request. Neither operation owns an additional fill producer.
    runner.max_dispatch_candidate_bars = 8
    _, chunked = raw(proxy, [{}])
    compare(output, chunked)
    _, repeated = raw(proxy, [{}])
    compare(output, repeated)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("max_loss_pct", [0.0, 0.5, 1.0])
def test_ema_gate_uses_constant_memory_without_a_history_queue(sides, max_loss_pct):
    proxy, _ = loss_gate_proxy(sides, max_loss_pct=max_loss_pct)
    runner, _ = raw(proxy, [{}])
    enabled = max_loss_pct < 1.0
    assert runner.loss_gate_enabled == enabled
    assert not hasattr(runner, "_unstuck_pnl_buffers")
    assert runner.settings[-1].item() == float(enabled)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("memory_span", [3.0, 8.0])
@pytest.mark.parametrize("unstuck", [False, True])
def test_ema_shared_decaying_gate_matches_rust(sides, coins, memory_span, unstuck):
    from backtest import run_backtest

    proxy, inputs = loss_gate_proxy(sides, coins, memory_span, unstuck=unstuck)
    runner, output = raw(proxy, [{}])
    fills, _, _ = run_backtest(*inputs)
    assert output["fill_count"].item() == len(fills)
    assert output["balance"].item() == pytest.approx(float(fills[-1, 5]), abs=0.002)
    for side in sides:
        expected = sum(float(fill[9]) for fill in fills if fill[13].endswith(side))
        key = "psize" if side == "long" else "short_psize"
        assert output[key].item() == pytest.approx(abs(expected), abs=2e-5)
    assert not hasattr(runner, "_unstuck_pnl_buffers")
    assert len(fills) > coins * len(sides)


def test_ema_shared_decay_gate_close_boundary_matches_rust():
    import numpy as np
    from backtest import run_backtest

    proxy, inputs = loss_gate_proxy(("long",), coins=1, memory_span=9.0)
    fills, _, _ = run_backtest(*inputs)
    # Continuous decay must admit a real fee-only close at the same CPU/GPU cut.
    assert len(fills) == 2
    assert str(fills[-1, 13]).startswith("close_")
    close_k = int(fills[-1, 0])
    for end in (close_k, close_k + 1):
        runner, output = raw(proxy, [{}], end_steps=np.array([end], dtype=np.int32))
        expected = [fill for fill in fills if int(fill[0]) < end]
        assert len(expected) == (1 if end == close_k else 2)
        assert not hasattr(runner, "_unstuck_pnl_buffers")
        assert output["fill_count"].item() == len(expected)
        assert output["balance"].item() == pytest.approx(float(expected[-1][5]), abs=0.002)
        assert output["psize"].item() == pytest.approx(abs(float(expected[-1][11])), abs=2e-5)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
def test_ema_loss_gate_memory_reuse_specialization_and_scratch(side, mode):
    proxy, _ = loss_gate_proxy((side,), hsl_mode=mode)
    candidates = [{}, {f"{side}_base_qty_pct": 0.3}, {}]
    runner, expected = raw(proxy, candidates)
    assert expected["balance"][0].item() != expected["balance"][1].item()
    assert runner.dispatch_hsl_disabled
    assert not hasattr(runner, "_unstuck_pnl_buffers")
    runner.hsl_disabled_specialization = False
    _, general = raw(proxy, candidates)
    compare(expected, general)
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate() * 2
    _, split = raw(proxy, candidates, profile=True)
    compare(expected, split)
    assert runner.last_profile["candidate_batch_count"] == 2
    _, repeated = raw(proxy, candidates)
    compare(expected, repeated)


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_loss_gate_reuses_memory_without_cpu_backtests(monkeypatch, sides, coins):
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
