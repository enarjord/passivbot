"""The native shared-account path retains canonical TM boundary operands."""

import pytest


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("case", ["half", "below_half", "tick", "off_tick"])
def test_native_decimal_boundaries_match_rust_without_cpu_substitution(monkeypatch, side, coins, case):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("Metal or CUDA required")
    from test_gpu_entry_sizing_parity import _decimal_boundary_inputs
    from optimization.gpu.service import MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension
    import backtest

    verify_loaded_runtime_extension()
    if case in {"half", "below_half"}:
        inputs = _decimal_boundary_inputs(side, coins, price=100.0,
            balance=1000.0 if case == "half" else 10000.0,
            initial_pct=.0037 if case == "half" else .05,
            qty_step=.001 if case == "half" else .1,
            factor=1.5 if case == "half" else 1.15 - 1e-12,
            threshold=.01, low=98.9, high=101.1)
    else:
        inputs = _decimal_boundary_inputs(side, coins, price=2.5, balance=100.0,
            initial_pct=.025, qty_step=.001, factor=1.0,
            threshold=.044 + (1e-12 if case == "off_tick" else 0.0),
            low=2.385, high=2.615)
    config, candles, markets, btc, timestamps = inputs
    fills, _, _ = backtest.run_backtest(candles, markets, config, "bybit", btc, timestamps)
    expected_size = abs(sum(float(fill[9]) for fill in fills))
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1,
        needed_metrics={"adg_strategy_eq"}, factual_hsl=True)
    runner = proxy.runners[side]
    original = runner.run
    captured = []

    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        captured.append(output)
        return output

    runner.run = capture
    monkeypatch.setattr(backtest, "execute_backtest",
                        lambda *_a, **_k: pytest.fail("CPU substitution in native replay"))
    proxy.evaluate([{}])
    output = captured[0]
    assert output["fill_count"].item() == len(fills)
    assert output["short_psize" if side == "short" else "psize"].item() == pytest.approx(
        expected_size, abs=2e-6)
