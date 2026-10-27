"""Metal reductions and prepared-service qualification with public fixtures."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Metal required")








@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_metal_native_parity_uses_shared_service(strategy, sides):
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_METRICS, DEFAULT_TOLERANCES
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "2", "--sides", sides, "--bars", "512",
    ]))
    report = run_comparison(inputs, "binance", DEFAULT_METRICS, DEFAULT_TOLERANCES, diagnostics=True)
    assert report["gpu_engine"] == "native" and report["gpu_replay"] == "shared_account"
    assert report["status"] == "passed", report




@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_metal_factual_hsl_matches_full_dispatch_through_policy_transitions(monkeypatch, strategy, mode):
    import backtest
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_hsl_factual_replay import _equal

    def forbidden(*args, **kwargs):
        pytest.fail("Metal HSL qualification cannot invoke a CPU simulation")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    sides = ("long", "short")
    proxy = make_proxy(mode, strategy, sides, minutes=256, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    assert runner.mps_coin_capacity == 2
    runner.max_dispatch_candidate_bars = None
    _, full = raw(proxy, [{}])
    full = {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in full.items()}
    runner.max_dispatch_candidate_bars = 64
    runner.hsl_fact_capacity_learned = 0
    _, chunked = raw(proxy, [{}])
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    assert runner.last_hsl_fact_retries == 0
    learned = runner.hsl_fact_capacity_learned
    _equal(chunked, full)
    disabled = {f"{side}_hsl_enabled": 0.0 for side in sides}
    raw(proxy, [disabled])
    assert runner.hsl_fact_capacity == 0
    assert runner.hsl_fact_capacity_learned == learned
    _, restored = raw(proxy, [{}])
    assert runner.hsl_fact_capacity == learned
    assert runner.last_hsl_fact_retries == 0
    _equal(restored, full)
