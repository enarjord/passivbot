"""Native HSL loss metrics share Rust reporting formulas outside optimization."""

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_hsl_episode_loss_matches_cpu_report(strategy, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", "coin",
    ]))
    for side in ("long", "short"):
        inputs[0]["bot"][side]["hsl"].update(
            drawdown_threshold_fraction=0.002, drawdown_fraction_ema_span_minutes=2.5,
            cooldown_after_red_minutes=5,
        )
    inputs[1][1500:, 0, :3] *= 0.7
    inputs[1][1800:, 1, :3] *= 1.3
    metrics = ("hsl_panic_episode_loss_max_fraction", "hsl_panic_episode_loss_mean_fraction")
    report = run_comparison(
        inputs, "binance", metrics,
        {metric: MetricTolerance(1e-5, 1e-3) for metric in metrics}, gpu_engine="native",
    )
    # Zero-loss episodes would miss a dead observation scalar. Both episode
    # aggregates require an actual positive realized HSL loss in this fixture.
    for engine in ("cpu", "gpu"):
        maximum, mean = (report["metrics"][metric][engine] for metric in metrics)
        assert maximum >= mean > 0
    assert report["passed"], report["metrics"]
