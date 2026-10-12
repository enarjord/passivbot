"""Native HSL loss metrics share Rust reporting formulas outside optimization."""

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_hsl_halt_loss_matches_cpu_report(strategy, sides):
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
            red_threshold=0.002, ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,
        )
    inputs[1][1500:, 0, :3] *= 0.7
    inputs[1][1800:, 1, :3] *= 1.3
    metric = "hard_stop_halt_to_restart_equity_loss_pct"
    report = run_comparison(
        inputs, "binance", [metric],
        {metric: MetricTolerance(1e-5, 1e-3)}, gpu_engine="native",
    )
    # A zero-loss fixture would miss the dead-scalar regression entirely.
    assert report["metrics"][metric]["cpu"] > 0
    assert report["passed"], report["metrics"]
