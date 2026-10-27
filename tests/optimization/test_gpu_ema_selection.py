"""Native EMA selection matches current ranking inputs rather than a cached set."""

import pytest


@pytest.mark.parametrize("mode", ["disabled", "coin", "pside", "portfolio"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_ema_current_ranking_matches_cpu(mode, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    from optimization.gpu.parity import MetricTolerance

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", sides, "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", mode,
    ]))
    for side in ("long", "short"):
        active = sides in (side, "both")
        inputs[0]["bot"][side]["risk"].update(
            max_positions=1 if active else 0,
            total_wallet_exposure_limit_ratio=2 if active else 0,
        )
        inputs[0]["bot"][side]["hsl"].update(
            drawdown_threshold_fraction=0.002, drawdown_fraction_ema_span_minutes=2.5, cooldown_after_red_minutes=5,
        )
    if mode == "portfolio":
        inputs[0]["bot"]["portfolio"]["hsl"].update(
            drawdown_threshold_fraction=0.002, drawdown_fraction_ema_span_minutes=2.5, cooldown_after_red_minutes=5,
        )
    metrics = ("cagr_fraction_strategy_equity", "drawdown_max_fraction_strategy_equity", "gross_turnover_ratio_per_day",
               "hsl_triggers_per_year", "hsl_triggers_per_year", "hsl_red_time_fraction")
    policies = {**DEFAULT_TOLERANCES,
                "hsl_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hsl_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hsl_red_time_fraction": MetricTolerance(1e-8, 1e-5)}
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    if mode != "disabled":
        assert report["metrics"]["hsl_triggers_per_year"]["cpu"] > 0
    assert report["passed"], report["metrics"]
