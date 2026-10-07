"""GPU lifecycle reporting includes open RED, GREEN recovery and retriggers."""

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode,expected", [
    (0, [2, 1, 1, 9, 2, 4, 2, 3, 7]),
    (1, [2, 1, 1, 3, 2, 2, 1, 3, 2]),
    (2, [1, 1, 0, 1, 1, 1, 1, 0, 1]),
    (3, [2, 1, 1, 5, 2, 4, 2, 3, 3]),
    (4, [1, 0, 0, 2, 1, 2, 1, 3, 2]),
], ids=["cooldown-retrigger", "green-recovery", "zero-cooldown", "renewed-exposure", "open-panic"])
def test_shared_hsl_observed_lifecycle(strategy, mode, expected):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = ("#define PASSIVBOT_HSL_CAPACITY 64\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
              "#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 1\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void lifecycle_probe(constant float* params, device HslNode* trees,
    device int* rows, constant int* mode, device float* out,
    uint b [[thread_position_in_grid]]) {
    HslState h = load_hsl(params, 0, 0);
    bind_hsl(h, trees, rows, 0, 64, 1, 1440, true, true);
    observe_hsl(h, 1000, 0, 0, false, 0, false);
    observe_hsl(h, 1000, 0, 0, true, 1, false);
    observe_hsl(h, 1000, 0, -200, true, 2, false);
    float last = 4;
    if (mode[0] == 0) {
        observe_hsl(h, 1000, 0, -100, true, 3, false);
        observe_hsl(h, 900, -100, 0, false, 4, true);
        observe_hsl(h, 900, -100, 0, false, 5, false);
        observe_hsl(h, 900, -100, 0, false, 9, false);
        observe_hsl(h, 900, -100, 0, true, 10, false);
        observe_hsl(h, 900, -100, -200, true, 11, false);
        observe_hsl(h, 900, -100, -200, true, 11, false);
        last = 13;
    } else if (mode[0] == 1) {
        observe_hsl(h, 1000, 0, 100, true, 3, false);
        observe_hsl(h, 1000, 0, -300, true, 4, false);
        last = 6;
    } else if (mode[0] == 2) {
        observe_hsl(h, 800, -200, 0, false, 3, true);
        last = 3;
    } else if (mode[0] == 3) {
        observe_hsl(h, 1000, 0, -100, true, 3, false);
        observe_hsl(h, 900, -100, 0, false, 4, true);
        observe_hsl(h, 900, -100, -200, true, 5, false);
        last = 7;
    }
    HslOutputAggregate report = init_hsl_output_aggregate(0, 0);
    accumulate_hsl_output(report, h, false, last);
    out[0] = report.triggers_long;
    out[1] = report.restarts_long;
    out[2] = report.restart_retrigger_count;
    out[3] = report.duration_sum;
    out[4] = report.duration_count;
    out[5] = report.flatten_time_sum;
    out[6] = report.flatten_time_count;
    out[7] = h.hsl.action;
    out[8] = report.duration_max;
}
""")
    device = gpu_device()
    params = torch.tensor([1, .05, 1, 0 if mode == 2 else 5, 0, 2, 1],
                          dtype=torch.float32, device=device)
    trees = torch.empty((36, 32), dtype=torch.uint8, device=device)
    rows = torch.empty(256, dtype=torch.int32, device=device)
    case = torch.tensor([mode], dtype=torch.int32, device=device)
    out = torch.zeros(9, dtype=torch.float32, device=device)
    compile_shader(source).lifecycle_probe(params, trees, rows, case, out, threads=1)
    assert out.cpu().tolist() == expected


@pytest.mark.parametrize("strategy,mode,sides", [
    (strategy, mode, sides)
    for strategy in ("ema_anchor", "trailing_martingale")
    for mode, sides in (("coin", "long"), ("coin", "short"), ("coin", "both"),
                        ("pside", "both"), ("unified", "both"))
])
def test_native_hsl_lifecycle_metrics_match_cpu(strategy, mode, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", mode,
    ]))
    for side in ("long", "short"):
        inputs[0]["bot"][side]["hsl"].update(
            red_threshold=.002, ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,
        )
    if mode == "unified":
        inputs[0]["bot"]["hsl"].update(
            red_threshold=.002, ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,
        )
    inputs[1][1500:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    # Flatten-time remains CPU-analysis-only at the public native API; the
    # shader probes above verify its raw reporting/censoring separately.
    metrics = [
        "hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
        "hard_stop_post_restart_retrigger_pct", "hard_stop_duration_minutes_mean",
        "hard_stop_duration_minutes_max", "hard_stop_trigger_drawdown_mean",
        "hard_stop_time_in_red_pct",
    ]
    report = run_comparison(
        inputs, "binance", metrics,
        {name: MetricTolerance(1e-5, 1e-3) for name in metrics}, gpu_engine="native",
    )
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["passed"], report["metrics"]
