from tools.run_ema_anchor_autoresearch_round import build_optimize_command


def test_build_baseline_optimize_command():
    cmd = build_optimize_command(
        config_path="configs/examples/ema_anchor.json",
        symbol="XMR",
        start_date="2024-10-01",
        end_date="2026-04-01",
        n_cpus=3,
        iterations=100000,
        results_dir="optimize_results/autoresearch_baseline_xmr",
    )

    assert cmd[:3] == ["passivbot", "optimize", "configs/examples/ema_anchor.json"]
    assert "--backtest.candle_interval_minutes" in cmd
    assert cmd[cmd.index("-t") + 1] == "optimize_results/autoresearch_baseline_xmr"
    assert "--start" not in cmd
    assert "--fine_tune_params" not in cmd


def test_build_candidate_optimize_command():
    cmd = build_optimize_command(
        config_path="configs/examples/ema_anchor.json",
        symbol="XMR",
        start_date="2024-10-01",
        end_date="2026-04-01",
        n_cpus=3,
        iterations=3000,
        results_dir="optimize_results/autoresearch_candidate_xmr",
        baseline_pareto="optimize_results/autoresearch_baseline_xmr/pareto",
        fine_tune_params=["long_offset", "short_offset"],
    )

    assert cmd[cmd.index("--start") + 1] == "optimize_results/autoresearch_baseline_xmr/pareto"
    assert cmd[cmd.index("--fine_tune_params") + 1] == "long_offset,short_offset"
