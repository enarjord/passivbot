"""CPU request preparation must work without an optional GPU execution stack."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hsl", ["disabled", "coin"])
def test_native_requests_prepare_without_gpu_imports_or_cpu_simulation(strategy, hsl):
    import passivbot_rust
    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real native Rust extension required")

    # A new process prevents earlier tests' GPU imports from hiding a dependency.
    script = textwrap.dedent("""
        import builtins
        import math
        import sys

        blocked = (
            "torch", "cupy", "optimization.gpu.runtime",
            "optimization.gpu.service", "optimization.gpu.mps_kernel",
            "optimization.gpu.cuda_kernel", "tools.gpu_proxy_benchmark",
        )
        original_import = builtins.__import__
        def cpu_only(name, *args, **kwargs):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
                raise AssertionError("CPU preparation imported " + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = cpu_only

        import passivbot_rust
        assert not getattr(passivbot_rust, "__is_stub__", False)
        import backtest
        def forbidden(*args, **kwargs):
            raise AssertionError("CPU request preparation ran a simulation")
        for name in ("execute_backtest", "run_backtest"):
            setattr(backtest, name, forbidden)
        backtest.pbr.run_backtest_bundle = forbidden

        from optimization.gpu.datasets import PreparedGpuDataset
        from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
        from optimize import Evaluator, config_to_individual
        from shared_arrays import SharedArraySpec
        from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

        strategy, hsl = sys.argv[1:]
        config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
            "--fixture", strategy, "--sides", "both", "--coins", "3",
            "--bars", "128", "--seed", "43", "--hsl", hsl, "--unstuck",
        ]))
        coins = config["backtest"]["coins"]["binance"]
        assert coins == ["COIN00", "COIN01", "COIN02"]
        assert candles.shape == (128, 3, 4)
        assert int(timestamps[-1]) - int(timestamps[0]) == 127 * 60_000
        assert candles[0, 0, 2] < candles[0, 1, 2] < candles[0, 2, 2]
        quantity = "base_qty_pct" if strategy == "ema_anchor" else "entry_initial_qty_pct"
        config["optimize"]["bounds"] = {f"long_{quantity}": [0.01, 0.05]}
        for side in ("long", "short"):
            for key in ("n_positions", "total_wallet_exposure_limit"):
                value = config["bot"][side]["risk"][key]
                config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["scoring"] = [
            {"metric": "adg_strategy_eq", "goal": "max"},
            {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
        ]
        config["optimize"]["limits"] = []
        spec = SharedArraySpec("candles", candles.shape, candles.dtype.str)
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", hlcvs=spec,
            btc=SharedArraySpec("btc", btc.shape, btc.dtype.str),
            timestamps=SharedArraySpec("timestamps", timestamps.shape, timestamps.dtype.str),
            candle_coins=coins, metrics=DEFAULT_METRICS,
        )
        evaluator = Evaluator({"binance": spec}, {}, {"binance": markets}, config)
        evaluator.evaluate = forbidden
        planner = NativeCandidatePlanner(evaluator, [ScenarioBinding("base", "dataset", dataset)])
        vector = config_to_individual(config, evaluator.bounds,
                                      optimization_shape=evaluator.optimization_shape)
        first = planner.prepare("first", vector)
        changed = list(vector)
        changed[[name for name, _ in evaluator.key_paths].index(f"long_{quantity}")] = 0.04
        second = planner.prepare("second", changed)
        assert first.stage == second.stage == "full"
        assert len(first.requests) == len(second.requests) == 1
        assert first.effective_key != second.effective_key
        request = second.requests[0]
        assert request.dataset_id == "dataset"
        assert request.parameters[f"long_{quantity}"] == 0.04
        assert request.parameters["long_hsl_enabled"] == float(hsl == "coin")
        assert request.parameters["short_hsl_enabled"] == float(hsl == "coin")
        assert all(math.isfinite(value) for value in request.parameters.values())
        assert not any(name == prefix or name.startswith(prefix + ".")
                       for name in sys.modules for prefix in blocked)
    """)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, strategy, hsl], cwd=root, env=env,
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
