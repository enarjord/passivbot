"""Real CPU entry points remain independent of optional GPU execution libraries."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


CPU_IMPORT_GUARD = """
import builtins
blocked = (
    "torch", "cupy", "optimization.gpu.runtime", "optimization.gpu.service",
    "optimization.gpu.mps_kernel", "optimization.gpu.cuda_kernel",
    "tools.gpu_proxy_benchmark",
)
original_import = builtins.__import__
def cpu_only(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise AssertionError("CPU entry point imported " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_only
"""

CPU_SETUP = """
import json
import sys
from pathlib import Path
from sitecustomize import blocked
import passivbot_rust
assert not getattr(passivbot_rust, "__is_stub__", False)
from tools.gpu_parity import build_parser, fixture_inputs
strategy, destination = sys.argv[1:3]
directory = Path(destination)
config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
    "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "512",
]))
"""


def run_cpu_process(tmp_path, strategy, source, *arguments):
    import passivbot_rust
    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real native Rust extension required")
    root = Path(__file__).resolve().parents[2]
    # Interpreter startup installs the guard in forkserver/spawn workers too.
    (tmp_path / "sitecustomize.py").write_text(CPU_IMPORT_GUARD)
    script = tmp_path / "cpu_check.py"
    body = CPU_SETUP + textwrap.dedent(source) + """
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in blocked)
"""
    script.write_text("if __name__ == '__main__':\n" + textwrap.indent(body, "    "))
    result = subprocess.run(
        [sys.executable, str(script), strategy, str(tmp_path), *arguments],
        cwd=root, env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(root / "src"))),
                          MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cpu_backtest_exports_and_plots_without_gpu_imports(tmp_path, strategy):
    run_cpu_process(tmp_path, strategy, """
        from backtest import build_backtest_payload, execute_backtest, post_process, BacktestPlotContext
        config["disable_plotting"] = ["coin_fills", "hard_stop"]
        payload = build_backtest_payload(
            candles, markets, config, "binance", btc, timestamps,
            metrics_only=False, skip_btc_analysis=True,
        )
        fills, equity, analysis = execute_backtest(payload, config)
        assert len(fills) > 0 and len(equity) > 0
        output = directory / "backtest"
        post_process(config, candles, fills, equity, btc, analysis, str(output) + "/", "binance",
                     plot_context=BacktestPlotContext.from_payload(payload))
        files = {path.name: path for path in output.rglob("*") if path.is_file()}
        assert {"analysis.json", "fills.csv", "balance_and_equity.csv.gz", "config.json"} <= files.keys()
        pictures = [path for name, path in files.items() if name.endswith(".png")]
        assert len(pictures) >= 3
        assert all(path.read_bytes().startswith(b"\\x89PNG\\r\\n\\x1a\\n") for path in pictures)
    """)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("backend", ["deap", "pymoo"])
def test_cpu_optimizer_and_resume_use_real_workers_without_gpu_imports(tmp_path, strategy, backend):
    run_cpu_process(tmp_path, strategy, """
        import asyncio
        from copy import deepcopy
        import msgpack
        import pytest
        import optimize

        backend = sys.argv[3]
        config["optimize"].update(backend=backend, population_size=4, iters=8, seed=12, n_cpus=2)
        config["optimize"]["scoring"] = [
            dict(metric="adg_strategy_eq", goal="max"),
            dict(metric="drawdown_worst_strategy_eq", goal="min"),
        ]
        config["optimize"]["limits"] = [
            dict(metric="backtest_completion_ratio", penalize_if="less_than", value=0.99),
        ]
        config["optimize"]["bounds"] = {}
        quantity = "base_qty_pct" if strategy == "ema_anchor" else "entry_initial_qty_pct"
        for side in ("long", "short"):
            for name in ("n_positions", "total_wallet_exposure_limit"):
                value = config["bot"][side]["risk"][name]
                config["optimize"]["bounds"][f"{side}_{name}"] = [value, value]
            config["optimize"]["bounds"][f"{side}_{quantity}"] = [0.01, 0.05]
        config["backtest"].update(suite_enabled=False, scenarios=[])
        config_path = directory / "input.json"
        config_path.write_text(json.dumps(config))
        async def offline(*args, **kwargs):
            return None
        async def prepared(*args, **kwargs):
            return list(config["backtest"]["coins"]["binance"]), candles, deepcopy(markets), "", "", btc, timestamps
        def entries(output):
            with (output / "all_results.bin").open("rb") as handle:
                return list(msgpack.Unpacker(handle, raw=False, strict_map_key=False))
        async def optimize_offline(argv):
            sys.argv = argv
            try:
                await optimize.main()
            except SystemExit as error:
                assert error.code == 0
        async def run():
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(optimize, "format_approved_ignored_coins", offline)
                patch.setattr(optimize, "prepare_hlcvs_mss", prepared)
                patch.chdir(directory)
                await optimize_offline(["passivbot optimize", str(config_path), "--offline", "y"])
                sessions = list((directory / "optimize_results").iterdir())
                assert len(sessions) == 1
                output = sessions[0]
                initial = len(entries(output))
                assert initial > 0 and list((output / "pareto").glob("*.json"))
                manifest = (output / "session.json").read_bytes()
                assert json.loads(manifest)["setup"]["config"]["optimize"]["backend"] == backend
                await optimize_offline(["passivbot optimize", str(config_path), "--offline", "y",
                                        "--resume", str(output), "-i", "12"])
                assert len(entries(output)) > initial
                assert (output / "session.json").read_bytes() == manifest
                assert list((directory / "optimize_results").iterdir()) == sessions
                assert list((output / "pareto").glob("*.json"))
        asyncio.run(run())
    """, backend)
