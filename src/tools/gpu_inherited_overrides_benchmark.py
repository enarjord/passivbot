"""Offline CUDA A/B/A benchmark of inherited coin override lookups."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import statistics
import time

import numpy as np
import torch

from optimization.gpu.parameters import prepare_candidate_parameters
from optimization.gpu.service import SharedAccountReplay
from rust_utils import verify_loaded_runtime_extension
from tools.gpu_parity import build_parser as fixture_parser, fixture_inputs


def benchmark_inputs(*, bars: int, coins: int):
    """Prepare canonical benchmark inputs before constructing a device replay."""
    fixture = fixture_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "long", "--coins", str(coins),
        "--bars", str(bars), "--seed", "43", "--hsl", "disabled", "--unstuck",
    ])
    config, candles, markets, btc, timestamps = fixture_inputs(fixture)
    config["bot"]["long"]["risk"].update(
        max_positions=min(7, coins - 1), total_wallet_exposure_limit=1.5
    )
    config["backtest"]["limit_order_fill_buffer_pct"] = 0.0005
    config["backtest"]["ew_half_life_days"] = 180.0
    return config, candles, markets, btc, timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=65536)
    parser.add_argument("--coins", type=int, default=25)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--warm-runs", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not (61 <= args.bars <= 100000 and 2 <= args.coins <= 64
            and 1 <= args.candidates <= 256 and 1 <= args.warm_runs <= 10):
        parser.error("bars=61..100000, coins=2..64, candidates=1..256, warm-runs=1..10")
    import passivbot_rust  # Verify the extension before constructing fixtures.
    verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps = benchmark_inputs(bars=args.bars, coins=args.coins)
    candidates = []
    for i in range(args.candidates):
        candidate = deepcopy(config)
        side = candidate["bot"]["long"]
        side["entry_cooldown"].update(
            base_duration_minutes=0.0, min_duration_minutes=float(i % 10),
            max_duration_minutes=180.0,
            weights_minutes={"exposure_ratio": float(10 + i % 100),
                            "adverse_directionality": float(20 + i % 200)},
        )
        side["forager"]["score_weights"]["unilateralness"] = 0.2
        side["forager"]["unilateralness_ema_span_1m"] = 15.0 + i % 166
        candidates.append(prepare_candidate_parameters(candidate, markets, "binance"))
    proxy = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange="binance", batch_size=args.candidates, factual_hsl=True,
        needed_metrics={"adg_strategy_eq", "adg_strategy_eq_ew",
                        "drawdown_worst_strategy_eq", "strategy_eq_underwater_pct_mean",
                        "fills_gap_time_weighted_mean_hours",
                        "position_held_time_weighted_mean_hours", "backtest_completion_ratio"},
    )
    runner = proxy.runners["long"]
    matrix = proxy._parameter_matrix(candidates, "long")
    reference_raw = reference_metrics = None
    results = []
    for mode in ("general", "inherited", "general_repeat"):
        runner.coin_overrides_absent = mode == "inherited"
        times = []
        for repeat in range(args.warm_runs + 1):
            start = time.perf_counter()
            metrics = proxy.evaluate_results(candidates)
            times.append(time.perf_counter() - start)
            print(json.dumps({"mode": mode, "repeat": repeat, "seconds": times[-1]}), flush=True)
        raw = {key: value.cpu().numpy().copy() for key, value in runner.run(matrix).items()
               if isinstance(value, torch.Tensor)}
        if reference_raw is None:
            reference_raw, reference_metrics = raw, metrics
        else:
            assert raw.keys() == reference_raw.keys()
            for key, value in raw.items():
                np.testing.assert_array_equal(value, reference_raw[key], err_msg=key)
            assert metrics == reference_metrics
        results.append({"mode": mode, "warm_seconds": times[1:],
                        "median_seconds": statistics.median(times[1:])})
    general = statistics.mean([results[0]["median_seconds"], results[2]["median_seconds"]])
    print(json.dumps({"results": results, "throughput_ratio": general / results[1]["median_seconds"],
                      "raw_arrays_identical": len(reference_raw), "metrics_identical": True,
                      "additional_workspace_bytes": 0}), flush=True)


if __name__ == "__main__":
    main()
