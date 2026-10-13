"""Offline matched native GPU coin-HSL cache benchmark with synthetic candles.

Run with ``python -m tools.gpu_hsl_coin_cache_benchmark``. The first run of
both variants includes shader compilation; only subsequent runs form the
reported speedup. Each run checks native replay outputs against cache-off.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import statistics
import time

import numpy as np

from tools.gpu_parity import fixture_inputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=65536)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--sides", choices=("long", "short", "both"), default="both")
    parser.add_argument("--report", help="Optional JSON report file")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (not 4096 <= args.bars <= 100_000 or not 1 <= args.candidates <= 64
            or not 1 <= args.lookback_days <= 60 or args.bars <= args.lookback_days * 1440):
        raise ValueError("benchmark requires 4096..100000 bars, 1..64 candidates and 1..60 lookback days shorter than history")
    import torch
    from optimization.gpu.service import MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension

    runtime = verify_loaded_runtime_extension()
    recipe = argparse.Namespace(
        fixture="trailing_martingale", exchange="binance", coins=1,
        bars=args.bars, sides=args.sides, seed=43, hsl="coin",
        hsl_red_threshold=0.02, hsl_ema_span_minutes=30.0,
        hsl_lookback_days=args.lookback_days,
    )
    config, candles, markets, btc, timestamps = fixture_inputs(recipe)
    # Long-held positions cross the lookback before shocks exercise
    # panic, cooldown and restart. This fixture has no network dependency.
    close = 100.0 + 0.1 * np.sin(np.arange(args.bars) / 37.0)
    close[args.bars * 3 // 4:] *= 0.85
    close[args.bars * 7 // 8:] *= 1.35
    candles[:, 0, 0] = close + 0.2
    candles[:, 0, 1] = close - 0.2
    candles[:, 0, 2] = close
    active = ("long", "short") if args.sides == "both" else (args.sides,)
    candidates = [
        {f"{side}_{key}": value for side in active for key, value in {
            "entry_initial_qty_pct": 0.3 + i * 0.001,
            "entry_initial_ema_dist": -0.005,
            "close_threshold_base_pct": 2.0,
            "close_retracement_base_pct": 2.0,
        }.items()}
        for i in range(args.candidates)
    ]
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="binance", batch_size=args.candidates,
        needed_metrics={"adg_strategy_eq", "backtest_completion_ratio"}, factual_hsl=True,
    )
    runner = proxy.fused_runner or proxy.runners[proxy.sides[0]]
    proxy.profile_enabled = True
    original_run = runner.run
    latest = {}
    def capture(*a, **kw):
        output = original_run(*a, **kw)
        latest.clear()
        latest.update({k: v.cpu().numpy().copy() for k, v in output.items()
                       if isinstance(v, torch.Tensor)})
        return output
    runner.run = capture
    records = []
    reference = None
    # Warm both libraries, then bracket two cache runs with cache-off runs.
    for label, enabled, warmup in (
        ("baseline_cold", False, True), ("cache_cold", True, True),
        ("baseline", False, False), ("cache", True, False),
        ("cache_repeat", True, False), ("baseline_repeat", False, False),
    ):
        runner.hsl_native_coin_cache_enabled = enabled
        started = time.perf_counter()
        rows = proxy.evaluate_results(candidates)
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = deepcopy(latest)
        else:
            assert reference.keys() == latest.keys()
            for key, value in latest.items():
                if key in {"alive", "fill_count", "hsl_triggers_long", "hsl_triggers_short",
                           "hsl_restart_retrigger_count"} or not np.issubdtype(value.dtype, np.floating):
                    np.testing.assert_array_equal(value, reference[key], err_msg=key)
                else:
                    np.testing.assert_allclose(value, reference[key], rtol=3e-5, atol=2e-6,
                                               equal_nan=True, err_msg=key)
        events = {key: float(np.sum(value)) for key, value in latest.items()
                  if key in {"fill_count", "hsl_triggers_long", "hsl_triggers_short",
                             "hsl_restart_retrigger_count"}}
        if sum(events.get(f"hsl_triggers_{side}", 0) for side in active) == 0:
            raise AssertionError("fixture must exercise actual HSL episodes")
        if len(rows) != args.candidates:
            raise AssertionError("incomplete candidate batch")
        record = dict(label=label, cache=enabled, warmup=warmup, seconds=elapsed,
                      backtests_per_second=args.candidates / elapsed, events=events,
                      profile=deepcopy(proxy.last_profile), checked_raw_outputs=len(latest))
        records.append(record)
        print(json.dumps(record), flush=True)
    baseline = statistics.median(r["seconds"] for r in records if not r["cache"] and not r["warmup"])
    cached = statistics.median(r["seconds"] for r in records if r["cache"] and not r["warmup"])
    report = dict(bars=args.bars, lookback_days=args.lookback_days, candidates=args.candidates, sides=args.sides,
                  device=str(runner.bars.device), runtime=runtime,
                  data_sha256=hashlib.sha256(candles.tobytes()+btc.tobytes()+timestamps.tobytes()).hexdigest(),
                  candidates_sha256=hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest(),
                  warm_speedup=baseline/cached, runs=records)
    if args.report:
        from pathlib import Path
        Path(args.report).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"warm_speedup": report["warm_speedup"], "raw_parity": "passed"}), flush=True)
    return report


if __name__ == "__main__":
    main()
