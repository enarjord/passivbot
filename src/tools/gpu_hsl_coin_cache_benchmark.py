"""Offline matched native GPU coin-HSL cache benchmark with synthetic candles.

Run with ``python -m tools.gpu_hsl_coin_cache_benchmark``. The first run of
both variants may include shader compilation; only subsequent runs form the
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
    parser.add_argument("--bars", type=int, default=8192)
    parser.add_argument("--lookback-days", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--case", choices=("long-held", "fill-heavy"), default="long-held")
    parser.add_argument("--sides", choices=("long", "short", "both"), default="both")
    parser.add_argument("--check-cpu", action="store_true", help="Also compare each candidate with exact Rust")
    parser.add_argument("--report", help="Optional JSON report file")
    return parser


def active_sides(sides):
    return ("long", "short") if sides == "both" else (sides,)


def benchmark_inputs(args):
    recipe = argparse.Namespace(
        fixture="trailing_martingale", exchange="binance", coins=1,
        bars=args.bars, sides=args.sides, seed=43, hsl="coin",
        hsl_red_threshold=0.02, hsl_ema_span_minutes=30.0,
        hsl_lookback_days=args.lookback_days,
    )
    config, candles, markets, btc, timestamps = fixture_inputs(recipe)
    # Long-held positions cross the lookback before shocks exercise
    # panic, cooldown and restart. This fixture has no network dependency.
    close = (100.0 + 0.1 * np.sin(np.arange(args.bars) / 37.0) if args.case == "fill-heavy"
             else 100.09375 + 0.03 * np.sin(np.arange(args.bars) / 37.0))
    # A brief profit-taking excursion leaves a partial fill inside the rolling
    # window while inventory predates it. An empty retained tail already has a
    # cheap native shortcut and would not represent this cache's bottleneck.
    partial = args.bars * 2 // 3
    close[partial:partial+1] *= 1.05
    close[args.bars * 3 // 4:] *= 0.85
    close[args.bars * 7 // 8:] *= 1.35
    candles[:, 0, 0] = close + 0.2
    candles[:, 0, 1] = close - 0.2
    candles[:, 0, 2] = close
    active = active_sides(args.sides)
    knobs = dict(entry_initial_ema_dist=-0.005, close_qty_pct=0.1,
                 close_threshold_base_pct=0.01, close_retracement_base_pct=0.001)
    if args.case == "long-held":
        # Isolate one retained partial close; default exposure/volatility
        # weights could otherwise create continuous closing and reentry.
        knobs.update(entry_initial_ema_dist=0.0, entry_threshold_base_pct=0.5,
            entry_threshold_we_weight=0.0, entry_threshold_volatility_1h_weight=0.0,
            entry_threshold_volatility_1m_weight=0.0, close_threshold_we_weight=0.0,
            close_threshold_volatility_1h_weight=0.0, close_threshold_volatility_1m_weight=0.0,
            close_retracement_volatility_1h_weight=0.0, close_retracement_volatility_1m_weight=0.0)
        markets["COIN00"].update(qty_step=0.125, min_qty=0.125, price_step=0.125)
    candidates = [{f"{side}_{key}": value for side in active for key, value in {
        **knobs, "entry_initial_qty_pct": (0.3+i*0.001 if args.case == "fill-heavy"
                                         else 0.3+(i%8)*0.025),
    }.items()} for i in range(args.candidates)]
    return config, candles, markets, btc, timestamps, candidates


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (not 4096 <= args.bars <= 100_000 or not 1 <= args.candidates <= 64
            or not 1 <= args.lookback_days <= 60 or args.bars <= args.lookback_days * 1440):
        raise ValueError("benchmark requires 4096..100000 bars, 1..64 candidates and 1..60 lookback days shorter than history")
    import torch
    from optimization.gpu.service import MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension

    runtime = verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps, candidates = benchmark_inputs(args)
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="binance", batch_size=args.candidates,
        needed_metrics={"adg_strategy_eq", "backtest_completion_ratio"}, factual_hsl=True,
    )
    cpu = []
    if args.check_cpu:
        from backtest import build_backtest_payload, execute_backtest
        from config.metrics import resolve_metric_value
        for i in range(args.candidates):
            reference_config = deepcopy(config)
            from optimization.gpu.model import TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
            paths = dict(TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
            for name, value in candidates[i].items():
                side, key = name.split("_", 1)
                target = reference_config["bot"][side]["strategy"]["trailing_martingale"]
                path = paths[key]
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
            reference_started = time.perf_counter()
            payload = build_backtest_payload(candles, markets, reference_config,
                "binance", btc, timestamps, metrics_only=True, skip_btc_analysis=True)
            _, _, analysis = execute_backtest(payload, reference_config)
            print(json.dumps(dict(phase="cpu_reference", candidate=i,
                                  seconds=time.perf_counter()-reference_started)), flush=True)
            cpu.append({key: resolve_metric_value(analysis, key)
                        for key in ("adg_strategy_eq", "backtest_completion_ratio")})
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
        if cpu:
            for row, reference_metrics in zip(rows, cpu):
                for key, expected in reference_metrics.items():
                    np.testing.assert_allclose(row.metrics[key], expected,
                        rtol=1e-4, atol=1e-7 if key == "adg_strategy_eq" else 1e-6,
                        err_msg=f"CPU {key}")
        events = {key: float(np.sum(value)) for key, value in latest.items()
                  if key in {"fill_count", "hsl_triggers_long", "hsl_triggers_short",
                             "hsl_restart_retrigger_count"}}
        if sum(events.get(f"hsl_triggers_{side}", 0) for side in active_sides(args.sides)) == 0:
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
    report = dict(case=args.case, bars=args.bars, lookback_days=args.lookback_days, candidates=args.candidates, sides=args.sides,
                  device=str(runner.bars.device), runtime=runtime,
                  data_sha256=hashlib.sha256(candles.tobytes()+btc.tobytes()+timestamps.tobytes()).hexdigest(),
                  candidates_sha256=hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest(),
                  warm_speedup=baseline/cached, cpu_metric_checks=len(cpu)*2*len(records), runs=records)
    if args.report:
        from pathlib import Path
        Path(args.report).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"warm_speedup": report["warm_speedup"], "raw_parity": "passed"}), flush=True)
    return report


if __name__ == "__main__":
    main()
