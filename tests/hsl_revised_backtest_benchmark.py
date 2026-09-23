"""Offline native replay benchmark with a digest of the complete result.

Run from the repository root with the current rebuilt extension:
    PYTHONPATH=src python tests/hsl_revised_backtest_benchmark.py

Compare the digest across builds before interpreting elapsed-time differences.
This exercises the shared native adapter without live exchange access.
Use --detailed for standalone backtest results, and add --hsl-detailed-report
to include per-minute HSL samples. The default
measures the metrics-only path used by CPU optimization. --runs reports repeated
timings and their median.
"""
import argparse
import hashlib
import json
import time
import statistics

import numpy as np
import passivbot_rust

from rust_utils import verify_loaded_runtime_extension
from test_hsl_revised_backtest_config import payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=10080)
    parser.add_argument("--lookback-days", type=float, default=7)
    parser.add_argument("--mode", choices=("coin", "pside", "unified"), default="coin")
    parser.add_argument("--red-threshold", type=float, default=.99)
    parser.add_argument("--detailed", action="store_true", help="Include standalone backtest results and HSL transitions")
    parser.add_argument("--hsl-detailed-report", action="store_true", help="Include per-minute HSL samples (requires --detailed)")
    parser.add_argument("--runs", type=int, default=3)
    options = parser.parse_args()
    if options.hsl_detailed_report and not options.detailed:
        parser.error("--hsl-detailed-report requires --detailed")
    if options.runs < 1:
        parser.error("--runs must be positive")
    if options.minutes < 41:
        parser.error("--minutes must be at least 41")
    artifact = verify_loaded_runtime_extension()
    args = list(payload(options.mode))
    marks = np.concatenate([np.full(40, 100.), np.linspace(98., 80., options.minutes - 40)])
    args[0] = np.array([[[p * 1.002, p * .998, p, 1000.]] for p in marks])
    args[1] = np.full(options.minutes, 50000.)
    params = args[-1]
    params.update(last_valid_indices=[options.minutes - 1], metrics_only=not options.detailed,
                  hsl_detailed_report=options.hsl_detailed_report,
                  pnls_max_lookback_days=options.lookback_days)
    hsl = params["equity_hard_stop_loss"]
    policies = list(hsl["sides"])
    for pair in hsl["coins"].values():
        policies.extend(pair)
    if hsl.get("portfolio") is not None:
        policies.append(hsl["portfolio"])
    for policy in policies:
        policy["red_threshold"] = options.red_threshold
    timings = []
    result = None
    for _ in range(options.runs):
        result = None  # Release the previous output outside the timed native call.
        start = time.perf_counter()
        result = passivbot_rust.run_backtest(*args)
        timings.append(time.perf_counter() - start)
    elapsed = statistics.median(timings)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True,
        default=lambda value: value.tolist()).encode()).hexdigest()
    print(json.dumps(dict(fixture=vars(options), seconds=elapsed, run_seconds=timings, result_sha256=digest,
        artifact=artifact, summary=result[4]["revised"]["summary"]), sort_keys=True))


if __name__ == "__main__":
    main()
