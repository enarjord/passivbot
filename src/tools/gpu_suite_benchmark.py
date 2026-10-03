"""Bounded, synthetic A/B benchmark for small GPU suite survivor batches.

Run with ``PYTHONPATH=src python -m tools.gpu_suite_benchmark``. No market
downloads, account credentials, or private configurations are used.
"""

import argparse
import json
import statistics
import time

import numpy as np

from optimization.gpu.runtime import synchronize
from tools.gpu_proxy_benchmark import _build_case, _require_mps_torch


def run_benchmark(*, bars, candidates, coins, repeats):
    proxy, population, *_, fixture_hash = _build_case(
        "tm-multicoin-overhead", candidates=candidates, dispatch_batch_size=512,
        single_bars=bars, multicoin_bars=bars, coins=coins, seed=7,
    )
    proxy.batch_size = 512
    scenarios = [
        [dict(candidate, long_n_positions=float(n)) for candidate in population]
        for n in range(4, 21, 2)
    ]
    combined = [candidate for scenario in scenarios for candidate in scenario]
    # Warm compilation before measuring either variant. Each variant uses the
    # same immutable candles, ordered candidates, dispatch cap and work limit.
    proxy.evaluate(combined[:1])
    samples = {"separate": [], "batched": []}
    reference = None
    for repeat in range(repeats):
        order = ("separate", "batched") if repeat % 2 == 0 else ("batched", "separate")
        for mode in order:
            synchronize()
            started = time.perf_counter()
            rows = (
                [row for scenario in scenarios for row in proxy.evaluate(scenario)]
                if mode == "separate" else proxy.evaluate(combined)
            )
            synchronize()
            elapsed = time.perf_counter() - started
            if reference is None:
                reference = rows
            else:
                # NaNs must match too; no tolerance hides a batching regression.
                np.testing.assert_equal(rows, reference)
            samples[mode].append(elapsed)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    return dict(
        fixture_sha256=fixture_hash, bars=bars, coins=coins,
        candidates_per_scenario=candidates, scenario_positions=list(range(4, 21, 2)),
        dispatch_cap=512, repeats=repeats, parity="exact",
        wall_seconds=samples, median_seconds=medians,
        speedup=medians["separate"] / medians["batched"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=12000)
    parser.add_argument("--candidates", type=int, default=103)
    parser.add_argument("--coins", type=int, default=26)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    for name, lower, upper in (
        ("bars", 120, 100000), ("candidates", 1, 512),
        ("coins", 2, 64), ("repeats", 1, 5),
    ):
        if not lower <= getattr(args, name) <= upper:
            parser.error(f"--{name} must be between {lower} and {upper}")
    _require_mps_torch(parser)
    print(json.dumps(run_benchmark(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
