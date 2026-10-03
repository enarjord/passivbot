"""Offline formula study; not a trading indicator or profitability benchmark.

Run: python src/tools/research_entry_timing.py --span 60
Only synthetic log returns are used. No exchange, credentials, or market downloads.
"""

import argparse
from collections import deque
import json
import math
import random
from time import perf_counter


def directionality(returns, span):
    """Return final EW efficiency, EW RMS, sign, price-EMA gap, and window ratio.

    Return EMAs are seeded at zero after a preceding flat market. Price EMAs
    start at 1. Float spans are preserved; only the window reference is rounded.
    """
    alpha = 2.0 / (span + 1.0)
    fast_alpha = 2.0 / (max(1.0, span / 4.0) + 1.0)
    mean = absolute = square = sign = 0.0
    fast = slow = price = 1.0
    window = deque(maxlen=max(1, round(span)))
    for r in returns:
        mean += alpha * (r - mean)
        absolute += alpha * (abs(r) - absolute)
        square += alpha * (r * r - square)
        sign += alpha * ((1.0 if r > 0 else -1.0 if r < 0 else 0.0) - sign)
        price *= math.exp(r)
        fast += fast_alpha * (price - fast)
        slow += alpha * (price - slow)
        window.append(r)
    path_length = sum(map(abs, window))
    return {
        "ew_efficiency": abs(mean) / absolute if absolute else 0.0,
        "ew_rms": abs(mean) / math.sqrt(square) if square else 0.0,
        "ew_sign": abs(sign),
        "price_ema_gap": abs(math.log(fast / slow)),
        "window_efficiency": abs(sum(window)) / path_length if path_length else 0.0,
    }


def additive(base, scores, weights_minutes, floor, ceiling):
    return min(
        ceiling, max(floor, base + sum(s * w for s, w in zip(scores, weights_minutes)))
    )


def multiplicative(base, scores, weights, floor, ceiling):
    return min(
        ceiling, max(floor, base * (1.0 + sum(s * w for s, w in zip(scores, weights))))
    )


def composition_experiment(seed=42, count=10000):
    rng = random.Random(seed)
    max_error = 0.0
    for _ in range(count):
        base = rng.uniform(0.0001, 120.0)
        scores = [rng.random() for _ in range(4)]
        weights = [rng.uniform(-1.0, 4.0) for _ in scores]
        floor, ceiling = rng.uniform(0.0, 5.0), rng.uniform(5.0, 240.0)
        a = additive(base, scores, [base * w for w in weights], floor, ceiling)
        m = multiplicative(base, scores, weights, floor, ceiling)
        max_error = max(max_error, abs(a - m))
    return {
        "cases": count,
        "max_rescaled_error_minutes": max_error,
        "zero_base_additive_minutes": additive(0.0, [0.75], [20.0], 0.0, 60.0),
        "zero_base_multiplicative_minutes": multiplicative(
            0.0, [0.75], [4.0], 0.0, 60.0
        ),
    }


def synthetic_cases(span):
    warm = max(100, math.ceil(span * 10))
    rng = random.Random(42)
    return {
        "flat": [0.0] * warm,
        "steady_up": [0.001] * warm,
        "steady_down": [-0.001] * warm,
        "alternating": [0.001, -0.001] * warm,
        "same_net_move_uneven_steps": [0.0001, 0.0019] * warm,
        "nine_up_one_large_down": [0.001] * 9 + [-0.009],
        "shock_now": [0.0] * warm + [-0.10],
        "shock_then_flat_one_span": [0.0] * warm + [-0.10] + [0.0] * round(span),
        "shock_then_flat_five_spans": [0.0] * warm + [-0.10] + [0.0] * round(5 * span),
        "up_then_down_half_span": [0.001] * warm + [-0.001] * round(span / 2),
        "seeded_noise": [rng.gauss(0, 0.001) for _ in range(warm)],
    }


def benchmark(span, updates=1000000):
    # Isolate two-EMA arithmetic from data generation and Python list allocation.
    alpha = 2.0 / (span + 1.0)
    signed = absolute = 0.0
    start = perf_counter()
    for i in range(updates):
        r = 0.001 if i & 1 else -0.0005
        signed += alpha * (r - signed)
        absolute += alpha * (abs(r) - absolute)
    elapsed = perf_counter() - start
    return {
        "updates": updates,
        "seconds": elapsed,
        "nanoseconds_per_update_python": elapsed * 1e9 / updates,
        "checksum": abs(signed) / absolute,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span", type=float, default=60.0)
    args = parser.parse_args()
    if not math.isfinite(args.span) or args.span < 1.0:
        parser.error("span must be finite and >= 1")
    print(
        json.dumps(
            {
                "span": args.span,
                "cases": {
                    name: directionality(rs, args.span)
                    for name, rs in synthetic_cases(args.span).items()
                },
                "composition": composition_experiment(),
                "benchmark": benchmark(args.span),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
