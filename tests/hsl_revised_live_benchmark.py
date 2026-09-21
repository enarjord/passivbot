"""Offline snapshot-adapter timing and full decision digest; no live activation.

    PYTHONPATH=src:tests python tests/hsl_revised_live_benchmark.py --mode unified

This synthetic no-fill history is one workload, not the complete readiness gate.
Compare result digests before interpreting timing changes across implementations.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import resource
import time

from live.hsl_revised_candles import Sources
from live.hsl_revised_inputs import Candle, CandleTape
from live.hsl_revised_runtime import capture, evaluate
from rust_utils import verify_loaded_runtime_extension
from test_hsl_revised_runtime import bot, quotes, NOW, SYMBOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("coin", "pside", "unified"), default="coin")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--coins", type=int, default=10)
    options = parser.parse_args()
    if not 1 <= options.lookback_days <= 90 or options.coins < 1:
        parser.error("lookback must be 1..90 days and coins must be positive")
    artifact = verify_loaded_runtime_extension()
    value = bot(options.mode)
    value.config["live"]["pnls_max_lookback_days"] = options.lookback_days
    value.config["bot"]["long"]["risk"]["n_positions"] = options.coins
    positions, marks, sources = {}, {}, {}
    minutes = options.lookback_days * 1440
    for i in range(options.coins):
        symbol = f"ASSET{i}/USDT:USDT"
        positions[symbol] = value.positions[SYMBOL]
        marks[symbol] = replace(quotes()[SYMBOL], symbol=symbol)
        value.c_mults[symbol], value.qty_steps[symbol] = 1., .1
        candles = tuple(Candle(NOW-(minutes-j)*60_000, 1, 100., 110., 90., 100., NOW)
                        for j in range(minutes))
        sources[symbol] = Sources((CandleTape(candles, ()),), (), 0)
    value.positions = positions
    start = time.perf_counter()
    requests, unavailable = capture(value, marks, sources, symbols=list(positions),
        now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
    captured = time.perf_counter()
    decisions = evaluate(requests)
    finished = time.perf_counter()
    result = dict(decisions=[asdict(d) for d in decisions],
                  unavailable=[asdict(u) for u in unavailable])
    print(json.dumps(dict(fixture=vars(options), artifact=artifact,
        capture_seconds=captured-start, evaluate_seconds=finished-captured,
        total_seconds=finished-start, payload_bytes=sum(len(r.payload) for r in requests),
        max_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        result_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()),
        sort_keys=True))


if __name__ == "__main__":
    main()
