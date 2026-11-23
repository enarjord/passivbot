"""Benchmark overlapping live candle reads on a deterministic local fixture.

Run with PYTHONPATH pointing at the checkout's src directory. This measures
process CPU for the candle workload without using an exchange. Each cycle
starts after that cycle's tail corrections, so timing covers the reads and
EMA calculation. The output digest includes the coverage and EMA results.
"""

import argparse
import asyncio
import cProfile
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

import candlestick_manager
from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS


DAY_MS = 1440 * ONE_MIN_MS
HOUR_MS = 60 * ONE_MIN_MS
START_DATE = date(2025, 1, 1)
DAYS = 7
START_MS = 1_735_689_600_000


def example_rows(day, step, symbol_number):
    minute = np.arange(DAY_MS // step)
    rows = np.empty(minute.size, dtype=CANDLE_DTYPE)
    rows["ts"] = START_MS + day * DAY_MS + minute * step
    close = 10 + symbol_number + day * 0.1 + minute * 0.001
    rows["o"] = rows["c"] = close
    rows["h"] = close + 0.1
    rows["l"] = close - 0.1
    rows["bv"] = 1 + minute % 40
    return rows


def create_fixture(manager, symbols, sparse):
    for number, symbol in enumerate(symbols):
        known_gaps = []
        for day in range(DAYS):
            key = (START_DATE + timedelta(days=day)).isoformat()
            rows = example_rows(day, ONE_MIN_MS, number)
            if sparse:
                removed = np.arange(rows.size) % 41 == 13 + number % 7
                known_gaps.extend(
                    {
                        "start_ts": int(ts),
                        "end_ts": int(ts),
                        "reason": "no_trades",
                        "retry_count": 0,
                        "added_at": int(ts),
                        "last_retry_at": int(ts),
                    }
                    for ts in rows["ts"][removed]
                )
                rows = rows[~removed]
            manager._save_shard(symbol, key, rows, timeframe="1m")
            manager._save_shard(
                symbol, key, example_rows(day, HOUR_MS, number), timeframe="1h"
            )
        if known_gaps:
            manager._save_known_gaps_enhanced(symbol, known_gaps)


def update_tail(manager, symbol, cycle):
    key = (START_DATE + timedelta(days=DAYS - 1)).isoformat()
    path = Path(manager._shard_path(symbol, key, timeframe="1m"))
    with path.open("rb") as stream:
        rows = np.load(stream, allow_pickle=False)
    rows["bv"][-45] += cycle + 1
    manager._save_shard(symbol, key, rows, timeframe="1m")


async def exercise(manager, symbols, cycle, now_ms):
    close_spans = (24.5, 520.25, 1270.75 + (cycle % 3) * 90)
    results = []
    for symbol in symbols:
        health = manager.get_completed_candle_health(
            symbol,
            {"1m": 1584 + (cycle % 3) * 720, "1h": 42},
            now_ms=now_ms,
        )
        minute = await manager.get_latest_ema_metric_spans(
            symbol,
            {"close": close_spans, "log_range": (400.5,), "qv": (70.25,)},
            timeframe="1m",
            allow_remote_fetch=False,
        )
        hourly = await manager.get_latest_ema_metric_spans(
            symbol,
            {"log_range": (48.5,)},
            timeframe="1h",
            allow_remote_fetch=False,
        )
        results.append((health, minute, hourly))
    return results


async def run(args):
    first_observation = START_MS + DAYS * DAY_MS - 20 * ONE_MIN_MS
    symbols = [f"TEST{i:02}/USDT:USDT" for i in range(args.symbols)]
    with tempfile.TemporaryDirectory() as directory:
        now_ms = first_observation

        def open_manager():
            manager = CandlestickManager(
                exchange_name="fixture", cache_dir=directory, archive_enabled=False
            )
            manager._now_ms_callback = lambda: now_ms
            return manager

        writer = open_manager()
        try:
            create_fixture(writer, symbols, args.scenario == "sparse")
        finally:
            writer.close()
        manager = open_manager()
        profile = cProfile.Profile() if args.profile else None
        cpu, wall, decodes = [], [], []
        result_hash = hashlib.sha256()
        decode_count = 0
        original_load = candlestick_manager.load_candle_array

        def counted_load(*args, **kwargs):
            nonlocal decode_count
            decode_count += 1
            return original_load(*args, **kwargs)

        candlestick_manager.load_candle_array = counted_load
        try:
            for cycle in range(args.cycles + 1):
                now_ms = first_observation + cycle * ONE_MIN_MS
                if cycle:
                    # Rotate the subset whose existing tail shard is replaced.
                    for number in range(cycle % 4, len(symbols), 4):
                        update_tail(manager, symbols[number], cycle)
                if profile and cycle:
                    profile.enable()
                decode_count = 0
                start_wall, start_cpu = time.perf_counter(), time.process_time()
                results = await exercise(manager, symbols, cycle, now_ms)
                spent_cpu, spent_wall = (
                    time.process_time() - start_cpu,
                    time.perf_counter() - start_wall,
                )
                if profile and cycle:
                    profile.disable()
                if cycle:
                    cpu.append(spent_cpu)
                    wall.append(spent_wall)
                    decodes.append(decode_count)
                    result_hash.update(json.dumps(results, sort_keys=True).encode())
                else:
                    cold_cpu, cold_decodes = spent_cpu, decode_count
            if profile:
                profile.dump_stats(args.profile)
        finally:
            candlestick_manager.load_candle_array = original_load
            manager.close()
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "symbols": args.symbols,
                "cycles": args.cycles,
                "median_cpu_ms": statistics.median(cpu) * 1000,
                "median_wall_ms": statistics.median(wall) * 1000,
                "cpu_ms": [round(value * 1000, 3) for value in cpu],
                "cold_cpu_ms": round(cold_cpu * 1000, 3),
                "cold_decodes": cold_decodes,
                "decodes": decodes,
                "output_sha256": result_hash.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("dense", "sparse"))
    parser.add_argument("--symbols", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=9)
    parser.add_argument("--profile")
    options = parser.parse_args()
    if options.symbols < 1 or not 1 <= options.cycles < 20:
        parser.error("use at least one symbol and between 1 and 19 cycles")
    asyncio.run(run(options))
