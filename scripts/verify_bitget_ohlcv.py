#!/usr/bin/env python3
import argparse
import asyncio
import datetime as dt
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for path in (ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import ccxt.async_support as ccxt_async

from procedures import assert_correct_ccxt_version, load_user_info


ONE_MIN_MS = 60_000


def _parse_ts(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    ts = dt.datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def _tf_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * ONE_MIN_MS
    if tf.endswith("h"):
        return int(tf[:-1]) * 60 * ONE_MIN_MS
    raise ValueError(f"unsupported timeframe {tf}")


def _collapse_missing(missing: List[int], step_ms: int) -> List[Tuple[int, int]]:
    spans = []
    missing = sorted(missing)
    if not missing:
        return spans
    start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts == prev + step_ms:
            prev = ts
            continue
        spans.append((start, prev))
        start = ts
        prev = ts
    spans.append((start, prev))
    return spans


async def fetch_range(
    exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    tf: str,
    limit: int,
    overlap: int,
):
    step_ms = _tf_to_ms(tf)
    since = start_ms
    rows = []
    seen = set()
    max_pages = 10_000
    pages = 0
    if overlap > 0:
        since = max(0, since - overlap * step_ms)
    while since <= end_ms and pages < max_pages:
        res = await exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        pages += 1
        if not res:
            break
        res = sorted(res, key=lambda x: x[0])
        for row in res:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            rows.append(row)
        last_ts = int(res[-1][0])
        if last_ts <= since:
            break
        new_since = last_ts + step_ms
        if overlap > 0:
            new_since = max(last_ts - overlap * step_ms, since + step_ms)
        if new_since <= since:
            break
        since = new_since
        if len(res) < limit and last_ts >= end_ms:
            break
    rows = sorted(rows, key=lambda x: x[0])
    return rows


def report(rows: List[list], start_ms: int, end_ms: int, tf: str, max_spans: int):
    step_ms = _tf_to_ms(tf)
    expected = int((end_ms - start_ms) // step_ms) + 1
    actual_ts = [int(r[0]) for r in rows if start_ms <= int(r[0]) <= end_ms]
    actual_set = set(actual_ts)
    missing = [
        ts for ts in range(start_ms, end_ms + step_ms, step_ms) if ts not in actual_set
    ]
    spans = _collapse_missing(missing, step_ms)
    print(f"expected={expected} fetched={len(actual_set)} missing={len(missing)} spans={len(spans)}")
    if rows:
        first_dt = dt.datetime.fromtimestamp(rows[0][0] / 1000, tz=dt.timezone.utc)
        last_dt = dt.datetime.fromtimestamp(rows[-1][0] / 1000, tz=dt.timezone.utc)
        print(f"first_ts={first_dt.isoformat()} last_ts={last_dt.isoformat()}")
    for s, e in spans[:max_spans]:
        s_iso = dt.datetime.fromtimestamp(s / 1000, tz=dt.timezone.utc).isoformat()
        e_iso = dt.datetime.fromtimestamp(e / 1000, tz=dt.timezone.utc).isoformat()
        print(f"missing_span: {s_iso} -> {e_iso} ({int((e - s)//step_ms + 1)} candles)")


async def main():
    p = argparse.ArgumentParser(description="Verify Bitget OHLCV coverage for a range.")
    p.add_argument("--user", default=None, help="api-keys.json user (optional)")
    p.add_argument("--exchange", default="bitget")
    p.add_argument("--symbol", default="AVAX/USDT:USDT")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-spans", type=int, default=10)
    p.add_argument("--overlap", type=int, default=0)
    args = p.parse_args()

    assert_correct_ccxt_version(ccxt=ccxt_async)

    config = {"enableRateLimit": True}
    if args.user:
        user_info = load_user_info(args.user)
        for k, v in user_info.items():
            if k == "exchange":
                continue
            config[k] = v
        user_opts = user_info.get("options", {}) if isinstance(user_info, dict) else {}
    else:
        user_opts = {}

    exchange_class = getattr(ccxt_async, args.exchange)
    ex = exchange_class(config)
    ex.options.update(user_opts)
    ex.options.setdefault("defaultType", "swap")

    start_ms = _parse_ts(args.start)
    end_ms = _parse_ts(args.end)
    if start_ms > end_ms:
        raise ValueError("start must be <= end")

    try:
        rows = await fetch_range(
            ex,
            args.symbol,
            start_ms,
            end_ms,
            args.timeframe,
            args.limit,
            args.overlap,
        )
        report(rows, start_ms, end_ms, args.timeframe, args.max_spans)
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
