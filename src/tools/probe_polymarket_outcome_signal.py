#!/usr/bin/env python3
"""Collect a verified public Polymarket actual-fill signal window."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import aiohttp

from outcome.adapters import polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.capture import capture_outcome_public_session
from outcome.candles import trades_to_1s_candles_by_native_book


GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets/{market_id}"


async def _fetch_gamma_market(gamma_market_id: str) -> dict:
    timeout = aiohttp.ClientTimeout(total=20, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(GAMMA_MARKET_URL.format(market_id=gamma_market_id)) as response:
            response.raise_for_status()
            payload = await response.json()
    if not isinstance(payload, dict):
        raise ValueError("unexpected Polymarket Gamma market response")
    return payload


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gamma-market-id",
        required=True,
        help="Gamma API market ID for a non-negative-risk binary CLOB market",
    )
    parser.add_argument(
        "--quote-asset",
        required=True,
        help="authoritative collateral identity for this market era, such as USDC.e or pUSD",
    )
    parser.add_argument("--observations", type=int, default=15)
    parser.add_argument("--max-wait-seconds", type=float, default=120.0)
    parser.add_argument(
        "--archive",
        default="caches/outcome_markets.sqlite",
        help="SQLite archive for normalized public fills and verified coverage",
    )
    args = parser.parse_args()
    if args.observations < 2:
        parser.error("--observations must be at least 2")

    raw_market = await _fetch_gamma_market(args.gamma_market_id)
    market = polymarket.normalize_market(raw_market, quote_asset=args.quote_asset)
    archive = OutcomeTradeArchive(Path(args.archive))
    collector_session = f"polymarket-public-{uuid.uuid4().hex}"
    try:
        capture = await capture_outcome_public_session(
            market,
            archive=archive,
            collector_session=collector_session,
            min_observations=args.observations,
            max_wait_seconds=args.max_wait_seconds,
        )
        window = capture.signal_window
        execution_candles = trades_to_1s_candles_by_native_book(
            window.covered_trades,
            verified_coverage=(window.coverage,),
        )
        print(
            json.dumps(
                {
                    "authenticated": False,
                    "mutations_performed": False,
                    "market": {
                        "gamma_market_id": args.gamma_market_id,
                        "market_id": market.market_id,
                        "title": market.title,
                        "quote_asset": market.quote_asset,
                        "yes_asset_id": market.yes_asset.asset_id,
                        "no_asset_id": market.no_asset.asset_id,
                        "price_grid": asdict(market.price_grid),
                        "min_order_qty": market.min_order_qty,
                        "scheduled_event_time_ms": market.lifecycle.scheduled_event_time_ms,
                    },
                    "collector_session": collector_session,
                    "archive": str(archive.db_path),
                    "coverage": asdict(window.coverage),
                    "observed_trade_records": len(window.trades),
                    "covered_trade_records": len(window.covered_trades),
                    "books_inserted": capture.books_inserted,
                    "books_ignored_as_duplicates": (
                        capture.books_ignored_as_duplicates
                    ),
                    "price_grid_changes_inserted": (
                        capture.price_grid_changes_inserted
                    ),
                    "price_grid_changes_ignored_as_duplicates": (
                        capture.price_grid_changes_ignored_as_duplicates
                    ),
                    "signal": {
                        "candles": len(window.candles),
                        "trade_seconds": sum(
                            candle.volume > 0.0 for candle in window.candles
                        ),
                        "zero_volume_seconds": sum(
                            candle.volume == 0.0 for candle in window.candles
                        ),
                        "trade_count": sum(
                            candle.trade_count for candle in window.candles
                        ),
                        "volume": sum(candle.volume for candle in window.candles),
                        "first_close": window.candles[0].close,
                        "last_close": window.candles[-1].close,
                        "low": min(candle.low for candle in window.candles),
                        "high": max(candle.high for candle in window.candles),
                    },
                    "execution_candles_by_native_book": {
                        side.value: {
                            "candles": len(candles),
                            "trade_seconds": sum(
                                candle.volume > 0.0 for candle in candles
                            ),
                            "zero_volume_seconds": sum(
                                candle.volume == 0.0 for candle in candles
                            ),
                            "volume": sum(candle.volume for candle in candles),
                        }
                        for side, candles in execution_candles.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        archive.close()


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
