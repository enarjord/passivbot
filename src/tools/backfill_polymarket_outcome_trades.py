#!/usr/bin/env python3
"""Backfill verified Polymarket OrderFilled events from a complete Polygon block range."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from outcome.adapters import polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.historical import archive_historical_batch
from outcome.polygon_rpc import (
    POLYGON_PUBLIC_RPC_URL,
    PolygonJsonRpc,
    download_polymarket_order_filled_range,
)


GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets/{market_id}"


async def _fetch_gamma_market(gamma_market_id: str) -> dict:
    timeout = aiohttp.ClientTimeout(total=20, connect=10)
    headers = {"User-Agent": "passivbot-outcome-archive/1"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(
            GAMMA_MARKET_URL.format(market_id=gamma_market_id)
        ) as response:
            response.raise_for_status()
            payload = await response.json()
    if not isinstance(payload, dict):
        raise ValueError("unexpected Polymarket Gamma market response")
    return payload


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-market-id", required=True)
    parser.add_argument(
        "--quote-asset",
        required=True,
        help="authoritative collateral identity for this market era, such as USDC.e or pUSD",
    )
    parser.add_argument("--start-ms", required=True, type=int)
    parser.add_argument("--end-ms", required=True, type=int)
    parser.add_argument(
        "--archive",
        default="caches/outcome_markets.sqlite",
        help="local SQLite outcome archive",
    )
    parser.add_argument(
        "--rpc-url",
        default=POLYGON_PUBLIC_RPC_URL,
        help="Polygon JSON-RPC endpoint; credentials, if required, remain transport-local",
    )
    parser.add_argument("--max-block-span", type=int, default=2_000)
    parser.add_argument(
        "--confirmation-blocks",
        type=int,
        default=128,
        help="exclude this many latest Polygon blocks from verified coverage",
    )
    parser.add_argument("--collector-session")
    args = parser.parse_args()

    market = polymarket.normalize_market(
        await _fetch_gamma_market(args.gamma_market_id),
        quote_asset=args.quote_asset,
    )
    async with PolygonJsonRpc(args.rpc_url) as rpc:
        download = await download_polymarket_order_filled_range(
            market,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            rpc=rpc,
            max_block_span=args.max_block_span,
            confirmation_blocks=args.confirmation_blocks,
        )
    collector_session = args.collector_session or (
        f"polygon-{args.gamma_market_id}-{args.start_ms}-{args.end_ms}"
    )
    archive = OutcomeTradeArchive(Path(args.archive))
    try:
        inserted, ignored = archive_historical_batch(
            archive,
            download.batch,
            collector_session=collector_session,
        )
    finally:
        archive.close()

    print(
        json.dumps(
            {
                "authenticated": False,
                "mutations_performed": False,
                "local_archive_written": True,
                "market": {
                    "gamma_market_id": args.gamma_market_id,
                    "market_id": market.market_id,
                    "title": market.title,
                },
                "coverage": {
                    "start_ms": args.start_ms,
                    "end_ms": args.end_ms,
                    "from_block": download.from_block,
                    "to_block": download.to_block,
                    "confirmed_head_block": download.confirmed_head_block,
                },
                "decoded_standard_market_logs": download.decoded_log_count,
                "target_market_logs": download.market_log_count,
                "condition_resolution_logs": download.resolution_log_count,
                "settlements": len(download.batch.settlements),
                "economic_trades": len(download.batch.trades),
                "archive": {
                    "path": str(Path(args.archive)),
                    "inserted": inserted,
                    "ignored": ignored,
                    "collector_session": collector_session,
                    "source_cursor": download.batch.source_cursor,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
