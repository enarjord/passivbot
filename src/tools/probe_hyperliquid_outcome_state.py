#!/usr/bin/env python3
"""Read-only HIP-4 account, inventory, order, fill, and top-of-book probe."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json

from outcome.adapters import hyperliquid
from outcome.hyperliquid_live import HyperliquidOutcomeLiveClient
from outcome.models import OutcomeSide
from tools.hyperliquid_probe_common import (
    add_public_probe_address_arg,
    create_hyperliquid_public_probe_session,
    mask_secret,
)


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_public_probe_address_arg(parser)
    parser.add_argument(
        "--underlying",
        default="BTC",
        help="active priceBinary market for the bounded top-of-book sample",
    )
    args = parser.parse_args()

    account_address = args.address
    session = create_hyperliquid_public_probe_session()
    try:
        meta = await session.publicPostInfo({"type": "outcomeMeta"})
        if not isinstance(meta, dict) or not isinstance(meta.get("outcomes"), list):
            raise ValueError("unexpected Hyperliquid outcomeMeta response")
        markets = []
        rejected_count = 0
        for raw_market in meta["outcomes"]:
            try:
                markets.append(hyperliquid.normalize_market(raw_market))
            except ValueError:
                rejected_count += 1
        if not markets:
            raise ValueError("no supported active HIP-4 priceBinary markets")
        selected = [
            market
            for market in markets
            if market.native_metadata["underlying"].casefold() == args.underlying.casefold()
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one active HIP-4 {args.underlying} priceBinary market, "
                f"got {len(selected)}"
            )
        client = HyperliquidOutcomeLiveClient(
            session,
            account_address=account_address,
            allow_mutations=False,
        )
        snapshot, yes_book, no_book = await asyncio.gather(
            client.fetch_account_snapshot((selected[0],)),
            client.fetch_book(selected[0], outcome=OutcomeSide.YES),
            client.fetch_book(selected[0], outcome=OutcomeSide.NO),
        )
        print(
            json.dumps(
                {
                    "wallet_address": mask_secret(account_address),
                    "mutations_enabled": client.allow_mutations,
                    "active_price_binary_markets": len(markets),
                    "unsupported_active_outcomes": rejected_count,
                    "selected_market": {
                        "market_id": selected[0].market_id,
                        "title": selected[0].title,
                        "quote_asset": selected[0].quote_asset,
                        "scheduled_event_time_ms": (
                            selected[0].lifecycle.scheduled_event_time_ms
                        ),
                    },
                    "collateral": asdict(snapshot.collateral)
                    | {
                        "unheld": snapshot.collateral.unheld,
                        "conservative_available": (
                            snapshot.collateral.conservative_available
                        ),
                    },
                    "nonzero_outcome_balances": [
                        asdict(balance)
                        for balance in snapshot.token_balances
                        if balance.total_qty > 0.0 or balance.held_qty > 0.0
                    ],
                    "open_outcome_orders": [
                        asdict(order) for order in snapshot.open_orders
                    ],
                    "recent_active_outcome_fills": [
                        asdict(fill) for fill in snapshot.recent_fills
                    ],
                    "unknown_outcome_coins": {
                        "balances": snapshot.unknown_outcome_balance_coins,
                        "orders": snapshot.unknown_outcome_order_coins,
                        "fills": snapshot.unknown_outcome_fill_coins,
                    },
                    "book_sample": {
                        "yes": {
                            "timestamp_ms": yes_book.timestamp_ms,
                            "best_bid": (
                                asdict(yes_book.bids[0]) if yes_book.bids else None
                            ),
                            "best_ask": (
                                asdict(yes_book.asks[0]) if yes_book.asks else None
                            ),
                        },
                        "no": {
                            "timestamp_ms": no_book.timestamp_ms,
                            "best_bid": (
                                asdict(no_book.bids[0]) if no_book.bids else None
                            ),
                            "best_ask": (
                                asdict(no_book.asks[0]) if no_book.asks else None
                            ),
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        await session.close()


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
