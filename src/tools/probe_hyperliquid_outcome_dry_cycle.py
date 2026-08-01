#!/usr/bin/env python3
"""Read-only HIP-4 EMA planning/reconciliation probe from live public fills."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.hyperliquid_live import HyperliquidOutcomeLiveClient
from outcome.live_bot import (
    run_hip4_outcome_collected_cycle,
)
from tools.hyperliquid_probe_common import (
    add_public_probe_address_arg,
    create_hyperliquid_public_probe_session,
    mask_secret,
)


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_public_probe_address_arg(parser)
    parser.add_argument("--underlying", default="BTC")
    parser.add_argument("--observations", type=int, default=5)
    parser.add_argument("--max-wait-seconds", type=float, default=120.0)
    parser.add_argument(
        "--archive",
        default="caches/outcome_markets.sqlite",
        help="SQLite archive for normalized public fills and coverage",
    )
    args = parser.parse_args()
    if args.observations < 2:
        parser.error("--observations must be at least 2")

    account_address = args.address
    session = create_hyperliquid_public_probe_session()
    archive = OutcomeTradeArchive(Path(args.archive))
    try:
        meta = await session.publicPostInfo({"type": "outcomeMeta"})
        if not isinstance(meta, dict) or not isinstance(meta.get("outcomes"), list):
            raise ValueError("unexpected Hyperliquid outcomeMeta response")
        markets = []
        for raw_market in meta["outcomes"]:
            try:
                markets.append(hyperliquid.normalize_market(raw_market))
            except ValueError:
                continue
        selected = [
            market
            for market in markets
            if market.native_metadata["underlying"].casefold() == args.underlying.casefold()
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one active HIP-4 {args.underlying} market, got {len(selected)}"
            )
        market = selected[0]
        collector_session = f"hip4-dry-{uuid.uuid4().hex}"
        client = HyperliquidOutcomeLiveClient(
            session,
            account_address=account_address,
            allow_mutations=False,
        )
        strategy_params = {
            "ema_span_fast_seconds": 2.0,
            "ema_span_slow_seconds": float(args.observations),
            "ema_warmup_seconds": args.observations,
            "quote_offset": 0.001,
            "inventory_skew": 0.002,
            "clip_qty": 25.0,
            "max_total_inventory_qty": 50.0,
            "max_abs_residual_qty": 25.0,
            "min_locked_pair_edge": 0.001,
            "estimated_fee_per_share": 0.0,
            "risk_reduction_only_ms_before_close": 30 * 60 * 1_000,
            "entry_cutoff_ms_before_close": 60 * 1_000,
            "execution_mode": "accumulate_pairs",
        }
        collected = await run_hip4_outcome_collected_cycle(
            client,
            market,
            strategy_params,
            min_observations=args.observations,
            max_wait_seconds=args.max_wait_seconds,
            archive=archive,
            collector_session=collector_session,
        )
        cycle = collected.cycle
        window = collected.signal_window
        collateral = {
            **asdict(cycle.account.collateral),
            "conservative_available": cycle.account.collateral.conservative_available,
        }
        account_integrity = {
            "fee_rates": {
                **asdict(cycle.account.fee_rates),
                "conservative_maker_rate": (
                    cycle.account.fee_rates.conservative_maker_rate
                ),
                "conservative_taker_rate": (
                    cycle.account.fee_rates.conservative_taker_rate
                ),
            },
            "token_balances": [
                asdict(balance) for balance in cycle.account.token_balances
            ],
            "open_orders_count": len(cycle.account.open_orders),
            "recent_fills_count": len(cycle.account.recent_fills),
            "unknown_outcome_balance_coins": cycle.account.unknown_outcome_balance_coins,
            "unknown_outcome_order_coins": cycle.account.unknown_outcome_order_coins,
            "unknown_outcome_fill_coins": cycle.account.unknown_outcome_fill_coins,
            "settlements_count": len(cycle.account.settlements),
        }
        lifecycle = {
            "state": cycle.lifecycle.state.value,
            "observed_at_ms": cycle.lifecycle.observed_at_ms,
            "settlement": (
                asdict(cycle.lifecycle.settlement)
                if cycle.lifecycle.settlement is not None
                else None
            ),
        }
        if window is None:
            print(
                json.dumps(
                    {
                        "wallet_address": mask_secret(account_address),
                        "mutations_enabled": client.allow_mutations,
                        "market": {
                            "market_id": market.market_id,
                            "title": market.title,
                            "quote_asset": market.quote_asset,
                        },
                        "collector_session": collector_session,
                        "archive": str(archive.db_path),
                        "planning_available": cycle.planning_available,
                        "planning_unavailable_reason": (
                            cycle.planning_unavailable_reason.value
                            if cycle.planning_unavailable_reason is not None
                            else None
                        ),
                        "lifecycle": lifecycle,
                        "reconciliation": asdict(cycle.reconciliation),
                        "collateral": collateral,
                        "account_integrity": account_integrity,
                        "dry_run": cycle.is_dry_run,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "wallet_address": mask_secret(account_address),
                    "mutations_enabled": client.allow_mutations,
                    "market": {
                        "market_id": market.market_id,
                        "title": market.title,
                        "quote_asset": market.quote_asset,
                    },
                    "collector_session": collector_session,
                    "archive": str(archive.db_path),
                    "coverage": asdict(window.coverage),
                    "native_trade_records": len(window.trades),
                    "signal_candles": len(window.candles),
                    "planning_available": cycle.planning_available,
                    "planning_unavailable_reason": None,
                    "lifecycle": lifecycle,
                    "plan": asdict(cycle.plan) if cycle.plan is not None else None,
                    "reconciliation": asdict(cycle.reconciliation),
                    "collateral": collateral,
                    "account_integrity": account_integrity,
                    "dry_run": cycle.is_dry_run,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        archive.close()
        await session.close()


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
