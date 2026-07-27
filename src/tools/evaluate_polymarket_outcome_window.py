#!/usr/bin/env python3
"""Evaluate EMA-anchor modes on one verified archived Polymarket fill window."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

import aiohttp

from outcome.adapters import polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.backtest_input import build_trade_derived_ema_anchor_input
from outcome.candles import VerifiedCoverage
from outcome.evaluation import (
    evaluate_ema_anchor_outcome_modes,
    summarize_outcome_strategy_modes,
)
from outcome.rust_runner import normalized_market_to_rust_spec


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


def _proves_interval(
    coverage: list[VerifiedCoverage],
    start_ms: int,
    end_ms: int,
) -> bool:
    cursor = start_ms
    for interval in sorted(coverage, key=lambda item: item.start_ms):
        if interval.end_ms <= cursor:
            continue
        if interval.start_ms > cursor:
            return False
        cursor = max(cursor, interval.end_ms)
        if cursor >= end_ms:
            return True
    return False


def _require_fee_free_market(market) -> None:
    if market.fee_metadata.formula != "venue_reported_zero":
        raise ValueError(
            "Polymarket evaluation requires an explicitly fee-free market; "
            f"unsupported fee metadata formula {market.fee_metadata.formula!r}"
        )


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
        help="SQLite archive containing normalized fills and verified coverage",
    )
    parser.add_argument("--starting-collateral", type=float, default=1_000.0)
    parser.add_argument("--qty-step", type=float, default=0.01)
    parser.add_argument("--ema-fast-seconds", type=float, default=5.0)
    parser.add_argument("--ema-slow-seconds", type=float, default=20.0)
    parser.add_argument("--quote-offset", type=float, default=0.01)
    parser.add_argument("--inventory-skew", type=float, default=0.02)
    parser.add_argument("--clip-qty", type=float, default=5.0)
    parser.add_argument("--max-total-inventory-qty", type=float, default=100.0)
    parser.add_argument("--max-abs-residual-qty", type=float, default=25.0)
    parser.add_argument("--min-locked-pair-edge", type=float, default=0.01)
    parser.add_argument("--risk-reduction-only-ms-before-close", type=int, default=0)
    parser.add_argument("--entry-cutoff-ms-before-close", type=int, default=0)
    args = parser.parse_args()
    if args.end_ms <= args.start_ms:
        parser.error("--end-ms must be greater than --start-ms")
    if args.start_ms % 1_000 or args.end_ms % 1_000:
        parser.error("--start-ms and --end-ms must be second-aligned")

    market = polymarket.normalize_market(
        await _fetch_gamma_market(args.gamma_market_id),
        quote_asset=args.quote_asset,
    )
    if market.price_grid.kind != "fixed_step" or market.price_grid.fixed_step is None:
        raise ValueError("Polymarket evaluation requires a fixed price grid")
    if market.min_order_qty is None:
        raise ValueError("Polymarket evaluation requires a minimum order quantity")
    _require_fee_free_market(market)

    archive = OutcomeTradeArchive(Path(args.archive))
    try:
        trades = []
        for asset in (market.yes_asset, market.no_asset):
            coverage = archive.load_verified_coverage(
                market.venue,
                market.market_id,
                asset.asset_id,
                start_ms=args.start_ms,
                end_ms=args.end_ms,
            )
            if not _proves_interval(coverage, args.start_ms, args.end_ms):
                raise ValueError(
                    f"archive does not prove complete coverage for {asset.side.value}"
                )
            trades.extend(
                archive.load_trades(
                    market.venue,
                    market.market_id,
                    asset.asset_id,
                    start_ms=args.start_ms,
                    end_ms=args.end_ms,
                )
            )
    finally:
        archive.close()

    market_spec = normalized_market_to_rust_spec(market, qty_step=args.qty_step)
    # This tool evaluates only the archived sample. These are deliberately synthetic
    # boundaries and must not be reported as a full-contract backtest.
    market_spec.update(
        {
            "trading_opens_ms": args.start_ms,
            "order_entry_opens_ms": args.start_ms,
            "trading_closes_ms": args.end_ms,
            "scheduled_event_ms": args.end_ms,
        }
    )
    strategy_params = {
        "ema_span_fast_seconds": args.ema_fast_seconds,
        "ema_span_slow_seconds": args.ema_slow_seconds,
        "ema_warmup_seconds": int(args.ema_slow_seconds),
        "quote_offset": args.quote_offset,
        "inventory_skew": args.inventory_skew,
        "clip_qty": args.clip_qty,
        "max_total_inventory_qty": args.max_total_inventory_qty,
        "max_abs_residual_qty": args.max_abs_residual_qty,
        "min_locked_pair_edge": args.min_locked_pair_edge,
        "estimated_fee_per_share": 0.0,
        "risk_reduction_only_ms_before_close": args.risk_reduction_only_ms_before_close,
        "entry_cutoff_ms_before_close": args.entry_cutoff_ms_before_close,
        "execution_mode": "accumulate_pairs",
    }
    payload = build_trade_derived_ema_anchor_input(
        market_spec=market_spec,
        trades=trades,
        verified_coverage=(VerifiedCoverage(args.start_ms, args.end_ms),),
        fee_schedule={
            # Current strategy orders are maker-only. The selected market's Gamma fee
            # schedule is retained below for audit; maker charges/rebates are zero here.
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "formula": "notional",
            "incidence": "every_fill",
            "settlement_rate": 0.0,
        },
        starting_collateral=args.starting_collateral,
        strategy_params=strategy_params,
        settlement_time_ms=args.end_ms,
        yes_fraction=0.0,
    )
    evaluations = evaluate_ema_anchor_outcome_modes(payload)
    summaries = summarize_outcome_strategy_modes(evaluations)
    print(
        json.dumps(
            {
                "authenticated": False,
                "mutations_performed": False,
                "sample_is_full_contract_backtest": False,
                "market": {
                    "gamma_market_id": args.gamma_market_id,
                    "market_id": market.market_id,
                    "title": market.title,
                    "fee_metadata": asdict(market.fee_metadata),
                },
                "coverage": {"start_ms": args.start_ms, "end_ms": args.end_ms},
                "actual_fill_records": len(trades),
                "signal_candles": len(payload["signal_candles"]),
                "execution_candles": len(payload["execution_candles"]),
                "assumptions": {
                    "qty_step": args.qty_step,
                    "maker_rate": 0.0,
                    "fee_incidence": "every_fill",
                    "settlement_rate": 0.0,
                    "synthetic_settlement_at_window_end": True,
                },
                "strategy_params": strategy_params,
                "mode_summaries": [asdict(summary) for summary in summaries],
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
