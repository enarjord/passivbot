#!/usr/bin/env python3
"""Evaluate EMA-anchor modes on one verified archived HIP-4 fill window."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path

import aiohttp

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.archive_replay import (
    consolidated_archived_market,
    load_verified_trade_window,
)
from outcome.backtest_input import build_trade_derived_ema_anchor_input
from outcome.evaluation import (
    ema_warmup_observations,
    evaluate_ema_anchor_outcome_modes,
    summarize_outcome_strategy_modes,
)
from outcome.models import NormalizedOutcomeMarket, OutcomeVenue
from outcome.rust_runner import normalized_market_to_rust_spec


INFO_URL = "https://api.hyperliquid.xyz/info"


async def _post_info(session: aiohttp.ClientSession, payload: dict) -> object:
    async with session.post(INFO_URL, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def _fetch_market(underlying: str):
    timeout = aiohttp.ClientTimeout(total=20, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        meta = await _post_info(session, {"type": "outcomeMeta"})
    if not isinstance(meta, dict) or not isinstance(meta.get("outcomes"), list):
        raise ValueError("unexpected Hyperliquid outcomeMeta response")
    markets = []
    for raw_market in meta["outcomes"]:
        try:
            market = hyperliquid.normalize_market(raw_market)
        except ValueError:
            continue
        if market.native_metadata["underlying"].casefold() == underlying.casefold():
            markets.append(market)
    if len(markets) != 1:
        raise ValueError(f"expected one active HIP-4 {underlying} market, got {len(markets)}")
    return markets[0]


def _window_market_spec(
    market: NormalizedOutcomeMarket,
    *,
    start_ms: int,
    end_ms: int,
    qty_step: float,
    min_order_qty: float,
    min_order_notional: float,
) -> dict:
    market_spec = normalized_market_to_rust_spec(
        market,
        qty_step=qty_step,
        min_order_qty=min_order_qty,
        min_order_notional=min_order_notional,
    )
    # This evaluator covers only the requested sample, so lifecycle gates and
    # inventory-time metrics must use the same synthetic window boundaries.
    market_spec.update(
        {
            "trading_opens_ms": start_ms,
            "order_entry_opens_ms": start_ms,
            "trading_closes_ms": end_ms,
            "scheduled_event_ms": end_ms,
        }
    )
    return market_spec


def _add_window_phase_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--risk-reduction-only-ms-before-close",
        type=int,
        default=0,
        help="Synthetic-window risk-reduction phase; defaults to disabled",
    )
    parser.add_argument(
        "--entry-cutoff-ms-before-close",
        type=int,
        default=0,
        help="Synthetic-window entry cutoff; defaults to disabled",
    )


def _add_constraint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--qty-step",
        required=True,
        type=float,
        help="Explicit authoritative quantity step or clearly labeled experiment assumption",
    )
    parser.add_argument(
        "--min-order-qty",
        required=True,
        type=float,
        help="Explicit authoritative minimum quantity or clearly labeled experiment assumption",
    )
    parser.add_argument(
        "--min-order-notional",
        required=True,
        type=float,
        help=(
            "Explicit authoritative minimum quote notional or clearly labeled "
            "experiment assumption"
        ),
    )


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    market_selector = parser.add_mutually_exclusive_group(required=True)
    market_selector.add_argument(
        "--market-id",
        help="Load retained HIP-4 metadata for this archived market ID",
    )
    market_selector.add_argument(
        "--underlying",
        help="Discover the one currently active HIP-4 contract for this underlying",
    )
    parser.add_argument("--start-ms", required=True, type=int)
    parser.add_argument("--end-ms", required=True, type=int)
    parser.add_argument("--archive", default="caches/outcome_markets.sqlite")
    parser.add_argument("--starting-collateral", type=float, default=1_000.0)
    parser.add_argument("--ema-fast-seconds", type=float, default=5.0)
    parser.add_argument("--ema-slow-seconds", type=float, default=30.0)
    parser.add_argument("--quote-offset", type=float, default=0.001)
    parser.add_argument("--inventory-skew", type=float, default=0.002)
    parser.add_argument("--clip-qty", type=float, default=25.0)
    parser.add_argument("--max-total-inventory-qty", type=float, default=500.0)
    parser.add_argument("--max-abs-residual-qty", type=float, default=50.0)
    parser.add_argument("--min-locked-pair-edge", type=float, default=0.001)
    _add_constraint_arguments(parser)
    _add_window_phase_arguments(parser)
    parser.add_argument(
        "--maker-rate",
        required=True,
        type=float,
        help="Explicit account/venue maker rate used for this replay",
    )
    parser.add_argument(
        "--taker-rate",
        required=True,
        type=float,
        help="Explicit account/venue taker rate used for this replay",
    )
    parser.add_argument(
        "--settlement-rate",
        required=True,
        type=float,
        help="Explicit payout-notional settlement rate; pass 0 only as a stated assumption",
    )
    parser.add_argument(
        "--fee-incidence",
        required=True,
        choices=("every_fill", "inventory_reduction_only"),
        help="Explicit incidence contract for the supplied HIP-4 fill rates",
    )
    args = parser.parse_args()
    if args.end_ms <= args.start_ms:
        parser.error("--end-ms must be greater than --start-ms")
    if args.start_ms % 1_000 or args.end_ms % 1_000:
        parser.error("--start-ms and --end-ms must be second-aligned")
    if (
        args.risk_reduction_only_ms_before_close < 0
        or args.entry_cutoff_ms_before_close < 0
    ):
        parser.error("synthetic-window close phases must be non-negative")
    if any(
        not math.isfinite(rate)
        for rate in (args.maker_rate, args.taker_rate, args.settlement_rate)
    ):
        parser.error("fee rates must be finite")
    if (
        not math.isfinite(args.qty_step)
        or args.qty_step <= 0.0
        or not math.isfinite(args.min_order_qty)
        or args.min_order_qty <= 0.0
        or not math.isfinite(args.min_order_notional)
        or args.min_order_notional < 0.0
    ):
        parser.error(
            "quantity constraints must be finite and positive, and minimum "
            "notional must be finite and non-negative"
        )

    archive = OutcomeTradeArchive(Path(args.archive))
    try:
        if args.market_id is not None:
            market_versions = archive.load_market_metadata(
                OutcomeVenue.HYPERLIQUID,
                args.market_id,
            )
            if not market_versions:
                raise ValueError(
                    f"outcome archive has no HIP-4 market metadata for {args.market_id}"
                )
            market = consolidated_archived_market(market_versions)
        else:
            market = await _fetch_market(args.underlying)
        trades, verified_coverage = load_verified_trade_window(
            archive,
            market,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
        )
    finally:
        archive.close()

    strategy_params = {
        "ema_span_fast_seconds": args.ema_fast_seconds,
        "ema_span_slow_seconds": args.ema_slow_seconds,
        "ema_warmup_seconds": ema_warmup_observations(args.ema_slow_seconds),
        "quote_offset": args.quote_offset,
        "inventory_skew": args.inventory_skew,
        "clip_qty": args.clip_qty,
        "max_total_inventory_qty": args.max_total_inventory_qty,
        "max_abs_residual_qty": args.max_abs_residual_qty,
        "min_locked_pair_edge": args.min_locked_pair_edge,
        "estimated_fee_per_share": max(0.0, args.maker_rate) * market.payout_unit,
        "risk_reduction_only_ms_before_close": (
            args.risk_reduction_only_ms_before_close
        ),
        "entry_cutoff_ms_before_close": args.entry_cutoff_ms_before_close,
        "execution_mode": "accumulate_pairs",
    }
    payload = build_trade_derived_ema_anchor_input(
        market_spec=_window_market_spec(
            market,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            qty_step=args.qty_step,
            min_order_qty=args.min_order_qty,
            min_order_notional=args.min_order_notional,
        ),
        trades=trades,
        verified_coverage=(verified_coverage,),
        fee_schedule={
            "maker_rate": args.maker_rate,
            "taker_rate": args.taker_rate,
            "formula": "notional",
            "incidence": args.fee_incidence,
            "settlement_rate": args.settlement_rate,
        },
        starting_collateral=args.starting_collateral,
        strategy_params=strategy_params,
        settlement_time_ms=args.end_ms,
        yes_fraction=0.0,
        candle_start_ms=args.start_ms,
    )
    summaries = summarize_outcome_strategy_modes(
        evaluate_ema_anchor_outcome_modes(payload)
    )
    print(
        json.dumps(
            {
                "authenticated": False,
                "mutations_performed": False,
                "sample_is_full_contract_backtest": False,
                "market": {
                    "market_id": market.market_id,
                    "title": market.title,
                    "quote_asset": market.quote_asset,
                },
                "coverage": {"start_ms": args.start_ms, "end_ms": args.end_ms},
                "actual_fill_records": len(trades),
                "signal_candles": len(payload["signal_candles"]),
                "execution_candles": len(payload["execution_candles"]),
                "assumptions": {
                    "maker_rate": args.maker_rate,
                    "taker_rate": args.taker_rate,
                    "fee_incidence": args.fee_incidence,
                    "settlement_rate": args.settlement_rate,
                    "qty_step": args.qty_step,
                    "min_order_qty": args.min_order_qty,
                    "min_order_notional": args.min_order_notional,
                    "settlement_scenarios": [0.0, 1.0],
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
