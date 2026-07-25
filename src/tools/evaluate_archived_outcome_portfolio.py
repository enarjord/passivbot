#!/usr/bin/env python3
"""Evaluate settled full-contract outcome archives through one shared wallet."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from outcome.archive import OutcomeTradeArchive
from outcome.archive_replay import build_archived_ema_anchor_replay
from outcome.models import NormalizedOutcomeMarket, OutcomeVenue
from outcome.orchestrator import (
    InsufficientCapitalPolicy,
    run_outcome_portfolio_backtest,
)
from outcome.rust_runner import make_rust_ema_anchor_outcome_job


DEFAULT_MODES = ("accumulate_pairs", "inventory_aware", "yes_only")


def _rust_fee_formula(market: NormalizedOutcomeMarket, override: str) -> str:
    if override != "archived":
        return override
    metadata = market.fee_metadata
    if metadata.formula == "polymarket_probability_curve":
        exponent = float(metadata.parameters.get("exponent", math.nan))
        if exponent != 1.0:
            raise ValueError(
                "archived Polymarket fee exponent is not supported by the Rust "
                "probability_variance formula; pass --fee-formula explicitly only "
                "when intentionally evaluating another formula"
            )
        return "probability_variance"
    if metadata.formula == "venue_reported_zero":
        return "notional"
    raise ValueError(
        f"archived fee formula {metadata.formula!r} has no Rust translation; "
        "pass --fee-formula explicitly"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", required=True, choices=[item.value for item in OutcomeVenue])
    parser.add_argument("--market-id", required=True, action="append")
    parser.add_argument("--archive", type=Path, default=Path("caches/outcome_markets.sqlite"))
    parser.add_argument("--starting-collateral", type=float, default=1_000.0)
    parser.add_argument("--allocation-per-market", type=float, default=100.0)
    parser.add_argument("--ema-fast-seconds", type=float, default=30.0)
    parser.add_argument("--ema-slow-seconds", type=float, default=300.0)
    parser.add_argument("--ema-warmup-seconds", type=int, default=300)
    parser.add_argument("--quote-offset", type=float, default=0.005)
    parser.add_argument("--inventory-skew", type=float, default=0.002)
    parser.add_argument("--clip-qty", type=float, default=5.0)
    parser.add_argument("--max-total-inventory-qty", type=float, default=100.0)
    parser.add_argument("--max-abs-residual-qty", type=float, default=5.0)
    parser.add_argument("--min-locked-pair-edge", type=float, default=0.005)
    parser.add_argument(
        "--risk-reduction-only-seconds-before-close",
        type=int,
        default=300,
    )
    parser.add_argument("--entry-cutoff-seconds-before-close", type=int, default=60)
    parser.add_argument("--maker-rate", required=True, type=float)
    parser.add_argument("--taker-rate", required=True, type=float)
    parser.add_argument("--settlement-rate", required=True, type=float)
    parser.add_argument(
        "--qty-step",
        type=float,
        help="Explicit share quantity step when retained venue metadata does not report one",
    )
    parser.add_argument(
        "--fee-formula",
        choices=("archived", "notional", "probability_variance"),
        default="archived",
        help="Use the retained venue formula when representable, or an explicit override",
    )
    parser.add_argument(
        "--fee-incidence",
        choices=("every_fill", "inventory_reduction_only"),
        default="inventory_reduction_only",
    )
    parser.add_argument(
        "--execution-mode",
        action="append",
        choices=DEFAULT_MODES,
        help="Repeat to evaluate selected modes; defaults to all three",
    )
    parser.add_argument(
        "--insufficient-capital-policy",
        choices=[item.value for item in InsufficientCapitalPolicy],
        default=InsufficientCapitalPolicy.FAIL.value,
    )
    args = parser.parse_args()

    if len(set(args.market_id)) != len(args.market_id):
        parser.error("--market-id values must be unique")
    positive_values = {
        "starting collateral": args.starting_collateral,
        "allocation per market": args.allocation_per_market,
        "EMA fast span": args.ema_fast_seconds,
        "EMA slow span": args.ema_slow_seconds,
        "clip quantity": args.clip_qty,
        "maximum total inventory": args.max_total_inventory_qty,
        "maximum residual": args.max_abs_residual_qty,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_values.values()):
        parser.error("collateral, spans, quantities, and limits must be finite and positive")
    if args.ema_warmup_seconds < 0:
        parser.error("--ema-warmup-seconds must be non-negative")
    if (
        args.risk_reduction_only_seconds_before_close < 0
        or args.entry_cutoff_seconds_before_close < 0
    ):
        parser.error("pre-close windows must be non-negative")
    if any(
        not math.isfinite(rate)
        for rate in (args.maker_rate, args.taker_rate, args.settlement_rate)
    ):
        parser.error("fee rates must be finite")
    if args.qty_step is not None and (
        not math.isfinite(args.qty_step) or args.qty_step <= 0.0
    ):
        parser.error("--qty-step must be finite and positive")

    venue = OutcomeVenue(args.venue)
    modes = tuple(args.execution_mode or DEFAULT_MODES)
    archive = OutcomeTradeArchive(args.archive)
    reports = []
    fee_schedules = {}
    try:
        for mode in modes:
            replays = []
            for market_id in args.market_id:
                market_versions = archive.load_market_metadata(venue, market_id)
                if not market_versions:
                    raise ValueError(
                        f"outcome archive has no market metadata for {market_id}"
                    )
                archived_market = market_versions[0]
                payout_unit = archived_market.payout_unit
                fee_schedule = {
                    "maker_rate": args.maker_rate,
                    "taker_rate": args.taker_rate,
                    "formula": _rust_fee_formula(
                        archived_market,
                        args.fee_formula,
                    ),
                    "incidence": args.fee_incidence,
                    "settlement_rate": args.settlement_rate,
                }
                fee_schedules[market_id] = fee_schedule
                strategy_params = {
                    "ema_span_fast_seconds": args.ema_fast_seconds,
                    "ema_span_slow_seconds": args.ema_slow_seconds,
                    "ema_warmup_seconds": args.ema_warmup_seconds,
                    "quote_offset": args.quote_offset,
                    "inventory_skew": args.inventory_skew,
                    "clip_qty": args.clip_qty,
                    "max_total_inventory_qty": args.max_total_inventory_qty,
                    "max_abs_residual_qty": args.max_abs_residual_qty,
                    "min_locked_pair_edge": args.min_locked_pair_edge,
                    "estimated_fee_per_share": max(0.0, args.maker_rate)
                    * payout_unit,
                    "risk_reduction_only_ms_before_close": (
                        args.risk_reduction_only_seconds_before_close * 1_000
                    ),
                    "entry_cutoff_ms_before_close": (
                        args.entry_cutoff_seconds_before_close * 1_000
                    ),
                    "execution_mode": mode,
                }
                replays.append(
                    build_archived_ema_anchor_replay(
                        archive,
                        venue=venue,
                        market_id=market_id,
                        fee_schedule=fee_schedule,
                        requested_collateral=args.allocation_per_market,
                        strategy_params=strategy_params,
                        qty_step=args.qty_step,
                    )
                )
            portfolio = run_outcome_portfolio_backtest(
                [
                    make_rust_ema_anchor_outcome_job(replay.payload)
                    for replay in replays
                ],
                starting_collateral=args.starting_collateral,
                insufficient_capital_policy=InsufficientCapitalPolicy(
                    args.insufficient_capital_policy
                ),
            )
            reports.append(
                {
                    "execution_mode": mode,
                    "contracts": [
                        {
                            "market_id": replay.market.market_id,
                            "title": replay.market.title,
                            "actual_fill_records": replay.actual_fill_records,
                            "coverage": asdict(replay.coverage),
                            "settlement": asdict(replay.settlement),
                        }
                        for replay in replays
                    ],
                    "portfolio": asdict(portfolio),
                }
            )
    finally:
        archive.close()

    print(
        json.dumps(
            {
                "authenticated": False,
                "mutations_performed": False,
                "archive": str(args.archive),
                "venue": venue.value,
                "market_ids": args.market_id,
                "fee_schedules": fee_schedules,
                "starting_collateral": args.starting_collateral,
                "allocation_per_market": args.allocation_per_market,
                "results": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
