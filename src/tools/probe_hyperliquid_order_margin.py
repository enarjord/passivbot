#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
from tools.hyperliquid_probe_common import (
    add_live_mutation_confirmation_arg,
    add_probe_identity_args,
    create_hyperliquid_probe_session,
    extract_balance_summary,
    extract_position_summary,
    load_hyperliquid_wallet,
    mask_secret,
    require_live_mutation_confirmation,
    round_to_step,
)


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mutating Hyperliquid diagnostic. Places one tiny post-only order, waits for balance "
            "state to settle, then cancels it and reports before/create/cancel snapshots."
        )
    )
    add_probe_identity_args(parser)
    add_live_mutation_confirmation_arg(parser)
    parser.add_argument("--symbol", default="BTC/USDC:USDC", help="swap symbol to probe")
    parser.add_argument(
        "--side",
        default="buy",
        choices=("buy", "sell"),
        help="resting side; buy posts below market, sell posts above market",
    )
    parser.add_argument(
        "--distance-pct",
        type=float,
        default=0.25,
        help="fractional price distance from mid so order stays resting (default 0.25 = 25%%)",
    )
    parser.add_argument(
        "--notional-usdc",
        type=float,
        default=11.5,
        help="target notional in USDC (kept just above Hyperliquid minimum by default)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="sleep after create/cancel before refetching balances",
    )
    parser.add_argument(
        "--set-margin-mode",
        choices=("cross", "isolated"),
        default=None,
        help="optionally set margin mode on the symbol before placing the test order",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=2,
        help="leverage to use if --set-margin-mode is supplied",
    )
    parser.add_argument(
        "--dump-raw-info",
        action="store_true",
        help="include raw balance info and fetched position payloads before/create/cancel",
    )
    args = parser.parse_args()
    require_live_mutation_confirmation(
        parser, args, action_description="probe_hyperliquid_order_margin"
    )

    _, wallet_address, private_key = load_hyperliquid_wallet(
        args.user, api_keys_path=args.api_keys
    )
    session = create_hyperliquid_probe_session(wallet_address, private_key)

    order = None
    order_id = None
    try:
        markets = await session.load_markets()
        if args.symbol not in markets:
            raise KeyError(f"symbol {args.symbol!r} not found in Hyperliquid markets")
        market = markets[args.symbol]
        set_margin_mode_response = None
        if args.set_margin_mode is not None:
            set_margin_mode_response = await session.set_margin_mode(
                args.set_margin_mode,
                symbol=args.symbol,
                params={"leverage": int(args.leverage)},
            )
        ticker = await session.fetch_ticker(args.symbol)
        last_price = float(ticker.get("last") or ticker.get("bid") or ticker.get("ask"))
        if not math.isfinite(last_price) or last_price <= 0.0:
            raise ValueError(f"invalid market price for {args.symbol}: {last_price}")

        qty_step = float(market.get("precision", {}).get("amount") or 0.0)
        price_step = float(market.get("precision", {}).get("price") or 0.0)
        min_cost = float((market.get("limits", {}).get("cost", {}) or {}).get("min") or 10.0)
        target_notional = max(float(args.notional_usdc), min_cost * 1.02)

        if args.side == "buy":
            raw_price = last_price * (1.0 - abs(float(args.distance_pct)))
            price = round_to_step(raw_price, price_step, mode="down")
            amount = round_to_step(target_notional / max(price, 1e-12), qty_step, mode="up")
        else:
            raw_price = last_price * (1.0 + abs(float(args.distance_pct)))
            price = round_to_step(raw_price, price_step, mode="up")
            amount = round_to_step(target_notional / max(price, 1e-12), qty_step, mode="up")

        before = await session.fetch_balance()
        before_positions = await session.fetch_positions(symbols=[args.symbol])
        order = await session.create_order(
            args.symbol,
            "limit",
            args.side,
            amount,
            price,
            {"timeInForce": "Alo"},
        )
        order_id = order.get("id")
        await asyncio.sleep(float(args.settle_seconds))
        after_create = await session.fetch_balance()
        after_create_positions = await session.fetch_positions(symbols=[args.symbol])
        open_orders = await session.fetch_open_orders(symbol=args.symbol)

        if order_id is not None:
            await session.cancel_order(order_id, symbol=args.symbol)
        else:
            for candidate in open_orders:
                if (
                    str(candidate.get("symbol") or "") == args.symbol
                    and str(candidate.get("side") or "").lower() == args.side
                    and float(candidate.get("amount") or 0.0) == amount
                    and float(candidate.get("price") or 0.0) == price
                ):
                    order_id = candidate.get("id")
                    await session.cancel_order(order_id, symbol=args.symbol)
                    break
        await asyncio.sleep(float(args.settle_seconds))
        after_cancel = await session.fetch_balance()
        after_cancel_positions = await session.fetch_positions(symbols=[args.symbol])

        output = {
            "user": args.user,
            "wallet_address": mask_secret(wallet_address),
            "symbol": args.symbol,
            "side": args.side,
            "last_price": last_price,
            "order": {
                "id": order_id,
                "price": price,
                "amount": amount,
                "target_notional": target_notional,
                "qty_step": qty_step,
                "price_step": price_step,
                "min_cost": min_cost,
            },
            "set_margin_mode": {
                "requested": args.set_margin_mode,
                "leverage": int(args.leverage),
                "response": set_margin_mode_response,
            },
            "before": extract_balance_summary(before),
            "after_create": extract_balance_summary(after_create),
            "after_cancel": extract_balance_summary(after_cancel),
            "positions_before": extract_position_summary(before_positions),
            "positions_after_create": extract_position_summary(after_create_positions),
            "positions_after_cancel": extract_position_summary(after_cancel_positions),
            "open_orders_after_create": [
                {
                    "id": candidate.get("id"),
                    "symbol": candidate.get("symbol"),
                    "side": candidate.get("side"),
                    "amount": candidate.get("amount"),
                    "price": candidate.get("price"),
                    "status": candidate.get("status"),
                }
                for candidate in open_orders
                if str(candidate.get("id") or "") == str(order_id or "")
            ],
        }
        if args.dump_raw_info:
            output["raw_info"] = {
                "before": before.get("info"),
                "after_create": after_create.get("info"),
                "after_cancel": after_cancel.get("info"),
            }
        print(json.dumps(output, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        try:
            if order_id is not None:
                try:
                    await session.cancel_order(order_id, symbol=args.symbol)
                except Exception:
                    pass
            await session.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
