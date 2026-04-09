#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

import ccxt.async_support as ccxt_async

from procedures import load_user_info


def _mask(value: str, *, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= prefix + suffix:
        return text
    return f"{text[:prefix]}...{text[-suffix:]}"


def _round_to_step(value: float, step: float, *, mode: str) -> float:
    if step <= 0.0:
        return float(value)
    q = Decimal(str(value)) / Decimal(str(step))
    rounding = ROUND_CEILING if mode == "up" else ROUND_FLOOR
    rounded = q.quantize(Decimal("1"), rounding=rounding) * Decimal(str(step))
    return float(rounded)


def _summary(balance: dict[str, Any]) -> dict[str, Any]:
    info = balance.get("info", {}) if isinstance(balance, dict) else {}
    margin_summary = info.get("marginSummary", {}) if isinstance(info, dict) else {}
    cross_margin_summary = info.get("crossMarginSummary", {}) if isinstance(info, dict) else {}
    return {
        "free_usdc": (balance.get("free") or {}).get("USDC"),
        "used_usdc": (balance.get("used") or {}).get("USDC"),
        "total_usdc": (balance.get("total") or {}).get("USDC"),
        "withdrawable": info.get("withdrawable"),
        "margin_account_value": margin_summary.get("accountValue"),
        "margin_total_margin_used": margin_summary.get("totalMarginUsed"),
        "margin_total_ntl_pos": margin_summary.get("totalNtlPos"),
        "margin_total_raw_usd": margin_summary.get("totalRawUsd"),
        "cross_account_value": cross_margin_summary.get("accountValue"),
        "cross_total_margin_used": cross_margin_summary.get("totalMarginUsed"),
        "cross_total_ntl_pos": cross_margin_summary.get("totalNtlPos"),
        "cross_total_raw_usd": cross_margin_summary.get("totalRawUsd"),
    }


def _position_summary(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for position in positions or []:
        summarized.append(
            {
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "contracts": position.get("contracts"),
                "entryPrice": position.get("entryPrice"),
                "marginMode": position.get("marginMode"),
                "isolated": position.get("isolated"),
                "leverage": position.get("leverage"),
                "info": position.get("info"),
            }
        )
    return summarized


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Place/cancel a tiny Hyperliquid post-only order and inspect balance fields"
    )
    parser.add_argument("--user", default="hyperliquid_01", help="user in api-keys.json")
    parser.add_argument("--api-keys", default="api-keys.json", help="path to api-keys.json")
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

    user_info = load_user_info(args.user, api_keys_path=args.api_keys)
    exchange = str(user_info.get("exchange") or "").lower()
    if exchange != "hyperliquid":
        raise ValueError(f"user {args.user!r} is exchange={exchange!r}, expected 'hyperliquid'")

    wallet_address = str(user_info.get("wallet_address") or "")
    private_key = str(user_info.get("private_key") or "")
    if not wallet_address or not private_key:
        raise ValueError(f"user {args.user!r} is missing wallet_address/private_key")

    session = ccxt_async.hyperliquid(
        {
            "walletAddress": wallet_address,
            "privateKey": private_key,
            "enableRateLimit": True,
            "timeout": 30_000,
        }
    )
    session.options["defaultType"] = "swap"
    session.options["fetchMarkets"] = {
        "types": ["swap", "hip3"],
        "hip3": {"dex": ["xyz"]},
    }

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
            price = _round_to_step(raw_price, price_step, mode="down")
            amount = _round_to_step(target_notional / max(price, 1e-12), qty_step, mode="up")
        else:
            raw_price = last_price * (1.0 + abs(float(args.distance_pct)))
            price = _round_to_step(raw_price, price_step, mode="up")
            amount = _round_to_step(target_notional / max(price, 1e-12), qty_step, mode="up")

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
            "wallet_address": _mask(wallet_address),
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
            "before": _summary(before),
            "after_create": _summary(after_create),
            "after_cancel": _summary(after_cancel),
            "positions_before": _position_summary(before_positions),
            "positions_after_create": _position_summary(after_create_positions),
            "positions_after_cancel": _position_summary(after_cancel_positions),
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
