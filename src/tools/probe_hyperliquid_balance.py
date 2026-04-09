#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
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


def _extract_balance_summary(balance: dict[str, Any]) -> dict[str, Any]:
    info = balance.get("info", {}) if isinstance(balance, dict) else {}
    margin_summary = info.get("marginSummary", {}) if isinstance(info, dict) else {}
    asset_positions = info.get("assetPositions", []) if isinstance(info, dict) else []
    return {
        "top_level_keys": sorted(balance.keys()) if isinstance(balance, dict) else [],
        "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        "margin_summary_keys": (
            sorted(margin_summary.keys()) if isinstance(margin_summary, dict) else []
        ),
        "account_value": margin_summary.get("accountValue"),
        "total_account_value": margin_summary.get("totalNtlPos"),
        "withdrawable": margin_summary.get("withdrawable"),
        "asset_positions_count": len(asset_positions) if isinstance(asset_positions, list) else None,
        "total_quote": (balance.get("total") or {}).get("USDC"),
        "free_quote": (balance.get("free") or {}).get("USDC"),
        "used_quote": (balance.get("used") or {}).get("USDC"),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Probe Hyperliquid fetch_balance for one user")
    parser.add_argument("--user", default="hyperliquid_01", help="user in api-keys.json")
    parser.add_argument(
        "--api-keys",
        default="api-keys.json",
        help="path to api-keys.json (default: repo root api-keys.json)",
    )
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="print the full raw fetch_balance payload after the summary",
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

    try:
        balance = await session.fetch_balance()
        summary = {
            "user": args.user,
            "exchange": "hyperliquid",
            "wallet_address": _mask(wallet_address),
            "is_vault": bool(user_info.get("is_vault")),
            "balance_summary": _extract_balance_summary(balance),
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        if args.dump_raw:
            print(json.dumps(balance, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        await session.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
