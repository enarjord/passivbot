#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from tools.hyperliquid_probe_common import (
    add_probe_identity_args,
    create_hyperliquid_probe_session,
    load_hyperliquid_wallet,
    mask_secret,
)


def _normalize_abstraction(raw) -> str:
    text = str(raw).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    return text or "unknown"


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Hyperliquid account abstraction probe. "
            "Queries the current userAbstraction mode and reports whether CCXT treats the account as unified."
        )
    )
    add_probe_identity_args(parser)
    args = parser.parse_args()

    user_info, wallet_address, private_key = load_hyperliquid_wallet(
        args.user, api_keys_path=args.api_keys
    )
    session = create_hyperliquid_probe_session(wallet_address, private_key)
    try:
        enabled, _ = await session.is_unified_enabled("fetchBalance", wallet_address, True, {})
        raw = await session.publicPostInfo({"type": "userAbstraction", "user": wallet_address})
        summary = {
            "user": args.user,
            "exchange": "hyperliquid",
            "wallet_address": mask_secret(wallet_address),
            "is_vault": bool(user_info.get("is_vault")),
            "raw_user_abstraction": raw,
            "abstraction": _normalize_abstraction(raw),
            "ccxt_is_unified_enabled": bool(enabled),
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    finally:
        await session.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
