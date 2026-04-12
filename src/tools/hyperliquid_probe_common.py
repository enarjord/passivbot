from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

import ccxt.async_support as ccxt_async

from procedures import load_user_info


def mask_secret(value: str, *, prefix: int = 6, suffix: int = 4) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= prefix + suffix:
        return text
    return f"{text[:prefix]}...{text[-suffix:]}"


def round_to_step(value: float, step: float, *, mode: str) -> float:
    if step <= 0.0:
        return float(value)
    q = Decimal(str(value)) / Decimal(str(step))
    rounding = ROUND_CEILING if mode == "up" else ROUND_FLOOR
    rounded = q.quantize(Decimal("1"), rounding=rounding) * Decimal(str(step))
    return float(rounded)


def extract_balance_summary(balance: dict[str, Any]) -> dict[str, Any]:
    info = balance.get("info", {}) if isinstance(balance, dict) else {}
    margin_summary = info.get("marginSummary", {}) if isinstance(info, dict) else {}
    cross_margin_summary = info.get("crossMarginSummary", {}) if isinstance(info, dict) else {}
    asset_positions = info.get("assetPositions", []) if isinstance(info, dict) else []
    return {
        "top_level_keys": sorted(balance.keys()) if isinstance(balance, dict) else [],
        "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
        "margin_summary_keys": (
            sorted(margin_summary.keys()) if isinstance(margin_summary, dict) else []
        ),
        "cross_margin_summary_keys": (
            sorted(cross_margin_summary.keys())
            if isinstance(cross_margin_summary, dict)
            else []
        ),
        "asset_positions_count": len(asset_positions) if isinstance(asset_positions, list) else None,
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


def extract_position_summary(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "unrealizedPnl": position.get("unrealizedPnl"),
                "info": position.get("info"),
            }
        )
    return summarized


def add_probe_identity_args(parser: argparse.ArgumentParser, *, require_user: bool = True) -> None:
    parser.add_argument(
        "--user",
        required=require_user,
        help="Hyperliquid user in api-keys.json",
    )
    parser.add_argument(
        "--api-keys",
        default="api-keys.json",
        help="path to api-keys.json",
    )


def add_live_mutation_confirmation_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "required acknowledgement for probes that place/cancel live orders or flatten positions "
            "on the specified Hyperliquid wallet"
        ),
    )


def require_live_mutation_confirmation(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    action_description: str,
) -> None:
    if getattr(args, "yes", False):
        return
    parser.error(
        f"{action_description} touches a live Hyperliquid wallet; rerun with --yes after "
        "verifying --user and --symbol"
    )


def load_hyperliquid_wallet(user: str, *, api_keys_path: str) -> tuple[dict[str, Any], str, str]:
    user_info = load_user_info(user, api_keys_path=api_keys_path)
    exchange = str(user_info.get("exchange") or "").lower()
    if exchange != "hyperliquid":
        raise ValueError(f"user {user!r} is exchange={exchange!r}, expected 'hyperliquid'")
    wallet_address = str(user_info.get("wallet_address") or "")
    private_key = str(user_info.get("private_key") or "")
    if not wallet_address or not private_key:
        raise ValueError(f"user {user!r} is missing wallet_address/private_key")
    return user_info, wallet_address, private_key


def create_hyperliquid_probe_session(wallet_address: str, private_key: str):
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
    return session


def hyperliquid_probe_vault_params(user_info: dict[str, Any]) -> dict[str, Any]:
    if not bool(user_info.get("is_vault")):
        return {}
    wallet_address = str(user_info.get("wallet_address") or "")
    if not wallet_address:
        raise ValueError("vault user is missing wallet_address")
    return {"vaultAddress": wallet_address}
