#!/usr/bin/env python3
"""
Fetch and print balance using ccxt for a given user.

Usage:
  python src/tools/fetch_balance.py --user USER

This script reads api keys from api-keys.json (by default from the repository root).
The expected shape is flexible; it will look for the specified user as a top-level key
or inside a "users" mapping. The user entry should include the exchange id (e.g. "binance")
and credentials (apiKey/key and secret). Example:

{
  "tester": {
    "exchange": "binance",
    "apiKey": "...",
    "secret": "..."
  }
}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import ccxt


def load_api_keys(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"api-keys file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_user_info(api_keys: Dict[str, Any], user: str) -> Dict[str, Any]:
    # Common shapes: { "user": {...} } or { "users": { "user": {...} } }
    if user in api_keys and isinstance(api_keys[user], dict):
        return api_keys[user]
    if isinstance(api_keys, dict) and "users" in api_keys and user in api_keys["users"]:
        return api_keys["users"][user]
    raise KeyError(f"user '{user}' not found in api-keys.json")


def build_exchange(user_info: Dict[str, Any]) -> ccxt.Exchange:
    # Accept multiple possible key names for convenience
    exchange_id = (
        user_info.get("exchange") or user_info.get("exchange_id") or user_info.get("exchangeId")
    )
    if not exchange_id:
        raise KeyError("missing 'exchange' in user info")

    # ccxt exposes exchanges as attributes on the ccxt module
    exchange_cls = getattr(ccxt, exchange_id, None) or getattr(ccxt, exchange_id.lower(), None)
    if exchange_cls is None:
        raise Exception(f"exchange '{exchange_id}' not found in ccxt")

    # Prefer perpetual futures (swap) when supported
    try:
        if not hasattr(exchange_cls, "options") or not isinstance(exchange_cls.options, dict):
            exchange_cls.options = {"defaultType": "swap"}
        else:
            exchange_cls.options["defaultType"] = "swap"
    except Exception:
        # best-effort: ignore failures to set class-level options
        pass

    api_key = user_info.get("apiKey") or user_info.get("key") or user_info.get("apikey")
    secret = user_info.get("secret") or user_info.get("apiSecret") or user_info.get("apisecret")
    password = user_info.get("password") or user_info.get("pwd") or user_info.get("passphrase")

    params = {"enableRateLimit": True}
    # Allow passing extra ccxt params through user_info if desired
    extra = user_info.get("ccxt", {})
    if isinstance(extra, dict):
        params.update(extra)

    # Build kwargs for the ccxt exchange constructor, normalizing common key names
    exchange_kwargs = dict(params)  # start with params (e.g. enableRateLimit, etc.)
    if api_key:
        exchange_kwargs["apiKey"] = api_key
    if secret:
        exchange_kwargs["secret"] = secret
    if password:
        exchange_kwargs["password"] = password

    # Include other useful fields from user_info (e.g., wallet_address, private_key),
    # but avoid copying control fields or credential aliases twice.
    for k, v in user_info.items():
        if k in ("exchange", "exchange_id", "exchangeId", "ccxt"):
            continue
        if k in (
            "key",
            "apiKey",
            "apikey",
            "secret",
            "apiSecret",
            "apisecret",
            "password",
            "pwd",
            "passphrase",
        ):
            continue
        # don't overwrite already set normalized keys
        if k not in exchange_kwargs:
            exchange_kwargs[k] = v

    exchange = exchange_cls(exchange_kwargs)
    return exchange


def pretty_print_balance(bal: Dict[str, Any]) -> None:
    # Print JSON that is stable and readable
    print(json.dumps(bal, indent=2, sort_keys=True, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and print exchange balance for a user")
    parser.add_argument("--user", required=True, help="user key in api-keys.json")
    parser.add_argument(
        "--api-keys",
        default="api-keys.json",
        help="path to api-keys.json (default: api-keys.json in repo root)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        keys_path = Path(args.api_keys)
        api_keys = load_api_keys(keys_path)
        user_info = get_user_info(api_keys, args.user)
        exchange = build_exchange(user_info)
        logging.info("Using exchange: %s", getattr(exchange, "id", type(exchange).__name__))
        logging.info("Fetching balance...")
        balance = exchange.fetch_balance()
        pretty_print_balance(balance)
    except Exception:
        logging.exception("Failed to fetch balance")
        sys.exit(1)


if __name__ == "__main__":
    main()
