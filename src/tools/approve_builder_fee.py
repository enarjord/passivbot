#!/usr/bin/env python3
"""
Approve the Hyperliquid builder fee for a Passivbot account.

Usage:
  python3 src/tools/approve_builder_fee.py --user hyperliquid_01
  python3 src/tools/approve_builder_fee.py --user hyperliquid_01 --api-keys path/to/api-keys.json

Builder fee approval must be signed by the main wallet. If api-keys.json contains
an agent wallet key, you will be prompted to enter your main wallet private key
interactively (it is never stored).
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
from pathlib import Path

# Allow running from repo root: `python3 src/tools/approve_builder_fee.py`
# src/ modules use bare imports (e.g. `from utils import ...`), so add src/ to path
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))

import ccxt.async_support as ccxt_async
from procedures import load_broker_code, load_user_info


def _is_positive_fee_value(value) -> bool:
    """Parse various fee value formats (int, float, "0.01%", etc.)."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            return float(text) > 0.0
        except ValueError:
            return False
    return False


def _extract_max_builder_fee(payload):
    """Recursively extract the max builder fee from various API response shapes."""
    if isinstance(payload, dict):
        for key in ("maxBuilderFee", "maxFeeRate"):
            if key in payload:
                return _extract_max_builder_fee(payload[key])
        for key in ("data", "response", "result"):
            if key in payload:
                return _extract_max_builder_fee(payload[key])
        return None
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_max_builder_fee(item)
            if extracted is not None:
                return extracted
        return None
    return payload


def _parse_builder_config(broker_code) -> dict:
    """Parse broker_codes.hjson hyperliquid entry into builder/feeRate."""
    if not isinstance(broker_code, dict):
        return {}
    builder = broker_code.get("builder")
    if not builder:
        return {}
    fee_rate = broker_code.get("fee_rate", broker_code.get("feeRate", "0.02%"))
    return {"builder": builder, "feeRate": fee_rate}


def _create_exchange(wallet_address: str, private_key: str) -> ccxt_async.hyperliquid:
    return ccxt_async.hyperliquid(
        {
            "walletAddress": wallet_address,
            "privateKey": private_key,
            "enableRateLimit": True,
        }
    )


def _ensure_secure_prompt_available() -> bool:
    # Fail closed if hidden input cannot be guaranteed.
    return bool(sys.stdin.isatty() and sys.stderr.isatty())


async def _check_approved(exchange, wallet_address: str, builder: str) -> bool:
    """Check if the builder fee is already approved for this wallet."""
    try:
        res = await exchange.fetch(
            "https://api.hyperliquid.xyz/info",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "type": "maxBuilderFee",
                    "user": wallet_address,
                    "builder": builder,
                }
            ),
        )
        max_fee = _extract_max_builder_fee(res)
        return _is_positive_fee_value(max_fee)
    except Exception as e:
        logging.debug("fee approval check failed (%s)", type(e).__name__)
        return False


async def _approve(exchange, builder: str, fee_rate: str) -> None:
    """Call CCXT approve_builder_fee."""
    await exchange.approve_builder_fee(builder, fee_rate)


async def main_async(user: str, api_keys_path: str) -> int:
    user_info = load_user_info(user, api_keys_path)

    if user_info.get("exchange", "").lower() != "hyperliquid":
        print(f"Error: user '{user}' is on exchange '{user_info.get('exchange')}', not hyperliquid")
        return 1

    broker_code = load_broker_code("hyperliquid")
    builder_cfg = _parse_builder_config(broker_code)
    if not builder_cfg:
        print("Error: no builder config found in broker_codes.hjson for hyperliquid")
        print("Expected a dict with 'builder' address. Check broker_codes.hjson.")
        return 1

    builder = builder_cfg["builder"]
    fee_rate = builder_cfg["feeRate"]
    wallet_address = user_info.get("wallet_address", "")

    if not wallet_address:
        print("Error: no wallet_address found for this user in api-keys.json")
        return 1

    print(f"Builder address: {builder}")
    print(f"Fee rate:        {fee_rate}")
    print(f"Wallet address:  {wallet_address}")
    print()

    # Step 1: Check if already approved
    private_key = user_info.get("private_key", "")
    if not private_key:
        print("Error: no private_key found for this user in api-keys.json")
        return 1

    exchange = _create_exchange(wallet_address, private_key)
    try:
        print("Checking current approval status...")
        if await _check_approved(exchange, wallet_address, builder):
            print("Builder fee is already approved! No action needed.")
            return 0

        # Step 2: Try approving with the key from api-keys.json
        print("Not yet approved. Attempting approval with key from api-keys.json...")
        try:
            await _approve(exchange, builder, fee_rate)
            print("Builder fee approved successfully!")
            return 0
        except Exception as e:
            logging.debug(
                "approval with api-keys.json key failed (%s)",
                type(e).__name__,
            )
            print(
                "Approval failed with the key in api-keys.json (likely an agent wallet)."
            )
    finally:
        await exchange.close()

    # Step 3: Prompt for main wallet private key
    print()
    print("Builder fee approval requires a signature from the main wallet.")
    print("Your main wallet private key will NOT be stored anywhere.")
    print()
    if not _ensure_secure_prompt_available():
        print("Cannot prompt securely for private key (stdin/stderr is not a TTY).")
        print("Run this command in an interactive terminal.")
        return 1
    main_key = getpass.getpass("Enter main wallet private key (hidden): ").strip()
    if not main_key:
        print("No key entered. Aborting.")
        return 1

    exchange = _create_exchange(wallet_address, main_key)
    try:
        print("Attempting approval with main wallet key...")
        await _approve(exchange, builder, fee_rate)
        # Verify
        if await _check_approved(exchange, wallet_address, builder):
            print("Builder fee approved successfully!")
            return 0
        else:
            print("Approval call succeeded but verification failed.")
            print("It may take a moment to propagate. Try running this tool again.")
            return 0
    except Exception as e:
        logging.debug("approval with main wallet key failed (%s)", type(e).__name__)
        print("Approval failed.")
        print()
        print("Possible causes:")
        print("  - Incorrect private key")
        print("  - Key does not match the wallet address")
        print("  - Network issue")
        return 1
    finally:
        await exchange.close()
        main_key = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approve the Hyperliquid builder fee for a Passivbot account"
    )
    parser.add_argument("--user", required=True, help="account name in api-keys.json")
    parser.add_argument(
        "--api-keys",
        default="api-keys.json",
        help="path to api-keys.json (default: api-keys.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        rc = asyncio.run(main_async(args.user, args.api_keys))
        sys.exit(rc)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logging.error("Unexpected error (%s)", type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
