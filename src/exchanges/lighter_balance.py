"""Lighter account validation shared by asynchronous trading and sync tools."""

import math

import ccxt


def validate_balance(result, account_index):
    rows = result["info"]["accounts"]
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or str(rows[0]["account_index"]) != str(account_index)
    ):
        raise ValueError("Lighter returned an unexpected account")
    value = rows[0]["collateral"]
    if isinstance(value, bool):
        raise ValueError("Lighter collateral must be numeric")
    collateral = float(value)
    if not math.isfinite(collateral):
        raise ValueError("Lighter invalid collateral")
    # The realized derivatives wallet excludes unrealized PnL.
    result["total"]["USDC"] = collateral
    result["USDC"]["total"] = collateral
    return result


class SyncLighterBalance(ccxt.lighter):
    def fetch_balance(self, params=None):
        return validate_balance(
            super().fetch_balance(params or {}), self.options["accountIndex"]
        )
