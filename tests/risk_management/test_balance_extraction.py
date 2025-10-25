from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.realtime import _extract_balance


def test_extract_balance_uses_settle_currency_total() -> None:
    balance_payload = {
        "total": {"USDT": "100.5"},
        "info": {"totalWalletBalance": "100.5"},
    }

    assert _extract_balance(balance_payload, "USDT") == pytest.approx(100.5)


def test_extract_balance_prefers_bybit_aggregate_fields() -> None:
    balance_payload = {
        "total": {"USDC": "250000", "USDT": "2506.92500033"},
        "info": {
            "retCode": "0",
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalMarginBalance": "252484.27769571",
                        "totalEquity": "252484.27769571",
                        "totalWalletBalance": "252495.83864295",
                        "coin": [
                            {"coin": "USDC", "equity": "250000", "walletBalance": "250000"},
                            {
                                "coin": "USDT",
                                "equity": "2496.86694033",
                                "walletBalance": "2506.92500033",
                            },
                        ],
                    }
                ]
            },
        },
    }

    result = _extract_balance(balance_payload, "USDT")

    assert result == pytest.approx(252484.27769571)
