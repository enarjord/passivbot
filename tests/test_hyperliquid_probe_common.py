import argparse

import pytest

from tools.hyperliquid_probe_common import (
    extract_balance_summary,
    mask_secret,
    require_live_mutation_confirmation,
    round_to_step,
)


def test_mask_secret_preserves_prefix_and_suffix():
    assert mask_secret("0x1234567890abcdef") == "0x1234...cdef"


def test_round_to_step_supports_up_and_down():
    assert round_to_step(1.001, 0.01, mode="up") == pytest.approx(1.01)
    assert round_to_step(1.009, 0.01, mode="down") == pytest.approx(1.0)


def test_extract_balance_summary_includes_cross_and_margin_fields():
    summary = extract_balance_summary(
        {
            "free": {"USDC": 10.0},
            "used": {"USDC": 1.0},
            "total": {"USDC": 11.0},
            "info": {
                "withdrawable": "9.5",
                "assetPositions": [{"position": {"coin": "BTC"}}],
                "marginSummary": {
                    "accountValue": "11.0",
                    "totalMarginUsed": "1.0",
                    "totalNtlPos": "20.0",
                    "totalRawUsd": "11.0",
                },
                "crossMarginSummary": {
                    "accountValue": "10.5",
                    "totalMarginUsed": "0.5",
                    "totalNtlPos": "15.0",
                    "totalRawUsd": "10.5",
                },
            },
        }
    )

    assert summary["free_usdc"] == 10.0
    assert summary["used_usdc"] == 1.0
    assert summary["margin_account_value"] == "11.0"
    assert summary["cross_account_value"] == "10.5"
    assert summary["asset_positions_count"] == 1


def test_require_live_mutation_confirmation_fails_without_yes():
    parser = argparse.ArgumentParser(prog="probe")
    args = argparse.Namespace(yes=False, user="hyperliquid_01", symbol="BTC/USDC:USDC")

    with pytest.raises(SystemExit) as exc:
        require_live_mutation_confirmation(
            parser,
            args,
            action_description="probe_hyperliquid_position_balance",
        )

    assert exc.value.code == 2


def test_require_live_mutation_confirmation_accepts_yes():
    parser = argparse.ArgumentParser(prog="probe")
    args = argparse.Namespace(yes=True, user="hyperliquid_01", symbol="BTC/USDC:USDC")

    require_live_mutation_confirmation(
        parser,
        args,
        action_description="probe_hyperliquid_position_balance",
    )
