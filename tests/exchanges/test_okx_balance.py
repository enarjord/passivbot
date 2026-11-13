from types import SimpleNamespace
from unittest.mock import AsyncMock

import ccxt.async_support as ccxt_async
import pytest

from exchanges.okx import OKXBot
from live.state_refresh import (
    AuthoritativeSurfaceUnavailable,
    DeferredAuthoritativeSurface,
    capture_balance_staged_snapshot,
)


@pytest.fixture
def bot():
    bot = OKXBot.__new__(OKXBot)
    bot.exchange = "okx"
    bot.quote = "USDT"
    return bot


def payload(details, **account_fields):
    return {
        # The unified quote amount is not the full trading account.
        "total": {"USDT": 151.25},
        "info": {"data": [{"details": details, **account_fields}]},
    }


@pytest.mark.parametrize("account_upl", ["", "12"])
def test_okx_removes_cross_and_isolated_upl_in_account_usd(bot, account_upl):
    # BTC contributes 0.02 BTC of cross and 0.01 BTC of isolated UPL.
    # The USDT row contributes -1200 USDT of cross UPL valued at 0.99 USD.
    fetched = payload(
        [
            {
                "ccy": "BTC",
                "eq": "0.1",
                "eqUsd": "6000",
                "upl": "0.03",
                "isoUpl": "0.01",
                "collateralEnabled": False,
            },
            {
                "ccy": "USDT",
                "eq": "-1000",
                "eqUsd": "-990",
                "upl": "-1200",
                "isoUpl": "0",
                "collateralEnabled": True,
            },
        ],
        totalEq="5010",
        upl=account_upl,
    )

    assert bot._get_balance(fetched) == pytest.approx(4398.0)


@pytest.mark.parametrize(
    ("total_equity", "details", "notional", "expected"),
    [
        ("0", [], "0", 0.0),
        ("0", [{"ccy": "BTC", "upl": "0", "eq": "0", "eqUsd": "0"}], "0", 0.0),
        ("1000", [{"ccy": "BTC", "upl": "0"}], "0", 1000.0),
        ("600", [{"ccy": "BTC", "upl": "0.01", "eq": "0.01", "eqUsd": "600"}], "10", 0.0),
    ],
)
def test_okx_preserves_proven_zero_and_does_not_require_quote_currency_row(
    bot, total_equity, details, notional, expected
):
    assert bot._get_balance(
        payload(details, totalEq=total_equity, notionalUsd=notional)
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("fetched", "reason"),
    [
        ({}, "invalid_okx_account"),
        ({"info": {"data": []}}, "invalid_okx_account"),
        ({"info": {"data": [{}, {}]}}, "invalid_okx_account"),
        (
            payload([{"ccy": "USDT", "cashBal": "100", "collateralEnabled": True}]),
            "invalid_okx_total_equity",
        ),
        (payload([], totalEq="NaN", notionalUsd="0"), "invalid_okx_total_equity"),
        (payload(None, totalEq="0", notionalUsd="0"), "invalid_okx_details"),
        (payload([], totalEq="100", notionalUsd="0"), "invalid_okx_details"),
        (payload([], totalEq="0", notionalUsd="10"), "invalid_okx_details"),
        (payload([], totalEq="0", notionalUsd="0", isoEq="10"), "invalid_okx_details"),
        (payload([], totalEq="0", notionalUsd="0", upl="-1"), "invalid_okx_details"),
        (payload([], totalEq="0"), "invalid_okx_notional"),
        (payload([None], totalEq="100"), "invalid_okx_currency"),
        (payload([{"ccy": "", "upl": "0"}], totalEq="100"), "invalid_okx_currency"),
        (
            payload([{"ccy": "USDT", "upl": "0"}, {"ccy": "usdt", "upl": "0"}], totalEq="100"),
            "invalid_okx_currency",
        ),
        (payload([{"ccy": "BTC"}], totalEq="100", upl="0"), "invalid_okx_unrealized_pnl"),
        (payload([{"ccy": "BTC", "upl": ""}], totalEq="100"), "invalid_okx_unrealized_pnl"),
        (
            payload([{"ccy": "BTC", "upl": False}], totalEq="100"),
            "invalid_okx_unrealized_pnl",
        ),
        (
            payload([{"ccy": "BTC", "upl": "0.02", "eq": "0", "eqUsd": "0"}], totalEq="100"),
            "unproven_okx_usd_conversion",
        ),
        (
            payload([{"ccy": "BTC", "upl": "0.02", "eq": "0.1"}], totalEq="100"),
            "invalid_okx_currency_equity_usd",
        ),
        (
            payload([{"ccy": "BTC", "upl": "0.02", "eq": "0.1", "eqUsd": "0"}], totalEq="100"),
            "unproven_okx_usd_conversion",
        ),
        (
            payload([{"ccy": "BTC", "upl": "0.02", "eq": "0.1", "eqUsd": "-6000"}], totalEq="100"),
            "unproven_okx_usd_conversion",
        ),
        (
            payload([{"ccy": "BTC", "upl": "1e308", "eq": "0.1", "eqUsd": "6000"}], totalEq="100"),
            "invalid_okx_unrealized_pnl_usd",
        ),
    ],
)
def test_okx_rejects_unavailable_wallet_snapshot(bot, fetched, reason):
    with pytest.raises(AuthoritativeSurfaceUnavailable) as exc:
        bot._get_balance(fetched)
    assert (exc.value.surface, exc.value.reason) == ("balance", reason)


@pytest.mark.asyncio
async def test_okx_direct_and_staged_capture_use_the_same_wallet_parser(bot):
    fetched = payload(
        [{"ccy": "BTC", "eq": "0.1", "eqUsd": "6000", "upl": "0.02"}],
        totalEq="6000",
    )
    bot.cca = SimpleNamespace(fetch_balance=AsyncMock(return_value=fetched))

    assert await bot.fetch_balance() == pytest.approx(4800.0)
    raw, diagnostic, balance = await capture_balance_staged_snapshot(bot)

    assert raw == fetched
    assert diagnostic["asset_balances"][0]["unrealized_pnl"] == 0.02
    assert balance == pytest.approx(4800.0)
    assert bot.cca.fetch_balance.await_count == 2


@pytest.mark.asyncio
async def test_okx_staged_capture_defers_when_scope_is_unproven(bot):
    fetched = payload([{"ccy": "USDT", "upl": "0"}])
    bot.cca = SimpleNamespace(fetch_balance=AsyncMock(return_value=fetched))

    raw, diagnostic, balance = await capture_balance_staged_snapshot(bot)
    assert raw is None
    assert diagnostic["status"] == "unavailable"
    assert balance == DeferredAuthoritativeSurface(
        surface="balance", reason="invalid_okx_total_equity"
    )
    bot.cca.fetch_balance.assert_awaited_once_with()



@pytest.mark.asyncio
async def test_okx_reads_the_unfiltered_trading_account_through_ccxt(bot):
    response = {
        "code": "0",
        "data": [
            {
                "totalEq": "6000",
                "details": [
                    {
                        "ccy": "BTC",
                        "cashBal": "0.08",
                        "eq": "0.1",
                        "eqUsd": "6000",
                        "upl": "0.02",
                    }
                ],
            }
        ],
        "msg": "",
    }
    exchange = ccxt_async.okx()
    exchange.markets = {}
    exchange.privateGetAccountBalance = AsyncMock(return_value=response)
    exchange.privateGetAssetBalances = AsyncMock()
    bot.cca = exchange
    try:
        raw, balance = await bot.capture_balance_snapshot()
    finally:
        await exchange.close()

    assert raw["info"] == response
    assert balance == pytest.approx(4800.0)
    exchange.privateGetAccountBalance.assert_awaited_once_with({})
    exchange.privateGetAssetBalances.assert_not_awaited()
