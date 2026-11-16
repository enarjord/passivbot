from decimal import Decimal
from unittest.mock import AsyncMock, call

import pytest
from ccxt.base.errors import AuthenticationError, InvalidOrder, NetworkError, OrderNotFound

from live.bitunix_create_evidence import (
    lookup_bitunix_create_evidence,
    parse_bitunix_create_evidence,
)
from live.bitunix_create_journal import BitunixCreateAttempt


SUBMITTED_MS = 1_780_000_000_000
DAY_MS = 86_400_000
CLIENT_ID = "0x0012abcdeffedcba02468defaceabcdef"
ORDER_ID = "900719925474099312345"
DETAIL = "/api/v1/futures/trade/get_order_detail"
HISTORY = "/api/v1/futures/trade/get_history_orders"


@pytest.fixture
def attempt():
    return BitunixCreateAttempt.from_wire(
        {
            "clientId": CLIENT_ID,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "tradeSide": "CLOSE",
            "reduceOnly": True,
            "positionId": "pos42",
            "orderType": "LIMIT",
            "qty": "12",
            "price": "50042.120",
            "effect": "POST_ONLY",
        },
        submitted_at_ms=SUBMITTED_MS,
    )


@pytest.fixture
def order_row():
    return {
        "orderId": ORDER_ID,
        "clientId": CLIENT_ID,
        "symbol": "BTCUSDT",
        "side": "SELL",
        "positionMode": "HEDGE",
        "reduceOnly": True,
        "type": "LIMIT",
        "effect": "POST_ONLY",
        "qty": "12.000",
        "tradeQty": "0",
        "price": "50042.12",
        "status": "NEW",
        "ctime": SUBMITTED_MS + 5,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "filled", "expected"),
    [
        ("NEW_", "0", "working"),
        ("NEW", "0.5", "working"),
        ("PART_FILLED", "4.1", "working"),
        ("CANCELED", "0", "terminal"),
        ("CANCELED", "2.1", "terminal"),
        ("PART_FILLED_CANCELED", "2.1", "terminal"),
        ("FILLED", "12.00", "terminal"),
        ("EXPIRED", "0", "terminal"),
    ],
)
async def test_exact_bitunix_create_detail_is_positive_evidence(
    attempt, order_row, status, filled, expected
):
    request = AsyncMock(return_value=[{**order_row, "status": status, "tradeQty": filled}])

    evidence = await lookup_bitunix_create_evidence(request, attempt)

    request.assert_awaited_once_with("GET", DETAIL, params={"clientId": CLIENT_ID}, private=True)
    assert evidence.order_id == ORDER_ID
    assert evidence.disposition == expected
    assert evidence.venue_status == status.rstrip("_")
    assert evidence.filled_qty == Decimal(filled)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"clientId": "other-order-with-identical-economics"},
        {"clientId": None},
        {"orderId": " "},
        {"symbol": "ETHUSDT"},
        {"side": "BUY"},
        {"reduceOnly": 1},
        {"positionMode": "ONE_WAY"},
        {"positionId": "another-position"},
        {"tradeSide": "OPEN"},
        {"orderType": "MARKET"},
        {"effect": None},
        {"price": "50042.13"},
        {"qty": "0"},
        {"ctime": SUBMITTED_MS - 60_001},
        {"status": "INIT"},
        {"status": "UNRECOGNIZED"},
        {"status": "FILLED"},
        {"status": "CANCELED", "tradeQty": None},
        {"status": "CANCELED", "tradeQty": ""},
        {"status": "CANCELED", "tradeQty": "NaN"},
        {"status": "CANCELED", "tradeQty": "-0.1"},
        {"status": "CANCELED", "tradeQty": "12.1"},
        {"status": "CANCELED", "tradeQty": True},
        {"status": "PART_FILLED", "tradeQty": "0"},
        {"orderStatus": "CANCELED"},
        {"dealAmount": "0.1"},
    ],
)
async def test_conflicting_or_incomplete_bitunix_detail_leaves_attempt_pending(
    attempt, order_row, change
):
    request = AsyncMock(return_value={**order_row, **change})

    assert await lookup_bitunix_create_evidence(request, attempt) is None
    request.assert_awaited_once()


def test_pending_order_explicit_aliases_preserve_fill_evidence(attempt, order_row):
    del order_row["status"]
    del order_row["tradeQty"]
    assert parse_bitunix_create_evidence(order_row, attempt) is None

    evidence = parse_bitunix_create_evidence(
        {**order_row, "orderStatus": "PART_FILLED", "dealAmount": "0.1"}, attempt
    )

    assert evidence.disposition == "working"
    assert evidence.filled_qty == Decimal("0.1")


@pytest.mark.asyncio
async def test_detail_miss_can_resolve_from_exact_canceled_history(attempt, order_row):
    request = AsyncMock(
        side_effect=[
            OrderNotFound("Bitunix error 20007"),
            {"orderList": [], "total": 0},
            {"orderList": [{**order_row, "status": "PART_FILLED_CANCELED", "tradeQty": "2.1"}]},
        ]
    )

    evidence = await lookup_bitunix_create_evidence(request, attempt, now_ms=SUBMITTED_MS + 1000)

    params = {
        "clientId": CLIENT_ID,
        "symbol": "BTCUSDT",
        "startTime": SUBMITTED_MS - 60_000,
        "limit": 2,
    }
    assert request.await_args_list == [
        call("GET", DETAIL, params={"clientId": CLIENT_ID}, private=True),
        call("GET", HISTORY, params={**params, "queryCanceled": "false"}, private=True),
        call("GET", HISTORY, params={**params, "queryCanceled": "true"}, private=True),
    ]
    assert evidence.disposition == "terminal"
    assert evidence.filled_qty == Decimal("2.1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("age_days", "requests"),
    [(0, 3), (4, 2), (91, 1)],
)
async def test_lookup_absence_keeps_old_attempts_unresolved(attempt, age_days, requests):
    request = AsyncMock(side_effect=[OrderNotFound(), {"orderList": []}, OrderNotFound()])

    assert await lookup_bitunix_create_evidence(
        request, attempt, now_ms=SUBMITTED_MS + age_days * DAY_MS
    ) is None
    assert request.await_count == requests


@pytest.mark.asyncio
async def test_exact_history_rejects_duplicate_or_incomplete_matches(attempt, order_row):
    canceled = {**order_row, "status": "CANCELED"}
    for history in (
        {"orderList": [canceled, canceled]},
        {"orderList": [canceled], "total": 2},
        {"orderList": [{**canceled, "clientId": "foreign-client-id"}], "total": 1},
    ):
        request = AsyncMock(side_effect=[OrderNotFound(), history])

        assert await lookup_bitunix_create_evidence(request, attempt, now_ms=SUBMITTED_MS) is None
        assert request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        NetworkError("unavailable"),
        AuthenticationError("permissions"),
        InvalidOrder("Bitunix error 30042"),
    ],
)
async def test_lookup_propagates_unexpected_exchange_failures(attempt, error):
    request = AsyncMock(side_effect=error)

    with pytest.raises(type(error), match=str(error)):
        await lookup_bitunix_create_evidence(request, attempt)
    request.assert_awaited_once()
