import asyncio
import hashlib
import json
import logging
import time
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from ccxt.base.errors import (
    AuthenticationError,
    ExchangeError,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    OrderNotFound,
    RateLimitExceeded,
)

from ccxt_contracts import build_contract_bot, get_bot_class
from custom_endpoint_overrides import (
    CustomEndpointConfigError,
    ResolvedEndpointOverride,
)
from exchanges.bitunix import BitunixBot, BitunixClient, BitunixOrderStream
from live.bitunix_create_journal import (
    BitunixCreateAttempt,
    BitunixCreateJournal,
    BitunixCreateJournalError,
)
from live.bitunix_create_recovery import BitunixCreateRecovery
from live.bitunix_create_wave import OrdinaryCloseWaveMember
from live import executor, hsl_live
from live.event_bus import ReasonCodes
from live.hsl_live import OrdinaryCreateWave, ProtectiveBatch
from live.market_snapshot import MarketSnapshot
from live.position_fill_sync import WriteDeferred, connector_context, defer_submission
from live.state_refresh import AuthoritativeSurfaceUnavailable
from fill_events_manager import (
    BitunixFetcher,
    DEFAULT_FALLBACK_FEE_FRACTION,
    FillEvent,
    _build_fetcher_for_bot,
    signed_fee_paid_from_payload,
)
from candlestick_manager import CandlestickManager

MARKET_ROW = {
    "symbol": "BTCUSDT",
    "base": "BTC",
    "quote": "USDT",
    "minTradeVolume": "0.0001",
    "basePrecision": 4,
    "quotePrecision": 1,
    "minLeverage": 1,
    "maxLeverage": 125,
    "defaultLeverage": 20,
    "defaultMarginMode": "CROSS",
    "symbolStatus": "OPEN",
    "isApiSupported": True,
}


def _market():
    return {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "active": True,
        "maker": BitunixClient.DEFAULT_MAKER_FEE,
        "taker": BitunixClient.DEFAULT_TAKER_FEE,
        "info": MARKET_ROW,
    }


def _market_for(market_id: str, symbol: str):
    market = deepcopy(_market())
    market["id"] = market_id
    market["symbol"] = symbol
    return market


def _prepared_client() -> BitunixClient:
    client = BitunixClient({"apiKey": "key", "secret": "secret"})
    client.markets = {"BTC/USDT:USDT": _market()}
    client.markets_by_id = {"BTCUSDT": client.markets["BTC/USDT:USDT"]}
    client.symbols = ["BTC/USDT:USDT"]
    return client


def _snapshot_bot() -> BitunixBot:
    bot = build_contract_bot("bitunix")
    bot.ws_enabled = True
    bot.config["live"]["api"] = {}
    return bot


def _kline_payload(
    market_id: str = "BTCUSDT",
    *,
    timestamp: int = 1_700_000_065_123,
    **data_overrides,
):
    data = {
        "o": "100",
        "h": "103",
        "l": "99",
        "c": "102",
        "b": "4.5",
        "q": "454.5",
    }
    data.update(data_overrides)
    return {
        "ch": "market_kline_1min",
        "symbol": market_id,
        "ts": timestamp,
        "data": data,
    }


class _PublicKlineSocket:
    def __init__(self):
        self.closed = False
        self.sent = []
        self.messages = asyncio.Queue()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive(self):
        return await self.messages.get()

    async def close(self):
        self.closed = True


class _PublicKlineSocketContext:
    def __init__(self, socket):
        self.socket = socket

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *_args):
        await self.socket.close()


class _PublicKlineSession:
    def __init__(self, socket):
        self.socket = socket
        self.connect_calls = []

    def ws_connect(self, url, **kwargs):
        self.connect_calls.append((url, kwargs))
        return _PublicKlineSocketContext(self.socket)


class _PublicKlineSessionPool:
    def __init__(self, sockets):
        self.sockets = iter(sockets)
        self.connect_calls = []

    def ws_connect(self, url, **kwargs):
        self.connect_calls.append((url, kwargs))
        return _PublicKlineSocketContext(next(self.sockets))


def _order_row(**overrides):
    row = {
        "orderId": "order-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "tradeQty": "0",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "price": "50000",
        "side": "BUY",
        "orderType": "LIMIT",
        "effect": "POST_ONLY",
        "clientId": "clock_entry_long_1",
        "reduceOnly": False,
        "status": "NEW",
        "ctime": 1_700_000_000_000,
        "mtime": 1_700_000_000_001,
    }
    row.update(overrides)
    return row


def _trade_row(**overrides):
    row = {
        "tradeId": "trade-1",
        "orderId": "order-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "price": "50000",
        "side": "BUY",
        "orderType": "MARKET",
        "effect": "",
        "clientId": "clock_entry_long_1",
        "reduceOnly": False,
        "fee": "0.01",
        "realizedPNL": "0",
        "ctime": 1_700_000_000_000,
        "roleType": "TAKER",
    }
    row.update(overrides)
    return row


def _position_row(**overrides):
    row = {
        "positionId": "position-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "entryValue": "50",
        "side": "LONG",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "unrealizedPNL": "0",
        "realizedPNL": "0",
        "avgOpenPrice": "50000",
        "ctime": 1_700_000_000_000,
        "mtime": 1_700_000_000_001,
    }
    row.update(overrides)
    return row


def test_ccxt_contract_registry_uses_bitunix_adapter():
    assert get_bot_class("bitunix") is BitunixBot


def test_candlestick_manager_uses_bitunix_kline_page_limit(tmp_path):
    client = _prepared_client()
    manager = CandlestickManager(
        exchange=client,
        exchange_name="bitunix",
        cache_dir=str(tmp_path / "caches"),
    )
    assert manager._ccxt_limit_default == 200


def test_rest_signature_matches_documented_double_sha256(monkeypatch):
    client = BitunixClient({"apiKey": "api", "secret": "secret"})
    monkeypatch.setattr(
        "exchanges.bitunix.uuid.uuid4", lambda: SimpleNamespace(hex="nonce")
    )
    monkeypatch.setattr(client, "milliseconds", lambda: 1_700_000_000_000)

    headers = client._signed_headers({"symbol": "BTCUSDT", "limit": 10}, "")

    first = hashlib.sha256(b"nonce1700000000000apilimit10symbolBTCUSDT").hexdigest()
    expected = hashlib.sha256(f"{first}secret".encode()).hexdigest()
    assert headers == {
        "api-key": "api",
        "nonce": "nonce",
        "timestamp": "1700000000000",
        "sign": expected,
        "Content-Type": "application/json",
    }


def test_private_websocket_signature_uses_seconds(monkeypatch):
    client = BitunixClient({"apiKey": "api", "secret": "secret"})
    stream = BitunixOrderStream(client)
    monkeypatch.setattr(
        "exchanges.bitunix.uuid.uuid4", lambda: SimpleNamespace(hex="nonce")
    )
    monkeypatch.setattr("exchanges.bitunix.time.time", lambda: 1_700_000_000.9)

    payload = stream._login_payload()

    first = hashlib.sha256(b"nonce1700000000api").hexdigest()
    expected = hashlib.sha256(f"{first}secret".encode()).hexdigest()
    assert payload["args"][0] == {
        "apiKey": "api",
        "timestamp": 1_700_000_000,
        "nonce": "nonce",
        "sign": expected,
    }


@pytest.mark.parametrize(
    "code,error_type",
    [
        (10004, AuthenticationError),
        (10001, NetworkError),
        (10006, RateLimitExceeded),
        (20003, InsufficientFunds),
        (30042, InvalidOrder),
    ],
)
def test_api_errors_map_to_ccxt_exception_contract(code, error_type):
    with pytest.raises(error_type) as exc_info:
        BitunixClient._raise_api_error(code, "test")

    assert exc_info.value.code == str(code)


def test_observed_code_one_network_envelope_is_retryable():
    with pytest.raises(NetworkError, match="Network Error"):
        BitunixClient._raise_api_error(1, "Network Error")


@pytest.mark.asyncio
async def test_load_markets_maps_base_quantity_and_tick_sizes():
    client = BitunixClient()
    client._request = AsyncMock(return_value=[MARKET_ROW])

    markets = await client.load_markets(True)

    market = markets["BTC/USDT:USDT"]
    assert market["id"] == "BTCUSDT"
    assert market["swap"] is True
    assert market["linear"] is True
    assert market["active"] is True
    assert market["contractSize"] == 1.0
    assert market["maker"] == pytest.approx(0.0002)
    assert market["taker"] == pytest.approx(0.0006)
    assert market["precision"] == {"amount": 0.0001, "price": 0.1}
    assert market["limits"]["amount"]["min"] == pytest.approx(0.0001)
    assert market["limits"]["leverage"]["max"] == pytest.approx(125)


@pytest.mark.asyncio
async def test_load_markets_retries_observed_network_error(monkeypatch):
    client = BitunixClient()
    client._request = AsyncMock(side_effect=[NetworkError("transient"), [MARKET_ROW]])
    sleep = AsyncMock()
    monkeypatch.setattr("exchanges.bitunix.asyncio.sleep", sleep)

    markets = await client.load_markets(True)

    assert "BTC/USDT:USDT" in markets
    assert client._request.await_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_list", [False, True])
async def test_balance_sums_disjoint_components_after_transfer_reconciliation(as_list):
    client = _prepared_client()
    raw = {
        "marginCoin": "USDT",
        "available": "1000",
        "frozen": "4",
        "margin": "10",
        "transfer": "1000",
        "bonus": "0",
        "positionMode": "HEDGE",
        "crossUnrealizedPNL": "2",
        "isolationUnrealizedPNL": "-1",
    }
    response = [raw] if as_list else raw
    client._request = AsyncMock(return_value=response)

    balance = await client.fetch_balance()

    assert balance["free"]["USDT"] == 1000.0
    assert balance["used"]["USDT"] == 14.0
    assert balance["total"]["USDT"] == 1014.0
    assert client._request.await_args.kwargs["params"] == {"marginCoin": "USDT"}


@pytest.mark.asyncio
async def test_balance_transfer_reconciliation_does_not_net_isolated_profit():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value={
            "marginCoin": "USDT",
            "available": "100",
            "frozen": "0",
            "margin": "10",
            "transfer": "90",
            "bonus": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "-10",
            "isolationUnrealizedPNL": "10",
        }
    )

    balance = await client.fetch_balance()

    assert balance["free"]["USDT"] == 100.0
    assert balance["used"]["USDT"] == 10.0
    assert balance["total"]["USDT"] == 110.0
    assert client._accepted_balance_components == (100.0, 0.0, 10.0)


@pytest.mark.asyncio
async def test_balance_zero_transfer_floor_cannot_confirm_locked_funds():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value={
            "marginCoin": "USDT",
            "available": "10",
            "frozen": "0",
            "margin": "10",
            "transfer": "0",
            "bonus": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "-10",
            "isolationUnrealizedPNL": "0",
        }
    )

    with pytest.raises(AuthoritativeSurfaceUnavailable) as exc_info:
        await client.fetch_balance()

    assert exc_info.value.reason == "balance_consistency_check"
    assert client._accepted_balance_components is None
    assert client._pending_balance_components == (10.0, 0.0, 10.0)


@pytest.mark.asyncio
async def test_balance_wallet_is_invariant_to_unrealized_pnl():
    client = _prepared_client()
    client._accepted_balance_components = (90.0, 4.0, 6.0)
    responses = []
    for upnl, transfer in (("-25", "65"), ("0", "90"), ("25", "90")):
        raw = {
            "marginCoin": "USDT",
            "available": "90",
            "frozen": "4",
            "margin": "6",
            "transfer": transfer,
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": upnl,
            "isolationUnrealizedPNL": "0",
        }
        client._request = AsyncMock(return_value=raw)
        responses.append(await client.fetch_balance())

    assert [response["total"]["USDT"] for response in responses] == [100.0] * 3


@pytest.mark.asyncio
async def test_balance_defers_duplicated_locked_funds_until_response_recovers():
    client = _prepared_client()
    client._accepted_balance_components = (85.3, 0.4, 1.2)

    def account(*, available, transfer):
        return {
            "marginCoin": "USDT",
            "available": str(available),
            "frozen": "0.4",
            "margin": "1.2",
            "transfer": str(transfer),
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "-0.2",
            "isolationUnrealizedPNL": "0",
        }

    client._request = AsyncMock(
        side_effect=[
            account(available=85.3, transfer=85.1),
            account(available=86.9, transfer=85.1),
            account(available=86.9, transfer=85.1),
            account(available=85.3, transfer=85.1),
        ]
    )

    initial = await client.fetch_balance()
    with pytest.raises(AuthoritativeSurfaceUnavailable) as first:
        await client.fetch_balance()
    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    recovered = await client.fetch_balance()

    assert initial["total"]["USDT"] == pytest.approx(86.9)
    assert first.value.surface == "balance"
    assert first.value.reason == "balance_consistency_check"
    assert recovered["total"]["USDT"] == pytest.approx(86.9)
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_accepts_exact_increase_when_transfer_reconciles():
    client = _prepared_client()
    client._accepted_balance_components = (90.0, 4.0, 6.0)
    client._request = AsyncMock(
        side_effect=[
            {
                "marginCoin": "USDT",
                "available": "90",
                "frozen": "4",
                "margin": "6",
                "transfer": "90",
                "positionMode": "HEDGE",
                "crossUnrealizedPNL": "0",
                "isolationUnrealizedPNL": "0",
            },
            {
                "marginCoin": "USDT",
                "available": "100",
                "frozen": "4",
                "margin": "6",
                "transfer": "100",
                "positionMode": "HEDGE",
                "crossUnrealizedPNL": "0",
                "isolationUnrealizedPNL": "0",
            },
            {
                "marginCoin": "USDT",
                "available": "100",
                "frozen": "4",
                "margin": "6",
                "transfer": "100",
                "positionMode": "HEDGE",
                "crossUnrealizedPNL": "0",
                "isolationUnrealizedPNL": "0",
            },
        ]
    )

    before = await client.fetch_balance()
    after = await client.fetch_balance()

    assert before["total"]["USDT"] == 100.0
    assert after["total"]["USDT"] == 110.0


@pytest.mark.asyncio
async def test_balance_restart_during_duplication_waits_for_consistent_recovery():
    client = _prepared_client()

    def account(available):
        return {
            "marginCoin": "USDT",
            "available": str(available),
            "frozen": "0.4",
            "margin": "1.2",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "-0.2",
            "isolationUnrealizedPNL": "0",
        }

    def account_with_transfer(available, transfer):
        result = account(available)
        result["transfer"] = str(transfer)
        return result

    client._request = AsyncMock(
        side_effect=[
            account_with_transfer(86.9, 85.1),
            account_with_transfer(86.9, 85.1),
            account_with_transfer(85.3, 85.1),
        ]
    )

    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    recovered = await client.fetch_balance()

    assert recovered["free"]["USDT"] == pytest.approx(85.3)
    assert recovered["used"]["USDT"] == pytest.approx(1.6)
    assert recovered["total"]["USDT"] == pytest.approx(86.9)
    assert client._accepted_balance_components == pytest.approx((85.3, 0.4, 1.2))
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_initial_locked_funds_require_transfer_reconciliation():
    client = _prepared_client()
    raw = {
        "marginCoin": "USDT",
        "available": "90",
        "frozen": "4",
        "margin": "6",
        "transfer": "90",
        "bonus": "0",
        "positionMode": "HEDGE",
        "crossUnrealizedPNL": "0",
        "isolationUnrealizedPNL": "0",
    }
    client._request = AsyncMock(return_value=raw)

    confirmed = await client.fetch_balance()

    assert confirmed["total"]["USDT"] == 100.0
    assert client._accepted_balance_components == (90.0, 4.0, 6.0)
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_ambiguous_state_never_ages_into_acceptance():
    client = _prepared_client()
    raw = {
        "marginCoin": "USDT",
        "available": "100",
        "frozen": "4",
        "margin": "6",
        "transfer": "90",
        "bonus": "0",
        "positionMode": "HEDGE",
        "crossUnrealizedPNL": "0",
        "isolationUnrealizedPNL": "0",
    }
    client._request = AsyncMock(return_value=raw)

    for _ in range(100):
        with pytest.raises(AuthoritativeSurfaceUnavailable) as exc_info:
            await client.fetch_balance()

    assert exc_info.value.reason == "balance_consistency_check"
    assert client._accepted_balance_components is None
    assert client._pending_balance_components == (100.0, 4.0, 6.0)


@pytest.mark.asyncio
async def test_balance_changed_locks_remain_deferred_until_transfer_reconciles():
    client = _prepared_client()
    client._accepted_balance_components = (90.0, 4.0, 6.0)

    def account(*, available, frozen, margin, transfer):
        return {
            "marginCoin": "USDT",
            "available": str(available),
            "frozen": str(frozen),
            "margin": str(margin),
            "transfer": str(transfer),
            "bonus": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "0",
            "isolationUnrealizedPNL": "0",
        }

    client._request = AsyncMock(
        side_effect=[
            account(available=100, frozen=4, margin=6, transfer=90),
            account(available=101, frozen=3, margin=7, transfer=90),
            account(available=91, frozen=3, margin=7, transfer=91),
        ]
    )

    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    reconciled = await client.fetch_balance()

    assert reconciled["total"]["USDT"] == 101.0
    assert client._accepted_balance_components == (91.0, 3.0, 7.0)
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_new_locked_funds_require_transfer_reconciliation():
    client = _prepared_client()
    client._accepted_balance_components = (90.0, 0.0, 0.0)

    def account(*, available, margin):
        return {
            "marginCoin": "USDT",
            "available": str(available),
            "frozen": "0",
            "margin": str(margin),
            "transfer": "90",
            "bonus": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "0",
            "isolationUnrealizedPNL": "0",
        }

    client._request = AsyncMock(
        side_effect=[
            account(available=100, margin=10),
            account(available=90, margin=10),
        ]
    )

    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await client.fetch_balance()
    reconciled = await client.fetch_balance()

    assert reconciled["total"]["USDT"] == 100.0
    assert client._accepted_balance_components == (90.0, 0.0, 10.0)
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_matching_locked_funds_still_require_transfer_reconciliation():
    client = _prepared_client()
    client._accepted_balance_components = (90.0, 0.0, 10.0)

    def account(*, available, transfer):
        return {
            "marginCoin": "USDT",
            "available": str(available),
            "frozen": "0",
            "margin": "10",
            "transfer": str(transfer),
            "bonus": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "0",
            "isolationUnrealizedPNL": "0",
        }

    client._request = AsyncMock(
        side_effect=[
            account(available=90, transfer=80),
            account(available=80, transfer=80),
        ]
    )

    with pytest.raises(AuthoritativeSurfaceUnavailable) as exc_info:
        await client.fetch_balance()
    reconciled = await client.fetch_balance()

    assert exc_info.value.reason == "balance_consistency_check"
    assert reconciled["free"]["USDT"] == 80.0
    assert reconciled["used"]["USDT"] == 10.0
    assert reconciled["total"]["USDT"] == 90.0
    assert client._accepted_balance_components == (80.0, 0.0, 10.0)
    assert client._pending_balance_components is None


@pytest.mark.asyncio
async def test_balance_initial_state_without_locked_funds_is_immediately_usable():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value={
            "marginCoin": "USDT",
            "available": "100",
            "frozen": "0",
            "margin": "0",
            "positionMode": "HEDGE",
            "crossUnrealizedPNL": "0",
            "isolationUnrealizedPNL": "0",
        }
    )

    balance = await client.fetch_balance()

    assert balance["total"]["USDT"] == 100.0
    assert client._accepted_balance_components == (100.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_bitunix_config_write_readiness_defers_unavailable_balance():
    bot = BitunixBot.__new__(BitunixBot)
    bot.balance_override = None
    bot.cca = SimpleNamespace(
        fetch_balance=AsyncMock(
            side_effect=AuthoritativeSurfaceUnavailable(
                "balance", "balance_consistency_check"
            )
        )
    )

    ready = await bot._exchange_config_write_ready()

    assert ready is False
    assert bot._last_authoritative_block_reason == "balance_consistency_check"
    bot.cca.fetch_balance.assert_awaited_once()


@pytest.mark.asyncio
async def test_bitunix_config_write_readiness_honors_balance_override():
    bot = BitunixBot.__new__(BitunixBot)
    bot.balance_override = 100.0
    bot.cca = SimpleNamespace(
        fetch_balance=AsyncMock(side_effect=AssertionError("must not fetch"))
    )

    ready = await bot._exchange_config_write_ready()

    assert ready is True
    bot.cca.fetch_balance.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balance_override",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -1.0,
        True,
        "not-a-number",
    ],
)
async def test_bitunix_config_write_readiness_rejects_invalid_balance_override(
    balance_override,
):
    bot = BitunixBot.__new__(BitunixBot)
    bot.balance_override = balance_override
    bot.cca = SimpleNamespace(
        fetch_balance=AsyncMock(side_effect=AssertionError("must not fetch"))
    )

    with pytest.raises(
        ValueError,
        match="balance_override must be a positive finite numeric value",
    ):
        await bot._exchange_config_write_ready()

    bot.cca.fetch_balance.assert_not_awaited()


def test_hedge_order_normalization_preserves_position_and_action_side():
    client = _prepared_client()

    entry = client._normalize_order(_order_row())
    close = client._normalize_order(
        _order_row(side="SELL", reduceOnly=True, clientId="clock_close_long_1")
    )
    short_close = client._normalize_order(
        _order_row(
            side="BUY",
            reduceOnly=True,
            clientId="clock_close_short_1",
        )
    )

    assert (entry["side"], entry["info"]["positionSide"], entry["reduceOnly"]) == (
        "buy",
        "LONG",
        False,
    )
    assert (close["side"], close["info"]["positionSide"], close["reduceOnly"]) == (
        "sell",
        "LONG",
        True,
    )
    assert (
        short_close["side"],
        short_close["info"]["positionSide"],
        short_close["reduceOnly"],
    ) == ("buy", "SHORT", True)


def test_order_status_accepts_live_trailing_enum_padding():
    client = _prepared_client()
    assert client._normalize_order(_order_row(status="NEW_"))["status"] == "open"


def test_pending_endpoint_conservatively_retains_safe_unknown_status(caplog):
    client = _prepared_client()

    with caplog.at_level(logging.WARNING):
        order = client._normalize_order(
            _order_row(status="CANCELING"), pending_snapshot=True
        )

    assert order["status"] == "open"
    assert "status=CANCELING" in caplog.text
    assert "retain_as_open_until_authoritative_absence" in caplog.text
    with pytest.raises(ValueError, match="Unknown Bitunix order status"):
        client._normalize_order(_order_row(status="CANCELING"))


@pytest.mark.parametrize(
    "status",
    [
        None,
        "",
        "bad status",
        "A" * 33,
        "CANCELING" + "_" * 24,
        "NEW" + "_" * 30,
    ],
)
def test_pending_endpoint_rejects_missing_or_malformed_status(status):
    client = _prepared_client()

    with pytest.raises(ValueError, match="Unknown Bitunix order status"):
        client._normalize_order(_order_row(status=status), pending_snapshot=True)


def test_detail_endpoint_rejects_oversized_padded_known_status():
    client = _prepared_client()

    with pytest.raises(ValueError, match="Unknown Bitunix order status"):
        client._normalize_order(_order_row(status="NEW" + "_" * 30))


def test_pending_unknown_status_warning_is_throttled(monkeypatch, caplog):
    client = _prepared_client()
    monotonic = iter((100.0, 101.0, 401.0))
    monkeypatch.setattr("exchanges.bitunix.time.monotonic", lambda: next(monotonic))

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            client._normalize_order(
                _order_row(status="CANCELING"), pending_snapshot=True
            )

    assert caplog.text.count("status=CANCELING") == 2


@pytest.mark.parametrize("price", [None, "", "0", "-1", "nan"])
def test_limit_order_requires_positive_finite_price(price):
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.price"):
        client._normalize_order(_order_row(price=price))


@pytest.mark.parametrize("qty", [None, "", "0", "-1", "nan"])
def test_order_requires_positive_finite_quantity(qty):
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.qty"):
        client._normalize_order(_order_row(qty=qty))


def test_terminal_market_order_allows_missing_request_price():
    client = _prepared_client()

    order = client._normalize_order(
        _order_row(orderType="MARKET", price=None, status="FILLED")
    )

    assert order["type"] == "market"
    assert order["price"] == 0.0


def test_open_market_order_requires_positive_price_for_reconciliation():
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.price"):
        client._normalize_order(
            _order_row(orderType="MARKET", price=None, status="NEW")
        )


@pytest.mark.asyncio
async def test_open_orders_paginate_to_stable_reported_total():
    client = _prepared_client()
    client.milliseconds = lambda: 1_700_001_000_000
    client._request = AsyncMock(
        side_effect=[
            {
                "orderList": [
                    _order_row(
                        orderId="order-2",
                        ctime=1_700_000_000_002,
                        mtime=1_700_000_000_002,
                    )
                ],
                "total": 2,
            },
            {"orderList": [_order_row()], "total": 2},
        ]
    )

    orders = await client.fetch_open_orders(limit=1)

    assert [order["id"] for order in orders] == ["order-1", "order-2"]
    assert [
        call.kwargs["params"]["skip"] for call in client._request.await_args_list
    ] == [0, 1]
    assert {
        call.kwargs["params"]["endTime"] for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"orderList": [_order_row()]},
        {"orderList": [_order_row()], "total": None},
        {"orderList": [_order_row()], "total": 0},
        {"orderList": [], "total": -1},
    ],
)
async def test_open_orders_reject_missing_or_inconsistent_total(payload):
    client = _prepared_client()
    client._request = AsyncMock(return_value=payload)

    with pytest.raises(ValueError, match="open-orders total"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_open_orders_reject_total_changes_between_pages():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"orderList": [_order_row(orderId="order-2")], "total": 2},
            {"orderList": [_order_row()], "total": 3},
        ]
    )

    with pytest.raises(ValueError, match="open-orders total changed"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_open_orders_reject_duplicate_or_incomplete_pagination():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"orderList": [_order_row()], "total": 2},
            {"orderList": [_order_row()], "total": 2},
        ]
    )

    with pytest.raises(ValueError, match="duplicate or incomplete"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_create_close_order_uses_hedge_side_and_position_id():
    client = _prepared_client()
    client._position_ids[("BTC/USDT:USDT", "long")] = "position-1"
    client._request = AsyncMock(
        return_value={"orderId": "order-1", "clientId": "clock_close_long_1"}
    )

    result = await client.create_order(
        "BTC/USDT:USDT",
        "limit",
        "sell",
        0.001,
        50_000,
        {
            "positionSide": "LONG",
            "clientOrderId": "clock_close_long_1",
            "reduceOnly": True,
            "effect": "POST_ONLY",
        },
    )

    body = client._request.await_args.kwargs["body"]
    assert body == {
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "side": "BUY",
        "tradeSide": "CLOSE",
        "orderType": "LIMIT",
        "reduceOnly": True,
        "price": "50000",
        "effect": "POST_ONLY",
        "clientId": "clock_close_long_1",
        "positionId": "position-1",
    }
    assert result["side"] == "sell"
    assert result["reduceOnly"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side,position_side,reduce_only",
    [
        ("buy", "LONG", False),
        ("sell", "LONG", True),
    ],
)
async def test_create_market_order_returns_identifier_acknowledgement(
    side, position_side, reduce_only
):
    client = _prepared_client()
    if reduce_only:
        client._position_ids[("BTC/USDT:USDT", "long")] = "position-1"
    client._request = AsyncMock(
        return_value={"orderId": "order-1", "clientId": "clock_market_long_1"}
    )

    result = await client.create_order(
        "BTC/USDT:USDT",
        "market",
        side,
        0.001,
        None,
        {
            "positionSide": position_side,
            "clientOrderId": "clock_market_long_1",
            "reduceOnly": reduce_only,
        },
    )

    assert result["id"] == "order-1"
    assert result["clientOrderId"] == "clock_market_long_1"
    assert result["type"] == "market"
    assert result["side"] == side
    assert result["price"] is None
    assert result["status"] is None
    assert result["info"]["positionSide"] == position_side
    assert result["reduceOnly"] is reduce_only
    assert build_contract_bot("bitunix").did_create_order(result) is True
    body = client._request.await_args.kwargs["body"]
    assert body["orderType"] == "MARKET"
    assert "price" not in body
    assert "effect" not in body
    if reduce_only:
        assert body["positionId"] == "position-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_side,expected",
    [("LONG", "long"), ("BUY", "long"), ("SHORT", "short"), ("SELL", "short")],
)
async def test_positions_accept_documented_and_live_hedge_side_aliases(
    raw_side, expected
):
    client = _prepared_client()
    client._request = AsyncMock(return_value=[_position_row(side=raw_side)])

    positions = await client.fetch_positions(["BTC/USDT:USDT"])

    assert positions[0]["side"] == expected
    assert client._position_ids[("BTC/USDT:USDT", expected)] == "position-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_price", [None, "", "0", "-1", "nan"])
async def test_nonzero_position_requires_positive_finite_entry_price(entry_price):
    client = _prepared_client()
    client._request = AsyncMock(return_value=[_position_row(avgOpenPrice=entry_price)])

    with pytest.raises(ValueError, match="entry price"):
        await client.fetch_positions(["BTC/USDT:USDT"])


@pytest.mark.asyncio
async def test_create_close_order_rejects_missing_live_position():
    client = _prepared_client()
    client.fetch_positions = AsyncMock(return_value=[])
    with pytest.raises(InvalidOrder, match="no live positionId"):
        await client.create_order(
            "BTC/USDT:USDT",
            "market",
            "sell",
            0.001,
            None,
            {
                "positionSide": "LONG",
                "clientOrderId": "clock_close_long_1",
                "reduceOnly": True,
            },
        )


@pytest.mark.asyncio
async def test_fill_history_paginates_and_normalizes_hedge_actions():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {
                "tradeList": [
                    _trade_row(
                        tradeId="trade-2",
                        side="SELL",
                        reduceOnly=True,
                        realizedPNL="0.5",
                        ctime=1_700_000_000_002,
                    )
                ],
                "total": 2,
            },
            {
                "tradeList": [_trade_row()],
                "total": 2,
            },
        ]
    )

    trades = await client.fetch_my_trades(
        since=1_699_999_000_000,
        limit=1,
        params={"until": 1_700_001_000_000},
    )

    assert [trade["id"] for trade in trades] == ["trade-1", "trade-2"]
    assert trades[0]["side"] == "buy"
    assert trades[0]["info"]["positionSide"] == "LONG"
    assert trades[1]["side"] == "sell"
    assert trades[1]["info"]["positionSide"] == "LONG"
    assert client._request.await_args_list[1].kwargs["params"]["skip"] == 1
    assert {
        call.kwargs["params"]["endTime"] for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
async def test_fill_history_freezes_end_time_when_until_is_omitted(monkeypatch):
    client = _prepared_client()
    monkeypatch.setattr(client, "milliseconds", lambda: 1_700_001_000_000)
    client._request = AsyncMock(
        side_effect=[
            {"tradeList": [_trade_row(tradeId="trade-2")], "total": 2},
            {"tradeList": [_trade_row()], "total": 2},
        ]
    )

    await client.fetch_my_trades(limit=1)

    assert [
        call.kwargs["params"]["skip"] for call in client._request.await_args_list
    ] == [0, 1]
    assert {
        call.kwargs["params"]["endTime"] for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"tradeList": [_trade_row()]},
        {"tradeList": [_trade_row()], "total": None},
        {"tradeList": [_trade_row()], "total": 0},
        {"tradeList": [_trade_row()], "total": -1},
    ],
)
async def test_fill_history_rejects_missing_or_inconsistent_total(payload):
    client = _prepared_client()
    client._request = AsyncMock(return_value=payload)

    with pytest.raises(ValueError, match="fill-history total"):
        await client.fetch_my_trades(limit=1)


@pytest.mark.asyncio
async def test_fill_history_rejects_total_changes_between_pages():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"tradeList": [_trade_row(tradeId="trade-2")], "total": 2},
            {"tradeList": [_trade_row()], "total": 3},
        ]
    )

    with pytest.raises(ValueError, match="fill-history total changed"):
        await client.fetch_my_trades(limit=1)


def test_trade_normalization_preserves_fee_rebate_sign():
    client = _prepared_client()

    trade = client._normalize_trade(_trade_row(fee="-0.01"))
    event = BitunixFetcher._normalize_trade(trade)

    assert trade["fee"]["cost"] == -0.01
    assert trade["fees"][0]["cost"] == -0.01
    assert signed_fee_paid_from_payload(event) == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_ohlcv_is_clamped_reversed_and_deduplicated():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 120_000,
                "open": "2",
                "high": "3",
                "low": "1",
                "close": "2.5",
                "quoteVol": "4",
                "baseVol": "10",
            },
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": "3",
                "baseVol": "4.5",
            },
        ]
    )

    candles = await client.fetch_ohlcv(
        "BTC/USDT:USDT",
        "1m",
        since=60_000,
        limit=1000,
        params={"until": 180_000},
    )

    assert candles == [
        [60_000, 1.0, 2.0, 0.5, 1.5, 3.0],
        [120_000, 2.0, 3.0, 1.0, 2.5, 4.0],
    ]
    query = client._request.await_args.kwargs["params"]
    assert query["limit"] == 200
    assert query["startTime"] == 60_000
    assert query["endTime"] == 180_000


@pytest.mark.asyncio
async def test_ohlcv_derives_forward_page_end_from_since_when_venue_ignores_start():
    client = _prepared_client()
    client._request = AsyncMock(return_value=[])

    await client.fetch_ohlcv("BTC/USDT:USDT", "1m", since=60_000, limit=200)

    query = client._request.await_args.kwargs["params"]
    assert query["startTime"] == 60_000
    assert query["endTime"] == 60_000 + 200 * 60_000


@pytest.mark.asyncio
@pytest.mark.parametrize("volume", [None, "", "nan", "-1"])
async def test_ohlcv_rejects_missing_or_invalid_base_volume(volume):
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": volume,
            }
        ]
    )

    with pytest.raises(ValueError, match="kline.quoteVol"):
        await client.fetch_ohlcv("BTC/USDT:USDT", "1m", since=60_000)


@pytest.mark.asyncio
async def test_ohlcv_accepts_explicit_zero_base_volume():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": "0",
            }
        ]
    )

    candles = await client.fetch_ohlcv("BTC/USDT:USDT", "1m", since=60_000)

    assert candles[0][5] == 0.0


def test_order_stream_advertises_native_ohlcv_and_normalizes_push_timestamp():
    client = _prepared_client()
    stream = BitunixOrderStream(client)

    symbol, rows = stream._normalize_ohlcv_payload(_kline_payload())

    assert stream.has["watchOHLCV"] is True
    assert symbol == "BTC/USDT:USDT"
    assert rows == [[1_700_000_040_000, 100.0, 103.0, 99.0, 102.0, 4.5]]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"b": None}, "base volume"),
        ({"b": "-1"}, "invalid price or volume"),
        ({"h": "101"}, "inconsistent OHLC"),
        ({"l": "103"}, "inconsistent OHLC"),
    ],
)
def test_order_stream_rejects_malformed_kline_payload(overrides, match):
    client = _prepared_client()
    stream = BitunixOrderStream(client)

    with pytest.raises(ValueError, match=match):
        stream._normalize_ohlcv_payload(_kline_payload(**overrides))


@pytest.mark.asyncio
async def test_order_stream_multiplexes_public_klines_and_uses_json_ping(
    monkeypatch,
    caplog,
):
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    socket = _PublicKlineSocket()
    session = _PublicKlineSession(socket)
    client._get_session = AsyncMock(return_value=session)
    stream = BitunixOrderStream(client)
    stream.PING_INTERVAL_SECONDS = 0.0
    monkeypatch.setattr("exchanges.bitunix.time.time", lambda: 1_700_000_000.9)

    btc_waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    eth_waiter = asyncio.create_task(stream.watch_ohlcv(eth_symbol, "1m"))
    for _ in range(100):
        subscribed = {
            arg["symbol"]
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
        }
        if subscribed == {"BTCUSDT", "ETHUSDT"}:
            break
        await asyncio.sleep(0.01)
    assert subscribed == {"BTCUSDT", "ETHUSDT"}
    assert len(session.connect_calls) == 1
    assert {"op": "ping", "ping": 1_700_000_000} in socket.sent

    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {"op": "ping", "pong": 1_700_000_000, "ping": 1_700_000_001}
            ),
        )
    )
    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload("BTCUSDT", l="101")),
        )
    )
    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload("BTCUSDT")),
        )
    )
    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload("ETHUSDT", c="101")),
        )
    )

    btc_rows, eth_rows = await asyncio.gather(btc_waiter, eth_waiter)
    assert btc_rows[0][4] == 102.0
    assert eth_rows[0][4] == 101.0
    assert "malformed kline updates dropped" in caplog.text
    assert "reason=inconsistent_ohlc" in caplog.text

    await stream.un_watch_ohlcv("BTC/USDT:USDT", "1m")
    await stream.un_watch_ohlcv(eth_symbol, "1m")
    assert socket.closed is True
    await stream.close()


def test_malformed_kline_warning_counts_repeats_across_symbols(caplog, monkeypatch):
    stream = BitunixOrderStream(_prepared_client())
    now = 100.0
    monkeypatch.setattr("exchanges.bitunix.time.monotonic", lambda: now)
    stream._log_malformed_kline("BTC", "inconsistent_ohlc", fallback=False)
    for symbol in ("ETH", "SOL", "XRP", "ADA"):
        now += 1
        stream._log_malformed_kline(symbol, "inconsistent_ohlc", fallback=True)
    assert len(caplog.records) == 1

    now = 400.0
    stream._log_malformed_kline("DOGE", "inconsistent_ohlc", fallback=False)
    assert len(caplog.records) == 2
    assert "drops=5 fallback_requests=4" in caplog.records[-1].message
    assert "sample=ETH,SOL,XRP" in caplog.records[-1].message


@pytest.mark.asyncio
async def test_order_stream_routes_persistent_malformed_klines_to_one_watcher():
    client = _prepared_client()
    socket = _PublicKlineSocket()
    client._get_session = AsyncMock(return_value=_PublicKlineSession(socket))
    stream = BitunixOrderStream(client)
    symbol = "BTC/USDT:USDT"
    queue = asyncio.Queue(maxsize=stream.KLINE_QUEUE_SIZE)
    stream._ohlcv_queues[symbol] = queue
    stream._ensure_ohlcv_task()
    for _ in range(100):
        if stream._ohlcv_ws is socket:
            break
        await asyncio.sleep(0.01)
    for _ in range(stream.MAX_CONSECUTIVE_MALFORMED_KLINES):
        await socket.messages.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(_kline_payload("BTCUSDT", l="101")),
            )
        )
    for _ in range(stream.KLINE_QUEUE_SIZE + 2):
        await socket.messages.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(_kline_payload("BTCUSDT")),
            )
        )
    for _ in range(100):
        if socket.messages.empty():
            break
        await asyncio.sleep(0.01)

    assert queue.qsize() == 1
    assert symbol in stream._ohlcv_fallback_pending
    with pytest.raises(NetworkError, match="consecutive malformed updates"):
        await stream.watch_ohlcv(symbol, "1m")
    assert symbol not in stream._ohlcv_fallback_pending
    assert stream._ohlcv_task is not None
    assert not stream._ohlcv_task.done()
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_public_failure_wakes_all_watchers_for_rest_fallback():
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    socket = _PublicKlineSocket()
    client._get_session = AsyncMock(return_value=_PublicKlineSession(socket))
    stream = BitunixOrderStream(client)

    waiters = [
        asyncio.create_task(stream.watch_ohlcv(symbol, "1m"))
        for symbol in ("BTC/USDT:USDT", eth_symbol)
    ]
    for _ in range(100):
        if stream._ohlcv_ws is socket:
            break
        await asyncio.sleep(0.01)
    await socket.messages.put(
        SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data="closed")
    )

    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert all(isinstance(result, NetworkError) for result in results)
    assert all("NetworkError" in str(result) for result in results)
    await stream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_surface", ["session", "socket"])
async def test_order_stream_setup_failure_wakes_shard_watchers(setup_surface):
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    if setup_surface == "session":
        client._get_session = AsyncMock(side_effect=NetworkError("session unavailable"))
    else:
        session = MagicMock()
        session.ws_connect.side_effect = NetworkError("socket unavailable")
        client._get_session = AsyncMock(return_value=session)
    stream = BitunixOrderStream(client)

    results = await asyncio.gather(
        stream.watch_ohlcv("BTC/USDT:USDT", "1m"),
        stream.watch_ohlcv(eth_symbol, "1m"),
        return_exceptions=True,
    )

    assert all(isinstance(result, NetworkError) for result in results)
    assert all(f"failed: {type(result).__name__}" in str(result) for result in results)
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_transport_failure_preserves_every_fallback_signal():
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    socket = _PublicKlineSocket()
    client._get_session = AsyncMock(return_value=_PublicKlineSession(socket))
    stream = BitunixOrderStream(client)
    symbols = {"BTC/USDT:USDT", eth_symbol}
    for symbol in symbols:
        stream._ohlcv_queues[symbol] = asyncio.Queue(maxsize=stream.KLINE_QUEUE_SIZE)
    stream._ensure_ohlcv_task()
    for _ in range(100):
        if stream._ohlcv_ws is socket:
            break
        await asyncio.sleep(0.01)

    await socket.messages.put(
        SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data="closed")
    )
    for _ in range(100):
        if stream._ohlcv_task is None:
            break
        await asyncio.sleep(0.01)

    assert stream._ohlcv_fallback_pending == symbols
    for symbol in symbols:
        queue = stream._ohlcv_queues[symbol]
        assert queue.qsize() == 1
        assert isinstance(queue.get_nowait(), NetworkError)
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_silence_timeout_wakes_waiter_for_rest_fallback():
    client = _prepared_client()
    socket = _PublicKlineSocket()
    session = _PublicKlineSession(socket)
    client._get_session = AsyncMock(return_value=session)
    stream = BitunixOrderStream(client)
    stream.KLINE_SILENCE_TIMEOUT_SECONDS = 0.02

    waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))

    with pytest.raises(NetworkError, match="failed: NetworkError"):
        await asyncio.wait_for(waiter, timeout=1.0)
    assert session.connect_calls[0][1]["receive_timeout"] == 45.0
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_tracks_kline_silence_per_subscription():
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    socket = _PublicKlineSocket()
    client._get_session = AsyncMock(return_value=_PublicKlineSession(socket))
    stream = BitunixOrderStream(client)
    stream.KLINE_SILENCE_TIMEOUT_SECONDS = 0.05

    btc_waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    eth_waiter = asyncio.create_task(stream.watch_ohlcv(eth_symbol, "1m"))
    for _ in range(100):
        subscribed = {
            arg["symbol"]
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
        }
        if subscribed == {"BTCUSDT", "ETHUSDT"}:
            break
        await asyncio.sleep(0.01)
    assert subscribed == {"BTCUSDT", "ETHUSDT"}

    eth_rows = None
    for _ in range(8):
        await socket.messages.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps({"op": "pong"}),
            )
        )
        await socket.messages.put(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(_kline_payload("ETHUSDT")),
            )
        )
        if eth_rows is None and eth_waiter.done():
            eth_rows = eth_waiter.result()
        await asyncio.sleep(0.01)

    with pytest.raises(NetworkError, match="subscription became silent"):
        await asyncio.wait_for(btc_waiter, timeout=1.0)
    if eth_rows is None:
        eth_rows = await asyncio.wait_for(eth_waiter, timeout=1.0)
    assert eth_rows[0][4] == 102.0
    assert eth_symbol not in stream._ohlcv_fallback_pending
    assert stream._ohlcv_task is not None
    assert not stream._ohlcv_task.done()
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_shards_kline_subscriptions_at_connection_limit():
    client = _prepared_client()
    symbols_by_id = {
        "ETHUSDT": "ETH/USDT:USDT",
        "SOLUSDT": "SOL/USDT:USDT",
    }
    for market_id, symbol in symbols_by_id.items():
        market = _market_for(market_id, symbol)
        client.markets[symbol] = market
        client.markets_by_id[market_id] = market
        client.symbols.append(symbol)
    sockets = [_PublicKlineSocket(), _PublicKlineSocket()]
    session = _PublicKlineSessionPool(sockets)
    client._get_session = AsyncMock(return_value=session)
    stream = BitunixOrderStream(client)
    stream.MAX_KLINE_SUBSCRIPTIONS = 2

    market_symbols = {
        "BTCUSDT": "BTC/USDT:USDT",
        **symbols_by_id,
    }
    waiters = {
        market_id: asyncio.create_task(stream.watch_ohlcv(symbol, "1m"))
        for market_id, symbol in market_symbols.items()
    }
    for _ in range(100):
        subscribed = {
            arg["symbol"]
            for socket in sockets
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
        }
        if subscribed == set(market_symbols):
            break
        await asyncio.sleep(0.01)

    assert subscribed == set(market_symbols)
    assert len(session.connect_calls) == 2
    assert len(stream._ohlcv_tasks) == 2
    subscription_counts = {market_id: 0 for market_id in market_symbols}
    for socket in sockets:
        socket_subscriptions = {
            arg["symbol"]
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
        }
        assert len(socket_subscriptions) <= stream.MAX_KLINE_SUBSCRIPTIONS
        for market_id in socket_subscriptions:
            subscription_counts[market_id] += 1
            await socket.messages.put(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps(_kline_payload(market_id)),
                )
            )

    results = await asyncio.gather(*waiters.values())
    assert all(rows[0][4] == 102.0 for rows in results)
    assert set(subscription_counts.values()) == {1}

    await stream.un_watch_ohlcv("SOL/USDT:USDT", "1m")
    assert len(stream._ohlcv_tasks) == 1
    assert sum(socket.closed for socket in sockets) == 1
    await stream.un_watch_ohlcv("BTC/USDT:USDT", "1m")
    await stream.un_watch_ohlcv("ETH/USDT:USDT", "1m")
    assert not stream._ohlcv_tasks
    assert all(socket.closed for socket in sockets)
    await stream.close()


@pytest.mark.parametrize(
    "rejection_fields",
    [{"code": 10001}, {"success": False}],
)
@pytest.mark.asyncio
async def test_order_stream_scopes_rejected_subscription_to_affected_watcher(
    rejection_fields,
    caplog,
):
    client = _prepared_client()
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    socket = _PublicKlineSocket()
    client._get_session = AsyncMock(return_value=_PublicKlineSession(socket))
    stream = BitunixOrderStream(client)

    btc_waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    eth_waiter = asyncio.create_task(stream.watch_ohlcv(eth_symbol, "1m"))
    for _ in range(100):
        subscribed = {
            arg["symbol"]
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
        }
        if subscribed == {"BTCUSDT", "ETHUSDT"}:
            break
        await asyncio.sleep(0.01)

    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "symbol": "BTCUSDT",
                            "ch": "market_kline_1min",
                        }
                    ],
                    "msg": "must not reach logs",
                    **rejection_fields,
                }
            ),
        )
    )
    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload("ETHUSDT")),
        )
    )

    with pytest.raises(NetworkError, match="subscription was rejected"):
        await asyncio.wait_for(btc_waiter, timeout=1.0)
    assert (await asyncio.wait_for(eth_waiter, timeout=1.0))[0][4] == 102.0
    assert "symbol=BTC/USDT:USDT action=rest_fallback" in caplog.text
    assert "must not reach logs" not in caplog.text
    assert stream._ohlcv_task is not None
    assert not stream._ohlcv_task.done()

    btc_recovery = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    # Subscription reconciliation polls at one-second intervals. Allow one
    # complete poll plus scheduler slack before declaring recovery absent.
    for _ in range(200):
        btc_subscriptions = sum(
            1
            for message in socket.sent
            if message.get("op") == "subscribe"
            for arg in message["args"]
            if arg["symbol"] == "BTCUSDT"
        )
        if btc_subscriptions >= 2:
            break
        await asyncio.sleep(0.01)
    assert btc_subscriptions >= 2
    await socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload("BTCUSDT")),
        )
    )
    assert (await asyncio.wait_for(btc_recovery, timeout=1.0))[0][4] == 102.0
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_can_resubscribe_immediately_after_last_unwatch():
    client = _prepared_client()
    first_socket = _PublicKlineSocket()
    second_socket = _PublicKlineSocket()
    sessions = iter(
        (_PublicKlineSession(first_socket), _PublicKlineSession(second_socket))
    )
    client._get_session = AsyncMock(side_effect=lambda: next(sessions))
    stream = BitunixOrderStream(client)

    first_waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    for _ in range(100):
        if stream._ohlcv_ws is first_socket:
            break
        await asyncio.sleep(0.01)
    await stream.un_watch_ohlcv("BTC/USDT:USDT", "1m")
    with pytest.raises(NetworkError, match="subscription was removed"):
        await first_waiter

    second_waiter = asyncio.create_task(stream.watch_ohlcv("BTC/USDT:USDT", "1m"))
    for _ in range(100):
        if stream._ohlcv_ws is second_socket:
            break
        await asyncio.sleep(0.01)
    await second_socket.messages.put(
        SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(_kline_payload()),
        )
    )
    assert (await second_waiter)[0][4] == 102.0
    await stream.close()


@pytest.mark.asyncio
async def test_order_stream_surfaces_unenriched_row_when_detail_is_not_found():
    client = _prepared_client()
    client.fetch_order = AsyncMock(side_effect=OrderNotFound("not found"))
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": {
                            "orderId": "order-1",
                            "symbol": "BTCUSDT",
                            "status": "FILLED",
                        },
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows == [
        {
            "orderId": "order-1",
            "symbol": "BTC/USDT:USDT",
            "status": "FILLED",
        }
    ]


@pytest.mark.asyncio
async def test_order_stream_sends_documented_json_ping_when_idle(monkeypatch):
    client = _prepared_client()
    stream = BitunixOrderStream(client)
    stream._last_ping_monotonic = time.monotonic() - stream.PING_INTERVAL_SECONDS - 1.0
    monkeypatch.setattr("exchanges.bitunix.time.time", lambda: 1_700_000_000.9)
    order_message = SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "ch": "order",
                "data": {
                    "orderId": "order-1",
                    "symbol": "BTCUSDT",
                    "status": "FILLED",
                },
            }
        ),
    )
    ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(return_value=order_message),
        send_json=AsyncMock(),
    )
    stream._ws = ws
    client.fetch_order = AsyncMock(
        return_value={"id": "order-1", "symbol": "BTC/USDT:USDT"}
    )

    rows = await stream.watch_orders()

    ws.send_json.assert_awaited_once_with({"op": "ping", "ping": 1_700_000_000})
    assert rows == [{"id": "order-1", "symbol": "BTC/USDT:USDT"}]


@pytest.mark.asyncio
async def test_order_stream_isolates_malformed_rest_detail_to_its_row():
    client = _prepared_client()
    valid_order = client._normalize_order(_order_row(orderId="order-2"))
    client.fetch_order = AsyncMock(
        side_effect=[ValueError("malformed order detail"), valid_order]
    )
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": [
                            {
                                "orderId": "order-1",
                                "symbol": "BTCUSDT",
                                "status": "FILLED",
                            },
                            {
                                "orderId": "order-2",
                                "symbol": "BTCUSDT",
                                "status": "FILLED",
                            },
                        ],
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows[0] == {
        "orderId": "order-1",
        "symbol": "BTC/USDT:USDT",
        "status": "FILLED",
    }
    assert rows[1] == valid_order


@pytest.mark.asyncio
async def test_order_stream_propagates_rest_transport_failure():
    client = _prepared_client()
    client.fetch_order = AsyncMock(side_effect=NetworkError("unavailable"))
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": {
                            "orderId": "order-1",
                            "symbol": "BTCUSDT",
                            "status": "FILLED",
                        },
                    }
                ),
            )
        ),
    )

    with pytest.raises(NetworkError, match="unavailable"):
        await stream.watch_orders()


@pytest.mark.asyncio
async def test_order_stream_forwards_malformed_rows_for_account_refresh():
    client = _prepared_client()
    client.fetch_order = AsyncMock()
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": [
                            {"symbol": "BTCUSDT", "status": "FILLED"},
                            "malformed",
                        ],
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows == [
        {
            "symbol": "BTC/USDT:USDT",
            "info": {"raw": {"symbol": "BTCUSDT", "status": "FILLED"}},
        },
        {"info": {"raw": "malformed"}},
    ]
    client.fetch_order.assert_not_awaited()

    bot = build_contract_bot("bitunix")
    bot.ccp = SimpleNamespace(has={"watchOrders": True})
    bot.stop_websocket = False
    bot._do_watch_orders = AsyncMock(side_effect=[rows, asyncio.CancelledError])
    bot._mark_account_critical_state_dirty = MagicMock()
    bot.handle_order_update = MagicMock()

    await bot.watch_orders()

    bot._mark_account_critical_state_dirty.assert_called_once_with(
        reason="order_ws_semantics_unavailable",
        symbols={"BTC/USDT:USDT"},
        source="bitunix_order_ws",
        level=logging.DEBUG,
    )
    bot.handle_order_update.assert_not_called()


@pytest.mark.asyncio
async def test_unenriched_order_row_requests_account_state_refresh():
    bot = build_contract_bot("bitunix")
    bot.ccp = SimpleNamespace(has={"watchOrders": True})
    bot.stop_websocket = False
    bot._do_watch_orders = AsyncMock(
        side_effect=[
            [
                {
                    "orderId": "order-1",
                    "symbol": "BTC/USDT:USDT",
                    "status": "FILLED",
                }
            ],
            asyncio.CancelledError,
        ]
    )
    bot._mark_account_critical_state_dirty = MagicMock()
    bot.handle_order_update = MagicMock()

    await bot.watch_orders()

    bot._mark_account_critical_state_dirty.assert_called_once_with(
        reason="order_ws_semantics_unavailable",
        symbols={"BTC/USDT:USDT"},
        source="bitunix_order_ws",
        level=logging.DEBUG,
    )
    bot.handle_order_update.assert_not_called()


@pytest.mark.asyncio
async def test_ticker_depth_fallback_is_capped_before_any_rest_fanout():
    client = _prepared_client()
    eth_market = {
        **_market(),
        "id": "ETHUSDT",
        "symbol": "ETH/USDT:USDT",
    }
    client.markets[eth_market["symbol"]] = eth_market
    client.markets_by_id[eth_market["id"]] = eth_market
    client.symbols.append(eth_market["symbol"])
    client.TICKER_WAIT_SECONDS = 0.0
    client.MAX_DEPTH_FALLBACK_SYMBOLS = 1
    client._ensure_ticker_tasks = lambda: None
    client._fetch_depth_ticker = AsyncMock()

    with pytest.raises(NetworkError, match="fallback limited"):
        await client.fetch_tickers(client.symbols)

    client._fetch_depth_ticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_rest_only_tickers_skip_public_websocket_and_use_depth():
    client = _prepared_client()
    client.ws_enabled = False
    client._ensure_ticker_tasks = MagicMock(
        side_effect=AssertionError("public websocket must stay disabled")
    )
    expected = {
        "symbol": "BTC/USDT:USDT",
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 1_700_000_000_000,
    }
    client._fetch_depth_ticker = AsyncMock(return_value=expected)

    result = await client.fetch_tickers(["BTC/USDT:USDT"])

    assert result == {"BTC/USDT:USDT": expected}
    client._ensure_ticker_tasks.assert_not_called()
    client._fetch_depth_ticker.assert_awaited_once_with("BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_rest_only_tickers_reject_unbounded_bulk_request():
    client = _prepared_client()
    client.ws_enabled = False
    client._fetch_depth_ticker = AsyncMock()

    with pytest.raises(NetworkError, match="requires explicit symbols"):
        await client.fetch_tickers()

    client._fetch_depth_ticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_depth_ticker_uses_midpoint_as_synthetic_last():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value={"bids": [["99", "1"]], "asks": [["101", "1"]]}
    )
    client.milliseconds = lambda: 1_700_000_000_000

    ticker = await client._fetch_depth_ticker("BTC/USDT:USDT")

    assert ticker["last"] == 100.0
    assert ticker["source"] == "bitunix_depth_mid"
    assert ticker["snapshot_received_ms"] == 1_700_000_000_000


@pytest.mark.asyncio
async def test_future_exchange_timestamp_does_not_extend_cache_freshness():
    client = _prepared_client()
    symbol = "BTC/USDT:USDT"
    client.TICKER_WAIT_SECONDS = 0.0
    client._ensure_ticker_tasks = lambda: None
    client.milliseconds = lambda: 100_000
    client._ticker_cache[symbol] = {
        "symbol": symbol,
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 10**15,
        "snapshot_received_ms": 60_000,
    }
    client._ticker_received_monotonic[symbol] = 60.0
    fallback = {
        "symbol": symbol,
        "bid": 98.0,
        "ask": 102.0,
        "last": 100.0,
        "timestamp": 100_000,
        "source": "bitunix_depth_mid",
    }
    client._fetch_depth_ticker = AsyncMock(return_value=fallback)

    with patch("exchanges.bitunix.time.monotonic", return_value=100.0):
        tickers = await client.fetch_tickers([symbol])

    assert tickers == {symbol: fallback}
    client._fetch_depth_ticker.assert_awaited_once_with(symbol)


@pytest.mark.asyncio
async def test_bot_ticker_normalization_preserves_source_and_timestamp():
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    bot.markets_dict = {symbol: _market()}
    bot.cca = SimpleNamespace(
        fetch_tickers=AsyncMock(
            return_value={
                symbol: {
                    "bid": 99.0,
                    "ask": 101.0,
                    "last": 100.0,
                    "timestamp": 1_700_000_000_000,
                    "snapshot_received_ms": 1_700_000_000_125,
                    "source": "bitunix_depth_mid",
                }
            }
        )
    )

    tickers = await bot.fetch_tickers_for_symbols([symbol])

    assert tickers[symbol] == {
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 1_700_000_000_000,
        "snapshot_received_ms": 1_700_000_000_125,
        "source": "bitunix_depth_mid",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("age_ms", "refresh", "strategy", "extra_markets", "expect_depth"),
    [
        (1_000, False, "symbols", 0, False),
        (25_000, False, "symbols", 0, True),
        (1_000, True, "symbols", 9, False),
        (25_000, True, "symbols", 9, True),
        (25_000, True, "bulk", 0, True),
    ],
)
async def test_bitunix_snapshot_respects_receipt_age_and_targeted_refresh(
    monkeypatch, age_ms, refresh, strategy, extra_markets, expect_depth
):
    import live.market_snapshot as snapshots

    now = 1_800_000_000_000
    monkeypatch.setattr(snapshots, "utc_ms", lambda: now)
    client = _prepared_client()
    client.milliseconds = lambda: now
    client._ensure_ticker_tasks = lambda: None
    client._ticker_ready.wait = AsyncMock(side_effect=AssertionError("unexpected WS wait"))
    symbol = "BTC/USDT:USDT"
    client._ticker_cache[symbol] = {
        "symbol": symbol,
        "bid": 89.0,
        "ask": 91.0,
        "last": 90.0,
        "timestamp": now + 60_000,  # venue time is not the local observation stamp
        "snapshot_received_ms": now - age_ms,
    }
    client._ticker_received_monotonic[symbol] = time.monotonic() - age_ms / 1_000
    for i in range(extra_markets):
        extra = f"COIN{i}/USDT:USDT"
        client.markets[extra] = {**_market(), "id": f"COIN{i}USDT", "symbol": extra}
        client._ticker_cache[extra] = {
            **client._ticker_cache[symbol], "symbol": extra, "snapshot_received_ms": now - 100,
        }
        client._ticker_received_monotonic[extra] = time.monotonic() - 0.1
    client._request = AsyncMock(return_value={"bids": [["99", "1"]], "asks": [["101", "1"]]})
    bot = _snapshot_bot()
    bot.cca = client
    bot.markets_dict = client.markets
    provider = snapshots.MarketSnapshotProvider(
        exchange_name="bitunix",
        fetch_tickers=bot.fetch_tickers,
        fetch_tickers_for_symbols=bot.fetch_tickers_for_symbols,
        fetch_market_tickers_for_symbols=bot.fetch_market_tickers_for_symbols,
        ticker_strategy=strategy,
    )
    quotes = await provider.get_snapshots(
        list(client.markets), max_age_ms=5_000, refresh=refresh
    )
    quote = quotes.pop(symbol)
    assert len(quotes) == extra_markets
    assert all(other.fetched_ms == now - 100 for other in quotes.values())
    if expect_depth:
        assert quote.last == 100.0
        assert quote.source == "bitunix_depth_mid"
        assert quote.fetched_ms == now
        assert quote.exchange_timestamp_ms is None
        client._request.assert_awaited_once_with(
            "GET", "/api/v1/futures/market/depth", params={"symbol": "BTCUSDT", "limit": 1}
        )
    else:
        assert quote.last == 90.0
        assert quote.fetched_ms == now - age_ms
        assert quote.exchange_timestamp_ms == now + 60_000
        client._request.assert_not_awaited()


def test_ticker_tasks_restart_when_active_market_ids_change():
    client = _prepared_client()
    existing_task = MagicMock()
    existing_task.done.return_value = False
    client._ticker_tasks = [existing_task]
    client._ticker_subscription_ids = ("BTCUSDT",)
    created_tasks = []

    def fake_create_task(coroutine, *, name):
        coroutine.close()
        task = MagicMock()
        task.done.return_value = False
        created_tasks.append((task, name))
        return task

    with patch(
        "exchanges.bitunix.asyncio.create_task", side_effect=fake_create_task
    ) as create_task:
        client._ensure_ticker_tasks()
        create_task.assert_not_called()
        existing_task.cancel.assert_not_called()

        eth_market = {
            **_market(),
            "id": "ETHUSDT",
            "symbol": "ETH/USDT:USDT",
        }
        client.markets[eth_market["symbol"]] = eth_market
        client._ensure_ticker_tasks()

    existing_task.cancel.assert_called_once_with()
    assert client._ticker_subscription_ids == ("BTCUSDT", "ETHUSDT")
    assert len(created_tasks) == 1
    assert client._ticker_tasks == [created_tasks[0][0]]


class _ResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _RecordingSession:
    def __init__(self, response):
        self.response = response
        self.call = None

    def request(self, *args, **kwargs):
        self.call = (args, kwargs)
        return _ResponseContext(self.response)


def _tracked_create_fixture(tmp_path, response):
    client = _prepared_client()
    journal = BitunixCreateJournal(tmp_path / "account" / "create_attempts.sqlite3")
    bot = SimpleNamespace(
        cca=client, _ensure_freshness_ledger=lambda: SimpleNamespace(epoch=7)
    )
    client.ordinary_create_recovery = BitunixCreateRecovery(bot, journal)
    response_text = (
        AsyncMock(side_effect=response)
        if isinstance(response, BaseException)
        else AsyncMock(return_value=response)
    )
    session = _RecordingSession(
        SimpleNamespace(status=200, headers={}, text=response_text)
    )
    client._get_session = AsyncMock(return_value=session)
    client._throttle = AsyncMock()
    return bot, journal, session


def _tracked_create_order(*, panic=False):
    return {
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "_dedicated_protective_market_panic": panic,
    }


async def _send_tracked_create(
    bot, order, *, market=False, client_id="test-close-long"
):
    return await bot.cca.create_order(
        order["symbol"],
        "market" if market else "limit",
        "sell",
        0.002,
        None if market else 51_000.0,
        {
            "positionSide": "LONG",
            "reduceOnly": True,
            "positionId": "position-1",
            "clientOrderId": client_id,
        },
    )


@pytest.mark.asyncio
async def test_bitunix_persists_exact_attempt_before_signed_post_and_through_ack(
    tmp_path, monkeypatch
):
    bot, journal, session = _tracked_create_fixture(
        tmp_path, '{"code":0,"data":{"orderId":"venue-1","clientId":"test-close-long"}}'
    )
    order = _tracked_create_order()
    request = session.request
    observed = []

    def observe_durable_attempt(*args, **kwargs):
        assert args[0] == "POST" and args[1].endswith(BitunixClient.PLACE_ORDER_PATH)
        assert kwargs["headers"]["sign"]
        (attempt,) = BitunixCreateJournal(journal.path).load()
        assert attempt == BitunixCreateAttempt.from_wire(
            json.loads(kwargs["data"]), submitted_at_ms=attempt.submitted_at_ms
        )
        assert (attempt.position_side, attempt.side, attempt.position_id) == (
            "long",
            "sell",
            "position-1",
        )
        observed.append(attempt)
        return request(*args, **kwargs)

    monkeypatch.setattr(session, "request", observe_durable_attempt)
    submitted = []
    with connector_context(bot, order, "create", recheck=lambda _: True):
        assert defer_submission(lambda: submitted.append(True))
        result = await _send_tracked_create(bot, order)

    assert result["id"] == "venue-1"
    assert submitted == [True]
    assert len(observed) == 1
    assert journal.load() == tuple(observed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (asyncio.TimeoutError(), NetworkError),
        (asyncio.CancelledError(), asyncio.CancelledError),
        ("invalid JSON", NetworkError),
        ('{"code":0,"data":{"clientId":"test-close-long"}}', ExchangeError),
        ('{"code":30042,"msg":"duplicate client ID","data":null}', InvalidOrder),
    ],
)
async def test_bitunix_ambiguous_place_result_retains_attempt(
    tmp_path, response, error
):
    bot, journal, session = _tracked_create_fixture(tmp_path, response)
    order = _tracked_create_order()

    with connector_context(bot, order, "create", recheck=lambda _: True) as context:
        with pytest.raises(error):
            await _send_tracked_create(bot, order)
        assert context.submitted

    assert session.call is not None
    (attempt,) = BitunixCreateJournal(journal.path).load()
    assert attempt.client_id == "test-close-long"


@pytest.mark.asyncio
async def test_bitunix_known_business_rejection_releases_exact_attempt(tmp_path):
    bot, journal, session = _tracked_create_fixture(
        tmp_path, '{"code":20003,"msg":"insufficient balance","data":null}'
    )
    order = _tracked_create_order()
    with connector_context(bot, order, "create", recheck=lambda _: True):
        with pytest.raises(InsufficientFunds):
            await _send_tracked_create(bot, order)
    assert session.call is not None
    assert journal.path.exists()
    assert journal.load() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["queue", "journal", "expired_after_fsync"])
async def test_bitunix_no_post_or_submit_bookkeeping_after_admission_failure(
    tmp_path, monkeypatch, denial
):
    bot, journal, session = _tracked_create_fixture(
        tmp_path, '{"code":0,"data":{"orderId":"venue-1"}}'
    )
    order = _tracked_create_order()
    authorized = True
    submitted = []
    record = journal.record

    async def queued(*, cancel=False):
        nonlocal authorized
        if denial == "queue":
            authorized = False

    def record_attempt(attempt):
        nonlocal authorized
        if denial == "journal":
            raise BitunixCreateJournalError("simulated failed durability")
        record(attempt)
        assert BitunixCreateJournal(journal.path).load() == (attempt,)
        if denial == "expired_after_fsync":
            authorized = False

    monkeypatch.setattr(bot.cca, "_throttle", queued)
    monkeypatch.setattr(journal, "record", record_attempt)
    with connector_context(
        bot, order, "create", recheck=lambda _: authorized
    ) as context:
        assert defer_submission(lambda: submitted.append(True))
        with pytest.raises(WriteDeferred):
            await _send_tracked_create(bot, order)
        assert not context.submitted

    assert session.call is None
    assert submitted == []
    assert journal.load() == ()
    assert journal.path.exists() == (denial == "expired_after_fsync")


@pytest.mark.asyncio
@pytest.mark.parametrize("market", [False, True])
async def test_bitunix_pending_attempt_allows_protective_batch_and_cancel(
    tmp_path, market
):
    ordinary_ack = (
        '{"code":0,"data":{"orderId":"venue-1","clientId":"test-close-long"}}'
    )
    bot, journal, session = _tracked_create_fixture(tmp_path, ordinary_ack)
    ordinary = _tracked_create_order()
    with connector_context(bot, ordinary, "create", recheck=lambda _: True):
        await _send_tracked_create(bot, ordinary)
    pending = journal.load()
    assert len(pending) == 1

    protective = _tracked_create_order(panic=True)
    batch = ProtectiveBatch(wave=None, cancels=(), creates=(protective,))
    bot._hsl_live = SimpleNamespace(_protective_batch=batch)
    session.call = None
    impostor = dict(protective)
    with connector_context(bot, impostor, "create", recheck=lambda _: True):
        with pytest.raises(WriteDeferred):
            await _send_tracked_create(bot, impostor, market=market)
    assert session.call is None

    session.response.text.return_value = '{"code":0,"data":{"orderId":"venue-2"}}'
    with connector_context(bot, protective, "create", recheck=lambda _: True):
        result = await _send_tracked_create(
            bot, protective, market=market, client_id="test-protection-close-long"
        )
    assert result["id"] == "venue-2"
    assert json.loads(session.call[1]["data"])["orderType"] == (
        "MARKET" if market else "LIMIT"
    )
    assert journal.load() == pending

    bot._hsl_live._protective_batch = None
    session.response.text.return_value = (
        '{"code":0,"data":{"successList":[{"orderId":"stale-order"}],"failureList":[]}}'
    )
    with connector_context(bot, ordinary, "cancel", recheck=lambda _: True):
        result = await bot.cca.cancel_order("stale-order", ordinary["symbol"])
    assert result["id"] == "stale-order"
    assert session.call[0][1].endswith("/cancel_orders")
    assert journal.load() == pending


@pytest.mark.asyncio
async def test_native_market_discovery_propagates_http_retry_after_without_early_retry():
    client = BitunixClient({})
    response = SimpleNamespace(
        status=429,
        headers={"retry-after": "45"},
        text=AsyncMock(return_value="server backpressure"),
    )
    client._get_session = AsyncMock(return_value=_RecordingSession(response))
    client._throttle = AsyncMock()

    with pytest.raises(RateLimitExceeded, match="Bitunix HTTP 429") as error:
        await client.load_markets()

    assert error.value.retry_after_seconds == 45.0
    response.text.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_request_uses_configured_rest_base_and_extra_headers():
    client = BitunixClient(
        {
            "apiKey": "key",
            "secret": "secret",
            "restUrl": "https://proxy.example/base/",
            "headers": {"X-Proxy": "enabled"},
        }
    )
    response = SimpleNamespace(
        status=200,
        text=AsyncMock(return_value='{"code": 0, "data": []}'),
    )
    session = _RecordingSession(response)
    client._get_session = AsyncMock(return_value=session)
    client._throttle = AsyncMock()

    await client._request("GET", "/private", private=True)

    args, kwargs = session.call
    assert args[:2] == ("GET", "https://proxy.example/base/private")
    assert kwargs["headers"]["X-Proxy"] == "enabled"
    assert kwargs["headers"]["api-key"] == "key"


@pytest.mark.parametrize(
    "header_name", ["api-key", "Api-Key", "NONCE", "Timestamp", "sIgN"]
)
def test_native_session_rejects_case_insensitive_auth_header_collision(
    header_name,
):
    with pytest.raises(ValueError, match="reserved authentication header"):
        BitunixClient(
            {
                "apiKey": "key",
                "secret": "secret",
                "headers": {header_name: "must-not-win"},
            }
        )


def test_native_session_applies_bitunix_endpoint_override():
    bot = build_contract_bot("bitunix")
    bot.user_info = {
        "exchange": "bitunix",
        "key": "key",
        "secret": "secret",
        "headers": {"Existing": "header"},
    }
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_url_overrides={"api": "https://proxy.example/bitunix"},
        rest_extra_headers={"X-Proxy": "enabled"},
        disable_ws=True,
    )
    bot.ws_enabled = False

    bot.create_ccxt_sessions()

    assert bot.cca.rest_url == "https://proxy.example/bitunix"
    assert bot.cca.headers == {
        "Existing": "header",
        "X-Proxy": "enabled",
    }
    assert bot.cca.ws_enabled is False
    assert bot.ccp is None
    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_native_session_applies_bitunix_domain_rewrite():
    bot = build_contract_bot("bitunix")
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_domain_rewrites={"https://fapi.bitunix.com": "https://proxy.example"},
    )
    bot.ws_enabled = False

    bot.create_ccxt_sessions()

    assert bot.cca.rest_url == "https://proxy.example"


def test_native_session_rejects_unsupported_endpoint_url_keys():
    bot = build_contract_bot("bitunix")
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_url_overrides={"private": "https://proxy.example"},
    )
    bot.ws_enabled = False

    with pytest.raises(
        CustomEndpointConfigError, match="unsupported REST URL override"
    ):
        bot.create_ccxt_sessions()


def test_market_settings_upgrade_legacy_cache_with_conservative_fees():
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    market = {
        **_market(),
        "precision": {"amount": 0.0001, "price": 0.1},
        "limits": {
            "amount": {"min": 0.0001, "max": None},
            "price": {"min": 0.1, "max": None},
            "cost": {"min": None, "max": None},
        },
        "contractSize": 1.0,
    }
    market.pop("maker")
    market.pop("taker")
    bot.markets_dict = {symbol: market}
    bot.eligible_symbols = {symbol}

    bot.set_market_specific_settings()

    assert bot.markets_dict[symbol]["maker"] == pytest.approx(0.0002)
    assert bot.markets_dict[symbol]["taker"] == pytest.approx(0.0006)


def test_bot_order_params_require_durable_hedge_semantics():
    bot = build_contract_bot("bitunix")

    params = bot._build_order_params(
        {
            "position_side": "short",
            "custom_id": "clock_close_short_1",
            "reduce_only": True,
            "type": "limit",
        }
    )

    assert params == {
        "positionSide": "SHORT",
        "clientOrderId": "clock_close_short_1",
        "reduceOnly": True,
        "effect": "GTC",
    }
    with pytest.raises(ValueError, match="invalid position_side"):
        bot._build_order_params(
            {
                "position_side": "",
                "custom_id": "clock_entry_long_1",
                "type": "limit",
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_mode,current_leverage,expect_margin,expect_leverage",
    [
        ("CROSS", 3, False, False),
        ("CROSS", 5, False, True),
        # A margin-mode change may reset venue leverage, so set it again even
        # when the pre-mutation leverage happened to match.
        ("ISOLATION", 3, True, True),
    ],
)
async def test_symbol_config_mutates_only_needed_state_and_reapplies_leverage(
    current_mode,
    current_leverage,
    expect_margin,
    expect_leverage,
):
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    bot.max_leverage = {symbol: 125}
    bot.markets_dict = {symbol: _market()}
    bot.cca = SimpleNamespace(
        fetch_leverage_margin_mode=AsyncMock(
            return_value={
                "marginMode": current_mode,
                "leverage": current_leverage,
            }
        ),
        set_margin_mode=AsyncMock(return_value={"code": 0}),
        set_leverage=AsyncMock(return_value={"code": 0}),
    )

    await bot.update_exchange_config_by_symbols([symbol])

    assert bot.cca.set_margin_mode.await_count == int(expect_margin)
    assert bot.cca.set_leverage.await_count == int(expect_leverage)
    if expect_leverage:
        bot.cca.set_leverage.assert_awaited_once_with(3, symbol)


@pytest.mark.asyncio
async def test_bitunix_fetcher_normalizes_accounting_fields():
    client = _prepared_client()
    client.fetch_my_trades = AsyncMock(
        return_value=[client._normalize_trade(_trade_row())]
    )
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(1_699_999_000_000, 1_700_001_000_000, cache)

    assert len(events) == 1
    assert events[0]["id"] == "trade-1"
    assert events[0]["position_side"] == "long"
    assert events[0]["side"] == "buy"
    assert events[0]["pnl"] == 0.0
    assert events[0]["c_mult"] == 1.0
    assert signed_fee_paid_from_payload(events[0]) == pytest.approx(-0.01)
    assert cache["trade-1"][0] == "clock_entry_long_1"


@pytest.mark.asyncio
@pytest.mark.parametrize("fee", ["missing", None, "", "0", "-0.01", "0.01"])
async def test_bitunix_missing_fee_uses_shared_fallback_and_preserves_reported_fees(
    fee,
):
    client = _prepared_client()
    raw = _trade_row()
    if fee == "missing":
        raw.pop("fee")
    else:
        raw["fee"] = fee
    client.fetch_my_trades = AsyncMock(return_value=[client._normalize_trade(raw)])
    events = await BitunixFetcher(client).fetch(None, None, {})
    event = FillEvent.from_dict(events[0])
    if fee in ("missing", None, ""):
        assert event.fee_paid == pytest.approx(
            -abs(event.qty * event.price) * DEFAULT_FALLBACK_FEE_FRACTION
        )
        assert event.fee_source == "fallback_pct"
        assert event.fee_quality == "fallback"
    else:
        assert event.fee_paid == pytest.approx(-float(fee))
        assert event.fee_source == "reported_quote"
        assert event.fee_quality == "exact"


@pytest.mark.asyncio
async def test_bitunix_fetcher_enriches_missing_fill_client_id_from_order_detail():
    client = _prepared_client()
    trade = client._normalize_trade(_trade_row(clientId=""))
    client.fetch_my_trades = AsyncMock(return_value=[trade])
    client.fetch_order = AsyncMock(
        return_value={
            "clientOrderId": "clock_entry_long_1",
            "info": {"clientId": "clock_entry_long_1"},
        }
    )
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(None, None, cache)

    assert events[0]["client_order_id"] == "clock_entry_long_1"
    assert events[0]["pb_order_type"]
    assert cache["trade-1"][0] == "clock_entry_long_1"


@pytest.mark.asyncio
async def test_bitunix_fetcher_retains_fill_when_order_detail_expired():
    client = _prepared_client()
    trade = client._normalize_trade(_trade_row(clientId=""))
    client.fetch_my_trades = AsyncMock(return_value=[trade])
    client.fetch_order = AsyncMock(side_effect=OrderNotFound("expired"))
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(None, None, cache)

    assert [event["id"] for event in events] == ["trade-1"]
    assert events[0]["client_order_id"] == ""
    assert events[0]["pb_order_type"] == "unknown"
    assert cache["trade-1"] == ("", "unknown")


def test_build_fetcher_for_bitunix():
    bot = SimpleNamespace(
        exchange="bitunix",
        cca=object(),
        user="bitunix_user",
        config={"live": {}},
    )
    assert isinstance(_build_fetcher_for_bot(bot, []), BitunixFetcher)


def test_setup_bot_bitunix_uses_native_adapter():
    from passivbot import setup_bot

    config = {"live": {"account": {"user": "bitunix_01"}}}
    with patch("passivbot.load_user_info", return_value={"exchange": "bitunix"}):
        with patch("exchanges.bitunix.BitunixBot") as mock_cls:
            result = setup_bot(config)
    assert result is mock_cls.return_value
    mock_cls.assert_called_once_with(config)


@pytest.mark.parametrize(
    "configured, expected",
    [(None, "symbols"), ("auto", "symbols"), ("symbols", "symbols"), ("bulk", "bulk")],
)
def test_bitunix_snapshot_default_is_scoped_but_preserves_override(
    configured, expected
):
    bot = _snapshot_bot()
    if configured is None:
        bot.config["live"].get("api", {}).pop("market_snapshot_ticker_strategy", None)
    else:
        bot.config["live"]["api"]["market_snapshot_ticker_strategy"] = configured
    assert bot._market_snapshot_ticker_strategy() == expected


@pytest.mark.asyncio
async def test_hsl_quote_refresh_does_not_wait_for_unrelated_bitunix_market(
    monkeypatch,
):
    """Exercise the real connector/provider/owner with offline websocket receipts."""
    from live.hsl_live import Owner
    from live.market_snapshot import MarketSnapshotProvider
    import live.market_snapshot as snapshots

    client = _prepared_client()
    symbol = "BTC/USDT:USDT"
    absent = {**_market(), "id": "ETHUSDT", "symbol": "ETH/USDT:USDT"}
    client.markets[absent["symbol"]] = absent
    client.markets_by_id[absent["id"]] = absent
    client.symbols.append(absent["symbol"])
    client._ensure_ticker_tasks = lambda: None
    # No packet will arrive for the unrelated market. Waiting for it is the bug.
    client._ticker_ready.wait = AsyncMock(
        side_effect=AssertionError("unrelated ticker wait")
    )
    client._fetch_depth_ticker = AsyncMock(
        side_effect=AssertionError("unneeded REST fallback")
    )
    clock = [1_000_000]
    monkeypatch.setattr(snapshots, "utc_ms", lambda: clock[0])
    bot = _snapshot_bot()
    bot.config["live"]["api"]["market_snapshot_ticker_strategy"] = "auto"
    bot.cca = client
    bot.markets_dict = client.markets
    provider = MarketSnapshotProvider(
        exchange_name="bitunix",
        fetch_tickers=bot.fetch_tickers,
        fetch_tickers_for_symbols=bot.fetch_tickers_for_symbols,
        fetch_market_tickers_for_symbols=bot.fetch_market_tickers_for_symbols,
        ticker_strategy=bot._market_snapshot_ticker_strategy(),
    )
    bot.market_snapshot_provider = provider
    client.milliseconds = lambda: clock[0]
    owner = Owner(bot)
    try:
        # Every pass expires the provider cache, reproducing repeated refreshes.
        for price in (100.0, 90.0, 105.0):
            clock[0] += 11_000
            client._ticker_cache[symbol] = dict(
                symbol=symbol,
                bid=price - 1,
                ask=price + 1,
                last=price,
                timestamp=clock[0],
                snapshot_received_ms=clock[0],
            )
            client._ticker_received_monotonic[symbol] = time.monotonic()
            quotes = await owner.acquire_quotes({symbol})
            assert quotes[symbol].last == price
            assert quotes[symbol].fetched_ms == clock[0]
            assert not owner._quote_tasks
        client._ticker_ready.wait.assert_not_awaited()
        client._fetch_depth_ticker.assert_not_awaited()
    finally:
        owner.cancel_inputs()
        provider.cancel_pending()


def test_order_stream_kline_rejection_does_not_expose_raw_field():
    client = _prepared_client()
    stream = BitunixOrderStream(client)
    with pytest.raises(ValueError, match="invalid high") as error:
        stream._normalize_ohlcv_payload(_kline_payload(h="UNBOUNDED_SENSITIVE_VALUE"))
    assert "UNBOUNDED_SENSITIVE_VALUE" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "sent_count", "ack_count", "pending_count"),
    [
        ("all_ack", 2, 2, 2),
        ("wrong_client_id", 1, 0, 1),
        ("missing_client_id", 1, 0, 1),
        ("old_venue_id", 1, 0, 1),
        ("boolean_success", 1, 0, 1),
        ("unexpected_fill_payload", 1, 0, 1),
        ("ack_storage_failure", 1, 0, 1),
        ("timeout_first", 1, 0, 1),
        ("timeout_second", 2, 1, 2),
        ("duplicate_venue_id", 2, 1, 2),
        ("rejected_second", 2, 1, 1),
        ("changed_account", 1, 1, 1),
        ("reconfirmed_account", 1, 1, 1),
        ("changed_wire", 1, 1, 1),
        ("market_moved", 1, 1, 1),
        ("expired_after_fsync", 1, 1, 1),
    ],
)
async def test_bitunix_ordinary_close_wave_transport_and_recovery(
    tmp_path, monkeypatch, outcome, sent_count, ack_count, pending_count
):
    bot, journal, _ = _tracked_create_fixture(tmp_path, "")
    client = bot.cca
    eth_symbol = "ETH/USDT:USDT"
    eth_market = _market_for("ETHUSDT", eth_symbol)
    client.markets[eth_symbol] = eth_market
    client.markets_by_id["ETHUSDT"] = eth_market
    client.symbols.append(eth_symbol)
    members = []
    quotes = {}
    clock = int(time.time() * 1000)
    for i, symbol in enumerate(("BTC/USDT:USDT", eth_symbol), start=1):
        order = dict(
            symbol=symbol,
            position_side="long",
            side="sell",
            type="limit",
            reduce_only=True,
            qty=0.002,
            price=51_000.0,
            custom_id=f"wave-member-{i}",
        )
        position_id = f"position-{i}"
        client._position_ids[(symbol, "long")] = position_id
        member = OrdinaryCloseWaveMember(
            order,
            symbol,
            "long",
            position_id,
            order["custom_id"],
            Decimal(str(order["qty"])),
            Decimal(str(order["price"])),
        )
        members.append(member)
        quotes[symbol] = MarketSnapshot(
            symbol,
            49_999.0,
            50_001.0,
            50_000.0,
            clock,
            "bitunix_depth_mid",
            clock,
        )
    snapshot = SimpleNamespace(
        symbols=tuple(quotes),
        invalid_details=lambda **_: [],
    )
    wave = OrdinaryCreateWave(
        tuple(members),
        snapshot,
        asyncio.get_running_loop().time() + 10,
        account_cohort=(0,),
        existing_order_ids=frozenset({"old-order"}),
    )
    bot._current_planning_snapshot = snapshot
    changed_account = False
    account_revision = 0
    bot._hsl_live = SimpleNamespace(
        _ordinary_create_wave=wave,
        _write_lock=asyncio.Lock(),
        admit=lambda order: not changed_account,
        quotes=quotes,
        _ordinary_account_cohort=lambda: (account_revision,),
    )
    bot._loss_gate_enabled = lambda: False
    bot._live_market_snapshot_fetch_max_age_ms = lambda: 10_000
    bot.live_value = lambda name: {"time_in_force": "post_only"}[name]
    bot.config = {"live": {"execution": {"time_in_force": "post_only"}}}
    bot.CLIENT_ORDER_ID_PATTERN = BitunixBot.CLIENT_ORDER_ID_PATTERN
    confirmations = []
    bot._request_authoritative_confirmation = lambda surfaces: confirmations.append(
        set(surfaces)
    )
    bot.did_create_order = lambda result: isinstance(result, dict) and bool(
        result.get("id")
    )
    bot._handle_order_write_failures = AsyncMock()
    monkeypatch.setattr(hsl_live.runtime, "enabled", lambda _: False)

    @hsl_live.connector_write("create")
    async def send(bot, order):
        params = BitunixBot._build_order_params(bot, order)
        return await client.create_order(
            order["symbol"],
            order["type"],
            order["side"],
            order["qty"],
            order["price"],
            params,
        )

    bot.execute_order = lambda order: send(bot, order)
    requests = []
    place = journal.record

    def persist(attempt, **kwargs):
        place(attempt, **kwargs)
        if (
            outcome == "expired_after_fsync"
            and attempt.client_id == members[1].client_id
        ):
            wave.deadline = asyncio.get_running_loop().time() - 1

    monkeypatch.setattr(journal, "record", persist)
    if outcome == "ack_storage_failure":

        def failed_observation(*_args):
            raise BitunixCreateJournalError("simulated write failure")

        monkeypatch.setattr(journal, "observe", failed_observation)

    def request(method, url, *, data, headers):
        nonlocal changed_account
        assert method == "POST" and url.endswith(BitunixClient.PLACE_ORDER_PATH)
        assert headers["sign"]
        index = len(requests)
        body = json.loads(data)
        requests.append(body)
        assert body["positionId"] == members[index].position_id
        assert body["effect"] == "POST_ONLY"
        pending = BitunixCreateJournal(journal.path).load()
        assert [attempt.client_id for attempt in pending] == [
            member.client_id for member in members[: index + 1]
        ]

        async def response():
            nonlocal changed_account, account_revision
            if index == 0:
                if outcome == "changed_account":
                    changed_account = True
                elif outcome == "reconfirmed_account":
                    account_revision += 1
                elif outcome == "changed_wire":
                    members[1].order["price"] += 1
                elif outcome == "market_moved":
                    quotes[eth_symbol] = MarketSnapshot(
                        eth_symbol,
                        51_000.0,
                        51_002.0,
                        51_001.0,
                        clock,
                        "bitunix_depth_mid",
                        clock,
                    )
            if (index, outcome) in {(0, "timeout_first"), (1, "timeout_second")}:
                raise asyncio.TimeoutError()
            if index == 1 and outcome == "rejected_second":
                return '{"code":20003,"msg":"insufficient balance","data":null}'
            ack = {"orderId": f"venue-{index}"}
            if index == 0 and outcome == "old_venue_id":
                ack["orderId"] = "old-order"
            if index == 1 and outcome == "duplicate_venue_id":
                ack["orderId"] = "venue-0"
            if not (index == 0 and outcome == "missing_client_id"):
                ack["clientId"] = (
                    "other"
                    if index == 0 and outcome == "wrong_client_id"
                    else body["clientId"]
                )
            if outcome == "unexpected_fill_payload":
                ack["tradeQty"] = "0.001"
            return json.dumps(
                {"code": False if outcome == "boolean_success" else 0, "data": ack}
            )

        return _ResponseContext(SimpleNamespace(status=200, headers={}, text=response))

    client._get_session = AsyncMock(return_value=SimpleNamespace(request=request))
    results = await executor._execute_ordinary_creates_serially(
        bot,
        [member.order for member in members],
        wave,
    )
    assert len(results) == 2
    assert len(requests) == sent_count == len(wave.submitted)
    assert len(wave.acknowledged_ids) == ack_count
    pending = BitunixCreateJournal(journal.path).load_pending_all()
    assert len(pending) == pending_count
    assert sum(observation is not None for _, observation in pending) == ack_count
    if not ack_count:
        assert results[1].reason_code == ReasonCodes.ORDINARY_CREATE_WAVE_STOPPED
    assert not confirmations
    wave.confirm(bot)
    wave.confirm(bot)
    assert confirmations == [{"balance", "positions", "open_orders", "fills"}]
