from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from copy import deepcopy
from decimal import Decimal
from typing import Any, Iterable
from urllib.parse import urlencode

import aiohttp
from ccxt.base.errors import (
    AuthenticationError,
    BadRequest,
    ExchangeError,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    OrderNotFound,
    RateLimitExceeded,
)

from config.access import require_live_value
from custom_endpoint_overrides import CustomEndpointConfigError
from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from live.balance_composition import normalize_bitunix_balance_composition
from live.state_refresh import AuthoritativeSurfaceUnavailable
from passivbot import logging
from utils import symbol_to_coin


def apply_bitunix_endpoint_override(
    config: dict, endpoint_override: Any | None
) -> dict:
    """Apply the native Bitunix REST override contract to a client config."""
    resolved = dict(config)
    if endpoint_override is None:
        return resolved
    unsupported_keys = set(endpoint_override.rest_url_overrides) - {"api"}
    if unsupported_keys:
        raise CustomEndpointConfigError(
            "bitunix: unsupported REST URL override key(s): "
            + ", ".join(sorted(unsupported_keys))
            + "; use 'api' for the native REST base"
        )
    rest_urls = endpoint_override.apply_to_api_urls(
        {"api": BitunixClient.REST_URL}
    )
    resolved["restUrl"] = rest_urls["api"]
    headers = dict(resolved.get("headers") or {})
    headers.update(endpoint_override.rest_extra_headers)
    resolved["headers"] = headers
    return resolved


def _float(value: Any, *, field: str, default: float | None = None) -> float:
    if value in (None, ""):
        if default is not None:
            return default
        raise ValueError(f"Bitunix response missing {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Bitunix response has non-finite {field}")
    return result


def _int(value: Any, *, field: str, default: int | None = None) -> int:
    if value in (None, ""):
        if default is not None:
            return default
        raise ValueError(f"Bitunix response missing {field}")
    return int(value)


def _decimal_string(value: Any) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("Bitunix request contains a non-finite number")
    return format(decimal, "f")


def _singleton_or_mapping(data: Any, *, endpoint: str) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    raise ValueError(f"Bitunix {endpoint} response has invalid data shape")


class BitunixClient:
    """Minimal async Bitunix futures client with a CCXT-compatible live boundary."""

    id = "bitunix"
    name = "Bitunix"
    precisionMode = 4  # ccxt.TICK_SIZE
    # The venue documents 10 requests/sec per UID and IP. Keep headroom for
    # rolling-window enforcement and other Passivbot account-state consumers.
    rateLimit = 160
    version = "v1"

    REST_URL = "https://fapi.bitunix.com"
    PUBLIC_WS_URL = "wss://fapi.bitunix.com/public/"
    PRIVATE_WS_URL = "wss://fapi.bitunix.com/private/"
    MAX_PAGE_SIZE = 100
    MAX_KLINE_PAGE_SIZE = 200
    MAX_OPEN_ORDER_PAGES = 1000
    MAX_FILL_PAGES = 1000
    TICKER_MAX_AGE_MS = 30_000
    TICKER_WAIT_SECONDS = 8.0
    MAX_DEPTH_FALLBACK_SYMBOLS = 8
    # Bitunix does not expose fee rates through the trading-pairs endpoint.
    # Use the documented VIP0 futures rates as conservative live-planning
    # inputs; higher VIP tiers pay equal or lower fees.
    DEFAULT_MAKER_FEE = 0.0002
    DEFAULT_TAKER_FEE = 0.0006
    RESERVED_AUTH_HEADERS = frozenset({"api-key", "nonce", "timestamp", "sign"})
    ORDER_STATUS_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9_]{0,31}")
    PENDING_STATUS_WARNING_INTERVAL_SECONDS = 300.0

    has = {
        "cancelOrder": True,
        "createOrder": True,
        "fetchBalance": True,
        "fetchMyTrades": True,
        "fetchOHLCV": True,
        "fetchOpenOrders": True,
        "fetchOrder": True,
        "fetchPositions": True,
        "fetchTickers": True,
        "setLeverage": True,
        "setMarginMode": True,
        "setPositionMode": True,
        "watchOrders": False,
    }

    timeframes = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }
    timeframe_milliseconds = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
        "3d": 3 * 24 * 60 * 60_000,
        "1w": 7 * 24 * 60 * 60_000,
        "1M": 30 * 24 * 60 * 60_000,
    }
    BALANCE_TRANSITION_REASON = "balance_consistency_check"

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.apiKey = str(config.get("apiKey") or config.get("key") or "")
        self.secret = str(config.get("secret") or "")
        self.rest_url = str(config.get("restUrl") or self.REST_URL).rstrip("/")
        configured_headers = {
            str(key): str(value)
            for key, value in dict(config.get("headers") or {}).items()
        }
        reserved_headers = sorted(
            key
            for key in configured_headers
            if key.lower() in self.RESERVED_AUTH_HEADERS
        )
        if reserved_headers:
            raise ValueError(
                "Bitunix headers may not override reserved authentication header(s): "
                + ", ".join(reserved_headers)
            )
        self.headers = configured_headers
        self.timeout = int(config.get("timeout") or 30_000)
        self.enableRateLimit = bool(config.get("enableRateLimit", True))
        self.ws_enabled = bool(config.get("wsEnabled", True))
        self.options: dict[str, Any] = {}
        self.markets: dict[str, dict] = {}
        self.markets_by_id: dict[str, dict] = {}
        self.symbols: list[str] = []
        self._session: aiohttp.ClientSession | None = None
        self._rate_lock = asyncio.Lock()
        self._last_request_monotonic = 0.0
        self._position_ids: dict[tuple[str, str], str] = {}
        self._ticker_cache: dict[str, dict] = {}
        self._ticker_received_monotonic: dict[str, float] = {}
        self._ticker_tasks: list[asyncio.Task] = []
        self._ticker_subscription_ids: tuple[str, ...] = ()
        self._ticker_ready = asyncio.Event()
        self._pending_status_warning_monotonic: dict[str, float] = {}
        self._accepted_balance_components: tuple[float, float, float] | None = None
        self._pending_balance_components: tuple[float, float, float] | None = None
        self._closed = False

    @staticmethod
    def _balance_values_close(left: float, right: float) -> bool:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)

    @classmethod
    def _balance_components_close(
        cls,
        left: tuple[float, float, float] | None,
        right: tuple[float, float, float],
    ) -> bool:
        return bool(
            left is not None
            and all(
                cls._balance_values_close(left_value, right_value)
                for left_value, right_value in zip(left, right)
            )
        )

    def _clear_pending_balance_transition(self) -> None:
        self._pending_balance_components = None

    def _defer_balance_candidate(
        self,
        current: tuple[float, float, float],
        *,
        defer_message: str,
    ) -> None:
        if not self._balance_components_close(
            self._pending_balance_components, current
        ):
            self._pending_balance_components = current
            logging.warning(defer_message)
        raise AuthoritativeSurfaceUnavailable(
            "balance", self.BALANCE_TRANSITION_REASON
        )

    def _available_is_transfer_reconciled(
        self,
        *,
        available: float,
        transfer: float | None,
        bonus: float,
        cross_unrealized_pnl: float,
    ) -> bool:
        """Corroborate available funds with Bitunix's maximum-transfer metric.

        Bitunix excludes non-transferable bonus funds and cross-margin unrealized
        losses from the amount transferable out of futures. Isolated PnL remains
        confined to its position margin and therefore must not offset a cross
        loss in this equation. Rearranging that relationship gives an
        exchange-calculated cross-check for ``available``. The known inconsistent
        account response changed ``available`` alone, so this remains false even
        if that response is repeated across a restart.
        """
        # ``transfer`` is floored at zero. At that floor it only proves that
        # ``available + min(cross_unrealized_pnl, 0) - bonus <= 0``; it cannot
        # independently disambiguate a locked-fund duplication. Keep the sample
        # unavailable until a positive transfer value restores an exact cross-check.
        if transfer is None or transfer <= 0.0:
            return False
        expected_available = transfer + bonus - min(cross_unrealized_pnl, 0.0)
        return self._balance_values_close(available, expected_available)

    def _validate_balance_transition(
        self,
        *,
        available: float,
        frozen: float,
        margin: float,
        transfer: float | None,
        bonus: float,
        cross_unrealized_pnl: float,
    ) -> None:
        """Reject Bitunix's transient duplication of locked funds into available.

        The known inconsistent response moves ``available`` by exactly the
        unchanged locked amount while continuing to report that amount as locked.
        Reconcile every new tuple with nonzero locked funds against Bitunix's
        separately calculated maximum-transfer value. This covers both restart
        and a transition that changes frozen or margin components while the
        inconsistent response is active. Keep the last accepted components
        unchanged until a response with locked funds passes the transfer
        reconciliation, even when its visible components match the last
        accepted tuple.
        """
        current = (available, frozen, margin)
        previous = self._accepted_balance_components
        transfer_reconciled = self._available_is_transfer_reconciled(
            available=available,
            transfer=transfer,
            bonus=bonus,
            cross_unrealized_pnl=cross_unrealized_pnl,
        )
        current_used = frozen + margin
        if current_used > 0.0 and not transfer_reconciled:
            self._defer_balance_candidate(
                current,
                defer_message=(
                    "[balance] Bitunix locked-fund balance did not reconcile "
                    "with maximum transfer; action=defer_authoritative_state"
                ),
            )
            return

        if self._balance_components_close(previous, current):
            if self._pending_balance_components is not None:
                logging.info(
                    "[balance] Bitunix balance returned to the last trusted state; "
                    "action=resume_authoritative_state"
                )
                self._clear_pending_balance_transition()
            return

        if self._pending_balance_components is not None:
            if transfer_reconciled:
                logging.info(
                    "[balance] Bitunix balance passed maximum-transfer reconciliation; "
                    "action=resume_authoritative_state"
                )
            else:
                logging.info(
                    "[balance] Bitunix balance cleared locked funds; "
                    "action=resume_authoritative_state"
                )
            self._clear_pending_balance_transition()
        self._accepted_balance_components = current

    def milliseconds(self) -> int:
        return int(time.time() * 1000)

    def nonce(self) -> int:
        return self.milliseconds()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout / 1000.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _throttle(self, *, cancel: bool = False) -> None:
        if not self.enableRateLimit:
            return
        minimum_spacing = 0.21 if cancel else self.rateLimit / 1000.0
        async with self._rate_lock:
            now = time.monotonic()
            delay = self._last_request_monotonic + minimum_spacing - now
            if delay > 0.0:
                await asyncio.sleep(delay)
            self._last_request_monotonic = time.monotonic()

    def _signed_headers(
        self,
        params: dict[str, Any] | None,
        body_text: str,
    ) -> dict[str, str]:
        if not self.apiKey or not self.secret:
            raise AuthenticationError("Bitunix API key and secret are required")
        nonce = uuid.uuid4().hex
        timestamp = str(self.milliseconds())
        query_text = "".join(
            f"{key}{value}"
            for key, value in sorted((params or {}).items())
            if value is not None
        )
        digest = hashlib.sha256(
            f"{nonce}{timestamp}{self.apiKey}{query_text}{body_text}".encode()
        ).hexdigest()
        signature = hashlib.sha256(f"{digest}{self.secret}".encode()).hexdigest()
        return {
            "api-key": self.apiKey,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": signature,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_api_error(code: Any, message: Any) -> None:
        code_text = str(code)
        safe_message = str(message or "Bitunix API error")[:240]
        text = f"Bitunix error {code_text}: {safe_message}"
        error_type: type[Exception]
        if code_text in {"403", "10003", "10004", "10007"}:
            error_type = AuthenticationError
        # The documented transport code is 10001. The live trading-pairs
        # endpoint also emits code 1 with the same message; classify both as
        # retryable network failures instead of permanent bad requests.
        elif code_text == "10001" or (
            code_text == "1" and safe_message.strip().lower() == "network error"
        ):
            error_type = NetworkError
        elif code_text in {"10005", "10006", "429"}:
            error_type = RateLimitExceeded
        elif code_text == "20007":
            error_type = OrderNotFound
        elif code_text in {"20003", "20004"}:
            error_type = InsufficientFunds
        elif code_text.startswith("3"):
            error_type = InvalidOrder
        elif code_text.startswith("2") or code_text.startswith("1"):
            error_type = BadRequest
        else:
            error_type = ExchangeError
        error = error_type(text)
        # Generic live diagnostics read only bounded structured attributes.
        # Retaining the numeric/code token here makes failed writes actionable
        # without publishing the response message or payload.
        error.code = code_text
        raise error

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        private: bool = False,
        cancel: bool = False,
    ) -> Any:
        params = {key: value for key, value in (params or {}).items() if value is not None}
        body_text = (
            json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            if body is not None
            else ""
        )
        headers = dict(self.headers)
        if private:
            headers.update(self._signed_headers(params, body_text))
        else:
            headers["Content-Type"] = "application/json"
        query = urlencode(sorted(params.items()))
        url = f"{self.rest_url}{path}" + (f"?{query}" if query else "")
        await self._throttle(cancel=cancel)
        session = await self._get_session()
        try:
            async with session.request(
                method,
                url,
                headers=headers,
                data=body_text if body is not None else None,
            ) as response:
                response_text = await response.text()
                if response.status == 429:
                    raise RateLimitExceeded("Bitunix HTTP 429")
                if response.status in {401, 403}:
                    raise AuthenticationError(f"Bitunix HTTP {response.status}")
                if response.status >= 400:
                    raise NetworkError(f"Bitunix HTTP {response.status}")
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise NetworkError(f"Bitunix request failed: {type(exc).__name__}") from exc
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise NetworkError("Bitunix returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ExchangeError("Bitunix returned a non-object response")
        code = payload.get("code")
        if code not in (0, "0"):
            self._raise_api_error(code, payload.get("msg"))
        if "data" not in payload:
            raise ExchangeError("Bitunix success response missing data")
        return payload["data"]

    async def load_markets(self, reload: bool = False) -> dict[str, dict]:
        if self.markets and not reload:
            return self.markets
        rows = None
        for attempt in range(3):
            try:
                rows = await self._request(
                    "GET", "/api/v1/futures/market/trading_pairs"
                )
                break
            except NetworkError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (2**attempt))
        if not isinstance(rows, list):
            raise ValueError("Bitunix trading-pairs response is not a list")
        markets: dict[str, dict] = {}
        markets_by_id: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Bitunix trading-pairs response contains a non-object row")
            market_id = str(row.get("symbol") or "")
            base = str(row.get("base") or "")
            quote = str(row.get("quote") or "")
            if not market_id or not base or not quote:
                raise ValueError("Bitunix market metadata missing symbol/base/quote")
            symbol = f"{base}/{quote}:{quote}"
            amount_precision = _int(
                row.get("basePrecision"), field=f"{market_id}.basePrecision"
            )
            price_precision = _int(
                row.get("quotePrecision"), field=f"{market_id}.quotePrecision"
            )
            amount_step = float(Decimal(1).scaleb(-amount_precision))
            price_step = float(Decimal(1).scaleb(-price_precision))
            min_amount = _float(
                row.get("minTradeVolume"),
                field=f"{market_id}.minTradeVolume",
            )
            max_leverage = _float(
                row.get("maxLeverage"), field=f"{market_id}.maxLeverage"
            )
            min_leverage = _float(
                row.get("minLeverage"),
                field=f"{market_id}.minLeverage",
                default=1.0,
            )
            market = {
                "id": market_id,
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "settle": quote,
                "baseId": base,
                "quoteId": quote,
                "settleId": quote,
                "type": "swap",
                "spot": False,
                "margin": False,
                "swap": True,
                "future": False,
                "option": False,
                "contract": True,
                "linear": True,
                "inverse": False,
                "maker": self.DEFAULT_MAKER_FEE,
                "taker": self.DEFAULT_TAKER_FEE,
                "active": (
                    str(row.get("symbolStatus") or "").upper() == "OPEN"
                    and row.get("isApiSupported") is True
                ),
                "contractSize": 1.0,
                "precision": {
                    "amount": amount_step,
                    "price": price_step,
                },
                "limits": {
                    "amount": {"min": min_amount, "max": None},
                    "price": {"min": price_step, "max": None},
                    "cost": {"min": None, "max": None},
                    "leverage": {"min": min_leverage, "max": max_leverage},
                },
                "marginModes": {"cross": True, "isolated": True},
                "info": deepcopy(row),
            }
            markets[symbol] = market
            markets_by_id[market_id] = market
        self.markets = markets
        self.markets_by_id = markets_by_id
        self.symbols = sorted(markets)
        return markets

    async def _ensure_markets(self) -> None:
        if not self.markets:
            await self.load_markets()

    def market(self, symbol: str) -> dict:
        if symbol in self.markets:
            return self.markets[symbol]
        if symbol in self.markets_by_id:
            return self.markets_by_id[symbol]
        raise BadRequest(f"Unknown Bitunix symbol {symbol!r}")

    def safe_symbol(self, market_id: str) -> str:
        market = self.markets_by_id.get(str(market_id))
        if market is None:
            raise ValueError(f"Unknown Bitunix market id {market_id!r}")
        return str(market["symbol"])

    async def fetch_balance(self, params: dict | None = None) -> dict:
        raw = _singleton_or_mapping(
            await self._request(
                "GET",
                "/api/v1/futures/account",
                params={"marginCoin": "USDT"},
                private=True,
            ),
            endpoint="account",
        )
        available = _float(raw.get("available"), field="account.available")
        frozen = _float(raw.get("frozen"), field="account.frozen")
        margin = _float(raw.get("margin"), field="account.margin")
        cross_upnl = _float(
            raw.get("crossUnrealizedPNL"),
            field="account.crossUnrealizedPNL",
        )
        _isolated_upnl = _float(
            raw.get("isolationUnrealizedPNL"),
            field="account.isolationUnrealizedPNL",
        )
        transfer = (
            None
            if raw.get("transfer") in (None, "")
            else _float(raw.get("transfer"), field="account.transfer")
        )
        bonus = _float(raw.get("bonus"), field="account.bonus", default=0.0)
        self._validate_balance_transition(
            available=available,
            frozen=frozen,
            margin=margin,
            transfer=transfer,
            bonus=bonus,
            cross_unrealized_pnl=cross_upnl,
        )
        # Bitunix documents these as disjoint available, order-locked, and
        # position-locked quantities. Unrealized PnL is mark-to-market state,
        # not a realized wallet component; subtracting it makes sizing move
        # with every mark-price update.
        wallet = available + frozen + margin
        if not math.isfinite(wallet) or wallet < 0.0:
            raise ValueError("Bitunix account response produced invalid wallet balance")
        return {
            "info": raw,
            "USDT": {"free": available, "used": frozen + margin, "total": wallet},
            "free": {"USDT": available},
            "used": {"USDT": frozen + margin},
            "total": {"USDT": wallet},
        }

    async def fetch_positions(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> list[dict]:
        await self._ensure_markets()
        requested_ids = (
            {self.market(symbol)["id"] for symbol in symbols} if symbols else None
        )
        query = dict(params or {})
        if requested_ids and len(requested_ids) == 1:
            query["symbol"] = next(iter(requested_ids))
        rows = await self._request(
            "GET",
            "/api/v1/futures/position/get_pending_positions",
            params=query,
            private=True,
        )
        if not isinstance(rows, list):
            raise ValueError("Bitunix positions response is not a list")
        positions = []
        fresh_ids: dict[tuple[str, str], str] = {}
        for row in rows:
            market_id = str(row.get("symbol") or "")
            if requested_ids and market_id not in requested_ids:
                continue
            raw_side = str(row.get("side") or "").upper().rstrip("_")
            side_aliases = {
                "LONG": "long",
                "BUY": "long",
                "SHORT": "short",
                "SELL": "short",
            }
            pside = side_aliases.get(raw_side)
            if pside is None:
                raise ValueError(
                    "Bitunix position missing LONG/SHORT or BUY/SELL side"
                )
            symbol = self.safe_symbol(market_id)
            position_id = str(row.get("positionId") or "")
            if not position_id:
                raise ValueError("Bitunix open position missing positionId")
            fresh_ids[(symbol, pside)] = position_id
            contracts = abs(_float(row.get("qty"), field="position.qty"))
            entry_price = _float(
                row.get("avgOpenPrice") or row.get("entryPrice"),
                field="position entry price",
                default=0.0 if contracts == 0.0 else None,
            )
            if contracts != 0.0 and entry_price <= 0.0:
                raise ValueError(
                    "Bitunix response has non-positive position entry price"
                )
            raw_margin_mode = str(row.get("marginMode") or "").upper()
            if raw_margin_mode not in {"CROSS", "ISOLATION"}:
                raise ValueError(
                    "Bitunix position missing CROSS/ISOLATION margin mode"
                )
            positions.append(
                {
                    "info": deepcopy(row),
                    "id": position_id,
                    "symbol": symbol,
                    "timestamp": _int(
                        row.get("ctime"), field="position.ctime", default=0
                    ),
                    "lastUpdateTimestamp": _int(
                        row.get("mtime"), field="position.mtime", default=0
                    ),
                    "side": pside,
                    "contracts": contracts,
                    "contractSize": 1.0,
                    "entryPrice": entry_price,
                    "markPrice": None,
                    "notional": _float(
                        row.get("entryValue"),
                        field="position.entryValue",
                        default=0.0,
                    ),
                    "leverage": _float(
                        row.get("leverage"), field="position.leverage", default=0.0
                    ),
                    "unrealizedPnl": _float(
                        row.get("unrealizedPNL"),
                        field="position.unrealizedPNL",
                        default=0.0,
                    ),
                    "realizedPnl": _float(
                        row.get("realizedPNL"),
                        field="position.realizedPNL",
                        default=0.0,
                    ),
                    "marginMode": (
                        "isolated"
                        if raw_margin_mode == "ISOLATION"
                        else "cross"
                    ),
                }
            )
        if requested_ids is None:
            self._position_ids = fresh_ids
        else:
            for key in [
                key
                for key in self._position_ids
                if self.market(key[0])["id"] in requested_ids
            ]:
                self._position_ids.pop(key, None)
            self._position_ids.update(fresh_ids)
        return positions

    @staticmethod
    def _order_status(raw_status: Any) -> str:
        # Live REST currently emits ``NEW_`` although the public schema says
        # ``NEW``.  Treat only trailing enum padding as equivalent.
        raw_token = str(raw_status or "").upper()
        if not BitunixClient.ORDER_STATUS_TOKEN_RE.fullmatch(raw_token):
            raise ValueError("Unknown Bitunix order status")
        status = raw_token.rstrip("_")
        mapping = {
            "INIT": "open",
            "NEW": "open",
            "PART_FILLED": "open",
            "FILLED": "closed",
            "CANCELED": "canceled",
            "PART_FILLED_CANCELED": "canceled",
        }
        if status not in mapping:
            raise ValueError(f"Unknown Bitunix order status {status!r}")
        return mapping[status]

    def _pending_order_status(self, raw_status: Any) -> str:
        """Classify one row returned by the authoritative pending endpoint.

        Endpoint membership proves the order is still pending even when the
        venue introduces a new code-like transition enum. Preserve that order
        conservatively until a later complete snapshot removes it. Missing or
        malformed status values remain invalid.
        """
        try:
            return self._order_status(raw_status)
        except ValueError:
            raw_token = str(raw_status or "").upper()
            if not self.ORDER_STATUS_TOKEN_RE.fullmatch(raw_token):
                raise
            token = raw_token.rstrip("_")
            now = time.monotonic()
            last = self._pending_status_warning_monotonic.get(token)
            if last is None or now - last >= self.PENDING_STATUS_WARNING_INTERVAL_SECONDS:
                logging.warning(
                    "[bitunix] [order] pending endpoint returned unrecognized status "
                    "| status=%s action=retain_as_open_until_authoritative_absence",
                    token,
                )
                self._pending_status_warning_monotonic[token] = now
            return "open"

    def _normalize_order(self, row: dict, *, pending_snapshot: bool = False) -> dict:
        market_id = str(row.get("symbol") or "")
        symbol = self.safe_symbol(market_id)
        raw_side = str(row.get("side") or "").upper()
        position_mode = str(row.get("positionMode") or "").upper()
        if position_mode not in {"HEDGE", "ONE_WAY"}:
            raise ValueError("Bitunix order missing HEDGE/ONE_WAY position mode")
        reduce_only = row.get("reduceOnly")
        if not isinstance(reduce_only, bool):
            raise ValueError("Bitunix order missing boolean reduceOnly")
        if position_mode == "HEDGE":
            if raw_side not in {"BUY", "SELL"}:
                raise ValueError("Bitunix hedge order missing BUY/SELL action side")
            action_side = raw_side.lower()
            pside = (
                ("short" if action_side == "buy" else "long")
                if reduce_only
                else ("long" if action_side == "buy" else "short")
            )
        else:
            action_side = raw_side.lower()
            if action_side not in {"buy", "sell"}:
                raise ValueError("Bitunix order missing buy/sell side")
            pside = (
                ("short" if action_side == "buy" else "long")
                if reduce_only
                else ("long" if action_side == "buy" else "short")
            )
        qty = _float(row.get("qty"), field="order.qty")
        if qty <= 0.0:
            raise ValueError("Bitunix response has non-positive order.qty")
        filled = abs(
            _float(
                row.get("tradeQty") or row.get("dealAmount"),
                field="order.tradeQty",
                default=0.0,
            )
        )
        timestamp = _int(row.get("ctime"), field="order.ctime")
        order_id = str(row.get("orderId") or "")
        if not order_id:
            raise ValueError("Bitunix order missing orderId")
        order_type = str(
            row.get("orderType") or row.get("type") or ""
        ).lower()
        if order_type not in {"limit", "market"}:
            raise ValueError("Bitunix order missing LIMIT/MARKET order type")
        raw_status = row.get("status") or row.get("orderStatus")
        status = (
            self._pending_order_status(raw_status)
            if pending_snapshot
            else self._order_status(raw_status)
        )
        price_required = order_type == "limit" or status == "open"
        price = _float(
            row.get("price"),
            field="order.price",
            default=0.0 if not price_required else None,
        )
        if price_required and price <= 0.0:
            raise ValueError("Bitunix response has non-positive order.price")
        info = deepcopy(row)
        info["positionSide"] = pside.upper()
        info["reduceOnly"] = reduce_only
        return {
            "info": info,
            "id": order_id,
            "clientOrderId": str(row.get("clientId") or ""),
            "timestamp": timestamp,
            "lastTradeTimestamp": _int(
                row.get("mtime"), field="order.mtime", default=timestamp
            ),
            "symbol": symbol,
            "type": order_type,
            "timeInForce": str(row.get("effect") or "").upper() or None,
            "postOnly": str(row.get("effect") or "").upper() == "POST_ONLY",
            "reduceOnly": reduce_only,
            "side": action_side,
            "price": price,
            "amount": qty,
            "filled": filled,
            "remaining": max(0.0, qty - filled),
            "average": _float(
                row.get("averagePrice"),
                field="order.averagePrice",
                default=0.0,
            )
            or None,
            "status": status,
            "fee": (
                {
                    "currency": "USDT",
                    "cost": _float(row.get("fee"), field="order.fee"),
                }
                if row.get("fee") not in (None, "")
                else None
            ),
        }

    async def _fetch_order_page(
        self, symbol: str | None, skip: int, limit: int, end_time: int
    ) -> tuple[list[dict], int]:
        await self._ensure_markets()
        params: dict[str, Any] = {
            "skip": skip,
            "limit": limit,
            "endTime": end_time,
        }
        if symbol:
            params["symbol"] = self.market(symbol)["id"]
        data = await self._request(
            "GET",
            "/api/v1/futures/trade/get_pending_orders",
            params=params,
            private=True,
        )
        if not isinstance(data, dict) or not isinstance(data.get("orderList"), list):
            raise ValueError("Bitunix open-orders response has invalid shape")
        try:
            total = _int(data.get("total"), field="open-orders total")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Bitunix response has invalid open-orders total"
            ) from exc
        if total < 0:
            raise ValueError("Bitunix response has negative open-orders total")
        return data["orderList"], total

    async def fetch_open_orders(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        page_size = min(self.MAX_PAGE_SIZE, max(1, int(limit or self.MAX_PAGE_SIZE)))
        snapshot_end = self.milliseconds()
        rows: dict[str, dict] = {}
        skip = 0
        expected_total: int | None = None
        for _ in range(self.MAX_OPEN_ORDER_PAGES):
            page, total = await self._fetch_order_page(
                symbol, skip, page_size, snapshot_end
            )
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise ValueError(
                    "Bitunix open-orders total changed during pagination"
                )
            for row in page:
                if not isinstance(row, dict):
                    raise ValueError(
                        "Bitunix open-orders response contains a non-object row"
                    )
                order_id = str(row.get("orderId") or "")
                if not order_id:
                    raise ValueError("Bitunix open-orders row missing orderId")
                rows[order_id] = row
            skip += len(page)
            if skip > total:
                raise ValueError(
                    "Bitunix open-orders total is smaller than returned rows"
                )
            if not page and skip < total:
                raise ValueError(
                    "Bitunix open-orders pagination ended before total"
                )
            if skip == total:
                break
        else:
            raise RuntimeError(
                "Bitunix open-orders pagination exceeded safety limit"
            )
        if expected_total is None or len(rows) != expected_total:
            raise ValueError(
                "Bitunix open-orders pagination returned duplicate or incomplete rows"
            )
        return sorted(
            (
                self._normalize_order(row, pending_snapshot=True)
                for row in rows.values()
            ),
            key=lambda order: order["timestamp"],
        )

    async def fetch_order(
        self, order_id: str, symbol: str | None = None, params: dict | None = None
    ) -> dict:
        query = {"orderId": str(order_id)}
        if not order_id and params and params.get("clientId"):
            query = {"clientId": str(params["clientId"])}
        row = await self._request(
            "GET",
            "/api/v1/futures/trade/get_order_detail",
            params=query,
            private=True,
        )
        return self._normalize_order(_singleton_or_mapping(row, endpoint="order detail"))

    async def _position_id(self, symbol: str, pside: str) -> str:
        position_id = self._position_ids.get((symbol, pside))
        if position_id:
            return position_id
        await self.fetch_positions([symbol])
        position_id = self._position_ids.get((symbol, pside))
        if not position_id:
            raise InvalidOrder(
                f"Bitunix close for {symbol} {pside} has no live positionId"
            )
        return position_id

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict:
        await self._ensure_markets()
        params = dict(params or {})
        pside = str(params.pop("positionSide", "")).lower()
        if pside not in {"long", "short"}:
            raise InvalidOrder("Bitunix order requires LONG/SHORT positionSide")
        reduce_only = bool(params.pop("reduceOnly", False))
        trade_side = "CLOSE" if reduce_only else "OPEN"
        raw_side = "BUY" if pside == "long" else "SELL"
        order_type = str(type).upper()
        if order_type not in {"LIMIT", "MARKET"}:
            raise InvalidOrder(f"Unsupported Bitunix order type {type!r}")
        body: dict[str, Any] = {
            "symbol": self.market(symbol)["id"],
            "qty": _decimal_string(abs(amount)),
            "side": raw_side,
            "tradeSide": trade_side,
            "orderType": order_type,
            "reduceOnly": reduce_only,
        }
        if order_type == "LIMIT":
            if price is None or float(price) <= 0.0:
                raise InvalidOrder("Bitunix limit order requires a positive price")
            body["price"] = _decimal_string(price)
            body["effect"] = str(
                params.pop("effect", params.pop("timeInForce", "GTC"))
            ).upper()
        client_id = str(params.pop("clientOrderId", params.pop("clientId", "")))
        if client_id:
            body["clientId"] = client_id
        if reduce_only:
            body["positionId"] = str(
                params.pop("positionId", "")
                or await self._position_id(symbol, pside)
            )
        if params:
            raise InvalidOrder(
                f"Unsupported Bitunix order params: {','.join(sorted(params))}"
            )
        data = _singleton_or_mapping(
            await self._request(
                "POST",
                "/api/v1/futures/trade/place_order",
                body=body,
                private=True,
            ),
            endpoint="place order",
        )
        order_id = str(data.get("orderId") or "")
        if not order_id:
            raise ExchangeError("Bitunix place-order response missing orderId")
        if order_type == "MARKET":
            # Bitunix acknowledges placement with identifiers only. A market
            # order may already be terminal when the response arrives, so do
            # not fabricate an unpriced open-order row from the request.
            return {
                "info": {
                    **data,
                    "positionSide": pside.upper(),
                    "reduceOnly": reduce_only,
                },
                "id": order_id,
                "clientOrderId": str(data.get("clientId") or client_id),
                "timestamp": self.milliseconds(),
                "symbol": symbol,
                "type": "market",
                "side": str(side).lower(),
                "price": None,
                "amount": abs(float(amount)),
                "filled": None,
                "remaining": None,
                "status": None,
                "reduceOnly": reduce_only,
            }
        raw = {
            **body,
            **data,
            # The placement request uses Bitunix's hedge-position side, while
            # order responses expose the actual buy/sell action.
            "side": str(side).upper(),
            "positionMode": "HEDGE",
            "marginMode": "",
            "status": "NEW",
            "ctime": self.milliseconds(),
            "tradeQty": "0",
        }
        return self._normalize_order(raw)

    async def cancel_order(
        self, order_id: str, symbol: str | None = None, params: dict | None = None
    ) -> dict:
        if not symbol:
            raise BadRequest("Bitunix cancel_order requires symbol")
        body = {
            "symbol": self.market(symbol)["id"],
            "orderList": [{"orderId": str(order_id)}],
        }
        data = await self._request(
            "POST",
            "/api/v1/futures/trade/cancel_orders",
            body=body,
            private=True,
            cancel=True,
        )
        if not isinstance(data, dict):
            raise ExchangeError("Bitunix cancel response has invalid shape")
        success = data.get("successList") or []
        failures = data.get("failureList") or []
        succeeded = any(
            str(row.get("orderId") or "") == str(order_id)
            for row in success
            if isinstance(row, dict)
        )
        if not succeeded:
            failure = next(
                (
                    row
                    for row in failures
                    if isinstance(row, dict)
                    and str(row.get("orderId") or "") == str(order_id)
                ),
                {},
            )
            code = failure.get("errorCode") or failure.get("code") or "20007"
            self._raise_api_error(code, failure.get("errorMsg") or failure.get("msg"))
        return {
            "id": str(order_id),
            "symbol": symbol,
            "status": "canceled",
            "info": deepcopy(data),
        }

    def _normalize_trade(self, row: dict) -> dict:
        market_id = str(row.get("symbol") or "")
        symbol = self.safe_symbol(market_id)
        raw_side = str(row.get("side") or "").upper()
        position_mode = str(row.get("positionMode") or "").upper()
        if position_mode not in {"HEDGE", "ONE_WAY"}:
            raise ValueError("Bitunix fill missing HEDGE/ONE_WAY position mode")
        reduce_only = row.get("reduceOnly")
        if not isinstance(reduce_only, bool):
            raise ValueError("Bitunix fill missing boolean reduceOnly")
        if position_mode == "HEDGE":
            if raw_side not in {"BUY", "SELL"}:
                raise ValueError("Bitunix fill missing BUY/SELL action side")
            action_side = raw_side.lower()
            pside = (
                ("short" if action_side == "buy" else "long")
                if reduce_only
                else ("long" if action_side == "buy" else "short")
            )
        else:
            action_side = raw_side.lower()
            if action_side not in {"buy", "sell"}:
                raise ValueError("Bitunix fill missing buy/sell side")
            pside = (
                ("short" if action_side == "buy" else "long")
                if reduce_only
                else ("long" if action_side == "buy" else "short")
            )
        trade_id = str(row.get("tradeId") or "")
        order_id = str(row.get("orderId") or "")
        if not trade_id or not order_id:
            raise ValueError("Bitunix fill missing tradeId/orderId")
        qty = abs(_float(row.get("qty"), field="fill.qty"))
        price = _float(row.get("price"), field="fill.price")
        timestamp = _int(row.get("ctime"), field="fill.ctime")
        fee = _float(row.get("fee"), field="fill.fee", default=0.0)
        info = deepcopy(row)
        info["positionSide"] = pside.upper()
        info["reduceOnly"] = reduce_only
        return {
            "info": info,
            "id": trade_id,
            "order": order_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "type": str(row.get("orderType") or row.get("type") or "").lower(),
            "side": action_side,
            "takerOrMaker": str(row.get("roleType") or "").lower() or None,
            "price": price,
            "amount": qty,
            "cost": qty * price,
            "fee": {"currency": "USDT", "cost": fee},
            "fees": [{"currency": "USDT", "cost": fee}],
            "clientOrderId": str(row.get("clientId") or ""),
        }

    async def fetch_my_trades(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        await self._ensure_markets()
        params = dict(params or {})
        until = params.pop("until", None)
        explicit_end_time = params.pop("endTime", None)
        snapshot_end = int(
            until
            if until is not None
            else explicit_end_time
            if explicit_end_time is not None
            else self.milliseconds()
        )
        page_size = min(self.MAX_PAGE_SIZE, max(1, int(limit or self.MAX_PAGE_SIZE)))
        query: dict[str, Any] = {
            "skip": 0,
            "limit": page_size,
            "endTime": snapshot_end,
        }
        if symbol:
            query["symbol"] = self.market(symbol)["id"]
        if since is not None:
            query["startTime"] = int(since)
        query.update(params)
        rows: dict[str, dict] = {}
        expected_total: int | None = None
        for _ in range(self.MAX_FILL_PAGES):
            data = await self._request(
                "GET",
                "/api/v1/futures/trade/get_history_trades",
                params=dict(query),
                private=True,
            )
            if not isinstance(data, dict) or not isinstance(data.get("tradeList"), list):
                raise ValueError("Bitunix fill-history response has invalid shape")
            page = data["tradeList"]
            try:
                total = _int(
                    data.get("total"),
                    field="fill-history total",
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Bitunix response has invalid fill-history total"
                ) from exc
            if total < 0:
                raise ValueError(
                    "Bitunix response has negative fill-history total"
                )
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise ValueError(
                    "Bitunix fill-history total changed during pagination"
                )
            for row in page:
                trade_id = str(row.get("tradeId") or "")
                if not trade_id:
                    raise ValueError("Bitunix fill-history row missing tradeId")
                rows[trade_id] = row
            query["skip"] += len(page)
            if query["skip"] > total:
                raise ValueError(
                    "Bitunix fill-history total is smaller than returned rows"
                )
            if not page and query["skip"] < total:
                raise ValueError(
                    "Bitunix fill-history pagination ended before total"
                )
            if query["skip"] == total:
                break
        else:
            raise RuntimeError("Bitunix fill-history pagination exceeded safety limit")
        if expected_total is None or len(rows) != expected_total:
            raise ValueError(
                "Bitunix fill-history pagination returned duplicate or incomplete rows"
            )
        return sorted(
            (self._normalize_trade(row) for row in rows.values()),
            key=lambda trade: trade["timestamp"],
        )

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[list[float]]:
        await self._ensure_markets()
        if timeframe not in self.timeframes:
            raise BadRequest(f"Unsupported Bitunix OHLCV timeframe {timeframe!r}")
        request_limit = min(
            self.MAX_KLINE_PAGE_SIZE,
            max(1, int(limit or self.MAX_KLINE_PAGE_SIZE)),
        )
        query: dict[str, Any] = {
            "symbol": self.market(symbol)["id"],
            "interval": self.timeframes[timeframe],
            "limit": request_limit,
            "type": "LAST_PRICE",
        }
        if since is not None:
            query["startTime"] = int(since)
        params = dict(params or {})
        requested_until = params.pop("until", None)
        if since is not None:
            # Bitunix ignores startTime for page anchoring and always returns
            # the rows immediately preceding endTime. Derive an end bound from
            # CCXT's forward-page contract so live pagination cannot skip to
            # the newest tail.
            derived_until = (
                int(since)
                + request_limit * self.timeframe_milliseconds[timeframe]
            )
            query["endTime"] = min(
                derived_until,
                int(requested_until)
                if requested_until is not None
                else derived_until,
            )
        elif requested_until is not None:
            query["endTime"] = int(requested_until)
        query.update(params)
        rows = await self._request(
            "GET", "/api/v1/futures/market/kline", params=query
        )
        if not isinstance(rows, list):
            raise ValueError("Bitunix kline response is not a list")
        normalized: dict[int, list[float]] = {}
        for row in rows:
            timestamp = _int(row.get("time"), field="kline.time")
            if since is not None and timestamp < int(since):
                continue
            if query.get("endTime") is not None and timestamp > int(query["endTime"]):
                continue
            normalized[timestamp] = [
                timestamp,
                _float(row.get("open"), field="kline.open"),
                _float(row.get("high"), field="kline.high"),
                _float(row.get("low"), field="kline.low"),
                _float(row.get("close"), field="kline.close"),
                # Bitunix's live payload names are inverted relative to their
                # units: quoteVol is base quantity, while baseVol is quote
                # notional (baseVol / quoteVol is approximately price).
                _float(row.get("quoteVol"), field="kline.quoteVol"),
            ]
            if normalized[timestamp][5] < 0.0:
                raise ValueError("Bitunix response has negative kline.quoteVol")
        return [normalized[key] for key in sorted(normalized)]

    async def _fetch_depth_ticker(self, symbol: str) -> dict:
        data = await self._request(
            "GET",
            "/api/v1/futures/market/depth",
            params={"symbol": self.market(symbol)["id"], "limit": 1},
        )
        if not isinstance(data, dict):
            raise ValueError("Bitunix depth response has invalid shape")
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids or not asks:
            raise ValueError(f"Bitunix depth missing top of book for {symbol}")
        bid = _float(bids[0][0], field="depth.bid")
        ask = _float(asks[0][0], field="depth.ask")
        if bid <= 0.0 or ask <= 0.0 or bid > ask:
            raise ValueError(f"Bitunix depth returned invalid top of book for {symbol}")
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": (bid + ask) / 2.0,
            "timestamp": self.milliseconds(),
            "source": "bitunix_depth_mid",
            "info": data,
        }

    async def _ticker_loop(self, market_ids: tuple[str, ...]) -> None:
        reconnect_delay = 1.0
        while not self._closed:
            try:
                await self._ensure_markets()
                if not market_ids:
                    raise ValueError("Bitunix has no active websocket ticker symbols")
                session = await self._get_session()
                async with session.ws_connect(
                    self.PUBLIC_WS_URL, heartbeat=20.0, receive_timeout=45.0
                ) as ws:
                    await ws.send_json(
                        {
                            "op": "subscribe",
                            "args": [
                                {"symbol": market_id, "ch": "tickers"}
                                for market_id in market_ids
                            ],
                        }
                    )
                    reconnect_delay = 1.0
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            payload = json.loads(message.data)
                            if payload.get("ch") != "tickers":
                                continue
                            data = payload.get("data")
                            rows = data if isinstance(data, list) else [data]
                            received_at = self.milliseconds()
                            received_monotonic = time.monotonic()
                            timestamp = int(payload.get("ts") or received_at)
                            for row in rows:
                                if not isinstance(row, dict):
                                    continue
                                market_id = str(row.get("s") or "")
                                market = self.markets_by_id.get(market_id)
                                if market is None:
                                    continue
                                bid = _float(row.get("bd"), field="ticker.bd")
                                ask = _float(row.get("ak"), field="ticker.ak")
                                last = _float(row.get("la"), field="ticker.la")
                                if bid <= 0.0 or ask <= 0.0 or last <= 0.0 or bid > ask:
                                    continue
                                self._ticker_cache[market["symbol"]] = {
                                    "symbol": market["symbol"],
                                    "bid": bid,
                                    "ask": ask,
                                    "last": last,
                                    "timestamp": timestamp,
                                    "info": deepcopy(row),
                                }
                                self._ticker_received_monotonic[
                                    market["symbol"]
                                ] = received_monotonic
                            if self._ticker_cache:
                                self._ticker_ready.set()
                        elif message.type in {
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        }:
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.debug(
                    "[ws] bitunix public ticker reconnect | error_type=%s",
                    type(exc).__name__,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(30.0, reconnect_delay * 2.0)

    def _ensure_ticker_tasks(self) -> None:
        if not self.ws_enabled:
            return
        market_ids = tuple(
            sorted(
                str(market["id"])
                for market in self.markets.values()
                if market.get("active")
            )
        )
        if self._ticker_tasks and all(
            not task.done() for task in self._ticker_tasks
        ) and market_ids == self._ticker_subscription_ids:
            return
        for task in self._ticker_tasks:
            task.cancel()
        self._ticker_tasks = []
        self._ticker_subscription_ids = market_ids
        self._ticker_ready.clear()
        for index in range(0, len(market_ids), 300):
            chunk = tuple(market_ids[index : index + 300])
            self._ticker_tasks.append(
                asyncio.create_task(
                    self._ticker_loop(chunk),
                    name=f"bitunix-public-tickers-{index // 300 + 1}",
                )
            )

    def _ticker_is_fresh(
        self, symbol: str, now_monotonic: float
    ) -> bool:
        if symbol not in self._ticker_cache:
            return False
        received_monotonic = self._ticker_received_monotonic.get(symbol)
        if received_monotonic is None:
            return False
        age_ms = (now_monotonic - received_monotonic) * 1000.0
        return 0.0 <= age_ms <= self.TICKER_MAX_AGE_MS

    async def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> dict[str, dict]:
        await self._ensure_markets()
        desired = list(
            dict.fromkeys(
                symbols
                or [
                    symbol
                    for symbol, market in self.markets.items()
                    if market.get("active")
                ]
            )
        )
        for symbol in desired:
            self.market(symbol)
        if not self.ws_enabled:
            if symbols is None:
                raise NetworkError(
                    "Bitunix REST-only ticker mode requires explicit symbols"
                )
            if len(desired) > self.MAX_DEPTH_FALLBACK_SYMBOLS:
                raise NetworkError(
                    "Bitunix REST-only ticker request has "
                    f"{len(desired)} symbols; depth requests limited to "
                    f"{self.MAX_DEPTH_FALLBACK_SYMBOLS}"
                )
            return {
                symbol: await self._fetch_depth_ticker(symbol)
                for symbol in desired
            }
        self._ensure_ticker_tasks()
        deadline = time.monotonic() + self.TICKER_WAIT_SECONDS
        while time.monotonic() < deadline:
            now_monotonic = time.monotonic()
            if all(
                self._ticker_is_fresh(symbol, now_monotonic)
                for symbol in desired
            ):
                break
            try:
                await asyncio.wait_for(
                    self._ticker_ready.wait(),
                    timeout=min(0.25, max(0.0, deadline - time.monotonic())),
                )
                self._ticker_ready.clear()
            except asyncio.TimeoutError:
                pass
        now_monotonic = time.monotonic()
        result = {
            symbol: deepcopy(self._ticker_cache[symbol])
            for symbol in desired
            if self._ticker_is_fresh(symbol, now_monotonic)
        }
        if symbols:
            missing = [symbol for symbol in desired if symbol not in result]
            if len(missing) > self.MAX_DEPTH_FALLBACK_SYMBOLS:
                raise NetworkError(
                    "Bitunix ticker websocket missing "
                    f"{len(missing)} symbols; REST depth fallback limited to "
                    f"{self.MAX_DEPTH_FALLBACK_SYMBOLS}"
                )
            for symbol in missing:
                result[symbol] = await self._fetch_depth_ticker(symbol)
        if not result:
            raise NetworkError("Bitunix ticker websocket produced no current quotes")
        return result

    async def fetch_position_mode(self) -> dict:
        raw = _singleton_or_mapping(
            await self._request(
                "GET",
                "/api/v1/futures/account",
                params={"marginCoin": "USDT"},
                private=True,
            ),
            endpoint="account",
        )
        mode = str(raw.get("positionMode") or "").upper()
        if mode not in {"HEDGE", "ONE_WAY"}:
            raise ValueError("Bitunix account response missing positionMode")
        return {"hedged": mode == "HEDGE", "info": raw}

    async def set_position_mode(
        self, hedged: bool, symbol: str | None = None, params: dict | None = None
    ) -> dict:
        mode = "HEDGE" if hedged else "ONE_WAY"
        data = await self._request(
            "POST",
            "/api/v1/futures/account/change_position_mode",
            body={"positionMode": mode},
            private=True,
        )
        return {"code": 0, "positionMode": mode, "info": data}

    async def fetch_leverage_margin_mode(self, symbol: str) -> dict:
        data = await self._request(
            "GET",
            "/api/v1/futures/account/get_leverage_margin_mode",
            params={"symbol": self.market(symbol)["id"], "marginCoin": "USDT"},
            private=True,
        )
        return _singleton_or_mapping(data, endpoint="leverage/margin mode")

    async def set_margin_mode(
        self, margin_mode: str, symbol: str | None = None, params: dict | None = None
    ) -> dict:
        if not symbol:
            raise BadRequest("Bitunix set_margin_mode requires symbol")
        normalized = str(margin_mode).lower()
        if normalized not in {"cross", "isolated", "isolation"}:
            raise BadRequest(f"Invalid Bitunix margin mode {margin_mode!r}")
        raw_mode = "CROSS" if normalized == "cross" else "ISOLATION"
        data = await self._request(
            "POST",
            "/api/v1/futures/account/change_margin_mode",
            body={
                "symbol": self.market(symbol)["id"],
                "marginCoin": "USDT",
                "marginMode": raw_mode,
            },
            private=True,
        )
        return {"code": 0, "marginMode": raw_mode, "info": data}

    async def set_leverage(
        self, leverage: int, symbol: str | None = None, params: dict | None = None
    ) -> dict:
        if not symbol:
            raise BadRequest("Bitunix set_leverage requires symbol")
        leverage_int = int(leverage)
        data = await self._request(
            "POST",
            "/api/v1/futures/account/change_leverage",
            body={
                "symbol": self.market(symbol)["id"],
                "marginCoin": "USDT",
                "leverage": leverage_int,
            },
            private=True,
        )
        return {"code": 0, "leverage": leverage_int, "info": data}

    async def close(self) -> None:
        self._closed = True
        for task in self._ticker_tasks:
            task.cancel()
        if self._ticker_tasks:
            await asyncio.gather(*self._ticker_tasks, return_exceptions=True)
        if self._session is not None and not self._session.closed:
            await self._session.close()


class BitunixOrderStream:
    """Native private-order and multiplexed public-candle WebSocket boundary."""

    id = "bitunix"
    has = {"watchOrders": True, "watchOHLCV": True}
    PING_INTERVAL_SECONDS = 15.0
    KLINE_CHANNELS = {"1m": "market_kline_1min"}
    MAX_KLINE_SUBSCRIPTIONS = 300
    KLINE_QUEUE_SIZE = 4
    MAX_CONSECUTIVE_MALFORMED_KLINES = 5
    KLINE_SILENCE_TIMEOUT_SECONDS = 45.0

    def __init__(self, rest: BitunixClient):
        self.rest = rest
        self.options: dict[str, Any] = {}
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._last_ping_monotonic = time.monotonic()
        self._ohlcv_ws: aiohttp.ClientWebSocketResponse | None = None
        self._ohlcv_task: asyncio.Task | None = None
        self._ohlcv_sockets: dict[int, aiohttp.ClientWebSocketResponse] = {}
        self._ohlcv_tasks: dict[int, asyncio.Task] = {}
        self._ohlcv_queues: dict[str, asyncio.Queue] = {}
        self._ohlcv_fallback_pending: set[str] = set()
        self._ohlcv_closed = False

    async def _receive_with_keepalive(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> aiohttp.WSMessage:
        """Receive one frame while satisfying Bitunix's JSON ping contract."""
        while True:
            elapsed = time.monotonic() - self._last_ping_monotonic
            timeout = max(0.0, self.PING_INTERVAL_SECONDS - elapsed)
            try:
                return await asyncio.wait_for(ws.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                ping_timestamp = int(time.time())
                await ws.send_json({"op": "ping", "ping": ping_timestamp})
                self._last_ping_monotonic = time.monotonic()

    def _login_payload(self) -> dict:
        if not self.rest.apiKey or not self.rest.secret:
            raise AuthenticationError("Bitunix API key and secret are required")
        nonce = uuid.uuid4().hex
        timestamp = int(time.time())
        digest = hashlib.sha256(
            f"{nonce}{timestamp}{self.rest.apiKey}".encode()
        ).hexdigest()
        signature = hashlib.sha256(
            f"{digest}{self.rest.secret}".encode()
        ).hexdigest()
        return {
            "op": "login",
            "args": [
                {
                    "apiKey": self.rest.apiKey,
                    "timestamp": timestamp,
                    "nonce": nonce,
                    "sign": signature,
                }
            ],
        }

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        session = await self.rest._get_session()
        ws = await session.ws_connect(
            self.rest.PRIVATE_WS_URL, heartbeat=20.0, receive_timeout=45.0
        )
        await ws.send_json(self._login_payload())
        while True:
            message = await ws.receive()
            if message.type != aiohttp.WSMsgType.TEXT:
                await ws.close()
                raise NetworkError("Bitunix private websocket closed during login")
            payload = json.loads(message.data)
            if payload.get("op") == "login":
                if payload.get("code") not in (None, 0, "0") or payload.get("success") is False:
                    await ws.close()
                    raise AuthenticationError("Bitunix private websocket login failed")
                break
        await ws.send_json({"op": "subscribe", "args": [{"ch": "order"}]})
        self._last_ping_monotonic = time.monotonic()
        self._ws = ws
        return ws

    async def watch_orders(self) -> list[dict]:
        ws = self._ws
        if ws is None or ws.closed:
            ws = await self._connect()
        while True:
            message = await self._receive_with_keepalive(ws)
            if message.type == aiohttp.WSMsgType.TEXT:
                payload = json.loads(message.data)
                if payload.get("ch") != "order":
                    continue
                data = payload.get("data")
                rows = data if isinstance(data, list) else [data]
                enriched = []
                for row in rows:
                    if not isinstance(row, dict) or not row.get("orderId"):
                        fallback = {"info": {"raw": deepcopy(row)}}
                        if isinstance(row, dict) and row.get("symbol"):
                            fallback["symbol"] = self.rest.safe_symbol(
                                str(row["symbol"])
                            )
                        enriched.append(fallback)
                        continue
                    try:
                        enriched.append(
                            await self.rest.fetch_order(
                                str(row["orderId"]),
                                self.rest.safe_symbol(str(row.get("symbol") or "")),
                            )
                        )
                    except (OrderNotFound, ValueError):
                        fallback = deepcopy(row)
                        fallback["symbol"] = self.rest.safe_symbol(
                            str(row.get("symbol") or "")
                        )
                        enriched.append(fallback)
                if enriched:
                    return enriched
            elif message.type in {
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                self._ws = None
                raise NetworkError("Bitunix private order websocket disconnected")

    def _normalize_ohlcv_payload(self, payload: Any) -> tuple[str, list[list[float]]]:
        if not isinstance(payload, dict):
            raise ValueError("Bitunix kline websocket payload is not an object")
        if payload.get("ch") != self.KLINE_CHANNELS["1m"]:
            raise ValueError("Bitunix kline websocket payload has unexpected channel")
        market_id = str(payload.get("symbol") or "")
        symbol = self.rest.safe_symbol(market_id)
        timestamp = _int(payload.get("ts"), field="kline websocket ts")
        if timestamp <= 0:
            raise ValueError("Bitunix kline websocket has invalid timestamp")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Bitunix kline websocket data is not an object")
        open_price = _float(data.get("o"), field="kline websocket open")
        high = _float(data.get("h"), field="kline websocket high")
        low = _float(data.get("l"), field="kline websocket low")
        close = _float(data.get("c"), field="kline websocket close")
        base_volume = _float(data.get("b"), field="kline websocket base volume")
        if min(open_price, high, low, close) <= 0.0 or base_volume < 0.0:
            raise ValueError("Bitunix kline websocket has invalid price or volume")
        if low > min(open_price, close) or high < max(open_price, close) or low > high:
            raise ValueError("Bitunix kline websocket has inconsistent OHLC values")
        bucket_timestamp = timestamp - timestamp % self.rest.timeframe_milliseconds["1m"]
        return symbol, [
            [bucket_timestamp, open_price, high, low, close, base_volume]
        ]

    @staticmethod
    def _queue_latest(queue: asyncio.Queue, item: Any, *, clear: bool = False) -> None:
        if clear:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        while queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(item)

    def _broadcast_ohlcv_error(
        self,
        error: BaseException,
        *,
        symbols: Iterable[str] | None = None,
    ) -> None:
        targets = tuple(self._ohlcv_queues) if symbols is None else tuple(symbols)
        for symbol in targets:
            self._queue_ohlcv_fallback(symbol, error)

    def _queue_ohlcv_fallback(self, symbol: str, error: BaseException) -> bool:
        queue = self._ohlcv_queues.get(symbol)
        if queue is None:
            return False
        self._ohlcv_fallback_pending.add(symbol)
        self._queue_latest(queue, error, clear=True)
        return True

    @staticmethod
    def _subscription_ack_market_ids(payload: dict[str, Any]) -> set[str]:
        market_ids: set[str] = set()
        symbol = payload.get("symbol")
        if symbol not in (None, ""):
            market_ids.add(str(symbol))
        for key in ("arg", "args"):
            value = payload.get(key)
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if isinstance(row, dict) and row.get("symbol") not in (None, ""):
                    market_ids.add(str(row["symbol"]))
        return market_ids

    def _ohlcv_shard_symbols(self, shard_index: int) -> list[str]:
        start = shard_index * self.MAX_KLINE_SUBSCRIPTIONS
        end = start + self.MAX_KLINE_SUBSCRIPTIONS
        return sorted(self._ohlcv_queues)[start:end]

    def _required_ohlcv_shards(self) -> int:
        if not self._ohlcv_queues:
            return 0
        return (
            len(self._ohlcv_queues) + self.MAX_KLINE_SUBSCRIPTIONS - 1
        ) // self.MAX_KLINE_SUBSCRIPTIONS

    def _ensure_ohlcv_task(self) -> None:
        if self._ohlcv_closed:
            raise NetworkError("Bitunix kline websocket is closed")
        for shard_index in range(self._required_ohlcv_shards()):
            task = self._ohlcv_tasks.get(shard_index)
            if task is not None and not task.done():
                continue
            task = asyncio.create_task(
                self._ohlcv_loop(shard_index),
                name=f"bitunix-public-klines-{shard_index}",
            )
            self._ohlcv_tasks[shard_index] = task
            if shard_index == 0:
                self._ohlcv_task = task

    async def _ohlcv_loop(self, shard_index: int) -> None:
        current_task = asyncio.current_task()
        # Establish the shard's failure scope before market/session/socket setup.
        # Any of those awaits may fail and must wake every current watcher.
        assigned_symbols = set(self._ohlcv_shard_symbols(shard_index))
        subscribed_symbols_by_id: dict[str, str] = {}
        try:
            await self.rest._ensure_markets()
            session = await self.rest._get_session()
            async with session.ws_connect(
                self.rest.PUBLIC_WS_URL,
                heartbeat=20.0,
                receive_timeout=45.0,
            ) as ws:
                self._ohlcv_sockets[shard_index] = ws
                if shard_index == 0:
                    self._ohlcv_ws = ws
                subscribed_ids: set[str] = set()
                malformed_counts: dict[str, int] = {}
                malformed_warned: set[str] = set()
                last_ping_monotonic = time.monotonic()
                last_frame_monotonic = last_ping_monotonic
                last_symbol_activity: dict[str, float] = {}
                pending_subscribe_ids: set[str] = set()
                while not self._ohlcv_closed:
                    shard_symbols = self._ohlcv_shard_symbols(shard_index)
                    assigned_symbols = set(shard_symbols)
                    desired_by_id = {
                        str(self.rest.market(symbol)["id"]): symbol
                        for symbol in shard_symbols
                    }
                    if not desired_by_id:
                        return
                    now_monotonic = time.monotonic()
                    for market_id in tuple(subscribed_ids):
                        symbol = subscribed_symbols_by_id.get(market_id)
                        if (
                            symbol is None
                            or symbol in self._ohlcv_fallback_pending
                        ):
                            continue
                        last_activity = last_symbol_activity.get(
                            symbol, now_monotonic
                        )
                        if (
                            now_monotonic - last_activity
                            < self.KLINE_SILENCE_TIMEOUT_SECONDS
                        ):
                            continue
                        if self._queue_ohlcv_fallback(
                            symbol,
                            NetworkError(
                                "Bitunix public kline websocket subscription "
                                "became silent"
                            ),
                        ):
                            logging.warning(
                                "[bitunix] [candle] websocket subscription silent "
                                "| symbol=%s action=rest_fallback",
                                symbol,
                            )
                    desired_ids = {
                        market_id
                        for market_id, symbol in desired_by_id.items()
                        if symbol not in self._ohlcv_fallback_pending
                    }
                    removed = sorted(subscribed_ids - desired_ids)
                    added = sorted(desired_ids - subscribed_ids)
                    if removed:
                        await ws.send_json(
                            {
                                "op": "unsubscribe",
                                "args": [
                                    {
                                        "symbol": market_id,
                                        "ch": self.KLINE_CHANNELS["1m"],
                                    }
                                    for market_id in removed
                                ],
                            }
                        )
                        for market_id in removed:
                            pending_subscribe_ids.discard(market_id)
                            removed_symbol = subscribed_symbols_by_id.pop(
                                market_id, None
                            )
                            if removed_symbol is not None:
                                last_symbol_activity.pop(removed_symbol, None)
                    if added:
                        await ws.send_json(
                            {
                                "op": "subscribe",
                                "args": [
                                    {
                                        "symbol": market_id,
                                        "ch": self.KLINE_CHANNELS["1m"],
                                    }
                                    for market_id in added
                                ],
                            }
                        )
                        pending_subscribe_ids.update(added)
                        for market_id in added:
                            symbol = desired_by_id[market_id]
                            subscribed_symbols_by_id[market_id] = symbol
                            last_symbol_activity[symbol] = now_monotonic
                    for market_id in desired_ids & subscribed_ids:
                        subscribed_symbols_by_id[market_id] = desired_by_id[
                            market_id
                        ]
                    subscribed_ids = desired_ids
                    if now_monotonic - last_ping_monotonic >= self.PING_INTERVAL_SECONDS:
                        await ws.send_json({"op": "ping", "ping": int(time.time())})
                        last_ping_monotonic = now_monotonic
                    silence_remaining = (
                        self.KLINE_SILENCE_TIMEOUT_SECONDS
                        - (now_monotonic - last_frame_monotonic)
                    )
                    if silence_remaining <= 0.0:
                        raise NetworkError("Bitunix public kline websocket became silent")
                    try:
                        message = await asyncio.wait_for(
                            ws.receive(), timeout=min(1.0, silence_remaining)
                        )
                    except asyncio.TimeoutError:
                        if (
                            time.monotonic() - last_frame_monotonic
                            >= self.KLINE_SILENCE_TIMEOUT_SECONDS
                        ):
                            raise NetworkError(
                                "Bitunix public kline websocket became silent"
                            )
                        continue
                    last_frame_monotonic = time.monotonic()
                    if message.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(message.data)
                        if isinstance(payload, dict):
                            operation = payload.get("op")
                            if operation == "subscribe":
                                acknowledged_ids = self._subscription_ack_market_ids(
                                    payload
                                )
                                if not acknowledged_ids:
                                    acknowledged_ids = set(pending_subscribe_ids)
                                pending_subscribe_ids.difference_update(
                                    acknowledged_ids
                                )
                                rejected = (
                                    payload.get("code") not in (None, 0, "0")
                                    or payload.get("success") is False
                                )
                                if rejected:
                                    if not acknowledged_ids:
                                        acknowledged_ids = set(subscribed_ids)
                                    for market_id in sorted(acknowledged_ids):
                                        symbol = desired_by_id.get(market_id)
                                        if symbol is None:
                                            symbol = subscribed_symbols_by_id.get(
                                                market_id
                                            )
                                        if symbol is None:
                                            continue
                                        if self._queue_ohlcv_fallback(
                                            symbol,
                                            NetworkError(
                                                "Bitunix public kline websocket "
                                                "subscription was rejected"
                                            ),
                                        ):
                                            logging.warning(
                                                "[bitunix] [candle] websocket "
                                                "subscription rejected | symbol=%s "
                                                "action=rest_fallback",
                                                symbol,
                                            )
                                    subscribed_ids.difference_update(
                                        acknowledged_ids
                                    )
                                else:
                                    for market_id in acknowledged_ids:
                                        symbol = subscribed_symbols_by_id.get(
                                            market_id
                                        )
                                        if symbol is not None:
                                            last_symbol_activity[symbol] = (
                                                last_frame_monotonic
                                            )
                                continue
                            if operation in {
                                "connect",
                                "ping",
                                "pong",
                                "unsubscribe",
                            }:
                                continue
                        try:
                            symbol, rows = self._normalize_ohlcv_payload(payload)
                        except ValueError:
                            # Bitunix has emitted transient internally inconsistent
                            # OHLC updates (for example, low one tick above open).
                            # Never clamp or persist them, and do not sacrifice
                            # unrelated multiplexed symbols. A bounded consecutive
                            # run wakes only this watcher into generic REST fallback.
                            market_id = (
                                str(payload.get("symbol") or "")
                                if isinstance(payload, dict)
                                and payload.get("ch") == self.KLINE_CHANNELS["1m"]
                                else ""
                            )
                            malformed_symbol = desired_by_id.get(market_id)
                            malformed_queue = self._ohlcv_queues.get(
                                malformed_symbol or ""
                            )
                            if malformed_symbol is None or malformed_queue is None:
                                if market_id in self.rest.markets_by_id:
                                    continue
                                raise
                            malformed_counts[malformed_symbol] = (
                                malformed_counts.get(malformed_symbol, 0) + 1
                            )
                            if malformed_symbol not in malformed_warned:
                                logging.warning(
                                    "[bitunix] [candle] malformed kline update dropped "
                                    "| symbol=%s action=drop_update "
                                    "rest_fallback_after=%d",
                                    malformed_symbol,
                                    self.MAX_CONSECUTIVE_MALFORMED_KLINES,
                                )
                                malformed_warned.add(malformed_symbol)
                            if (
                                malformed_counts[malformed_symbol]
                                >= self.MAX_CONSECUTIVE_MALFORMED_KLINES
                            ):
                                self._queue_ohlcv_fallback(
                                    malformed_symbol,
                                    NetworkError(
                                        "Bitunix public kline websocket received "
                                        "consecutive malformed updates"
                                    ),
                                )
                                malformed_counts[malformed_symbol] = 0
                            continue
                        market_id = str(payload.get("symbol") or "")
                        if desired_by_id.get(market_id) != symbol:
                            continue
                        last_symbol_activity[symbol] = last_frame_monotonic
                        malformed_counts[symbol] = 0
                        queue = self._ohlcv_queues.get(symbol)
                        if (
                            queue is not None
                            and symbol not in self._ohlcv_fallback_pending
                        ):
                            self._queue_latest(queue, rows)
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise NetworkError("Bitunix public kline websocket disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.debug(
                "[ws] bitunix public kline reconnect | shard=%d error_type=%s",
                shard_index,
                type(exc).__name__,
            )
            self._broadcast_ohlcv_error(
                NetworkError(
                    "Bitunix public kline websocket failed: "
                    f"{type(exc).__name__}"
                ),
                symbols=(
                    assigned_symbols
                    | set(self._ohlcv_shard_symbols(shard_index))
                    | set(subscribed_symbols_by_id.values())
                ),
            )
        finally:
            if self._ohlcv_sockets.get(shard_index) is not None:
                self._ohlcv_sockets.pop(shard_index, None)
            if self._ohlcv_tasks.get(shard_index) is current_task:
                self._ohlcv_tasks.pop(shard_index, None)
            if shard_index == 0:
                self._ohlcv_ws = None
                if self._ohlcv_task is current_task:
                    self._ohlcv_task = None

    async def watch_ohlcv(self, symbol: str, timeframe: str = "1m") -> list[list[float]]:
        if timeframe not in self.KLINE_CHANNELS:
            raise BadRequest(f"Unsupported Bitunix WebSocket OHLCV timeframe {timeframe!r}")
        await self.rest._ensure_markets()
        self.rest.market(symbol)
        queue = self._ohlcv_queues.get(symbol)
        if queue is None:
            queue = asyncio.Queue(maxsize=self.KLINE_QUEUE_SIZE)
            self._ohlcv_queues[symbol] = queue
        self._ensure_ohlcv_task()
        item = await queue.get()
        if isinstance(item, BaseException):
            self._ohlcv_fallback_pending.discard(symbol)
            raise item
        if item is None:
            raise NetworkError("Bitunix kline websocket subscription was removed")
        return item

    async def un_watch_ohlcv(self, symbol: str, timeframe: str = "1m") -> None:
        if timeframe not in self.KLINE_CHANNELS:
            raise BadRequest(f"Unsupported Bitunix WebSocket OHLCV timeframe {timeframe!r}")
        queue = self._ohlcv_queues.pop(symbol, None)
        self._ohlcv_fallback_pending.discard(symbol)
        if queue is not None:
            self._queue_latest(queue, None, clear=True)
        required_shards = self._required_ohlcv_shards()
        obsolete_tasks = [
            task
            for shard_index, task in tuple(self._ohlcv_tasks.items())
            if shard_index >= required_shards
        ]
        for task in obsolete_tasks:
            task.cancel()
        if obsolete_tasks:
            await asyncio.gather(*obsolete_tasks, return_exceptions=True)
        if required_shards == 0:
            self._ohlcv_task = None
            self._ohlcv_ws = None

    async def close(self) -> None:
        self._ohlcv_closed = True
        for queue in tuple(self._ohlcv_queues.values()):
            self._queue_latest(queue, None, clear=True)
        self._ohlcv_queues.clear()
        self._ohlcv_fallback_pending.clear()
        ohlcv_tasks = tuple(self._ohlcv_tasks.values())
        for task in ohlcv_tasks:
            task.cancel()
        if ohlcv_tasks:
            await asyncio.gather(*ohlcv_tasks, return_exceptions=True)
        self._ohlcv_tasks.clear()
        self._ohlcv_sockets.clear()
        self._ohlcv_task = None
        self._ohlcv_ws = None
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None


class BitunixBot(CCXTBot):
    """Bitunix USDT-margined perpetual-futures connector."""

    MAX_OPEN_ORDERS = 100
    CLIENT_ORDER_ID_PATTERN = re.compile(r"^[.A-Z:/a-z0-9_-]{1,36}$")

    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36
        self.quote = "USDT"
        self.hedge_mode = True

    def create_ccxt_sessions(self) -> None:
        ccxt_config = self._build_ccxt_config()
        ccxt_config["wsEnabled"] = bool(self.ws_enabled)
        ccxt_config = apply_bitunix_endpoint_override(
            ccxt_config, getattr(self, "endpoint_override", None)
        )
        self.cca = BitunixClient(ccxt_config)
        self.cca.options.update(self.user_info.get("options", {}))
        if self.ws_enabled:
            self.ccp = BitunixOrderStream(self.cca)
        else:
            self.ccp = None
            logging.info("bitunix: WebSocket disabled, using REST polling")

    def _market_snapshot_ticker_strategy(self) -> str:
        if not self.ws_enabled:
            return "symbols"
        return super()._market_snapshot_ticker_strategy()

    def _normalize_tickers(self, fetched: dict) -> dict:
        tickers = super()._normalize_tickers(fetched)
        for symbol, ticker in tickers.items():
            raw = fetched[symbol]
            if raw.get("timestamp") is not None:
                ticker["timestamp"] = int(raw["timestamp"])
            source = raw.get("source")
            if isinstance(source, str) and source.strip():
                ticker["source"] = source.strip()
        return tickers

    def _normalize_balance_diagnostics(self, fetched: object) -> dict:
        """Expose only documented Bitunix account components for diagnostics."""
        return normalize_bitunix_balance_composition(fetched)

    async def _exchange_config_write_ready(self) -> bool:
        """Gate position-mode writes on an authoritative Bitunix balance sample."""
        balance_override = getattr(self, "balance_override", None)
        if balance_override is not None:
            try:
                if isinstance(balance_override, bool):
                    raise ValueError("boolean balance override")
                parsed_override = float(balance_override)
            except (TypeError, ValueError):
                raise ValueError(
                    "balance_override must be a positive finite numeric value"
                ) from None
            if not math.isfinite(parsed_override) or parsed_override <= 0.0:
                raise ValueError(
                    "balance_override must be a positive finite numeric value"
                )
            return True
        try:
            await self.cca.fetch_balance()
        except AuthoritativeSurfaceUnavailable as exc:
            if exc.surface != "balance":
                raise
            self._last_authoritative_block_reason = exc.reason
            return False
        return True

    async def update_exchange_config(self) -> None:
        current = await self.cca.fetch_position_mode()
        if current.get("hedged") is True:
            logging.debug("[config] bitunix account already uses HEDGE position mode")
            return
        result = await self.cca.set_position_mode(True)
        logging.info(
            "[config] bitunix set HEDGE position mode result=%s",
            format_exchange_config_response(result),
        )

    async def update_exchange_config_by_symbols(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            current = await self.cca.fetch_leverage_margin_mode(symbol)
            current_mode = str(current.get("marginMode") or "").upper()
            desired_mode = self._get_margin_mode_for_symbol(symbol)
            desired_raw_mode = "CROSS" if desired_mode == "cross" else "ISOLATION"
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
            margin_mode_changed = current_mode != desired_raw_mode
            if margin_mode_changed:
                result = await self.cca.set_margin_mode(desired_mode, symbol)
                logging.info(
                    "[config] %s set Bitunix %s margin result=%s",
                    log_symbol,
                    desired_mode,
                    format_exchange_config_response(result),
                )
            leverage = self._calc_leverage_for_symbol(symbol)
            raw_leverages = [
                current.get("leverage"),
                current.get("longLeverage"),
                current.get("shortLeverage"),
            ]
            known = {
                int(float(value))
                for value in raw_leverages
                if value not in (None, "")
            }
            if margin_mode_changed or known != {leverage}:
                result = await self.cca.set_leverage(leverage, symbol)
                logging.info(
                    "[config] %s set Bitunix leverage=%sx result=%s",
                    log_symbol,
                    leverage,
                    format_exchange_config_response(result),
                )

    def set_market_specific_settings(self) -> None:
        sizing_symbols = sorted(self.symbols_requiring_market_sizing())
        for symbol in sizing_symbols:
            market = self.markets_dict[symbol]
            # Caches written by the initial native connector release predate
            # fee metadata. Upgrade those records in memory so existing users
            # do not need to delete a fresh markets cache manually.
            if market.get("maker_fee") is None and market.get("maker") is None:
                market["maker"] = BitunixClient.DEFAULT_MAKER_FEE
            if market.get("taker_fee") is None and market.get("taker") is None:
                market["taker"] = BitunixClient.DEFAULT_TAKER_FEE
        super().set_market_specific_settings()
        for symbol in sizing_symbols:
            # Fail during initialization, before entering the execution loop,
            # if native metadata ever stops providing mandatory planning fees.
            self._get_exchange_fee_rates(symbol)
            raw = self.markets_dict[symbol].get("info") or {}
            max_leverage = _float(
                raw.get("maxLeverage"), field=f"{symbol}.maxLeverage"
            )
            self.max_leverage[symbol] = int(max_leverage)

    def _build_order_params(self, order: dict) -> dict:
        position_side = str(order.get("position_side") or "").lower()
        if position_side not in {"long", "short"}:
            raise ValueError(
                f"Bitunix order has invalid position_side: {position_side!r}"
            )
        client_order_id = str(order.get("custom_id") or "")
        if not self.CLIENT_ORDER_ID_PATTERN.fullmatch(client_order_id):
            raise ValueError(
                "Bitunix client order id must match ^[.A-Z:/a-z0-9_-]{1,36}$"
            )
        params = {
            "positionSide": position_side.upper(),
            "clientOrderId": client_order_id,
            "reduceOnly": bool(order.get("reduce_only")),
        }
        if order.get("type", "limit") == "limit":
            params["effect"] = (
                "POST_ONLY"
                if require_live_value(self.config, "time_in_force") == "post_only"
                else "GTC"
            )
        return params

    def _get_position_side_for_order(self, order: dict) -> str:
        info = order.get("info") or {}
        position_side = str(info.get("positionSide") or "").lower()
        if position_side not in {"long", "short"}:
            raise ValueError("Bitunix order missing explicit LONG/SHORT positionSide")
        return position_side

    def _canonical_open_order_reduce_only(self, order: dict) -> bool:
        raw = (order.get("info") or {}).get("reduceOnly")
        if not isinstance(raw, bool):
            raise ValueError("Bitunix open order missing boolean reduceOnly")
        return raw

    def _get_position_side_from_trade(self, trade: dict) -> str:
        position_side = str(
            (trade.get("info") or {}).get("positionSide") or ""
        ).lower()
        if position_side not in {"long", "short"}:
            raise ValueError("Bitunix fill missing explicit LONG/SHORT positionSide")
        return position_side

    async def close(self) -> None:
        self.stop_data_maintainers()
        if self.ccp is not None:
            await self.ccp.close()
        await self.cca.close()
        self._close_live_event_pipeline(timeout=2.0)
