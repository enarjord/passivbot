from __future__ import annotations

import math
import re
import time

import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

from config.access import require_live_value
from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
from utils import symbol_to_coin


class _WeexSuccessEnvelopeMixin:
    """Accept WEEX's documented mutation success envelope.

    CCXT 4.5.66 treats every response containing ``msg`` as an error, including
    ``{"code": "200", "msg": "success"}`` from margin/leverage mutations.
    Keep the workaround exact so genuine WEEX errors still use CCXT's mapping.
    """

    def handle_errors(
        self,
        code,
        reason,
        url,
        method,
        headers,
        body,
        response,
        request_headers,
        request_body,
    ):
        if isinstance(response, dict):
            response_code = str(response.get("code") or "")
            response_message = str(response.get("msg") or "").lower()
            if response_code == "200" and response_message == "success":
                return None
        return super().handle_errors(
            code,
            reason,
            url,
            method,
            headers,
            body,
            response,
            request_headers,
            request_body,
        )


class AsyncWeex(_WeexSuccessEnvelopeMixin, ccxt_async.weex):
    pass


class ProWeex(_WeexSuccessEnvelopeMixin, ccxt_pro.weex):
    pass


class WeexBot(CCXTBot):
    """WEEX USDT-margined perpetual-futures adapter.

    WEEX V3 uses base-asset quantities at the unified CCXT boundary and
    configures combined-position/margin mode per symbol. Its 24-hour ticker
    feed omits bid/ask, so live quotes come from the contract book-ticker endpoint.
    """

    MAX_OPEN_ORDERS = 100
    CLIENT_ORDER_ID_PATTERN = re.compile(r"^[.A-Z:/a-z0-9_-]{1,36}$")

    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36
        self.quote = "USDT"
        self.hedge_mode = True

    def create_ccxt_sessions(self):
        ccxt_config = self._build_ccxt_config()
        user_options = self.user_info.get("options", {})

        self.cca = AsyncWeex(ccxt_config)
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options.update(user_options)
        self.cca.options["defaultType"] = "swap"
        self._apply_endpoint_override(self.cca)

        if self.ws_enabled:
            self.ccp = ProWeex(ccxt_config)
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options.update(user_options)
            self.ccp.options["defaultType"] = "swap"
            self._apply_endpoint_override(self.ccp)
        else:
            self.ccp = None
            logging.info("weex: WebSocket disabled, using REST polling")

    async def update_exchange_config(self):
        """WEEX has no account-wide position-mode endpoint.

        Position and margin mode are applied per active symbol by
        update_exchange_config_by_symbols() after markets are loaded.
        """
        logging.debug(
            "[config] weex position and margin mode are configured per symbol"
        )

    async def update_exchange_config_by_symbols(self, symbols: list[str]):
        for symbol in symbols:
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            leverage = self._calc_leverage_for_symbol(symbol)
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol

            position_mode = await self.cca.fetch_position_mode(symbol)
            current_margin = await self.cca.fetch_margin_mode(symbol)
            is_separated = bool(position_mode.get("hedged"))
            current_margin_mode = str(
                current_margin.get("marginMode") or ""
            ).lower()

            if is_separated or current_margin_mode != margin_mode:
                started = time.time()
                result = await self.cca.set_position_mode(
                    False,
                    symbol=symbol,
                    params={"marginMode": margin_mode},
                )
                logging.debug(
                    "[config] %s set WEEX combined-position/margin mode result=%s elapsed_ms=%.1f",
                    log_symbol,
                    format_exchange_config_response(result),
                    (time.time() - started) * 1000.0,
                )

            started = time.time()
            result = await self.cca.set_leverage(
                leverage,
                symbol=symbol,
                params={"marginMode": margin_mode},
            )
            logging.debug(
                "[config] %s set WEEX leverage=%sx result=%s elapsed_ms=%.1f",
                log_symbol,
                leverage,
                format_exchange_config_response(result),
                (time.time() - started) * 1000.0,
            )

    def _build_order_params(self, order: dict) -> dict:
        """Build the exact WEEX V3 hedge-order contract.

        WEEX closes are selected by side + positionSide.  The V3 order API
        does not expose reduceOnly, so do not leak that unsupported field into
        the signed request.
        """
        position_side = str(order.get("position_side") or "").lower()
        if position_side not in {"long", "short"}:
            raise ValueError(f"WEEX order has invalid position_side: {position_side!r}")
        client_order_id = str(order.get("custom_id") or "")
        if not self.CLIENT_ORDER_ID_PATTERN.fullmatch(client_order_id):
            raise ValueError(
                "WEEX client order id must match ^[.A-Z:/a-z0-9_-]{1,36}$"
            )
        params = {
            "positionSide": position_side.upper(),
            "clientOrderId": client_order_id,
        }
        if order.get("type", "limit") == "limit":
            params["timeInForce"] = (
                "POST_ONLY"
                if require_live_value(self.config, "time_in_force") == "post_only"
                else "GTC"
            )
        return params

    def _get_position_side_from_trade(self, trade: dict) -> str:
        info = trade.get("info") or {}
        position_side = str(info.get("positionSide") or "").lower()
        if position_side not in {"long", "short"}:
            raise ValueError(
                "weex fill missing explicit LONG/SHORT positionSide"
            )
        return position_side

    async def _do_fetch_tickers(self) -> list[dict]:
        return await self.cca.contract_get_capi_v3_market_ticker_bookticker()

    async def _do_fetch_tickers_for_symbols(self, symbols: list[str]) -> list[dict]:
        # One bulk book-ticker call is cheaper and more consistent than one
        # request per symbol.  Callers filter the normalized mapping.
        return await self._do_fetch_tickers()

    def _normalize_tickers(self, fetched: list[dict] | dict) -> dict:
        rows = fetched if isinstance(fetched, list) else [fetched]
        tickers = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            market_id = str(row.get("symbol") or "")
            symbol = self.symbol_ids_inv.get(market_id)
            if symbol not in self.markets_dict:
                continue
            try:
                bid = float(row["bidPrice"])
                ask = float(row["askPrice"])
                timestamp = int(row["time"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(bid) or not math.isfinite(ask):
                continue
            if bid <= 0.0 or ask <= 0.0 or bid > ask or timestamp <= 0:
                continue
            tickers[symbol] = {
                "bid": bid,
                "ask": ask,
                "last": (bid + ask) / 2.0,
                "timestamp": timestamp,
            }
        return tickers

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.symbols_requiring_market_sizing():
            market = self.markets_dict[symbol]
            if not (
                market.get("swap")
                and market.get("linear")
                and market.get("settle") == self.quote
            ):
                continue
            # WEEX's unified order quantity is already expressed in base units.
            # contractVal is metadata, not a contracts-to-base multiplier here.
            self.c_mults[symbol] = 1.0
            raw_max_leverage = (market.get("limits") or {}).get("leverage", {}).get(
                "max"
            )
            if raw_max_leverage is None:
                raw_max_leverage = (market.get("info") or {}).get("maxLeverage")
            max_leverage = float(raw_max_leverage)
            if not math.isfinite(max_leverage) or max_leverage <= 0.0:
                raise ValueError(
                    f"{symbol}: invalid WEEX max leverage metadata: {raw_max_leverage!r}"
                )
            self.max_leverage[symbol] = int(max_leverage)
