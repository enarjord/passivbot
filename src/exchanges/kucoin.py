from __future__ import annotations
from exchanges.ccxt_bot import CCXTBot
from passivbot import logging
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import asyncio
import passivbot_rust as pbr
from utils import ts_to_date, utc_ms
from procedures import assert_correct_ccxt_version
from collections import defaultdict
import hmac
import hashlib
import base64

calc_order_price_diff = pbr.calc_order_price_diff

# ---------------------------------------------------------------------------
# Broker mixin classes for injecting KC-BROKER-NAME on KuCoin futures requests.
#
# When a broker configuration is provided via the ``options['partner']``
# dictionary (see ``create_ccxt_sessions`` below), CCXT automatically adds
# ``KC-API-PARTNER``, ``KC-API-PARTNER-SIGN`` and ``KC-API-PARTNER-VERIFY``
# headers to private API calls.  However, the human friendly broker name
# (``KC-BROKER-NAME``) is only attached on broker-specific endpoints.  To
# ensure KuCoin attributes futures trades to the broker, we override the
# ``sign`` method to append the broker name on private and futuresPrivate
# requests.  These classes should be used instead of the vanilla CCXT
# exchange classes when broker codes are defined.


class AsyncKucoinBrokerFutures(ccxt_async.kucoinfutures):
    """Asynchronous KuCoin futures exchange with broker tagging support."""

    def __init__(self, config=None):
        super().__init__(config)

    @property
    def checkConflictingProxies(self):
        """Ensure camelCase version always points to snake_case"""
        return self.check_conflicting_proxies

    def sign(self, path, api="public", method="GET", params=None, headers=None, body=None):
        signed = super().sign(path, api, method, params or {}, headers, body)
        try:
            if api in {"private", "futuresPrivate", "broker"}:
                partner_root = getattr(self, "options", {}).get("partner") or {}
                partner_cfg = (
                    partner_root.get("future", partner_root) if isinstance(partner_root, dict) else {}
                )
                broker_name = partner_cfg.get("name") if isinstance(partner_cfg, dict) else None
                if broker_name:
                    hdrs = dict(signed.get("headers") or {})
                    hdrs["KC-BROKER-NAME"] = broker_name
                    signed["headers"] = hdrs
        except Exception:
            pass
        return signed


class ProKucoinBrokerFutures(ccxt_pro.kucoinfutures):
    """Websocket-enabled KuCoin futures exchange with broker tagging support."""

    def __init__(self, config=None):
        super().__init__(config)

    @property
    def checkConflictingProxies(self):
        """Ensure camelCase version always points to snake_case"""
        return self.check_conflicting_proxies

    def sign(self, path, api="public", method="GET", params=None, headers=None, body=None):
        signed = super().sign(path, api, method, params or {}, headers, body)
        try:
            if api in {"private", "futuresPrivate", "broker"}:
                partner_root = getattr(self, "options", {}).get("partner") or {}
                partner_cfg = (
                    partner_root.get("future", partner_root) if isinstance(partner_root, dict) else {}
                )
                broker_name = partner_cfg.get("name") if isinstance(partner_cfg, dict) else None
                if broker_name:
                    hdrs = dict(signed.get("headers") or {})
                    hdrs["KC-BROKER-NAME"] = broker_name
                    signed["headers"] = hdrs
        except Exception:
            pass
        return signed


assert_correct_ccxt_version(ccxt=ccxt_async)


class KucoinBot(CCXTBot):
    MAX_OPEN_ORDERS = 150

    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 40
        self.quote = "USDT"
        self.hedge_mode = True

    def _get_partner_signature(self, timestamp: str) -> str:
        prehash = f"{timestamp}{self.partner}{self.api_key}"
        digest = hmac.new(self.broker_key.encode(), prehash.encode(), hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def create_ccxt_sessions(self) -> None:
        """Initialise CCXT sessions for KuCoin futures with optional broker support.

        If broker codes are defined under ``self.broker_code['futures']``, these
        values are used to configure partner signing so that private/futures
        requests include the correct broker metadata.  Otherwise the bot
        instantiates the standard CCXT classes.
        """
        broker_cfg = self.broker_code.get("futures", {}) if isinstance(self.broker_code, dict) else {}
        partner_id = broker_cfg.get("partner")
        partner_secret = broker_cfg.get("broker-key")
        broker_name = broker_cfg.get("broker-name")
        options = {}
        if partner_id and partner_secret:
            options = {
                "partner": {
                    "future": {
                        "id": partner_id,
                        "secret": partner_secret,
                        "name": broker_name,
                    }
                }
            }
        base_kwargs = {
            "apiKey": self.user_info["key"],
            "secret": self.user_info["secret"],
            "password": self.user_info["passphrase"],
            "enableRateLimit": True,
        }
        if options:
            base_kwargs["options"] = options

        async_cls = (
            AsyncKucoinBrokerFutures
            if partner_id and partner_secret and broker_name
            else ccxt_async.kucoinfutures
        )
        pro_cls = (
            ProKucoinBrokerFutures
            if partner_id and partner_secret and broker_name
            else ccxt_pro.kucoinfutures
        )

        self.cca = async_cls(dict(base_kwargs))
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options["defaultType"] = "swap"
        self._apply_endpoint_override(self.cca)

        if self.ws_enabled:
            self.ccp = pro_cls(dict(base_kwargs))
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options["defaultType"] = "swap"
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Kucoin websocket session due to custom endpoint override.")

    async def watch_ohlcvs_1m(self):
        """KuCoin: No-op - OHLCV websocket not used."""
        return

    async def watch_ohlcv_1m_single(self, symbol):
        """KuCoin: No-op - OHLCV websocket not used."""
        return

    def _get_position_side_for_order(self, order: dict) -> str:
        """KuCoin: Derive position_side from position state."""
        return self.determine_pos_side(order)

    def determine_pos_side(self, order):
        # non hedge mode
        if self.has_position("long", order["symbol"]):
            return "long"
        elif self.has_position("short", order["symbol"]):
            return "short"
        elif order["side"] == "buy":
            return "long"
        elif order["side"] == "sell":
            return "short"
        raise Exception(f"unknown side {order['side']}")

    async def fetch_open_orders(self, symbol: str = None) -> list:
        """KuCoin: Fetch open orders with pagination.

        Returns:
            list: Orders sorted by timestamp with normalized fields.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        open_orders = []
        page_size = 100
        current_page = 1
        while True:
            params = {"pageSize": page_size, "currentPage": current_page}
            fetched = await self.cca.fetch_open_orders(symbol=symbol, params=params)
            if not fetched:
                break
            for order in fetched:
                order["position_side"] = self.determine_pos_side(order)
                order["qty"] = order["amount"]
            open_orders.extend(fetched)
            if len(fetched) < page_size:
                break
            if len(open_orders) >= self.MAX_OPEN_ORDERS:
                break
            current_page += 1
        return sorted(open_orders, key=lambda x: x["timestamp"])

    def _get_balance(self, fetched: dict) -> float:
        """KuCoin uses marginBalance in info.data."""
        return float(fetched["info"]["data"]["marginBalance"])

    async def calc_ideal_orders(self):
        # KuCoin enforces a 150 open-order cap; keep only the closest price targets.
        ideal_orders = await super().calc_ideal_orders()
        flattened = []
        for symbol, orders in ideal_orders.items():
            if not orders:
                continue
            market_price = await self.cm.get_current_close(symbol, max_age_ms=10_000)
            for order in orders:
                price_diff = calc_order_price_diff(order["side"], order["price"], market_price)
                flattened.append((price_diff, symbol, order))
        limit = getattr(self, "MAX_OPEN_ORDERS", 150)
        flattened.sort(key=lambda x: x[0])
        trimmed = flattened[:limit] if limit and limit > 0 else flattened
        filtered: dict[str, list] = {symbol: [] for symbol in self.active_symbols}
        for _, symbol, order in trimmed:
            filtered.setdefault(symbol, []).append(order)
        for symbol in ideal_orders:
            filtered.setdefault(symbol, ideal_orders[symbol])
        return filtered

    async def fetch_fills(self, start_time=None, end_time=None, limit=None):
        if start_time is None:
            logging.warning(
                "fetch_fills called without start_time; "
                "consider setting pnls_max_lookback_days in config to limit fetch"
            )
        all_fills = []
        params = {}
        if end_time:
            params["until"] = int(end_time)
        day = 1000 * 60 * 60 * 24
        now_ms = self.get_exchange_time()
        while True:
            fills = await self.cca.fetch_my_trades(params=params)
            if fills:
                new_until = fills[0]["timestamp"]
                if "until" in params and new_until == params["until"]:
                    new_until -= day
            else:
                if "until" in params:
                    new_until = params["until"] - day
                else:
                    new_until = now_ms - day
            params["until"] = new_until
            all_fills = fills + all_fills
            if start_time is not None and new_until <= start_time + day:
                break
            logging.info(
                f"fetched {len(fills)} fill{'' if len(fills) == 1 else 's'}"
                f" {ts_to_date(new_until)[:19]}"
            )
        for i in range(len(all_fills)):
            all_fills[i]["qty"] = all_fills[i]["amount"]
            all_fills[i]["pnl"] = 0.0
            if all_fills[i]["side"] == "buy":
                all_fills[i]["position_side"] = (
                    "long" if float(all_fills[i]["info"]["closeFeePay"]) == 0.0 else "short"
                )
            elif all_fills[i]["side"] == "sell":
                all_fills[i]["position_side"] = (
                    "short" if float(all_fills[i]["info"]["closeFeePay"]) == 0.0 else "long"
                )
            else:
                raise Exception(f"invalid side {all_fills[i]}")
        deduped = {x["id"]: x for x in all_fills}
        if start_time:
            deduped = {k: v for k, v in deduped.items() if v["timestamp"] >= start_time}
        if end_time:
            deduped = {k: v for k, v in deduped.items() if v["timestamp"] <= end_time}
        return sorted(deduped.values(), key=lambda x: x["timestamp"])

    async def fetch_positions_history(self, start_time=None, end_time=None, limit=None):
        if start_time is None:
            logging.warning(
                "fetch_positions_history called without start_time; "
                "consider setting pnls_max_lookback_days in config to limit fetch"
            )
        all_ph = []
        params = {}
        if end_time:
            params["until"] = int(end_time)
        day = 1000 * 60 * 60 * 24
        now_ms = self.get_exchange_time()
        while True:
            ph = await self.cca.fetch_positions_history(params=params)
            ph = sorted(ph, key=lambda x: x["lastUpdateTimestamp"])
            if ph:
                new_until = ph[0]["lastUpdateTimestamp"]
                if "until" in params and new_until == params["until"]:
                    new_until -= day
            else:
                if "until" in params:
                    new_until = params["until"] - day
                else:
                    new_until = now_ms - day
            params["until"] = new_until
            all_ph = ph + all_ph
            if start_time is not None and new_until <= start_time + day:
                break
            logging.info(
                f"fetched {len(ph)} pos histor{'y' if len(ph) == 1 else 'ies'}"
                f" {ts_to_date(new_until)[:19]}"
            )
        deduped = {x["info"]["closeId"]: x for x in all_ph}
        if start_time:
            deduped = {k: v for k, v in deduped.items() if v["lastUpdateTimestamp"] >= start_time}
        if end_time:
            deduped = {k: v for k, v in deduped.items() if v["lastUpdateTimestamp"] <= end_time}
        return sorted(deduped.values(), key=lambda x: x["lastUpdateTimestamp"])

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        # fetch fills...
        mt = await self.fetch_fills(start_time=start_time, end_time=end_time)
        closes = [
            x
            for x in mt
            if (x["side"] == "sell" and x["position_side"] == "long")
            or (x["side"] == "buy" and x["position_side"] == "short")
        ]
        if not closes:
            return mt
        # fetch pos history for pnls
        ph = await self.fetch_positions_history(
            start_time=closes[0]["timestamp"] - 60000, end_time=closes[-1]["timestamp"] + 60000
        )

        # match up...
        cld, phd = defaultdict(list), defaultdict(list)
        for x in closes:
            cld[x["symbol"]].append(x)
        for x in ph:
            phd[x["symbol"]].append(x)
        matches = []
        seen_trade_id = set()
        for symbol in phd:
            if symbol not in cld:
                logging.debug(f"no fills for pos close {symbol} {phd[symbol]}")
                continue
            for p in phd[symbol]:
                with_td = sorted(
                    [x for x in cld[symbol] if x["id"] not in seen_trade_id],
                    key=lambda x: abs(p["lastUpdateTimestamp"] - x["timestamp"]),
                )
                if not with_td:
                    logging.debug(f"no matching fill for {p}")
                    continue
                best_match = with_td[0]
                matches.append((p, best_match))
                timedelta = best_match["timestamp"] - p["lastUpdateTimestamp"]
                if timedelta > 1000:
                    logging.debug(
                        f"best match fill and pos close {symbol} timedelta>1000ms: {best_match['timestamp'] - p['lastUpdateTimestamp']}ms"
                    )
                seen_trade_id.add(best_match["id"])
            if len(phd[symbol]) != len(cld[symbol]):
                logging.debug(
                    f"len mismatch between closes and positions_history for {symbol}: {len(cld[symbol])} {len(phd[symbol])}"
                )
        # add pnls, dedup and return
        deduped = {}
        for p, c in matches:
            c["pnl"] = p["realizedPnl"]
            if c["id"] in deduped:
                logging.debug(f"unexpected duplicate {c}")
                continue
            deduped[c["id"]] = c
        for t in mt:
            if t["id"] not in deduped:
                deduped[t["id"]] = t

        return sorted(deduped.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for KuCoin.

        Returns:
            list: Fill events with normalized fields.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time, limit=limit)
        events = []
        for fill in fills:
            events.append(
                {
                    "id": fill.get("id") or fill.get("orderId"),
                    "timestamp": fill.get("timestamp"),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "position_side": fill.get("position_side"),
                    "qty": fill.get("qty") or fill.get("amount"),
                    "price": fill.get("price"),
                    "pnl": fill.get("pnl"),
                    "fee": fill.get("fee"),
                    "info": fill.get("info"),
                }
            )
        return events

    def _build_order_params(self, order: dict) -> dict:
        return {
            "timeInForce": "GTC",
            "reduceOnly": order.get("reduce_only", False),
            "marginMode": "CROSS",
            "clientOid": order.get("custom_id", None),
            "positionSide": order.get("position_side", "").upper(),
        }

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return order is not None and order["id"] in executed.get("info", {}).get("data", {}).get(
                "cancelledOrderIds", []
            )
        except (KeyError, TypeError, AttributeError):
            return False

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ticker("BTC/USDT:USDT")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def update_exchange_config(self):
        """Ensure account-level settings (hedge mode) are applied."""
        try:
            # Hedge mode enabled so both long/short can coexist.
            if hasattr(self.cca, "set_position_mode"):
                res = await self.cca.set_position_mode(True)
                logging.info(f"set_position_mode hedged=True {res}")
            else:
                logging.info("set_position_mode not supported by current KuCoin client; continuing")
        except Exception as e:
            logging.warning(f"set_position_mode hedged=True not applied: {e}")

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call = []
        for symbol in symbols:
            try:
                params = {
                    "marginMode": "cross",
                    "symbol": symbol,
                }
                coros_to_call.append(
                    (
                        symbol,
                        "set_margin_mode",
                        asyncio.create_task(self.cca.set_margin_mode(**params)),
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error set_margin_mode {e}")
        for symbol, task_name, task in coros_to_call:
            res = None
            to_print = ""
            try:
                res = await task
                to_print += f"{task_name} {res}"
            except Exception as e:
                logging.error(f"{symbol} error {task_name} {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

        coros_to_call = []
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.config_get(["live", "leverage"], symbol=symbol),
                        )
                    ),
                    "symbol": symbol,
                    "params": {"marginMode": "cross"},
                }
                coros_to_call.append(
                    (symbol, "set_leverage", asyncio.create_task(self.cca.set_leverage(**params)))
                )
            except Exception as e:
                logging.error(f"{symbol}: error set_margin_mode {e}")
        for symbol, task_name, task in coros_to_call:
            res = None
            to_print = ""
            try:
                res = await task
                to_print += f"{task_name} {res}"
            except Exception as e:
                logging.error(f"{symbol} error {task_name} {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")
