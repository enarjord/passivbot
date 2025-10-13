from __future__ import annotations
from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import asyncio
import traceback
import numpy as np
import passivbot_rust as pbr
from utils import ts_to_date, utc_ms
from pure_funcs import (
    floatify,
    calc_hash,
    shorten_custom_id,
)
from procedures import print_async_exception, assert_correct_ccxt_version
from collections import defaultdict
from typing import Any
import hmac
import hashlib
import base64

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


class KucoinBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 40
        self.quote = "USDT"
        self.hedge_mode = False

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
        try:
            self.cca.options["defaultType"] = "swap"
        except Exception:
            pass
        self._apply_endpoint_override(self.cca)

        if self.ws_enabled:
            self.ccp = pro_cls(dict(base_kwargs))
            try:
                self.ccp.options["defaultType"] = "swap"
            except Exception:
                pass
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Kucoin websocket session due to custom endpoint override.")

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = int(elm["limits"]["leverage"]["max"])

    async def watch_ohlcvs_1m(self):
        # print("debug watch_ohlcvs_1m")
        return

    async def watch_ohlcv_1m_single(self, symbol):
        # print('debug watch_ohlcv_1m_single', symbol)
        return

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for order in res:
                    order["position_side"] = self.determine_pos_side(order)
                    order["qty"] = order["amount"]
                self.handle_order_update(res)
            except Exception as e:
                logging.error(f"exception watch_orders {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

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

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol)
            for order in fetched:
                order["position_side"] = self.determine_pos_side(order)
                order["qty"] = order["amount"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions, fetched_balance = None, None
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(),
                self.cca.fetch_balance(),
            )
            positions = []
            for p in fetched_positions:
                positions.append(
                    {
                        **p,
                        **{
                            "symbol": p["symbol"],
                            "position_side": p["side"],
                            "size": float(p["contracts"]),
                            "price": float(p["entryPrice"]),
                        },
                    }
                )
            balance = fetched_balance["info"]["data"]["marginBalance"]
            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(fetched_positions)
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fetch_tickers()
            return fetched
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

    async def fetch_fills(self, start_time=None, end_time=None, limit=None):
        all_fills = []
        ids_seen = set()
        params = {}
        if end_time:
            params["until"] = int(end_time)
        day = 1000 * 60 * 60 * 24
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
                    new_until = self.get_exchange_time() - day
            params["until"] = new_until
            all_fills = fills + all_fills
            if start_time is None or new_until <= start_time + day:
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
        all_ph = []
        ids_seen = set()
        params = {}
        if end_time:
            params["until"] = int(end_time)
        day = 1000 * 60 * 60 * 24
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
                    new_until = self.get_exchange_time() - day
            params["until"] = new_until
            all_ph = ph + all_ph
            if start_time is None or new_until <= start_time + day:
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
                print(f"debug no fills for pos close {symbol} {phd[symbol]}")
                continue
            for p in phd[symbol]:
                with_td = sorted(
                    [x for x in cld[symbol] if x["id"] not in seen_trade_id],
                    key=lambda x: abs(p["lastUpdateTimestamp"] - x["timestamp"]),
                )
                if not with_td:
                    print(f"debug no matching fill for {p}")
                    continue
                best_match = with_td[0]
                matches.append((p, best_match))
                timedelta = best_match["timestamp"] - p["lastUpdateTimestamp"]
                if timedelta > 1000:
                    print(
                        f"debug best match fill and pos close {symbol} timedelta>1000ms: {best_match['timestamp'] - p['lastUpdateTimestamp']}ms"
                    )
                seen_trade_id.add(best_match["id"])
            if len(phd[symbol]) != len(cld[symbol]):
                print(
                    f"debug len mismatch between closes and positions_history for {symbol}: {len(cld[symbol])} {len(phd[symbol])}"
                )
        # add pnls, dedup and return
        deduped = {}
        for p, c in matches:
            c["pnl"] = p["realizedPnl"]
            if c["id"] in deduped:
                print(f"debug unexpected duplicate {c}")
                continue
            deduped[c["id"]] = c
        for t in mt:
            if t["id"] not in deduped:
                deduped[t["id"]] = t

        return sorted(deduped.values(), key=lambda x: x["timestamp"])

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        return {
            "timeInForce": "GTC",
            "reduceOnly": order.get("reduce_only", False),
            "marginMode": "CROSS",
            "clientOid": order.get("custom_id", None),
        }

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return order is not None and order["id"] in executed.get("info", {}).get("data", {}).get(
                "cancelledOrderIds", []
            )
        except:
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
