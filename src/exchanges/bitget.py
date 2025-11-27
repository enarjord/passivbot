from passivbot import Passivbot, logging, custom_id_to_snake, clip_by_timestamp
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from utils import utc_ms, ts_to_date
from config_utils import require_live_value
from pure_funcs import (
    multi_replace,
    floatify,
    calc_hash,
    shorten_custom_id,
)
import passivbot_rust as pbr

calc_order_price_diff = pbr.calc_order_price_diff
from procedures import print_async_exception, assert_correct_ccxt_version
import passivbot_rust as pbr

assert_correct_ccxt_version(ccxt=ccxt_async)


def deduce_side_pside(fill: dict) -> tuple[str, str]:
    """Infer standard ``(side, pside)`` for a Bitget fill payload."""

    trade_side = str(fill.get("tradeSide", "")).lower()
    raw_side = str(fill.get("side", "")).lower()
    pos_mode = str(fill.get("posMode", "")).lower()

    def _canonical(side: str, pside: str) -> tuple[str, str]:
        side = side or ("buy" if pside == "long" else "sell")
        return side, pside

    # Normalize hedge mode strings first.
    if pos_mode == "hedge_mode":
        if "close_long" in trade_side:
            return _canonical("sell", "long")
        if "close_short" in trade_side:
            return _canonical("buy", "short")
        if trade_side == "open":
            if raw_side == "sell":
                return _canonical("sell", "short")
            return _canonical("buy", "long")
        if trade_side == "close":
            if raw_side == "buy":
                return _canonical("sell", "long")
            if raw_side == "sell":
                return _canonical("buy", "long")
            return _canonical("sell", "long")
        if "long" in trade_side:
            return _canonical("buy", "long")
        if "short" in trade_side:
            return _canonical("sell", "short")

    # One-way mode ("single") encodes direction explicitly.
    if "buy_single" in trade_side:
        return _canonical("buy", "long")
    if "sell_single" in trade_side:
        return _canonical("sell", "short")
    if "reduce_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "reduce_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "burst_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "burst_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "delivery_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "delivery_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "dte_sys_adl_buy_in_single_side_mode" in trade_side:
        return _canonical("buy", "long")
    if "dte_sys_adl_sell_in_single_side_mode" in trade_side:
        return _canonical("sell", "short")

    # Generic fallback: look for keywords.
    if "close_long" in trade_side:
        return _canonical("sell", "long")
    if "close_short" in trade_side:
        return _canonical("buy", "short")
    if "buy" in trade_side:
        return _canonical("buy", "long")
    if "sell" in trade_side:
        return _canonical("sell", "short")

    if raw_side == "sell":
        return _canonical("sell", "long")
    if raw_side == "buy":
        return _canonical("buy", "long")

    return _canonical(raw_side or "buy", "long")


class BitgetBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.position_side_map = {
            "buy": {"open": "long", "close": "short"},
            "sell": {"open": "short", "close": "long"},
        }
        self.custom_id_max_length = 64

    def create_ccxt_sessions(self):
        if self.ws_enabled:
            self.ccp = getattr(ccxt_pro, self.exchange)(
                {
                    "apiKey": self.user_info["key"],
                    "secret": self.user_info["secret"],
                    "password": self.user_info["passphrase"],
                    "enableRateLimit": True,
                }
            )
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options["defaultType"] = "swap"
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Bitget websocket session due to custom endpoint override.")
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
                "enableRateLimit": True,
            }
        )
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options["defaultType"] = "swap"
        self._apply_endpoint_override(self.cca)

    def get_symbol_id(self, symbol):
        """Return the exchange-native identifier for `symbol`, caching defaults. Overrides from parent"""
        try:
            return self.symbol_ids[symbol]
            if symbol in self.symbol_ids:
                return self.symbol_ids[symbol]
            # use heuristics to guess
            guess = symbol.replace("/USDT:", "")
            logging.warning(
                f"failed to map {symbol} to its exchange specifec symbol id. Using heursistics to guess {guess}"
            )
            return guess
        except:
            logging.info(f"debug: symbol {symbol} missing from self.symbol_ids. Using {symbol}")
            self.symbol_ids[symbol] = symbol
            return symbol

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

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = max(
                5.1, 0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = res[i]["info"]["posSide"]
                    res[i]["qty"] = res[i]["amount"]
                    res[i]["side"] = self.determine_side(res[i])
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_side(self, order: dict) -> str:
        if "info" in order:
            if all([x in order["info"] for x in ["tradeSide", "reduceOnly", "posSide"]]):
                if order["info"]["tradeSide"] == "close":
                    if order["info"]["posSide"] == "long":
                        return "sell"
                    elif order["info"]["posSide"] == "short":
                        return "buy"
                elif order["info"]["tradeSide"] == "open":
                    if order["info"]["posSide"] == "long":
                        return "buy"
                    elif order["info"]["posSide"] == "short":
                        return "sell"
        raise Exception(f"failed to determine side {order}")

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders()
            for i in range(len(fetched)):
                fetched[i]["position_side"] = fetched[i]["info"]["posSide"]
                fetched[i]["qty"] = fetched[i]["amount"]
                fetched[i]["custom_id"] = fetched[i]["clientOrderId"]
                fetched[i]["side"] = self.determine_side(fetched[i])
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions = None
        try:
            fetched_positions = await self.cca.fetch_positions()
            for i in range(len(fetched_positions)):
                fetched_positions[i]["position_side"] = fetched_positions[i]["side"]
                fetched_positions[i]["size"] = fetched_positions[i]["contracts"]
                fetched_positions[i]["price"] = fetched_positions[i]["entryPrice"]
            return fetched_positions
        except Exception as e:
            logging.error(f"error fetching positions {e}")
            print_async_exception(fetched_positions)
            traceback.print_exc()
            return False

    async def fetch_balance(self):
        fetched_balance = None
        try:
            fetched_balance = await self.cca.fetch_balance()
            balance_info = [x for x in fetched_balance["info"] if x["marginCoin"] == self.quote][0]
            if (
                "assetMode" in balance_info
                and "unionTotalMargin" in balance_info
                and balance_info["assetMode"] == "union"
            ):
                balance = float(balance_info["unionTotalMargin"]) - float(
                    balance_info["unrealizedPL"]
                )
                if not hasattr(self, "previous_hysteresis_balance"):
                    self.previous_hysteresis_balance = balance
                self.previous_hysteresis_balance = pbr.hysteresis(
                    balance,
                    self.previous_hysteresis_balance,
                    self.hyst_pct,
                )
                balance = self.previous_hysteresis_balance
            else:
                balance = float(balance_info["available"])
            return balance
        except Exception as e:
            logging.error(f"error fetching balance {e}")
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            tickers = await self.cca.fetch_tickers()
            return tickers
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            if "bybit does not have market symbol" in str(e):
                # ccxt is raising bad symbol error
                # restart might help...
                raise Exception("ccxt gives bad symbol error... attempting bot restart")
            return False

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        fetched = None
        try:
            fetched = await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        params = {"productType": "USDT-FUTURES"}
        if start_time:
            start_time = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        if limit:
            params["limit"] = min(100, limit)
        side_pos_side_map = {"buy": "long", "sell": "short"}
        data_d = {}
        while True:
            fetched = await self.cca.private_mix_get_v2_mix_order_fill_history(params)
            end_id = fetched["data"]["endId"]
            data = fetched["data"]["fillList"]
            if data is None:
                # print("debug a")
                break
            if not data:
                # print("debug b")
                break
            with_hashes = {calc_hash(x): x for x in data}
            if all([h in data_d for h in with_hashes]):
                # print("debug c")
                break
            for h, x in with_hashes.items():
                data_d[h] = x
                data_d[h]["pnl"] = float(x["profit"])
                data_d[h]["price"] = float(x["price"])
                data_d[h]["amount"] = float(x["baseVolume"])
                data_d[h]["id"] = x["tradeId"]
                data_d[h]["timestamp"] = float(x["cTime"])
                data_d[h]["datetime"] = ts_to_date(data_d[h]["timestamp"])
                data_d[h]["position_side"] = side_pos_side_map[x["side"]]
                data_d[h]["symbol"] = self.get_symbol_id_inv(x["symbol"])
            if start_time is None:
                # print("debug d")
                break
            last_ts = float(data[-1]["cTime"])
            if last_ts < start_time:
                # print("debug e")
                break
            logging.info(f"fetched {len(data)} fills until {ts_to_date(last_ts)[:19]}")
            params["endTime"] = int(last_ts)
        return sorted(data_d.values(), key=lambda x: x["timestamp"])

    async def _throttled_order_detail(self, order_id: str, symbol: str):
        """Rate limited wrapper for clientOid lookups."""
        if not hasattr(self, "_detail_fetch_timestamps"):
            self._detail_fetch_timestamps = []
        n_sec = 5
        max_calls = 30
        while True:
            now = utc_ms()
            self._detail_fetch_timestamps = [
                ts for ts in self._detail_fetch_timestamps if ts > now - n_sec * 1000
            ]
            if len(self._detail_fetch_timestamps) < max_calls:
                self._detail_fetch_timestamps.append(now)
                break
            await asyncio.sleep(0.1)
        print("fetching order detail for", symbol, order_id)
        return await self.cca.private_mix_get_v2_mix_order_detail(
            params={
                "productType": "USDT-FUTURES",
                "orderId": order_id,
                "symbol": symbol,
            }
        )

    async def _ensure_client_oid_for_event(self, event: dict) -> None:
        if not event.get("id"):
            return
        if not hasattr(self, "_client_oid_cache"):
            self._client_oid_cache = {}
        cached = self._client_oid_cache.get(event["id"])
        if cached:
            event["client_order_id"], event["pb_order_type"] = cached
            return
        try:
            order_details = await self._throttled_order_detail(
                event["id"], self.get_symbol_id(event["symbol"])
            )
            client_oid = order_details.get("data", {}).get("clientOid")
            if client_oid:
                pb_type = custom_id_to_snake(client_oid)
                event["client_order_id"] = client_oid
                event["pb_order_type"] = pb_type
                self._client_oid_cache[event["id"]] = (client_oid, pb_type)
            else:
                logging.debug(
                    "bitget order detail missing clientOid for id=%s symbol=%s",
                    event["id"],
                    event["symbol"],
                )
        except Exception as exc:
            logging.warning(
                "failed to fetch bitget order detail for id=%s symbol=%s: %s",
                event["id"],
                event["symbol"],
                exc,
            )

    def _prime_client_oid_cache(self) -> None:
        if not hasattr(self, "_client_oid_cache"):
            self._client_oid_cache = {}
        if self._client_oid_cache:
            return
        source_events: List[dict] = []
        if hasattr(self, "fill_events") and self.fill_events:
            source_events.extend(self.fill_events)
        cache_path = getattr(self, "fill_events_cache_path", None)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as fh:
                    cached_events = json.load(fh)
                    if isinstance(cached_events, list):
                        source_events.extend(cached_events)
            except Exception:
                pass
        for evt in source_events:
            evt_id = evt.get("id")
            cid = evt.get("client_order_id")
            pb = evt.get("pb_order_type")
            if evt_id and cid and evt_id not in self._client_oid_cache:
                self._client_oid_cache[evt_id] = (cid, pb)

    async def fetch_fill_events(self, start_time=None, end_time=None, limit=None):

        def _extract_fill(elm: dict) -> dict:
            timestamp = int(elm["cTime"])
            side, position_side = deduce_side_pside(elm)
            return {
                "id": elm.get("orderId"),
                "timestamp": timestamp,
                "datetime": ts_to_date(timestamp),
                "symbol": self.get_symbol_id_inv(elm["symbol"]),
                "side": side,
                "qty": float(elm["baseVolume"]),
                "price": float(elm["price"]),
                "pnl": float(elm.get("profit", 0.0)),
                "fees": elm.get("feeDetail"),
                "pb_order_type": None,
                "position_side": position_side,
                "client_order_id": None,
                "info": elm,
            }

        self._prime_client_oid_cache()

        # max limit is 100
        limit = 100 if limit is None else min(limit, 100)
        max_n_fetches = 200
        buffer_step_ms = int(1000 * 60 * 60 * 24)
        end_time = int(utc_ms() + 3600000 if end_time is None else end_time)
        events_map: Dict[str, dict] = {}
        params = {
            "productType": "USDT-FUTURES",
            "endTime": end_time,
            "limit": limit,
        }
        count = 0

        async def _fetch_window() -> List[dict]:
            nonlocal count
            count += 1
            if count >= max_n_fetches:
                logging.warning(f"over {count} calls to fetch_fill_events. Breaking.")
                return []
            fetched = await self.cca.private_mix_get_v2_mix_order_fill_history(params)
            fill_events = [
                _extract_fill(x) for x in (fetched.get("data", {}).get("fillList", []) or [])
            ]
            fill_events.sort(key=lambda x: x["timestamp"])
            if count > 1:
                n_fe = len(fill_events)
                if n_fe == 1:
                    logging.info(f"fetched 1 fill at {fill_events[0]['datetime'][:19]}")
                elif n_fe > 2:
                    logging.info(
                        f"fetched {n_fe} fills from {fill_events[0]['datetime'][:19]} to {fill_events[-1]['datetime'][:19]}"
                    )
            if not fill_events:
                return []
            return fill_events

        async def _enrich_events(fill_events: List[dict]) -> None:
            pending: List[Tuple[str, dict, asyncio.Task]] = []
            for event in fill_events:
                event_id = event.get("id")
                if not event_id:
                    continue
                if event_id in events_map:
                    continue
                cached = self._client_oid_cache.get(event_id)
                if cached:
                    event["client_order_id"], event["pb_order_type"] = cached
                if not event.get("client_order_id"):
                    pending.append(
                        (
                            event_id,
                            event,
                            asyncio.create_task(self._ensure_client_oid_for_event(event)),
                        )
                    )
                else:
                    self._client_oid_cache[event_id] = (
                        event["client_order_id"],
                        event.get("pb_order_type"),
                    )
                events_map[event_id] = event
            if pending:
                await asyncio.gather(*(task for _, _, task in pending), return_exceptions=True)
                for event_id, event, _ in pending:
                    if event.get("client_order_id"):
                        self._client_oid_cache[event_id] = (
                            event["client_order_id"],
                            event.get("pb_order_type"),
                        )

        while True:
            fill_events = await _fetch_window()
            if not fill_events:
                break
            await _enrich_events(fill_events)
            if len(fill_events) < limit:
                if start_time is None or params["endTime"] - start_time < buffer_step_ms:
                    logging.debug(
                        f"broke loop private_mix_get_v2_mix_order_fill_history on n fill_events {len(fill_events)}"
                    )
                    break
                else:
                    new_end_time = int(
                        fill_events[0]["timestamp"] + 1
                        if fill_events
                        else params["endTime"] - buffer_step_ms
                    )
                    if params["endTime"] == new_end_time:
                        new_end_time -= buffer_step_ms
                    params["endTime"] = new_end_time
                    continue
            if start_time is None or fill_events[0]["timestamp"] < start_time:
                logging.debug(
                    f"broke loop private_mix_get_v2_mix_order_fill_history on start time exceeded"
                )
                break
            if params["endTime"] == fill_events[0]["timestamp"]:
                logging.debug(
                    f"broke loop private_mix_get_v2_mix_order_fill_history on two successive identical endTimes"
                )
                break
            params["endTime"] = int(fill_events[0]["timestamp"])
        final_result = sorted(events_map.values(), key=lambda x: x["timestamp"])
        return final_result

    async def fetch_closed_orders(self, start_time, end_time, limit=100):
        def extract_fill_event_from_co(elm):
            timestamp = int(elm["lastUpdateTimestamp"])
            price = float(elm["price"])
            qty = float(elm["filled"])
            pb_order_type = custom_id_to_snake(elm.get("clientOrderId"))
            if not pb_order_type or pb_order_type == "unknown":
                if not hasattr(self, "pb_order_type_missing_logged"):
                    self.pb_order_type_missing_logged = set()
                key = json.dumps(elm)
                if key not in self.pb_order_type_missing_logged:
                    logging.info(
                        "bitget fill without pb_order_type id=%s symbol=%s clientOrderId=%s %s %s %s @ %s",
                        elm.get("id"),
                        elm.get("symbol"),
                        elm.get("clientOrderId"),
                        elm.get("side"),
                        elm.get("info", {}).get("posSide"),
                        qty,
                        price,
                    )
                self.pb_order_type_missing_logged.add(key)
            return {
                "id": elm.get("id"),
                "timestamp": timestamp,
                "datetime": ts_to_date(timestamp),
                "symbol": elm["symbol"],
                "side": elm["side"],
                "qty": qty,
                "price": price,
                "pnl": float(elm["info"]["totalProfits"]),
                "fees": elm.get("fees"),
                "pb_order_type": pb_order_type,
                "position_side": elm["info"]["posSide"],
                "client_order_id": elm.get("clientOrderId"),
            }

        # max limit is 100
        limit = min(limit, 100) if limit is not None else 100
        max_n_fetches = 200
        buffer_step_ms = int(1000 * 60 * 60 * 24)
        end_time = int(utc_ms() + 3600000 if end_time is None else end_time)
        params = {"until": end_time}
        closed_orders_all = []
        count = 0
        while True:
            count += 1
            if count >= max_n_fetches:
                logging.warning(f"over {count} calls to fetch_closed_orders. Breaking.")
                break
            closed_orders = await self.cca.fetch_closed_orders(
                limit=limit,
                params=params,
            )
            if count > 1:
                line = f"fetched {len(closed_orders)} fill{'' if len(closed_orders) == 1 else 's'}"
                if len(closed_orders) > 2:
                    line += f" from {closed_orders[0]['datetime'][:19]} to {closed_orders[-1]['datetime'][:19]}"
                logging.info(line)
            closed_orders_all.extend(closed_orders)
            if len(closed_orders) < limit:
                if start_time is None or params["until"] - start_time < buffer_step_ms:
                    logging.debug(
                        f"broke loop fetch_closed_orders on n closed_orders {len(closed_orders)}"
                    )
                    break
                else:
                    params["until"] = int(
                        closed_orders[0]["timestamp"] + 1
                        if closed_orders
                        else params["until"] - buffer_step_ms
                    )
                    continue
            if start_time is None or closed_orders[0]["timestamp"] < start_time:
                logging.debug(f"broke loop fetch_closed_orders on start time exceeded")
                break
            if params["until"] == closed_orders[0]["timestamp"]:
                logging.debug(f"broke loop fetch_closed_orders on two successive identical endTimes")
                break
            params["until"] = int(closed_orders[0]["timestamp"])
        final_result = sorted(
            [extract_fill_event_from_co(x) for x in closed_orders_all],
            key=lambda x: x["timestamp"],
        )

        deduped = []
        seen = set()
        for evt in final_result:
            fees_key = json.dumps(evt.get("fees"))
            key = (
                evt["id"],
                evt["symbol"],
                evt["qty"],
                evt["price"],
                evt["timestamp"],
                evt.get("pb_order_type"),
                fees_key,
                evt.get("client_order_id"),
            )
            if key in seen:
                logging.debug(f"removed duplicate fill event {evt}")
                continue
            seen.add(key)
            deduped.append(evt)

        return clip_by_timestamp(deduped, start_time, end_time)

    async def fetch_pnls_old(self, start_time=None, end_time=None, limit=None):
        wait_between_fetches_minimum_seconds = 0.5
        all_res = {}
        until = int(end_time) if end_time else None
        since = int(start_time) if start_time else None
        retry_count = 0
        first_fetch = True
        while True:
            if since and until and since >= until:
                # print("debug fetch_pnls g")
                break
            sts = utc_ms()
            res = await (
                self.cca.fetch_closed_orders(since=since)
                if until is None
                else self.cca.fetch_closed_orders(since=since, params={"until": until})
            )
            if first_fetch:
                if not res:
                    # print("debug fetch_pnls e")
                    break
                first_fetch = False
            if not res:
                # print("debug fetch_pnls a retry_count:", retry_count)
                if retry_count >= 10:
                    break
                retry_count += 1
                until = int(until - 1000 * 60 * 60 * 4)
                continue
            resd = {elm["id"]: elm for elm in res}
            # if len(resd) != len(res):
            #    print("debug fetch_pnls b", len(resd), len(res))
            if all(id_ in all_res for id_ in resd):
                # print("debug fetch_pnls c retry_count:", retry_count)
                if retry_count >= 10:
                    break
                retry_count += 1
                until = int(until - 1000 * 60 * 60 * 4)
                continue
            retry_count = 0
            for k, v in resd.items():
                all_res[k] = v
                all_res[k]["pnl"] = float(v["info"]["totalProfits"])
                all_res[k]["position_side"] = v["info"]["posSide"]
            if start_time is None and end_time is None:
                break
            if since and res[0]["timestamp"] <= since:
                # print("debug fetch_pnls e")
                break
            until = int(res[0]["timestamp"])
            # print(
            #    "debug fetch_pnls d len(res):",
            #    len(res),
            #    res[0]["datetime"],
            #    res[-1]["datetime"],
            #    (res[-1]["timestamp"] - res[0]["timestamp"]) / (1000 * 60 * 60),
            # )
            wait_time_seconds = max(
                0.0, wait_between_fetches_minimum_seconds - (utc_ms() - sts) / 1000
            )
            await asyncio.sleep(wait_time_seconds)
        all_res_list = sorted(all_res.values(), key=lambda x: x["timestamp"])
        return all_res_list

    async def fetch_fills_with_types(self, start_time=None, end_time=None):
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time)
        print("n fills", len(fills))
        order_details_tasks = []
        for fill in fills:
            order_details_tasks.append(
                asyncio.create_task(
                    self.cca.fetch_order(fill.get("orderId", fill.get("id")), fill["symbol"])
                )
            )
        order_details_results = {}
        for task in order_details_tasks:
            result = await task
            order_details_results[result.get("orderId", result.get("id"))] = result
        fills_by_id = {fill.get("orderId", result.get("id")): fill for fill in fills}
        for id_ in order_details_results:
            if id_ in fills_by_id:
                fills_by_id[id_].update(order_details_results[id_])
            else:
                logging.warning(f"fetch_fills_with_types id missing {id_}")
        return sorted(fills_by_id.values(), key=lambda x: x["timestamp"])

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        return {
            "timeInForce": (
                "PO" if require_live_value(self.config, "time_in_force") == "post_only" else "GTC"
            ),
            "holdSide": order["position_side"],
            "reduceOnly": order["reduce_only"],
            "oneWayMode": False,
            "clientOid": order["custom_id"],
        }

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        "cross",
                        symbol=symbol,
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self.cca.set_leverage(
                        int(self.config_get(["live", "leverage"], symbol=symbol)), symbol=symbol
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: a error setting leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f" set leverage {res} "
            except Exception as e:
                logging.error(f"{symbol} error setting leverage {e} {res}")
            res = None
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                logging.error(f"{symbol} error setting cross mode {e} {res}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def calc_ideal_orders(self, allow_unstuck: bool = True):
        # Bitget returns max 100 open orders per fetch_open_orders.
        # Only create 100 open orders.
        # Drop orders whose pprice diff is greatest.
        ideal_orders = await super().calc_ideal_orders(allow_unstuck=allow_unstuck)
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append(
                    (
                        calc_order_price_diff(
                            x["side"],
                            x["price"],
                            await self.cm.get_current_close(s, max_age_ms=10_000),
                        ),
                        {**x, **{"symbol": s}},
                    )
                )
        ideal_orders_tmp = [x[1] for x in sorted(ideal_orders_tmp, key=lambda item: item[0])][:100]
        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for x in ideal_orders_tmp:
            ideal_orders[x["symbol"]].append(x)
        return ideal_orders

    async def update_exchange_config(self):
        res = None
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode {e} {res}")

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (self.broker_code + "#" + formatted)[: self.custom_id_max_length]
