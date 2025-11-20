from passivbot import Passivbot, logging, clip_by_timestamp
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
from ccxt.base.errors import InvalidNonce
import pprint
import asyncio
import traceback
import numpy as np
import passivbot_rust as pbr
from collections import defaultdict
from utils import ts_to_date, utc_ms
from config_utils import require_live_value
from pure_funcs import (
    multi_replace,
    floatify,
    calc_hash,
    determine_pos_side_ccxt,
    flatten,
)
from procedures import print_async_exception, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class BybitBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)

    def create_ccxt_sessions(self):
        if self.ws_enabled:
            self.ccp = getattr(ccxt_pro, self.exchange)(
                {
                    "apiKey": self.user_info["key"],
                    "secret": self.user_info["secret"],
                    "password": self.user_info["passphrase"],
                    "headers": {"referer": self.broker_code} if self.broker_code else {},
                    "enableRateLimit": True,
                }
            )
            self.ccp.options.update(self._build_ccxt_options())
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Bybit websocket session due to custom endpoint override.")
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
                "headers": {"referer": self.broker_code} if self.broker_code else {},
                "enableRateLimit": True,
            }
        )
        self.cca.options.update(self._build_ccxt_options())
        self._apply_endpoint_override(self.cca)

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

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = determine_pos_side_ccxt(res[i])
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None) -> [dict]:
        fetched = None
        open_orders = {}
        limit = 50
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol, limit=limit)
            while True:
                if all([elm["id"] in open_orders for elm in fetched]):
                    break
                next_page_cursor = None
                for elm in fetched:
                    elm["position_side"] = determine_pos_side_ccxt(elm)
                    elm["qty"] = elm["amount"]
                    open_orders[elm["id"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                if len(fetched) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched = await self.cca.fetch_open_orders(
                    symbol=symbol, limit=limit, params={"cursor": next_page_cursor}
                )
            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions, fetched_balance = None, None
        positions = {}
        limit = 200
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(params={"limit": limit}), self.cca.fetch_balance()
            )
            balinfo = fetched_balance["info"]["result"]["list"][0]
            if balinfo["accountType"] == "UNIFIED":
                balance = 0.0
                for elm in balinfo["coin"]:
                    if elm["marginCollateral"] and elm["collateralSwitch"]:
                        balance += float(elm["usdValue"]) + float(elm["unrealisedPnl"])
                if not hasattr(self, "previous_hysteresis_balance"):
                    self.previous_hysteresis_balance = balance
                self.previous_hysteresis_balance = pbr.hysteresis(
                    balance,
                    self.previous_hysteresis_balance,
                    self.hyst_pct,
                )
                balance = self.previous_hysteresis_balance
            else:
                balance = fetched_balance[self.quote]["total"]
            while True:
                if all([elm["symbol"] + elm["side"] in positions for elm in fetched_positions]):
                    break
                next_page_cursor = None
                for elm in fetched_positions:
                    elm["position_side"] = determine_pos_side_ccxt(elm)
                    elm["size"] = float(elm["contracts"])
                    elm["price"] = float(elm["entryPrice"])
                    positions[elm["symbol"] + elm["side"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                    positions[elm["symbol"] + elm["side"]] = elm
                if len(fetched_positions) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched_positions = await self.cca.fetch_positions(
                    params={"cursor": next_page_cursor, "limit": limit}
                )
            return sorted(positions.values(), key=lambda x: x["timestamp"]), balance
        except InvalidNonce as e:
            logging.warning("Invalid nonce while fetching positions/balance: %s", e)
            return False
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
            if "bybit does not have market symbol" in str(e):
                # ccxt is raising bad symbol error
                # restart might help...
                raise Exception("ccxt gives bad symbol error... attempting bot restart")
            return False

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

    async def fetch_pnls_sub(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
        if start_time is None:
            pnls = await self.fetch_pnl(start_time=start_time, end_time=end_time)
        else:
            week = 1000 * 60 * 60 * 24 * 7
            pnls = []
            if end_time is None:
                end_time = int(self.get_exchange_time() + 1000 * 60 * 60 * 24)
            # bybit has limit of 7 days per paginated fetch
            # fetch multiple times
            i = 1
            while i < 52:  # limit n fetches to 52 (one year)
                sts = end_time - week * i
                ets = sts + week
                sts = max(sts, start_time)
                fetched = await self.fetch_pnl(start_time=sts, end_time=ets)
                pnls.extend(fetched)
                if sts <= start_time:
                    break
                i += 1
                logging.info(f"fetched pnls for more than a week {ts_to_date(sts)}")
        return sorted(pnls, key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        fetched = None
        all_pnls = []
        ids_seen = set()
        if limit is None:
            limit = 100
        try:
            params = {"category": "linear", "limit": limit}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            fetched = (await self.cca.private_get_v5_position_closed_pnl(params))["result"]
            while True:
                fetched["list"] = sorted(
                    floatify(fetched["list"]), key=lambda x: float(x["updatedTime"])
                )
                for i in range(len(fetched["list"])):
                    fetched["list"][i]["timestamp"] = float(fetched["list"][i]["updatedTime"])
                    fetched["list"][i]["symbol"] = self.get_symbol_id_inv(
                        fetched["list"][i]["symbol"]
                    )
                    fetched["list"][i]["pnl"] = float(fetched["list"][i]["closedPnl"])
                    fetched["list"][i]["side"] = fetched["list"][i]["side"].lower()
                    fetched["list"][i]["position_side"] = (
                        "long" if fetched["list"][i]["side"] == "sell" else "short"
                    )
                if fetched["list"] == []:
                    break
                if (
                    fetched["list"][0]["orderId"] in ids_seen
                    and fetched["list"][-1]["orderId"] in ids_seen
                ):
                    break
                all_pnls.extend(fetched["list"])
                for elm in fetched["list"]:
                    ids_seen.add(elm["orderId"])
                if start_time is None:
                    break
                if fetched["list"][0]["updatedTime"] <= start_time:
                    break
                if not fetched["nextPageCursor"]:
                    break
                if len(fetched["list"]) < limit:
                    break
                logging.info(
                    f"fetched pnls from {ts_to_date(fetched['list'][-1]['updatedTime'])} n pnls: {len(fetched['list'])}"
                )
                params["cursor"] = fetched["nextPageCursor"]
                fetched = (await self.cca.private_get_v5_position_closed_pnl(params))["result"]
            return sorted(all_pnls, key=lambda x: x["updatedTime"])
        except Exception as e:
            logging.error(f"error fetching pnls {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return []

    async def fetch_fills(self, start_time, end_time, limit=None):
        if start_time is None:
            result = await self.cca.fetch_my_trades()
            return sorted(result, key=lambda x: x["timestamp"])
        if end_time is None:
            end_time = int(self.get_exchange_time() + 1000 * 60 * 60 * 4)
        all_fetched_fills = []
        prev_hash = ""
        for _ in range(100):
            fills = await self.cca.fetch_my_trades(
                limit=limit, params={"paginate": True, "endTime": int(end_time)}
            )
            if not fills:
                break
            fills.sort(key=lambda x: x["timestamp"])
            all_fetched_fills += fills
            if fills[0]["timestamp"] <= start_time:
                break
            new_hash = calc_hash([x["id"] for x in fills])
            if new_hash == prev_hash:
                break
            prev_hash = new_hash
            logging.info(
                f"fetched fills from {fills[0]['datetime']} to {fills[-1]['datetime']} n fills: {len(fills)}"
            )
            end_time = fills[0]["timestamp"]
            limit = 1000
        else:
            logging.error(f"more than 100 calls to ccxt fetch_my_trades")
        return sorted(all_fetched_fills, key=lambda x: x["timestamp"])

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        # fetch fills first, then pnls (bybit has them in separate endpoints)
        if start_time:
            if self.get_exchange_time() - start_time < 1000 * 60 * 60 * 4 and limit == 100:
                # set start time to None (fetch latest) if start time is recent
                start_time = None
        fills = await self.fetch_fills(start_time=start_time, end_time=end_time, limit=limit)
        if start_time:
            fills = [x for x in fills if x["timestamp"] >= start_time - 1000 * 60 * 60]
        if not fills:
            return []
        start_time = fills[0]["timestamp"]
        pnls = await self.fetch_pnls_sub(start_time=start_time, end_time=end_time)

        fillsd = defaultdict(list)
        for x in fills:
            x["orderId"] = x["info"]["orderId"]
            x["position_side"] = self.determine_pos_side(x)
            x["pnl"] = 0.0
            fillsd[x["orderId"]].append(x)
        pnls_ids = set()
        for x in pnls:
            pnls_ids.add(x["orderId"])
            if x["orderId"] in fillsd:
                fillsd[x["orderId"]][-1]["pnl"] = x["pnl"]
            else:
                # logging.info(f"debug missing order id in fills {x['orderId']} {x}")
                x["info"] = {"execId": uuid4().hex}
                x["id"] = x["orderId"]
                fillsd[x["orderId"]] = [x]
        joined = {x["info"]["execId"]: x for x in flatten(fillsd.values())}
        return sorted(joined.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for equity reconstruction (draft implementation)."""

        def extract_fill_event_from_ph(elm):
            event = {
                "id": elm["info"]["orderId"],
                "timestamp": int(float(elm.get("lastUpdateTimestamp", elm.get("timestamp", 0.0)))),
                "symbol": elm["symbol"],
                "side": str(elm["info"].get("side", "")).lower(),
                "qty": float(elm.get("contracts", elm.get("qty", 0.0))),
                "price": float(elm.get("lastPrice", elm.get("price", 0.0))),
                "pnl": float(elm.get("realizedPnl", 0.0)),
                "fee": None,
                "custom_id": elm.get("info", {}).get("orderLinkId"),
                "position_side": None,
            }
            if event["side"] == "buy":
                event["position_side"] = "long" if event["pnl"] == 0.0 else "short"
            elif event["side"] == "sell":
                event["position_side"] = "short" if event["pnl"] == 0.0 else "long"
            else:
                raise Exception(f"malformed side {event['side']}")
            return event

        def extract_fill_event_from_mt(elm):
            event = {
                "id": elm["info"]["orderId"],
                "timestamp": elm["timestamp"],
                "symbol": elm["symbol"],
                "side": elm["side"],
                "qty": float(elm["amount"]),
                "price": float(elm["price"]),
                "pnl": None,
                "fee": elm.get("fee"),
                "custom_id": elm.get("info", {}).get("orderLinkId"),
                "position_side": None,
            }
            closed_size = float(elm["info"].get("closedSize", 0.0))
            if event["side"] == "buy":
                event["position_side"] = "long" if closed_size == 0.0 else "short"
                if closed_size == 0.0:
                    event["pnl"] = 0.0
            elif event["side"] == "sell":
                event["position_side"] = "short" if closed_size == 0.0 else "long"
                if closed_size == 0.0:
                    event["pnl"] = 0.0
            else:
                raise Exception(f"malformed side {event['side']}")
            return event

        def is_equal(x0, x1):
            for key in ["id", "symbol", "qty", "side", "position_side", "price"]:
                if x0[key] != x1[key]:
                    return False
            return True

        def merge_events(x0, x1):
            merged_list = []
            if is_equal(x0, x1):
                event = {}
                for key in x0:
                    event[key] = x0.get(key, x1.get(key))
                return [event]
            else:
                return [x0, x1]

        def get_dedup_key(event):
            return tuple(
                [event[k] for k in ["id", "symbol", "qty", "side", "position_side", "price"]]
            )

        if end_time is None:
            end_time = int(self.get_exchange_time() + 1000 * 60 * 60)
        if start_time is None:
            start_time = end_time - 1000 * 60 * 60 * 24 * 3

        # fetch concurrently
        try:
            my_trades, positions_history = await asyncio.gather(
                self.fetch_my_trades(start_time, end_time),
                self.fetch_positions_history(start_time, end_time),
            )
        except Exception as exc:
            logging.error(f"error fetching my_trades, positions_history {exc}")
            my_trades, positions_history = [], []

        # extract events
        mt_events = sorted(
            [extract_fill_event_from_mt(x) for x in my_trades], key=lambda x: x["timestamp"]
        )
        ph_events = sorted(
            [extract_fill_event_from_ph(x) for x in positions_history], key=lambda x: x["timestamp"]
        )
        mt_events = clip_by_timestamp(mt_events, start_time, end_time)
        ph_events = clip_by_timestamp(ph_events, start_time, end_time)

        pnls = defaultdict(float)
        for event in ph_events:
            pnls[event["id"]] += event["pnl"]
        unified = []
        for event in mt_events[::-1]:
            if event["id"] in pnls:
                event["pnl"] = pnls.pop(event["id"])
            unified.append(event)
        if len(pnls) > 0:
            print("debug positions_history events without corresponding my_trades")
        unified.sort(key=lambda x: x["timestamp"])
        return unified

    async def fetch_my_trades(self, start_time, end_time, limit=100):
        # wrapper for ccxt.fetch_my_trades
        # multiple fetches to find all fills inside given date range
        # limit is max 100
        # The time range between startTime and endTime cannot exceed 7 days
        # if start time is given without end time, will fetch fills closes to one week after start time
        # strategy: fetch backwards from end time to start time
        limit = min(limit, 100)
        max_n_fetches = 200
        week_with_buffer_ms = int(1000 * 60 * 60 * 24 * 6.5)
        end_time = int(utc_ms() + 3600000 if end_time is None else end_time)
        params = {"type": "swap", "subType": "linear", "limit": limit, "endTime": end_time}
        my_trades_all = []
        count = 0
        while True:
            count += 1
            my_trades = await self.cca.fetch_my_trades(params=params)
            my_trades_all.extend(my_trades)
            if len(my_trades) < limit:
                if start_time is None or params["endTime"] - start_time < week_with_buffer_ms:
                    logging.debug(f"broke loop fetch_my_trades on n my_trades {len(my_trades)}")
                    break
                else:
                    params["endTime"] = int(
                        my_trades[0]["timestamp"] + 1
                        if my_trades
                        else params["endTime"] - week_with_buffer_ms
                    )
                    continue
            if start_time is None or my_trades[0]["timestamp"] < start_time:
                logging.debug(f"broke loop fetch_my_trades on start time exceeded")
                break
            if params["endTime"] == my_trades[0]["timestamp"]:
                logging.debug(f"broke loop fetch_my_trades on two successive identical endTimes")
                break
            params["endTime"] = int(my_trades[0]["timestamp"] + 1)
            if count > 1:
                logging.info(
                    f"fetched {len(my_trades)} fills from {my_trades[0]['datetime'][:19]} to {my_trades[-1]['datetime'][:19]}"
                )
        return sorted(my_trades_all, key=lambda x: x["timestamp"])

    async def fetch_positions_history(self, start_time, end_time, limit=100):
        # wrapper for ccxt.fetch_positions_history
        # limit is max 100

        # The start timestamp (ms)
        # startTime and endTime are not passed, return 7 days by default
        # Only startTime is passed, return range between startTime and startTime+7 days
        # Only endTime is passed, return range between endTime-7 days and endTime
        # If both are passed, the rule is endTime - startTime <= 7 days

        # The time range between startTime and endTime cannot exceed 7 days
        # if start time is given without end time, will fetch positions closes to one week after start time
        # strategy: fetch backwards from end time to start time
        limit = min(limit, 100)
        max_n_fetches = 200
        week_with_buffer_ms = int(1000 * 60 * 60 * 24 * 6.5)
        end_time = int(utc_ms() + 3600000 if end_time is None else end_time)
        params = {"limit": limit, "endTime": end_time}
        positions_history_all = []
        count = 0
        while True:
            count += 1
            positions_history = await self.cca.fetch_positions_history(params=params)
            positions_history.sort(key=lambda x: x["timestamp"])
            positions_history_all.extend(positions_history)
            if len(positions_history) < limit:
                if start_time is None or params["endTime"] - start_time < week_with_buffer_ms:
                    logging.debug(f"broke loop fetch_positions_history on n {len(positions_history)}")
                    break
                else:
                    params["endTime"] = int(params["endTime"] - week_with_buffer_ms)
                    continue
            if start_time is None or positions_history[0]["timestamp"] < start_time:
                logging.debug("broke loop fetch_positions_history on start time exceeded")
                break
            if params["endTime"] == positions_history[0]["timestamp"]:
                logging.debug(
                    "broke loop fetch_positions_history on two successive identical endTimes"
                )
                break
            params["endTime"] = int(positions_history[0]["timestamp"])
            if count > 1:
                logging.info(
                    f"fetched {len(positions_history)} positions_history from {positions_history[0]['datetime'][:19]} to {positions_history[-1]['datetime'][:19]}"
                )
        return sorted(positions_history_all, key=lambda x: x["timestamp"])

    def determine_pos_side(self, x):
        if x["side"] == "buy":
            return "short" if float(x["info"]["closedSize"]) != 0.0 else "long"
        return "long" if float(x["info"]["closedSize"]) != 0.0 else "short"

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        return {
            "positionIdx": 1 if order["position_side"] == "long" else 2,
            "timeInForce": (
                "postOnly"
                if require_live_value(self.config, "time_in_force") == "post_only"
                else "GTC"
            ),
            "orderLinkId": order["custom_id"],
        }

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        "cross",
                        symbol=symbol,
                        params={
                            "leverage": int(self.config_get(["live", "leverage"], symbol=symbol))
                        },
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
                if '"retCode":110043' in e.args[0]:
                    to_print += f" leverage: {e}"
                else:
                    logging.error(f"{symbol} error setting leverage {e}")
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                if '"retCode":110026' in e.args[0]:
                    to_print += f" set cross mode: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting cross mode {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode {e}")

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        if since is None:
            result = await self.cca.fetch_ohlcv(symbol, timeframe="1m", limit=n_candles_limit)
            return result
        since = since // 60000 * 60000
        max_n_fetches = 5000 // n_candles_limit
        all_fetched = []
        for i in range(max_n_fetches):
            fetched = await self.cca.fetch_ohlcv(
                symbol, timeframe="1m", since=int(since), limit=n_candles_limit
            )
            all_fetched += fetched
            if len(fetched) < n_candles_limit:
                break
            since = fetched[-1][0]
        all_fetched_d = {x[0]: x for x in all_fetched}
        return sorted(all_fetched_d.values(), key=lambda x: x[0])
