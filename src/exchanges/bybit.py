from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
from uuid import uuid4
import asyncio
import ccxt
from collections import defaultdict
from utils import ts_to_date, utc_ms
from config.access import require_live_value
from pure_funcs import (
    floatify,
    calc_hash,
    flatten,
)


class BybitBot(CCXTBot):
    def __init__(self, config: dict):
        super().__init__(config)

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    # ═══════════════════ BYBIT-SPECIFIC METHODS ═══════════════════

    async def fetch_open_orders(self, symbol: str = None) -> list:
        """Bybit: Handle nextPageCursor pagination."""
        open_orders = {}
        limit = 50
        fetched = await self.cca.fetch_open_orders(symbol=symbol, limit=limit)

        while True:
            if all(elm["id"] in open_orders for elm in fetched):
                break
            next_page_cursor = None
            for elm in fetched:
                elm["position_side"] = self._get_position_side_for_order(elm)
                elm["qty"] = elm["amount"]
                self._record_live_margin_mode_from_payload(elm)
                open_orders[elm["id"]] = elm
                if "nextPageCursor" in elm.get("info", {}):
                    next_page_cursor = elm["info"]["nextPageCursor"]
            if len(fetched) < limit or next_page_cursor is None:
                break
            fetched = await self.cca.fetch_open_orders(
                symbol=symbol, limit=limit, params={"cursor": next_page_cursor}
            )

        return sorted(open_orders.values(), key=lambda x: x["timestamp"])

    async def fetch_positions(self) -> list:
        """Bybit: Handle nextPageCursor pagination."""
        positions = {}
        limit = 200
        fetched = await self.cca.fetch_positions(params={"limit": limit})

        while True:
            if all(elm["symbol"] + elm["side"] in positions for elm in fetched):
                break
            next_page_cursor = None
            for elm in fetched:
                key = elm["symbol"] + elm["side"]
                normalized = {
                    "symbol": elm["symbol"],
                    "position_side": elm.get("side", "long").lower(),
                    "size": float(elm["contracts"]),
                    "price": float(elm["entryPrice"]),
                }
                margin_mode = self._extract_live_margin_mode(elm)
                if margin_mode is not None:
                    normalized["margin_mode"] = margin_mode
                self._record_live_margin_mode(elm["symbol"], margin_mode)
                positions[key] = normalized
                if "nextPageCursor" in elm.get("info", {}):
                    next_page_cursor = elm["info"]["nextPageCursor"]
            if len(fetched) < limit or next_page_cursor is None:
                break
            fetched = await self.cca.fetch_positions(
                params={"cursor": next_page_cursor, "limit": limit}
            )

        return list(positions.values())

    async def fetch_balance(self) -> float:
        """Bybit: Complex UNIFIED account balance calculation."""
        fetched_balance = await self.cca.fetch_balance()
        balinfo = fetched_balance["info"]["result"]["list"][0]
        if balinfo["accountType"] == "UNIFIED":
            balance = 0.0
            for elm in balinfo["coin"]:
                if elm["marginCollateral"] and elm["collateralSwitch"]:
                    balance += float(elm["usdValue"]) + float(elm["unrealisedPnl"])
        else:
            balance = fetched_balance[self.quote]["total"]
        return balance

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
        params = {"category": "linear", "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        fetched = (await self.cca.private_get_v5_position_closed_pnl(params))["result"]
        while True:
            fetched["list"] = sorted(floatify(fetched["list"]), key=lambda x: float(x["updatedTime"]))
            for i in range(len(fetched["list"])):
                fetched["list"][i]["timestamp"] = float(fetched["list"][i]["updatedTime"])
                fetched["list"][i]["symbol"] = self.get_symbol_id_inv(fetched["list"][i]["symbol"])
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

    def _build_order_params(self, order: dict) -> dict:
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
        for symbol in symbols:
            to_print = ""
            leverage = self._calc_leverage_for_symbol(symbol)
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            try:
                res = await self.cca.set_margin_mode(
                    margin_mode,
                    symbol=symbol,
                    params={"leverage": leverage},
                )
                to_print += f"margin={format_exchange_config_response(res)} "
            except ccxt.BadRequest as e:
                err_str = str(e).lower()
                if "110026" in err_str or "not modified" in err_str:
                    logging.debug(f"{symbol}: margin mode already set (not modified)")
                else:
                    raise
            try:
                res = await self.cca.set_leverage(leverage, symbol=symbol)
                to_print += f"leverage={format_exchange_config_response(res)}"
            except ccxt.BadRequest as e:
                err_str = str(e).lower()
                if "110043" in err_str or "not modified" in err_str:
                    logging.debug(f"{symbol}: leverage already set (not modified)")
                else:
                    raise
            if to_print:
                logging.info(f"{symbol}: {to_print.strip()}")

    async def update_exchange_config(self):
        res = await self.cca.set_position_mode(True)
        logging.debug("[config] set hedge mode response: %s", res)
