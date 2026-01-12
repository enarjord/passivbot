"""
DefxBot: Defx-specific exchange connector.

Extends CCXTBot with Defx-specific logic for one-way mode positions,
custom balance fetching, and leverage configuration.
"""

import asyncio

import passivbot_rust as pbr
from exchanges.ccxt_bot import CCXTBot
from passivbot import logging
from utils import utc_ms


class DefxBot(CCXTBot):
    """Defx exchange bot with one-way mode position handling."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36  # adjust if needed
        self.quote = "USDC"
        self.hedge_mode = False

    def _get_position_side_for_order(self, order: dict) -> str:
        """Defx: Derive from position state (one-way mode)."""
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

    async def fetch_open_orders(self, symbol: str = None):
        fetched = await self.cca.fetch_open_orders(symbol=symbol)
        for order in fetched:
            order["position_side"] = self.determine_pos_side(order)
            order["qty"] = order["amount"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def fetch_positions(self):
        fetched_positions = await self.cca.fetch_positions()
        positions = []
        for p in fetched_positions:
            positions.append(
                {
                    **p,
                    **{
                        "symbol": p["symbol"],
                        "position_side": p["info"]["positionSide"].lower(),
                        "size": float(p["contracts"]),
                        "price": float(p["entryPrice"]),
                    },
                }
            )
        return positions

    async def fetch_wallet_collaterals(self):
        fetched = await self.cca.fetch2(
            path="api/wallet/balance/collaterals",
            api=["v1", "private"],  # tuple-like fallback
            method="GET",
            params={},
        )
        for i in range(len(fetched)):
            for k in fetched[i]:
                try:
                    fetched[i][k] = float(fetched[i][k])
                except (ValueError, TypeError):
                    # Some fields (IDs, strings) can't be converted to float - skip them
                    pass
        return fetched

    async def fetch_balance(self):
        fetched_balance = await self.fetch_wallet_collaterals()
        return sum([x["marginValue"] for x in fetched_balance])

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        # TODO: impl start_time and end_time
        res = await self.cca.fetch_my_trades()
        for i in range(len(res)):
            res[i]["qty"] = res[i]["amount"]
            res[i]["pnl"] = float(res[i]["info"]["pnl"])
            if res[i]["side"] == "buy":
                res[i]["position_side"] = "long" if res[i]["pnl"] == 0.0 else "short"
            elif res[i]["side"] == "sell":
                res[i]["position_side"] = "short" if res[i]["pnl"] == 0.0 else "long"
            else:
                raise Exception(f"invalid side {res[i]}")
        return res

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for dYdX/DeFX adapter (draft placeholder)."""
        events = []
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time, limit=limit)
        for fill in fills:
            events.append(
                {
                    "id": fill.get("id"),
                    "timestamp": fill.get("timestamp"),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "position_side": fill.get("position_side"),
                    "qty": fill.get("qty"),
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
        }

    async def update_exchange_config(self):
        """Defx uses one-way mode; no hedge mode configuration needed."""
        pass

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ticker("BTC/USDC:USDC")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_leverage = {}
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.config_get(["live", "leverage"], symbol=symbol),
                            pbr.round_up(
                                max(
                                    self.get_wallet_exposure_limit("long", symbol),
                                    self.get_wallet_exposure_limit("short", symbol),
                                )
                                * 1.1,
                                1,
                            ),
                        )
                    ),
                    "symbol": symbol,
                }
                logging.debug(f"update_exchange_config_by_symbols {params}")
                coros_to_call_leverage[symbol] = asyncio.create_task(self.cca.set_leverage(**params))
            except Exception as e:
                logging.error(f"{symbol}: error setting leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_leverage[symbol]
                to_print += f"set leverage {res}"
            except Exception as e:
                if '"code":"59107"' in str(e):
                    to_print += f" cross mode and leverage: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting leverage {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")
        return
