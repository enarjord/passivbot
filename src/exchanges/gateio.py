from exchanges.ccxt_bot import CCXTBot
from passivbot import logging

from utils import ts_to_date, utc_ms
from config_utils import require_live_value


class GateIOBot(CCXTBot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ohlcvs_1m_init_duration_seconds = (
            120  # gateio has stricter rate limiting on fetching ohlcvs
        )
        self.hedge_mode = False
        max_cancel = int(require_live_value(config, "max_n_cancellations_per_batch"))
        self.config["live"]["max_n_cancellations_per_batch"] = min(max_cancel, 20)
        max_create = int(require_live_value(config, "max_n_creations_per_batch"))
        self.config["live"]["max_n_creations_per_batch"] = min(max_create, 10)
        self.custom_id_max_length = 28

    def create_ccxt_sessions(self):
        """GateIO: Add broker header to CCXT config."""
        super().create_ccxt_sessions()
        # Add broker header to both clients
        headers = {"X-Gate-Channel-Id": self.broker_code} if self.broker_code else {}
        for client in [self.cca, self.ccp]:
            if client is not None:
                client.headers.update(headers)

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    def _get_position_side_for_order(self, order: dict) -> str:
        """GateIO: Derive position side from order side + reduceOnly (one-way mode)."""
        return self.determine_pos_side(order)

    def determine_pos_side(self, order):
        """GateIO-specific logic for one-way mode position side derivation."""
        if order["side"] == "buy":
            return "short" if order["reduceOnly"] else "long"
        if order["side"] == "sell":
            return "long" if order["reduceOnly"] else "short"
        raise Exception(f"unsupported order side {order['side']}")

    # ═══════════════════ GATEIO-SPECIFIC METHODS ═══════════════════

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # GateIO uses fetch_ohlcv for this
        result = await self.cca.fetch_ohlcv("BTC/USDT:USDT", timeframe="1m")
        self.utc_offset = round((result[-1][0] - utc_ms()) / (1000 * 60 * 60)) * (1000 * 60 * 60)
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def fetch_balance(self) -> float:
        """GateIO: Fetch balance with special UID logic for websockets.

        GateIO requires UID for websocket subscriptions, which is obtained
        from the balance response. Also handles classic vs multi_currency
        margin modes.
        """
        balance_fetched = await self.cca.fetch_balance()
        if not hasattr(self, "uid") or not self.uid:
            self.uid = balance_fetched["info"][0]["user"]
            self.cca.uid = self.uid
            if self.ccp is not None:
                self.ccp.uid = self.uid
        margin_mode_name = balance_fetched["info"][0]["margin_mode_name"]
        self.log_once(f"account margin mode: {margin_mode_name}")
        if margin_mode_name == "classic":
            balance = float(balance_fetched[self.quote]["total"])
        elif margin_mode_name == "multi_currency":
            balance = float(balance_fetched["info"][0]["cross_available"])
        else:
            raise Exception(f"unknown margin_mode_name {balance_fetched}")
        return balance

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit=None,
    ):
        if start_time is None:
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        if limit is None:
            limit = 1000
        offset = 0
        while True:
            fetched = await self.fetch_pnl(offset=offset, limit=limit)
            if not fetched:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            if fetched[0]["timestamp"] <= start_time:
                break
            logging.debug(f"fetching pnls {ts_to_date(fetched[-1]['timestamp'])}")
            offset += limit
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for Gate.io."""
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
                    "qty": fill.get("amount") or fill.get("filled"),
                    "price": fill.get("price"),
                    "pnl": fill.get("pnl"),
                    "fee": fill.get("fee"),
                    "info": fill.get("info"),
                }
            )
        return events

    async def fetch_pnl(
        self,
        offset=0,
        limit=None,
    ):
        n_pnls_limit = 1000 if limit is None else limit
        fetched = await self.cca.fetch_closed_orders(limit=n_pnls_limit, params={"offset": offset})
        for i in range(len(fetched)):
            fetched[i]["pnl"] = float(fetched[i]["info"]["pnl"])
            fetched[i]["position_side"] = self.determine_pos_side(fetched[i])
        return sorted(fetched, key=lambda x: x["timestamp"])

    def did_cancel_order(self, executed, order=None):
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return executed.get("id", "") == order["id"] and executed.get("status", "") == "canceled"
        except Exception:
            return False

    def _build_order_params(self, order: dict) -> dict:
        order_type = order["type"] if "type" in order else "limit"
        params = {
            "reduce_only": order["reduce_only"],
            "text": order["custom_id"],
        }
        if order_type == "limit":
            params["timeInForce"] = (
                "poc" if require_live_value(self.config, "time_in_force") == "post_only" else "gtc"
            )
        return params

    def did_create_order(self, executed):
        try:
            return "status" in executed and executed["status"] != "rejected"
        except Exception:
            return False

    async def update_exchange_config_by_symbols(self, symbols):
        """GateIO: No per-symbol configuration needed."""
        pass

    async def update_exchange_config(self):
        """GateIO: No exchange-level configuration needed."""
        pass
