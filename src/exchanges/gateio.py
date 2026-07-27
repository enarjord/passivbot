import math

from exchanges.ccxt_bot import CCXTBot
from passivbot import logging

from utils import symbol_to_coin, to_ccxt_client_id, ts_to_date
from config.access import require_live_value
from custom_endpoint_overrides import (
    get_custom_endpoint_source,
    resolve_custom_endpoint_override_with_aliases,
)


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
        endpoint_override = resolve_custom_endpoint_override_with_aliases("gateio", ("gate",))
        if endpoint_override != self.endpoint_override:
            self.endpoint_override = endpoint_override
            self.ws_enabled = endpoint_override is None or not endpoint_override.disable_ws
            if endpoint_override is not None:
                logging.info(
                    "Custom endpoint override active for gateio/gate "
                    "(disable_ws=%s, source=%s)",
                    endpoint_override.disable_ws,
                    get_custom_endpoint_source() or "auto-discovered",
                )
        # CCXT 4.5.66 exposes Gate.io clients under ``gate``. Keep Passivbot's
        # exchange identity canonical everywhere outside client construction.
        canonical_ccxt_id = self.exchange_ccxt_id
        try:
            self.exchange_ccxt_id = to_ccxt_client_id(canonical_ccxt_id)
            super().create_ccxt_sessions()
        finally:
            self.exchange_ccxt_id = canonical_ccxt_id
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
        return self._normalize_one_way_position_side(order)

    # ═══════════════════ GATEIO-SPECIFIC METHODS ═══════════════════

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        unavailable_symbols: set[str] = set()
        for symbol in self.symbols_requiring_market_sizing():
            market = self.markets_dict[symbol]
            raw_max_leverage = (market.get("limits") or {}).get("leverage", {}).get(
                "max"
            )
            if raw_max_leverage is None:
                raw_max_leverage = (market.get("info") or {}).get("leverage_max")
            try:
                max_leverage = float(raw_max_leverage)
            except (TypeError, ValueError, OverflowError):
                max_leverage = float("nan")
            if not math.isfinite(max_leverage) or max_leverage <= 0.0:
                unavailable_symbols.add(symbol)
                self.max_leverage.pop(symbol, None)
                configured_symbols = getattr(
                    self, "already_updated_exchange_config_symbols", None
                )
                if isinstance(configured_symbols, set):
                    configured_symbols.discard(symbol)
                logging.warning(
                    "%s: Gate.io max leverage metadata unavailable; "
                    "deferring exposure-increasing orders",
                    symbol_to_coin(symbol, verbose=False) or symbol,
                )
                continue
            effective_max_leverage = int(max_leverage)
            previous_max_leverage = self.max_leverage.get(symbol)
            self.max_leverage[symbol] = effective_max_leverage
            if (
                previous_max_leverage is not None
                and previous_max_leverage != effective_max_leverage
            ):
                configured_symbols = getattr(
                    self, "already_updated_exchange_config_symbols", None
                )
                if isinstance(configured_symbols, set):
                    configured_symbols.discard(symbol)
        self._gate_leverage_metadata_unavailable_symbols = unavailable_symbols

    async def fetch_balance(self) -> float:
        """GateIO: Fetch balance using the same parser as staged snapshots."""
        balance_fetched = await self._do_fetch_balance()
        return self._get_balance(balance_fetched)

    def _get_balance(self, balance_fetched: dict) -> float:
        """Extract Gate.io futures balance for classic and multi-currency margin modes.

        Staged refresh calls CCXTBot.capture_balance_snapshot(), which bypasses
        fetch_balance() and calls this hook directly. Keep Gate.io's margin-mode
        specific parsing here so legacy and staged paths use the same balance.
        """
        info = balance_fetched.get("info")
        if not isinstance(info, list) or not info:
            raise KeyError(f"{self.exchange}: fetch_balance response missing info[0]")
        primary = info[0]
        if not hasattr(self, "uid") or not self.uid:
            # Gate's REST payload currently returns ``user`` as an integer,
            # while CCXT Pro's private futures subscription treats the UID as
            # a string and calls len() on it while signing the request.
            raw_uid = primary["user"]
            if isinstance(raw_uid, bool) or not isinstance(raw_uid, (str, int)):
                raise ValueError(
                    f"{self.exchange}: fetch_balance response has invalid info[0].user"
                )
            uid = str(raw_uid).strip()
            if not uid:
                raise ValueError(
                    f"{self.exchange}: fetch_balance response has empty info[0].user"
                )
            self.uid = uid
            self.cca.uid = self.uid
            if self.ccp is not None:
                self.ccp.uid = self.uid
        margin_mode_name = primary["margin_mode_name"]
        self.log_once(f"account margin mode: {margin_mode_name}")
        if margin_mode_name == "classic":
            balance = float(balance_fetched[self.quote]["total"])
        elif margin_mode_name == "multi_currency":
            # ``cross_available`` is spendable margin, not account equity. It
            # falls when resting orders reserve margin and rises again when
            # those orders are cancelled, which would make Passivbot resize its
            # ideal orders on every reconciliation cycle. Reconstruct the
            # stable cross-margin balance from the same authoritative account
            # payload by adding back position and resting-order initial margin.
            margin_balance = sum(
                float(primary[key])
                for key in (
                    "cross_available",
                    "cross_initial_margin",
                    "cross_order_margin",
                )
            )
            # Margin balance includes unrealized cross-position PnL. Passivbot
            # adds position PnL separately when deriving equity, so remove it
            # here to retain wallet-balance semantics and avoid double-counting.
            balance = margin_balance - float(primary["cross_unrealised_pnl"])
            if not math.isfinite(balance):
                raise ValueError(
                    f"{self.exchange}: fetch_balance response has non-finite "
                    "multi-currency margin balance"
                )
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
            # Gate.io requires contract order text to start with "t-". CCXT maps
            # clientOrderId to that exchange text field while preserving our marker.
            "clientOrderId": order["custom_id"],
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
        """Apply the configured leverage and margin mode through Gate's leverage endpoint."""
        for symbol in symbols:
            if symbol in set(
                getattr(
                    self,
                    "_gate_leverage_metadata_unavailable_symbols",
                    set(),
                )
                or set()
            ):
                raise ValueError(
                    f"{symbol}: Gate.io max leverage metadata unavailable"
                )
            leverage = self._calc_leverage_for_symbol(symbol)
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            await self.cca.set_leverage(
                leverage,
                symbol=symbol,
                params={"marginMode": margin_mode},
            )
            logging.info(
                "%s: set %s leverage to %sx",
                symbol_to_coin(symbol, verbose=False) or symbol,
                margin_mode,
                leverage,
            )

    def _order_requires_exchange_config_before_create(self, order: dict) -> bool:
        """Gate leverage is an entry prerequisite, not a close prerequisite."""
        return self._extract_order_reduce_only(order) is not True

    def _pending_exchange_config_consumes_error_budget(
        self, blocked_orders: list[dict]
    ) -> bool:
        """Keep persistent Gate entry-configuration failures restart-visible."""
        return bool(blocked_orders)

    async def update_exchange_config(self):
        """GateIO: No exchange-level configuration needed."""
        pass
