from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging, custom_id_to_snake, clip_by_timestamp
import asyncio
import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Tuple
from bitget_normalization import (
    bitget_payload_context as _bitget_payload_context,
    deduce_side_pside,
    normalize_uta_fill_payload,
)
from utils import symbol_to_coin, utc_ms, ts_to_date
from config.access import require_live_value
import passivbot_rust as pbr

calc_order_price_diff = pbr.calc_order_price_diff

_CLASSIC_ACCOUNT_MODE_CODES = {"40084", "25245"}


def _bitget_error_code_from_exception(exc: Exception) -> str:
    payload = getattr(exc, "response", None) or getattr(exc, "body", None)
    if isinstance(payload, bytes):
        payload = payload.decode(errors="replace")
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except Exception:
            decoded = None
        if isinstance(decoded, dict) and decoded.get("code") is not None:
            return str(decoded["code"])
    elif isinstance(payload, dict) and payload.get("code") is not None:
        return str(payload["code"])

    msg = str(exc)
    match = re.search(r"['\"]code['\"]\s*:\s*['\"]?(\d+)['\"]?", msg)
    if match:
        return match.group(1)
    match = re.search(r"\b(40084|25245)\b", msg)
    return match.group(1) if match else ""


class BitgetBot(CCXTBot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 64
        # Whether this API key drives a UTA / Elite (copy-trading) account.
        # Auto-detected once in update_exchange_config(); classic accounts keep
        # is_uta == False and every code path below behaves exactly as before.
        self.is_uta = False
        self._account_mode_detected = False

    def create_ccxt_sessions(self):
        """Bitget: set Passivbot channel code for broker rebate attribution."""
        super().create_ccxt_sessions()
        if not isinstance(self.broker_code, str) or not self.broker_code:
            raise ValueError("Bitget broker code must be a non-empty string")
        for client in [self.cca, self.ccp]:
            if client is not None:
                client.options["broker"] = self.broker_code
                # Preserve UTA routing across ccxt session re-creation (reconnects)
                # once the account mode has been detected.
                if getattr(self, "is_uta", False):
                    client.options["uta"] = True

    async def _detect_account_mode(self):
        """Detect whether this API key is a UTA / Elite (copy-trading) account
        (Bitget v3 API) or a classic v2/mix account, and route ccxt accordingly.

        Probes the v3 account-assets endpoint: it succeeds on UTA/Elite keys.
        Explicit classic-account responses select classic mode; any other
        failure is inconclusive and must fail loudly so a UTA key is not routed
        into classic v2/mix calls for the whole process lifetime."""
        if getattr(self, "_account_mode_detected", False):
            return
        try:
            await self.cca.private_uta_get_v3_account_assets()
            is_uta = True
        except Exception as e:
            code = _bitget_error_code_from_exception(e)
            if code not in _CLASSIC_ACCOUNT_MODE_CODES:
                raise RuntimeError(
                    "bitget: UTA account-mode detection failed inconclusively; "
                    f"error_code={code or 'unknown'}"
                ) from e
            is_uta = False
        self.is_uta = is_uta
        for client in (getattr(self, "cca", None), getattr(self, "ccp", None)):
            if client is not None:
                if getattr(client, "options", None) is None:
                    client.options = {}
                client.options["uta"] = is_uta
        self._account_mode_detected = True
        logging.info(
            "[bitget] account mode: %s",
            "UTA / Elite copy-trading (v3)" if is_uta else "Classic (v2/mix)",
        )

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    def _get_position_side_for_order(self, order: dict) -> str:
        """Bitget provides posSide in info."""
        return order.get("info", {}).get("posSide", "long").lower()

    def get_symbol_id(self, symbol):
        """Return the exchange-native identifier for `symbol`, caching defaults."""
        if symbol in self.symbol_ids:
            return self.symbol_ids[symbol]
        logging.debug(f"symbol {symbol} missing from self.symbol_ids, using as-is")
        self.symbol_ids[symbol] = symbol
        return symbol

    def set_market_specific_settings(self):
        """Bitget override: higher minimum cost floor (5.1 USDT)."""
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            # Bitget requires minimum 5.1 USDT per order
            self.min_costs[symbol] = max(5.1, elm["limits"]["cost"]["min"] or 0.1)

    def _normalize_order_update(self, order: dict) -> dict:
        """Bitget override: derive side from tradeSide/posSide."""
        order["position_side"] = self._get_position_side_for_order(order)
        order["qty"] = order["amount"]
        order["side"] = self._determine_side(order)
        return order

    def _determine_side(self, order: dict) -> str:
        if getattr(self, "is_uta", False):
            # UTA orders carry posSide (long/short) and reduceOnly (yes/no) but
            # no tradeSide. Derive the order side from those.
            info = order.get("info", {})
            pos_side = str(info.get("posSide", order.get("position_side", "long"))).lower()
            reduce_raw = info.get("reduceOnly", order.get("reduceOnly", False))
            reduce_only = str(reduce_raw).strip().lower() in ("yes", "true", "1")
            if reduce_only:
                return "sell" if pos_side == "long" else "buy"
            return "buy" if pos_side == "long" else "sell"
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

    def _normalize_open_orders(self, fetched: list) -> list:
        """Bitget override: derive side from tradeSide/posSide."""
        for elm in fetched:
            elm["position_side"] = elm["info"]["posSide"]
            elm["qty"] = elm["amount"]
            elm["custom_id"] = elm["clientOrderId"]
            elm["side"] = self._determine_side(elm)
            self._record_live_margin_mode_from_payload(elm)
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def fetch_open_orders(self, symbol: str = None):
        fetched = await self._do_fetch_open_orders(symbol=symbol)
        return self._normalize_open_orders(fetched)

    async def fetch_positions(self):
        """Bitget: use CCXT unified fields (contracts, entryPrice, side)."""
        fetched = await self.cca.fetch_positions()
        for elm in fetched:
            elm["position_side"] = elm["side"]
            elm["size"] = elm["contracts"]
            elm["price"] = elm["entryPrice"]
            margin_mode = self._extract_live_margin_mode(elm)
            if margin_mode is not None:
                elm["margin_mode"] = margin_mode
                self._record_live_margin_mode(elm["symbol"], margin_mode)
        return fetched

    async def capture_positions_snapshot(self) -> tuple[list, list]:
        fetched = await self._do_fetch_positions()
        return fetched, self.fetch_positions_from_fetched(deepcopy(fetched))

    def fetch_positions_from_fetched(self, fetched: list) -> list:
        for elm in fetched:
            elm["position_side"] = elm["side"]
            elm["size"] = elm["contracts"]
            elm["price"] = elm["entryPrice"]
            margin_mode = self._extract_live_margin_mode(elm)
            if margin_mode is not None:
                elm["margin_mode"] = margin_mode
                self._record_live_margin_mode(elm["symbol"], margin_mode)
        return fetched

    async def _do_fetch_balance(self) -> dict:
        if getattr(self, "is_uta", False):
            return await self.cca.private_uta_get_v3_account_assets()
        return await super()._do_fetch_balance()

    def _get_balance(self, fetched: dict) -> float:
        """Bitget override: handle union margin mode (classic) and UTA/elite."""
        if getattr(self, "is_uta", False):
            data = fetched.get("data") if isinstance(fetched, dict) else None
            if isinstance(data, dict):
                # Passivbot balance is wallet balance: equity - upnl.
                # Bitget UTA's effEquity is discounted effective margin value,
                # not account balance.
                if data.get("usdtEquity") not in (None, "") and data.get(
                    "usdtUnrealisedPnl"
                ) not in (None, ""):
                    return float(data["usdtEquity"]) - float(data["usdtUnrealisedPnl"])
                if data.get("accountEquity") not in (None, "") and data.get(
                    "unrealisedPnl"
                ) not in (None, ""):
                    return float(data["accountEquity"]) - float(data["unrealisedPnl"])
            raise ValueError(
                "bitget: UTA balance response missing required account equity/upnl fields; "
                f"context={_bitget_payload_context(fetched)}"
            )
        balance_info = [x for x in fetched["info"] if x["marginCoin"] == self.quote][0]
        if (
            "assetMode" in balance_info
            and "unionTotalMargin" in balance_info
            and balance_info["assetMode"] == "union"
        ):
            return float(balance_info["unionTotalMargin"]) - float(balance_info["unrealizedPL"])
        return float(balance_info["available"])

    # ═══════════════════ BITGET-SPECIFIC METHODS ═══════════════════

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        raise NotImplementedError(
            "Bitget fetch_pnls legacy PnL path is unsupported; use fetch_fill_events"
        )

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
        logging.debug(f"fetching order detail for {symbol} {order_id}")
        if getattr(self, "is_uta", False):
            return await self.cca.private_uta_get_v3_trade_order_info(
                params={
                    "category": "USDT-FUTURES",
                    "orderId": order_id,
                }
            )
        return await self.cca.private_mix_get_v2_mix_order_detail(
            params={
                "productType": "USDT-FUTURES",
                "orderId": order_id,
                "symbol": symbol,
            }
        )

    async def _ensure_client_oid_for_event(self, event: dict) -> None:
        order_id = event.get("order_id") or event.get("id")
        if not order_id:
            return
        if not hasattr(self, "_client_oid_cache"):
            self._client_oid_cache = {}
        cached = self._client_oid_cache.get(order_id)
        if cached:
            event["client_order_id"], event["pb_order_type"] = cached
            return
        try:
            order_details = await self._throttled_order_detail(
                order_id, self.get_symbol_id(event["symbol"])
            )
            client_oid = order_details.get("data", {}).get("clientOid")
            if client_oid:
                pb_type = custom_id_to_snake(client_oid)
                event["client_order_id"] = client_oid
                event["pb_order_type"] = pb_type
                self._client_oid_cache[order_id] = (client_oid, pb_type)
            else:
                logging.debug(
                    "bitget order detail missing clientOid for id=%s symbol=%s",
                    order_id,
                    symbol_to_coin(event["symbol"] or "", verbose=False)
                    or event["symbol"],
                )
        except Exception as exc:
            logging.warning(
                "failed to fetch bitget order detail for id=%s symbol=%s: %s",
                order_id,
                symbol_to_coin(event["symbol"] or "", verbose=False)
                or event["symbol"],
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
            evt_id = evt.get("order_id") or evt.get("id")
            cid = evt.get("client_order_id")
            pb = evt.get("pb_order_type")
            if evt_id and cid and evt_id not in self._client_oid_cache:
                self._client_oid_cache[evt_id] = (cid, pb)

    async def fetch_fill_events(self, start_time=None, end_time=None, limit=None):
        if getattr(self, "is_uta", False):
            return await self._fetch_fill_events_uta(start_time, end_time, limit)

        def _extract_fill(elm: dict) -> dict:
            timestamp = int(elm["cTime"])
            side, position_side = deduce_side_pside(elm)
            order_id = str(elm.get("orderId") or "")
            fill_id = (
                elm.get("tradeId")
                or elm.get("fillId")
                or elm.get("execId")
                or elm.get("id")
            )
            if fill_id:
                event_id = str(fill_id)
            else:
                event_id = json.dumps(
                    [
                        order_id,
                        timestamp,
                        side,
                        position_side,
                        elm.get("baseVolume"),
                        elm.get("price"),
                    ],
                    separators=(",", ":"),
                )
            return {
                "id": event_id,
                "order_id": order_id,
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
                order_id = event.get("order_id") or event_id
                cached = self._client_oid_cache.get(order_id)
                if cached:
                    event["client_order_id"], event["pb_order_type"] = cached
                if not event.get("client_order_id"):
                    pending.append(
                        (
                            order_id,
                            event,
                            asyncio.create_task(self._ensure_client_oid_for_event(event)),
                        )
                    )
                else:
                    self._client_oid_cache[order_id] = (
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

    async def _fetch_fill_events_uta(self, start_time=None, end_time=None, limit=None):
        """UTA/elite variant of fetch_fill_events using v3 trade fills.

        v3 fills already include clientOid and execPnl, so no per-order detail
        enrichment is needed. Deduped by execId."""

        def _extract(elm: dict) -> dict:
            event = normalize_uta_fill_payload(elm, self.get_symbol_id_inv, custom_id_to_snake)
            event["info"] = elm
            return event

        limit = 100 if limit is None else min(limit, 100)
        max_n_fetches = 200
        buffer_step_ms = int(1000 * 60 * 60 * 24)
        max_window_ms = 30 * buffer_step_ms - 1
        end_time = int(utc_ms() + 3600000 if end_time is None else end_time)
        events_map: Dict[str, dict] = {}
        params = {"category": "USDT-FUTURES", "limit": str(limit)}
        count = 0
        cursor = None
        while True:
            count += 1
            if count >= max_n_fetches:
                logging.warning(f"over {count} calls to fetch_fill_events (uta). Breaking.")
                break
            request_params = dict(params)
            request_params["endTime"] = end_time
            window_start = None
            if start_time is not None:
                window_start = max(int(start_time), int(end_time) - max_window_ms)
                request_params["startTime"] = window_start
            if cursor:
                request_params["cursor"] = cursor
            fetched = await self.cca.private_uta_get_v3_trade_fills(request_params)
            data = fetched.get("data") or {}
            rows = data.get("list") or []
            fill_events = [_extract(x) for x in rows]
            fill_events.sort(key=lambda x: x["timestamp"])
            if not fill_events:
                if start_time is not None and window_start is not None and window_start > start_time:
                    next_end = int(window_start) - 1
                    if next_end <= start_time:
                        break
                    end_time = next_end
                    cursor = None
                    continue
                break
            added = False
            for fe in fill_events:
                key = fe["id"]
                if key not in events_map:
                    events_map[key] = fe
                    added = True
            if not added:
                break
            next_cursor = data.get("cursor")
            if len(fill_events) >= limit and next_cursor:
                cursor = str(next_cursor)
                continue
            cursor = None
            if len(fill_events) < limit:
                if start_time is None:
                    break
                if window_start is not None and window_start > start_time:
                    next_end = int(window_start) - 1
                    if next_end <= start_time:
                        break
                    end_time = next_end
                    continue
                if end_time - start_time < buffer_step_ms:
                    break
                end_time = int(fill_events[0]["timestamp"] + 1)
                continue
            if start_time is None or fill_events[0]["timestamp"] < start_time:
                break
            if end_time == fill_events[0]["timestamp"]:
                break
            end_time = int(fill_events[0]["timestamp"])
        return sorted(events_map.values(), key=lambda x: x["timestamp"])

    async def fetch_closed_orders(self, start_time, end_time, limit=100):
        if getattr(self, "is_uta", False):
            return await self._fetch_fill_events_uta(start_time, end_time, limit)

        def extract_fill_event_from_co(elm):
            timestamp = int(elm["lastUpdateTimestamp"])
            price = float(elm["price"])
            qty = float(elm["filled"])
            info = elm.get("info")
            if not isinstance(info, dict):
                raise ValueError(
                    "bitget closed-order payload missing info; "
                    f"context={_bitget_payload_context(elm)}"
                )
            if info.get("totalProfits") in (None, ""):
                raise ValueError(
                    "bitget closed-order payload missing info.totalProfits; "
                    f"context={_bitget_payload_context(elm)}"
                )
            try:
                pnl = float(info["totalProfits"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "bitget closed-order info.totalProfits is not numeric; "
                    f"value={info.get('totalProfits')!r} context={_bitget_payload_context(elm)}"
                ) from exc
            position_side = str(info.get("posSide") or "").lower()
            if position_side not in ("long", "short"):
                raise ValueError(
                    "bitget closed-order payload missing valid info.posSide; "
                    f"context={_bitget_payload_context(elm)}"
                )
            pb_order_type = custom_id_to_snake(elm.get("clientOrderId"))
            if not pb_order_type or pb_order_type == "unknown":
                if not hasattr(self, "pb_order_type_missing_logged"):
                    self.pb_order_type_missing_logged = set()
                key = json.dumps(elm)
                if key not in self.pb_order_type_missing_logged:
                    logging.info(
                        "bitget fill without pb_order_type id=%s symbol=%s clientOrderId=%s %s %s %s @ %s",
                        elm.get("id"),
                        symbol_to_coin(elm.get("symbol") or "", verbose=False)
                        or elm.get("symbol"),
                        elm.get("clientOrderId"),
                        elm.get("side"),
                        position_side,
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
                "pnl": pnl,
                "fees": elm.get("fees"),
                "pb_order_type": pb_order_type,
                "position_side": position_side,
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

    def _build_order_params(self, order: dict) -> dict:
        use_post_only = require_live_value(self.config, "time_in_force") == "post_only"
        if getattr(self, "is_uta", False):
            # UTA/elite hedge-mode orders use posSide; holdSide/oneWayMode are
            # v2-only and rejected by v3 with code 25236.
            params = {
                "posSide": order["position_side"],
                "reduceOnly": order["reduce_only"],
                "clientOid": order["custom_id"],
            }
            if use_post_only:
                params["postOnly"] = True
            else:
                params["timeInForce"] = "gtc"
            return params
        tif = "PO" if use_post_only else "GTC"
        return {
            "timeInForce": tif,
            "holdSide": order["position_side"],
            "reduceOnly": order["reduce_only"],
            "oneWayMode": False,
            "clientOid": order["custom_id"],
        }

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
            # UTA/elite: ccxt has no v3 routing for set_margin_mode and the elite
            # portfolio is cross-only, so skip it (avoids a failing v2 call).
            if not getattr(self, "is_uta", False):
                try:
                    coros_to_call_margin_mode[symbol] = asyncio.create_task(
                        self.cca.set_margin_mode(
                            margin_mode,
                            symbol=symbol,
                        )
                    )
                except Exception as e:
                    logging.error(f"{log_symbol}: error setting {margin_mode} mode {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self.cca.set_leverage(self._calc_leverage_for_symbol(symbol), symbol=symbol)
                )
            except Exception as e:
                logging.error(f"{log_symbol}: error setting leverage {e}")
        for symbol in symbols:
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f"leverage={format_exchange_config_response(res)} "
            except Exception as e:
                logging.error(f"{log_symbol} error setting leverage {e}")
            res = None
            if symbol in coros_to_call_margin_mode:
                try:
                    res = await coros_to_call_margin_mode[symbol]
                    to_print += f"margin={format_exchange_config_response(res)}"
                except Exception as e:
                    logging.error(
                        f"{log_symbol} error setting {self._get_margin_mode_for_symbol(symbol)} mode {e}"
                    )
            if to_print:
                logging.info(f"{log_symbol}: {to_print}")

    async def calc_ideal_orders(self):
        # Bitget returns max 100 open orders per fetch_open_orders.
        # Only create 100 open orders.
        # Drop orders whose pprice diff is greatest.
        ideal_orders = await super().calc_ideal_orders()
        market_prices = await self._get_live_last_prices(
            ideal_orders.keys(),
            max_age_ms=10_000,
            context="bitget_order_cap_sort",
            allow_completed_candle_fallback=True,
        )
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append(
                    (
                        calc_order_price_diff(
                            x["side"],
                            x["price"],
                            market_prices[s],
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
        # Detect classic vs UTA/elite once, before any balance/position/order call.
        await self._detect_account_mode()
        res = None
        try:
            res = await self.cca.set_position_mode(True)
            logging.debug("[config] set hedge mode response: %s", res)
        except Exception as e:
            logging.error("[config] error setting hedge mode: %s %s", e, res)
            raise

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        if getattr(self, "is_uta", False):
            return formatted[:32]
        return (self.broker_code + "#" + formatted)[: self.custom_id_max_length]
