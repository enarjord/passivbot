"""Complete Lighter trade history and one-way position-side normalization."""

from __future__ import annotations

from decimal import Decimal

from exchanges.lighter import decode_client_id, finite
from fill_events_manager import BaseFetcher, _fill_datetime
from passivbot import custom_id_to_snake, custom_id_has_explicit_passivbot_marker


class LighterFetcher(BaseFetcher):
    def __init__(self, api, *, trade_limit=100):
        self.api = api
        self.trade_limit = max(1, min(100, int(trade_limit)))

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        await self.api.load_markets()
        await self.api.prepare_api_key()
        params = {
            "account_index": self.api.options["accountIndex"],
            "market_type": "perp",
            "sort_by": "timestamp",
            "sort_dir": "desc",
            "limit": self.trade_limit,
        }
        if until_ms is not None:
            params["from"] = int(until_ms)
        cursors, trades, events = set(), {}, []
        for _ in range(10000):
            response = await self.api.privateGetTrades(params)
            rows = response["trades"]
            if not isinstance(rows, list):
                raise ValueError("Lighter trade page must contain a list")
            oldest = None
            added = 0
            for row in rows:
                timestamp = int(row["timestamp"])
                if timestamp <= 0:
                    raise ValueError("Lighter invalid trade timestamp")
                oldest = timestamp if oldest is None else min(oldest, timestamp)
                identity = str(row["trade_id"])
                if identity in trades:
                    if row != trades[identity]:
                        raise ValueError("Lighter conflicting duplicate trade")
                    continue
                trades[identity] = row
                added += 1
                if since_ms is not None and timestamp < since_ms:
                    continue
                if until_ms is not None and timestamp > until_ms:
                    continue
                events.extend(self.normalize_trade(row))
            cursor = response.get("next_cursor")
            if not cursor or (
                since_ms is not None and oldest is not None and oldest < since_ms
            ):
                break
            if cursor in cursors or not rows or not added:
                raise ValueError("Lighter trade pagination made no progress")
            cursors.add(cursor)
            params["cursor"] = cursor
        else:
            raise ValueError("Lighter trade pagination exceeded its safety bound")
        events.sort(key=lambda e: (e["timestamp"], int(e["id"].split(":")[0]), e["id"]))
        if on_batch and events:
            on_batch(events)
        return events

    def normalize_trade(self, row):
        return normalize_lighter_trade(
            row,
            int(self.api.options["accountIndex"]),
            self.api.markets_by_id[str(row["market_id"])][0],
        )


def normalize_lighter_trade(row, account, market):
    ask, bid = int(row["ask_account_id"]), int(row["bid_account_id"])
    if (ask == account) == (bid == account):
        raise ValueError("Lighter trade must identify exactly one account side")
    leg, side = ("ask", "sell") if ask == account else ("bid", "buy")
    maker_ask = row["is_maker_ask"]
    if not isinstance(maker_ask, bool):
        raise ValueError("Lighter is_maker_ask must be boolean")
    role = "maker" if maker_ask == (leg == "ask") else "taker"
    before = finite(row[f"{role}_position_size_before"], "pre-fill position")
    entry_quote = finite(row[f"{role}_entry_quote_before"], "pre-fill entry quote")
    if entry_quote < 0 or (before == 0 and entry_quote != 0):
        raise ValueError("Lighter contradictory pre-fill position and entry quote")
    quantity = finite(row["size"], "fill size", positive=True)
    price = finite(row["price"], "fill price", positive=True)
    if not market["swap"] or market["settle"] != "USDC":
        raise ValueError("Lighter fill is not a USDC perpetual")
    delta = quantity if side == "buy" else -quantity
    reducing = before * delta < 0
    close_qty = min(abs(before), quantity) if reducing else 0.0
    pnl_key = f"{leg}_account_pnl"
    # Opening/increasing exposure has no realized price PnL. Reductions
    # require the venue's reported PnL, including an explicitly reported zero.
    if reducing and pnl_key not in row:
        # Sparse JSON may omit a zero PnL. Accept that only if the exact
        # exchange before-state proves zero at USDC's six-decimal precision.
        native_before = abs(Decimal(str(row[f"{role}_position_size_before"])))
        gross = (
            (
                Decimal(str(row["price"]))
                - Decimal(str(row[f"{role}_entry_quote_before"])) / native_before
            )
            * min(native_before, Decimal(str(row["size"])))
            * (1 if before > 0 else -1)
        )
        if abs(gross) >= Decimal("0.000001"):
            raise ValueError("Lighter reducing fill is missing realized PnL")
        pnl = 0.0
    else:
        pnl = finite(row[pnl_key], "realized PnL") if reducing else 0.0
    if not reducing and pnl_key in row and finite(row[pnl_key], "realized PnL") != 0:
        raise ValueError("Lighter opening fill has nonzero realized PnL")
    client_id = decode_client_id(row[f"{leg}_client_id"])
    pb_type = (
        custom_id_to_snake(client_id)
        if custom_id_has_explicit_passivbot_marker(client_id)
        else "unknown"
    )
    fees = None
    fee_key = f"{role}_fee"
    if fee_key in row:
        # Official lighter-go FeeTick is 1,000,000. Missing fee evidence
        # remains absent for the canonical observable best-effort fee policy.
        rate = finite(row[fee_key], "fee rate") / 1_000_000
        integrator_key = f"integrator_{role}_fee"
        if integrator_key in row:
            rate += finite(row[integrator_key], "integrator fee rate") / 1_000_000
        fees = {
            "currency": "USDC",
            "cost": float(Decimal(str(row["usd_amount"])) * Decimal(str(rate))),
        }
    pieces = []
    if close_qty:
        pieces.append(("close", "long" if before > 0 else "short", close_qty, pnl))
    remainder = float(Decimal(str(quantity)) - Decimal(str(close_qty)))
    if remainder > 0:
        pieces.append(("open", "long" if delta > 0 else "short", remainder, 0.0))
    events = []
    for kind, pside, qty, piece_pnl in pieces:
        fee = None if fees is None else {**fees, "cost": fees["cost"] * qty / quantity}
        timestamp = int(row["timestamp"])
        if kind == "close":
            after_size = max(0.0, abs(before) - qty)
            after_price = entry_quote / abs(before) if after_size else 0.0
        elif reducing:
            after_size, after_price = qty, price
        else:
            after_size = abs(before) + qty
            after_price = (entry_quote + qty * price) / after_size
        event_id = str(row["trade_id"]) + (f":{kind}" if len(pieces) > 1 else "")
        events.append(
            {
                "id": event_id,
                # Each flip leg is a distinct execution component. The raw
                # trade ID alone would make the cache replace one with the other.
                "source_ids": [event_id],
                "order_id": str(row[f"{leg}_id"]),
                "timestamp": timestamp,
                "datetime": _fill_datetime(timestamp),
                "symbol": market["symbol"],
                "side": side,
                "qty": qty,
                "price": price,
                "pnl": piece_pnl,
                "fees": fee,
                "position_side": pside,
                "client_order_id": client_id,
                "pb_order_type": pb_type,
                "c_mult": 1.0,
                "raw": [
                    {
                        "source": "lighter_trades",
                        "data": dict(row),
                        "account_index": account,
                        "piece": kind,
                    }
                ],
                "psize": after_size,
                "pprice": after_price,
            }
        )
    return events


def apply_lighter_raw_position_overrides(events):
    """Rebuild after-state from per-fill exchange evidence even on a truncated window."""
    for event in events:
        rows = [
            r
            for r in event.get("raw", [])
            if isinstance(r, dict) and r.get("source") == "lighter_trades"
        ]
        if not rows:
            continue
        if len(rows) != 1:
            raise ValueError(
                "Lighter fill must retain one authoritative trade component"
            )
        raw = rows[0]
        pieces = normalize_lighter_trade(
            raw["data"],
            raw["account_index"],
            {"symbol": event["symbol"], "swap": True, "settle": "USDC"},
        )
        matches = [
            p
            for p in pieces
            if p["id"] == event["id"] and p["position_side"] == event["position_side"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "Lighter fill identity contradicts its raw position evidence"
            )
        match = matches[0]
        for field in ("side", "timestamp", "client_order_id"):
            if str(event[field]) != str(match[field]):
                raise ValueError("Lighter fill contradicts its raw execution evidence")
        if (
            abs(abs(float(event["qty"])) - match["qty"]) > 1e-12
            or float(event["price"]) != match["price"]
        ):
            raise ValueError("Lighter fill quantity or price contradicts raw evidence")
        for field in ("psize", "pprice"):
            event[field] = matches[0][field]
