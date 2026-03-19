from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hjson

from exchanges.ccxt_bot import CCXTBot
from ohlcv_utils import load_ohlcv_data
from pure_funcs import ts_to_date

logger = logging.getLogger(__name__)


def _parse_time_to_ms(value: Any) -> int:
    if value is None:
        raise ValueError("Fake scenario timestamp is required")
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts < 10**11:
            ts *= 1000
        return ts
    text = str(value).strip()
    if not text:
        raise ValueError("Fake scenario timestamp is empty")
    try:
        ts = int(text)
        if ts < 10**11:
            ts *= 1000
        return ts
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load_fake_scenario(path: str | Path) -> dict:
    scenario_path = Path(path)
    try:
        with scenario_path.open("r", encoding="utf-8") as handle:
            data = hjson.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load fake scenario {scenario_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Fake scenario {scenario_path} must contain a mapping at the top level")
    data["_scenario_path"] = str(scenario_path)
    return data


def _copy_order(order: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(order)
    copied["info"] = dict(order.get("info") or {})
    return copied


def _parse_timeframe_to_ms(timeframe: str) -> int:
    text = str(timeframe or "").strip().lower()
    if text == "1m":
        return 60_000
    if len(text) < 2:
        raise ValueError(f"Unsupported fake timeframe {timeframe!r}")
    unit = text[-1]
    try:
        count = int(text[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported fake timeframe {timeframe!r}") from exc
    if count <= 0:
        raise ValueError(f"Unsupported fake timeframe {timeframe!r}")
    if unit == "m":
        return count * 60_000
    if unit == "h":
        return count * 3_600_000
    if unit == "d":
        return count * 86_400_000
    raise ValueError(f"Unsupported fake timeframe {timeframe!r}")


class FakeCCXTClient:
    id = "fake"

    def __init__(self, scenario: dict, *, quote: str = "USDT") -> None:
        self.has = {
            "fetchBalance": True,
            "fetchPositions": True,
            "fetchOpenOrders": True,
            "fetchTickers": True,
            "fetchTicker": True,
            "fetchOHLCV": True,
            "fetchMyTrades": True,
            "fetchTime": True,
            "createOrder": True,
            "cancelOrder": True,
            "setPositionMode": True,
            "setLeverage": True,
            "setMarginMode": True,
            "watchOrders": False,
        }
        self.options: Dict[str, Any] = {}
        self.urls: Dict[str, Any] = {}
        self.quote = quote
        self.scenario = copy.deepcopy(scenario)
        self.scenario_name = str(
            self.scenario.get("name")
            or Path(str(self.scenario.get("_scenario_path", "scenario"))).stem
        )

        self.tick_interval_ms = int(float(self.scenario.get("tick_interval_seconds", 60)) * 1000)
        if self.tick_interval_ms <= 0:
            raise ValueError("Fake scenario tick_interval_seconds must be > 0")
        self.start_time_ms = (
            _parse_time_to_ms(self.scenario.get("start_time"))
            if self.scenario.get("start_time") is not None
            else None
        )

        account = self.scenario.get("account") or {}
        self.balance_total = float(account.get("balance", 0.0))
        self.balance_free = float(account.get("balance", 0.0))
        self.realized_pnl = 0.0
        self.realized_fees = 0.0
        self.position_mode = True
        self.leverage_by_symbol: Dict[str, int] = {}
        self.margin_mode_by_symbol: Dict[str, str] = {}

        self.markets = self._build_markets(self.scenario.get("symbols") or {})
        self.markets_by_id = {market["id"]: market for market in self.markets.values()}
        self.symbols = sorted(self.markets)

        self.timeline = self._build_timeline(self.scenario.get("timeline") or [])
        if not self.timeline:
            self.timeline = self._build_replay_timeline(self.scenario.get("replay") or {})
        if not self.timeline:
            raise ValueError("Fake scenario must define timeline rows or replay candles")
        if self.start_time_ms is None:
            self.start_time_ms = int(self.timeline[0]["timestamp"])

        self.boot_index = int(self.scenario.get("boot_index", 0))
        if not 0 <= self.boot_index < len(self.timeline):
            raise ValueError(
                f"Fake scenario boot_index {self.boot_index} is outside timeline size {len(self.timeline)}"
            )
        self.current_index = self.boot_index
        self.now_ms = int(self.timeline[self.current_index]["timestamp"])

        self.positions: Dict[Tuple[str, str], Dict[str, float | str]] = {}
        for symbol in self.symbols:
            for pside in ("long", "short"):
                self.positions[(symbol, pside)] = {
                    "symbol": symbol,
                    "position_side": pside,
                    "size": 0.0,
                    "entry_price": 0.0,
                }
        for position in account.get("positions") or []:
            self._load_boot_position(position)

        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.fills: List[Dict[str, Any]] = []
        self._next_order_id = 1
        self._next_trade_id = 1

        for fill in account.get("fills") or []:
            self._load_boot_fill(fill)
        for order in account.get("open_orders") or []:
            self._load_boot_order(order)
        self._process_resting_orders_for_current_step()

    @classmethod
    def from_config(cls, config: dict, user_info: dict) -> "FakeCCXTClient":
        live = config.get("live") or {}
        scenario_path = live.get("fake_scenario_path") or user_info.get("fake_scenario_path")
        if not scenario_path:
            raise ValueError(
                "Fake exchange requires live.fake_scenario_path or api-keys fake_scenario_path"
            )
        scenario = load_fake_scenario(scenario_path)
        quote = str(user_info.get("quote") or scenario.get("quote") or "USDT")
        return cls(scenario, quote=quote)

    def _build_markets(self, symbols_config: dict) -> Dict[str, dict]:
        if not symbols_config:
            raise ValueError("Fake scenario must define symbols")
        markets: Dict[str, dict] = {}
        for symbol, meta in symbols_config.items():
            if not isinstance(meta, dict):
                raise TypeError(f"Fake symbol config for {symbol} must be a mapping")
            base, quote = self._split_symbol(symbol)
            qty_step = float(meta.get("qty_step", 0.001))
            price_step = float(meta.get("price_step", 0.1))
            min_qty = float(meta.get("min_qty", qty_step))
            min_cost = float(meta.get("min_cost", 5.0))
            contract_size = float(meta.get("contractSize", 1.0))
            markets[symbol] = {
                "id": str(meta.get("id") or symbol.replace("/", "").replace(":", "_")),
                "symbol": symbol,
                "base": base,
                "quote": quote,
                "settle": quote,
                "type": "swap",
                "swap": True,
                "linear": True,
                "inverse": False,
                "active": True,
                "precision": {"amount": qty_step, "price": price_step},
                "limits": {"amount": {"min": min_qty}, "cost": {"min": min_cost}},
                "contractSize": contract_size,
                "maker": float(meta.get("maker_fee", 0.0002)),
                "taker": float(meta.get("taker_fee", 0.00055)),
                "info": dict(meta),
            }
        return markets

    @staticmethod
    def _split_symbol(symbol: str) -> Tuple[str, str]:
        if "/" not in symbol:
            raise ValueError(f"Fake symbol '{symbol}' must look like BASE/QUOTE:SETTLE")
        left, right = symbol.split("/", 1)
        quote = right.split(":")[0]
        return left, quote

    def _build_timeline(self, timeline_rows: List[dict]) -> List[Dict[str, Any]]:
        if not timeline_rows:
            return []
        if self.start_time_ms is None:
            raise ValueError("Fake scripted timeline requires scenario start_time")
        prev_prices: Dict[str, float] = {}
        candles_by_symbol: Dict[str, List[List[float]]] = {symbol: [] for symbol in self.markets}
        normalized: List[Dict[str, Any]] = []
        for idx, row in enumerate(timeline_rows):
            if not isinstance(row, dict):
                raise TypeError("Fake timeline rows must be mappings")
            step_num = int(row.get("t", idx))
            prices_update = row.get("prices") or {}
            if not isinstance(prices_update, dict):
                raise TypeError("Fake timeline row prices must be a mapping")
            prices = {symbol: float(prev_prices[symbol]) for symbol in prev_prices}
            for symbol, price in prices_update.items():
                prices[symbol] = float(price)
            missing = [symbol for symbol in self.markets if symbol not in prices]
            if missing:
                raise ValueError(
                    f"Fake timeline row {idx} missing prices for symbols: {', '.join(sorted(missing))}"
                )
            timestamp = self.start_time_ms + step_num * self.tick_interval_ms
            step_candles: Dict[str, List[float]] = {}
            for symbol, close_price in prices.items():
                open_price = float(prev_prices.get(symbol, close_price))
                high_price = max(open_price, close_price)
                low_price = min(open_price, close_price)
                candle = [
                    float(timestamp),
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    float(row.get("volume", 0.0)),
                ]
                candles_by_symbol[symbol].append(candle)
                step_candles[symbol] = candle
            normalized.append(
                {
                    "index": idx,
                    "t": step_num,
                    "timestamp": timestamp,
                    "datetime": ts_to_date(timestamp),
                    "prices": dict(prices),
                    "candles": step_candles,
                }
            )
            prev_prices = prices
        self._candles_by_symbol = candles_by_symbol
        return normalized

    def _build_replay_timeline(self, replay_config: dict) -> List[Dict[str, Any]]:
        if not replay_config:
            return []
        symbol_specs = replay_config.get("symbols")
        if not symbol_specs:
            if len(self.symbols) != 1:
                raise ValueError("Fake replay config without replay.symbols requires exactly one market")
            symbol_specs = {self.symbols[0]: replay_config}
        if not isinstance(symbol_specs, dict):
            raise TypeError("Fake replay.symbols must be a mapping")

        replay_start = (
            _parse_time_to_ms(replay_config["start_time"])
            if replay_config.get("start_time") is not None
            else None
        )
        replay_end = (
            _parse_time_to_ms(replay_config["end_time"])
            if replay_config.get("end_time") is not None
            else None
        )
        source_dir = replay_config.get("source_dir")

        per_symbol_rows: Dict[str, List[List[float]]] = {}
        for symbol in self.symbols:
            spec = symbol_specs.get(symbol)
            if spec is None:
                raise ValueError(f"Fake replay missing symbol spec for {symbol}")
            rows = self._load_replay_rows(symbol, spec, source_dir=source_dir)
            if replay_start is not None:
                rows = [row for row in rows if int(row[0]) >= replay_start]
            if replay_end is not None:
                rows = [row for row in rows if int(row[0]) <= replay_end]
            if not rows:
                raise ValueError(f"Fake replay for {symbol} produced no candles after filtering")
            per_symbol_rows[symbol] = rows

        all_timestamps = sorted({int(row[0]) for rows in per_symbol_rows.values() for row in rows})
        if not all_timestamps:
            return []

        last_candle_by_symbol: Dict[str, List[float]] = {}
        row_index_by_symbol = {symbol: 0 for symbol in self.symbols}
        candles_by_symbol: Dict[str, List[List[float]]] = {symbol: [] for symbol in self.symbols}
        normalized: List[Dict[str, Any]] = []

        for index, timestamp in enumerate(all_timestamps):
            step_candles: Dict[str, List[float]] = {}
            prices: Dict[str, float] = {}
            for symbol in self.symbols:
                rows = per_symbol_rows[symbol]
                row_index = row_index_by_symbol[symbol]
                next_row = rows[row_index] if row_index < len(rows) else None
                if next_row is not None and int(next_row[0]) == timestamp:
                    candle = list(next_row)
                    row_index_by_symbol[symbol] += 1
                    last_candle_by_symbol[symbol] = candle
                elif symbol in last_candle_by_symbol:
                    last_close = float(last_candle_by_symbol[symbol][4])
                    candle = [float(timestamp), last_close, last_close, last_close, last_close, 0.0]
                    last_candle_by_symbol[symbol] = candle
                else:
                    raise ValueError(
                        f"Fake replay for {symbol} is missing an initial candle at {timestamp}"
                    )
                candles_by_symbol[symbol].append(candle)
                step_candles[symbol] = candle
                prices[symbol] = float(candle[4])
            normalized.append(
                {
                    "index": index,
                    "t": index,
                    "timestamp": int(timestamp),
                    "datetime": ts_to_date(int(timestamp)),
                    "prices": prices,
                    "candles": step_candles,
                }
            )

        self._candles_by_symbol = candles_by_symbol
        return normalized

    def _load_replay_rows(
        self,
        symbol: str,
        spec: dict,
        *,
        source_dir: Optional[str] = None,
    ) -> List[List[float]]:
        if not isinstance(spec, dict):
            raise TypeError(f"Fake replay spec for {symbol} must be a mapping")

        rows: List[List[float]] = []
        if spec.get("candles"):
            rows.extend(self._normalize_inline_candles(spec["candles"]))

        files: List[Path] = []
        if spec.get("file"):
            files.append(self._resolve_replay_path(str(spec["file"]), source_dir=source_dir))
        for value in spec.get("files") or []:
            files.append(self._resolve_replay_path(str(value), source_dir=source_dir))
        if spec.get("glob"):
            pattern = self._resolve_replay_path(str(spec["glob"]), source_dir=source_dir)
            files.extend(sorted(pattern.parent.glob(pattern.name)))

        for path in files:
            df = load_ohlcv_data(str(path))
            rows.extend(
                [
                    [
                        float(row["timestamp"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    ]
                    for _, row in df.iterrows()
                ]
            )

        deduped: Dict[int, List[float]] = {}
        for row in rows:
            if len(row) != 6:
                raise ValueError(f"Fake replay candle for {symbol} must have 6 columns, got {row}")
            deduped[int(row[0])] = [float(value) for value in row]
        return [deduped[key] for key in sorted(deduped)]

    def _normalize_inline_candles(self, candles: List[Any]) -> List[List[float]]:
        rows: List[List[float]] = []
        for candle in candles:
            if isinstance(candle, dict):
                rows.append(
                    [
                        float(_parse_time_to_ms(candle["timestamp"])),
                        float(candle["open"]),
                        float(candle["high"]),
                        float(candle["low"]),
                        float(candle["close"]),
                        float(candle.get("volume", 0.0)),
                    ]
                )
                continue
            if isinstance(candle, (list, tuple)) and len(candle) == 6:
                row = list(candle)
                row[0] = float(_parse_time_to_ms(row[0]))
                rows.append([float(value) for value in row])
                continue
            raise TypeError(f"Unsupported fake replay candle shape: {candle!r}")
        return rows

    def _resolve_replay_path(self, value: str, *, source_dir: Optional[str] = None) -> Path:
        path = Path(value)
        if not path.is_absolute():
            base = self._scenario_base_dir()
            if source_dir:
                source = Path(source_dir)
                base = source if source.is_absolute() else base / source
            path = base / path
        return path

    def _scenario_base_dir(self) -> Path:
        scenario_path = self.scenario.get("_scenario_path")
        if scenario_path:
            return Path(str(scenario_path)).resolve().parent
        return Path.cwd()

    def _load_boot_position(self, position: dict) -> None:
        symbol = str(position["symbol"])
        pside = str(position["position_side"]).lower()
        qty = abs(float(position["qty"]))
        price = float(position["price"])
        self.positions[(symbol, pside)] = {
            "symbol": symbol,
            "position_side": pside,
            "size": qty,
            "entry_price": price,
        }

    def _load_boot_order(self, order: dict) -> None:
        symbol = str(order["symbol"])
        order_id = str(order.get("id") or self._next_order_id)
        try:
            self._next_order_id = max(self._next_order_id, int(order_id) + 1)
        except ValueError:
            self._next_order_id += 1
        position_side = str(order.get("position_side") or "long").lower()
        amount = abs(float(order["amount"]))
        side = str(order["side"]).lower()
        price = float(order["price"])
        client_order_id = str(order.get("clientOrderId") or order.get("custom_id") or "")
        self.open_orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "type": str(order.get("type") or "limit").lower(),
            "side": side,
            "amount": amount,
            "price": price,
            "timestamp": int(order.get("timestamp") or self.now_ms),
            "clientOrderId": client_order_id,
            "status": "open",
            "reduceOnly": bool(order.get("reduce_only") or order.get("reduceOnly")),
            "filled": 0.0,
            "remaining": amount,
            "info": {
                "positionSide": position_side.upper(),
                "reduceOnly": bool(order.get("reduce_only") or order.get("reduceOnly")),
            },
        }

    def _load_boot_fill(self, fill: dict) -> None:
        symbol = str(fill["symbol"])
        if symbol not in self.markets:
            raise KeyError(f"Unknown fake symbol in boot fill: {symbol}")
        trade_id = str(fill.get("id") or fill.get("trade_id") or self._next_trade_id)
        try:
            self._next_trade_id = max(self._next_trade_id, int(trade_id) + 1)
        except ValueError:
            self._next_trade_id += 1
        order_id = str(fill.get("order") or fill.get("order_id") or fill.get("orderId") or "")
        if order_id:
            try:
                self._next_order_id = max(self._next_order_id, int(order_id) + 1)
            except ValueError:
                pass
        timestamp = int(_parse_time_to_ms(fill.get("timestamp") or self.now_ms))
        side = str(fill["side"]).lower()
        amount = abs(
            float(
                fill.get("amount")
                if fill.get("amount") is not None
                else fill.get("qty")
                if fill.get("qty") is not None
                else fill.get("size")
                if fill.get("size") is not None
                else fill.get("contracts")
            )
        )
        price = float(fill["price"])
        if amount <= 0.0:
            raise ValueError(f"Fake boot fill amount must be > 0 for {symbol}")
        if price <= 0.0:
            raise ValueError(f"Fake boot fill price must be > 0 for {symbol}")
        info = dict(fill.get("info") or {})
        position_side = str(
            fill.get("position_side") or fill.get("pside") or info.get("positionSide") or "long"
        ).lower()
        if position_side not in ("long", "short"):
            position_side = "long"
        fee_obj = fill.get("fee")
        if isinstance(fee_obj, dict):
            fee_cost = float(fee_obj.get("cost", 0.0) or 0.0)
        elif fee_obj is None:
            fee_cost = float(fill.get("fee_cost", 0.0) or 0.0)
        else:
            fee_cost = float(fee_obj)
        boot_fill = {
            "id": trade_id,
            "order": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "pnl": float(fill.get("pnl", 0.0) or 0.0),
            "fee": {"currency": self.quote, "cost": fee_cost},
            "clientOrderId": str(
                fill.get("clientOrderId")
                or fill.get("client_order_id")
                or fill.get("custom_id")
                or ""
            ),
            "position_side": position_side,
            "reduceOnly": bool(fill.get("reduceOnly") or fill.get("reduce_only")),
            "info": {
                "positionSide": position_side.upper(),
                "reduceOnly": bool(fill.get("reduceOnly") or fill.get("reduce_only")),
                "liquidity": str(info.get("liquidity") or fill.get("liquidity") or "historical"),
                "realizedPnl": float(fill.get("pnl", 0.0) or 0.0),
                "fee": fee_cost,
                "contractMultiplier": self._c_mult(symbol),
            },
        }
        self.realized_pnl += float(boot_fill["pnl"])
        self.realized_fees += fee_cost
        self.fills.append(boot_fill)

    def get_current_step(self) -> dict:
        return self.timeline[self.current_index]

    def has_next_step(self) -> bool:
        return self.current_index < len(self.timeline) - 1

    async def load_markets(self, reload: bool = True) -> Dict[str, dict]:
        return copy.deepcopy(self.markets)

    async def fetch_balance(self) -> dict:
        return {
            "timestamp": self.now_ms,
            "datetime": ts_to_date(self.now_ms),
            "free": {self.quote: float(self.balance_free)},
            "used": {self.quote: 0.0},
            "total": {self.quote: float(self.balance_total)},
            "info": {"balance": float(self.balance_total)},
        }

    async def fetch_positions(self) -> List[dict]:
        positions: List[dict] = []
        for state in self.positions.values():
            size = float(state["size"])
            if size == 0.0:
                continue
            positions.append(
                {
                    "symbol": state["symbol"],
                    "contracts": size,
                    "entryPrice": float(state["entry_price"]),
                    "side": str(state["position_side"]),
                    "info": {"positionSide": str(state["position_side"]).upper()},
                }
            )
        return positions

    async def fetch_open_orders(self, symbol: str = None) -> List[dict]:
        orders = []
        for order in self.open_orders.values():
            if symbol is not None and order["symbol"] != symbol:
                continue
            orders.append(_copy_order(order))
        return sorted(orders, key=lambda item: (item["timestamp"], item["id"]))

    async def fetch_tickers(self) -> Dict[str, dict]:
        prices = self.get_current_step()["prices"]
        return {symbol: self._ticker(symbol, prices[symbol]) for symbol in self.symbols}

    async def fetch_ticker(self, symbol: str) -> dict:
        prices = self.get_current_step()["prices"]
        if symbol not in prices:
            raise KeyError(f"Fake ticker price missing for {symbol}")
        return self._ticker(symbol, prices[symbol])

    def _ticker(self, symbol: str, last_price: float) -> dict:
        return {
            "symbol": symbol,
            "timestamp": self.now_ms,
            "datetime": ts_to_date(self.now_ms),
            "bid": float(last_price),
            "ask": float(last_price),
            "last": float(last_price),
            "info": {"source": "fake"},
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ) -> List[List[float]]:
        candles = self._candles_by_symbol.get(symbol)
        if candles is None:
            raise KeyError(f"Unknown fake symbol {symbol}")
        timeframe_ms = _parse_timeframe_to_ms(timeframe)
        params = params or {}
        until_ms = params.get("until")
        rows = [list(row) for row in candles[: self.current_index + 1]]
        if timeframe_ms != 60_000:
            rows = self._aggregate_candles(rows, timeframe_ms)
        if since is not None:
            rows = [row for row in rows if int(row[0]) >= int(since)]
        if until_ms is not None:
            rows = [row for row in rows if int(row[0]) <= int(until_ms)]
        if limit is not None:
            rows = rows[-int(limit) :]
        return rows

    async def fetch_my_trades(
        self,
        symbol: str = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict] = None,
    ) -> List[dict]:
        params = params or {}
        since_ms = since
        if since_ms is None and params.get("since") is not None:
            since_ms = int(params["since"])
        until_ms = params.get("until")
        trades = []
        for fill in self.fills:
            if symbol is not None and fill["symbol"] != symbol:
                continue
            if since_ms is not None and int(fill["timestamp"]) < int(since_ms):
                continue
            if until_ms is not None and int(fill["timestamp"]) > int(until_ms):
                continue
            trades.append(copy.deepcopy(fill))
        trades.sort(key=lambda item: (item["timestamp"], item["id"]))
        if limit is not None:
            trades = trades[-int(limit) :]
        return trades

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> dict:
        params = params or {}
        order_type = str(type or "limit").lower()
        order_side = str(side).lower()
        if symbol not in self.markets:
            raise KeyError(f"Unknown fake symbol {symbol}")
        position_side = str(
            params.get("positionSide")
            or params.get("position_side")
            or ("LONG" if order_side == "buy" else "SHORT")
        ).lower()
        if position_side not in ("long", "short"):
            position_side = "long" if order_side == "buy" else "short"
        client_order_id = str(
            params.get("clientOrderId")
            or params.get("newClientOrderId")
            or params.get("orderLinkId")
            or ""
        )
        reduce_only = bool(params.get("reduceOnly") or params.get("reduce_only"))
        order_id = str(self._next_order_id)
        self._next_order_id += 1
        order_price = float(price) if price is not None else None
        if order_type == "limit" and order_price is None:
            raise ValueError("Fake limit order requires price")
        last_price = float(self.get_current_step()["prices"][symbol])
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": order_side,
            "amount": abs(float(amount)),
            "price": float(last_price if order_type == "market" else order_price),
            "timestamp": self.now_ms,
            "datetime": ts_to_date(self.now_ms),
            "clientOrderId": client_order_id,
            "status": "open",
            "filled": 0.0,
            "remaining": abs(float(amount)),
            "reduceOnly": reduce_only,
            "info": {
                "positionSide": position_side.upper(),
                "reduceOnly": reduce_only,
            },
        }

        if order_type == "market":
            self._fill_order(order, fill_price=last_price, liquidity="taker")
            return _copy_order(order)

        if self._limit_crossed_now(order):
            self._fill_order(order, fill_price=float(order["price"]), liquidity="maker")
            return _copy_order(order)

        self.open_orders[order_id] = order
        return _copy_order(order)

    async def cancel_order(self, order_id: str, symbol: str = None, params: Optional[dict] = None) -> dict:
        order = self.open_orders.pop(str(order_id), None)
        if order is None:
            raise KeyError(f"Fake order {order_id} not found")
        if symbol is not None and order["symbol"] != symbol:
            raise KeyError(f"Fake order {order_id} belongs to {order['symbol']}, not {symbol}")
        order["status"] = "canceled"
        order["info"] = dict(order.get("info") or {})
        order["info"]["canceled"] = True
        return _copy_order(order)

    async def fetch_time(self) -> int:
        return int(self.now_ms)

    async def set_position_mode(self, hedge_mode: bool, **kwargs) -> dict:
        self.position_mode = bool(hedge_mode)
        return {"hedgeMode": self.position_mode}

    async def set_leverage(self, leverage: int, symbol: str = None, **kwargs) -> dict:
        if symbol is not None:
            self.leverage_by_symbol[symbol] = int(leverage)
        return {"symbol": symbol, "leverage": int(leverage)}

    async def set_margin_mode(self, margin_mode: str, symbol: str = None, **kwargs) -> dict:
        if symbol is not None:
            self.margin_mode_by_symbol[symbol] = str(margin_mode)
        return {"symbol": symbol, "marginMode": str(margin_mode)}

    async def close(self) -> None:
        return None

    def advance_time(self, steps: int = 1) -> bool:
        advanced = False
        for _ in range(max(1, int(steps))):
            if not self.has_next_step():
                break
            self.current_index += 1
            self.now_ms = int(self.timeline[self.current_index]["timestamp"])
            self._process_resting_orders_for_current_step()
            advanced = True
        return advanced

    def get_fill_events(self, since_ms: Optional[int], until_ms: Optional[int]) -> List[dict]:
        events = []
        for fill in self.fills:
            ts = int(fill["timestamp"])
            if since_ms is not None and ts < int(since_ms):
                continue
            if until_ms is not None and ts > int(until_ms):
                continue
            events.append(self._fill_to_event(fill))
        return sorted(events, key=lambda item: (item["timestamp"], item["id"]))

    def export_state(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "scenario_path": self.scenario.get("_scenario_path"),
            "now_ms": int(self.now_ms),
            "datetime": ts_to_date(self.now_ms),
            "boot_index": int(self.boot_index),
            "current_index": int(self.current_index),
            "balance_total": float(self.balance_total),
            "balance_free": float(self.balance_free),
            "realized_pnl": float(self.realized_pnl),
            "realized_fees": float(self.realized_fees),
            "open_orders": [self._serialize_order(order) for order in self.open_orders.values()],
            "positions": self.export_positions(),
            "fills": [copy.deepcopy(fill) for fill in self.fills],
            "prices": dict(self.get_current_step()["prices"]),
        }

    def export_positions(self) -> List[dict]:
        result = []
        for state in self.positions.values():
            size = float(state["size"])
            if size == 0.0:
                continue
            result.append(
                {
                    "symbol": str(state["symbol"]),
                    "position_side": str(state["position_side"]),
                    "size": size,
                    "entry_price": float(state["entry_price"]),
                }
            )
        result.sort(key=lambda item: (item["symbol"], item["position_side"]))
        return result

    def _serialize_order(self, order: Dict[str, Any]) -> dict:
        dumped = _copy_order(order)
        dumped["reduceOnly"] = bool(dumped.get("reduceOnly"))
        return dumped

    def _process_resting_orders_for_current_step(self) -> None:
        current_candles = self.get_current_step()["candles"]
        fill_ids = []
        for order_id, order in self.open_orders.items():
            candle = current_candles[order["symbol"]]
            low_price = float(candle[3])
            high_price = float(candle[2])
            order_price = float(order["price"])
            if order["side"] == "buy" and low_price <= order_price:
                fill_ids.append(order_id)
            elif order["side"] == "sell" and high_price >= order_price:
                fill_ids.append(order_id)
        for order_id in fill_ids:
            order = self.open_orders.pop(order_id)
            self._fill_order(order, fill_price=float(order["price"]), liquidity="maker")

    def _limit_crossed_now(self, order: Dict[str, Any]) -> bool:
        last_price = float(self.get_current_step()["prices"][order["symbol"]])
        order_price = float(order["price"])
        if order["side"] == "buy":
            return last_price <= order_price
        return last_price >= order_price

    def _aggregate_candles(self, rows: List[List[float]], timeframe_ms: int) -> List[List[float]]:
        if timeframe_ms <= 60_000:
            return rows
        buckets: Dict[int, List[float]] = {}
        ordered: List[int] = []
        for row in rows:
            bucket_ts = int(row[0]) // timeframe_ms * timeframe_ms
            if bucket_ts not in buckets:
                buckets[bucket_ts] = [
                    float(bucket_ts),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
                ordered.append(bucket_ts)
                continue
            bucket = buckets[bucket_ts]
            bucket[2] = max(float(bucket[2]), float(row[2]))
            bucket[3] = min(float(bucket[3]), float(row[3]))
            bucket[4] = float(row[4])
            bucket[5] = float(bucket[5]) + float(row[5])
        return [buckets[key] for key in ordered]

    def _fill_order(self, order: Dict[str, Any], *, fill_price: float, liquidity: str) -> None:
        order["status"] = "closed"
        order["filled"] = float(order["amount"])
        order["remaining"] = 0.0
        order["price"] = float(fill_price)
        info = dict(order.get("info") or {})
        position_side = str(info.get("positionSide") or "LONG").lower()
        position = self.positions[(order["symbol"], position_side)]
        qty = abs(float(order["amount"]))
        pnl = 0.0
        if position_side == "long":
            if order["side"] == "buy":
                new_size = float(position["size"]) + qty
                if new_size == 0.0:
                    new_entry = 0.0
                else:
                    new_entry = (
                        float(position["entry_price"]) * float(position["size"]) + fill_price * qty
                    ) / new_size
                position["size"] = new_size
                position["entry_price"] = new_entry
            else:
                close_qty = min(float(position["size"]), qty)
                pnl = (fill_price - float(position["entry_price"])) * close_qty * self._c_mult(order["symbol"])
                position["size"] = max(0.0, float(position["size"]) - close_qty)
                if float(position["size"]) == 0.0:
                    position["entry_price"] = 0.0
        else:
            if order["side"] == "sell":
                new_size = float(position["size"]) + qty
                if new_size == 0.0:
                    new_entry = 0.0
                else:
                    new_entry = (
                        float(position["entry_price"]) * float(position["size"]) + fill_price * qty
                    ) / new_size
                position["size"] = new_size
                position["entry_price"] = new_entry
            else:
                close_qty = min(float(position["size"]), qty)
                pnl = (float(position["entry_price"]) - fill_price) * close_qty * self._c_mult(order["symbol"])
                position["size"] = max(0.0, float(position["size"]) - close_qty)
                if float(position["size"]) == 0.0:
                    position["entry_price"] = 0.0

        fee_rate = self._fee_rate(order["symbol"], liquidity)
        fee_cost = fill_price * qty * self._c_mult(order["symbol"]) * fee_rate
        self.realized_pnl += pnl
        self.realized_fees += fee_cost
        self.balance_total += pnl - fee_cost
        self.balance_free = self.balance_total

        trade_id = str(self._next_trade_id)
        self._next_trade_id += 1
        fill = {
            "id": trade_id,
            "order": str(order["id"]),
            "timestamp": self.now_ms,
            "datetime": ts_to_date(self.now_ms),
            "symbol": str(order["symbol"]),
            "side": str(order["side"]),
            "amount": qty,
            "price": float(fill_price),
            "pnl": float(pnl),
            "fee": {"currency": self.quote, "cost": float(fee_cost)},
            "clientOrderId": str(order.get("clientOrderId") or ""),
            "position_side": position_side,
            "reduceOnly": bool(order.get("reduceOnly")),
            "info": {
                "positionSide": position_side.upper(),
                "reduceOnly": bool(order.get("reduceOnly")),
                "liquidity": liquidity,
                "realizedPnl": float(pnl),
                "fee": float(fee_cost),
                "contractMultiplier": self._c_mult(order["symbol"]),
            },
        }
        self.fills.append(fill)

    def _fill_to_event(self, fill: dict) -> dict:
        return {
            "id": str(fill["id"]),
            "order_id": str(fill.get("order") or ""),
            "source_ids": [str(fill["id"])],
            "timestamp": int(fill["timestamp"]),
            "datetime": str(fill.get("datetime") or ts_to_date(int(fill["timestamp"]))),
            "symbol": str(fill["symbol"]),
            "side": str(fill["side"]).lower(),
            "qty": abs(float(fill["amount"])),
            "price": float(fill["price"]),
            "pnl": float(fill.get("pnl") or 0.0),
            "fees": fill.get("fee"),
            "pb_order_type": "unknown",
            "position_side": str(fill.get("position_side") or "long"),
            "client_order_id": str(fill.get("clientOrderId") or ""),
            "raw": [{"source": "fake_fill_ledger", "data": copy.deepcopy(fill)}],
            "c_mult": self._c_mult(str(fill["symbol"])),
        }

    def _c_mult(self, symbol: str) -> float:
        return float(self.markets[symbol].get("contractSize", 1.0))

    def _fee_rate(self, symbol: str, liquidity: str) -> float:
        key = "maker" if liquidity == "maker" else "taker"
        return float(self.markets[symbol].get(key, 0.0))


class FakeBot(CCXTBot):
    def create_ccxt_sessions(self):
        self.cca = FakeCCXTClient.from_config(self.config, self.user_info)
        self.ccp = None
        self.ws_enabled = False
        logger.info(
            "fake: loaded scenario %s from %s",
            self.cca.scenario_name,
            self.cca.scenario.get("_scenario_path"),
        )

    def _build_order_params(self, order: dict) -> dict:
        params = super()._build_order_params(order)
        params["reduceOnly"] = bool(order.get("reduce_only", False))
        return params
