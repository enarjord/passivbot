"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # pragma: no cover - optional dependency for tests
    import ccxt.async_support as ccxt_async
    from ccxt.base.errors import BaseError
except ModuleNotFoundError:  # pragma: no cover - allow tests without ccxt
    ccxt_async = None  # type: ignore[assignment]

    class BaseError(Exception):
        """Fallback error when ccxt is unavailable."""

        pass

from .configuration import AccountConfig, RealtimeConfig

logger = logging.getLogger(__name__)


@dataclass()
class AccountClientProtocol:
    """Protocol describing realtime account clients."""

    async def fetch(self) -> Dict[str, Any]:  # pragma: no cover - protocol definition
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - protocol definition
        raise NotImplementedError


class CCXTAccountClient(AccountClientProtocol):
    """Realtime account client backed by ccxt asynchronous exchanges."""

    def __init__(self, config: AccountConfig):
        self.config = config
        exchange_id = config.exchange
        if ccxt_async is None:  # pragma: no cover - configuration error
            raise RuntimeError("ccxt is required to instantiate CCXTAccountClient")
        try:
            exchange_class = getattr(ccxt_async, exchange_id)
        except AttributeError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.") from exc
        params = dict(config.credentials)
        params.setdefault("enableRateLimit", True)
        self.client = exchange_class(params)
        self._balance_params = dict(config.params.get("balance", {}))
        self._positions_params = dict(config.params.get("positions", {}))

    async def fetch(self) -> Dict[str, Any]:
        await self.client.load_markets()
        balance_raw = await self.client.fetch_balance(params=self._balance_params)
        balance_value = _extract_balance(balance_raw, self.config.settle_currency)
        positions_raw: Iterable[Mapping[str, Any]] = []
        positions: List[Dict[str, Any]] = []
        if hasattr(self.client, "fetch_positions"):
            try:
                positions_raw = await self.client.fetch_positions(params=self._positions_params)
            except BaseError as exc:
                logger.warning(
                    "Failed to fetch positions for %s: %s", self.config.name, exc, exc_info=True
                )
        for position_raw in positions_raw or []:
            parsed = _parse_position(position_raw, balance_value)
            if parsed is not None:
                positions.append(parsed)
        return {
            "name": self.config.name,
            "balance": balance_value,
            "positions": positions,
        }

    async def close(self) -> None:
        await self.client.close()


class RealtimeDataFetcher:
    """Fetch realtime snapshots across multiple accounts."""

    def __init__(
        self,
        config: RealtimeConfig,
        account_clients: Sequence[AccountClientProtocol] | None = None,
    ) -> None:
        self.config = config
        if account_clients is None:
            if ccxt_async is None:
                raise RuntimeError(
                    "ccxt is required to create realtime exchange clients."
                )
            self._account_clients = [CCXTAccountClient(account) for account in config.accounts]
        else:
            self._account_clients = list(account_clients)

    async def fetch_snapshot(self) -> Dict[str, Any]:
        tasks = [client.fetch() for client in self._account_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        accounts_payload: List[Dict[str, Any]] = []
        account_messages: Dict[str, str] = {}
        for account_config, result in zip(self.config.accounts, results):
            if isinstance(result, Exception):
                message = f"{account_config.name}: {result}"
                logger.exception("Failed to fetch snapshot for %s", account_config.name, exc_info=result)
                account_messages[account_config.name] = message
                accounts_payload.append({"name": account_config.name, "balance": 0.0, "positions": []})
            else:
                accounts_payload.append(result)
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "accounts": accounts_payload,
            "alert_thresholds": self.config.alert_thresholds,
            "notification_channels": self.config.notification_channels,
        }
        if account_messages:
            snapshot["account_messages"] = account_messages
        return snapshot

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._account_clients))


def _extract_balance(balance: Mapping[str, Any], settle_currency: str) -> float:
    """Extract a numeric balance from ccxt balance payloads."""

    total = balance.get("total") if isinstance(balance, Mapping) else None
    if isinstance(total, Mapping) and total:
        if settle_currency in total and total[settle_currency] is not None:
            try:
                return float(total[settle_currency])
            except (TypeError, ValueError):
                logger.debug("Non-numeric balance for %s: %s", settle_currency, total[settle_currency])
        try:
            return float(sum(float(v or 0.0) for v in total.values()))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Unable to aggregate total balances: %s", total)
    info = balance.get("info") if isinstance(balance, Mapping) else None
    if isinstance(info, Mapping):
        for key in (
            "totalWalletBalance",
            "totalMarginBalance",
            "equity",
            "totalEquity",
            "marginBalance",
            "totalBalance",
        ):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    logger.debug("Non-numeric balance info %s=%s", key, value)
    balances = [
        balance.get(settle_currency) if isinstance(balance, Mapping) else None,
        balance.get("USDT") if isinstance(balance, Mapping) else None,
    ]
    for entry in balances:
        if isinstance(entry, Mapping):
            value = entry.get("total") or entry.get("free") or entry.get("used")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        elif entry is not None:
            try:
                return float(entry)
            except (TypeError, ValueError):
                continue
    return 0.0


def _parse_position(position: Mapping[str, Any], balance: float) -> Dict[str, Any] | None:
    size = _first_float(
        position.get("contracts"),
        position.get("size"),
        position.get("amount"),
        position.get("info", {}).get("positionAmt") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("size") if isinstance(position.get("info"), Mapping) else None,
    )
    if size is None or abs(size) < 1e-12:
        return None
    side = "long" if size > 0 else "short"
    entry_price = _first_float(
        position.get("entryPrice"),
        position.get("entry_price"),
        position.get("info", {}).get("entryPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("avgEntryPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    mark_price = _first_float(
        position.get("markPrice"),
        position.get("mark_price"),
        position.get("info", {}).get("markPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("last") if isinstance(position.get("info"), Mapping) else None,
    )
    liquidation_price = _first_float(
        position.get("liquidationPrice"),
        position.get("info", {}).get("liquidationPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    unrealized = _first_float(
        position.get("unrealizedPnl"),
        position.get("info", {}).get("unRealizedProfit") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("unrealisedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("upl") if isinstance(position.get("info"), Mapping) else None,
    ) or 0.0
    contract_size = _first_float(
        position.get("contractSize"),
        position.get("info", {}).get("contractSize") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("ctVal") if isinstance(position.get("info"), Mapping) else None,
    ) or 1.0
    notional = _first_float(
        position.get("notional"),
        position.get("notionalValue"),
        position.get("info", {}).get("notionalValue") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("notionalUsd") if isinstance(position.get("info"), Mapping) else None,
    )
    if notional is None:
        reference_price = mark_price or entry_price or 0.0
        notional = abs(size) * contract_size * reference_price
    take_profit = _first_float(
        position.get("takeProfitPrice"),
        position.get("tpPrice"),
        position.get("info", {}).get("takeProfitPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("tpTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    stop_loss = _first_float(
        position.get("stopLossPrice"),
        position.get("slPrice"),
        position.get("info", {}).get("stopLossPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("slTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    wallet_exposure = None
    if balance:
        wallet_exposure = abs(notional) / balance if balance else None
    return {
        "symbol": str(position.get("symbol") or position.get("id") or "unknown"),
        "side": side,
        "notional": float(notional or 0.0),
        "entry_price": float(entry_price or 0.0),
        "mark_price": float(mark_price or 0.0),
        "liquidation_price": float(liquidation_price) if liquidation_price is not None else None,
        "wallet_exposure_pct": float(wallet_exposure) if wallet_exposure is not None else None,
        "unrealized_pnl": float(unrealized),
        "max_drawdown_pct": None,
        "take_profit_price": float(take_profit) if take_profit is not None else None,
        "stop_loss_price": float(stop_loss) if stop_loss is not None else None,
    }


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
