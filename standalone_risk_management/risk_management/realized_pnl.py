"""Helpers for computing realized PnL history across exchanges."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency in some environments
    from ccxt.base.errors import BaseError  # type: ignore
except Exception:  # pragma: no cover - fallback when ccxt is missing
    class BaseError(Exception):
        """Fallback error when ccxt is unavailable."""

        pass


from ._utils import (
    coerce_float as _coerce_float,
    coerce_int as _coerce_int,
    first_float as _first_float,
)

logger = logging.getLogger(__name__)


def _dedupe_symbols(symbols: Optional[Sequence[Optional[str]]]) -> Sequence[Optional[str]]:
    if not symbols:
        return [None]
    unique: list[Optional[str]] = []
    seen: set[Optional[str]] = set()
    for symbol in symbols:
        if symbol is None:
            if None not in seen:
                seen.add(None)
                unique.append(None)
            continue
        if isinstance(symbol, str):
            stripped = symbol.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            unique.append(stripped)
    return unique or [None]


async def fetch_realized_pnl_history(
    exchange_id: str,
    client: Any,
    *,
    since: int,
    until: int,
    params: Optional[Mapping[str, Any]] = None,
    limit: Optional[int] = None,
    symbols: Optional[Sequence[Optional[str]]] = None,
    account_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
    debug_api_payloads: bool = False,
) -> Optional[float]:
    """Return realised PnL from exchange history endpoints when available."""

    logger_instance = log or logger
    identifier = account_name or exchange_id

    if since >= until:
        return 0.0

    params_base = dict(params or {})

    try:
        if exchange_id in {"binanceusdm", "binancecoinm", "binancecm"}:
            fetch_income = getattr(client, "fetch_income", None)
            if fetch_income is None:
                return None
            request = dict(params_base)
            request.setdefault("incomeType", "REALIZED_PNL")
            request.setdefault("startTime", since)
            request.setdefault("endTime", until)
            limit_value = _coerce_int(limit)
            if limit_value and limit_value > 0:
                request.setdefault("limit", limit_value)
            incomes = await fetch_income(params=request)
            total = 0.0
            for entry in incomes or []:
                amount = _coerce_float(entry.get("amount"))
                if amount is None and isinstance(entry.get("info"), Mapping):
                    info = entry["info"]
                    amount = _first_float(
                        info.get("amount"),
                        info.get("income"),
                        info.get("realizedPnl"),
                        info.get("realisedPnl"),
                    )
                if amount is None:
                    continue
                total += float(amount)
            return total

        if exchange_id == "bybit":
            fetch_closed_pnl = getattr(client, "private_get_v5_position_closed_pnl", None)
            if fetch_closed_pnl is None:
                return None
            limit_value = _coerce_int(limit)
            if limit_value is None or limit_value <= 0:
                limit_value = 200
            total = 0.0
            cursor: Optional[str] = None
            while True:
                request = dict(params_base)
                request.setdefault("startTime", since)
                request.setdefault("endTime", until)
                request.setdefault("limit", limit_value)
                if cursor:
                    request["cursor"] = cursor
                response = await fetch_closed_pnl(request)
                result = response.get("result") if isinstance(response, Mapping) else None
                rows = result.get("list") if isinstance(result, Mapping) else None
                entries = rows or []
                for entry in entries:
                    pnl = _first_float(entry.get("pnl"), entry.get("closedPnl"))
                    if pnl is None:
                        continue
                    total += float(pnl)
                cursor = (
                    result.get("nextPageCursor") if isinstance(result, Mapping) else None
                )
                if not cursor or not entries:
                    break
            return total

        if exchange_id == "okx":
            fetch_trades = getattr(client, "fetch_my_trades", None)
            if fetch_trades is None:
                return None
            limit_value = _coerce_int(limit)
            if limit_value is None or limit_value <= 0:
                limit_value = 200
            params_base.setdefault("until", until)
            total = 0.0
            for symbol in _dedupe_symbols(symbols):
                request = dict(params_base)
                try:
                    trades = await fetch_trades(
                        symbol,
                        since=since,
                        limit=limit_value,
                        params=request,
                    )
                except BaseError as exc:
                    logger_instance.debug(
                        "[%s] fetch_my_trades failed for %s: %s",
                        identifier,
                        symbol or "*",
                        exc,
                        exc_info=debug_api_payloads,
                    )
                    continue
                if not trades:
                    continue
                for trade in trades:
                    pnl = _first_float(
                        trade.get("pnl"),
                        trade.get("realizedPnl"),
                        trade.get("realisedPnl"),
                    )
                    info = trade.get("info") if isinstance(trade.get("info"), Mapping) else None
                    if pnl is None and info:
                        pnl = _first_float(
                            info.get("fillPnl"),
                            info.get("pnl"),
                            info.get("realizedPnl"),
                            info.get("realisedPnl"),
                        )
                    if pnl is None:
                        continue
                    total += float(pnl)
            return total

    except BaseError as exc:
        logger_instance.debug(
            "[%s] Failed to fetch realised PnL via history: %s",
            identifier,
            exc,
            exc_info=debug_api_payloads,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger_instance.debug(
            "[%s] Unexpected error while fetching realised PnL: %s",
            identifier,
            exc,
            exc_info=debug_api_payloads,
        )

    return None
