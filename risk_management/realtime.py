"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from custom_endpoint_overrides import (
    CustomEndpointConfigError,
    configure_custom_endpoint_loader,
    load_custom_endpoint_config,
)

try:  # pragma: no cover - optional dependency when running tests
    from ccxt.base.errors import AuthenticationError
except (ModuleNotFoundError, ImportError):  # pragma: no cover - ccxt is optional for tests

    class AuthenticationError(Exception):
        """Fallback authentication error used when ccxt is unavailable."""

        pass

from .account_clients import AccountClientProtocol, CCXTAccountClient
from .configuration import CustomEndpointSettings, RealtimeConfig
from .dashboard import evaluate_alerts, parse_snapshot
from .email_notifications import EmailAlertSender

logger = logging.getLogger(__name__)


def _build_search_paths(config_root: Path | None) -> tuple[str, ...]:
    """Return candidate custom endpoint paths prioritising the config directory."""

    candidates: list[str] = []
    if config_root is not None:
        candidate = (config_root / "custom_endpoints.json").resolve()
        candidates.append(str(candidate))
    default_path = os.path.join("configs", "custom_endpoints.json")
    if default_path not in candidates:
        candidates.append(default_path)
    # Remove duplicates while preserving order
    ordered = list(dict.fromkeys(candidates))
    return tuple(ordered)


def _configure_custom_endpoints(
    settings: CustomEndpointSettings | None, config_root: Path | None
) -> None:
    """Initialise custom endpoint overrides before creating ccxt clients."""

    search_paths = _build_search_paths(config_root)

    if settings is None or (not settings.path and settings.autodiscover):
        preloaded = None
        try:
            preloaded = load_custom_endpoint_config(search_paths=search_paths)
        except CustomEndpointConfigError as exc:
            logger.warning("Failed to load custom endpoint config via discovery: %s", exc)
        configure_custom_endpoint_loader(None, autodiscover=True, preloaded=preloaded)
        source = preloaded.source_path if preloaded else None
        if source:
            logger.info("Using custom endpoints from %s", source)
        else:
            logger.info("No custom endpoint overrides found; using exchange defaults")
        return

    path = settings.path
    autodiscover = settings.autodiscover
    preloaded = None

    if path:
        try:
            preloaded = load_custom_endpoint_config(path)
        except CustomEndpointConfigError as exc:
            raise ValueError(f"Failed to load custom endpoint config '{path}': {exc}") from exc

    configure_custom_endpoint_loader(path, autodiscover=autodiscover, preloaded=preloaded)
    if path:
        logger.info("Using custom endpoints from %s", path)


class RealtimeDataFetcher:
    """Fetch realtime snapshots across multiple accounts."""

    def __init__(
        self,
        config: RealtimeConfig,
        account_clients: Sequence[AccountClientProtocol] | None = None,
    ) -> None:
        self.config = config
        _configure_custom_endpoints(config.custom_endpoints, config.config_root)
        if account_clients is None:
            clients: List[AccountClientProtocol] = []
            for account in config.accounts:
                try:
                    clients.append(CCXTAccountClient(account))
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Unable to create realtime clients. Install ccxt or provide custom account clients."
                    ) from exc
                except Exception as exc:
                    logger.error(
                        "Failed to initialise account client for %s: %s", account.name, exc, exc_info=True
                    )
                    raise
            self._account_clients = clients
        else:
            self._account_clients = list(account_clients)
        self._last_auth_errors: Dict[str, str] = {}
        if config.debug_api_payloads:
            logger.info(
                "Exchange API payload debug logging enabled for realtime fetcher"
            )
        for account in config.accounts:
            if account.debug_api_payloads and not config.debug_api_payloads:
                logger.info(
                    "Debug API payload logging enabled for account %s", account.name
                )
        self._email_sender = EmailAlertSender(config.email) if config.email else None
        self._email_recipients = self._extract_email_recipients()
        self._active_alerts: set[str] = set()

    def _extract_email_recipients(self) -> List[str]:
        recipients: List[str] = []
        for channel in self.config.notification_channels:
            if not isinstance(channel, str):
                continue
            if channel.lower().startswith("email:"):
                address = channel.split(":", 1)[1].strip()
                if address:
                    recipients.append(address)
        return recipients

    async def fetch_snapshot(self) -> Dict[str, Any]:
        tasks = [client.fetch() for client in self._account_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        accounts_payload: List[Dict[str, Any]] = []
        account_messages: Dict[str, str] = {}
        for account_config, result in zip(self.config.accounts, results):
            if isinstance(result, Exception):
                if isinstance(result, AuthenticationError):
                    message = (
                        f"{account_config.name}: authentication failed - {result}"
                    )

                    error_message = str(result)
                    previous_error = self._last_auth_errors.get(account_config.name)
                    if previous_error != error_message:
                        logger.warning(
                            "Authentication failed for %s: %s",
                            account_config.name,
                            result,
                        )
                        self._last_auth_errors[account_config.name] = error_message
                    else:
                        logger.debug(
                            "Authentication failure for %s unchanged: %s",
                            account_config.name,
                            result,
                        )

                else:
                    message = f"{account_config.name}: {result}"
                    logger.exception(
                        "Failed to fetch snapshot for %s", account_config.name, exc_info=result
                    )
                account_messages[account_config.name] = message
                accounts_payload.append({"name": account_config.name, "balance": 0.0, "positions": []})
            else:
                accounts_payload.append(result)
                if account_config.name in self._last_auth_errors:
                    logger.info(
                        "Authentication for %s restored", account_config.name
                    )
                    self._last_auth_errors.pop(account_config.name, None)
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "accounts": accounts_payload,
            "alert_thresholds": self.config.alert_thresholds,
            "notification_channels": self.config.notification_channels,
        }
        if account_messages:
            snapshot["account_messages"] = account_messages
        self._dispatch_email_alerts(snapshot)
        return snapshot

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._account_clients))

    async def execute_kill_switch(
        self, account_name: str | None = None, symbol: str | None = None
    ) -> Dict[str, Any]:
        targets: List[AccountClientProtocol] = []
        for client in self._account_clients:
            if account_name is None or client.config.name == account_name:
                targets.append(client)
        if account_name is not None and not targets:
            raise ValueError(f"Account '{account_name}' is not configured for realtime monitoring.")
        results: Dict[str, Any] = {}
        for client in targets:
            try:
                results[client.config.name] = await client.kill_switch(symbol)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Kill switch failed for %s", client.config.name, exc_info=exc)
                results[client.config.name] = {"error": str(exc)}
        return results

    def _dispatch_email_alerts(self, snapshot: Mapping[str, Any]) -> None:
        if not self._email_sender or not self._email_recipients:
            return
        try:
            _, accounts, thresholds, _ = parse_snapshot(dict(snapshot))
            alerts = evaluate_alerts(accounts, thresholds)
        except Exception as exc:  # pragma: no cover - snapshot parsing errors are logged for diagnostics
            logger.debug("Skipping email alert dispatch due to parsing error: %s", exc, exc_info=True)
            return
        alerts_set = set(alerts)
        new_alerts = [alert for alert in alerts if alert not in self._active_alerts]
        self._active_alerts = alerts_set
        if not new_alerts:
            return
        generated_at = snapshot.get("generated_at")
        timestamp = (
            generated_at
            if isinstance(generated_at, str)
            else datetime.now(timezone.utc).isoformat()
        )
        lines = [f"Exposure thresholds were exceeded at {timestamp}.", "", "Alerts:"]
        lines.extend(f"- {alert}" for alert in new_alerts)
        body = "\n".join(lines)
        subject = "Risk alert: exposure threshold breached"
        self._email_sender.send(subject, body, self._email_recipients)


def _extract_balance(balance: Mapping[str, Any], settle_currency: str) -> float:
    """Extract a numeric balance from ccxt balance payloads."""

    if not isinstance(balance, Mapping):
        return 0.0

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    aggregate_keys = (
        "totalMarginBalance",
        "totalEquity",
        "totalWalletBalance",
        "marginBalance",
        "totalBalance",
    )

    def _find_nested_aggregate(value: Any) -> float | None:
        if isinstance(value, Mapping):
            for key in aggregate_keys:
                candidate = _to_float(value.get(key))
                if candidate is not None:
                    return candidate
            for child in value.values():
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        elif isinstance(value, (list, tuple)):
            for child in value:
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        return None

    # Some exchanges expose aggregate balances directly on the top-level payload.
    for key in (*aggregate_keys, "equity"):
        candidate = _to_float(balance.get(key))
        if candidate is not None:
            return candidate

    info = balance.get("info")
    if isinstance(info, Mapping):
        for key in (*aggregate_keys, "equity"):
            candidate = _to_float(info.get(key))
            if candidate is not None:
                return candidate
        nested = _find_nested_aggregate(info)
        if nested is not None:
            return nested

    total = balance.get("total")
    if isinstance(total, Mapping) and total:
        if settle_currency in total:
            candidate = _to_float(total.get(settle_currency))
            if candidate is not None:
                return candidate
        summed = 0.0
        found_value = False
        for value in total.values():
            candidate = _to_float(value)
            if candidate is None:
                continue
            summed += candidate
            found_value = True
        if found_value:
            return summed

    for currency_key in (settle_currency, "USDT"):
        entry = balance.get(currency_key)
        if isinstance(entry, Mapping):
            for key in ("total", "free", "used"):
                candidate = _to_float(entry.get(key))
                if candidate is not None:
                    return candidate
        else:
            candidate = _to_float(entry)
            if candidate is not None:
                return candidate

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
    notional_value = float(notional or 0.0)
    if size < 0 and notional_value > 0:
        signed_notional = -abs(notional_value)
    elif size > 0 and notional_value < 0:
        signed_notional = abs(notional_value)
    else:
        signed_notional = notional_value
    abs_notional = abs(signed_notional)
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
        wallet_exposure = abs_notional / balance if balance else None
    return {
        "symbol": str(position.get("symbol") or position.get("id") or "unknown"),
        "side": side,
        "notional": abs_notional,
        "entry_price": float(entry_price or 0.0),
        "mark_price": float(mark_price or 0.0),
        "liquidation_price": float(liquidation_price) if liquidation_price is not None else None,
        "wallet_exposure_pct": float(wallet_exposure) if wallet_exposure is not None else None,
        "unrealized_pnl": float(unrealized),
        "max_drawdown_pct": None,
        "take_profit_price": float(take_profit) if take_profit is not None else None,
        "stop_loss_price": float(stop_loss) if stop_loss is not None else None,
        "size": float(size),
        "signed_notional": signed_notional,
    }


def _parse_order(order: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(order, Mapping):
        return None
    symbol = order.get("symbol") or order.get("id")
    if not symbol:
        return None
    price = _first_float(
        order.get("price"),
        order.get("triggerPrice"),
        order.get("stopPrice"),
        order.get("info", {}).get("price") if isinstance(order.get("info"), Mapping) else None,
    )
    amount = _first_float(
        order.get("amount"),
        order.get("contracts"),
        order.get("size"),
        order.get("info", {}).get("origQty") if isinstance(order.get("info"), Mapping) else None,
    )
    if amount is None:
        return None
    remaining = _first_float(
        order.get("remaining"),
        order.get("remainingAmount"),
        order.get("info", {}).get("leavesQty") if isinstance(order.get("info"), Mapping) else None,
    )
    reduce_only_raw = order.get("reduceOnly")
    if isinstance(order.get("info"), Mapping):
        reduce_only_raw = reduce_only_raw or order["info"].get("reduceOnly")
    reduce_only = bool(reduce_only_raw)
    stop_price = _first_float(
        order.get("stopPrice"),
        order.get("triggerPrice"),
        order.get("info", {}).get("stopPrice") if isinstance(order.get("info"), Mapping) else None,
    )
    timestamp_raw = order.get("timestamp")
    created_at = None
    if isinstance(timestamp_raw, (int, float)):
        created_at = datetime.fromtimestamp(float(timestamp_raw) / 1000, timezone.utc).isoformat()
    else:
        datetime_str = order.get("datetime")
        if isinstance(datetime_str, str) and datetime_str:
            created_at = datetime_str
    notional = price * amount if price is not None else None
    return {
        "order_id": str(order.get("id") or order.get("clientOrderId") or ""),
        "symbol": str(symbol),
        "side": str(order.get("side") or "").lower(),
        "type": str(order.get("type") or "").lower(),
        "price": price,
        "amount": amount,
        "remaining": remaining,
        "status": str(order.get("status") or ""),
        "reduce_only": reduce_only,
        "stop_price": stop_price,
        "notional": notional,
        "created_at": created_at,
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
