"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone, date, time
from types import TracebackType
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

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
from .telegram_notifications import TelegramNotifier

logger = logging.getLogger(__name__)

def _exception_info(
    exc: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Return a ``logging`` compatible ``exc_info`` tuple for ``exc``."""

    return (type(exc), exc, exc.__traceback__)


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
    settings: Optional[CustomEndpointSettings], config_root: Optional[Path]
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
        account_clients: Optional[Sequence[AccountClientProtocol]] = None,
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
        self._telegram_targets = self._extract_telegram_targets()
        self._telegram_notifier = TelegramNotifier() if self._telegram_targets else None
        self._active_alerts: set[str] = set()
        self._daily_snapshot_tz = ZoneInfo("America/New_York")
        self._daily_snapshot_sent_date: Optional[date] = None
        self._portfolio_stop_loss: Optional[Dict[str, Any]] = None
        self._last_portfolio_balance: Optional[float] = None

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

    def _extract_telegram_targets(self) -> List[tuple[str, str]]:
        targets: List[tuple[str, str]] = []
        for channel in self.config.notification_channels:
            if not isinstance(channel, str):
                continue
            if not channel.lower().startswith("telegram:"):
                continue
            payload = channel.split(":", 1)[1]
            token = ""
            chat_id = ""
            if "@" in payload:
                token, _, chat_id = payload.partition("@")
            elif "/" in payload:
                token, _, chat_id = payload.partition("/")
            else:
                parts = payload.split(":", 1)
                if len(parts) == 2:
                    token, chat_id = parts
            token = token.strip()
            chat_id = chat_id.strip()
            if token and chat_id:
                targets.append((token, chat_id))
        return targets

    async def fetch_snapshot(self) -> Dict[str, Any]:
        tasks = [client.fetch() for client in self._account_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        accounts_payload: List[Dict[str, Any]] = []
        account_messages: Dict[str, str] = dict(self.config.account_messages)
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
                    logger.error(
                        "Failed to fetch snapshot for %s",
                        account_config.name,
                        exc_info=_exception_info(result),
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
        portfolio_balance = sum(
            float(account.get("balance", 0.0)) for account in accounts_payload
        )
        self._last_portfolio_balance = portfolio_balance
        stop_loss_state = self._update_portfolio_stop_loss_state(portfolio_balance)
        if stop_loss_state:
            snapshot["portfolio_stop_loss"] = stop_loss_state
        self._maybe_send_daily_balance_snapshot(snapshot, portfolio_balance)
        self._dispatch_notifications(snapshot)
        return snapshot

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._account_clients))

    async def execute_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        scope = account_name or "all accounts"
        symbol_desc = f" for {symbol}" if symbol else ""
        logger.info("Kill switch requested for %s%s", scope, symbol_desc)
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
                logger.exception("Kill switch failed for %s", client.config.name, exc_info=True)
                results[client.config.name] = {"error": str(exc)}
        logger.info("Kill switch completed for %s", scope)
        return results

    def _dispatch_notifications(self, snapshot: Mapping[str, Any]) -> None:
        if not (self._email_sender or self._telegram_notifier):
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
        if self._email_sender and self._email_recipients:
            self._email_sender.send(subject, body, self._email_recipients)
        if self._telegram_notifier and self._telegram_targets:
            message = f"Exposure alert at {timestamp}\n" + "\n".join(new_alerts)
            for token, chat_id in self._telegram_targets:
                self._telegram_notifier.send(token, chat_id, message)

    def _maybe_send_daily_balance_snapshot(
        self, snapshot: Mapping[str, Any], portfolio_balance: float
    ) -> None:
        if not self._email_sender or not self._email_recipients:
            return
        now_ny = datetime.now(self._daily_snapshot_tz)
        current_date = now_ny.date()
        if self._daily_snapshot_sent_date and current_date > self._daily_snapshot_sent_date:
            self._daily_snapshot_sent_date = None
        if now_ny.time() < time(16, 0):
            return
        if self._daily_snapshot_sent_date == current_date:
            return
        accounts = snapshot.get("accounts", [])
        lines = [
            f"Daily portfolio snapshot ({now_ny.strftime('%Y-%m-%d')} 16:00 ET)",
            f"Total balance: ${portfolio_balance:,.2f}",
            "",
            "Accounts:",
        ]
        for account in accounts or []:
            if not isinstance(account, Mapping):
                continue
            name = str(account.get("name", "unknown"))
            balance = float(account.get("balance", 0.0))
            realised = float(account.get("daily_realized_pnl", 0.0))
            lines.append(
                f"- {name}: balance ${balance:,.2f}, daily realised PnL ${realised:,.2f}"
            )
        body = "\n".join(lines)
        subject = "Daily portfolio balance snapshot"
        self._email_sender.send(subject, body, self._email_recipients)
        self._daily_snapshot_sent_date = current_date

    def _update_portfolio_stop_loss_state(
        self, portfolio_balance: float
    ) -> Optional[Dict[str, Any]]:
        if self._portfolio_stop_loss is None:
            return None
        state = dict(self._portfolio_stop_loss)
        state.setdefault("active", True)
        state.setdefault("triggered", False)
        state.setdefault("threshold_pct", 0.0)
        if state.get("baseline_balance") is None and portfolio_balance:
            state["baseline_balance"] = portfolio_balance
        baseline = state.get("baseline_balance")
        threshold_pct = state.get("threshold_pct")
        drawdown: Optional[float] = None
        if baseline and baseline > 0:
            drawdown = max(0.0, (baseline - portfolio_balance) / baseline)
        state["current_balance"] = portfolio_balance
        state["current_drawdown_pct"] = drawdown
        if (
            isinstance(threshold_pct, (int, float))
            and threshold_pct > 0
            and drawdown is not None
            and drawdown >= float(threshold_pct) / 100.0
            and not state.get("triggered")
        ):
            state["triggered"] = True
            state["triggered_at"] = datetime.now(timezone.utc).isoformat()
        self._portfolio_stop_loss = state
        return dict(state)

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        if self._portfolio_stop_loss is None:
            return None
        return dict(self._portfolio_stop_loss)

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        if threshold_pct <= 0:
            raise ValueError("Portfolio stop loss threshold must be greater than zero.")
        state = {
            "threshold_pct": float(threshold_pct),
            "baseline_balance": self._last_portfolio_balance,
            "triggered": False,
            "triggered_at": None,
            "active": True,
        }
        self._portfolio_stop_loss = state
        return dict(state)

    async def clear_portfolio_stop_loss(self) -> None:
        self._portfolio_stop_loss = None

    def _resolve_account_client(self, account_name: str) -> AccountClientProtocol:
        for client in self._account_clients:
            if client.config.name == account_name:
                return client
        raise ValueError(f"Account '{account_name}' is not configured for realtime monitoring.")

    async def place_order(
        self,
        account_name: str,
        *,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        normalized_amount = float(amount)
        normalized_price = float(price) if price is not None else None
        return await client.create_order(
            symbol, order_type, side, normalized_amount, normalized_price, params=params
        )

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        normalized_id = str(order_id)
        return await client.cancel_order(normalized_id, symbol, params=params)

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        client = self._resolve_account_client(account_name)
        return await client.close_position(symbol)

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        client = self._resolve_account_client(account_name)
        return await client.list_order_types()


def _extract_balance(balance: Mapping[str, Any], settle_currency: str) -> float:
    """Extract a numeric balance from ccxt balance payloads."""

    if not isinstance(balance, Mapping):
        return 0.0

    def _to_float(value: Any) -> Optional[float]:
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

    def _find_nested_aggregate(value: Any) -> Optional[float]:
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


def _parse_position(position: Mapping[str, Any], balance: float) -> Optional[Dict[str, Any]]:
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
    realized = _first_float(
        position.get("dailyRealizedPnl"),
        position.get("realizedPnl"),
        position.get("realisedPnl"),
        position.get("info", {}).get("dailyRealizedPnl")
        if isinstance(position.get("info"), Mapping)
        else None,
        position.get("info", {}).get("realizedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("realisedPnl") if isinstance(position.get("info"), Mapping) else None,
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
        "daily_realized_pnl": float(realized),
        "max_drawdown_pct": None,
        "take_profit_price": float(take_profit) if take_profit is not None else None,
        "stop_loss_price": float(stop_loss) if stop_loss is not None else None,
        "size": float(size),
        "signed_notional": signed_notional,
    }


def _parse_order(order: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
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


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
