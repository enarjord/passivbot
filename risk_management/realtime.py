"""Realtime data gathering for the risk management dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from types import TracebackType
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

from ._notifications import NotificationCoordinator
from ._parsing import (
    extract_balance as _extract_balance,
    parse_order as _parse_order,
    parse_position as _parse_position,
)
from .account_clients import AccountClientProtocol, CCXTAccountClient
from .configuration import CustomEndpointSettings, RealtimeConfig

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
        self._notifications = NotificationCoordinator(config)
        self._portfolio_stop_loss: Optional[Dict[str, Any]] = None
        self._last_portfolio_balance: Optional[float] = None

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
        self._notifications.send_daily_snapshot(snapshot, portfolio_balance)
        self._notifications.dispatch_alerts(snapshot)
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
