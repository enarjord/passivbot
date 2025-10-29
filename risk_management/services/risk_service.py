"""Service abstractions orchestrating realtime risk management operations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

from ..configuration import RealtimeConfig
from ..realtime import RealtimeDataFetcher


class RiskServiceProtocol(Protocol):
    """Protocol describing the operations exposed by :class:`RiskService`."""

    async def fetch_snapshot(self) -> Dict[str, Any]:
        """Return a realtime snapshot for all configured accounts."""

    async def close(self) -> None:
        """Release any resources held by the underlying fetcher."""

    async def trigger_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        """Cancel orders and close positions for the requested scope."""

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
        """Submit an order on behalf of the specified account."""

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Cancel a previously submitted order."""

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        """Close an open position on the provided trading pair."""

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        """Return the order types supported by the underlying account client."""

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        """Return the currently configured portfolio level stop loss if active."""

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        """Activate a portfolio level stop loss at the provided threshold."""

    async def clear_portfolio_stop_loss(self) -> None:
        """Disable any active portfolio level stop loss."""

    def get_account_stop_loss(self, account_name: str) -> Optional[Dict[str, Any]]:
        """Return the account specific stop loss configuration if one exists."""

    async def set_account_stop_loss(self, account_name: str, threshold_pct: float) -> Dict[str, Any]:
        """Configure an account level stop loss."""

    async def clear_account_stop_loss(self, account_name: str) -> None:
        """Remove an account level stop loss if present."""

    async def cancel_all_orders(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        """Cancel all open orders for the provided scope."""

    async def close_all_positions(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        """Close all open positions for the provided scope."""


class RiskService(RiskServiceProtocol):
    """Concrete implementation of :class:`RiskServiceProtocol` backed by ``RealtimeDataFetcher``."""

    def __init__(self, fetcher: RealtimeDataFetcher) -> None:
        self._fetcher = fetcher

    @classmethod
    def from_config(cls, config: RealtimeConfig) -> "RiskService":
        """Convenience constructor that instantiates the underlying fetcher from ``config``."""

        return cls(RealtimeDataFetcher(config))

    async def fetch_snapshot(self) -> Dict[str, Any]:
        return await self._fetcher.fetch_snapshot()

    async def close(self) -> None:
        await self._fetcher.close()

    async def trigger_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        return await self._fetcher.execute_kill_switch(account_name, symbol)

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
        return await self._fetcher.place_order(
            account_name,
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        return await self._fetcher.cancel_order(
            account_name,
            order_id,
            symbol=symbol,
            params=params,
        )

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        return await self._fetcher.close_position(account_name, symbol)

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        return await self._fetcher.list_order_types(account_name)

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        state = self._fetcher.get_portfolio_stop_loss()
        return dict(state) if state is not None else None

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        return await self._fetcher.set_portfolio_stop_loss(threshold_pct)

    async def clear_portfolio_stop_loss(self) -> None:
        await self._fetcher.clear_portfolio_stop_loss()

    def get_account_stop_loss(self, account_name: str) -> Optional[Dict[str, Any]]:
        state = self._fetcher.get_account_stop_loss(account_name)
        return dict(state) if state is not None else None

    async def set_account_stop_loss(self, account_name: str, threshold_pct: float) -> Dict[str, Any]:
        return await self._fetcher.set_account_stop_loss(account_name, threshold_pct)

    async def clear_account_stop_loss(self, account_name: str) -> None:
        await self._fetcher.clear_account_stop_loss(account_name)

    async def cancel_all_orders(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        return await self._fetcher.cancel_all_orders(account_name, symbol)

    async def close_all_positions(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        return await self._fetcher.close_all_positions(account_name, symbol)

    @property
    def fetcher(self) -> RealtimeDataFetcher:
        """Expose the underlying fetcher for advanced scenarios."""

        return self._fetcher
