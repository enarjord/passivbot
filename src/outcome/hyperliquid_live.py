from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Mapping, Protocol, Sequence

from outcome.adapters import hyperliquid
from outcome.models import (
    NormalizedOutcomeMarket,
    OutcomeAccountFill,
    OutcomeBookSnapshot,
    OutcomeCollateralBalance,
    OutcomeOpenOrder,
    OutcomeOrderSide,
    OutcomeSide,
    OutcomeSettlementEvidence,
    OutcomeTokenBalance,
    OutcomeVenue,
)


class HyperliquidOutcomeSession(Protocol):
    async def publicPostInfo(self, payload: Mapping[str, Any]) -> Any: ...

    async def privatePostExchange(self, payload: Mapping[str, Any]) -> Any: ...

    def milliseconds(self) -> int: ...

    def sign_l1_action(
        self,
        action: Mapping[str, Any],
        nonce: int,
        vault_address: str | None = None,
        expires_after: int | None = None,
    ) -> Mapping[str, Any]: ...


class OutcomeMutationDisabled(RuntimeError):
    pass


class OutcomeActionRejected(RuntimeError):
    pass


class HyperliquidOutcomeLifecycleState(str, Enum):
    ACTIVE = "active"
    EXPIRED_AWAITING_SETTLEMENT = "expired_awaiting_settlement"
    SETTLED = "settled"


@dataclass(frozen=True)
class HyperliquidOutcomeFeeRates:
    """Current account rates reported by Hyperliquid's userFees endpoint."""

    user_add_rate: float
    user_cross_rate: float
    user_spot_add_rate: float
    user_spot_cross_rate: float

    def __post_init__(self) -> None:
        for name in (
            "user_add_rate",
            "user_cross_rate",
            "user_spot_add_rate",
            "user_spot_cross_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"HIP-4 account {name} must be finite and in [-1, 1]")

    @property
    def conservative_maker_rate(self) -> float:
        """Conservative outcome maker rate until HIP-4-specific incidence is authoritative."""

        return max(0.0, self.user_add_rate, self.user_spot_add_rate)

    @property
    def conservative_taker_rate(self) -> float:
        """Conservative outcome taker rate until HIP-4-specific incidence is authoritative."""

        return max(0.0, self.user_cross_rate, self.user_spot_cross_rate)


@dataclass(frozen=True)
class HyperliquidOutcomeAccountSnapshot:
    received_time_ms: int
    collateral: OutcomeCollateralBalance
    fee_rates: HyperliquidOutcomeFeeRates
    token_balances: tuple[OutcomeTokenBalance, ...]
    open_orders: tuple[OutcomeOpenOrder, ...]
    recent_fills: tuple[OutcomeAccountFill, ...]
    unknown_outcome_balance_coins: tuple[str, ...]
    unknown_outcome_order_coins: tuple[str, ...]
    unknown_outcome_fill_coins: tuple[str, ...]
    settlements: tuple[OutcomeSettlementEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.received_time_ms < 0:
            raise ValueError("outcome account snapshot receive time must be non-negative")

    def balance(self, market_id: str, outcome: OutcomeSide) -> OutcomeTokenBalance:
        matches = [
            balance
            for balance in self.token_balances
            if balance.market_id == market_id and balance.outcome is outcome
        ]
        if len(matches) != 1:
            raise ValueError(
                f"outcome snapshot does not contain exactly one {market_id} {outcome.value} balance"
            )
        return matches[0]

    def settlement(self, market_id: str) -> OutcomeSettlementEvidence | None:
        matches = [
            settlement
            for settlement in self.settlements
            if settlement.market_id == market_id
        ]
        if not matches:
            return None
        yes_fractions = {settlement.yes_fraction for settlement in matches}
        if len(yes_fractions) != 1:
            raise ValueError(f"outcome snapshot has conflicting settlement for {market_id}")
        return max(
            matches,
            key=lambda settlement: (
                settlement.settlement_time_ms,
                settlement.received_time_ms,
                settlement.source_event_id,
            ),
        )


@dataclass(frozen=True)
class HyperliquidOutcomeLifecycleSnapshot:
    market_id: str
    state: HyperliquidOutcomeLifecycleState
    observed_at_ms: int
    settlement: OutcomeSettlementEvidence | None

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("HIP-4 lifecycle market_id must not be empty")
        if self.observed_at_ms < 0:
            raise ValueError("HIP-4 lifecycle observed_at_ms must be non-negative")
        if (self.state is HyperliquidOutcomeLifecycleState.SETTLED) != (
            self.settlement is not None
        ):
            raise ValueError("HIP-4 settled lifecycle requires exactly one settlement evidence")


@dataclass(frozen=True)
class HyperliquidOutcomeMutationResult:
    kind: str
    order_id: str | None
    filled_qty: float
    average_price: float | None
    raw_response: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in {"resting", "filled", "cancelled"}:
            raise ValueError(f"unsupported HIP-4 mutation result kind {self.kind!r}")
        if not math.isfinite(self.filled_qty) or self.filled_qty < 0.0:
            raise ValueError("HIP-4 mutation filled quantity must be finite and non-negative")
        if self.average_price is not None and (
            not math.isfinite(self.average_price)
            or not 0.0 <= self.average_price <= 1.0
        ):
            raise ValueError("HIP-4 mutation average price must be in [0, 1]")


def _selected_identifiers(markets: Sequence[NormalizedOutcomeMarket]) -> set[str]:
    identifiers = set()
    for market in markets:
        if market.venue is not OutcomeVenue.HYPERLIQUID:
            raise ValueError("HIP-4 live client only accepts Hyperliquid outcome markets")
        for asset in (market.yes_asset, market.no_asset):
            identifiers.update(
                {
                    asset.asset_id,
                    asset.market_data_symbol,
                    asset.order_asset_id,
                }
            )
    return identifiers


def _unknown_outcome_coins(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_identifiers: set[str],
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                coin
                for row in rows
                if (coin := str(row.get("coin", ""))).startswith(prefixes)
                and coin not in selected_identifiers
            }
        )
    )


def _normalize_user_fee_rates(payload: Mapping[str, Any]) -> HyperliquidOutcomeFeeRates:
    def required_rate(native_key: str) -> float:
        try:
            value = float(payload[native_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Hyperliquid userFees response omitted valid {native_key}"
            ) from exc
        return value

    return HyperliquidOutcomeFeeRates(
        user_add_rate=required_rate("userAddRate"),
        user_cross_rate=required_rate("userCrossRate"),
        user_spot_add_rate=required_rate("userSpotAddRate"),
        user_spot_cross_rate=required_rate("userSpotCrossRate"),
    )


class HyperliquidOutcomeLiveClient:
    """Isolated HIP-4 state and action adapter.

    Mutation methods are disabled unless the caller explicitly opts in. The client does not share
    Passivbot's perpetual position model and does not infer absent lifecycle, fee, inventory, or
    collateral fields.
    """

    def __init__(
        self,
        session: HyperliquidOutcomeSession,
        *,
        account_address: str,
        allow_mutations: bool = False,
        vault_address: str | None = None,
    ) -> None:
        if not account_address.strip():
            raise ValueError("HIP-4 account address must not be empty")
        self.session = session
        self.account_address = account_address
        self.allow_mutations = bool(allow_mutations)
        self.vault_address = vault_address

    async def fetch_account_snapshot(
        self,
        markets: Sequence[NormalizedOutcomeMarket],
    ) -> HyperliquidOutcomeAccountSnapshot:
        market_list = tuple(markets)
        if not market_list:
            raise ValueError("HIP-4 account snapshot requires at least one market")
        quote_assets = {market.quote_asset for market in market_list}
        if len(quote_assets) != 1:
            raise ValueError("one HIP-4 account snapshot cannot mix quote assets")
        identifiers = _selected_identifiers(market_list)
        state, open_orders, fills, user_fees = await asyncio.gather(
            self.session.publicPostInfo(
                {
                    "type": "spotClearinghouseState",
                    "user": self.account_address,
                }
            ),
            self.session.publicPostInfo(
                {
                    "type": "frontendOpenOrders",
                    "user": self.account_address,
                }
            ),
            self.session.publicPostInfo(
                {
                    "type": "userFills",
                    "user": self.account_address,
                    "aggregateByTime": False,
                }
            ),
            self.session.publicPostInfo(
                {
                    "type": "userFees",
                    "user": self.account_address,
                }
            ),
        )
        if not isinstance(state, Mapping):
            raise ValueError("Hyperliquid spotClearinghouseState response must be an object")
        if not isinstance(open_orders, list) or not all(
            isinstance(row, Mapping) for row in open_orders
        ):
            raise ValueError("Hyperliquid frontendOpenOrders response must be an object array")
        if not isinstance(fills, list) or not all(isinstance(row, Mapping) for row in fills):
            raise ValueError("Hyperliquid userFills response must be an object array")
        if not isinstance(user_fees, Mapping):
            raise ValueError("Hyperliquid userFees response must be an object")
        balances = state.get("balances")
        if not isinstance(balances, list) or not all(
            isinstance(row, Mapping) for row in balances
        ):
            raise ValueError("Hyperliquid spot balances response must be an object array")
        quote_asset = next(iter(quote_assets))
        received_time_ms = int(time.time() * 1_000)
        normalized_fills = hyperliquid.normalize_account_fills(fills, market_list)
        settlements = tuple(
            settlement
            for market in market_list
            for settlement in hyperliquid.normalize_settlement_evidence(
                normalized_fills,
                market,
                received_time_ms=received_time_ms,
                evidence_source="hyperliquid_user_fills",
            )
        )
        return HyperliquidOutcomeAccountSnapshot(
            received_time_ms=received_time_ms,
            collateral=hyperliquid.normalize_collateral_balance(
                state,
                quote_asset=quote_asset,
            ),
            fee_rates=_normalize_user_fee_rates(user_fees),
            token_balances=hyperliquid.normalize_token_balances(state, market_list),
            open_orders=hyperliquid.normalize_open_orders(open_orders, market_list),
            recent_fills=normalized_fills,
            unknown_outcome_balance_coins=_unknown_outcome_coins(
                balances,
                selected_identifiers=identifiers,
                prefixes=("+",),
            ),
            unknown_outcome_order_coins=_unknown_outcome_coins(
                open_orders,
                selected_identifiers=identifiers,
                prefixes=("#",),
            ),
            unknown_outcome_fill_coins=_unknown_outcome_coins(
                fills,
                selected_identifiers=identifiers,
                prefixes=("#",),
            ),
            settlements=settlements,
        )

    async def fetch_book(
        self,
        market: NormalizedOutcomeMarket,
        *,
        outcome: OutcomeSide,
    ) -> OutcomeBookSnapshot:
        _selected_identifiers((market,))
        asset = market.yes_asset if outcome is OutcomeSide.YES else market.no_asset
        payload = await self.session.publicPostInfo(
            {
                "type": "l2Book",
                "coin": asset.market_data_symbol,
            }
        )
        received_time_ms = int(time.time() * 1_000)
        if not isinstance(payload, Mapping):
            raise ValueError("Hyperliquid l2Book response must be an object")
        return hyperliquid.normalize_l2_book(
            payload,
            market,
            outcome=outcome,
            received_time_ms=received_time_ms,
        )

    async def fetch_market_lifecycle(
        self,
        market: NormalizedOutcomeMarket,
        *,
        account: HyperliquidOutcomeAccountSnapshot | None = None,
        now_ms: int | None = None,
    ) -> HyperliquidOutcomeLifecycleSnapshot:
        _selected_identifiers((market,))
        payload = await self.session.publicPostInfo({"type": "outcomeMeta"})
        if not isinstance(payload, Mapping) or not isinstance(payload.get("outcomes"), list):
            raise ValueError("Hyperliquid outcomeMeta response is unavailable")
        matching = [
            row
            for row in payload["outcomes"]
            if isinstance(row, Mapping) and str(row.get("outcome", "")) == market.market_id
        ]
        if len(matching) > 1:
            raise ValueError(f"HIP-4 market {market.market_id} metadata is ambiguous")
        if matching:
            refreshed = hyperliquid.normalize_market(matching[0])
            if (
                refreshed.description != market.description
                or refreshed.quote_asset != market.quote_asset
                or refreshed.yes_asset != market.yes_asset
                or refreshed.no_asset != market.no_asset
                or refreshed.lifecycle.scheduled_event_time_ms
                != market.lifecycle.scheduled_event_time_ms
            ):
                raise ValueError(f"HIP-4 market {market.market_id} metadata changed")
        event_time_ms = market.lifecycle.scheduled_event_time_ms
        if event_time_ms is None:
            raise ValueError(f"HIP-4 market {market.market_id} has no scheduled expiry")
        observed_at_ms = int(time.time() * 1_000) if now_ms is None else int(now_ms)
        if observed_at_ms < 0:
            raise ValueError("HIP-4 lifecycle observation time must be non-negative")
        settlement = account.settlement(market.market_id) if account is not None else None
        if settlement is None and observed_at_ms >= event_time_ms:
            historical_settlements = await self.fetch_settlement_evidence(
                market,
                start_time_ms=event_time_ms,
                end_time_ms=observed_at_ms,
            )
            if historical_settlements:
                yes_fractions = {
                    evidence.yes_fraction for evidence in historical_settlements
                }
                if len(yes_fractions) != 1:
                    raise ValueError(
                        f"HIP-4 market {market.market_id} has conflicting settlement history"
                    )
                settlement = max(
                    historical_settlements,
                    key=lambda evidence: (
                        evidence.settlement_time_ms,
                        evidence.received_time_ms,
                        evidence.source_event_id,
                    ),
                )
        if settlement is not None:
            if settlement.settlement_time_ms < event_time_ms:
                raise ValueError("HIP-4 settlement predates the scheduled event")
            return HyperliquidOutcomeLifecycleSnapshot(
                market_id=market.market_id,
                state=HyperliquidOutcomeLifecycleState.SETTLED,
                observed_at_ms=observed_at_ms,
                settlement=settlement,
            )
        if matching and observed_at_ms < event_time_ms:
            return HyperliquidOutcomeLifecycleSnapshot(
                market_id=market.market_id,
                state=HyperliquidOutcomeLifecycleState.ACTIVE,
                observed_at_ms=observed_at_ms,
                settlement=None,
            )
        if not matching and observed_at_ms < event_time_ms:
            raise ValueError(
                f"HIP-4 market {market.market_id} disappeared before scheduled expiry"
            )
        return HyperliquidOutcomeLifecycleSnapshot(
            market_id=market.market_id,
            state=HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT,
            observed_at_ms=observed_at_ms,
            settlement=None,
        )

    async def fetch_settlement_evidence(
        self,
        market: NormalizedOutcomeMarket,
        *,
        start_time_ms: int,
        end_time_ms: int,
    ) -> tuple[OutcomeSettlementEvidence, ...]:
        """Retrieve account settlement rows from Hyperliquid's bounded fill history."""

        _selected_identifiers((market,))
        if start_time_ms < 0 or end_time_ms < start_time_ms:
            raise ValueError("HIP-4 settlement history range is invalid")
        payload = await self.session.publicPostInfo(
            {
                "type": "userFillsByTime",
                "user": self.account_address,
                "startTime": int(start_time_ms),
                "endTime": int(end_time_ms),
                "aggregateByTime": False,
            }
        )
        received_time_ms = int(time.time() * 1_000)
        if not isinstance(payload, list) or not all(
            isinstance(row, Mapping) for row in payload
        ):
            raise ValueError("Hyperliquid userFillsByTime response must be an object array")
        fills = hyperliquid.normalize_account_fills(payload, (market,))
        return hyperliquid.normalize_settlement_evidence(
            fills,
            market,
            received_time_ms=received_time_ms,
            evidence_source="hyperliquid_user_fills_by_time",
        )

    async def assert_market_current(self, market: NormalizedOutcomeMarket) -> None:
        lifecycle = await self.fetch_market_lifecycle(market)
        if lifecycle.state is not HyperliquidOutcomeLifecycleState.ACTIVE:
            raise ValueError(
                f"HIP-4 market {market.market_id} is not active: {lifecycle.state.value}"
            )

    def _require_mutations_enabled(self) -> None:
        if not self.allow_mutations:
            raise OutcomeMutationDisabled(
                "HIP-4 exchange mutations are disabled for this client"
            )

    async def submit_limit_order(
        self,
        market: NormalizedOutcomeMarket,
        *,
        outcome: OutcomeSide,
        side: OutcomeOrderSide,
        native_price: float | str,
        qty: float | str,
        client_order_id: str | None = None,
        post_only: bool = True,
    ) -> HyperliquidOutcomeMutationResult:
        self._require_mutations_enabled()
        if not post_only:
            raise ValueError("initial HIP-4 live integration only permits post-only orders")
        action = hyperliquid.build_limit_order_action(
            market,
            outcome=outcome,
            side=side,
            native_price=native_price,
            qty=qty,
            client_order_id=client_order_id,
            time_in_force="Alo",
        )
        await self.assert_market_current(market)
        snapshot, book = await asyncio.gather(
            self.fetch_account_snapshot((market,)),
            self.fetch_book(market, outcome=outcome),
        )
        price = float(native_price)
        quantity = float(qty)
        if side is OutcomeOrderSide.BUY:
            conservative_fee_reserve = (
                quantity
                * market.payout_unit
                * snapshot.fee_rates.conservative_maker_rate
            )
            required = price * quantity + conservative_fee_reserve
            if required > snapshot.collateral.conservative_available + 1e-12:
                raise ValueError(
                    f"insufficient HIP-4 collateral: required {required}, "
                    f"available {snapshot.collateral.conservative_available}"
                )
            if book.asks and price >= book.asks[0].native_price:
                raise ValueError("post-only HIP-4 buy would cross the current ask")
        else:
            available = snapshot.balance(market.market_id, outcome).available_qty
            if quantity > available + 1e-12:
                raise ValueError(
                    f"insufficient HIP-4 {outcome.value} inventory: "
                    f"required {quantity}, available {available}"
                )
            if book.bids and price <= book.bids[0].native_price:
                raise ValueError("post-only HIP-4 sell would cross the current bid")
        response = await self._send_action(action)
        return _parse_order_response(response)

    async def cancel_order(
        self,
        market: NormalizedOutcomeMarket,
        *,
        outcome: OutcomeSide,
        order_id: int,
        expected_client_order_id: str,
    ) -> HyperliquidOutcomeMutationResult:
        self._require_mutations_enabled()
        if not expected_client_order_id:
            raise ValueError("HIP-4 cancel requires the expected client-order ID")
        snapshot = await self.fetch_account_snapshot((market,))
        await self.fetch_market_lifecycle(market, account=snapshot)
        matches = [
            order
            for order in snapshot.open_orders
            if (
                order.order_id == str(order_id)
                and order.outcome is outcome
                and order.client_order_id == expected_client_order_id
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "HIP-4 cancel target is not an authoritative open order for this "
                "market/side/client-order ID"
            )
        action = hyperliquid.build_cancel_action(
            market,
            outcome=outcome,
            order_id=order_id,
        )
        response = await self._send_action(action)
        return _parse_cancel_response(response)

    async def _send_action(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        nonce = int(self.session.milliseconds())
        signature = self.session.sign_l1_action(
            action,
            nonce,
            self.vault_address,
        )
        request: dict[str, Any] = {
            "action": dict(action),
            "nonce": nonce,
            "signature": dict(signature),
        }
        if self.vault_address is not None:
            request["vaultAddress"] = self.vault_address
        response = await self.session.privatePostExchange(request)
        if not isinstance(response, Mapping):
            raise OutcomeActionRejected("Hyperliquid exchange response was not an object")
        return response


def _response_statuses(response: Mapping[str, Any], expected_type: str) -> list[Any]:
    if response.get("status") != "ok":
        raise OutcomeActionRejected(f"Hyperliquid action failed: {response!r}")
    response_body = response.get("response")
    if not isinstance(response_body, Mapping) or response_body.get("type") != expected_type:
        raise OutcomeActionRejected(
            f"Hyperliquid action returned unexpected response type: {response!r}"
        )
    data = response_body.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("statuses"), list):
        raise OutcomeActionRejected("Hyperliquid action response omitted statuses")
    statuses = data["statuses"]
    if len(statuses) != 1:
        raise OutcomeActionRejected("single HIP-4 action returned an unexpected status count")
    return statuses


def _parse_order_response(
    response: Mapping[str, Any],
) -> HyperliquidOutcomeMutationResult:
    status = _response_statuses(response, "order")[0]
    if isinstance(status, str):
        raise OutcomeActionRejected(f"Hyperliquid rejected HIP-4 order: {status}")
    if not isinstance(status, Mapping):
        raise OutcomeActionRejected("Hyperliquid HIP-4 order status was malformed")
    resting = status.get("resting")
    if isinstance(resting, Mapping):
        order_id = resting.get("oid")
        if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id < 0:
            raise OutcomeActionRejected("Hyperliquid resting order omitted a valid oid")
        return HyperliquidOutcomeMutationResult(
            kind="resting",
            order_id=str(order_id),
            filled_qty=0.0,
            average_price=None,
            raw_response=dict(response),
        )
    filled = status.get("filled")
    if isinstance(filled, Mapping):
        order_id = filled.get("oid")
        if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id < 0:
            raise OutcomeActionRejected("Hyperliquid filled order omitted a valid oid")
        try:
            filled_qty = float(filled["totalSz"])
            average_price = float(filled["avgPx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OutcomeActionRejected("Hyperliquid filled order status was malformed") from exc
        return HyperliquidOutcomeMutationResult(
            kind="filled",
            order_id=str(order_id),
            filled_qty=filled_qty,
            average_price=average_price,
            raw_response=dict(response),
        )
    error = status.get("error")
    if error is not None:
        raise OutcomeActionRejected(f"Hyperliquid rejected HIP-4 order: {error}")
    raise OutcomeActionRejected(f"unrecognized Hyperliquid HIP-4 order status: {status!r}")


def _parse_cancel_response(
    response: Mapping[str, Any],
) -> HyperliquidOutcomeMutationResult:
    status = _response_statuses(response, "cancel")[0]
    if status != "success":
        raise OutcomeActionRejected(f"Hyperliquid rejected HIP-4 cancel: {status!r}")
    return HyperliquidOutcomeMutationResult(
        kind="cancelled",
        order_id=None,
        filled_qty=0.0,
        average_price=None,
        raw_response=dict(response),
    )
