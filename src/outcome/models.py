from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class OutcomeVenue(str, Enum):
    HYPERLIQUID = "hyperliquid"
    POLYMARKET = "polymarket"


class OutcomeSide(str, Enum):
    YES = "yes"
    NO = "no"


class OutcomeOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OutcomeCollateralBalance:
    asset: str
    total: float
    held: float
    available_after_maintenance: float | None = None

    def __post_init__(self) -> None:
        if not self.asset.strip():
            raise ValueError("collateral asset must not be empty")
        for name in ("total", "held"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"collateral {name} must be finite and non-negative")
        if self.held > self.total + 1e-12:
            raise ValueError("collateral held amount exceeds total")
        if self.available_after_maintenance is not None and (
            not math.isfinite(self.available_after_maintenance)
            or self.available_after_maintenance < 0.0
        ):
            raise ValueError(
                "collateral available_after_maintenance must be finite and non-negative"
            )

    @property
    def unheld(self) -> float:
        return max(0.0, self.total - self.held)

    @property
    def conservative_available(self) -> float:
        if self.available_after_maintenance is None:
            return self.unheld
        return min(self.unheld, self.available_after_maintenance)


@dataclass(frozen=True)
class OutcomeTokenBalance:
    market_id: str
    asset_id: str
    outcome: OutcomeSide
    total_qty: float
    held_qty: float
    entry_notional: float

    def __post_init__(self) -> None:
        if not self.market_id or not self.asset_id:
            raise ValueError("outcome token balance requires market and asset IDs")
        for name in ("total_qty", "held_qty", "entry_notional"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"outcome token {name} must be finite and non-negative")
        if self.held_qty > self.total_qty + 1e-12:
            raise ValueError("outcome token held quantity exceeds total")

    @property
    def available_qty(self) -> float:
        return max(0.0, self.total_qty - self.held_qty)


@dataclass(frozen=True)
class OutcomeOpenOrder:
    market_id: str
    order_id: str
    asset_id: str
    outcome: OutcomeSide
    side: OutcomeOrderSide
    native_price: float
    qty: float
    original_qty: float
    timestamp_ms: int
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        if not self.market_id or not self.order_id or not self.asset_id:
            raise ValueError("outcome open order requires market, order, and asset IDs")
        if not math.isfinite(self.native_price) or not 0.0 <= self.native_price <= 1.0:
            raise ValueError("outcome open-order price must be in [0, 1]")
        for name in ("qty", "original_qty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"outcome open-order {name} must be finite and positive")
        if self.qty > self.original_qty + 1e-12:
            raise ValueError("outcome open-order remaining quantity exceeds original")
        if self.timestamp_ms < 0:
            raise ValueError("outcome open-order timestamp must be non-negative")


@dataclass(frozen=True)
class OutcomeAccountFill:
    market_id: str
    trade_id: str
    transaction_hash: str
    order_id: str
    asset_id: str
    outcome: OutcomeSide
    side: OutcomeOrderSide
    native_price: float
    qty: float
    fee: float
    fee_asset: str
    is_maker: bool
    timestamp_ms: int
    direction: str
    start_position_qty: float

    def __post_init__(self) -> None:
        for name in ("market_id", "trade_id", "transaction_hash", "order_id", "asset_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"outcome account fill {name} must not be empty")
        if not math.isfinite(self.native_price) or not 0.0 <= self.native_price <= 1.0:
            raise ValueError("outcome account-fill price must be in [0, 1]")
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ValueError("outcome account-fill quantity must be finite and positive")
        if not math.isfinite(self.fee):
            raise ValueError("outcome account-fill fee must be finite")
        if not self.fee_asset:
            raise ValueError("outcome account-fill fee asset must not be empty")
        if self.timestamp_ms < 0:
            raise ValueError("outcome account-fill timestamp must be non-negative")
        if not math.isfinite(self.start_position_qty):
            raise ValueError("outcome account-fill start position must be finite")

    @property
    def is_settlement(self) -> bool:
        return self.direction.casefold() == "settlement"


@dataclass(frozen=True)
class OutcomeSettlementEvidence:
    """Authoritative market-level resolution evidence, separate from ordinary trades."""

    venue: OutcomeVenue
    market_id: str
    yes_fraction: float
    payout_unit: float
    settlement_time_ms: int
    capital_release_time_ms: int | None
    received_time_ms: int
    source_event_id: str
    evidence_source: str
    observed_yes_qty: float
    observed_no_qty: float
    collateral_payout: float
    fee: float
    fee_asset: str
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("market_id", "source_event_id", "evidence_source", "fee_asset"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"outcome settlement {name} must not be empty")
        if not math.isfinite(self.yes_fraction) or not 0.0 <= self.yes_fraction <= 1.0:
            raise ValueError("outcome settlement yes_fraction must be in [0, 1]")
        if not math.isfinite(self.payout_unit) or self.payout_unit <= 0.0:
            raise ValueError("outcome settlement payout_unit must be finite and positive")
        if self.settlement_time_ms < 0 or self.received_time_ms < 0:
            raise ValueError("outcome settlement timestamps must be non-negative")
        if self.capital_release_time_ms is not None:
            if self.capital_release_time_ms < self.settlement_time_ms:
                raise ValueError(
                    "outcome capital release must not predate resolution evidence"
                )
        for name in (
            "observed_yes_qty",
            "observed_no_qty",
            "collateral_payout",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"outcome settlement {name} must be finite and non-negative")
        if not math.isfinite(self.fee):
            raise ValueError("outcome settlement fee must be finite")


@dataclass(frozen=True)
class OutcomeBookLevel:
    native_price: float
    qty: float
    order_count: int | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.native_price) or not 0.0 <= self.native_price <= 1.0:
            raise ValueError("outcome book price must be in [0, 1]")
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ValueError("outcome book quantity must be finite and positive")
        if self.order_count is not None and self.order_count <= 0:
            raise ValueError("outcome book order count must be positive")


@dataclass(frozen=True)
class OutcomeBookSnapshot:
    venue: OutcomeVenue
    market_id: str
    asset_id: str
    outcome: OutcomeSide
    timestamp_ms: int
    received_time_ms: int
    bids: tuple[OutcomeBookLevel, ...]
    asks: tuple[OutcomeBookLevel, ...]
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.market_id or not self.asset_id:
            raise ValueError("outcome book requires market and asset IDs")
        if self.timestamp_ms < 0 or self.received_time_ms < 0:
            raise ValueError("outcome book timestamps must be non-negative")
        if any(
            self.bids[index].native_price <= self.bids[index + 1].native_price
            for index in range(len(self.bids) - 1)
        ):
            raise ValueError("outcome book bids must be strictly descending")
        if any(
            self.asks[index].native_price >= self.asks[index + 1].native_price
            for index in range(len(self.asks) - 1)
        ):
            raise ValueError("outcome book asks must be strictly ascending")
        if self.bids and self.asks and self.bids[0].native_price >= self.asks[0].native_price:
            raise ValueError("outcome book must not be crossed")


@dataclass(frozen=True)
class NormalizedOutcomeAsset:
    side: OutcomeSide
    label: str
    asset_id: str
    market_data_symbol: str
    order_asset_id: str

    def __post_init__(self) -> None:
        for name in ("label", "asset_id", "market_data_symbol", "order_asset_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"outcome asset {name} must not be empty")


@dataclass(frozen=True)
class OutcomeCapabilities:
    complementary_books_merged: bool
    sell_requires_inventory: bool
    supports_split: bool
    supports_merge: bool
    supports_redeem: bool
    supports_post_only: bool
    supports_gtd: bool


@dataclass(frozen=True)
class OutcomeFeeMetadata:
    """Venue-reported fee configuration, not a computed or assumed fee."""

    formula: str
    maker_rate: float | None = None
    taker_rate: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.formula.strip():
            raise ValueError("fee formula must not be empty")
        for name in ("maker_rate", "taker_rate"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class OutcomePriceGridMetadata:
    kind: str
    fixed_step: float | None = None
    max_significant_figures: int | None = None
    max_decimal_places: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "fixed_step":
            if (
                self.fixed_step is None
                or not math.isfinite(self.fixed_step)
                or self.fixed_step <= 0.0
                or self.max_significant_figures is not None
                or self.max_decimal_places is not None
            ):
                raise ValueError("fixed price grid requires only a positive fixed_step")
        elif self.kind == "significant_figures":
            if (
                self.fixed_step is not None
                or self.max_significant_figures is None
                or self.max_significant_figures <= 0
                or self.max_decimal_places is None
                or self.max_decimal_places < 0
            ):
                raise ValueError(
                    "significant-figure grid requires max_significant_figures and max_decimal_places"
                )
        else:
            raise ValueError(f"unsupported outcome price grid kind {self.kind!r}")


@dataclass(frozen=True)
class OutcomePriceGridChange:
    venue: OutcomeVenue
    market_id: str
    timestamp_ms: int
    received_time_ms: int
    old_grid: OutcomePriceGridMetadata
    new_grid: OutcomePriceGridMetadata
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.market_id:
            raise ValueError("price-grid change market_id must not be empty")
        if self.timestamp_ms < 0 or self.received_time_ms < 0:
            raise ValueError("price-grid change timestamps must be non-negative")
        if self.old_grid == self.new_grid:
            raise ValueError("price-grid change must change the grid")


@dataclass(frozen=True)
class MarketLifecycle:
    discovery_time_ms: int | None = None
    trading_open_time_ms: int | None = None
    order_acceptance_time_ms: int | None = None
    trading_close_time_ms: int | None = None
    scheduled_event_time_ms: int | None = None
    resolution_time_ms: int | None = None
    settlement_time_ms: int | None = None
    accepting_orders: bool | None = None
    resolved: bool | None = None
    yes_payout_fraction: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "discovery_time_ms",
            "trading_open_time_ms",
            "order_acceptance_time_ms",
            "trading_close_time_ms",
            "scheduled_event_time_ms",
            "resolution_time_ms",
            "settlement_time_ms",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.yes_payout_fraction is not None and not 0.0 <= self.yes_payout_fraction <= 1.0:
            raise ValueError("yes_payout_fraction must be in [0, 1]")


@dataclass(frozen=True)
class NormalizedOutcomeMarket:
    venue: OutcomeVenue
    market_id: str
    title: str
    description: str
    quote_asset: str
    yes_asset: NormalizedOutcomeAsset
    no_asset: NormalizedOutcomeAsset
    payout_unit: float
    price_grid: OutcomePriceGridMetadata
    qty_step: float | None
    min_order_qty: float | None
    min_order_notional: float | None
    lifecycle: MarketLifecycle
    capabilities: OutcomeCapabilities
    fee_metadata: OutcomeFeeMetadata
    native_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("market_id must not be empty")
        if self.yes_asset.side is not OutcomeSide.YES or self.no_asset.side is not OutcomeSide.NO:
            raise ValueError("yes_asset and no_asset must have their corresponding canonical sides")
        if not math.isfinite(self.payout_unit) or self.payout_unit <= 0.0:
            raise ValueError("payout_unit must be finite and positive")
        for name in ("qty_step", "min_order_qty", "min_order_notional"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")

    def asset_for_id(self, asset_id: str) -> NormalizedOutcomeAsset:
        normalized = str(asset_id)
        for asset in (self.yes_asset, self.no_asset):
            if normalized in {asset.asset_id, asset.market_data_symbol, asset.order_asset_id}:
                return asset
        raise ValueError(f"asset {asset_id!r} does not belong to market {self.market_id!r}")


@dataclass(frozen=True)
class NormalizedOutcomeTrade:
    venue: OutcomeVenue
    market_id: str
    asset_id: str
    outcome: OutcomeSide
    native_side: OutcomeOrderSide
    native_price: float
    canonical_yes_price: float
    qty: float
    exchange_time_ms: int
    received_time_ms: int
    source_event_id: str | None = None
    economic_event_id: str | None = None
    sequence_id: str | None = None
    collector_sequence: int | None = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.market_id or not self.asset_id:
            raise ValueError("trade market_id and asset_id must not be empty")
        for name in ("native_price", "canonical_yes_price"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ValueError("trade qty must be finite and positive")
        if self.exchange_time_ms < 0 or self.received_time_ms < 0:
            raise ValueError("trade timestamps must be non-negative")
        if self.collector_sequence is not None and self.collector_sequence < 0:
            raise ValueError("collector_sequence must be non-negative")

    @property
    def canonical_exposure_delta(self) -> float:
        sign = 1.0 if self.native_side is OutcomeOrderSide.BUY else -1.0
        return self.qty * sign * (1.0 if self.outcome is OutcomeSide.YES else -1.0)

    @property
    def deduplication_key(self) -> tuple[str, ...] | None:
        if self.source_event_id is not None:
            return (self.venue.value, self.market_id, self.asset_id, "event", self.source_event_id)
        if self.sequence_id is not None:
            return (self.venue.value, self.market_id, self.asset_id, "sequence", self.sequence_id)
        return None

    @property
    def economic_deduplication_key(self) -> tuple[str, ...] | None:
        if self.economic_event_id is None:
            return None
        return (self.venue.value, self.market_id, "economic", self.economic_event_id)


@dataclass(frozen=True)
class OutcomeCandle1s:
    timestamp_ms: int
    source_outcome: OutcomeSide
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    carried_forward: bool

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.timestamp_ms % 1_000 != 0:
            raise ValueError("one-second candle timestamp must be non-negative and second-aligned")
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"candle {name} must be finite and in [0, 1]")
        if self.high < self.low or not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise ValueError("invalid candle price ordering")
        if not math.isfinite(self.volume) or self.volume < 0.0:
            raise ValueError("candle volume must be finite and non-negative")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if self.carried_forward != (self.trade_count == 0):
            raise ValueError("carried_forward must be true exactly for no-trade candles")
        if self.carried_forward and self.volume != 0.0:
            raise ValueError("carried-forward candle volume must be zero")


@dataclass(frozen=True)
class OutcomeSignalCandle1s:
    """Canonical YES candle combining actual fills from both native books."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    carried_forward: bool

    def __post_init__(self) -> None:
        # Reuse the same validation contract without assigning mixed trades to a native book.
        OutcomeCandle1s(
            timestamp_ms=self.timestamp_ms,
            source_outcome=OutcomeSide.YES,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            trade_count=self.trade_count,
            carried_forward=self.carried_forward,
        )
