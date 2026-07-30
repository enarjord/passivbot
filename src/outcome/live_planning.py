from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from outcome.hyperliquid_live import HyperliquidOutcomeAccountSnapshot
from outcome.models import (
    NormalizedOutcomeMarket,
    OutcomeOrderSide,
    OutcomeSide,
    OutcomeSignalCandle1s,
    OutcomeTokenBalance,
)
from outcome.order_ownership import is_managed_outcome_client_order_id
from outcome.rust_runner import (
    normalized_market_to_rust_spec,
    plan_outcome_ema_anchor,
)


DEFAULT_OUTCOME_MAX_ACCOUNT_AGE_MS = 5_000
DEFAULT_OUTCOME_MAX_SIGNAL_AGE_MS = 5_000


class OutcomeSignalPlanningUnavailable(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        if reason not in {
            "incomplete_verified_signal",
            "stale_verified_signal",
            "market_constraints_unavailable",
        }:
            raise ValueError("unsupported outcome signal unavailability reason")
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class OutcomeLiveOrderIntent:
    slot: str
    outcome: OutcomeSide
    side: OutcomeOrderSide
    native_price: float
    canonical_yes_price: float
    qty: float

    def __post_init__(self) -> None:
        if self.slot not in {"canonical_bid", "canonical_ask"}:
            raise ValueError(f"unsupported live outcome intent slot {self.slot!r}")
        for name in ("native_price", "canonical_yes_price"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"live outcome intent {name} must be finite and non-negative")
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ValueError("live outcome intent qty must be finite and positive")


@dataclass(frozen=True)
class OutcomeLivePlan:
    strategy_kind: str
    market_id: str
    observation_start_ms: int
    observation_end_ms: int
    observation_count: int
    ema_fast: float
    ema_slow: float
    inventory_shift: float
    configured_estimated_fee_per_share: float
    effective_estimated_fee_per_share: float
    estimated_fee_source: str
    intents: tuple[OutcomeLiveOrderIntent, ...]


def _average_cost(entry_notional: float, qty: float) -> float:
    return entry_notional / qty if qty > 0.0 else 0.0


def _planning_free_collateral(
    market: NormalizedOutcomeMarket,
    account: HyperliquidOutcomeAccountSnapshot,
) -> float:
    """Include collateral reclaimable from this market's managed replacement orders."""

    managed_buy_reserve = sum(
        order.native_price * order.qty
        for order in account.open_orders
        if order.market_id == market.market_id
        and order.side is OutcomeOrderSide.BUY
        and is_managed_outcome_client_order_id(
            order.client_order_id,
            market.market_id,
        )
    )
    reclaimable = min(account.collateral.held, managed_buy_reserve)
    return min(
        account.collateral.total,
        account.collateral.conservative_available + reclaimable,
    )


def _planning_available_inventory(
    market: NormalizedOutcomeMarket,
    account: HyperliquidOutcomeAccountSnapshot,
    balance: OutcomeTokenBalance,
) -> float:
    """Include inventory reclaimable only from this market's managed sell orders."""

    managed_sell_reserve = sum(
        order.qty
        for order in account.open_orders
        if order.market_id == market.market_id
        and order.asset_id == balance.asset_id
        and order.outcome is balance.outcome
        and order.side is OutcomeOrderSide.SELL
        and is_managed_outcome_client_order_id(
            order.client_order_id,
            market.market_id,
        )
    )
    reclaimable = min(balance.held_qty, managed_sell_reserve)
    return min(balance.total_qty, balance.available_qty + reclaimable)


def build_ema_anchor_outcome_live_plan(
    market: NormalizedOutcomeMarket,
    strategy_params: Mapping[str, Any],
    signal_candles: Sequence[OutcomeSignalCandle1s],
    account: HyperliquidOutcomeAccountSnapshot,
    *,
    now_ms: int,
    max_account_age_ms: int = DEFAULT_OUTCOME_MAX_ACCOUNT_AGE_MS,
    max_signal_age_ms: int = DEFAULT_OUTCOME_MAX_SIGNAL_AGE_MS,
) -> OutcomeLivePlan:
    """Create a restart-reproducible Rust plan from exchange state and archived candles."""

    if now_ms < 0:
        raise ValueError("live outcome planning now_ms must be non-negative")
    account_age_ms = now_ms - account.received_time_ms
    if account_age_ms < 0 or account_age_ms > max_account_age_ms:
        raise ValueError("HIP-4 account snapshot is stale or from the future")
    if not signal_candles:
        raise OutcomeSignalPlanningUnavailable(
            "incomplete_verified_signal",
            "live outcome planning requires signal candles",
        )
    for index, candle in enumerate(signal_candles):
        if index > 0 and candle.timestamp_ms != signal_candles[index - 1].timestamp_ms + 1_000:
            raise OutcomeSignalPlanningUnavailable(
                "incomplete_verified_signal",
                "live outcome signal candles must be contiguous",
            )
    latest = signal_candles[-1]
    signal_age_ms = now_ms - (latest.timestamp_ms + 1_000)
    if signal_age_ms < 0 or signal_age_ms > max_signal_age_ms:
        raise OutcomeSignalPlanningUnavailable(
            "stale_verified_signal",
            "live outcome signal candle is stale or incomplete",
        )

    yes = account.balance(market.market_id, OutcomeSide.YES)
    no = account.balance(market.market_id, OutcomeSide.NO)
    configured_params = dict(strategy_params)
    try:
        configured_fee_per_share = float(configured_params["estimated_fee_per_share"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "outcome strategy estimated_fee_per_share must be configured"
        ) from exc
    if not math.isfinite(configured_fee_per_share) or configured_fee_per_share < 0.0:
        raise ValueError(
            "outcome strategy estimated_fee_per_share must be finite and non-negative"
        )
    account_fee_floor = account.fee_rates.conservative_maker_rate * market.payout_unit
    effective_fee_per_share = max(configured_fee_per_share, account_fee_floor)
    configured_params["estimated_fee_per_share"] = effective_fee_per_share
    if (
        market.qty_step is None
        or market.min_order_qty is None
        or market.min_order_notional is None
    ):
        raise OutcomeSignalPlanningUnavailable(
            "market_constraints_unavailable",
            "HIP-4 order constraints are unavailable for live planning",
        )
    payload = {
        "market": normalized_market_to_rust_spec(market),
        "strategy_params": configured_params,
        "observations": [
            {
                "timestamp_ms": candle.timestamp_ms,
                "close": candle.close,
            }
            for candle in signal_candles
        ],
        "inventory": {
            "yes_qty": yes.total_qty,
            "no_qty": no.total_qty,
            "yes_available_qty": _planning_available_inventory(market, account, yes),
            "no_available_qty": _planning_available_inventory(market, account, no),
            "yes_average_cost": _average_cost(yes.entry_notional, yes.total_qty),
            "no_average_cost": _average_cost(no.entry_notional, no.total_qty),
            "free_collateral": _planning_free_collateral(market, account),
        },
    }
    output = plan_outcome_ema_anchor(payload)
    quotes = output.get("quotes")
    if not isinstance(quotes, Mapping):
        raise TypeError("Rust outcome planner omitted quotes")
    intents = []
    for key in ("canonical_bid", "canonical_ask"):
        raw_intent = quotes.get(key)
        if raw_intent is None:
            continue
        if not isinstance(raw_intent, Mapping):
            raise TypeError(f"Rust outcome planner returned malformed {key}")
        native_price = float(raw_intent["native_price"])
        canonical_yes_price = float(raw_intent["canonical_yes_price"])
        if not (
            0.0 <= native_price <= market.payout_unit
            and 0.0 <= canonical_yes_price <= market.payout_unit
        ):
            raise ValueError("Rust outcome planner returned a price outside the payout range")
        intents.append(
            OutcomeLiveOrderIntent(
                slot=key,
                outcome=OutcomeSide(str(raw_intent["outcome"])),
                side=OutcomeOrderSide(str(raw_intent["side"])),
                native_price=native_price,
                canonical_yes_price=canonical_yes_price,
                qty=float(raw_intent["qty"]),
            )
        )
    return OutcomeLivePlan(
        strategy_kind=str(output["strategy_kind"]),
        market_id=market.market_id,
        observation_start_ms=int(output["observation_start_ms"]),
        observation_end_ms=int(output["observation_end_ms"]),
        observation_count=int(output["observation_count"]),
        ema_fast=float(quotes["ema_fast"]),
        ema_slow=float(quotes["ema_slow"]),
        inventory_shift=float(quotes["inventory_shift"]),
        configured_estimated_fee_per_share=configured_fee_per_share,
        effective_estimated_fee_per_share=effective_fee_per_share,
        estimated_fee_source=(
            "configured"
            if configured_fee_per_share >= account_fee_floor
            else "hyperliquid_user_fees_conservative_maker_floor"
        ),
        intents=tuple(intents),
    )
