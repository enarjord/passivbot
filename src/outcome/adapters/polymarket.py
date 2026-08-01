from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Mapping

from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeAsset,
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeBookLevel,
    OutcomeBookSnapshot,
    OutcomeCapabilities,
    OutcomeFeeMetadata,
    OutcomeOrderSide,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
    OutcomeSettlementEvidence,
    OutcomeSide,
    OutcomeVenue,
)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _json_pair(value: Any, name: str) -> list[Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON array") from exc
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return parsed


def _iso_time_ms(value: Any, name: str) -> int | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000)


def _fee_metadata(payload: Mapping[str, Any]) -> OutcomeFeeMetadata:
    enabled = payload.get("feesEnabled")
    if enabled is False:
        return OutcomeFeeMetadata(formula="venue_reported_zero", maker_rate=0.0, taker_rate=0.0)
    if enabled is not True:
        return OutcomeFeeMetadata(formula="authoritative_fill_or_venue_schedule_required")
    schedule = payload.get("feeSchedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("feesEnabled Polymarket market is missing feeSchedule")
    required = {"exponent", "rate", "takerOnly", "rebateRate"}
    if not required.issubset(schedule):
        raise ValueError("Polymarket feeSchedule is incomplete")
    exponent = _finite_float(schedule["exponent"], "feeSchedule.exponent")
    rate = _finite_float(schedule["rate"], "feeSchedule.rate")
    rebate_rate = _finite_float(schedule["rebateRate"], "feeSchedule.rebateRate")
    if exponent < 0.0 or rate < 0.0 or rebate_rate < 0.0:
        raise ValueError("Polymarket feeSchedule values must be non-negative")
    if not isinstance(schedule["takerOnly"], bool):
        raise ValueError("feeSchedule.takerOnly must be boolean")
    return OutcomeFeeMetadata(
        formula="polymarket_probability_curve",
        parameters={
            "exponent": exponent,
            "rate": rate,
            "taker_only": schedule["takerOnly"],
            "rebate_rate": rebate_rate,
            "fee_type": str(payload.get("feeType", "")),
        },
    )


def normalize_market(
    payload: Mapping[str, Any],
    *,
    quote_asset: str | None = None,
) -> NormalizedOutcomeMarket:
    if payload.get("negRisk") is not False:
        raise ValueError("Polymarket negative-risk markets are outside the binary foundation scope")
    if payload.get("enableOrderBook") is not True:
        raise ValueError("Polymarket market is not CLOB-enabled")

    labels = [str(value).strip() for value in _json_pair(payload.get("outcomes"), "outcomes")]
    token_ids = [str(value).strip() for value in _json_pair(payload.get("clobTokenIds"), "clobTokenIds")]
    if not all(labels) or not all(token_ids) or len(set(token_ids)) != 2:
        raise ValueError("Polymarket outcomes and token IDs must be non-empty and distinct")

    lower_labels = [label.casefold() for label in labels]
    if set(lower_labels) == {"yes", "no"}:
        yes_index = lower_labels.index("yes")
        no_index = lower_labels.index("no")
        canonical_reference_is_semantic_yes = True
    else:
        yes_index, no_index = 0, 1
        canonical_reference_is_semantic_yes = False

    market_id = str(payload.get("conditionId", "")).strip()
    if not market_id:
        raise ValueError("Polymarket conditionId must not be empty")
    yes_asset = NormalizedOutcomeAsset(
        side=OutcomeSide.YES,
        label=labels[yes_index],
        asset_id=token_ids[yes_index],
        market_data_symbol=token_ids[yes_index],
        order_asset_id=token_ids[yes_index],
    )
    no_asset = NormalizedOutcomeAsset(
        side=OutcomeSide.NO,
        label=labels[no_index],
        asset_id=token_ids[no_index],
        market_data_symbol=token_ids[no_index],
        order_asset_id=token_ids[no_index],
    )
    price_tick = _finite_float(payload.get("orderPriceMinTickSize"), "orderPriceMinTickSize")
    min_order_qty = _finite_float(payload.get("orderMinSize"), "orderMinSize")
    if price_tick <= 0.0 or min_order_qty <= 0.0:
        raise ValueError("Polymarket order constraints must be positive")
    accepting_orders = payload.get("acceptingOrders")
    if accepting_orders is not None and not isinstance(accepting_orders, bool):
        raise ValueError("Polymarket acceptingOrders must be boolean when present")
    reported_quote_asset = payload.get("denominationToken")
    if reported_quote_asset in (None, ""):
        reported_quote_asset = payload.get("collateralToken")
    normalized_quote_asset = str(
        reported_quote_asset if reported_quote_asset not in (None, "") else quote_asset or ""
    ).strip()
    if not normalized_quote_asset:
        raise ValueError(
            "Polymarket collateral identity is absent; supply quote_asset explicitly"
        )

    return NormalizedOutcomeMarket(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market_id,
        title=str(payload.get("question", "")).strip(),
        description=str(payload.get("description", "")),
        quote_asset=normalized_quote_asset,
        yes_asset=yes_asset,
        no_asset=no_asset,
        payout_unit=1.0,
        price_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=price_tick),
        qty_step=None,
        min_order_qty=min_order_qty,
        min_order_notional=None,
        lifecycle=MarketLifecycle(
            discovery_time_ms=_iso_time_ms(payload.get("createdAt"), "createdAt"),
            trading_open_time_ms=_iso_time_ms(payload.get("startDate"), "startDate"),
            order_acceptance_time_ms=_iso_time_ms(
                payload.get("acceptingOrdersTimestamp"), "acceptingOrdersTimestamp"
            ),
            trading_close_time_ms=_iso_time_ms(payload.get("closedTime"), "closedTime"),
            scheduled_event_time_ms=_iso_time_ms(payload.get("endDate"), "endDate"),
            accepting_orders=accepting_orders,
        ),
        capabilities=OutcomeCapabilities(
            complementary_books_merged=False,
            sell_requires_inventory=True,
            supports_split=True,
            supports_merge=True,
            supports_redeem=True,
            supports_post_only=True,
            supports_gtd=True,
        ),
        fee_metadata=_fee_metadata(payload),
        native_metadata={
            "gamma_market_id": str(payload.get("id", "")),
            "slug": str(payload.get("slug", "")),
            "question_id": str(payload.get("questionID", "")),
            "resolution_source": str(payload.get("resolutionSource", "")),
            "restricted": payload.get("restricted"),
            "active": payload.get("active"),
            "closed": payload.get("closed"),
            "yes_outcome_index": yes_index,
            "no_outcome_index": no_index,
            "canonical_reference_is_semantic_yes": canonical_reference_is_semantic_yes,
            "denomination_token": payload.get("denominationToken"),
            "collateral_token": payload.get("collateralToken"),
            "uma_resolution_status": payload.get("umaResolutionStatus"),
            "resolved_by": payload.get("resolvedBy"),
        },
    )


def normalize_condition_resolution(
    event: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
) -> OutcomeSettlementEvidence:
    """Normalize one on-chain CTF ConditionResolution event.

    Gamma's terminal prices are useful discovery metadata, but the CTF payout vector is the
    authoritative redemption contract. The vector is indexed by the venue's original outcomes
    array, which can differ from canonical YES/NO ordering.
    """

    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("market is not a Polymarket outcome market")
    args = event.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("Polymarket ConditionResolution args must be an object")
    if str(args.get("conditionId", "")).casefold() != market.market_id.casefold():
        raise ValueError("Polymarket ConditionResolution conditionId does not match market")
    outcome_slot_count = int(args.get("outcomeSlotCount", -1))
    payouts = args.get("payoutNumerators")
    if outcome_slot_count != 2 or not isinstance(payouts, list) or len(payouts) != 2:
        raise ValueError("Polymarket binary ConditionResolution requires two payout numerators")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in payouts):
        raise ValueError("Polymarket payout numerators must be non-negative integers")
    denominator = sum(payouts)
    if denominator <= 0:
        raise ValueError("Polymarket payout denominator must be positive")
    yes_index = market.native_metadata.get("yes_outcome_index")
    no_index = market.native_metadata.get("no_outcome_index")
    if (
        isinstance(yes_index, bool)
        or not isinstance(yes_index, int)
        or isinstance(no_index, bool)
        or not isinstance(no_index, int)
        or {yes_index, no_index} != {0, 1}
    ):
        raise ValueError("Polymarket market metadata lacks canonical outcome indices")
    settlement_time_ms = int(event.get("blockTimeMs", -1))
    if settlement_time_ms < 0:
        raise ValueError("Polymarket ConditionResolution requires a non-negative blockTimeMs")
    transaction_hash = str(event.get("transactionHash", "")).strip()
    log_index = event.get("logIndex")
    if (
        not transaction_hash.startswith("0x")
        or isinstance(log_index, bool)
        or not isinstance(log_index, int)
        or log_index < 0
    ):
        raise ValueError("Polymarket ConditionResolution requires transaction and log identity")
    return OutcomeSettlementEvidence(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        yes_fraction=payouts[yes_index] / denominator,
        payout_unit=market.payout_unit,
        settlement_time_ms=settlement_time_ms,
        capital_release_time_ms=None,
        received_time_ms=int(received_time_ms),
        source_event_id=f"{transaction_hash}:{log_index}",
        evidence_source="polymarket_ctf_condition_resolution",
        observed_yes_qty=0.0,
        observed_no_qty=0.0,
        collateral_payout=0.0,
        fee=0.0,
        fee_asset=market.quote_asset,
        raw_payload=dict(event),
    )


def normalize_public_trade(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
) -> NormalizedOutcomeTrade:
    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("market is not a Polymarket outcome market")
    if str(payload.get("conditionId", "")) != market.market_id:
        raise ValueError("Polymarket trade conditionId does not match market")
    asset = market.asset_for_id(str(payload.get("asset", "")))
    side_raw = str(payload.get("side", "")).upper()
    if side_raw not in {"BUY", "SELL"}:
        raise ValueError("Polymarket trade side must be BUY or SELL")
    native_side = OutcomeOrderSide.BUY if side_raw == "BUY" else OutcomeOrderSide.SELL
    native_price = _finite_float(payload.get("price"), "price")
    qty = _finite_float(payload.get("size"), "size")
    if not 0.0 <= native_price <= 1.0 or qty <= 0.0:
        raise ValueError("invalid Polymarket trade price or quantity")
    timestamp_seconds = _finite_float(payload.get("timestamp"), "timestamp")
    if timestamp_seconds < 0.0 or not timestamp_seconds.is_integer():
        raise ValueError("Polymarket public trade timestamp must be integer Unix seconds")
    canonical_price = native_price if asset.side is OutcomeSide.YES else 1.0 - native_price
    outcome_label = str(payload.get("outcome", ""))
    if outcome_label and outcome_label != asset.label:
        raise ValueError("Polymarket trade outcome label does not match asset")
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=asset.side,
        native_side=native_side,
        native_price=native_price,
        canonical_yes_price=canonical_price,
        qty=qty,
        exchange_time_ms=int(timestamp_seconds * 1_000),
        received_time_ms=int(received_time_ms),
        # The public Data API exposes a transaction hash but no log index or unique trade ID.
        # Treat it as source metadata, not as a safe deduplication key.
        source_event_id=None,
        raw_payload=dict(payload),
    )


def normalize_market_ws_trade(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
    collector_sequence: int | None = None,
) -> NormalizedOutcomeTrade:
    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("market is not a Polymarket outcome market")
    if payload.get("event_type") != "last_trade_price":
        raise ValueError("Polymarket market websocket payload is not a trade event")
    if str(payload.get("market", "")) != market.market_id:
        raise ValueError("Polymarket websocket trade market does not match")
    asset = market.asset_for_id(str(payload.get("asset_id", "")))
    side_raw = str(payload.get("side", "")).upper()
    if side_raw not in {"BUY", "SELL"}:
        raise ValueError("Polymarket trade side must be BUY or SELL")
    native_side = OutcomeOrderSide.BUY if side_raw == "BUY" else OutcomeOrderSide.SELL
    native_price = _finite_float(payload.get("price"), "price")
    qty = _finite_float(payload.get("size"), "size")
    if not 0.0 <= native_price <= 1.0 or qty <= 0.0:
        raise ValueError("invalid Polymarket trade price or quantity")
    timestamp_ms = _finite_float(payload.get("timestamp"), "timestamp")
    if timestamp_ms < 0.0 or not timestamp_ms.is_integer():
        raise ValueError("Polymarket websocket trade timestamp must be integer milliseconds")
    canonical_price = native_price if asset.side is OutcomeSide.YES else 1.0 - native_price
    explicit_id = payload.get("id")
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=asset.side,
        native_side=native_side,
        native_price=native_price,
        canonical_yes_price=canonical_price,
        qty=qty,
        exchange_time_ms=int(timestamp_ms),
        received_time_ms=int(received_time_ms),
        # Current public market-channel examples have a transaction hash but no log index.
        # Only use a source identity if the venue actually supplies an event ID.
        source_event_id=str(explicit_id) if explicit_id not in (None, "") else None,
        collector_sequence=collector_sequence,
        raw_payload=dict(payload),
    )


def normalize_market_ws_price_grid_change(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
) -> OutcomePriceGridChange:
    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("market is not a Polymarket outcome market")
    if payload.get("event_type") != "tick_size_change":
        raise ValueError("Polymarket websocket payload is not a tick-size change")
    if str(payload.get("market", "")) != market.market_id:
        raise ValueError("Polymarket tick-size change market does not match")
    asset_id = str(payload.get("asset_id", ""))
    market.asset_for_id(asset_id)
    timestamp_ms = _finite_float(payload.get("timestamp"), "timestamp")
    if timestamp_ms < 0.0 or not timestamp_ms.is_integer():
        raise ValueError("Polymarket tick-size timestamp must be integer milliseconds")
    old_step = _finite_float(payload.get("old_tick_size"), "old_tick_size")
    new_step = _finite_float(payload.get("new_tick_size"), "new_tick_size")
    return OutcomePriceGridChange(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        timestamp_ms=int(timestamp_ms),
        received_time_ms=int(received_time_ms),
        old_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=old_step),
        new_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=new_step),
        raw_payload=dict(payload),
    )


def normalize_market_ws_book(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
) -> OutcomeBookSnapshot:
    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("market is not a Polymarket outcome market")
    if payload.get("event_type") != "book":
        raise ValueError("Polymarket market websocket payload is not a book event")
    if str(payload.get("market", "")) != market.market_id:
        raise ValueError("Polymarket websocket book market does not match")
    asset = market.asset_for_id(str(payload.get("asset_id", "")))
    timestamp_ms = _finite_float(payload.get("timestamp"), "timestamp")
    if timestamp_ms < 0.0 or not timestamp_ms.is_integer():
        raise ValueError("Polymarket websocket book timestamp must be integer milliseconds")

    def normalize_side(name: str, *, reverse: bool) -> tuple[OutcomeBookLevel, ...]:
        raw_levels = payload.get(name)
        if not isinstance(raw_levels, list):
            raise ValueError(f"Polymarket websocket book {name} must be an array")
        levels = []
        for raw_level in raw_levels:
            if not isinstance(raw_level, Mapping):
                raise ValueError("Polymarket websocket book level must be an object")
            levels.append(
                OutcomeBookLevel(
                    native_price=_finite_float(raw_level.get("price"), "book price"),
                    qty=_finite_float(raw_level.get("size"), "book size"),
                    order_count=None,
                )
            )
        return tuple(
            sorted(
                levels,
                key=lambda level: level.native_price,
                reverse=reverse,
            )
        )

    return OutcomeBookSnapshot(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=asset.side,
        timestamp_ms=int(timestamp_ms),
        received_time_ms=int(received_time_ms),
        bids=normalize_side("bids", reverse=True),
        asks=normalize_side("asks", reverse=False),
        raw_payload=dict(payload),
    )
