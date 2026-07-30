from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeAsset,
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeAccountFill,
    OutcomeBookLevel,
    OutcomeBookSnapshot,
    OutcomeCapabilities,
    OutcomeCollateralBalance,
    OutcomeFeeMetadata,
    OutcomeOpenOrder,
    OutcomeOrderSide,
    OutcomePriceGridMetadata,
    OutcomeSide,
    OutcomeSettlementEvidence,
    OutcomeTokenBalance,
    OutcomeVenue,
)


_DESCRIPTION_KEYS = {"class", "underlying", "expiry", "targetPrice", "period"}
_PERIOD_RE = re.compile(r"^[1-9][0-9]*[smhd]$")
_PERIOD_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
}


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


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(parsed) != str(value).strip() or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def parse_price_binary_description(description: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for segment in str(description).split("|"):
        key, separator, value = segment.partition(":")
        if not separator or not key or not value or key in fields:
            raise ValueError("invalid HIP-4 priceBinary description")
        fields[key] = value
    if set(fields) != _DESCRIPTION_KEYS or fields["class"] != "priceBinary":
        raise ValueError("unsupported HIP-4 outcome description")
    underlying = fields["underlying"].strip()
    if not underlying or any(ch.isspace() for ch in underlying):
        raise ValueError("HIP-4 underlying must be a non-empty symbol")
    target_price = _finite_float(fields["targetPrice"], "targetPrice")
    if target_price <= 0.0:
        raise ValueError("targetPrice must be positive")
    if not _PERIOD_RE.fullmatch(fields["period"]):
        raise ValueError("unsupported HIP-4 period")
    try:
        expiry = datetime.strptime(fields["expiry"], "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid HIP-4 expiry") from exc
    expiry_time_ms = int(expiry.timestamp() * 1_000)
    period_ms = int(fields["period"][:-1]) * _PERIOD_UNIT_MS[fields["period"][-1]]
    return {
        "underlying": underlying,
        "target_price": target_price,
        "expiry_time_ms": expiry_time_ms,
        "trading_open_time_ms": expiry_time_ms - period_ms,
        "period": fields["period"],
    }


def normalize_market(payload: Mapping[str, Any]) -> NormalizedOutcomeMarket:
    outcome_id = _non_negative_int(payload.get("outcome"), "outcome")
    side_specs = payload.get("sideSpecs")
    if not isinstance(side_specs, list) or side_specs != [{"name": "Yes"}, {"name": "No"}]:
        raise ValueError("HIP-4 binary outcome must expose ordered Yes and No sideSpecs")
    parsed = parse_price_binary_description(str(payload.get("description", "")))
    quote_token = str(payload.get("quoteToken", "")).strip()
    if not quote_token:
        raise ValueError("HIP-4 quoteToken must not be empty")

    yes_encoding = 10 * outcome_id
    no_encoding = yes_encoding + 1
    yes_asset = NormalizedOutcomeAsset(
        side=OutcomeSide.YES,
        label="Yes",
        asset_id=f"+{yes_encoding}",
        market_data_symbol=f"#{yes_encoding}",
        order_asset_id=str(100_000_000 + yes_encoding),
    )
    no_asset = NormalizedOutcomeAsset(
        side=OutcomeSide.NO,
        label="No",
        asset_id=f"+{no_encoding}",
        market_data_symbol=f"#{no_encoding}",
        order_asset_id=str(100_000_000 + no_encoding),
    )
    target_text = str(payload["description"]).split("targetPrice:", 1)[1].split("|", 1)[0]
    title = (
        f"{parsed['underlying']} above {target_text} at "
        f"{datetime.fromtimestamp(parsed['expiry_time_ms'] / 1_000, tz=timezone.utc).isoformat()}"
    )
    return NormalizedOutcomeMarket(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id=str(outcome_id),
        title=title,
        description=str(payload["description"]),
        quote_asset=quote_token,
        yes_asset=yes_asset,
        no_asset=no_asset,
        payout_unit=1.0,
        price_grid=OutcomePriceGridMetadata(
            kind="significant_figures",
            max_significant_figures=5,
            max_decimal_places=8,
        ),
        # outcomeMeta does not currently expose side-token size precision or
        # market-specific order minima.  Keep them unavailable so live planning
        # and order construction fail closed instead of treating generic spot
        # assumptions as authoritative HIP-4 constraints.
        qty_step=None,
        min_order_qty=None,
        min_order_notional=None,
        lifecycle=MarketLifecycle(
            trading_open_time_ms=parsed["trading_open_time_ms"],
            trading_close_time_ms=parsed["expiry_time_ms"],
            scheduled_event_time_ms=parsed["expiry_time_ms"],
        ),
        capabilities=OutcomeCapabilities(
            complementary_books_merged=True,
            sell_requires_inventory=True,
            supports_split=False,
            supports_merge=False,
            supports_redeem=False,
            supports_post_only=True,
            supports_gtd=False,
        ),
        fee_metadata=OutcomeFeeMetadata(
            formula="authoritative_fill_or_venue_schedule_required"
        ),
        native_metadata={
            "class": "priceBinary",
            "underlying": parsed["underlying"],
            "target_price": parsed["target_price"],
            "period": parsed["period"],
            "outcome_name": str(payload.get("name", "")),
        },
    )


def normalize_trade(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
    collector_sequence: int | None = None,
) -> NormalizedOutcomeTrade:
    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("market is not a Hyperliquid outcome market")
    coin = str(payload.get("coin", ""))
    asset = market.asset_for_id(coin)
    side_raw = str(payload.get("side", "")).upper()
    if side_raw not in {"B", "A"}:
        raise ValueError("Hyperliquid trade side must be B or A")
    native_side = OutcomeOrderSide.BUY if side_raw == "B" else OutcomeOrderSide.SELL
    native_price = _finite_float(payload.get("px"), "px")
    qty = _finite_float(payload.get("sz"), "sz")
    if not 0.0 <= native_price <= 1.0 or qty <= 0.0:
        raise ValueError("invalid Hyperliquid outcome trade price or quantity")
    exchange_time_ms = _non_negative_int(payload.get("time"), "time")
    tid = _non_negative_int(payload.get("tid"), "tid")
    canonical_price = native_price if asset.side is OutcomeSide.YES else 1.0 - native_price
    transaction_hash = str(payload.get("hash", "")).strip()
    users = payload.get("users")
    users_key = ""
    if isinstance(users, list) and len(users) == 2:
        users_key = ":".join(sorted(str(user) for user in users))
    economic_event_id = (
        f"{exchange_time_ms}:{transaction_hash}:{canonical_price:.12g}:{qty:.12g}:{users_key}"
        if transaction_hash
        else None
    )
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=asset.side,
        native_side=native_side,
        native_price=native_price,
        canonical_yes_price=canonical_price,
        qty=qty,
        exchange_time_ms=exchange_time_ms,
        received_time_ms=int(received_time_ms),
        source_event_id=f"{exchange_time_ms}:{coin}:{tid}",
        economic_event_id=economic_event_id,
        collector_sequence=collector_sequence,
        raw_payload=dict(payload),
    )


def _asset_lookup(
    markets: Iterable[NormalizedOutcomeMarket],
) -> dict[str, tuple[NormalizedOutcomeMarket, NormalizedOutcomeAsset]]:
    lookup: dict[str, tuple[NormalizedOutcomeMarket, NormalizedOutcomeAsset]] = {}
    for market in markets:
        if market.venue is not OutcomeVenue.HYPERLIQUID:
            raise ValueError("Hyperliquid outcome state requires Hyperliquid markets")
        for asset in (market.yes_asset, market.no_asset):
            for identifier in (
                asset.asset_id,
                asset.market_data_symbol,
                asset.order_asset_id,
            ):
                if identifier in lookup:
                    raise ValueError(f"duplicate Hyperliquid outcome asset identifier {identifier!r}")
                lookup[identifier] = (market, asset)
    return lookup


def normalize_collateral_balance(
    payload: Mapping[str, Any],
    *,
    quote_asset: str,
) -> OutcomeCollateralBalance:
    balances = payload.get("balances")
    if not isinstance(balances, list):
        raise ValueError("Hyperliquid spotClearinghouseState balances must be an array")
    matching = [
        row
        for row in balances
        if isinstance(row, Mapping) and str(row.get("coin", "")) == quote_asset
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Hyperliquid spotClearinghouseState must contain one {quote_asset} balance"
        )
    row = matching[0]
    total = _finite_float(row.get("total"), "collateral total")
    held = _finite_float(row.get("hold"), "collateral hold")
    if total < 0.0 or held < 0.0:
        raise ValueError("Hyperliquid collateral balance must be non-negative")

    available_after_maintenance = None
    token = row.get("token")
    maintenance = payload.get("tokenToAvailableAfterMaintenance")
    if token is not None:
        if not isinstance(maintenance, list):
            raise ValueError(
                "Hyperliquid spotClearinghouseState maintenance availability must be an array"
            )
        token_matches = [
            item
            for item in maintenance
            if isinstance(item, list) and len(item) == 2 and str(item[0]) == str(token)
        ]
        if len(token_matches) > 1:
            raise ValueError("duplicate Hyperliquid maintenance availability token")
        if token_matches:
            available_after_maintenance = _finite_float(
                token_matches[0][1],
                "available after maintenance",
            )
    return OutcomeCollateralBalance(
        asset=quote_asset,
        total=total,
        held=held,
        available_after_maintenance=available_after_maintenance,
    )


def normalize_token_balances(
    payload: Mapping[str, Any],
    markets: Sequence[NormalizedOutcomeMarket],
) -> tuple[OutcomeTokenBalance, ...]:
    """Normalize an authoritative state, explicitly materializing absent selected tokens as zero."""

    balances = payload.get("balances")
    if not isinstance(balances, list):
        raise ValueError("Hyperliquid spotClearinghouseState balances must be an array")
    lookup = _asset_lookup(markets)
    seen: set[str] = set()
    normalized: list[OutcomeTokenBalance] = []
    for row in balances:
        if not isinstance(row, Mapping):
            raise ValueError("Hyperliquid spot balance row must be an object")
        coin = str(row.get("coin", ""))
        resolved = lookup.get(coin)
        if resolved is None:
            continue
        market, asset = resolved
        if asset.asset_id in seen:
            raise ValueError(f"duplicate Hyperliquid outcome balance for {asset.asset_id}")
        seen.add(asset.asset_id)
        total = _finite_float(row.get("total"), "outcome balance total")
        held = _finite_float(row.get("hold"), "outcome balance hold")
        entry_notional = _finite_float(row.get("entryNtl"), "outcome entryNtl")
        if total < 0.0 or held < 0.0 or entry_notional < 0.0:
            raise ValueError("Hyperliquid outcome balances must be non-negative")
        normalized.append(
            OutcomeTokenBalance(
                market_id=market.market_id,
                asset_id=asset.asset_id,
                outcome=asset.side,
                total_qty=total,
                held_qty=held,
                entry_notional=entry_notional,
            )
        )
    for market in markets:
        for asset in (market.yes_asset, market.no_asset):
            if asset.asset_id not in seen:
                normalized.append(
                    OutcomeTokenBalance(
                        market_id=market.market_id,
                        asset_id=asset.asset_id,
                        outcome=asset.side,
                        total_qty=0.0,
                        held_qty=0.0,
                        entry_notional=0.0,
                    )
                )
    return tuple(sorted(normalized, key=lambda item: (int(item.market_id), item.outcome.value)))


def normalize_open_orders(
    payload: Sequence[Mapping[str, Any]],
    markets: Sequence[NormalizedOutcomeMarket],
) -> tuple[OutcomeOpenOrder, ...]:
    lookup = _asset_lookup(markets)
    normalized = []
    seen: set[str] = set()
    for row in payload:
        coin = str(row.get("coin", ""))
        resolved = lookup.get(coin)
        if resolved is None:
            continue
        market, asset = resolved
        order_id = str(_non_negative_int(row.get("oid"), "oid"))
        if order_id in seen:
            raise ValueError(f"duplicate Hyperliquid outcome order ID {order_id}")
        seen.add(order_id)
        side_raw = str(row.get("side", "")).upper()
        if side_raw not in {"B", "A"}:
            raise ValueError("Hyperliquid open-order side must be B or A")
        qty = _finite_float(row.get("sz"), "open-order sz")
        original_qty = _finite_float(row.get("origSz"), "open-order origSz")
        normalized.append(
            OutcomeOpenOrder(
                market_id=market.market_id,
                order_id=order_id,
                asset_id=asset.asset_id,
                outcome=asset.side,
                side=(
                    OutcomeOrderSide.BUY
                    if side_raw == "B"
                    else OutcomeOrderSide.SELL
                ),
                native_price=_finite_float(row.get("limitPx"), "open-order limitPx"),
                qty=qty,
                original_qty=original_qty,
                timestamp_ms=_non_negative_int(row.get("timestamp"), "timestamp"),
                client_order_id=(
                    str(row["cloid"]) if row.get("cloid") not in (None, "") else None
                ),
            )
        )
    return tuple(sorted(normalized, key=lambda item: (item.timestamp_ms, item.order_id)))


def normalize_account_fills(
    payload: Sequence[Mapping[str, Any]],
    markets: Sequence[NormalizedOutcomeMarket],
) -> tuple[OutcomeAccountFill, ...]:
    lookup = _asset_lookup(markets)
    normalized = []
    seen: set[tuple[int, str, str]] = set()
    for row in payload:
        coin = str(row.get("coin", ""))
        resolved = lookup.get(coin)
        if resolved is None:
            continue
        market, asset = resolved
        timestamp_ms = _non_negative_int(row.get("time"), "fill time")
        trade_id = str(_non_negative_int(row.get("tid"), "fill tid"))
        transaction_hash = str(row.get("hash", "")).strip()
        identity = (timestamp_ms, coin, trade_id)
        if identity in seen:
            continue
        seen.add(identity)
        side_raw = str(row.get("side", "")).upper()
        if side_raw not in {"B", "A"}:
            raise ValueError("Hyperliquid account-fill side must be B or A")
        crossed = row.get("crossed")
        if not isinstance(crossed, bool):
            raise ValueError("Hyperliquid account-fill crossed must be boolean")
        normalized.append(
            OutcomeAccountFill(
                market_id=market.market_id,
                trade_id=trade_id,
                transaction_hash=transaction_hash,
                order_id=str(_non_negative_int(row.get("oid"), "fill oid")),
                asset_id=asset.asset_id,
                outcome=asset.side,
                side=(
                    OutcomeOrderSide.BUY
                    if side_raw == "B"
                    else OutcomeOrderSide.SELL
                ),
                native_price=_finite_float(row.get("px"), "fill px"),
                qty=_finite_float(row.get("sz"), "fill sz"),
                fee=_finite_float(row.get("fee"), "fill fee"),
                fee_asset=str(row.get("feeToken", "")).strip(),
                is_maker=not crossed,
                timestamp_ms=timestamp_ms,
                direction=str(row.get("dir", "")).strip(),
                start_position_qty=_finite_float(
                    row.get("startPosition"),
                    "fill startPosition",
                ),
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda item: (item.timestamp_ms, item.asset_id, int(item.trade_id)),
        )
    )


def normalize_settlement_evidence(
    fills: Sequence[OutcomeAccountFill],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
    evidence_source: str,
) -> tuple[OutcomeSettlementEvidence, ...]:
    """Collapse authoritative HIP-4 Settlement fills into market-level payout evidence."""

    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("HIP-4 settlement evidence requires a Hyperliquid market")
    if received_time_ms < 0:
        raise ValueError("HIP-4 settlement receive time must be non-negative")
    if not evidence_source.strip():
        raise ValueError("HIP-4 settlement evidence source must not be empty")
    selected = [
        fill
        for fill in fills
        if fill.market_id == market.market_id and fill.is_settlement
    ]
    grouped: dict[tuple[int, str], list[OutcomeAccountFill]] = {}
    for fill in selected:
        grouped.setdefault((fill.timestamp_ms, fill.transaction_hash), []).append(fill)

    evidence = []
    for (timestamp_ms, transaction_hash), group in sorted(grouped.items()):
        fractions = []
        observed_yes_qty = 0.0
        observed_no_qty = 0.0
        collateral_payout = 0.0
        fee = 0.0
        fee_assets = set()
        raw_rows = []
        for fill in group:
            if fill.side is not OutcomeOrderSide.SELL:
                raise ValueError("HIP-4 Settlement fill must sell settled inventory")
            if fill.start_position_qty <= 0.0 or not math.isclose(
                fill.qty,
                fill.start_position_qty,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("HIP-4 Settlement fill must consume the full starting inventory")
            native_fraction = fill.native_price / market.payout_unit
            if fill.outcome is OutcomeSide.YES:
                fractions.append(native_fraction)
                observed_yes_qty += fill.qty
            else:
                fractions.append(1.0 - native_fraction)
                observed_no_qty += fill.qty
            collateral_payout += fill.qty * fill.native_price
            fee += fill.fee
            fee_assets.add(fill.fee_asset)
            raw_rows.append(
                {
                    "asset_id": fill.asset_id,
                    "outcome": fill.outcome.value,
                    "native_price": fill.native_price,
                    "qty": fill.qty,
                    "fee": fill.fee,
                    "fee_asset": fill.fee_asset,
                    "trade_id": fill.trade_id,
                    "order_id": fill.order_id,
                }
            )
        yes_fraction = fractions[0]
        if any(
            not math.isclose(
                fraction,
                yes_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for fraction in fractions[1:]
        ):
            raise ValueError("HIP-4 Settlement fills disagree on the resolved outcome")
        if not (
            math.isclose(yes_fraction, 0.0, abs_tol=1e-12)
            or math.isclose(yes_fraction, 1.0, abs_tol=1e-12)
        ):
            raise ValueError("HIP-4 priceBinary settlement must resolve to 0 or 1")
        if len(fee_assets) != 1:
            raise ValueError("HIP-4 Settlement fills disagree on fee asset")
        evidence.append(
            OutcomeSettlementEvidence(
                venue=OutcomeVenue.HYPERLIQUID,
                market_id=market.market_id,
                yes_fraction=0.0 if yes_fraction < 0.5 else 1.0,
                payout_unit=market.payout_unit,
                settlement_time_ms=timestamp_ms,
                capital_release_time_ms=timestamp_ms,
                received_time_ms=received_time_ms,
                source_event_id=f"{market.market_id}:{transaction_hash}:{timestamp_ms}",
                evidence_source=evidence_source,
                observed_yes_qty=observed_yes_qty,
                observed_no_qty=observed_no_qty,
                collateral_payout=collateral_payout,
                fee=fee,
                fee_asset=next(iter(fee_assets)),
                raw_payload={
                    "transaction_hash": transaction_hash,
                    "fills": raw_rows,
                },
            )
        )
    return tuple(evidence)


def normalize_l2_book(
    payload: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    outcome: OutcomeSide,
    received_time_ms: int,
) -> OutcomeBookSnapshot:
    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("HIP-4 book normalization requires a Hyperliquid market")
    asset = market.yes_asset if outcome is OutcomeSide.YES else market.no_asset
    if str(payload.get("coin", "")) != asset.market_data_symbol:
        raise ValueError("Hyperliquid outcome book returned the wrong coin")
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) != 2:
        raise ValueError("Hyperliquid outcome book levels must contain bids and asks")

    normalized_sides: list[tuple[OutcomeBookLevel, ...]] = []
    for raw_side in levels:
        if not isinstance(raw_side, list):
            raise ValueError("Hyperliquid outcome book side must be an array")
        normalized = []
        for raw_level in raw_side:
            if not isinstance(raw_level, Mapping):
                raise ValueError("Hyperliquid outcome book level must be an object")
            normalized.append(
                OutcomeBookLevel(
                    native_price=_finite_float(raw_level.get("px"), "book px"),
                    qty=_finite_float(raw_level.get("sz"), "book sz"),
                    order_count=_non_negative_int(raw_level.get("n"), "book n"),
                )
            )
        normalized_sides.append(tuple(normalized))
    return OutcomeBookSnapshot(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=outcome,
        timestamp_ms=_non_negative_int(payload.get("time"), "book time"),
        received_time_ms=_non_negative_int(received_time_ms, "book received time"),
        bids=normalized_sides[0],
        asks=normalized_sides[1],
        raw_payload=dict(payload),
    )


def _decimal_text(value: Any, name: str) -> str:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be decimal-compatible") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _is_step_aligned(value: Decimal, step: Decimal) -> bool:
    if step <= 0:
        return False
    return value % step == 0


def _validate_native_order_values(
    market: NormalizedOutcomeMarket,
    *,
    native_price: Any,
    qty: Any,
    allow_below_min_qty: bool = False,
) -> tuple[str, str]:
    price_text = _decimal_text(native_price, "native_price")
    qty_text = _decimal_text(qty, "qty")
    price = Decimal(price_text)
    quantity = Decimal(qty_text)
    if not Decimal("0") < price < Decimal(str(market.payout_unit)):
        raise ValueError("HIP-4 order price must be strictly between zero and payout")
    if quantity <= 0:
        raise ValueError("HIP-4 order quantity must be positive")
    if market.qty_step is None:
        raise ValueError("HIP-4 quantity step is unavailable")
    if not _is_step_aligned(quantity, Decimal(str(market.qty_step))):
        raise ValueError("HIP-4 order quantity is not step-aligned")
    if market.min_order_qty is None:
        raise ValueError("HIP-4 minimum order quantity is unavailable")
    if not allow_below_min_qty and quantity < Decimal(str(market.min_order_qty)):
        raise ValueError("HIP-4 order quantity is below the minimum")
    if market.min_order_notional is None:
        raise ValueError("HIP-4 minimum order notional is unavailable")
    if price * quantity < Decimal(str(market.min_order_notional)):
        raise ValueError("HIP-4 order notional is below the minimum")

    grid = market.price_grid
    if grid.kind == "fixed_step":
        assert grid.fixed_step is not None
        if not _is_step_aligned(price, Decimal(str(grid.fixed_step))):
            raise ValueError("HIP-4 order price is not step-aligned")
    elif grid.kind == "significant_figures":
        assert grid.max_significant_figures is not None
        assert grid.max_decimal_places is not None
        normalized = price.normalize()
        significant_digits = len(normalized.as_tuple().digits)
        decimal_places = max(0, -normalized.as_tuple().exponent)
        if significant_digits > grid.max_significant_figures:
            raise ValueError("HIP-4 order price has too many significant figures")
        if decimal_places > grid.max_decimal_places:
            raise ValueError("HIP-4 order price has too many decimal places")
    else:  # pragma: no cover - model validation already rejects this
        raise ValueError(f"unsupported HIP-4 price grid {grid.kind!r}")
    return price_text, qty_text


def build_limit_order_action(
    market: NormalizedOutcomeMarket,
    *,
    outcome: OutcomeSide,
    side: OutcomeOrderSide,
    native_price: Any,
    qty: Any,
    client_order_id: str | None = None,
    time_in_force: str = "Alo",
    allow_below_min_qty: bool = False,
) -> dict[str, Any]:
    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("HIP-4 order action requires a Hyperliquid market")
    if time_in_force not in {"Alo", "Gtc", "Ioc"}:
        raise ValueError("HIP-4 limit time-in-force must be Alo, Gtc, or Ioc")
    price_text, qty_text = _validate_native_order_values(
        market,
        native_price=native_price,
        qty=qty,
        allow_below_min_qty=allow_below_min_qty,
    )
    asset = market.yes_asset if outcome is OutcomeSide.YES else market.no_asset
    order = {
        "a": int(asset.order_asset_id),
        "b": side is OutcomeOrderSide.BUY,
        "p": price_text,
        "s": qty_text,
        "r": False,
        "t": {"limit": {"tif": time_in_force}},
    }
    if client_order_id is not None:
        if not re.fullmatch(r"0x[0-9a-fA-F]{32}", client_order_id):
            raise ValueError("Hyperliquid client order ID must be a 16-byte hex string")
        order["c"] = client_order_id.lower()
    return {"type": "order", "orders": [order], "grouping": "na"}


def build_cancel_action(
    market: NormalizedOutcomeMarket,
    *,
    outcome: OutcomeSide,
    order_id: int,
) -> dict[str, Any]:
    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("HIP-4 cancel action requires a Hyperliquid market")
    if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id < 0:
        raise ValueError("Hyperliquid order ID must be a non-negative integer")
    asset = market.yes_asset if outcome is OutcomeSide.YES else market.no_asset
    return {
        "type": "cancel",
        "cancels": [{"a": int(asset.order_asset_id), "o": order_id}],
    }
