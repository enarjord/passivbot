from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage
from outcome.models import (
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeOrderSide,
    OutcomeSettlementEvidence,
    OutcomeSide,
    OutcomeVenue,
)


def _utc_ms() -> int:
    return int(time.time() * 1_000)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), hex(parsed)}:
        raise ValueError(f"{name} must be a non-negative base-10 integer")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    parsed = _non_negative_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _iso_time_ms(value: Any, name: str) -> int:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000)


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


def _ordered_sequence(block_number: int, transaction_index: int, log_index: int) -> str:
    return f"{block_number:020d}:{transaction_index:08d}:{log_index:08d}"


@dataclass(frozen=True)
class OutcomeHistoricalBatch:
    venue: OutcomeVenue
    market_id: str
    market: NormalizedOutcomeMarket
    source_cursor: str
    trades: tuple[NormalizedOutcomeTrade, ...]
    coverage_by_asset: Mapping[str, tuple[VerifiedCoverage, ...]]
    settlements: tuple[OutcomeSettlementEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.market_id.strip():
            raise ValueError("historical market ID must not be empty")
        if not self.source_cursor.strip():
            raise ValueError("historical source cursor must not be empty")
        if any(trade.venue is not self.venue for trade in self.trades):
            raise ValueError("historical batch contains a trade from another venue")
        if self.market.venue is not self.venue or self.market.market_id != self.market_id:
            raise ValueError("historical batch metadata belongs to another market")
        if any(trade.market_id != self.market_id for trade in self.trades):
            raise ValueError("historical batch contains a trade from another market")
        if any(settlement.venue is not self.venue for settlement in self.settlements):
            raise ValueError("historical batch contains settlement evidence from another venue")
        if any(settlement.market_id != self.market_id for settlement in self.settlements):
            raise ValueError("historical batch contains settlement evidence from another market")
        if any(not asset_id or not intervals for asset_id, intervals in self.coverage_by_asset.items()):
            raise ValueError("historical coverage requires non-empty asset IDs and intervals")


def archive_historical_batch(
    archive: OutcomeTradeArchive,
    batch: OutcomeHistoricalBatch,
    *,
    collector_session: str,
) -> tuple[int, int]:
    """Archive metadata, immutable evidence, and continuity as one atomic source batch."""

    if not collector_session.strip():
        raise ValueError("collector_session must not be empty")
    inserted = 0
    ignored = 0
    with archive.write_transaction():
        archive.append_market_metadata(
            batch.market,
            observed_at_ms=_utc_ms(),
            observation_source=batch.source_cursor,
            collector_session=collector_session,
        )
        for trade in batch.trades:
            if archive.append_trade(
                trade,
                collector_session=collector_session,
                source_cursor=batch.source_cursor,
            ):
                inserted += 1
            else:
                ignored += 1
        for settlement in batch.settlements:
            archive.append_settlement(
                settlement,
                collector_session=collector_session,
                source_cursor=batch.source_cursor,
            )
        for asset_id, intervals in batch.coverage_by_asset.items():
            for interval in intervals:
                archive.record_verified_coverage(
                    batch.venue,
                    batch.market_id,
                    asset_id,
                    interval,
                    collector_session=collector_session,
                )
    return inserted, ignored


def parse_hyperliquid_node_fills_by_block(
    lines: Iterable[str],
    market: NormalizedOutcomeMarket,
    *,
    source_cursor: str,
    received_time_ms: int | None = None,
) -> OutcomeHistoricalBatch:
    """Parse one complete official `node_fills_by_block` NDJSON object stream.

    The source writes every block in order. Consecutive block numbers are required before the
    interval between the first and last block is marked as verified. Each economic match appears
    once in the output even though the node file contains one account fill per participant.
    """

    if market.venue is not OutcomeVenue.HYPERLIQUID:
        raise ValueError("Hyperliquid historical fills require a Hyperliquid market")
    received_ms = _utc_ms() if received_time_ms is None else _non_negative_int(
        received_time_ms, "received_time_ms"
    )
    symbols = {
        identifier
        for asset in (market.yes_asset, market.no_asset)
        for identifier in (asset.asset_id, asset.market_data_symbol, asset.order_asset_id)
    }
    first_block_time_ms: int | None = None
    last_block_time_ms: int | None = None
    previous_block: int | None = None
    trades: list[NormalizedOutcomeTrade] = []
    settlement_payloads: list[Mapping[str, Any]] = []
    seen_economic_fills: set[tuple[str, int, str, str, str, int]] = set()
    previous_fill_time_ms: int | None = None
    previous_fill_position: tuple[int, int] | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            block = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Hyperliquid block JSON at line {line_number}") from exc
        if not isinstance(block, Mapping):
            raise ValueError(f"Hyperliquid block line {line_number} must be an object")
        block_number = _non_negative_int(block.get("block_number"), "block_number")
        if previous_block is not None and block_number != previous_block + 1:
            raise ValueError(
                "Hyperliquid node_fills_by_block source has a block gap or duplicate: "
                f"{previous_block} followed by {block_number}"
            )
        previous_block = block_number
        block_time_ms = _iso_time_ms(block.get("block_time"), "block_time")
        if first_block_time_ms is None:
            first_block_time_ms = block_time_ms
        last_block_time_ms = block_time_ms
        events = block.get("events")
        if not isinstance(events, list):
            raise ValueError("Hyperliquid block events must be an array")
        for event_index, event in enumerate(events):
            if (
                not isinstance(event, list)
                or len(event) != 2
                or not isinstance(event[1], Mapping)
            ):
                raise ValueError("Hyperliquid fill event must be [user_address, fill]")
            user_address = str(event[0]).strip()
            fill = event[1]
            coin = str(fill.get("coin", ""))
            if coin not in symbols:
                continue
            exchange_time_ms = _non_negative_int(fill.get("time"), "fill time")
            if (
                previous_fill_time_ms is not None
                and exchange_time_ms < previous_fill_time_ms
            ):
                assert previous_fill_position is not None
                raise ValueError(
                    "Hyperliquid node_fills_by_block fill times contradict block/event "
                    f"order: block {previous_fill_position[0]} event "
                    f"{previous_fill_position[1]} has {previous_fill_time_ms}, followed by "
                    f"block {block_number} event {event_index} with {exchange_time_ms}"
                )
            previous_fill_time_ms = exchange_time_ms
            previous_fill_position = (block_number, event_index)
            if str(fill.get("dir", "")).casefold() == "settlement":
                payload = dict(fill)
                payload["historical_user"] = user_address
                payload["historical_block_number"] = block_number
                payload["historical_event_index"] = event_index
                settlement_payloads.append(payload)
                continue
            tid = _non_negative_int(fill.get("tid"), "fill tid")
            identity = (
                coin,
                tid,
                str(fill.get("hash", "")),
                str(fill.get("px", "")),
                str(fill.get("sz", "")),
                exchange_time_ms,
            )
            if identity in seen_economic_fills:
                continue
            seen_economic_fills.add(identity)
            payload = dict(fill)
            payload["historical_user"] = user_address
            payload["historical_block_number"] = block_number
            payload["historical_event_index"] = event_index
            normalized = hyperliquid.normalize_trade(
                payload,
                market,
                received_time_ms=received_ms,
            )
            trades.append(
                replace(
                    normalized,
                    sequence_id=_ordered_sequence(block_number, 0, event_index),
                    raw_payload=payload,
                )
            )
    if first_block_time_ms is None or last_block_time_ms is None:
        raise ValueError("Hyperliquid historical source contains no blocks")

    trades.sort(
        key=lambda trade: (
            trade.exchange_time_ms,
            str(trade.sequence_id),
            trade.asset_id,
        )
    )
    # The supplied first block proves nothing about an unsupplied predecessor that may share
    # its wall-clock second. Start only after the first block's complete second; subsequent
    # consecutive block numbers then prove that every block in the retained interval was seen.
    start_ms = ((first_block_time_ms // 1_000) + 1) * 1_000
    end_ms = (last_block_time_ms // 1_000) * 1_000
    coverage = (
        (VerifiedCoverage(start_ms, end_ms),)
        if start_ms < end_ms
        else ()
    )
    settlement_fills = hyperliquid.normalize_account_fills(
        settlement_payloads,
        (market,),
    )
    settlements = hyperliquid.normalize_settlement_evidence(
        settlement_fills,
        market,
        received_time_ms=received_ms,
        evidence_source="hyperliquid_node_fills_by_block",
    )
    return OutcomeHistoricalBatch(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id=market.market_id,
        market=market,
        source_cursor=source_cursor,
        trades=tuple(trades),
        coverage_by_asset=(
            {
                market.yes_asset.asset_id: coverage,
                market.no_asset.asset_id: coverage,
            }
            if coverage
            else {}
        ),
        settlements=settlements,
    )


def _polymarket_event_args(event: Mapping[str, Any]) -> Mapping[str, Any]:
    args = event.get("args", event)
    if not isinstance(args, Mapping):
        raise ValueError("Polymarket decoded OrderFilled args must be an object")
    return args


def _polymarket_order_fill(
    event: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
    *,
    received_time_ms: int,
    collateral_scale: int,
    outcome_scale: int,
) -> NormalizedOutcomeTrade | None:
    args = _polymarket_event_args(event)
    contract_address = str(event.get("address", "")).casefold()
    taker = str(args.get("taker", "")).casefold()
    if event.get("is_taker_aggregate") is True or (
        contract_address and taker == contract_address
    ):
        return None

    if "tokenId" in args and "side" in args:
        token_id = str(args["tokenId"])
        side_value = _non_negative_int(args["side"], "OrderFilled.side")
        if side_value not in {0, 1}:
            raise ValueError("Polymarket v2 OrderFilled.side must be BUY=0 or SELL=1")
        side = OutcomeOrderSide.BUY if side_value == 0 else OutcomeOrderSide.SELL
        maker_amount = _positive_int(args.get("makerAmountFilled"), "makerAmountFilled")
        taker_amount = _positive_int(args.get("takerAmountFilled"), "takerAmountFilled")
        collateral_amount = maker_amount if side is OutcomeOrderSide.BUY else taker_amount
        outcome_amount = taker_amount if side is OutcomeOrderSide.BUY else maker_amount
        schema = "ctf_exchange_v2"
    elif "makerAssetId" in args and "takerAssetId" in args:
        maker_asset_id = str(args["makerAssetId"])
        taker_asset_id = str(args["takerAssetId"])
        maker_amount = _positive_int(args.get("makerAmountFilled"), "makerAmountFilled")
        taker_amount = _positive_int(args.get("takerAmountFilled"), "takerAmountFilled")
        if maker_asset_id == "0" and taker_asset_id != "0":
            token_id = taker_asset_id
            side = OutcomeOrderSide.BUY
            collateral_amount = maker_amount
            outcome_amount = taker_amount
        elif taker_asset_id == "0" and maker_asset_id != "0":
            token_id = maker_asset_id
            side = OutcomeOrderSide.SELL
            collateral_amount = taker_amount
            outcome_amount = maker_amount
        else:
            raise ValueError(
                "Polymarket v1 maker OrderFilled must exchange one outcome token with collateral"
            )
        schema = "ctf_exchange_v1"
    else:
        raise ValueError("unsupported decoded Polymarket OrderFilled schema")

    asset = market.asset_for_id(token_id)
    price = (collateral_amount / collateral_scale) / (outcome_amount / outcome_scale)
    qty = outcome_amount / outcome_scale
    if not 0.0 <= price <= market.payout_unit or not math.isfinite(qty) or qty <= 0.0:
        raise ValueError("decoded Polymarket OrderFilled has invalid price or quantity")
    block_number = _non_negative_int(event.get("blockNumber"), "blockNumber")
    transaction_index = _non_negative_int(event.get("transactionIndex"), "transactionIndex")
    log_index = _non_negative_int(event.get("logIndex"), "logIndex")
    timestamp_ms = _non_negative_int(event.get("blockTimeMs"), "blockTimeMs")
    transaction_hash = str(event.get("transactionHash", "")).strip()
    if not transaction_hash:
        raise ValueError("decoded Polymarket OrderFilled requires transactionHash")
    source_event_id = f"{transaction_hash}:{log_index}"
    canonical_price = price if asset.side is OutcomeSide.YES else market.payout_unit - price
    payload = dict(event)
    payload["historical_schema"] = schema
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        asset_id=asset.asset_id,
        outcome=asset.side,
        native_side=side,
        native_price=price,
        canonical_yes_price=canonical_price,
        qty=qty,
        exchange_time_ms=timestamp_ms,
        received_time_ms=received_time_ms,
        source_event_id=source_event_id,
        economic_event_id=source_event_id,
        sequence_id=_ordered_sequence(block_number, transaction_index, log_index),
        raw_payload=payload,
    )


def parse_polymarket_order_filled_logs(
    events: Iterable[Mapping[str, Any]],
    market: NormalizedOutcomeMarket,
    *,
    source_cursor: str,
    coverage: VerifiedCoverage,
    collateral_decimals: int = 6,
    outcome_decimals: int = 6,
    received_time_ms: int | None = None,
) -> OutcomeHistoricalBatch:
    """Normalize a complete, decoded Polygon `eth_getLogs` range for CTF Exchange v1 or v2.

    The RPC/indexer layer must supply block time and prove the requested range was complete before
    passing `coverage`. Logs are sorted by the canonical `(block, transaction, log)` order.
    """

    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("Polymarket historical logs require a Polymarket market")
    if not 0 <= collateral_decimals <= 30 or not 0 <= outcome_decimals <= 30:
        raise ValueError("token decimals must be between 0 and 30")
    received_ms = _utc_ms() if received_time_ms is None else _non_negative_int(
        received_time_ms, "received_time_ms"
    )
    collateral_scale = 10**collateral_decimals
    outcome_scale = 10**outcome_decimals
    ordered_events = sorted(
        events,
        key=lambda event: (
            _non_negative_int(event.get("blockNumber"), "blockNumber"),
            _non_negative_int(event.get("transactionIndex"), "transactionIndex"),
            _non_negative_int(event.get("logIndex"), "logIndex"),
        ),
    )
    seen_positions: set[tuple[int, int, int]] = set()
    trades = []
    for event in ordered_events:
        position = (
            _non_negative_int(event.get("blockNumber"), "blockNumber"),
            _non_negative_int(event.get("transactionIndex"), "transactionIndex"),
            _non_negative_int(event.get("logIndex"), "logIndex"),
        )
        if position in seen_positions:
            raise ValueError(f"duplicate Polymarket log position {position}")
        seen_positions.add(position)
        trade = _polymarket_order_fill(
            event,
            market,
            received_time_ms=received_ms,
            collateral_scale=collateral_scale,
            outcome_scale=outcome_scale,
        )
        if trade is not None:
            if not coverage.start_ms <= trade.exchange_time_ms < coverage.end_ms:
                raise ValueError(
                    "decoded Polymarket OrderFilled timestamp falls outside proven coverage"
                )
            trades.append(trade)
    return OutcomeHistoricalBatch(
        venue=OutcomeVenue.POLYMARKET,
        market_id=market.market_id,
        market=market,
        source_cursor=source_cursor,
        trades=tuple(trades),
        coverage_by_asset={
            market.yes_asset.asset_id: (coverage,),
            market.no_asset.asset_id: (coverage,),
        },
    )
