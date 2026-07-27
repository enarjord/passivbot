from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import time
from typing import Any

import aiohttp

from outcome.candles import VerifiedCoverage
from outcome.historical import OutcomeHistoricalBatch, parse_polymarket_order_filled_logs
from outcome.adapters import polymarket
from outcome.models import NormalizedOutcomeMarket, OutcomeVenue


POLYGON_PUBLIC_RPC_URL = "https://polygon.drpc.org"

POLYMARKET_CTF_EXCHANGE_V1 = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
POLYMARKET_CTF_EXCHANGE_V2 = "0xe111180000d2663c0091e4f400237545b87b996b"
POLYMARKET_CONDITIONAL_TOKENS = "0x4d97dcd97ec945f40cf65f87097ace5ea0476045"

# keccak256 of the event signatures published by the official v1 and v2 contracts.
POLYMARKET_ORDER_FILLED_V1_TOPIC = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)
POLYMARKET_ORDER_FILLED_V2_TOPIC = (
    "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
)
POLYMARKET_CONDITION_RESOLUTION_TOPIC = (
    "0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894"
)

RpcCall = Callable[[str, Sequence[Any]], Awaitable[Any]]


class PolygonRpcError(RuntimeError):
    pass


class PolygonRpcRangeError(PolygonRpcError):
    """A provider rejected only the requested log range size, so splitting is safe."""


def _utc_ms() -> int:
    return int(time.time() * 1_000)


def _classify_rpc_error(message: str, *, http_status: int | None = None) -> PolygonRpcError:
    lowered = message.casefold()
    range_markers = (
        "block range",
        "range limit",
        "query exceeds",
        "response size",
        "too many results",
        "more than ",
    )
    if http_status == 413 or any(marker in lowered for marker in range_markers):
        return PolygonRpcRangeError(message)
    return PolygonRpcError(message)


def _hex_int(value: Any, name: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{name} must be a 0x-prefixed hexadecimal string")
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _hex_bytes(value: Any, name: str, *, length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{name} must be a 0x-prefixed hexadecimal string")
    payload = value[2:]
    if len(payload) % 2 != 0:
        raise ValueError(f"{name} must contain whole bytes")
    try:
        decoded = bytes.fromhex(payload)
    except ValueError as exc:
        raise ValueError(f"{name} contains invalid hexadecimal data") from exc
    if length is not None and len(decoded) != length:
        raise ValueError(f"{name} must contain exactly {length} bytes")
    return decoded


def _word_uint(word: bytes) -> int:
    return int.from_bytes(word, "big", signed=False)


def _topic_address(topic: Any, name: str) -> str:
    raw = _hex_bytes(topic, name, length=32)
    if any(raw[:12]):
        raise ValueError(f"{name} is not a canonical indexed address")
    return f"0x{raw[12:].hex()}"


def _topic_bytes32(topic: Any, name: str) -> str:
    return f"0x{_hex_bytes(topic, name, length=32).hex()}"


def _data_words(data: Any, expected: int) -> list[bytes]:
    raw = _hex_bytes(data, "log data", length=expected * 32)
    return [raw[index : index + 32] for index in range(0, len(raw), 32)]


def decode_polymarket_order_filled_log(
    raw_log: Mapping[str, Any],
    *,
    block_time_ms: int,
) -> dict[str, Any]:
    """Decode one official standard-market CTF Exchange v1 or v2 OrderFilled log."""

    topics = raw_log.get("topics")
    if not isinstance(topics, Sequence) or isinstance(topics, (str, bytes)):
        raise ValueError("Polygon log topics must be an array")
    if len(topics) != 4:
        raise ValueError("Polymarket OrderFilled log must contain four topics")
    topic0 = str(topics[0]).casefold()
    address = str(raw_log.get("address", "")).casefold()
    expected_address_by_topic = {
        POLYMARKET_ORDER_FILLED_V1_TOPIC: POLYMARKET_CTF_EXCHANGE_V1,
        POLYMARKET_ORDER_FILLED_V2_TOPIC: POLYMARKET_CTF_EXCHANGE_V2,
    }
    expected_address = expected_address_by_topic.get(topic0)
    if expected_address is None:
        raise ValueError("unsupported Polymarket OrderFilled topic")
    if address != expected_address:
        raise ValueError(
            "Polymarket OrderFilled log address does not match its event version"
        )
    common = {
        "address": address,
        "blockNumber": _hex_int(raw_log.get("blockNumber"), "blockNumber"),
        "transactionIndex": _hex_int(
            raw_log.get("transactionIndex"), "transactionIndex"
        ),
        "logIndex": _hex_int(raw_log.get("logIndex"), "logIndex"),
        "blockTimeMs": block_time_ms,
        "transactionHash": str(raw_log.get("transactionHash", "")),
        "removed": raw_log.get("removed", False),
    }
    if common["removed"] is True:
        raise ValueError("removed Polygon logs cannot be archived as verified history")
    if not common["transactionHash"].startswith("0x"):
        raise ValueError("Polygon log requires a transaction hash")
    indexed = {
        "orderHash": _topic_bytes32(topics[1], "OrderFilled.orderHash"),
        "maker": _topic_address(topics[2], "OrderFilled.maker"),
        "taker": _topic_address(topics[3], "OrderFilled.taker"),
    }

    if topic0 == POLYMARKET_ORDER_FILLED_V1_TOPIC:
        words = _data_words(raw_log.get("data"), 5)
        args = {
            **indexed,
            "makerAssetId": _word_uint(words[0]),
            "takerAssetId": _word_uint(words[1]),
            "makerAmountFilled": _word_uint(words[2]),
            "takerAmountFilled": _word_uint(words[3]),
            "fee": _word_uint(words[4]),
        }
    elif topic0 == POLYMARKET_ORDER_FILLED_V2_TOPIC:
        words = _data_words(raw_log.get("data"), 7)
        side = _word_uint(words[0])
        if side not in {0, 1}:
            raise ValueError("Polymarket v2 OrderFilled.side must be BUY=0 or SELL=1")
        args = {
            **indexed,
            "side": side,
            "tokenId": _word_uint(words[1]),
            "makerAmountFilled": _word_uint(words[2]),
            "takerAmountFilled": _word_uint(words[3]),
            "fee": _word_uint(words[4]),
            "builder": f"0x{words[5].hex()}",
            "metadata": f"0x{words[6].hex()}",
        }
    return {**common, "args": args, "raw_log": dict(raw_log)}


def decode_polymarket_condition_resolution_log(
    raw_log: Mapping[str, Any],
    *,
    block_time_ms: int,
) -> dict[str, Any]:
    """Decode the official CTF ConditionResolution event for a binary condition."""

    topics = raw_log.get("topics")
    if not isinstance(topics, Sequence) or isinstance(topics, (str, bytes)):
        raise ValueError("Polygon log topics must be an array")
    if len(topics) != 4:
        raise ValueError("Polymarket ConditionResolution log must contain four topics")
    if str(topics[0]).casefold() != POLYMARKET_CONDITION_RESOLUTION_TOPIC:
        raise ValueError("unsupported Polymarket ConditionResolution topic")
    if str(raw_log.get("address", "")).casefold() != POLYMARKET_CONDITIONAL_TOKENS:
        raise ValueError(
            "Polymarket ConditionResolution log address is not Conditional Tokens"
        )
    raw_data = _hex_bytes(raw_log.get("data"), "ConditionResolution data")
    if len(raw_data) != 5 * 32:
        raise ValueError("binary ConditionResolution data must contain five ABI words")
    words = [raw_data[index : index + 32] for index in range(0, len(raw_data), 32)]
    outcome_slot_count = _word_uint(words[0])
    payout_offset = _word_uint(words[1])
    payout_length = _word_uint(words[2])
    if outcome_slot_count != 2 or payout_offset != 64 or payout_length != 2:
        raise ValueError("ConditionResolution is not a canonical binary payout vector")
    common = {
        "address": str(raw_log.get("address", "")).casefold(),
        "blockNumber": _hex_int(raw_log.get("blockNumber"), "blockNumber"),
        "transactionIndex": _hex_int(
            raw_log.get("transactionIndex"), "transactionIndex"
        ),
        "logIndex": _hex_int(raw_log.get("logIndex"), "logIndex"),
        "blockTimeMs": block_time_ms,
        "transactionHash": str(raw_log.get("transactionHash", "")),
        "removed": raw_log.get("removed", False),
    }
    if common["address"] != POLYMARKET_CONDITIONAL_TOKENS:
        raise ValueError("ConditionResolution came from an unexpected contract")
    if common["removed"] is True:
        raise ValueError("removed Polygon logs cannot be archived as verified history")
    if not common["transactionHash"].startswith("0x"):
        raise ValueError("Polygon log requires a transaction hash")
    args = {
        "conditionId": _topic_bytes32(topics[1], "ConditionResolution.conditionId"),
        "oracle": _topic_address(topics[2], "ConditionResolution.oracle"),
        "questionId": _topic_bytes32(topics[3], "ConditionResolution.questionId"),
        "outcomeSlotCount": outcome_slot_count,
        "payoutNumerators": [_word_uint(words[3]), _word_uint(words[4])],
    }
    if sum(args["payoutNumerators"]) <= 0:
        raise ValueError("ConditionResolution payout denominator must be positive")
    return {**common, "args": args, "raw_log": dict(raw_log)}


def _event_references_market(
    event: Mapping[str, Any],
    market: NormalizedOutcomeMarket,
) -> bool:
    args = event["args"]
    token_ids = {market.yes_asset.asset_id, market.no_asset.asset_id}
    if "tokenId" in args:
        return str(args["tokenId"]) in token_ids
    return (
        str(args["makerAssetId"]) in token_ids
        or str(args["takerAssetId"]) in token_ids
    )


class PolygonJsonRpc:
    """Small fail-closed JSON-RPC client for complete Polygon log ranges."""

    def __init__(
        self,
        rpc_url: str = POLYGON_PUBLIC_RPC_URL,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        call: RpcCall | None = None,
    ):
        if not rpc_url.strip():
            raise ValueError("Polygon RPC URL must not be empty")
        if timeout_seconds <= 0.0:
            raise ValueError("Polygon RPC timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("Polygon RPC max_attempts must be positive")
        self.rpc_url = rpc_url
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._injected_call = call
        self._next_id = 1
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> PolygonJsonRpc:
        if self._injected_call is None and self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            headers = {"User-Agent": "passivbot-outcome-archive/1"}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def call(self, method: str, params: Sequence[Any]) -> Any:
        if self._injected_call is not None:
            return await self._injected_call(method, params)
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            request_id = self._next_id
            self._next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": list(params),
            }
            try:
                if self._session is None:
                    await self.__aenter__()
                if self._session is None:
                    raise PolygonRpcError("Polygon RPC session was not initialized")
                async with self._session.post(self.rpc_url, json=payload) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise _classify_rpc_error(
                            f"Polygon RPC HTTP {response.status}: {body[:300]}",
                            http_status=response.status,
                        )
                decoded = json.loads(body)
                if not isinstance(decoded, Mapping) or decoded.get("id") != request_id:
                    raise PolygonRpcError("Polygon RPC returned a malformed response")
                if "error" in decoded:
                    raise _classify_rpc_error(f"Polygon RPC error: {decoded['error']}")
                if "result" not in decoded:
                    raise PolygonRpcError("Polygon RPC response has no result")
                return decoded["result"]
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
                PolygonRpcError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    await asyncio.sleep(0.25 * (2**attempt))
        if isinstance(last_error, PolygonRpcError):
            raise last_error
        raise PolygonRpcError(
            f"Polygon RPC {method} failed after {self.max_attempts} attempts"
        ) from last_error

    async def head_block_number(self) -> int:
        return _hex_int(await self.call("eth_blockNumber", []), "head block number")

    async def block_timestamp_seconds(self, block_number: int) -> int:
        if block_number < 0:
            raise ValueError("block number must be non-negative")
        block = await self.call("eth_getBlockByNumber", [hex(block_number), False])
        if not isinstance(block, Mapping):
            raise PolygonRpcError(f"Polygon block {block_number} was not found")
        actual_number = _hex_int(block.get("number"), "block.number")
        if actual_number != block_number:
            raise PolygonRpcError(
                f"Polygon RPC returned block {actual_number} for requested block {block_number}"
            )
        return _hex_int(block.get("timestamp"), "block.timestamp")

    async def first_block_at_or_after(
        self,
        timestamp_seconds: int,
        *,
        head_block: int | None = None,
    ) -> int:
        if timestamp_seconds < 0:
            raise ValueError("timestamp must be non-negative")
        head = await self.head_block_number() if head_block is None else head_block
        if head < 0:
            raise ValueError("head block must be non-negative")
        if await self.block_timestamp_seconds(head) < timestamp_seconds:
            raise PolygonRpcError(
                "requested historical end is newer than the current finalized RPC head"
            )
        low = 0
        high = head
        while low < high:
            middle = (low + high) // 2
            if await self.block_timestamp_seconds(middle) < timestamp_seconds:
                low = middle + 1
            else:
                high = middle
        return low

    async def _logs_chunk(
        self,
        *,
        address: str,
        topic: str,
        indexed_topics: Sequence[str | None] = (),
        from_block: int,
        to_block: int,
    ) -> list[Mapping[str, Any]]:
        result = await self.call(
            "eth_getLogs",
            [
                {
                    "address": address,
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "topics": [topic, *indexed_topics],
                }
            ],
        )
        if not isinstance(result, list) or any(not isinstance(log, Mapping) for log in result):
            raise PolygonRpcError("Polygon eth_getLogs result must be an array of objects")
        return result

    async def complete_logs(
        self,
        *,
        address: str,
        topic: str,
        indexed_topics: Sequence[str | None] = (),
        from_block: int,
        to_block: int,
        max_block_span: int = 2_000,
    ) -> list[Mapping[str, Any]]:
        if from_block < 0 or to_block < from_block:
            raise ValueError("invalid Polygon log block range")
        if max_block_span <= 0:
            raise ValueError("max_block_span must be positive")

        async def fetch_adaptive(start: int, end: int) -> list[Mapping[str, Any]]:
            try:
                return await self._logs_chunk(
                    address=address,
                    topic=topic,
                    indexed_topics=indexed_topics,
                    from_block=start,
                    to_block=end,
                )
            except PolygonRpcRangeError:
                if start == end:
                    raise
                middle = (start + end) // 2
                left = await fetch_adaptive(start, middle)
                right = await fetch_adaptive(middle + 1, end)
                return [*left, *right]

        logs: list[Mapping[str, Any]] = []
        cursor = from_block
        while cursor <= to_block:
            chunk_end = min(to_block, cursor + max_block_span - 1)
            logs.extend(await fetch_adaptive(cursor, chunk_end))
            cursor = chunk_end + 1
        return logs


@dataclass(frozen=True)
class PolymarketPolygonDownload:
    batch: OutcomeHistoricalBatch
    from_block: int
    to_block: int
    confirmed_head_block: int
    decoded_log_count: int
    market_log_count: int
    resolution_log_count: int


async def download_polymarket_order_filled_range(
    market: NormalizedOutcomeMarket,
    *,
    start_ms: int,
    end_ms: int,
    rpc: PolygonJsonRpc | None = None,
    received_time_ms: int | None = None,
    max_block_span: int = 2_000,
    confirmation_blocks: int = 128,
) -> PolymarketPolygonDownload:
    """Download a complete standard-market OrderFilled range and grant exact time coverage."""

    if market.venue is not OutcomeVenue.POLYMARKET:
        raise ValueError("Polygon OrderFilled download requires a Polymarket market")
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("historical coverage requires start_ms < end_ms")
    if start_ms % 1_000 != 0 or end_ms % 1_000 != 0:
        raise ValueError("historical coverage bounds must align to whole seconds")
    if (
        isinstance(confirmation_blocks, bool)
        or not isinstance(confirmation_blocks, int)
        or confirmation_blocks < 0
    ):
        raise ValueError("confirmation_blocks must be a non-negative integer")
    if rpc is None:
        async with PolygonJsonRpc() as owned_rpc:
            return await download_polymarket_order_filled_range(
                market,
                start_ms=start_ms,
                end_ms=end_ms,
                rpc=owned_rpc,
                received_time_ms=received_time_ms,
                max_block_span=max_block_span,
                confirmation_blocks=confirmation_blocks,
            )
    client = rpc
    live_head = await client.head_block_number()
    if live_head < confirmation_blocks:
        raise PolygonRpcError("Polygon head is lower than the required confirmation depth")
    confirmed_head = live_head - confirmation_blocks
    from_block = await client.first_block_at_or_after(
        start_ms // 1_000,
        head_block=confirmed_head,
    )
    end_block = await client.first_block_at_or_after(
        end_ms // 1_000,
        head_block=confirmed_head,
    )
    to_block = end_block - 1

    raw_logs: list[Mapping[str, Any]] = []
    raw_resolutions: list[Mapping[str, Any]] = []
    if to_block >= from_block:
        for address, topic in (
            (POLYMARKET_CTF_EXCHANGE_V1, POLYMARKET_ORDER_FILLED_V1_TOPIC),
            (POLYMARKET_CTF_EXCHANGE_V2, POLYMARKET_ORDER_FILLED_V2_TOPIC),
        ):
            raw_logs.extend(
                await client.complete_logs(
                    address=address,
                    topic=topic,
                    from_block=from_block,
                    to_block=to_block,
                    max_block_span=max_block_span,
                )
            )
        raw_resolutions = await client.complete_logs(
            address=POLYMARKET_CONDITIONAL_TOKENS,
            topic=POLYMARKET_CONDITION_RESOLUTION_TOPIC,
            indexed_topics=(market.market_id,),
            from_block=from_block,
            to_block=to_block,
            max_block_span=max_block_span,
        )

    positions: set[tuple[int, int, int]] = set()
    block_numbers: set[int] = set()
    for raw_log in [*raw_logs, *raw_resolutions]:
        position = (
            _hex_int(raw_log.get("blockNumber"), "blockNumber"),
            _hex_int(raw_log.get("transactionIndex"), "transactionIndex"),
            _hex_int(raw_log.get("logIndex"), "logIndex"),
        )
        if not from_block <= position[0] <= to_block:
            raise PolygonRpcError("Polygon RPC returned a log outside the requested block range")
        if position in positions:
            raise PolygonRpcError(f"Polygon RPC returned duplicate log position {position}")
        positions.add(position)
        block_numbers.add(position[0])

    block_times = {
        block_number: await client.block_timestamp_seconds(block_number)
        for block_number in sorted(block_numbers)
    }
    decoded = [
        decode_polymarket_order_filled_log(
            raw_log,
            block_time_ms=block_times[
                _hex_int(raw_log.get("blockNumber"), "blockNumber")
            ]
            * 1_000,
        )
        for raw_log in raw_logs
    ]
    decoded.sort(
        key=lambda event: (
            event["blockNumber"],
            event["transactionIndex"],
            event["logIndex"],
        )
    )
    market_events = [event for event in decoded if _event_references_market(event, market)]
    decoded_resolutions = [
        decode_polymarket_condition_resolution_log(
            raw_log,
            block_time_ms=block_times[
                _hex_int(raw_log.get("blockNumber"), "blockNumber")
            ]
            * 1_000,
        )
        for raw_log in raw_resolutions
    ]
    if len(decoded_resolutions) > 1:
        raise PolygonRpcError("Polygon RPC returned multiple resolutions for one condition")
    coverage = VerifiedCoverage(start_ms, end_ms)
    received_ms = _utc_ms() if received_time_ms is None else received_time_ms
    source_cursor = (
        f"polygon:{from_block}:{to_block}:"
        f"{POLYMARKET_CTF_EXCHANGE_V1},{POLYMARKET_CTF_EXCHANGE_V2},"
        f"{POLYMARKET_CONDITIONAL_TOKENS}"
    )
    batch = parse_polymarket_order_filled_logs(
        market_events,
        market,
        source_cursor=source_cursor,
        coverage=coverage,
        received_time_ms=received_ms,
    )
    batch = replace(
        batch,
        settlements=tuple(
            polymarket.normalize_condition_resolution(
                event,
                market,
                received_time_ms=received_ms,
            )
            for event in decoded_resolutions
        ),
    )
    return PolymarketPolygonDownload(
        batch=batch,
        from_block=from_block,
        to_block=to_block,
        confirmed_head_block=confirmed_head,
        decoded_log_count=len(decoded),
        market_log_count=len(market_events),
        resolution_log_count=len(decoded_resolutions),
    )
