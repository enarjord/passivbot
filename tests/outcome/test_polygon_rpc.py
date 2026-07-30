from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from outcome.adapters import polymarket
from outcome.models import OutcomeOrderSide, OutcomeSide
from outcome.polygon_rpc import (
    POLYMARKET_CONDITION_RESOLUTION_TOPIC,
    POLYMARKET_CONDITIONAL_TOKENS,
    POLYMARKET_CTF_EXCHANGE_V1,
    POLYMARKET_CTF_EXCHANGE_V2,
    POLYMARKET_ORDER_FILLED_V1_TOPIC,
    POLYMARKET_ORDER_FILLED_V2_TOPIC,
    PolygonJsonRpc,
    PolygonRpcError,
    PolygonRpcRangeError,
    decode_polymarket_condition_resolution_log,
    decode_polymarket_order_filled_log,
    download_polymarket_order_filled_range,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def word(value: int) -> bytes:
    return value.to_bytes(32, "big")


def topic_address(value: str) -> str:
    return f"0x{bytes.fromhex(value.removeprefix('0x')).rjust(32, b'\x00').hex()}"


def raw_log(
    *,
    address: str,
    topic0: str,
    data_words: list[bytes],
    block: int = 10,
    transaction_index: int = 2,
    log_index: int = 3,
    taker: str = "0x2222222222222222222222222222222222222222",
) -> dict[str, Any]:
    return {
        "address": address,
        "topics": [
            topic0,
            f"0x{'11' * 32}",
            topic_address("0x1111111111111111111111111111111111111111"),
            topic_address(taker),
        ],
        "data": f"0x{b''.join(data_words).hex()}",
        "blockNumber": hex(block),
        "transactionIndex": hex(transaction_index),
        "logIndex": hex(log_index),
        "transactionHash": f"0x{'33' * 32}",
        "removed": False,
    }


def raw_resolution_log(
    condition_id: str,
    *,
    payouts: tuple[int, int],
    block: int = 11,
    transaction_index: int = 3,
    log_index: int = 8,
) -> dict[str, Any]:
    return {
        "address": POLYMARKET_CONDITIONAL_TOKENS,
        "topics": [
            POLYMARKET_CONDITION_RESOLUTION_TOPIC,
            condition_id,
            topic_address("0x3333333333333333333333333333333333333333"),
            f"0x{'44' * 32}",
        ],
        "data": f"0x{b''.join([word(2), word(64), word(2), word(payouts[0]), word(payouts[1])]).hex()}",
        "blockNumber": hex(block),
        "transactionIndex": hex(transaction_index),
        "logIndex": hex(log_index),
        "transactionHash": f"0x{'55' * 32}",
        "removed": False,
    }


def test_decodes_official_v1_and_v2_order_filled_layouts():
    v1 = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V1,
        topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        data_words=[word(0), word(99), word(335_000), word(1_000_000), word(10)],
    )
    v2 = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V2,
        topic0=POLYMARKET_ORDER_FILLED_V2_TOPIC,
        data_words=[
            word(1),
            word(99),
            word(1_000_000),
            word(335_000),
            word(10),
            bytes.fromhex("44" * 32),
            bytes.fromhex("55" * 32),
        ],
    )

    decoded_v1 = decode_polymarket_order_filled_log(v1, block_time_ms=10_000)
    decoded_v2 = decode_polymarket_order_filled_log(v2, block_time_ms=10_000)

    assert decoded_v1["args"]["makerAssetId"] == 0
    assert decoded_v1["args"]["takerAssetId"] == 99
    assert decoded_v1["args"]["makerAmountFilled"] == 335_000
    assert decoded_v2["args"]["side"] == 1
    assert decoded_v2["args"]["tokenId"] == 99
    assert decoded_v2["args"]["builder"] == f"0x{'44' * 32}"
    assert decoded_v2["args"]["metadata"] == f"0x{'55' * 32}"


def test_decodes_authoritative_binary_condition_resolution():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    decoded = decode_polymarket_condition_resolution_log(
        raw_resolution_log(market.market_id, payouts=(1, 1)),
        block_time_ms=12_000,
    )

    assert decoded["args"]["conditionId"] == market.market_id
    assert decoded["args"]["outcomeSlotCount"] == 2
    assert decoded["args"]["payoutNumerators"] == [1, 1]
    assert decoded["blockTimeMs"] == 12_000


def test_rejects_malformed_or_zero_condition_resolution_vector():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    zero = raw_resolution_log(market.market_id, payouts=(0, 0))
    with pytest.raises(ValueError, match="denominator"):
        decode_polymarket_condition_resolution_log(zero, block_time_ms=12_000)

    malformed = raw_resolution_log(market.market_id, payouts=(1, 0))
    malformed["data"] = f"0x{b''.join([word(2), word(96), word(2), word(1), word(0)]).hex()}"
    with pytest.raises(ValueError, match="canonical binary"):
        decode_polymarket_condition_resolution_log(malformed, block_time_ms=12_000)


def test_removed_or_noncanonical_logs_are_rejected():
    removed = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V1,
        topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        data_words=[word(0), word(99), word(335_000), word(1_000_000), word(0)],
    )
    removed["removed"] = True
    with pytest.raises(ValueError, match="removed"):
        decode_polymarket_order_filled_log(removed, block_time_ms=10_000)

    bad_address = dict(removed, removed=False)
    bad_address["topics"] = list(removed["topics"])
    bad_address["topics"][2] = f"0x01{'00' * 31}"
    with pytest.raises(ValueError, match="canonical indexed address"):
        decode_polymarket_order_filled_log(bad_address, block_time_ms=10_000)

    wrong_contract = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V2,
        topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        data_words=[word(0), word(99), word(335_000), word(1_000_000), word(0)],
    )
    with pytest.raises(ValueError, match="address does not match"):
        decode_polymarket_order_filled_log(wrong_contract, block_time_ms=10_000)


def test_logs_require_explicit_boolean_non_removed_evidence():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    fill = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V1,
        topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        data_words=[word(0), word(99), word(335_000), word(1_000_000), word(0)],
    )
    resolution = raw_resolution_log(market.market_id, payouts=(1, 1))

    for decoder, raw in (
        (decode_polymarket_order_filled_log, fill),
        (decode_polymarket_condition_resolution_log, resolution),
    ):
        missing = dict(raw)
        missing.pop("removed")
        with pytest.raises(ValueError, match="explicitly boolean false"):
            decoder(missing, block_time_ms=10_000)
        for invalid in (True, 0, "false", None):
            with pytest.raises(ValueError, match="explicitly boolean false"):
                decoder(dict(raw, removed=invalid), block_time_ms=10_000)


def test_logs_require_canonical_transaction_hashes():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    fill = raw_log(
        address=POLYMARKET_CTF_EXCHANGE_V1,
        topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        data_words=[word(0), word(99), word(335_000), word(1_000_000), word(0)],
    )
    resolution = raw_resolution_log(market.market_id, payouts=(1, 1))

    for decoder, raw in (
        (decode_polymarket_order_filled_log, fill),
        (decode_polymarket_condition_resolution_log, resolution),
    ):
        for invalid_hash in (
            "0xdead",
            f"0x{'gg' * 32}",
            f"0x{'11' * 31}",
            f"0x{'11' * 33}",
            f"{'11' * 32}",
        ):
            with pytest.raises(ValueError, match="transactionHash"):
                decoder(
                    dict(raw, transactionHash=invalid_hash),
                    block_time_ms=10_000,
                )


@pytest.mark.asyncio
async def test_download_proves_exact_range_and_filters_other_markets():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    yes_id = int(market.yes_asset.asset_id)
    no_id = int(market.no_asset.asset_id)
    logs = {
        (POLYMARKET_CTF_EXCHANGE_V1, 10, 11): [
            raw_log(
                address=POLYMARKET_CTF_EXCHANGE_V1,
                topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
                data_words=[
                    word(0),
                    word(yes_id),
                    word(335_000),
                    word(1_000_000),
                    word(0),
                ],
                block=10,
                log_index=3,
            ),
            raw_log(
                address=POLYMARKET_CTF_EXCHANGE_V1,
                topic0=POLYMARKET_ORDER_FILLED_V1_TOPIC,
                data_words=[word(0), word(123), word(500_000), word(1_000_000), word(0)],
                block=10,
                log_index=4,
            ),
        ],
        (POLYMARKET_CTF_EXCHANGE_V2, 10, 11): [
            raw_log(
                address=POLYMARKET_CTF_EXCHANGE_V2,
                topic0=POLYMARKET_ORDER_FILLED_V2_TOPIC,
                data_words=[
                    word(1),
                    word(no_id),
                    word(2_000_000),
                    word(1_330_000),
                    word(0),
                    bytes(32),
                    bytes(32),
                ],
                block=11,
                log_index=2,
            )
        ],
        (POLYMARKET_CONDITIONAL_TOKENS, 10, 11): [
            raw_resolution_log(market.market_id, payouts=(1, 0))
        ],
    }
    block_times = {0: 0, 9: 9, 10: 10, 11: 10, 12: 12, 20: 20}

    async def call(method: str, params):
        if method == "eth_blockNumber":
            return hex(20)
        if method == "eth_getBlockByNumber":
            block = int(params[0], 16)
            timestamp = block_times.get(block, block)
            return {"number": hex(block), "timestamp": hex(timestamp)}
        if method == "eth_getLogs":
            query = params[0]
            key = (
                query["address"],
                int(query["fromBlock"], 16),
                int(query["toBlock"], 16),
            )
            return logs.get(key, [])
        raise AssertionError(method)

    result = await download_polymarket_order_filled_range(
        market,
        start_ms=10_000,
        end_ms=12_000,
        rpc=PolygonJsonRpc(call=call),
        received_time_ms=30_000,
        confirmation_blocks=0,
    )

    assert (result.from_block, result.to_block) == (10, 11)
    assert result.confirmed_head_block == 20
    assert result.decoded_log_count == 3
    assert result.market_log_count == 2
    assert [trade.outcome for trade in result.batch.trades] == [
        OutcomeSide.YES,
        OutcomeSide.NO,
    ]
    assert [trade.native_side for trade in result.batch.trades] == [
        OutcomeOrderSide.BUY,
        OutcomeOrderSide.SELL,
    ]
    assert result.batch.coverage_by_asset[market.yes_asset.asset_id][0].start_ms == 10_000
    assert result.batch.coverage_by_asset[market.yes_asset.asset_id][0].end_ms == 12_000
    assert result.resolution_log_count == 1
    assert len(result.batch.settlements) == 1
    assert result.batch.settlements[0].yes_fraction == 1.0
    assert result.batch.settlements[0].settlement_time_ms == 10_000
    assert result.batch.settlements[0].capital_release_time_ms is None


@pytest.mark.asyncio
async def test_complete_logs_splits_provider_limited_ranges_without_gaps():
    requested: list[tuple[int, int]] = []

    async def call(method: str, params):
        assert method == "eth_getLogs"
        query = params[0]
        start = int(query["fromBlock"], 16)
        end = int(query["toBlock"], 16)
        requested.append((start, end))
        if end - start + 1 > 2:
            raise PolygonRpcRangeError("provider range limit")
        return [
            {
                "blockNumber": hex(block),
                "transactionIndex": "0x0",
                "logIndex": hex(block),
            }
            for block in range(start, end + 1)
        ]

    logs = await PolygonJsonRpc(call=call).complete_logs(
        address=POLYMARKET_CTF_EXCHANGE_V1,
        topic=POLYMARKET_ORDER_FILLED_V1_TOPIC,
        from_block=10,
        to_block=14,
        max_block_span=10,
    )

    assert [int(log["blockNumber"], 16) for log in logs] == [10, 11, 12, 13, 14]
    assert requested == [(10, 14), (10, 12), (10, 11), (12, 12), (13, 14)]


@pytest.mark.asyncio
async def test_single_block_rpc_failure_never_claims_coverage():
    async def call(method: str, params):
        if method == "eth_getLogs":
            raise PolygonRpcError("unavailable")
        raise AssertionError(method)

    with pytest.raises(PolygonRpcError, match="unavailable"):
        await PolygonJsonRpc(call=call).complete_logs(
            address=POLYMARKET_CTF_EXCHANGE_V1,
            topic=POLYMARKET_ORDER_FILLED_V1_TOPIC,
            from_block=10,
            to_block=10,
        )


@pytest.mark.asyncio
async def test_archive_authorization_failure_is_not_recursively_split():
    requested: list[tuple[int, int]] = []

    async def call(method: str, params):
        assert method == "eth_getLogs"
        query = params[0]
        requested.append((int(query["fromBlock"], 16), int(query["toBlock"], 16)))
        raise PolygonRpcError("archive requests require a personal token")

    with pytest.raises(PolygonRpcError, match="personal token"):
        await PolygonJsonRpc(call=call).complete_logs(
            address=POLYMARKET_CTF_EXCHANGE_V1,
            topic=POLYMARKET_ORDER_FILLED_V1_TOPIC,
            from_block=10,
            to_block=14,
            max_block_span=10,
        )
    assert requested == [(10, 14)]


@pytest.mark.asyncio
async def test_unconfirmed_interval_cannot_receive_verified_coverage():
    async def call(method: str, params):
        if method == "eth_blockNumber":
            return hex(200)
        if method == "eth_getBlockByNumber":
            block = int(params[0], 16)
            return {"number": hex(block), "timestamp": hex(block)}
        raise AssertionError(method)

    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    with pytest.raises(PolygonRpcError, match="newer than"):
        await download_polymarket_order_filled_range(
            market,
            start_ms=190_000,
            end_ms=195_000,
            rpc=PolygonJsonRpc(call=call),
            confirmation_blocks=10,
        )
