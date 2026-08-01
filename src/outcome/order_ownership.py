from __future__ import annotations

import hashlib


_CLOID_NAMESPACE = bytes.fromhex("50424f4d")  # "PBOM": Passivbot outcome market.
_CLOID_PREFIX = "0x"


def managed_outcome_client_order_id(
    market_id: str,
    *,
    slot: str,
    observation_end_ms: int,
) -> str:
    """Return a deterministic 16-byte Hyperliquid client-order ID."""

    if slot not in {"canonical_bid", "canonical_ask"}:
        raise ValueError(f"unsupported outcome order slot {slot!r}")
    if observation_end_ms < 0 or observation_end_ms % 1_000 != 0:
        raise ValueError("outcome client-order timestamp must be second-aligned")
    timestamp_seconds = observation_end_ms // 1_000
    if timestamp_seconds >= 1 << 56:
        raise ValueError("outcome client-order timestamp exceeds encoding")
    market_digest = hashlib.sha256(str(market_id).encode()).digest()[:4]
    slot_byte = b"\x00" if slot == "canonical_bid" else b"\x01"
    encoded = (
        _CLOID_NAMESPACE
        + market_digest
        + slot_byte
        + timestamp_seconds.to_bytes(7, "big")
    )
    return _CLOID_PREFIX + encoded.hex()


def is_managed_outcome_client_order_id(
    client_order_id: str | None,
    market_id: str,
) -> bool:
    return managed_outcome_client_order_slot(client_order_id, market_id) is not None


def managed_outcome_client_order_slot(
    client_order_id: str | None,
    market_id: str,
) -> str | None:
    if client_order_id is None or not client_order_id.startswith(_CLOID_PREFIX):
        return None
    try:
        encoded = bytes.fromhex(client_order_id[2:])
    except ValueError:
        return None
    if len(encoded) != 16 or encoded[:4] != _CLOID_NAMESPACE:
        return None
    expected_market = hashlib.sha256(str(market_id).encode()).digest()[:4]
    if encoded[4:8] != expected_market or encoded[8] not in {0, 1}:
        return None
    return "canonical_bid" if encoded[8] == 0 else "canonical_ask"
