"""Best-effort prices for asset age, separate from venue candle coverage."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3
import time

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import portalocker

from asset_inception_catalog import (
    AssetIdentity,
    asset_venue_bindings,
    is_non_crypto_instrument,
    resolve_asset_identity,
)
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvRange, OhlcvStore
from simulation_data import is_offline, record_range, snapshot_file
from utils import _atomic_write_json, load_ccxt_instance, to_standard_exchange_name


ASSET_INCEPTION_POLICY_VERSION = 1
MINUTE_MS = 60_000
DAY_MS = 86_400_000
_CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v1/cryptocurrency/map"
_CMC_BATCH_SIZE = 8
_CMC_TIMEOUT_SECONDS = 5
_VENUE_TIMEOUT_SECONDS = 12
_VENUE_CONCURRENCY = 4
_VENUE_PAGE_DAYS = 31
_VENUE_PAGE_MINUTES = 60
_VENUE_EVIDENCE_EXCHANGES = frozenset({"okx", "kucoin", "bitunix"})
_MAX_LOCAL_MONTHS = 3


@dataclass(frozen=True)
class InceptionEvidence:
    timestamp_ms: int
    source: str
    asset_id: str | None = None
    is_provisional: bool = False


VenueMarket = tuple[str, Mapping]


def market_inception_identity(market):
    """Identify a market and its age evidence boundary across metadata refreshes."""
    if market is None:
        return None
    info = market.get("info")
    info = info if isinstance(info, Mapping) else {}
    fields = ("id", "symbol", "base", "baseId", "quote", "settle", "swap")
    age_fields = (
        "instIdCode", "instCategory", "underlying", "marketType", "assetClass",
        "isRwa", "underlyingType", "contractType", "listTime", "preMktSwTime",
        "firstOpenDate", "preMarketToPerpDate", "launchTime", "marketStage",
        "ruleType", "isPreListing", "is_pre_market", "state", "status", "symbolStatus",
    )
    return tuple(market.get(field) for field in fields), tuple(info.get(field) for field in age_fields)


def _positive_ms(value, *, now_ms):
    if type(value) is int or isinstance(value, str) and len(value) <= 16 and value.isascii() and value.isdigit():
        parsed = int(value)
        if 0 < parsed <= now_ms:
            return parsed
    return None


def _published_time_ms(value, now_ms):
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return _positive_ms(int(parsed.timestamp() * 1000), now_ms=now_ms)
    except (OverflowError, ValueError):
        return None
    return None


def _asset_id(identity: AssetIdentity | None):
    return f"coinmarketcap:{identity.cmc_id}" if identity is not None else None


def _catalog_evidence(identity, now_ms):
    if identity is None or not identity.observation_source:
        return None
    timestamp = _positive_ms(identity.observation_ms, now_ms=now_ms)
    if timestamp is None or timestamp < identity.earliest_eligible_ms:
        return None
    return InceptionEvidence(timestamp, f"catalog:{identity.observation_source}", _asset_id(identity))


def _floor_from_listing_and_conversion(info, listing_key, conversion_key, now_ms):
    listing = _positive_ms(info.get(listing_key), now_ms=now_ms)
    conversion = info.get(conversion_key)
    if listing is None or conversion in ("", None, "0", 0):
        return listing
    conversion_ms = _positive_ms(conversion, now_ms=now_ms)
    return max(listing, conversion_ms) if conversion_ms is not None else None


def _regular_trading_floor(exchange, market, identity, now_ms):
    """Bind a price search to a normal-trading interval for this exact instrument.

    A current regular status does not certify an earlier launch field. Gate CC
    and Bitget LIT both retain their documented pre-market launch after conversion.
    """
    info = market.get("info")
    market_id = market.get("id")
    if not isinstance(info, Mapping) or not isinstance(market_id, str) or market.get("swap") is not True:
        return None
    if exchange == "okx" and info.get("instId") == market_id:
        category = info.get("instCategory")
        if identity is None and not (category == "1" or type(category) is int and category == 1):
            return None
        if info.get("state") != "live" or info.get("ruleType") not in {"normal", "xperp"}:
            return None
        if info.get("ruleType") == "xperp" and info.get("preMktSwTime") in (None, "", "0", 0):
            return None
        listing = _floor_from_listing_and_conversion(info, "listTime", "preMktSwTime", now_ms)
    elif exchange == "kucoin" and info.get("symbol") == market_id:
        if (
            info.get("status") != "Open"
            or info.get("marketStage") != "NORMAL"
            or info.get("marketType") != "CRYPTO"
            or "preMarketToPerpDate" not in info
        ):
            return None
        listing = _floor_from_listing_and_conversion(
            info, "firstOpenDate", "preMarketToPerpDate", now_ms
        )
    elif exchange == "bitunix" and info.get("symbol") == market_id:
        # This feed mixes crypto and stock contracts without an asset-class field.
        if identity is None or info.get("symbolStatus") != "OPEN":
            return None
        listing = _positive_ms(info.get("launchTime"), now_ms=now_ms)
    else:
        return None
    if listing is None:
        return None
    return max(listing, identity.earliest_eligible_ms) if identity is not None else listing


def _cache_path(cache_dir, key):
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / "asset_inception" / f"{digest}.json"


def _read_cached(cache_dir, key, floor_ms, now_ms):
    path = _cache_path(cache_dir, key)
    try:
        with path.open(encoding="utf-8") as file:
            cached = json.load(file)
        if (not isinstance(cached, dict) or type(cached.get("version")) is not int
                or cached["version"] != ASSET_INCEPTION_POLICY_VERSION):
            return None
        timestamp = _positive_ms(cached.get("timestamp_ms"), now_ms=now_ms)
        if cached.get("key") == key and timestamp is not None and timestamp >= floor_ms:
            snapshot_file(path)
            return timestamp
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logging.warning("[age] asset evidence cache unreadable | error_type=%s", type(exc).__name__)
    return None


def _save_cached(cache_dir, key, timestamp, floor_ms, now_ms):
    path = _cache_path(cache_dir, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(f"{path}.lock", timeout=0, fail_when_locked=True):
            old = _read_cached(cache_dir, key, floor_ms, now_ms)
            if old is None or timestamp < old:
                _atomic_write_json(
                    str(path),
                    {"version": ASSET_INCEPTION_POLICY_VERSION, "key": key, "timestamp_ms": timestamp},
                )
    except (OSError, portalocker.exceptions.LockException) as exc:
        logging.debug("[age] asset cache write deferred | error_type=%s", type(exc).__name__)


async def _fetch_cmc(session, identities, now_ms):
    expected = {item.cmc_id: item for item in identities}
    params = {
        "symbol": ",".join(sorted({item.cmc_symbol for item in identities})),
        "aux": "first_historical_data,is_active",
    }
    async with session.get(_CMC_URL, params=params, allow_redirects=False) as response:
        response.raise_for_status()
        payload = await response.json()
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("status"), dict)
        or type(payload["status"].get("error_code")) is not int
        or payload["status"]["error_code"] != 0
        or not isinstance(payload.get("data"), list)
    ):
        raise ValueError("unexpected cryptocurrency map response")
    matches = {}
    duplicates = set()
    for row in payload["data"]:
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ValueError("invalid cryptocurrency map row")
        asset_id = row["id"]
        if asset_id not in expected:
            continue
        if asset_id in matches:
            duplicates.add(asset_id)
        matches[asset_id] = row
    result = {}
    for asset_id, row in matches.items():
        identity = expected[asset_id]
        if asset_id in duplicates or row.get("symbol") != identity.cmc_symbol:
            continue
        timestamp = _published_time_ms(row.get("first_historical_data"), now_ms)
        if timestamp is not None and timestamp >= identity.earliest_eligible_ms:
            result[asset_id] = timestamp
    return result


async def _resolve_cmc(identities, cache_dir, now_ms):
    timeout = aiohttp.ClientTimeout(total=_CMC_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for offset in range(0, len(identities), _CMC_BATCH_SIZE):
            batch = identities[offset:offset + _CMC_BATCH_SIZE]
            try:
                timestamps = await _fetch_cmc(session, batch, now_ms)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                logging.warning("[age] asset price provider unavailable | provider=coinmarketcap error_type=%s",
                                type(exc).__name__)
                break
            for identity in batch:
                timestamp = timestamps.get(identity.cmc_id)
                if timestamp is not None:
                    key = _asset_id(identity)
                    _save_cached(cache_dir, key, timestamp, identity.earliest_eligible_ms, now_ms)
                    yield InceptionEvidence(timestamp, key, key)


def _local_exchange_candle(store, exchange, symbol, floor_ms, ceiling_ms):
    """Find the first verified positive-volume 1m candle in the leading local chunks."""
    start_ms = (floor_ms + MINUTE_MS - 1) // MINUTE_MS * MINUTE_MS
    end_ms = (ceiling_ms - MINUTE_MS) // MINUTE_MS * MINUTE_MS
    if start_ms > end_ms:
        return None
    chunks = store.catalog.list_chunks(exchange, "1m", symbol, start_ms, end_ms)
    for chunk in chunks[:_MAX_LOCAL_MONTHS]:
        values, valid = store.read_verified_chunk(chunk)
        start = (max(start_ms, chunk.start_ts) - chunk.start_ts) // MINUTE_MS
        stop = (min(end_ms, chunk.end_ts) - chunk.start_ts) // MINUTE_MS + 1
        window = values[start:stop]
        real_price = (
            valid[start:stop]
            & np.isfinite(window).all(axis=1)
            & (window[:, :3] > 0).all(axis=1)
            & (window[:, 1] <= window[:, 2])
            & (window[:, 2] <= window[:, 0])
            & (window[:, 3] > 0)
        )
        if real_price.any():
            index = start + int(real_price.argmax())
            observed = chunk.start_ts + index * MINUTE_MS
            record_range(
                exchange, symbol, observed, observed,
                OhlcvRange(
                    np.asarray([observed], dtype=np.int64),
                    values[index:index + 1],
                    valid[index:index + 1],
                ),
            )
            return observed + MINUTE_MS
    return None


def _valid_venue_candle(row, since, end, interval_ms):
    if not isinstance(row, (list, tuple)) or len(row) < 6 or type(row[0]) is not int:
        return False
    timestamp, opened, high, low, closed, volume = row[:6]
    return (
        since <= timestamp <= end
        and timestamp % interval_ms == 0
        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in row[1:6])
        and low <= opened <= high
        and low <= closed <= high
    )


def _new_venue_client(exchange, market):
    if exchange == "bitunix":
        from custom_endpoint_overrides import resolve_custom_endpoint_override
        from exchanges.bitunix import BitunixClient, apply_bitunix_endpoint_override

        client = BitunixClient(apply_bitunix_endpoint_override(
            {"enableRateLimit": True, "timeout": 5_000, "wsEnabled": False},
            resolve_custom_endpoint_override(exchange),
        ))
        client.markets = {market["symbol"]: dict(market)}
    else:
        client = load_ccxt_instance(exchange, timeout_ms=5_000)
        client.set_markets([dict(market)])
    return client


def _venue_candle_end_params(exchange, end_ms):
    if exchange == "okx":
        return {"type": "HistoryCandles", "until": end_ms}
    if exchange == "kucoin":
        return {"to": end_ms - 1}
    if exchange == "bitunix":
        return {"until": end_ms - 1}
    raise ValueError(f"Unsupported asset-age venue {exchange!r}")


async def _fetch_venue_candle(exchange, market, floor_ms, now_ms):
    """Verify an actual trade during normal trading, without walking unbounded history."""
    first_minute = (floor_ms + MINUTE_MS - 1) // MINUTE_MS * MINUTE_MS
    if first_minute + MINUTE_MS > now_ms:
        return None
    client = _new_venue_client(exchange, market)
    try:
        for timeframe, interval_ms, page_size in (
            ("1m", MINUTE_MS, _VENUE_PAGE_MINUTES),
            ("1d", DAY_MS, _VENUE_PAGE_DAYS),
        ):
            start_ms = (floor_ms + interval_ms - 1) // interval_ms * interval_ms
            end_ms = min(start_ms + page_size * interval_ms, now_ms // interval_ms * interval_ms)
            if start_ms >= end_ms:
                continue
            rows = await client.fetch_ohlcv(
                market["symbol"], since=start_ms, timeframe=timeframe, limit=page_size,
                params=_venue_candle_end_params(exchange, end_ms),
            )
            if not isinstance(rows, list) or len(rows) > page_size:
                raise ValueError("unexpected venue history response")
            observed = [
                row[0] + interval_ms for row in rows
                if _valid_venue_candle(row, start_ms, end_ms - interval_ms, interval_ms)
            ]
            if observed:
                return min(observed)
    finally:
        await client.close()
    return None


async def _resolve_venue(requests, cache_dir, now_ms):
    async def query(exchange, market, floor, cache_key, identity):
        try:
            value = await _fetch_venue_candle(exchange, market, floor, now_ms)
        except (ccxt.BaseError, asyncio.TimeoutError, aiohttp.ClientError, RuntimeError, ValueError) as exc:
            logging.warning("[age] venue price evidence unavailable | exchange=%s error_type=%s",
                            exchange, type(exc).__name__)
            return None
        if value is not None:
            _save_cached(cache_dir, cache_key, value, floor, now_ms)
            return InceptionEvidence(value, f"venue_candle:{exchange}", _asset_id(identity), True)
        return None

    results = {}
    deadline = asyncio.get_running_loop().time() + _VENUE_TIMEOUT_SECONDS
    for offset in range(0, len(requests), _VENUE_CONCURRENCY):
        batch = requests[offset:offset + _VENUE_CONCURRENCY]
        tasks = {asyncio.create_task(query(*value)): key for key, value in batch}
        try:
            done, pending = await asyncio.wait(tasks, timeout=max(0, deadline - asyncio.get_running_loop().time()))
            for task in done:
                if (evidence := task.result()) is not None:
                    results.setdefault(tasks[task], []).append(evidence)
            if pending or asyncio.get_running_loop().time() >= deadline:
                logging.warning("[age] venue price lookup deadline reached")
                break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return results


def _load_local_markets(cache_dir, exchange, loaded):
    if exchange not in loaded:
        path = cache_dir / exchange / "markets.json"
        try:
            with path.open(encoding="utf-8") as file:
                metadata = json.load(file)
        except FileNotFoundError:
            metadata = {}
        except (OSError, ValueError) as exc:
            logging.debug("[age] local market binding unavailable | exchange=%s error_type=%s",
                          exchange, type(exc).__name__)
            metadata = {}
        loaded[exchange] = metadata if isinstance(metadata, dict) else {}
    return loaded[exchange]


def _earliest(evidence):
    return min(evidence, key=lambda item: (item.timestamp_ms, item.is_provisional, item.source))


def _collect_local_evidence(bound, candidates, cache_dir, allow_remote, now_ms):
    store_root = cache_dir / "ohlcvs"
    store = None
    if bound and (catalog_path := store_root / "catalog.sqlite").is_file():
        store = OhlcvStore(store_root, OhlcvCatalog(catalog_path, read_only=True))
    requests = []
    cached_metadata = {}
    for key, (venues, identity) in bound.items():
        active_sources = {(exchange, market.get("symbol")) for exchange, market in venues}
        sources = [(exchange, market, None) for exchange, market in venues]
        if identity is not None:
            for exchange, native_id, symbol in asset_venue_bindings(identity):
                # These are the only venues with a supported normal-trading historical bound.
                if (exchange, symbol) in active_sources or exchange not in _VENUE_EVIDENCE_EXCHANGES:
                    continue
                metadata = _load_local_markets(cache_dir, exchange, cached_metadata)
                market = metadata.get(symbol)
                if (
                    isinstance(market, Mapping)
                    and market.get("id") == native_id
                    and resolve_asset_identity(exchange, market) == identity
                ):
                    sources.append((exchange, market, cache_dir / exchange / "markets.json"))
        eligible_requests = []
        for exchange, market, metadata_path in sources:
            symbol = market.get("symbol")
            floor = _regular_trading_floor(exchange, market, identity, now_ms)
            if not isinstance(symbol, str) or floor is None:
                continue
            asset_id = _asset_id(identity)
            binding = f"{exchange}:{market['id']}:{symbol}:{floor}:{asset_id}"
            for source in ("local_candle", "venue_candle"):
                cache_key = f"{source}:{binding}"
                if cached := _read_cached(cache_dir, cache_key, floor, now_ms):
                    if metadata_path is not None:
                        snapshot_file(metadata_path)
                    candidates[key].append(
                        InceptionEvidence(cached, f"{source}:{exchange}", asset_id, True)
                    )
            ceiling = _earliest(candidates[key]).timestamp_ms if candidates[key] else now_ms
            if store is not None and floor < ceiling:
                try:
                    local = _local_exchange_candle(store, exchange, symbol, floor, ceiling)
                except (OSError, ValueError, sqlite3.Error, portalocker.exceptions.LockException) as exc:
                    logging.debug("[age] local asset price evidence unavailable | exchange=%s error_type=%s",
                                  exchange, type(exc).__name__)
                else:
                    if local is not None:
                        if metadata_path is not None:
                            snapshot_file(metadata_path)
                        if allow_remote:
                            _save_cached(cache_dir, f"local_candle:{binding}", local, floor, now_ms)
                        candidates[key].append(
                            InceptionEvidence(local, f"local_candle:{exchange}", asset_id, True)
                        )
            if allow_remote and metadata_path is None:
                eligible_requests.append(
                    (key, (exchange, market, floor, f"venue_candle:{binding}", identity))
                )
        if not candidates[key]:
            requests.extend(eligible_requests)

    return requests


async def resolve_asset_inception(
    markets: Mapping[str, Sequence[VenueMarket]], *, allow_remote: bool = True, cache_dir="caches",
) -> dict[str, InceptionEvidence]:
    """Resolve age by exact instrument, omitting unknown or conflicting identities.

    The per-asset provider snapshot and candle evidence are shared across callers. An
    unbound instrument can contribute only local or venue data for that exact market.
    """
    cache_dir = Path(cache_dir)
    now_ms = int(time.time() * 1000)
    allow_remote = allow_remote and not is_offline()
    bound = {}
    candidates = {}
    get_cmc = {}
    for key, venues in markets.items():
        resolved = [(to_standard_exchange_name(exchange), market) for exchange, market in venues]
        if not resolved:
            continue
        if any(is_non_crypto_instrument(exchange, market) for exchange, market in resolved):
            logging.debug("[age] excluded non-crypto market from asset-age lookup | market=%s", key)
            continue
        identities = [resolve_asset_identity(exchange, market) for exchange, market in resolved]
        if len(resolved) > 1 and (identities[0] is None or any(i != identities[0] for i in identities)):
            logging.warning("[age] conflicting cross-venue asset identity | market=%s", key)
            continue
        identity = identities[0]
        bound[key] = (resolved, identity)
        candidates[key] = []
        if identity is not None:
            provider_key = _asset_id(identity)
            if reviewed := _catalog_evidence(identity, now_ms):
                candidates[key].append(reviewed)
            if cached := _read_cached(cache_dir, provider_key, identity.earliest_eligible_ms, now_ms):
                candidates[key].append(InceptionEvidence(cached, provider_key, provider_key))
            if not candidates[key]:
                get_cmc[provider_key] = identity
    if allow_remote and get_cmc:
        refreshed = {
            fact.asset_id: fact async for fact in _resolve_cmc(list(get_cmc.values()), cache_dir, now_ms)
        }
        for key, (_, identity) in bound.items():
            if fact := refreshed.get(_asset_id(identity)):
                candidates[key].append(fact)

    if not bound:
        return {}
    requests = await asyncio.to_thread(
        _collect_local_evidence, bound, candidates, cache_dir, allow_remote, now_ms
    )

    if requests:
        for key, facts in (await _resolve_venue(requests, cache_dir, now_ms)).items():
            candidates[key].extend(facts)
    result = {key: _earliest(facts) for key, facts in candidates.items() if facts}
    if result:
        logging.debug("[age] asset price evidence resolved | assets=%d provisional=%d",
                      len(result), sum(fact.is_provisional for fact in result.values()))
    return result
