"""Bounded, source-specific observations of unavailable leading candle history.

These facts say what a public historical endpoint returned for explicit closed
intervals. They do not establish market inception, no trading, or coin age.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np


MINUTE_MS = 60_000
MAX_PAGE_MINUTES = 1000
MAX_PREFIX_PAGES = 4
MAX_PROBE_ATTEMPTS = 4
MAX_PRIOR_GAP_SUMMARIES = 8
PROOF_KIND = "source_leading_unavailable"
PROOF_VERSION = 1


def _integer(value):
    if type(value) is not int:
        raise ValueError("Leading history timestamps and limits must be integers")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def prior_gap_provenance(gaps):
    """Retain bounded prior-fact references, never recursively escaped proofs."""
    digest = hashlib.sha256()
    count = 0
    summaries = []
    for gap in gaps:
        if not gap.note:
            continue
        if not isinstance(gap.note, str):
            raise ValueError("Prior source history notes must be text")
        summary = {
            "start_ts": int(gap.start_ts), "end_ts": int(gap.end_ts),
            "reason": str(gap.reason),
            "note_sha256": hashlib.sha256(gap.note.encode()).hexdigest(),
        }
        encoded = _json(summary).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
        summaries.append(summary)
        if len(summaries) > MAX_PRIOR_GAP_SUMMARIES:
            summaries.pop(0)
    return {"count": count, "sha256": digest.hexdigest(), "summaries": summaries}


def request(exchange, market_id, start, end, *, control=False):
    """One explicit closed minute interval, never an unbounded first-row query."""
    start, end = _integer(start), _integer(end)
    if start <= 0 or start % MINUTE_MS or end % MINUTE_MS or end < start:
        raise ValueError("Leading history probe requires an aligned closed interval")
    count = (end - start) // MINUTE_MS + 1
    if count > MAX_PAGE_MINUTES or (control and count != 1):
        raise ValueError("Leading history probe exceeds its bounded request size")
    if not isinstance(market_id, str) or not market_id or not market_id.isalnum():
        raise ValueError("Leading history probe requires an exact venue market id")
    limit = 1 if control else MAX_PAGE_MINUTES
    if exchange == "binance":
        return {"method": "fapiPublicGetKlines", "params": {
            "symbol": market_id, "interval": "1m", "startTime": start,
            "endTime": end + MINUTE_MS - 1, "limit": limit,
        }}
    if exchange == "bybit":
        return {"method": "publicGetV5MarketKline", "params": {
            "category": "linear", "symbol": market_id, "interval": "1",
            "start": start, "end": end + MINUTE_MS - 1, "limit": limit,
        }}
    raise ValueError("Leading history probes support only Binance and Bybit linear candles")


def plan(exchange, market_id, listing_timestamp, first_valid):
    listing_timestamp, first_valid = _integer(listing_timestamp), _integer(first_valid)
    start = listing_timestamp // MINUTE_MS * MINUTE_MS
    if listing_timestamp <= 0 or first_valid % MINUTE_MS or first_valid <= start:
        raise ValueError("Leading history proof requires candles after the published listing bound")
    if first_valid - start > MAX_PAGE_MINUTES * MAX_PREFIX_PAGES * MINUTE_MS:
        raise ValueError("Leading history proof exceeds its bounded discovery budget")
    pages = []
    while start < first_valid:
        end = min(first_valid - MINUTE_MS, start + (MAX_PAGE_MINUTES - 1) * MINUTE_MS)
        pages.append(request(exchange, market_id, start, end))
        start = end + MINUTE_MS
    return pages, request(exchange, market_id, first_valid, first_valid, control=True)


def _bound_symbol(symbol, market_id):
    return (isinstance(symbol, str) and symbol.endswith("/USDT:USDT")
            and market_id == symbol.split(":", 1)[0].replace("/", ""))


def response_rows(exchange, market_id, query, response):
    """Validate successful source envelopes and every returned OHLCV row."""
    params = query["params"]
    start = params["startTime" if exchange == "binance" else "start"]
    end = params["endTime" if exchange == "binance" else "end"]
    if exchange == "bybit":
        if not isinstance(response, dict) or type(response.get("retCode")) is not int or response["retCode"] != 0:
            raise ValueError("Bybit leading history response was not successful")
        result = response.get("result")
        if not isinstance(result, dict) or result.get("symbol") != market_id or result.get("category") != "linear":
            raise ValueError("Bybit leading history response has a different source binding")
        rows = result.get("list")
    else:
        rows = response
    if not isinstance(rows, list) or len(rows) > params["limit"]:
        raise ValueError("Leading history response is not a bounded candle list")
    parsed = []
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 6 or isinstance(row[0], bool):
            raise ValueError("Malformed leading history candle")
        if not isinstance(row[0], (str, int)):
            raise ValueError("Leading history candle timestamp must be an integer")
        if any(type(value) not in (str, int, float) for value in row[1:6]):
            raise ValueError("Leading history OHLCV fields must be numeric scalars")
        timestamp = int(row[0])
        values = np.asarray(row[1:6], dtype=np.float64)
        if (values.shape != (5,) or timestamp % MINUTE_MS
                or not start <= timestamp <= end or timestamp in seen
                or not np.isfinite(values).all() or np.any(values[:4] <= 0) or values[4] < 0
                or values[2] > values[1] or not values[2] <= values[0] <= values[1]
                or not values[2] <= values[3] <= values[1]):
            raise ValueError("Invalid or out-of-range leading history candle")
        with np.errstate(over="ignore"):
            if not np.isfinite(values.astype(np.float32)).all():
                raise ValueError("Leading history candle exceeds canonical store precision")
        seen.add(timestamp)
        parsed.append((timestamp, [float(values[i]) for i in (1, 2, 3, 4)]))
    return sorted(parsed)


def _control_digest(values):
    values = np.asarray(values, dtype="<f4")
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Leading history control requires one genuine HLCV row")
    return hashlib.sha256(values.tobytes()).hexdigest()


async def collect(*, exchange, symbol, market_id, listing_timestamp, first_valid,
                  control_values, observed_at_ms, fetch):
    """Return a complete observation or genuine earlier rows for ordinary repair."""
    if not _bound_symbol(symbol, market_id):
        raise ValueError("Leading history symbol does not match its linear venue market id")
    queries, control_query = plan(exchange, market_id, listing_timestamp, first_valid)
    pages = []
    for query in queries:
        response = await fetch(deepcopy(query))
        rows = response_rows(exchange, market_id, query, response)
        if rows:
            return None, rows
        pages.append({"request": query, "response": response, "response_sha256": _digest(response)})
    response = await fetch(deepcopy(control_query))
    rows = response_rows(exchange, market_id, control_query, response)
    expected = _control_digest(control_values)
    if len(rows) != 1 or rows[0][0] != first_valid or _control_digest(rows[0][1]) != expected:
        raise ValueError("Leading history historical control does not match the cached candle")
    proof = {"kind": PROOF_KIND, "version": PROOF_VERSION,
             "exchange": exchange, "timeframe": "1m", "symbol": symbol, "market_id": market_id,
             "listing_field": "onboardDate" if exchange == "binance" else "launchTime",
             "published_listing_timestamp_ms": listing_timestamp,
             "first_valid_timestamp_ms": first_valid,
             "observed_at_ms": _integer(observed_at_ms), "pages": pages,
             "control": {"request": control_query, "response": response,
                         "response_sha256": _digest(response)},
             "control_hlcv_float32_sha256": expected}
    # Reuse the reader's validation before creating a persistent fact.
    if not validates(proof, exchange=exchange, symbol=symbol, first_valid=first_valid):
        raise ValueError("Leading history observation failed its own proof validation")
    return proof, []


def validates(proof, *, exchange, symbol, first_valid, control_values=None):
    """Old observation-only notes and incomplete/foreign certificates fail closed."""
    try:
        if (not isinstance(proof, dict) or proof.get("kind") != PROOF_KIND
                or type(proof.get("version")) is not int or proof["version"] != PROOF_VERSION
                or proof.get("exchange") != exchange or proof.get("symbol") != symbol
                or proof.get("timeframe") != "1m"
                or _integer(proof.get("first_valid_timestamp_ms")) != _integer(first_valid)
                or proof.get("listing_field") != ("onboardDate" if exchange == "binance" else "launchTime")):
            return False
        market_id = proof["market_id"]
        if not _bound_symbol(symbol, market_id):
            return False
        if control_values is not None and _control_digest(control_values) != proof["control_hlcv_float32_sha256"]:
            return False
        listing = proof["published_listing_timestamp_ms"]
        queries, control_query = plan(exchange, market_id, listing, first_valid)
        observed = _integer(proof["observed_at_ms"])
        if observed <= first_valid or len(proof["pages"]) != len(queries):
            return False
        for page, query in zip(proof["pages"], queries, strict=True):
            if (_json(page["request"]) != _json(query)
                    or page["response_sha256"] != _digest(page["response"])
                    or response_rows(exchange, market_id, query, page["response"])):
                return False
        control = proof["control"]
        rows = response_rows(exchange, market_id, control_query, control["response"])
        return (len(rows) == 1 and rows[0][0] == first_valid
                and _json(control["request"]) == _json(control_query)
                and control["response_sha256"] == _digest(control["response"])
                and _control_digest(rows[0][1]) == proof["control_hlcv_float32_sha256"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def note(proof):
    encoded = _json(proof)
    if len(encoded) > 65_536:
        raise ValueError("Leading history certificate exceeds its bounded fact size")
    return encoded


def from_note(value):
    try:
        if not isinstance(value, str) or len(value) > 65_536:
            return None
        return json.loads(value)
    except (TypeError, ValueError):
        return None
