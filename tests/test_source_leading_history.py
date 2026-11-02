"""Source-leading absence is bounded evidence, distinct from market inception."""

from copy import deepcopy
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest

import hlcv_preparation as hp
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from simulation_data import OfflineDataError, has_current_offline_proof, simulation_data_policy
import source_leading_history as proof


START = 1_774_915_200_000
LISTING = START + 10_123
FIRST = START + 1500 * 60_000
END = FIRST + 9 * 60_000
SYMBOL = "NEW/USDT:USDT"
VALUES = np.asarray([101., 99., 100., 3.], dtype=np.float32)


def envelope(exchange, rows):
    if exchange == "binance":
        return rows
    return {"retCode": 0, "retMsg": "OK",
            "result": {"symbol": "NEWUSDT", "category": "linear", "list": rows}}


def candle(timestamp=FIRST, volume="3"):
    return [str(timestamp), "100", "101", "99", "100", volume]


async def observation(exchange="bybit"):
    queries = []

    async def fetch(query):
        queries.append(query)
        p = query["params"]
        start = p.get("start", p.get("startTime"))
        return envelope(exchange, [candle()] if start == FIRST else [])

    result, earlier = await proof.collect(
        exchange=exchange, symbol=SYMBOL, market_id="NEWUSDT", listing_timestamp=LISTING,
        first_valid=FIRST, control_values=VALUES, observed_at_ms=END + 60_000, fetch=fetch,
    )
    assert earlier == []
    return result, queries


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "bybit"])
async def test_complete_bounded_pages_and_matching_historical_control_are_required(exchange):
    result, queries = await observation(exchange)
    assert len(queries) == 3  # Two closed negative pages and one adjacent positive control.
    assert proof.validates(result, exchange=exchange, symbol=SYMBOL, first_valid=FIRST)
    assert proof.from_note(proof.note(result)) == result
    assert result["kind"] == "source_leading_unavailable"
    assert result["listing_field"] == ("onboardDate" if exchange == "binance" else "launchTime")
    starts = [q["params"].get("start", q["params"].get("startTime")) for q in queries]
    ends = [q["params"].get("end", q["params"].get("endTime")) for q in queries]
    assert starts == [START, START + 1000 * 60_000, FIRST]
    assert ends == [START + 1000 * 60_000 - 1, FIRST - 1, FIRST + 60_000 - 1]
    assert [q["params"]["limit"] for q in queries] == [1000, 1000, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["old_version", "missing_page", "source", "symbol", "market_id",
    "request", "response_hash", "control_hash", "control_value", "listing", "boolean_version",
    "float_boundary"])
async def test_incomplete_or_foreign_proofs_cannot_be_reused(mutation):
    result, _ = await observation()
    if mutation == "old_version": result["version"] = 0
    elif mutation == "boolean_version": result["version"] = True
    elif mutation == "float_boundary": result["first_valid_timestamp_ms"] = float(FIRST)
    elif mutation == "missing_page": result["pages"].pop()
    elif mutation == "source": result["exchange"] = "binance"
    elif mutation == "symbol": result["symbol"] = "OTHER/USDT:USDT"
    elif mutation == "market_id": result["market_id"] = "OTHERUSDT"
    elif mutation == "request": result["pages"][0]["request"]["params"]["end"] -= 60_000
    elif mutation == "response_hash": result["pages"][0]["response_sha256"] = "0" * 64
    elif mutation == "control_hash": result["control_hlcv_float32_sha256"] = "0" * 64
    elif mutation == "control_value": result["control"]["response"]["result"]["list"][0][5] = "4"
    elif mutation == "listing": result["published_listing_timestamp_ms"] -= 60_000
    assert not proof.validates(result, exchange="bybit", symbol=SYMBOL, first_valid=FIRST)


def test_probe_budget_and_source_binding_do_not_waive_unknown_history():
    assert len(proof.plan("bybit", "NEWUSDT", LISTING, START + 4000 * 60_000)[0]) == 4
    for listing, first in ((None, FIRST), (True, FIRST), (LISTING, START),
                            (LISTING, START + 4001 * 60_000)):
        with pytest.raises(ValueError):
            proof.plan("bybit", "NEWUSDT", listing, first)
    assert proof.from_note("confirmed_by_v2_fetch") is None
    assert not has_current_offline_proof({"version": 2, "boundary_proof_version": 2})


@pytest.mark.parametrize("response", [None, {}, {"retCode": 1}, {"retCode": False},
    {"retCode": 0, "result": {"symbol": "OTHERUSDT", "category": "linear", "list": []}},
    {"retCode": 0, "result": {"symbol": "NEWUSDT", "category": "inverse", "list": []}},
    {"retCode": 0, "result": {"symbol": "NEWUSDT", "category": "linear"}}])
def test_unsuccessful_or_ambiguous_responses_are_not_empty_history(response):
    query = proof.request("bybit", "NEWUSDT", START, START)
    with pytest.raises(ValueError):
        proof.response_rows("bybit", "NEWUSDT", query, response)


@pytest.mark.parametrize("row", [candle(START - 60_000), candle(START + 1),
    [str(START), "100", "99", "101", "100", "3"],
    [str(START), "100", "101", "99", "100", "-1"],
    [str(START), "100", "101", "99", "nan", "3"]])
def test_malformed_or_out_of_query_candles_are_not_absence_evidence(row):
    query = proof.request("bybit", "NEWUSDT", START, START)
    with pytest.raises(ValueError):
        proof.response_rows("bybit", "NEWUSDT", query, envelope("bybit", [row]))


@pytest.mark.parametrize("exchange", ["binance", "bybit"])
@pytest.mark.parametrize("value", [True, False, ["100"], ("100",), {"value": "100"}, None])
@pytest.mark.parametrize("field", range(1, 6))
def test_earlier_candle_fields_must_be_flat_numeric_scalars(exchange, value, field):
    row = candle(START)
    row[field] = value
    query = proof.request(exchange, "NEWUSDT", START, START)
    with pytest.raises(ValueError, match="numeric scalars"):
        proof.response_rows(exchange, "NEWUSDT", query, envelope(exchange, [row]))


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "bybit"])
@pytest.mark.parametrize("malformed", ["nested", "boolean"])
async def test_malformed_matching_control_cannot_create_certificate(exchange, malformed):
    if malformed == "nested":
        row = [str(FIRST), ["100"], ["101"], ["99"], ["100"], ["3"]]
        control_values = VALUES
    else:
        row = [str(FIRST), True, True, True, True, False]
        control_values = np.asarray([1., 1., 1., 0.], dtype=np.float32)

    async def fetch(query):
        params = query["params"]
        start = params.get("start", params.get("startTime"))
        return envelope(exchange, [row] if start == FIRST else [])

    with pytest.raises(ValueError, match="numeric scalars"):
        await proof.collect(
            exchange=exchange, symbol=SYMBOL, market_id="NEWUSDT", listing_timestamp=LISTING,
            first_valid=FIRST, control_values=control_values, observed_at_ms=END + 60_000,
            fetch=fetch,
        )


@pytest.mark.asyncio
async def test_real_earlier_rows_are_returned_for_repair_without_certifying_absence():
    async def fetch(query):
        return envelope("bybit", [candle(START)])
    result, earlier = await proof.collect(
        exchange="bybit", symbol=SYMBOL, market_id="NEWUSDT", listing_timestamp=LISTING,
        first_valid=FIRST, control_values=VALUES, observed_at_ms=END + 60_000, fetch=fetch,
    )
    assert result is None and earlier == [(START, VALUES.tolist())]


def owner(exchange, *, mode="ok"):
    field = "onboardDate" if exchange == "binance" else "launchTime"
    calls = []

    async def fetch(symbol, query):
        assert symbol == SYMBOL
        calls.append(deepcopy(query))
        p = query["params"]
        start = p.get("start", p.get("startTime"))
        rows = [candle()] if start == FIRST else []
        if mode == "mismatch" and start == FIRST: rows = [candle(volume="4")]
        if mode == "earlier": rows = [candle(START)]
        return envelope(exchange, rows)

    return SimpleNamespace(exchange=exchange, cm=None, gap_tolerance_ohlcvs_minutes=0,
        markets={SYMBOL: {"id": "NEWUSDT", "symbol": SYMBOL,
                          "info": {"symbol": "NEWUSDT", field: LISTING}}},
        get_symbol=lambda _coin: SYMBOL, load_first_timestamp=lambda _coin: FIRST,
        fetch_leading_history_page=fetch, calls=calls)


def cached(tmp_path, exchange):
    catalog = OhlcvCatalog(tmp_path / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "ohlcvs", catalog)
    ts = np.arange(FIRST, END + 60_000, 60_000, dtype=np.int64)
    store.write_rows(exchange, "1m", SYMBOL, ts, np.tile(VALUES, (len(ts), 1)))
    catalog.mark_gap(exchange=exchange, timeframe="1m", symbol=SYMBOL, start_ts=START,
        end_ts=FIRST - 60_000, reason="leading_unavailable", persistent=True, retry_count=3,
        next_retry_at=END + 999999999999, note="confirmed_by_v2_fetch")
    return catalog, store


def expire(catalog, exchange):
    gap = catalog.get_persistent_gaps(exchange, "1m", SYMBOL, START, FIRST - 60_000)[0]
    catalog.mark_gap(exchange=exchange, timeframe="1m", symbol=SYMBOL,
        start_ts=gap.start_ts, end_ts=gap.end_ts, reason=gap.reason, persistent=True,
        retry_count=gap.retry_count, last_attempt_at=gap.last_attempt_at,
        next_retry_at=0, note=gap.note)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "bybit"])
@pytest.mark.parametrize("refresh", ["not_due", "expired", "forced"])
async def test_online_raw_resolution_honors_retry_and_accepts_just_refreshed_proof(
    tmp_path, monkeypatch, exchange, refresh,
):
    catalog, store = cached(tmp_path, exchange)
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(catalog=catalog, store=store, exchange=exchange, coin="NEW", symbol=SYMBOL,
                start_ts=START, end_ts=END, allow_remote_fetch=True,
                local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert await hp._resolve_v2_store_range(om=owner(exchange), **args) is not None
    if refresh == "expired":
        expire(catalog, exchange)
    fresh_owner = owner(exchange)
    fresh_owner.force_refetch_gaps = refresh == "forced"
    result = await hp._resolve_v2_store_range(om=fresh_owner, **args)
    assert result.timestamps[0] == FIRST and result.valid.all()
    assert len(fresh_owner.calls) == (0 if refresh == "not_due" else 3)
    gap = catalog.get_persistent_gaps(exchange, "1m", SYMBOL, START, FIRST - 60_000)[0]
    assert not hp._gap_retry_due(gap)
    if refresh == "forced":
        # This exercises inner refresh plus outer wrapper admission. The exact
        # new proof is accepted, without a second refresh loop or old-fact reuse.
        assert len(fresh_owner._leading_history_verified_notes) == 1
        expire(catalog, exchange)
        assert hp._confirmed_catalog_prefix_end(
            catalog, exchange, SYMBOL, START, FIRST - 60_000, om=fresh_owner,
            coin="NEW", control_values=VALUES, allow_remote_fetch=True) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["offline", "cache_only"])
async def test_expired_certificate_remains_available_without_remote_refresh(
    tmp_path, monkeypatch, mode,
):
    catalog, store = cached(tmp_path, "bybit")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(catalog=catalog, store=store, exchange="bybit", coin="NEW", symbol=SYMBOL,
                start_ts=START, end_ts=END, allow_remote_fetch=True,
                local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert await hp._resolve_v2_store_range(om=owner("bybit"), **args) is not None
    expire(catalog, "bybit")
    fresh_owner = owner("bybit")
    fresh_owner.force_refetch_gaps = True
    async def no_network(*args, **kwargs): pytest.fail("cache-only reuse must not probe")
    fresh_owner.fetch_leading_history_page = no_network
    monkeypatch.setattr(hp, "get_first_timestamps_unified", no_network)
    args["allow_remote_fetch"] = False
    with simulation_data_policy({"backtest": {"offline": mode == "offline"}}):
        result = await hp._resolve_v2_store_range(om=fresh_owner, **args)
    assert result.timestamps[0] == FIRST and result.valid.all()


@pytest.mark.asyncio
@pytest.mark.parametrize("forced", [False, True])
async def test_failed_due_refresh_cannot_fall_back_to_old_certificate(tmp_path, monkeypatch, forced):
    catalog, store = cached(tmp_path, "bybit")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(catalog=catalog, store=store, exchange="bybit", coin="NEW", symbol=SYMBOL,
                start_ts=START, end_ts=END, allow_remote_fetch=True,
                local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert await hp._resolve_v2_store_range(om=owner("bybit"), **args) is not None
    if not forced:
        expire(catalog, "bybit")
    old_note = catalog.get_persistent_gaps("bybit", "1m", SYMBOL, START, FIRST - 60_000)[0].note
    fresh_owner = owner("bybit", mode="mismatch")
    fresh_owner.force_refetch_gaps = forced
    async def no_ordinary_repair(**kwargs): return None
    monkeypatch.setattr(hp, "_fetch_invalid_windows_into_v2_store", no_ordinary_repair)
    assert await hp._resolve_v2_store_range(om=fresh_owner, **args) is None
    assert len(fresh_owner.calls) == 3
    assert catalog.get_persistent_gaps("bybit", "1m", SYMBOL, START, FIRST - 60_000)[0].note == old_note


@pytest.mark.asyncio
async def test_explicit_force_refetch_bypasses_automatic_prepared_cache(tmp_path, monkeypatch):
    import backtest
    from config.schema import get_expanded_config_prototype
    monkeypatch.chdir(tmp_path)
    config = get_expanded_config_prototype()
    config["backtest"]["offline"] = False
    calls = []
    def forbidden_cache(*args, **kwargs): pytest.fail("force refetch must bypass prepared cache")
    async def raw_preparation(config, exchange, *, force_refetch_gaps):
        calls.append((exchange, force_refetch_gaps))
        raise ValueError("raw preparation reached")
    monkeypatch.setattr(backtest, "load_hlcvs_data_override", lambda *args: None)
    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", forbidden_cache)
    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", raw_preparation)
    with pytest.raises(ValueError, match="raw preparation reached"):
        await backtest.prepare_hlcvs_mss(config, "binance", force_refetch_gaps=True)
    assert calls == [("binance", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "bybit"])
async def test_verified_source_absence_reuses_offline_without_inception_or_network(
    tmp_path, monkeypatch, exchange,
):
    catalog, store = cached(tmp_path, exchange)
    om = owner(exchange)
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(om=om, catalog=catalog, store=store, exchange=exchange, coin="NEW", symbol=SYMBOL,
                start_ts=START, end_ts=END, allow_remote_fetch=True,
                local_hit_log_label="hit", remote_fetch_log_label="fetch")
    result = await hp._resolve_v2_store_range(**args)
    assert result.timestamps[0] == FIRST and result.valid.all()
    assert len(om.calls) == 3
    gaps = catalog.get_persistent_gaps(exchange, "1m", SYMBOL, START, FIRST - 60_000)
    assert all(gap.reason == "leading_unavailable" for gap in gaps)
    attested = proof.from_note(gaps[0].note)
    assert attested["previous_gap_notes"][0]["note"] == "confirmed_by_v2_fetch"
    assert hp._venue_listing_boundary(om, exchange, "NEW", SYMBOL) == LISTING
    async def no_network(*args, **kwargs): pytest.fail("offline proof reuse must not fetch")
    om.fetch_leading_history_page = no_network
    monkeypatch.setattr(hp, "get_first_timestamps_unified", no_network)
    with simulation_data_policy({"backtest": {"offline": True}}):
        again = await hp._resolve_v2_store_range(**args)
    np.testing.assert_array_equal(again.values, result.values)
    np.testing.assert_array_equal(again.timestamps, result.timestamps)
    assert om.load_first_timestamp("NEW") == FIRST  # Observation/age facts were not rewritten.


@pytest.mark.asyncio
async def test_mismatching_control_preserves_old_notes_and_keeps_prefix_unknown(tmp_path, monkeypatch):
    catalog, store = cached(tmp_path, "bybit")
    om = owner("bybit", mode="mismatch")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    result = await hp._resolve_v2_store_range(om=om, catalog=catalog, store=store,
        exchange="bybit", coin="NEW", symbol=SYMBOL, start_ts=START, end_ts=END,
        allow_remote_fetch=True, local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert result is None
    assert len(om.calls) == 3
    assert all(g.note == "confirmed_by_v2_fetch" for g in
               catalog.get_persistent_gaps("bybit", "1m", SYMBOL, START, FIRST - 60_000))
    with simulation_data_policy({"backtest": {"offline": True}}), pytest.raises(OfflineDataError):
        await hp._resolve_v2_store_range(om=om, catalog=catalog, store=store,
            exchange="bybit", coin="NEW", symbol=SYMBOL, start_ts=START, end_ts=END,
            allow_remote_fetch=True, local_hit_log_label="hit", remote_fetch_log_label="fetch")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["listing", "control", "earlier"])
async def test_changed_source_facts_invalidate_offline_certificate(tmp_path, monkeypatch, change):
    catalog, store = cached(tmp_path, "bybit")
    om = owner("bybit")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    args = dict(om=om, catalog=catalog, store=store, exchange="bybit", coin="NEW", symbol=SYMBOL,
                start_ts=START, end_ts=END, allow_remote_fetch=True,
                local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert await hp._resolve_v2_store_range(**args) is not None
    if change == "listing":
        om.markets[SYMBOL]["info"]["launchTime"] += 60_000
    else:
        timestamp = FIRST if change == "control" else FIRST - 60_000
        values = VALUES.copy()
        values[3] += 1
        store.write_rows("bybit", "1m", SYMBOL, np.asarray([timestamp]), values[None, :])
    async def no_network(*args, **kwargs): pytest.fail("offline validation must not fetch")
    om.fetch_leading_history_page = no_network
    monkeypatch.setattr(hp, "get_first_timestamps_unified", no_network)
    with simulation_data_policy({"backtest": {"offline": True}}), pytest.raises(OfflineDataError):
        await hp._resolve_v2_store_range(**args)


@pytest.mark.asyncio
async def test_genuine_earlier_response_repairs_cache_without_absence_fact(tmp_path, monkeypatch):
    catalog, store = cached(tmp_path, "bybit")
    om = owner("bybit", mode="earlier")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    result = await hp._verify_source_leading_unavailable(
        om=om, catalog=catalog, store=store, exchange="bybit", coin="NEW", symbol=SYMBOL,
        start_ts=START, first_valid_ts=FIRST, control_values=VALUES,
    )
    assert result == "repaired" and len(om.calls) == 1
    repaired = store.read_range("bybit", "1m", SYMBOL, START, START)
    assert repaired.valid.all()
    np.testing.assert_array_equal(repaired.values[0], VALUES)
    assert all(proof.from_note(g.note) is None for g in
               catalog.get_persistent_gaps("bybit", "1m", SYMBOL, START, FIRST - 60_000))
    assert om.load_first_timestamp("NEW") == FIRST


@pytest.mark.asyncio
async def test_cache_only_request_does_not_collect_a_new_certificate(tmp_path, monkeypatch):
    catalog, store = cached(tmp_path, "bybit")
    om = owner("bybit")
    async def first_timestamps(*args, **kwargs): return {"NEW": FIRST}
    monkeypatch.setattr(hp, "get_first_timestamps_unified", first_timestamps)
    result = await hp._resolve_v2_store_range(om=om, catalog=catalog, store=store,
        exchange="bybit", coin="NEW", symbol=SYMBOL, start_ts=START, end_ts=END,
        allow_remote_fetch=False, local_hit_log_label="hit", remote_fetch_log_label="fetch")
    assert result is None and not om.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["binance", "bybit"])
async def test_owner_dispatches_exact_public_request_and_offline_rejects_it(exchange):
    calls = []
    @asynccontextmanager
    async def locked(symbol, timeframe):
        calls.append(("lock", symbol, timeframe))
        yield
    async def backoff(): calls.append(("backoff",))
    async def spacing(**kwargs): calls.append(("spacing", kwargs))
    async def endpoint(params):
        calls.append(("request", deepcopy(params)))
        return envelope(exchange, [])
    query = proof.request(exchange, "NEWUSDT", START, START)
    cm = SimpleNamespace(_acquire_fetch_lock=locked, _apply_rate_limit_backoff=backoff,
                         _apply_remote_fetch_spacing=spacing)
    cc = SimpleNamespace(**{query["method"]: endpoint})
    om = SimpleNamespace(exchange=exchange, cm=cm, cc=cc, load_cc=lambda: None,
                         get_symbol=lambda coin: SYMBOL)
    response = await hp.HLCVManager.fetch_leading_history_page(om, SYMBOL, query)
    assert response == envelope(exchange, [])
    assert calls == [("lock", SYMBOL, "1m"), ("backoff",),
                     ("spacing", {"symbol": SYMBOL, "tf": "1m"}), ("request", query["params"])]
    calls.clear()
    with simulation_data_policy({"backtest": {"offline": True}}), pytest.raises(OfflineDataError):
        await hp.HLCVManager.fetch_leading_history_page(om, SYMBOL, query)
    assert not calls
