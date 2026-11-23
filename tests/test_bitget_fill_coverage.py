import json
from types import SimpleNamespace

import pytest

from fill_events_manager import BitgetFetcher, FillEventsManager, PNL_CONTRACT_CURRENT
from live.state_refresh import AuthoritativeSurfaceUnavailable


DAY = 86_400_000


def fill(uta, identifier, timestamp=1000):
    shared = {
        "symbol": "BTCUSDT",
        "tradeSide": "open",
        "side": "buy",
        "clientOid": "external",
    }
    if uta:
        return {
            **shared,
            "execId": str(identifier),
            "orderId": "order",
            "createdTime": str(timestamp),
            "execQty": "1",
            "execPrice": "100",
            "execPnl": "0",
        }
    return {
        **shared,
        "posMode": "hedge_mode",
        "tradeId": str(identifier),
        "cTime": str(timestamp),
        "baseVolume": "1",
        "price": "100",
        "profit": "0",
    }


def page(uta, rows, cursor=None):
    data = {"list" if uta else "fillList": rows}
    if cursor is not None:
        data["cursor" if uta else "endId"] = cursor
    return {"code": "00000", "data": data}


def make_fetcher(uta, pages, limit=2, now=1000):
    calls = []

    async def fetch(params):
        calls.append(params)
        return pages.pop(0)

    api = SimpleNamespace(
        options={"uta": uta},
        private_uta_get_v3_trade_fills=fetch,
        private_mix_get_v2_mix_order_fill_history=fetch,
    )
    return (
        BitgetFetcher(
            api,
            symbol_resolver=lambda symbol: symbol,
            history_limit=limit,
            now_func=lambda: now,
        ),
        calls,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"code": "00000"},
        {"code": "00000", "data": None},
        {"code": "00000", "data": {}},
        {"code": "00000", "data": []},
        "null_rows",
        "wrong_rows",
        "error_code",
        "missing_code",
    ],
)
async def test_malformed_fill_page_cannot_certify_empty_window(tmp_path, uta, bad):
    if bad == "null_rows":
        bad = page(uta, None)
    elif bad == "wrong_rows":
        bad = page(uta, {})
    elif bad == "error_code":
        bad = {**page(uta, []), "code": "40000"}
    elif bad == "missing_code":
        bad = {"data": page(uta, [])["data"]}
    fetcher, _ = make_fetcher(uta, [bad])
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await manager.refresh_for_lookback(100)
    assert not manager.get_coverage_status(start_ms=100)["ready"]
    assert manager.cache.load_metadata()["last_refresh_ms"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
async def test_saturated_boundary_preserves_all_executions_at_requested_start(uta):
    rows = [fill(uta, i) for i in (7, 6, 5, 4, 3)]
    fetcher, calls = make_fetcher(
        uta, [page(uta, rows[:2], "6"), page(uta, rows[2:4], "4"), page(uta, rows[4:])]
    )
    fills = await fetcher.fetch(1000, 1000, {})
    assert {f["id"] for f in fills} == {"7", "6", "5", "4", "3"}
    assert all(c["startTime"] == c["endTime"] == 1000 for c in calls)
    assert [c.get("cursor" if uta else "idLessThan") for c in calls] == [None, "6", "4"]


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
@pytest.mark.parametrize("invalid", [None, "", "  ", " 7", False, 123, [], {}])
async def test_full_page_requires_wire_cursor(uta, invalid):
    fetcher, calls = make_fetcher(
        uta, [page(uta, [fill(uta, 10), fill(uta, 9)], invalid)]
    )
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="cursor|endId"):
        await fetcher.fetch(1000, 1000, {})
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
@pytest.mark.parametrize("repeated", ["cursor", "rows"])
async def test_pagination_rejects_stalled_pages(uta, repeated):
    initial = [fill(uta, 10), fill(uta, 9)]
    second = initial if repeated == "rows" else [fill(uta, 8), fill(uta, 7)]
    pages = [
        page(uta, initial, "9"),
        page(uta, second, "7" if repeated == "rows" else "9"),
    ]
    fetcher, _ = make_fetcher(uta, pages)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="progress|repeated"):
        await fetcher.fetch(1000, 1000, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("uta,window", [(False, 7 * DAY), (True, 30 * DAY)])
async def test_empty_window_still_checks_singleton_at_inclusive_start(uta, window):
    pages = [page(uta, []), page(uta, [fill(uta, 1)])]
    fetcher, calls = make_fetcher(uta, list(reversed(pages)) if uta else pages)
    assert [event["id"] for event in await fetcher.fetch(1000, 1000 + window, {})] == [
        "1"
    ]
    expected = [(1001, 1000 + window), (1000, 1000)]
    assert [(c["startTime"], c["endTime"]) for c in calls] == (
        list(reversed(expected)) if uta else expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["all", "window"])
async def test_old_bitget_coverage_needs_new_proof_after_restart(tmp_path, scope):
    metadata = {
        "pnl_contract": PNL_CONTRACT_CURRENT,
        "covered_start_ms": 80,
        "history_scope": scope,
        "last_refresh_ms": 999,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    fetcher, calls = make_fetcher(False, [page(False, [])])
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await manager.ensure_loaded()
    assert not manager.get_coverage_status(start_ms=100)["ready"]
    assert manager.cache.load_metadata()["last_refresh_ms"] == 0
    await manager.refresh_for_lookback(100)
    assert [(c["startTime"], c["endTime"]) for c in calls] == [(100, 1000)]
    restarted = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert restarted.get_coverage_status(start_ms=100)["ready"]
    assert not restarted.get_coverage_status(start_ms=80)["ready"]
    assert restarted.cache.load_metadata()["last_refresh_ms"] > 0


@pytest.mark.asyncio
async def test_reproof_keeps_cached_rows_and_adds_previously_missed_fill(tmp_path):
    fetcher, _ = make_fetcher(False, [page(False, [fill(False, 1)])])
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await manager.refresh_for_lookback(100)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["coverage_contract"]
    path.write_text(json.dumps(metadata))
    fetcher, calls = make_fetcher(False, [page(False, [fill(False, 2)])])
    restarted = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert [event.id for event in restarted.get_events()] == ["1"]
    assert not restarted.get_coverage_status(start_ms=100)["ready"]
    await restarted.refresh_for_lookback(100)
    assert calls[0]["startTime"] == 100
    assert {event.id for event in restarted.get_events()} == {"1", "2"}
    assert restarted.get_coverage_status(start_ms=100)["ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
async def test_default_api_window_does_not_certify_lifetime(tmp_path, uta):
    fetcher, _ = make_fetcher(uta, [page(uta, [])])
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await manager.refresh_latest()
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="bounded lookback"):
        manager.set_history_scope("all")
    assert not manager.get_coverage_status(start_ms=None)["ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
async def test_short_page_with_continuation_is_exhausted(uta):
    fetcher, calls = make_fetcher(
        uta, [page(uta, [fill(uta, 3)], "3"), page(uta, [fill(uta, 2)])]
    )
    assert {ev["id"] for ev in await fetcher.fetch(500, 1000, {})} == {"2", "3"}
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
async def test_page_outside_queried_window_is_not_evidence(uta):
    fetcher, _ = make_fetcher(uta, [page(uta, [fill(uta, 1, timestamp=1001)])])
    with pytest.raises(
        AuthoritativeSurfaceUnavailable, match="outside requested window"
    ):
        await fetcher.fetch(500, 1000, {})


@pytest.mark.asyncio
async def test_classic_end_id_must_move_backwards():
    fetcher, _ = make_fetcher(
        False,
        [page(False, [fill(False, 9)], "9"), page(False, [fill(False, 10)], "10")],
        limit=1,
    )
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="older ID"):
        await fetcher.fetch(500, 1000, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("uta", [False, True])
@pytest.mark.parametrize("cursor", ["next", False, {}, " "])
async def test_empty_with_continuation_cannot_finish_traversal(uta, cursor):
    fetcher, _ = make_fetcher(uta, [page(uta, [], cursor)])
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="empty fill page"):
        await fetcher.fetch(100, 1000, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("end", [None, 85 * DAY])
async def test_expired_uta_prefix_fails_before_empty_pages_can_certify_it(
    tmp_path, end
):
    now = 200 * DAY
    fetcher, calls = make_fetcher(True, [page(True, [])] * 5, now=now)
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="90-day retention"):
        await manager.refresh_for_lookback(80 * DAY, end_ms=end)
    assert not calls
    assert not manager.get_coverage_status(start_ms=80 * DAY)["ready"]
    assert manager.cache.load_metadata()["last_refresh_ms"] == 0


@pytest.mark.asyncio
async def test_uta_cold_starter_then_restart_bridges_certified_prefix(tmp_path):
    now = 200 * DAY
    start = now - 90 * DAY
    fetcher, calls = make_fetcher(True, [page(True, [])] * 4, now=now + 17)
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    from passivbot import Passivbot
    from test_realized_loss_gate import _set_memory_horizon

    bot = object.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.is_uta = True
    bot._pnls_manager = manager
    _set_memory_horizon(bot, minutes=90 * 1440, now_ms=now)
    assert bot._realized_memory_start_ms(now - 25) == start - 25
    assert getattr(bot, "_realized_memory_origin_ms", None) is None
    await manager.refresh_for_lookback(start, observed_at_ms=now)
    assert bot._realized_memory_start_ms(now) == start
    assert bot._realized_memory_start_ms(now + DAY) == start
    assert calls[0]["startTime"] == calls[0]["endTime"] == start
    assert max(call["endTime"] for call in calls) == now
    assert manager.cache.load_metadata()["covered_through_ms"] == now
    fetcher, calls = make_fetcher(
        True, [page(True, [fill(True, "late", now + 1)])], now=now + DAY
    )
    restarted = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await restarted.refresh_for_lookback(start)
    assert calls == [
        {"category": "USDT-FUTURES", "limit": 2, "startTime": now, "endTime": now + DAY}
    ]
    assert [event.id for event in restarted.get_events()] == ["late"]
    assert restarted.cache.get_covered_start_ms() == start
    assert restarted.cache.load_metadata()["covered_through_ms"] == now + DAY


@pytest.mark.asyncio
async def test_uta_retention_gap_cannot_be_bridged_by_unrelated_recent_refresh(
    tmp_path,
):
    old = 200 * DAY
    now = old + 100 * DAY
    start = old - DAY
    fetcher, _ = make_fetcher(True, [page(True, [])], now=old)
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await manager.refresh_for_lookback(start)
    fetcher, calls = make_fetcher(True, [page(True, [])] * 4, now=now)
    restarted = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert (
        restarted.get_coverage_status(start_ms=start)["reason"] == "uta_retention_gap"
    )
    assert restarted.get_coverage_status(start_ms=start, end_ms=old)["ready"]
    await restarted.refresh(start_ms=now - 1000, end_ms=now)
    assert restarted.cache.load_metadata()["covered_through_ms"] == old
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="90-day retention"):
        await restarted.refresh_for_lookback(start)
    assert len(calls) == 1
    await restarted.refresh_for_lookback(now - DAY)
    assert restarted.cache.get_covered_start_ms() == now - DAY
    assert not restarted.get_coverage_status(start_ms=start)["ready"]


@pytest.mark.asyncio
async def test_uta_incremental_overlap_ignores_retained_execution_before_certified_window(
    tmp_path,
):
    old = 100 * DAY
    fetcher, _ = make_fetcher(True, [page(True, [fill(True, "old", old)])], now=old)
    manager = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await manager.refresh_for_lookback(old)

    now = 300 * DAY
    start = now - 90 * DAY
    fetcher, calls = make_fetcher(True, [page(True, [])] * 10, now=now)
    restarted = FillEventsManager(
        exchange="bitget", user="fixture", fetcher=fetcher, cache_path=tmp_path
    )
    await restarted.refresh_for_lookback(start)
    assert [event.id for event in restarted.get_events()] == ["old"]
    assert restarted.cache.get_covered_start_ms() == start
    calls.clear()
    await restarted.refresh_for_lookback(start)
    assert calls
    assert [(c["startTime"], c["endTime"]) for c in calls] == [(now, now)]
    calls.clear()
    restarted.cache.load_metadata()["last_refresh_ms"] = now
    await restarted.refresh_latest(last_refresh_overlap_ms=200 * DAY)
    assert min(c["startTime"] for c in calls) == start
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="90-day retention"):
        await restarted.refresh_for_lookback(old)
