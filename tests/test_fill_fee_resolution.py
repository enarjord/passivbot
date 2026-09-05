from copy import deepcopy
from datetime import datetime, timezone

import pytest

import fill_events_manager as fem


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        ms = 1_700_000_000_000

        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(cls.ms / 1000, tz=tz or timezone.utc)

    monkeypatch.setattr(fem, "datetime", Clock)
    return Clock


class TickerApi:
    markets = {"BNB/USDT": {}}

    def __init__(self, clock):
        self.clock = clock
        self.calls = 0
        self.price = 300.0
        self.quote_age_ms = 0
        self.failed = False

    async def fetch_ticker(self, symbol):
        self.calls += 1
        if self.failed:
            raise RuntimeError("ticker unavailable")
        return {"last": self.price, "timestamp": self.clock.ms - self.quote_age_ms}


class FillFetcher(fem.BaseFetcher):
    def __init__(self, api, rows=()):
        self.api = api
        self.rows = list(rows)

    async def fetch(self, since_ms, until_ms, detail_cache, on_batch=None):
        return deepcopy(self.rows)


def fill(fees, timestamp, event_id="close"):
    return {
        "id": event_id,
        "timestamp": timestamp,
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "position_side": "long",
        "qty": 1.0,
        "price": 1000.0,
        "pnl": 2.0,
        "pb_order_type": "",
        "client_order_id": "",
        "fees": fees,
    }


def manager(tmp_path, api, rows=(), max_age_ms=86_400_000):
    return fem.FillEventsManager(
        exchange="fake",
        user="test",
        fetcher=FillFetcher(api, rows),
        cache_path=tmp_path,
        fee_conversion_max_age_ms=max_age_ms,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fees,expected_calls,expected_source",
    [
        (0, 0, fem.FEE_SOURCE_REPORTED_QUOTE),
        (0.0, 0, fem.FEE_SOURCE_REPORTED_QUOTE),
        ("0", 0, fem.FEE_SOURCE_REPORTED_QUOTE),
        ({"currency": "USDT", "cost": 0.0}, 0, fem.FEE_SOURCE_REPORTED_QUOTE),
        ({"currency": "USDT", "totalFee": "0"}, 0, fem.FEE_SOURCE_REPORTED_QUOTE),
        ({"currency": "BNB", "cost": 0.0}, 0, fem.FEE_SOURCE_REPORTED_CONVERTED),
        (
            [{"currency": "USDT", "cost": 0.1}, {"currency": "USDT", "cost": -0.1}],
            0,
            fem.FEE_SOURCE_REPORTED_QUOTE,
        ),
        (
            [{"currency": "BNB", "cost": 0.001}, {"currency": "BNB", "cost": -0.001}],
            1,
            fem.FEE_SOURCE_REPORTED_CONVERTED,
        ),
        (
            [{"currency": "BNB", "cost": 0.001}, {"currency": "USDT", "cost": -0.3}],
            1,
            fem.FEE_SOURCE_REPORTED_CONVERTED,
        ),
    ],
)
async def test_zero_fee_and_resolved_zero_sum_survive_refresh_and_reload(
    tmp_path, clock, fees, expected_calls, expected_source
):
    api = TickerApi(clock)
    fills = manager(tmp_path, api, [fill(fees, clock.ms)])

    await fills.refresh()
    event = fills.get_events()[0]
    assert event.fee_paid == pytest.approx(0.0)
    assert event.fee_source == expected_source
    assert fills.get_pnl_sum() == pytest.approx(2.0)
    assert api.calls == expected_calls

    reloaded = manager(tmp_path, api)
    await reloaded.ensure_loaded()
    assert reloaded.get_events()[0].fee_source == expected_source
    assert reloaded.get_pnl_sum() == pytest.approx(2.0)
    assert api.calls == expected_calls


@pytest.mark.parametrize(
    "fees",
    [None, {}]
    + [{"currency": "USDT", "cost": value} for value in (None, "", "nan", "inf", False)],
)
def test_missing_or_unusable_fee_is_not_reported_zero(fees):
    fee_paid, metadata = fem._normalize_fee_paid_from_payload(fill(fees, 1_700_000_000_000))
    assert fee_paid == pytest.approx(-0.2)
    assert metadata["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT


@pytest.mark.asyncio
@pytest.mark.parametrize("quote_age_ms,advance_ms", [(0, 10_001), (8000, 3000)])
async def test_cached_conversion_expires_by_fetch_and_quote_age(
    tmp_path, clock, quote_age_ms, advance_ms
):
    api = TickerApi(clock)
    api.quote_age_ms = quote_age_ms
    fills = manager(tmp_path, api, max_age_ms=10_000)
    fees = {"currency": "BNB", "cost": 0.001}
    fills.fetcher.rows = [fill(fees, clock.ms, "first")]
    await fills.refresh()
    assert fills.get_events()[0].fee_paid == pytest.approx(-0.3)
    assert api.calls == 1

    clock.ms += advance_ms
    api.price = 600.0
    api.quote_age_ms = 0
    fills.fetcher.rows = [fill(fees, clock.ms, "second"), fill(fees, clock.ms, "third")]
    await fills.refresh()
    assert [event.fee_paid for event in fills.get_events()] == pytest.approx([-0.3, -0.6, -0.6])
    assert api.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("max_age_ms,retry_ms", [(10_000, 10_000), (120_000, 60_000), (0, 0)])
async def test_failed_conversion_recovers_after_bounded_retry_without_batch_duplicates(
    tmp_path, clock, max_age_ms, retry_ms
):
    api = TickerApi(clock)
    api.failed = True
    fills = manager(tmp_path, api, max_age_ms=max_age_ms)
    fees = {"currency": "BNB", "cost": 0.001}
    first_batch = [fill(fees, clock.ms, str(i)) for i in range(3)]
    await fills._apply_fee_policy_to_batch(first_batch)
    assert api.calls == 1
    assert all(row["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT for row in first_batch)

    api.failed = False
    if retry_ms:
        clock.ms += retry_ms - 1
        assert await fills._fee_conversion_rate("BNB", "USDT", clock.ms) is None
        assert api.calls == 1
        clock.ms += 1

    second_batch = [fill(fees, clock.ms, str(i)) for i in range(3)]
    await fills._apply_fee_policy_to_batch(second_batch)
    assert api.calls == 2
    assert all(row["fee_source"] == fem.FEE_SOURCE_REPORTED_CONVERTED for row in second_batch)
    assert [row["fee_paid"] for row in second_batch] == pytest.approx([-0.3] * 3)


@pytest.mark.asyncio
async def test_cached_quote_is_rechecked_against_each_fill_timestamp(tmp_path, clock):
    api = TickerApi(clock)
    api.quote_age_ms = 9000
    fills = manager(tmp_path, api, max_age_ms=10_000)
    assert await fills._fee_conversion_rate("BNB", "USDT", clock.ms) == 300.0
    # Both the quote and fill are near now, but are too far apart from each other.
    api.price = 600.0
    api.quote_age_ms = 0
    assert await fills._fee_conversion_rate("BNB", "USDT", clock.ms + 9000) == 600.0
    assert api.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("price", [float("inf"), float("nan")])
async def test_nonfinite_quote_does_not_become_a_converted_fee(tmp_path, clock, price):
    api = TickerApi(clock)
    api.price = price
    fills = manager(tmp_path, api)
    assert await fills._fee_conversion_rate("BNB", "USDT", clock.ms) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("first_is_old", [True, False], ids=["old-fill-first", "new-fill-first"])
async def test_fill_specific_quote_rejection_does_not_poison_pair_cache(
    tmp_path, clock, first_is_old
):
    api = TickerApi(clock)
    # Each timestamp is individually within the age limit relative to now;
    # the old fill and the quote are too far apart from each other.
    api.quote_age_ms = -2000
    fills = manager(tmp_path, api, max_age_ms=10_000)
    fee = {"currency": "BNB", "cost": 0.001}
    old = fill(fee, clock.ms - 9000, "old")
    recent = fill(fee, clock.ms, "recent")
    batch = [old, recent] if first_is_old else [recent, old]
    await fills._apply_fee_policy_to_batch(batch)

    assert old["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT
    assert recent["fee_source"] == fem.FEE_SOURCE_REPORTED_CONVERTED
    assert recent["fee_paid"] == pytest.approx(-0.3)
    # A following current fill still reuses the pair's positive quote.
    subsequent = fill(fee, clock.ms, "subsequent")
    await fills._apply_fee_policy_to_batch([subsequent])
    assert subsequent["fee_paid"] == pytest.approx(-0.3)
    assert api.calls == (1 if first_is_old else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_cost", [None, "", "invalid", "nan", "inf", False])
@pytest.mark.parametrize("valid_first", [True, False], ids=["valid-first", "invalid-first"])
async def test_coalescing_preserves_unresolved_fee_through_refresh_and_reload(
    tmp_path, clock, invalid_cost, valid_first
):
    invalid = fill({"currency": "USDT", "cost": invalid_cost}, clock.ms, "invalid")
    valid = fill({"currency": "USDT", "cost": 0}, clock.ms, "valid")
    rows = [valid, invalid] if valid_first else [invalid, valid]
    coalesced = fem._coalesce_events(rows)
    assert len(coalesced) == 1
    api = TickerApi(clock)
    fills = manager(tmp_path, api, coalesced)
    await fills.refresh()
    event = fills.get_events()[0]
    assert event.fee_source == fem.FEE_SOURCE_FALLBACK_PCT
    assert event.fee_paid == pytest.approx(-0.4)
    assert fills.get_pnl_sum() == pytest.approx(3.6)
    reloaded = manager(tmp_path, api)
    await reloaded.ensure_loaded()
    assert reloaded.get_events()[0].fee_source == fem.FEE_SOURCE_FALLBACK_PCT
    assert reloaded.get_pnl_sum() == pytest.approx(3.6)


@pytest.mark.asyncio
async def test_coalescing_preserves_missing_cost_and_explicit_zero_distinction(tmp_path, clock):
    api = TickerApi(clock)
    rows = fem._coalesce_events(
        [
            fill({"currency": "USDT"}, clock.ms, "missing"),
            fill({"currency": "USDT", "cost": 0}, clock.ms + 1, "zero"),
        ]
    )
    fills = manager(tmp_path, api, rows)
    await fills.refresh()
    events = {event.id: event for event in fills.get_events()}
    assert events["missing"].fee_source == fem.FEE_SOURCE_FALLBACK_PCT
    assert events["missing"].fee_paid == pytest.approx(-0.2)
    assert events["zero"].fee_source == fem.FEE_SOURCE_REPORTED_QUOTE
    assert events["zero"].fee_paid == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["bybit", "okx"])
@pytest.mark.parametrize("cost", [None, "", "invalid", "nan", "inf", False, 0.0])
async def test_exchange_fee_producers_preserve_unknown_vs_zero_after_coalescing(
    tmp_path, clock, exchange, cost
):
    api = TickerApi(clock)
    if exchange == "bybit":
        row = fem.BybitFetcher._normalize_trade(
            {
                "id": "trade",
                "timestamp": clock.ms,
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "amount": 1.0,
                "price": 1000.0,
                "pnl": 2.0,
                "fee": {"currency": "USDT", "cost": cost},
                "info": {"closedSize": 1.0},
            }
        )
    else:
        row = fem.OkxFetcher(api)._normalize_fill(
            {
                "tradeId": "trade",
                "ts": clock.ms,
                "instId": "BTC-USDT-SWAP",
                "side": "sell",
                "fillSz": 1.0,
                "fillPx": 1000.0,
                "fillPnl": 2.0,
                "posSide": "long",
                "feeCcy": "USDT",
                "fee": cost,
            }
        )
    fills = manager(tmp_path, api, fem._coalesce_events([row]))
    await fills.refresh()
    event = fills.get_events()[0]
    exact_zero = isinstance(cost, float) and cost == 0.0
    assert event.fee_source == (
        fem.FEE_SOURCE_REPORTED_QUOTE if exact_zero else fem.FEE_SOURCE_FALLBACK_PCT
    )
    assert event.fee_paid == pytest.approx(0.0 if exact_zero else -0.2)
    assert fills.get_pnl_sum() == pytest.approx(2.0 if exact_zero else 1.8)


@pytest.mark.parametrize("cost", [None, "", "invalid", "nan", "inf", False])
def test_bybit_overlap_fee_normalization_does_not_fabricate_zero(cost):
    fee = fem.FillEventsManager._extract_bybit_fee_from_trade_row(
        {"fee": {"currency": "USDT", "cost": cost}}
    )
    coalesced = fem._coalesce_events([fill(fee, 1_700_000_000_000)])
    fee_paid, metadata = fem._normalize_fee_paid_from_payload(coalesced[0])
    assert fee_paid == pytest.approx(-0.2)
    assert metadata["fee_source"] == fem.FEE_SOURCE_FALLBACK_PCT
