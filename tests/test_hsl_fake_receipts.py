"""Offline acquisition clocks retain receipt identity and actual UTC freshness."""

import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from live.hsl_candles import Sources, CandleSourceReader
from live.hsl_inputs import capture_candles
from live.hsl_live import Owner
from live.hsl_runtime import capture, observe_fills
from live.market_snapshot import MarketSnapshot
from tools.run_fake_live import _FakeHslFactClocks, _refuse_unsettled_fake_bar
from test_hsl_runtime import NOW, SYMBOL, bot


@pytest.mark.parametrize("kind", ["quote", "candle", "fill_start", "fill_end"])
def test_receipt_identity_distinguishes_equal_utc_reads_in_different_fake_bars(kind):
    clocks = _FakeHslFactClocks()
    old = MarketSnapshot(SYMBOL, 100., 100., 100., NOW, "fake")
    new = replace(old)
    clocks.record(kind, old, (NOW,), NOW-60_000)
    clocks.record(kind, new, (NOW,), NOW)
    assert clocks.project(kind, old, NOW) == NOW-60_000
    assert clocks.project(kind, new, NOW) == NOW
    with pytest.raises(ValueError, match="retime"):
        clocks.record(kind, old, (NOW,), NOW)
    with pytest.raises(ValueError, match="unknown"):
        clocks.project(kind, replace(old), NOW)
    with pytest.raises(ValueError, match="unknown"):
        clocks.project(kind, old, NOW+1)


def test_capture_projection_keeps_old_facts_fixed_and_quote_ttl_in_actual_utc():
    value = bot()
    value.get_exchange_time = lambda: NOW
    clocks = _FakeHslFactClocks()
    quote = MarketSnapshot(SYMBOL, 100., 100., 100., NOW-1000, "fake")
    tape = capture_candles([
        dict(ts=NOW-120_000, o=100., h=100., l=100., c=100.)
    ], minutes=1, observed_at=NOW-1000)
    source = Sources((tape,), (), 0)
    value._hsl_fill_observation = observe_fills(value, (NOW-1002, NOW-1000))
    receipt = value._hsl_fill_observation
    clocks.record("quote", quote, (quote.fetched_ms,), NOW-60_000)
    clocks.record("candle", tape, (NOW-1000,), NOW-60_000)
    clocks.record("fill_start", receipt, (NOW-1002,), NOW-60_000)
    clocks.record("fill_end", receipt, (NOW-1000,), NOW-60_000)

    def read(utc):
        requests, missing = capture(
            value, {SYMBOL: quote}, {SYMBOL: source},
            symbols={"long": [SYMBOL], "short": []}, now_ms=NOW,
            utc_now_ms=utc, max_current_age_ms=120_000,
            observation_clock_offset_ms=NOW-utc,
            account_clock_projector=lambda *args: NOW,
            fact_clock_projector=clocks.project, use_observed_fills=True,
        )
        assert not missing and len(requests) == 1
        return requests[0]

    first, second = read(NOW+1), read(NOW+2)
    assert first.metadata == second.metadata
    pair = json.loads(first.metadata)["snapshot"]["pairs"][0]
    assert pair["mark_at"] == pair["fills_at"] == NOW-60_000
    assert first.mark_observed_ms == second.mark_observed_ms == (NOW-1000,)
    assert tape.candles[0].available_at == NOW-1000
    requests, missing = capture(
        value, {SYMBOL: quote}, {SYMBOL: source},
        symbols={"long": [SYMBOL], "short": []}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=2000,
        fact_clock_projector=clocks.project, account_clock_projector=lambda *args: NOW,
    )
    assert not requests
    assert len(missing) == 1 and missing[0].reason == "current_mark_unavailable"
    # A projector is not a UTC freshness renewal.
    for surface in ("balance", "positions"):
        value._ensure_freshness_ledger().stamp(surface, now_ms=NOW+121_000)
    _, missing = capture(
        value, {SYMBOL: quote}, {SYMBOL: source},
        symbols={"long": [SYMBOL], "short": []}, now_ms=NOW,
        utc_now_ms=NOW+121_000, max_current_age_ms=120_000,
        fact_clock_projector=clocks.project, account_clock_projector=lambda *args: NOW,
    )
    assert len(missing) == 1 and missing[0].reason == "current_mark_unavailable"


def test_capture_without_fact_hook_retains_utc_quote_admission_with_explicit_offset():
    value = bot()
    value.get_exchange_time = lambda: NOW + 1000
    quote = MarketSnapshot(SYMBOL, 100., 100., 100., NOW-100, "fake")
    requests, missing = capture(
        value, {SYMBOL: quote}, {},
        symbols={"long": [SYMBOL], "short": []}, now_ms=NOW+1000,
        utc_now_ms=NOW, max_current_age_ms=500,
        observation_clock_offset_ms=-1000,
        account_clock_projector=lambda *args: NOW+1000,
    )
    assert not missing and len(requests) == 1
    pair = json.loads(requests[0].metadata)["snapshot"]["pairs"][0]
    assert pair["mark_at"] == quote.fetched_ms - 1000
    assert NOW + 1000 - pair["mark_at"] > 500
    assert requests[0].mark_observed_ms == (quote.fetched_ms,)


@pytest.mark.parametrize("jitter", [0, 1])
def test_flat_capture_skew_uses_projected_receipts_only_when_hooked(jitter):
    value = bot()
    value.positions = {}
    value.get_exchange_time = lambda: NOW
    value._ensure_freshness_ledger().stamp("positions", now_ms=NOW-200-jitter)
    value._hsl_fill_observation = observe_fills(value, (NOW-201, NOW-200))
    receipt = value._hsl_fill_observation
    clocks = _FakeHslFactClocks()
    clocks.record("fill_start", receipt, (NOW-201,), NOW)
    clocks.record("fill_end", receipt, (NOW-200,), NOW)
    arguments = dict(
        symbols={"long": [SYMBOL], "short": []}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000,
        use_observed_fills=True, account_clock_projector=lambda *args: NOW,
    )
    projected, missing = capture(
        value, {}, {}, fact_clock_projector=clocks.project, **arguments
    )
    assert not missing and len(projected) == 1
    assert "snapshot_skew" not in projected[0].reasons
    original, missing = capture(value, {}, {}, **arguments)
    assert not missing and len(original) == 1
    assert ("snapshot_skew" in original[0].reasons) is bool(jitter)


@pytest.mark.asyncio
async def test_source_cohort_is_owned_before_a_scheduled_close_can_remove_exposure():
    value = bot()
    value.get_exchange_time = lambda: NOW
    instance = Owner(value)
    instance.history_symbols = lambda: set()
    acquired = []

    async def acquire(symbol, **kwargs):
        acquired.append(symbol)
        return Sources((), (), 0)

    instance._reader = SimpleNamespace(acquire=acquire)
    instance.schedule_sources()
    value.positions = {}
    await instance._source_task
    assert acquired == [SYMBOL]
    assert set(instance.sources) == {SYMBOL}


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [.001, .020])
async def test_reader_records_each_actual_immutable_tape_after_a_delayed_read(delay):
    from test_hsl_candles import rows

    returned = rows(0, 1)

    async def get_candles(*args, **kwargs):
        await asyncio.sleep(delay)
        return returned

    clocks, receipts = _FakeHslFactClocks(), []

    def observe(tape):
        receipts.append(tape)
        clocks.record("candle", tape, (c.available_at for c in tape.candles), NOW)

    manager = SimpleNamespace(exchange=SimpleNamespace(timeframes={"1m": "1m"}),
                              get_candles=get_candles)
    reader = CandleSourceReader(manager, observation_clock=lambda: NOW-2,
                               receipt_observer=observe)
    result = await reader.acquire(SYMBOL, start=0, end=120_000,
                                  timeout_seconds=.25, allow_remote_fetch=False)
    assert len(receipts) == len(result.tapes) == 1
    assert receipts[0] is result.tapes[0]
    assert all(c.available_at == NOW-2 for c in result.tapes[0].candles)
    assert clocks.project("candle", result.tapes[0], NOW-2) == NOW
    assert [c.close for c in result.tapes[0].candles] == [100., 101.]


def test_inflight_input_refuses_market_advance_and_retains_explicit_partial_report(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    summaries = [dict(timestamp=NOW, result="pending")]
    with pytest.raises(RuntimeError, match="cannot advance"):
        _refuse_unsettled_fake_bar(dict(input_acquisition_pending=True), summaries, snapshots)
    saved = json.loads((tmp_path / "incomplete.json").read_text())
    assert saved["step_summaries"] == summaries
    assert saved["market_clock_advanced_after_pending"] is False
    # Ordinary-only pending remains bounded and may service the next scenario step.
    _refuse_unsettled_fake_bar(dict(preparation_pending=True), summaries, snapshots)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", [None, "source", "quote", "retired_quote", "provider", "reader"])
@pytest.mark.parametrize("raises", [False, True])
async def test_account_outage_cannot_bypass_an_owned_input_acquisition(channel, raises, tmp_path):
    from ccxt.base.errors import NetworkError
    from live.hsl_live import owner
    from tools.run_fake_live import _run_fake_cycle_ready

    async def unavailable(**kwargs):
        if raises:
            raise NetworkError("offline injected account outage")
        return False

    value = SimpleNamespace(config={"live": {}},
                            refresh_protective_authoritative_state=unavailable,
                            _begin_live_event_cycle=lambda **kwargs: None)
    instance = owner(value)
    release = asyncio.Event()
    task = asyncio.create_task(release.wait()) if channel else None
    if channel == "source":
        instance._source_task = task
    elif channel == "quote":
        instance._quote_tasks[SYMBOL] = task
    elif channel == "retired_quote":
        instance._discard_quote_tasks.add(task)
    elif channel == "provider":
        value.market_snapshot_provider = SimpleNamespace(_fetch_task=task)
    elif channel == "reader":
        instance._reader = SimpleNamespace(_pending={SYMBOL: task})
    try:
        result = await asyncio.wait_for(_run_fake_cycle_ready(value), 4.)
        assert result["updated"] is False
        if raises:
            assert result["current_io_unavailable"]
            assert "ordinary_executed" not in result
        if channel:
            assert result["input_acquisition_pending"]
            assert result["passes"] == 8
            with pytest.raises(RuntimeError, match="cannot advance"):
                _refuse_unsettled_fake_bar(result, [result], tmp_path / "snapshots")
        else:
            assert result["passes"] == 1
            _refuse_unsettled_fake_bar(result, [result], None)
    finally:
        release.set()
        if task is not None:
            await task


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["reader", "provider"])
async def test_final_wait_inspects_a_new_underlying_read_after_its_wrapper_finishes(
    monkeypatch, channel, tmp_path
):
    from live import hsl_live
    from tools.run_fake_live import _run_fake_cycle_ready

    calls, children = 0, []
    release = asyncio.Event()
    reader = SimpleNamespace(_pending={})
    provider = SimpleNamespace(_symbol_fetch_tasks={})

    async def wrapper():
        child = asyncio.create_task(release.wait())
        children.append(child)
        if channel == "reader":
            reader._pending[SYMBOL] = child
        else:
            provider._symbol_fetch_tasks[SYMBOL] = child

    async def cycle():
        nonlocal calls
        calls += 1
        if calls == 8:
            instance._source_task = asyncio.create_task(wrapper())
        return dict(updated=True, ordinary_executed=False)

    instance = SimpleNamespace(
        cycle=cycle, _ordinary=None, _source_task=None,
        _quote_tasks={}, _discard_quote_tasks=set(), _reader=reader,
    )
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    value = SimpleNamespace(market_snapshot_provider=provider)
    try:
        result = await _run_fake_cycle_ready(value)
        assert calls == 8 and instance._source_task.done()
        assert len(children) == 1 and not children[0].done()
        assert result["input_acquisition_pending"]
        with pytest.raises(RuntimeError, match="cannot advance"):
            _refuse_unsettled_fake_bar(result, [result], tmp_path / "snapshots")
    finally:
        release.set()
        await asyncio.gather(*children)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["source", "provider"])
@pytest.mark.parametrize("finishes", [False, True])
async def test_skipped_initial_cycle_settles_or_refuses_boot_inputs_without_trading(
    monkeypatch, channel, finishes, tmp_path
):
    from exchanges.fake import FakeCCXTClient
    from tools import run_fake_live as runner
    from test_run_fake_live import _scenario

    release, reads, trading = asyncio.Event(), [], []

    async def read():
        reads.append(True)
        if not finishes:
            await release.wait()

    task = asyncio.create_task(read())
    instance = SimpleNamespace(_quote_tasks={}, _discard_quote_tasks=set())
    provider = SimpleNamespace(_fetch_task=task if channel == "provider" else None)
    if channel == "source":
        instance._source_task = task
    value = SimpleNamespace(_hsl_live=instance, market_snapshot_provider=provider)
    client = FakeCCXTClient(_scenario())
    initial_index = client.current_index

    async def cycle(bot):
        assert task.done() and client.current_index == initial_index+1
        trading.append(True)
        return dict(updated=True, ordinary_executed=True)

    monkeypatch.setattr(runner, "_run_fake_cycle", cycle)
    try:
        if finishes:
            result = await runner._run_fake_bot(
                value, client, 1, run_initial_cycle=False
            )
            assert len(result) == len(trading) == 1
        else:
            with pytest.raises(RuntimeError, match="cannot advance"):
                await runner._run_fake_bot(
                    value, client, 1, snapshot_dir=tmp_path / "snapshots",
                    run_initial_cycle=False,
                )
            assert client.current_index == initial_index and not trading
            assert json.loads((tmp_path / "incomplete.json").read_text())["step_summaries"] == []
        assert reads == [True]
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_fake_quote_cache_requires_both_original_utc_and_original_scenario_freshness():
    from ccxt.base.errors import NetworkError
    from exchanges.fake import FakeCCXTClient
    from live.market_snapshot import MarketSnapshotProvider, MarketSnapshotUnavailable
    from tools.run_fake_live import _install_runtime_overrides
    from test_run_fake_live import _scenario
    from utils import utc_ms

    calls = []

    async def broken_fetch():
        calls.append(True)
        raise NetworkError("offline injected quote outage")

    client = FakeCCXTClient(_scenario())
    provider = MarketSnapshotProvider(exchange_name="fake", fetch_tickers=broken_fetch)
    value = SimpleNamespace(cca=client, market_snapshot_provider=provider,
                            _record_authoritative_surface=lambda *args: None)
    _install_runtime_overrides(value, {})
    symbol, actual_utc = client.symbols[0], int(utc_ms())
    ticker = dict(last=100., bid=100., ask=100.)
    first_bar = client.now_ms
    old = provider._snapshot_from_ticker(symbol, ticker, fetched_ms=actual_utc)
    provider._cache[symbol] = old
    assert provider.get_cached(symbol, now_ms=actual_utc, max_age_ms=1000) is old
    client.now_ms += 60_000
    assert provider.get_cached(symbol, now_ms=actual_utc, max_age_ms=1000) is None
    with pytest.raises((NetworkError, MarketSnapshotUnavailable)):
        await provider.get_snapshots([symbol], max_age_ms=1000)
    assert calls == [True]
    assert value._hsl_fact_clock_projector("quote", old, old.fetched_ms) == first_bar
    # Equal UTC millisecond does not identify a new synthetic-bar receipt.
    new = provider._snapshot_from_ticker(symbol, ticker, fetched_ms=actual_utc)
    provider._cache[symbol] = new
    assert new is not old and new.fetched_ms == old.fetched_ms
    assert provider.get_cached(symbol, now_ms=actual_utc, max_age_ms=1000) is new
    assert value._hsl_fact_clock_projector("quote", new, new.fetched_ms) == client.now_ms
    assert provider.get_cached(symbol, now_ms=actual_utc+1001, max_age_ms=1000) is None
    provider.begin_shutdown()
    await provider.wait_pending()


def test_fake_retry_clock_moves_only_required_deadlines_and_respects_per_bar_cap():
    from exchanges.fake import FakeCCXTClient
    from live.hsl_live import owner
    from tools.run_fake_live import _install_runtime_overrides
    from test_run_fake_live import _scenario

    client = FakeCCXTClient(_scenario())
    value = SimpleNamespace(cca=client, _record_authoritative_surface=lambda *args: None)
    _install_runtime_overrides(value, {})
    instance = owner(value)
    beginning = instance._schedule_clock()
    instance._next_history = beginning + 5.
    instance.required_fill_facts = lambda: None
    for _ in range(4):
        value._advance_fake_hsl_readiness(instance)
    assert instance._schedule_clock() == beginning
    instance.required_fill_facts = lambda: ((), False)
    for _ in range(5):
        instance._next_history = instance._schedule_clock() + 5.
        value._advance_fake_hsl_readiness(instance)
    assert instance._schedule_clock() == beginning + 15.
    assert client.now_ms / 1000. == beginning


@pytest.mark.asyncio
@pytest.mark.parametrize("crosses_bar", [False, True])
async def test_fill_projection_requires_the_actual_new_same_bar_observation(crosses_bar):
    from exchanges.fake import FakeCCXTClient
    from live.hsl_inputs import FillTape
    from live.hsl_runtime import FillObservation
    from tools.run_fake_live import _install_runtime_overrides
    from test_run_fake_live import _scenario

    client = FakeCCXTClient(_scenario())
    value = SimpleNamespace(cca=client, _record_authoritative_surface=lambda *args: None)
    created = []

    async def actual_refresh(**kwargs):
        receipt = FillObservation(object(), 0, FillTape((), ()), (NOW, NOW))
        value._hsl_fill_observation = receipt
        created.append(receipt)
        if crosses_bar:
            client.now_ms += 60_000
        return True

    value.update_pnls = actual_refresh
    _install_runtime_overrides(value, {})
    initial = client.now_ms
    if crosses_bar:
        with pytest.raises(ValueError, match="crossed an unsettled"):
            await value.update_pnls(source="hsl")
        with pytest.raises(ValueError, match="unknown"):
            value._hsl_fact_clock_projector("fill_end", created[0], NOW)
    else:
        assert await value.update_pnls(source="hsl")
        client.now_ms += 60_000
        assert await value.update_pnls(source="hsl")
        for index, receipt in enumerate(created):
            for kind in ("fill_start", "fill_end"):
                assert value._hsl_fact_clock_projector(kind, receipt, NOW) == initial + index*60_000
