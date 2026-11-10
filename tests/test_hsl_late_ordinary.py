"""Only the post-protection owner may commit an ordinary preparation."""

import asyncio
from dataclasses import replace

import pytest

from live import market_snapshot
from live.hsl_candles import Sources
from live.hsl_runtime import observe_fills
from passivbot import Passivbot
from test_hsl_ordinary_freshness import planning_bot
from test_hsl_runtime import NOW, SYMBOL


OTHER = "OTHER/USDT:USDT"
SYMBOLS = sorted((SYMBOL, OTHER))
ACCOUNT_SURFACES = ("balance", "positions", "open_orders")


@pytest.fixture
def owner_handoff(monkeypatch):
    fixture = planning_bot(monkeypatch)
    bot, owner = fixture.bot, fixture.owner
    ledger = fixture.ledger = bot._ensure_freshness_ledger()
    fixture.events, fixture.quote_reads, fixture.captures = [], [], []
    fixture.raw_balance = 1000.0
    fixture.ticker_price = 100.0
    fixture.anchors = {}
    bot.exchange = "fixture"
    bot.get_raw_balance = lambda: fixture.raw_balance
    bot.active_symbols = SYMBOLS
    bot._build_live_symbol_universe = lambda: SYMBOLS
    bot._get_last_position_change_anchors = lambda: fixture.anchors
    bot._completed_candle_health_now_ms = lambda: fixture.clock[0]
    bot._live_market_snapshot_fetch_max_age_ms = lambda: 5_000
    bot._orchestrator_ema_bundle_allows_quote_retry = True
    bot._begin_live_event_cycle = lambda **kwargs: None
    owner.schedule_history = owner.schedule_sources = owner.schedule_account_reporting = lambda: None
    owner.sources[SYMBOL] = Sources((), (), 0)
    bot._hsl_fill_observation = observe_fills(bot, (NOW, NOW))
    monkeypatch.setattr(market_snapshot, "utc_ms", lambda: fixture.clock[0])

    async def fetch_tickers():
        fixture.quote_reads.append(fixture.clock[0])
        price = fixture.ticker_price
        return {symbol: {"bid": price - 1, "ask": price + 1, "last": price} for symbol in SYMBOLS}

    bot.market_snapshot_provider = market_snapshot.MarketSnapshotProvider(
        exchange_name=bot.exchange, fetch_tickers=fetch_tickers
    )

    for name in (
        "_prepare_orchestrator_inputs",
        "_orchestrator_preparation_inputs",
        "_orchestrator_preparation_is_current",
        "_get_orchestrator_market_snapshots",
        "_record_market_snapshot_surface",
        "_market_snapshot_signature",
    ):
        setattr(bot, name, getattr(Passivbot, name).__get__(bot))

    async def refresh(*, require_balance):
        assert require_balance
        fixture.events.append("account")
        ledger.begin_epoch(now_ms=fixture.clock[0])
        for surface in ACCOUNT_SURFACES:
            ledger.stamp(surface, now_ms=fixture.clock[0])
        return True

    async def protect(**kwargs):
        fixture.events.append("protection")
        return False

    async def reconcile(strategy, quotes, receipts):
        fixture.events.append("calculate")
        wave = owner.capture(quotes, ordinary_receipts=receipts)
        fixture.captures.append((strategy, quotes, wave))
        bot._hsl_planning_wave = wave
        bot._current_planning_snapshot = object()
        return [], []

    async def execute(cancels, creates):
        assert not cancels and not creates
        fixture.events.append("execute")

    bot.refresh_protective_authoritative_state = refresh
    owner.protect = protect
    bot._reconcile_prepared_ordinary = reconcile
    bot.execute_order_plan_to_exchange = execute
    return fixture


async def start_preparation(fixture):
    result = await fixture.owner.cycle()
    assert result["updated"] and not result["ordinary_completed"]
    try:
        await asyncio.wait_for(fixture.entered.wait(), 1.0)
    except BaseException:
        await stop_preparation(fixture.owner)
        raise
    return fixture.owner._ordinary


async def stop_preparation(owner, *previous):
    tasks = {task for task in (*previous, owner._ordinary) if task is not None}
    owner.cancel_inputs()
    await asyncio.gather(*tasks, return_exceptions=True)


def expire_prior_account(fixture):
    fixture.ledger.stamp(
        "balance",
        now_ms=fixture.clock[0] - fixture.bot._live_market_snapshot_max_age_ms() - 1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_fresh", [True, False], ids=["fresh", "stale"])
async def test_entry_ready_preparation_preserves_protection_when_refresh_fails(
    owner_handoff, prior_fresh
):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    task = await start_preparation(fixture)
    try:
        fixture.release.set()
        await asyncio.wait_for(task, 1.0)
        if not prior_fresh:
            expire_prior_account(fixture)
        prior_epoch = ledger.epoch
        fixture.events.clear()

        async def protect(**kwargs):
            assert ledger.epoch == prior_epoch
            assert not owner._current_account_unavailable(fixture.clock[0])
            fixture.events.append("protective_write")
            return True

        async def unavailable(*, require_balance):
            assert require_balance
            fixture.events.append("account_unavailable")
            return False

        owner.protect = protect
        bot.refresh_protective_authoritative_state = unavailable
        result = await owner.cycle()

        assert result["ordinary_completed"] and not result["updated"]
        assert not result["ordinary_executed"]
        assert result["protective_work"] is prior_fresh
        assert fixture.events == (["protective_write"] if prior_fresh else []) + [
            "account_unavailable"
        ]
        assert not fixture.captures
        assert ledger.epoch == prior_epoch
        assert ledger.surfaces["market_snapshot"].revision == 0
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
async def test_write_confirmation_delays_next_preparation_until_confirming_epoch(owner_handoff):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    prepare_universe = bot.prepare_planning_universe
    reconcile = bot._reconcile_prepared_ordinary
    refresh = bot.refresh_protective_authoritative_state
    confirming_preparation, release = asyncio.Event(), asyncio.Event()
    acquired_epochs = []
    order = {"symbol": SYMBOL}

    async def observe_preparation():
        acquired_epochs.append(ledger.epoch)
        if len(acquired_epochs) == 2:
            fixture.events.append("confirmed_preparation")
            assert ledger.surface_epoch("open_orders") == ledger.epoch
            assert not bot._authoritative_pending_confirmations
            confirming_preparation.set()
            await release.wait()
        return await prepare_universe()

    async def reconcile_create(*args):
        cancels, creates = await reconcile(*args)
        assert not creates
        return cancels, [order]

    async def execute(cancels, creates):
        assert not cancels and creates == [order]
        fixture.events.append("execute")
        Passivbot._request_authoritative_confirmation(bot, {"open_orders"})

    bot.prepare_planning_universe = observe_preparation
    bot._reconcile_prepared_ordinary = reconcile_create
    bot.execute_order_plan_to_exchange = execute
    original = await start_preparation(fixture)
    initial_epoch = ledger.epoch
    try:
        fixture.release.set()
        await asyncio.wait_for(original, 1.0)
        result = await owner.cycle()

        assert result["ordinary_completed"] and result["ordinary_executed"]
        required_epoch = ledger.epoch + 1
        assert bot._authoritative_pending_confirmations == {"open_orders": required_epoch}
        assert owner._ordinary is None
        assert acquired_epochs == [initial_epoch]

        async def refresh_unavailable(*, require_balance):
            assert require_balance
            ledger.begin_epoch(now_ms=fixture.clock[0])
            return False

        bot.refresh_protective_authoritative_state = refresh_unavailable
        result = await owner.cycle()
        assert not result["updated"] and not result["ordinary_executed"]
        assert ledger.epoch == required_epoch
        assert ledger.surface_epoch("open_orders") < required_epoch
        assert owner._ordinary is None and acquired_epochs == [initial_epoch]

        async def refresh_with_followup(**kwargs):
            result = await refresh(**kwargs)
            Passivbot._request_authoritative_confirmation(bot, {"open_orders"})
            return result

        bot.refresh_protective_authoritative_state = refresh_with_followup
        result = await owner.cycle()
        assert result["updated"] and not result["ordinary_completed"]
        assert not result["ordinary_executed"]
        confirmation_epoch = ledger.epoch + 1
        assert bot._authoritative_pending_confirmations == {"open_orders": confirmation_epoch}
        assert owner._ordinary is None and acquired_epochs == [initial_epoch]

        async def confirm(**kwargs):
            result = await refresh(**kwargs)
            Passivbot._clear_authoritative_confirmations(bot, {"open_orders"})
            return result

        async def protect_while_preparing(**kwargs):
            fixture.events.append("protection_start")
            await asyncio.wait_for(confirming_preparation.wait(), 1.0)
            assert not release.is_set()
            fixture.events.append("protection_done")
            return False

        bot.refresh_protective_authoritative_state = confirm
        owner.protect = protect_while_preparing
        result = await asyncio.wait_for(owner.cycle(), 1.0)

        assert result["updated"] and not result["ordinary_completed"]
        assert not result["ordinary_executed"]
        assert acquired_epochs == [initial_epoch, confirmation_epoch]
        assert fixture.events[-4:] == [
            "account", "protection_start", "confirmed_preparation", "protection_done"
        ]
        assert owner._ordinary is not None and not owner._ordinary.done()
    finally:
        await stop_preparation(owner, original)


@pytest.mark.asyncio
async def test_protection_can_make_account_pending_before_preparation_first_turn(owner_handoff):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    prepare_universe = bot.prepare_planning_universe
    refresh, protect = bot.refresh_protective_authoritative_state, owner.protect
    acquired_epochs = []

    async def observe_preparation():
        assert not owner._current_account_unavailable(fixture.clock[0])
        acquired_epochs.append(ledger.epoch)
        return await prepare_universe()

    async def write_without_yielding(**kwargs):
        assert owner._ordinary is not None and not owner._ordinary.done()
        assert not acquired_epochs
        Passivbot._request_authoritative_confirmation(bot, {"open_orders"})
        return True

    bot.prepare_planning_universe = observe_preparation
    owner.protect = write_without_yielding
    task = None
    try:
        result = await owner.cycle()
        assert result["updated"] and result["protective_work"]
        assert not result["ordinary_completed"] and not result["ordinary_executed"]
        task = owner._ordinary
        assert not task.done()
        assert await asyncio.wait_for(task, 1.0) is None
        assert not acquired_epochs and not fixture.entered.is_set()
        assert not fixture.quote_reads and not fixture.captures
        [defer] = fixture.defers
        assert defer["defer_reason"] == "current_account_inputs_unavailable"
        assert defer["missing"] == ["open_orders"]

        async def confirm(**kwargs):
            result = await refresh(**kwargs)
            Passivbot._clear_authoritative_confirmations(bot, {"open_orders"})
            return result

        bot.refresh_protective_authoritative_state = confirm
        owner.protect = protect
        assert (await owner.cycle())["updated"]
        confirming_epoch = ledger.epoch
        fixture.release.set()
        await asyncio.wait_for(owner._ordinary, 1.0)
        assert acquired_epochs == [confirming_epoch]
        assert (await owner.cycle())["ordinary_executed"]
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", ["ready_at_entry", "same_cycle"])
async def test_late_handoff_uses_current_account_and_exact_unstamped_quote_cohort(
    owner_handoff, handoff
):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    market = ledger.surfaces["market_snapshot"]
    task = await start_preparation(fixture)
    try:
        # The owner services another pass while the exact same task remains blocked.
        result = await asyncio.wait_for(owner.cycle(), 1.0)
        assert result["updated"] and not result["ordinary_completed"]
        assert fixture.events == ["account", "protection", "account", "protection"]
        assert owner._ordinary is task and not task.done()
        assert not fixture.quote_reads and not fixture.captures

        fixture.clock[0] += 15_000
        fixture.raw_balance = 1000.001

        async def finish_preparation():
            fixture.release.set()
            preparation = await asyncio.wait_for(task, 1.0)
            assert set(preparation.quotes) == set(SYMBOLS)
            assert market.revision == 0
            assert not fixture.captures

        if handoff == "ready_at_entry":
            await finish_preparation()

        refresh = bot.refresh_protective_authoritative_state

        async def fresh_account(**kwargs):
            result = await refresh(**kwargs)
            fixture.raw_balance = 1000.002
            return result

        current_quotes = {}

        async def protect(**kwargs):
            fixture.events.append("protection_start")
            if handoff == "same_cycle":
                await finish_preparation()
            assert fixture.captures == []
            assert market.revision == 0
            # The provider acquired a newer cohort during protection.
            fixture.clock[0] += 5_001
            fixture.ticker_price = 101.0
            current_quotes.update(await bot.market_snapshot_provider.get_snapshots(
                SYMBOLS, max_age_ms=5_000
            ))
            fixture.clock[0] += 50
            fixture.events.append("protection_done")
            return False

        bot.refresh_protective_authoritative_state = fresh_account
        owner.protect = protect
        result = await owner.cycle()

        assert result["ordinary_completed"] and result["ordinary_executed"]
        assert fixture.events[-5:] == [
            "account", "protection_start", "protection_done", "calculate", "execute"
        ]
        [(strategy, quotes, wave)] = fixture.captures
        assert strategy.symbols == SYMBOLS
        assert set(quotes) == set(SYMBOLS)
        assert all(quotes[symbol] is current_quotes[symbol] for symbol in SYMBOLS)
        assert all(snapshot.fetched_ms == fixture.clock[0] - 50 for snapshot in quotes.values())
        assert fixture.quote_reads == [NOW + 15_000, NOW + 20_001]
        assert market.epoch == ledger.epoch and market.updated_ms == fixture.clock[0]
        assert market.revision == 1
        assert market.signature == bot._market_snapshot_signature(SYMBOLS, current_quotes)
        assert wave.balance == 1000.002
        assert wave.balance_observed_ms == NOW + 15_000
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ("generation", "account_cohort_changed_during_protection"),
        ("raw_balance", "account_cohort_changed_during_protection"),
        ("positions", "account_facts_changed_during_preparation"),
        ("account_receipt", "account_cohort_changed_during_protection"),
        ("anchors", "prepared_trailing_inputs_changed"),
        ("incomplete_quotes", "prepared_market_inputs_unavailable"),
    ],
)
async def test_protective_changes_fail_closed_before_market_commit(owner_handoff, changed, reason):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    task = await start_preparation(fixture)
    try:
        fixture.release.set()
        await asyncio.wait_for(task, 1.0)
        assert ledger.surfaces["market_snapshot"].revision == 0
        expire_prior_account(fixture)

        async def protect(**kwargs):
            fixture.events.append("protection")
            if changed == "generation":
                bot._account_invalidation_generation = 1
            elif changed == "raw_balance":
                fixture.raw_balance = 1000.001
            elif changed == "positions":
                bot.positions[SYMBOL]["long"]["size"] += 1.0
            elif changed == "account_receipt":
                ledger.stamp("open_orders", now_ms=fixture.clock[0])
            elif changed == "anchors":
                fixture.anchors = {SYMBOL: {"long": {"timestamp": NOW, "epoch": "fill"}}}
            elif changed == "incomplete_quotes":
                # A complete but old preparation must never stand in for the current provider.
                bot.market_snapshot_provider._cache.pop(OTHER)
            return False

        owner.protect = protect
        result = await owner.cycle()

        assert result["ordinary_completed"] and not result["ordinary_executed"]
        assert fixture.events == ["account", "protection", "account", "protection"]
        assert fixture.quote_reads == [NOW]
        assert not fixture.captures
        assert ledger.surfaces["market_snapshot"].revision == 0
        assert [details["defer_reason"] for details in fixture.defers] == [reason]
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["fill", "candle"])
@pytest.mark.parametrize("stage", ["protection", "after_capture"])
async def test_receipt_identity_is_bound_at_consumption(owner_handoff, kind, stage):
    fixture = owner_handoff
    owner, bot = fixture.owner, fixture.bot
    task = await start_preparation(fixture)
    try:
        fixture.release.set()
        await asyncio.wait_for(task, 1.0)

        if kind == "fill":
            original = bot._hsl_fill_observation
            renew = lambda: setattr(bot, "_hsl_fill_observation", replace(original))
            consumed = lambda receipts: receipts.fills
        else:
            original = owner.sources[SYMBOL]
            renew = lambda: owner.sources.update({SYMBOL: replace(original)})
            consumed = lambda receipts: dict(receipts.sources)[SYMBOL]

        if stage == "protection":
            expire_prior_account(fixture)
            protect = owner.protect

            async def renew_during_protection(**kwargs):
                renew()
                return await protect(**kwargs)

            owner.protect = renew_during_protection
        else:
            reconcile = bot._reconcile_prepared_ordinary

            async def renew_after_capture(*args):
                result = await reconcile(*args)
                renew()
                return result

            bot._reconcile_prepared_ordinary = renew_after_capture

        result = await owner.cycle()
        [(_, _, wave)] = fixture.captures
        receipt = consumed(wave.ordinary_receipts)

        assert result["ordinary_completed"]
        assert result["ordinary_executed"] == (stage == "protection")
        assert receipt == original
        if stage == "protection":
            assert receipt is consumed(owner.ordinary_receipts())
            assert receipt is not original
            assert fixture.events[-1] == "execute"
            assert not fixture.defers
        else:
            assert receipt is original
            assert receipt is not consumed(owner.ordinary_receipts())
            assert fixture.events[-1] == "calculate"
            assert [details["defer_reason"] for details in fixture.defers] == [
                f"hsl_{kind}_receipt_replaced"
            ]
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_ages_out", [False, True], ids=["ready", "expired-again"])
async def test_expired_entry_quotes_get_one_nonblocking_retry(owner_handoff, retry_ages_out):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    provider = bot.market_snapshot_provider
    market = ledger.surfaces["market_snapshot"]
    entered, release = asyncio.Event(), asyncio.Event()
    ema_calls, quote_options = [], []
    original_ema = bot._load_orchestrator_ema_bundle
    original_fetch, get_snapshots = provider._fetch_tickers, provider.get_snapshots

    async def ema(*args):
        ema_calls.append(True)
        return await original_ema(*args)

    async def get_quotes(symbols, *, max_age_ms, refresh=False):
        quote_options.append((max_age_ms, refresh))
        return await get_snapshots(symbols, max_age_ms=max_age_ms, refresh=refresh)

    async def delayed_fetch():
        fixture.events.append("quote_fetch_started")
        entered.set()
        await release.wait()
        return await original_fetch()

    bot._load_orchestrator_ema_bundle = ema
    provider.get_snapshots = get_quotes
    original = await start_preparation(fixture)
    retry = None
    try:
        fixture.release.set()
        preparation = await asyncio.wait_for(original, 1.0)
        assert all(snapshot.fetched_ms == NOW for snapshot in preparation.quotes.values())
        assert market.revision == 0

        fixture.clock[0] += 5_001
        fixture.ticker_price = 101.0
        provider._fetch_tickers = delayed_fetch
        refresh = bot.refresh_protective_authoritative_state

        async def refresh_while_quotes_pending(**kwargs):
            result = await refresh(**kwargs)
            # Run the spawned request during account refresh and keep it pending
            # through protection and the end of this outer cycle.
            await entered.wait()
            assert not release.is_set()
            return result

        bot.refresh_protective_authoritative_state = refresh_while_quotes_pending
        result = await asyncio.wait_for(owner.cycle(), 1.0)
        assert result["updated"] and not result["ordinary_completed"]
        retry = owner._ordinary
        assert retry is not original and not retry.done()
        assert fixture.events[-3:] == ["account", "quote_fetch_started", "protection"]
        assert fixture.quote_reads == [NOW]
        assert quote_options == [(5_000, False), (5_000, True)]
        assert ema_calls == [True]

        result = await asyncio.wait_for(owner.cycle(), 1.0)
        assert result["updated"] and not result["ordinary_completed"]
        assert owner._ordinary is retry and not retry.done()
        assert fixture.events[-2:] == ["account", "protection"]
        assert not fixture.captures and market.revision == 0

        bot.refresh_protective_authoritative_state = refresh
        release.set()
        renewed = await asyncio.wait_for(retry, 1.0)
        assert renewed.strategy is preparation.strategy
        assert market.revision == 0
        assert fixture.quote_reads == [NOW, NOW + 5_001]
        if retry_ages_out:
            fixture.clock[0] += 5_001

        result = await owner.cycle()

        assert result["ordinary_completed"]
        assert result["ordinary_executed"] == (not retry_ages_out)
        assert quote_options == [(5_000, False), (5_000, True)]
        assert ema_calls == [True]
        if retry_ages_out:
            assert not fixture.captures and market.revision == 0
            assert [details["defer_reason"] for details in fixture.defers] == [
                "prepared_market_inputs_unavailable"
            ]
        else:
            [(strategy, quotes, _)] = fixture.captures
            assert strategy is preparation.strategy
            assert all(quote is renewed.quotes[symbol] for symbol, quote in quotes.items())
            assert all(quote.fetched_ms == NOW + 5_001 for quote in quotes.values())
            assert market.revision == 1 and market.epoch == ledger.epoch
            assert fixture.events[-4:] == ["account", "protection", "calculate", "execute"]
    finally:
        release.set()
        await stop_preparation(owner, original, retry)
        provider.cancel_pending()
        await provider.wait_pending(timeout_seconds=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", ["fresh", "at_entry", "during_protection"])
async def test_age_bounded_ema_inputs_require_full_preparation_after_quote_expiry(
    owner_handoff, expiry
):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    market = ledger.surfaces["market_snapshot"]
    load_emas = bot._load_orchestrator_ema_bundle
    ema_calls = []

    async def load_age_bounded_emas(*args):
        ema_calls.append(fixture.clock[0])
        bundle = await load_emas(*args)
        bot._orchestrator_ema_bundle_allows_quote_retry = False
        return bundle

    bot._load_orchestrator_ema_bundle = load_age_bounded_emas
    original = await start_preparation(fixture)
    replacement = None
    try:
        fixture.release.set()
        preparation = await asyncio.wait_for(original, 1.0)
        assert not preparation.strategy.allow_quote_retry
        fixture.clock[0] += 5_001 if expiry == "at_entry" else 4_999
        protect = owner.protect
        if expiry == "during_protection":
            prior_epoch = ledger.epoch

            async def expire_after_account_refresh(**kwargs):
                result = await protect(**kwargs)
                if ledger.epoch != prior_epoch:
                    fixture.clock[0] += 2
                return result

            owner.protect = expire_after_account_refresh

        result = await owner.cycle()
        assert result["updated"] and result["ordinary_completed"]
        assert ema_calls == fixture.quote_reads == [NOW]

        if expiry == "fresh":
            expected_strategy = preparation.strategy
            assert not fixture.defers
        else:
            assert not result["ordinary_executed"]
            assert not fixture.captures and market.revision == 0
            assert [details["defer_reason"] for details in fixture.defers] == [
                "prepared_market_inputs_unavailable"
            ]

            owner.protect = protect
            replacement = owner._ordinary
            prepared_again = await asyncio.wait_for(replacement, 1.0)
            expected_strategy = prepared_again.strategy
            assert expected_strategy is not preparation.strategy
            assert not expected_strategy.allow_quote_retry
            assert ema_calls == fixture.quote_reads == [NOW, NOW + 5_001]
            assert market.revision == 0
            result = await owner.cycle()

        assert result["ordinary_completed"] and result["ordinary_executed"]
        [(strategy, quotes, _)] = fixture.captures
        assert strategy is expected_strategy
        assert {quote.fetched_ms for quote in quotes.values()} == {fixture.quote_reads[-1]}
        assert market.revision == 1
    finally:
        await stop_preparation(owner, original, replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("rollover", ["before_request", "during_request"])
async def test_quote_retry_discards_preparation_across_completed_minute(owner_handoff, rollover):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    provider = bot.market_snapshot_provider
    next_minute = NOW + 60_000
    prepared_at = next_minute - 5_002
    fixture.clock[0] = prepared_at
    original = await start_preparation(fixture)
    retry = None
    try:
        fixture.release.set()
        await asyncio.wait_for(original, 1.0)
        assert fixture.quote_reads == [prepared_at]
        fixture.clock[0] = next_minute - 1

        if rollover == "during_request":
            fetch = provider._fetch_tickers

            async def fetch_across_minute():
                assert fixture.clock[0] == next_minute - 1
                fixture.clock[0] = next_minute
                return await fetch()

            provider._fetch_tickers = fetch_across_minute

        result = await owner.cycle()
        assert result["updated"] and not result["ordinary_completed"]
        retry = owner._ordinary
        if rollover == "before_request":
            fixture.clock[0] = next_minute
        done, pending = await asyncio.wait((retry,), timeout=1.0)
        assert done == {retry} and not pending

        result = await owner.cycle()

        assert result["updated"] and result["ordinary_completed"]
        assert not result["ordinary_executed"]
        expected_reads = [prepared_at, next_minute] if rollover == "during_request" else [prepared_at]
        assert fixture.quote_reads == expected_reads
        assert not fixture.captures
        assert ledger.surfaces["market_snapshot"].revision == 0
        [defer] = fixture.defers
        assert defer["defer_reason"] == "prepared_strategy_inputs_expired"
        phase = "consumption" if rollover == "during_request" else "preparation"
        assert defer["context"] == f"ordinary {phase}: prepared_strategy_inputs_expired"
    finally:
        await stop_preparation(owner, original, retry)
        provider.cancel_pending()
        await provider.wait_pending(timeout_seconds=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_ms", [False, True], ids=["mixed-age", "same-ms-hit"])
async def test_symbol_scoped_retry_renews_mixed_age_quote_cohort(owner_handoff, same_ms):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    requests = []
    fetch_delay_ms = 0

    async def fetch_tickers_for_symbols(symbols):
        requests.append(tuple(symbols))
        fixture.clock[0] += fetch_delay_ms
        return {symbol: {"bid": 99.0, "ask": 101.0, "last": 100.0} for symbol in symbols}

    provider = bot.market_snapshot_provider = market_snapshot.MarketSnapshotProvider(
        exchange_name=bot.exchange,
        fetch_tickers=None,
        fetch_tickers_for_symbols=fetch_tickers_for_symbols,
        ticker_strategy="symbols",
    )
    await provider.get_snapshots([SYMBOL], max_age_ms=5_000)
    fixture.clock[0] += 4_000
    await provider.get_snapshots([OTHER], max_age_ms=5_000)

    original = await start_preparation(fixture)
    retry = None
    try:
        fixture.release.set()
        preparation = await asyncio.wait_for(original, 1.0)
        assert preparation.quotes[SYMBOL].fetched_ms == NOW
        assert preparation.quotes[OTHER].fetched_ms == NOW + 4_000
        assert requests == [(SYMBOL,), (OTHER,)]

        fixture.clock[0] += 1_001
        other_fetched_ms = NOW + 4_000
        if same_ms:
            recent = await provider.get_snapshots([OTHER], max_age_ms=5_000, refresh=True)
            other_fetched_ms = recent[OTHER].fetched_ms
            assert other_fetched_ms == fixture.clock[0]
        expected_requests = [*requests, tuple(SYMBOLS)]
        # Only SYMBOL is expired at retry start. If the retry fetches just
        # SYMBOL, OTHER will expire while the connector is acquiring it.
        fetch_delay_ms = 5_001 - (fixture.clock[0] - other_fetched_ms)
        renewed_at = fixture.clock[0] + fetch_delay_ms
        result = await owner.cycle()
        assert result["updated"] and not result["ordinary_executed"]
        assert not fixture.captures

        retry = owner._ordinary
        renewed = await asyncio.wait_for(retry, 1.0)
        assert requests == expected_requests
        assert fixture.clock[0] - other_fetched_ms == 5_001
        assert renewed.strategy is preparation.strategy
        assert {quote.fetched_ms for quote in renewed.quotes.values()} == {renewed_at}
        assert ledger.surfaces["market_snapshot"].revision == 0

        # Forced acquisition does not replace the ordinary admission cutoff.
        fixture.clock[0] += 5_000
        result = await owner.cycle()

        assert result["ordinary_completed"] and result["ordinary_executed"]
        assert requests == expected_requests
        [(strategy, quotes, _)] = fixture.captures
        assert strategy is preparation.strategy
        assert all(quotes[symbol] is renewed.quotes[symbol] for symbol in SYMBOLS)
        assert ledger.surfaces["market_snapshot"].epoch == ledger.epoch
    finally:
        await stop_preparation(owner, original, retry)
        provider.cancel_pending()
        await provider.wait_pending(timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_quotes_expired_during_protection_retry_on_next_turn(owner_handoff):
    fixture = owner_handoff
    owner, bot, ledger = fixture.owner, fixture.bot, fixture.ledger
    market = ledger.surfaces["market_snapshot"]
    load_emas = bot._load_orchestrator_ema_bundle
    ema_calls = []

    async def count_emas(*args):
        ema_calls.append(True)
        return await load_emas(*args)

    bot._load_orchestrator_ema_bundle = count_emas
    original = await start_preparation(fixture)
    retry = None
    try:
        fixture.release.set()
        preparation = await asyncio.wait_for(original, 1.0)
        assert all(quote.fetched_ms == NOW for quote in preparation.quotes.values())
        fixture.clock[0] += 4_999
        protect = owner.protect
        prior_epoch = ledger.epoch

        async def expire_while_protecting(**kwargs):
            if ledger.epoch == prior_epoch:
                fixture.events.append("early_protection")
                return False
            result = await protect(**kwargs)
            fixture.clock[0] += 2
            return result

        owner.protect = expire_while_protecting
        result = await owner.cycle()

        assert result["updated"] and result["ordinary_completed"]
        assert not result["ordinary_executed"]
        assert fixture.events[-3:] == ["early_protection", "account", "protection"]
        assert fixture.quote_reads == [NOW]
        assert not fixture.captures and market.revision == 0
        assert [details["defer_reason"] for details in fixture.defers] == [
            "prepared_market_inputs_unavailable"
        ]

        owner.protect = protect
        fixture.ticker_price = 101.0
        result = await owner.cycle()
        assert result["updated"] and not result["ordinary_completed"]
        assert not result["ordinary_executed"]
        assert fixture.events[-2:] == ["account", "protection"]
        assert not fixture.captures and market.revision == 0

        retry = owner._ordinary
        renewed = await asyncio.wait_for(retry, 1.0)
        assert renewed.strategy is preparation.strategy
        assert fixture.quote_reads == [NOW, NOW + 5_001]
        assert ema_calls == [True]
        assert market.revision == 0

        result = await owner.cycle()

        assert result["updated"] and result["ordinary_completed"] and result["ordinary_executed"]
        assert fixture.events[-4:] == ["account", "protection", "calculate", "execute"]
        [(strategy, quotes, _)] = fixture.captures
        assert strategy is preparation.strategy
        assert set(quotes) == set(SYMBOLS)
        assert all(quote is renewed.quotes[symbol] for symbol, quote in quotes.items())
        assert all(quote.fetched_ms == NOW + 5_001 for quote in quotes.values())
        assert ema_calls == [True] and market.revision == 1
    finally:
        await stop_preparation(owner, original, retry)


@pytest.mark.asyncio
async def test_held_trailing_derives_from_corrected_cache_after_protection(owner_handoff, tmp_path):
    from candlestick_manager import CandlestickManager
    from test_hsl_trailing_consumption import NoNetworkExchange
    from test_unstucking_safeguards import _make_candles

    fixture = owner_handoff
    owner, bot = fixture.owner, fixture.bot
    exchange = NoNetworkExchange()
    cm = bot.cm = CandlestickManager(exchange=exchange, cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms = lambda: fixture.clock[0]
    anchor = NOW - 240_000
    fixture.anchors = {SYMBOL: {"long": {"timestamp": anchor, "epoch": "fill:held"}}}
    rows = [
        (NOW - 180_000, 100.0, 105.0, 99.0, 102.0, 1.0),
        (NOW - 120_000, 102.0, 109.0, 101.0, 108.0, 1.0),
        (NOW - 60_000, 108.0, 112.0, 107.0, 110.0, 1.0),
    ]
    cm._persist_batch(SYMBOL, _make_candles(rows), merge_cache=True, last_refresh_ms=NOW)
    bot.is_trailing = lambda symbol, side: (symbol, side) == (SYMBOL, "long")
    for name in (
        "has_position",
        "refresh_market_state_if_needed",
        "_preload_trailing_candles",
        "_completed_trailing_candle_subset",
        "update_trailing_data",
    ):
        setattr(bot, name, getattr(Passivbot, name).__get__(bot))

    async def update_ohlcvs():
        return True

    bot.update_ohlcvs_1m_for_actives = update_ohlcvs
    read_candles = cm.get_candles_with_resolution_ladder
    reads = []

    async def record_read(symbol, *, allow_remote_fetch=True, **kwargs):
        assert symbol == SYMBOL and kwargs["start_ts"] == anchor + 60_000
        fixture.events.append("trailing_read")
        result = await read_candles(symbol, allow_remote_fetch=allow_remote_fetch, **kwargs)
        reads.append((allow_remote_fetch, result.candles["h"].tolist()))
        return result

    cm.get_candles_with_resolution_ladder = record_read
    task = await start_preparation(fixture)
    try:
        fixture.release.set()
        preparation = await asyncio.wait_for(task, 1.0)
        assert preparation.trailing_anchors == fixture.anchors
        assert reads == [(True, [105.0, 109.0, 112.0])]
        assert not getattr(bot, "trailing_prices", {})
        prior_epoch = fixture.ledger.epoch

        async def protect(**kwargs):
            if fixture.ledger.epoch == prior_epoch:
                fixture.events.append("early_protection")
                return False
            cm._persist_batch(
                SYMBOL,
                _make_candles([(NOW - 60_000, 108.0, 118.0, 107.0, 110.0, 1.0)]),
                merge_cache=True,
                last_refresh_ms=NOW,
            )
            fixture.events.append("protection_done")
            return False

        reconcile = bot._reconcile_prepared_ordinary

        async def check_trailing(*args):
            assert bot._orchestrator_trailing_unavailable_psides == {}
            assert bot.trailing_prices[SYMBOL]["long"]["max_since_open"] == 118.0
            return await reconcile(*args)

        owner.protect = protect
        bot._reconcile_prepared_ordinary = check_trailing
        result = await owner.cycle()

        assert result["ordinary_completed"] and result["ordinary_executed"]
        assert reads == [
            (True, [105.0, 109.0, 112.0]),
            (False, [105.0, 109.0, 118.0]),
        ]
        assert fixture.events[-6:] == [
            "early_protection", "account", "protection_done", "trailing_read", "calculate", "execute"
        ]
        assert exchange.calls == []
    finally:
        await stop_preparation(owner, task)


@pytest.mark.asyncio
async def test_late_handoff_sends_refreshed_raw_balance_to_native_and_executes(tmp_path, monkeypatch):
    import json

    import hjson
    import passivbot
    import utils
    from config import prepare_config
    from config.hsl import generated_template
    from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture, set_risk_memory_spans
    from live import hsl_live
    from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
    from tools import run_fake_live as runner

    legacy = prepare_config(load_fake_hsl_config(), target="canonical", runtime=None, verbose=False)
    config = generated_template(mutable_runtime_fixture(legacy), "coin")
    for side in ("long", "short"):
        config["bot"][side]["unstuck"]["enabled"] = False
        config["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 0.0
    config["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.0
    config["bot"]["long"]["universe"]["approved_coins"] = ["BTC"]
    config["bot"]["long"]["hsl"].update(
        enabled=True, drawdown_fraction_ema_span_minutes=1.0, drawdown_threshold_fraction=0.5
    )
    set_risk_memory_spans(config, 288.0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({key: value for key, value in config.items() if not key.startswith("_")}))
    scenario = hjson.loads((REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text())
    scenario["run_initial_cycle"] = True
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        owner = hsl_live.owner(bot)
        clock = [int(utils.utc_ms())]
        monkeypatch.setattr(utils, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(passivbot, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(market_snapshot, "utc_ms", lambda: clock[0])
        assert await bot.refresh_protective_authoritative_state(require_balance=True)
        initial_raw = bot.get_raw_balance()
        initial_snapped = bot.get_hysteresis_snapped_balance()
        market = bot._ensure_freshness_ledger().surfaces["market_snapshot"]
        initial_market_revision = market.revision
        entered, release = asyncio.Event(), asyncio.Event()
        load_emas = bot._load_orchestrator_ema_bundle

        async def slow_emas(*args):
            prepared = await load_emas(*args)
            entered.set()
            await release.wait()
            return prepared

        monkeypatch.setattr(bot, "_load_orchestrator_ema_bundle", slow_emas)
        await owner.cycle()
        task = owner._ordinary
        try:
            await asyncio.wait_for(entered.wait(), 5.0)
            clock[0] += 15_000
            release.set()
            preparation = await asyncio.wait_for(task, 5.0)
            assert preparation is not None
            assert market.revision == initial_market_revision

            native_calls = []
            protect_done = False
            protect = owner.protect
            compute = passivbot.pbr.compute_ideal_orders_json

            async def protect_first(**kwargs):
                nonlocal protect_done
                result = await protect(**kwargs)
                protect_done = True
                return result

            def capture_native(payload):
                assert protect_done
                native_calls.append((json.loads(payload), bot._current_planning_snapshot))
                return compute(payload)

            def create_count():
                return sum(call["method"] == "create_order" for call in bot.cca.export_request_log())

            monkeypatch.setattr(owner, "protect", protect_first)
            monkeypatch.setattr(passivbot.pbr, "compute_ideal_orders_json", capture_native)
            bot.cca.balance_total += 0.001
            creates_before = create_count()
            result = await owner.cycle()

            assert result["ordinary_completed"] and result["ordinary_executed"]
            [(payload, snapshot)] = native_calls
            [quote] = snapshot.market_snapshots
            [symbol_input] = payload["symbols"]
            assert quote.fetched_ms == preparation.quotes[quote.symbol].fetched_ms
            assert symbol_input["order_book"] == {"bid": quote.bid, "ask": quote.ask}
            [stamp] = [surface for surface in snapshot.surfaces if surface.name == "market_snapshot"]
            assert stamp.epoch == snapshot.epoch
            assert stamp.signature[0][4] == quote.fetched_ms
            assert payload["balance"] == bot.get_hysteresis_snapped_balance() == initial_snapped
            assert payload["balance_raw"] == bot.get_raw_balance() == pytest.approx(initial_raw + 0.001)
            assert create_count() > creates_before
            completed.append(True)
            return {"late_native_order": True}
        finally:
            release.set()
            await stop_preparation(owner, task)

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    user = f"fake_hsl_late_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    try:
        await runner._run_fake_case(
            config_path=str(config_path), scenario_path=str(scenario_path), user=user,
            max_steps=1, output_dir=tmp_path / "run", log_level=1,
            snapshot_each_step=False, enforce_assertions=False,
        )
        assert completed
    finally:
        _cleanup_fake_user_state(user)
