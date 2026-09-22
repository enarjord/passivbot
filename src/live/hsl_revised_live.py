"""Revised HSL live orchestration, separate from legacy recovery authority.

Explicit revised selection chooses this owner. Nothing in this module persists
a permission: a wave carries an immutable observation and
must be admitted again from current facts immediately before a connector write.
"""
from dataclasses import dataclass, replace
from functools import wraps
import logging
from uuid import uuid4

from config.hsl_revised import engine
from live import hsl_revised_runtime as runtime
from live.market_snapshot import MarketSnapshotUnavailable


def selected(bot):
    return engine(getattr(bot, "config", {})) == "revised"


def policy(bot, side, symbol=None):
    if bot.config["live"]["hsl_signal_mode"] == "unified":
        return dict(bot.config["bot"]["hsl"])
    result = dict(bot.config["bot"][side]["hsl"])
    if symbol is not None and bot.config["live"]["hsl_signal_mode"] == "coin":
        result.update(bot.coin_overrides.get(symbol, {}).get("bot", {}).get(side, {}).get("hsl", {}))
    return result


def owner(bot):
    instance = getattr(bot, "_hsl_revised_live", None)
    if instance is None:
        instance = Owner(bot)
        bot._hsl_revised_live = instance
    return instance


def connector_write(action):
    """Serialize revised admission and writes, including adapter batch overrides.

    A write may change aggregate risk before its REST result is reflected locally.
    The next write needs a newer complete account read, even after an ambiguous
    failure. Legacy batching and admission remain unchanged.
    """
    def decorate(operation):
        @wraps(operation)
        async def submit(bot, order):
            if not selected(bot):
                return await operation(bot, order)
            from live import executor
            instance = owner(bot)
            async with instance._write_lock:
                if not instance.admit(order):
                    return (executor.DeferredOrderCreation() if action == "create"
                            else executor.DeferredOrderCancellation())
                if action == "cancel":
                    executor.record_cancel_connector_admission(bot, order)
                try:
                    return await operation(bot, order)
                finally:
                    # Reads completed while the call was pending cannot confirm
                    # its final outcome. Mark all current account inputs pending.
                    bot._request_authoritative_confirmation({"balance", "positions", "open_orders"})
        return submit
    return decorate


def matches(scope, symbol, side):
    return ((scope.symbol is None or scope.symbol == symbol)
            and (scope.pside is None or scope.pside == side))


@dataclass(frozen=True)
class Wave:
    captured_ms: int
    position_observed_ms: int
    mark_observed_ms: tuple[int, ...]
    decisions: tuple
    unavailable: tuple
    positions: str
    open_orders: tuple
    balance: float
    generation: int
    required_fills: tuple | None = None

    def permission(self, symbol, side):
        if any(matches(item.scope, symbol, side) for item in self.unavailable):
            return "unavailable", None
        decisions = [d for d in self.decisions if matches(d.scope, symbol, side)]
        if len(decisions) > 1:
            raise runtime.InvalidHslOutput("overlapping revised HSL scopes")
        return (decisions[0].action, decisions[0].execution_type) if decisions else (None, None)


class Owner:
    def __init__(self, bot):
        import asyncio
        self._write_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        from time import monotonic
        self._schedule_clock = monotonic
        self._next_sources = self._next_history = 0.
        self._ordinary = None
        self._cycle_running = False
        self._running = False
        self.bot = bot
        self.sources = {}
        self.quotes = {}
        self._waves = {}
        self._quote_tasks = {}
        self._quote_started = {}
        self._discard_quote_tasks = set()
        self._quote_cursor = 0
        self._position_observation = None

    def capture(self, quotes=None, *, target=None):
        # Factual caches only. The adapter clips every observation again to the
        # current window; neither prior GREEN nor prior RED is an input.
        bot = self.bot
        if quotes is not None:
            self.quotes.update(quotes)
        now_utc, max_age = self.remember_position()
        requests, unavailable = runtime.capture(
            bot, self.quotes, self.sources,
            symbols={side: set(bot.approved_coins_minus_ignored_coins[side])
                     for side in ("long", "short")},
            now_ms=int(bot.get_exchange_time()), utc_now_ms=now_utc,
            max_current_age_ms=max_age, position_observation=self._position_observation,
            use_observed_fills=True, target=target)
        decisions = runtime.evaluate(requests)
        wave = Wave(now_utc, self._position_observation.observed_ms,
                    tuple(stamp for request in requests for stamp in request.mark_observed_ms),
                    decisions, unavailable, runtime.observe_positions(bot).payload,
                    runtime.observe_open_orders(bot), bot.get_raw_balance(),
                    int(getattr(bot, "_account_invalidation_generation", 0)))
        return wave

    def remember_position(self):
        from utils import utc_ms
        bot = self.bot
        current_position = runtime.observe_positions(bot)
        previous = self._position_observation
        now_utc = int(utc_ms())
        max_age = int(bot._live_market_snapshot_max_age_ms())
        if (previous is None or previous.payload != current_position.payload
                or previous.generation != current_position.generation
                or not 0 <= now_utc - previous.observed_ms <= max_age // 2):
            self._position_observation = current_position
        return now_utc, max_age

    def report(self, wave):
        from live.hsl_revised_diagnostics import record
        record(self.bot, wave)

    def _account_matches(self, wave, now):
        bot = self.bot
        if self._shutdown_requested() or self._refresh_lock.locked():
            return False
        if wave.required_fills is not None and wave.required_fills != self.required_fill_facts():
            return False
        ledger = bot._ensure_freshness_ledger()
        pending = getattr(bot, '_authoritative_pending_confirmations', {})
        for surface in ('balance', 'positions', 'open_orders'):
            state = ledger.surfaces[surface]
            if (state.updated_ms <= 0 or not 0 <= now - state.updated_ms <= bot._live_market_snapshot_max_age_ms()
                    or int(pending.get(surface, 0)) > state.epoch):
                return False
        # A revised panic is not a durable commitment and cannot take legacy's
        # account-generation bypass. Changed size/basis also requires replanning.
        if (wave.generation != int(getattr(bot, "_account_invalidation_generation", 0))
                or wave.positions != runtime.observe_positions(bot).payload
                or wave.open_orders != runtime.observe_open_orders(bot)):
            return False
        if wave.balance != bot.get_raw_balance():
            return False
        return True

    def admit(self, order):
        from utils import utc_ms
        wave = self._waves.get(order.get("_hsl_revised_wave"))
        if not isinstance(wave, Wave) or not self._account_matches(wave, int(utc_ms())):
            return False
        # Capture is side-effect free with respect to diagnostic sinks. Reporting
        # happens after the protective wave, outside the write freshness budget.
        if (order.get("_hsl_revised_retired_panic")
                and self.bot.get_forced_PB_mode(order["position_side"], order["symbol"]) in {"panic", "manual"}):
            return False
        current = self.capture(target=(order["symbol"], order["position_side"]))
        now = int(utc_ms())
        # A long synchronous reconstruction can consume the remaining freshness
        # budget even without an await. Recheck at the actual write boundary.
        if not self._account_matches(wave, now):
            return False
        scopes = [d.scope for d in current.decisions
                  if matches(d.scope, order['symbol'], order['position_side'])]
        for symbol, sides in self.bot.positions.items():
            if any(p['size'] != 0 and any(matches(scope, symbol, side) for scope in scopes)
                   for side, p in sides.items()):
                quote = self.quotes.get(symbol)
                if (quote is None or not quote.is_valid()
                        or not 0 < quote.fetched_ms <= now
                        or now - quote.fetched_ms > self.bot._live_market_snapshot_max_age_ms()):
                    return False
        before = wave.permission(order["symbol"], order["position_side"])
        after = current.permission(order["symbol"], order["position_side"])
        return after[0] != "unavailable" and before == after

    def _shutdown_requested(self):
        from passivbot import Passivbot
        return Passivbot._shutdown_requested(self.bot)

    def bind(self, wave, cancels, creates, *, ordinary=False):
        if ordinary:
            wave = replace(wave, required_fills=self.required_fill_facts())
        # A fresh owner must not recycle an old order's receipt identifier.
        token = uuid4().hex
        self._waves[token] = wave
        while len(self._waves) > 8:
            del self._waves[next(iter(self._waves))]
        for order in (*cancels, *creates):
            order["_hsl_revised_wave"] = token

    async def protect(self, *, deferred_reports=None):
        """One finite wave over all currently evaluable RED scopes.

        Account refresh belongs to the caller. This does not wait for flattening,
        history repair, cooldown, or another scope's close to fill.
        An outer execution pass may collect reports until its ordinary writes finish.
        """
        bot = self.bot
        if self._shutdown_requested():
            return False
        symbols = {symbol for symbol, sides in bot.positions.items()
                   if any(position["size"] != 0 for position in sides.values())}
        symbols.update(symbol for symbol, orders in bot.open_orders.items() if orders)
        quotes = await self.acquire_quotes(symbols)
        if self._shutdown_requested():
            return False
        wave = self.capture(quotes)
        try:
            targets, execution_types = {}, {}
            for symbol in sorted(symbols):
                for side in ("long", "short"):
                    action, execution_type = wave.permission(symbol, side)
                    if action in {"panic", "halted"}:
                        targets.setdefault(symbol, set()).add(side)
                        execution_types[symbol, side] = execution_type
            from live import reconciler
            green_symbols = [symbol for symbol, orders in bot.open_orders.items()
                             if orders and any(wave.permission(symbol, side)[0] == "normal"
                                               for side in ("long", "short"))]
            actual = reconciler.snapshot_actual_orders(bot, green_symbols) if green_symbols else {}
            retired = [dict(order, _hsl_revised_retired_panic=True) for symbol, orders in actual.items()
                       for order in orders
                       if wave.permission(symbol, order["position_side"])[0] == "normal"
                       and order["pb_order_type"].rsplit("_", 1)[0] == "close_panic"
                       and bot.get_forced_PB_mode(order["position_side"], symbol) not in {"panic", "manual"}]
            if not targets and not retired:
                return False
            bot._record_market_snapshot_surface(sorted(quotes), quotes)
            cancels, creates = [], []
            if targets:
                cancels, creates = await bot.calc_protective_panic_orders_to_cancel_and_create(
                    target_psides_by_symbol=targets, market_snapshots=quotes,
                    execution_types=execution_types)
            cancels.extend(retired)
            if self._shutdown_requested():
                return False
            self.bind(wave, cancels, creates)
            await bot.execute_order_plan_to_exchange(cancels, creates, configure_creations=False)
            return bool(cancels or creates)
        finally:
            # Synchronous projection/logging must not age a wave before its
            # protective writes. Report even when no targets or orders remain.
            if deferred_reports is None:
                self.report(wave)
            else:
                deferred_reports.append(wave)

    def history_symbols(self):
        bot = self.bot
        if bot._pnls_manager is None:
            return set()
        now = int(bot.get_exchange_time())
        start = max(0, now - round(bot.config['live']['pnls_max_lookback_days'] * 86_400_000))
        tape = runtime.capture_fills(bot._pnls_manager.get_events(start_ms=start), bot.c_mults)
        return {pair.symbol for pair in tape.pairs
                if any(start <= fill.timestamp <= now for fill in pair.fills)}

    async def acquire_quotes(self, symbols):
        import asyncio
        from time import monotonic
        from ccxt.base.errors import NetworkError
        # Reuse slots occupied by cancellation-resistant requests. No delayed
        # symbol can grow an unbounded queue or suspend independent protection.
        quotes = {}
        def collect():
            for symbol, task in list(self._quote_tasks.items()):
                if not task.done():
                    continue
                del self._quote_tasks[symbol]
                del self._quote_started[symbol]
                discarded = task in self._discard_quote_tasks
                self._discard_quote_tasks.discard(task)
                if task.cancelled():
                    continue
                try:
                    result = task.result()
                except (NetworkError, MarketSnapshotUnavailable):
                    self.quotes.pop(symbol, None)
                    continue
                if not discarded:
                    quotes.update(result)
        collect()
        ordered = sorted(symbols)
        if ordered:
            start = self._quote_cursor % len(ordered)
            ordered = ordered[start:] + ordered[:start]
        for symbol in ordered:
            if symbol not in self._quote_tasks and len(self._quote_tasks) < 8:
                self._quote_tasks[symbol] = asyncio.create_task(
                    self.bot._get_orchestrator_market_snapshots([symbol]))
                self._quote_started[symbol] = monotonic()
                self._quote_cursor += 1
        if self._quote_tasks:
            await asyncio.wait(tuple(self._quote_tasks.values()), timeout=.25)
            for symbol, task in self._quote_tasks.items():
                # The wave's time slice is not the network deadline. Keep a
                # slow, useful read alive across waves; after five seconds its
                # slot remains owned until cancellation actually completes.
                if (not task.done() and task not in self._discard_quote_tasks
                        and monotonic() - self._quote_started[symbol] >= 5.):
                    self._discard_quote_tasks.add(task)
                    task.cancel()
        collect()
        self.quotes.update(quotes)
        # Earlier factual quotes remain usable only until their original TTL;
        # capture validates each fetched time, including held marks.
        return {symbol: quote for symbol, quote in self.quotes.items() if symbol in symbols}

    def poll_inputs(self):
        for name in ('_fill_task', '_source_task'):
            task = getattr(self, name, None)
            if task is not None and task.done():
                task.result()

    def schedule_history(self):
        import asyncio
        task = getattr(self, '_fill_task', None)
        if task is not None:
            if not task.done():
                return
            task.result()  # Unexpected failures remain fatal, including late ones.
        # Ordinary fill consumers keep their own readiness gates. This task only
        # refreshes their canonical manager; it does not stamp synthetic readiness.
        now = self._schedule_clock()
        kwargs = {}
        if now >= getattr(self, '_next_history_window', 0.):
            end = int(self.bot.get_exchange_time())
            kwargs['since_ms'] = max(0, end - round(self.bot.config['live']['pnls_max_lookback_days'] * 86_400_000))
            self._next_history_window = now + 900.
        self._fill_task = asyncio.create_task(self._refresh_history(kwargs))

    async def _refresh_history(self, kwargs):
        import asyncio
        from ccxt.base.errors import NetworkError
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from passivbot_exceptions import FillEventDataError
        # Give the actual millisecond clock a chance to separate the account
        # read from the tail read. A tied observation stays explicitly uncertain.
        await asyncio.sleep(.001)
        try:
            ready = await self.bot.update_pnls(source='hsl_revised', **kwargs)
        except (NetworkError, OSError, AuthoritativeSurfaceUnavailable, FillEventDataError) as exc:
            logging.warning('[risk] revised history repair unavailable | error_type=%s', type(exc).__name__)
            ready = False
        if not ready:
            # A failed/incomplete refresh cannot leave an older tape certified
            # for ordinary fill consumers. The canonical successful refresh owns
            # renewal; account-only and protective actions do not require fills.
            self.bot._request_authoritative_confirmation({'fills'})
        return ready

    def schedule_sources(self):
        import asyncio
        from live.hsl_revised_candles import CandleSourceReader
        task = getattr(self, '_source_task', None)
        if task is not None:
            if not task.done():
                return
            task.result()
        now = self._schedule_clock()
        if now < self._next_sources:
            return
        self._next_sources = now + 60.
        if not hasattr(self, '_reader'):
            from utils import utc_ms
            self._reader = CandleSourceReader(self.bot.cm, observation_clock=lambda: int(utc_ms()))
        self._source_task = asyncio.create_task(self._read_sources())

    async def _read_sources(self):
        bot = self.bot
        symbols = {symbol for symbol, sides in bot.positions.items()
                   if any(position['size'] != 0 for position in sides.values())}
        symbols.update(self.history_symbols())
        self.sources = {symbol: source for symbol, source in self.sources.items() if symbol in symbols}
        now = int(bot.get_exchange_time())
        start = max(0, now - round(bot.config['live']['pnls_max_lookback_days'] * 86_400_000))
        # Publish each factual result as it arrives; no all-symbol barrier.
        for symbol in sorted(symbols):
            self.sources[symbol] = await self._reader.acquire(
                symbol, start=start, end=now, timeout_seconds=15., allow_remote_fetch=True)

    def required_fill_facts(self):
        bot = self.bot
        if 'fills' not in bot._staged_planner_required_surfaces(include_market_snapshot=False):
            return None
        state = bot._ensure_freshness_ledger().surfaces['fills']
        minimum = max(1, int(getattr(bot, '_authoritative_pending_confirmations', {}).get('fills', 0)))
        return (state.signature, state.epoch >= minimum)

    def account_facts(self):
        bot = self.bot
        return (runtime.observe_positions(bot), runtime.observe_open_orders(bot), bot.get_hysteresis_snapped_balance(),
                self.required_fill_facts())

    @staticmethod
    def same_account_facts(before, after):
        # Observation timestamps may advance on a confirming unchanged read.
        # Changing facts or invalidation still void the entire prepared universe.
        return (before[0].payload == after[0].payload and before[0].generation == after[0].generation
                and before[1:] == after[1:])

    async def _ordinary_plan(self):
        bot = self.bot
        account = self.account_facts()
        await bot.prepare_planning_universe()
        if not self.same_account_facts(account, self.account_facts()):
            return None
        if not await bot.refresh_market_state_if_needed():
            return None
        if not self.same_account_facts(account, self.account_facts()):
            return None
        ready, _ = bot._staged_execution_ready_state(
            include_market_snapshot=False, context='revised ordinary planning')
        if not ready:
            return None
        cancels, creates = await bot.calc_orders_to_cancel_and_create()
        if not self.same_account_facts(account, self.account_facts()):
            return None
        # Preparation may span confirming raw-balance changes. Ordinary Rust
        # risk consumes the latest raw balance at its actual calculation, so
        # reconciliation and submission must retain that exact observation.
        if bot._hsl_revised_planning_wave.balance != bot.get_raw_balance():
            return None
        # Empty plans need the same immutable receipt as nonempty plans. An
        # order dictionary is not the authority for the completed calculation.
        wave = replace(bot._hsl_revised_planning_wave, required_fills=self.required_fill_facts())
        return cancels, creates, bot._current_planning_snapshot, wave

    def cancel_inputs(self):
        for task in (self._ordinary, getattr(self, '_fill_task', None), getattr(self, '_source_task', None),
                     *self._quote_tasks.values()):
            if task is not None:
                task.cancel()
                task.add_done_callback(_retrieve_on_shutdown)
        reader = getattr(self, '_reader', None)
        if reader is not None:
            reader.cancel_pending()

    async def warmup(self):
        from ccxt.base.errors import NetworkError
        from candlestick_manager import OhlcvFetchError
        try:
            await self.bot.warmup_trading_ready_candles()
        except (NetworkError, OSError, OhlcvFetchError) as exc:
            logging.info('[boot] trading-ready candle warmup skipped | error_type=%s', type(exc).__name__)

    async def during_preparation(self, operation):
        """Keep protection scheduled during slow startup preparation."""
        import asyncio
        from ccxt.base.errors import NetworkError
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        # init_markets is reused by hourly maintenance. Once the main owner is
        # running it alone schedules protection; maintenance must remain a reader.
        if self._running:
            return await operation
        task = asyncio.create_task(operation)
        try:
            while not task.done() and not self._shutdown_requested():
                self.poll_inputs()
                try:
                    if await self.bot.refresh_protective_authoritative_state(require_balance=True):
                        self.remember_position()
                        self.schedule_history()
                        self.schedule_sources()
                        await self.protect()
                except (NetworkError, AuthoritativeSurfaceUnavailable, MarketSnapshotUnavailable) as exc:
                    logging.warning('[risk] revised startup current I/O unavailable | error_type=%s', type(exc).__name__)
                await asyncio.wait((task,), timeout=1.)
            if task.done():
                return task.result()
        finally:
            if not task.done():
                task.cancel()
                task.add_done_callback(_retrieve_on_shutdown)

    async def cycle(self):
        """One finite production pass; offline scenario stepping uses this same owner.

        Background task ownership spans passes. A caller must not overlap cycles;
        slow ordinary work never becomes a prerequisite for the protective pass.
        """
        import asyncio
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from ccxt.base.errors import NetworkError
        if self._cycle_running:
            raise RuntimeError("revised HSL execution cycle is already running")
        self._cycle_running = True
        was_running, self._running = self._running, True
        bot = self.bot
        reports = []
        try:
            from utils import utc_ms
            bot._begin_live_event_cycle(loop_start_ms=int(utc_ms()))
            bot.execution_scheduled = False
            bot.state_change_detected_by_symbol = set()
            self.poll_inputs()
            plan = None
            completed_plan = self._ordinary is not None and self._ordinary.done()
            # Retrieve completed failures before any fallible account refresh.
            if completed_plan:
                completed, self._ordinary = self._ordinary, None
                try:
                    plan = completed.result()
                except RuntimeError as exc:
                    handled, details = bot._handle_staged_execution_precondition_error(exc)
                    if not handled:
                        raise
                    bot._log_staged_execution_defer(details)
            protective_work = False
            if self._shutdown_requested():
                return dict(updated=False, ordinary_completed=completed_plan, ordinary_executed=False)
            if plan is not None:
                # Service ready work before a new confirming account read can
                # invalidate it. Protection still gets first turn; every write
                # retains its current-input and exact raw-balance admission.
                protective_work = await self.protect(deferred_reports=reports)
                if self._shutdown_requested():
                    return dict(updated=False, ordinary_completed=completed_plan,
                                ordinary_executed=False, protective_work=protective_work)
                cancels, creates, snapshot, wave = plan
                if self._account_matches(wave, int(utc_ms())):
                    bot._current_planning_snapshot = snapshot
                    await bot.execute_order_plan_to_exchange(cancels, creates)
                else:
                    plan = None
            if (not await bot.refresh_protective_authoritative_state(require_balance=True)
                    or self._shutdown_requested()):
                return dict(updated=False, ordinary_completed=completed_plan,
                            ordinary_executed=plan is not None, protective_work=protective_work)
            self.remember_position()
            now = self._schedule_clock()
            if now >= self._next_history:
                self.schedule_history()
                self._next_history = now + 5.
            self.schedule_sources()
            protective_work = await self.protect(deferred_reports=reports) or protective_work
            if self._ordinary is None and not self._shutdown_requested():
                self._ordinary = asyncio.create_task(self._ordinary_plan())
            return dict(updated=True, ordinary_completed=completed_plan,
                        ordinary_executed=plan is not None, protective_work=protective_work)
        except (NetworkError, AuthoritativeSurfaceUnavailable, MarketSnapshotUnavailable) as exc:
            logging.warning('[risk] revised current I/O unavailable | error_type=%s', type(exc).__name__)
            return dict(updated=False, current_io_unavailable=True)
        finally:
            self._running = was_running
            self._cycle_running = False
            for wave in reports:
                self.report(wave)

    async def run(self):
        """Serialized writes with periodic protection while preparation is pending."""
        from utils import utc_ms
        bot = self.bot
        if self._running:
            raise RuntimeError("revised HSL execution owner is already running")
        self._running = True
        try:
            while not self._shutdown_requested():
                started = int(utc_ms())
                result = await self.cycle()
                if not result['updated'] and not result.get('current_io_unavailable'):
                    await bot._sleep_unless_shutdown(.5, stage='revised_current_inputs')
                    continue
                bot._last_loop_duration_ms = int(utc_ms()) - started
                bot._maybe_log_health_summary()
                await bot._sleep_unless_shutdown(
                    max(.05, float(bot.live_value('execution_delay_seconds'))),
                    stage='revised_execution_delay')
        finally:
            self._running = False
            self.cancel_inputs()


def _retrieve_on_shutdown(task):
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logging.error('[risk] revised background task failed during shutdown | error_type=%s', type(error).__name__)
