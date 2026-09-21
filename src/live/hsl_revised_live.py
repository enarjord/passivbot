"""Revised HSL live orchestration, separate from legacy recovery authority.

The public runtime guard stays closed while this owner is integrated. Nothing in
this module persists a permission: a wave carries an immutable observation and
must be admitted again from current facts immediately before a connector write.
"""
from dataclasses import dataclass
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
    decisions: tuple
    unavailable: tuple
    positions: str
    open_orders: tuple
    balance: float
    generation: int

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
        self._running = False
        self.bot = bot
        self.sources = {}
        self.quotes = {}
        self._diagnostic = None
        self._waves = {}
        self._quote_tasks = {}
        self._quote_started = {}
        self._discard_quote_tasks = set()
        self._quote_cursor = 0
        self._position_observation = None

    def capture(self, quotes=None):
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
            use_observed_fills=True)
        decisions = runtime.evaluate(requests)
        return Wave(decisions, unavailable, runtime.observe_positions(bot).payload, runtime.observe_open_orders(bot), bot.get_raw_balance(),
                    int(getattr(bot, "_account_invalidation_generation", 0)))

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
        signature = (tuple((d.scope, d.action, d.reasons) for d in wave.decisions), wave.unavailable)
        if signature != self._diagnostic:
            for d in wave.decisions:
                logging.info("[risk] revised HSL | mode=%s side=%s symbol=%s action=%s estimates=%s",
                             d.scope.mode, d.scope.pside, d.scope.symbol, d.action,
                             ",".join(d.reasons) or "none")
            for item in wave.unavailable:
                logging.warning("[risk] revised HSL cannot evaluate current inputs | scope=%s reason=%s",
                                item.scope, item.reason)
            self._diagnostic = signature

    def _account_matches(self, wave, now):
        bot = self.bot
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
                or wave.open_orders != runtime.observe_open_orders(bot) or wave.balance != bot.get_raw_balance()):
            return False
        return True

    def admit(self, order):
        from utils import utc_ms
        wave = self._waves.get(order.get("_hsl_revised_wave"))
        if not isinstance(wave, Wave) or not self._account_matches(wave, int(utc_ms())):
            return False
        current = self.capture()
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

    def bind(self, wave, cancels, creates):
        # A fresh owner must not recycle an old order's receipt identifier.
        token = uuid4().hex
        self._waves[token] = wave
        while len(self._waves) > 8:
            del self._waves[next(iter(self._waves))]
        for order in (*cancels, *creates):
            order["_hsl_revised_wave"] = token

    async def protect(self):
        """One finite wave over all currently evaluable RED scopes.

        Account refresh belongs to the caller. This does not wait for flattening,
        history repair, cooldown, or another scope's close to fill.
        """
        bot = self.bot
        symbols = {symbol for symbol, sides in bot.positions.items()
                   if any(position["size"] != 0 for position in sides.values())}
        symbols.update(symbol for symbol, orders in bot.open_orders.items() if orders)
        quotes = await self.acquire_quotes(symbols)
        wave = self.capture(quotes)
        self.report(wave)
        targets, execution_types = {}, {}
        for symbol in sorted(symbols):
            for side in ("long", "short"):
                action, execution_type = wave.permission(symbol, side)
                if action in {"panic", "halted"}:
                    targets.setdefault(symbol, set()).add(side)
                    execution_types[symbol, side] = execution_type
        if not targets:
            return False
        bot._record_market_snapshot_surface(sorted(quotes), quotes)
        cancels, creates = await bot.calc_protective_panic_orders_to_cancel_and_create(
            target_psides_by_symbol=targets, market_snapshots=quotes,
            execution_types=execution_types)
        self.bind(wave, cancels, creates)
        await bot.execute_order_plan_to_exchange(cancels, creates, configure_creations=False)
        return bool(cancels or creates)

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
        from time import monotonic
        now = monotonic()
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
            return await self.bot.update_pnls(source='hsl_revised', **kwargs)
        except (NetworkError, OSError, AuthoritativeSurfaceUnavailable, FillEventDataError) as exc:
            logging.warning('[risk] revised history repair unavailable | error_type=%s', type(exc).__name__)
            return False

    def schedule_sources(self):
        import asyncio
        from live.hsl_revised_candles import CandleSourceReader
        task = getattr(self, '_source_task', None)
        if task is not None:
            if not task.done():
                return
            task.result()
        if not hasattr(self, '_reader'):
            self._reader = CandleSourceReader(self.bot.cm)
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

    def account_facts(self):
        bot = self.bot
        return (runtime.observe_positions(bot), runtime.observe_open_orders(bot), bot.get_raw_balance())

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
        return cancels, creates, bot._current_planning_snapshot

    def cancel_inputs(self):
        for task in (getattr(self, '_fill_task', None), getattr(self, '_source_task', None),
                     *self._quote_tasks.values()):
            if task is not None:
                task.cancel()
                task.add_done_callback(_retrieve_on_shutdown)
        reader = getattr(self, '_reader', None)
        if reader is not None:
            reader.cancel_pending()

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
            while not task.done() and not self.bot.stop_signal_received:
                self.poll_inputs()
                try:
                    if await self.bot.refresh_protective_authoritative_state(require_balance=True):
                        self.remember_position()
                        self.schedule_history()
                        if getattr(self, '_source_task', None) is None:
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

    async def run(self):
        """Serialized writes with periodic protection while preparation is pending."""
        import asyncio
        from time import monotonic
        from utils import utc_ms
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from ccxt.base.errors import NetworkError
        bot = self.bot
        if self._running:
            raise RuntimeError("revised HSL execution owner is already running")
        self._running = True
        ordinary = None
        next_history = next_sources = 0.
        try:
            while not bot.stop_signal_received:
                started = int(utc_ms())
                bot._begin_live_event_cycle(loop_start_ms=started)
                bot.execution_scheduled = False
                bot.state_change_detected_by_symbol = set()
                try:
                    self.poll_inputs()
                    plan = None
                    # Completed producer failures must surface even when the next
                    # account refresh cannot succeed. Plans still need fresh input
                    # admission below; a blocked cycle may discard a valid plan.
                    if ordinary is not None and ordinary.done():
                        completed, ordinary = ordinary, None
                        try:
                            plan = completed.result()
                        except RuntimeError as exc:
                            handled, details = bot._handle_staged_execution_precondition_error(exc)
                            if not handled:
                                raise
                            bot._log_staged_execution_defer(details)
                    if not await bot.refresh_protective_authoritative_state(require_balance=True):
                        await bot._sleep_unless_shutdown(.5, stage='revised_current_inputs')
                        continue
                    self.remember_position()
                    now = monotonic()
                    if now >= next_history:
                        self.schedule_history()
                        next_history = now + 5.
                    if now >= next_sources:
                        self.schedule_sources()
                        next_sources = now + 60.
                    # A pending limit panic never monopolizes the owner. Other
                    # RED scopes and ordinary ready scopes get a pass each wave.
                    await self.protect()
                    if plan is not None:
                        cancels, creates, snapshot = plan
                        bot._current_planning_snapshot = snapshot
                        await bot.execute_order_plan_to_exchange(cancels, creates)
                    if ordinary is None:
                        ordinary = asyncio.create_task(self._ordinary_plan())
                except (NetworkError, AuthoritativeSurfaceUnavailable, MarketSnapshotUnavailable) as exc:
                    logging.warning('[risk] revised current I/O unavailable | error_type=%s', type(exc).__name__)
                bot._last_loop_duration_ms = int(utc_ms()) - started
                bot._maybe_log_health_summary()
                await bot._sleep_unless_shutdown(
                    max(.05, float(bot.live_value('execution_delay_seconds'))),
                    stage='revised_execution_delay')
        finally:
            # Keep every task owned; cancellation-resistant reads are not replaced
            # with overlapping retries. Shutdown never waits indefinitely on them.
            self._running = False
            self.cancel_inputs()
            if ordinary is not None:
                ordinary.cancel()
                ordinary.add_done_callback(_retrieve_on_shutdown)


def _retrieve_on_shutdown(task):
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logging.error('[risk] revised background task failed during shutdown | error_type=%s', type(error).__name__)
