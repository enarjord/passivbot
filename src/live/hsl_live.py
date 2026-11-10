"""HSL live orchestration.

Nothing in this module persists a permission: a wave carries an immutable observation and
must be admitted again from current facts immediately before a connector write.
"""

from dataclasses import dataclass, replace
from functools import wraps
import logging
import weakref
from uuid import uuid4

from live import hsl_runtime as runtime
from live import position_fill_sync
from live.console_health import observe_ordinary_acceptance, observe_ordinary_outer_cycle
from live.diagnostic_safety import log_process_recovery
from live.events import run_diagnostic_step
from live.market_snapshot import MarketSnapshotUnavailable
from live.ordinary_diagnostics import describe_ordinary_admission


def policy(bot, side, symbol=None):
    if bot.config["bot"]["portfolio"]["hsl"]["scope"] == "portfolio":
        return dict(bot.config["bot"]["portfolio"]["hsl"])
    result = dict(bot.config["bot"][side]["hsl"])
    if symbol is not None and bot.config["bot"]["portfolio"]["hsl"]["scope"] == "coin":
        result.update(
            bot.coin_overrides.get(symbol, {})
            .get("bot", {})
            .get(side, {})
            .get("hsl", {})
        )
    return result


def owner(bot):
    instance = getattr(bot, "_hsl_live", None)
    if instance is None:
        instance = Owner(bot)
        bot._hsl_live = instance
    return instance


def connector_write(action):
    """Serialize hsl admission and writes, including adapter batch overrides.

    A write may change aggregate risk before its REST result is reflected locally.
    The next write needs a newer complete account read, even after an ambiguous
    failure. All writes share bounded position/fill settling.
    """

    def decorate(operation):
        @wraps(operation)
        async def submit(bot, order):
            from live import executor

            deferred = (
                executor.DeferredOrderCreation
                if action == "create"
                else executor.DeferredOrderCancellation
            )
            if not position_fill_sync.permits(bot, order):
                return deferred()
            try:
                with position_fill_sync.connector_context(bot, order):
                    instance = owner(bot)
                    async with instance._write_lock:
                        if not instance.admit(order):
                            return deferred()
                        if action == "cancel":
                            executor.record_cancel_connector_admission(bot, order)
                        try:
                            return await operation(bot, order)
                        finally:
                            # Reads completed while the call was pending cannot
                            # confirm its final outcome.
                            bot._request_authoritative_confirmation(
                                {"balance", "positions", "open_orders"}
                            )
            except position_fill_sync.WriteDeferred:
                return deferred()

        return submit

    return decorate


def matches(scope, symbol, side):
    return (scope.symbol is None or scope.symbol == symbol) and (
        scope.pside is None or scope.pside == side
    )


class OrdinaryPlanDeferred(RuntimeError):
    """Discard preparation when its current-account facts cannot be confirmed."""

    def __init__(self, reason, missing=()):
        self.reason = str(reason)
        self.missing = tuple(missing)
        super().__init__(self.reason)


@dataclass(frozen=True, eq=False)
class OrdinaryReceipts:
    """Strong references retained from actual ordinary input consumption."""

    sources: tuple
    fills: object

    def mismatch_reason(self, other):
        if self.fills is not other.fills:
            return "hsl_fill_receipt_replaced"
        if len(self.sources) != len(other.sources) or any(
            symbol != current_symbol or receipt is not current_receipt
            for (symbol, receipt), (current_symbol, current_receipt)
            in zip(self.sources, other.sources)
        ):
            return "hsl_candle_receipt_replaced"
        return None

    def matches(self, other):
        return self.mismatch_reason(other) is None



@dataclass(frozen=True)
class _OrdinaryPreparation:
    account: tuple
    strategy: object
    strategy_minute: int
    trailing_anchors: dict
    quotes: dict
    quotes_reacquired: bool = False


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
    ordinary_receipts: OrdinaryReceipts | None = None
    balance_observed_ms: int | None = None
    ordinary_receipt_ref: object | None = None

    def permission(self, symbol, side):
        if any(matches(item.scope, symbol, side) for item in self.unavailable):
            return "unavailable", None
        decisions = [d for d in self.decisions if matches(d.scope, symbol, side)]
        if len(decisions) > 1:
            raise runtime.InvalidHslOutput("overlapping HSL scopes")
        return (
            (decisions[0].action, decisions[0].execution_type)
            if decisions
            else (None, None)
        )


class Owner:
    def __init__(self, bot):
        import asyncio

        self._write_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        from time import monotonic

        self._schedule_clock = monotonic
        self._next_sources = self._next_history = 0.0
        self._ordinary = None
        self._ordinary_quote_retry = None
        self._account_report = None
        self._ordinary_started_ms = None
        self._cycle_running = False
        self._running = False
        self.bot = bot
        self.sources = {}
        self.quotes = {}
        self._waves = {}
        self._retired_ordinary_receipts = weakref.WeakSet()
        self._quote_tasks = {}
        self._quote_started = {}
        self._discard_quote_tasks = set()
        self._quote_cursor = 0
        self._position_observation = None
        # Transport clock configuration survives replacement of disposable
        # HSL observation/replay owners; real live uses UTC by default.
        self._observation_clock_offset = getattr(bot, "_hsl_observation_clock_offset", None)
        self._account_clock_projector = getattr(bot, "_hsl_account_clock_projector", None)

    def capture(self, quotes=None, *, target=None, ordinary_receipts=None):
        # Factual caches only. The adapter clips every observation again to the
        # current window; neither prior GREEN nor prior RED is an input.
        bot = self.bot
        if quotes is not None:
            self.quotes.update(quotes)
        now_utc, max_age = self.remember_position()
        balance_observed_ms = bot._ensure_freshness_ledger().surfaces["balance"].updated_ms
        requests, unavailable = runtime.capture(
            bot,
            self.quotes,
            self.sources,
            symbols={
                side: set(bot.approved_coins_minus_ignored_coins[side])
                for side in ("long", "short")
            },
            now_ms=int(bot.get_exchange_time()),
            utc_now_ms=now_utc,
            max_current_age_ms=max_age,
            position_observation=self._position_observation,
            use_observed_fills=True,
            target=target,
            observation_clock_offset_ms=(
                0 if self._observation_clock_offset is None
                else self._observation_clock_offset(now_utc)
            ),
            account_clock_projector=self._account_clock_projector,
            fact_clock_projector=getattr(bot, "_hsl_fact_clock_projector", None),
        )
        evaluated = runtime.evaluate(requests)
        if any(not isinstance(item, (runtime.Decision, runtime.Unavailable)) for item in evaluated):
            raise runtime.InvalidHslOutput("invalid HSL evaluation result type")
        decisions = tuple(item for item in evaluated if isinstance(item, runtime.Decision))
        unavailable += tuple(item for item in evaluated if isinstance(item, runtime.Unavailable))
        wave = Wave(
            now_utc,
            self._position_observation.observed_ms,
            tuple(stamp for request in requests for stamp in request.mark_observed_ms),
            decisions,
            unavailable,
            runtime.observe_positions(bot).payload,
            runtime.observe_open_orders(bot),
            bot.get_raw_balance(),
            int(getattr(bot, "_account_invalidation_generation", 0)),
            balance_observed_ms=balance_observed_ms,
            ordinary_receipts=ordinary_receipts,
        )
        return wave

    def remember_position(self):
        from utils import utc_ms

        bot = self.bot
        current_position = runtime.observe_positions(bot)
        previous = self._position_observation
        now_utc = int(utc_ms())
        max_age = int(bot._live_market_snapshot_max_age_ms())
        retain = (
            previous is not None
            and previous.payload == current_position.payload
            and previous.generation == current_position.generation
            and 0 <= now_utc - previous.observed_ms <= max_age // 2
        )
        if retain and self._account_clock_projector is not None:
            observed = self._account_clock_projector(
                "positions", previous.revision, previous.observed_ms
            )
            now_event = int(bot.get_exchange_time())
            if type(observed) is not int or not 0 < observed <= now_event:
                raise ValueError("invalid projected HSL account clock")
            retain = now_event - observed <= max_age // 2
        if not retain:
            self._position_observation = current_position
        return now_utc, max_age

    def report(self, wave):
        from live.hsl_diagnostics import record

        record(self.bot, wave)

    def _account_matches(self, wave, now):
        return self._account_mismatch_reason(wave, now) is None

    def _account_mismatch_reason(self, wave, now):
        bot = self.bot
        if self._shutdown_requested():
            return "shutdown"
        if self._refresh_lock.locked():
            return "account_refresh_in_progress"
        receipts = wave.ordinary_receipts
        if wave.ordinary_receipt_ref is not None:
            receipts = wave.ordinary_receipt_ref()
            if receipts is None:
                return "ordinary_plan_retired"
        if receipts is not None:
            if receipts in self._retired_ordinary_receipts:
                return "ordinary_plan_retired"
            reason = receipts.mismatch_reason(self.ordinary_receipts())
            if reason is not None:
                return reason
        if (
            wave.required_fills is not None
            and wave.required_fills != self.required_fill_facts()
        ):
            return "required_fill_facts_changed"
        if self._current_account_unavailable(now):
            return "current_account_inputs_unavailable"
        return self._account_facts_mismatch_reason(wave)

    def _account_facts_mismatch_reason(self, wave):
        bot = self.bot
        # Every write requires the current account generation. Changed size/basis
        # requires replanning, including for a panic close.
        if (
            wave.generation != int(getattr(bot, "_account_invalidation_generation", 0))
            or wave.positions != runtime.observe_positions(bot).payload
            or wave.open_orders != runtime.observe_open_orders(bot)
        ):
            return "account_facts_changed"
        if wave.balance != bot.get_raw_balance():
            return "raw_balance_changed"
        return None

    def _can_retry_expired_account(self, wave, now):
        """Check whether an account-unavailable rejection was caused only by expiry."""
        bot = self.bot
        ledger = bot._ensure_freshness_ledger()
        pending = getattr(bot, "_authoritative_pending_confirmations", {})
        for surface in ("balance", "positions", "open_orders"):
            state = ledger.surfaces[surface]
            if not 0 < state.updated_ms <= now or int(pending.get(surface, 0)) > state.epoch:
                return False
        return self._account_facts_mismatch_reason(wave) is None

    def _defer_ordinary(self, reason, *, phase, missing=()):
        # This describes an existing discard, never an admission decision.
        from live.events import run_diagnostic_step

        bot = self.bot
        details = dict(
            context=f"ordinary {phase}: {reason}",
            defer_reason=reason,
            missing=list(missing),
            required=["balance", "positions", "open_orders"],
            invalid={"ordinary_plan": [{"reason": reason}]},
            epoch=bot._ensure_freshness_ledger().epoch,
        )
        run_diagnostic_step(
            "observe ordinary plan discard",
            lambda: bot._log_staged_execution_defer(details),
        )

    async def prepare_ordinary_consumption(self, prepared_account):
        """Rebuild fill-dependent inputs after remote work, then freeze their cohort."""
        candles = await self.bot._preload_trailing_candles(allow_remote_fetch=False)
        return await self._consume_ordinary_candles(prepared_account, candles)

    async def _consume_ordinary_candles(self, prepared_account, candles):
        if prepared_account is not None:
            self.require_ordinary_account(prepared_account)
        receipts = self.ordinary_receipts()
        # The prepared-candle route is a complete nonyielding derivation: no
        # confirmation state can leak from an abandoned cache-read cohort.
        await self.bot.update_trailing_data(
            allow_remote_fetch=False, prepared_candles=candles
        )
        if prepared_account is not None:
            self.require_ordinary_account(prepared_account)
        reason = receipts.mismatch_reason(self.ordinary_receipts())
        if reason is not None:
            raise OrdinaryPlanDeferred(reason)
        return receipts

    def _current_account_unavailable(self, now):
        bot = self.bot
        ledger = bot._ensure_freshness_ledger()
        pending = getattr(bot, "_authoritative_pending_confirmations", {})
        return tuple(
            surface
            for surface in ("balance", "positions", "open_orders")
            if (
                (state := ledger.surfaces[surface]).updated_ms <= 0
                or not 0 <= now - state.updated_ms <= bot._live_market_snapshot_max_age_ms()
                or int(pending.get(surface, 0)) > state.epoch
            )
        )

    async def refresh_ordinary_account(self, prepared_account, *, allow_new_refresh: bool = True) -> bool:
        """Confirm unchanged facts after slow preparation, before capturing permission.

        A confirming read may advance receipt times, never rewrite an old one.
        Changed economic facts still invalidate the prepared universe. This work
        stays in the ordinary task. Protection never awaits EMA preparation.
        With no new read allowed, return whether an existing refresh was joined.
        """
        from utils import utc_ms

        bot = self.bot
        if self._refresh_lock.locked():
            # An existing protective refresh may already supply the needed
            # confirmation. Wait for its complete cohort without cancelling it.
            async with self._refresh_lock:
                pass
        elif not allow_new_refresh:
            return False
        if self._shutdown_requested():
            raise OrdinaryPlanDeferred("shutdown")
        if not self.same_account_facts(prepared_account, self.account_facts()):
            raise OrdinaryPlanDeferred("account_facts_changed_during_preparation")
        if allow_new_refresh and self._current_account_unavailable(int(utc_ms())):
            from ccxt.base.errors import InvalidNonce, NetworkError
            from live.state_refresh import AuthoritativeSurfaceUnavailable

            try:
                refreshed = await bot.refresh_protective_authoritative_state(
                    require_balance=True
                )
            except InvalidNonce:
                raise
            except (NetworkError, AuthoritativeSurfaceUnavailable) as exc:
                raise OrdinaryPlanDeferred(
                    "current_account_refresh_unavailable",
                    self._current_account_unavailable(int(utc_ms())),
                ) from exc
            if not refreshed:
                raise OrdinaryPlanDeferred(
                    "current_account_refresh_unavailable",
                    self._current_account_unavailable(int(utc_ms())),
                )
        self.require_ordinary_account(prepared_account)
        return True

    def require_ordinary_account(self, prepared_account):
        from utils import utc_ms

        if self._shutdown_requested():
            raise OrdinaryPlanDeferred("shutdown")
        if not self.same_account_facts(prepared_account, self.account_facts()):
            raise OrdinaryPlanDeferred("account_facts_changed_during_preparation")
        missing = self._current_account_unavailable(int(utc_ms()))
        if self._refresh_lock.locked() or missing:
            raise OrdinaryPlanDeferred("current_account_inputs_unavailable", missing)

    def admit(self, order):
        from utils import utc_ms

        if not position_fill_sync.permits(self.bot, order):
            return False
        wave = self._waves.get(order.get("_hsl_wave"))
        if not isinstance(wave, Wave) or not self._account_matches(wave, int(utc_ms())):
            return False
        # Capture is side-effect free with respect to diagnostic sinks. Reporting
        # happens after the protective wave, outside the write freshness budget.
        if order.get("_hsl_retired_panic") and self.bot.get_forced_PB_mode(
            order["position_side"], order["symbol"]
        ) in {"panic", "manual"}:
            return False
        current = self.capture(target=(order["symbol"], order["position_side"]))
        now = int(utc_ms())
        # A long synchronous reconstruction can consume the remaining freshness
        # budget even without an await. Recheck at the actual write boundary.
        if not self._account_matches(wave, now):
            return False
        scopes = [
            d.scope
            for d in current.decisions
            if matches(d.scope, order["symbol"], order["position_side"])
        ]
        for symbol, sides in self.bot.positions.items():
            if any(
                p["size"] != 0 and any(matches(scope, symbol, side) for scope in scopes)
                for side, p in sides.items()
            ):
                quote = self.quotes.get(symbol)
                if (
                    quote is None
                    or not quote.is_valid()
                    or not 0 < quote.fetched_ms <= now
                    or now - quote.fetched_ms
                    > self.bot._live_market_snapshot_max_age_ms()
                ):
                    return False
        before = wave.permission(order["symbol"], order["position_side"])
        after = current.permission(order["symbol"], order["position_side"])
        return after[0] != "unavailable" and before == after

    def _shutdown_requested(self):
        from passivbot import Passivbot

        return Passivbot._shutdown_requested(self.bot)

    def bind(self, wave, cancels, creates, *, ordinary=False):
        if ordinary:
            wave = replace(
                wave,
                required_fills=self.required_fill_facts(),
                ordinary_receipt_ref=(
                    weakref.ref(wave.ordinary_receipts)
                    if wave.ordinary_receipts is not None else None
                ),
                ordinary_receipts=None,
            )
        # A fresh owner must not recycle an old order's receipt identifier.
        token = uuid4().hex
        self._waves[token] = wave
        while len(self._waves) > 8:
            del self._waves[next(iter(self._waves))]
        for order in (*cancels, *creates):
            order["_hsl_wave"] = token

    async def protect(self, *, deferred_reports=None):
        """One finite wave over all currently evaluable RED scopes.

        Account refresh belongs to the caller. This does not wait for flattening,
        history repair, cooldown, or another scope's close to fill.
        An outer execution pass may collect reports until its ordinary writes finish.
        """
        bot = self.bot
        if self._shutdown_requested():
            return False
        if not runtime.enabled(bot):
            return False
        symbols = {
            symbol
            for symbol, sides in bot.positions.items()
            if any(position["size"] != 0 for position in sides.values())
        }
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

            green_symbols = [
                symbol
                for symbol, orders in bot.open_orders.items()
                if orders
                and any(
                    wave.permission(symbol, side)[0] == "normal"
                    for side in ("long", "short")
                )
            ]
            actual = (
                reconciler.snapshot_actual_orders(bot, green_symbols)
                if green_symbols
                else {}
            )
            retired = [
                dict(order, _hsl_retired_panic=True)
                for symbol, orders in actual.items()
                for order in orders
                if wave.permission(symbol, order["position_side"])[0] == "normal"
                and order["pb_order_type"].rsplit("_", 1)[0] == "close_panic"
                and bot.get_forced_PB_mode(order["position_side"], symbol)
                not in {"panic", "manual"}
            ]
            if not targets and not retired:
                return False
            bot._record_market_snapshot_surface(sorted(quotes), quotes)
            cancels, creates = [], []
            if targets:
                cancels, creates = (
                    await bot.calc_protective_panic_orders_to_cancel_and_create(
                        target_psides_by_symbol=targets,
                        market_snapshots=quotes,
                        execution_types=execution_types,
                    )
                )
            cancels.extend(retired)
            if self._shutdown_requested():
                return False
            self.bind(wave, cancels, creates)
            await bot.execute_order_plan_to_exchange(
                cancels, creates, configure_creations=False
            )
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
        if not runtime.enabled(bot) or bot._pnls_manager is None:
            return set()
        now = int(bot.get_exchange_time())
        start = runtime.history_start_ms(bot, now)
        tape = runtime.capture_fills(
            bot._pnls_manager.get_events(start_ms=start), bot.c_mults
        )
        return {
            pair.symbol
            for pair in tape.pairs
            if any(start <= fill.timestamp <= now for fill in pair.fills)
        }

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
                    self.bot._get_orchestrator_market_snapshots(
                        [symbol], record_surface=False
                    )
                )
                self._quote_started[symbol] = monotonic()
                self._quote_cursor += 1
        if self._quote_tasks:
            await asyncio.wait(tuple(self._quote_tasks.values()), timeout=0.25)
            for symbol, task in self._quote_tasks.items():
                # The wave's time slice is not the network deadline. Keep a
                # slow, useful read alive across waves; after five seconds its
                # slot remains owned until cancellation actually completes.
                if (
                    not task.done()
                    and task not in self._discard_quote_tasks
                    and monotonic() - self._quote_started[symbol] >= 5.0
                ):
                    self._discard_quote_tasks.add(task)
                    task.cancel()
        collect()
        self.quotes.update(quotes)
        # Earlier factual quotes remain usable only until their original TTL;
        # capture validates each fetched time, including held marks.
        return {
            symbol: quote for symbol, quote in self.quotes.items() if symbol in symbols
        }

    def poll_inputs(self):
        for name in ("_fill_task", "_source_task"):
            task = getattr(self, name, None)
            if task is not None and task.done():
                task.result()

    async def wait_for_ordinary_fill_refresh(self):
        """Give the owned fill refresh a bounded head start before ordinary capture."""
        import asyncio

        if self._refresh_lock.locked():
            # The owner schedules its next fill task after this account read,
            # before yielding to waiters. Select the task after that handoff.
            async with self._refresh_lock:
                pass
        if self._shutdown_requested():
            raise OrdinaryPlanDeferred("shutdown")
        task = getattr(self, "_fill_task", None)
        if task is None:
            return
        if not task.done():
            # Neither a timeout nor ordinary cancellation cancels the owner's read.
            await asyncio.wait((task,), timeout=2.0)
            if self._shutdown_requested():
                raise OrdinaryPlanDeferred("shutdown")
        if task.done():
            task.result()

    def schedule_history(self):
        import asyncio

        task = getattr(self, "_fill_task", None)
        if not position_fill_sync.fetch_ready(self.bot):
            return
        if task is not None:
            if not task.done():
                return
            task.result()  # Unexpected failures remain fatal, including late ones.
        # Ordinary fill consumers keep their own readiness gates. This task only
        # refreshes their canonical manager; it does not stamp synthetic readiness.
        now = self._schedule_clock()
        kwargs = {}
        if runtime.enabled(self.bot) and now >= getattr(
            self, "_next_history_window", 0.0
        ):
            end = int(self.bot.get_exchange_time())
            kwargs["since_ms"] = runtime.history_start_ms(self.bot, end)
            self._next_history_window = now + 900.0
        self._fill_task = asyncio.create_task(self._refresh_history(kwargs))

    async def _refresh_history(self, kwargs):
        import asyncio
        from ccxt.base.errors import NetworkError
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from passivbot_exceptions import FillEventDataError

        # Give the actual millisecond clock a chance to separate the account
        # read from the tail read. A tied observation stays explicitly uncertain.
        await asyncio.sleep(0.001)
        try:
            ready = await self.bot.update_pnls(source="hsl", **kwargs)
        except (
            NetworkError,
            OSError,
            AuthoritativeSurfaceUnavailable,
            FillEventDataError,
        ) as exc:
            logging.warning(
                "[risk] hsl history repair unavailable | error_type=%s",
                type(exc).__name__,
            )
            ready = False
        if not ready:
            # A failed/incomplete refresh cannot leave an older tape certified
            # for ordinary fill consumers. The canonical successful refresh owns
            # renewal; account-only and protective actions do not require fills.
            self.bot._request_authoritative_confirmation({"fills"})
        return ready

    def schedule_sources(self):
        if not runtime.enabled(self.bot):
            return
        import asyncio
        from live.hsl_candles import CandleSourceReader

        task = getattr(self, "_source_task", None)
        if task is not None:
            if not task.done():
                return
            task.result()
        now = self._schedule_clock()
        if now < self._next_sources:
            return
        self._next_sources = now + 60.0
        if not hasattr(self, "_reader"):
            from utils import utc_ms

            self._reader = CandleSourceReader(
                self.bot.cm, observation_clock=lambda: int(utc_ms()),
                receipt_observer=getattr(self.bot, "_hsl_candle_receipt_observer", None),
            )
        bot = self.bot
        symbols = {
            symbol
            for symbol, sides in bot.positions.items()
            if any(position["size"] != 0 for position in sides.values())
        }
        symbols.update(self.history_symbols())
        # Own the requested cohort before protection can close its last held
        # position and before the canonical closing fill becomes visible.
        self._source_task = asyncio.create_task(self._read_sources(tuple(sorted(symbols))))

    async def _read_sources(self, symbols):
        bot = self.bot
        self.sources = {
            symbol: source
            for symbol, source in self.sources.items()
            if symbol in symbols
        }
        now = int(bot.get_exchange_time())
        start = runtime.history_start_ms(bot, now)
        # Publish each factual result as it arrives; no all-symbol barrier.
        for symbol in sorted(symbols):
            self.sources[symbol] = await self._reader.acquire(
                symbol,
                start=start,
                end=now,
                timeout_seconds=15.0,
                allow_remote_fetch=True,
            )

    def required_fill_facts(self):
        bot = self.bot
        if "fills" not in bot._staged_planner_required_surfaces(
            include_market_snapshot=False
        ):
            return None
        state = bot._ensure_freshness_ledger().surfaces["fills"]
        minimum = max(
            1,
            int(
                getattr(bot, "_authoritative_pending_confirmations", {}).get("fills", 0)
            ),
        )
        return (state.signature, state.epoch >= minimum)

    def account_facts(self):
        bot = self.bot
        return (
            runtime.observe_positions(bot),
            runtime.observe_open_orders(bot),
            bot.get_hysteresis_snapped_balance(),
            self.required_fill_facts(),
        )

    @staticmethod
    def same_account_facts(before, after):
        # Observation timestamps may advance on a confirming unchanged read.
        # Changing facts or invalidation still void the entire prepared universe.
        return (
            before[0].payload == after[0].payload
            and before[0].generation == after[0].generation
            and before[1:] == after[1:]
        )

    def ordinary_receipts(self):
        # Quotes acquired by preparation are deliberately excluded. HSL candle
        # and fill acquisitions are immutable; even equal replacement payloads
        # require a plan prepared against the new actual receipt.
        return OrdinaryReceipts(
            tuple(sorted(self.sources.items())),
            getattr(self.bot, "_hsl_fill_observation", None),
        )

    async def _ordinary_plan(self):
        bot = self.bot
        account = self.account_facts()
        hsl_enabled = runtime.enabled(bot)
        initial_receipts = None if hsl_enabled else self.ordinary_receipts()
        plan = None
        try:
            plan = await self._prepare_ordinary_plan(account, initial_receipts)
            return plan
        finally:
            self._release_planning_wave(plan)

    async def _prepare_ordinary_universe(self, account):
        bot = self.bot
        await bot.prepare_planning_universe()
        if not self.same_account_facts(account, self.account_facts()):
            self._defer_ordinary("account_facts_changed", phase="preparation")
            return None
        if not await bot.refresh_market_state_if_needed(trailing_acquisition_only=True):
            self._defer_ordinary("market_inputs_unavailable", phase="preparation")
            return None
        if not self.same_account_facts(account, self.account_facts()):
            self._defer_ordinary("account_facts_changed", phase="preparation")
            return None
        ready, details = bot._staged_execution_ready_state(
            include_market_snapshot=False, context="hsl ordinary planning"
        )
        if not ready:
            self._defer_ordinary(
                "staged_inputs_unavailable", phase="preparation",
                missing=details.get("missing", ()),
            )
            return None
        return True

    async def _prepare_ordinary_plan(self, account, initial_receipts):
        bot = self.bot
        if not await self._prepare_ordinary_universe(account):
            return None
        try:
            cancels, creates = await bot.calc_orders_to_cancel_and_create()
        except OrdinaryPlanDeferred as exc:
            self._defer_ordinary(exc.reason, phase="consumption", missing=exc.missing)
            return None
        return self._freeze_ordinary_plan(account, initial_receipts, cancels, creates)

    async def _acquire_ordinary_inputs(self):
        from live import reconciler
        from utils import utc_ms

        if self._shutdown_requested():
            return None
        missing = self._current_account_unavailable(int(utc_ms()))
        if missing:
            self._defer_ordinary(
                "current_account_inputs_unavailable", phase="preparation", missing=missing
            )
            return None
        bot = self.bot
        account = self.account_facts()
        if not await self._prepare_ordinary_universe(account):
            return None
        try:
            symbols = sorted(set(bot.active_symbols))
            strategy_minute = bot._completed_candle_health_now_ms() // 60_000
            strategy = await bot._prepare_orchestrator_inputs(symbols)
            await self.wait_for_ordinary_fill_refresh()
            anchors = bot._get_last_position_change_anchors()
            quotes = await self._acquire_ordinary_quotes(symbols)
            if not self.same_account_facts(account, self.account_facts()):
                raise OrdinaryPlanDeferred("account_facts_changed_during_preparation")
            return _OrdinaryPreparation(account, strategy, strategy_minute, anchors, quotes)
        except Exception:
            reconciler.record_ordinary_planning_failure(bot)
            raise

    async def _acquire_ordinary_quotes(self, symbols, *, refresh=False):
        try:
            return await self.bot._get_orchestrator_market_snapshots(
                symbols, record_surface=False, refresh=refresh
            )
        except MarketSnapshotUnavailable as exc:
            raise OrdinaryPlanDeferred("prepared_market_inputs_unavailable") from exc

    def _require_reusable_strategy(self, preparation):
        if (
            not preparation.strategy.allow_quote_retry
            or preparation.strategy_minute != self.bot._completed_candle_health_now_ms() // 60_000
        ):
            raise OrdinaryPlanDeferred("prepared_strategy_inputs_expired")

    async def _reacquire_prepared_quotes(self, preparation):
        from live import reconciler

        try:
            self._require_reusable_strategy(preparation)
            # Refresh the full set once. Successive partial reads can keep
            # aging the previous cache hits on symbol-scoped connectors.
            quotes = await self._acquire_ordinary_quotes(
                preparation.strategy.symbols, refresh=True
            )
            return replace(preparation, quotes=quotes, quotes_reacquired=True)
        except Exception:
            reconciler.record_ordinary_planning_failure(self.bot)
            raise

    def _ready_ordinary_quotes(self, preparation):
        from utils import utc_ms

        bot = self.bot
        symbols = preparation.strategy.symbols
        if not symbols:
            return {}
        max_age_ms = bot._live_market_snapshot_fetch_max_age_ms()
        quotes = bot.market_snapshot_provider.get_cached_snapshots(
            symbols, max_age_ms=max_age_ms
        )
        if quotes is not None or bot.exchange != "hyperliquid":
            return quotes
        # Hyperliquid's explicit canonical fallback need not populate the provider.
        now = int(utc_ms())
        fallback = preparation.quotes
        if set(fallback) == set(symbols) and all(
            0 < snap.fetched_ms <= now
            and now - snap.fetched_ms <= max_age_ms
            and snap.is_valid()
            for snap in fallback.values()
        ):
            return fallback
        return None

    def _ordinary_account_cohort(self):
        bot = self.bot
        ledger = bot._ensure_freshness_ledger()
        return (
            ledger.epoch,
            int(getattr(bot, "_account_invalidation_generation", 0)),
            bot.get_raw_balance(),
            tuple(ledger.surfaces[name].revision for name in ("balance", "positions", "open_orders")),
        )

    async def _plan_with_current_account(self, preparation, account_cohort):
        """Calculate and reconcile under the account cohort that just ran protection."""
        from live import reconciler

        bot = self.bot
        plan = None
        reconciling = False
        try:
            if self._ordinary_account_cohort() != account_cohort:
                raise OrdinaryPlanDeferred("account_cohort_changed_during_protection")
            self.require_ordinary_account(preparation.account)
            strategy = preparation.strategy
            if not bot._orchestrator_preparation_is_current(strategy):
                raise OrdinaryPlanDeferred("prepared_universe_changed")
            anchors = preparation.trailing_anchors
            if anchors != bot._get_last_position_change_anchors():
                raise OrdinaryPlanDeferred("prepared_trailing_inputs_changed")
            quotes = self._ready_ordinary_quotes(preparation)
            if quotes is None:
                raise OrdinaryPlanDeferred("prepared_market_inputs_unavailable")
            candles = {}
            if anchors:
                # The remote preload ran in preparation. Read the current local
                # rows after protection, including same-minute corrections.
                candles = await bot._preload_trailing_candles(allow_remote_fetch=False)
                self.poll_inputs()
                if self._ordinary_account_cohort() != account_cohort:
                    raise OrdinaryPlanDeferred("account_cohort_changed_during_consumption")
                if anchors != bot._get_last_position_change_anchors():
                    raise OrdinaryPlanDeferred("prepared_trailing_inputs_changed")
                if not bot._orchestrator_preparation_is_current(strategy):
                    raise OrdinaryPlanDeferred("prepared_universe_changed")
                quotes = self._ready_ordinary_quotes(preparation)
                if quotes is None:
                    raise OrdinaryPlanDeferred("prepared_market_inputs_unavailable")
            if preparation.quotes_reacquired:
                self._require_reusable_strategy(preparation)
            receipts = await self._consume_ordinary_candles(preparation.account, candles)
            if self._ordinary_account_cohort() != account_cohort:
                raise OrdinaryPlanDeferred("account_cohort_changed_during_consumption")
            if strategy.symbols:
                bot._record_market_snapshot_surface(strategy.symbols, quotes)
            reconciling = True
            cancels, creates = await bot._reconcile_prepared_ordinary(strategy, quotes, receipts)
            plan = self._freeze_ordinary_plan(preparation.account, None, cancels, creates)
            return plan
        except Exception:
            if not reconciling:
                reconciler.record_ordinary_planning_failure(bot)
            raise
        finally:
            self._release_planning_wave(plan)

    def _release_planning_wave(self, plan):
        # The returned plan owns the consumed receipts until execution finishes.
        wave = getattr(self.bot, "_hsl_planning_wave", None)
        if isinstance(wave, Wave) and wave.ordinary_receipts is not None:
            if plan is None:
                self._retired_ordinary_receipts.add(wave.ordinary_receipts)
            self.bot._hsl_planning_wave = replace(wave, ordinary_receipts=None)

    def _freeze_ordinary_plan(self, account, initial_receipts, cancels, creates):
        bot = self.bot
        if not self.same_account_facts(account, self.account_facts()):
            self._defer_ordinary("account_facts_changed", phase="after_reconciliation")
            return None
        receipts = bot._hsl_planning_wave.ordinary_receipts
        if not isinstance(receipts, OrdinaryReceipts):
            raise RuntimeError("ordinary plan is missing its consumed input receipt")
        current_receipts = self.ordinary_receipts()
        reason = (
            initial_receipts.mismatch_reason(current_receipts)
            if initial_receipts is not None else None
        ) or receipts.mismatch_reason(current_receipts)
        if reason is not None:
            self._defer_ordinary(reason, phase="after_reconciliation")
            return None
        # Preparation may span confirming raw-balance changes. Ordinary Rust
        # risk consumes the latest raw balance at its actual calculation, so
        # reconciliation and submission must retain that exact observation.
        if bot._hsl_planning_wave.balance != bot.get_raw_balance():
            self._defer_ordinary("raw_balance_changed", phase="after_reconciliation")
            return None
        # Empty plans need the same immutable receipt as nonempty plans. An
        # order dictionary is not the authority for the completed calculation.
        wave = replace(
            bot._hsl_planning_wave,
            required_fills=self.required_fill_facts(),
            ordinary_receipts=receipts,
        )
        return cancels, creates, bot._current_planning_snapshot, wave

    def schedule_account_reporting(self):
        import asyncio
        from live.state_refresh import publish_protective_account_report

        if self._account_report is not None and self._account_report.done():
            _retrieve_on_shutdown(self._account_report)
            self._account_report = None
        if (
            self._account_report is None
            and getattr(self.bot, "_protective_account_report", None) is not None
            and not self._shutdown_requested()
        ):
            self._account_report = asyncio.create_task(
                publish_protective_account_report(self.bot)
            )

    def cancel_inputs(self):
        self._ordinary_quote_retry = None
        for task in (
            self._ordinary,
            self._account_report,
            getattr(self, "_fill_task", None),
            getattr(self, "_source_task", None),
            *self._quote_tasks.values(),
        ):
            if task is not None:
                task.cancel()
                task.add_done_callback(_retrieve_on_shutdown)
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.cancel_pending()

    async def warmup(self):
        from ccxt.base.errors import NetworkError
        from candlestick_manager import OhlcvFetchError

        try:
            await self.bot.warmup_trading_ready_candles()
        except (NetworkError, OSError, OhlcvFetchError) as exc:
            logging.info(
                "[boot] trading-ready candle warmup skipped | error_type=%s",
                type(exc).__name__,
            )

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
                    if await self.bot.refresh_protective_authoritative_state(
                        require_balance=True
                    ):
                        self.remember_position()
                        self.schedule_history()
                        self.schedule_sources()
                        await self.protect()
                        self.schedule_account_reporting()
                except (
                    NetworkError,
                    AuthoritativeSurfaceUnavailable,
                    MarketSnapshotUnavailable,
                ) as exc:
                    logging.warning(
                        "[risk] hsl startup current I/O unavailable | error_type=%s",
                        type(exc).__name__,
                    )
                await asyncio.wait((task,), timeout=1.0)
            if task.done():
                return task.result()
        finally:
            if not task.done():
                task.cancel()
                task.add_done_callback(_retrieve_on_shutdown)

    def _collect_completed_ordinary(self, now_ms):
        task, self._ordinary = self._ordinary, None
        started_ms, self._ordinary_started_ms = self._ordinary_started_ms, None
        elapsed_ms = None if started_ms is None else max(0, now_ms - started_ms)
        try:
            return task.result(), elapsed_ms
        except OrdinaryPlanDeferred as exc:
            self._defer_ordinary(exc.reason, phase="preparation", missing=exc.missing)
            return None, elapsed_ms
        except RuntimeError as exc:
            handled, details = self.bot._handle_staged_execution_precondition_error(exc)
            if not handled:
                raise
            self.bot._log_staged_execution_defer(details)
            return None, elapsed_ms

    async def _service_ordinary(self, plan, timings, *, now):
        from time import perf_counter
        from utils import utc_ms

        bot = self.bot
        cancels, creates, snapshot, wave = plan
        reason = self._account_mismatch_reason(wave, now)
        if reason is not None:
            return reason
        bot._current_planning_snapshot = snapshot
        started = perf_counter()
        await bot.execute_order_plan_to_exchange(cancels, creates)
        if not self._shutdown_requested():
            run_diagnostic_step(
                "record ordinary acceptance",
                lambda: observe_ordinary_acceptance(bot, int(utc_ms())),
            )
            log_process_recovery(bot)
        timings["ordinary_execute"] += int((perf_counter() - started) * 1000)
        return None

    def _describe_ordinary_rejection(self, plan, *, now_ms, handoff, reason):
        cancels, creates = plan[:2]
        return run_diagnostic_step(
            "describe ordinary admission",
            lambda: describe_ordinary_admission(
                self.bot, now_ms=now_ms, handoff=handoff, reason=reason,
                create_count=len(creates), cancel_count=len(cancels),
            ),
        )

    async def cycle(self):
        """One finite production pass; offline scenario stepping uses this same owner.

        Background task ownership spans passes. A caller must not overlap cycles;
        slow ordinary work never becomes a prerequisite for the protective pass.
        """
        import asyncio
        from live.state_refresh import AuthoritativeSurfaceUnavailable
        from ccxt.base.errors import NetworkError, InvalidNonce

        if self._cycle_running:
            raise RuntimeError("HSL execution cycle is already running")
        self._cycle_running = True
        was_running, self._running = self._running, True
        bot = self.bot
        reports = []
        from utils import utc_ms
        from time import perf_counter

        started_ms = int(utc_ms())
        completed_cycle = False
        ordinary_executed = False
        retry_after_refresh = False
        timings = dict(authoritative=0, protection=0, ordinary_calculate=0, ordinary_execute=0)
        cycle_id = None
        ordinary_admission = None
        ordinary_elapsed_ms = None
        completed_receipts = None
        try:
            bot._begin_live_event_cycle(loop_start_ms=started_ms)
            cycle_id = getattr(bot, "_live_event_current_cycle_id", None)
            bot.execution_scheduled = False
            bot.state_change_detected_by_symbol = set()
            self.poll_inputs()
            plan = None
            preparation = None
            late_calculation = callable(
                getattr(bot, "_prepare_orchestrator_inputs", None)
            ) and runtime.enabled(bot)
            completed_plan = self._ordinary is not None and self._ordinary.done()
            # Retrieve completed failures before any fallible account refresh.
            if completed_plan:
                plan, ordinary_elapsed_ms = self._collect_completed_ordinary(int(utc_ms()))
                if isinstance(plan, _OrdinaryPreparation):
                    preparation, plan = plan, None
                if plan is not None:
                    completed_receipts = plan[3].ordinary_receipts
            protective_work = False
            if self._shutdown_requested():
                return dict(
                    updated=False,
                    ordinary_completed=completed_plan,
                    ordinary_executed=False,
                )
            if plan is not None or (
                preparation is not None
                and not self._refresh_lock.locked()
                and not self._current_account_unavailable(int(utc_ms()))
            ):
                # Preserve the protective opportunity on a still-current
                # cohort before the scheduled account read can block or fail.
                phase_started = perf_counter()
                protective_work = await self.protect(deferred_reports=reports)
                timings["protection"] += int((perf_counter() - phase_started) * 1000)
                if self._shutdown_requested():
                    return dict(
                        updated=False,
                        ordinary_completed=completed_plan,
                        ordinary_executed=False,
                        protective_work=protective_work,
                    )
            if plan is not None:
                gate_ms = int(utc_ms())
                reason = await self._service_ordinary(plan, timings, now=gate_ms)
                ordinary_executed = reason is None
                if reason is not None:
                    retry_after_refresh = (
                        reason == "current_account_inputs_unavailable"
                        and self._can_retry_expired_account(plan[3], gate_ms)
                    )
                    ordinary_admission = self._describe_ordinary_rejection(
                        plan, now_ms=gate_ms, handoff="ready_at_entry", reason=reason
                    )
                    if not retry_after_refresh:
                        self._defer_ordinary(reason, phase="after_protection")
            quote_retry, self._ordinary_quote_retry = self._ordinary_quote_retry, None
            if (
                preparation is not None
                and not preparation.quotes_reacquired
                and preparation.strategy.allow_quote_retry
                and self._ready_ordinary_quotes(preparation) is None
            ):
                quote_retry = preparation
                preparation, completed_plan = None, False
            if quote_retry is not None:
                # Renew expired quotes alongside the normal owner read while
                # keeping the completed candle and EMA preparation.
                self._ordinary_started_ms = int(utc_ms())
                self._ordinary = asyncio.create_task(
                    self._reacquire_prepared_quotes(quote_retry)
                )
            phase_started = perf_counter()
            refreshed = await bot.refresh_protective_authoritative_state(
                require_balance=True
            )
            timings["authoritative"] = int((perf_counter() - phase_started) * 1000)
            if not refreshed or self._shutdown_requested():
                if retry_after_refresh and not self._shutdown_requested():
                    self._defer_ordinary(
                        "current_account_inputs_unavailable", phase="after_protection"
                    )
                return dict(
                    updated=False,
                    ordinary_completed=completed_plan,
                    ordinary_executed=ordinary_executed,
                    protective_work=protective_work,
                )
            self.remember_position()
            now = self._schedule_clock()
            if now >= self._next_history:
                self.schedule_history()
                self._next_history = now + 5.0
            self.schedule_sources()
            if (
                late_calculation
                and self._ordinary is None
                and preparation is None
                and not self._current_account_unavailable(int(utc_ms()))
            ):
                # A previous write can leave no usable preparation. Start
                # from the confirming account cohort while protection runs.
                self._ordinary_started_ms = int(utc_ms())
                self._ordinary = asyncio.create_task(self._acquire_ordinary_inputs())
            account_cohort = self._ordinary_account_cohort() if late_calculation else None
            phase_started = perf_counter()
            protective_work = (
                await self.protect(deferred_reports=reports) or protective_work
            )
            timings["protection"] += int((perf_counter() - phase_started) * 1000)
            # Recheck a plan blocked only by expiry, or hand off a task that
            # just finished. Both use the protection pass just completed and
            # the same receipt and write admission checks.
            ready_at_tail = (
                not completed_plan
                and self._ordinary is not None
                and self._ordinary.done()
            )
            if (retry_after_refresh or ready_at_tail or preparation is not None) and not self._shutdown_requested():
                self.poll_inputs()  # Surface completed input failures before any write.
                if ready_at_tail:
                    completed_plan = True
                    plan, ordinary_elapsed_ms = self._collect_completed_ordinary(int(utc_ms()))
                    if isinstance(plan, _OrdinaryPreparation):
                        preparation, plan = plan, None
                    if plan is not None:
                        completed_receipts = plan[3].ordinary_receipts
                if preparation is not None:
                    phase_started = perf_counter()
                    try:
                        plan = await self._plan_with_current_account(preparation, account_cohort)
                    except OrdinaryPlanDeferred as exc:
                        if (
                            exc.reason == "prepared_market_inputs_unavailable"
                            and not preparation.quotes_reacquired
                            and preparation.strategy.allow_quote_retry
                        ):
                            self._ordinary_quote_retry = preparation
                        self._defer_ordinary(exc.reason, phase="consumption", missing=exc.missing)
                    except RuntimeError as exc:
                        handled, details = bot._handle_staged_execution_precondition_error(exc)
                        if not handled:
                            raise
                        bot._log_staged_execution_defer(details)
                    finally:
                        timings["ordinary_calculate"] = int((perf_counter() - phase_started) * 1000)
                    if plan is not None:
                        completed_receipts = plan[3].ordinary_receipts
                if plan is not None:
                    gate_ms = int(utc_ms())
                    reason = await self._service_ordinary(plan, timings, now=gate_ms)
                    ordinary_executed = reason is None
                    if reason is not None:
                        handoff = "after_protection_calculation" if preparation is not None else (
                            "after_refresh_retry" if retry_after_refresh else "same_cycle"
                        )
                        ordinary_admission = self._describe_ordinary_rejection(
                            plan, now_ms=gate_ms, handoff=handoff, reason=reason
                        )
                        self._defer_ordinary(reason, phase="after_protection")
            self.schedule_account_reporting()
            if (
                self._ordinary is None
                and self._ordinary_quote_retry is None
                and not self._shutdown_requested()
                and (not late_calculation or not self._current_account_unavailable(int(utc_ms())))
            ):
                self._ordinary_started_ms = int(utc_ms())
                self._ordinary = asyncio.create_task(
                    self._acquire_ordinary_inputs() if late_calculation else self._ordinary_plan()
                )
            completed_cycle = not self._shutdown_requested()
            return dict(
                updated=True,
                ordinary_completed=completed_plan,
                ordinary_executed=ordinary_executed,
                protective_work=protective_work,
            )
        except InvalidNonce:
            raise
        except (
            NetworkError,
            AuthoritativeSurfaceUnavailable,
            MarketSnapshotUnavailable,
        ) as exc:
            logging.warning(
                "[risk] hsl current I/O unavailable | error_type=%s", type(exc).__name__
            )
            return dict(updated=False, current_io_unavailable=True)
        finally:
            if completed_receipts is not None:
                self._retired_ordinary_receipts.add(completed_receipts)
            self._running = was_running
            self._cycle_running = False
            for wave in reports:
                self.report(wave)
            if completed_cycle:
                run_diagnostic_step(
                    "record ordinary cycle progress",
                    lambda: observe_ordinary_outer_cycle(
                        bot, int(utc_ms()), accepted=ordinary_executed
                    ),
                )
                try:
                    from live.event_emitters import emit_live_cycle_completed

                    emit_live_cycle_completed(
                        bot,
                        cycle_id=cycle_id,
                        loop_start_ms=started_ms,
                        timings_ms=timings,
                        execution_owner="hsl",
                        ordinary_executed=ordinary_executed,
                        ordinary_admission=ordinary_admission,
                        ordinary_pending=self._ordinary is not None
                        and not self._ordinary.done(),
                        ordinary_prepare_elapsed_ms=ordinary_elapsed_ms,
                        ordinary_pending_age_ms=(
                            max(0, int(utc_ms()) - self._ordinary_started_ms)
                            if self._ordinary is not None
                            and not self._ordinary.done()
                            and self._ordinary_started_ms is not None
                            else None
                        ),
                    )
                except Exception as exc:
                    logging.debug(
                        "[event] HSL cycle timing unavailable | error_type=%s",
                        type(exc).__name__,
                    )

    async def run(self):
        """Serialized writes with periodic protection while preparation is pending."""
        from utils import utc_ms

        bot = self.bot
        if self._running:
            raise RuntimeError("HSL execution owner is already running")
        self._running = True
        try:
            while not self._shutdown_requested():
                started = int(utc_ms())
                from ccxt.base.errors import ExchangeError, InvalidNonce

                try:
                    result = await self.cycle()
                except (ExchangeError, InvalidNonce) as exc:
                    if self._shutdown_requested():
                        break
                    if not await bot._maybe_recover_exchange_time_sync(
                        exc, source="run_execution_loop"
                    ):
                        raise
                    bot._health_errors += 1
                    await bot._sleep_unless_shutdown(0.5, stage="exchange_time_sync")
                    continue
                if not result["updated"] and not result.get("current_io_unavailable"):
                    await bot._sleep_unless_shutdown(0.5, stage="hsl_current_inputs")
                    continue
                bot._last_loop_duration_ms = int(utc_ms()) - started
                bot._maybe_log_health_summary()
                try:
                    bot._maybe_log_trailing_status()
                except Exception as exc:
                    logging.debug(
                        "[trailing] status presentation failed | error_type=%s",
                        type(exc).__name__,
                    )
                from live.events import run_diagnostic_step

                run_diagnostic_step(
                    "unstuck status", lambda: bot._maybe_log_unstuck_status()
                )
                await bot._sleep_unless_shutdown(
                    max(0.05, float(bot.live_value("execution_delay_seconds"))),
                    stage="hsl_execution_delay",
                )
        finally:
            self._running = False
            self.cancel_inputs()


def _retrieve_on_shutdown(task):
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logging.error(
                "[risk] hsl background task failed during shutdown | error_type=%s",
                type(error).__name__,
            )
