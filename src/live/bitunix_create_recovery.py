"""Recover ordinary Bitunix creates without replaying a trading decision."""

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ccxt.base.errors import ExchangeError, NetworkError

from live.bitunix_create_evidence import (
    BitunixCreateEvidence,
    lookup_bitunix_create_evidence,
    parse_bitunix_create_evidence,
)
from live.bitunix_create_journal import (
    MAX_CREATE_WAVE_SIZE,
    BitunixCreateAttempt,
    BitunixCreateJournalError,
    BitunixCreateObservation,
)
from live.position_fill_sync import WriteDeferred


ACCOUNT = {"balance", "positions", "open_orders"}
LOOKUP_INTERVAL_SECONDS = 10.0
REPORT_INTERVAL_SECONDS = 60.0


def _utc_ms():
    return time.time_ns() // 1_000_000


@dataclass
class _PendingCreate:
    attempt: BitunixCreateAttempt
    account_epoch: int
    fill_after_ms: int
    observation: BitunixCreateObservation | None = None
    unpersisted: BitunixCreateEvidence | None = None
    terminal: BitunixCreateEvidence | None = None
    next_lookup: float = 0.0
    lookup_error: str | None = None

    @property
    def has_conflicting_identity(self):
        return (
            self.observation is not None
            and self.observation.conflicting_order_id is not None
        )


class BitunixCreateRecovery:
    def __init__(self, bot, journal):
        self.bot = bot
        self.journal = journal
        self.attempts: dict[str, _PendingCreate] = {}
        self._active_wave = None
        self._wave_records: set[str] = set()
        self.fill_receipt = None
        self.task = None
        self._lookup_client_id = None
        self._retry_after = 0.0
        self.error = False
        self.next_report = 0.0
        self.last_report = None

    def _report(self, reason):
        now = time.monotonic()
        if self.last_report != reason or now >= self.next_report:
            logging.warning(
                "[bitunix] ordinary create recovery pending | reason=%s", reason
            )
            self.last_report = reason
            self.next_report = now + REPORT_INTERVAL_SECONDS

    def _bind(self, attempt):
        state = _PendingCreate(
            attempt,
            self.bot._ensure_freshness_ledger().epoch + 1,
            _utc_ms(),
        )
        self.attempts[attempt.client_id] = state
        return state

    @property
    def pending(self):
        return self.error or bool(self.attempts)

    def record(self, body, context):
        from live.hsl_live import active_ordinary_create_wave, active_protective_batch

        if context is not None and body["reduceOnly"] is True:
            batch = active_protective_batch(context.bot)
            if batch is not None and batch.contains(context.order, "create"):
                return None
        wave = active_ordinary_create_wave(self.bot)
        if wave is not self._active_wave:
            self._active_wave = None
            self._wave_records.clear()
        if wave is not None and (
            context is None
            or context.bot is not self.bot
            or context.action != "create"
            or context.order.get("custom_id") != body.get("clientId")
            or body["reduceOnly"] is not True
            or body["orderType"] != "LIMIT"
            or not wave.permits(self.bot, context.order, "create")
        ):
            raise WriteDeferred()
        if self.error or (self.attempts and self._active_wave is None):
            raise WriteDeferred()
        if wave is not None and (
            len(self._wave_records) >= MAX_CREATE_WAVE_SIZE
            or body["clientId"] in self._wave_records
            or self._wave_records - self.attempts.keys()
        ):
            raise WriteDeferred()
        attempt = BitunixCreateAttempt.from_wire(body, submitted_at_ms=_utc_ms())
        try:
            if wave is None:
                self.journal.record(attempt)
            else:
                # A restart cannot rejoin a durable wave. A live continuation
                # still needs the durable attempts it recorded before sending.
                pending = self.journal.load_pending_all()
                if {durable.client_id: durable for durable, _ in pending} != {
                    client_id: state.attempt
                    for client_id, state in self.attempts.items()
                }:
                    raise WriteDeferred()
                self.journal.record(
                    attempt,
                    wave_id=wave.wave_id,
                    expected_client_ids=frozenset(self._wave_records),
                )
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            raise WriteDeferred() from exc
        self._bind(attempt)
        if wave is not None:
            self._active_wave = wave
            self._wave_records.add(attempt.client_id)
        return attempt

    def record_acknowledgment(self, attempt, order_id) -> bool:
        """Persist the venue identity from an exact ACK while retaining the attempt."""
        state = self.attempts.get(attempt.client_id)
        if state is None or state.attempt != attempt:
            self.error = True
            self._report("missing_record")
            return False
        # Zero is a lower bound. An ACK says nothing about the current fill.
        return (
            self._persist_observation(state, order_id, Decimal(0))
            and not state.has_conflicting_identity
        )

    def resolve(self, attempt, reason, *, evidence=None) -> bool:
        try:
            values = (
                {}
                if evidence is None
                else {
                    "order_id": evidence.order_id,
                    "filled_qty": evidence.filled_qty,
                }
            )
            if not self.journal.resolve(attempt.client_id, **values):
                return False
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return False
        level = (
            logging.info
            if _utc_ms() - attempt.submitted_at_ms >= 15_000
            else logging.debug
        )
        self.attempts.pop(attempt.client_id, None)
        if not self.attempts:
            self.last_report = None
        level("[bitunix] ordinary create recovered | outcome=%s", reason)
        return True

    def poll(self, request):
        """Consume committed account evidence and schedule missing detail off the plan path."""
        try:
            pending = {
                attempt.client_id: (attempt, observation)
                for attempt, observation in self.journal.load_pending_all()
            }
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return
        if self.attempts.keys() - pending.keys():
            self.error = True
            self._report("missing_record")
            return
        if any(
            state.attempt != pending[client_id][0]
            for client_id, state in self.attempts.items()
        ):
            self.error = True
            self._report("conflicting_journal_record")
            return
        self.error = False
        for client_id, (attempt, observation) in pending.items():
            state = self.attempts.get(client_id) or self._bind(attempt)
            state.observation = observation
        for state in self.attempts.values():
            if state.unpersisted is not None:
                self._observe_lookup(state, state.unpersisted)
        looking_up = self.attempts.get(self._lookup_client_id)
        if (
            self.task is not None
            and self.task.done()
            and (looking_up is None or looking_up.unpersisted is None)
        ):
            task, self.task = self.task, None
            self._lookup_client_id = None
            attempt, evidence, error, retry_after = task.result()
            self._retry_after = max(self._retry_after, retry_after)
            state = self.attempts.get(attempt.client_id)
            if state is not None and state.attempt == attempt:
                state.lookup_error = error
                if evidence is not None:
                    self._observe_lookup(state, evidence)
        now = time.monotonic()
        ready = []
        reasons = set()
        for state in tuple(self.attempts.values()):
            if state.unpersisted is not None:
                continue
            if state.has_conflicting_identity:
                reasons.add("conflicting_order_identity")
                continue
            if self._resolve_from_account(state):
                continue
            if (
                not state.has_conflicting_identity
                and now >= state.next_lookup
                and self._account_current(state)
            ):
                ready.append(state)
            if _utc_ms() - state.attempt.submitted_at_ms >= 15_000:
                reason = (
                    "awaiting_terminal_confirmation"
                    if state.terminal is not None
                    else "awaiting_order_evidence"
                )
                reasons.add(state.lookup_error or reason)
        if reasons and not self.error:
            self._report(",".join(sorted(reasons)))
        if self.task is None and now >= self._retry_after and ready:
            # First give the normal post-write account refresh an opportunity to
            # adopt a working order. Startup attempts also require a new cohort.
            state = min(ready, key=lambda item: item.next_lookup)
            state.next_lookup = now + LOOKUP_INTERVAL_SECONDS
            self._lookup_client_id = state.attempt.client_id
            self.task = asyncio.create_task(self._lookup(request, state.attempt))

    async def _lookup(self, request, attempt):
        try:
            return (
                attempt,
                await lookup_bitunix_create_evidence(request, attempt),
                None,
                0.0,
            )
        except (ExchangeError, NetworkError) as exc:
            # Lookup failure degrades only create admission. The caller keeps
            # running the normal account, cancellation and protection paths.
            retry = getattr(exc, "retry_after_seconds", 0.0)
            retry_after = time.monotonic() + retry if retry else 0.0
            return attempt, None, type(exc).__name__, retry_after

    def _persist_observation(self, state, order_id, filled_qty):
        try:
            observation = self.journal.observe(
                state.attempt.client_id, order_id, filled_qty
            )
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return False
        if observation is None:
            self.error = True
            self._report("missing_record")
            return False
        state.observation = observation
        return True

    def _observe_lookup(self, state, evidence):
        previous = state.observation
        if not self._persist_observation(state, evidence.order_id, evidence.filled_qty):
            state.unpersisted = evidence
            return False
        state.unpersisted = None
        terminal = (
            evidence.disposition == "terminal"
            and evidence != state.terminal
            and evidence.filled_qty >= state.observation.filled_qty
        )
        if terminal:
            state.terminal = evidence
        if terminal or previous != state.observation:
            # Evidence acquired by a separate lookup requires a later account
            # cohort. Repeating the same proof does not reset its confirmation.
            state.account_epoch = self.bot._ensure_freshness_ledger().epoch + 1
            state.fill_after_ms = _utc_ms()
        return True

    def _account_current(self, state):
        ledger = self.bot._ensure_freshness_ledger()
        pending = self.bot._authoritative_pending_confirmations
        return (
            ledger.epoch >= state.account_epoch
            and ACCOUNT <= ledger.surfaces_at_epoch()
            and all(ledger.epoch >= pending.get(surface, 0) for surface in ACCOUNT)
        )

    def _consistent(self, state, evidence):
        return (
            state.observation is not None
            and not state.has_conflicting_identity
            and state.observation.order_id == evidence.order_id
            and state.observation.filled_qty <= evidence.filled_qty
        )

    def _resolve_from_account(self, state):
        if not self._account_current(state):
            return False
        attempt = state.attempt
        lookup_pending = self._lookup_client_id == attempt.client_id
        venue_order_id = (
            state.observation.order_id if state.observation is not None else None
        )
        matches = [
            row
            for row in self.bot.fetched_open_orders
            if row.get("clientOrderId") == attempt.client_id
            or venue_order_id is not None
            and row.get("id") == venue_order_id
        ]
        observations = []
        for row in matches:
            evidence = parse_bitunix_create_evidence(
                row.get("info"),
                attempt,
                confirmed_order_id=venue_order_id,
            )
            if evidence is not None:
                if not self._persist_observation(
                    state, evidence.order_id, evidence.filled_qty
                ):
                    state.unpersisted = evidence
                    return True
                observations.append(evidence)
        if state.terminal is not None:
            if (
                not matches
                and not lookup_pending
                and self._consistent(state, state.terminal)
                and self._fills_confirmed(state, state.terminal, require_refresh=True)
            ):
                return self.resolve(attempt, "terminal", evidence=state.terminal)
        elif len(matches) == 1 and observations and not lookup_pending:
            evidence = observations[0]
            if (
                evidence.disposition == "working"
                and self._consistent(state, evidence)
                and self._fills_confirmed(state, evidence, require_refresh=False)
            ):
                return self.resolve(attempt, "working", evidence=evidence)
        return False

    def _fills_confirmed(self, state, evidence, *, require_refresh):
        bot = self.bot
        manager = getattr(bot, "_pnls_manager", None)
        if not require_refresh and evidence.filled_qty == 0:
            return (
                manager is None
                or _observed_filled_qty(manager, state.attempt, evidence.order_id) == 0
            )
        if self.fill_receipt is None or manager is None:
            return False
        observed_manager, interval = self.fill_receipt
        ledger = bot._ensure_freshness_ledger()
        if (
            observed_manager is not manager
            or interval[0] <= state.fill_after_ms
            or ledger.surface_epoch("fills")
            < bot._authoritative_pending_confirmations.get("fills", 0)
            or ledger.surface_updated_ms("fills") < interval[1]
        ):
            return False
        return (
            _observed_filled_qty(manager, state.attempt, evidence.order_id)
            == evidence.filled_qty
        )

    def record_fill_refresh(self, manager, interval):
        self.fill_receipt = (manager, interval)

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                # The attempt is already durable.
                pass


def _observed_filled_qty(manager, attempt, order_id):
    # FillEventsManager can aggregate several trade rows into one fill. Use
    # unique venue trade IDs and the original decimal quantities.
    quantities = {}
    for event in manager.get_events(start_ms=max(0, attempt.submitted_at_ms - 60_000)):
        for source in event.raw or ():
            trade = source.get("data", {})
            if (
                source.get("source") != "fetch_my_trades"
                or trade.get("order") != order_id
            ):
                continue
            info = trade.get("info", {})
            if (
                info.get("symbol") != attempt.symbol
                or event.position_side != attempt.position_side
            ):
                return None
            if trade.get("clientOrderId") not in (None, "", attempt.client_id):
                return None
            trade_id = trade.get("id")
            try:
                qty = Decimal(str(info.get("qty")))
            except InvalidOperation:
                return None
            if not trade_id or not qty.is_finite() or qty <= 0:
                return None
            if quantities.setdefault(trade_id, qty) != qty:
                return None
    return sum(quantities.values(), Decimal(0))
