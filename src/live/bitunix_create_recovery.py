"""Recover one ordinary Bitunix create without replaying a trading decision."""

import asyncio
import logging
import time
from decimal import Decimal, InvalidOperation

from ccxt.base.errors import ExchangeError, NetworkError

from live.bitunix_create_evidence import (
    lookup_bitunix_create_evidence,
    parse_bitunix_create_evidence,
)
from live.bitunix_create_journal import BitunixCreateAttempt, BitunixCreateJournalError
from live.position_fill_sync import WriteDeferred


ACCOUNT = {"balance", "positions", "open_orders"}
LOOKUP_INTERVAL_SECONDS = 10.0
REPORT_INTERVAL_SECONDS = 60.0


def _utc_ms():
    return time.time_ns() // 1_000_000


class BitunixCreateRecovery:
    def __init__(self, bot, journal):
        self.bot = bot
        self.journal = journal
        self.attempt = None
        self.account_epoch = 0
        self.fill_after_ms = 0
        self.fill_receipt = None
        self.observation = None
        self.unpersisted = None
        self.terminal = None
        self.task = None
        self.next_lookup = 0.0
        self.lookup_error = None
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
        self.attempt = attempt
        self.account_epoch = self.bot._ensure_freshness_ledger().epoch + 1
        self.fill_after_ms = _utc_ms()
        self.observation = None
        self.unpersisted = None
        self.terminal = None
        self.next_lookup = 0.0
        self.lookup_error = None
        self.last_report = None

    @property
    def pending(self):
        return self.error or self.attempt is not None

    def record(self, body, context):
        if context is not None and body["reduceOnly"] is True:
            from live.hsl_live import active_protective_batch

            batch = active_protective_batch(context.bot)
            if batch is not None and batch.contains(context.order, "create"):
                return None
        if self.pending:
            raise WriteDeferred()
        attempt = BitunixCreateAttempt.from_wire(body, submitted_at_ms=_utc_ms())
        try:
            self.journal.record(attempt)
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            raise WriteDeferred() from exc
        self._bind(attempt)
        return attempt

    def resolve(self, attempt, reason, *, evidence=None):
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
                return
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return
        level = (
            logging.info
            if _utc_ms() - attempt.submitted_at_ms >= 15_000
            else logging.debug
        )
        self.attempt = self.terminal = None
        self.observation = None
        self.error = False
        self.last_report = None
        level("[bitunix] ordinary create recovered | outcome=%s", reason)

    def poll(self, request):
        """Consume committed account evidence and schedule missing detail off the plan path."""
        try:
            pending = self.journal.load_pending()
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return
        if not pending and self.attempt is not None:
            self.error = True
            self._report("missing_record")
            return
        self.error = False
        if pending is not None:
            attempt, observation = pending
            if attempt != self.attempt:
                self._bind(attempt)
            self.observation = observation
        if self.unpersisted is not None and not self._observe_lookup(self.unpersisted):
            return
        if self.task is not None and self.task.done():
            task, self.task = self.task, None
            attempt, evidence, error = task.result()
            if attempt == self.attempt:
                self.lookup_error = error
                if evidence is not None and not self._observe_lookup(evidence):
                    return
        if self.attempt is None:
            return
        if (
            self.observation is not None
            and self.observation.conflicting_order_id is not None
        ):
            self._report("conflicting_order_identity")
            return
        if self._resolve_from_account():
            return
        now = time.monotonic()
        if self.task is None and now >= self.next_lookup:
            # First give the normal post-write account refresh an opportunity to
            # adopt a working order. Startup attempts also require a new cohort.
            if self._account_current():
                self.next_lookup = now + LOOKUP_INTERVAL_SECONDS
                self.task = asyncio.create_task(self._lookup(request, self.attempt))
        if _utc_ms() - self.attempt.submitted_at_ms >= 15_000:
            reason = (
                "awaiting_terminal_confirmation"
                if self.terminal is not None
                else "awaiting_order_evidence"
            )
            self._report(self.lookup_error or reason)

    async def _lookup(self, request, attempt):
        try:
            return attempt, await lookup_bitunix_create_evidence(request, attempt), None
        except (ExchangeError, NetworkError) as exc:
            # Lookup failure degrades only create admission. The caller keeps
            # running the normal account, cancellation and protection paths.
            retry = getattr(exc, "retry_after_seconds", 0.0)
            if retry:
                self.next_lookup = max(self.next_lookup, time.monotonic() + retry)
            return attempt, None, type(exc).__name__

    def _persist_observation(self, evidence):
        try:
            observation = self.journal.observe(
                self.attempt.client_id, evidence.order_id, evidence.filled_qty
            )
        except BitunixCreateJournalError as exc:
            self.error = True
            self._report(type(exc).__name__)
            return False
        if observation is None:
            self.error = True
            self._report("missing_record")
            return False
        self.observation = observation
        return True

    def _observe_lookup(self, evidence):
        previous = self.observation
        if not self._persist_observation(evidence):
            self.unpersisted = evidence
            return False
        self.unpersisted = None
        terminal = (
            evidence.disposition == "terminal"
            and evidence != self.terminal
            and evidence.filled_qty >= self.observation.filled_qty
        )
        if terminal:
            self.terminal = evidence
        if terminal or previous != self.observation:
            # Evidence acquired by a separate lookup requires a later account
            # cohort. Repeating the same proof does not reset its confirmation.
            self.account_epoch = self.bot._ensure_freshness_ledger().epoch + 1
            self.fill_after_ms = _utc_ms()
        return True

    def _account_current(self):
        ledger = self.bot._ensure_freshness_ledger()
        pending = self.bot._authoritative_pending_confirmations
        return (
            ledger.epoch >= self.account_epoch
            and ACCOUNT <= ledger.surfaces_at_epoch()
            and all(ledger.epoch >= pending.get(surface, 0) for surface in ACCOUNT)
        )

    def _consistent(self, evidence):
        return (
            self.observation is not None
            and self.observation.conflicting_order_id is None
            and self.observation.order_id == evidence.order_id
            and self.observation.filled_qty <= evidence.filled_qty
        )

    def _resolve_from_account(self):
        if not self._account_current():
            return False
        venue_order_id = (
            self.observation.order_id if self.observation is not None else None
        )
        matches = [
            row
            for row in self.bot.fetched_open_orders
            if row.get("clientOrderId") == self.attempt.client_id
            or venue_order_id is not None
            and row.get("id") == venue_order_id
        ]
        observations = []
        for row in matches:
            evidence = parse_bitunix_create_evidence(
                row.get("info"),
                self.attempt,
                confirmed_order_id=venue_order_id,
            )
            if evidence is not None:
                if not self._persist_observation(evidence):
                    self.unpersisted = evidence
                    return True
                observations.append(evidence)
        if self.terminal is not None:
            if (
                not matches
                and self.task is None
                and self._consistent(self.terminal)
                and self._fills_confirmed(self.terminal, require_refresh=True)
            ):
                self.resolve(self.attempt, "terminal", evidence=self.terminal)
                return True
        elif len(matches) == 1 and observations and self.task is None:
            evidence = observations[0]
            if (
                evidence.disposition == "working"
                and self._consistent(evidence)
                and self._fills_confirmed(evidence, require_refresh=False)
            ):
                self.resolve(self.attempt, "working", evidence=evidence)
                return True
        return False

    def _fills_confirmed(self, evidence, *, require_refresh):
        bot = self.bot
        manager = getattr(bot, "_pnls_manager", None)
        if not require_refresh and evidence.filled_qty == 0:
            return (
                manager is None
                or _observed_filled_qty(manager, self.attempt, evidence.order_id) == 0
            )
        if self.fill_receipt is None or manager is None:
            return False
        observed_manager, interval = self.fill_receipt
        ledger = bot._ensure_freshness_ledger()
        if (
            observed_manager is not manager
            or interval[0] <= self.fill_after_ms
            or ledger.surface_epoch("fills")
            < bot._authoritative_pending_confirmations.get("fills", 0)
            or ledger.surface_updated_ms("fills") < interval[1]
        ):
            return False
        return (
            _observed_filled_qty(manager, self.attempt, evidence.order_id)
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
