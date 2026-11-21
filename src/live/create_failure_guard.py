"""Detect a sustained loss of exchange order creation from transport outcomes."""

import logging
import time

from ccxt.base.errors import InsufficientFunds, InvalidOrder

from live import reconciler
from live.diagnostic_safety import (
    bounded_exception_type,
    bounded_exchange_error_context,
    bounded_exchange_error_context_from_mapping,
    hyperliquid_wallet_rejected,
)
from live.event_bus import EventTags, EventTypes, ReasonCodes
from passivbot_exceptions import FatalOrderWriteError

FAILURE_WINDOW_SECONDS = 60 * 60
MIN_FAILED_CREATES = 5
CONFIRMATION_GRACE_SECONDS = 60
MAX_UNCONFIRMED_IDS = 256


class CreateFailureGuard:
    """Keep failure evidence across internal restarts of one live invocation."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.in_flight = set()
        self._reset_streak()

    def _reset_streak(self):
        self.first_at = None
        self.last_ambiguous_at = None
        self.stop_at = None
        self.failures = 0
        self.unconfirmed = 0
        self.unconfirmed_ids = {}
        self.latest_reason = {}

    def submitted(self, order):
        self.in_flight.add(id(order))

    def completed(self, bot, order, *, result=None, error=None):
        key = id(order)
        if key not in self.in_flight:
            return
        self.in_flight.remove(key)
        status = str(result.get("status") or "").strip().lower() if isinstance(result, dict) else ""
        if (
            error is None
            and isinstance(result, dict)
            and str(result.get("id") or "").strip()
            and status not in ("failed", "expired", "canceled", "cancelled", "rejected")
            and bot.did_create_order(result)
        ):
            self._accepted(bot)
            return
        now = self.clock()
        if self.first_at is None:
            self.first_at = now
        self.failures += 1
        if error is not None:
            self.latest_reason = bounded_exchange_error_context(error)
            self.latest_reason["error_type"] = bounded_exception_type(error)
        elif isinstance(result, dict):
            self.latest_reason = bounded_exchange_error_context_from_mapping(result)
        else:
            self.latest_reason = {}

        if not (
            isinstance(error, (InvalidOrder, InsufficientFunds))
            or status in ("rejected", "failed")
        ):
            self.unconfirmed += 1
            self.last_ambiguous_at = now
            client_id = reconciler.canonical_passivbot_custom_id(
                reconciler.extract_order_custom_id(order)
            )
            if client_id:
                self.unconfirmed_ids[(order["symbol"], client_id)] = None
                if len(self.unconfirmed_ids) > MAX_UNCONFIRMED_IDS:
                    del self.unconfirmed_ids[next(iter(self.unconfirmed_ids))]
        if self.failures == MIN_FAILED_CREATES:
            self._emit(
                bot, "degraded", "warning", ReasonCodes.CREATE_FAILURE_STREAK, "create"
            )
        if (
            self.stop_at is None
            and self.failures >= MIN_FAILED_CREATES
            and now - self.first_at >= FAILURE_WINDOW_SECONDS
        ):
            self.stop_at = now
            if self.last_ambiguous_at is not None:
                self.stop_at = max(
                    now, self.last_ambiguous_at + CONFIRMATION_GRACE_SECONDS
                )

    def cancelled(self, order):
        self.in_flight.discard(id(order))

    def confirm(self, bot, identities):
        """An authoritative open order or fill proves its exact client ID was accepted."""
        if not self.unconfirmed_ids:
            return
        for symbol, client_id in identities:
            canonical_id = reconciler.canonical_passivbot_custom_id(client_id)
            if (symbol, canonical_id) in self.unconfirmed_ids:
                self._accepted(bot)
                return

    def _accepted(self, bot):
        if self.failures >= MIN_FAILED_CREATES:
            self._emit(
                bot, "succeeded", "info", ReasonCodes.CREATE_FAILURE_STREAK, "create",
                details={},
            )
        self._reset_streak()

    def check(self, bot):
        # Wait for concurrently submitted siblings to finish. An acknowledged
        # sibling clears the streak even if other orders in the batch failed.
        if (
            not self.in_flight
            and self.stop_at is not None
            and self.clock() >= self.stop_at
        ):
            self._emit(
                bot, "failed", "error", ReasonCodes.CREATE_FAILURE_STREAK, "create"
            )
            raise FatalOrderWriteError("Sustained exchange order creation failure")

    def check_write_error(self, bot, action, error):
        if bot.exchange != "hyperliquid" or not hyperliquid_wallet_rejected(error):
            return
        details = bounded_exchange_error_context(error)
        details["error_type"] = bounded_exception_type(error)
        self._emit(
            bot, "failed", "error", ReasonCodes.ORDER_WRITE_SIGNER_REJECTED, action,
            details=details,
        )
        raise FatalOrderWriteError("Hyperliquid does not recognize the API wallet") from None

    def _emit(self, bot, status, level, reason, action, *, details=None):
        elapsed = 0 if self.first_at is None else int(self.clock() - self.first_at)
        data = {
            "action": action,
            "next_action": "stop" if status == "failed" else "continue",
            "failed_creates": self.failures,
            "unconfirmed_creates": self.unconfirmed,
            "elapsed_seconds": elapsed,
            **(self.latest_reason if details is None else details),
        }
        try:
            emitted = bot._emit_live_event(
                EventTypes.EXECUTION_CREATE_HEALTH,
                level=level,
                component="execution.order_write",
                tags=(EventTags.EXECUTION, EventTags.ORDER, action),
                status=status,
                reason_code=reason,
                data=data,
                require_enqueue=status == "failed",
            )
            if emitted is not None:
                return
        except Exception:
            pass
        # Sink failures must not change the decision or the process exit.
        try:
            logging.log(
                getattr(logging, level.upper()),
                "[order] write health %s action=%s reason=%s failed_creates=%d "
                "unconfirmed_creates=%d elapsed_seconds=%d",
                status, action, reason, self.failures, self.unconfirmed, elapsed,
            )
        except Exception:
            pass
