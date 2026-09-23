"""Bounded, scope-local position/fill settling shared by fetch and write admission.

This is observation scheduling, never a certificate of fill completeness or risk
permission. Expiry releases only this gate. Current account/quote and Rust intent
checks still apply. A failed/missing fill can never renew the deadline.
"""

from dataclasses import dataclass
import logging
import time

SETTLE_SECONDS = 5.0
MAX_WAIT_SECONDS = 15.0


class Settling(Exception):
    """A position changed while an asynchronous history refresh was preparing."""


@dataclass
class Pending:
    first: float
    changed: float
    revision: int
    expired_reported: bool = False


class PositionFillSync:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.positions = None
        self.pending = {}
        self.revision = 0

    def observe(self, positions):
        now = self.clock()
        positions = dict(positions)
        if self.positions is not None:
            for key in self.positions.keys() | positions.keys():
                if self.positions.get(key, (0.0, 0.0)) == positions.get(
                    key, (0.0, 0.0)
                ):
                    continue
                self.revision += 1
                old = self.pending.get(key)
                self.pending[key] = Pending(
                    old.first if old else now,
                    now,
                    self.revision,
                    old.expired_reported if old else False,
                )
        self.positions = positions

    def expired(self, key):
        pending = self.pending.get(key)
        if pending is None or self.clock() - pending.first < MAX_WAIT_SECONDS:
            return False
        if not pending.expired_reported:
            logging.warning(
                "[state] position/fill synchronization expired | symbol=%s pside=%s "
                "wait_seconds=%s action=use_current_inputs_with_best_effort_history",
                *key,
                MAX_WAIT_SECONDS
            )
            pending.expired_reported = True
        return True

    def blocked(self, key):
        return key in self.pending and not self.expired(key)

    def fetch_ready(self):
        # A shared account endpoint may also fetch unrelated scopes. Only receipts
        # whose capture starts after a scope's settling time can release it.
        return not self.pending or any(
            self.expired(key) or self.clock() - p.changed >= SETTLE_SECONDS
            for key, p in self.pending.items()
        )

    def begin_fetch(self):
        now = self.clock()
        return {
            key: p.revision
            for key, p in self.pending.items()
            if now - p.changed >= SETTLE_SECONDS
        }

    def finish_fetch(self, receipt):
        for key, revision in receipt.items():
            if key in self.pending and self.pending[key].revision == revision:
                del self.pending[key]


def state(bot):
    value = getattr(bot, "_position_fill_sync", None)
    if value is None:
        value = bot._position_fill_sync = PositionFillSync()
    return value


def observe(bot, positions):
    state(bot).observe(positions)


def fetch_ready(bot):
    return state(bot).fetch_ready()


def permits(bot, order):
    gate = getattr(bot, "_position_fill_sync", None)
    if gate is None:
        return True
    key = (order.get("symbol"), order.get("position_side", order.get("pside")))
    # All actions are scoped, including cancels and protective market orders.
    # Aggregate HSL decisions consume every contributing position on that scope.
    keys = {key}
    config = getattr(bot, "config", {})
    if config.get("live", {}).get("hsl_engine") == "revised":
        mode = config.get("live", {}).get("hsl_signal_mode", "coin")
        if mode == "unified":
            keys.update(gate.pending)
        elif mode == "pside":
            keys.update(k for k in gate.pending if k[1] == key[1])
    return not any(gate.blocked(k) for k in keys)
