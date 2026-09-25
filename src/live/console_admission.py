"""Bounded, presentation-only admission for explicitly opted-in state summaries."""
from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Callable, Hashable


@dataclass
class _State:
    fingerprint: Hashable
    last_written: float
    repeats: int = 0


class ConsoleAdmission:
    """Families own semantic identity; this class only owns repeat accounting.

    Keep one state per scope, not per observed fingerprint. Eviction/restart makes
    the next observation visible. No events, payloads or rendered text are retained.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, capacity: int = 256):
        self._clock = clock
        self._capacity = max(1, capacity)
        self._states: OrderedDict[Hashable, _State] = OrderedDict()
        self._lock = threading.RLock()

    def write(self, key: Hashable, fingerprint: Hashable, message: str,
              emit: Callable[[str], None], *, reminder_seconds: float | None) -> str | None:
        with self._lock:
            now = self._clock()
            previous = self._states.get(key)
            if previous is not None and previous.fingerprint == fingerprint:
                repeats = min(previous.repeats + 1, 999_999_999)
                if reminder_seconds is None or now - previous.last_written < reminder_seconds:
                    previous.repeats = repeats
                    self._states.move_to_end(key)
                    return None
                # Reserve room for the usual timestamp/level/exchange prefix.
                message = ((message if len(message) <= 130 else message[:127] + "...") + f" repeats={repeats} over="
                           f"{min(int(max(0, now - previous.last_written)), 999_999_999)}s")
            # Failed delivery must not consume the first occurrence or reminder.
            emit(message)
            self._states[key] = _State(fingerprint, now)
            self._states.move_to_end(key)
            while len(self._states) > self._capacity:
                self._states.popitem(last=False)
            return message
