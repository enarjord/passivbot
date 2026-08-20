from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Callable


class OptimizerInterruptLatch:
    """Latch SIGINT until optimizer code reaches a safe cancellation point.

    Native runtimes such as PyTorch MPS may return normally after a SIGINT
    delivered during a native dispatch. Remembering the signal separately
    prevents the optimizer from silently continuing into another generation.
    """

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._previous_handler = None
        self._installed = False

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        self._requested.set()

    def raise_if_requested(self) -> None:
        if self._requested.is_set():
            raise KeyboardInterrupt

    def install(self) -> "OptimizerInterruptLatch":
        if threading.current_thread() is not threading.main_thread():
            return self
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self._installed = True
        return self

    def restore(self) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous_handler)
            self._installed = False

    def __enter__(self) -> "OptimizerInterruptLatch":
        return self.install()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.restore()


def no_interrupt_requested() -> None:
    """Default cancellation checkpoint for callers without a SIGINT latch."""


InterruptCheck = Callable[[], None]


class OptimizerBackendInterrupted(KeyboardInterrupt):
    """Carry backend-owned resources into the CLI's common shutdown path."""

    def __init__(self, *, pool, pool_terminated: bool) -> None:
        super().__init__()
        self.pool = pool
        self.pool_terminated = bool(pool_terminated)
