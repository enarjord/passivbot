import signal

import pytest

from optimization.interrupts import (
    OptimizerBackendInterrupted,
    OptimizerInterruptLatch,
)


def test_sigint_latch_survives_native_dispatch_boundary():
    latch = OptimizerInterruptLatch()

    latch._handle_sigint(signal.SIGINT, None)
    latch._handle_sigint(signal.SIGINT, None)

    assert latch.requested is True
    with pytest.raises(KeyboardInterrupt):
        latch.raise_if_requested()


def test_sigint_latch_restores_previous_handler(monkeypatch):
    previous = object()
    installed = []
    monkeypatch.setattr(signal, "getsignal", lambda _signum: previous)
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.append((signum, handler)),
    )
    latch = OptimizerInterruptLatch()

    with latch:
        assert installed == [(signal.SIGINT, latch._handle_sigint)]

    assert installed[-1] == (signal.SIGINT, previous)


def test_backend_interrupt_carries_terminated_pool_to_common_cleanup():
    pool = object()

    exc = OptimizerBackendInterrupted(pool=pool, pool_terminated=True)

    assert isinstance(exc, KeyboardInterrupt)
    assert exc.pool is pool
    assert exc.pool_terminated is True
