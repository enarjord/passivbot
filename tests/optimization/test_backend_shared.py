"""Tests for shared optimizer backend coordination helpers."""

import logging

from optimization import backend_shared


class _DelayedResult:
    """Minimal asynchronous result which becomes ready on its second poll."""

    def __init__(self, payload):
        self.payload = payload
        self.polls = 0

    def ready(self):
        """Return false once so the progress heartbeat can run while pending."""
        self.polls += 1
        return self.polls > 1

    def get(self):
        """Return the configured worker payload."""
        return self.payload


def test_drain_async_results_logs_progress_while_population_is_pending(monkeypatch, caplog):
    """A long-running population emits a useful heartbeat before any result completes."""
    times = iter((0.0, 301.0, 302.0))
    monkeypatch.setattr(backend_shared.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(backend_shared.time, "sleep", lambda _seconds: None)
    result = _DelayedResult({"fitness": 1.0})

    with caplog.at_level(logging.INFO):
        completed = backend_shared.drain_async_results(
            {result: "candidate"},
            on_result=lambda _context, _payload: None,
            progress_label="Optimizer population",
        )

    assert completed == 1
    assert (
        "Optimizer population progress | completed=0/1 pending=1 submitted=1 "
        "elapsed=301.0s rate=0.000/s eta=unknown"
    ) in caplog.messages


def test_stream_async_results_reports_total_and_completed_rate(monkeypatch, caplog):
    """Streaming evaluation heartbeats include bounded totals and an ETA."""
    times = iter((0.0, 100.0, 301.0, 302.0))
    monkeypatch.setattr(backend_shared.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(backend_shared.time, "sleep", lambda _seconds: None)
    results = [_DelayedResult(1), _DelayedResult(2)]
    results[0].polls = 1

    with caplog.at_level(logging.INFO):
        completed = backend_shared.stream_async_results(
            range(2),
            submit=lambda index: (results[index], index),
            on_result=lambda _context, _payload: None,
            max_pending=2,
            progress_label="Optimizer starting configs",
            progress_total=2,
        )

    assert completed == 2
    assert any(
        message.startswith(
            "Optimizer starting configs progress | completed=1/2 pending=1 submitted=2 "
        )
        and "eta=" in message
        for message in caplog.messages
    )
