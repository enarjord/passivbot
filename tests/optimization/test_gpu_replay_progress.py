import logging

import pytest

from optimization.gpu.replay_progress import TemporalReplayProgress, suite_replay_context
from optimization.progress import gpu_work_context, work_scope, publish_progress


def test_replays_have_distinct_ids_and_explicit_completion(caplog):
    caplog.set_level(logging.INFO)
    first = TemporalReplayProgress(64, 100)
    first.log("progress", 90, 60.0)
    first.log("complete", 100, 65.0)
    second = TemporalReplayProgress(64, 100)
    assert first.replay_id != second.replay_id
    messages = [record.getMessage() for record in caplog.records]
    assert "replay start" in messages[0]
    assert f"replay={first.replay_id}" in messages[1]
    assert "progress=90.0% bars=90/100 elapsed=1m00s" in messages[1]
    assert "replay complete" in messages[2] and "bars=100/100" in messages[2]
    assert f"replay={second.replay_id}" in messages[3] and "history_bars=100" in messages[3]


def test_suite_context_restores_after_nested_failure(caplog):
    caplog.set_level(logging.INFO)
    with suite_replay_context(pass_index=1, pass_count=2, labels=["small"],
                             exchanges=["combined"], evaluation_stage="screening"):
        with pytest.raises(RuntimeError):
            with suite_replay_context(pass_index=2, pass_count=2, labels=["large"],
                                     exchanges=["combined"], evaluation_stage="full"):
                TemporalReplayProgress(2, 10)
                raise RuntimeError("interrupted")
        TemporalReplayProgress(2, 10)
    TemporalReplayProgress(2, 10)
    messages = [record.getMessage() for record in caplog.records]
    assert "scenarios=large" in messages[0] and "stage=full" in messages[0]
    assert "scenarios=small" in messages[1] and "stage=screening" in messages[1]
    assert "suite_pass" not in messages[2]
    assert not any("complete" in m for m in messages)


def test_suite_context_bounds_and_sanitizes_labels(caplog):
    caplog.set_level(logging.INFO)
    with suite_replay_context(pass_index=1, pass_count=1,
                             labels=["very\nlong\x1b" + "x" * 100] * 9,
                             exchanges=["combined"] * 9, evaluation_stage="screening"):
        TemporalReplayProgress(1024, 2500000)
    message = caplog.records[0].getMessage()
    assert "\n" not in message and "\x1b" not in message
    assert ",+6 exchange=combined" in message
    assert len(message) <= 240


def test_replay_progress_cadence_eta_and_debug_details(caplog):
    caplog.set_level(logging.DEBUG)
    replay = TemporalReplayProgress(512, 1000)
    replay.log("progress", 300, 30.0)
    replay.log("progress", 599, 59.999)
    replay.log("progress", 600, 60.0)
    replay.log("progress", 900, 90.0)
    replay.log("complete", 1000, 100.0)
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 3
    assert "progress=60.0%" in info[1]
    assert "rate=10 bars/s eta_batch=40s" in info[1]
    assert "complete" in info[2] and "progress=100.0%" in info[2]
    assert sum(r.levelno == logging.DEBUG for r in caplog.records) == 3
    assert all(len(message) <= 240 for message in info)


def test_empty_replay_does_not_divide_by_zero(caplog):
    caplog.set_level(logging.INFO)
    replay = TemporalReplayProgress(1, 0)
    replay.log("complete", 0, 0.0)
    assert "progress=100.0%" in caplog.records[-1].getMessage()
    assert "eta_batch=unknown" in caplog.records[-1].getMessage()


def test_generation_phase_and_kernel_counts_survive_nested_interruption(caplog):
    caplog.set_level(logging.INFO)
    ticks = []
    with gpu_work_context(7, "gpu_proxy", lambda: ticks.append(1)):
        with suite_replay_context(pass_index=2, pass_count=3, labels=["full"],
                                 exchanges=["combined"], evaluation_stage="full"):
            replay = TemporalReplayProgress(128, 16384, history_chunk_bars=8192)
            replay.log("progress", 8192, 60, kernel_dispatches=1)
            replay.log("complete", 16384, 120, kernel_dispatches=2)
            with pytest.raises(KeyboardInterrupt):
                with gpu_work_context(0, "seed_proxy"):
                    assert work_scope() == "gen=0 phase=seed_proxy"
                    raise KeyboardInterrupt
            assert work_scope() == "gen=7 phase=gpu_proxy"
    assert work_scope() == "phase=gpu_proxy"
    assert len(ticks) == 3
    assert "history_chunk_bars=8192" in caplog.text
    assert "kernel_dispatches=1" in caplog.text and "kernel_dispatches=2" in caplog.text
    assert all("gen=7 phase=gpu_proxy group=2/3" in r.getMessage() for r in caplog.records)
    assert all(len(r.getMessage()) <= 240 for r in caplog.records)


def test_optional_progress_callback_failure_does_not_abort_replay(caplog):
    def failed_callback():
        raise OSError("private detail excluded")
    with gpu_work_context(1, "gpu_proxy", failed_callback):
        publish_progress()
        replay = TemporalReplayProgress(1, 10)
        replay.log("complete", 10, 60)
    assert "private detail" not in caplog.text


def test_generation_eta_is_scoped_bounded_and_unknown_on_overrun():
    from optimization.progress import GenerationMilestone

    tick = [0.0]
    estimate = GenerationMilestone(clock=lambda: tick[0])
    assert estimate.eta() == "unknown"
    estimate.begin()
    tick[0] += 100
    assert estimate.eta() == "unknown"  # no invented first-generation estimate
    estimate.finish()
    estimate.begin()
    tick[0] += 30
    assert estimate.eta() == "1m10s"
    tick[0] += 100
    assert estimate.eta() == "unknown"
    estimate.finish()
    assert estimate.eta() == "unknown"  # CPU admission wait isn't a GPU generation
    for _ in range(8):
        estimate.begin()
        tick[0] += 60
        estimate.finish()
    assert len(estimate.completed) == 5
    estimate.begin()
    tick[0] += 10
    assert estimate.eta() == "50s"
