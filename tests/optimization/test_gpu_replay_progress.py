import logging

import pytest

from optimization.gpu.replay_progress import TemporalReplayProgress, suite_replay_context


def test_replays_have_distinct_ids_and_explicit_completion(caplog):
    caplog.set_level(logging.INFO)
    first = TemporalReplayProgress(64, 100)
    first.log("progress", 90, 30.5)
    first.log("complete", 100, 34.0)
    second = TemporalReplayProgress(64, 100)
    assert first.replay_id != second.replay_id
    messages = [record.getMessage() for record in caplog.records]
    assert "replay start" in messages[0]
    assert f"replay={first.replay_id} candidates=64 bars=90/100 elapsed=30.5s" in messages[1]
    assert "replay complete" in messages[2] and "bars=100/100" in messages[2]
    assert f"replay={second.replay_id}" in messages[3] and "bars=0/100" in messages[3]


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
