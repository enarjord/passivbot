"""Host-only accounting boundaries; no replay, runtime or search imports."""

import logging
from types import SimpleNamespace

import pytest

from optimization import progress


def test_cohort_counts_attempts_separately_from_rows_and_overlapping_owner_time(monkeypatch, caplog):
    ticks = [0.0, 0.0]
    monkeypatch.setattr(progress.time, "perf_counter", lambda: ticks[0])
    monkeypatch.setattr(progress.time, "thread_time", lambda: ticks[1])
    caplog.set_level(logging.DEBUG)
    counts = dict(completed=100, screened=200, prefilter_rejected=15)
    transport = dict(admitted_attempts=0, cache_attempts=0, alias_attempts=0,
                     submitted_rows=0, replay_rows=0, reused_rows=0, delivered_attempts=0)
    owner = dict(batches=0, rows=0, execution_seconds=0.0,
                 accumulation_seconds=0.0, preparation_seconds=0.0)
    session = SimpleNamespace(diagnostics_snapshot=lambda: dict(transport),
                              service=SimpleNamespace(diagnostics_snapshot=lambda: dict(owner)))
    cohort = progress.CohortDiagnostics(generation=7, phase="generation", counts=lambda: counts,
                                        session=session)
    for name, wall, cpu in (("prepare", 3, 2), ("poll", 7, 1), ("consume", 2, 1)):
        with cohort.measure(name):
            ticks[0] += wall
            ticks[1] += cpu
    ticks[0] += 1
    ticks[1] += .5
    counts.update(completed=103, screened=201, prefilter_rejected=16)
    transport.update(admitted_attempts=3, cache_attempts=1, alias_attempts=1,
                     submitted_rows=6, replay_rows=6, reused_rows=2, delivered_attempts=3)
    owner.update(batches=2, rows=6, execution_seconds=8.0, accumulation_seconds=2.0,
                 preparation_seconds=.5)
    cohort.finish("complete")
    info = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert len(info) == 1
    assert "gen=7 phase=generation status=complete" in info[0]
    assert "attempts=4 prefilter=1 gpu_rows=6 wall_s=13.000 attempts/s=0.3077 gpu_rows/s=0.4615" in info[0]
    assert "full_attempts=3 screened_attempts=1 main_thread_cpu_s=4.500000" in caplog.text
    assert "prepare_wall_s=3.000000 prepare_cpu_s=2.000000" in caplog.text
    assert "poll_wall_s=7.000000 poll_cpu_s=1.000000" in caplog.text
    assert "consume_wall_s=2.000000" in caplog.text and "consume_cpu_s=1.000000" in caplog.text
    assert "other_wall_s=1.000000 other_cpu_s=0.500000" in caplog.text
    assert "overlapping_owner_execution_seconds=8.000000" in caplog.text
    assert "overlapping_owner_accumulation_seconds=2.000000" in caplog.text
    assert "cache_attempts=1 alias_attempts=1" in caplog.text
    assert all(len(record.getMessage()) <= 240 for record in caplog.records)


def test_disabled_diagnostics_do_not_read_clocks_counters_or_service(monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "isEnabledFor", lambda _level: False)
    def forbidden():
        pytest.fail("disabled diagnostics must not take clocks or snapshots")
    monkeypatch.setattr(progress.time, "perf_counter", forbidden)
    monkeypatch.setattr(progress.time, "thread_time", forbidden)
    cohort = progress.CohortDiagnostics(generation=1, phase="generation", counts=forbidden,
                                        session=None)
    with cohort.measure("prepare"):
        pass
    cohort.finish("complete")
    with progress.search_boundary(1, "ask"):
        pass


def test_optional_snapshots_and_logging_preserve_original_exception(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    counts = dict(completed=0, screened=0, prefilter_rejected=0)
    def unavailable():
        raise OSError("private observer detail")
    session = SimpleNamespace(diagnostics_snapshot=unavailable,
                              service=SimpleNamespace(diagnostics_snapshot=unavailable))
    cohort = progress.CohortDiagnostics(generation=1, phase="generation", counts=lambda: counts,
                                        session=session)
    failure = ValueError("original planner error")
    with pytest.raises(ValueError) as raised:
        try:
            with cohort.measure("prepare"):
                raise failure
        finally:
            cohort.finish("failed")
    assert raised.value is failure
    assert "status=failed" in caplog.text and "gpu_rows=unknown" in caplog.text
    assert "private observer detail" not in caplog.text
    monkeypatch.setattr(logging.getLogger(), "log", lambda *_args, **_kwargs: unavailable())
    with pytest.raises(ValueError) as raised:
        with progress.search_boundary(1, "tell"):
            raise failure
    assert raised.value is failure


def test_timing_failure_does_not_replace_interruption(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    counts = dict(completed=0, screened=0, prefilter_rejected=0)
    session = SimpleNamespace(service=SimpleNamespace())
    cohort = progress.CohortDiagnostics(generation=1, phase="screening", counts=lambda: counts,
                                        session=session)
    def unavailable():
        raise OSError("clock failed")
    interruption = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised:
        with cohort.measure("poll"):
            monkeypatch.setattr(progress.time, "thread_time", unavailable)
            raise interruption
    cohort.finish("interrupted")
    assert raised.value is interruption


def test_serial_search_barriers_have_separate_wall_and_thread_cpu_spans(monkeypatch, caplog):
    ticks = [0.0, 0.0]
    monkeypatch.setattr(progress.time, "perf_counter", lambda: ticks[0])
    monkeypatch.setattr(progress.time, "thread_time", lambda: ticks[1])
    caplog.set_level(logging.DEBUG)
    with progress.search_boundary(4, "tell"):
        ticks[:] = [5.0, 2.0]
    assert "gen=4 operation=tell | wall_s=5.000000 main_thread_cpu_s=2.000000" in caplog.text
    assert all(record.levelno == logging.DEBUG for record in caplog.records)
