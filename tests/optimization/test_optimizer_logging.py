import logging

from optimization.progress import SeedBootstrapProgress
from pareto_store import ParetoStore


def entry(index):
    return {
        "bot": {"long": {"param": index}, "short": {}},
        "metrics": {"objectives": {"metric1": index, "metric2": 101 - index},
                    "constraint_violation": 0.0},
    }


def test_pareto_summary_throttles_without_throttling_persistence(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [0.0]
    monkeypatch.setattr("pareto_store.time.monotonic", lambda: now[0])
    store = ParetoStore(str(tmp_path))
    store.scoring_keys = ["metric1", "metric2"]
    for index in range(100):
        assert store.add_entry(entry(index))
    assert len(caplog.records) == 1
    assert "eval=1 front=1 feasible=1 changes=+1/-0" in caplog.text
    assert len(list((tmp_path / "pareto").glob("*.json"))) == 100
    assert len(store.get_front()) == 100
    now[0] = 59.999
    store._emit_front_summary()
    assert len(caplog.records) == 1
    now[0] = 60.0
    store._emit_front_summary()
    assert len(caplog.records) == 2
    assert "eval=100 front=100 feasible=100 changes=+99/-0" in caplog.text
    assert all(len(record.getMessage()) <= 240 for record in caplog.records)
    store.flush_now()
    assert len(caplog.records) == 2


def test_pareto_flush_reports_tail_and_debug_keeps_detail(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("pareto_store.time.monotonic", lambda: 0.0)
    store = ParetoStore(str(tmp_path))
    store.scoring_keys = ["metric1", "metric2"]
    store.add_entry(entry(0))
    store.add_entry(entry(1))
    duplicate = entry(1)
    duplicate["bot"]["long"]["param"] = -1
    assert not store.add_entry(duplicate)
    store.flush_now()
    store.flush_now()
    info = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    debug = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(info) == 2
    assert "eval=3 front=2" in info[-1]
    assert any("metric1:" in message for message in debug)
    assert any("Dropping candidate" in message for message in debug)


def test_seed_progress_reports_waiting_and_resume_without_fabricated_eta(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [10.0]
    monkeypatch.setattr("optimization.progress.time.monotonic", lambda: now[0])
    progress = SeedBootstrapProgress(10, completed=4, workers=2)
    progress.update(4, inflight=2, queued=4)
    assert len(caplog.records) == 1
    now[0] = 69.999
    progress.update(4, inflight=2, queued=4)
    assert len(caplog.records) == 1
    now[0] = 70.0
    progress.update(4, inflight=2, queued=4)
    assert "completed=4/10 inflight=2 queued=4 elapsed=60s eta_seed=unknown" in caplog.text
    now[0] = 130.0
    progress.update(6, inflight=2, queued=2)
    assert "eta_seed=240s" in caplog.records[-1].getMessage()
    now[0] = 135.0
    progress.update(10, inflight=0, queued=0, force=True)
    assert "exact complete" in caplog.records[-1].getMessage()
    assert "eta_seed=0s" in caplog.records[-1].getMessage()
    assert all(len(record.getMessage()) < 240 for record in caplog.records)


def test_seed_clamps_are_one_warning_with_bounded_samples(caplog):
    from optimize import _flush_seed_bounds_adjustments, _record_seed_bounds_adjustment
    from optimization.bounds import Bound

    caplog.set_level(logging.DEBUG)
    caplog.clear()
    collector = {}
    for index in range(25):
        _record_seed_bounds_adjustment(
            source="seed.json", bound_key=f"long_param_{index}",
            path=("bot", "long", f"param_{index}"), original=2.0,
            adjusted=1.0, bound=Bound(0.0, 1.0), context="starting config",
            collector=collector,
        )
    _flush_seed_bounds_adjustments(collector)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    details = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(warnings) == 1 and len(details) == 25
    assert "adjustments=25 keys=25" in warnings[0]
    assert "+22 more" in warnings[0] and len(warnings[0]) <= 240
    assert all("bounds=[0.0, 1.0] | clamped=1.0 | source=seed.json" in m for m in details)



def test_resumed_pareto_summaries_exclude_reconstructed_history(tmp_path, caplog):
    from optimize import ResultRecorder

    caplog.set_level(logging.INFO)
    caplog.clear()
    options = dict(
        results_dir=str(tmp_path), sig_digits=6, flush_interval=60,
        scoring_keys=["metric1", "metric2"], compress=False, write_all_results=False,
    )
    original = ResultRecorder(**options)
    for index in range(5):
        original.record(entry(index))
    original.flush()
    original.close()
    caplog.clear()

    resumed = ResultRecorder(**options, starting_iters=123)
    assert resumed.store.n_iters == 123
    assert len(resumed.store.get_front()) == 5
    resumed.flush()
    assert not caplog.records

    # A duplicate increments exact evaluation count but does not change the front.
    resumed.record(entry(0))
    assert not caplog.records
    resumed.record(entry(5))
    assert len(caplog.records) == 1
    assert "eval=125 front=6 feasible=6 changes=+1/-0" in caplog.text
    resumed.record(entry(6))
    resumed.flush()
    resumed.close()
    assert len(caplog.records) == 2
    assert "eval=126 front=7 feasible=7 changes=+1/-0" in caplog.records[-1].getMessage()
    assert len(list((tmp_path / "pareto").glob("*.json"))) == 7
