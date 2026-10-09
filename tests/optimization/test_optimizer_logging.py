import logging

from optimization.progress import OptimizerProgress, duration, log_tokens
from pareto_store import ParetoStore


def entry(index):
    return {
        "bot": {"long": {"param": index}, "short": {}},
        "metrics": {"objectives": {"metric1": index, "metric2": 101 - index},
                    "constraint_violation": 0.0},
    }


def test_pareto_reports_every_accepted_member_without_throttling_persistence(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [0.0]
    monkeypatch.setattr("pareto_store.time.monotonic", lambda: now[0])
    store = ParetoStore(str(tmp_path))
    store.scoring_keys = ["metric1", "metric2"]
    for index in range(100):
        assert store.add_entry(entry(index))
    assert len(caplog.records) == 200
    assert "eval=1 front=1 feasible=1 changes=+1/-0" in caplog.text
    assert len(list((tmp_path / "pareto").glob("*.json"))) == 100
    assert len(store.get_front()) == 100
    assert "eval=100 front=100 feasible=100 changes=+1/-0" in caplog.text
    assert "metric1=[0,99] metric2=[2*,101]" in caplog.records[-1].getMessage()
    assert all(len(record.getMessage()) <= 240 for record in caplog.records)
    store.flush_now()
    assert len(caplog.records) == 200


def test_pareto_flush_is_quiet_and_debug_keeps_detail(tmp_path, monkeypatch, caplog):
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
    assert len(info) == 4
    assert "eval=2 front=2" in info[-2]
    assert any("metric1:" in message for message in debug)
    assert any("Dropping candidate" in message for message in debug)




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
    assert len(caplog.records) == 2
    assert "eval=125 front=6 feasible=6 changes=+1/-0" in caplog.text
    resumed.record(entry(6))
    resumed.flush()
    resumed.close()
    assert len(caplog.records) == 4
    assert "eval=126 front=7 feasible=7 changes=+1/-0" in caplog.records[-2].getMessage()
    assert len(list((tmp_path / "pareto").glob("*.json"))) == 7


def test_pareto_all_objective_ranges_goals_tradeoffs_and_resume(tmp_path, caplog):
    from config.scoring import ObjectiveSpec

    caplog.set_level(logging.INFO)
    specs = [ObjectiveSpec("adg_strategy_eq", "max"), ObjectiveSpec("adg_strategy_eq_w", "max")]
    specs += [ObjectiveSpec(name, "min") for name in (
        "strategy_eq_underwater_pct_mean", "strategy_eq_recovery_days_max",
        "drawdown_worst_strategy_eq", "drawdown_worst_mean_1pct_strategy_eq",
        "position_held_time_weighted_mean_hours",
    )]
    def candidate(index, values, violation=0):
        result = entry(index)
        result["optimize"] = {"scoring": [spec.to_config() for spec in specs]}
        result["metrics"] = {"objectives": dict(zip((s.metric for s in specs), values)),
                             "constraint_violation": violation}
        return result
    store = ParetoStore(str(tmp_path))
    assert store.add_entry(candidate(0, [1, 4, 1, 1, 1, 1, 1]))
    caplog.clear()
    assert store.add_entry(candidate(1, [3, 2, 2, 2, 2, 2, 2]))
    assert "adg_strategy_eq=[1,3*] adg_strategy_eq_w=[2,4]" in caplog.text
    for spec in specs[2:]:
        assert f"{spec.metric}=[1,2]" in caplog.text
    assert "+5 metrics" not in caplog.text
    assert all(len(r.getMessage()) <= 240 for r in caplog.records)
    caplog.clear()
    # A new tradeoff point improves none of the existing extremes.
    assert store.add_entry(candidate(2, [2, 3, 1.5, 1.5, 1.5, 1.5, 1.5]))
    assert "quality=unchanged_tradeoff" in caplog.text
    assert not any("*" in r.getMessage() for r in caplog.records)
    caplog.clear()
    resumed = ParetoStore(str(tmp_path))
    assert not caplog.records
    assert resumed.progress_snapshot()["pareto_added"] == 0
    assert resumed.progress_snapshot()["last_pareto_age"] == "unknown"
    assert resumed.add_entry(candidate(3, [4, 2, 2, 2, 2, 2, 2]))
    assert "adg_strategy_eq=[1,4*] adg_strategy_eq_w=[2,4]" in caplog.text
    assert "changes=+1/-1" in caplog.text


def test_pareto_labels_infeasible_then_feasible_front(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    store = ParetoStore(str(tmp_path))
    store.scoring_keys = ["metric1", "metric2"]
    first = entry(0)
    first["metrics"]["constraint_violation"] = 1
    assert store.add_entry(first)
    assert "best_scope=infeasible_front" in caplog.text
    caplog.clear()
    assert store.add_entry(entry(1))
    assert "best_scope=feasible_front" in caplog.text
    assert "metric1=[1*,1] metric2=[100*,100]" in caplog.text


def test_pareto_marks_best_endpoints_only_when_ranges_change(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    store = ParetoStore(str(tmp_path))

    def candidate(index, gain, drawdown, underwater):
        result = entry(index)
        result["optimize"] = {"scoring": [
            {"metric": "adg_strategy_eq", "goal": "max"},
            {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
            {"metric": "strategy_eq_underwater_pct_mean", "goal": "min"},
        ]}
        result["metrics"]["objectives"] = {
            "adg_strategy_eq": gain, "drawdown_worst_strategy_eq": drawdown,
            "strategy_eq_underwater_pct_mean": underwater,
        }
        return result

    assert store.add_entry(candidate(0, 1, 2, 1))
    assert "adg_strategy_eq=[1,1*]" in caplog.text
    assert "drawdown_worst_strategy_eq=[2*,2]" in caplog.text
    caplog.clear()
    assert store.add_entry(candidate(1, 2, 3, 3))
    assert "adg_strategy_eq=[1,2*]" in caplog.text
    assert "drawdown_worst_strategy_eq=[2,3]" in caplog.text
    caplog.clear()
    assert store.add_entry(candidate(2, .5, 1, 4))
    assert "adg_strategy_eq=[0.5,2]" in caplog.text
    assert "drawdown_worst_strategy_eq=[1*,3]" in caplog.text
    caplog.clear()
    # This point widens the worse drawdown endpoint but improves no best.
    assert store.add_entry(candidate(3, 1.5, 4, 2))
    assert "quality=unchanged_tradeoff" in caplog.text
    assert "adg_strategy_eq=[0.5,2]" in caplog.text
    assert "drawdown_worst_strategy_eq=[1,4]" in caplog.text
    assert "strategy_eq_underwater_pct_mean=[1,4]" in caplog.text
    assert not any("*" in record.getMessage() for record in caplog.records)


def test_pareto_range_scope_excludes_infeasible_outliers(tmp_path, caplog):
    from config.scoring import ObjectiveSpec

    caplog.set_level(logging.INFO)
    store = ParetoStore(str(tmp_path))
    store.scoring_specs = [ObjectiveSpec("adg_strategy_eq", "max"),
                           ObjectiveSpec("drawdown_worst_strategy_eq", "min")]
    store.scoring_keys = [spec.metric for spec in store.scoring_specs]
    # Exercise scope selection directly with infeasible outliers at both ends.
    store._front = ["first", "second", "infeasible"]
    store._objectives = {"first": (2, 2), "second": (3, 3), "infeasible": (999, -1)}
    store._violations = {"first": 0, "second": 0, "infeasible": 1}
    store._log_front_state(added=1, removed=0)
    assert "best_scope=feasible_front" in caplog.text
    assert "adg_strategy_eq=[2,3*]" in caplog.text
    assert "drawdown_worst_strategy_eq=[2*,3]" in caplog.text
    assert "999" not in caplog.text
    assert "[-1" not in caplog.text


def test_console_failure_does_not_prevent_pareto_persistence(tmp_path):
    class BrokenConsole(logging.Handler):
        def emit(self, record):
            raise OSError("injected console failure")
    store = ParetoStore(str(tmp_path), log_name="broken_optimizer_console")
    store._log.setLevel(logging.INFO)
    handler = BrokenConsole()
    store._log.addHandler(handler)
    try:
        assert store.add_entry(entry(0))
        assert store.add_entry(entry(1))
        store.flush_now()
        assert store.n_iters == 2
        assert len(store.get_front()) == len(list((tmp_path / "pareto").glob("*.json"))) == 2
    finally:
        store._log.removeHandler(handler)


def test_optimizer_snapshot_cadence_phases_and_no_hidden_fields(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [0.0]
    monkeypatch.setattr("optimization.progress.time.monotonic", lambda: now[0])
    snapshot = dict(gen=2, evolution_proxy_completed_run=1024,
        seed_exact=10, evolution_exact="5/100", evolution_pending=2, front=4,
        pareto_added=3, last_pareto_age="unknown")
    progress = OptimizerProgress(lambda: snapshot)
    progress.transition("gpu_proxy")
    count = len(caplog.records)
    now[0] = 59.999
    progress.report()
    assert len(caplog.records) == count
    now[0] = 60
    progress.report()
    assert len(caplog.records) == 2 * count
    assert "run_elapsed=1m00s" in caplog.text
    progress.transition("exact_wait")
    assert "gen=2 phase=exact_wait" in caplog.text
    assert "evolution_pending=2" in caplog.text and "pareto_added=3" in caplog.text
    assert all(len(r.getMessage()) <= 240 for r in caplog.records)
    assert snapshot["gen"] == 2




def test_durations_and_long_record_splitting(caplog):
    caplog.set_level(logging.INFO)
    assert duration(3723) == "1h02m03s"
    assert duration(None) == "unknown"
    assert duration(0) == "0s"
    long = "metric=" + "x" * 600
    log_tokens("progress |", [long, "following=1\nunsafe=2"])
    assert all(len(r.getMessage()) <= 240 for r in caplog.records)
    assert sum(r.getMessage().count("x") for r in caplog.records) == 600
    assert "following=1_unsafe=2" in caplog.text


def test_optional_snapshot_failure_is_visible_and_retried(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    now = [0.0]
    monkeypatch.setattr("optimization.progress.time.monotonic", lambda: now[0])
    def unavailable():
        raise OSError("private detail")
    progress = OptimizerProgress(unavailable)
    progress.transition("gpu_proxy")
    assert "snapshot=unavailable error_type=OSError" in caplog.text
    assert "private detail" not in caplog.text
    progress.snapshot = lambda: dict(gen=2, evolution_exact="1/10")
    now[0] = 60
    progress.report()
    assert "gen=2 phase=gpu_proxy" in caplog.records[-1].getMessage()
