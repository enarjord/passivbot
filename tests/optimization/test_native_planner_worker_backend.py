from collections import Counter
import json
from types import SimpleNamespace

import pytest

from optimization.backends import gpu_backend
from optimization.native_checkpoint import load_checkpoint
from shared_arrays import SharedArrayManager
from test_native_backend import FakeService, execute, guard_cpu, inputs


@pytest.fixture
def file_backed_planner(monkeypatch):
    from optimization import native_datasets

    guard_cpu(monkeypatch)
    monkeypatch.setattr(native_datasets, "SharedArrayManager", lambda: SharedArrayManager(file_backed=True))
    manager = SharedArrayManager(file_backed=True)
    try:
        yield inputs(manager)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("mode,iters,expected_capacity", [
    ("conservative", 8, []),
    ("max", 1, []),
    ("max", 8, [4]),
])
def test_cuda_worker_selection_is_bounded_and_keeps_serial_fallback(
    monkeypatch, tmp_path, file_backed_planner, mode, iters, expected_capacity,
):
    base = file_backed_planner
    base.config["optimize"]["iters"] = iters
    base.config["optimize"]["gpu"].update(tuning_mode=mode, batch_size=4)
    observed = []

    def no_capacity(planner, lifetime, *, device, available_rows, interrupt_check):
        observed.append(available_rows)
        assert device == "cuda"
        assert callable(planner.prepare)
        assert callable(lifetime.enter_context)
        assert callable(interrupt_check)
        return None

    monkeypatch.setattr(gpu_backend, "_open_preparation_workers", no_capacity)
    rows = []
    execute(base, SimpleNamespace(record=rows.append), tmp_path / "selection.pkl")
    assert observed == expected_capacity
    assert len(rows) == iters


def test_large_search_budget_does_not_inflate_cpu_preparation_headroom(
    monkeypatch, tmp_path, file_backed_planner,
):
    from optimization import native_planner_capacity as capacity
    from optimization.gpu import native

    class WorkersCanStart(Exception):
        pass

    base = file_backed_planner
    base.config["optimize"]["iters"] = 65536
    base.config["optimize"]["gpu"].update(tuning_mode="max", batch_size=None)
    monkeypatch.setattr(native, "cuda_initial_candidate_width", lambda: 512)
    monkeypatch.setattr(native, "NativeBacktestService",
                        lambda **kwargs: FakeService(**{**kwargs, "batch_size": 512}))

    # The registered metadata is small, so each candidate reserves 256 KiB.
    # The live allocation admits at least four workers for 512 plans, but
    # charging the entire 8192-slot service ceiling would admit no pool.
    monkeypatch.setattr(capacity, "_host_capacity", lambda: (20, 3 * 2**30))
    select_count = capacity.cuda_plan_worker_count

    def stop_after_selection(planner, *, admission_slots, available_rows):
        assert admission_slots == available_rows == 512
        assert select_count(planner, admission_slots=8192, available_rows=available_rows) == 0
        assert select_count(planner, admission_slots=admission_slots, available_rows=available_rows) == 4
        raise WorkersCanStart

    monkeypatch.setattr(capacity, "cuda_plan_worker_count", stop_after_selection)
    checkpoint = tmp_path / "wide-budget.pkl"
    recorder = SimpleNamespace(record=lambda row: pytest.fail("stopped before evaluation"))
    with pytest.raises(WorkersCanStart):
        execute(base, recorder, checkpoint)
    state = load_checkpoint(checkpoint, base.config)
    assert state["phase"] == "seeds"
    assert state["completed"] == 0
    assert state["max_search_allocated"] == len(state["population"]) == 512


def test_existing_worker_pool_is_reused_and_tiny_tail_stays_serial(monkeypatch):
    from optimization import native_planner_capacity

    search = object.__new__(gpu_backend._Search)
    search._preparation_lifetime = object()
    search._preparation_device = "cuda"
    search._preparation_workers = None
    search._largest_preparation_window = 0
    search.session = SimpleNamespace(max_candidates=12)
    search.registry = SimpleNamespace(planner=SimpleNamespace(bindings=()))
    search.interrupt_check = lambda: None
    pool = SimpleNamespace(worker_count=4, pending_count=0)
    chosen = []

    def select(_planner, _lifetime, *, device, available_rows, interrupt_check):
        assert device == "cuda"
        chosen.append(available_rows)
        return pool

    monkeypatch.setattr(gpu_backend, "_open_preparation_workers", select)
    monkeypatch.setattr(native_planner_capacity, "cuda_plan_pool_fits", lambda *args, **kwargs: True)
    assert search._workers_for_window(0) is None
    assert search._workers_for_window(1) is None
    assert search._workers_for_window(32) is pool
    assert search._workers_for_window(3) is None
    assert search._workers_for_window(4) is pool
    assert chosen == [12]


def test_existing_worker_pool_retires_before_next_cohort_on_host_pressure(monkeypatch):
    from optimization import native_planner_capacity

    search = object.__new__(gpu_backend._Search)
    search._preparation_workers = None
    search._preparation_device = "cuda"
    search._largest_preparation_window = 0
    search.session = SimpleNamespace(max_candidates=128)
    planner = object()
    search.registry = SimpleNamespace(planner=planner)
    search.interrupt_check = lambda: None
    phases = []
    search._preparation_lifetime = SimpleNamespace(close=lambda: phases.append("retired"))
    pool = SimpleNamespace(worker_count=4, pending_count=0)

    def open_pool(*_args, **_kwargs):
        phases.append("open")
        return pool

    def estimate(candidate_planner, admission_slots):
        assert candidate_planner is planner
        phases.append(("estimate", admission_slots))
        return admission_slots, 45

    def fits(*, parent_bytes, workers):
        assert workers == 4
        return parent_bytes <= 128

    monkeypatch.setattr(gpu_backend, "_open_preparation_workers", open_pool)
    monkeypatch.setattr(native_planner_capacity, "cuda_plan_preparation_memory", estimate)
    monkeypatch.setattr(native_planner_capacity, "cuda_plan_pool_fits", fits)

    assert search._workers_for_window(128) is pool
    pool.pending_count = 1
    with pytest.raises(RuntimeError, match="pending rows across cohorts"):
        search._workers_for_window(128)
    assert search._preparation_workers is pool and phases == ["open"]
    pool.pending_count = 0
    # A nominal wider phase is still safe if resumption has only 128 pending.
    search.session.max_candidates = 512
    assert search._workers_for_window(128) is pool
    assert search._workers_for_window(512) is None
    assert search._workers_for_window(128) is None
    assert search._preparation_workers is None
    assert phases == ["open", ("estimate", 128), ("estimate", 512), "retired"]


def test_cuda_worker_startup_is_interruptible_before_admission(monkeypatch, tmp_path, file_backed_planner):
    from optimization import native_planner_capacity

    base = file_backed_planner
    base.config["optimize"]["gpu"].update(tuning_mode="max", batch_size=4)
    checkpoint = tmp_path / "startup.pkl"
    rows = []
    stop = False

    def interrupt_check():
        if stop:
            raise KeyboardInterrupt

    def capacity(planner, *, admission_slots, available_rows):
        from optimization import native_planner_workers

        class InterruptingStartup:
            def __init__(self, *_args, interrupt_check, **_kwargs):
                self.interrupt_check = interrupt_check

            def __enter__(self):
                nonlocal stop
                stop = True
                self.interrupt_check()

            def __exit__(self, *_args):
                pytest.fail("A context that failed to start must not be entered")

        monkeypatch.setattr(native_planner_workers, "OrderedCandidateWorkers", InterruptingStartup)
        return 2

    monkeypatch.setattr(native_planner_capacity, "cuda_plan_worker_count", capacity)
    with pytest.raises(KeyboardInterrupt):
        execute(base, SimpleNamespace(record=rows.append), checkpoint, interrupt_check=interrupt_check)
    state = load_checkpoint(checkpoint, base.config)
    assert state["phase"] == "seeds" and state["sequence"] == state["completed"] == 0
    assert not rows and not FakeService.rows


@pytest.mark.parametrize("interrupt", [False, True])
def test_ordered_workers_preserve_gpu_request_plans_records_and_resume(
    monkeypatch, tmp_path, file_backed_planner, interrupt,
):
    import passivbot_rust

    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real native Rust extension required for spawned planners")
    from optimization import native_planner_capacity

    base = file_backed_planner
    base.config["optimize"]["gpu"].update(tuning_mode="max", batch_size=4)
    serial_rows, parallel_rows = [], []
    serial_path, parallel_path = tmp_path / "serial.pkl", tmp_path / "parallel.pkl"
    launches = []
    use_workers = False

    def worker_count(planner, *, admission_slots, available_rows):
        if not use_workers:
            return 0
        launches.append((admission_slots, available_rows))
        return 2

    def record(row):
        if interrupt and len(parallel_rows) == 3:
            raise KeyboardInterrupt
        parallel_rows.append(row)

    monkeypatch.setattr(native_planner_capacity, "cuda_plan_worker_count", worker_count)
    monkeypatch.setattr(native_planner_capacity, "cuda_plan_pool_fits", lambda *args, **kwargs: True)
    assert execute(base, SimpleNamespace(record=serial_rows.append), serial_path) == {
        "pool": None, "pool_terminated": False,
    }
    expected_vectors = list(FakeService.rows)
    FakeService.rows.clear()
    use_workers = True
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            execute(base, SimpleNamespace(record=record), parallel_path)
        saved = load_checkpoint(parallel_path, base.config)
        assert saved["completed"] == len(parallel_rows) == 3
        assert saved["phase"] != "idle"
        assert all("worker" not in key for key in saved)
        execute(base, SimpleNamespace(record=parallel_rows.append), parallel_path, resume=True)
    else:
        execute(base, SimpleNamespace(record=record), parallel_path)
        assert FakeService.rows == expected_vectors
    assert launches == [(4, 4)] * (2 if interrupt else 1)
    assert load_checkpoint(parallel_path, base.config)["phase"] == "idle"
    assert len(parallel_rows) == len(serial_rows) == 8
    assert Counter(json.dumps(row, sort_keys=True) for row in parallel_rows) == Counter(
        json.dumps(row, sort_keys=True) for row in serial_rows
    )
