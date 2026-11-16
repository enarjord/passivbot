from types import SimpleNamespace

import pytest

from optimization import native_planner_capacity as capacity


def test_cgroup_headroom_respects_ancestor_memory_high(tmp_path):
    mount = tmp_path / "cgroup with spaces"
    child = mount / "worker"
    child.mkdir(parents=True)
    (mount / "memory.max").write_text("5000")
    (mount / "memory.high").write_text("3000")
    (mount / "memory.current").write_text("1500")
    (mount / "cpu.max").write_text("450000 100000")
    (child / "memory.max").write_text("4000")
    (child / "memory.current").write_text("1000")
    (child / "cpu.max").write_text("600000 100000")
    (tmp_path / "cgroup").write_text("0::/worker\n")
    encoded_mount = str(mount).replace(" ", "\\040")
    (tmp_path / "mountinfo").write_text(f"35 27 0:32 / {encoded_mount} rw - cgroup2 cgroup rw\n")

    assert capacity._read_cgroup_limits(tmp_path) == (4.5, 1500)


@pytest.mark.parametrize(
    "host,available_rows,expected",
    [(None, 512, 0), ((20, 8192), 40, 0), ((5, 8192), 512, 0),
     ((6, 8192), 512, 4), ((20, 8192), 80, 4), ((20, 8192), 512, 8),
     ((20, 2350), 512, 0), ((20, 3000), 512, 4)],
)
def test_worker_count_respects_cpu_memory_and_useful_work(monkeypatch, host, available_rows, expected):
    planner = SimpleNamespace(bindings=())
    capacity_sample = None if host is None else (host[0], host[1] * 2**20)
    monkeypatch.setattr(capacity, "_host_capacity", lambda: capacity_sample)

    assert capacity.cuda_plan_worker_count(planner, admission_slots=256, available_rows=available_rows) == expected


def test_packing_and_wide_admission_keep_headroom_for_cuda_owner(monkeypatch):
    candle_spec = SimpleNamespace(shape=(1_000_000, 30, 4), dtype="float64")
    dataset = SimpleNamespace(
        time_range=(0, 1_000_000),
        hlcvs=candle_spec,
        coin_indices=list(range(30)),
        config_json="{}",
        markets_json="{}",
    )
    planner = SimpleNamespace(bindings=(SimpleNamespace(dataset=dataset),))
    monkeypatch.setattr(capacity, "_host_capacity", lambda: (20, 6 * 2**30))

    wide = capacity.cuda_plan_worker_count(planner, admission_slots=8192, available_rows=512)
    small = capacity.cuda_plan_worker_count(planner, admission_slots=512, available_rows=512)
    assert wide == 0
    assert small == 4


def test_parent_admission_does_not_duplicate_registered_markets_for_each_candidate():
    dataset = SimpleNamespace(
        time_range=(0, 128),
        hlcvs=SimpleNamespace(shape=(128, 1, 4), dtype="float64"),
        coin_indices=[0],
        config_json="{}",
        markets_json="{}",
    )
    planner = SimpleNamespace(bindings=(SimpleNamespace(dataset=dataset),))
    before_parent, before_worker = capacity.cuda_plan_preparation_memory(planner, 4096)
    dataset.markets_json += " " * 2**20
    after_parent, after_worker = capacity.cuda_plan_preparation_memory(planner, 4096)
    assert after_parent - before_parent == 2**20
    assert after_worker > before_worker


@pytest.mark.parametrize("host,expected", [(None, None), ((5, 8192), None), ((20, 8192), 8192)])
def test_parallel_window_preserves_device_ceiling_when_host_cannot_improve_admission(monkeypatch, host, expected):
    sample = None if host is None else (host[0], host[1] * 2**20)
    monkeypatch.setattr(capacity, "_host_capacity", lambda: sample)
    assert capacity.cuda_plan_parallel_window(SimpleNamespace(bindings=()), 8192) == expected


@pytest.mark.parametrize("available,window", [(7000, 2048), (7400, 3072)])
def test_parallel_window_can_reduce_admission_to_retain_more_workers(monkeypatch, available, window):
    planner = SimpleNamespace(bindings=())
    monkeypatch.setattr(capacity, "_host_capacity", lambda: (20, available))
    monkeypatch.setattr(capacity, "cuda_plan_preparation_memory", lambda _planner, slots: (slots, 512))
    assert capacity.cuda_plan_worker_count(planner, admission_slots=4096, available_rows=4096) == 4
    assert capacity.cuda_plan_parallel_window(planner, 4096) == window


@pytest.mark.parametrize("host,expected", [(None, False), ((9, 30), False), ((20, 19), False), ((10, 20), True)])
def test_resident_pool_reserves_remaining_parent_headroom_without_double_charging_workers(monkeypatch, host, expected):
    monkeypatch.setattr(capacity, "_host_capacity", lambda: host)
    assert capacity.cuda_plan_pool_fits(parent_bytes=20, workers=8) is expected
