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


@pytest.mark.parametrize("host,expected", [(None, False), ((9, 30), False), ((20, 19), False), ((10, 20), True)])
def test_resident_pool_reserves_remaining_parent_headroom_without_double_charging_workers(monkeypatch, host, expected):
    monkeypatch.setattr(capacity, "_host_capacity", lambda: host)
    assert capacity.cuda_plan_pool_fits(parent_bytes=20, workers=8) is expected
