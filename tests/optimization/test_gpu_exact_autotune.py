"""Exact tuning changes execution capacity, never validation allocations."""

import argparse
import json
from types import SimpleNamespace

import pytest

from optimization.gpu import exact_autotune as tune


def controller(tmp_path, *, proxies=(), mode="auto", context=None):
    tick = [0.0]
    item = tune.ExactQueueController(
        4,
        8,
        proxies,
        cache_dir=tmp_path,
        mode=mode,
        clock=lambda: tick[0],
        hardware={"device": "test"},
        context=context,
    )
    item.tick = tick
    item.gen = 0
    return item


def window(item, seconds=2, work=1, wait=0):
    for _ in range(tune.WINDOW):
        item.tick[0] += seconds
        item.gen += 1
        item.record(work, wait)
        item.update(item.gen)


def test_queue_waits_for_evidence_and_excludes_cold_window(tmp_path):
    item = controller(tmp_path)
    window(item, 0.01)
    assert not item.warmed and item.limit == 16
    item.tick[0] += 31
    item.update(item.gen)
    assert item.warmed and item.limit == 16
    window(item)
    assert item.limit == 24 and item.baseline[0] == 16
    window(item, 1.5)
    assert item.limit == 24 and item.baseline is None and item.cooldown == 1


def test_queue_failure_rolls_back_and_small_plateau_is_preferred(tmp_path):
    item = controller(tmp_path)
    window(item)
    window(item)
    window(item, 3)
    assert item.limit == 16 and item.cooldown == 3
    for _ in range(3):
        window(item)
        assert item.limit == 16
    window(item)
    assert item.limit == 8
    window(item, 2.02)
    assert item.limit == 8 and item.baseline is None
    item.cooldown = 0
    window(item)
    assert item.limit >= item.floor


def test_queue_median_rejects_spike_invalid_samples_and_bounds_records(tmp_path):
    item = controller(tmp_path)
    for _ in range(item.ceiling * 4):
        item.record(1, 0)
    assert len(item.collected) == item.ceiling
    item.reset(0)
    for bad in [0, -1, float("nan"), float("inf")]:
        item.record(bad, 0)
        item.record(1, bad)
    item.update(0)
    assert item.completed == 1  # zero queue delay is valid
    item.reset(0)
    item.tick[0] = 60
    for _ in range(24):
        item.record(1, 0)
    item.update(0)  # Requires several generations, too.
    assert not item.warmed
    item.update(4)
    assert item.warmed


def test_proxy_and_queue_trials_are_coordinated_at_generation_boundary(tmp_path):
    batch = SimpleNamespace(baseline=None)
    proxy_tuner = SimpleNamespace(revision=0, controllers={"shape": batch})
    proxy = SimpleNamespace(batch_tuner=proxy_tuner, checkpoint_contract={}, needed_metrics=[])
    item = controller(tmp_path, proxies=[proxy])
    window(item)
    window(item)
    assert not proxy_tuner.allow_trial()
    # Collector timing cannot make a queue decision until main-thread update.
    item.record(1, 0)
    assert item.limit == 24
    window(item, 1.5)
    assert proxy_tuner.allow_trial()
    item.record(1, 0)
    proxy_tuner.revision += 1
    item.update(item.gen)
    assert not item.samples and item.completed == 0
    batch.baseline = (16, 1)
    window(item)
    assert item.completed == 0


def test_queue_cache_reuse_refresh_and_workload_separation(tmp_path):
    item = controller(tmp_path)
    window(item)
    window(item)
    window(item, 1.5)
    assert controller(tmp_path).limit == 24
    assert controller(tmp_path, mode="refresh").limit == 16
    assert controller(tmp_path, context={"bounds": "different"}).limit == 16
    path = tmp_path / (item.key + ".json")
    path.write_text(
        json.dumps({"version": 1, "max_pending_exact": 999, "candidates_per_second": 1})
    )
    assert controller(tmp_path).limit == 16


def test_queue_cache_failure_keeps_tuning(tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    item = controller(blocked)
    window(item)
    window(item)
    assert item.limit == 24 and item.cache.cache_write_disabled
    assert "continuing with in-memory tuning" in caplog.text


def test_workers_use_cores_memory_and_largest_shared_replay(monkeypatch):
    contexts = [SimpleNamespace(shared_hlcvs_np={"x": SimpleNamespace(nbytes=100 * tune.MIB)})] * 3
    evaluator = SimpleNamespace(contexts=contexts)
    monkeypatch.setattr(
        tune,
        "resource_snapshot",
        lambda: dict(cores=8, available=8 * 1024 * tune.MIB, rss=512 * tune.MIB),
    )
    assert tune.prepared_bytes(evaluator) == 100 * tune.MIB
    assert tune.initial_workers(None, 4, evaluator, mode="auto") == 6
    assert tune.initial_workers(None, 4, evaluator, mode="auto", pending=2) == 2
    monkeypatch.setattr(
        tune, "resource_snapshot", lambda: dict(cores=1, available=0, rss=512 * tune.MIB)
    )
    assert tune.initial_workers(None, 4, evaluator, mode="auto") == 1


@pytest.mark.parametrize(
    "requested,mode,expected", [(0, "auto", 4), (2, "auto", 2), (None, "off", 4)]
)
def test_fixed_and_legacy_workers_do_not_inspect_resources(monkeypatch, requested, mode, expected):
    monkeypatch.setattr(tune, "resource_snapshot", lambda: pytest.fail("hardware queried"))
    assert tune.initial_workers(requested, 4, None, mode=mode) == expected


def test_worker_detection_failure_is_observable(monkeypatch, caplog):
    def fail():
        raise OSError("unavailable")

    monkeypatch.setattr(tune, "resource_snapshot", fail)
    assert tune.initial_workers(None, 4, None, mode="auto") == 4
    assert "auto-sizing unavailable" in caplog.text


def test_cpu_affinity_and_container_limits_are_respected(monkeypatch):
    monkeypatch.setattr(tune.psutil, "cpu_count", lambda logical=True: 16 if logical else 8)
    monkeypatch.setattr(
        tune.psutil,
        "Process",
        lambda: SimpleNamespace(
            cpu_affinity=lambda: [0, 1], memory_info=lambda: SimpleNamespace(rss=123)
        ),
    )
    monkeypatch.setattr(tune.psutil, "virtual_memory", lambda: SimpleNamespace(available=1000))
    monkeypatch.setattr(tune, "_cgroup_limits", lambda: (1, 500))
    assert tune.resource_snapshot() == dict(cores=1, available=500, rss=123)


def test_cgroup_process_and_ancestor_limits(tmp_path):
    root = tmp_path / "cgroup"
    child = root / "slice/job"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/slice/job\n")
    (root / "cpu.max").write_text("max 100000")
    (child / "cpu.max").write_text("600000 100000")
    (root / "slice/cpu.max").write_text("250000 100000")
    (root / "memory.max").write_text("1000")
    (root / "memory.current").write_text("200")
    (child / "memory.max").write_text("2000")
    (child / "memory.current").write_text("500")
    assert tune._cgroup_limits(root, membership) == (2, 800)
    membership.write_text("0::/../../escape\n")
    assert tune._cgroup_limits(root, membership) == (None, None)


@pytest.mark.parametrize("field", ["exact_workers", "max_pending_exact"])
@pytest.mark.parametrize("value", [None, "auto", " AUTO "])
def test_auto_exact_options_roundtrip_and_cli(field, value):
    from config_utils import get_template_config, add_arguments_recursively, format_config
    from optimization.backends.gpu_backend import _resolve_options

    config = get_template_config()
    assert config["optimize"]["gpu"][field] is None
    config["optimize"]["gpu"][field] = value
    assert _resolve_options(format_config(config))[field] is None
    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, {"optimize": {"gpu": {field: 4}}})
    args = parser.parse_args(["--optimize.gpu." + field, "auto"])
    assert vars(args)["optimize.gpu." + field] == "auto"
    config["optimize"]["gpu"][field] = 0
    assert _resolve_options(config)[field] == 0
    config["optimize"]["gpu"][field] = -1
    with pytest.raises(ValueError, match=field):
        _resolve_options(config)
