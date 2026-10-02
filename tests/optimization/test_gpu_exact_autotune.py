"""Exact tuning changes execution capacity, never validation allocations."""

import argparse
import json
import math
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


def window(item, seconds=2, work=None, wait=0):
    count = max(tune.WINDOW, math.ceil(tune.MIN_SECONDS * item.workers / seconds))
    for _ in range(count):
        started = item.tick[0]
        item.tick[0] += seconds
        item.gen += 1
        item.record(seconds if work is None else work, wait, started, item.tick[0], epoch=item.epoch)
        item.update(item.gen)


def test_queue_waits_for_real_work_evidence_and_excludes_cold_window(tmp_path):
    item = controller(tmp_path)
    for index in range(tune.WINDOW):
        item.record(0.01, 0, index * 0.01, (index + 1) * 0.01, epoch=item.epoch)
    item.update(0)
    assert not item.warmed and item.limit == 16
    item.tick[0] += 36000  # GPU-only time cannot manufacture CPU evidence.
    item.update(100)
    assert not item.warmed and item.limit == 16
    item.reset(100)
    window(item)
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
        item.record(1, 0, item.tick[0], item.tick[0] + 1, epoch=item.epoch)
    assert len(item.collected) == item.ceiling
    item.reset(0)
    for bad in [0, -1, float("nan"), float("inf")]:
        item.record(bad, 0, 0, 1, epoch=item.epoch)
        item.record(1, bad, 0, 1, epoch=item.epoch)
    item.update(0)
    assert item.completed == 1  # zero queue delay is valid
    item.reset(0)
    for index in range(24):
        item.record(5, 0, index * 5, (index + 1) * 5, epoch=item.epoch)
    item.update(0)  # Enough CPU work permits calibration within seed generation.
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
    item.record(1, 0, item.tick[0], item.tick[0] + 1, epoch=item.epoch)
    assert item.limit == 24
    window(item, 1.5)
    assert proxy_tuner.allow_trial()
    item.record(1, 0, item.tick[0], item.tick[0] + 1, epoch=item.epoch)
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


def test_worker_sizing_excludes_gpu_coordinator_rss_but_uses_remaining_ram(monkeypatch):
    rss = [512 * tune.MIB]
    monkeypatch.setattr(tune.psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=rss[0]),
    ))
    baseline = tune.capture_worker_rss(None, mode="auto")
    rss[0] = 10 * 1024 * tune.MIB  # GPU history remains only in the coordinator.
    monkeypatch.setattr(tune, "resource_snapshot", lambda: dict(
        cores=8, available=8 * 1024 * tune.MIB, rss=rss[0],
    ))
    evaluator = SimpleNamespace(shared_hlcvs_np={
        "x": SimpleNamespace(nbytes=100 * tune.MIB),
    })
    assert tune.initial_workers(None, 4, evaluator, mode="auto", baseline_rss=baseline) == 6
    monkeypatch.setattr(tune, "resource_snapshot", lambda: dict(
        cores=8, available=1024 * tune.MIB, rss=rss[0],
    ))
    assert tune.initial_workers(None, 4, evaluator, mode="auto", baseline_rss=baseline) == 1


@pytest.mark.parametrize("requested,mode", [(0, "auto"), (2, "auto"), (None, "off")])
def test_worker_baseline_skips_fixed_and_off(monkeypatch, requested, mode):
    monkeypatch.setattr(tune.psutil, "Process", lambda: pytest.fail("hardware queried"))
    assert tune.capture_worker_rss(requested, mode=mode) is None


def test_worker_baseline_failure_warns_and_keeps_conservative_fallback(monkeypatch, caplog):
    def fail():
        raise OSError("unavailable")
    monkeypatch.setattr(tune.psutil, "Process", fail)
    assert tune.capture_worker_rss(None, mode="auto") is None
    assert "memory baseline unavailable" in caplog.text


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


def test_lazy_suite_worker_sizing_uses_prepared_views_not_empty_context_maps(monkeypatch):
    contexts = [SimpleNamespace(shared_hlcvs_np={}, exchanges=["x"], length=size)
                for size in (256 * tune.MIB, 2 * 1024 * tune.MIB)]
    calls = []

    def prepared(ctx, exchange):
        calls.append((ctx, exchange))
        return SimpleNamespace(nbytes=ctx.length), None, [0]

    evaluator = SimpleNamespace(contexts=contexts, get_prepared_context_data=prepared)
    monkeypatch.setattr(tune, "resource_snapshot", lambda: dict(
        cores=16, available=8 * 1024 * tune.MIB, rss=512 * tune.MIB,
    ))
    assert tune.prepared_bytes(evaluator) == 2 * 1024 * tune.MIB
    assert tune.initial_workers(None, 4, evaluator, mode="auto") == 1
    assert len(calls) == 4  # each context once per sizing pass; no sum across scenarios


def test_lazy_suite_prepared_bytes_preserves_time_view_and_coin_mapping():
    import numpy as np
    from optimize import SuiteEvaluator
    master = np.zeros((100, 4, 4), dtype=np.float64)
    spec = SimpleNamespace(name="fixture")
    ctx = SimpleNamespace(
        exchanges=["x"], shared_hlcvs_np={}, master_hlcvs_specs={"x": spec},
        time_slice={"x": (20, 50)}, coin_slice_indices={"x": [0, 2]},
        master_btc_specs=None,
    )
    evaluator = SuiteEvaluator.__new__(SuiteEvaluator)
    evaluator.contexts = [ctx, ctx]
    evaluator._master_arrays = {"hlcvs": {"fixture": master}, "btc": {}}
    view, _, coins = evaluator.get_prepared_context_data(ctx, "x")
    assert np.shares_memory(view, master) and coins == [0, 2]
    assert tune.prepared_bytes(evaluator) == master[20:50].nbytes
    assert ctx.shared_hlcvs_np == {}


def test_long_validations_calibrate_during_seed_admission_without_four_generations(tmp_path):
    item = controller(tmp_path)
    def complete_wave(start, duration):
        for _ in range(4):
            item.record(duration, 0, start, start + duration, epoch=item.epoch)
        item.update(0)
    complete_wave(0, 120)
    assert item.warmed and item.limit == 16
    complete_wave(10000, 120)
    assert item.limit == 24 and item.baseline[1] == pytest.approx(4 / 120)
    complete_wave(20000, 90)
    # Four jobs alone still need the generous 120-second active window.
    assert item.baseline is not None
    complete_wave(20090, 90)
    assert item.baseline is None and item.limit == 24
    assert item.cooldown == 1


def test_bootstrap_exit_rolls_back_unfinished_queue_trial_and_unblocks_batch(tmp_path):
    batch = SimpleNamespace(baseline=None)
    tuner = SimpleNamespace(revision=0, controllers={"shape": batch})
    proxy = SimpleNamespace(batch_tuner=tuner, checkpoint_contract={}, needed_metrics=[])
    item = controller(tmp_path, proxies=[proxy])
    window(item)
    window(item)
    assert item.limit == 24 and not tuner.allow_trial()
    item.finish_bootstrap(0)
    assert item.limit == 16 and item.baseline is None and tuner.allow_trial()
    assert not item.samples


def test_temporal_batch_and_exact_queue_trials_keep_revision_and_cooldown_boundaries(tmp_path):
    from optimization.gpu.autotune import ProxyBatchTuner, proxy_batches, record_replay_chunk
    proxy = SimpleNamespace(checkpoint_contract={}, needed_metrics=[],
                            max_dispatch_candidate_bars=1000000)
    proxy.batch_tuner = ProxyBatchTuner(
        proxy, mode="refresh", hardware={"device": "test"}, context={}, cache_dir=tmp_path,
    )
    proxy.batch_tuner._headroom = lambda: True
    item = controller(tmp_path, proxies=[proxy])
    item.baseline = (16, 1)
    batches = proxy_batches(proxy, list(range(2048)), 512)
    next(batches)
    for _ in range(128):
        record_replay_chunk(512, 4096, 2538000, 2)
    assert len(next(batches)[1]) == 512
    assert proxy.batch_tuner.revision == 0
    item.baseline = None
    for _ in range(128):
        record_replay_chunk(512, 4096, 2538000, 2)
    assert len(next(batches)[1]) == 256
    assert proxy.batch_tuner.revision == 1
    item.record(120, 0, 0, 120, epoch=item.epoch)
    item.update(0)
    assert item.completed == 0 and not item.samples
    batches.close()


@pytest.mark.parametrize("direction,limit", [(1, 16), (-1, 32)])
@pytest.mark.parametrize("long_work", [False, True])
def test_queue_trial_ignores_jobs_admitted_before_grow_or_shrink(tmp_path, direction, limit, long_work):
    item = controller(tmp_path)
    item.warmed = True
    item.direction = direction
    item.limit = limit
    old_epoch = item.epoch
    window(item)
    assert item.limit == 24 and item.baseline[0] == limit
    assert item.epoch != old_epoch
    # Even a complete regular/expensive window of old-queue work cannot
    # resolve this trial or cache the new admission setting.
    for index in range(4 if long_work else 24):
        start = 10000 if long_work else 10000 + index * 5
        item.record(120 if long_work else 5, 0, start,
                    start + (120 if long_work else 5), epoch=old_epoch)
    item.update(0)
    assert item.baseline is not None and not item.samples and item.completed == 0
    assert item.cache.read(item.key, "max_pending_exact", item.floor, item.ceiling) == limit
    window(item, 1.5)
    assert item.baseline is None and item.limit == 24
    assert item.cache.read(item.key, "max_pending_exact", item.floor, item.ceiling) == 24


def test_accepted_temporal_batch_trial_invalidates_collected_cpu_evidence(tmp_path):
    from optimization.gpu.autotune import ProxyBatchTuner, proxy_batches, record_replay_chunk
    proxy = SimpleNamespace(checkpoint_contract={}, needed_metrics=[],
                            max_dispatch_candidate_bars=1000000)
    proxy.batch_tuner = ProxyBatchTuner(
        proxy, mode="refresh", hardware={"device": "test"}, context={}, cache_dir=tmp_path,
    )
    proxy.batch_tuner._headroom = lambda: True
    item = controller(tmp_path, proxies=[proxy])
    batches = proxy_batches(proxy, list(range(2048)), 512)
    next(batches)
    for _ in range(128):
        record_replay_chunk(512, 4096, 2538000, 2)
    next(batches)
    assert proxy.batch_tuner.revision == 1
    item.update(0)  # Marks the active trial and invalidates pre-trial admissions.
    epoch = item.epoch
    for _ in range(4):
        item.record(120, 0, 0, 120, epoch=epoch)
    for _ in range(128):
        record_replay_chunk(256, 4096, 2538000, 0.8)
    assert len(next(batches)[1]) == 256  # Accepted smaller plateau; width stays put.
    assert proxy.batch_tuner.revision == 2
    assert not proxy.batch_tuner.controllers[next(iter(proxy.batch_tuner.controllers))].baseline
    item.update(0)
    assert item.epoch != epoch and not item.samples and item.completed == 0
    # Late collectors also cannot resurrect evidence from that final trial pass.
    for _ in range(4):
        item.record(120, 0, 0, 120, epoch=epoch)
    item.update(0)
    assert not item.warmed
    batches.close()


def test_one_shot_seed_batch_trial_retires_before_exact_queue_evidence(tmp_path):
    from optimization.gpu.autotune import ProxyBatchTuner, proxy_batches, record_replay_chunk
    proxy = SimpleNamespace(checkpoint_contract={}, needed_metrics=[],
                            max_dispatch_candidate_bars=1000000)
    proxy.batch_tuner = ProxyBatchTuner(
        proxy, mode="refresh", hardware={"device": "test"}, context={}, cache_dir=tmp_path,
    )
    proxy.batch_tuner._headroom = lambda: True
    item = controller(tmp_path, proxies=[proxy])
    batches = proxy_batches(proxy, list(range(768)), 512)
    next(batches)
    for _ in range(128):
        record_replay_chunk(512, 4096, 2538000, 2)
    next(batches)
    # Seed demand ends before an expensive trial has enough evidence.
    record_replay_chunk(256, 4096, 2538000, 2)
    with pytest.raises(StopIteration):
        next(batches)
    seed = next(iter(proxy.batch_tuner.controllers.values()))
    assert seed.baseline is not None
    item.finish_seed_screen(0)
    assert seed.width == 512 and seed.baseline is None and seed.cooldown == 1
    assert not item.proxy_state()[1]
    assert proxy.batch_tuner.revision == 2  # proposal and retirement invalidate evidence.
    item.update(0)
    window(item)
    window(item)
    assert item.limit == 24 and item.baseline is not None  # Seed-only class cannot block.
    item.finish_bootstrap(0)
    evolution = proxy_batches(proxy, list(range(1024)), 512)
    next(evolution)
    assert len(proxy.batch_tuner.controllers) == 2
    assert not item.proxy_state()[1]
    evolution.close()


def test_inactive_class_trial_rolls_back_without_caching_unproven_width(tmp_path):
    from optimization.gpu.autotune import ProxyBatchTuner, proxy_batches, record_replay_chunk
    proxy = SimpleNamespace(checkpoint_contract={}, needed_metrics=[],
                            max_dispatch_candidate_bars=1000000)
    proxy.batch_tuner = ProxyBatchTuner(
        proxy, mode="refresh", hardware={"device": "test"}, context={}, cache_dir=tmp_path,
    )
    proxy.batch_tuner._headroom = lambda: True
    batches = proxy_batches(proxy, list(range(1024)), 512)
    next(batches)
    for _ in range(128):
        record_replay_chunk(512, 4096, 2538000, 2)
    next(batches)
    batch = next(iter(proxy.batch_tuner.controllers.values()))
    assert batch.width == 256 and batch.baseline is not None
    batches.close()
    assert batch.baseline is not None  # Same class can continue on its next call.
    assert proxy.batch_tuner.controller(512, 1024, None) is batch
    assert batch.baseline is not None
    proxy.batch_tuner.controller(512, 2048, None)
    assert batch.width == 512 and batch.baseline is None
    cached = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert cached['batch_size'] == 512


@pytest.mark.parametrize('long_work', [False, True])
def test_shrink_trial_counts_queue_caused_admission_idle_gaps(tmp_path, long_work):
    item = controller(tmp_path)
    item.warmed = True
    item.direction = -1
    window(item)
    assert item.limit == 8 and item.baseline[0] == 16
    if long_work:
        # Establish an expensive baseline with the same 4-worker service rate.
        item.baseline = (16, 4 / 120)
        for _ in range(4):
            item.record(120, 0, 10120, 10240, epoch=item.epoch,
                        admission_stall=(10000, 10120))
        item.update(0)
    else:
        # Same service time as the baseline, but each admission wave now needs
        # another GPU pass after the shallow queue has drained.
        for index in range(60):
            if index % 8 == 0:
                stall = (item.tick[0], item.tick[0] + 100)
                item.tick[0] += 100
            start = item.tick[0]
            item.tick[0] += 2
            item.record(2, 0, start, item.tick[0], epoch=item.epoch,
                        admission_stall=stall)
            item.update(0)
    assert item.baseline is None and item.limit == 16 and item.cooldown == 3
    assert item.cache.read(item.key, 'max_pending_exact', 8, 32) == 16


def test_overlapping_admission_stalls_count_once_and_do_not_manufacture_evidence(tmp_path):
    item = controller(tmp_path)
    for _ in range(4):
        item.record(30, 0, 10000, 10030, epoch=item.epoch,
                    admission_stall=(0, 10000))
    item.update(0)
    assert not item.warmed  # Large idle gap cannot replace 120 active seconds.
    item.reset(0)
    item.warmed = True
    for _ in range(4):
        item.record(120, 0, 10120, 10240, epoch=item.epoch,
                    admission_stall=(10000, 10120))
    item.update(0)
    assert item.baseline[1] == pytest.approx(4 / 240)  # Not four times the stall.


@pytest.mark.parametrize('allowed,expected', [([0, 1, 2, 3], 2), ([0, 2, 4, 6], 4)])
def test_affinity_counts_allowed_package_core_pairs(tmp_path, monkeypatch, allowed, expected):
    root = tmp_path / 'cpu'
    for cpu in range(8):
        topology = root / f'cpu{cpu}/topology'
        topology.mkdir(parents=True)
        (topology / 'physical_package_id').write_text('0')
        (topology / 'core_id').write_text(str(cpu // 2))
    assert tune._affinity_physical_cores(allowed, 8, 16, root) == expected
    monkeypatch.setattr(tune.psutil, 'cpu_count', lambda logical=True: 16 if logical else 8)
    monkeypatch.setattr(tune.psutil, 'Process', lambda: SimpleNamespace(
        cpu_affinity=lambda: allowed, memory_info=lambda: SimpleNamespace(rss=0)))
    monkeypatch.setattr(tune.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1 << 40))
    monkeypatch.setattr(tune, '_cgroup_limits', lambda: (None, None))
    original = tune._affinity_physical_cores
    monkeypatch.setattr(tune, '_affinity_physical_cores',
                        lambda ids, physical, logical: original(ids, physical, logical, root))
    assert tune.resource_snapshot()['cores'] == expected
    assert tune.initial_workers(None, 4, SimpleNamespace(), mode='auto') == max(1, expected - 1)


def test_affinity_missing_topology_uses_conservative_smt_ratio(tmp_path):
    assert tune._affinity_physical_cores([0, 1, 2, 3], 8, 16, tmp_path) == 2
    assert tune._affinity_physical_cores([0, 1, 2, 3], 8, 8, tmp_path) == 4
    assert tune._affinity_physical_cores([0], 8, 16, tmp_path) == 1


def worker_controller(tmp_path, monkeypatch, *, workers=2, ceiling=4, mode="auto"):
    monkeypatch.setattr(
        tune, "resource_snapshot", lambda: {"available": 8 * 1024 * tune.MIB}
    )
    return tune.ExactWorkerController(
        workers,
        ceiling,
        [],
        per_worker=512 * tune.MIB,
        mode=mode,
        hardware={"implementation": "test"},
        cache_dir=tmp_path,
    )


def worker_window(item, seconds=10, *, gap=0):
    count = tune.WINDOW if seconds < 30 else max(4, 2 * item.workers)
    origin = getattr(item, "test_tick", 0) + gap
    for i in range(count):
        began = origin + (i // item.workers) * seconds
        item.record(seconds, 0, began, began + seconds, epoch=item.epoch)
    item.test_tick = origin + ((count + item.workers - 1) // item.workers) * seconds
    item.update()


def test_worker_trial_measures_parallel_capacity_and_drains_before_application(
    tmp_path, monkeypatch
):
    item = worker_controller(tmp_path, monkeypatch)
    worker_window(item)
    assert item.warmed and item.target == 2
    worker_window(item, gap=10000)  # unrelated GPU-only pauses excluded
    assert item.target == 3 and item.workers == 2 and item.baseline[0] == 2
    epoch = item.epoch
    item.record(10, 0, 0, 10, epoch=epoch)  # late old-pool jobs while draining
    assert not item.samples
    item.applied()
    assert item.workers == 3 and item.epoch > epoch
    worker_window(item)  # replacement pool cold window
    assert item.baseline is not None
    worker_window(item)
    assert item.baseline is None and item.target == 3
    assert item.cache.read(item.key, "exact_workers", 1, 4) == 3


def test_worker_regression_rolls_back_and_memory_blocks_growth(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    worker_window(item)
    worker_window(item)
    item.applied()
    worker_window(item, seconds=20)
    worker_window(item, seconds=20)
    assert item.target == 2 and item.workers == 3 and item.cooldown == 3
    item.applied()
    assert item.workers == 2
    monkeypatch.setattr(tune, "resource_snapshot", lambda: {"available": 0})
    item.warmed = True
    item.cooldown = 0
    item.direction = 1
    worker_window(item)
    assert item.target == 2 and item.baseline is None


def test_worker_evidence_is_bounded_epoch_scoped_and_coordinated(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    for epoch, duration in ((-1, 10), (0, float("nan")), (0, 0), (0, -1)):
        item.record(duration, 0, 0, 10, epoch=epoch)
    assert not item.samples
    for i in range(100):
        item.record(0.01, 0, i * 0.01, (i + 1) * 0.01, epoch=item.epoch)
    assert len(item.samples) == 24
    item.update()
    assert not item.warmed
    item.allow_trial = lambda: False
    item.update()
    epoch = item.epoch
    item.update()
    assert item.epoch == epoch and not item.samples  # no repeated invalidation
    item.allow_trial = lambda: True
    worker_window(item)
    assert item.warmed


def test_worker_expensive_evidence_and_cache_refresh(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    worker_window(item, seconds=60)
    assert item.warmed
    worker_window(item, seconds=60)
    assert item.target == 3
    item.applied()
    worker_window(item, seconds=60)
    worker_window(item, seconds=60)
    assert item.target == 3 and item.baseline is None
    assert worker_controller(tmp_path, monkeypatch).workers == 3
    assert worker_controller(tmp_path, monkeypatch, mode="refresh").workers == 2
    monkeypatch.setattr(tune, "resource_snapshot", lambda: {"available": 0})
    # Cache cannot authorize an unsafe growth allocation.
    assert (
        tune.ExactWorkerController(
            2,
            4,
            [],
            per_worker=512 * tune.MIB,
            hardware={"implementation": "test"},
            cache_dir=tmp_path,
        ).workers
        == 2
    )


def test_worker_trial_includes_queue_backpressure_idle_gaps(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    worker_window(item)
    worker_window(item)
    item.applied()
    worker_window(item)
    origin = item.test_tick + 1000
    for i in range(24):
        began = origin + (i // item.workers) * 10
        item.record(
            10,
            0,
            began,
            began + 10,
            epoch=item.epoch,
            admission_stall=(origin - 300, origin),
        )
    item.update()
    assert item.target == 2  # A faster pool cannot hide queue-induced starvation.


def test_worker_invalid_stall_is_not_evidence(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    for stall in ((2, 1), (0, 11), (float("nan"), 1)):
        item.record(10, 0, 0, 10, epoch=item.epoch, admission_stall=stall)
    assert not item.samples and item.work_seconds == 0


def test_worker_cache_reuses_page_noise_but_guards_full_startup_pool(
    tmp_path, monkeypatch
):
    memory = [8 * 1024 * tune.MIB]
    monkeypatch.setattr(
        tune, "resource_snapshot", lambda: {"available": memory[0], "cores": 8}
    )

    def make(estimate):
        return tune.ExactWorkerController(
            2,
            7,
            [],
            per_worker=estimate,
            hardware={"implementation": "fixture"},
            cache_dir=tmp_path,
        )

    first = make(700 * tune.MIB + 4096)
    first.cache.write(first.key, "exact_workers", 4, 1.0, 120)
    noisy = make(700 * tune.MIB + 12288)
    assert noisy.key == first.key and noisy.workers == 4
    assert make(900 * tune.MIB).key != first.key
    memory[0] = 4 * 1024 * tune.MIB
    constrained = make(700 * tune.MIB + 12288)
    assert constrained.can_grow(
        4
    )  # Incremental growth would fit once two workers exist.
    assert not constrained.can_grow(4, startup=True)
    assert constrained.workers == 2


def test_cached_ceiling_starts_with_shrink_direction(tmp_path, monkeypatch):
    item = worker_controller(tmp_path, monkeypatch)
    item.cache.write(item.key, "exact_workers", 4, 1.0, 120)
    cached = worker_controller(tmp_path, monkeypatch)
    assert cached.workers == 4 and cached.direction == -1
