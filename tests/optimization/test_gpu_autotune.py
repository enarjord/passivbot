"""Execution adaptation must preserve work, bounds, and search policy."""

from types import SimpleNamespace

import pytest

from optimization.gpu import autotune as tune


def window(controller, seconds=2.0):
    width = controller.width
    for _ in range(tune.WINDOW + (width not in controller.seen)):
        controller.observe(width, seconds)


def proxy():
    return SimpleNamespace(
        checkpoint_contract={
            "hlcvs": {"shape": [1000, 2, 4], "sha256": "data-a"},
            "timestamps": {"sha256": "time-a"},
            "backtest": {"first_timestamp_ms": 123, "global_warmup_bars": 100},
            "base_params": {"hsl_enabled": 0},
        },
        needed_metrics={"adg"},
        max_dispatch_candidate_bars=1000000,
    )


def tuner(tmp_path, *, mode="auto", item=None):
    result = tune.ProxyBatchTuner(
        item or proxy(), mode=mode, hardware={"device": "cuda"}, context={}, cache_dir=tmp_path
    )
    result._headroom = lambda: True
    return result


def test_waits_for_full_window_and_minimum_duration():
    controller = tune.BatchController(1024, 128)
    for _ in range(100):
        controller.observe(127, 5.0)  # partial tails never count
    assert not controller.samples
    window(controller, 0.1)
    assert controller.width == 128
    assert len(controller.samples) == tune.WINDOW
    controller.observe(128, 30.0)
    assert controller.width == 256


def test_noise_does_not_trigger_early_and_failed_trial_rolls_back():
    saved = []
    controller = tune.BatchController(1024, 128, save=lambda *args: saved.append(args))
    controller.observe(128, 1000)  # cold compile excluded
    for index in range(tune.WINDOW):
        controller.observe(128, 50 if index == 12 else 2)
    assert controller.width == 256
    assert saved[0][1] == 64  # rolling median rejects isolated stall
    window(controller, 5)  # larger batch is slower per candidate
    assert controller.width == 128
    assert controller.cooldown == 3
    for _ in range(3):
        window(controller)
        assert controller.width == 128
    window(controller)
    assert controller.width == 64


def test_successful_growth_and_smaller_plateau_are_accepted():
    controller = tune.BatchController(256, 128)
    window(controller)
    window(controller, 3)  # 85.3/s vs 64/s
    assert controller.width == 256
    assert controller.baseline is None
    controller.cooldown = 0
    controller.direction = -1
    window(controller, 4)
    assert controller.width == 128
    window(controller, 2.01)  # nearly equal throughput with half the allocation
    assert controller.width == 128
    assert controller.baseline is None


def test_ceiling_memory_pressure_and_invalid_timings():
    controller = tune.BatchController(150, 128, can_grow=lambda: False)
    for value in [0, -1, float("nan"), float("inf")]:
        controller.observe(128, value)
    assert not controller.samples
    window(controller)
    assert controller.width == 128
    controller.cooldown = 0
    controller.can_grow = lambda: True
    window(controller)
    assert controller.width == 150


def test_real_work_generator_preserves_order_and_no_sample_after_failure():
    controller = tune.BatchController(8, 2)
    tick = [0.0]
    item = SimpleNamespace(batch_tuner=SimpleNamespace(controller=lambda *args: controller))
    rows = list(range(17))
    seen = []
    for start, chunk in tune.proxy_batches(item, rows, 8, clock=lambda: tick[0]):
        assert start == len(seen)
        seen.extend(chunk)
        tick[0] += 2.0
        if start == 2:
            controller.width = 4
    assert seen == rows
    samples = len(controller.samples)
    work = tune.proxy_batches(item, rows, 8, clock=lambda: tick[0])
    next(work)
    work.close()
    assert len(controller.samples) == samples


def test_fixed_batches_do_not_read_clock():
    def forbidden():
        raise AssertionError("fixed sizing does not collect timings")

    assert list(tune.proxy_batches(SimpleNamespace(), list(range(5)), 3, clock=forbidden)) == [
        (0, [0, 1, 2]),
        (3, [3, 4]),
    ]


def test_cache_reuse_refresh_and_classes(tmp_path):
    first = tuner(tmp_path)
    controller = first.controller(512, 1024, None)
    controller.save(256, 100.0, 60.0)
    assert tuner(tmp_path).controller(512, 1024, None).width == 256
    assert tuner(tmp_path, mode="refresh").controller(512, 1024, None).width == 128
    assert tuner(tmp_path).controller(512, 2048, None).width == 128
    assert tuner(tmp_path).controller(512, 1024, 500).width == 128
    changed = proxy()
    changed.checkpoint_contract["base_params"]["hsl_enabled"] = 1
    assert tuner(tmp_path, item=changed).controller(512, 1024, None).width == 128
    # Equal-length new data/dates reuse execution evidence, not trading results.
    changed = proxy()
    changed.checkpoint_contract["hlcvs"]["sha256"] = "data-b"
    changed.checkpoint_contract["timestamps"]["sha256"] = "time-b"
    changed.checkpoint_contract["backtest"]["first_timestamp_ms"] = 456
    assert tuner(tmp_path, item=changed).controller(512, 1024, None).width == 256
    changed.checkpoint_contract["hlcvs"]["shape"][0] *= 2
    assert tuner(tmp_path, item=changed).controller(512, 1024, None).width == 128


def test_cache_limits_and_corrupt_or_unwritable_cache(tmp_path, caplog):
    item = tuner(tmp_path)
    item.controller(512, 1024, None).save(256, 100, 60)
    cache = next(tmp_path.glob("*.json"))
    cache.write_text('{"version": 1, "batch_size": 999999}')
    assert tuner(tmp_path).controller(512, 1024, None).width == 128
    assert "cache unavailable" in caplog.text
    cache.write_text("not json")
    assert tuner(tmp_path).controller(512, 1024, None).width == 128
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    offline = tuner(blocked)
    controller = offline.controller(512, 1024, None)
    window(controller)
    assert controller.width == 256  # cache I/O failure cannot disable useful work
    for demand in range(1, 20):
        item.controller(512, demand, None)
    assert len(item.controllers) == 8


def test_numeric_and_off_do_not_inspect_hardware(monkeypatch):
    monkeypatch.setattr(tune, "hardware_identity", lambda *args: pytest.fail("hardware queried"))
    tune.configure_batch_tuning(
        [], {"optimize": {"gpu": {"batch_size": 64}}}, {"tuning_mode": "auto"}
    )
    tune.configure_batch_tuning([], {}, {"tuning_mode": "off"})


@pytest.mark.parametrize("value", [None, "auto", " AUTO "])
def test_auto_gpu_options_and_canonical_cli(value):
    import argparse
    from config_utils import get_template_config, add_arguments_recursively
    from optimization.backends.gpu_backend import _resolve_options

    config = get_template_config()
    config["optimize"]["gpu"].update(
        batch_size=value, population_size=value, max_dispatch_candidate_bars=value
    )
    options = _resolve_options(config)
    assert options["batch_size"] == 4096
    assert options["population_size"] == 1024
    assert options["tuning_mode"] == "auto"
    # Parsing must allow auto even when the input file contains explicit numbers.
    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, {"optimize": {"gpu": {"batch_size": 64}}})
    args = parser.parse_args(["--optimize.gpu.batch_size", "auto"])
    assert vars(args)["optimize.gpu.batch_size"] == "auto"


def test_bad_mode_fails_before_gpu_work():
    from config_utils import get_template_config
    from optimization.backends.gpu_backend import _resolve_options

    config = get_template_config()
    config["optimize"]["gpu"]["tuning_mode"] = "fastest"
    with pytest.raises(ValueError, match="tuning_mode"):
        _resolve_options(config)


@pytest.mark.parametrize(
    "case",
    [
        "ema-single-long",
        "tm-single-long",
        "tm-single-long-hsl",
        "ema-multicoin-overhead",
        "tm-multicoin-overhead",
    ],
)
def test_gpu_changed_batch_width_preserves_every_metric(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("GPU unavailable")
    from tools.gpu_proxy_benchmark import _build_case

    item, candidates, *_ = _build_case(
        case,
        candidates=17,
        dispatch_batch_size=16,
        single_bars=256,
        multicoin_bars=256,
        coins=3,
        seed=7,
    )
    if case == "tm-multicoin-overhead":
        # Force several temporal chunks per candidate replay.
        item.runners["long"].max_dispatch_candidate_bars = 2048
    expected = item.evaluate(candidates)

    # Change sizes between real batches, including an irregular final remainder.
    class Changing:
        width = 2

        def observe(self, count, seconds):
            self.width = min(8, self.width * 2)

    controller = Changing()
    item.batch_tuner = SimpleNamespace(controller=lambda *args: controller)
    actual = item.evaluate(candidates)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert left.keys() == right.keys()
        for key in left:
            # Host metric reductions can differ at float64 roundoff across shapes.
            assert left[key] == pytest.approx(right[key], rel=1e-12, abs=1e-12, nan_ok=True), key


def test_adaptive_progress_reports_actual_chunks(monkeypatch, caplog):
    from optimization.gpu.service import _new_gpu_dispatch_progress, _update_gpu_dispatch_progress

    ticks = iter([0.0, 61.0, 122.0])
    monkeypatch.setattr("optimization.gpu.service.time.monotonic", lambda: next(ticks))
    progress = _new_gpu_dispatch_progress(16, 16, adaptive=True)
    with caplog.at_level("INFO"):
        _update_gpu_dispatch_progress(progress, completed_candidates=2, strategy="test")
        _update_gpu_dispatch_progress(progress, completed_candidates=16, strategy="test")
    assert "batches_done=1 scenario_evals=2/16" in caplog.records[0].message
    assert "batches_done=2 scenario_evals=16/16" in caplog.records[1].message


def test_corrupt_cache_is_repaired_and_low_memory_ignores_large_cached_width(tmp_path):
    item = tuner(tmp_path)
    controller = item.controller(512, 1024, None)
    controller.save(256, 100, 60)
    constrained = tuner(tmp_path)
    constrained._headroom = lambda: False
    assert constrained.controller(512, 1024, None).width == 128
    next(tmp_path.glob("*.json")).write_text("broken")
    repair = tuner(tmp_path).controller(512, 1024, None)
    repair.save(64, 80, 60)
    assert tuner(tmp_path).controller(512, 1024, None).width == 64


def test_hardware_identity_distinguishes_nvidia_and_rejects_rocm(monkeypatch):
    import sys

    monkeypatch.setattr(tune, "_implementation_identity", lambda: "kernel-v1")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="test"))
    props = SimpleNamespace(
        name="test GPU", total_memory=8 * 1024**3, multi_processor_count=40, major=8, minor=6
    )
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test", hip=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda index: props,
        ),
    )
    identity = tune.hardware_identity(torch)
    assert identity["name"] == "test GPU"
    assert identity["memory"] == 8 * 1024**3
    assert identity["processors"] == 40
    torch.version.hip = "test"
    with pytest.raises(RuntimeError, match="NVIDIA"):
        tune.hardware_identity(torch)


def test_memory_growth_checks_backend_limits(tmp_path):
    item = tuner(tmp_path)
    del item._headroom
    item.proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (1, 1024**3)))
    assert not item._headroom()
    item.proxy._torch.cuda.mem_get_info = lambda: (2 * 1024**3, 4 * 1024**3)
    assert item._headroom()
    item.hardware["device"] = "mps"
    item.proxy._torch = SimpleNamespace(
        mps=SimpleNamespace(driver_allocated_memory=lambda: 80, recommended_max_memory=lambda: 100)
    )
    assert not item._headroom()


def test_cache_is_bounded_and_preserves_unrelated_files(tmp_path):
    item = tuner(tmp_path)
    unrelated = tmp_path / "user-notes.json"
    unrelated.write_text("keep")
    for demand in range(1, 131):
        controller = item.controller(512, demand, None)
        controller.save(controller.width, 1.0, 30.0)
    assert len(list(tmp_path.glob("*.json"))) == 129
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("batch_config", [{}, {"batch_size": None}, {"batch_size": "auto"}])
def test_mps_automatic_startup_uses_supported_device_name_api(monkeypatch, batch_config):
    from config_utils import get_template_config
    from optimization.backends.gpu_backend import _resolve_options

    monkeypatch.setattr(tune, "_implementation_identity", lambda: "kernel-v1")
    calls = []

    def device_name():
        calls.append("get_name")
        return "Apple test GPU"

    # PyTorch 2.13 exports get_name from torch.backends.mps. CUDA probing must
    # not be required when the MPS backend is available.
    torch = SimpleNamespace(
        __version__="2.13.0",
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True, get_name=device_name)
        ),
        mps=SimpleNamespace(recommended_max_memory=lambda: 8 * 1024**3),
    )
    item = proxy()
    item._torch = torch
    config = get_template_config()
    config["optimize"]["gpu"].pop("batch_size")
    config["optimize"]["gpu"].update(batch_config)
    tune.configure_batch_tuning([item], config, _resolve_options(config))
    assert calls == ["get_name"]
    assert item.batch_tuner.hardware["device"] == "mps"
    assert item.batch_tuner.hardware["name"] == "Apple test GPU"
    assert item.batch_tuner.hardware["memory"] == 8 * 1024**3


def test_real_mps_automatic_startup_device_identity():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple MPS unavailable")
    from config_utils import get_template_config
    from optimization.backends.gpu_backend import _resolve_options

    item = proxy()
    item._torch = torch
    config = get_template_config()
    tune.configure_batch_tuning([item], config, _resolve_options(config))
    hardware = item.batch_tuner.hardware
    assert hardware["device"] == "mps"
    assert hardware["name"] == torch.backends.mps.get_name()
    assert hardware["name"]
    assert hardware["memory"] == torch.mps.recommended_max_memory()
    assert hardware["memory"] > 0
