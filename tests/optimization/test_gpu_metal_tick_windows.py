"""Bounded immutable tick transport keeps replay indices and bytes unchanged."""

import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TICK_WINDOW_METRICS = (
    "cagr_fraction_strategy_equity", "cagr_fraction_strategy_equity_ew", "drawdown_max_fraction_strategy_equity",
    "ulcer_index_fraction_strategy_equity", "pain_index_fraction_strategy_equity",
    "backtest_completion_fraction", "peak_recovery_max_days_strategy_equity",
    "hsl_red_time_fraction", "position_held_time_weighted_mean_hours", "gross_turnover_ratio_per_day",
)


@pytest.fixture
def helpers():
    # These transport/source contracts need neither an accelerator nor Torch.
    source = ROOT / "src/optimization/gpu/mps_kernel.py"
    names = {"_METAL_WINDOWED_TICKS", "_with_metal_tick_windows",
             "_metal_tick_window_paths", "_metal_tick_window_bounds",
             "_metal_tick_window_arguments"}
    parsed = ast.parse(source.read_text())
    nodes = [node for node in parsed.body
             if ((isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in names
                             for target in node.targets)))]
    namespace = dict(Path=Path, re=re, np=np, contextmanager=contextmanager,
                     logging=logging)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def tick_data(tmp_path, *, n=19, coins=3):
    data = dict(n=n, n_coins=coins, stream_tick_inputs=True)
    for ordinal, (name, width) in enumerate((
        ("fill_ticks", 2), ("touch_ticks", 2), ("touch_nearest_ticks", 1),
        ("touch_min_qty_bits", 1), ("touch_min_qty_relation", 1),
    )):
        shape = (n, coins) + ((2,) if width == 2 else ())
        values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + ordinal * 1000
        data[name] = tmp_path / f"{name}.npy"
        np.save(data[name], values, allow_pickle=False)
    return data


def test_source_windows_preserve_public_pointer_abi_and_all_absolute_indices(helpers):
    source = (ROOT / "passivbot-rust/src/gpu/mps_trailing_martingale_multicoin.metal").read_text()
    result = helpers["_with_metal_tick_windows"](source)
    assert "return values[absolute_index - first_element];" in result
    assert "constant float* bars" in result
    assert "constant int* market_execution_ticks" in result
    # All loads retain the original absolute integer expression.
    original_loads = re.findall(
        r"\b(?:fill_ticks|touch_ticks|touch_nearest_ticks|touch_min_qty_bits|touch_min_qty_relation)\[[^\n]+",
        source,
    )
    assert original_loads
    for load in original_loads:
        assert load in result
    for name, width in helpers["_METAL_WINDOWED_TICKS"]:
        assert result.count(f"constant int* window_{name}") == 3
        assert result.count(f"replay_range[2] * sizes[2] * {width}}};") >= 3
        assert f"constant int* {name}" not in result
    assert result.count("struct PassivbotTickWindow") == 1
    for signature in re.findall(r"kernel void \w+\((.*?)\) \{", result, flags=re.DOTALL):
        assert "PassivbotTickWindow" not in signature


def test_source_declaration_or_wrapper_drift_fails_before_compilation(helpers):
    source = (ROOT / "passivbot-rust/src/gpu/mps_trailing_martingale_multicoin.metal").read_text()
    with pytest.raises(RuntimeError, match="declaration contract"):
        helpers["_with_metal_tick_windows"](source.replace("constant int* fill_ticks", "constant int* other", 1))
    with pytest.raises(RuntimeError, match="wrapper contract"):
        helpers["_with_metal_tick_windows"](source.replace(
            "kernel void passivbot_trailing_martingale_multicoin(", "kernel void renamed(", 1))


@pytest.mark.parametrize("begin,stop,available,expected", [
    (1, 1, 3, (0, 2)), (1, 5, 19, (0, 6)),
    (5, 11, 19, (4, 12)), (11, 18, 19, (10, 19)),
])
def test_windows_include_previous_touch_and_next_fill_rows(helpers, begin, stop, available, expected):
    assert helpers["_metal_tick_window_bounds"](begin, stop, available) == expected


@pytest.mark.parametrize("begin,stop,available", [(0, 3, 19), (5, 4, 19), (5, 5, 19), (1, 19, 19)])
def test_invalid_windows_are_rejected(helpers, begin, stop, available):
    with pytest.raises(ValueError, match="input window"):
        helpers["_metal_tick_window_bounds"](begin, stop, available)


def test_disk_input_contract_rejects_partial_wrong_dtype_and_unsupported_modes(helpers, tmp_path):
    data = tick_data(tmp_path)
    check = helpers["_metal_tick_window_paths"]
    assert len(check(data, device="mps", supported=True, temporal=True)) == 5
    for kwargs in (dict(device="cuda", supported=True, temporal=True),
                   dict(device="mps", supported=False, temporal=True),
                   dict(device="mps", supported=True, temporal=False)):
        with pytest.raises(ValueError, match="temporal Metal"):
            check(data, **kwargs)
    with pytest.raises(ValueError, match="every tick"):
        check(dict(data, fill_ticks=np.zeros((19, 3, 2), dtype=np.int32)),
              device="mps", supported=True, temporal=True)
    with pytest.raises(ValueError, match="streamed-input contract"):
        check(dict(data, stream_tick_inputs=False), device="mps", supported=True, temporal=True)
    np.save(data["fill_ticks"], np.zeros((19, 3, 2), dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="Invalid streamed GPU tick array: fill_ticks"):
        check(data, device="mps", supported=True, temporal=True)


def test_each_uploaded_window_is_byte_identical_and_lives_through_existing_sync(helpers, tmp_path, monkeypatch):
    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    original_load = np.load
    mappings, uploads, waits = [], [], []

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        mappings.append(array)
        return array

    def as_tensor(array, *, device):
        assert device == "mps"
        uploads.append(array)
        return SimpleNamespace(contiguous=lambda: np.array(array, copy=True))

    wait = lambda: waits.append(1)
    helpers.update(torch=SimpleNamespace(as_tensor=as_tensor,
                                         mps=SimpleNamespace(synchronize=wait)), synchronize=wait)
    monkeypatch.setattr(np, "load", load)
    sentinel = object()
    args = (sentinel, sentinel, *paths.values(), sentinel)
    for begin, stop in ((1, 5), (5, 11), (11, 18)):
        with helpers["_metal_tick_window_arguments"](
            args, paths, begin=begin, stop=stop, available=data["n"],
        ) as (actual, start):
            end = min(data["n"], stop + 1)
            assert start == begin - 1
            assert actual[:2] == args[:2] and actual[-1] is sentinel
            for slot, path in enumerate(paths.values(), start=2):
                expected = original_load(path)[start:end]
                np.testing.assert_array_equal(actual[slot], expected)
            assert all(not mapped._mmap.closed for mapped in mappings[-5:])
            helpers["synchronize"]()  # Production's existing replay boundary.
        assert all(mapped._mmap.closed for mapped in mappings)
    assert len(waits) == 3  # No additional normal-path wait.


def test_partial_upload_failure_drains_queue_and_closes_every_mapping(helpers, tmp_path, monkeypatch):
    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    mappings, waits = [], []
    original_load = np.load

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        mappings.append(array)
        return array

    def as_tensor(array, *, device):
        if len(mappings) == 2:
            raise RuntimeError("upload failed")
        return SimpleNamespace(contiguous=lambda: array)

    wait = lambda: waits.append(1)
    helpers.update(torch=SimpleNamespace(as_tensor=as_tensor,
                                         mps=SimpleNamespace(synchronize=wait)), synchronize=wait)
    monkeypatch.setattr(np, "load", load)
    with pytest.raises(RuntimeError, match="upload failed"):
        with helpers["_metal_tick_window_arguments"](
            (None,) * 8, paths, begin=1, stop=5, available=19,
        ):
            pytest.fail("partial upload must not dispatch")
    assert waits == [1]
    assert all(mapped._mmap.closed for mapped in mappings)


def test_full_resident_arguments_do_not_load_or_synchronize(helpers):
    args = (object(), object())
    with helpers["_metal_tick_window_arguments"](args, {}, begin=1, stop=3, available=4) as actual:
        assert actual == (args, 0)


def test_streamed_pages_are_advised_after_existing_wait_without_an_extra_wait(helpers, tmp_path, monkeypatch):
    from optimization.gpu import residency

    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    waits, advised = [], []
    wait = lambda: waits.append(1)
    helpers.update(torch=SimpleNamespace(as_tensor=lambda array, **kwargs:
        SimpleNamespace(contiguous=lambda: np.array(array, copy=True)),
        mps=SimpleNamespace(synchronize=wait)), synchronize=wait)

    def advise(array, completed_bytes, discarded_bytes=0):
        assert waits == [1]
        assert not array._mmap.closed
        advised.append((array.shape, completed_bytes, discarded_bytes))
        return discarded_bytes

    monkeypatch.setattr(residency, "_discard_completed_upload_pages", advise)
    with helpers["_metal_tick_window_arguments"](
        (None,) * 8, paths, begin=5, stop=11, available=19,
    ):
        helpers["synchronize"]()
    assert len(advised) == 5 and waits == [1]
    for shape, completed_bytes, discarded_bytes in advised:
        assert completed_bytes == 12 * int(np.prod(shape[1:])) * 4
        assert discarded_bytes % residency.mmap.PAGESIZE == 0


def test_failed_window_wait_retains_every_source_until_confirmed_quiescence(
    helpers, tmp_path, monkeypatch, caplog,
):
    from optimization.gpu import residency

    assert not residency._mps_upload_quarantine
    data = tick_data(tmp_path)
    paths = {name: data[name] for name, _ in helpers["_METAL_WINDOWED_TICKS"]}
    mappings = []
    original_load = np.load
    original_error = RuntimeError("window upload failed")

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        mappings.append(array)
        return array

    def as_tensor(array, **kwargs):
        if len(mappings) == 2:
            raise original_error
        return SimpleNamespace(contiguous=lambda: np.array(array, copy=True))

    def failed_wait():
        raise RuntimeError("window queue status unknown")

    fake_torch = SimpleNamespace(as_tensor=as_tensor, mps=SimpleNamespace(synchronize=failed_wait))
    helpers.update(torch=fake_torch)
    monkeypatch.setattr(np, "load", load)
    try:
        with pytest.raises(RuntimeError) as caught:
            with helpers["_metal_tick_window_arguments"](
                (None,) * 8, paths, begin=1, stop=5, available=19,
            ):
                pytest.fail("failed upload cannot dispatch")
        assert caught.value is original_error
        assert len(residency._mps_upload_quarantine) == len(mappings) == 2
        assert all(not array._mmap.closed for array in mappings)
        assert "input-window cleanup synchronization failed" in caplog.text
    finally:
        fake_torch.mps.synchronize = lambda: None
        residency._synchronize_mps_uploads(fake_torch)
    assert not residency._mps_upload_quarantine
    assert all(array._mmap.closed for array in mappings)


def test_metal_residency_uploads_only_retained_inputs_and_admits_resident_bytes(tmp_path, monkeypatch):
    from optimization.gpu.residency import (
        METAL_STREAMED_TICK_FIELDS, MetalSuiteResidency,
        check_mps_invariant_memory, multicoin_invariant_bytes, multicoin_resident_bytes,
    )

    n, coins = 20_000, 2
    full = multicoin_invariant_bytes(n, coins, device="mps")
    resident = multicoin_resident_bytes(n, coins, device="mps", stream_tick_inputs=True)
    assert full == n * coins * 56
    assert resident == (n + 8194) * coins * 28
    assert resident < full
    uploads, waits = [], []

    def upload(array, *, device):
        assert device == "mps"
        uploads.append(Path(array.filename).stem)
        return SimpleNamespace(contiguous=lambda: object())

    fake_torch = SimpleNamespace(
        as_tensor=upload,
        mps=SimpleNamespace(recommended_max_memory=lambda: 4_000_000,
                            driver_allocated_memory=lambda: 0,
                            synchronize=lambda: waits.append(1), empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(MemoryError, match="45%"):
        check_mps_invariant_memory(fake_torch, full)
    manager = MetalSuiteResidency()

    def build(directory):
        data = tick_data(directory, n=n, coins=coins)
        for name, shape in (("bars", (n, coins, 4)),
                            ("market_execution_ticks", (n, coins, 2)),
                            ("hour_log_ranges", (n, coins)),
                            ("coin_settings", (coins, 15))):
            data[name] = directory / f"{name}.npy"
            dtype = np.int32 if name == "market_execution_ticks" else np.float32
            np.save(data[name], np.zeros(shape, dtype=dtype), allow_pickle=False)
        data.update(invariant_bytes=full, resident_bytes=resident)
        return data

    class Owner:
        def __init__(self, data):
            self.data, self.runners, self.fused_runner = data, {}, None

        def _create_runners(self):
            assert all(isinstance(self.data[name], Path) for name in METAL_STREAMED_TICK_FIELDS)
            self.runners["long"] = object()

    try:
        data = manager.prepare("market", build)
        first, second = Owner(data), Owner(data)
        manager.register(first)
        manager.register(second)
        manager.activate(first)
        assert set(uploads) == {"bars", "market_execution_ticks", "hour_log_ranges", "coin_settings"}
        assert first.runners
        manager.activate(second)
        assert not first.runners and second.runners
        assert len(uploads) == 4  # A new scratch owner reuses retained market tensors.
        directory = manager._directory.name
    finally:
        manager.close()
    assert not first.runners and not second.runners
    assert manager._active is None and manager._active_owner is None
    assert not Path(directory).exists()
    assert waits  # Owners are synchronized before their buffers are released.


@pytest.fixture
def metal_runtime():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal required")
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    return torch


def _tick_window_fixture(sides, hsl):
    from tools.gpu_parity import build_parser, fixture_inputs

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--coins", "2", "--sides", sides,
        "--bars", "1537", "--seed", "43", "--hsl", hsl, "--unstuck",
        "--hsl-drawdown-threshold-fraction", ".002", "--hsl-drawdown-fraction-ema-span-minutes", "2.5",
        "--hsl-cooldown-minutes", "120", "--hsl-memory-ema-span-minutes", "1440",
        "--price-shock", "0", "701", ".7", "--price-shock", "1", "911", "1.3",
    ]))
    config, candles, markets, _btc, _timestamps = inputs
    config["backtest"]["ew_half_life_days"] = 1.5
    # The span crosses a UTC-day boundary and many completed hours. The second
    # coin lists later and delists before the replay ends, exercising both halos
    # and distant clamped marks, which deliberately remain full resident.
    markets["COIN01"].update(first_valid_index=103, last_valid_index=1231)
    candles[:103, 1] = np.nan
    candles[1232:, 1] = np.nan
    for side in ("long", "short"):
        bot = config["bot"][side]
        if bot["enabled"]:
            bot["risk"]["max_positions"] = 1
        bot["entry_cooldown"].update(base_duration_minutes=1.5, max_duration_minutes=12.0)
        bot["entry_cooldown"]["weights_minutes"].update(
            adverse_directionality=3.0, exposure_ratio=2.0,
        )
        bot["forager"].update(unilateralness_ema_span_minutes=3.25,
                               volume_ema_span_minutes=7.5,
                               log_range_ema_span_minutes=11.25)
        bot["forager"]["score_weights"]["unilateralness"] = 0.35
        strategy = bot["strategy"]["trailing_martingale"]
        strategy["entry"].update(ema_span_0_minutes=7.5, ema_span_1_minutes=13.25)
        strategy.update(log_range_ema_span_minutes=9.5, log_range_ema_span_1h_candles=2.5)
        bot["unstuck"].update(ema_span_0_minutes=5.5, ema_span_1_minutes=11.25)
    return inputs


def test_tick_window_fixture_admits_requested_metrics_without_device_execution(monkeypatch):
    original_import = builtins.__import__
    def cpu_only(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("host metric/config validation must not import Torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", cpu_only)
    from config.ew import validate_ew_half_life_days
    from config.gpu import validate_gpu_backtest_config

    # Exercise the actual host validator and its complete metric declarations,
    # without importing the unrelated tensor reductions or retaining a fake
    # Torch module that could contaminate later device qualification tests.
    source = ROOT / "src/optimization/gpu/metrics.py"
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "validate_gpu_metric_names":
                nodes.append(node)
                break
            continue
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module in {
            "optimization.gpu.entry_intervals", "optimization.gpu.exponential_equity",
        }:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Assert)):
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    validate_gpu_metric_names = namespace["validate_gpu_metric_names"]

    config, *_ = _tick_window_fixture("both", "coin")
    validate_gpu_backtest_config(config)
    assert validate_ew_half_life_days(config["backtest"]["ew_half_life_days"]) == 1.5
    assert set(validate_gpu_metric_names(TICK_WINDOW_METRICS, native_replay=True,
                                         device="mps")) == set(TICK_WINDOW_METRICS)


def _assert_tick_window_output_exact(torch, expected, actual):
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{key}: {message}")
        elif isinstance(value, dict):
            _assert_tick_window_output_exact(torch, value, other)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=key)
        elif isinstance(value, float):
            assert value.hex() == other.hex(), key
        else:
            assert value == other, key


@pytest.mark.parametrize("sides,hsl,chunk_bars", [
    ("long", "disabled", 17), ("short", "disabled", 61),
    ("both", "disabled", 17), ("long", "coin", 61),
    ("both", "coin", 17), ("short", "portfolio", 17),
    ("both", "portfolio", 61),
])
def test_metal_service_streamed_ticks_match_resident_outputs_and_reset(
    metal_runtime, monkeypatch, sides, hsl, chunk_bars,
):
    torch = metal_runtime
    import backtest
    from optimization.gpu import mps_kernel, service as replay_module
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import NativeBacktestService
    from optimization.gpu import residency
    from tools.gpu_parity import _native_dataset

    def forbidden(*args, **kwargs):
        pytest.fail("Metal tick transport qualification cannot invoke CPU simulations")

    for owner, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                        (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(owner, name, forbidden)
    original_prepare = replay_module._prepared_multicoin_data
    original_run = mps_kernel._MulticoinReplayRunner.run
    mode = {"streamed": False}
    prepared_modes, raw_outputs, dispatches = [], [], []

    def prepare(*args, **kwargs):
        # Force the constrained-memory preference for this small qualification
        # fixture, then disable only transport for the resident reference.
        assert kwargs["stream_tick_inputs"] is True
        if not mode["streamed"]:
            kwargs["stream_tick_inputs"] = False
            kwargs["compact_hourly_ranges"] = False
        result = original_prepare(*args, **kwargs)
        prepared_modes.append(bool(result["stream_tick_inputs"]))
        return result

    def capture(runner, *args, **kwargs):
        result = original_run(runner, *args, **kwargs)
        raw_outputs.append({key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                                 else deepcopy(value)) for key, value in result.items()})
        dispatches.append(deepcopy(runner._last_temporal_dispatch))
        assert bool(runner._streamed_tick_paths) is mode["streamed"]
        assert runner.compact_hourly_ranges is mode["streamed"]
        return result

    monkeypatch.setattr(residency, "prefer_streamed_ticks", lambda *args, **kwargs: True)
    monkeypatch.setattr(replay_module, "_prepared_multicoin_data", prepare)
    monkeypatch.setattr(mps_kernel._MulticoinReplayRunner, "run", capture)
    requested = TICK_WINDOW_METRICS
    sides_count = 2 if sides == "both" else 1
    results = []
    with _native_dataset(_tick_window_fixture(sides, hsl), "binance", requested) as dataset:
        for streamed in (False, True):
            mode["streamed"] = streamed
            with NativeBacktestService(
                batch_size=1, max_pending=1, tuning_mode="off",
                max_dispatch_candidate_bars=chunk_bars * 2 * sides_count,
            ) as worker:
                worker.register_dataset("market", dataset)
                # A second request must reset candidate state and replay windows.
                for cycle in range(2):
                    results.append(worker.submit(BacktestRequest(
                        f"{streamed}-{cycle}", "market", {},
                    )).result(timeout=180))
            assert worker._residency is None and not worker._prepared_cache
    assert prepared_modes == [False, True]
    assert len(raw_outputs) == 4
    for other in raw_outputs[1:]:
        _assert_tick_window_output_exact(torch, raw_outputs[0], other)
    for other in results[1:]:
        assert other.liquidated is results[0].liquidated
        assert {key: float(value).hex() for key, value in other.metrics.items()} == {
            key: float(value).hex() for key, value in results[0].metrics.items()
        }
    assert all(item["dispatch_count"] > 1 for item in dispatches)
    assert all(item["temporal_chunk_bars"] == chunk_bars for item in dispatches)
    assert all(item.get("streamed_tick_inputs", False) is False for item in dispatches[:2])
    assert all(item["streamed_tick_inputs"] is True for item in dispatches[2:])
