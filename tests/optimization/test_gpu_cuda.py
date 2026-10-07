"""CUDA launch contract, independent of the shared strategy regression suite."""

import sys
import inspect
import importlib
from contextlib import contextmanager
from copy import deepcopy
from threading import get_ident
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.cuda_kernel import cuda_source
from optimization.gpu.runtime import gpu_device


def test_runtime_selects_cuda_without_mps():
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert gpu_device(torch) == "cuda"
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="GPU optimization requires"):
        gpu_device(torch)


@pytest.mark.parametrize(
    "source", ["kernel void missing_index() {}", "kernel void k(uint i [[other]]) {}"]
)
def test_unknown_shader_dialect_is_rejected(source):
    with pytest.raises(ValueError, match="Unsupported GPU shader"):
        cuda_source(source)


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    return torch, CudaShaderLibrary


@pytest.mark.parametrize("count", [1, 63, 64, 65, 129])
def test_cuda_dispatch_bounds_and_current_stream(cuda, count):
    torch, library_cls = cuda
    library = library_cls("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void transform(constant float* input, device float* output,
                              uint i [[thread_position_in_grid]]) {
            output[i] = fma(input[i], 2.0f, 1.0f);
        }
    """)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.arange(count + 64, device="cuda", dtype=torch.float32) * 3
        output = torch.full_like(values, -1)
        library.transform(values, output, threads=(count, 1, 1))
        actual = output.cpu().numpy()
    np.testing.assert_array_equal(
        actual[:count], np.arange(count, dtype=np.float32) * 6 + 1
    )
    np.testing.assert_array_equal(actual[count:], -np.ones(64))
    for invalid in [0, 2**32]:
        with pytest.raises(ValueError, match="dispatch must contain"):
            library.transform(values, output, threads=invalid)
    with pytest.raises(ValueError, match="one-dimensional thread groups"):
        library.transform(values, output, threads=count, group_size=(64, 2, 1))
    with pytest.raises(ValueError, match="contiguous"):
        library.transform(values[::2], output, threads=count)
    with pytest.raises(ValueError, match="contiguous"):
        library.transform(values.cpu(), output, threads=count)


def test_cuda_vector_and_bitcast_contract(cuda):
    torch, library_cls = cuda
    library = library_cls("""
        #include <metal_stdlib>
        using namespace metal;
        kernel void check(device float* output, uint i [[thread_position_in_grid]]) {
            float3 value = clamp(2.0f / (float3(1.0f, 3.0f, 7.0f) + 1.0f), 0.0f, 1.0f);
            value = fma(value, float3(8.0f) - float3(4.0f), float3(4.0f));
            output[0] = value.x; output[1] = value.y; output[2] = value.z;
            output[3] = as_type<float>(as_type<uint>(1.0f) - 1u);
            output[4] = INFINITY;
            output[5] = NAN;
        }
    """)
    output = torch.empty(6, device="cuda", dtype=torch.float32)
    library.check(output, threads=1)
    np.testing.assert_array_equal(output[:3].cpu().numpy(), [8, 6, 5])
    assert output[3].item() == np.nextafter(np.float32(1), np.float32(0))
    assert torch.isinf(output[4])
    assert torch.isnan(output[5])


def test_shader_buffer_annotations_preserve_argument_order():
    source = """kernel void copy(constant float* x [[buffer(0)]],
        device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
        y[i] = x[i];
    }"""
    assert "[[" not in cuda_source(source)
    with pytest.raises(ValueError, match="positional argument order"):
        cuda_source(source.replace("buffer(1)", "buffer(0)"))
    with pytest.raises(ValueError, match="positional argument order"):
        cuda_source(
            source.replace(" [[buffer(0)]]", "").replace("buffer(1)", "buffer(0)")
        )


def test_checkpoint_keeps_mps_compatibility_and_identifies_cuda(monkeypatch):
    import sys
    from optimization.gpu.runtime import checkpoint_runtime

    runtime = SimpleNamespace(
        __version__="test-torch",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda: (8, 6)
        ),
    )
    assert checkpoint_runtime(runtime) == {}
    runtime.backends.mps.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="test-cupy"))
    assert checkpoint_runtime(runtime) == {
        "cuda_runtime": {
            "kernel_contract": 1,
            "torch": "test-torch",
            "cupy": "test-cupy",
            "compute_capability": [8, 6],
        }
    }


def test_cuda_constant_reference_buffer(cuda):
    torch, library_cls = cuda
    library = library_cls("""
        kernel void scale(constant int& factor [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
            output[i] = float(factor) * float(i + 1);
        }
    """)
    factor = torch.tensor([3], dtype=torch.int32, device="cuda")
    output = torch.empty(5, device="cuda")
    for value in [factor, 3]:
        library.scale(value, output, threads=5)
        np.testing.assert_array_equal(output.cpu().numpy(), [3, 6, 9, 12, 15])


@pytest.mark.parametrize("capacity", [1, 2, 8, 64])
def test_cuda_coin_capacity_specializes_only_the_declared_limit(capacity):
    source = """constant int MAX_COINS = 64;
        kernel void size(device int* output, uint i [[thread_position_in_grid]]) {
            output[i] = MAX_COINS;
        }"""
    assert f"constexpr int MAX_COINS = {capacity};" in cuda_source(
        source, coin_capacity=capacity
    )
    assert "constexpr int MAX_COINS = 64;" in cuda_source(source)


@pytest.mark.parametrize("capacity", [0, -1, 65, 8.5, True])
def test_cuda_coin_capacity_rejects_invalid_limits(capacity):
    with pytest.raises(ValueError, match="coin capacity"):
        cuda_source("constant int MAX_COINS = 64;", coin_capacity=capacity)


@pytest.mark.parametrize("source", ["", "constant int MAX_COINS = 64;" * 2])
def test_cuda_coin_capacity_requires_one_known_declaration(source):
    with pytest.raises(ValueError, match="one MAX_COINS declaration"):
        cuda_source(source, coin_capacity=8)


def test_mps_compilation_does_not_apply_cuda_coin_specialization(monkeypatch):
    import sys
    from optimization.gpu.runtime import compile_shader

    sources = []
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(compile_shader=lambda source: sources.append(source)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    source = "constant int MAX_COINS = 64;"
    compile_shader(source, cuda_coin_capacity=8)
    assert sources == [source]


@pytest.fixture
def source_kernel():
    """Source-only checks run without Torch and cannot poison later device tests."""
    from optimization import gpu

    missing = object()
    name = "optimization.gpu.mps_kernel"
    old_torch = sys.modules.get("torch", missing)
    old_kernel = sys.modules.pop(name, missing)
    old_attribute = getattr(gpu, "mps_kernel", missing)
    sys.modules["torch"] = SimpleNamespace()
    try:
        yield importlib.import_module(name)
    finally:
        for module_name, previous in (("torch", old_torch), (name, old_kernel)):
            if previous is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        if old_attribute is missing:
            delattr(gpu, "mps_kernel")
        else:
            gpu.mps_kernel = old_attribute


def test_disabled_hsl_specialization_requires_explicit_shader_guard(source_kernel):
    """Only guarded multicoin sources may opt into the compact HSL state."""
    _with_hsl_disabled = source_kernel._with_hsl_disabled
    _with_hsl_features = source_kernel._with_hsl_features

    guarded = (
        "#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED\n"
        "#if PASSIVBOT_HSL_DISABLED\nint compact;\n#endif\n"
    )
    assert _with_hsl_disabled(guarded, False) == guarded
    compact = _with_hsl_disabled(guarded, True)
    assert compact == ("#define PASSIVBOT_HSL_DISABLED 1\n" + guarded)
    assert "#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 0" not in compact
    assert (
        _with_hsl_features(
            compact,
            ema_tail_enabled=False,
            raw_drawdown_enabled=False,
            raw_tail_enabled=False,
        )
        == compact
    )
    with pytest.raises(RuntimeError, match="disabled-HSL feature guard"):
        _with_hsl_disabled("kernel void unguarded() {}", True)


def test_disabled_hsl_specialization_excludes_fused_layout(source_kernel):
    """The compact one-side HSL arrays must never back the fused kernel."""
    MpsEmaAnchorMulticoinFusedRunner = source_kernel.MpsEmaAnchorMulticoinFusedRunner

    assert MpsEmaAnchorMulticoinFusedRunner.hsl_disabled_specialization is False


@pytest.mark.parametrize(
    "enabled,modes,label,coin_enabled,supported,expected",
    [
        ([0, 0], [0, 1], "EMA", False, True, True),
        ([0, 0.5], [1, 1], "EMA", False, True, True),
        ([0, 1], [0, 0], "EMA", False, True, False),
        ([0, np.nan], [0, 0], "EMA", False, True, False),
        ([0, 0], [0, 2], "EMA", False, True, False),
        ([0, 0], [0, 2.1], "EMA", False, True, False),
        ([0, 0], [0, np.nan], "EMA", False, True, False),
        ([0, 0], [0, 0], "EMA", True, True, False),
        ([0, 0], [0, 0], "EMA", False, False, False),
        ([0, 0], [0, 0], "Trailing Martingale", False, True, False),
    ],
)
def test_disabled_hsl_selection_preserves_diagnostic_topology(
    source_kernel, enabled, modes, label, coin_enabled, supported, expected
):
    keys = source_kernel.EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    matrix = np.zeros((2, len(keys)), dtype=np.float32)
    matrix[:, keys.index("hsl_enabled")] = enabled
    matrix[:, keys.index("hsl_signal_mode")] = modes
    runner = SimpleNamespace(
        coin_override_label=label,
        coin_hsl_may_enable=coin_enabled,
        hsl_disabled_specialization=supported,
    )
    assert source_kernel.MpsEmaAnchorMulticoinRunner._use_disabled_hsl_specialization(
        runner, matrix
    ) is expected


def test_disabled_hsl_source_removes_hsl_portfolio_scans():
    """The ordinary disabled-HSL candle path has no pre-fill or HSL update scan."""
    source = (
        Path(__file__).parents[2]
        / "passivbot-rust/src/gpu/mps_ema_anchor_multicoin_long.metal"
    ).read_text()

    assert (
        "#if PASSIVBOT_HSL_DISABLED\n        float hsl_equity_before_fills = 0.0f;"
        in source
    )
    assert "#if !PASSIVBOT_HSL_DISABLED\n        if (can_generate && alive" in source
    assert "#if !PASSIVBOT_HSL_DISABLED\n    bind_hsl_multicoin_hsl" in source


@pytest.mark.parametrize("case", ["ema-multicoin-overhead", "tm-multicoin-overhead"])
@pytest.mark.parametrize("coins", [1, 2, 3, 5, 9, 17, 33, 64])
def test_cuda_coin_capacity_matches_full_capacity_outputs(cuda, case, coins):
    torch, _library_cls = cuda
    from tools.gpu_proxy_benchmark import _build_case

    def evaluate(full_capacity):
        proxy, candidates, *_ = _build_case(
            case,
            candidates=4,
            dispatch_batch_size=4,
            single_bars=256,
            multicoin_bars=256,
            coins=coins,
            seed=7,
        )
        runner = proxy.runners["long"]
        expected_capacity = 1 << (coins - 1).bit_length()
        assert runner.cuda_coin_capacity == expected_capacity
        loader, arguments = runner._library_cache_call()
        bound = inspect.signature(loader).bind(*arguments)
        assert bound.arguments["cuda_coin_capacity"] == expected_capacity
        if full_capacity:
            runner.cuda_coin_capacity = 64
        if case.startswith("tm-"):
            # Exercise the variant-specific replay-state ABI across temporal chunks.
            runner.max_dispatch_candidate_bars = 4 * coins * 31
        output = runner.run(proxy._parameter_matrix(candidates, "long"))
        return {
            key: value.cpu().numpy().copy()
            for key, value in output.items()
            if isinstance(value, torch.Tensor)
        }

    specialized = evaluate(False)
    baseline = evaluate(True)
    assert specialized.keys() == baseline.keys()
    for key in baseline:
        np.testing.assert_array_equal(specialized[key], baseline[key], err_msg=key)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("market_orders_allowed", [False, True])
def test_cuda_shared_account_one_coin_reuses_isolated_candidate_state(
    cuda, monkeypatch, strategy, sides, market_orders_allowed
):
    """One coin uses the same account kernel, including fused directional state."""
    from optimization.gpu.executor import BacktestRequest, GpuBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    import backtest

    def forbidden(*_args, **_kwargs):
        raise AssertionError("shared-account replay must not call a CPU backtest")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "1", "--bars", "512",
    ]))
    config["backtest"]["market_orders_allowed"] = market_orders_allowed
    replay = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange="binance", batch_size=2, needed_metrics=set(DEFAULT_METRICS),
    )
    if strategy == "trailing_martingale":
        runners = [replay.fused_runner] if replay.fused_runner is not None else replay.runners.values()
        for runner in runners:
            runner.max_dispatch_candidate_bars = 2 * 31
    candidates = [{}, {f"{side}_total_wallet_exposure_limit": 0.5 for side in replay.sides}, {}]
    expected = [replay.evaluate([candidate])[0] for candidate in candidates]
    assert expected[0]["fills_per_day"] > 0
    with GpuBacktestService(batch_size=2, max_pending=3) as service:
        service.register_dataset("one-coin", replay)
        for repeat in range(2):
            futures = [service.submit(BacktestRequest(f"{repeat}:{i}", "one-coin", candidate))
                       for i, candidate in enumerate(candidates)]
            actual = [future.result(timeout=60).metrics for future in futures]
            for baseline, row in zip(expected, actual):
                assert baseline.keys() == row.keys()
                for name in baseline:
                    np.testing.assert_array_equal(row[name], baseline[name], err_msg=name)


@pytest.mark.parametrize(
    "case",
    [
        "ema-single-long",
        "tm-single-long",
        "ema-multicoin-overhead",
        "tm-multicoin-overhead",
    ],
)
def test_cuda_async_service_reuses_replay_and_preserves_metrics(cuda, case):
    """Cold compilation and repeated dispatches work on the owning service thread."""
    from concurrent.futures import as_completed
    from optimization.gpu.executor import BacktestRequest, GpuBacktestService
    from tools.gpu_proxy_benchmark import _build_case

    proxy, candidates, *_ = _build_case(
        case,
        candidates=7,
        dispatch_batch_size=7,
        single_bars=256,
        multicoin_bars=256,
        coins=3,
        seed=11,
    )
    with GpuBacktestService(batch_size=2, max_batch_delay=0.001) as service:
        service.register_dataset("fixture", proxy)
        observations = []
        for repeat in range(2):
            futures = [
                service.submit(BacktestRequest(f"{repeat}:{i}", "fixture", candidate))
                for i, candidate in enumerate(candidates)
            ]
            completed = [future.result() for future in as_completed(futures, timeout=60)]
            assert {result.request_id for result in completed} == {
                f"{repeat}:{i}" for i in range(len(candidates))
            }
            assert all(result.dataset_id == "fixture" for result in completed)
            rows = {result.request_id: result.metrics for result in completed}
            observations.append([rows[f"{repeat}:{i}"] for i in range(len(candidates))])

    # Different batch shapes and a different calling thread cannot change simulation.
    expected = proxy.evaluate(candidates)
    for actual in observations:
        for row, baseline in zip(actual, expected):
            assert row.keys() == baseline.keys()
            for metric, value in baseline.items():
                np.testing.assert_allclose(
                    row[metric], value, rtol=1e-6, atol=1e-7, err_msg=metric
                )


def test_cuda_disabled_hsl_specialization_matches_full_hsl_state(cuda):
    """Removing unreachable per-coin HSL state must preserve every GPU output."""
    torch, _library_cls = cuda
    from tools.gpu_proxy_benchmark import _build_case

    def evaluate(compact_hsl):
        proxy, candidates, *_ = _build_case(
            "ema-multicoin-overhead",
            candidates=4,
            dispatch_batch_size=4,
            single_bars=128,
            multicoin_bars=128,
            coins=5,
            seed=11,
        )
        runner = proxy.runners["long"]
        if not compact_hsl:
            runner.coin_hsl_may_enable = True
        output = runner.run(proxy._parameter_matrix(candidates, "long"))
        assert runner.dispatch_hsl_disabled is compact_hsl
        return {
            key: value.cpu().numpy().copy()
            for key, value in output.items()
            if isinstance(value, torch.Tensor)
        }

    compact = evaluate(True)
    baseline = evaluate(False)
    assert compact.keys() == baseline.keys()
    for key in baseline:
        np.testing.assert_array_equal(compact[key], baseline[key], err_msg=key)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("coins", [1, 3])
def test_cuda_factory_owns_replay_lifetime_and_matches_direct_metrics(cuda, monkeypatch, strategy, coins):
    from optimization.gpu.executor import BacktestRequest, GpuBacktestService
    from optimization.gpu.service import MpsMulticoinProxy, MpsSingleCoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    import backtest

    def cpu_backtest_forbidden(*_args, **_kwargs):
        raise AssertionError("GPU execution must not call a CPU backtest")

    monkeypatch.setattr(backtest, "execute_backtest", cpu_backtest_forbidden)
    monkeypatch.setattr(backtest, "run_backtest", cpu_backtest_forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", str(coins), "--bars", "128",
    ]))
    cls = MpsSingleCoinProxy if coins == 1 else MpsMulticoinProxy

    def construct():
        return cls(config=deepcopy(config), hlcvs=candles, mss=markets, btc=btc,
                   timestamps=timestamps, exchange="binance", batch_size=2,
                   needed_metrics=set(DEFAULT_METRICS))

    events, references = [], []

    @contextmanager
    def factory():
        events.append(("enter", get_ident()))
        replay = construct()
        references.append(weakref.ref(replay))
        try:
            yield replay
        finally:
            events.append(("exit", get_ident()))

    main_thread = get_ident()
    with GpuBacktestService(batch_size=2) as service:
        service.register_dataset_factory("fixture", factory)
        assert not events
        for repeat in range(2):
            futures = [service.submit(BacktestRequest(f"{repeat}:{i}", "fixture", {})) for i in range(3)]
            metrics = [future.result(timeout=60).metrics for future in futures]
            assert len(events) == 1
    gc.collect()
    assert events == [("enter", service._thread.ident), ("exit", service._thread.ident)]
    assert service._thread.ident != main_thread
    assert references[0]() is None
    expected = construct().evaluate([{}])[0]
    for row in metrics:
        for name, value in expected.items():
            np.testing.assert_allclose(row[name], value, rtol=1e-6, atol=1e-7, err_msg=name)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("coins", [1, 3])
def test_cuda_prepared_service_reuses_packing_and_bounds_resident_scenarios(cuda, monkeypatch, strategy, coins):
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu import service as replay_module
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    from shared_arrays import SharedArrayManager
    import backtest

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepared service must not call a CPU backtest")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", str(coins), "--bars", "512",
    ]))
    manager = SharedArrayManager()
    instances, prepared, subset_builds = [], [], []
    original_class = replay_module.MpsMulticoinProxy
    original_builder = replay_module.build_mps_multicoin_data
    from optimization.gpu.residency import CudaSuiteResidency
    original_subset = CudaSuiteResidency.prepare_coin_subset

    def construct(**kwargs):
        instance = original_class(**kwargs)
        instances.append(weakref.ref(instance))
        return instance
    def pack(*args, **kwargs):
        prepared.append((get_ident(), kwargs.get("spill_dir")))
        return original_builder(*args, **kwargs)
    def subset(self, values, indices):
        subset_builds.append(tuple(indices))
        return original_subset(self, values, indices)
    monkeypatch.setattr(replay_module, "MpsMulticoinProxy", construct)
    monkeypatch.setattr(replay_module, "build_mps_multicoin_data", pack)
    monkeypatch.setattr(CudaSuiteResidency, "prepare_coin_subset", subset)
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        common = dict(hlcvs=specs[0], btc=specs[1], timestamps=specs[2], exchange="binance",
                      candle_coins=config["backtest"]["coins"]["binance"],
                      markets=markets, metrics=DEFAULT_METRICS)
        base = PreparedGpuDataset(config=config, **common)
        changed_config = deepcopy(config)
        changed_config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.5
        changed = PreparedGpuDataset(config=changed_config, **common)
        indices = (0,) if coins == 1 else (0, 2)
        subset_config = deepcopy(config)
        subset_config["backtest"]["coins"]["binance"] = [
            config["backtest"]["coins"]["binance"][index] for index in indices
        ]
        sliced = PreparedGpuDataset(config=subset_config, coin_indices=indices, **common)
        with CudaBacktestService(batch_size=2, max_pending=4) as service:
            for name, dataset in (("base", base), ("changed", changed), ("subset", sliced), ("subset-copy", sliced)):
                service.register_dataset(name, dataset)
            assert instances == prepared == subset_builds == []
            rows = {}
            for i, name in enumerate(("base", "changed", "subset", "subset-copy", "base", "changed", "subset")):
                row = service.submit(BacktestRequest(str(i), name, {})).result(timeout=60)
                assert row.liquidated is False
                if name in rows:
                    assert row.metrics == rows[name]
                rows[name] = row.metrics
                alive = [ref() for ref in instances if ref() is not None]
                owners = [instance for instance in alive if instance.fused_runner is not None or instance.runners]
                assert len(owners) == 1
                assert owners[0]._cuda_residency is service._residency
                assert len(service._residency._entries) == (2 if coins > 1 and "subset" in rows else 1)
                del alive, owners
            assert rows["subset-copy"] == rows["subset"]
            owner_thread = service._executor._thread.ident
        gc.collect()
        assert all(reference() is None for reference in instances)
        assert all(thread == owner_thread and isinstance(path, Path) and not path.exists()
                   for thread, path in prepared)
        assert len(prepared) == (1 if coins == 1 else 2)
        assert subset_builds == ([] if coins == 1 else [(0, 2)])
        assert not service._prepared_cache and not service._subset_cache
        assert service._residency is None
        for name, candidate_config, values in (
            ("base", config, candles), ("changed", changed_config, candles),
            ("subset", subset_config, candles[:, indices, :]),
        ):
            expected = original_class(
                config=candidate_config, hlcvs=values, mss=markets, btc=btc, timestamps=timestamps,
                exchange="binance", batch_size=2, needed_metrics=DEFAULT_METRICS,
            ).evaluate([{}])[0]
            assert rows[name].keys() == expected.keys()
            for metric in expected:
                np.testing.assert_array_equal(rows[name][metric], expected[metric], err_msg=metric)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("failure", ["attachment", "interrupt"])
def test_cuda_prepared_service_cleans_up_after_setup_or_interrupt(cuda, monkeypatch, failure):
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.residency import CudaSuiteResidency
    import optimization.gpu.datasets as datasets
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    from shared_arrays import SharedArrayManager, SharedArraySpec
    import backtest

    def forbidden(*_args, **_kwargs):
        pytest.fail("failed GPU requests must not fall back to CPU backtests")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--coins", "3", "--bars", "128",
    ]))
    config["backtest"]["coins"]["binance"] = ["COIN00", "COIN02"]
    closed, directories = [], []
    original_attach = datasets.attach_shared_array
    original_directory = CudaSuiteResidency._spill_directory
    def tracked_attach(spec):
        attachment = original_attach(spec)
        original_close = attachment.close
        def close():
            closed.append(spec.name)
            original_close()
        attachment.close = close
        return attachment
    def tracked_directory(residency):
        path = original_directory(residency)
        directories.append(path)
        return path
    monkeypatch.setattr(datasets, "attach_shared_array", tracked_attach)
    monkeypatch.setattr(CudaSuiteResidency, "_spill_directory", tracked_directory)
    def interrupt():
        raise KeyboardInterrupt("GPU optimization interrupted")
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        if failure == "attachment":
            specs[1] = SharedArraySpec("missing-" + specs[1].name, specs[1].shape, specs[1].dtype)
        dataset = PreparedGpuDataset(
            config=config, markets=markets, hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
            candle_coins=["COIN00", "COIN01", "COIN02"], coin_indices=(0, 2),
            exchange="binance", metrics=DEFAULT_METRICS,
        )
        with CudaBacktestService(interrupt_check=interrupt if failure == "interrupt" else None) as service:
            service.register_dataset("scenario", dataset)
            expected = KeyboardInterrupt if failure == "interrupt" else FileNotFoundError
            with pytest.raises(expected):
                service.submit(BacktestRequest("candidate", "scenario", {})).result(timeout=60)
            with pytest.raises(RuntimeError, match="service failed"):
                service.submit(BacktestRequest("next", "scenario", {}))
        expected_closed = [spec.name for spec in reversed(specs)] if failure == "interrupt" else [specs[0].name]
        assert closed == expected_closed
        assert bool(directories) == (failure == "interrupt")
        assert all(not path.exists() for path in directories)
        assert service._residency is None
        assert not service._prepared_cache and not service._subset_cache
    finally:
        manager.cleanup()


def test_cuda_prepared_service_preserves_actual_liquidation_per_candidate(cuda, monkeypatch):
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.executor import BacktestRequest
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    from shared_arrays import SharedArrayManager
    import backtest

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--coins", "1", "--bars", "128",
    ]))
    candles[:, :, :3] = [101.0, 99.0, 100.0]
    candles[80:, :, :3] *= 0.01
    bot = config["bot"]["long"]
    bot["risk"]["total_wallet_exposure_limit"] = 10.0
    strategy = bot["strategy"]["trailing_martingale"]
    strategy["entry"].update(initial_qty_pct=1.0, ema_gate_mode="disabled")
    strategy["close"]["threshold_base_pct"] = 1.0
    parameters = [{}, {"long_total_wallet_exposure_limit": 0.1, "long_entry_initial_qty_pct": 0.01}]
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", candle_coins=["COIN00"],
            hlcvs=specs[0], btc=specs[1], timestamps=specs[2], metrics=DEFAULT_METRICS,
        )
        def forbidden(*_args, **_kwargs):
            pytest.fail("native service must not run a CPU backtest")
        with monkeypatch.context() as guard:
            guard.setattr(backtest, "execute_backtest", forbidden)
            guard.setattr(backtest, "run_backtest", forbidden)
            with CudaBacktestService(batch_size=2, max_batch_delay=0.05) as service:
                service.register_dataset("crash", dataset)
                futures = [service.submit(BacktestRequest(str(i), "crash", row)) for i, row in enumerate(parameters)]
                results = [future.result(timeout=60) for future in futures]
        assert [result.liquidated for result in results] == [True, False]
        for index, result in enumerate(results):
            candidate = deepcopy(config)
            if index:
                candidate["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.1
                candidate["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.01
            payload = backtest.build_backtest_payload(
                candles, markets, candidate, "binance", btc, timestamps, metrics_only=True,
            )
            analysis = backtest.execute_backtest(payload, candidate)[2]
            assert analysis["liquidated"] is result.liquidated
    finally:
        manager.cleanup()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cuda_first_coin_patch_does_not_replace_global_defaults(cuda, strategy):
    from optimization.gpu.service import MpsMulticoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    if strategy == "trailing_martingale":
        parameter = "entry_initial_qty_pct"
        global_value = config["bot"]["long"]["strategy"][strategy]["entry"]["initial_qty_pct"]
        patch = {"entry": {"initial_qty_pct": 0.5}}
    else:
        parameter = "base_qty_pct"
        global_value = config["bot"]["long"]["strategy"][strategy][parameter]
        patch = {parameter: 0.5}
    config["coin_overrides"] = {"COIN00": {"bot": {"long": {"strategy": {strategy: patch}}}}}
    replay = MpsMulticoinProxy(config=config, mss=markets, exchange="binance",
                              hlcvs=candles, btc=btc, timestamps=timestamps,
                              needed_metrics=DEFAULT_METRICS, batch_size=2)
    assert replay.base_params["long"][parameter] == global_value
    values = replay._parameter_matrix([{}], "long")
    assert values[0, replay.param_keys.index(parameter)] == pytest.approx(global_value)
    contract = replay.coin_override_contract["exact_overrides_by_side"]["long"]
    assert contract[0]["bot"]["long"]["strategy"][strategy] == patch
    assert contract[1:] == [{}, {}]


def test_cuda_async_completions_drive_canonical_suite_scoring_without_cpu_backtests(cuda, monkeypatch):
    from concurrent.futures import as_completed
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.executor import BacktestRequest
    from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
    from optimize import Evaluator, SuiteEvaluator
    from metrics_schema import build_scenario_metrics
    from suite_runner import ScenarioResult, SuiteScenario
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    from shared_arrays import SharedArrayManager
    import backtest

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "512",
    ]))
    config["optimize"]["bounds"] = {"long_entry_initial_qty_pct": [0.01, 0.03]}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max", "scenario": "base"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min", "scenario": None, "aggregate": "max"},
    ]
    config["optimize"]["limits"] = [
        {"metric": "backtest_completion_ratio", "penalize_if": "less_than", "value": 0.99},
    ]
    requested_metrics = (*DEFAULT_METRICS, "backtest_completion_ratio")
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        base = Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets}, config)
        contexts = [SimpleNamespace(label=label, exchanges=["binance"]) for label in ("base", "stress")]
        suite = SuiteEvaluator(base, contexts, {"default": "mean"})
        scorer = CanonicalResultScorer(suite)
        scenario_configs = {"base": config, "stress": deepcopy(config)}
        scenario_configs["stress"]["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 0.5
        def forbidden(*_args, **_kwargs):
            pytest.fail("native completion scoring must not run CPU simulations")
        monkeypatch.setattr(backtest, "execute_backtest", forbidden)
        monkeypatch.setattr(backtest, "run_backtest", forbidden)
        monkeypatch.setattr(base, "evaluate", forbidden)
        monkeypatch.setattr(suite, "evaluate", forbidden)
        candidate_values = [0.03, 0.02, 0.01]
        parameters = [{"long_entry_initial_qty_pct": value} for value in candidate_values]
        collectors, final, received = {}, {}, {}
        with CudaBacktestService(batch_size=2, max_pending=8) as service:
            for label, scenario_config in scenario_configs.items():
                service.register_dataset(label, PreparedGpuDataset(
                    config=scenario_config, markets=markets, exchange="binance",
                    candle_coins=config["backtest"]["coins"]["binance"],
                    hlcvs=specs[0], btc=specs[1], timestamps=specs[2], metrics=requested_metrics,
                ))
            requests = []
            owners = {}
            for index, candidate in enumerate(parameters):
                graph = [ResultSlot(f"{index}:{label}", label, label, "binance", requested_metrics)
                         for label in scenario_configs]
                vector = [candidate_values[index] if key == "long_entry_initial_qty_pct" else bound.low
                          for (key, _path), bound in zip(base.key_paths, base.bounds, strict=True)]
                collectors[str(index)] = CandidateEvaluation(str(index), vector, graph, scorer)
                for slot in graph:
                    owners[slot.request_id] = str(index)
                    requests.append(BacktestRequest(slot.request_id, slot.dataset_id, candidate))
            # Scenario locality is an execution choice; fan-in still identifies each candidate.
            futures = [service.submit(request) for request in sorted(requests, key=lambda item: item.dataset_id)]
            for future in as_completed(futures, timeout=60):
                row = future.result()
                received[row.request_id] = row
                complete = collectors[owners[row.request_id]].add_result(row)
                if complete is not None:
                    final[complete.candidate_id] = complete.require_full()
        assert set(final) == {"0", "1", "2"}
        for candidate_id, payload in final.items():
            rows = []
            for label in scenario_configs:
                result = received[f"{candidate_id}:{label}"]
                per_exchange = {"binance": dict(result.metrics, liquidated=result.liquidated)}
                rows.append(ScenarioResult(SuiteScenario(label, None, None, None, None), per_exchange,
                                           build_scenario_metrics(per_exchange), 0.0, None))
            expected = suite.score_scenario_results(rows)
            assert payload["fitness"] == expected["objectives"]
            assert payload["metrics"]["suite_metrics"] == expected["suite_metrics"]
            assert payload["constraint_violation"] == expected["constraint_violation"]
    finally:
        manager.cleanup()


def test_cuda_cpu_session_prepares_effective_suite_requests_and_reuses_duplicates(cuda, monkeypatch):
    import time
    import backtest
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
    from optimization.native_session import NativeEvaluationSession
    from optimize import Evaluator, SuiteEvaluator, config_to_individual
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "512",
    ]))
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["bounds"][f"{side}_entry_initial_qty_pct"] = [0.01, 0.05]
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.025,
    }
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
    ]
    config["optimize"]["limits"] = []
    scenario_configs = {label: deepcopy(config) for label in ("base", "stress")}
    for label, value in scenario_configs.items():
        value["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.025
        value["bot"]["short"] = deepcopy(value["bot"]["long"])
        if label == "stress":
            value["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.03

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU orchestration must never run a CPU simulation")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    reference = {label: MpsMulticoinProxy(
        config=value, mss=markets, exchange="binance", hlcvs=candles, btc=btc,
        timestamps=timestamps, needed_metrics=DEFAULT_METRICS, batch_size=1,
    ).evaluate_results([{}])[0] for label, value in scenario_configs.items()}
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        base = Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets}, config)
        contexts = [SimpleNamespace(
            label=label, exchanges=["binance"], config=deepcopy(config), msss={"binance": markets},
            overrides={"bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.03}
            if label == "stress" else {},
        ) for label in scenario_configs]
        suite = SuiteEvaluator(base, contexts, {"default": "mean"})
        monkeypatch.setattr(base, "evaluate", forbidden)
        monkeypatch.setattr(suite, "evaluate", forbidden)
        bindings = [ScenarioBinding(label, label, PreparedGpuDataset(
            config=value, markets=markets, exchange="binance", hlcvs=specs[0], btc=specs[1],
            timestamps=specs[2], candle_coins=config["backtest"]["coins"]["binance"], metrics=DEFAULT_METRICS,
        )) for label, value in scenario_configs.items()]
        planner = NativeCandidatePlanner(suite, bindings)
        vector = config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape)
        vectors = [[qty if key.endswith("entry_initial_qty_pct") else original
                    for (key, _path), original in zip(base.key_paths, vector, strict=True)]
                   for qty in (0.01, 0.05, 0.03)]
        plans = [planner.prepare(str(index), values) for index, values in enumerate(vectors)]
        assert len({plan.effective_key for plan in plans}) == 1
        submitted = []
        with CudaBacktestService(batch_size=2, max_pending=2) as service:
            for binding in bindings:
                service.register_dataset(binding.dataset_id, binding.dataset)
            original_submit = service.submit
            def submit(request):
                future = original_submit(request)
                submitted.append((request, future))
                return future
            monkeypatch.setattr(service, "submit", submit)
            session = NativeEvaluationSession(service, planner.scorer, max_candidates=2)
            for plan in plans[:2]:
                session.admit(plan)
            final = {}
            deadline = time.monotonic() + 60
            while len(final) < 2 and time.monotonic() < deadline:
                final.update((row.candidate_id, row.require_full()) for row in session.poll(timeout=0.05))
            assert set(final) == {"0", "1"}
            session.admit(plans[2])
            final["2"] = session.poll()[0].require_full()
            assert session.active_candidate_ids == ()
            assert session.pending_request_count == 0
            assert len(submitted) == 2
        for request, future in submitted:
            row = future.result()
            assert row.metrics == pytest.approx(reference[request.dataset_id].metrics, rel=1e-7, abs=1e-9)
            assert row.liquidated == reference[request.dataset_id].liquidated
        assert final["0"]["fitness"] == final["1"]["fitness"] == final["2"]["fitness"]
        assert [final[str(i)]["evaluation_vector"] for i in range(3)] == [list(plan.vector) for plan in plans]
    finally:
        manager.cleanup()


@pytest.mark.parametrize("pending_queries", [0, 2])
def test_cuda_completion_wait_yields_until_ready(monkeypatch, pending_queries):
    import sys
    from optimization.gpu import runtime

    calls = []
    ready = iter([False] * pending_queries + [True])
    event = SimpleNamespace(
        record=lambda: calls.append("record"),
        query=lambda: next(ready),
    )
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            Event=lambda: event,
            synchronize=lambda: calls.append("device_sync"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    ticks = iter([0.0] + [0.5] * pending_queries)
    monkeypatch.setattr(
        runtime,
        "time",
        SimpleNamespace(
            perf_counter=lambda: next(ticks),
            sleep=lambda seconds: calls.append(seconds),
        ),
    )
    runtime.synchronize()
    assert calls == ["record", *([0.001] * pending_queries), "device_sync"]


def test_cuda_completion_error_propagates(monkeypatch):
    import sys
    from optimization.gpu import runtime

    def failed_query():
        raise RuntimeError("asynchronous kernel failure")

    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            Event=lambda: SimpleNamespace(record=lambda: None, query=failed_query),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="asynchronous kernel failure"):
        runtime.synchronize()


def test_mps_completion_keeps_existing_synchronization(monkeypatch):
    import sys
    from optimization.gpu import runtime

    calls = []
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps_sync")),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    runtime.wait_for_cuda_stream()
    runtime.synchronize()
    assert calls == ["mps_sync"]


def test_cuda_completion_waits_for_current_stream(cuda):
    torch, _library_cls = cuda
    from optimization.gpu.runtime import wait_for_cuda_stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output = torch.empty(16, device="cuda")
        torch.cuda._sleep(10_000_000)
        output.fill_(42)
        wait_for_cuda_stream()
        assert stream.query()
    np.testing.assert_array_equal(output.cpu().numpy(), np.full(16, 42))


def test_cuda_synchronize_still_waits_for_other_streams(cuda):
    torch, _library_cls = cuda
    from optimization.gpu.runtime import synchronize

    other = torch.cuda.Stream()
    with torch.cuda.stream(other):
        torch.cuda._sleep(10_000_000)
    synchronize()
    assert other.query()


def test_cuda_completion_keeps_short_wait_active(monkeypatch):
    import sys
    from optimization.gpu import runtime

    calls = []
    ready = iter([False, False, True])
    ticks = iter([0.0, 0.1, 0.49])
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            Event=lambda: SimpleNamespace(
                record=lambda: None, query=lambda: next(ready)
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        runtime,
        "time",
        SimpleNamespace(
            perf_counter=lambda: next(ticks),
            sleep=lambda seconds: calls.append(seconds),
        ),
    )
    runtime.wait_for_cuda_stream()
    assert calls == []


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("weighted_volume", [False, True])
def test_tm_unchunked_dispatch_keeps_apple_launch_options(monkeypatch, device, weighted_volume):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel

    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: device)
    runner = SimpleNamespace(
        **{
            name: object()
            for name in (
                "bars",
                "fill_ticks",
                "touch_ticks",
                "touch_nearest_ticks",
                "touch_min_qty_bits",
                "touch_min_qty_relation",
                "hour_log_ranges",
                "coin_settings",
                "coin_overrides",
                "settings",
            )
        },
        btc_prices_enabled=False,
        equity_balance_diff_enabled=False,
        entry_interval_enabled=False,
        recovery_distribution_enabled=False,
        weighted_volume_enabled=weighted_volume,
        weighted_equity_cols=0,
        max_dispatch_candidate_bars=None,
        hsl_capacity=0,
        unstuck_pnl_capacity=0,
    )
    calls = []
    library = SimpleNamespace(
        passivbot_trailing_martingale_multicoin=lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )
    )
    buffers = [object() for _ in range(12)]
    mps_kernel.MpsTrailingMartingaleMulticoinRunner._dispatch(
        runner, library, *buffers, weighted_equity_samples=None, batch_size=65
    )
    assert len(calls) == 1
    assert len(calls[0][0]) == 17 + int(weighted_volume)
    if weighted_volume:
        assert calls[0][0][-1] is buffers[-1]
    expected = {"threads": (65, 1, 1)}
    if device == "cuda":
        expected["group_size"] = (32, 1, 1)
    assert calls[0][1] == expected


@pytest.mark.parametrize("count", [33, 129])
@pytest.mark.parametrize("coins", [2, 8])
def test_cuda_tm_unchunked_blocks_preserve_raw_outputs(cuda, monkeypatch, count, coins):
    torch, library_cls = cuda
    from tools.gpu_proxy_benchmark import _build_case

    proxy, candidates, *_ = _build_case(
        "tm-multicoin-overhead",
        candidates=count,
        dispatch_batch_size=count,
        single_bars=256,
        multicoin_bars=256,
        coins=coins,
        seed=7,
    )
    runner = proxy.runners["long"]
    runner.max_dispatch_candidate_bars = None
    matrix = proxy._parameter_matrix(candidates, "long")

    def evaluate():
        return {
            key: value.cpu().numpy().copy()
            for key, value in runner.run(matrix).items()
            if isinstance(value, torch.Tensor)
        }

    actual = evaluate()
    assert actual
    original = library_cls.__getattr__

    def old_block(self, name):
        launch = original(self, name)
        if name != "passivbot_trailing_martingale_multicoin":
            return launch

        def launch_64(*args, **kwargs):
            kwargs["group_size"] = (64, 1, 1)
            return launch(*args, **kwargs)

        return launch_64

    monkeypatch.setattr(library_cls, "__getattr__", old_block)
    baseline = evaluate()
    assert actual.keys() == baseline.keys()
    for key in baseline:
        np.testing.assert_array_equal(actual[key], baseline[key], err_msg=key)


@pytest.mark.parametrize("capacity", [None, 4])
def test_cuda_multicoin_relation_bytes_preserve_signed_values(cuda, capacity):
    torch, library_cls = cuda
    source = """
        constant int MAX_COINS = 64;
        kernel void relations(constant int* touch_min_qty_relation,
                              device int* output,
                              uint i [[thread_position_in_grid]]) {
            output[i] = touch_min_qty_relation[i];
        }
    """
    assert "const signed char* touch_min_qty_relation" in cuda_source(
        source, coin_capacity=4
    )
    assert "const signed char* touch_min_qty_relation" in cuda_source(source)
    library = library_cls(source, coin_capacity=capacity)
    values = torch.tensor([-1, 0, 1] * 50, device="cuda", dtype=torch.int8)
    output = torch.empty(len(values), device="cuda", dtype=torch.int32)
    library.relations(values, output, threads=len(values))
    np.testing.assert_array_equal(output.cpu().numpy(), values.cpu().numpy())
    with pytest.raises(ValueError, match="signed int8"):
        library.relations(values.to(torch.int32), output, threads=len(values))


@pytest.mark.parametrize("coins", [3, 28, 64])
@pytest.mark.parametrize("count", [1023, 1025])
def test_cuda_temporal_batch_increase_preserves_outputs_and_partial_tail(
    cuda, coins, count
):
    torch, _ = cuda
    from tools.gpu_proxy_benchmark import _build_case

    proxy, candidates, *_ = _build_case(
        "tm-multicoin-overhead",
        candidates=count,
        dispatch_batch_size=1024,
        single_bars=256,
        multicoin_bars=1513,
        coins=coins,
        seed=7,
    )
    runner = proxy.runners["long"]
    runner.max_dispatch_candidate_bars = 1024 * coins * 47
    matrix = proxy._parameter_matrix(candidates, "long")
    ends = np.resize(np.asarray([1, 123, 1513], dtype=np.int32), count)

    def evaluate(batch):
        chunks = []
        for offset in range(0, count, batch):
            params = matrix[offset : offset + batch]
            raw = runner.run(params, end_steps=ends[offset : offset + batch])
            chunks.append(
                {
                    key: value.cpu().numpy().copy()
                    for key, value in raw.items()
                    if isinstance(value, torch.Tensor)
                }
            )
        return {
            key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
        }

    baseline = evaluate(512)
    assert baseline
    # Different candidate partitions also change history chunk boundaries. Reuse
    # the runner to cover buffer reallocation and the one-candidate final batch.
    for _ in range(2):
        actual = evaluate(1024)
        assert actual.keys() == baseline.keys()
        for key in baseline:
            np.testing.assert_array_equal(actual[key], baseline[key], err_msg=key)


def test_mps_coin_capacity_is_explicit_and_preserves_other_source(monkeypatch):
    import sys
    from optimization.gpu.runtime import compile_shader

    sources = []
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(compile_shader=lambda source: sources.append(source)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    source = "constant int MAX_COINS = 64; // 64 remains elsewhere"
    compile_shader(source, mps_coin_capacity=4)
    assert sources == ["constant int MAX_COINS = 4; // 64 remains elsewhere"]
