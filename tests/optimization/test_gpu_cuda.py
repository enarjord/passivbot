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


@pytest.mark.parametrize("case", ["ema-multicoin-overhead", "tm-multicoin-overhead"])
@pytest.mark.parametrize("coins", [2, 3, 5, 9, 17, 33, 64])
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
def test_tm_unchunked_dispatch_keeps_apple_launch_options(monkeypatch, device):
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
    mps_kernel.MpsTrailingMartingaleMulticoinRunner._dispatch(
        runner, library, *[object() for _ in range(11)], batch_size=65
    )
    assert len(calls) == 1
    assert len(calls[0][0]) == 17
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
