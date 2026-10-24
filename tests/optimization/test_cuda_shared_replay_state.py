"""Block-local replay state must never alias candidates or directional sides."""

import numpy as np
import pytest

from optimization.gpu.cuda_kernel import cuda_source, _shared_side_replay


def source(count=128, *, native=True, chunked=True):
    return f"""
#define PASSIVBOT_HSL_FACTUAL_ONLY {int(native)}
#define PASSIVBOT_EMA_MULTICOIN_CHUNKED {int(chunked)}
#ifndef PASSIVBOT_GPU_SIDE_STATE
#define PASSIVBOT_GPU_SIDE_STATE(TYPE, NAME, SLOT) TYPE NAME
#endif
struct State {{ float values[{count}]; }};
kernel void state_probe(device float* output, uint b [[thread_position_in_grid]]) {{
    PASSIVBOT_GPU_SIDE_STATE(State, long_side, 0);
    PASSIVBOT_GPU_SIDE_STATE(State, short_side, 1);
    for (int c = 0; c < {count}; ++c) {{
        long_side.values[c] = float(b * 2 + c);
        short_side.values[c] = float(int(b) * 3 - c);
    }}
    for (int c = 0; c < {count}; ++c) {{
        output[b * {count} * 2 + c] = long_side.values[c];
        output[b * {count} * 2 + {count} + c] = short_side.values[c];
    }}
}}
"""


@pytest.mark.parametrize(
    "native,chunked", [(False, False), (False, True), (True, False), (True, True)]
)
def test_shared_workspace_is_only_selected_for_native_temporal_replay(native, chunked):
    shader = source(native=native, chunked=chunked)
    assert _shared_side_replay(shader) == (native and chunked)
    lowered = cuda_source(shader)
    assert ("NAME##_storage" in lowered) == (native and chunked)
    assert "NAME##_storage" not in cuda_source(shader, shared_replay_state=False)


@pytest.mark.parametrize("count", [128, 6144, 6145])
def test_actual_cuda_side_slots_and_size_fallback(count):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    from optimization.gpu.cuda_kernel import CudaShaderLibrary

    library = CudaShaderLibrary(source(count))
    kernel = library._module.get_function("state_probe")
    shared = kernel.attributes["shared_size_bytes"]
    assert shared == (count * 4 * 2 if count <= 6144 else 0)
    width = 65
    output = torch.full((width * count * 2,), float("nan"), device="cuda")
    library.state_probe(output, threads=width, group_size=(1, 1, 1))
    expected = np.empty((width, 2, count), dtype=np.float32)
    for b in range(width):
        expected[b, 0] = b * 2 + np.arange(count)
        expected[b, 1] = b * 3 - np.arange(count)
    np.testing.assert_array_equal(
        output.cpu().numpy().reshape(expected.shape), expected
    )
    if shared:
        # Reject before any launch can corrupt a different candidate's state.
        for block in (2, 32, 64):
            with pytest.raises(ValueError, match="one candidate per thread block"):
                library.state_probe(output, threads=width, group_size=(block, 1, 1))
    else:
        library.state_probe(output, threads=width, group_size=(64, 1, 1))
        np.testing.assert_array_equal(
            output.cpu().numpy().reshape(expected.shape), expected
        )


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_native_hsl_shared_state_is_bit_identical_to_private_state(
    monkeypatch, strategy, mode, sides
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    from optimization.gpu import cuda_kernel, mps_kernel
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    proxy = make_proxy(
        mode, strategy, sides, minutes=256, coin_count=3, factual_hsl=True
    )
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    runner.max_dispatch_candidate_bars = 3 * 3 * len(sides) * 17
    candidates = [{"hsl_red_threshold": 1e-6}, {"hsl_red_threshold": 0.03}, {}]
    ends = np.array([runner.n - 1, 73, 1], dtype=np.int32)
    loaders = (
        mps_kernel._ema_anchor_multicoin_shader_library,
        mps_kernel._trailing_martingale_multicoin_shader_library,
    )
    original = cuda_kernel.cuda_source
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                cuda_kernel,
                "cuda_source",
                lambda source, **kwargs: original(
                    source, shared_replay_state=False, **kwargs
                ),
            )
            for loader in loaders:
                loader.cache_clear()
            _, private = raw(proxy, candidates, end_steps=ends)
        for loader in loaders:
            loader.cache_clear()
        _, shared = raw(proxy, candidates, end_steps=ends)
        equal_outputs(private, shared)
        assert runner._last_temporal_dispatch["dispatch_count"] > 1
    finally:
        # Do not leave a diagnostic memory-policy variant in the global cache.
        for loader in loaders:
            loader.cache_clear()


@pytest.mark.parametrize(
    "strategy,sides,shocks",
    [
        ("ema_anchor", "both", False),
        ("trailing_martingale", "short", True),
    ],
)
def test_seed_43_coin_hsl_fixture_matches_private_replay(
    monkeypatch, strategy, sides, shocks
):
    """Longer ordering fixtures must retain every baseline GPU summary."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    from optimization.gpu import cuda_kernel, mps_kernel
    from tools.gpu_parity import build_parser, fixture_inputs
    from optimization.gpu.service import SharedAccountReplay
    from test_gpu_hsl_multicoin import raw
    from test_gpu_side_specialization import equal_outputs

    config, candles, markets, btc, timestamps = fixture_inputs(
        build_parser().parse_args(
            [
                "--fixture",
                strategy,
                "--sides",
                sides,
                "--coins",
                "2",
                "--bars",
                "3000",
                "--seed",
                "43",
                "--hsl",
                "coin",
            ]
        )
    )
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(
            n_positions=1 if sides == "both" or side == sides else 0,
            total_wallet_exposure_limit=2 if sides == "both" or side == sides else 0,
        )
        config["bot"][side]["hsl"].update(
            drawdown_ratio_threshold=0.002, ema_span_minutes=2.5, cooldown_minutes_after_red=5
        )
    if shocks:
        candles[1500:, 0, :3] *= 0.7
        candles[1800:, 1, :3] *= 1.3
    metrics = {
        "adg_strategy_eq",
        "drawdown_worst_strategy_eq",
        "fills_per_day",
        "hard_stop_triggers_per_year",
        "hard_stop_restarts_per_year",
        "hard_stop_time_in_red_pct",
    }
    proxy = SharedAccountReplay(
        config=config,
        hlcvs=candles,
        mss=markets,
        btc=btc,
        timestamps=timestamps,
        exchange="binance",
        batch_size=3,
        needed_metrics=metrics,
        factual_hsl=True,
    )
    loader = getattr(mps_kernel, f"_{strategy}_multicoin_shader_library")
    original = cuda_kernel.cuda_source
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                cuda_kernel,
                "cuda_source",
                lambda source, **kwargs: original(
                    source, shared_replay_state=False, **kwargs
                ),
            )
            loader.cache_clear()
            _, private = raw(proxy, [{}, {}])
        loader.cache_clear()
        _, shared = raw(proxy, [{}, {}])
        equal_outputs(private, shared)
    finally:
        loader.cache_clear()
