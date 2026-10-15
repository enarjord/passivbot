"""Exact daily tail selection, horizon capacity, and query/replay isolation."""
import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.mps_kernel import _raw_drawdown_tail_capacity, _with_hsl_features


@pytest.mark.parametrize("days,capacity", [(1, 1), (199, 1), (200, 2), (299, 2),
                                          (300, 4), (799, 8), (800, 8), (3000, 32)])
def test_tail_capacity_covers_registered_horizon(days, capacity):
    assert _raw_drawdown_tail_capacity(days) == capacity


@pytest.mark.parametrize("days", [0, -1, True, 1.0])
def test_tail_capacity_rejects_invalid_horizon(days):
    with pytest.raises(ValueError, match="positive prepared day count"):
        _raw_drawdown_tail_capacity(days)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.0])
def test_direct_tail_source_rejects_invalid_capacity(capacity):
    source = ("#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED\n"
              "#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED\nbody")
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        _with_hsl_features(source, ema_tail_enabled=False, raw_drawdown_enabled=True,
                           raw_tail_enabled=True, raw_tail_capacity=capacity)


_PROBE = r"""
kernel void daily_tail_probe(constant float* equity, constant int* days,
    constant int* sizes, device float* out, uint b [[thread_position_in_grid]]) {
    HslStrategyEquityStats stats = init_hsl_strategy_equity_stats();
    HslStrategyEquityStats uninterrupted = init_hsl_strategy_equity_stats();
    for (int k = 0; k < sizes[0]; ++k) {
        update_hsl_strategy_equity_stats(stats, equity[k], days[k]);
        update_hsl_strategy_equity_stats(uninterrupted, equity[k], days[k]);
        if (k == sizes[1]) {
            // Query before copying the state at a temporal replay boundary.
            out[0] = hsl_strategy_equity_drawdown_mean_worst_1pct(stats);
            HslStrategyEquityStats snapshot = stats;
            stats = snapshot;
        }
    }
    out[1] = hsl_strategy_equity_drawdown_mean_worst_1pct(stats);
    out[2] = hsl_strategy_equity_drawdown_mean_worst_1pct(stats);
    out[3] = hsl_strategy_equity_drawdown_mean_worst_1pct(uninterrupted);
    out[4] = hsl_strategy_equity_drawdown_max(stats);
    out[5] = sizeof(HslStrategyEquityStats);
}
"""


def _daily_reference(equity, days):
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(np.abs(peak), np.float32(1e-12))
    daily = [dd[days == day].max() for day in np.unique(days)]
    return float(np.mean(sorted(daily, reverse=True)[:max(len(daily) // 100, 1)], dtype=np.float64))


def _cases():
    rows = []
    for count in (199, 200, 299, 300):
        eq = np.full(count, 985, dtype=np.float32)
        eq[:3] = [1000, 983, 984]
        rows.append((f"same-bin-{count}", eq, np.arange(count, dtype=np.int32), count))
    rng = np.random.default_rng(4319)
    for count in (300, 3000):
        eq = rng.uniform(500, 1100, count * 3).astype(np.float32)
        days = np.repeat(np.arange(count, dtype=np.int32) * 2 + 19723, 3)
        rows.append((f"intraday-gaps-{count}", eq, days, int(days[-1] - days[0]) + 1))
    rows.extend([
        ("early-truncation", np.array([1000, 970, 980], dtype=np.float32),
         np.arange(3, dtype=np.int32), 3000),
        ("unfinished-worst-day", np.r_[np.full(299, 1000), 500].astype(np.float32),
         np.arange(300, dtype=np.int32), 300),
        ("flat", np.full(300, 1000, dtype=np.float32), np.arange(300, dtype=np.int32), 300),
    ])
    return rows


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cuda_exact_daily_tail_and_query_snapshot_isolation(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    base = ("#define PASSIVBOT_HSL_CAPACITY 64\n#define PASSIVBOT_HSL_TREE_SIZE 1\n"
            "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
            + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")())
    libraries = {}
    for name, eq, days, horizon in _cases():
        capacity = _raw_drawdown_tail_capacity(horizon)
        if capacity not in libraries:
            source = _with_hsl_features(base, ema_tail_enabled=False,
                raw_drawdown_enabled=True, raw_tail_enabled=True, raw_tail_capacity=capacity)
            libraries[capacity] = compile_shader(source + _PROBE)
        cut = max(len(eq) // 2, 1)
        args = [torch.tensor(a, device="cuda") for a in
                (eq, days, np.array([len(eq), cut], dtype=np.int32))]
        out = torch.empty(6, dtype=torch.float32, device="cuda")
        libraries[capacity].daily_tail_probe(*args, out, threads=1)
        got = out.cpu().tolist()
        assert got[0] == pytest.approx(_daily_reference(eq[:cut+1], days[:cut+1]), abs=1e-7), name
        assert got[1] == pytest.approx(_daily_reference(eq, days), abs=1e-7), name
        assert got[1] == got[2] == got[3], name
        assert got[4] == pytest.approx(float(np.max(
            (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq))), abs=1e-7), name
        assert got[5] <= 44 + capacity * 4

    # Explicitly undersized direct source must not fabricate a cutoff-bin tail.
    source = _with_hsl_features(base, ema_tail_enabled=False,
        raw_drawdown_enabled=True, raw_tail_enabled=True, raw_tail_capacity=1)
    eq = np.full(300, 985, dtype=np.float32); eq[0] = 1000
    out = torch.empty(6, dtype=torch.float32, device="cuda")
    compile_shader(source + _PROBE).daily_tail_probe(
        torch.tensor(eq, device="cuda"), torch.arange(300, dtype=torch.int32, device="cuda"),
        torch.tensor([300, 149], dtype=torch.int32, device="cuda"), out, threads=1)
    assert math.isnan(out.cpu().tolist()[1])


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_cuda_long_replay_tail_ablation_capacity_and_temporal_parity(strategy, sides):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu import mps_kernel
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    count = 210 * 96 + 7  # 15-minute candles; partial initial/final UTC days.
    close = np.tile([100., 120.], (count, 1))
    close[3000:14000, 0] *= .8
    close[7000:17000, 1] *= 1.2
    _, row, run, data = _multicoin_exposure_fixture(strategy,
        "long" if sides == "both" else sides, count=count, closes=close,
        return_context=True, interval_minutes=15)
    cls = {
        ("ema_anchor", False): mps_kernel.MpsEmaAnchorMulticoinRunner,
        ("ema_anchor", True): mps_kernel.MpsEmaAnchorMulticoinFusedRunner,
        ("trailing_martingale", False): mps_kernel.MpsTrailingMartingaleMulticoinRunner,
        ("trailing_martingale", True): mps_kernel.MpsTrailingMartingaleMulticoinFusedRunner,
    }[strategy, sides == "both"]
    params = np.asarray([row] * 3, dtype=np.float64)
    if sides == "both":
        params = np.concatenate((params, params), axis=1)
    kwargs = {} if sides == "both" else {"side": sides}
    ends = np.array([799, 15000, count], dtype=np.int32)
    tail_keys = {"hsl_drawdown_raw_mean_worst_1pct_long", "hsl_drawdown_raw_mean_worst_1pct_short"}

    def replay(enabled, capacity=None, temporal=False):
        runner = cls(run, data, hsl_raw_drawdown_enabled=True,
                     hsl_raw_tail_enabled=enabled, **kwargs)
        expected_capacity = _raw_drawdown_tail_capacity(data["n_days"]) if enabled else 1
        assert runner.hsl_raw_tail_capacity == expected_capacity
        if capacity is not None:
            runner.hsl_raw_tail_capacity = capacity
        if temporal:
            runner.max_dispatch_candidate_bars = 3 * 2 * 197
        loader, arguments = runner._library_cache_call()
        bound = inspect.signature(loader).bind(*arguments)
        assert bound.arguments["hsl_raw_tail_capacity"] == runner.hsl_raw_tail_capacity
        return {name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in runner.run(params, end_steps=ends).items()}

    disabled, exact, larger = replay(False), replay(True), replay(True, capacity=8)
    assert exact.keys() == disabled.keys() == larger.keys()
    for name in exact:
        if isinstance(exact[name], torch.Tensor):
            torch.testing.assert_close(exact[name], larger[name], rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{name}: {message}")
            if name not in tail_keys:
                torch.testing.assert_close(exact[name], disabled[name], rtol=0, atol=0, equal_nan=True,
                    msg=lambda message: f"{name}: {message}")
        else:
            assert exact[name] == disabled[name] == larger[name]
    assert any(exact[name].max().item() > 0 for name in tail_keys)
    for name in tail_keys:
        assert torch.isfinite(exact[name]).all()
    if strategy == "trailing_martingale":
        chunked = replay(True, temporal=True)
        for name, value in exact.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(chunked[name], value, rtol=0, atol=0, equal_nan=True,
                    msg=lambda message: f"{name}: {message}")
