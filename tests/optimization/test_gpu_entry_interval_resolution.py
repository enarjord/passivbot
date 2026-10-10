"""Native entry percentiles retain exact gap counts without exposing histories."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.entry_intervals import (
    ENTRY_INTERVAL_NAMES, MAX_EXACT_ENTRY_BARS,
    entry_interval_history_bytes, entry_intervals_from_counts,
)


def _counts(populations, n=1004):
    result = torch.zeros((len(populations), n + 2), dtype=torch.int32)
    for row, gaps in enumerate(populations):
        result[row, 0] = len(gaps)
        for gap in gaps:
            result[row, gap + 1] += 1
    return result


def _reference(gaps, interval_ms):
    if not len(gaps):
        return np.zeros(5)
    values = np.asarray(gaps, dtype=np.float64) * interval_ms / 3_600_000
    return np.asarray([values.mean(), *np.quantile(values, [.5, .95, .99]), values.max()])


@pytest.mark.parametrize("interval_ms", [60_000, 300_000, 3_600_000])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ordered_gap_percentiles_match_sorted_population(interval_ms, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(43)
    populations = [[], [0], [1004], [0, 0, 1, 1004], [5, 8, 8, 17],
                   *[rng.integers(0, 1005, size=n) for n in (99, 100, 199, 200, 1003)]]
    counts = _counts(populations).to(device)
    original = counts.clone()
    actual = entry_intervals_from_counts(counts, interval_ms)
    assert actual.device.type == device and actual.dtype == torch.float64
    np.testing.assert_allclose(actual.cpu(), [_reference(p, interval_ms) for p in populations],
                               rtol=2e-15, atol=2e-13)
    assert torch.equal(counts, original)


def test_large_integer_multiplicities_and_noncontiguous_counts():
    counts = torch.zeros((2, 8), dtype=torch.int32)
    counts[0, 0], counts[0, 1], counts[0, 4] = 16_777_218, 16_777_217, 1
    counts[1, 0], counts[1, 3], counts[1, 7] = 2_147_483_647, 1_000_000_000, 1_147_483_647
    expected = [[3 / 16_777_218, 0, 0, 0, 3],
                [(2 * 1_000_000_000 + 6 * 1_147_483_647) / 2_147_483_647, 6, 6, 6, 6]]
    backing = torch.zeros((2, 16), dtype=torch.int32)
    backing[:, ::2] = counts
    assert not backing[:, ::2].is_contiguous()
    np.testing.assert_allclose(entry_intervals_from_counts(backing[:, ::2], 3_600_000), expected,
                               rtol=2e-15, atol=1e-15)


def test_malformed_gap_counts_are_rejected():
    with pytest.raises(ValueError, match="int32"):
        entry_intervals_from_counts(torch.zeros((1, 5)), 60_000)
    overflow = _counts([[1, 2]], 3)
    overflow[0, 0] = -1
    with pytest.raises(RuntimeError, match="overflow"):
        entry_intervals_from_counts(overflow, 60_000)
    inconsistent = _counts([[1, 2]], 3)
    inconsistent[0, 0] += 1
    with pytest.raises(RuntimeError, match="disagrees"):
        entry_intervals_from_counts(inconsistent, 60_000)
    with pytest.raises(ValueError, match="milliseconds"):
        entry_intervals_from_counts(_counts([], 3), 0)


def test_compact_native_payload_uses_exact_metrics_and_rejects_malformed_values():
    from optimization.gpu.metrics import _entry_interval_metrics
    from types import SimpleNamespace
    compact = entry_intervals_from_counts(_counts([[0, 1, 10, 40]]), 60_000)
    output = {"fill_count": torch.tensor([7.]), "entry_interval_native_metrics": compact}
    result = _entry_interval_metrics(output, SimpleNamespace(interval_ms=60_000), "trailing_martingale")
    np.testing.assert_allclose([result[name].item() for name in ENTRY_INTERVAL_NAMES],
                               _reference([0, 1, 10, 40], 60_000), rtol=2e-15)
    output["entry_interval_native_metrics"] = compact[:, :4]
    with pytest.raises(RuntimeError, match="malformed"):
        _entry_interval_metrics(output, SimpleNamespace(interval_ms=60_000), "trailing_martingale")


def test_native_gap_allocation_reset_budget_and_index_bounds(monkeypatch):
    from optimization.gpu import mps_kernel
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *args: "cpu")
    runner = object.__new__(mps_kernel._MulticoinReplayRunner)
    runner.n, runner.native_factual_hsl, runner.entry_interval_enabled = 1004, True, True
    runner._entry_interval_stat_buffers, runner._entry_interval_count_buffers = {}, {}
    runner._history_bytes_per_candidate = lambda: entry_interval_history_bytes(runner.n)
    runner.hsl_scratch_budget_bytes = 2 * runner._history_bytes_per_candidate()
    stats, counts = runner._entry_interval_buffers(2)
    assert counts.shape == (2, 1006) and counts.dtype == torch.int32
    counts.fill_(1)
    stats.fill_(1)
    repeated = runner._entry_interval_buffers(2)
    assert repeated[1].data_ptr() == counts.data_ptr()
    assert counts.eq(0).all() and stats.eq(0).all()
    with pytest.raises(ValueError, match="scratch budget"):
        runner._entry_interval_buffers(3)
    runner.hsl_scratch_budget_bytes = 1 << 60
    with pytest.raises(ValueError, match="addressing"):
        runner._entry_interval_buffers(np.iinfo(np.int32).max // (runner.n + 2) + 1)
    for n in (0, MAX_EXACT_ENTRY_BARS + 1):
        with pytest.raises(ValueError, match="bars"):
            entry_interval_history_bytes(n)
    runner.entry_interval_enabled = False
    assert runner._entry_interval_buffers(2) == (None, None)
    runner.entry_interval_enabled, runner.native_factual_hsl = True, False
    runner._entry_interval_stat_buffers, runner._entry_interval_count_buffers = {}, {}
    assert runner._entry_interval_buffers(2)[1].shape == (2, 129)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_native_gap_capture_temporal_partial_and_compact_join(sides, interval_minutes, monkeypatch):
    from copy import deepcopy
    import backtest
    from tools.gpu_parity import build_parser, fixture_inputs
    from optimization.gpu import mps_kernel
    from optimization.gpu.service import MpsMulticoinProxy

    def forbidden(*args, **kwargs):
        pytest.fail("native gap replay must not use CPU backtests")
    for owner, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                        (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(owner, name, forbidden)
    inputs = list(fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", sides,
        "--coins", "2", "--bars", "1504", "--seed", "43",
    ])))
    if interval_minutes != 1:
        inputs[0]["backtest"]["candle_interval_minutes"] = interval_minutes
        interval_ms = interval_minutes * 60_000
        first_ts = int(inputs[4][0]) // interval_ms * interval_ms
        inputs[4] = first_ts + np.arange(len(inputs[4]), dtype=np.int64) * interval_ms
        inputs[2]["__meta__"].update(
            data_interval_minutes=interval_minutes, requested_start_ts=int(inputs[4][0])
        )
        # Prepared validity bounds retain the producer's one-minute index units.
        for coin in inputs[0]["backtest"]["coins"]["binance"]:
            inputs[2][coin]["last_valid_index"] = len(inputs[1]) * interval_minutes - 1
    config, candles, markets, btc, timestamps = inputs
    replay = MpsMulticoinProxy(config=deepcopy(config), hlcvs=candles, mss=markets,
        btc=btc, timestamps=timestamps, exchange="binance", batch_size=3,
        needed_metrics=set(ENTRY_INTERVAL_NAMES), factual_hsl=True)
    runner = replay.fused_runner if sides == "both" else replay.runners[sides]
    candidates = [{}, {f"{replay.sides[0]}_entry_initial_qty_pct": .03}, {}]
    params = np.concatenate([replay._parameter_matrix(candidates, side) for side in replay.sides], axis=1)
    runner.max_dispatch_candidate_bars = None
    full = {k: v.clone() for k, v in runner.run(params).items()}
    counts = runner._entry_interval_count_buffers[3].cpu().numpy()
    assert counts[:, 0].max() > 0
    populations = [np.repeat(np.arange(counts.shape[1] - 1), row[1:]) for row in counts]
    np.testing.assert_allclose(full["entry_interval_native_metrics"].cpu(),
        [_reference(p, interval_minutes * 60_000) for p in populations], rtol=2e-15, atol=2e-13)
    assert "entry_interval_hist" not in full
    runner.max_dispatch_candidate_bars = 3 * runner.n_coins * runner.replay_sides * 73
    chunked = runner.run(params)
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    for key in full:
        torch.testing.assert_close(chunked[key], full[key], rtol=0, atol=0, equal_nan=True)
    ends = np.array([1, runner.n // 2, runner.n], dtype=np.int32)
    partial = {k: v.clone() for k, v in runner.run(params, end_steps=ends).items()}
    assert partial["entry_interval_native_metrics"][0].eq(0).all()
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    split = runner.run(params, end_steps=ends)
    for key in partial:
        torch.testing.assert_close(split[key], partial[key], rtol=0, atol=0, equal_nan=True)
    assert split["entry_interval_native_metrics"].shape == (3, 5)
    assert set(runner._entry_interval_count_buffers) == {1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rejected_factual_attempt_does_not_reduce_or_reuse_entry_counts(monkeypatch):
    from optimization.gpu import mps_kernel
    from test_gpu_weighted_equity_capture import _runner_context
    runner, params = _runner_context("trailing_martingale", "both", shock=True,
                                    factual_hsl=True, count=512)
    uncaptured = {key: value.clone() if isinstance(value, torch.Tensor) else value
                  for key, value in runner.run(params).items()}
    runner.entry_interval_enabled = True
    captured = runner.run(params)
    for key, value in uncaptured.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(captured[key], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert captured[key] == value
    assert set(captured) == set(uncaptured) | {"entry_interval_native_metrics"}
    expected = captured["entry_interval_native_metrics"].clone()
    real_reduce = mps_kernel.entry_intervals_from_counts
    accepted = []

    def reduced(counts, interval):
        accepted.append(counts.shape[0])
        return real_reduce(counts, interval)

    monkeypatch.setattr(mps_kernel, "entry_intervals_from_counts", reduced)
    runner.hsl_fact_capacity_learned = 1
    actual = runner.run(params)
    assert runner.last_hsl_fact_retries > 0
    assert accepted == [3]
    torch.testing.assert_close(actual["entry_interval_native_metrics"], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_gap_accumulator_zero_gap_empty_population_and_invalid_counts():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    from optimization.gpu.runtime import compile_shader
    from optimization.gpu.mps_kernel import _with_hsl
    verify_loaded_runtime_extension()
    source = ("#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 32\n"
              "#define PASSIVBOT_ENTRY_INTERVAL_ENABLED 1\n"
              + _with_hsl(passivbot_rust.mps_trailing_martingale_multicoin_source_py(), 34, 1) + r"""
kernel void entry_gap_probe(device float* stats, device int* counts,
    uint b [[thread_position_in_grid]]) {
    float last = -1.0f;
    if (b == 0) {
        record_initial_entry_interval(stats, counts, b, last, 5.0f, 32);
        record_initial_entry_interval(stats, counts, b, last, 5.0f, 32);
        record_initial_entry_interval(stats, counts, b, last, 9.0f, 32);
        record_initial_entry_interval(stats, counts, b, last, 11.0f, 32);
    } else if (b < 3) {
        last = 5.0f;
        counts[ulong(b) * 33] = b == 1 ? 2147483647 : 0;
        record_initial_entry_interval(stats, counts, b, last, b == 1 ? 6.0f : 40.0f, 32);
    } else if (b < 5) {
        last = 5.0f;
        record_initial_entry_interval(stats, counts, b, last, b == 3 ? 5.5f : 4.0f, 32);
    } else {
        record_initial_entry_interval(stats, counts, b, last, 5.0f, 32);
    }
}
""")
    stats = torch.zeros((6, 2), device="cuda")
    counts = torch.zeros((6, 33), dtype=torch.int32, device="cuda")
    compile_shader(source).entry_gap_probe(stats, counts, threads=6)
    actual = counts.cpu()
    assert actual[0, 0] == 3
    assert actual[0, 1] == actual[0, 3] == actual[0, 5] == 1
    assert actual[1:5, 0].eq(-1).all()
    assert actual[5].eq(0).all()
