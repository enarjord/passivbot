"""Strict completion guards must accept full replay and reject early endpoints."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.metrics import _backtest_completion_ratio


@pytest.mark.parametrize("count", [126143, 238463, 2749823])
@pytest.mark.parametrize("interval", [60_000, 300_000, 420_000])
def test_long_native_full_replay_is_exactly_one_and_one_candle_short_is_not(count, interval):
    origin = 1618790400000
    run = SimpleNamespace(interval_ms=interval, requested_start_ts_ms=origin + 90 * interval)
    # The timestamp reflects the old f32 product, including its rounding error.
    step = torch.tensor([count - 1, count - 2, count // 2, -1], dtype=torch.int64)
    output = {"last_eq_step": step,
              "last_eq_ts": (step.float() * interval).double() + origin,
              "alive": torch.tensor([True, True, False, True])}
    ratio = _backtest_completion_ratio(output, run, {"ts0": origin, "n": count}, step >= 0)
    assert ratio.dtype == torch.float64
    assert ratio[0].item() == 1.0
    assert 0 < ratio[1].item() < 1.0  # surviving but explicitly truncated
    assert 0 < ratio[2].item() < 1.0  # early liquidation
    assert ratio[3].item() == 0.0
    expected = (count - 91) / (count - 90)
    assert ratio[1].item() == expected


@pytest.mark.parametrize("step", [torch.tensor([3.5]), torch.tensor([100]), torch.tensor([-2])])
def test_native_malformed_endpoint_does_not_fall_back_to_timestamp(step):
    run = SimpleNamespace(interval_ms=60_000, requested_start_ts_ms=0)
    with pytest.raises(RuntimeError, match="completion candle endpoint"):
        _backtest_completion_ratio({"last_eq_step": step, "last_eq_ts": torch.tensor([99 * 60_000.])},
                                   run, {"ts0": 0, "n": 100}, torch.tensor([True]))


def test_requested_boundary_candle_is_not_an_extra_required_interval():
    run = SimpleNamespace(interval_ms=60_000, requested_start_ts_ms=0, requested_end_ts_ms=99 * 60_000)
    output = {"last_eq_step": torch.tensor([98, 97], dtype=torch.int64)}
    ratio = _backtest_completion_ratio(output, run, {"ts0": 0, "n": 100}, torch.tensor([True, True]))
    assert ratio[0].item() == 1.0
    assert ratio[1].item() == 98 / 99
    run.requested_end_ts_ms = 100 * 60_000
    assert _backtest_completion_ratio(output, run, {"ts0": 0, "n": 100}, torch.tensor([True, True]))[0] < 1.0


def test_native_decoder_preserves_integer_endpoint_and_legacy_timestamp_shape():
    from optimization.gpu.mps_kernel import _decode_outputs, _decode_multicoin_fused_outputs
    daily = torch.zeros((1, 1, 9))
    scalars = torch.zeros((1, 82))
    scalars[0, 8] = 2749822
    gaps = torch.zeros((1, 1024))
    for decode in (_decode_outputs, _decode_multicoin_fused_outputs):
        native = decode(daily, scalars, gaps, native_factual_hsl=True, interval_ms=300_000)
        assert native["last_eq_step"].dtype == torch.int64
        assert native["last_eq_step"].item() == 2749822
        assert native["last_eq_ts"].item() == (scalars[:, 8] * 300_000).item()
        legacy = decode(daily, scalars, gaps)
        assert "last_eq_step" not in legacy
        assert legacy["last_eq_ts"].item() == scalars[0, 8].item()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
def test_native_hardware_full_completion_matches_exact_rust(strategy, sides):
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("CUDA or Metal device required")
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    inputs = list(fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "2", "--sides", sides, "--bars", "512",
    ])))
    import numpy as np
    # The public fixture ends at 23:59; add the explicit requested-end boundary
    # candle so the last executed candle covers the complete requested day.
    inputs[1] = np.concatenate([inputs[1], inputs[1][-1:]], axis=0)
    inputs[3] = np.concatenate([inputs[3], inputs[3][-1:]], axis=0)
    inputs[4] = np.concatenate([inputs[4], [int(inputs[4][-1]) + 60_000]])
    report = run_comparison(inputs, "binance", ["backtest_completion_ratio"], DEFAULT_TOLERANCES, diagnostics=True)
    assert report["status"] == "passed", report
    assert report["metrics"]["backtest_completion_ratio"]["gpu"] == 1.0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("terminal_fill", [False, True])
def test_native_hardware_early_liquidation_remains_incomplete(strategy, terminal_fill):
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("CUDA or Metal device required")
    from test_gpu_hsl_ordering import _liquidation_inputs
    from tools.gpu_parity import run_comparison, DEFAULT_TOLERANCES
    inputs = _liquidation_inputs(strategy, "long", terminal_fill)
    from datetime import datetime, timezone
    first = int(inputs[4][0])
    end = ((int(inputs[4][-1]) // 86_400_000) + 1) * 86_400_000
    inputs[0]["backtest"].update(start_date=datetime.fromtimestamp(first / 1000, timezone.utc).isoformat(),
                                end_date=datetime.fromtimestamp(end / 1000, timezone.utc).isoformat())
    report = run_comparison(inputs, "bybit",
                            ["backtest_completion_ratio"], DEFAULT_TOLERANCES, diagnostics=True)
    assert report["status"] == "passed", report
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert 0 <= report["metrics"]["backtest_completion_ratio"]["gpu"] < 1.0
