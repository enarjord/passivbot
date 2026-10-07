"""Raw strategy growth uses its factual curve, including terminal loss marks."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.metric_registry import RAW_STRATEGY_EQUITY_METRICS
from optimization.gpu.metrics import _raw_strategy_equity_metrics


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantized", [False, True], ids=["f64", "f32"])
def test_raw_daily_reducer_matches_actual_rust_producer(device, quantized):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "gpu_weighted_equity.json").read_text())
    for case in fixture["cases"]:
        values = np.asarray(case["equities"], dtype=np.float32 if quantized else np.float64).astype(np.float64)
        day_ids = (case["first_timestamp_ms"] + np.arange(len(values)) * case["interval_ms"]) // 86_400_000
        # Leading/trailing padding exercises actual compact buffer masking.
        ends, minima, dd = [0.], [float("inf")], [0.]
        peak = -float("inf")
        for day in np.unique(day_ids):
            sample = values[day_ids == day]
            worst = 0.
            for value in sample:
                peak = max(peak, value)
                worst = max(worst, (peak - value) / max(abs(peak), 1e-12))
            ends.append(sample[-1]); minima.append(sample.min()); dd.append(worst)
        ends.append(0.); minima.append(float("inf")); dd.append(1000.)
        output = {name: torch.tensor([data], device=device, dtype=torch.float64) for name, data in (
            ("raw_strategy_day_end_eq", ends), ("raw_strategy_day_min_eq", minima),
            ("raw_strategy_day_max_dd", dd),
        )}
        expected = case["expected_raw_full_f32" if quantized else "expected_raw_full"]
        actual = _raw_strategy_equity_metrics(output, RAW_STRATEGY_EQUITY_METRICS)
        assert set(actual) == set(expected)
        for name, golden in expected.items():
            value = actual[name].item()
            if golden is None:
                assert not np.isfinite(value), (case["case"], name, value)
            else:
                assert value == pytest.approx(golden, abs=1e-8, rel=1e-12), (case["case"], name, value, golden)
            single = _raw_strategy_equity_metrics(output, {name})
            torch.testing.assert_close(single[name], actual[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["mark", "panic-fill"])
def test_native_raw_growth_retains_liquidation_mark(strategy, sides, terminal_fill):
    from test_gpu_hsl_ordering import _liquidation_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = list(_liquidation_inputs(strategy, sides, terminal_fill))
    origin = int(inputs[4][0]) // 86_400_000 * 86_400_000 + 23 * 3_600_000 + 56 * 60_000
    inputs[4] = origin + np.arange(len(inputs[4]), dtype=np.int64) * 60_000
    account = {name.replace("_strategy_eq", "_usd") for name in RAW_STRATEGY_EQUITY_METRICS
               if name not in {"adg_rolling_hmean_strategy_eq", "adg_time_integrated_strategy_eq",
                               "positive_gain_participation_strategy_eq"}}
    names = RAW_STRATEGY_EQUITY_METRICS | account | {
        "adg_per_exposure_long_usd", "adg_strategy_eq_w", "adg_w_usd",
        "drawdown_worst_strategy_eq", "drawdown_worst_usd",
    }
    report = run_comparison(tuple(inputs), "bybit", tuple(sorted(names)),
        {name: MetricTolerance(1e-8, 1e-5) for name in names},
        diagnostics=True, gpu_engine="native")
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["metrics"]["adg_strategy_eq"]["cpu"] == -1.
    assert report["metrics"]["adg_usd"]["cpu"] > -1.
    assert report["passed"], report["metrics"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("raw_risk", [False, True])
@pytest.mark.parametrize("btc_risk", [False, True])
def test_raw_growth_ablation_preserves_all_outputs(strategy, sides, raw_risk, btc_risk):
    from test_gpu_weighted_equity_capture import _runner_context
    from optimization.gpu.metric_registry import WEIGHTED_EQUITY_METRICS

    kwargs = dict(requested=WEIGHTED_EQUITY_METRICS, raw_risk=raw_risk, btc_risk=btc_risk)
    baseline, params = _runner_context(strategy, sides, **kwargs)
    enabled, _ = _runner_context(strategy, sides, raw_growth=True, **kwargs)
    assert enabled.daily_cols == baseline.daily_cols + 2
    assert enabled._history_bytes_per_candidate() - baseline._history_bytes_per_candidate() == 8 * enabled.n_days
    expected = baseline.run(params)
    actual = enabled.run(params)
    assert set(actual) == set(expected) | {"raw_strategy_day_end_eq", "raw_strategy_day_min_eq"}
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{name}: {message}")
        else:
            assert actual[name] == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("side", ["long", "short"])
def test_temporal_raw_growth_preserves_partial_days(side):
    from test_gpu_weighted_equity_capture import _runner_context
    from optimization.gpu.metric_registry import WEIGHTED_EQUITY_METRICS

    kwargs = dict(requested=WEIGHTED_EQUITY_METRICS, raw_growth=True, raw_risk=True)
    generic, params = _runner_context("trailing_martingale", side, **kwargs)
    temporal, _ = _runner_context("trailing_martingale", side, chunked=True, **kwargs)
    ends = np.asarray([2, 799, 1512], dtype=np.int32)
    expected = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in generic.run(params, end_steps=ends).items()}
    actual = temporal.run(params, end_steps=ends)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{name}: {message}")
        else:
            assert actual[name] == value


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("btc_risk", [False, True])
def test_daily_decoder_uses_explicit_feature_ownership(fused, btc_risk):
    from optimization.gpu.mps_kernel import _decode_outputs, _decode_multicoin_fused_outputs
    # Raw closes/minima/risk make twelve daily columns without any BTC metrics.
    # Shape cannot identify optional column ownership.
    daily = torch.arange(15, dtype=torch.float32).reshape(1, 1, 15)
    scalars = torch.zeros((1, 72), dtype=torch.float32)
    decode = _decode_multicoin_fused_outputs if fused else _decode_outputs
    actual = decode(daily, scalars, torch.zeros((1, 128)), btc_risk_enabled=btc_risk)
    btc = {"btc_day_end_eq", "btc_day_min_eq", "btc_day_max_dd"}
    assert set(actual) & btc == (btc if btc_risk else set())
    if btc_risk:
        assert actual["btc_day_end_eq"].item() == 9.
        assert actual["btc_day_min_eq"].item() == 10.
        assert actual["btc_day_max_dd"].item() == 11.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("hsl", ["disabled", "coin", "pside", "unified"])
def test_native_raw_growth_on_active_shock_replay(strategy, sides, hsl):
    from test_gpu_side_equity_sampling import _side_equity_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    # Bounds apply only to this public seed-43 shock. Actual Rust producers
    # match reductions of CPU, quantized CPU and GPU curves independently.
    # Existing replay differences change near-zero ADG and derived ratios;
    # their relative errors alone are not useful acceptance criteria.
    growth = {"adg_strategy_eq", "mdg_strategy_eq", "adg_rolling_hmean_strategy_eq",
              "adg_time_integrated_strategy_eq", "expected_shortfall_1pct_strategy_eq"}
    policies = {name: MetricTolerance(1e-4, 0) if name in growth
                else MetricTolerance(1e-4, .01) for name in RAW_STRATEGY_EQUITY_METRICS}
    policies["positive_gain_participation_strategy_eq"] = MetricTolerance(1e-6, .001)
    if strategy == "trailing_martingale" and sides != "long":
        # A near-zero two-day ADG affects short-side ratios most; all observed
        # absolute ratio gaps remain below .0005 in those ill-conditioned cases.
        for name in ("calmar_ratio_strategy_eq", "sterling_ratio_strategy_eq",
                     "sharpe_ratio_strategy_eq", "sortino_ratio_strategy_eq"):
            policies[name] = MetricTolerance(5e-4, .01)
    report = run_comparison(_side_equity_inputs(strategy, sides, hsl), "binance",
        tuple(sorted(RAW_STRATEGY_EQUITY_METRICS)), policies, diagnostics=True, gpu_engine="native")
    assert not report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["passed"], report["metrics"]
