"""Metal reductions and prepared-service qualification with public fixtures."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Metal required")


@pytest.mark.parametrize("curve", ["raw_strategy", "account"])
def test_metal_weighted_suffixes_match_f64_reference_with_integer_epoch(curve):
    from test_gpu_weighted_equity import FIXTURE, _inputs
    from optimization.gpu.weighted_equity import weighted_equity_from_samples
    # Rust-generated goldens include short, discarded-peak, signed and flat paths.
    for case in FIXTURE["cases"]:
        samples, kwargs = _inputs([case], device="cpu", dtype=torch.float32, curve=curve)
        names = case["expected_raw"] if curve == "raw_strategy" else case["expected_account"]["initial_fill"]
        kwargs["first_timestamps_ms"] += 1_728_000_000_000  # UTC midnight
        expected = weighted_equity_from_samples(samples, requested=names, **kwargs)
        kwargs["first_timestamps_ms"] = torch.where(
            kwargs["sample_counts"] > 0, kwargs["first_timestamps_ms"], 0).to(torch.int64)
        device_kwargs = {key: value.to("mps") if isinstance(value, torch.Tensor) else value
                         for key, value in kwargs.items()}
        device_kwargs["first_timestamps_ms"] = kwargs["first_timestamps_ms"].to(torch.int64).to("mps")
        actual = weighted_equity_from_samples(samples.to("mps"), requested=names, **device_kwargs)
        for name in names:
            assert actual[name].device.type == "cpu"
            assert actual[name].dtype == torch.float64
            torch.testing.assert_close(actual[name].cpu().double(), expected[name],
                                       rtol=2e-5, atol=2e-6, equal_nan=True, msg=f"{case['case']}: {name}")


def test_metal_ew_retains_exact_utc_metadata_and_missing_clocks():
    from test_gpu_exponential_equity import fixture
    from config.ew import EW_METRICS
    from optimization.gpu.exponential_equity import exponential_equity_metrics
    origin = 1_728_000_000_000
    out, run, data = fixture(origin=origin)
    expected, metadata = exponential_equity_metrics(out, run, data, EW_METRICS)
    relative = deepcopy(out)
    for name in ("first_eq_ts", "last_eq_ts"):
        relative[name] -= origin
    relative = {name: value.float().to("mps") if value.is_floating_point() else value.to("mps")
                for name, value in relative.items()}
    actual, sidecar = exponential_equity_metrics(relative, run, data, EW_METRICS, relative_timestamps=True)
    for name in EW_METRICS:
        torch.testing.assert_close(actual[name].cpu().double(), expected[name], rtol=2e-5, atol=2e-6)
    for name in ("observation_count", "first_return_end_ms", "last_return_end_ms"):
        assert torch.equal(sidecar[name].cpu(), metadata[name])
    relative["first_eq_ts"].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="clocks disagree"):
        exponential_equity_metrics(relative, run, data, EW_METRICS, relative_timestamps=True)
    relative["last_eq_ts"].fill_(torch.nan)
    missing, sidecar = exponential_equity_metrics(relative, run, data, EW_METRICS, relative_timestamps=True)
    assert sidecar["observation_count"].item() == 0
    assert all(value.item() == -torch.inf for value in missing.values())


def test_metal_entry_interval_quantiles_match_integer_rank_reference():
    from optimization.gpu.entry_intervals import entry_intervals_from_counts
    counts = torch.tensor([[7, 0, 1, 2, 1, 3], [0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    expected = entry_intervals_from_counts(counts, 300_000)
    actual = entry_intervals_from_counts(counts.to("mps"), 300_000)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_metal_native_parity_uses_shared_service(strategy, sides):
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_METRICS, DEFAULT_TOLERANCES
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "2", "--sides", sides, "--bars", "512",
    ]))
    report = run_comparison(inputs, "binance", DEFAULT_METRICS, DEFAULT_TOLERANCES, diagnostics=True)
    assert report["gpu_engine"] == "native" and report["gpu_replay"] == "shared_account"
    assert report["status"] == "passed", report


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_metal_native_ew_only_omits_legacy_history(monkeypatch, strategy):
    from test_gpu_exponential_native_cuda import test_native_ew_only_omits_legacy_history_and_nested_analysis
    from optimization.gpu.native import NativeBacktestService
    test_native_ew_only_omits_legacy_history_and_nested_analysis(
        monkeypatch, strategy, service_class=NativeBacktestService)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
def test_metal_factual_hsl_matches_full_dispatch_through_policy_transitions(monkeypatch, strategy, mode):
    import backtest
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_hsl_factual_replay import _equal

    def forbidden(*args, **kwargs):
        pytest.fail("Metal HSL qualification cannot invoke a CPU simulation")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    sides = ("long", "short")
    proxy = make_proxy(mode, strategy, sides, minutes=256, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    assert runner.mps_coin_capacity == 2
    runner.max_dispatch_candidate_bars = None
    _, full = raw(proxy, [{}])
    full = {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in full.items()}
    runner.max_dispatch_candidate_bars = 64
    runner.hsl_fact_capacity_learned = 1
    _, chunked = raw(proxy, [{}])
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    assert runner.last_hsl_fact_retries > 0
    learned = runner.hsl_fact_capacity_learned
    _equal(chunked, full)
    disabled = {f"{side}_hsl_enabled": 0.0 for side in sides}
    raw(proxy, [disabled])
    assert runner.hsl_fact_capacity == 0
    assert runner.hsl_fact_capacity_learned == learned
    _, restored = raw(proxy, [{}])
    assert runner.hsl_fact_capacity == learned
    assert runner.last_hsl_fact_retries == 0
    _equal(restored, full)
