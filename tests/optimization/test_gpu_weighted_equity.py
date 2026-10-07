"""Controlled cross-producer references; histories remain on the selected device."""

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from optimization.gpu import weighted_equity as reducer

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "gpu_weighted_equity.json").read_text()
)


def _inputs(cases, *, device, dtype, curve, variant="initial_fill"):
    capacity = max((len(case["equities"]) for case in cases), default=0)
    samples = torch.full((len(cases), capacity), float("nan"), dtype=dtype, device=device)
    for row, case in enumerate(cases):
        samples[row, :len(case["equities"])] = torch.tensor(case["equities"], dtype=dtype, device=device)
    kwargs = dict(
        first_timestamps_ms=torch.tensor([
            case["first_timestamp_ms"] if case["equities"] else float("nan")
            for case in cases
        ], dtype=torch.float64, device=device),
        sample_counts=torch.tensor([len(case["equities"]) for case in cases], dtype=torch.long, device=device),
        interval_ms=cases[0]["interval_ms"],
        n_days=max((case["first_timestamp_ms"] + max(0, len(case["equities"]) - 1) * case["interval_ms"]) // 86_400_000
                   - case["first_timestamp_ms"] // 86_400_000 + 1 for case in cases),
        curve=curve,
    )
    if curve == "account":
        kwargs["fill_counts"] = torch.tensor([len(case["fill_indices"][variant]) for case in cases], device=device)
    return samples, kwargs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("curve,variant", [
    ("raw_strategy", "initial_fill"), ("account", "initial_fill"),
    ("account", "sparse_fills"), ("account", "no_fills"),
])
def test_weighted_equity_current_rust_references(device, dtype, curve, variant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    key = "expected_raw" if curve == "raw_strategy" else "expected_account"
    if dtype == torch.float32:
        key += "_f32"
    # Independent clocks and padding share a dispatch only at equal cadence.
    for interval in {case["interval_ms"] for case in FIXTURE["cases"]}:
        cases = [case for case in FIXTURE["cases"] if case["interval_ms"] == interval]
        expected = [case[key] if curve == "raw_strategy" else case[key][variant] for case in cases]
        samples, kwargs = _inputs(cases, device=device, dtype=dtype, curve=curve, variant=variant)
        outputs = reducer.weighted_equity_from_samples(samples, requested=set(expected[0]), **kwargs)
        assert set(outputs) == set(expected[0])
        for name, values in outputs.items():
            assert values.device == samples.device
            assert values.dtype == torch.float64
            for row, reference in enumerate(expected):
                actual = float(values[row])
                golden = reference[name]
                if golden is None:
                    assert not math.isfinite(actual), (cases[row]["case"], name, actual)
                else:
                    assert actual == pytest.approx(golden, abs=1e-8, rel=1e-12), (
                        cases[row]["case"], name, actual, golden
                    )


@pytest.mark.parametrize("curve", ["raw_strategy", "account"])
@pytest.mark.parametrize("stem", reducer.WEIGHTED_EQUITY_STEMS)
def test_requested_subset_matches_full_reduction(curve, stem):
    case = next(c for c in FIXTURE["cases"] if c["case"] == "partial_day_dip")
    samples, kwargs = _inputs([case], device="cpu", dtype=torch.float64, curve=curve)
    expected = case["expected_raw"] if curve == "raw_strategy" else case["expected_account"]["initial_fill"]
    name = f"{stem}_strategy_eq_w" if curve == "raw_strategy" else f"{stem}_w_usd"
    full = reducer.weighted_equity_from_samples(samples, requested=expected, **kwargs)
    subset = reducer.weighted_equity_from_samples(samples, requested=[name], **kwargs)
    assert set(subset) == {name}
    torch.testing.assert_close(subset[name], full[name], rtol=0, atol=0)


def test_requested_reduction_skips_unused_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unused weighted reducer executed")
    for name in ("_smoothed_adg", "_sharpe_sortino", "_mean_worst_one_pct_largest", "_omega_ratio"):
        monkeypatch.setattr(reducer, name, forbidden)
    case = next(c for c in FIXTURE["cases"] if c["case"] == "discarded_peak")
    samples, kwargs = _inputs([case], device="cpu", dtype=torch.float64, curve="raw_strategy")
    output = reducer.weighted_equity_from_samples(samples, requested=["mdg_strategy_eq_w"], **kwargs)
    assert float(output["mdg_strategy_eq_w"][0]) == pytest.approx(case["expected_raw"]["mdg_strategy_eq_w"])
    # An unrequested family allocates nothing and never consumes a history.
    assert reducer.weighted_equity_from_samples(None, requested=[], curve="raw_strategy",
        first_timestamps_ms=None, sample_counts=None, interval_ms=None, n_days=None) == {}


@pytest.mark.parametrize("problem", ["count_negative", "count_overflow", "fractional_count",
    "clock_nan", "clock_negative", "sample_nan", "sample_inf", "calendar_overflow",
    "calendar_bool", "interval_zero", "metadata_shape", "fill_negative", "fill_nan", "fill_missing"])
def test_required_history_inputs_reject_invalid_data(problem):
    case = next(c for c in FIXTURE["cases"] if c["case"] == "short_4")
    samples, kwargs = _inputs([case], device="cpu", dtype=torch.float64, curve="account")
    if problem == "count_negative": kwargs["sample_counts"][0] = -1
    elif problem == "count_overflow": kwargs["sample_counts"][0] = 5
    elif problem == "fractional_count": kwargs["sample_counts"] = torch.tensor([3.5])
    elif problem == "clock_nan": kwargs["first_timestamps_ms"][0] = float("nan")
    elif problem == "clock_negative": kwargs["first_timestamps_ms"][0] = -1
    elif problem == "sample_nan": samples[0, 2] = float("nan")
    elif problem == "sample_inf": samples[0, 2] = float("inf")
    elif problem == "calendar_overflow": kwargs["n_days"] = 3
    elif problem == "calendar_bool": kwargs["n_days"] = True
    elif problem == "interval_zero": kwargs["interval_ms"] = 0
    elif problem == "metadata_shape": kwargs["first_timestamps_ms"] = torch.zeros(2)
    elif problem == "fill_negative": kwargs["fill_counts"][0] = -1
    elif problem == "fill_nan": kwargs["fill_counts"] = torch.tensor([float("nan")])
    elif problem == "fill_missing": del kwargs["fill_counts"]
    with pytest.raises(ValueError):
        reducer.weighted_equity_from_samples(samples, requested=["adg_w_usd"], **kwargs)


def test_curve_and_metric_ownership_is_explicit():
    case = next(c for c in FIXTURE["cases"] if c["case"] == "short_2")
    samples, kwargs = _inputs([case], device="cpu", dtype=torch.float64, curve="account")
    with pytest.raises(ValueError, match="unsupported"):
        reducer.weighted_equity_from_samples(samples, requested=["adg_strategy_eq_w"], **kwargs)
    kwargs["curve"] = "implicit"
    with pytest.raises(ValueError, match="explicit"):
        reducer.weighted_equity_from_samples(samples, requested=[], **kwargs)


def test_empty_history_capacity_returns_defaults_without_a_clock():
    samples = torch.empty((3, 0), dtype=torch.float64)
    out = reducer.weighted_equity_from_samples(samples,
        first_timestamps_ms=torch.full((3,), float("nan")), sample_counts=torch.zeros(3, dtype=torch.long),
        interval_ms=60000, n_days=1, curve="raw_strategy", requested=["adg_strategy_eq_w"])
    torch.testing.assert_close(out["adg_strategy_eq_w"], torch.zeros(3, dtype=torch.float64))


def test_late_first_observation_padding_does_not_exceed_owned_calendar():
    short = next(c for c in FIXTURE["cases"] if c["case"] == "short_2")
    cases = [
        {**short, "first_timestamp_ms": 23 * 3_600_000, "interval_ms": 21_600_000},
        {**short, "first_timestamp_ms": 0, "interval_ms": 21_600_000,
         "equities": [100., 101., 102., 103., 104., 105., 106., 107.]},
    ]
    samples, kwargs = _inputs(cases, device="cpu", dtype=torch.float64, curve="raw_strategy")
    assert kwargs["n_days"] == 2
    # The first candidate's unused clock would enter day 2, outside this calendar.
    result = reducer.weighted_equity_from_samples(samples,
        requested=["adg_strategy_eq_w"], **kwargs)
    assert float(result["adg_strategy_eq_w"][0]) == pytest.approx(
        short["expected_raw"]["adg_strategy_eq_w"], abs=1e-8, rel=1e-12
    )


def test_interleaved_resident_curves_are_not_mutated_or_assumed_contiguous():
    case = next(c for c in FIXTURE["cases"] if c["case"] == "discarded_peak")
    samples, kwargs = _inputs([case], device="cpu", dtype=torch.float64, curve="raw_strategy")
    interleaved = torch.stack((samples, samples * 2), dim=2)
    original = interleaved.clone()
    view = interleaved[:, :, 0]
    assert not view.is_contiguous()
    expected = reducer.weighted_equity_from_samples(samples,
        requested=case["expected_raw"], **kwargs)
    actual = reducer.weighted_equity_from_samples(view,
        requested=case["expected_raw"], **kwargs)
    torch.testing.assert_close(interleaved, original, rtol=0, atol=0, equal_nan=True)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)
