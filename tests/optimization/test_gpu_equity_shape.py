"""Daily shape metrics follow Rust references at f64 and f32 input precision."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.metrics import compute_objectives

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "gpu_weighted_equity.json").read_text()
)
SHAPE_METRICS = frozenset(
    f"{stem}{suffix}_usd"
    for stem in ("equity_choppiness", "equity_jerkiness", "exponential_fit_error")
    for suffix in ("", "_w")
)


def _daily_output(case, variant, quantized, device):
    values = np.asarray(case["equities"], dtype=np.float64)
    if quantized:
        values = values.astype(np.float32).astype(np.float64)
    first, interval = case["first_timestamp_ms"], case["interval_ms"]
    times = first + np.arange(len(values), dtype=np.int64) * interval
    days = times // 86_400_000 - first // 86_400_000
    width = int(days[-1]) + 1 if len(days) else 1
    ends = np.zeros(width)
    minima = np.full(width, np.inf)
    for day in np.unique(days):
        samples = values[days == day]
        ends[day], minima[day] = samples[-1], samples.min()
    fills = case["fill_indices"][variant]

    def scalar(value):
        return torch.tensor([value], dtype=torch.float64, device=device)

    return dict(
        day_end_eq=torch.tensor(ends[None, :], dtype=torch.float64, device=device),
        day_min_eq=torch.tensor(minima[None, :], dtype=torch.float64, device=device),
        day_max_dd=torch.zeros((1, width), dtype=torch.float64, device=device),
        day_volume=torch.zeros((1, width), dtype=torch.float64, device=device),
        day_has_fill=torch.zeros((1, width), dtype=torch.bool, device=device),
        fill_count=scalar(len(fills)),
        first_fill_ts=scalar(times[fills[0]] if fills else math.nan),
        last_fill_ts=scalar(times[fills[-1]] if fills else math.nan),
        first_eq_ts=scalar(times[0] if len(times) else math.nan),
        last_eq_ts=scalar(times[-1] if len(times) else math.nan),
        last_high_ts=scalar(math.nan),
        recovery_max_ms=scalar(0),
        held_max_ms=scalar(0),
        gap_max_ms=scalar(0),
        max_dd=scalar(0),
    )


@pytest.mark.parametrize("unit", ["usd", "btc"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantized", [False, True], ids=["f64", "f32"])
@pytest.mark.parametrize("variant", ["initial_fill", "sparse_fills", "no_fills"])
def test_account_shape_matches_current_rust_producer(unit, device, quantized, variant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    key = "expected_account_shape_f32" if quantized else "expected_account_shape"
    for case in FIXTURE["cases"]:
        assert set(case[key][variant]) == SHAPE_METRICS
        out = _daily_output(case, variant, quantized, device)
        needed = {name.removesuffix("usd") + unit for name in SHAPE_METRICS}
        data = {"ts0": case["first_timestamp_ms"], "n": len(case["equities"])}
        if unit == "btc":
            # A unit price makes the actual BTC reducer use the same Rust curve.
            # This checks metric routing, not variable-price conversion or replay.
            data.update(
                btc_prices=np.ones(max(1, data["n"])),
                btc_day_end_price=np.ones(out["day_end_eq"].shape[1]),
            )
        result = compute_objectives(
            out,
            SimpleNamespace(
                interval_ms=case["interval_ms"],
                requested_start_ts_ms=case["first_timestamp_ms"],
            ),
            data,
            needed=needed,
        )
        assert set(result) == needed
        for source, reference in case[key][variant].items():
            name = source.removesuffix("usd") + unit
            actual = float(result[name][0])
            if reference == "positive_infinity":
                assert actual == math.inf, (case["case"], name, actual)
            else:
                assert actual == pytest.approx(reference, abs=1e-10, rel=1e-12), (
                    case["case"], variant, name, actual, reference
                )
