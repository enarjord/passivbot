"""Explicit daily EW contract and optimizer legacy-pass ablation regressions."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config.ew import EW_METRICS
from optimization.gpu.exponential_equity import DAY_MS, exponential_equity_metrics
from optimization.gpu.metric_registry import (
    RAW_STRATEGY_EQUITY_METRICS,
)
from optimization.gpu.metrics import compute_objectives, validate_gpu_metric_names


def fixture(equities=(100.0, 110.0, 99.0, 118.8), *, origin=0, first=0, last=None):
    values = torch.tensor([equities], dtype=torch.float64)
    last = len(equities) * DAY_MS - 60_000 if last is None else last
    zero = torch.zeros(1, dtype=torch.float64)
    out = {
        "day_end_eq": values,
        "day_min_eq": values.clone(),
        "raw_strategy_day_end_eq": values.clone(),
        "raw_strategy_day_min_eq": values.clone(),
        "btc_day_end_eq": values.clone() / 10,
        "btc_day_min_eq": values.clone() / 10,
        "day_max_dd": torch.zeros_like(values),
        "day_volume": torch.zeros_like(values),
        "day_has_fill": torch.zeros_like(values, dtype=torch.bool),
        "first_eq_ts": torch.tensor([origin + first], dtype=torch.float64),
        "last_eq_ts": torch.tensor([origin + last], dtype=torch.float64),
        "last_high_ts": torch.tensor([origin + last], dtype=torch.float64),
        "first_fill_ts": torch.full_like(zero, torch.nan),
        "last_fill_ts": torch.full_like(zero, torch.nan),
        "fill_count": zero,
        "max_dd": zero,
        "held_max_ms": zero,
        "gap_max_ms": zero,
        "recovery_max_ms": zero,
        "liq_step": torch.tensor([-1]),
    }
    run = SimpleNamespace(interval_ms=60_000, requested_start_ts_ms=origin)
    data = {"ts0": origin, "n": len(equities) * 1440, "ew_half_life_days": 1.0}
    return out, run, data


def test_all_twelve_metrics_match_hand_calculated_daily_return_weights():
    out, run, data = fixture()
    values, metadata = exponential_equity_metrics(out, run, data, EW_METRICS)
    returns = (0.1, -0.1, 0.2)
    p = (1 / 7, 2 / 7, 4 / 7)
    mean = sum(weight * ret for weight, ret in zip(p, returns))
    variance = sum(weight * (ret - mean) ** 2 for weight, ret in zip(p, returns))
    downside = sum(weight * min(ret, 0) ** 2 for weight, ret in zip(p, returns))
    expected = {
        "adg": math.expm1(sum(weight * math.log1p(ret) for weight, ret in zip(p, returns))),
        "sharpe_ratio": mean / math.sqrt(variance),
        "sortino_ratio": mean / math.sqrt(downside),
        "omega_ratio": (p[0] * .1 + p[2] * .2) / (p[1] * .1),
    }
    assert set(values) == EW_METRICS
    for suffix in ("_strategy_eq_ew", "_ew_usd", "_ew_btc"):
        for stem, expected_value in expected.items():
            assert values[stem + suffix].item() == pytest.approx(expected_value, abs=2e-14)
    assert metadata["observation_count"].item() == 3
    assert metadata["effective_sample_size"].item() == pytest.approx(49 / 21)
    assert metadata["first_return_end_ms"].item() == 2 * DAY_MS
    assert metadata["last_return_end_ms"].item() == 4 * DAY_MS
    assert all(value["status"].item() == 0 for value in metadata["curves"].values())


def test_warmup_partial_days_and_boundary_sample_are_excluded():
    out, run, data = fixture(first=DAY_MS + 60_000, last=3 * DAY_MS + 60_000)
    values, metadata = exponential_equity_metrics(out, run, data, {"adg_ew_usd"})
    # Day 1's partial history still supplies its last candle as the boundary
    # for complete day 2; partial day 3 contributes no return.
    assert metadata["observation_count"].item() == 1
    assert metadata["first_return_end_ms"].item() == 3 * DAY_MS
    assert metadata["curves"]["usd"]["status"].item() == 1
    assert values["adg_ew_usd"].item() == -math.inf


@pytest.mark.parametrize("equities,expected", [
    ((100, 100, 100, 100), {"adg": 0, "sharpe_ratio": 0, "sortino_ratio": 0, "omega_ratio": 0}),
    ((100, 110, 121, 133.1), {"adg": .1, "sharpe_ratio": 1000, "sortino_ratio": 1000, "omega_ratio": 1000}),
])
def test_flat_and_constant_positive_returns(equities, expected):
    out, run, data = fixture(equities)
    names = {name + "_ew_usd" for name in expected}
    values, _ = exponential_equity_metrics(out, run, data, names)
    for stem, value in expected.items():
        assert values[stem + "_ew_usd"].item() == pytest.approx(value)


def test_distinct_raw_usd_btc_equity_provenance():
    out, run, data = fixture((100, 100, 100, 100))
    out["raw_strategy_day_end_eq"] = torch.tensor([[100., 110., 121., 133.1]])
    out["raw_strategy_day_min_eq"] = out["raw_strategy_day_end_eq"].clone()
    out["btc_day_end_eq"] = torch.tensor([[100., 90., 81., 72.9]], dtype=torch.float64)
    out["btc_day_min_eq"] = out["btc_day_end_eq"].clone()
    names = {"adg_strategy_eq_ew", "adg_ew_usd", "adg_ew_btc"}
    values, _ = exponential_equity_metrics(out, run, data, names)
    assert values["adg_strategy_eq_ew"].item() == pytest.approx(.1, abs=1e-7)
    assert values["adg_ew_usd"].item() == 0
    assert values["adg_ew_btc"].item() == pytest.approx(-.1)


@pytest.mark.parametrize("key", ["day_end_eq", "day_min_eq"])
def test_active_nonfinite_daily_observation_is_fatal(key):
    out, run, data = fixture()
    out[key][0, 1] = math.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        exponential_equity_metrics(out, run, data, {"adg_ew_usd"})


def test_missing_required_btc_and_raw_curves_are_fatal():
    out, run, data = fixture()
    for key, metric in (("btc_day_end_eq", "adg_ew_btc"), ("raw_strategy_day_min_eq", "adg_strategy_eq_ew")):
        saved = out.pop(key)
        with pytest.raises(RuntimeError, match="factual curve output is missing"):
            exponential_equity_metrics(out, run, data, {metric})
        out[key] = saved


def test_intraday_nonpositive_equity_on_partial_day_is_unavailable():
    out, run, data = fixture(last=3 * DAY_MS + 60_000)
    out["raw_strategy_day_min_eq"][0, -1] = -1
    values, metadata = exponential_equity_metrics(out, run, data, {"adg_strategy_eq_ew"})
    assert metadata["observation_count"].item() == 2
    assert metadata["curves"]["strategy_eq"]["status"].item() == 2
    assert values["adg_strategy_eq_ew"].item() == -math.inf


def test_liquidation_cannot_return_an_ew_score_from_earlier_complete_days():
    out, run, data = fixture()
    out["liq_step"][0] = 4321
    values, metadata = exponential_equity_metrics(out, run, data, EW_METRICS)
    assert all(value.item() == -math.inf for value in values.values())
    assert all(value["status"].item() == 3 for value in metadata["curves"].values())


def test_relative_f32_clocks_restore_long_candle_grid():
    origin = 20_000 * DAY_MS
    out, run, data = fixture([100.] * 400, origin=origin)
    absolute, absolute_meta = exponential_equity_metrics(out, run, data, {"adg_ew_usd"})
    out["first_eq_ts"] = (out["first_eq_ts"] - origin).float()
    out["last_eq_ts"] = (out["last_eq_ts"] - origin).float()
    relative, relative_meta = exponential_equity_metrics(out, run, data, {"adg_ew_usd"}, relative_timestamps=True)
    assert torch.equal(absolute["adg_ew_usd"], relative["adg_ew_usd"])
    assert absolute_meta["observation_count"].item() == relative_meta["observation_count"].item() == 399


def test_ew_only_objectives_bypass_every_legacy_multi_pass_helper(monkeypatch):
    import optimization.gpu.metrics as gpu_metrics
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy nested suffix analysis must be skipped")
    for name in ("_weighted_subsets", "_weighted_subset_context", "_weighted_strategy_eq_metrics",
                 "_weighted_daily_series_metrics", "_weighted_pnl_metrics", "_weighted_adg"):
        assert not hasattr(gpu_metrics, name)
    out, run, data = fixture()
    values = compute_objectives(out, run, data, needed=EW_METRICS)
    expected, _ = exponential_equity_metrics(out, run, data, EW_METRICS)
    assert set(values) == EW_METRICS
    assert all(torch.equal(values[name], expected[name]) for name in EW_METRICS)


def test_supplied_native_ew_scalars_skip_reduction(monkeypatch):
    import optimization.gpu.metrics as gpu_metrics
    out, run, data = fixture()
    out["adg_ew_usd"] = torch.tensor([.375], dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError("native EW scalar already reduced")
    monkeypatch.setattr(gpu_metrics, "exponential_equity_metrics", forbidden)
    assert compute_objectives(out, run, data, needed={"adg_ew_usd"})["adg_ew_usd"].item() == .375


def test_metric_registry_adds_ew_without_weighted_capture_or_aliases():
    assert validate_gpu_metric_names(EW_METRICS) == EW_METRICS
    assert {name for name in EW_METRICS if name.endswith("strategy_eq_ew")} <= RAW_STRATEGY_EQUITY_METRICS


def test_disabled_half_life_rejects_explicit_ew_request():
    out, run, data = fixture()
    data["ew_half_life_days"] = None
    with pytest.raises(ValueError, match="ew_half_life_days"):
        compute_objectives(out, run, data, needed={"adg_ew_usd"})


@pytest.mark.parametrize("proxy_name", ["SharedAccountReplay"])
def test_exponential_metal_still_requires_configured_half_life(monkeypatch, proxy_name):
    from config.schema import get_template_config
    from optimization.gpu import service

    monkeypatch.setattr(service, "gpu_device", lambda *_args: "mps")
    config = get_template_config()
    config["backtest"]["ew_half_life_days"] = None
    with pytest.raises(ValueError, match="require backtest.ew_half_life_days"):
        getattr(service, proxy_name)(config=config, hlcvs=None, mss={}, btc=None,
                                    timestamps=None, exchange="binance", batch_size=1,
                                    needed_metrics={"adg_strategy_eq_ew"})


def test_metadata_sidecar_materializes_status_and_unavailable_clocks():
    from optimization.gpu.service import _exponential_metadata_row, _exponential_metadata_to_host

    out, run, data = fixture()
    for key, value in list(out.items()):
        out[key] = value.repeat((2,) + (1,) * (value.ndim - 1))
    out["first_eq_ts"][1] = math.nan
    out["last_eq_ts"][1] = math.nan
    _, metadata = exponential_equity_metrics(out, run, data, {"adg_ew_usd"})
    host = _exponential_metadata_to_host(metadata)
    available = _exponential_metadata_row(host, 0)
    unavailable = _exponential_metadata_row(host, 1)
    assert available["curves"]["usd"] == {"status": "available", "available": True}
    assert unavailable["curves"]["usd"] == {"status": "insufficient_observations", "available": False}
    assert unavailable["first_return_end_ms"] is None
    assert unavailable["last_return_end_ms"] is None
    assert unavailable["observation_count"] == 0


@pytest.mark.parametrize("value", [True, 0, .5, math.inf, math.nan])
def test_invalid_half_life_is_rejected(value):
    out, run, data = fixture()
    data["ew_half_life_days"] = value
    with pytest.raises(ValueError, match="finite number >= 1"):
        exponential_equity_metrics(out, run, data, {"adg_ew_usd"})


def test_empty_history_returns_explicit_unavailable_without_nan():
    out, run, data = fixture((0.,))
    out["first_eq_ts"][:] = math.nan
    out["last_eq_ts"][:] = math.nan
    for key in ("day_min_eq", "raw_strategy_day_min_eq", "btc_day_min_eq"):
        out[key][:] = math.inf
    values, metadata = exponential_equity_metrics(out, run, data, EW_METRICS)
    assert metadata["observation_count"].item() == 0
    assert metadata["effective_sample_size"].item() == 0
    assert metadata["first_return_end_ms"].item() == -1
    assert all(value.item() == -math.inf for value in values.values())


def test_batched_candidate_start_and_stop_bounds_are_independent():
    out, run, data = fixture()
    # Same full replay shape, but the second candidate starts after the first
    # complete UTC day and stops during the last UTC day.
    for key, value in list(out.items()):
        out[key] = value.repeat((2,) + (1,) * (value.ndim - 1))
    out["first_eq_ts"][1] = DAY_MS + 60_000
    out["last_eq_ts"][1] = 3 * DAY_MS + 60_000
    values, metadata = exponential_equity_metrics(out, run, data, {"adg_ew_usd"})
    assert metadata["observation_count"].tolist() == [3, 1]
    assert metadata["curves"]["usd"]["status"].tolist() == [0, 1]
    assert math.isfinite(values["adg_ew_usd"][0].item())
    assert values["adg_ew_usd"][1].item() == -math.inf


def test_off_grid_interval_and_malformed_daily_shape_are_fatal():
    out, run, data = fixture()
    run.interval_ms = 70_000
    with pytest.raises(ValueError, match="aligned UTC candle grid"):
        exponential_equity_metrics(out, run, data, {"adg_ew_usd"})
    run.interval_ms = 60_000
    out["day_min_eq"] = torch.ones((1, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="inconsistent shape"):
        exponential_equity_metrics(out, run, data, {"adg_ew_usd"})


@pytest.mark.parametrize("interval_minutes", [1, 5])
@pytest.mark.parametrize("half_life", [1.0, 180.5])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seeded_daily_reduction_matches_current_rust(interval_minutes, half_life, partial, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(pbr, "__is_stub__", False)
    verify_loaded_runtime_extension()
    interval = interval_minutes * 60_000
    origin = 20_000 * DAY_MS
    n = 5 * DAY_MS // interval
    clocks = origin + np.arange(n, dtype=np.int64) * interval
    rng = np.random.default_rng(209)
    account = 1_000 * np.exp(np.cumsum(rng.normal(0.00001, 0.0004, n)))
    raw = 1_000 * np.exp(np.cumsum(rng.normal(0.00002, 0.0005, n)))
    btc = account / (30_000 * np.exp(np.cumsum(rng.normal(0, 0.0007, n))))
    first = 13 if partial else 0
    stop = n - 19 if partial else n
    out = {
        "first_eq_ts": torch.tensor([clocks[first]], dtype=torch.float64, device=device),
        "last_eq_ts": torch.tensor([clocks[stop - 1]], dtype=torch.float64, device=device),
        "liq_step": torch.tensor([-1], device=device),
    }
    run = SimpleNamespace(interval_ms=interval)
    data = {"ts0": origin, "n": n, "ew_half_life_days": half_life}
    canonical = {}
    for curve, equities, end_key, min_key in (
        ("usd", account, "day_end_eq", "day_min_eq"),
        ("strategy_eq", raw, "raw_strategy_day_end_eq", "raw_strategy_day_min_eq"),
        ("btc", btc, "btc_day_end_eq", "btc_day_min_eq"),
    ):
        daily_ends, daily_minima = [], []
        for day in range(5):
            mask = (clocks[first:stop] // DAY_MS) == origin // DAY_MS + day
            samples = equities[first:stop][mask]
            daily_ends.append(samples[-1])
            daily_minima.append(samples.min())
        out[end_key] = torch.tensor([daily_ends], dtype=torch.float64, device=device)
        out[min_key] = torch.tensor([daily_minima], dtype=torch.float64, device=device)
        canonical[curve] = pbr.calculate_ew_metrics(
            equities[first:stop].tolist(), clocks[first:stop].tolist(), half_life, interval_minutes,
        )
    actual, metadata = exponential_equity_metrics(out, run, data, EW_METRICS)
    assert all(value.device.type == device for value in actual.values())
    for curve, reference in canonical.items():
        assert reference["metadata"]["status"] == "available"
        suffix = "_strategy_eq_ew" if curve == "strategy_eq" else "_ew_" + curve
        for stem, expected in reference["metrics"].items():
            assert actual[stem + suffix].item() == pytest.approx(expected, rel=1e-11, abs=1e-13)
        for key in ("observation_count", "first_return_end_ms", "last_return_end_ms"):
            assert metadata[key].item() == reference["metadata"][key]
        assert metadata["effective_sample_size"].item() == pytest.approx(reference["metadata"]["effective_sample_size"], abs=1e-13)
