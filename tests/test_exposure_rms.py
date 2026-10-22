"""Candle RMS definitions, runtime budgets, and canonical scoring contracts."""
from copy import deepcopy
import math

import numpy as np
import pytest

from config.metrics import EXPOSURE_RMS_METRICS, canonicalize_metric_name
from config.scoring import default_objective_goal, extract_objective_specs
from config.schema import get_template_config
from metrics_schema import MetricAggregationError, build_scenario_metrics


NAMES = (
    "total_wallet_exposure_rms_long", "total_wallet_exposure_rms_short",
    "total_wallet_exposure_rms_gross", "total_wallet_exposure_utilization_rms_long",
    "total_wallet_exposure_utilization_rms_short", "position_exposure_utilization_rms_long",
    "position_exposure_utilization_rms_short",
)


def fixture_inputs(strategy="trailing_martingale", sides="both", scenario="ordinary"):
    from tools.gpu_parity import build_parser, fixture_inputs as make_inputs
    inputs = make_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "3", "--bars", "2880",
        "--seed", "209",
    ]))
    config, candles, markets, _, _ = inputs
    if scenario == "budgets":
        for side in ("long", "short"):
            config["bot"][side]["risk"]["we_excess_allowance_pct"] = 0.5
        config["coin_overrides"] = {"COIN00": {"bot": {
            side: {"risk": {"wallet_exposure_limit": 0.2}}
            for side in ("long", "short")
        }}}
        # A later listing changes runtime allocation.
        markets["COIN02"]["first_valid_index"] = 720
        markets["COIN02"]["warmup_minutes"] = 60
        candles[:720, 2, :] = np.nan
    elif scenario == "flat":
        config["backtest"]["filter_by_min_effective_cost"] = True
        for coin in ("COIN00", "COIN01", "COIN02"):
            markets[coin]["min_cost"] = 1e9
    return inputs


def run_cpu(inputs):
    from backtest import build_backtest_payload, execute_backtest
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps = inputs
    payload = build_backtest_payload(candles, markets, deepcopy(config), "binance",
                                     btc, timestamps, metrics_only=False)
    fills, equities, analysis = execute_backtest(payload, config)
    return payload, fills, equities, analysis


def reference_moments(inputs, payload, fills, equities):
    """Independent fill-led reconstruction of each recorded candle's inventory."""
    config, candles, markets, _, timestamps = inputs
    coins = payload.backtest_params["coins"]
    params = payload.bot_params_list
    limits = [params[0][side]["total_wallet_exposure_limit"] for side in ("long", "short")]
    slots = [params[0][side]["n_positions"] for side in ("long", "short")]
    budgets = np.array([[p[side]["wallet_exposure_limit"] if p[side]["wallet_exposure_limit"] >= 0
                         else limits[s] / slots[s] if slots[s] else 0.0
                         for p in params] for s, side in enumerate(("long", "short"))])
    max_seen = [0, 0]
    sizes, prices = np.zeros((2, len(coins))), np.zeros((2, len(coins)))
    balance = config["backtest"]["starting_balance"]
    cursor = 0
    values = []
    start = payload.backtest_params["trade_start_indices"]
    last = payload.backtest_params["last_valid_indices"]
    for timestamp in equities[:, 0]:
        k = int(np.searchsorted(timestamps, timestamp))
        while cursor < len(fills) and int(fills[cursor][0]) <= k:
            fill = fills[cursor]
            s = 1 if str(fill[13]).endswith("short") else 0
            c = coins.index(str(fill[2]))
            balance = float(fill[5])
            sizes[s, c], prices[s, c] = abs(float(fill[11])), float(fill[12])
            cursor += 1
        for s, side in enumerate(("long", "short")):
            eligible = [c for c, p in enumerate(params)
                        if start[c] <= k <= last[c] and p[side]["entry_eligible"]
                        and p[side]["wallet_exposure_limit"] != 0.0
                        and not np.isnan(candles[k, c, :3]).all()]
            max_seen[s] = max(max_seen[s], len(eligible))
            effective = min(slots[s], max_seen[s]) if config["backtest"]["dynamic_wel_by_tradability"] else slots[s]
            for c in eligible:
                if params[c][side]["wallet_exposure_limit"] < 0:
                    budgets[s, c] = limits[s] / effective if effective else 0.0
        multipliers = np.array([markets[coin]["c_mult"] for coin in coins])
        e = sizes * prices * multipliers / balance
        total = e.sum(axis=1)
        ratios = [total[s] / limits[s] if total[s] else 0.0 for s in (0, 1)]
        position = [sum(e[s, c] ** 2 / (limits[s] * budgets[s, c])
                        for c in range(len(coins)) if e[s, c]) for s in (0, 1)]
        values.append([total[0] ** 2, total[1] ** 2, sum(total) ** 2,
                       ratios[0] ** 2, ratios[1] ** 2, *position])
    return dict(zip(NAMES, np.sqrt(np.mean(values, axis=0))))


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("scenario", ["ordinary", "budgets"])
def test_real_rust_candle_metrics_match_independent_inventory_reconstruction(strategy, sides, scenario):
    inputs = fixture_inputs(strategy, sides, scenario)
    payload, fills, equities, analysis = run_cpu(inputs)
    assert len(equities) > 1000
    assert len(fills) > 0
    expected = reference_moments(inputs, payload, fills, equities)
    assert {name: analysis[name] for name in NAMES} == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert not {f"{name}_{suffix}" for name in NAMES for suffix in ("usd", "btc", "w")} & analysis.keys()


def test_rms_names_goals_defaults_and_nonfinite_aggregation():
    from tools.gpu_parity import DEFAULT_TOLERANCES
    assert EXPOSURE_RMS_METRICS == frozenset(NAMES)
    for name in NAMES:
        assert canonicalize_metric_name(name) == name
        assert default_objective_goal(name) == "min"
        assert DEFAULT_TOLERANCES[name].absolute == 1e-5
        assert DEFAULT_TOLERANCES[name].relative == 1e-4
        assert build_scenario_metrics({"venue": {name: 0.0}})["stats"][name]["mean"] == 0.0
        with pytest.raises(MetricAggregationError, match="non-finite"):
            build_scenario_metrics({"venue": {name: math.nan}})
    assert not EXPOSURE_RMS_METRICS & {s.metric for s in extract_objective_specs(get_template_config())}


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_real_rust_flat_history_includes_idle_candles(strategy):
    _, fills, equities, analysis = run_cpu(fixture_inputs(strategy, "both", "flat"))
    assert len(equities) > 1000
    assert len(fills) == 0
    assert [analysis[name] for name in NAMES] == [0.0] * 7


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("optional_cols", [0, 2, 4, 6])
def test_native_scalar_transport_preserves_legacy_tail_and_optional_columns(fused, optional_cols):
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _decode_outputs, _decode_multicoin_fused_outputs
    from optimization.gpu.metrics import validate_gpu_metric_names
    from optimization.gpu.service import CORE_OUTPUT_KEYS
    assert validate_gpu_metric_names(NAMES) == frozenset(NAMES)
    assert EXPOSURE_RMS_METRICS <= CORE_OUTPUT_KEYS
    columns = (67 if fused else 62) + optional_cols
    scalars = torch.arange(columns + 7, dtype=torch.float32)[None, :]
    scalars[0, 9] = -1
    daily = torch.ones((1, 1, 9))
    gaps = torch.zeros((1, 512))
    decode = _decode_multicoin_fused_outputs if fused else _decode_outputs
    legacy = decode(daily, scalars[:, :-7], gaps)
    native = decode(daily, scalars, gaps, native_factual_hsl=True, exposure_rms_enabled=True)
    native_without_rms = decode(daily, scalars[:, :-7], gaps, native_factual_hsl=True)
    assert not EXPOSURE_RMS_METRICS & native_without_rms.keys()
    assert native_without_rms["hsl_drawdown_ema_max_portfolio"].item() == scalars[0, 17].item()
    for key in legacy:
        if key in {"first_eq_ts", "last_eq_ts"}:
            assert torch.equal(native[key], legacy[key] * 60_000), key
            prefix, column = ("first_eq", 7) if key == "first_eq_ts" else ("last_eq", 8)
            assert torch.equal(native[f"{prefix}_step"], scalars[:, column].to(torch.int64))
        else:
            assert torch.equal(native[key], legacy[key]), key
    assert [native[name].item() for name in NAMES] == list(range(columns, columns + 7))


def test_utilization_objectives_and_absolute_limits_use_canonical_scorer():
    from optimize import Evaluator
    from optimization.native_results import CanonicalResultScorer
    config = get_template_config()
    metric, limit = NAMES[3], NAMES[2]
    config["optimize"]["scoring"] = [{"metric": metric, "goal": "min"}]
    config["optimize"]["limits"] = [{"metric": limit, "penalize_if": "greater_than", "value": 0.5}]
    evaluator = Evaluator(hlcvs_specs={"binance": None}, btc_usd_specs={}, msss={}, config=config)
    scorer = CanonicalResultScorer(evaluator)
    assert scorer.required_metrics("base") == {metric, limit}
    low = scorer.score([], {"base": {"binance": {metric: 0.2, limit: 0.4, "liquidated": False}}})
    high = scorer.score([], {"base": {"binance": {metric: 0.3, limit: 0.6, "liquidated": False}}})
    assert high["constraint_violation"] > low["constraint_violation"]


def test_canonical_config_accepts_utilization_objectives_and_absolute_limits():
    from config import prepare_config
    config = get_template_config()
    config["optimize"]["bounds"] = {}
    objectives = sorted(name for name in NAMES if "utilization" in name)
    config["optimize"]["scoring"] = [{"metric": name} for name in objectives]
    config["optimize"]["limits"] = [{"metric": NAMES[2], "penalize_if": "greater_than", "value": 0.7}]
    prepared = prepare_config(config, verbose=False)
    assert [(spec.metric, spec.goal) for spec in extract_objective_specs(prepared)] == [
        (name, "min") for name in objectives
    ]
    assert prepared["optimize"]["limits"] == config["optimize"]["limits"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_shader_compiles_rms_only_when_requested(monkeypatch, strategy, enabled):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel
    builder = getattr(mps_kernel, f"_{strategy}_multicoin_shader_library")
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "cuda")
    monkeypatch.setattr(mps_kernel, "compile_shader", lambda source, **_: source)
    builder.cache_clear()
    try:
        source = builder(hsl_factual_only=True, exposure_rms_enabled=enabled)
        assert f"#define PASSIVBOT_EXPOSURE_RMS_ENABLED {int(enabled)}\n" in source
        assert "#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n" in source
    finally:
        builder.cache_clear()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("native,device,accepted", [
    (True, "cuda", True), (True, "mps", False), (False, "cuda", False),
    (False, "mps", False),
])
def test_rms_gpu_capabilities_are_native_cuda_only(name, native, device, accepted):
    pytest.importorskip("torch")
    from optimization.gpu.metrics import validate_gpu_metric_names
    if accepted:
        assert validate_gpu_metric_names([name], native_replay=native, device=device) == {name}
    else:
        with pytest.raises(ValueError, match="require native NVIDIA/CUDA replay"):
            validate_gpu_metric_names([name], native_replay=native, device=device)
    assert validate_gpu_metric_names(["adg_strategy_eq"], native_replay=native, device=device) == {"adg_strategy_eq"}


@pytest.mark.parametrize("name", NAMES)
def test_legacy_parity_rejects_rms_before_preparing_inputs(monkeypatch, capsys, name):
    import json
    pytest.importorskip("torch")
    from tools import gpu_parity
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported metrics must be rejected before input preparation or replay")
    monkeypatch.setattr(gpu_parity, "fixture_inputs", unexpected)
    monkeypatch.setattr(gpu_parity, "run_comparison", unexpected)
    assert gpu_parity.main(["--fixture", "ema_anchor", "--gpu-engine", "legacy", "--metrics", name]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unsupported"
    assert report["stage"] == "metric_contract"
    assert "native NVIDIA/CUDA" in report["error"]["message"]


def test_legacy_parity_rejects_rms_limit_before_replay(monkeypatch, capsys):
    import json
    pytest.importorskip("torch")
    from tools import gpu_parity
    config = get_template_config()
    config["optimize"]["limits"] = [{"metric": NAMES[2], "penalize_if": "greater_than", "value": 0.5}]
    monkeypatch.setattr(gpu_parity, "fixture_inputs", lambda *_: (config,))
    monkeypatch.setattr(gpu_parity, "run_comparison", lambda *_a, **_k: pytest.fail("unsupported limit reached replay"))
    assert gpu_parity.main(["--fixture", "ema_anchor", "--gpu-engine", "legacy", "--metrics", "adg_strategy_eq"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unsupported"
    assert report["stage"] == "metric_contract"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("proxy_kind,native,device", [
    ("single", False, "cuda"), ("multi", False, "cuda"), ("multi", True, "mps"),
])
def test_unsupported_rms_service_requests_fail_before_payload_preparation(monkeypatch, name, proxy_kind, native, device):
    import numpy as np
    pytest.importorskip("torch")
    from optimization.gpu import service
    monkeypatch.setattr(service, "gpu_device", lambda *_: device)
    proxy = service.MpsSingleCoinProxy if proxy_kind == "single" else service.MpsMulticoinProxy
    extra = {"factual_hsl": native} if proxy_kind == "multi" else {}
    import backtest
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported RMS capability must reject before payload preparation")
    monkeypatch.setattr(backtest, "build_backtest_payload", forbidden)
    message = "NativeBacktestService" if proxy_kind == "single" else "require native NVIDIA/CUDA replay"
    with pytest.raises(ValueError, match=message):
        proxy(config=get_template_config(), hlcvs=np.ones((2, 2, 4)), mss={},
              btc=np.ones(2), timestamps=np.arange(2) * 60_000, exchange="binance",
              batch_size=1, needed_metrics=[name], **extra)
