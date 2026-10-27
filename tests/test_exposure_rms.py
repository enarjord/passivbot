"""Independent inventory/mark/budget oracles for the three core exposure RMS metrics."""

from copy import deepcopy
import json

import numpy as np
import pytest

from config.metrics import EXPOSURE_RMS_METRICS, canonicalize_metric_name, reject_retired_metric
from config.scoring import default_objective_goal, extract_objective_specs
from config.schema import get_expanded_config_prototype, get_template_config
from metrics_schema import MetricAggregationError, build_scenario_metrics


NAMES = (
    "gross_exposure_rms_ratio_strategy_equity",
    "position_exposure_utilization_rms_ratio_long",
    "position_exposure_utilization_rms_ratio_short",
)
RETIRED_NAMES = (
    "wallet_exposure_rms_ratio_long", "wallet_exposure_rms_ratio_short",
    "gross_wallet_exposure_rms_ratio", "wallet_exposure_utilization_rms_ratio_long",
    "wallet_exposure_utilization_rms_ratio_short",
)


def fixture_inputs(strategy="trailing_martingale", sides="both", scenario="ordinary"):
    from tools.gpu_parity import build_parser, fixture_inputs as make_inputs

    inputs = make_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "3", "--bars", "2880",
        "--seed", "209",
    ]))
    config, candles, markets, _, _ = inputs
    # Different contract multipliers make a missing conversion visible.
    for coin, multiplier in zip(("COIN00", "COIN01", "COIN02"), (1.0, 2.0, 0.5)):
        markets[coin]["c_mult"] = multiplier
    if scenario == "budgets":
        for side in ("long", "short"):
            config["bot"][side]["risk"]["we_excess_allowance_fraction"] = 0.5
        config["coin_overrides"] = {"COIN00": {"bot": {
            side: {"wallet_exposure_limit_ratio": 0.2}
            for side in ("long", "short")
        }}}
        # A later listing changes contemporaneous nominal allocation.
        markets["COIN02"]["first_valid_index"] = 720
        markets["COIN02"]["warmup_minutes"] = 60
        candles[:720, 2, :] = np.nan
    elif scenario == "flat":
        config["backtest"]["filter_by_min_effective_cost"] = True
        for coin in ("COIN00", "COIN01", "COIN02"):
            markets[coin]["min_cost"] = 1e9
    elif scenario == "inactive":
        # Evaluation begins before trading is admitted, so idle observations count.
        for coin in ("COIN00", "COIN01", "COIN02"):
            markets[coin]["warmup_minutes"] = 720
    return inputs


def run_cpu(inputs):
    from backtest import build_backtest_payload, execute_backtest
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    config, candles, markets, btc, timestamps = inputs
    payload = build_backtest_payload(
        candles, markets, deepcopy(config), "binance", btc, timestamps, metrics_only=False,
    )
    fills, equities, analysis = execute_backtest(payload, config)
    return payload, fills, equities, analysis


def reference_samples(inputs, payload, fills, equities):
    """Reconstruct factual inventory independently at exported baseline/close clocks.

    A close at boundary k contains fills executed in candle k-1, including forced
    closes at that boundary. Ordinary opening fills of candle k occur afterwards.
    These fixtures finish normally, with no same-clock early-terminal replacement.
    """
    config, candles, markets, _, timestamps = inputs
    assert config["backtest"]["btc_collateral_target_ratio"] == 0.0
    coins = payload.backtest_params["coins"]
    params = payload.bot_params_list
    sides = ("long", "short")
    limits = [params[0][side]["total_wallet_exposure_limit_ratio"] for side in sides]
    slots = [params[0][side]["n_positions"] for side in sides]
    budgets = np.array([
        [p[side]["wallet_exposure_limit_ratio"]
         if p[side]["wallet_exposure_limit_ratio"] > 0
         else limits[s] / slots[s] if slots[s] else 0.0 for p in params]
        for s, side in enumerate(sides)
    ])
    multipliers = np.array([markets[coin]["c_mult"] for coin in coins])
    max_seen = [0, 0]
    sizes, prices = np.zeros((2, len(coins))), np.zeros((2, len(coins)))
    marks = np.full(len(coins), np.nan)
    balance = config["backtest"]["starting_balance"]
    cursor = 0
    values, entry_gross = [], []
    start = payload.backtest_params["trade_start_indices"]
    first = payload.backtest_params["first_valid_indices"]
    last = payload.backtest_params["last_valid_indices"]
    for sample, row in enumerate(equities):
        boundary = int(np.searchsorted(timestamps, int(row[0])))
        assert int(timestamps[boundary]) == int(row[0])
        k = boundary - 1
        while cursor < len(fills) and int(fills[cursor][0]) <= k:
            fill = fills[cursor]
            side = 1 if str(fill[13]).endswith("short") else 0
            coin = coins.index(str(fill[2]))
            balance = float(fill[5])
            sizes[side, coin], prices[side, coin] = abs(float(fill[11])), float(fill[12])
            cursor += 1
        valid = [c for c in range(len(coins))
                 if first[c] <= k <= last[c] and np.isfinite(candles[k, c, :3]).all()]
        marks[valid] = candles[k, valid, 2]
        if sample:
            for s, side in enumerate(sides):
                eligible = [c for c in valid
                            if start[c] <= k and params[c][side]["entry_eligible"]
                            and params[c][side]["wallet_exposure_limit_ratio"] != 0.0]
                max_seen[s] = max(max_seen[s], len(eligible))
                effective = (min(slots[s], max_seen[s])
                             if config["backtest"]["dynamic_wel_by_tradability"] else slots[s])
                for c in eligible:
                    if params[c][side]["wallet_exposure_limit_ratio"] < 0:
                        budgets[s, c] = limits[s] / effective if effective else 0.0
        entry_exposure = sizes * prices * multipliers / balance
        utilization_squared = [
            sum(entry_exposure[s, c] ** 2 / (limits[s] * budgets[s, c])
                for c in range(len(coins)) if sizes[s, c])
            for s in (0, 1)
        ]
        held = np.any(sizes != 0.0, axis=0)
        gross = (sizes[:, held] * marks[held] * multipliers[held]).sum() / row[3]
        values.append([gross ** 2, *utilization_squared])
        entry_gross.append(entry_exposure.sum())
    assert cursor == len(fills)
    return np.asarray(values), np.asarray(entry_gross)


def reference_moments(inputs, payload, fills, equities):
    values, _ = reference_samples(inputs, payload, fills, equities)
    return dict(zip(NAMES, np.sqrt(np.mean(values, axis=0))))


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("scenario", ["ordinary", "budgets"])
def test_real_rust_metrics_match_independent_inventory_mark_and_budget_reconstruction(strategy, sides, scenario):
    inputs = fixture_inputs(strategy, sides, scenario)
    payload, fills, equities, analysis = run_cpu(inputs)
    assert len(equities) > 1000 and len(fills) > 0
    expected = reference_moments(inputs, payload, fills, equities)
    assert {name: analysis[name] for name in NAMES} == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert analysis["metric_metadata"]["equity_observation_count"] == len(equities)
    assert not set(RETIRED_NAMES) & analysis.keys()
    assert not {f"{name}_{suffix}" for name in NAMES for suffix in ("usd", "btc", "w")} & analysis.keys()
    if sides != "both":
        inactive = NAMES[2] if sides == "long" else NAMES[1]
        assert analysis[inactive] == 0.0
    _, entry_gross = reference_samples(inputs, payload, fills, equities)
    # Gross marked exposure/strategy equity is distinct from entry exposure/balance.
    assert abs(analysis[NAMES[0]] - np.sqrt(np.mean(entry_gross ** 2))) > 1e-6


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("scenario", ["flat", "inactive"])
def test_real_rust_flat_and_initially_inactive_history_retains_all_observations(strategy, scenario):
    inputs = fixture_inputs(strategy, "both", scenario)
    payload, fills, equities, analysis = run_cpu(inputs)
    assert len(equities) > 1000
    assert equities[0, 0] == inputs[-1][61]
    assert analysis["metric_metadata"]["equity_observation_count"] == len(equities)
    values, _ = reference_samples(inputs, payload, fills, equities)
    if scenario == "flat":
        assert len(fills) == 0
        assert [analysis[name] for name in NAMES] == [0.0] * len(NAMES)
    else:
        assert len(fills) > 0
        assert int(fills[0][0]) >= 720
        assert not values[:600].any()
        assert {name: analysis[name] for name in NAMES} == pytest.approx(
            dict(zip(NAMES, np.sqrt(values.mean(axis=0)))), rel=1e-11, abs=1e-12,
        )


def test_rms_catalog_goals_and_selected_nonfinite_aggregation():
    from tools.gpu_parity import DEFAULT_TOLERANCES

    assert EXPOSURE_RMS_METRICS == frozenset(NAMES)
    for name in NAMES:
        assert canonicalize_metric_name(name) == name
        assert default_objective_goal(name) == "min"
        assert DEFAULT_TOLERANCES[name].absolute == 1e-5
        assert DEFAULT_TOLERANCES[name].relative == 1e-4
        source = {"metric_metadata": {"definition_version": 1}, name: 0.0}
        assert build_scenario_metrics({"venue": source}, required_metrics=[name])["stats"][name]["mean"] == 0.0
        source[name] = float("nan")
        with pytest.raises(MetricAggregationError, match="malformed finite metric"):
            build_scenario_metrics({"venue": source}, required_metrics=[name])
    for name in RETIRED_NAMES:
        with pytest.raises(ValueError, match="unknown or retired core metric"):
            reject_retired_metric(name, path="metric")
    assert not EXPOSURE_RMS_METRICS & {s.metric for s in extract_objective_specs(get_template_config())}


@pytest.mark.parametrize("fused", [False, True])
def test_native_scalar_transport_rejects_retired_exposure_appendage(fused):
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _decode_outputs, _decode_multicoin_fused_outputs

    columns = 67 if fused else 62
    scalars = torch.arange(columns, dtype=torch.float32)[None, :]
    scalars[0, 9] = -1
    daily = torch.ones(1)
    decode = _decode_multicoin_fused_outputs if fused else _decode_outputs
    from inspect import signature
    assert "exposure_rms_enabled" not in signature(decode).parameters
    current = decode(daily, scalars)
    assert not EXPOSURE_RMS_METRICS & current.keys()
    with pytest.raises(ValueError, match="Unsupported current"):
        decode(daily, torch.cat([scalars, torch.zeros((1, 7))], dim=1))


def test_utilization_objective_and_gross_exposure_limit_use_canonical_scorer():
    from optimize import Evaluator
    from optimization.native_results import CanonicalResultScorer

    config = get_expanded_config_prototype()
    metric, limit = NAMES[1], NAMES[0]
    config["optimize"]["scoring"] = [{"metric": metric, "goal": "min"}]
    config["optimize"]["limits"] = [{"metric": limit, "penalize_if": "greater_than", "value": 0.5}]
    evaluator = Evaluator(hlcvs_specs={"binance": None}, btc_usd_specs={}, msss={}, config=config)
    scorer = CanonicalResultScorer(evaluator)
    assert scorer.required_metrics("base") == {metric, limit}
    def source(values):
        return {**values, "liquidated": False, "metric_metadata": {"definition_version": 1}}
    low = scorer.score([], {"base": {"binance": source({metric: 0.2, limit: 0.4})}})
    high = scorer.score([], {"base": {"binance": source({metric: 0.3, limit: 0.6})}})
    assert high["constraint_violation"] > low["constraint_violation"]


def test_canonical_config_accepts_position_utilization_objectives_and_gross_limits():
    from config import prepare_config

    config = get_template_config()
    config["optimize"]["bounds"] = {}
    config["optimize"]["scoring"] = [{"metric": name} for name in NAMES[1:]]
    config["optimize"]["limits"] = [{"metric": NAMES[0], "penalize_if": "greater_than", "value": 0.7}]
    prepared = prepare_config(config, verbose=False)
    assert [(spec.metric, spec.goal) for spec in extract_objective_specs(prepared)] == [
        (name, "min") for name in NAMES[1:]
    ]
    assert prepared["optimize"]["limits"] == config["optimize"]["limits"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_shader_compiles_current_position_budget_support_and_core_mask(monkeypatch, strategy, enabled):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel
    from optimization.gpu.core_metrics import metric_mask

    builder = getattr(mps_kernel, f"_{strategy}_multicoin_shader_library")
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "cuda")
    monkeypatch.setattr(mps_kernel, "compile_shader", lambda source, **_: source)
    builder.cache_clear()
    try:
        mask = metric_mask(NAMES if enabled else ())
        source = builder(hsl_factual_only=True, exposure_rms_enabled=enabled, core_metric_mask=mask)
        assert f"#define PASSIVBOT_EXPOSURE_RMS_ENABLED {int(enabled)}\n" in source
        assert f"#define PASSIVBOT_CORE_REQUEST_MASK {mask}u\n" in source
        assert "#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n" in source
    finally:
        builder.cache_clear()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("native,device,accepted", [
    (True, "cuda", True), (True, "mps", True), (False, "cuda", False),
    (False, "mps", False), (True, "cpu", False),
])
def test_rms_gpu_capabilities_require_current_native_metal_or_cuda(name, native, device, accepted):
    from optimization.gpu.metrics import validate_gpu_metric_names

    if accepted:
        assert validate_gpu_metric_names([name], native_replay=native, device=device) == {name}
    else:
        with pytest.raises(ValueError, match="require native factual replay|require Metal or CUDA"):
            validate_gpu_metric_names([name], native_replay=native, device=device)


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("surface", ["requested", "limit"])
def test_parity_rms_preflight_accepts_current_native_devices_without_replay(monkeypatch, capsys, device, surface):
    pytest.importorskip("torch")
    from tools import gpu_parity
    from optimization.gpu import runtime

    monkeypatch.setattr(runtime, "gpu_device", lambda: device)
    config = get_template_config()
    config["optimize"]["limits"] = (
        [{"metric": NAMES[0], "penalize_if": "greater_than", "value": 0.5}]
        if surface == "limit" else []
    )
    monkeypatch.setattr(gpu_parity, "fixture_inputs", lambda *_: (config,))
    def compared(_inputs, _exchange, metrics, *_args, **_kwargs):
        assert NAMES[0] in metrics
        return {"passed": True}
    monkeypatch.setattr(gpu_parity, "run_comparison", compared)
    requested = NAMES[0] if surface == "requested" else "cagr_fraction_strategy_equity"
    assert gpu_parity.main(["--fixture", "ema_anchor", "--metrics", requested]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]


def test_parity_ordinary_metrics_do_not_probe_device(monkeypatch, capsys):
    from tools import gpu_parity
    from optimization.gpu import runtime

    def forbidden():
        pytest.fail("ordinary metric validation must not probe a GPU device")
    monkeypatch.setattr(runtime, "gpu_device", forbidden)
    config = get_template_config()
    config["optimize"]["limits"] = []
    monkeypatch.setattr(gpu_parity, "fixture_inputs", lambda *_: (config,))
    monkeypatch.setattr(gpu_parity, "run_comparison", lambda *_a, **_k: {"passed": True})
    assert gpu_parity.main(["--fixture", "ema_anchor", "--metrics", "cagr_fraction_strategy_equity"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]


def test_proportional_position_and_nominal_budget_scaling_preserves_native_utilization():
    """Constant marks and zero cashflows isolate nominal-budget normalization."""
    from tools.gpu_parity import build_parser, fixture_inputs as make_inputs

    # End before the nominal exposure cap introduces a quantized last clip.
    inputs = make_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", "long", "--coins", "3", "--bars", "128",
    ]))
    config, candles, markets, _, _ = inputs
    candles[:, :, :3] = [100.0, 99.0, 100.0]
    config["backtest"]["maker_fee_override_fraction"] = 0.0
    config["backtest"]["taker_fee_override_fraction"] = 0.0
    config["bot"]["long"]["strategy"]["ema_anchor"]["inventory_offset_exposure_coefficient"] = 0.0
    for coin in ("COIN00", "COIN01", "COIN02"):
        markets[coin]["qty_step"] = markets[coin]["min_qty"] = 1e-9
    traces = []
    for scale in (1.0, 1.5):
        scaled = deepcopy(inputs)
        scaled[0]["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] = scale
        payload, fills, equities, analysis = run_cpu(scaled)
        assert len(fills) > 0
        assert all("entry" in str(fill[13]) for fill in fills)
        assert all(float(fill[5]) == 1000.0 for fill in fills)
        expected = reference_moments(scaled, payload, fills, equities)
        assert {name: analysis[name] for name in NAMES} == pytest.approx(expected, rel=1e-11, abs=1e-12)
        traces.append((fills, analysis))
    first_fills, first = traces[0]
    scaled_fills, scaled = traces[1]
    assert [fill[1] for fill in scaled_fills] == [fill[1] for fill in first_fills]
    assert [float(fill[9]) for fill in scaled_fills] == pytest.approx(
        [1.5 * float(fill[9]) for fill in first_fills], rel=1e-7, abs=2e-9,
    )
    assert scaled[NAMES[1]] == pytest.approx(first[NAMES[1]], rel=1e-7)
    assert scaled[NAMES[2]] == first[NAMES[2]] == 0.0
    assert scaled[NAMES[0]] > first[NAMES[0]] > 0.0
