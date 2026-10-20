from copy import deepcopy
import math
import sys
from types import SimpleNamespace

import pytest

import optimization.native_planning as planning
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
from optimize import Evaluator, SuiteEvaluator, config_to_individual
from shared_arrays import SharedArraySpec
from test_native_planning import set_value
from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS


def projection_inputs(*, strategy="ema_anchor", suite=False, fixed=False, mirror=False,
                      selected_indices=(2,), max_positions=None):
    config, candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    candle_coins = tuple(config["backtest"]["coins"]["binance"])
    if selected_indices is not None:
        selected = [candle_coins[index] for index in selected_indices]
        config["backtest"]["coins"]["binance"] = selected
        for side in ("long", "short"):
            config['bot'][side]['universe']['approved_coins'] = list(selected)
            config["bot"][side]["risk"]["max_positions"] = len(selected)
    if max_positions is not None:
        for side in ("long", "short"):
            config["bot"][side]["risk"]["max_positions"] = max_positions
    config["optimize"]["bounds"] = {
        "global": {"unstuck": {"memory_ema_span_minutes": [1000.0, 50000.0]}}
    }
    for side in ("long", "short"):
        config["optimize"]["bounds"][side] = {"risk": {
            key: [config["bot"][side]["risk"][key]] * 2
            for key in ("max_positions", "total_wallet_exposure_limit")
        }}
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
    ]
    config["optimize"]["limits"] = []
    if fixed:
        config["optimize"]["fixed_runtime_overrides"] = {"bot.global.unstuck.memory_ema_span_minutes": 4096.5}
    if mirror:
        config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    spec = SharedArraySpec("candles", candles.shape, candles.dtype.str)
    base = Evaluator({"binance": spec}, {}, {"binance": markets}, config)
    contexts = [SimpleNamespace(
        label=label, exchanges=["binance"], config=deepcopy(base.config), msss={"binance": markets},
        overrides={"bot.global.unstuck.memory_ema_span_minutes": 2048.5} if label == "stress" else {},
    ) for label in ("base", "stress")]
    evaluator = SuiteEvaluator(base, contexts, {"default": "mean"}) if suite else base
    bindings = []
    for context in contexts if suite else contexts[:1]:
        effective = evaluator.build_scenario_candidate_config(base.config, context) if suite else base.config
        dataset = PreparedGpuDataset(
            config=effective, markets=markets, exchange="binance", hlcvs=spec,
            btc=SharedArraySpec("btc", (len(candles),), "<f8"),
            timestamps=SharedArraySpec("timestamps", (len(candles),), "<i8"),
            candle_coins=candle_coins, coin_indices=selected_indices, metrics=DEFAULT_METRICS,
        )
        bindings.append(ScenarioBinding(context.label, context.label, dataset))
    planner = NativeCandidatePlanner(evaluator, bindings)
    vector = config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
    return planner, vector


def with_variant(planner, *, strategy="trailing_martingale", scenario="base", first=False, selected_indices=None):
    other, _vector = projection_inputs(strategy=strategy, selected_indices=selected_indices)
    variant = ScenarioBinding(scenario, f"{scenario}:{strategy}", other.bindings[0].dataset)
    bindings = [variant, *planner.bindings] if first else [*planner.bindings, variant]
    return NativeCandidatePlanner(planner.scorer.evaluator, bindings)


def set_path(config, path, value):
    node = config
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def outcome(call):
    try:
        return ("value", call())
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        return (type(error), str(error))


def require_eligible_projection_calls(monkeypatch, planner):
    projected_key = planner._execution_key
    def eligible_only(pair, config):
        assert planner._projection_eligible[pair], "ineligible pairs must call execution_key directly"
        return projected_key(pair, config)
    monkeypatch.setattr(planner, "_execution_key", eligible_only)


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_public_execution_key_does_not_construct_projection_paths(monkeypatch, strategy):
    planner, vector = projection_inputs(strategy=strategy)
    config = planner.prepare("base", vector).record_config
    expected = planning.execution_key(config)
    monkeypatch.setattr(planning, "_request_owned_paths", lambda _config: pytest.fail("legacy key used projection paths"))
    assert planning.execution_key(config) == expected


@pytest.mark.parametrize("suite,fixed,mirror", [
    (False, False, False), (True, False, False), (True, True, False),
    (False, False, True), (True, True, True),
])
@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_planner_projection_reuses_only_validated_static_inputs(monkeypatch, suite, fixed, mirror, strategy):
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner, vector = projection_inputs(strategy=strategy, suite=suite, fixed=fixed, mirror=mirror)
    first = planner.prepare("first", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 1000.0))
    calls.clear()
    second = planner.prepare("second", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 50000.0))
    assert not calls
    assert [request.dataset_id for request in first.requests] == [request.dataset_id for request in second.requests]
    assert len(planner._execution_projections) == (2 if suite else 1)
    assert all(planner._projection_eligible.values())
    for pair, projection in planner._execution_projections.items():
        effective = second.record_config
        if suite:
            context = next(ctx for ctx in planner.scorer.evaluator.contexts if ctx.label == pair[0])
            effective = planner.scorer.evaluator.build_scenario_candidate_config(effective, context)
        assert projection.matches(effective)
        assert projection.key == planning.execution_key(effective)
    # Public callers do not acquire a trusted-canonical shortcut.
    calls.clear()
    planning.execution_key(second.record_config)
    planning.execution_key(second.record_config)
    assert len(calls) == 2


@pytest.mark.parametrize("variant", ["tm", "ema", "mixed_tm_first", "mixed_ema_first"])
def test_multicoin_and_ineligible_mixed_pairs_never_skip_cleaning(monkeypatch, variant):
    planner, vector = projection_inputs(strategy="trailing_martingale" if variant == "tm" else "ema_anchor",
                                        selected_indices=None)
    if variant.startswith("mixed"):
        planner = with_variant(planner, first=variant == "mixed_tm_first")
    require_eligible_projection_calls(monkeypatch, planner)
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner.prepare("warmup", vector)
    calls.clear()
    for candidate, value in (("low", 1000.0), ("high", 50000.0)):
        planner.prepare(candidate, set_value(planner, vector, "unstuck_memory_ema_span_minutes", value))
    assert len(calls) == 2
    assert planner._projection_eligible == {("base", "binance"): False}
    assert planner._execution_projections == {}


@pytest.mark.parametrize("indices,eligible", [((2,), True), ((0, 2), False)])
@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_projection_uses_selected_indices_instead_of_source_column_count(monkeypatch, indices, eligible, strategy):
    planner, vector = projection_inputs(strategy=strategy, selected_indices=indices)
    require_eligible_projection_calls(monkeypatch, planner)
    dataset = planner.bindings[0].dataset
    assert len(dataset.candle_coins) == dataset.hlcvs.shape[1] == 3
    assert dataset.coin_indices == indices
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner.prepare("warmup", vector)
    calls.clear()
    low = planner.prepare("low", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 1000.0))
    high = planner.prepare("high", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 50000.0))
    assert len(calls) == (0 if eligible else 2)
    assert planner._projection_eligible == {("base", "binance"): eligible}
    assert len(planner._execution_projections) == int(eligible)
    assert low.effective_key != high.effective_key
    assert low.requests[0].dataset_id == high.requests[0].dataset_id
    assert [plan.requests[0].parameters["unstuck_memory_ema_span_minutes"]
            for plan in (low, high)] == [1000.0, 50000.0]


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_single_position_in_a_multicoin_view_does_not_qualify(monkeypatch, strategy):
    planner, vector = projection_inputs(strategy=strategy, selected_indices=None, max_positions=1)
    require_eligible_projection_calls(monkeypatch, planner)
    assert len(planner.bindings[0].dataset.coin_indices) == 3
    assert planner.base.config["bot"]["long"]["risk"]["max_positions"] == 1
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner.prepare("warmup", vector)
    calls.clear()
    planner.prepare("next", vector)
    assert len(calls) == 1
    assert planner._projection_eligible == {("base", "binance"): False}
    assert planner._execution_projections == {}


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("variant_strategy", ["trailing_martingale", "ema_anchor"])
def test_one_multicoin_variant_disables_a_single_coin_pair(monkeypatch, first, strategy, variant_strategy):
    planner, vector = projection_inputs(strategy=strategy)
    planner = with_variant(planner, strategy=variant_strategy, first=first)
    require_eligible_projection_calls(monkeypatch, planner)
    assert sorted(len(binding.dataset.coin_indices) for binding in planner.bindings) == [1, 3]
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner.prepare("warmup", vector)
    calls.clear()
    planner.prepare("next", vector)
    assert len(calls) == 1
    assert planner._projection_eligible == {("base", "binance"): False}
    assert planner._execution_projections == {}


@pytest.mark.parametrize("first", [False, True])
def test_single_coin_variants_qualify_across_strategies(monkeypatch, first):
    planner, vector = projection_inputs()
    planner = with_variant(planner, first=first, selected_indices=(2,))
    assert [len(binding.dataset.coin_indices) for binding in planner.bindings] == [1, 1]
    planner.prepare("warmup", vector)
    monkeypatch.setattr(planning, "execution_key", lambda _config: pytest.fail("qualified warm pair was cleaned again"))
    planner.prepare("next", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 50000.0))
    assert planner._projection_eligible == {("base", "binance"): True}
    assert len(planner._execution_projections) == 1


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_single_coin_pair_qualifies_with_multiple_execution_variants(monkeypatch, strategy):
    planner, vector = projection_inputs(strategy=strategy)
    changed = deepcopy(planner.base.config)
    changed["backtest"]["starting_balance"] = 2000.0
    original = planner.bindings[0]
    variant = ScenarioBinding("base", "base:other-balance", original.dataset.with_config(changed))
    planner = NativeCandidatePlanner(planner.scorer.evaluator, [*planner.bindings, variant])
    planner.prepare("warmup", vector)
    monkeypatch.setattr(planning, "execution_key", lambda _config: pytest.fail("eligible warm single-coin pair was cleaned again"))
    planner.prepare("next", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 50000.0))
    assert planner._projection_eligible == {("base", "binance"): True}
    assert len(planner._execution_projections) == 1


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_scenario_pairs_decide_eligibility_independently(monkeypatch, strategy):
    cleaner = planning.clean_config
    calls = []
    def observed(config):
        calls.append(config)
        return cleaner(config)
    monkeypatch.setattr(planning, "clean_config", observed)
    planner, vector = projection_inputs(strategy=strategy, suite=True)
    planner = with_variant(planner, scenario="stress")
    require_eligible_projection_calls(monkeypatch, planner)
    planner.prepare("warmup", vector)
    calls.clear()
    planner.prepare("next", set_value(planner, vector, "unstuck_memory_ema_span_minutes", 50000.0))
    assert len(calls) == 1
    assert planner._projection_eligible == {("base", "binance"): True, ("stress", "binance"): False}
    assert set(planner._execution_projections) == {("base", "binance")}


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -1.0, 1e-300, 1e300])
@pytest.mark.parametrize("strategy,hsl", [
    ("trailing_martingale", "coin"), ("ema_anchor", "coin"),
    ("trailing_martingale", "portfolio"), ("ema_anchor", "portfolio"),
])
def test_every_permitted_finite_leaf_preserves_the_full_cleaned_static_key(value, strategy, hsl):
    config, *_rest = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--hsl", hsl, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    key = planning.execution_key(config)
    projection = planning._ExecutionProjection(config, key)
    compared = 0
    for path in planning._request_owned_paths(config):
        node = config
        for part in path:
            if not isinstance(node, dict) or part not in node:
                break
            node = node[part]
        else:
            if type(node) not in (int, float):
                continue
            changed = deepcopy(config)
            set_path(changed, path, value)
            if projection.matches(changed):
                assert planning.execution_key(changed) == key, path
                compared += 1
    assert compared >= 30


@pytest.mark.parametrize("weights", [
    {"volume": 0.0, "ema_readiness": 0.0, "volatility": 0.0, "unilateralness": 0.0},
    {"volume": sys.float_info.max, "ema_readiness": sys.float_info.max,
     "volatility": 0.0, "unilateralness": 0.0},
    {"volume": 0.1, "ema_readiness": 0.2, "volatility": 0.3, "unilateralness": 0.4},
])
def test_valid_joint_weight_changes_preserve_the_full_cleaned_static_key(weights):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    changed["bot"]["long"]["forager"]["score_weights"] = weights
    assert planner._execution_projections[pair].matches(changed)
    assert planning.execution_key(changed) == planning.execution_key(config)


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, True, "0.5"])
def test_unsupported_weight_values_retain_exact_legacy_validation(value):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    changed["bot"]["long"]["forager"]["score_weights"]["volume"] = value
    assert not planner._execution_projections[pair].matches(changed)
    assert outcome(lambda: planner._execution_key(pair, changed)) == outcome(lambda: planning.execution_key(changed))


@pytest.mark.parametrize("path,value", [
    (("backtest", "btc_collateral_cap"), -0.0),
    (("backtest", "starting_balance"), 1000),
    (("backtest", "filter_by_min_effective_cost"), 0),
])
def test_static_json_float_sign_and_scalar_types_do_not_alias(path, value):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    previous = planning.execution_key(config)
    changed = deepcopy(config)
    set_path(changed, path, value)
    expected = planning.execution_key(changed)
    assert expected != previous
    assert not planner._execution_projections[pair].matches(changed)
    assert planner._execution_key(pair, changed) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "bad", True, 10**1000])
def test_malformed_dynamic_values_follow_unchanged_legacy_outcome(value):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    changed["bot"]["long"]["risk"]["max_positions"] = value
    assert not planner._execution_projections[pair].matches(changed)
    assert outcome(lambda: planner._execution_key(pair, changed)) == outcome(lambda: planning.execution_key(changed))


@pytest.mark.parametrize("value", [True, "2"])
def test_initial_non_numeric_transport_leaves_do_not_acquire_a_shortcut(value):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    config["bot"]["long"]["risk"]["max_positions"] = value
    key = planning.execution_key(config)
    projection = planning._ExecutionProjection(config, key)
    assert projection.guard is None
    assert not projection.matches(config)


@pytest.mark.parametrize("mode", ["disabled", "initial", "reentry", "invalid"])
def test_request_owned_string_selector_changes_still_take_the_full_path(mode):
    planner, vector = projection_inputs(strategy="trailing_martingale")
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    projection = planning._ExecutionProjection(config, planning.execution_key(config))
    changed = deepcopy(config)
    changed["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_gate_mode"] = mode
    assert not projection.matches(changed)
    assert outcome(lambda: planner._execution_key(pair, changed)) == outcome(lambda: planning.execution_key(changed))


@pytest.mark.parametrize("change", ["version", "retired", "portfolio_bounds", "weights", "duration"])
def test_changed_cleaner_validation_is_not_skipped(change):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    if change == "version":
        changed["config_version"] = "v8.0.0"
    elif change == "retired":
        changed["live"]["hsl_engine"] = "legacy"
    elif change == "portfolio_bounds":
        changed["optimize"]["bounds"].setdefault("global", {})["hsl"] = []
    elif change == "weights":
        changed["bot"]["long"]["forager"]["score_weights"]["volume"] = -0.1
    else:
        changed["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = math.inf
    expected = outcome(lambda: planning.execution_key(changed))
    assert expected[0] != "value"
    assert not planner._execution_projections[pair].matches(changed)
    assert outcome(lambda: planner._execution_key(pair, changed)) == expected


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_dynamic_side_enablement_still_selects_a_distinct_static_contract(value):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    changed["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = value
    if value == 0.0:
        assert not planner._execution_projections[pair].matches(changed)
        assert planning.execution_key(changed) != planning.execution_key(config)
    assert planner._execution_key(pair, changed) == planning.execution_key(changed)


@pytest.mark.parametrize("path,value", [
    ("backtest.starting_balance", 2000.0),
    ('bot.global.execution.market_orders_allowed', True),
    ("bot.global.loss_gate.max_drawdown_ratio", 0.5),
])
def test_post_construction_static_policies_still_require_a_compatible_dataset(path, value):
    planner, vector = projection_inputs()
    planner.prepare("base", vector)
    planner.base.config["optimize"]["fixed_runtime_overrides"] = {path: value}
    with pytest.raises(ValueError, match="dataset-owned execution inputs"):
        planner.prepare("changed", vector)
    assert len(planner._execution_projections) == 1


@pytest.mark.parametrize("change", ["float", "dict", "list", "key"])
def test_scalar_container_and_key_subclasses_use_the_legacy_path(change):
    class Number(float):
        pass
    class Mapping(dict):
        pass
    class Sequence(list):
        pass
    class Key(str):
        pass
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    changed = deepcopy(config)
    if change == "float":
        changed["bot"]["long"]["risk"]["max_positions"] = Number(2.0)
    elif change == "dict":
        changed["backtest"] = Mapping(changed["backtest"])
    elif change == "list":
        changed["backtest"]["exchanges"] = Sequence(changed["backtest"]["exchanges"])
    else:
        changed["backtest"] = {Key(key): value for key, value in changed["backtest"].items()}
    assert not planner._execution_projections[pair].matches(changed)
    assert outcome(lambda: planner._execution_key(pair, changed)) == outcome(lambda: planning.execution_key(changed))


def test_projection_never_retains_mutable_config_values_or_grows_after_fallbacks():
    planner, vector = projection_inputs()
    first = planner.prepare("first", vector)
    pair = next(iter(planner._variants))
    original = planner._execution_projections[pair]
    first.record_config["backtest"]["starting_balance"] = 2000.0
    second = planner.prepare("second", vector)
    assert original.matches(second.record_config)
    for balance in (1001.0, 1002.0, 1003.0):
        changed = deepcopy(second.record_config)
        changed["backtest"]["starting_balance"] = balance
        assert planner._execution_key(pair, changed) == planning.execution_key(changed)
    assert planner._execution_projections == {pair: original}


def test_metadata_replacement_and_unsupported_objects_disable_the_guard(monkeypatch):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    pair = next(iter(planner._variants))
    previous = planning.get_strategy_spec
    monkeypatch.setattr(planning, "get_strategy_spec", lambda kind: deepcopy(previous(kind)))
    assert not planner._execution_projections[pair].matches(config)
    assert planner._execution_key(pair, config) == planning.execution_key(config)
    opaque = deepcopy(config)
    opaque["_opaque"] = object()
    projection = planning._ExecutionProjection(opaque, planning.execution_key(opaque))
    assert projection.guard is None
    assert not projection.matches(opaque)


@pytest.mark.parametrize("kind", ["cycle", "deep", "subclass"])
def test_ignored_unsupported_metadata_does_not_introduce_a_new_error(kind):
    planner, vector = projection_inputs()
    config = planner.prepare("base", vector).record_config
    if kind == "cycle":
        metadata = []
        metadata.append(metadata)
    elif kind == "deep":
        metadata = None
        for _ in range(40):
            metadata = [metadata]
    else:
        class Opaque(dict):
            def items(self):
                raise AssertionError("the optional projection must not inspect subclass methods")
        metadata = Opaque()
    config["_opaque"] = metadata
    expected = planning.execution_key(config)
    projection = planning._ExecutionProjection(config, expected)
    assert projection.guard is None
    assert not projection.matches(config)
