"""Source snapshots stay lean; runtime, identity and saved results stay effective."""
from copy import deepcopy
import json

import pytest

from config.bot_inheritance import (
    apply_bot_config_patch, bot_policy_source_view, get_bot_policy, install_bot_policy,
)
from config.inherited_bounds import get_bound_plan
from config.load import prepare_config
from config.universe import universe_source_view
from config_utils import clean_config
from optimization.evaluation_contract import build_evaluation_contract
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import execution_key
from optimizer_overrides import optimizer_overrides
from session_artifacts import effective_setup_config
from shared_arrays import SharedArraySpec


def source(kind="trailing_martingale"):
    return {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "long": {"enabled": True}, "global": { "universe": {"approved_coins": ["AAA"], "minimum_coin_age_days": 0.0}},
        "short": {"enabled": False}},
        "optimize": {"bounds": {"global": {"risk": {"total_wallet_exposure_limit_ratio": [1.0, 3.0]}}}}}


@pytest.mark.parametrize("enabled", [False, True])
def test_resume_compares_authored_bounds_and_resolved_activation_not_hydrated_domains(enabled):
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import CONTRACT_KEY
    raw = source()
    raw["bot"]["long"]["enabled"] = enabled
    current = load(raw)
    saved = clean_config(current)
    saved["backtest"]["coins"] = deepcopy(current["backtest"]["coins"])
    saved[CONTRACT_KEY] = build_evaluation_contract(current)
    assert raw["bot"]["long"] == {"enabled": enabled}
    assert set(saved["optimize"]["bounds"]) == {"global"}
    assert set(current["optimize"]["bounds"]) != {"global"}
    assert _resume_config_mismatches(saved, current) == []
    changed = deepcopy(saved)
    changed["optimize"]["bounds"]["global"]["risk"]["total_wallet_exposure_limit_ratio"] = [1.0, 4.0]
    assert any("optimize.bounds" in item for item in _resume_config_mismatches(changed, current))


def test_actual_suite_candidate_source_handoff_preserves_omitted_ignored_and_numeric_gene_authority(tmp_path):
    from types import SimpleNamespace
    from optimize import SuiteEvaluator
    from config.bot_inheritance import apply_bot_policy_patch
    ignored = tmp_path / "ignored.json"
    ignored.write_text('["BBB"]')
    raw = source()
    raw["bot"]["global"]["universe"]["ignored_coins"] = str(ignored)
    context = load(raw)
    current_raw = source()
    current_raw["bot"]["long"] = {"universe": {"ignored_coins": []}}
    current = load(current_raw)
    policy = apply_bot_policy_patch(get_bot_policy(current),
        {"bot.global.risk.total_wallet_exposure_limit_ratio": 2.75}, channel="gene")
    policy = apply_bot_policy_patch(policy,
        {"bot.short.unstuck.close_fraction": 0.08}, channel="derived")
    install_bot_policy(current, policy)
    ignored.unlink()
    ctx = SimpleNamespace(config=context, overrides={}, label="base", msss={"binance": {"AAA": {}}})
    evaluator = SimpleNamespace(base=SimpleNamespace(scoring_specs=[], limit_checks=[]), objective_bases=[])
    result = SuiteEvaluator.build_scenario_candidate_config(evaluator, current, ctx)
    exported = bot_policy_source_view(result)["bot"]
    assert "ignored_coins" not in exported.get("long", {}).get("universe", {})
    assert "ignored_coins" not in exported.get("short", {}).get("universe", {})
    assert exported["global"]["universe"]["ignored_coins"] == str(ignored)
    assert result["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 2.75
    assert ("bot", "global", "risk", "total_wallet_exposure_limit_ratio") in get_bot_policy(result)._genes
    assert result["bot"]["short"]["unstuck"]["close_fraction"] == 0.08
    assert get_bot_policy(result)._derived == policy._derived
    assert list(result["bot"]["long"]["universe"]["ignored_coins"]) == ["BBB"]
    assert execution_key(result) == execution_key(dataset(result).policy_config())


def load(raw):
    result = prepare_config(raw, verbose=False)
    result["backtest"]["coins"] = {"binance": ["AAA"]}
    return result


def dataset(config):
    return PreparedGpuDataset(config=config, markets={"AAA": {"price_step": 0.01}},
        hlcvs=SharedArraySpec("candles", (10, 1, 4), "<f8"),
        btc=SharedArraySpec("btc", (10,), "<f8"),
        timestamps=SharedArraySpec("timestamps", (10,), "<i8"),
        candle_coins=["AAA"], exchange="binance", metrics=["fills_per_day"])


def test_evaluation_identity_records_resolved_fixed_hysteresis_pair():
    raw = source()
    raw["bot"]["global"]["forager"] = {"score_hysteresis_margin": 0.03}
    raw["bot"]["short"]["forager"] = {"score_hysteresis_margin": 0.07}
    config = load(raw)
    contract = build_evaluation_contract(config)
    assert contract["forager_hysteresis"] == {"abi": 1, "policy": {"long": 0.03, "short": 0.07}}
    apply_bot_config_patch(config, {"bot.global.forager.score_hysteresis_margin": 0.04})
    changed = build_evaluation_contract(config)
    assert changed != contract
    assert changed["forager_hysteresis"]["policy"] == {"long": 0.04, "short": 0.07}


def test_setup_reads_effective_simulation_policy_before_source_export():
    config = load(source())
    assert "universe" not in clean_config(config)["bot"]["short"]
    setup = effective_setup_config(config)
    assert setup["backtest"]["filter_by_min_effective_cost"] == {"long": True, "short": True}
    assert "universe" not in setup["bot"]["short"]


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_prepared_dataset_reconstructs_effective_policy_without_source_reread(tmp_path, kind):
    path = tmp_path / "coins.json"
    path.write_text('["AAA"]')
    raw = source(kind)
    raw["bot"]["global"]["universe"]["approved_coins"] = str(path)
    prepared = load(raw)
    registered = dataset(prepared)
    assert "risk" not in json.loads(registered.config_json)["bot"]["short"]
    path.unlink()
    policy, execution = registered.policy_config(), registered.execution_config()
    assert get_bot_policy(policy) is not None
    assert policy["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.5
    assert execution["bot"]["long"]["strategy"][kind]
    assert execution["bot"]["short"]["universe"]["approved_coins"] == ["AAA"]
    policy["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] = 99
    assert registered.policy_config()["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.5


def test_request_owned_common_gene_reuses_execution_identity_but_equal_pin_changes_authority():
    config = load(source())
    plan = get_bound_plan(config)
    key = execution_key(config)
    initial_contract = build_evaluation_contract(config)
    values = [binding.bound.low for binding in plan.bindings]
    values[[binding.key for binding in plan.bindings].index("bot.global.risk.total_wallet_exposure_limit_ratio")] = 2.75
    changed = deepcopy(config)
    install_bot_policy(changed, plan.apply_vector(get_bot_policy(config), values))
    assert execution_key(changed) == key
    assert build_evaluation_contract(changed) == initial_contract
    raw = source()
    raw["bot"]["short"]["risk"] = {"total_wallet_exposure_limit_ratio": 1.5}
    assert execution_key(load(raw)) != key


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("transform", ["mirror_short_from_long", "couple_unstuck_ema_spans"])
def test_saved_derived_snapshot_reloads_as_ordinary_policy_without_changing_authored_source(kind, transform):
    raw = source(kind)
    raw["bot"]["long"] = {"unstuck": {"ema_span_0_minutes": 123.5, "ema_span_1_minutes": 345.5}}
    config = load(raw)
    original = bot_policy_source_view(config)["bot"]
    config["optimize"]["enable_overrides"] = [transform]
    optimizer_overrides([transform], config, None)
    for side in ("long", "short"):
        optimizer_overrides([transform], config, side)
    assert bot_policy_source_view(config)["bot"] == original
    snapshot = clean_config(universe_source_view(config, candidate=True))
    snapshot["optimize"]["enable_overrides"] = []
    replay = prepare_config(json.loads(json.dumps(snapshot)), verbose=False)
    for side in ("long", "short"):
        assert replay["bot"][side]["unstuck"] == config["bot"][side]["unstuck"]
    assert get_bot_policy(config)._derived


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scenario_span", [None, 71.25])
def test_scenario_coupling_preserves_transformed_candidate_before_explicit_overrides(kind, scenario_span):
    from config.bot_inheritance import apply_bot_policy_patch
    from optimization.warmup import _finalize_optimizer_vector_config

    raw = source(kind)
    spans = {"ema_span_0_minutes": 17.25, "ema_span_1_minutes": 211.75}
    strategy = {"entry": spans} if kind == "trailing_martingale" else spans
    raw["bot"]["long"] = {"strategy": {kind: deepcopy(strategy)}}
    raw["bot"]["short"]["strategy"] = {kind: deepcopy(strategy)}
    suffix = f"strategy.{kind}" + (".entry" if kind == "trailing_martingale" else "")
    raw["optimize"]["enable_overrides"] = ["mirror_short_from_long", "couple_unstuck_ema_spans"]
    overrides = {} if scenario_span is None else {f"bot.short.{suffix}.ema_span_0_minutes": scenario_span}
    raw["backtest"] = {"scenarios": [{"label": "candidate", "overrides": overrides}]}
    config = load(raw)
    policy = apply_bot_policy_patch(get_bot_policy(config),
        {f"bot.long.{suffix}.ema_span_0_minutes": 17.2}, channel="gene")
    install_bot_policy(config, policy)
    candidate = _finalize_optimizer_vector_config(config)
    source_before_repeat = bot_policy_source_view(candidate)["bot"]
    saved = candidate["backtest"]["scenarios"][0]["overrides"]
    assert saved["bot.short.unstuck.ema_span_0_minutes"] == (17.2 if scenario_span is None else scenario_span)
    effective_strategy = candidate["bot"]["short"]["strategy"][kind]
    if kind == "trailing_martingale":
        effective_strategy = effective_strategy["entry"]
    assert effective_strategy["ema_span_0_minutes"] == 17.2
    repeated = _finalize_optimizer_vector_config(deepcopy(candidate))
    assert repeated == candidate
    assert bot_policy_source_view(repeated)["bot"] == source_before_repeat


def test_normalization_retains_effective_numeric_types_without_rewriting_source_types():
    raw = source()
    raw["bot"]["global"].update({
        "forager": {"score_weights": {"unilateralness": 0, "ema_readiness": 1, "volume": 0, "log_range": 0}},
        "risk": {"max_positions": 7.0}})
    prepared = load(raw)
    assert type(prepared["bot"]["long"]["forager"]["score_weights"]["unilateralness"]) is float
    assert type(prepared["bot"]["long"]["risk"]["max_positions"]) is int
    retained = bot_policy_source_view(prepared)["bot"]["global"]
    assert type(retained["forager"]["score_weights"]["unilateralness"]) is int
    assert type(retained["risk"]["max_positions"]) is float


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_actual_optimizer_patch_owner_records_equal_side_declarations(kind):
    from optimize import _apply_config_overrides

    config = load(source(kind))
    _apply_config_overrides(config, {"bot.short.risk.total_wallet_exposure_limit_ratio": 1.5})
    raw = bot_policy_source_view(config)["bot"]
    assert raw["short"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.5
    _apply_config_overrides(config, {"bot.global.risk.total_wallet_exposure_limit_ratio": 2.75})
    assert config["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == 2.75
    assert config["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.5


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_actual_optimizer_patch_owner_rebuilds_declared_bound_authority(kind):
    from config.inherited_bounds import optimize_bounds_source_view
    from optimize import _apply_config_overrides

    config = load(source(kind))
    before = get_bound_plan(config).identity()
    _apply_config_overrides(config, {
        "optimize.bounds.global.risk.total_wallet_exposure_limit_ratio": [1.0, 4.0]})
    assert get_bound_plan(config).identity() != before
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == {
        "global": {"risk": {"total_wallet_exposure_limit_ratio": [1.0, 4.0]}}}


@pytest.mark.parametrize("value", [None, [], 0, "invalid"])
def test_bound_capability_lookup_rejects_a_malformed_optimize_section(value):
    with pytest.raises(TypeError, match="config.optimize"):
        get_bound_plan({"optimize": value})


def test_actual_optimizer_bound_patch_rejects_unknown_fields_without_changing_authority():
    from optimize import _apply_config_overrides
    from config.inherited_bounds import optimize_bounds_source_view
    config = load(source())
    before = optimize_bounds_source_view(config)
    with pytest.raises(ValueError, match="unknown optimizer field"):
        _apply_config_overrides(config, {"optimize.bounds.global.risk.invented": [1.0, 4.0]})
    assert optimize_bounds_source_view(config) == before


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("sample,expected", [(71.0, 70.0), (0.5, 1.0), (1.26, 1.5)])
def test_actual_canonicalizer_repairs_only_outside_endpoints_before_transform(monkeypatch, kind, sample, expected):
    import optimize
    from optimization.shape import build_optimization_shape
    raw = source(kind)
    raw["optimize"]["bounds"]["global"]["risk"]["total_wallet_exposure_limit_ratio"] = [1.0, 70.0, 0.5]
    config = load(raw)
    shape = build_optimization_shape(config)
    vector = optimize.config_to_individual(config, shape.bounds, shape.sig_digits, optimization_shape=shape)
    index = [key for key, _ in shape.key_paths].index("bot.global.risk.total_wallet_exposure_limit_ratio")
    vector[index] = sample
    observed = []
    original = optimize.individual_to_config
    def observe(values, *args, **kwargs):
        observed.append(float(values[index]))
        return original(values, *args, **kwargs)
    monkeypatch.setattr(optimize, "individual_to_config", observe)
    result = optimize._canonicalize_optimizer_individual(
        vector, config, shape.bounds, shape.sig_digits, shape.key_paths, [])
    assert observed[0] == max(1.0, min(70.0, sample))
    assert vector[index] == expected
    assert result["bot"]["long"]["risk"]["total_wallet_exposure_limit_ratio"] == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_actual_canonicalizer_never_repairs_nonfinite_genes(value):
    import optimize
    from optimization.shape import build_optimization_shape
    config = load(source())
    shape = build_optimization_shape(config)
    vector = optimize.config_to_individual(config, shape.bounds, shape.sig_digits, optimization_shape=shape)
    index = [key for key, _ in shape.key_paths].index("bot.global.risk.total_wallet_exposure_limit_ratio")
    vector[index] = value
    with pytest.raises(TypeError, match="finite numeric"):
        optimize._canonicalize_optimizer_individual(
            vector, config, shape.bounds, shape.sig_digits, shape.key_paths, [])


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_runtime_patch_snapshot_preserves_unrelated_mirror_and_reusable_base(tmp_path, kind):
    from config.bot import normalize_bot_forager_config
    from optimization.warmup import _apply_config_overrides
    from test_explicit_optimize_bounds import mirrored_weight_source, mirrored_weight_candidate

    raw = mirrored_weight_source(kind)
    approvals = tmp_path / "approved.json"
    approvals.write_text('["AAA"]')
    raw["bot"]["global"]["universe"]["approved_coins"] = str(approvals)
    base = prepare_config(raw, verbose=False)
    base_source = bot_policy_source_view(base)
    base_policy = get_bot_policy(base)
    identity = get_bound_plan(base).identity()
    approvals.unlink()
    candidate = mirrored_weight_candidate(base, 0.9)
    assert get_bot_policy(candidate)._derived
    expected_short = deepcopy(candidate["bot"]["short"]["forager"]["score_weights"])
    patch = {"bot.long.forager.score_weights": {
        "volume": 0.2, "log_range": 0.3, "ema_readiness": 0.4, "unilateralness": 0.1}}
    _apply_config_overrides(candidate, patch, preserve_prepared_universe=True)
    normalize_bot_forager_config(candidate, verbose=False)
    assert candidate["bot"]["short"]["forager"]["score_weights"] == expected_short
    assert candidate["bot"]["long"]["forager"]["score_weights"] == pytest.approx(patch["bot.long.forager.score_weights"])
    assert list(candidate["bot"]["short"]["universe"]["approved_coins"]) == ["AAA"]
    assert get_bound_plan(candidate).identity() == identity
    assert get_bound_plan(base).identity() == identity
    assert get_bot_policy(base) == base_policy
    assert bot_policy_source_view(base) == base_source
    previous = deepcopy(candidate)
    _apply_config_overrides(candidate, patch, preserve_prepared_universe=True)
    normalize_bot_forager_config(candidate, verbose=False)
    assert candidate == previous


@pytest.mark.parametrize("value,expected", [(0.0, 0), (0.5, 0), (1.5, 2), (4.5, 4), (5.5, 6)])
def test_continuous_count_gene_uses_existing_canonical_count_adapter(value, expected):
    from config.bot_inheritance import apply_bot_policy_patch
    config = load(source())
    policy = apply_bot_policy_patch(get_bot_policy(config),
        {"bot.global.risk.max_positions": value}, channel="gene")
    assert type(policy.effective_view()["long"]["risk"]["max_positions"]) is int
    assert policy.effective_view()["long"]["risk"]["max_positions"] == expected
    if not value.is_integer():
        with pytest.raises(TypeError, match="serialization type"):
            apply_bot_policy_patch(get_bot_policy(config), {"bot.global.risk.max_positions": value})


def test_compiled_aliases_are_effective_only_and_cannot_become_authored_fields():
    from config.runtime_compile import compile_runtime_config
    prepared = load(source())
    authored = bot_policy_source_view(prepared)["bot"]
    compiled = compile_runtime_config(prepared, runtime="backtest")
    assert compiled["bot"]["long"]["risk_entry_cooldown_minutes"] == 24.1
    assert bot_policy_source_view(compiled)["bot"] == authored
    assert bot_policy_source_view(compile_runtime_config(compiled, runtime="backtest"))["bot"] == authored
    compiled["bot"]["long"]["risk_entry_cooldown_minutes"] = 0.0
    with pytest.raises(ValueError, match="compiled native aliases"):
        bot_policy_source_view(compiled)
    raw = source()
    raw["bot"]["long"] = {"risk_entry_cooldown_minutes": 1.0}
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(raw, verbose=False)
