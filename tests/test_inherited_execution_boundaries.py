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
        "global": {"enabled": True, "universe": {"approved_coins": ["AAA"], "minimum_coin_age_days": 0.0}},
        "short": {"enabled": False}},
        "optimize": {"bounds": {"global": {"risk": {"total_wallet_exposure_limit": [1.0, 3.0]}}}}}


@pytest.mark.parametrize("enabled", [False, True])
def test_resume_compares_authored_bounds_and_resolved_activation_not_hydrated_domains(enabled):
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import CONTRACT_KEY
    raw = source()
    raw["bot"]["global"]["enabled"] = enabled
    current = load(raw)
    saved = clean_config(current)
    saved["backtest"]["coins"] = deepcopy(current["backtest"]["coins"])
    saved[CONTRACT_KEY] = build_evaluation_contract(current)
    assert "long" not in raw["bot"]
    assert set(saved["optimize"]["bounds"]) == {"global"}
    assert set(current["optimize"]["bounds"]) != {"global"}
    assert _resume_config_mismatches(saved, current) == []
    changed = deepcopy(saved)
    changed["optimize"]["bounds"]["global"]["risk"]["total_wallet_exposure_limit"] = [1.0, 4.0]
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
        {"bot.global.risk.total_wallet_exposure_limit": 2.75}, channel="gene")
    install_bot_policy(current, policy)
    ignored.unlink()
    ctx = SimpleNamespace(config=context, overrides={}, label="base", msss={"binance": {"AAA": {}}})
    evaluator = SimpleNamespace(base=SimpleNamespace(scoring_specs=[], limit_checks=[]), objective_bases=[])
    result = SuiteEvaluator.build_scenario_candidate_config(evaluator, current, ctx)
    exported = bot_policy_source_view(result)["bot"]
    assert "ignored_coins" not in exported.get("long", {}).get("universe", {})
    assert "ignored_coins" not in exported.get("short", {}).get("universe", {})
    assert exported["global"]["universe"]["ignored_coins"] == str(ignored)
    assert result["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 2.75
    assert ("bot", "global", "risk", "total_wallet_exposure_limit") in get_bot_policy(result)._genes
    assert list(result["bot"]["long"]["universe"]["ignored_coins"]) == ["BBB"]


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
    assert policy["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert execution["bot"]["long"]["strategy"][kind]
    assert execution["bot"]["short"]["universe"]["approved_coins"] == ["AAA"]
    policy["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 99
    assert registered.policy_config()["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.5


def test_request_owned_common_gene_reuses_execution_identity_but_equal_pin_changes_authority():
    config = load(source())
    plan = get_bound_plan(config)
    key = execution_key(config)
    initial_contract = build_evaluation_contract(config)
    values = [binding.bound.low for binding in plan.bindings]
    values[[binding.key for binding in plan.bindings].index("bot.global.risk.total_wallet_exposure_limit")] = 2.75
    changed = deepcopy(config)
    install_bot_policy(changed, plan.apply_vector(get_bot_policy(config), values))
    assert execution_key(changed) == key
    assert build_evaluation_contract(changed) == initial_contract
    raw = source()
    raw["bot"]["short"]["risk"] = {"total_wallet_exposure_limit": 1.5}
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


def test_normalization_retains_effective_numeric_types_without_rewriting_source_types():
    raw = source()
    raw["bot"]["global"].update({
        "forager": {"score_weights": {"unilateralness": 0, "ema_readiness": 1, "volume": 0, "volatility": 0}},
        "risk": {"max_positions": 7.0}})
    prepared = load(raw)
    assert type(prepared["bot"]["long"]["forager"]["score_weights"]["unilateralness"]) is float
    assert type(prepared["bot"]["long"]["risk"]["max_positions"]) is int
    retained = bot_policy_source_view(prepared)["bot"]["global"]
    assert type(retained["forager"]["score_weights"]["unilateralness"]) is int
    assert type(retained["risk"]["max_positions"]) is float


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
