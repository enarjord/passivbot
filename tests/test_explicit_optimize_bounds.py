"""Only authored search domains may vary; released v8 ranges migrate explicitly."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.optimize_bounds import flatten_optimize_bounds, sort_optimize_bounds_in_place
from config.schema import get_template_config
from config_utils import clean_config, strip_config_metadata
from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths
from optimization.warmup import build_optimizer_vector_config
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate
from tools.optimizer_bounds import apply_recommended_bounds, main


def flattened(config):
    return flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=config["bot"]["global"]["strategy_kind"])


def gene_bounds(config):
    return dict(zip((key for key, _path in get_optimization_key_paths(config)),
                    extract_bounds_tuple_list_from_config(config)))


@pytest.mark.parametrize("omission", ["optimize", "bounds", "empty", "side", "leaf"])
@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_omitted_dimensions_freeze_effective_values_through_exports(omission, strategy):
    source = get_template_config()
    source["bot"]["global"]["strategy_kind"] = strategy
    if strategy == "trailing_martingale":
        source["bot"]["long"]["strategy"][strategy]["entry"]["ema_span_0_minutes"] = 123.5
        branch = source["optimize"]["bounds"]["long"]["strategy"][strategy]["entry"]
    else:
        source["bot"]["long"]["strategy"][strategy]["ema_span_0_minutes"] = 123.5
        branch = source["optimize"]["bounds"]["long"]["strategy"][strategy]
    if omission == "optimize":
        source.pop("optimize")
    elif omission == "bounds":
        source["optimize"].pop("bounds")
    elif omission == "empty":
        source["optimize"]["bounds"] = {}
    elif omission == "side":
        source["optimize"]["bounds"].pop("long")
    else:
        branch.pop("ema_span_0_minutes")
    original = deepcopy(source)
    prepared = prepare_config(source, verbose=False)
    assert source == original
    for result in (prepared, clean_config(source),
                   prepare_config(json.loads(json.dumps(strip_config_metadata(prepared))), verbose=False)):
        bound = gene_bounds(result)["long_ema_span_0"]
        assert bound.low == bound.high == 123.5
        if omission in {"optimize", "bounds", "empty"}:
            assert all(item.low == item.high for item in gene_bounds(result).values())


@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_candidate_vectors_vary_only_the_authored_dimension(fraction):
    source = get_template_config()
    source["optimize"]["bounds"] = {
        "long": {"strategy": {"trailing_martingale": {"entry": {"ema_span_0_minutes": [100.5, 200.5, 0.5]}}}}
    }
    prepared = prepare_config(source, verbose=False)
    bounds = gene_bounds(prepared)
    assert {key for key, value in bounds.items() if value.low != value.high} == {"long_ema_span_0"}
    candidate = build_optimizer_vector_config([b.low + fraction * (b.high - b.low) for b in bounds.values()], prepared)
    for key, path in get_optimization_key_paths(prepared):
        old, new = prepared, candidate
        for part in path:
            old, new = old[part], new[part]
        if key == "long_ema_span_0":
            assert new == 100.5 + fraction * 100.0
        else:
            assert new == old, key


def test_cleaning_and_reload_preserve_disabled_zero_weight_coordinates():
    source = get_template_config()
    source["optimize"]["bounds"] = {}
    prepared = prepare_config(clean_config(source), verbose=False)
    bounds = gene_bounds(prepared)
    candidate = build_optimizer_vector_config([bound.low for bound in bounds.values()], prepared)
    for name, value in source["bot"]["short"]["forager"]["score_weights"].items():
        assert value == 0.0
        assert bounds[f"short_forager_score_weights_{name}"].low == 0.0
        assert candidate["bot"]["short"]["forager"]["score_weights"][name] == value


def test_sorting_does_not_open_template_ranges_and_null_stays_outside_genome():
    bounds = {"long": {"risk": {"max_positions": [3]}}}
    sort_optimize_bounds_in_place(bounds, strategy_kind="trailing_martingale")
    assert bounds == {"long": {"risk": {"max_positions": [3, 3]}}}
    source = get_template_config()
    source["optimize"]["bounds"] = {}
    prepared = prepare_config(source, verbose=False)
    assert "long_entry_cooldown_max_duration_minutes" not in gene_bounds(prepared)
    source["optimize"]["bounds"] = {"long": {"entry_cooldown": {"max_duration_minutes": [30.0, 60.0]}}}
    assert gene_bounds(prepare_config(source, verbose=False))["long_entry_cooldown_max_duration_minutes"].high == 60.0


def weight_source(bounds):
    source = get_template_config()
    source["optimize"]["bounds"] = {"long": {"forager": bounds}}
    return source


def test_partial_weight_search_cannot_move_omitted_positive_coordinates():
    source = weight_source({"score_weights": {"volume": [0.1, 1.0]}})
    with pytest.raises(ValueError, match=r"bot.long.forager.score_weights.*ema_readiness, volatility.*freeze the whole"):
        prepare_config(source, verbose=False)


def test_weight_validation_respects_the_supplied_search_space():
    from optimization.config_adapter import validate_optimize_bounds_against_bot_config

    config = prepare_config(get_template_config(), verbose=False)
    with pytest.raises(ValueError, match="partial normalized-weight search"):
        validate_optimize_bounds_against_bot_config(config, {
            "long": {"forager": {"score_weights": {"volume": [0.1, 1.0]}}}
        })


def test_whole_positive_weight_group_keeps_normalization_and_zero_omission():
    source = weight_source({"score_weights": {name: [0.1, 1.0] for name in ("volume", "volatility", "ema_readiness")}})
    config = prepare_config(source, verbose=False)
    bounds = gene_bounds(config)
    result = build_optimizer_vector_config([bound.high for bound in bounds.values()], config)
    weights = result["bot"]["long"]["forager"]["score_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["unilateralness"] == 0.0


@pytest.mark.parametrize("low,allowed", [(0.1, True), (0.0, False)])
def test_omitted_zero_weight_is_safe_except_at_an_unauthorized_fallback(low, allowed):
    source = weight_source({"score_weights": {"volume": [low, 1.0]}})
    source["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 1.0, "volatility": 0.0, "ema_readiness": 0.0, "unilateralness": 0.0,
    }
    if allowed:
        config = prepare_config(source, verbose=False)
        bounds = gene_bounds(config)
        result = build_optimizer_vector_config([bound.low for bound in bounds.values()], config)
        assert result["bot"]["long"]["forager"]["score_weights"] == source["bot"]["long"]["forager"]["score_weights"]
    else:
        with pytest.raises(ValueError, match="all-zero normalization fallback"):
            prepare_config(source, verbose=False)


def test_fixed_weight_leaf_is_rejected_but_fixing_whole_group_is_safe():
    source = get_template_config()
    source["optimize"]["fixed_params"] = ["long.forager.score_weights.volume"]
    with pytest.raises(ValueError, match="partial normalized-weight search"):
        prepare_config(source, verbose=False)
    source["optimize"]["fixed_params"] = ["long.forager.score_weights"]
    config = prepare_config(source, verbose=False)
    from optimize import apply_fine_tune_bounds
    apply_fine_tune_bounds(config, [], set())
    assert all(bound.low == bound.high for key, bound in gene_bounds(config).items() if "forager_score_weights" in key)


@pytest.mark.parametrize("selector,value", [
    ("bot.long.forager.score_weights.volume", 0.2),
    ("bot.long.forager.score_weights.volatility", 0.2),
])
@pytest.mark.parametrize("mirrored", [False, True])
def test_runtime_weight_pins_cannot_be_renormalized_by_search(selector, value, mirrored):
    source = get_template_config()
    if mirrored:
        source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    source["optimize"]["fixed_runtime_overrides"] = {selector: value}
    with pytest.raises(ValueError, match=rf"partial normalized-weight search.*{selector.split('.')[-1]}"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("patch", [
    {"bot.long.forager.score_weights.volume": 0.5},
    {"bot": {"long": {"forager": {"score_weights": {"volume": 0.5}}}}},
    {"bot.long.forager.score_weights.unilateralness": 0.5},
])
def test_scenario_weight_pins_fail_before_evaluation(patch):
    source = get_template_config()
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "pinned", "overrides": patch}])
    with pytest.raises(ValueError, match=r"scenarios\[0\].*pinned.*bot.long.forager.score_weights.*partial normalized-weight"):
        prepare_config(source, verbose=False)


def test_scenario_activation_cannot_move_an_omitted_positive_coordinate():
    source = get_template_config()
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "short-active", "overrides": {
        "bot.short.risk.total_wallet_exposure_limit": 1.0,
        "bot.short.forager.score_weights.unilateralness": 0.5,
    }}])
    with pytest.raises(ValueError, match=r"short-active.*bot.short.forager.score_weights.*unilateralness"):
        prepare_config(source, verbose=False)


def mirrored_weight_source(strategy="trailing_martingale"):
    source = weight_source({"score_weights": {
        name: [0.0, 1.0] for name in ("volume", "volatility", "ema_readiness")
    }})
    source["bot"]["global"]["strategy_kind"] = strategy
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    return source


def mirrored_weight_candidate(config, volume):
    bounds = gene_bounds(config)
    selected = {"volume": volume, "volatility": 0.2, "ema_readiness": 0.3}
    return build_optimizer_vector_config([
        selected.get(key.removeprefix("long_forager_score_weights_"), bound.low)
        if key.startswith("long_forager_score_weights_") else bound.low
        for key, bound in bounds.items()
    ], config)


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("weight", ["volume", "unilateralness"])
def test_mirrored_short_scenario_pins_validate_the_active_source_domains(strategy, weight):
    source = mirrored_weight_source(strategy)
    assert source["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    source["backtest"].update(suite_enabled=True, scenarios=[{
        "label": "mirrored-pin", "overrides": {
            f"bot.short.forager.score_weights.{weight}": 0.5,
        },
    }])
    before = deepcopy(source)
    with pytest.raises(ValueError, match=rf"mirrored-pin.*bot.short.forager.score_weights.*{weight}.*Mirrored search domains come from bot.long"):
        prepare_config(source, verbose=False)
    assert source == before


def test_mirroring_shadows_short_runtime_pins_and_keeps_directional_selection():
    source = mirrored_weight_source()
    source["bot"]["long"]["forager"]["always_include"] = ["BTC"]
    source["bot"]["short"]["forager"]["always_include"] = ["ETH"]
    source["optimize"]["fixed_runtime_overrides"] = {
        "bot.short.risk.total_wallet_exposure_limit": 0.0,
        "bot.short.forager.score_weights.volume": 0.5,
    }
    config = prepare_config(source, verbose=False)
    volumes = []
    for volume in (0.1, 0.9):
        candidate = mirrored_weight_candidate(config, volume)
        assert candidate["bot"]["short"]["risk"]["total_wallet_exposure_limit"] > 0
        assert candidate["bot"]["short"]["forager"]["score_weights"] == candidate["bot"]["long"]["forager"]["score_weights"]
        assert candidate["bot"]["short"]["forager"]["always_include"] == ["ETH"]
        volumes.append(candidate["bot"]["short"]["forager"]["score_weights"]["volume"])
    assert volumes[0] != volumes[1]


def test_source_runtime_pins_fix_mirrored_domains_before_scenario_pins():
    from config.bot import normalize_bot_forager_config
    from optimization.warmup import _apply_config_overrides

    source = mirrored_weight_source()
    fixed = {"volume": 0.2, "volatility": 0.3, "ema_readiness": 0.4, "unilateralness": 0.1}
    source["optimize"]["fixed_runtime_overrides"] = {
        f"bot.long.forager.score_weights.{name}": value for name, value in fixed.items()
    }
    patch = {"bot.short.forager.score_weights.volume": 0.5}
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "source-fixed", "overrides": patch}])
    config = prepare_config(source, verbose=False)
    results = []
    for volume in (0.1, 0.9):
        candidate = mirrored_weight_candidate(config, volume)
        _apply_config_overrides(candidate, patch)
        normalize_bot_forager_config(candidate, verbose=False)
        results.append(candidate["bot"]["short"]["forager"]["score_weights"])
    assert results[0] == results[1]
    assert results[0]["volume"] == pytest.approx(0.5 / 1.3)


def test_postmirror_long_scenario_pins_do_not_fix_short_source_domains():
    from config.bot import normalize_bot_forager_config
    from optimization.warmup import _apply_config_overrides

    source = mirrored_weight_source()
    fixed = {"volume": 0.2, "volatility": 0.3, "ema_readiness": 0.4, "unilateralness": 0.1}
    patch = {"bot.long.forager.score_weights": fixed}
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "postmirror", "overrides": patch}])
    config = prepare_config(source, verbose=False)
    short_volumes = []
    for volume in (0.1, 0.9):
        candidate = mirrored_weight_candidate(config, volume)
        _apply_config_overrides(candidate, patch)
        normalize_bot_forager_config(candidate, verbose=False)
        assert candidate["bot"]["long"]["forager"]["score_weights"] == pytest.approx(fixed)
        short_volumes.append(candidate["bot"]["short"]["forager"]["score_weights"]["volume"])
    assert short_volumes[0] != short_volumes[1]
    patch["bot.short.forager.score_weights.volume"] = 0.5
    with pytest.raises(ValueError, match=r"postmirror.*bot.short.forager.score_weights.*partial normalized-weight"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("field", ["max_positions", "total_wallet_exposure_limit"])
@pytest.mark.parametrize("activity", ["runtime_source_disabled", "scenario_target_disabled", "scenario_target_reactivated"])
def test_mirrored_weight_pins_follow_effective_runtime_and_scenario_activity(field, activity):
    from optimization.warmup import _apply_config_overrides

    source = mirrored_weight_source()
    patch = {"bot.short.forager.score_weights.volume": 0.5}
    if activity != "scenario_target_disabled":
        source["optimize"]["fixed_runtime_overrides"] = {f"bot.long.risk.{field}": 0}
    if activity != "runtime_source_disabled":
        patch[f"bot.short.risk.{field}"] = int(activity == "scenario_target_reactivated")
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": activity, "overrides": patch}])
    if activity == "scenario_target_reactivated":
        with pytest.raises(ValueError, match=rf"{activity}.*bot.short.forager.score_weights.*partial normalized-weight"):
            prepare_config(source, verbose=False)
    else:
        config = prepare_config(source, verbose=False)
        candidate = mirrored_weight_candidate(config, 0.9)
        _apply_config_overrides(candidate, patch)
        assert candidate["bot"]["short"]["risk"][field] == 0


def test_atomic_partial_weight_object_fails_before_evaluation():
    source = get_template_config()
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "partial-object", "overrides": {
        "bot.long.forager.score_weights": {"volume": 0.5},
    }}])
    with pytest.raises(ValueError, match=r"partial-object.*removes required bot.long.forager.score_weights.*complete object"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("whole_forager", [False, True])
@pytest.mark.parametrize("mirrored", [False, True])
def test_complete_scenario_weight_group_is_fixed_after_normalization(whole_forager, mirrored):
    from optimization.warmup import _apply_config_overrides
    from config.bot import normalize_bot_forager_config

    source = get_template_config()
    side = "short" if mirrored else "long"
    if mirrored:
        source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    fixed = {"volume": 0.2, "volatility": 0.3, "ema_readiness": 0.4, "unilateralness": 0.1}
    path, value = f"bot.{side}.forager.score_weights", fixed
    if whole_forager:
        path, value = f"bot.{side}.forager", deepcopy(source["bot"][side]["forager"])
        value["score_weights"] = fixed
    patch = {path: value}
    source["backtest"].update(suite_enabled=True, scenarios=[{"label": "fixed-group", "overrides": patch}])
    config = prepare_config(source, verbose=False)
    bounds = gene_bounds(config)
    for fraction in (0.1, 0.9):
        candidate = build_optimizer_vector_config([b.low + fraction * (b.high - b.low) for b in bounds.values()], config)
        _apply_config_overrides(candidate, patch)
        normalize_bot_forager_config(candidate, verbose=False)
        assert candidate["bot"][side]["forager"]["score_weights"] == pytest.approx(fixed)


@pytest.mark.asyncio
@pytest.mark.parametrize("mirrored", [False, True])
async def test_external_suite_pins_fail_before_market_or_dataset_preparation(monkeypatch, mirrored):
    import optimize_suite
    from optimize import _materialize_suite_run_contract

    source = get_template_config()
    if mirrored:
        source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    config = prepare_config(source, verbose=False)
    side = "short" if mirrored else "long"
    suite = {"enabled": True, "scenarios": [{"label": "external", "overrides": {
        f"bot.{side}.forager.score_weights.volume": 0.5,
    }}]}
    async def denied(*_args, **_kwargs):
        pytest.fail("weight authority must fail before loading markets")
    monkeypatch.setattr(optimize_suite, "load_markets", denied)
    with pytest.raises(ValueError, match=r"external.*partial normalized-weight search"):
        await optimize_suite.prepare_suite_contexts(config, suite, shared_array_manager=object())
    with pytest.raises(ValueError, match=r"external.*partial normalized-weight search"):
        _materialize_suite_run_contract(config, suite)


@pytest.mark.parametrize("anchored", [False, True])
def test_partial_fine_tune_weight_selection_fails_before_candidate_evaluation(anchored):
    from optimize import apply_fine_tune_bounds, install_anchored_fine_tune_plan

    config = prepare_config(get_template_config(), verbose=False)
    with pytest.raises(ValueError, match="partial normalized-weight search"):
        if anchored:
            install_anchored_fine_tune_plan(config, ["long.forager.score_weights.volume"], "unused",
                                           starting_configs_override=[get_template_config()])
        else:
            apply_fine_tune_bounds(config, ["long.forager.score_weights.volume"], set())


def test_explicit_recommended_preset_supplies_visible_ranges_and_is_offline(tmp_path, monkeypatch):
    import socket
    def denied(*_args, **_kwargs):
        pytest.fail("preset authoring must not contact any network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    source = get_template_config()
    source["optimize"]["bounds"] = {}
    before = deepcopy(source)
    prescribed = apply_recommended_bounds(source)
    assert source == before
    assert sum(bound.low != bound.high for bound in gene_bounds(prescribed).values()) > 1
    assert flattened(prescribed)["long_ema_span_0"] == [100.0, 2880.0, 10.0]
    infile, outfile = tmp_path / "input.json", tmp_path / "search.json"
    infile.write_text(json.dumps(source))
    assert main([str(infile), str(outfile)]) == 0
    assert gene_bounds(prepare_config(json.loads(outfile.read_text()), verbose=False)) == gene_bounds(prescribed)
    with pytest.raises(FileExistsError):
        main([str(infile), str(outfile)])


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_recommended_preset_preserves_strategy_hsl_and_positive_optional_weight(strategy, scope):
    from config.hsl import generated_template

    source = generated_template(get_template_config(), scope)
    source["bot"]["global"]["strategy_kind"] = strategy
    source["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 0.5
    source["optimize"]["bounds"] = {}
    prescribed = apply_recommended_bounds(source)
    assert prescribed["bot"]["global"]["strategy_kind"] == strategy
    assert prescribed["bot"]["global"]["hsl"]["scope"] == scope
    assert flattened(prescribed)["long_forager_score_weights_unilateralness"] == [0.0, 1.0, 0.01]
    assert gene_bounds(prepare_config(prescribed, verbose=False)) == gene_bounds(prescribed)


@pytest.mark.parametrize("missing", ["leaf", "bounds", "optimize"])
def test_v8_implicit_ranges_are_materialized_before_new_omission_semantics(missing):
    source = released_v8_policy()
    if missing == "leaf":
        del source["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"]
    elif missing == "bounds":
        source["optimize"]["bounds"] = {}
    else:
        source["optimize"].pop("bounds")
    source["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 42.5
    migrated = migrate(source, memory_span_minutes=123.5)
    bounds = flattened(migrated.config)
    assert bounds["long_ema_span_0"] == [100.0, 2880.0, 10.0]
    assert bounds["long_hsl_red_threshold"] == [0.15, 0.15, 0.001]
    assert bounds["long_unstuck_ema_dist"] == [-0.2, 0.01, 0.0001]
    assert bounds["long_unilateralness_ema_span_1m"] == [42.5, 42.5]
    assert any("materialized released-v8 inherited search" in item["reason"] for item in migrated.changes)


def test_v8_positive_omitted_weight_keeps_its_domain_and_requires_an_explicit_remedy():
    source = released_v8_policy()
    source["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 0.5
    for best_effort in (False, True):
        with pytest.raises(ValueError, match=r"bot.long.forager.score_weights.*unilateralness.*freeze the whole"):
            migrate(source, memory_span_minutes=123.5, best_effort=best_effort)


def test_v8_runtime_weight_pin_requires_freezing_its_normalized_group():
    source = released_v8_policy()
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.forager.score_weights.volume": 0.2}
    with pytest.raises(ValueError, match=r"partial normalized-weight search.*volume"):
        migrate(source, memory_span_minutes=123.5)
    source["optimize"]["fixed_params"] = ["bot.long.forager.score_weights"]
    migrated = migrate(source, memory_span_minutes=123.5)
    assert migrate(migrated.config).config == migrated.config


@pytest.mark.parametrize("parts", [
    ("future_scope", "unknown"),
    ("global", "future_policy", "unknown"),
    ("global", "unstuck", "future_span"),
    ("long", "forager", "score_weights", "future_signal"),
    ("long", "risk", "position_exposure_enforcer", "future_threshold"),
    ("long", "strategy", "trailing_martingale", "entry", "future_span"),
])
def test_v8_materializing_ranges_preserves_unknown_bounds_for_strict_rejection(parts):
    source = released_v8_policy()
    target = source["optimize"]["bounds"]
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = [0.0, 1.0]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="unknown configuration field"):
        migrate(source, memory_span_minutes=123.5)
    assert source == original


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("owner,legacy", [
    ("position_exposure_enforcer", "position_exposure_enforcer_threshold"),
    ("total_exposure_enforcer", "total_exposure_enforcer_threshold"),
])
def test_v8_materializing_ranges_cannot_erase_conflicting_feature_bound_aliases(reverse, owner, legacy):
    source = released_v8_policy()
    risk = source["optimize"]["bounds"]["long"]["risk"]
    aliases = [(legacy, [0.5, 0.75]), (owner, {"threshold": [0.9, 1.0]})]
    risk.pop(legacy)
    risk.update(reversed(aliases) if reverse else aliases)
    with pytest.raises(ValueError, match="conflict"):
        migrate(source, memory_span_minutes=123.5)


def test_frozen_range_materializer_overlays_only_missing_canonical_fields():
    from config.migrations.released_v8 import materialize_released_v8_optimizer_bounds

    source = get_template_config()
    source["optimize"]["bounds"] = {
        "global": {"unknown": {"future": [2.0, 3.0]}},
        "long": {
            "risk": {"position_exposure_enforcer": {'threshold_ratio': [0.9, 0.95], "future": [4.0, 5.0]}},
            "forager": {"score_weights": {"volume": [0.2, 0.8], "future": [6.0, 7.0]}},
        },
    }
    authored = deepcopy(source["optimize"]["bounds"])
    materialize_released_v8_optimizer_bounds(source, changes=[])
    bounds = source["optimize"]["bounds"]
    assert bounds["global"] == authored["global"]
    assert bounds["long"]["risk"]["position_exposure_enforcer"] == authored["long"]["risk"]["position_exposure_enforcer"]
    for name in ("volume", "future"):
        assert bounds["long"]["forager"]["score_weights"][name] == authored["long"]["forager"]["score_weights"][name]
    assert bounds["long"]["forager"]["score_weights"]["volatility"] == [0, 1, 0.01]
    assert bounds["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0_minutes"] == [100.0, 2880.0, 10.0]


@pytest.mark.parametrize("transform,source_parts,target_parts,configured,domain", [
    ("couple_unstuck_ema_spans", ("strategy", "trailing_martingale", "entry", "ema_span_0_minutes"),
     ("long", "unstuck", "ema_span_0_minutes"), 42.5, [100.0, 200.0]),
    ("mirror_short_from_long", ("risk", "total_wallet_exposure_limit"),
     ("short", "risk", "total_wallet_exposure_limit"), 0.0, [0.5, 1.5]),
    ("lossless_close_trailing", ("strategy", "trailing_martingale", "close", "retracement_base_fraction"),
     ("long", "strategy", "trailing_martingale", "close", "threshold_base_fraction"), 0.0, [0.01, 0.02]),
])
def test_explicit_relationships_derive_effective_targets_without_opening_target_genes(
    transform, source_parts, target_parts, configured, domain,
):
    source = get_template_config()
    target = source["bot"]
    for part in target_parts[:-1]:
        target = target[part]
    target[target_parts[-1]] = configured
    source["optimize"]["enable_overrides"] = [transform]
    source["optimize"]["bounds"] = {"long": {}}
    target = source["optimize"]["bounds"]["long"]
    for part in source_parts[:-1]:
        target = target.setdefault(part, {})
    target[source_parts[-1]] = domain
    config = prepare_config(source, verbose=False)
    key_paths = get_optimization_key_paths(config)
    bounds = extract_bounds_tuple_list_from_config(config)
    varied = [path for (_key, path), bound in zip(key_paths, bounds) if bound.low != bound.high]
    assert varied == [("bot", "long", *source_parts)]
    if transform == "couple_unstuck_ema_spans":
        assert ("bot", *target_parts) not in [path for _key, path in key_paths]
    else:
        gene = next(bound for (_key, path), bound in zip(key_paths, bounds) if path == ("bot", *target_parts))
        assert gene.low == gene.high == configured
    values = []
    for corner in ("low", "high"):
        candidate = build_optimizer_vector_config([getattr(bound, corner) for bound in bounds], config)
        target = candidate["bot"]
        for part in target_parts:
            target = target[part]
        values.append(target)
    assert values == domain
