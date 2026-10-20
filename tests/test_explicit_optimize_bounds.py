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
    return flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=config["live"]["strategy_kind"])


def gene_bounds(config):
    return dict(zip((key for key, _path in get_optimization_key_paths(config)),
                    extract_bounds_tuple_list_from_config(config)))


@pytest.mark.parametrize("omission", ["optimize", "bounds", "empty", "side", "leaf"])
@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
def test_omitted_dimensions_freeze_effective_values_through_exports(omission, strategy):
    source = get_template_config()
    source["live"]["strategy_kind"] = strategy
    if strategy == "trailing_martingale":
        source["bot"]["long"]["strategy"][strategy]["entry"]["ema_span_0"] = 123.5
        branch = source["optimize"]["bounds"]["long"]["strategy"][strategy]["entry"]
    else:
        source["bot"]["long"]["strategy"][strategy]["ema_span_0"] = 123.5
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
        branch.pop("ema_span_0")
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
        "long": {"strategy": {"trailing_martingale": {"entry": {"ema_span_0": [100.5, 200.5, 0.5]}}}}
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
    source = weight_source({"score_weights_volume": [0.1, 1.0]})
    with pytest.raises(ValueError, match=r"bot.long.forager.score_weights.*ema_readiness, volatility.*freeze the whole"):
        prepare_config(source, verbose=False)


def test_weight_validation_respects_the_supplied_search_space():
    from optimization.config_adapter import validate_optimize_bounds_against_bot_config

    config = prepare_config(get_template_config(), verbose=False)
    with pytest.raises(ValueError, match="partial normalized-weight search"):
        validate_optimize_bounds_against_bot_config(config, {
            "long": {"forager": {"score_weights_volume": [0.1, 1.0]}}
        })


def test_whole_positive_weight_group_keeps_normalization_and_zero_omission():
    source = weight_source({f"score_weights_{name}": [0.1, 1.0] for name in ("volume", "volatility", "ema_readiness")})
    config = prepare_config(source, verbose=False)
    bounds = gene_bounds(config)
    result = build_optimizer_vector_config([bound.high for bound in bounds.values()], config)
    weights = result["bot"]["long"]["forager"]["score_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["unilateralness"] == 0.0


@pytest.mark.parametrize("low,allowed", [(0.1, True), (0.0, False)])
def test_omitted_zero_weight_is_safe_except_at_an_unauthorized_fallback(low, allowed):
    source = weight_source({"score_weights_volume": [low, 1.0]})
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
def test_runtime_weight_pins_cannot_be_renormalized_by_search(selector, value):
    source = get_template_config()
    source["optimize"]["fixed_runtime_overrides"] = {selector: value}
    with pytest.raises(ValueError, match=rf"partial normalized-weight search.*{selector.split('.')[-1]}"):
        prepare_config(source, verbose=False)


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
@pytest.mark.parametrize("scope", ["coin", "pside", "portfolio"])
def test_recommended_preset_preserves_strategy_hsl_and_positive_optional_weight(strategy, scope):
    from config.hsl import generated_template

    source = generated_template(get_template_config(), scope)
    source["live"]["strategy_kind"] = strategy
    source["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 0.5
    source["optimize"]["bounds"] = {}
    prescribed = apply_recommended_bounds(source)
    assert prescribed["live"]["strategy_kind"] == strategy
    assert prescribed["bot"]["risk"]["hsl"]["scope"] == scope
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
