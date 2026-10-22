"""Logical optimizer dimensions expand onto stable directional native fields."""

from copy import deepcopy

import pytest

from config.bot_inheritance import install_bot_policy, resolve_bot_policy
from config.inherited_bounds import prepare_bound_plan

def real_prepared(*, short_enabled=True, short_pin=False):
    from config.schema import get_template_config
    from test_bot_inheritance import real_catalog
    config = get_template_config()
    raw = {"portfolio": deepcopy(config["bot"]["portfolio"]), "global": {"enabled": True},
           "short": {"enabled": short_enabled}}
    if short_pin:
        raw["short"]["unstuck"] = {"close_fraction": 0.041}
    install_bot_policy(config, resolve_bot_policy(raw, real_catalog()))
    config["optimize"]["bounds"] = {"global": {"unstuck": {"close_fraction": [0.02, 0.06]}}}
    prepare_bound_plan(config)
    return config


@pytest.mark.parametrize("short_enabled", [False, True])
@pytest.mark.parametrize("short_pin", [False, True])
def test_shape_and_adapter_use_one_common_coordinate_without_losing_side_pins(short_enabled, short_pin):
    from optimization.config_adapter import get_optimization_key_paths, resolve_optimization_bound_path
    from optimization.shape import build_optimization_shape
    config = real_prepared(short_enabled=short_enabled, short_pin=short_pin)
    shape = build_optimization_shape(config)
    selected = [(key, path, bound) for (key, path), bound in zip(shape.key_paths, shape.bounds)
                if path[-2:] == ("unstuck", "close_fraction")]
    assert selected[0][0] == "bot.global.unstuck.close_fraction"
    assert selected[0][1] == ("bot", "global", "unstuck", "close_fraction")
    assert selected[0][2].low == 0.02 and selected[0][2].high == 0.06
    assert len(selected) == (2 if short_pin else 1)
    if short_pin:
        assert selected[1][0] == "bot.short.unstuck.close_fraction"
        assert selected[1][2].low == selected[1][2].high == 0.041
    assert shape.key_paths == tuple(get_optimization_key_paths(config))
    assert resolve_optimization_bound_path(config, "long_unstuck_close_pct") == selected[0][1]


def test_adapter_target_fixed_selector_does_not_create_a_directional_axis():
    from optimization.shape import build_optimization_shape
    config = real_prepared()
    config["optimize"]["fixed_params"] = ["short.unstuck.close_fraction"]
    prepare_bound_plan(config)
    shape = build_optimization_shape(config)
    selected = [(key, bound) for (key, path), bound in zip(shape.key_paths, shape.bounds)
                if path[-2:] == ("unstuck", "close_fraction")]
    assert len(selected) == 1 and selected[0][0] == "bot.global.unstuck.close_fraction"
    assert selected[0][1].low == selected[0][1].high == 0.041


@pytest.mark.parametrize("short_pin", [False, True])
def test_actual_vector_writer_fans_common_gene_out_and_retains_sparse_source(short_pin):
    from config.universe import universe_source_view
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config
    config = real_prepared(short_pin=short_pin)
    before = deepcopy(config)
    shape = build_optimization_shape(config)
    vector = [0.055 if key == "bot.global.unstuck.close_fraction" else bound.low
              for (key, _), bound in zip(shape.key_paths, shape.bounds)]
    candidate = build_optimizer_vector_config(vector, config, key_paths=shape.key_paths)
    assert candidate["bot"]["long"]["unstuck"]["close_fraction"] == 0.055
    assert candidate["bot"]["short"]["unstuck"]["close_fraction"] == (0.041 if short_pin else 0.055)
    source = universe_source_view(candidate)["bot"]
    assert source["global"]["unstuck"]["close_fraction"] == 0.055
    assert ("unstuck" in source.get("short", {})) is short_pin
    assert "long" not in source
    assert universe_source_view(config) == universe_source_view(before)


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_fixed_same_file_replay_keeps_frozen_membership_without_rereading(tmp_path, monkeypatch, field):
    from config.bot_inheritance import get_bot_policy
    from config.universe import materialize_side_universe_sources, get_side_universe_sources
    from optimization.warmup import _apply_config_overrides
    config = real_prepared()
    source = str(tmp_path / "coins.json")
    from config.bot_inheritance import apply_bot_config_patch
    apply_bot_config_patch(config, {f"bot.global.universe.{field}": source})
    materialize_side_universe_sources(config, field, {"long": ["BTC"], "short": ["BTC"]})
    prepare_bound_plan(config)
    config.setdefault("backtest", {})["coins"] = {"binance": ["BTC"]}
    monkeypatch.setattr(type(tmp_path), "open", lambda *_a, **_k: pytest.fail("candidate replay must not reopen files"))
    _apply_config_overrides(config, {f"bot.global.universe.{field}": source}, preserve_prepared_universe=True)
    assert get_side_universe_sources(config, field) == {"long": ["BTC"], "short": ["BTC"]}
    assert get_bot_policy(config) is not None
    with pytest.raises(ValueError, match="prepare its membership"):
        _apply_config_overrides(config, {f"bot.global.universe.{field}": source + ".new"}, preserve_prepared_universe=True)


def test_fine_tune_effective_target_freezes_common_domain_without_a_side_copy():
    from config.inherited_bounds import optimize_bounds_source_view
    from optimize import apply_fine_tune_bounds
    config = real_prepared()
    config["optimize"]["fixed_params"] = ["short.unstuck.close_fraction"]
    prepare_bound_plan(config)
    apply_fine_tune_bounds(config, [], set())
    raw = optimize_bounds_source_view(config)["optimize"]["bounds"]
    assert raw["global"]["unstuck"]["close_fraction"] == [0.041, 0.041]
    assert "short" not in raw


def test_anchored_vector_installs_common_fixed_values_through_gene_authority():
    from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY
    from optimization.warmup import build_optimizer_vector_config
    from config.universe import universe_source_view
    config = real_prepared(short_pin=True)
    config[ANCHOR_PLAN_KEY] = {"anchors": [{"source": "public synthetic fixture", "fixed_values": [
        {"key": "bot.global.unstuck.close_fraction", "path": ["bot", "global", "unstuck", "close_fraction"], "value": 0.05}
    ]}], "key_paths": [], "tunable_keys": [], "fixed_keys": ["bot.global.unstuck.close_fraction"]}
    candidate = build_optimizer_vector_config([0.0], config)
    assert candidate["bot"]["long"]["unstuck"]["close_fraction"] == 0.05
    assert candidate["bot"]["short"]["unstuck"]["close_fraction"] == 0.041
    source = universe_source_view(candidate)["bot"]
    assert source["global"]["unstuck"]["close_fraction"] == 0.05
    assert source["short"]["unstuck"]["close_fraction"] == 0.041
    assert "long" not in source


def test_mirrored_and_coupled_targets_do_not_create_unused_independent_axes():
    from config.inherited_bounds import optimize_bounds_source_view
    from optimization.config_adapter import get_optimization_key_paths
    config = real_prepared(short_pin=True)
    raw = optimize_bounds_source_view(config)["optimize"]["bounds"]
    raw["short"] = {"unstuck": {"close_fraction": [0.01, 0.04]}}
    raw["global"]["unstuck"]["ema_span_0_minutes"] = [100.0, 200.0]
    prepare_bound_plan(config, authored_bounds=raw)
    config["optimize"]["enable_overrides"] = ["mirror_short_from_long", "couple_unstuck_ema_spans"]
    paths = get_optimization_key_paths(config)
    assert ("bot.global.unstuck.close_fraction", ("bot", "global", "unstuck", "close_fraction")) in paths
    assert not any(path[:2] == ("bot", "short") for _, path in paths)
    assert not any(path[2:3] == ("unstuck",) and path[-1] in {"ema_span_0_minutes", "ema_span_1_minutes"}
                   for _, path in paths)
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == raw


def test_fixed_source_install_preserves_sparse_policy_and_leaves_numeric_pins_for_their_owner(tmp_path):
    from config.universe import universe_source_view, materialize_side_universe_sources, get_side_universe_sources
    from optimization.warmup import apply_optimizer_universe_sources
    from config.bot_inheritance import apply_bot_config_patch
    config = real_prepared()
    apply_bot_config_patch(config, {"bot.global.universe.ignored_coins": ["ETH"]})
    materialize_side_universe_sources(config, "ignored_coins", {"long": ["ETH"], "short": ["ETH"]})
    source = tmp_path / "new.json"
    source.write_text('["BTC"]')
    config["optimize"]["fixed_runtime_overrides"] = {
        "bot.global.universe.approved_coins": str(source),
        "bot.global.risk.total_wallet_exposure_limit": 99.0,
    }
    before = universe_source_view(config)
    apply_optimizer_universe_sources(config)
    after = universe_source_view(config)
    assert after["bot"]["global"]["universe"]["approved_coins"] == str(source)
    assert "universe" not in after["bot"]["short"]
    assert "long" not in after["bot"]
    assert after["bot"]["global"].get("risk") == before["bot"]["global"].get("risk")
    assert get_side_universe_sources(config, "ignored_coins") == {"long": ["ETH"], "short": ["ETH"]}


def weight_search_config(*, side_domain=False):
    config = real_prepared()
    raw = {"global": {"forager": {"score_weights": {"volume": [0.1, 1.0]}}}}
    if side_domain:
        raw["short"] = deepcopy(raw["global"])
    prepare_bound_plan(config, authored_bounds=raw)
    return config


def test_global_complete_runtime_weight_pin_freezes_the_inherited_group():
    from optimization.config_adapter import validate_forager_weight_search
    config = weight_search_config()
    config["optimize"]["fixed_runtime_overrides"] = {
        f"bot.global.forager.score_weights.{name}": value for name, value in {
            "volume": 0.2, "volatility": 0.3, "ema_readiness": 0.5, "unilateralness": 0.0,
        }.items()
    }
    validate_forager_weight_search(config)


def test_global_runtime_pin_cannot_freeze_an_independently_owned_side_weight_domain():
    from optimization.config_adapter import validate_forager_weight_search
    config = weight_search_config(side_domain=True)
    config["optimize"]["fixed_runtime_overrides"] = {
        f"bot.global.forager.score_weights.{name}": value for name, value in {
            "volume": 0.2, "volatility": 0.3, "ema_readiness": 0.5, "unilateralness": 0.0,
        }.items()
    }
    with pytest.raises(ValueError, match=r"bot.short.forager.score_weights.*partial normalized-weight"):
        validate_forager_weight_search(config)


def test_parent_side_replacement_inherits_required_weights_instead_of_removing_them():
    from optimization.config_adapter import validate_forager_weight_search
    config = real_prepared()
    config["backtest"].update(suite_enabled=True, scenarios=[{
        "label": "inherit", "overrides": {"bot.long": {"enabled": True}},
    }])
    validate_forager_weight_search(config)


@pytest.mark.parametrize("sample,expected", [(4.5, 4), (5.5, 6), (0.49, 0)])
def test_count_gene_uses_existing_runtime_rounding_without_weakening_authored_integer_type(sample, expected):
    from config.bot_inheritance import get_bot_policy, apply_bot_policy_patch
    from config.universe import universe_source_view
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config
    config = real_prepared()
    prepare_bound_plan(config, authored_bounds={"global": {"risk": {"max_positions": [0.0, 8.0]}}})
    shape = build_optimization_shape(config)
    vector = [sample if key == "bot.global.risk.max_positions" else bound.low
              for (key, _), bound in zip(shape.key_paths, shape.bounds)]
    candidate = build_optimizer_vector_config(vector, config, key_paths=shape.key_paths)
    for side in ("long", "short"):
        assert candidate["bot"][side]["risk"]["max_positions"] == expected
    assert universe_source_view(candidate)["bot"]["global"]["risk"]["max_positions"] == expected
    with pytest.raises(TypeError, match="serialization type"):
        apply_bot_policy_patch(get_bot_policy(config), {"bot.global.risk.max_positions": 4.5})


@pytest.mark.parametrize("side_pin", [False, True])
def test_actual_anchor_install_uses_effective_targets_and_preserves_base_side_pins(side_pin):
    from config.schema import get_template_config
    from config.universe import universe_source_view
    from optimize import install_anchored_fine_tune_plan
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config
    config = real_prepared(short_pin=side_pin)
    anchor = get_template_config()
    anchor["bot"]["global"]["unstuck"]["close_fraction"] = 0.045
    anchor["bot"]["long"] = {"unstuck": {"close_fraction": 0.052}}
    anchor["bot"]["short"] = {"unstuck": {"close_fraction": 0.052 if not side_pin else 0.041}}
    install_anchored_fine_tune_plan(config, ["global.risk.total_wallet_exposure_limit"],
                                  "unused", starting_configs_override=[anchor])
    shape = build_optimization_shape(config)
    candidate = build_optimizer_vector_config([bound.low for bound in shape.bounds], config)
    assert candidate["bot"]["long"]["unstuck"]["close_fraction"] == 0.052
    assert candidate["bot"]["short"]["unstuck"]["close_fraction"] == (0.041 if side_pin else 0.052)
    source = universe_source_view(candidate)["bot"]
    assert source["global"]["unstuck"]["close_fraction"] == 0.052
    assert ("unstuck" in source.get("short", {})) is side_pin


def test_anchor_and_seed_reject_values_that_one_shared_coordinate_cannot_represent(caplog):
    from config.schema import get_template_config
    from optimize import install_anchored_fine_tune_plan, _build_starting_seed_config, config_to_individual
    from optimization.shape import build_optimization_shape
    config = real_prepared()
    anchor = get_template_config()
    anchor["bot"]["long"] = {"unstuck": {"close_fraction": 0.052}}
    anchor["bot"]["short"] = {"unstuck": {"close_fraction": 0.048}}
    with pytest.raises(ValueError, match="at least one valid"):
        install_anchored_fine_tune_plan(config, ["global.risk.total_wallet_exposure_limit"],
                                      "unused", starting_configs_override=[anchor])
    assert "declare explicit side bounds" in caplog.text
    seed = _build_starting_seed_config(anchor)
    shape = build_optimization_shape(config)
    with pytest.raises(ValueError, match="shared coordinate.*explicit side bounds"):
        config_to_individual(seed, shape.bounds, optimization_shape=shape)


def test_seed_stream_retains_an_existing_prepared_binding_plan_when_shape_is_omitted():
    from optimize import configs_to_individuals
    from optimization.shape import build_optimization_shape
    config = real_prepared(short_pin=True)
    shape = build_optimization_shape(config)
    vectors = configs_to_individuals([config], shape.bounds, 6)
    assert len(vectors) == 1 and len(vectors[0]) == len(shape.bounds)
    values = dict(zip((key for key, _ in shape.key_paths), vectors[0]))
    assert values["bot.global.unstuck.close_fraction"] == 0.041
    assert values["bot.short.unstuck.close_fraction"] == 0.041
