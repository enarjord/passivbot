"""Review regressions for selector compatibility, nullable overrides and search domains."""

from copy import deepcopy

import pytest

from config import get_template_config, prepare_config
from config.overrides import parse_overrides
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_bound_selectors
from config_utils import clean_config
from optimization.config_adapter import validate_optimize_bounds_against_bot_config


@pytest.mark.parametrize(
    "selector,sides",
    [
        ("long.risk.*", ("long",)),
        ("bot.short.risk.*", ("short",)),
        ("risk.*", ("long", "short")),
        ("*.risk.*", ("long", "short")),
        ("risk", ("long", "short")),
        ("long.risk", ("long",)),
    ],
)
def test_legacy_risk_selectors_include_only_the_original_cooldown_leaf(selector, sides):
    cfg = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(cfg["optimize"]["bounds"], strategy_kind="trailing_martingale")
    for side in ("long", "short"):
        bounds[f"{side}_entry_cooldown_weights_minutes_exposure_ratio"] = [0.0, 10.0]
    selected = resolve_bound_selectors(cfg, [selector], bounds)
    assert {k for k in selected if "cooldown" in k} == {
        f"{side}_risk_entry_cooldown_minutes" for side in sides
    }
    assert all(k.split("_", 1)[0] in sides for k in selected)


def test_coin_override_can_clear_ceiling_after_disabling_inherited_weights():
    cfg = get_template_config()
    cfg["bot"]["long"]["entry_cooldown"].update(
        max_duration_minutes=30.0,
        weights_minutes={"exposure_ratio": 10.0, "adverse_directionality": 20.0},
    )
    cfg["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "entry_cooldown": {
                        "base_duration_minutes": 45.0,
                        "max_duration_minutes": None,
                        "weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
                    }
                }
            }
        }
    }
    prepared = parse_overrides(prepare_config(cfg, verbose=False), verbose=False)
    patch = prepared["coin_overrides"]["BTC"]["bot"]["long"]["entry_cooldown"]
    assert "max_duration_minutes" in patch and patch["max_duration_minutes"] is None
    assert patch["base_duration_minutes"] == 45.0
    again = parse_overrides(prepare_config(clean_config(prepared), verbose=False), verbose=False)
    assert again["coin_overrides"] == prepared["coin_overrides"]
    invalid = deepcopy(cfg)
    invalid["coin_overrides"]["BTC"]["bot"]["long"]["entry_cooldown"]["weights_minutes"][
        "exposure_ratio"
    ] = 1.0
    with pytest.raises(ValueError, match="finite"):
        parse_overrides(prepare_config(invalid, verbose=False), verbose=False)


@pytest.mark.parametrize(
    "key,bound",
    [
        ("unilateralness_ema_span_1m", [0.0, 60.0]),
        ("unilateralness_ema_span_1m", [60.0, 100001.0]),
        ("unilateralness_ema_span_1m", [60.0, float("nan")]),
        ("entry_cooldown_weights_minutes_exposure_ratio", [-1.0, 10.0]),
        ("entry_cooldown_weights_minutes_adverse_directionality", [-1.0, 10.0]),
        ("entry_cooldown_weights_minutes_adverse_directionality", [0.0, float("inf")]),
        ("forager_score_weights_unilateralness", [-1.0, 1.0]),
        ("entry_cooldown_min_duration_minutes", [-1.0, 10.0]),
        ("entry_cooldown_max_duration_minutes", [0.0, float("inf")]),
    ],
)
@pytest.mark.parametrize("side", ["long", "short"])
def test_adaptive_optimizer_rejects_invalid_domains_before_sampling(key, bound, side):
    cfg = get_template_config()
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    with pytest.raises(ValueError, match="optimize.bounds"):
        validate_optimize_bounds_against_bot_config(cfg, {f"{side}_{key}": bound})


def test_prepare_config_rejects_nested_invalid_search_domain():
    cfg = get_template_config()
    cfg["optimize"]["bounds"]["long"]["forager"]["unilateralness_ema_span_1m"] = [0.0, 60.0]
    with pytest.raises(ValueError, match="optimize.bounds"):
        prepare_config(cfg, verbose=False)


def test_adaptive_optimizer_accepts_float_span_boundaries_and_zero_weights():
    cfg = get_template_config()
    validate_optimize_bounds_against_bot_config(
        cfg,
        {
            "long_unilateralness_ema_span_1m": [1.0, 100000.0, 0.5],
            "short_unilateralness_ema_span_1m": [10.5, 20.5],
            "long_entry_cooldown_weights_minutes_exposure_ratio": [0.0, 10.0],
            "short_entry_cooldown_weights_minutes_adverse_directionality": 0.0,
        },
    )


def test_risk_wildcards_reach_fine_tune_fixed_params_and_composition():
    from types import SimpleNamespace
    from optimize import _resolve_fine_tune_key_sets
    from config.overrides import get_allowed_modifications
    from tools.compose_coin_overrides import _resolve_override_params

    cfg = prepare_config(get_template_config(), verbose=False)
    _, tunable, _, fixed = _resolve_fine_tune_key_sets(cfg, ["long.risk.*"])
    assert "long_risk_entry_cooldown_minutes" in tunable
    assert "long_risk_entry_cooldown_minutes" not in fixed
    assert "short_risk_entry_cooldown_minutes" in fixed
    cfg["optimize"]["fixed_params"] = ["risk.*"]
    _, _, configured_fixed, _ = _resolve_fine_tune_key_sets(cfg, [])
    assert {
        "long_risk_entry_cooldown_minutes",
        "short_risk_entry_cooldown_minutes",
    } <= configured_fixed
    paths, _ = _resolve_override_params(
        "long.risk.*", [SimpleNamespace(config=cfg)], cfg, get_allowed_modifications()
    )
    assert ("bot", "long", "entry_cooldown", "base_duration_minutes") in paths
    assert not any("weights_minutes" in path for path in paths)


@pytest.mark.parametrize("leaf", ["base_duration_minutes", "min_duration_minutes"])
def test_only_the_nullable_ceiling_accepts_null(leaf):
    cfg = get_template_config()
    cfg["coin_overrides"] = {"BTC": {"bot": {"long": {"entry_cooldown": {leaf: None}}}}}
    with pytest.raises(TypeError, match="may not be null"):
        parse_overrides(prepare_config(cfg, verbose=False), verbose=False)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "floor,ceiling,bounds,valid",
    [
        (0, 30, {"min_duration_minutes": [0, 20], "max_duration_minutes": [10, 30]}, False),
        (0, 30, {"min_duration_minutes": [0, 10], "max_duration_minutes": [10, 30]}, True),
        (0, 10, {"min_duration_minutes": [0, 20]}, False),
        (20, 30, {"max_duration_minutes": [10, 30]}, False),
        (0, None, {"min_duration_minutes": [0, 20]}, True),
    ],
)
def test_cooldown_search_floor_cannot_exceed_any_reachable_ceiling(side, floor, ceiling, bounds, valid):
    cfg = get_template_config()
    cfg["bot"][side]["entry_cooldown"].update(
        min_duration_minutes=floor, max_duration_minutes=ceiling
    )
    cfg["optimize"]["bounds"][side]["entry_cooldown"].update(bounds)
    if valid:
        prepare_config(cfg, verbose=False)
    else:
        with pytest.raises(ValueError, match="highest min_duration_minutes.*lowest max_duration_minutes"):
            prepare_config(cfg, verbose=False)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
@pytest.mark.parametrize("bound_span", [None, 240.5])
def test_rms_history_budget_covers_optimizer_only_consumers(side, consumer, bound_span):
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["max_warmup_minutes"] = 1
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 90.0
    bounds = cfg["optimize"]["bounds"][side]
    bounds["risk"].update(n_positions=[1, 1], total_wallet_exposure_limit=[1.0, 1.0])
    if consumer == "forager":
        bounds["forager"]["score_weights"] = {"unilateralness": [0.0, 1.0]}
    else:
        bounds["entry_cooldown"]["weights_minutes"] = {"adverse_directionality": [0.0, 30.0]}
    if bound_span is not None:
        bounds["forager"]["unilateralness_ema_span_1m"] = [60.0, bound_span]
    cfg = prepare_config(cfg, verbose=False)
    import math
    assert compute_backtest_warmup_minutes(cfg) == math.ceil(20 * (bound_span or 60.0)) + 1


@pytest.mark.parametrize("side", ["long", "short"])
def test_optimizer_can_sample_ceiling_from_default_null(side):
    from optimization.config_adapter import get_optimization_key_paths
    from optimization.warmup import build_optimizer_max_config

    cfg = get_template_config()
    assert cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] is None
    cfg["optimize"]["bounds"][side]["entry_cooldown"]["max_duration_minutes"] = [10.0, 30.0]
    cfg = prepare_config(cfg, verbose=False)
    paths = dict(get_optimization_key_paths(cfg))
    assert paths[f"{side}_entry_cooldown_max_duration_minutes"] == (
        "bot", side, "entry_cooldown", "max_duration_minutes"
    )
    candidate = build_optimizer_max_config(cfg)
    # The short side may be disabled by template bounds, in which case its lower bound is used.
    assert candidate["bot"][side]["entry_cooldown"]["max_duration_minutes"] in (10.0, 30.0)


@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
def test_optimizer_rms_history_does_not_delay_disabled_candidate(consumer):
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["max_warmup_minutes"] = 1
    cfg["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 90.0
    bounds = cfg["optimize"]["bounds"]["long"]
    bounds["forager"]["unilateralness_ema_span_1m"] = [60.0, 240.5]
    if consumer == "forager":
        bounds["forager"]["score_weights"] = {"unilateralness": [0.0, 1.0]}
    else:
        bounds["entry_cooldown"]["weights_minutes"] = {"adverse_directionality": [0.0, 30.0]}
    cfg = prepare_config(cfg, verbose=False)
    assert compute_backtest_warmup_minutes(cfg) == 4811
    assert compute_backtest_warmup_minutes(cfg, for_trade_activation=True) == 1


def test_active_rms_consumer_uses_larger_span_bound_and_zero_consumers_need_no_rms_history():
    from warmup_utils import compute_backtest_warmup_minutes

    cfg = get_template_config()
    cfg["live"]["max_warmup_minutes"] = 1
    cfg["optimize"]["bounds"]["long"]["forager"]["unilateralness_ema_span_1m"] = [60.0, 240.5]
    cfg = prepare_config(cfg, verbose=False)
    assert compute_backtest_warmup_minutes(cfg) == 1
    cfg["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = 1.0
    assert compute_backtest_warmup_minutes(cfg) == 4811


def test_unbounded_seed_ceiling_projects_to_finite_search_upper_bound():
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config
    from optimize import config_to_individual

    cfg = get_template_config()
    cfg["optimize"]["bounds"]["long"]["entry_cooldown"]["max_duration_minutes"] = [10.0, 30.0]
    cfg = prepare_config(cfg, verbose=False)
    shape = build_optimization_shape(cfg)
    vector = config_to_individual(cfg, shape.bounds, optimization_shape=shape, clamp_context="seed")
    candidate = build_optimizer_vector_config(vector, cfg, key_paths=shape.key_paths)
    assert candidate["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] == 30.0
    assert cfg["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] is None


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("modifier", ["exposure_ratio", "adverse_directionality"])
def test_joint_ceiling_and_modifier_search_from_null_default(side, modifier):
    from optimization.warmup import validate_optimizer_effective_configs

    cfg = get_template_config()
    bounds = cfg["optimize"]["bounds"][side]["entry_cooldown"]
    bounds["max_duration_minutes"] = [30.0, 90.0]
    bounds["weights_minutes"] = {modifier: [0.0, 20.0]}
    assert cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] is None
    cfg = prepare_config(cfg, verbose=False)
    validate_optimizer_effective_configs(cfg)
    del cfg["optimize"]["bounds"][side]["entry_cooldown"]["max_duration_minutes"]
    with pytest.raises(ValueError, match="finite when searching modifier weights"):
        prepare_config(cfg, verbose=False)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
@pytest.mark.parametrize("positive_bound", [False, True])
def test_aggregated_candle_config_rejects_reachable_rms_before_optimization(side, consumer, positive_bound):
    cfg = get_template_config()
    cfg["backtest"]["candle_interval_minutes"] = 5
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 90.0
    bounds = cfg["optimize"]["bounds"][side]
    bounds["risk"].update(n_positions=[1, 1], total_wallet_exposure_limit=[1.0, 1.0])
    weight_bound = [0.0, 1.0 if positive_bound else 0.0]
    if consumer == "forager":
        bounds["forager"]["score_weights"] = {"unilateralness": weight_bound}
    else:
        bounds["entry_cooldown"]["weights_minutes"] = {"adverse_directionality": weight_bound}
    cfg = prepare_config(cfg, verbose=False)
    from optimization.warmup import validate_optimizer_dataset_intervals
    from test_backtest_directional_eligibility import _synthetic_inputs
    _, markets, _, _ = _synthetic_inputs()
    cfg["live"]["approved_coins"] = {"long": ["LONGCOIN"], "short": ["SHORTCOIN"]}
    cfg["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    if positive_bound:
        with pytest.raises(ValueError, match="RMS unilateralness requires.*one-minute"):
            validate_optimizer_dataset_intervals(cfg, markets, "binance")
        cfg["live"]["approved_coins"][side] = ["UNLOADEDCOIN"]
    validate_optimizer_dataset_intervals(cfg, markets, "binance")


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "bounds,patch,valid",
    [
        ({"min_duration_minutes": [0, 20]}, {"max_duration_minutes": 10}, False),
        ({"max_duration_minutes": [10, 60]}, {"min_duration_minutes": 20}, False),
        ({"min_duration_minutes": [0, 20]}, {"max_duration_minutes": 20}, True),
        # Coin pins supersede global genes, including the searched leaf itself.
        ({"min_duration_minutes": [0, 20]}, {"min_duration_minutes": 5, "max_duration_minutes": 10}, True),
        ({"max_duration_minutes": [10, 60]}, {"min_duration_minutes": 20, "max_duration_minutes": 30}, True),
        ({"min_duration_minutes": [0, 20]}, {"max_duration_minutes": None}, True),
    ],
)
@pytest.mark.parametrize("origin", ["inline", "file"])
def test_cooldown_search_corners_respect_effective_coin_pins(side, bounds, patch, valid, origin):
    cfg = get_template_config()
    cfg["bot"][side]["entry_cooldown"].update(max_duration_minutes=60.0)
    cfg["optimize"]["bounds"][side]["entry_cooldown"].update(bounds)
    coin_patch = {"bot": {side: {"entry_cooldown": patch}}}
    cfg["coin_overrides"] = {"BTC": coin_patch if origin == "inline" else {}}

    def prepared():
        return parse_overrides(
            prepare_config(cfg, verbose=False), verbose=False,
            override_loader=lambda config, coin: coin_patch if origin == "file" else {},
        )

    if valid:
        result = prepared()
        assert result["optimize"]["bounds"][side]["entry_cooldown"] == cfg["optimize"]["bounds"][side]["entry_cooldown"]
        validate_optimize_bounds_against_bot_config(result, result["optimize"]["bounds"])
    else:
        with pytest.raises(ValueError, match="coin_overrides.BTC.*highest min_duration_minutes"):
            prepared()


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("gate", ["n_positions", "total_wallet_exposure_limit"])
@pytest.mark.parametrize("consumer", ["forager", "adverse_cooldown"])
@pytest.mark.parametrize("reactivates", [False, True])
def test_aggregated_rms_ignores_only_statically_disabled_sides(side, gate, consumer, reactivates):
    cfg = get_template_config()
    cfg["backtest"]["candle_interval_minutes"] = 5
    cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1.0)
    cfg["bot"][side]["risk"][gate] = 0
    cfg["bot"][side]["risk"]["total_wallet_exposure_limit"] = 0.0
    cfg["optimize"]["bounds"][side]["risk"].update(
        n_positions=[1, 1], total_wallet_exposure_limit=[1.0, 1.0]
    )
    cfg["optimize"]["bounds"][side]["risk"][gate] = [0, 1 if reactivates else 0]
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    if consumer == "forager":
        cfg["bot"][side]["forager"]["score_weights"]["unilateralness"] = 1.0
    else:
        cfg["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 10.0
    prepared = prepare_config(cfg, verbose=False)
    from optimization.warmup import validate_optimizer_dataset_intervals
    from test_backtest_directional_eligibility import _synthetic_inputs
    _, markets, _, _ = _synthetic_inputs()
    prepared["live"]["approved_coins"] = {"long": ["LONGCOIN"], "short": ["SHORTCOIN"]}
    prepared["backtest"]["coins"] = {"binance": ["LONGCOIN", "SHORTCOIN"]}
    if reactivates:
        with pytest.raises(ValueError, match="RMS unilateralness requires.*one-minute"):
            validate_optimizer_dataset_intervals(prepared, markets, "binance")
    else:
        validate_optimizer_dataset_intervals(prepared, markets, "binance")
        from warmup_utils import compute_backtest_warmup_minutes
        reference = deepcopy(prepared)
        reference["bot"][side]["forager"]["score_weights"]["unilateralness"] = 0.0
        reference["bot"][side]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 0.0
        for activation in (False, True):
            assert compute_backtest_warmup_minutes(prepared, for_trade_activation=activation) == compute_backtest_warmup_minutes(reference, for_trade_activation=activation)


@pytest.mark.parametrize("side", ["long", "short"])
def test_mixed_optimizer_bounds_preserve_all_adaptive_dimensions(side):
    from optimization.config_adapter import get_optimization_key_paths
    cfg = get_template_config()
    bounds = cfg["optimize"]["bounds"]
    bounds[f"{side}_n_positions"] = [2, 3]
    del bounds[side]["risk"]["n_positions"]
    bounds[side]["entry_cooldown"].update(
        min_duration_minutes=[0, 5], max_duration_minutes=[30, 90],
        weights_minutes={"exposure_ratio": [0, 10], "adverse_directionality": [0, 20]},
    )
    bounds[side]["forager"].update(
        unilateralness_ema_span_1m=[10.5, 60.5], score_weights={"unilateralness": [0, 1]},
    )
    expected = flatten_optimize_bounds(bounds, strategy_kind="trailing_martingale")
    prepared = prepare_config(cfg, verbose=False)
    for candidate in (prepared, prepare_config(clean_config(prepared), verbose=False)):
        actual = flatten_optimize_bounds(candidate["optimize"]["bounds"], strategy_kind="trailing_martingale")
        assert actual == expected
        paths = dict(get_optimization_key_paths(candidate))
        for leaf in ("min_duration_minutes", "max_duration_minutes"):
            assert paths[f"{side}_entry_cooldown_{leaf}"] == ("bot", side, "entry_cooldown", leaf)
        for modifier in ("exposure_ratio", "adverse_directionality"):
            assert paths[f"{side}_entry_cooldown_weights_minutes_{modifier}"] == ("bot", side, "entry_cooldown", "weights_minutes", modifier)
        assert f"{side}_unilateralness_ema_span_1m" in paths
        assert f"{side}_forager_score_weights_unilateralness" in paths


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("modifier", ["exposure_ratio", "adverse_directionality"])
@pytest.mark.parametrize("origin", ["inline", "file"])
def test_coin_zero_modifier_pin_can_clear_searched_ceiling(side, modifier, origin):
    cfg = get_template_config()
    cfg["optimize"]["bounds"][side]["entry_cooldown"].update(
        max_duration_minutes=[30, 90], weights_minutes={modifier: [0, 20]},
    )
    patch = {"bot": {side: {"entry_cooldown": {
        "max_duration_minutes": None, "weights_minutes": {modifier: 0.0},
    }}}}
    cfg["coin_overrides"] = {"BTC": patch if origin == "inline" else {}}
    result = parse_overrides(prepare_config(cfg, verbose=False), verbose=False,
        override_loader=lambda config, coin: patch if origin == "file" else {})
    assert result["coin_overrides"]["BTC"] == patch
    assert result["optimize"]["bounds"][side]["entry_cooldown"] == cfg["optimize"]["bounds"][side]["entry_cooldown"]
    # A positive effective coin modifier still requires a ceiling.
    patch["bot"][side]["entry_cooldown"]["weights_minutes"][modifier] = 1.0
    with pytest.raises(ValueError, match="finite"):
        parse_overrides(prepare_config(cfg, verbose=False), verbose=False,
            override_loader=lambda config, coin: patch if origin == "file" else {})


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("parser_ceiling", [None, 60.0])
@pytest.mark.parametrize("spelling", ["dotted", "underscored"])
@pytest.mark.parametrize("argument,expected", [(None, 60.0), ("null", None), ("none", None), ("90.5", 90.5)])
def test_nullable_cooldown_ceiling_cli_roundtrip(side, parser_ceiling, spelling, argument, expected):
    import argparse
    from config_utils import add_config_arguments, update_config_with_args
    template = get_template_config()
    template["bot"][side]["entry_cooldown"]["max_duration_minutes"] = parser_ceiling
    parser = argparse.ArgumentParser()
    keys = add_config_arguments(parser, template)
    key = f"bot.{side}.entry_cooldown.max_duration_minutes"
    option = key if spelling == "dotted" else key.replace(".", "_")
    args = parser.parse_args([] if argument is None else [f"--{option}", argument])
    if argument is None:
        assert key not in vars(args)
    cfg = get_template_config()
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    update_config_with_args(cfg, args, allowed_keys=keys)
    result = prepare_config(cfg, verbose=False)
    assert result["bot"][side]["entry_cooldown"]["max_duration_minutes"] == expected


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("eligibility", ["eligible", "excluded", "zero_wel", "zero_weight"])
def test_optimizer_data_registration_checks_final_coin_rms_policy(side, eligibility):
    from optimize import _register_exchange_data
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs
    cfg = _ema_anchor_config(True)
    cfg["backtest"]["candle_interval_minutes"] = 5
    cfg["optimize"]["enable_overrides"] = []
    cfg["bot"][side]["entry_cooldown"]["max_duration_minutes"] = 60.0
    cfg["optimize"]["bounds"] = {
        f"{side}_n_positions": [1, 1],
        f"{side}_total_wallet_exposure_limit": [1, 1],
        f"{side}_entry_cooldown_weights_minutes_adverse_directionality": [0, 10],
    }
    if eligibility == "excluded":
        cfg["live"]["approved_coins"][side] = ["UNLOADEDCOIN"]
    elif eligibility in ("zero_wel", "zero_weight"):
        coin = "LONGCOIN" if side == "long" else "SHORTCOIN"
        patch = ({"wallet_exposure_limit": 0.0} if eligibility == "zero_wel" else
                 {"entry_cooldown": {"weights_minutes": {"adverse_directionality": 0.0}}})
        cfg["coin_overrides"] = {coin: {"bot": {side: patch}}}
    cfg = parse_overrides(prepare_config(cfg, verbose=False), verbose=False)
    cfg["backtest"]["coins"] = {}
    hlcvs, mss, btc, timestamps = _synthetic_inputs()

    class Arrays:
        def __init__(self):
            self.count = 0

        def create_from(self, array):
            self.count += 1
            return object(), array

    arrays = Arrays()

    def register():
        return _register_exchange_data(
            "binance", (["LONGCOIN", "SHORTCOIN"], hlcvs, mss, None, None, btc, timestamps),
            cfg, msss={}, hlcvs_specs={}, btc_usd_specs={}, timestamps_dict={}, array_manager=arrays,
        )

    if eligibility == "eligible":
        with pytest.raises(ValueError, match="RMS unilateralness requires.*one-minute"):
            register()
        assert arrays.count == 0
    else:
        register()
        assert arrays.count == 2
