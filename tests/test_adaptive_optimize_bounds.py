from copy import deepcopy

import pytest

from config import get_template_config, prepare_config
from config.optimize_bounds import flatten_optimize_bounds
from config_utils import clean_config, dump_config, load_config
from optimization.config_adapter import (
    extract_bounds_tuple_list_from_config,
    get_optimization_key_paths,
)
from optimization.warmup import (
    build_optimizer_vector_config,
    compute_optimizer_per_coin_warmup_minutes,
)


def omit_unilateralness_bounds(config):
    for side in ("long", "short"):
        bounds = config["optimize"]["bounds"][side]["forager"]
        bounds.pop("score_weights_unilateralness", None)
        bounds.pop("unilateralness_ema_span_1m", None)


def omit_cooldown_bounds(config):
    for side in ("long", "short"):
        bounds = config["optimize"]["bounds"][side]["entry_cooldown"]
        bounds.pop("weights_minutes", None)
        bounds.pop("min_duration_minutes", None)
        bounds.pop("max_duration_minutes", None)


def flat_bounds(config):
    return flatten_optimize_bounds(
        config["optimize"]["bounds"], strategy_kind=config["live"]["strategy_kind"]
    )


def test_parsed_template_bounds_freeze_adaptive_defaults():
    cfg = prepare_config(get_template_config(), verbose=False)
    bounds = flat_bounds(cfg)
    for side in ("long", "short"):
        forager = cfg["bot"][side]["forager"]
        weight = forager["score_weights"]["unilateralness"]
        span = forager["unilateralness_ema_span_1m"]
        assert bounds[f"{side}_forager_score_weights_unilateralness"] == [weight, weight]
        assert bounds[f"{side}_unilateralness_ema_span_1m"] == [span, span]
        cooldown = cfg["bot"][side]["entry_cooldown"]
        for key, value in cooldown["weights_minutes"].items():
            assert bounds[f"{side}_entry_cooldown_weights_minutes_{key}"] == [value, value]
        minimum = cooldown["min_duration_minutes"]
        assert bounds[f"{side}_entry_cooldown_min_duration_minutes"] == [minimum, minimum]
        assert f"{side}_entry_cooldown_max_duration_minutes" not in bounds


@pytest.mark.parametrize("side", ["long", "short"])
def test_template_edits_derive_adaptive_bounds_from_configured_values(side):
    source = get_template_config()
    original = flat_bounds(source)
    for key in (
        "forager_score_weights_unilateralness", "unilateralness_ema_span_1m",
        "entry_cooldown_weights_minutes_exposure_ratio",
        "entry_cooldown_weights_minutes_adverse_directionality",
        "entry_cooldown_min_duration_minutes", "entry_cooldown_max_duration_minutes",
    ):
        assert f"{side}_{key}" not in original
    source["bot"][side]["forager"].update(
        score_weights={"volume": 1.0, "ema_readiness": 0.0,
                       "volatility": 0.0, "unilateralness": 1.0},
        unilateralness_ema_span_1m=42.5,
    )
    source["bot"][side]["entry_cooldown"].update(
        min_duration_minutes=20.0, max_duration_minutes=90.0,
        weights_minutes={"exposure_ratio": 3.25, "adverse_directionality": 2.5},
    )
    expected = {
        "forager_score_weights_unilateralness": 0.5,
        "unilateralness_ema_span_1m": 42.5,
        "entry_cooldown_weights_minutes_exposure_ratio": 3.25,
        "entry_cooldown_weights_minutes_adverse_directionality": 2.5,
        "entry_cooldown_min_duration_minutes": 20.0,
        "entry_cooldown_max_duration_minutes": 90.0,
    }
    parsed = prepare_config(source, verbose=False)
    for cfg in (parsed, clean_config(source)):
        bounds = flat_bounds(cfg)
        for key, value in expected.items():
            assert bounds[f"{side}_{key}"] == [value, value]
    bounds = extract_bounds_tuple_list_from_config(parsed)
    vector = [bound.low for bound in bounds]
    for index, (_, path) in enumerate(get_optimization_key_paths(parsed)):
        if path[:4] == ("bot", side, "forager", "score_weights"):
            vector[index] = parsed["bot"][side]["forager"]["score_weights"][path[-1]]
    candidate = build_optimizer_vector_config(vector, parsed)
    assert candidate["bot"][side]["forager"]["score_weights"]["unilateralness"] == 0.5
    assert candidate["bot"][side]["forager"]["unilateralness_ema_span_1m"] == 42.5
    for key in ("weights_minutes", "min_duration_minutes", "max_duration_minutes"):
        assert candidate["bot"][side]["entry_cooldown"][key] == parsed["bot"][side]["entry_cooldown"][key]


@pytest.mark.parametrize("missing,wrapped", [
    (missing, wrapped)
    for missing in ("leaves", "forager", "bounds", "optimize")
    for wrapped in (False, True)
    if not (wrapped and missing == "optimize")
])
@pytest.mark.parametrize("live_only", [False, True])
def test_missing_bounds_freeze_normalized_side_values(missing, wrapped, live_only, tmp_path):
    source = get_template_config()
    for side, span, weight in (("long", 12.5, 2.0), ("short", 81.25, 3.0)):
        source["bot"][side]["forager"].update(
            unilateralness_ema_span_1m=span,
            score_weights={"volume": 1.0, "ema_readiness": 1.0,
                           "volatility": 0.0, "unilateralness": weight},
        )
    omit_unilateralness_bounds(source)
    if missing == "forager":
        for side in ("long", "short"):
            source["optimize"]["bounds"][side].pop("forager")
    elif missing == "bounds":
        source["optimize"].pop("bounds")
    elif missing == "optimize":
        source.pop("optimize")
    cfg = prepare_config({"config": source} if wrapped else source,
                         live_only=live_only, verbose=False)
    for candidate in (cfg, clean_config(cfg)):
        bounds = flat_bounds(candidate)
        for side in ("long", "short"):
            forager = cfg["bot"][side]["forager"]
            weight = forager["score_weights"]["unilateralness"]
            span = forager["unilateralness_ema_span_1m"]
            assert bounds[f"{side}_forager_score_weights_unilateralness"] == [weight, weight]
            assert bounds[f"{side}_unilateralness_ema_span_1m"] == [span, span]
    path = str(tmp_path / "config.json")
    dump_config(cfg, path, clean=True)
    assert flat_bounds(load_config(path, live_only=live_only, verbose=False)) == flat_bounds(cfg)


@pytest.mark.parametrize("form", ["nested", "grouped", "flat"])
def test_explicit_bounds_survive_hydration_cleaning_and_loading(form, tmp_path):
    source = get_template_config()
    omit_unilateralness_bounds(source)
    for side in ("long", "short"):
        forager = source["optimize"]["bounds"][side]["forager"]
        if form == "nested":
            forager["score_weights"] = {"unilateralness": [0.0, 1.0, 0.01]}
            forager["unilateralness_ema_span_1m"] = [20.5, 240.5, 0.5]
        elif form == "grouped":
            forager["score_weights_unilateralness"] = [0.0, 1.0, 0.01]
            forager["unilateralness_ema_span_1m"] = [20.5, 240.5, 0.5]
        else:
            source["optimize"]["bounds"][f"{side}_forager_score_weights_unilateralness"] = [0.0, 1.0, 0.01]
            source["optimize"]["bounds"][f"{side}_unilateralness_ema_span_1m"] = [20.5, 240.5, 0.5]
    cfg = prepare_config(source, verbose=False)
    path = str(tmp_path / "config.json")
    dump_config(cfg, path, clean=True)
    for candidate in (cfg, clean_config(cfg), load_config(path, verbose=False)):
        for side in ("long", "short"):
            assert flat_bounds(candidate)[f"{side}_forager_score_weights_unilateralness"] == [0.0, 1.0, 0.01]
            assert flat_bounds(candidate)[f"{side}_unilateralness_ema_span_1m"] == [20.5, 240.5, 0.5]


def test_direct_cleaning_freezes_effective_weight_without_mutating_source():
    source = get_template_config()
    omit_unilateralness_bounds(source)
    source["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 1.0, "ema_readiness": 1.0, "volatility": 0.0, "unilateralness": 2.0,
    }
    source["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 42.5
    snapshot = deepcopy(source)
    cleaned = clean_config(source)
    assert source == snapshot
    assert flat_bounds(cleaned)["long_forager_score_weights_unilateralness"] == [0.5, 0.5]
    assert flat_bounds(cleaned)["long_unilateralness_ema_span_1m"] == [42.5, 42.5]
    assert flat_bounds(prepare_config(cleaned, verbose=False)) == flat_bounds(cleaned)


@pytest.mark.parametrize("weight", [0.0, 0.4])
@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_frozen_bounds_preserve_candidates_search_ranges_and_warmup(weight, fraction):
    from optimize import _canonicalize_optimizer_individual

    source = get_template_config()
    omit_unilateralness_bounds(source)
    omit_cooldown_bounds(source)
    source["bot"]["long"]["forager"]["score_weights"]["unilateralness"] = weight
    source["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 75.25
    source["bot"]["long"]["entry_cooldown"].update(
        min_duration_minutes=2.5, max_duration_minutes=90.25,
        weights_minutes={"exposure_ratio": 3.25, "adverse_directionality": 2.5},
    )
    hydrated = prepare_config(source, verbose=False)
    previous = deepcopy(hydrated)
    omit_unilateralness_bounds(previous)
    omit_cooldown_bounds(previous)

    def shape(cfg):
        return dict(zip((key for key, _ in get_optimization_key_paths(cfg)),
                        extract_bounds_tuple_list_from_config(cfg)))

    old_shape, new_shape = shape(previous), shape(hydrated)
    assert {key: bound for key, bound in old_shape.items() if bound.low != bound.high} == {
        key: bound for key, bound in new_shape.items() if bound.low != bound.high
    }
    candidates = []
    canonical_candidates = []
    for cfg in (previous, hydrated):
        bounds = list(shape(cfg).values())
        vector = [bound.quantize(bound.low + fraction * (bound.high - bound.low))
                  for bound in bounds]
        candidates.append(build_optimizer_vector_config(vector, cfg))
        canonical_candidates.append(_canonicalize_optimizer_individual(
            vector, cfg, bounds, cfg["optimize"]["round_to_n_significant_digits"],
            get_optimization_key_paths(cfg), cfg["optimize"]["enable_overrides"],
        ))
    assert candidates[0]["bot"] == candidates[1]["bot"]
    assert canonical_candidates[0]["bot"] == canonical_candidates[1]["bot"]
    for activation in (False, True):
        assert compute_optimizer_per_coin_warmup_minutes(previous, for_trade_activation=activation) == (
            compute_optimizer_per_coin_warmup_minutes(hydrated, for_trade_activation=activation)
        )


@pytest.mark.parametrize("ceiling", [None, 90.25])
@pytest.mark.parametrize("missing", ["leaves", "cooldown", "bounds", "optimize"])
def test_missing_cooldown_bounds_freeze_each_side_and_preserve_null_ceiling(ceiling, missing, tmp_path):
    source = get_template_config()
    omit_cooldown_bounds(source)
    for side, minimum in (("long", 2.5), ("short", 3.75)):
        source["bot"][side]["entry_cooldown"].update(
            min_duration_minutes=minimum, max_duration_minutes=ceiling,
            weights_minutes={
                "exposure_ratio": minimum if ceiling is not None else 0.0,
                "adverse_directionality": minimum / 2 if ceiling is not None else 0.0,
            },
        )
    if missing == "cooldown":
        for side in ("long", "short"):
            source["optimize"]["bounds"][side].pop("entry_cooldown")
    elif missing == "bounds":
        source["optimize"].pop("bounds")
    elif missing == "optimize":
        source.pop("optimize")
    parsed = prepare_config(source, verbose=False)
    path = str(tmp_path / "config.json")
    dump_config(parsed, path, clean=True)
    for candidate in (parsed, clean_config(parsed), clean_config(source), load_config(path, verbose=False)):
        bounds = flat_bounds(candidate)
        for side in ("long", "short"):
            cooldown = parsed["bot"][side]["entry_cooldown"]
            for key, value in cooldown["weights_minutes"].items():
                assert bounds[f"{side}_entry_cooldown_weights_minutes_{key}"] == [value, value]
            minimum = cooldown["min_duration_minutes"]
            assert bounds[f"{side}_entry_cooldown_min_duration_minutes"] == [minimum, minimum]
            key = f"{side}_entry_cooldown_max_duration_minutes"
            if ceiling is None:
                assert key not in bounds
            else:
                assert bounds[key] == [ceiling, ceiling]


@pytest.mark.parametrize("form", ["nested", "grouped", "flat"])
def test_explicit_cooldown_bounds_survive_roundtrip_with_null_bot_ceiling(form, tmp_path):
    source = get_template_config()
    omit_cooldown_bounds(source)
    expected = {
        "weights_minutes_exposure_ratio": [0.0, 10.0, 0.25],
        "weights_minutes_adverse_directionality": [0.0, 20.0, 0.5],
        "min_duration_minutes": [1.0, 10.0, 0.25],
        "max_duration_minutes": [20.0, 100.0, 0.5],
    }
    for side in ("long", "short"):
        target = source["optimize"]["bounds"][side]["entry_cooldown"]
        for key, value in expected.items():
            if form == "flat":
                source["optimize"]["bounds"][f"{side}_entry_cooldown_{key}"] = value
            elif form == "nested" and key.startswith("weights_minutes_"):
                target.setdefault("weights_minutes", {})[key.removeprefix("weights_minutes_")] = value
            else:
                target[key] = value
    cfg = prepare_config(source, verbose=False)
    path = str(tmp_path / "config.json")
    dump_config(cfg, path, clean=True)
    for candidate in (cfg, clean_config(cfg), load_config(path, verbose=False)):
        for side in ("long", "short"):
            assert candidate["bot"][side]["entry_cooldown"]["max_duration_minutes"] is None
            for key, value in expected.items():
                assert flat_bounds(candidate)[f"{side}_entry_cooldown_{key}"] == value


@pytest.mark.parametrize("ceiling", [float("inf"), float("nan")])
def test_cleaning_rejects_nonfinite_cooldown_ceiling_instead_of_hydrating_invalid_bound(ceiling):
    source = get_template_config()
    source["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = ceiling
    with pytest.raises(ValueError, match="max_duration_minutes must be finite"):
        clean_config(source)
