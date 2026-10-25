"""Post-transform patches retain logical source and formatter authority."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import (
    apply_bot_policy_patch, bot_policy_source_view, get_bot_policy, install_bot_policy,
    reconciled_bot_policy,
)
from config.bot import normalize_bot_forager_config, normalize_forager_score_weights
from config.inherited_bounds import get_bound_plan, prepare_bound_plan
from config.schema import get_template_config
from optimization.warmup import _apply_config_overrides, _finalize_optimizer_vector_config
from optimizer_overrides import apply_coupled_unstuck_ema_spans


@pytest.mark.parametrize("derived", [False, True])
def test_forager_formatter_repeats_from_raw_input_without_changing_source_or_origins(derived):
    raw = get_template_config()
    weights = {"volume": 0.2, "log_range": 0.3, "ema_readiness": 0.4, "unilateralness": 0.1}
    raw["bot"]["global"]["forager"]["score_weights"] = deepcopy(weights)
    raw["bot"]["long"] = {"forager": {"score_weights": deepcopy(weights)}}
    config = prepare_config(raw, verbose=False)
    source = bot_policy_source_view(config)
    short_input = {"volume": 0.9, "log_range": 0.2, "ema_readiness": 0.3, "unilateralness": 0.0}
    if derived:
        install_bot_policy(config, apply_bot_policy_patch(get_bot_policy(config), {
            f"bot.short.forager.score_weights.{name}": value for name, value in short_input.items()
        }, channel="derived"))
    expected = {"long": normalize_forager_score_weights(weights, path="long"),
                "short": normalize_forager_score_weights(short_input if derived else weights, path="short")}
    for _ in range(3):
        normalize_bot_forager_config(config, verbose=False)
        policy = get_bot_policy(config)
        assert reconciled_bot_policy(config) == policy
        assert bot_policy_source_view(config) == source
        for side in ("long", "short"):
            assert config["bot"][side]["forager"]["score_weights"] == expected[side]
            assert policy.effective_view()[side]["forager"]["score_weights"] == expected[side]
        origins = {origin.target: origin for origin in policy.resolve()[1]}
        assert origins[("bot", "long", "forager", "score_weights", "volume")].kind == "side"
        assert origins[("bot", "short", "forager", "score_weights", "volume")].kind == (
            "derived" if derived else "global")
    if derived:
        updated = apply_bot_policy_patch(policy, {"bot.short.forager.score_weights.volume": 0.45}, channel="derived")
        assert updated.effective_view()["short"]["forager"]["score_weights"]["volume"] == 0.45


@pytest.mark.parametrize("scope", ["global", "long"])
def test_normalized_only_policy_keeps_untouched_weights_and_recomputes_changed_group(scope):
    raw = get_template_config()
    weights = {"volume": 2.0, "log_range": 3.0, "ema_readiness": 5.0, "unilateralness": 0.0}
    raw["bot"]["global"]["forager"]["score_weights"] = deepcopy(weights)
    raw["bot"]["long"] = {"forager": {"score_weights": deepcopy(weights)}}
    config = prepare_config(raw, verbose=False)
    assert get_bot_policy(config)._derived == ()
    normalized = deepcopy(config["bot"]["long"]["forager"]["score_weights"])
    _apply_config_overrides(config, {"bot.long.unstuck.close_fraction": 0.2})
    assert get_bot_policy(config)._derived == ()
    for side in ("long", "short"):
        assert config["bot"][side]["forager"]["score_weights"] == normalized
    source = bot_policy_source_view(config)
    assert source["bot"]["global"]["forager"]["score_weights"] == weights
    assert source["bot"]["long"]["forager"]["score_weights"] == weights
    _apply_config_overrides(config, {f"bot.{scope}.forager.score_weights.volume": 9.0})
    updated = {"volume": 9.0 / 17, "log_range": 3.0 / 17,
               "ema_readiness": 5.0 / 17, "unilateralness": 0.0}
    assert config["bot"]["long"]["forager"]["score_weights"] == (
        updated if scope == "long" else normalized)
    assert config["bot"]["short"]["forager"]["score_weights"] == (
        updated if scope == "global" else normalized)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_scenario_patch_preserves_derived_spans_and_unrelated_normalized_pins(kind):
    raw = get_template_config()
    raw["bot"]["portfolio"]["strategy_kind"] = kind
    weights = {"volume": 2.0, "log_range": 3.0, "ema_readiness": 5.0, "unilateralness": 0.0}
    raw["bot"]["global"]["forager"]["score_weights"] = deepcopy(weights)
    raw["bot"]["long"] = {"forager": {"score_weights": deepcopy(weights)}}
    spans = {"ema_span_0_minutes": [2.0, 20.0]}
    raw["optimize"]["bounds"] = {"global": {"strategy": {
        kind: {"entry": spans} if kind == "trailing_martingale" else spans,
    }}}
    raw["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    base = prepare_config(raw, verbose=False)
    original = bot_policy_source_view(base)
    plan = get_bound_plan(base)
    entry = "entry." if kind == "trailing_martingale" else ""
    coordinate = f"bot.global.strategy.{kind}.{entry}ema_span_0_minutes"
    identities = []
    for value in (2.0, 20.0, 2.0):
        candidate = deepcopy(base)
        install_bot_policy(candidate, plan.apply_values(get_bot_policy(candidate), {coordinate: value}))
        candidate = _finalize_optimizer_vector_config(candidate)
        normalized = deepcopy(candidate["bot"]["long"]["forager"]["score_weights"])
        _apply_config_overrides(candidate, {
            f"bot.long.strategy.{kind}.{entry}ema_span_1_minutes": 3.5,
        })
        apply_coupled_unstuck_ema_spans(candidate)
        identities.append(prepare_bound_plan(candidate).identity())
        policy = get_bot_policy(candidate)
        source = bot_policy_source_view(candidate)["bot"]
        assert "unstuck" not in source["long"]
        assert source["global"]["forager"]["score_weights"] == weights
        assert source["long"]["forager"]["score_weights"] == weights
        assert candidate["bot"]["long"]["forager"]["score_weights"] == normalized
        assert candidate["bot"]["short"]["forager"]["score_weights"] == normalized
        assert candidate["bot"]["long"]["unstuck"]["ema_span_0_minutes"] == value
        assert candidate["bot"]["long"]["unstuck"]["ema_span_1_minutes"] == 3.5
        origins = {origin.target: origin for origin in policy.resolve()[1]}
        assert origins[("bot", "long", "unstuck", "ema_span_0_minutes")].kind == "derived"
        assert origins[("bot", "long", "forager", "score_weights", "volume")].source == (
            "bot", "long", "forager", "score_weights", "volume",
        )
        assert not any(entry.path == ("bot", "long", "unstuck", "ema_span_0_minutes")
                       for entry in policy._normalized)
        updated = {"volume": 9.0 / 17, "log_range": 3.0 / 17,
                   "ema_readiness": 5.0 / 17, "unilateralness": 0.0}
        for scope in ("global", "long"):
            changed = deepcopy(candidate)
            _apply_config_overrides(changed, {f"bot.{scope}.forager.score_weights.volume": 9.0})
            assert changed["bot"]["long"]["forager"]["score_weights"] == (
                updated if scope == "long" else normalized)
            assert changed["bot"]["short"]["forager"]["score_weights"] == (
                updated if scope == "global" else normalized)
            assert bot_policy_source_view(changed)["bot"][scope]["forager"]["score_weights"]["volume"] == 9.0
        _apply_config_overrides(candidate, {"bot.long.unstuck.ema_span_0_minutes": 9.0})
        assert candidate["bot"]["long"]["unstuck"]["ema_span_0_minutes"] == 9.0
        assert bot_policy_source_view(candidate)["bot"]["long"]["unstuck"]["ema_span_0_minutes"] == 9.0
    assert len(set(identities)) == 1
    assert bot_policy_source_view(base) == original
    assert get_bound_plan(base) is plan
