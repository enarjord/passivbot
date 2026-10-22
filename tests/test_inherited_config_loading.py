"""Actual authored/loader boundaries for common policy and sparse side pins."""
from copy import deepcopy

import pytest

from config.authoring import compact_config, resolved_config
from config.bot_inheritance import (
    PreparedBotPolicy, apply_bot_config_patch, apply_bot_policy_patch,
    bot_policy_source_view, get_bot_policy, install_bot_policy,
)
from config.load import prepare_config
from config.schema import get_template_config, get_expanded_config_prototype
from config_utils import clean_config


def source(kind="trailing_martingale"):
    return {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"enabled": True, "universe": {"approved_coins": ["BTC", "ETH"],
                                                    "minimum_coin_age_days": 0.0},
                   "risk": {"total_wallet_exposure_limit": 2.25}},
        "short": {"enabled": False}}, "live": {}}


def test_lean_template_has_common_profile_and_no_side_permission_pins():
    raw = get_template_config()
    assert raw["bot"]["long"] == raw["bot"]["short"] == {}
    assert raw["bot"]["global"]["enabled"] is False
    assert raw["optimize"]["bounds"] == {}
    raw["bot"]["global"]["enabled"] = True
    prepared = prepare_config(raw, verbose=False)
    assert prepared["bot"]["long"]["enabled"] is True
    assert prepared["bot"]["short"]["enabled"] is True
    assert prepared["bot"]["long"]["risk"] == prepared["bot"]["short"]["risk"]
    assert get_expanded_config_prototype()["bot"]["global"]["enabled"] is False


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("export", [clean_config, compact_config, resolved_config])
def test_source_roundtrip_keeps_equal_pin_and_common_edit_propagation(kind, export):
    raw = source(kind)
    raw["bot"]["short"]["risk"] = {"total_wallet_exposure_limit": 2.25}
    before = deepcopy(raw)
    prepared = prepare_config(raw, verbose=False)
    assert type(prepared["bot"]) is PreparedBotPolicy
    result = export(prepared)
    assert result["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 2.25
    assert "risk" not in result["bot"].get("long", {})
    result["bot"]["global"]["risk"]["total_wallet_exposure_limit"] = 3.5
    loaded = prepare_config(result, verbose=False)
    assert loaded["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 3.5
    assert loaded["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 2.25
    assert raw == before


def test_zero_false_empty_and_ignored_null_are_explicit_pins():
    raw = source()
    raw["bot"]["global"]["universe"]["ignored_coins"] = ["ETH"]
    raw["bot"]["short"].update({"risk": {"total_wallet_exposure_limit": 0.0},
                                "universe": {"approved_coins": [], "ignored_coins": None}})
    prepared = prepare_config(raw, verbose=False)
    assert prepared["bot"]["short"]["enabled"] is False
    assert prepared["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert prepared["bot"]["short"]["universe"]["approved_coins"] == []
    assert prepared["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]
    assert bot_policy_source_view(prepared)["bot"]["short"]["universe"]["ignored_coins"] is None
    raw["bot"]["global"]["universe"]["ignored_coins"] = None
    assert prepare_config(raw, verbose=False)["bot"]["long"]["universe"]["ignored_coins"] == []
    raw["bot"]["global"]["universe"]["approved_coins"] = None
    with pytest.raises(TypeError, match="null"):
        prepare_config(raw, verbose=False)


def test_prepared_global_ignored_file_and_approval_survive_deleted_sources(tmp_path):
    approved, ignored = tmp_path / "approved.json", tmp_path / "ignored.json"
    approved.write_text('["BTC", "ETH"]')
    ignored.write_text('["ETH"]')
    raw = source()
    raw["bot"]["global"]["universe"].update(approved_coins=str(approved), ignored_coins=str(ignored))
    prepared = prepare_config(raw, verbose=False)
    approved.unlink(); ignored.unlink()
    prepared["bot"]["global"]["risk"]["total_wallet_exposure_limit"] = 3.0
    loaded = prepare_config(prepared, verbose=False)
    assert loaded["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]
    assert loaded["bot"]["short"]["universe"]["ignored_coins"] == ["ETH"]
    assert loaded["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 3.0


def test_normalization_overlay_is_not_source_or_candidate_pin_and_clears_on_gene_write():
    raw = source()
    weights = {"volume": 2.0, "volatility": 1.0, "ema_readiness": 1.0, "unilateralness": 0.0}
    raw["bot"]["global"]["forager"] = {"score_weights": weights}
    raw["bot"]["short"]["forager"] = {"score_weights": deepcopy(weights)}
    prepared = prepare_config(raw, verbose=False)
    assert prepared["bot"]["long"]["forager"]["score_weights"]["volume"] == 0.5
    assert bot_policy_source_view(prepared)["bot"] == raw["bot"]
    origins = {origin.target: origin for origin in get_bot_policy(prepared).resolve()[1]}
    assert origins[("bot", "long", "forager", "score_weights", "volume")].source == (
        "bot", "global", "forager", "score_weights", "volume")
    assert origins[("bot", "short", "forager", "score_weights", "volume")].source == (
        "bot", "short", "forager", "score_weights", "volume")
    assert "forager" not in get_bot_policy(prepared).candidate_view().get("long", {})
    assert prepare_config(prepared, verbose=False)["bot"]["long"]["forager"]["score_weights"]["volume"] == 0.5
    policy = apply_bot_policy_patch(get_bot_policy(prepared),
        {"bot.global.forager.score_weights.volume": 6.0}, channel="gene")
    assert policy._normalized == ()
    install_bot_policy(prepared, policy)
    assert prepared["bot"]["long"]["forager"]["score_weights"]["volume"] == 6.0
    assert prepare_config(prepared, verbose=False)["bot"]["long"]["forager"]["score_weights"]["volume"] == 0.75
    assert prepare_config(prepared, verbose=False)["bot"]["short"]["forager"]["score_weights"]["volume"] == 0.5


def test_parent_patch_keeps_equal_pin_but_incomplete_atomic_weights_still_fail():
    prepared = prepare_config(source(), verbose=False)
    apply_bot_config_patch(prepared, {"bot.short.risk": {"total_wallet_exposure_limit": 2.25}})
    apply_bot_config_patch(prepared, {"bot.global.risk.total_wallet_exposure_limit": 4.0})
    prepared = prepare_config(prepared, verbose=False)
    assert prepared["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 4.0
    assert prepared["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 2.25
    with pytest.raises(ValueError, match="missing required keys"):
        apply_bot_config_patch(prepared, {"bot.short.forager.score_weights": {"volume": 1.0}})


def test_owned_projection_keeps_full_policy_and_source_presence_guard():
    from optimization.native_planning import _compile_projection_guard
    prepared = prepare_config(source(), verbose=False)
    guard = _compile_projection_guard(prepared, set())
    assert guard is not None and guard(deepcopy(prepared))
    changed = deepcopy(prepared)
    apply_bot_config_patch(changed, {"bot.short.risk.total_wallet_exposure_limit": 2.25})
    assert not guard(changed)  # Equal effective values still change source authority.
    changed = deepcopy(prepared)
    changed["bot"]["long"]["universe"]["approved_coins"].source_present = True
    assert not guard(changed)
    class Unowned(type(prepared["bot"])):
        pass
    changed = deepcopy(prepared)
    changed["bot"] = Unowned(changed["bot"], get_bot_policy(changed))
    assert _compile_projection_guard(changed, set()) is None
