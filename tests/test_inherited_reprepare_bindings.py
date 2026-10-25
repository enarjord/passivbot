"""Repreparing edited policy retains selected file bindings and frozen siblings."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import bot_policy_source_view
from config.universe import get_side_universe_sources, resolve_universe_sources


def paths(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "coins.json").write_text('["BTC"]')
    (b / "coins.json").write_text('["ETH"]')
    return a, b


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
@pytest.mark.parametrize("scope", ["global", "long"])
def test_reprepared_new_relative_source_stays_bound_after_context_change(tmp_path, field, scope):
    a, b = paths(tmp_path)
    config = prepare_config({"config_version": "v9.0.0"}, live_only=True,
                            base_config_path=str(a / "config.json"), verbose=False)
    config["bot"][scope]["universe"][field] = "coins.json"
    config = prepare_config(config, live_only=True, base_config_path=str(a / "config.json"), verbose=False)
    before = deepcopy(config)
    config["live"]["base_config_path"] = str(b / "config.json")
    values, _ = resolve_universe_sources(config, field)
    assert values["long"] == ["BTC"]
    assert bot_policy_source_view(config)["bot"][scope]["universe"][field] == str(a / "coins.json")
    assert before["bot"][scope]["universe"][field] == str(a / "coins.json")
    (a / "coins.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_universe_sources(config, field)


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_unchanged_frozen_sources_reprepare_without_reopening_deleted_files(tmp_path, field):
    a, b = paths(tmp_path)
    source = {"config_version": "v9.0.0", "bot": {"global": {"universe": {field: str(a / "coins.json")}}}}
    config = prepare_config(source, base_config_path=str(a / "config.json"), verbose=False)
    expected = deepcopy(get_side_universe_sources(config, field))
    (a / "coins.json").unlink()
    loaded = prepare_config(config, base_config_path=str(b / "config.json"), verbose=False)
    assert get_side_universe_sources(loaded, field) == expected
    assert bot_policy_source_view(loaded)["bot"]["global"]["universe"][field] == str(a / "coins.json")


def test_new_shared_binding_keeps_independently_pinned_deleted_file_membership(tmp_path):
    a, b = paths(tmp_path)
    pinned = a / "pinned.json"
    pinned.write_text('["AAA"]')
    config = prepare_config({"config_version": "v9.0.0", "bot": {
        "long": {"universe": {"approved_coins": str(pinned)}},
        "short": {"universe": {"approved_coins": []}},
    }}, base_config_path=str(a / "config.json"), verbose=False)
    pinned.unlink()
    config["bot"]["global"]["universe"]["approved_coins"] = "coins.json"
    loaded = prepare_config(config, base_config_path=str(a / "config.json"), verbose=False)
    assert get_side_universe_sources(loaded, "approved_coins") == {"long": ["AAA"], "short": []}
    assert bot_policy_source_view(loaded)["bot"]["long"]["universe"]["approved_coins"] == str(pinned)


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
@pytest.mark.parametrize("long_source", ["file", None])
def test_common_edit_invalidates_only_inherited_fact_sources(tmp_path, field, long_source):
    a, _ = paths(tmp_path)
    pinned = a / "pinned.json"
    pinned.write_text('["ETH"]')
    declared = str(pinned) if long_source == "file" else None
    source = {"config_version": "v9.0.0", "bot": {
        "global": {"universe": {field: ["BTC"]}},
        "long": {"universe": {field: declared}},
    }}
    config = prepare_config(source, base_config_path=str(a / "config.json"), verbose=False)
    pinned.unlink()
    config["bot"]["global"]["universe"][field] = ["SOL"]
    loaded = prepare_config(config, base_config_path=str(a / "config.json"), verbose=False)
    expected_long = ["ETH"] if long_source == "file" else ["SOL"] if field == "approved_coins" else []
    assert get_side_universe_sources(loaded, field) == {"long": expected_long, "short": ["SOL"]}
    assert bot_policy_source_view(loaded)["bot"]["long"]["universe"][field] == declared
    assert "risk" not in bot_policy_source_view(loaded)["bot"]["long"]


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_binding_canonicalization_preserves_numeric_origins_and_sparse_equal_pins(tmp_path, kind):
    from config.bot_inheritance import get_bot_policy
    from config.inherited_bounds import get_bound_plan
    a, _ = paths(tmp_path)
    weights = {"volume": 2.0, "volatility": 1.0, "ema_readiness": 0.0, "unilateralness": 0.0}
    raw = {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"risk": {"total_wallet_exposure_limit_ratio": 1.5},
                   "forager": {"score_weights": weights}},
        "short": {"risk": {"total_wallet_exposure_limit_ratio": 1.5}},
    }, "optimize": {"bounds": {"global": {"risk": {"total_wallet_exposure_limit_ratio": [1.0, 2.0]}}}}}
    config = prepare_config(raw, live_only=True, verbose=False)
    config["bot"]["global"]["universe"]["approved_coins"] = "coins.json"
    loaded = prepare_config(config, live_only=True, base_config_path=str(a / "config.json"), verbose=False)
    source = bot_policy_source_view(loaded)["bot"]
    assert source["global"]["forager"]["score_weights"] == weights
    assert source["short"]["risk"]["total_wallet_exposure_limit_ratio"] == 1.5
    assert "risk" not in source.get("long", {}) and "forager" not in source["short"]
    assert loaded["bot"]["long"]["forager"]["score_weights"]["volume"] == pytest.approx(2.0 / 3.0)
    policy = get_bot_policy(loaded)
    plan = get_bound_plan(loaded)
    plan.apply_values(policy, {})
    binding = next(binding for binding in plan.bindings if binding.key == "bot.global.risk.total_wallet_exposure_limit_ratio")
    assert binding.effective_targets == (("bot", "long", "risk", "total_wallet_exposure_limit_ratio"),)
