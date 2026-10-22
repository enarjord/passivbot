"""Prepared source membership retains declaration presence outside JSON policy."""

from copy import deepcopy
import json

import pytest

from config.universe import ResolvedCoinList


def policy_config(source):
    from config.bot_inheritance import SideFieldCatalog, install_bot_policy, resolve_bot_policy
    from config.field_spec import FieldRegistry, FieldSpec, FieldType

    defaults = {"universe": {"approved_coins": [], "ignored_coins": []},
                "risk": {"capacity": 1.0}}
    fields = [FieldSpec(("bot", side, "universe", field),
                        (FieldType.ARRAY, FieldType.STRING), nullable=True)
              for side in ("long", "short")
              for field in ("approved_coins", "ignored_coins")]
    fields.extend(FieldSpec(("bot", side, "risk", "capacity"), (FieldType.FLOAT,))
                  for side in ("long", "short"))
    catalog = SideFieldCatalog.from_registry(FieldRegistry(fields), common_defaults=defaults)
    config = {"live": {}}
    install_bot_policy(config, resolve_bot_policy(source, catalog))
    return config


@pytest.mark.parametrize("present", [False, True])
def test_prepared_source_presence_survives_copy_but_is_not_json_authority(present):
    value = ResolvedCoinList(["BTC"], None, ["BTC"], source_present=present)
    copied = deepcopy(value)
    assert copied.source_present is present
    assert copied.source is None
    assert copied.shared_source == ["BTC"]
    assert json.loads(json.dumps(value)) == ["BTC"]
    assert type(json.loads(json.dumps(value))) is list


@pytest.mark.parametrize("present", [None, 0, 1, "false", []])
def test_prepared_source_presence_requires_an_exact_internal_boolean(present):
    with pytest.raises(TypeError):
        ResolvedCoinList(["BTC"], None, ["BTC"], source_present=present)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_inherited_shared_sources_read_once_and_keep_sparse_presence(tmp_path, monkeypatch, typed, field):
    from config.universe import (materialize_side_universe_sources,
                                 resolve_universe_sources, universe_source_view,
                                 get_side_universe_sources)
    source = tmp_path / "coins.json"
    source.write_text('["BTC", "ETH"]')
    raw = {"global": {"universe": {field: str(source)}}}
    config = policy_config(raw) if typed else {"bot": deepcopy(raw)}
    calls = []
    original = type(source).open

    def opened(path, *args, **kwargs):
        if path == source:
            calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "open", opened)
    values, errors = resolve_universe_sources(config, field)
    assert not errors and values == {side: ["BTC", "ETH"] for side in ("long", "short")}
    assert calls == [source]
    materialize_side_universe_sources(config, field, values)
    source.unlink()
    assert get_side_universe_sources(config, field) == values
    for side in ("long", "short"):
        membership = config["bot"][side]["universe"][field]
        assert type(membership) is ResolvedCoinList
        assert membership.source_present is False
        assert membership.shared_source == str(source)
    exported = universe_source_view(config)["bot"]
    assert exported["global"] == raw["global"]
    assert all(field not in exported.get(side, {}).get("universe", {}) for side in ("long", "short"))
    if typed:
        assert exported == raw


@pytest.mark.parametrize("typed", [False, True])
def test_ignored_omission_inherits_but_explicit_null_clears(tmp_path, typed):
    from config.universe import resolve_universe_sources, materialize_side_universe_sources, universe_source_view
    source = tmp_path / "ignored.json"
    source.write_text('["BTC"]')
    raw = {"global": {"universe": {"ignored_coins": str(source)}},
           "short": {"universe": {"ignored_coins": None}}}
    config = policy_config(raw) if typed else {"bot": deepcopy(raw)}
    values, _ = resolve_universe_sources(config, "ignored_coins")
    assert values == {"long": ["BTC"], "short": []}
    materialize_side_universe_sources(config, "ignored_coins", values)
    assert not config["bot"]["long"]["universe"]["ignored_coins"].source_present
    assert config["bot"]["short"]["universe"]["ignored_coins"].source_present
    exported = universe_source_view(config)["bot"]
    assert exported["global"] == raw["global"]
    assert "ignored_coins" not in exported.get("long", {}).get("universe", {})
    assert exported["short"] == raw["short"]
    if typed:
        assert exported == raw


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_independent_side_sources_do_not_read_unused_shared_file(tmp_path, field):
    from config.universe import resolve_universe_sources
    config = policy_config({"global": {"universe": {field: str(tmp_path / "missing.json")}},
                            **{side: {"universe": {field: ["BTC"]}}
                               for side in ("long", "short")}})
    assert resolve_universe_sources(config, field) == ({"long": ["BTC"], "short": ["BTC"]}, {})


def test_prepared_views_retain_numeric_shape_and_source_authority_without_io(tmp_path, monkeypatch):
    from config.bot_inheritance import (apply_bot_policy_patch, get_bot_policy, install_bot_policy)
    from config.universe import (materialize_side_universe_sources, materialize_universe_snapshot,
                                 runtime_universe_snapshot, universe_source_view, fixed_runtime_universe_view)
    raw = {"global": {"universe": {"approved_coins": "./coins.json", "ignored_coins": ["ETH"]}},
           "short": {"universe": {"approved_coins": ["BTC"], "ignored_coins": None}}}
    config = policy_config(raw)
    for field, values in (("approved_coins", {"long": ["BTC"], "short": ["BTC"]}),
                          ("ignored_coins", {"long": ["ETH"], "short": []})):
        materialize_side_universe_sources(config, field, values)
    policy = apply_bot_policy_patch(get_bot_policy(config), {"bot.global.risk.capacity": 2.0})
    install_bot_policy(config, policy)
    monkeypatch.setattr(type(tmp_path), "open", lambda *_a, **_k: pytest.fail("prepared views must not read files"))
    snapshot = runtime_universe_snapshot(config)
    restored = materialize_universe_snapshot(config, snapshot)
    assert restored["bot"]["long"]["risk"]["capacity"] == 2.0
    assert universe_source_view(restored)["bot"] == {**raw, "global": {
        **raw["global"], "risk": {"capacity": 2.0}}}
    fixed = fixed_runtime_universe_view(restored)
    assert get_bot_policy(fixed) is None
    assert fixed["bot"]["long"]["universe"]["approved_coins"] == ["BTC"]
    assert fixed["bot"]["long"]["risk"]["capacity"] == 2.0


def test_source_view_adopts_changed_public_binding_without_pinning_other_side():
    from config.universe import get_side_universe_sources, universe_source_view
    config = policy_config({"global": {"universe": {"approved_coins": ["BTC"]}}})
    config["bot"]["global"]["universe"]["approved_coins"] = ["ETH"]
    assert get_side_universe_sources(config, "approved_coins") == {"long": ["ETH"], "short": ["ETH"]}
    assert universe_source_view(config)["bot"] == {"global": {"universe": {"approved_coins": ["ETH"]}}}


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_resolved_source_projection_preserves_sparse_and_equal_pin_authority(tmp_path, field):
    from config.authoring import _resolve_coin_sources
    from config.universe import universe_source_view
    common = tmp_path / "common.json"
    common.write_text('["BTC"]')
    source = {"global": {"universe": {field: str(common)}},
              "short": {"universe": {field: ["BTC"]}}}
    config = policy_config(source)
    resolved, pending = _resolve_coin_sources(config, require_complete=True)
    assert not pending
    exported = universe_source_view(resolved)["bot"]
    assert exported["global"]["universe"][field] == ["BTC"]
    assert exported["short"]["universe"][field] == ["BTC"]
    assert "long" not in exported
    assert universe_source_view(config)["bot"] == source


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_resolved_source_projection_keeps_null_and_resolves_unused_shared_file(tmp_path, field):
    from config.authoring import _resolve_coin_sources
    from config.universe import universe_source_view
    common = tmp_path / "common.json"
    common.write_text('["ETH"]')
    source = {"global": {"universe": {field: str(common)}},
              "long": {"universe": {field: None}},
              "short": {"universe": {field: []}}}
    config = policy_config(source)
    resolved, _ = _resolve_coin_sources(config, require_complete=True)
    exported = universe_source_view(resolved)["bot"]
    assert exported["global"]["universe"][field] == ["ETH"]
    assert exported["long"]["universe"][field] is None
    assert exported["short"]["universe"][field] == []
    from config.authoring import _resolved_patch_value
    path = ("bot", "long", "universe", field)
    assert _resolved_patch_value(None, path, resolved) is None
    parent = ("bot", "long", "universe")
    assert _resolved_patch_value({field: None}, parent, resolved) == {field: None}


def test_declared_coin_file_global_defaults_project_without_hydrating_or_widening():
    from config.overrides import _project_declared_file_defaults, _extract_allowed_patch
    kind = "trailing_martingale"
    raw = {"bot": {"global": {
        "enabled": True,
        "universe": {"approved_coins": ["BTC"], "ignored_coins": []},
        "risk": {"total_wallet_exposure_limit": 2.0, "max_positions": 3},
        "strategy": {kind: {"entry": {"ema_span_0_minutes": 12.5}},
                     "ema_anchor": {"fast_span_minutes": 42.0}},
        "unstuck": {"ema_span_0_minutes": 30.5},
    }, "short": {"strategy": {kind: {"entry": {"ema_span_0_minutes": 17.5}}}},
        "portfolio": {"strategy_kind": kind, "hedge_mode": False}}}
    before = deepcopy(raw)
    projected = _project_declared_file_defaults(raw, strategy_kind=kind)
    patch = _extract_allowed_patch(projected, coin="BTC", source="coin.json", strict=False,
                                   strategy_kind=kind, hsl_signal_mode="coin")
    assert raw == before
    assert set(patch["bot"]) == {"long", "short"}
    assert patch["bot"]["long"]["strategy"][kind] == {"entry": {"ema_span_0_minutes": 12.5}}
    assert patch["bot"]["short"]["strategy"][kind] == {"entry": {"ema_span_0_minutes": 17.5}}
    for side in ("long", "short"):
        assert patch["bot"][side]["unstuck"] == {"ema_span_0_minutes": 30.5}
        assert "enabled" not in patch["bot"][side]
        assert "universe" not in patch["bot"][side]
        assert "risk" not in patch["bot"][side]
        assert set(patch["bot"][side]["strategy"]) == {kind}
    with pytest.raises(ValueError, match="not overridable"):
        _extract_allowed_patch(raw, coin="BTC", source="inline", strict=True,
                               strategy_kind=kind, hsl_signal_mode="coin")


def test_inspection_reports_global_origin_without_treating_equal_side_pin_as_inherited():
    from config.authoring import _inherited_sources
    config = policy_config({"global": {"risk": {"capacity": 1.0},
                                      "universe": {"ignored_coins": ["BTC"]}},
                            "short": {"risk": {"capacity": 1.0},
                                      "universe": {"ignored_coins": None}}})
    assert _inherited_sources(config, ("bot", "long", "risk", "capacity")) == [
        (("bot", "long", "risk", "capacity"), ("bot", "global", "risk", "capacity"))]
    assert _inherited_sources(config, ("bot", "short", "risk", "capacity")) == []
    assert _inherited_sources(config, ("bot", "long", "universe", "ignored_coins")) == [
        (("bot", "long", "universe", "ignored_coins"), ("bot", "global", "universe", "ignored_coins"))]
    assert _inherited_sources(config, ("bot", "short", "universe", "ignored_coins")) == []


def test_materialization_reconciles_direct_source_edit_before_capturing_members():
    from config.universe import (get_side_universe_sources, materialize_side_universe_sources,
                                 universe_source_view)
    config = policy_config({"global": {"universe": {"approved_coins": ["BTC"]}}})
    materialize_side_universe_sources(config, "approved_coins", {"long": ["BTC"], "short": ["BTC"]})
    config["bot"]["global"]["universe"]["approved_coins"] = ["ETH"]
    config["bot"]["global"]["risk"]["capacity"] = 2.0
    materialize_side_universe_sources(config, "approved_coins", {"long": ["ETH"], "short": ["ETH"]})
    assert get_side_universe_sources(config, "approved_coins") == {"long": ["ETH"], "short": ["ETH"]}
    assert config["bot"]["short"]["risk"]["capacity"] == 2.0
    assert universe_source_view(config)["bot"] == {"global": {
        "universe": {"approved_coins": ["ETH"]}, "risk": {"capacity": 2.0}}}


def test_candidate_snapshot_is_an_explicit_projection_separate_from_authored_sources():
    from config.bot_inheritance import apply_bot_policy_patch, get_bot_policy, install_bot_policy
    from config.universe import anchored_universe_source_view, universe_source_view
    raw = {"global": {"risk": {"capacity": 1.0}}, "short": {"risk": {"capacity": 1.0}}}
    config = policy_config(raw)
    policy = apply_bot_policy_patch(get_bot_policy(config), {"bot.long.risk.capacity": 2.0}, channel="derived")
    install_bot_policy(config, policy)
    assert universe_source_view(config)["bot"] == raw
    candidate = anchored_universe_source_view(config, candidate=True)
    assert candidate["bot"]["long"]["risk"]["capacity"] == 2.0
    assert candidate["bot"]["short"]["risk"]["capacity"] == 1.0
    assert candidate["bot"]["global"]["universe"] == {"approved_coins": [], "ignored_coins": []}
    assert "long" not in universe_source_view(config, include_defaults=True)["bot"]
    assert json.loads(json.dumps(candidate)) == candidate
