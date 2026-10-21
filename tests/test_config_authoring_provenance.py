"""Coin explain views identify the actual policy layer and exact market key."""

from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from config.authoring import explain_config
from config.schema import get_template_config
from config.universe import set_side_universe_sources


COINS = ("BTC", "BTC/USDT:USDT", "bybit::BTC/USDT:USDT", "BTC/USD:BTC")
ROOT_PATH = "bot.long.unstuck.close_fraction"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("explain must not access the network")

    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)
    monkeypatch.setattr(socket.socket, "connect", reject)


def policy():
    source = get_template_config()
    set_side_universe_sources(source, "approved_coins", {"long": ["BTC"], "short": []})
    set_side_universe_sources(source, "ignored_coins", {"long": [], "short": []})
    source["optimize"]["bounds"] = {}
    source["optimize"]["enable_overrides"] = []
    source["bot"]["long"]["unstuck"]["close_fraction"] = 0.05
    return source


def coin_patch(value):
    return {"bot": {"long": {"unstuck": {"close_fraction": value}}}}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def selected_report(source, coin, form, *, base, scenario=None, field="close_fraction"):
    path = f"bot.long.unstuck.{field}"
    if form == "direct":
        return explain_config(source, f"coin_overrides.{coin}.{path}",
                              base_config_path=base, scenario=scenario, optimize=True)
    return explain_config(source, path, coin=coin, base_config_path=base,
                          scenario=scenario, optimize=True)


def suite_policy(source, *, base, scenario=None, optimize=False):
    """Replay the suite's local policy stages without selecting a dataset."""
    from config.load import prepare_config
    from config.overrides import parse_overrides
    from config.scenario_overrides import normalize_scenario_overrides
    from optimization.warmup import _apply_config_overrides
    from optimizer_overrides import apply_coupled_unstuck_ema_spans, optimizer_overrides
    from suite_runner import apply_scenario_overrides

    context = parse_overrides(prepare_config(source, base_config_path=base, verbose=False), verbose=False)
    if optimize:
        _apply_config_overrides(context, context["optimize"]["fixed_runtime_overrides"])
        transforms = context["optimize"]["enable_overrides"]
        for side in (None, "long", "short"):
            context = optimizer_overrides(transforms, context, side)
    if scenario is not None:
        selected = next(item for item in context["backtest"]["scenarios"] if item["label"] == scenario)
        apply_scenario_overrides(context, normalize_scenario_overrides(selected.get("overrides", {})))
        if optimize:
            context = apply_coupled_unstuck_ema_spans(context)
        context = parse_overrides(context, verbose=False)
    return context


@pytest.mark.parametrize("coin", COINS)
@pytest.mark.parametrize("form", ["selected", "direct"])
@pytest.mark.parametrize("scenario_form", [None, "leaf", "coin", "map"])
def test_coin_layer_sources_match_fixed_and_scenario_authority(tmp_path, coin, form, scenario_form):
    override = write_json(tmp_path / "coin.json", coin_patch(0.11))
    source = policy()
    source["coin_overrides"] = {key: coin_patch(0.14 + index / 100)
                                for index, key in enumerate(COINS)}
    source["coin_overrides"][coin] = {"override_config_path": override.name, **coin_patch(0.12)}
    coin_path = f"coin_overrides.{coin}.{ROOT_PATH}"
    source["optimize"]["fixed_runtime_overrides"] = {ROOT_PATH: 0.06, coin_path: 0.21}
    scenario = None
    if scenario_form is not None:
        if scenario_form == "leaf":
            selector, value = coin_path, 0.31
        elif scenario_form == "coin":
            selector, value = f"coin_overrides.{coin}", coin_patch(0.31)
        else:
            selector, value = "coin_overrides", {coin: coin_patch(0.31)}
        source["backtest"]["scenarios"] = [{"label": "one", "overrides": {
            ROOT_PATH: 0.07, selector: value,
        }}]
        scenario = "one"
    original = deepcopy(source)
    file_bytes = override.read_bytes()
    report = selected_report(source, coin, form, base=str(tmp_path / "input.json"), scenario=scenario)

    expected_kind = "fixed_runtime_override" if scenario is None else "scenario_override"
    assert report["value"] == (0.21 if scenario is None else 0.31)
    assert report["sources"][-1]["kind"] == expected_kind
    assert report["sources"][-1]["path"] == (coin_path if scenario is None else selector)
    assert report["search"]["status"] == expected_kind
    assert report["search"]["gene_status"] == "omitted_fixed"
    assert {"kind": "fixed_runtime_override", "path": coin_path} in report["search"]["shadowed_by"]
    layer_events = [(event["kind"], event["path"]) for event in report["sources"]
                    if event["kind"] in {"fixed_runtime_override", "scenario_override"}]
    expected_events = [("fixed_runtime_override", coin_path)]
    if form == "selected":
        expected_events.insert(0, ("fixed_runtime_override", ROOT_PATH))
    if scenario is not None:
        if form == "selected":
            expected_events.append(("scenario_override", ROOT_PATH))
        expected_events.append(("scenario_override", selector))
    assert layer_events == expected_events
    assert not any(event["kind"] == "normalization" for event in report["sources"])
    assert not any(event["kind"] == "coin_override" and event.get("value") == report["value"]
                   for event in report["sources"])
    assert source == original
    assert override.read_bytes() == file_bytes


@pytest.mark.parametrize("coin", COINS)
@pytest.mark.parametrize("field,value", [("close_fraction", 0.11), ("ema_span_0_minutes", 17.5)])
def test_direct_coin_file_query_reports_the_same_source_as_selected_coin(tmp_path, coin, field, value):
    override = write_json(tmp_path / "coin.json", {"bot": {"long": {"unstuck": {field: value}}}})
    source = policy()
    source["coin_overrides"] = {key: coin_patch(0.14 + index / 100)
                                for index, key in enumerate(COINS)}
    source["coin_overrides"][coin] = {"override_config_path": override.name}
    original = deepcopy(source)
    original_bytes = override.read_bytes()
    reports = [selected_report(source, coin, form, base=str(tmp_path / "input.json"), field=field)
               for form in ("selected", "direct")]

    for report in reports:
        assert report["value"] == value
        assert report["sources"][-1]["kind"] == "override_file"
        assert report["sources"][-1]["path"] == f"coin_overrides.{coin}.bot.long.unstuck.{field}"
        assert Path(report["sources"][-1]["file"]) == override
        assert report["search"]["status"] == "coin_override"
        assert report["search"]["gene_status"] == "omitted_fixed"
        assert not any(event["kind"] == "normalization" for event in report["sources"])
    assert reports[0]["search"] == reports[1]["search"]
    assert source == original
    assert override.read_bytes() == original_bytes


@pytest.mark.parametrize("coin", COINS)
def test_coin_selection_does_not_invent_a_pin_for_an_inherited_root_default(coin):
    source = {"config_version": "v9.0.0", "bot": {"long": {}}, "live": {},
              "coin_overrides": {coin: coin_patch(0.2)}}
    report = explain_config(source, "bot.long.unstuck.ema_span_0_minutes", coin=coin)
    assert report["sources"] == [{"kind": "default_or_derived", "path": "bot.long.unstuck.ema_span_0_minutes"}]
    assert report["search"]["status"] == "omitted_fixed"
    with pytest.raises(KeyError, match="unknown or inactive"):
        explain_config(source, f"coin_overrides.{coin}.bot.long.unstuck.ema_span_0_minutes")


@pytest.mark.parametrize("coin", COINS)
def test_coin_selection_keeps_simulation_null_inheritance_attributed_to_global_policy(coin):
    source = policy()
    source["backtest"]["filter_by_min_effective_cost"] = None
    source["coin_overrides"] = {coin: coin_patch(0.2)}
    report = explain_config(source, "backtest.filter_by_min_effective_cost", coin=coin)
    assert report["value"] is True
    assert report["sources"][-1] == {
        "kind": "inherited", "path": "bot.global.universe.filter_by_min_effective_cost", "value": True,
    }
    assert report["search"]["status"] == "not_in_bounds"


@pytest.mark.parametrize("selected", ["SHIB", "1000SHIB"])
@pytest.mark.parametrize("form", ["selected", "direct"])
@pytest.mark.parametrize("origin", ["file", "inline"])
def test_normalized_coin_alias_retains_its_authored_source(tmp_path, selected, form, origin):
    source = policy()
    override = write_json(tmp_path / "alias.json", coin_patch(0.12))
    source["coin_overrides"] = {"1000SHIB": {"override_config_path": str(override)}
                                if origin == "file" else coin_patch(0.12)}
    original = deepcopy(source)
    report = selected_report(source, selected, form, base=str(tmp_path / "input.json"))
    assert report["value"] == 0.12
    assert report["sources"][-1]["kind"] == ("override_file" if origin == "file" else "coin_override")
    assert report["sources"][-1]["path"] == "coin_overrides.SHIB.bot.long.unstuck.close_fraction"
    if origin == "file":
        assert report["sources"][-1]["file"] == str(override)
    assert report["search"]["status"] == "coin_override"
    assert source == original


@pytest.mark.parametrize("form", ["selected", "direct"])
@pytest.mark.parametrize("scope", ["leaf", "coin", "map"])
def test_alias_scenario_layer_is_visible_under_the_canonical_coin(tmp_path, form, scope):
    source = policy()
    source["coin_overrides"] = {"1000SHIB": coin_patch(0.12)}
    if scope == "leaf":
        selector, value = "coin_overrides.SHIB.bot.long.unstuck.close_fraction", 0.31
        expected_path = "coin_overrides.SHIB.bot.long.unstuck.close_fraction"
    elif scope == "coin":
        selector, value = "coin_overrides.SHIB", coin_patch(0.31)
        expected_path = "coin_overrides.SHIB"
    else:
        selector, value = "coin_overrides", {"1000SHIB": coin_patch(0.31)}
        expected_path = selector
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {selector: value}}]
    original = deepcopy(source)
    expected = suite_policy(source, base=str(tmp_path / "input.json"), scenario="one", optimize=True)
    report = selected_report(source, "SHIB", form, base=str(tmp_path / "input.json"), scenario="one")
    assert report["value"] == expected["coin_overrides"]["SHIB"]["bot"]["long"]["unstuck"]["close_fraction"] == 0.31
    assert report["sources"][-1]["kind"] == "scenario_override"
    assert report["sources"][-1]["path"] == expected_path
    assert report["search"]["status"] == "scenario_override"
    assert report["search"]["shadowed_by"] == [{"kind": "scenario_override", "path": expected_path}]
    assert not any(event["kind"] == "normalization" for event in report["sources"])
    assert source == original


@pytest.mark.parametrize("authored_coin,selector_coin", [
    ("1000SHIB", "1000SHIB"), ("1000SHIB", "SHIB"), ("SHIB", "SHIB"),
])
def test_explain_preserves_strict_fixed_runtime_alias_selector_authority(tmp_path, authored_coin, selector_coin):
    source = policy()
    source["coin_overrides"] = {authored_coin: coin_patch(0.12)}
    selector = f"coin_overrides.{selector_coin}.bot.long.unstuck.close_fraction"
    source["optimize"]["fixed_runtime_overrides"] = {selector: 0.21}
    if authored_coin == "1000SHIB":
        with pytest.raises((KeyError, ValueError), match="Unknown override path") as owner_error:
            suite_policy(source, base=str(tmp_path / "input.json"), optimize=True)
        with pytest.raises(owner_error.type, match="Unknown override path"):
            selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"))
    else:
        expected = suite_policy(source, base=str(tmp_path / "input.json"), optimize=True)
        report = selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"))
        assert report["value"] == expected["coin_overrides"]["SHIB"]["bot"]["long"]["unstuck"]["close_fraction"] == 0.21
        assert report["sources"][-1]["kind"] == "fixed_runtime_override"
        assert report["search"]["status"] == "fixed_runtime_override"


@pytest.mark.parametrize("authored_coin,canonical_coin", [("BTC", "BTC"), ("1000SHIB", "SHIB")])
@pytest.mark.parametrize("form", ["selected", "direct"])
@pytest.mark.parametrize("origin", ["file", "inline"])
@pytest.mark.parametrize("field", ["close_fraction", "ema_span_0_minutes"])
def test_coupling_keeps_untouched_sources_and_identifies_its_generated_span(
    tmp_path, authored_coin, canonical_coin, form, origin, field,
):
    source = policy()
    patch = {"bot": {"long": {"unstuck": {"close_fraction": 0.12, "ema_span_0_minutes": 17.5},
                              "strategy": {"trailing_martingale": {"entry": {"ema_span_0_minutes": 321.5}}}}}}
    override = write_json(tmp_path / "coupled.json", patch)
    source["coin_overrides"] = {authored_coin: {"override_config_path": str(override)}
                                if origin == "file" else patch}
    source["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    original = deepcopy(source)
    original_file = override.read_bytes()
    expected = suite_policy(source, base=str(tmp_path / "input.json"), optimize=True)
    report = selected_report(source, canonical_coin, form, base=str(tmp_path / "input.json"), field=field)
    assert report["value"] == expected["coin_overrides"][canonical_coin]["bot"]["long"]["unstuck"][field]
    event = report["sources"][-1]
    assert event["path"] == f"coin_overrides.{canonical_coin}.bot.long.unstuck.{field}"
    if field == "ema_span_0_minutes":
        assert report["value"] == 321.5
        assert event["kind"] == "optimizer_transform"
        assert event["name"] == "couple_unstuck_ema_spans"
        assert event["dependencies"] == [{
            "path": f"coin_overrides.{canonical_coin}.bot.long.strategy.trailing_martingale.entry.ema_span_0_minutes",
            "path_parts": ["coin_overrides", canonical_coin, "bot", "long", "strategy",
                           "trailing_martingale", "entry", "ema_span_0_minutes"],
        }]
        assert report["search"]["status"] == "optimizer_transform"
        assert report["search"]["gene_status"] == "not_in_genome"
    else:
        assert report["value"] == 0.12
        assert event["kind"] == ("override_file" if origin == "file" else "coin_override")
        if origin == "file":
            assert event["file"] == str(override)
        assert report["search"]["status"] == "coin_override"
    assert source == original
    assert override.read_bytes() == original_file


@pytest.mark.parametrize("scope", ["coin", "map"])
@pytest.mark.parametrize("replacement", ["inline", "file", "removed"])
def test_coupling_scenario_replacement_does_not_reuse_an_earlier_file_source(tmp_path, scope, replacement):
    source = policy()
    old_file = write_json(tmp_path / "old.json", coin_patch(0.12))
    new_file = write_json(tmp_path / "new.json", coin_patch(0.31))
    source["coin_overrides"] = {"1000SHIB": {"override_config_path": str(old_file)}}
    source["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    patch = (coin_patch(0.31) if replacement == "inline" else
             {"override_config_path": str(new_file)} if replacement == "file" else
             {"bot": {"long": {"unstuck": {"ema_span_0_minutes": 17.5}}}})
    selector = "coin_overrides.SHIB" if scope == "coin" else "coin_overrides"
    value = patch if scope == "coin" else {"SHIB": patch}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {selector: value}}]
    original = deepcopy(source)
    originals = {path: path.read_bytes() for path in (old_file, new_file)}
    expected = suite_policy(source, base=str(tmp_path / "input.json"), scenario="one", optimize=True)
    report = selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"), scenario="one")
    assert report["value"] == expected["coin_overrides"]["SHIB"]["bot"]["long"]["unstuck"].get(
        "close_fraction", expected["bot"]["long"]["unstuck"]["close_fraction"],
    )
    assert report["value"] == (source["bot"]["long"]["unstuck"]["close_fraction"]
                               if replacement == "removed" else 0.31)
    file_events = [event for event in report["sources"] if event["kind"] == "override_file"]
    if replacement == "file":
        assert report["sources"][-1]["kind"] == "override_file"
        assert [event["file"] for event in file_events] == [str(new_file)]
    else:
        assert not file_events
        assert report["sources"][-1]["kind"] == "scenario_override"
    assert not any(event["kind"] == "normalization" for event in report["sources"])
    assert source == original
    assert {path: path.read_bytes() for path in originals} == originals


def test_alias_coupling_scenario_keeps_the_suite_owner_selector_rejection(tmp_path):
    source = policy()
    override = write_json(tmp_path / "alias.json", coin_patch(0.12))
    source["coin_overrides"] = {"1000SHIB": {"override_config_path": str(override)}}
    source["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"]
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {
        "coin_overrides.1000SHIB": coin_patch(0.31),
    }}]
    with pytest.raises(KeyError, match="1000SHIB") as owner_error:
        suite_policy(source, base=str(tmp_path / "input.json"), scenario="one", optimize=True)
    with pytest.raises(owner_error.type, match="1000SHIB"):
        selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"), scenario="one")


@pytest.mark.parametrize("scope", ["leaf", "coin"])
def test_raw_alias_scenario_selector_does_not_gain_authority_in_explain(tmp_path, scope):
    source = policy()
    source["coin_overrides"] = {"1000SHIB": coin_patch(0.12)}
    selector = ("coin_overrides.1000SHIB.bot.long.unstuck.close_fraction"
                if scope == "leaf" else "coin_overrides.1000SHIB")
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {
        selector: 0.31 if scope == "leaf" else coin_patch(0.31),
    }}]
    with pytest.raises(KeyError, match="1000SHIB") as owner_error:
        suite_policy(source, base=str(tmp_path / "input.json"), scenario="one")
    with pytest.raises(owner_error.type, match="1000SHIB"):
        selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"), scenario="one")


def test_duplicate_authored_aliases_remain_rejected_by_suite_and_explain(tmp_path):
    source = policy()
    source["coin_overrides"] = {"1000SHIB": coin_patch(0.12), "SHIB": coin_patch(0.31)}
    with pytest.raises(ValueError, match="both normalize to 'SHIB'"):
        suite_policy(source, base=str(tmp_path / "input.json"))
    with pytest.raises(ValueError, match="both normalize to 'SHIB'"):
        selected_report(source, "SHIB", "selected", base=str(tmp_path / "input.json"))


@pytest.mark.parametrize("selected", ["SHIB", "1000SHIB"])
@pytest.mark.parametrize("origin", ["inline", "file", "mixed"])
def test_whole_coin_alias_queries_report_the_aggregate_authored_and_file_sources(tmp_path, selected, origin):
    source = policy()
    override = write_json(tmp_path / "container.json", {"bot": {"long": {"unstuck": {
        "close_fraction": 0.12, "ema_span_0_minutes": 17.5,
    }}}})
    patch = (coin_patch(0.31) if origin == "inline" else {"override_config_path": str(override)})
    if origin == "mixed":
        patch.update(coin_patch(0.31))
    source["coin_overrides"] = {"1000SHIB": patch}
    original = deepcopy(source)
    expected = suite_policy(source, base=str(tmp_path / "input.json"))["coin_overrides"]["SHIB"]
    report = explain_config(source, f"coin_overrides.{selected}", base_config_path=str(tmp_path / "input.json"))
    assert report["path"] == "coin_overrides.SHIB"
    assert report["value"] == expected
    assert any(event["kind"] == "authored" and event["value"] == patch for event in report["sources"])
    assert not any(event["kind"] in {"default_or_derived", "normalization"} for event in report["sources"])
    files = [event["file"] for event in report["sources"] if event["kind"] == "override_file"]
    assert set(files) == (set() if origin == "inline" else {str(override)})
    assert report["search"]["status"] == "not_in_bounds"
    assert source == original
