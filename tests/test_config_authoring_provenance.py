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
ROOT_PATH = "bot.long.unstuck.close_pct"


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
    source["bot"]["long"]["unstuck"]["close_pct"] = 0.05
    return source


def coin_patch(value):
    return {"bot": {"long": {"unstuck": {"close_pct": value}}}}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def selected_report(source, coin, form, *, base, scenario=None, field="close_pct"):
    path = f"bot.long.unstuck.{field}"
    if form == "direct":
        return explain_config(source, f"coin_overrides.{coin}.{path}",
                              base_config_path=base, scenario=scenario, optimize=True)
    return explain_config(source, path, coin=coin, base_config_path=base,
                          scenario=scenario, optimize=True)


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
@pytest.mark.parametrize("field,value", [("close_pct", 0.11), ("ema_span_0", 17.5)])
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
    report = explain_config(source, "bot.long.unstuck.ema_span_0", coin=coin)
    assert report["sources"] == [{"kind": "default_or_derived", "path": "bot.long.unstuck.ema_span_0"}]
    assert report["search"]["status"] == "omitted_fixed"
    with pytest.raises(KeyError, match="unknown or inactive"):
        explain_config(source, f"coin_overrides.{coin}.bot.long.unstuck.ema_span_0")


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
