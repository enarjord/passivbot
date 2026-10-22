"""Authored exports preserve sparse declarations while inspection expands values."""

from copy import deepcopy
import json

import pytest

from config.authoring import compact_config, resolved_config
from config.bot_inheritance import get_bot_policy
from config.load import prepare_config
from config.overrides import parse_overrides
from config.universe import universe_source_view
from tools.clean_config import cleanup_config, render_source


def authored(kind="trailing_martingale"):
    return {"config_version": "v9.0.0", "live": {}, "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"enabled": False,
                   "universe": {"approved_coins": ["BTC", "ETH"], "ignored_coins": []},
                   "risk": {"total_wallet_exposure_limit": 1.5}},
        "long": {"enabled": True},
        "short": {"enabled": False, "risk": {"total_wallet_exposure_limit": 1.5},
                  "universe": {"ignored_coins": None}},
    }}


def load(source, base=""):
    return prepare_config(deepcopy(source), base_config_path=str(base),
                          live_only=True, verbose=False, log_config_transforms=False)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("export", [compact_config, resolved_config])
def test_authored_export_reload_preserves_equal_pins_and_missing_leaf_inheritance(kind, export):
    source = authored(kind)
    before = deepcopy(source)
    prepared = load(source)
    assert get_bot_policy(prepared) is not None
    output = export(prepared)
    assert source == before
    assert output["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert "risk" not in output["bot"]["long"]
    assert output["bot"]["short"]["universe"]["ignored_coins"] is None
    assert "universe" not in output["bot"]["long"]
    reloaded_json = json.loads(json.dumps(output))
    reloaded_json["bot"]["global"].setdefault("risk", {})["total_wallet_exposure_limit"] = 2.0
    reloaded = load(reloaded_json)
    assert reloaded["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 2.0
    assert reloaded["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert output == export(output)


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_resolved_common_file_and_side_null_keep_their_distinct_authority(tmp_path, field):
    source = authored()
    coins = tmp_path / "coins.json"
    coins.write_text('["BTC"]')
    source["bot"]["global"]["universe"][field] = "./coins.json"
    source["bot"]["short"]["universe"][field] = None
    base = tmp_path / "input.json"
    original = deepcopy(source)
    output = resolved_config(source, base_config_path=str(base))
    assert source == original
    assert output["bot"]["global"]["universe"][field] == ["BTC"]
    assert output["bot"]["short"]["universe"][field] is None
    assert field not in output["bot"]["long"].get("universe", {})
    coins.unlink()
    prepared = load(json.loads(json.dumps(output)), tmp_path / "moved.json")
    view = universe_source_view(prepared)
    assert view["bot"]["global"]["universe"][field] == ["BTC"]
    assert view["bot"]["short"]["universe"][field] is None


def test_compact_keeps_selected_coin_file_alias_when_output_moves(tmp_path):
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    first, second = inputs / "first.json", inputs / "second.json"
    first.write_text('["BTC"]')
    second.write_text('["ETH"]')
    alias = inputs / "coins.json"
    alias.symlink_to(first.name)
    source = authored()
    source["bot"]["global"]["universe"]["approved_coins"] = "./coins.json"
    output = compact_config(source, base_config_path=str(inputs / "input.json"),
                            destination_path=str(outputs / "compact.json"))
    assert output["bot"]["global"]["universe"]["approved_coins"] == "../inputs/coins.json"
    alias.unlink()
    alias.symlink_to(second.name)
    prepared = prepare_config(output, base_config_path=str(outputs / "compact.json"), verbose=False)
    assert prepared["bot"]["long"]["universe"]["approved_coins"] == ["ETH"]
    assert source["bot"]["global"]["universe"]["approved_coins"] == "./coins.json"


def test_format_preserves_raw_declarations_and_semantic_cleanup_keeps_equal_pins(tmp_path):
    source = authored()
    text = json.dumps(source)
    formatted = render_source(text, source=tmp_path / "input.json", mode="format")
    assert json.loads(formatted) == source
    output = cleanup_config(source)
    assert output["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert "risk" not in output["bot"]["long"]
    assert source == json.loads(text)


@pytest.mark.parametrize("kind,path", [
    ("trailing_martingale", ("entry", "ema_span_0_minutes")),
    ("ema_anchor", ("ema_span_1_minutes",)),
])
def test_lean_referenced_coin_file_imports_declared_global_parameters_only(tmp_path, kind, path):
    from config.strategy_spec import get_strategy_spec
    spec = get_strategy_spec(kind)
    parameter = next(item for item in spec["parameters"]
                     if item["config_path"] == ["strategy", "long", *path])
    value = float(parameter["default"]) + 0.125
    strategy = {}
    current = strategy
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value
    file_config = {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"enabled": True, "strategy": {kind: strategy},
                   "universe": {"approved_coins": ["ETH"]}},
    }}
    reference = tmp_path / "coin.json"
    reference.write_text(json.dumps(file_config))
    source = authored(kind)
    source["coin_overrides"] = {"BTC": {"override_config_path": "./coin.json"}}
    prepared = parse_overrides(load(source, tmp_path / "input.json"), verbose=False)
    patch = prepared["coin_overrides"]["BTC"]
    assert set(patch["bot"]) == {"long", "short"}
    for side in ("long", "short"):
        assert patch["bot"][side] == {"strategy": {kind: strategy}}
    assert json.loads(reference.read_text()) == file_config
