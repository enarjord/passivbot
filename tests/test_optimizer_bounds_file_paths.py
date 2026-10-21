"""Recommended-bound exports preserve the authority of referenced local files."""

from copy import deepcopy
import json
import socket

import pytest

from config import load_prepared_config, prepare_config
from config.authored import validate_authored_config
from config.overrides import parse_overrides
from config.schema import get_template_config
from config.scenario_overrides import normalize_scenario_overrides
from optimization.warmup import _apply_config_overrides
from tools.optimizer_bounds import apply_recommended_bounds, main


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*_args, **_kwargs):
        pytest.fail("bounds authoring must remain offline")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, denied)
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, denied)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def strategy_patch(**values):
    return {"bot": {"long": {"strategy": {"trailing_martingale": {"entry": values}}}}}


def entry(patch):
    return patch["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]


def source_config():
    config = get_template_config()
    config["optimize"]["bounds"] = {}
    return config


@pytest.mark.parametrize("context", ["argument", "metadata", "cwd", "cli"])
def test_moved_export_keeps_original_coin_file_and_inline_precedence(tmp_path, monkeypatch, context):
    original_dir, output_dir = tmp_path / "source", tmp_path / "output"
    original_dir.mkdir()
    output_dir.mkdir()
    original_file = write_json(original_dir / "coin.json", strategy_patch(ema_span_0=171.5, ema_span_1=222.5))
    decoy_file = write_json(output_dir / "coin.json", strategy_patch(ema_span_0=999.5, ema_span_1=999.5))
    config = source_config()
    config["coin_overrides"] = {"BTC": {"override_config_path": "coin.json", **strategy_patch(ema_span_0=345.5)}}
    source_path, output_path = original_dir / "source.json", output_dir / "search.json"
    if context == "metadata":
        config["live"]["base_config_path"] = str(source_path)
    write_json(source_path, config)
    before = deepcopy(config)
    original_bytes = {path: path.read_bytes() for path in (source_path, original_file, decoy_file)}
    monkeypatch.chdir(original_dir if context == "cwd" else output_dir)
    if context == "cli":
        assert main([str(source_path), str(output_path)]) == 0
        exported = json.loads(output_path.read_text())
    else:
        kwargs = {"base_config_path": str(source_path)} if context == "argument" else {}
        exported = apply_recommended_bounds(config, **kwargs)
        write_json(output_path, exported)
    assert config == before
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes
    assert exported["coin_overrides"]["BTC"]["override_config_path"] == str(original_file.resolve())
    assert "base_config_path" not in exported["live"]
    validate_authored_config(exported, get_template_config())
    monkeypatch.chdir(output_dir)
    reloaded = load_prepared_config(str(output_path), verbose=False)
    resolved = parse_overrides(reloaded, verbose=False)
    assert entry(resolved["coin_overrides"]["BTC"])["ema_span_0"] == 345.5
    assert entry(resolved["coin_overrides"]["BTC"])["ema_span_1"] == 222.5
    # The file remains authoritative: later edits still apply without changing
    # the exported inline patch or accidentally selecting the destination decoy.
    write_json(original_file, strategy_patch(ema_span_0=444.5, ema_span_1=333.5))
    resolved = parse_overrides(load_prepared_config(str(output_path), verbose=False), verbose=False)
    assert entry(resolved["coin_overrides"]["BTC"])["ema_span_0"] == 345.5
    assert entry(resolved["coin_overrides"]["BTC"])["ema_span_1"] == 333.5
    assert source_path.read_bytes() == original_bytes[source_path]
    assert decoy_file.read_bytes() == original_bytes[decoy_file]


@pytest.mark.parametrize("atomic_coin", [False, True])
def test_scenario_coin_files_keep_their_scope_and_fixed_scalar_pins(tmp_path, monkeypatch, atomic_coin):
    source_dir, destination = tmp_path / "source", tmp_path / "destination"
    source_dir.mkdir()
    destination.mkdir()
    root_file = write_json(source_dir / "root.json", strategy_patch(ema_span_0=101.5))
    scenario_file = write_json(source_dir / "scenario.json", strategy_patch(ema_span_0=202.5, ema_span_1=303.5))
    write_json(destination / "root.json", strategy_patch(ema_span_0=901.5))
    write_json(destination / "scenario.json", strategy_patch(ema_span_0=902.5))
    config = source_config()
    config["coin_overrides"] = {"BTC": {"override_config_path": root_file.name}}
    patch = {"override_config_path": scenario_file.name, **strategy_patch(ema_span_0=404.5)}
    selector = "coin_overrides.BTC" if atomic_coin else "coin_overrides"
    overrides = {selector: patch if atomic_coin else {"BTC": patch}}
    config["backtest"]["scenarios"] = [{"label": "coin-file", "overrides": overrides}]
    fixed = {"bot.long.strategy.trailing_martingale.entry.ema_span_1_minutes": 505.5}
    config["optimize"]["fixed_runtime_overrides"] = fixed
    source_path = write_json(source_dir / "source.json", config)
    source_bytes = source_path.read_bytes()
    before = deepcopy(config)
    exported = apply_recommended_bounds(config, base_config_path=str(source_path))
    assert config == before
    assert exported["optimize"]["fixed_runtime_overrides"] == fixed
    exported_patch = exported["backtest"]["scenarios"][0]["overrides"][selector]
    exported_patch = exported_patch if atomic_coin else exported_patch["BTC"]
    assert set(exported["backtest"]["scenarios"][0]["overrides"]) == {selector}
    assert exported_patch == {**patch, "override_config_path": str(scenario_file.resolve())}
    output_path = write_json(destination / "search.json", exported)
    monkeypatch.chdir(destination)
    effective = load_prepared_config(str(output_path), verbose=False)
    assert entry(parse_overrides(effective, verbose=False)["coin_overrides"]["BTC"])["ema_span_0"] == 101.5
    _apply_config_overrides(effective, effective["optimize"]["fixed_runtime_overrides"])
    assert entry(effective)["ema_span_1"] == 505.5
    _apply_config_overrides(effective, normalize_scenario_overrides(effective["backtest"]["scenarios"][0]["overrides"]))
    resolved = parse_overrides(effective, verbose=False)
    assert entry(resolved["coin_overrides"]["BTC"]) == {"ema_span_0": 404.5, "ema_span_1": 303.5}
    assert source_path.read_bytes() == source_bytes


def test_retained_universe_files_use_original_resolver_in_scenarios_and_fixed_pins(tmp_path, monkeypatch):
    source_dir, destination = tmp_path / "source", tmp_path / "destination"
    source_dir.mkdir()
    destination.mkdir()
    approved = write_json(source_dir / "approved.json", ["bybit::BTCUSDT", "ETH"])
    ignored = write_json(source_dir / "ignored.json", ["SOL"])
    short = write_json(source_dir / "short.json", ["XRP"])
    for path in (approved, ignored, short):
        write_json(destination / path.name, ["DECOY"])
    config = source_config()
    universe = {"approved_coins": approved.name, "ignored_coins": [ignored.name]}
    config["bot"]["long"]["universe"] = deepcopy(universe)
    config["optimize"]["fixed_runtime_overrides"] = {"bot.short.universe.approved_coins": short.name}
    config["backtest"]["scenarios"] = [{"label": "lists", "overrides": {"bot.long.universe": deepcopy(universe)}}]
    source_path = write_json(source_dir / "source.json", config)
    source_bytes = {path: path.read_bytes() for path in (source_path, approved, ignored, short)}
    before = deepcopy(config)
    monkeypatch.chdir(source_dir)
    exported = apply_recommended_bounds(config, base_config_path=str(source_path))
    assert config == before
    assert exported["bot"]["long"]["universe"] == {"approved_coins": ["bybit::BTCUSDT", "ETH"], "ignored_coins": ["SOL"]}
    assert exported["optimize"]["fixed_runtime_overrides"] == {"bot.short.universe.approved_coins": str(short.resolve())}
    scenario = exported["backtest"]["scenarios"][0]["overrides"]
    assert scenario == {"bot.long.universe": {"approved_coins": str(approved.resolve()), "ignored_coins": [str(ignored.resolve())]}}
    output_path = write_json(destination / "search.json", exported)
    monkeypatch.chdir(destination)
    reloaded = load_prepared_config(str(output_path), verbose=False)
    _apply_config_overrides(reloaded, reloaded["optimize"]["fixed_runtime_overrides"])
    _apply_config_overrides(reloaded, normalize_scenario_overrides(scenario))
    normalized = prepare_config(reloaded, base_config_path=str(output_path), verbose=False)
    assert normalized["bot"]["long"]["universe"] == exported["bot"]["long"]["universe"]
    assert normalized["bot"]["short"]["universe"]["approved_coins"] == ["XRP"]
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


def test_missing_override_file_fails_before_creating_output(tmp_path):
    config = source_config()
    config["coin_overrides"] = {"BTC": {"override_config_path": "missing.json"}}
    source_path = write_json(tmp_path / "source.json", config)
    source_bytes = source_path.read_bytes()
    output_path = tmp_path / "output.json"
    with pytest.raises(FileNotFoundError, match="missing.json"):
        main([str(source_path), str(output_path)])
    assert not output_path.exists()
    assert source_path.read_bytes() == source_bytes


def test_file_references_do_not_weaken_overwrite_guards(tmp_path):
    write_json(tmp_path / "coin.json", strategy_patch(ema_span_0=171.5))
    config = source_config()
    config["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    source_path = write_json(tmp_path / "source.json", config)
    output_path = tmp_path / "output.json"
    output_path.write_text("existing output\n")
    source_bytes = source_path.read_bytes()
    with pytest.raises(FileExistsError):
        main([str(source_path), str(output_path)])
    with pytest.raises(SystemExit) as error:
        main([str(source_path), str(source_path)])
    assert error.value.code == 2
    assert source_path.read_bytes() == source_bytes
    assert output_path.read_text() == "existing output\n"
