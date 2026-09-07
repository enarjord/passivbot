from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from pathlib import Path

import pytest
import tools.compose_coin_overrides as compose_tool

from config.load import prepare_config
from config.overrides import apply_allowed_modifications, get_allowed_modifications, parse_overrides
from config.schema import get_template_config
from tools.compose_coin_overrides import compose_directory, main


def _single_coin_config(coin: str) -> dict:
    config = get_template_config()
    config["live"]["approved_coins"] = {"long": [coin], "short": [coin]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    config["live"]["user"] = "test_user"
    config["coin_overrides"] = {}
    for side in ("long", "short"):
        risk = config["bot"][side]["risk"]
        risk["n_positions"] = 1.0
        risk["position_exposure_enforcer_enabled"] = False
        risk["total_exposure_enforcer_enabled"] = False
        config["bot"][side]["hsl"]["enabled"] = False
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.0
    config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0.0
    return config


def _write(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def test_composes_minimal_overrides_and_canonicalizes_disabled_features(tmp_path: Path):
    first = _single_coin_config("BTC")
    second = _single_coin_config("ETH")

    first["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "initial_qty_pct"
    ] = 0.01
    second["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
        "initial_qty_pct"
    ] = 0.02
    first["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 1.0
    second["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 2.0
    first["bot"]["long"]["forager"]["volatility_ema_span_1m"] = 100.0
    second["bot"]["long"]["forager"]["volatility_ema_span_1m"] = 200.0
    first["live"]["leverage"] = 3
    second["live"]["leverage"] = 5

    first["bot"]["long"]["hsl"]["red_threshold"] = 0.21
    second["bot"]["long"]["hsl"]["red_threshold"] = 0.11
    first["optimize"]["bounds"]["long"]["hsl"]["red_threshold"] = [0.03, 0.25, 0.01]
    first["bot"]["long"]["risk"]["position_exposure_enforcer_threshold"] = 0.9
    second["bot"]["long"]["risk"]["position_exposure_enforcer_threshold"] = 0.8
    first["optimize"]["bounds"]["long"]["risk"][
        "position_exposure_enforcer_threshold"
    ] = [0.7, 1.0, 0.01]

    _write(tmp_path / "a_btc.json", first)
    _write(tmp_path / "b_eth.json", second)

    composed, report = compose_directory(tmp_path)

    assert set(composed) == {
        "bot",
        "coin_overrides",
        "config_version",
        "live",
        "logging",
        "monitor",
    }
    assert composed["live"]["approved_coins"] == {
        "long": ["BTC", "ETH"],
        "short": ["BTC", "ETH"],
    }
    assert composed["bot"]["long"]["risk"]["n_positions"] == 2.0
    assert composed["bot"]["short"]["risk"]["n_positions"] == 1.0
    assert report.position_count_changes == {"long": (1.0, 2.0)}
    assert composed["bot"]["long"]["hsl"]["enabled"] is False
    assert composed["bot"]["long"]["hsl"]["red_threshold"] == 0.03
    assert (
        composed["bot"]["long"]["risk"]["position_exposure_enforcer_threshold"]
        == 0.7
    )

    assert "BTC" not in composed["coin_overrides"]
    eth = composed["coin_overrides"]["ETH"]
    assert (
        eth["bot"]["long"]["strategy"]["trailing_martingale"]["entry"][
            "initial_qty_pct"
        ]
        == 0.02
    )
    assert eth["bot"]["long"]["risk"]["entry_cooldown_minutes"] == 2.0
    assert eth["live"]["leverage"] == 5
    assert "hsl" not in eth["bot"]["long"]
    assert "position_exposure_enforcer_threshold" not in eth["bot"]["long"]["risk"]
    assert "bot.long.forager.volatility_ema_span_1m" in report.account_wide_conflicts

    prepared = prepare_config(composed, verbose=False, log_config_transforms=False)
    parse_overrides(prepared, verbose=False)


def test_selected_master_supplies_globals_and_optional_sections(tmp_path: Path):
    first = _single_coin_config("BTC")
    selected = _single_coin_config("ETH")
    first["bot"]["long"]["forager"]["volume_ema_span_1m"] = 300.0
    selected["bot"]["long"]["forager"]["volume_ema_span_1m"] = 600.0
    first["backtest"]["start_date"] = "2020-01-01"
    selected["backtest"]["start_date"] = "2021-01-01"
    first["optimize"]["iters"] = 100
    selected["optimize"]["iters"] = 200
    _write(tmp_path / "a_btc.json", first)
    selected_path = tmp_path / "b_eth.json"
    _write(selected_path, selected)

    composed, report = compose_directory(
        tmp_path,
        master_config=Path("b_eth.json"),
        include_backtest_optimize=True,
    )

    assert report.master_path == selected_path.resolve()
    assert report.master_was_selected is True
    assert composed["bot"]["long"]["forager"]["volume_ema_span_1m"] == 600.0
    assert composed["backtest"]["start_date"] == "2021-01-01"
    assert composed["optimize"]["iters"] == 200


def test_mixed_feature_enablement_omits_inert_dependent_overrides(tmp_path: Path):
    master = _single_coin_config("BTC")
    disabled = _single_coin_config("ETH")
    for config in (master, disabled):
        config["bot"]["long"]["hsl"]["enabled"] = True
        config["bot"]["long"]["unstuck"]["enabled"] = True
        config["bot"]["long"]["risk"]["position_exposure_enforcer_enabled"] = True
    disabled["bot"]["long"]["hsl"].update(
        {"enabled": False, "red_threshold": 0.24, "ema_span_minutes": 12.0}
    )
    disabled["bot"]["long"]["unstuck"].update(
        {"enabled": False, "threshold": 0.88, "close_pct": 0.11}
    )
    disabled["bot"]["long"]["risk"].update(
        {
            "position_exposure_enforcer_enabled": False,
            "position_exposure_enforcer_threshold": 0.75,
        }
    )
    _write(tmp_path / "a_btc.json", master)
    _write(tmp_path / "b_eth.json", disabled)

    composed, _report = compose_directory(tmp_path)

    eth_long = composed["coin_overrides"]["ETH"]["bot"]["long"]
    assert eth_long["hsl"] == {"enabled": False}
    assert eth_long["unstuck"] == {"enabled": False}
    assert eth_long["risk"] == {"position_exposure_enforcer_enabled": False}


def test_rejects_non_single_and_precomposed_inputs(tmp_path: Path):
    valid = _single_coin_config("BTC")
    invalid = _single_coin_config("ETH")
    invalid["live"]["approved_coins"]["long"].append("XRP")
    _write(tmp_path / "a.json", valid)
    _write(tmp_path / "b.json", invalid)

    with pytest.raises(ValueError, match="expected exactly one approved coin"):
        compose_directory(tmp_path)

    invalid = _single_coin_config("ETH")
    invalid["coin_overrides"] = {"ETH": {"live": {"leverage": 4}}}
    _write(tmp_path / "b.json", invalid)
    with pytest.raises(ValueError, match="must not contain coin_overrides"):
        compose_directory(tmp_path)


def test_rejects_nested_precomposed_input(tmp_path: Path):
    _write(tmp_path / "a.json", _single_coin_config("BTC"))
    nested = _single_coin_config("ETH")
    nested["coin_overrides"] = {"ETH": {"live": {"leverage": 4}}}
    _write(tmp_path / "b.json", {"config": nested})

    with pytest.raises(ValueError, match="must not contain coin_overrides"):
        compose_directory(tmp_path)


def test_rejects_all_as_single_coin(tmp_path: Path):
    _write(tmp_path / "a.json", _single_coin_config("BTC"))
    wildcard = _single_coin_config("ETH")
    wildcard["live"]["approved_coins"] = "all"
    _write(tmp_path / "b.json", wildcard)

    with pytest.raises(ValueError, match="'all' sentinel"):
        compose_directory(tmp_path)


def test_rejects_duplicate_coin_and_missing_master(tmp_path: Path):
    _write(tmp_path / "a.json", _single_coin_config("BTC"))
    _write(tmp_path / "b.json", _single_coin_config("BTC"))

    with pytest.raises(ValueError, match="duplicate single-coin config for BTC"):
        compose_directory(tmp_path)
    with pytest.raises(FileNotFoundError, match="selected master config does not exist"):
        compose_directory(tmp_path, master_config=tmp_path / "missing.json")


@pytest.mark.parametrize("relative_path", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_external_master_supplies_baseline_without_adding_coins(
    tmp_path: Path, monkeypatch, relative_path: bool, wrapped: bool
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    master = _single_coin_config("SOL")
    master["live"]["approved_coins"]["long"].append("XRP")
    master["live"]["ignored_coins"]["long"] = ["BTC", "DOGE"]
    master["live"]["leverage"] = 7
    master["bot"]["long"]["forager"]["volume_ema_span_1m"] = 777.0
    master["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 12.0
    master["optimize"]["iters"] = 321
    master["optimize"]["backend"] = "gpu"
    master["backtest"]["start_date"] = "2021-01-01"
    for name, coin, cooldown in (("a.json", "BTC", 3.0), ("b.json", "ETH", 6.0)):
        config = _single_coin_config(coin)
        config["bot"]["long"]["risk"]["entry_cooldown_minutes"] = cooldown
        _write(inputs / name, config)
    master_path = tmp_path / "master.hjson"
    _write(master_path, {"config": master} if wrapped else master)
    selected = master_path
    if relative_path:
        monkeypatch.chdir(tmp_path)
        selected = Path("master.hjson")

    composed, report = compose_directory(
        inputs, master_config=selected, include_backtest_optimize=True
    )

    assert report.master_path == master_path.resolve()
    assert report.master_was_selected is True
    assert report.source_paths == [inputs / "a.json", inputs / "b.json"]
    assert report.coins == ["BTC", "ETH"]
    assert composed["live"]["approved_coins"] == {
        "long": ["BTC", "ETH"], "short": ["BTC", "ETH"]
    }
    assert composed["live"]["ignored_coins"]["long"] == ["DOGE"]
    assert composed["live"]["leverage"] == 7
    assert composed["bot"]["long"]["forager"]["volume_ema_span_1m"] == 777.0
    assert composed["bot"]["long"]["risk"]["n_positions"] == 2.0
    assert composed["optimize"]["backend"] == "gpu"
    assert composed["optimize"]["iters"] == 321
    assert composed["backtest"]["start_date"] == "2021-01-01"
    assert set(composed["coin_overrides"]) == {"BTC", "ETH"}
    for coin, cooldown in (("BTC", 3.0), ("ETH", 6.0)):
        assert composed["coin_overrides"][coin]["bot"]["long"]["risk"][
            "entry_cooldown_minutes"
        ] == cooldown
    prepared = prepare_config(composed, verbose=False, log_config_transforms=False)
    parse_overrides(prepared, verbose=False)


def test_external_single_coin_master_is_not_an_extra_input(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    master = _single_coin_config("BTC")
    master["bot"]["long"]["hsl"]["enabled"] = True
    _write(tmp_path / "master.json", master)
    for name, coin in (("a.json", "BTC"), ("b.json", "ETH")):
        _write(inputs / name, _single_coin_config(coin))

    output = tmp_path / "composed.json"
    assert main([
        str(inputs), str(output), "--master-config", str(tmp_path / "master.json")
    ]) == 0
    composed = json.loads(output.read_text())
    assert "optimize" not in composed
    assert "backtest" not in composed
    assert composed["bot"]["long"]["hsl"]["enabled"] is True
    for coin in ("BTC", "ETH"):
        assert composed["coin_overrides"][coin]["bot"]["long"]["hsl"] == {"enabled": False}


@pytest.mark.parametrize("key,value", [
    ("strategy_kind", "ema_anchor"), ("hsl_signal_mode", "unified")
])
def test_external_master_must_match_input_strategy_and_hsl_mode(tmp_path: Path, key, value):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    master = _single_coin_config("SOL")
    master["live"][key] = value
    _write(tmp_path / "master.json", master)
    with pytest.raises(ValueError, match=f"same live.{key}"):
        compose_directory(inputs, master_config=tmp_path / "master.json")


@pytest.mark.parametrize("wrapped", [False, True])
def test_external_master_rejects_existing_overrides(tmp_path: Path, wrapped: bool):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    master = _single_coin_config("SOL")
    master["coin_overrides"] = {"BTC": {"live": {"leverage": 10}}}
    _write(tmp_path / "master.json", {"config": master} if wrapped else master)
    with pytest.raises(ValueError, match="master config must not contain coin_overrides"):
        compose_directory(inputs, master_config=tmp_path / "master.json")


def test_external_master_rejects_non_config_file(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    _write(tmp_path / "master.txt", _single_coin_config("SOL"))
    with pytest.raises(ValueError, match="master config must be a JSON/HJSON file"):
        compose_directory(inputs, master_config=tmp_path / "master.txt")


def test_rejects_duplicate_resolved_market_aliases(tmp_path: Path, monkeypatch):
    def fake_coin_to_symbol(identifier, exchange, **_kwargs):
        if exchange == "binance" and identifier in {"BTC", "BTCUSDT"}:
            return "BTC/USDT:USDT"
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", fake_coin_to_symbol)
    _write(tmp_path / "a.json", _single_coin_config("BTC"))
    _write(tmp_path / "b.json", _single_coin_config("BTCUSDT"))

    with pytest.raises(ValueError, match="resolve to the same market"):
        compose_directory(tmp_path)


def test_removes_ignored_alias_of_approved_market(tmp_path: Path, monkeypatch):
    def fake_coin_to_symbol(identifier, exchange, **_kwargs):
        if exchange == "binance" and identifier in {"BTC", "BTCUSDT"}:
            return "BTC/USDT:USDT"
        if identifier in {"ETH", "DOGE"}:
            return f"{identifier}/USDT:USDT"
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", fake_coin_to_symbol)
    master = _single_coin_config("ETH")
    master["live"]["ignored_coins"]["long"] = ["BTCUSDT", "DOGE"]
    _write(tmp_path / "a.json", master)
    _write(tmp_path / "b.json", _single_coin_config("BTC"))

    composed, _report = compose_directory(tmp_path)

    assert composed["live"]["ignored_coins"]["long"] == ["DOGE"]


@pytest.mark.parametrize("field", ["approved", "ignored"])
def test_rejects_unresolved_exact_market_identifiers(
    tmp_path: Path, monkeypatch, field: str
):
    def unavailable_market(*_args, **_kwargs):
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", unavailable_market)
    first = _single_coin_config("ETH")
    second = _single_coin_config("BTC")
    if field == "approved":
        second["live"]["approved_coins"] = {
            "long": ["hyperliquid::12345"],
            "short": ["hyperliquid::12345"],
        }
    else:
        first["live"]["ignored_coins"]["long"] = ["hyperliquid::12345"]
    _write(tmp_path / "a.json", first)
    _write(tmp_path / "b.json", second)

    with pytest.raises(ValueError, match="could not resolve exact market identifier"):
        compose_directory(tmp_path)


@pytest.mark.parametrize("field", ["approved", "ignored"])
def test_rejects_exact_identifier_resolving_to_different_venue_markets(
    tmp_path: Path, monkeypatch, field: str
):
    def fake_coin_to_symbol(identifier, exchange, **_kwargs):
        if identifier == "12345":
            return {
                "binance": "ABC/USDT:USDT",
                "bybit": "OTHER/USDT:USDT",
            }[exchange]
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", fake_coin_to_symbol)
    first = _single_coin_config("ETH")
    second = _single_coin_config("12345" if field == "approved" else "BTC")
    if field == "ignored":
        first["live"]["ignored_coins"]["long"] = ["12345"]
    _write(tmp_path / "a.json", first)
    _write(tmp_path / "b.json", second)

    with pytest.raises(ValueError, match="resolves to different contracts"):
        compose_directory(tmp_path)


def test_qualified_identifier_venue_participates_in_alias_resolution(
    tmp_path: Path, monkeypatch
):
    def fake_coin_to_symbol(identifier, exchange, **_kwargs):
        if exchange == "hyperliquid" and identifier in {
            "hyperliquid::12345",
            "xyz:TSLA",
        }:
            return "xyz:TSLA/USDC:USDC"
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", fake_coin_to_symbol)
    _write(tmp_path / "a.json", _single_coin_config("hyperliquid::12345"))
    _write(tmp_path / "b.json", _single_coin_config("xyz:TSLA"))

    with pytest.raises(ValueError, match="resolve to the same market"):
        compose_directory(tmp_path)


def test_qualified_identifier_venue_removes_ignored_alias(tmp_path: Path, monkeypatch):
    def fake_coin_to_symbol(identifier, exchange, **_kwargs):
        if exchange == "hyperliquid" and identifier in {
            "hyperliquid::12345",
            "xyz:TSLA",
        }:
            return "xyz:TSLA/USDC:USDC"
        if identifier == "ETH":
            return "ETH/USDC:USDC"
        raise compose_tool.MarketIdentifierResolutionError("unavailable")

    monkeypatch.setattr(compose_tool, "coin_to_symbol", fake_coin_to_symbol)
    master = _single_coin_config("ETH")
    master["live"]["ignored_coins"]["long"] = ["hyperliquid::12345"]
    _write(tmp_path / "a.json", master)
    _write(tmp_path / "b.json", _single_coin_config("xyz:TSLA"))

    composed, _report = compose_directory(tmp_path)

    assert composed["live"]["ignored_coins"]["long"] == []


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_retains_gpu_optimizer_with_coin_overrides(tmp_path: Path, strategy_kind: str):
    master = _single_coin_config("BTC")
    other = _single_coin_config("ETH")
    for config in (master, other):
        config["live"]["strategy_kind"] = strategy_kind
    master["optimize"]["backend"] = "gpu"
    master["optimize"]["iters"] = 123
    master["backtest"]["start_date"] = "2021-01-01"
    if strategy_kind == "ema_anchor":
        master["bot"]["long"]["strategy"][strategy_kind]["offset"] = 0.01
        other["bot"]["long"]["strategy"][strategy_kind]["offset"] = 0.02
        expected_strategy = {"offset": 0.02}
    else:
        master["bot"]["long"]["strategy"][strategy_kind]["entry"]["initial_qty_pct"] = 0.01
        other["bot"]["long"]["strategy"][strategy_kind]["entry"]["initial_qty_pct"] = 0.02
        expected_strategy = {"entry": {"initial_qty_pct": 0.02}}
    _write(tmp_path / "a.json", master)
    _write(tmp_path / "b.json", other)

    lean, _report = compose_directory(tmp_path)
    assert "optimize" not in lean
    assert "backtest" not in lean

    output = tmp_path / "composed.json"
    assert main([str(tmp_path), str(output), "--include-backtest-optimize"]) == 0
    composed = json.loads(output.read_text(encoding="utf-8"))
    assert composed["optimize"]["backend"] == "gpu"
    assert composed["optimize"]["gpu"] == master["optimize"]["gpu"]
    assert composed["optimize"]["iters"] == 123
    assert composed["backtest"]["start_date"] == "2021-01-01"
    assert composed["live"]["approved_coins"]["long"] == ["BTC", "ETH"]
    assert composed["bot"]["long"]["risk"]["n_positions"] == 2.0
    assert composed["coin_overrides"] == {
        "ETH": {"bot": {"long": {"strategy": {strategy_kind: expected_strategy}}}}
    }
    prepared = prepare_config(composed, verbose=False, log_config_transforms=False)
    parse_overrides(prepared, verbose=False)
    assert prepared["optimize"]["backend"] == "gpu"


def test_cli_writes_sorted_config_and_protects_existing_output(tmp_path: Path, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    output = tmp_path / "composed.json"

    assert main([str(inputs), str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    report_output = capsys.readouterr().out
    assert "Master source (alphabetically first)" in report_output
    assert "Set bot.long.risk.n_positions: 1 -> 2 (approved coin count)" in report_output

    assert main([str(inputs), str(output)]) == 2
    assert "pass --overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hsl_signal_mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("external_master", [False, True])
def test_verbose_overrides_survive_master_edits(
    tmp_path: Path, strategy_kind: str, hsl_signal_mode: str, external_master: bool
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name, coin in (("a.json", "BTC"), ("b.json", "ETH")):
        config = _single_coin_config(coin)
        config["live"]["strategy_kind"] = strategy_kind
        config["live"]["hsl_signal_mode"] = hsl_signal_mode
        config["live"]["leverage"] = 3
        config["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
        config["bot"]["long"]["hsl"]["red_threshold"] = 0.123
        _write(inputs / name, config)
    master_path = None
    if external_master:
        master_path = tmp_path / "master.json"
        _write(master_path, config)

    sources, master, selected = compose_tool.load_single_coin_directory(
        inputs, master_config=master_path
    )
    original_sources = deepcopy((sources, master))
    lean, lean_report = compose_tool.compose_configs(sources, master_source=master)
    verbose, verbose_report = compose_tool.compose_configs(
        sources, master_source=master, master_was_selected=selected, override_mode="verbose"
    )
    assert (sources, master) == original_sources
    assert lean["coin_overrides"] == {}
    assert set(verbose["coin_overrides"]) == {"BTC", "ETH"}
    assert lean_report.override_mode == "lean"
    assert verbose_report.override_mode == "verbose"
    assert verbose_report.canonicalized_features == []
    assert verbose_report.account_wide_conflicts == lean_report.account_wide_conflicts

    # Exercise the real override loader after changing global values which were
    # equal at composition time, including a disabled feature's enable flag.
    verbose["live"]["leverage"] = 9
    verbose["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 99.0
    verbose["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 2.0
    verbose["bot"]["long"]["strategy"][strategy_kind] = deepcopy(
        verbose["bot"]["short"]["strategy"][strategy_kind]
    )
    if strategy_kind == "ema_anchor":
        verbose["bot"]["long"]["strategy"][strategy_kind]["offset"] = 0.15
    else:
        verbose["bot"]["long"]["strategy"][strategy_kind]["entry"]["initial_qty_pct"] = 0.15
    if hsl_signal_mode == "coin":
        verbose["bot"]["long"]["hsl"]["enabled"] = True
        verbose["bot"]["long"]["hsl"]["red_threshold"] = 0.2
    prepared = prepare_config(verbose, verbose=False, log_config_transforms=False)
    parsed = parse_overrides(prepared, verbose=False)
    policy = get_allowed_modifications(hsl_signal_mode=hsl_signal_mode)
    for source in sources:
        patch = parsed["coin_overrides"][source.coin]
        effective = apply_allowed_modifications(prepared, patch, policy)
        assert effective["live"]["leverage"] == 3
        assert effective["bot"]["long"]["risk"]["entry_cooldown_minutes"] == 7.0
        assert effective["bot"]["long"]["strategy"] == source.config["bot"]["long"]["strategy"]
        assert effective["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 2.0
        assert "n_positions" not in patch["bot"]["long"]["risk"]
        if hsl_signal_mode == "coin":
            assert effective["bot"]["long"]["hsl"]["enabled"] is False
            assert effective["bot"]["long"]["hsl"]["red_threshold"] == 0.123
        else:
            assert "hsl" not in patch["bot"]["long"]


def test_verbose_cli_keeps_original_inactive_values(tmp_path: Path, capsys):
    for name, coin, threshold in (("a.json", "BTC", 0.123), ("b.json", "ETH", 0.234)):
        config = _single_coin_config(coin)
        config["bot"]["long"]["hsl"]["red_threshold"] = threshold
        _write(tmp_path / name, config)
    output = tmp_path / "result.json"
    assert main([str(tmp_path), str(output), "--override-mode", "verbose"]) == 0
    payload = json.loads(output.read_text())
    for coin, threshold in (("BTC", 0.123), ("ETH", 0.234)):
        assert payload["coin_overrides"][coin]["bot"]["long"]["hsl"]["red_threshold"] == threshold
    assert "Coin override mode: verbose" in capsys.readouterr().out
    assert main([str(tmp_path), str(output), "--override-mode", "lean", "--overwrite"]) == 0
    assert json.loads(output.read_text())["coin_overrides"] == {}


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_refuses_overwriting_external_master(tmp_path: Path, symlink: bool, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    master = tmp_path / "master.json"
    _write(master, _single_coin_config("SOL"))
    original = master.read_bytes()
    output = master
    if symlink:
        output = tmp_path / "link.json"
        output.symlink_to(master)
    assert main([
        str(inputs), str(output), "--master-config", str(master), "--overwrite"
    ]) == 2
    assert "must not overwrite the selected master" in capsys.readouterr().err
    assert master.read_bytes() == original


def test_failed_output_replacement_preserves_existing_file(tmp_path: Path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("original content")

    def fail_replace(source, destination):
        assert json.loads(Path(source).read_text()) == {"new": "content"}
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(compose_tool.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        compose_tool.write_config({"new": "content"}, output, overwrite=True)
    assert output.read_text() == "original content"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077])
def test_new_output_uses_normal_umask_permissions(tmp_path: Path, mask):
    output = tmp_path / "new.json"
    previous = os.umask(mask)
    try:
        compose_tool.write_config({"new": "content"}, output)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(output.stat().st_mode) == 0o666 & ~mask


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_overwrite_preserves_output_permissions_and_ownership(tmp_path: Path, mode):
    output = tmp_path / "existing.json"
    output.write_text("original")
    output.chmod(mode)
    before = output.stat()
    compose_tool.write_config({"new": "content"}, output, overwrite=True)
    after = output.stat()
    assert stat.S_IMODE(after.st_mode) == mode
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert json.loads(output.read_text()) == {"new": "content"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership semantics")
def test_failed_ownership_preservation_leaves_original_output(tmp_path: Path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")
    real_stat = Path.stat

    def foreign_owner(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == output:
            values = list(result)
            values[4] += 1
            return os.stat_result(values)
        return result

    def denied_chown(path, uid, gid):
        raise PermissionError("cannot preserve output ownership")

    monkeypatch.setattr(Path, "stat", foreign_owner)
    monkeypatch.setattr(compose_tool.os, "chown", denied_chown)
    with pytest.raises(PermissionError, match="cannot preserve output ownership"):
        compose_tool.write_config({"new": "content"}, output, overwrite=True)
    assert output.read_text() == "original"
    assert list(tmp_path.iterdir()) == [output]


def test_output_created_during_composition_is_not_overwritten(tmp_path: Path, monkeypatch):
    output = tmp_path / "result.json"
    real_link = compose_tool.os.link

    def concurrent_link(source, destination):
        output.write_text("concurrent output")
        real_link(source, destination)

    monkeypatch.setattr(compose_tool.os, "link", concurrent_link)
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        compose_tool.write_config({"new": "content"}, output)
    assert output.read_text() == "concurrent output"
    assert list(tmp_path.iterdir()) == [output]


def test_cli_refuses_excluding_and_overwriting_single_coin_input(tmp_path: Path, capsys):
    for coin in ("BTC", "ETH", "SOL"):
        _write(tmp_path / f"{coin}.json", _single_coin_config(coin))
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert main([str(tmp_path), str(tmp_path / "ETH.json"), "--overwrite"]) == 2
    assert "must not overwrite a single-coin input" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original


@pytest.mark.parametrize("mode", ["lean", "verbose"])
@pytest.mark.parametrize("strategy_kind", ["ema_anchor", "trailing_martingale"])
def test_custom_params_pin_selection_and_inherit_global_unstuck(tmp_path: Path, mode, strategy_kind):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    master = _single_coin_config("SOL")
    master["live"]["strategy_kind"] = strategy_kind
    master["bot"]["long"]["unstuck"].update({"enabled": False, "threshold": 0.83})
    master["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 7.0
    master_path = tmp_path / "master.json"
    _write(master_path, master)
    for coin, cooldown in (("BTC", 7.0), ("ETH", 9.0)):
        source = deepcopy(master)
        source["live"]["approved_coins"] = {"long": [coin], "short": [coin]}
        source["bot"]["long"]["risk"]["entry_cooldown_minutes"] = cooldown
        source["bot"]["long"]["unstuck"].update({"enabled": True, "threshold": 0.6})
        _write(inputs / f"{coin}.json", source)

    output = tmp_path / "result.json"
    selectors = "long.strategy,long.risk.entry_cooldown_minutes"
    assert main([
        str(inputs), str(output), "--master-config", str(master_path),
        "--override-mode", mode, "--override-params", selectors,
    ]) == 0
    composed = json.loads(output.read_text())
    for coin, cooldown in (("BTC", 7.0), ("ETH", 9.0)):
        source = compose_tool.load_single_coin_config(inputs / f"{coin}.json")
        assert composed["coin_overrides"][coin] == {
            "bot": {"long": {
                "strategy": source.config["bot"]["long"]["strategy"],
                "risk": {"entry_cooldown_minutes": cooldown},
            }}
        }
    assert composed["bot"]["long"]["unstuck"] == master["bot"]["long"]["unstuck"]
    composed["bot"]["long"]["unstuck"].update({"enabled": True, "threshold": 0.91})
    composed["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 100.0
    prepared = prepare_config(composed, verbose=False, log_config_transforms=False)
    parsed = parse_overrides(prepared, verbose=False)
    policy = get_allowed_modifications(hsl_signal_mode=prepared["live"]["hsl_signal_mode"])
    for coin, cooldown in (("BTC", 7.0), ("ETH", 9.0)):
        effective = apply_allowed_modifications(prepared, parsed["coin_overrides"][coin], policy)
        assert effective["bot"]["long"]["risk"]["entry_cooldown_minutes"] == cooldown
        assert effective["bot"]["long"]["unstuck"]["enabled"] is True
        assert effective["bot"]["long"]["unstuck"]["threshold"] == 0.91


@pytest.mark.parametrize("selectors", [
    "bot.long.risk.entry_cooldown_minutes",
    " long.risk.entry_cooldown_minutes , bot.long.risk.entry_cooldown_minutes ",
])
def test_custom_leaf_aliases_and_overlap_are_deduplicated(tmp_path: Path, selectors):
    for coin in ("BTC", "ETH"):
        _write(tmp_path / f"{coin}.json", _single_coin_config(coin))
    composed, report = compose_directory(tmp_path, override_params=selectors)
    assert report.override_mode == "custom"
    assert report.canonicalized_features == []
    for patch in composed["coin_overrides"].values():
        assert patch == {"bot": {"long": {"risk": {
            "entry_cooldown_minutes": composed["bot"]["long"]["risk"]["entry_cooldown_minutes"]
        }}}}


@pytest.mark.parametrize("selector", ["*.risk.entry_cooldown_minutes", "entry_cooldown_minutes"])
def test_custom_wildcards_and_leaf_suffixes_match_both_sides(tmp_path: Path, selector):
    for coin in ("BTC", "ETH"):
        _write(tmp_path / f"{coin}.json", _single_coin_config(coin))
    composed, _ = compose_directory(tmp_path, override_params=selector)
    for patch in composed["coin_overrides"].values():
        assert set(patch["bot"]) == {"long", "short"}
        for side in ("long", "short"):
            assert set(patch["bot"][side]) == {"risk"}
            assert set(patch["bot"][side]["risk"]) == {"entry_cooldown_minutes"}


@pytest.mark.parametrize("selector", [
    "", " ", "long.strategy,", "long..strategy", "long.stratgey",
    "long.strategy,long.stratgey",
    "long.risk.n_positions", "long.forager", "long.strategy.ema_anchor",
])
def test_custom_rejects_empty_invalid_or_non_overridable_selectors(tmp_path: Path, selector):
    for coin in ("BTC", "ETH"):
        _write(tmp_path / f"{coin}.json", _single_coin_config(coin))
    with pytest.raises(ValueError, match="--override-params"):
        compose_directory(tmp_path, override_params=selector)


def test_custom_group_only_includes_allowed_leaves(tmp_path: Path):
    for coin in ("BTC", "ETH"):
        _write(tmp_path / f"{coin}.json", _single_coin_config(coin))
    composed, _ = compose_directory(tmp_path, override_params="long.risk,live.leverage")
    for patch in composed["coin_overrides"].values():
        assert set(patch["bot"]["long"]["risk"]) == {
            "entry_cooldown_minutes", "position_exposure_enforcer_enabled",
            "position_exposure_enforcer_threshold", "we_excess_allowance_pct",
        }
        assert patch["live"] == {"leverage": composed["live"]["leverage"]}


def test_custom_strategy_leaf_uses_active_strategy_shorthand(tmp_path: Path):
    for coin in ("BTC", "ETH"):
        config = _single_coin_config(coin)
        config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.03
        _write(tmp_path / f"{coin}.json", config)
    composed, _ = compose_directory(tmp_path, override_params="long.strategy.entry.initial_qty_pct")
    for patch in composed["coin_overrides"].values():
        assert patch == {"bot": {"long": {"strategy": {
            "trailing_martingale": {"entry": {"initial_qty_pct": 0.03}}
        }}}}


def test_custom_hsl_selection_obeys_signal_mode_and_preserves_inactive_value(tmp_path: Path):
    for coin in ("BTC", "ETH"):
        config = _single_coin_config(coin)
        config["bot"]["long"]["hsl"]["red_threshold"] = 0.123
        _write(tmp_path / f"{coin}.json", config)
    composed, _ = compose_directory(tmp_path, override_params="long.hsl.red_threshold")
    assert composed["coin_overrides"]["BTC"] == {
        "bot": {"long": {"hsl": {"red_threshold": 0.123}}}
    }
    for coin in ("BTC", "ETH"):
        config["live"]["approved_coins"] = {"long": [coin], "short": [coin]}
        config["live"]["hsl_signal_mode"] = "unified"
        _write(tmp_path / f"{coin}.json", config)
    with pytest.raises(ValueError, match="matches no overridable input fields"):
        compose_directory(tmp_path, override_params="long.hsl")
