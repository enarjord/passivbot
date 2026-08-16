from __future__ import annotations

import json
from pathlib import Path

import pytest
import tools.compose_coin_overrides as compose_tool

from config.load import prepare_config
from config.overrides import parse_overrides
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


def test_rejects_duplicate_coin_and_master_outside_directory(tmp_path: Path):
    _write(tmp_path / "a.json", _single_coin_config("BTC"))
    _write(tmp_path / "b.json", _single_coin_config("BTC"))

    with pytest.raises(ValueError, match="duplicate single-coin config for BTC"):
        compose_directory(tmp_path)
    with pytest.raises(ValueError, match="selected master config is not"):
        compose_directory(tmp_path, master_config=tmp_path / "missing.json")


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


def test_rejects_retaining_gpu_optimizer_for_composed_config(tmp_path: Path):
    master = _single_coin_config("BTC")
    master["optimize"]["backend"] = "gpu"
    _write(tmp_path / "a.json", master)
    _write(tmp_path / "b.json", _single_coin_config("ETH"))

    lean, _report = compose_directory(tmp_path)
    assert "optimize" not in lean
    with pytest.raises(ValueError, match="optimize.backend='gpu'"):
        compose_directory(tmp_path, include_backtest_optimize=True)


def test_cli_writes_sorted_config_and_protects_existing_output(tmp_path: Path, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs / "a.json", _single_coin_config("BTC"))
    _write(inputs / "b.json", _single_coin_config("ETH"))
    output = tmp_path / "composed.json"

    assert main([str(inputs), str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == sorted(payload)
    assert "Master source (alphabetically first)" in capsys.readouterr().out

    assert main([str(inputs), str(output)]) == 2
    assert "pass --overwrite" in capsys.readouterr().err
