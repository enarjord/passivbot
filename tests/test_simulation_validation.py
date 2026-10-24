"""Empty simulations fail before acquisition; valid override-driven runs still proceed."""

from copy import deepcopy
import json
import socket

import pytest

from config import prepare_config
from config.bot_inheritance import apply_bot_config_patch
from config.schema import get_template_config
from config.universe import materialize_side_universe_sources
from passivbot_cli import main as cli
from passivbot_exceptions import SimulationConfigError
from simulation_validation import validate_simulation_config
from suite_runner import build_scenarios, validate_simulation_scenarios


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("simulation validation attempted network access")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    original_connect = socket.socket.connect

    def connect(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        return denied(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


def config(*, enabled=True, coins=("BTC",)):
    source = get_template_config()
    source["bot"]["long"]["enabled"] = enabled
    source["bot"]["global"]["universe"]["approved_coins"] = list(coins)
    return prepare_config(source, verbose=False)


def test_schema_defaults_and_live_loading_stay_unchanged():
    source = get_template_config()
    original = deepcopy(source)
    prepared = prepare_config(source, live_only=True, verbose=False)
    with pytest.raises(SimulationConfigError) as error:
        validate_simulation_config(prepared)
    assert "Both position sides are disabled" in str(error.value)
    assert "No coins are approved" in str(error.value)
    assert source == original == get_template_config()
    assert not any(prepared["bot"][side]["enabled"] for side in ("long", "short"))


@pytest.mark.parametrize("field", ["max_positions", "total_wallet_exposure_limit"])
def test_zero_capacity_reports_capacity_without_empty_approval_claim(field):
    prepared = config()
    apply_bot_config_patch(prepared, {f"bot.long.risk.{field}": 0})
    with pytest.raises(SimulationConfigError, match="no trading capacity") as error:
        validate_simulation_config(prepared)
    assert "No coins are approved" not in str(error.value)


def test_disabled_side_approvals_do_not_make_enabled_side_runnable():
    prepared = config(coins=())
    apply_bot_config_patch(prepared, {"bot.short.universe.approved_coins": ["BTC"]})
    with pytest.raises(SimulationConfigError, match="No coins are approved") as error:
        validate_simulation_config(prepared)
    assert "disabled" not in str(error.value)


@pytest.mark.parametrize("source", ["all", ["BTC"], "BTC,ETH"])
def test_nonempty_sources_pass_static_validation(source):
    prepared = config()
    apply_bot_config_patch(prepared, {"bot.long.universe.approved_coins": source})
    validate_simulation_config(prepared)


def test_external_coin_sources_are_acquired_only_by_the_resolution_owner(tmp_path):
    coins = tmp_path / "coins.json"
    coins.write_text("[]")
    prepared = config()
    apply_bot_config_patch(prepared, {"bot.long.universe.approved_coins": str(coins)})
    coins.unlink()
    # Preflight must not reread a file or fabricate membership from its path.
    validate_simulation_config(prepared)
    materialize_side_universe_sources(prepared, "approved_coins", {"long": [], "short": []})
    with pytest.raises(SimulationConfigError, match="No approved coins remain"):
        validate_simulation_config(prepared, resolved=True)


def test_resolved_validation_does_not_reread_mutable_sources(tmp_path):
    coins = tmp_path / "coins.json"
    coins.write_text('["BTC"]')
    prepared = config()
    apply_bot_config_patch(prepared, {"bot.long.universe.approved_coins": str(coins)})
    materialize_side_universe_sources(prepared, "approved_coins", {"long": ["BTC"], "short": []})
    coins.unlink()
    validate_simulation_config(prepared, resolved=True)
    materialize_side_universe_sources(prepared, "approved_coins", {"long": [], "short": []})
    with pytest.raises(SimulationConfigError, match="after coin resolution and ignored-coin filtering"):
        validate_simulation_config(prepared, resolved=True)


@pytest.mark.parametrize("command", ["backtest", "optimize"])
@pytest.mark.parametrize("suite", ["n", "y"])
def test_bare_cli_fails_cleanly_without_market_or_candle_requests(command, suite, capsys, monkeypatch):
    import backtest
    import optimize
    import suite_runner

    async def forbidden(*args, **kwargs):
        pytest.fail("invalid simulation reached market or candle preparation")

    for module in (backtest, optimize, suite_runner):
        monkeypatch.setattr(module, "load_markets", forbidden)
    assert cli.main([command, "--suite", suite, "-sd", "2026", "-ed", "2026-10-20"]) == 2
    error = capsys.readouterr().err
    assert f"Cannot start {command}" in error
    assert "Both position sides are disabled" in error
    assert f"{command} configs/examples/btc_long.json" in error
    assert "--bot.long.enabled y -s BTC,ETH" in error
    assert "Traceback" not in error


class ReachedMarkets(BaseException):
    pass


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_cli_only_configuration_reaches_market_preparation(command, monkeypatch):
    import backtest
    import optimize

    async def reached(*args, **kwargs):
        raise ReachedMarkets

    monkeypatch.setattr(backtest if command == "backtest" else optimize, "load_markets", reached)
    with pytest.raises(ReachedMarkets):
        cli.main([command, "--suite", "n", "--bot.long.enabled", "y", "-s", "BTC,ETH"])


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_resolved_empty_cli_stops_before_candles(command, monkeypatch, capsys):
    import backtest
    import optimize

    module = backtest if command == "backtest" else optimize

    async def markets(*args, **kwargs):
        return {}

    async def empty(prepared, *args, **kwargs):
        materialize_side_universe_sources(prepared, "approved_coins", {"long": [], "short": []})

    async def forbidden(*args, **kwargs):
        pytest.fail("empty resolved universe reached candle acquisition")

    monkeypatch.setattr(module, "load_markets", markets)
    monkeypatch.setattr(module, "format_approved_ignored_coins", empty)
    monkeypatch.setattr(module, "prepare_hlcvs_mss", forbidden)
    assert cli.main([command, "--suite", "n", "--bot.long.enabled", "y", "-s", "BTC"]) == 2
    error = capsys.readouterr().err
    assert "after coin resolution and ignored-coin filtering" in error
    assert "No coin data found" not in error


def test_suite_preflight_uses_scenario_coins_and_activation_overrides():
    prepared = config(enabled=False, coins=())
    original = deepcopy(prepared)
    scenarios, _ = build_scenarios({"scenarios": [{
        "label": "enabled_here", "coins": ["BTC"],
        "overrides": {"bot.long.enabled": True},
    }]})
    validate_simulation_scenarios(prepared, scenarios)
    assert prepared == original
    scenarios, _ = build_scenarios({"scenarios": [{
        "label": "empty_here", "overrides": {"bot.long.enabled": True},
    }]})
    with pytest.raises(SimulationConfigError) as error:
        validate_simulation_scenarios(prepared, scenarios)
    assert error.value.context == "empty_here"
    assert "No coins are approved" in str(error.value)


def test_suite_cli_accepts_disabled_empty_base_when_scenario_supplies_inputs(tmp_path, monkeypatch):
    import suite_runner

    source = get_template_config()
    source["backtest"]["scenarios"] = [{
        "label": "enabled_here", "coins": ["BTC"],
        "overrides": {"bot.long.enabled": True},
    }]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source))

    async def reached(*args, **kwargs):
        raise ReachedMarkets

    monkeypatch.setattr(suite_runner, "load_markets", reached)
    with pytest.raises(ReachedMarkets):
        cli.main(["backtest", str(path), "--suite", "y"])


def test_optimizer_preflight_respects_reachable_capacity_and_fixed_overrides():
    from optimization.warmup import validate_optimizer_simulation_config

    source = get_template_config()
    source["bot"]["long"]["enabled"] = True
    source["bot"]["global"]["universe"]["approved_coins"] = ["BTC"]
    source["bot"]["long"]["risk"] = {"total_wallet_exposure_limit": 0.0}
    source["optimize"]["bounds"] = {"long": {"risk": {"total_wallet_exposure_limit": [0.0, 1.0]}}}
    prepared = prepare_config(source, verbose=False)
    validate_optimizer_simulation_config(prepared, suite_cfg={"enabled": False})
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.enabled": False}
    prepared = prepare_config(source, verbose=False)
    with pytest.raises(SimulationConfigError, match="Both position sides are disabled"):
        validate_optimizer_simulation_config(prepared, suite_cfg={"enabled": False})


@pytest.mark.parametrize("exception", [RuntimeError("unexpected defect"), ValueError("unexpected value")])
def test_cli_does_not_hide_unexpected_errors(monkeypatch, exception):
    def broken(*args, **kwargs):
        raise exception

    monkeypatch.setattr(cli, "_invoke_module_main", broken)
    with pytest.raises(type(exception), match=str(exception)):
        cli.main(["backtest"])


def test_resolved_suite_rejects_empty_membership_before_dataset_acquisition(monkeypatch):
    import suite_runner

    prepared = config()
    materialize_side_universe_sources(prepared, "approved_coins", {"long": [], "short": []})
    scenarios, _ = build_scenarios({"scenarios": [{"label": "filtered_out"}]})
    monkeypatch.setattr(suite_runner, "apply_scenario", lambda *a, **kw: (prepared, ["BTC"]))
    with pytest.raises(SimulationConfigError) as error:
        suite_runner._prepare_scenario_universes(
            prepared, scenarios, master_coins=["BTC"], master_ignored=[],
            available_exchanges=["binance"], require_runnable=True,
        )
    assert error.value.context == "filtered_out"
    assert "No approved coins remain" in str(error.value)


def test_optimizer_suite_configuration_error_cleans_up_without_traceback(monkeypatch, capsys):
    import optimize

    cleaned = []

    class Arrays:
        def cleanup(self):
            cleaned.append(True)

    async def markets(*args, **kwargs):
        return {}

    async def resolved(prepared, *args, **kwargs):
        materialize_side_universe_sources(prepared, "approved_coins", {"long": ["BTC"], "short": ["BTC"]})

    async def invalid(*args, **kwargs):
        raise SimulationConfigError("No approved coins remain after scenario filtering.", context="filtered_out")

    monkeypatch.setattr(optimize, "load_markets", markets)
    monkeypatch.setattr(optimize, "format_approved_ignored_coins", resolved)
    monkeypatch.setattr(optimize, "SharedArrayManager", Arrays)
    monkeypatch.setattr(optimize, "prepare_suite_contexts", invalid)
    assert cli.main(["optimize", "--suite", "y", "--scenarios", "base",
                     "--bot.long.enabled", "y", "-s", "BTC"]) == 2
    error = capsys.readouterr().err
    assert "scenario 'filtered_out'" in error
    assert "Traceback" not in error
    assert cleaned == [True]
