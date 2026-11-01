"""Transport selection cannot weaken simulation coverage or offline policy."""
import asyncio
from types import SimpleNamespace

import pytest

import hlcv_preparation as hp
from config import prepare_config
from config.schema import get_template_config
from simulation_data import (
    OfflineDataError, acquisition_mode, is_offline, simulation_data_policy,
)
from session_artifacts import effective_setup_config


@pytest.mark.parametrize("mode", ["auto", "rest"])
def test_canonical_acquisition_mode(mode):
    config = get_template_config()
    config["backtest"]["acquisition_mode"] = mode
    assert prepare_config(config, verbose=False)["backtest"]["acquisition_mode"] == mode


@pytest.mark.parametrize("mode", [None, True, "archive", "REST", 1, []])
def test_bad_acquisition_mode_fails_before_preparation(mode):
    with pytest.raises(ValueError, match="backtest.acquisition_mode"):
        with simulation_data_policy({"backtest": {"acquisition_mode": mode}}):
            pytest.fail("invalid policy entered")


@pytest.mark.asyncio
async def test_acquisition_mode_is_task_local_and_resets_on_failure():
    async def selected(mode):
        with simulation_data_policy({"backtest": {"acquisition_mode": mode}}):
            await asyncio.sleep(0)
            return acquisition_mode()
    assert await asyncio.gather(selected("rest"), selected("auto")) == ["rest", "auto"]
    assert acquisition_mode() == "auto"
    with pytest.raises(RuntimeError):
        with simulation_data_policy({"backtest": {"acquisition_mode": "rest"}}):
            raise RuntimeError("fixture")
    assert acquisition_mode() == "auto"


def test_rest_transport_disables_both_archive_owners(monkeypatch):
    seen = []
    monkeypatch.setattr(hp, "CandlestickManager", lambda **kw: seen.append(kw) or SimpleNamespace())
    with simulation_data_policy({"backtest": {"acquisition_mode": "rest"}}):
        manager = hp.HLCVManager("binance", cc=SimpleNamespace())
        manager.load_cc()
        assert seen[0]["archive_enabled"] is False
        assert hp._binance_archives_enabled(manager, "binance") is False
        with pytest.raises(ValueError, match="archives are disabled"):
            manager.get_binance_archive_client()
    assert manager.acquisition_mode == "rest"
    assert hp._binance_archives_enabled(manager, "binance") is False


@pytest.mark.asyncio
async def test_rest_transport_uses_strict_owner_fetch(monkeypatch):
    calls = []
    async def ccxt_fetch(**kwargs):
        calls.append(kwargs)
        return hp.V2FetchResult(False, "leading_unavailable")
    monkeypatch.setattr(hp, "_fetch_coin_range_via_ccxt_into_v2_store", ccxt_fetch)
    manager = SimpleNamespace(get_binance_archive_client=lambda: pytest.fail("archive requested"))
    with simulation_data_policy({"backtest": {"acquisition_mode": "rest"}}):
        result = await hp._fetch_coin_range_into_v2_store(
            om=manager, catalog=None, store=None, exchange="binance", coin="BTC",
            symbol="BTC/USDT:USDT", start_ts=1, end_ts=10,
        )
    assert not result.ok
    assert result.reason == "leading_unavailable"
    assert calls[0]["allow_pre_inception_prefix"] is False


@pytest.mark.parametrize("mode", ["auto", "rest"])
def test_offline_policy_cannot_be_bypassed_by_transport(mode):
    with simulation_data_policy({"backtest": {"offline": True, "acquisition_mode": mode}}):
        assert is_offline()
        with simulation_data_policy({"backtest": {"offline": False, "acquisition_mode": mode}}):
            with pytest.raises(OfflineDataError):
                hp.HLCVManager("binance").get_binance_archive_client()
    assert not is_offline()
    assert acquisition_mode() == "auto"


def test_transport_selector_does_not_change_semantic_setup_identity():
    config = prepare_config(get_template_config(), verbose=False)
    before = effective_setup_config(config, optimize=True)
    config["backtest"]["acquisition_mode"] = "rest"
    assert effective_setup_config(config, optimize=True) == before


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_simulation_cli_acquisition_mode(command):
    import argparse
    from config_utils import (
        add_config_arguments, project_template_config_for_cli, update_config_with_args,
    )
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser, project_template_config_for_cli(get_template_config(), command), command=command,
    )
    args = parser.parse_args(["--acquisition-mode", "rest"])
    config = get_template_config()
    update_config_with_args(config, args, allowed_keys=allowed)
    assert config["backtest"]["acquisition_mode"] == "rest"


def test_acquisition_selector_not_exposed_on_live_cli():
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, project_template_config_for_cli(get_template_config(), "live"), command="live")
    with pytest.raises(SystemExit):
        parser.parse_args(["--acquisition-mode", "rest"])


def test_transport_does_not_change_cache_or_evaluation_identity():
    from backtest import get_cache_hash
    from optimization.evaluation_contract import build_evaluation_contract
    config = get_template_config()
    config["backtest"].update(start_date="2024-01-01", end_date="2024-01-10")
    config = prepare_config(config, verbose=False)
    cache_hash = get_cache_hash(config, "binance")
    contract = build_evaluation_contract(config)
    config["backtest"]["acquisition_mode"] = "rest"
    assert get_cache_hash(config, "binance") == cache_hash
    assert build_evaluation_contract(config) == contract
