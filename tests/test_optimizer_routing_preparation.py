"""Optimizer source capture follows required routing-metadata preparation."""

import json

import pytest

from config.bot_inheritance import bot_policy_source_view
from config.inherited_bounds import get_bound_plan
from config.universe import runtime_universe_snapshot, universe_snapshot_signature
from test_hsl_offline_runtime import offline_cli_config


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize("source_exchange", [None, "bybit"])
async def test_optimizer_freezes_same_routing_facts_on_cold_and_warm_preparation(
    tmp_path, monkeypatch, suite, source_exchange,
):
    import optimize
    import utils

    config = offline_cli_config(tmp_path, monkeypatch, "portfolio")
    expected_exchanges = ["binance"]
    if source_exchange is not None:
        metadata = tmp_path / "caches" / source_exchange / "markets.json"
        metadata.parent.mkdir()
        metadata.write_bytes((tmp_path / "caches/binance/markets.json").read_bytes())
        config["backtest"]["coin_sources"] = {"BTC": source_exchange}
        expected_exchanges.append(source_exchange)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    assert not (tmp_path / "caches/binance/coin_to_symbol_map.json").exists()
    load_markets = utils.load_markets
    captured = []

    class Captured(BaseException):
        pass

    async def prepare_metadata(exchange, **kwargs):
        assert kwargs["offline"] is True
        assert exchange in expected_exchanges
        return await load_markets(exchange, **kwargs)

    def capture(current):
        # Observe the real formatter and binding refresh at their downstream
        # consumer boundary, before any candle work starts.
        captured.append((universe_snapshot_signature(runtime_universe_snapshot(
            current, available_coins=["BTC"], available_exchanges=("binance",),
        )), get_bound_plan(current).identity(), bot_policy_source_view(current)["bot"]))
        raise Captured

    monkeypatch.setattr(optimize, "load_markets", prepare_metadata, raising=False)
    monkeypatch.setattr(optimize, "build_optimizer_data_config", capture)
    monkeypatch.setattr(optimize.sys, "argv", [
        "optimize", str(path), "--suite", "y" if suite else "n",
    ])
    for attempt in range(2):
        with pytest.raises(Captured):
            await optimize.main()
        if attempt == 0:
            # Model the required data owner's metadata acquisition between an
            # initially cold run and a new preparation of the same source.
            for exchange in expected_exchanges:
                await load_markets(exchange, offline=True)
    assert captured[0] == captured[1]
    assert captured[0][0][0]["available_members"] == ["binance::BTC/USDT:USDT"]
    assert path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
async def test_warm_alias_cache_does_not_bypass_required_offline_metadata(tmp_path, monkeypatch, suite):
    import optimize
    import utils
    from simulation_data import OfflineDataError

    config = offline_cli_config(tmp_path, monkeypatch, "portfolio")
    await utils.load_markets("binance", offline=True)
    (tmp_path / "caches/binance/markets.json").unlink()
    assert (tmp_path / "caches/binance/coin_to_symbol_map.json").exists()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()

    async def forbidden(*_args, **_kwargs):
        pytest.fail("source capture must follow required metadata acquisition")

    monkeypatch.setattr(optimize, "format_approved_ignored_coins", forbidden)
    monkeypatch.setattr(optimize.sys, "argv", [
        "optimize", str(path), "--suite", "y" if suite else "n",
    ])
    with pytest.raises(OfflineDataError, match="market metadata missing or invalid.*markets.json"):
        await optimize.main()
    assert path.read_bytes() == original
