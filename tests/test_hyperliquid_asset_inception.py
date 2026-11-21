"""Numeric CCXT Hyperliquid IDs retain exact reviewed asset-age bindings."""

from copy import deepcopy
import socket

import ccxt
import pytest

import asset_inception as age
from asset_inception_catalog import resolve_asset_identity
from simulation_data import simulation_data_policy


ASSETS = [("BTC", 0, 1), ("ETH", 1, 1027), ("SOL", 5, 5426), ("HYPE", 159, 32196)]


def hyperliquid_market(name="HYPE", index=159):
    """Parse synthetic public metadata with the installed, pinned CCXT adapter."""
    return ccxt.hyperliquid().parse_market({
        "name": name, "baseId": index, "szDecimals": "4",
        "maxLeverage": "10", "markPx": "100",
    })


@pytest.mark.parametrize("name,index,cmc_id", ASSETS)
@pytest.mark.parametrize("index_type", [int, str])
def test_numeric_hyperliquid_market_resolves_reviewed_asset(name, index, cmc_id, index_type):
    """Use the declared instrument, never a hardcoded universe-index table."""
    market = hyperliquid_market(name, index_type(index))
    before = deepcopy(market)

    identity = resolve_asset_identity("hyperliquid", market)

    assert identity is not None
    assert identity.cmc_id == cmc_id
    assert market == before


@pytest.mark.parametrize("name,index,cmc_id", ASSETS)
def test_named_hyperliquid_market_remains_supported(name, index, cmc_id):
    """Older named-ID metadata keeps its existing exact binding."""
    market = hyperliquid_market(name, index)
    market.update(id=name, baseId=name)
    assert resolve_asset_identity("hyperliquid", market).cmc_id == cmc_id


@pytest.mark.parametrize("patch", [
    {"baseId": "160"},
    {"baseId": 159},
    {"info": None},
    {"info": []},
    {"info": {}},
    {"info": {"name": "HYPE", "baseId": 160}},
    {"info": {"name": "HYPE", "baseId": True}},
    {"info": {"name": "HYPE", "baseId": 159.0}},
    {"info": {"name": "HYPE"}},
    {"info": {"name": None, "baseId": 159}},
    {"info": {"name": "BTC", "baseId": 159}},
    {"info": {"name": "xyz:HYPE", "baseId": 159}},
    {"info": {"name": "HYPE", "baseId": 159, "isRwa": True}},
    {"info": {"name": "HYPE", "baseId": 159, "isPreListing": True}},
    {"info": {"name": "HYPE", "baseId": 159, "is_pre_market": True}},
    {"base": "HYPER"},
    {"symbol": "HYPER/USDC:USDC"},
    {"quote": "USDT"},
    {"settle": "USDT"},
    {"swap": False},
    {"id": "１５９", "baseId": "１５９"},
    {"id": "159.0", "baseId": "159.0"},
    {"id": 159},
])
def test_numeric_binding_rejects_inconsistent_or_non_crypto_metadata(patch):
    """An index or ticker alone cannot bypass the reviewed instrument checks."""
    market = hyperliquid_market()
    market.update(patch)
    assert resolve_asset_identity("hyperliquid", market) is None


@pytest.mark.parametrize("name", ["NOT_IN_REVIEWED_CATALOG", "xyz:TSLA", "xyz:HYPE"])
def test_unknown_or_builder_market_cannot_inherit_crypto_age(name):
    """Builder names remain qualified and unknown assets remain unresolved."""
    assert resolve_asset_identity("hyperliquid", hyperliquid_market(name)) is None


@pytest.mark.parametrize("exchange", ["bybit", "okx", "binance"])
def test_numeric_name_resolution_is_hyperliquid_only(exchange):
    """Other exchange IDs cannot opt into Hyperliquid's metadata interpretation."""
    assert resolve_asset_identity(exchange, hyperliquid_market()) is None


@pytest.fixture
def offline_age(tmp_path, monkeypatch):
    """Isolate all evidence files and fail on any provider or transport access."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(age.time, "time", lambda: 1_800_000_000)

    def unexpected_request(*args, **kwargs):
        pytest.fail("asset-age resolution attempted a network request")

    monkeypatch.setattr(socket.socket, "connect", unexpected_request)
    monkeypatch.setattr(age, "_resolve_cmc", unexpected_request)
    monkeypatch.setattr(age, "_resolve_venue", unexpected_request)
    return tmp_path / "cache"


@pytest.mark.asyncio
async def test_offline_numeric_markets_share_reviewed_cross_venue_evidence(offline_age):
    """Resolve several numeric markets without copying candles or fetching prices."""
    markets = {name: [("hyperliquid", hyperliquid_market(name, index))]
               for name, index, _ in ASSETS}
    markets["HYPE"].append(("bybit", {
        "id": "HYPEUSDT", "base": "HYPE", "symbol": "HYPE/USDT:USDT",
        "quote": "USDT", "settle": "USDT", "swap": True,
    }))
    with simulation_data_policy({"backtest": {"offline": True}}):
        facts = await age.resolve_asset_inception(markets, cache_dir=offline_age)

    assert set(facts) == {name for name, _, _ in ASSETS}
    for name, index, cmc_id in ASSETS:
        identity = resolve_asset_identity("hyperliquid", hyperliquid_market(name, index))
        assert facts[name].asset_id == f"coinmarketcap:{cmc_id}"
        assert facts[name].timestamp_ms == identity.observation_ms
        assert not facts[name].is_provisional


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange", ["hyperliquid", "combined"])
async def test_prepared_unknown_age_recovers_from_numeric_market(offline_age, monkeypatch, exchange):
    """The actual preparation consumer retries unknown age and preserves 60-day policy."""
    import backtest as bt
    from config.schema import get_expanded_config_prototype

    config = get_expanded_config_prototype()
    config["backtest"].update(exchanges=["bybit", "hyperliquid"], offline=True)
    config["bot"]["long"]["enabled"] = True
    config["bot"]["long"]["universe"].update(
        approved_coins=["HYPE"], ignored_coins=[], minimum_coin_age_days=60.0,
    )
    settings = hyperliquid_market()
    settings.update(exchange="hyperliquid", inception_timestamp_ms=None,
                    asset_inception_policy_version=bt.ASSET_INCEPTION_POLICY_VERSION)
    original = deepcopy(settings)
    # Use the real resolver, with only its cache location redirected to the fixture.
    async def resolve(markets, *, allow_remote):
        assert not allow_remote
        return await age.resolve_asset_inception(markets, allow_remote=False, cache_dir=offline_age)

    monkeypatch.setattr(bt, "resolve_asset_inception", resolve)
    with simulation_data_policy(config):
        resolved = await bt._resolve_prepared_inception(config, exchange, ["HYPE"], {"HYPE": settings})

    assert resolved["HYPE"]["inception_timestamp_ms"] == 1_732_866_300_000
    assert resolved["HYPE"]["asset_inception_asset_id"] == "coinmarketcap:32196"
    assert not resolved["HYPE"]["asset_inception_provisional"]
    assert config["bot"]["long"]["universe"]["minimum_coin_age_days"] == 60.0
    assert settings == original
