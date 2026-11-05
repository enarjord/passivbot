from types import SimpleNamespace

import pytest

from exchanges.okx import OKXBot


@pytest.mark.parametrize("host", [
    "ws.okx.com", "wspap.okx.com", "wseea.okx.com", "wseeapap.okx.com",
    "wsus.okx.com", "wsuspap.okx.com",
])
def test_okx_updates_official_retired_websocket_port(host):
    bot = OKXBot.__new__(OKXBot)
    client = SimpleNamespace(options={}, urls={
        "api": {"ws": f"wss://{host}:8443/ws/v5", "rest": "https://rest.invalid"},
        "test": {"ws": f"wss://{host}:8443/ws/v5?test=1"},
    })
    bot._apply_exchange_market_options(client, "okx")
    assert client.urls["api"]["ws"] == f"wss://{host}:443/ws/v5"
    assert client.urls["test"]["ws"] == f"wss://{host}:443/ws/v5?test=1"
    assert client.urls["api"]["rest"] == "https://rest.invalid"
    assert client.options["fetchMarkets"] == {"types": ["swap"]}


@pytest.mark.parametrize("url", [
    "wss://proxy.invalid:8443/ws/v5", "wss://ws.okx.com:443/ws/v5",
    "wss://ws.okx.com/ws/v5", "wss://ws.okx.com.evil.invalid:8443/ws/v5",
])
def test_okx_preserves_unrelated_or_current_endpoints(url):
    bot = OKXBot.__new__(OKXBot)
    client = SimpleNamespace(options={}, urls={"api": {"ws": url}})
    bot._apply_exchange_market_options(client, "okx")
    assert client.urls["api"]["ws"] == url


@pytest.mark.asyncio
async def test_created_okx_clients_use_current_port_before_endpoint_override():
    bot = OKXBot.__new__(OKXBot)
    bot.exchange = "okx"
    bot.ws_enabled = True
    bot.user_info = {"options": {"brokerId": "test-broker"}}
    bot._build_ccxt_config = lambda: {}
    bot._build_ccxt_options = lambda: {}
    bot._apply_endpoint_override = lambda client: None
    bot.create_ccxt_sessions()
    try:
        assert ":443/" in bot.ccp.get_url("users", "private")
        assert ":443/" in bot.ccp.get_url("candle1m", "public")
        assert bot.ccp.options["brokerId"] == "test-broker"
        # The standard callback applies after exchange defaults. An operator's
        # deliberate proxy endpoint remains authoritative.
        def proxy(client):
            client.urls["api"]["ws"] = "wss://proxy.invalid:8443/custom"
        bot._apply_endpoint_override = proxy
        other = OKXBot.__new__(OKXBot)
        other.__dict__.update(bot.__dict__)
        other.create_ccxt_sessions()
        try:
            assert other.ccp.urls["api"]["ws"] == "wss://proxy.invalid:8443/custom"
        finally:
            await other.cca.close()
            await other.ccp.close()
    finally:
        await bot.cca.close()
        await bot.ccp.close()
