import pytest

from exchanges.hyperliquid import HyperliquidBot


@pytest.mark.asyncio
async def test_fetch_tickers_for_symbols_uses_dex_inference_for_hip3():
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {
            "base": "XYZ-SP500",
            "baseName": "xyz:SP500",
            "info": {"name": "xyz:SP500"},
        },
        "BTC/USDC:USDC": {"base": "BTC", "info": {}},
    }
    bot._get_hl_dex_for_symbol = lambda symbol: (
        "xyz" if symbol.startswith("XYZ-") else None
    )

    async def fake_fetch_tickers():
        return {"BTC/USDC:USDC": {"last": 100_000.0, "bid": 99_999.0, "ask": 100_001.0}}

    async def fake_fetch_hip3_tickers_for_symbols(dex, symbols):
        assert dex == "xyz"
        assert symbols == ["XYZ-SP500/USDC:USDC"]
        return {
            "XYZ-SP500/USDC:USDC": {"last": 7_171.2, "bid": 7_171.2, "ask": 7_171.2}
        }

    bot.fetch_tickers = fake_fetch_tickers
    bot._fetch_hip3_tickers_for_symbols = fake_fetch_hip3_tickers_for_symbols

    out = await bot.fetch_tickers_for_symbols(["XYZ-SP500/USDC:USDC", "BTC/USDC:USDC"])

    assert out["XYZ-SP500/USDC:USDC"]["last"] == pytest.approx(7_171.2)
    assert out["BTC/USDC:USDC"]["last"] == pytest.approx(100_000.0)


@pytest.mark.asyncio
async def test_fetch_hip3_tickers_matches_asset_suffix_names():
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {
            "base": "XYZ-SP500",
            "baseName": "xyz:SP500",
            "info": {"name": "xyz:SP500"},
        }
    }

    class FakeCCA:
        async def fetch(self, url, method=None, headers=None, body=None):
            return [
                {"universe": [{"name": "SP500"}]},
                [{"midPx": "7171.2", "markPx": "7171.4"}],
            ]

    bot.cca = FakeCCA()
    bot._hl_info_url = lambda: "https://example.invalid/info"

    out = await bot._fetch_hip3_tickers_for_symbols("xyz", ["XYZ-SP500/USDC:USDC"])

    assert out["XYZ-SP500/USDC:USDC"]["last"] == pytest.approx(7171.2)
