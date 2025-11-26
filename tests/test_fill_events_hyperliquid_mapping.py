from types import SimpleNamespace

from fill_events_manager import _build_fetcher_for_bot, HyperliquidFetcher


def test_build_fetcher_hyperliquid():
    bot = SimpleNamespace()
    bot.exchange = "hyperliquid"
    bot.cca = "dummy"
    bot.user = "u"
    bot.markets_dict = {}
    bot.coin_to_symbol = lambda x, verbose=False: x
    fetcher = _build_fetcher_for_bot(bot, symbols=["BTC"])
    assert isinstance(fetcher, HyperliquidFetcher)
