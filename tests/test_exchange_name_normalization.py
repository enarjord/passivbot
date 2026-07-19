from types import SimpleNamespace

import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

import utils
from exchanges.ccxt_bot import CCXTBot
from exchanges.gateio import GateIOBot
from hlcv_preparation import HLCVManager


def test_gateio_session_boundary_resolves_current_rest_and_pro_clients(monkeypatch):
    resolved_ids = []

    def create_sessions(bot):
        resolved_ids.append(bot.exchange_ccxt_id)
        bot.cca = SimpleNamespace(headers={})
        bot.ccp = SimpleNamespace(headers={})

    monkeypatch.setattr(CCXTBot, "create_ccxt_sessions", create_sessions)
    bot = GateIOBot.__new__(GateIOBot)
    bot.exchange_ccxt_id = "gateio"
    bot.broker_code = None

    bot.create_ccxt_sessions()

    assert resolved_ids == ["gate"]
    assert bot.exchange_ccxt_id == "gateio"
    assert getattr(ccxt_async, resolved_ids[0]).__name__ == "gate"
    assert getattr(ccxt_pro, resolved_ids[0]).__name__ == "gate"
    assert utils.to_ccxt_exchange_id("gateio") == "gateio"
    assert utils.to_standard_exchange_name("gateio") == "gateio"


def test_gateio_hlcv_identity_keeps_cache_and_fee_overrides():
    manager = HLCVManager("gateio", start_date="2026-01-01", end_date="2026-01-02")
    manager.markets = {
        "BTC/USDT:USDT": {
            "base": "BTC",
            "maker": 0.001,
            "taker": 0.001,
            "contractSize": 1.0,
            "limits": {"cost": {"min": 1.0}, "amount": {"min": 0.001}},
            "precision": {"price": 0.1, "amount": 0.001},
        }
    }

    settings = manager.get_market_specific_settings("BTC")

    assert manager.exchange == "gateio"
    assert manager.cache_filepaths["markets"] == "caches/gateio/markets.json"
    assert settings["maker_fee"] == 0.0002
    assert settings["taker_fee"] == 0.0005
