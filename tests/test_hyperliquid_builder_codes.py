import importlib
import sys
import types

import pytest


@pytest.fixture
def stubbed_modules(monkeypatch):
    pr_module = types.ModuleType("passivbot_rust")
    pr_module.round_ = lambda x, step: x
    pr_module.round_dynamic = lambda x, y=None: x
    pr_module.round_dynamic_up = lambda x, y=None: x
    pr_module.round_dynamic_dn = lambda x, y=None: x
    pr_module.__getattr__ = lambda name: (lambda *args, **kwargs: 0)
    monkeypatch.setitem(sys.modules, "passivbot_rust", pr_module)

    errors_module = types.ModuleType("ccxt.base.errors")
    errors_module.NetworkError = Exception
    errors_module.RateLimitExceeded = Exception
    monkeypatch.setitem(sys.modules, "ccxt.base.errors", errors_module)

    ccxt_base = types.ModuleType("ccxt.base")
    ccxt_base.errors = errors_module
    monkeypatch.setitem(sys.modules, "ccxt.base", ccxt_base)

    ccxt_async = types.ModuleType("ccxt.async_support")
    ccxt_async.hyperliquid = None
    monkeypatch.setitem(sys.modules, "ccxt.async_support", ccxt_async)

    ccxt_pro = types.ModuleType("ccxt.pro")
    ccxt_pro.hyperliquid = None
    monkeypatch.setitem(sys.modules, "ccxt.pro", ccxt_pro)

    ccxt_module = types.ModuleType("ccxt")
    ccxt_module.__version__ = "4.5.22"
    ccxt_module.base = ccxt_base
    ccxt_module.async_support = ccxt_async
    ccxt_module.pro = ccxt_pro
    monkeypatch.setitem(sys.modules, "ccxt", ccxt_module)

    proc_module = types.ModuleType("procedures")
    proc_module.assert_correct_ccxt_version = lambda *args, **kwargs: None
    proc_module.print_async_exception = lambda *args, **kwargs: None
    proc_module.load_broker_code = lambda *args, **kwargs: {}
    proc_module.load_user_info = lambda *args, **kwargs: {"exchange": "hyperliquid"}
    proc_module.get_first_timestamps_unified = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "procedures", proc_module)

    yield

    if "exchanges.hyperliquid" in sys.modules:
        sys.modules.pop("exchanges.hyperliquid", None)


def _make_bot(HyperliquidBot):
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot._builder_settings = {}
    bot._builder_approval_last_check_ms = 0
    bot._builder_approval_is_active = False
    bot.user_info = {"wallet_address": "0xabc"}
    bot.ccp = None
    return bot


class DummyCCA:
    def __init__(self, response):
        self.options = {}
        self.response = response

    async def fetch(self, *args, **kwargs):
        return self.response


def test_normalize_builder_settings(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)
    bot.broker_code = {
        "ref": "PASSIVBOT",
        "builder": "0x123",
        "feeRate": "0.01%",
        "feeInt": 10,
        "builderFee": True,
    }

    settings = bot._normalize_builder_settings()
    assert settings["ref"] == "PASSIVBOT"
    assert settings["builder"] == "0x123"
    assert settings["feeRate"] == "0.01%"
    assert settings["feeInt"] == 10
    assert settings["builderFee"] is True
    assert bot._builder_feature_enabled() is True


def test_extract_and_parse_builder_fee_values(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)

    extracted = bot._extract_max_builder_fee({"data": {"maxBuilderFee": "0.01%"}})
    assert extracted == "0.01%"
    assert bot._is_positive_builder_fee_value(extracted) is True
    assert bot._is_positive_builder_fee_value("0") is False
    assert bot._is_positive_builder_fee_value("0%") is False
    assert bot._is_positive_builder_fee_value(0) is False


@pytest.mark.asyncio
async def test_builder_approval_active_when_ccxt_flag_set(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)
    bot._builder_settings = {"builder": "0x123", "builderFee": True}
    bot.cca = DummyCCA(response={"maxBuilderFee": "0"})
    bot.cca.options["approvedBuilderFee"] = True

    assert await bot._is_builder_approval_active() is True


@pytest.mark.asyncio
async def test_builder_approval_detected_from_info_endpoint(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)
    bot._builder_settings = {"builder": "0x123", "builderFee": True}
    bot.cca = DummyCCA(response={"maxBuilderFee": "0.01%"})

    assert await bot._is_builder_approval_active() is True
    assert bot.cca.options["approvedBuilderFee"] is True

