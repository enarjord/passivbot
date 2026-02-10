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


def _make_bot(HyperliquidBot, broker_code=None):
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.broker_code = broker_code or {}
    bot._builder_approval_last_check_ms = 0
    bot._builder_initialized = False
    bot._builder_pending_approval = False
    bot._builder_disabled_ts = None
    bot._builder_thank_you_printed = False
    bot._builder_settings = {}
    bot.user_info = {"wallet_address": "0xabc", "is_vault": False}
    bot.cca = None
    bot.ccp = None
    return bot


class DummyCCA:
    def __init__(self, response=None, approve_raises=None):
        self.options = {}
        self.response = response
        self.approve_raises = approve_raises
        self.approve_called = False

    async def fetch(self, *args, **kwargs):
        return self.response

    async def approve_builder_fee(self, builder, fee_rate):
        self.approve_called = True
        if self.approve_raises:
            raise self.approve_raises
        return {"status": "ok"}


def test_apply_builder_code_options_dict(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA()
    bot.ccp = DummyCCA()
    bot._apply_builder_code_options()

    assert bot.cca.options["ref"] == "PASSIVBOT"
    assert bot.cca.options["builder"] == "0x123"
    assert bot.cca.options["feeRate"] == "0.01%"
    assert bot.cca.options["feeInt"] == 10
    assert bot.cca.options["builderFee"] is False
    assert bot.cca.options["approvedBuilderFee"] is False
    assert bot.ccp.options["builder"] == "0x123"
    assert bot._has_builder_config() is True


def test_apply_builder_code_options_string(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot, broker_code="PASSIVBOT")
    bot.cca = DummyCCA()
    bot._apply_builder_code_options()

    assert bot.cca.options["ref"] == "PASSIVBOT"
    assert "builder" not in bot.cca.options
    assert bot._has_builder_config() is False


def test_apply_builder_code_options_invalid_fee_int(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "builder": "0x123",
            "feeInt": "abc",
        },
    )
    bot.cca = DummyCCA()
    bot._apply_builder_code_options()
    assert bot.cca.options["feeInt"] == 10


def test_apply_builder_code_options_disabled_builder_fee(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "builderFee": False,
        },
    )
    bot.cca = DummyCCA()
    bot._apply_builder_code_options()

    assert bot.cca.options["ref"] == "PASSIVBOT"
    assert "builder" not in bot.cca.options
    assert "builderFee" not in bot.cca.options
    assert bot._has_builder_config() is False


def test_apply_builder_code_options_disabled_builder_fee_string(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "builder": "0x123",
            "builder_fee": "false",
        },
    )
    bot.cca = DummyCCA()
    bot._apply_builder_code_options()

    assert "builder" not in bot.cca.options
    assert bot._has_builder_config() is False


def test_normalize_builder_settings_snake_case_aliases(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "fee_rate": "0.02%",
            "fee_int": 20,
            "builder_fee": "true",
        },
    )
    settings = bot._normalize_builder_settings()
    assert settings["feeRate"] == "0.02%"
    assert settings["feeInt"] == 20
    assert settings["builderFee"] is True


def test_extract_and_parse_builder_fee_values(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)

    assert bot._extract_max_builder_fee({"data": {"maxBuilderFee": "0.01%"}}) == "0.01%"
    assert bot._extract_max_builder_fee({"response": {"maxFeeRate": "0.02%"}}) == "0.02%"
    assert bot._extract_max_builder_fee([None, {"maxBuilderFee": "0.05%"}]) == "0.05%"
    assert bot._extract_max_builder_fee({}) is None
    assert bot._extract_max_builder_fee(None) is None

    assert bot._is_positive_builder_fee_value("0.01%") is True
    assert bot._is_positive_builder_fee_value("0") is False
    assert bot._is_positive_builder_fee_value("0%") is False
    assert bot._is_positive_builder_fee_value(0) is False


def test_is_builder_fee_error(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot)
    assert bot._is_builder_fee_error(Exception("Builder fee has not been approved")) is True
    assert bot._is_builder_fee_error(Exception("BUILDER FEE HAS NOT BEEN APPROVED")) is True
    assert bot._is_builder_fee_error(Exception("Order must have minimum value of $10")) is False


@pytest.mark.asyncio
async def test_path_a_already_approved(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0.01%")
    bot._apply_builder_code_options()
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot.cca.approve_called is False
    assert bot._builder_pending_approval is False
    assert bot._builder_thank_you_printed is True


@pytest.mark.asyncio
async def test_path_b_main_wallet_auto_approve(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0")
    bot._apply_builder_code_options()
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot.cca.approve_called is True
    assert bot._builder_pending_approval is False
    assert bot._builder_thank_you_printed is True


@pytest.mark.asyncio
async def test_path_c_agent_wallet_pending(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0", approve_raises=Exception("only main wallet can approve"))
    bot._apply_builder_code_options()
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot.cca.approve_called is True
    assert bot._builder_pending_approval is True


@pytest.mark.asyncio
async def test_maybe_reenable_builder_codes_not_approved(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0")
    bot._apply_builder_code_options()
    bot._builder_pending_approval = True
    bot._builder_disabled_ts = 0
    bot.BUILDER_NAG_INTERVAL_MS = 0
    await bot._maybe_reenable_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot._builder_pending_approval is True
    assert bot._builder_disabled_ts is None


@pytest.mark.asyncio
async def test_maybe_reenable_builder_codes_approved(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "ref": "PASSIVBOT",
            "builder": "0x123",
            "feeRate": "0.01%",
            "feeInt": 10,
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0.01%")
    bot._apply_builder_code_options()
    bot._builder_pending_approval = True
    bot._builder_disabled_ts = 0
    bot.BUILDER_NAG_INTERVAL_MS = 0
    await bot._maybe_reenable_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot._builder_pending_approval is False
    assert bot._builder_disabled_ts is None


@pytest.mark.asyncio
async def test_check_builder_fee_approved_cache(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(
        HyperliquidBot,
        broker_code={
            "builder": "0x123",
            "builderFee": True,
        },
    )
    bot.cca = DummyCCA(response="0")
    bot._apply_builder_code_options()
    result_1 = await bot._check_builder_fee_approved()
    assert result_1 is False
    bot.cca.options["approvedBuilderFee"] = True
    result_2 = await bot._check_builder_fee_approved()
    assert result_2 is True


@pytest.mark.asyncio
async def test_no_builder_config_skips_init(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HyperliquidBot, broker_code="PASSIVBOT")
    bot.cca = DummyCCA()
    bot._apply_builder_code_options()
    await bot._init_builder_codes()
    assert bot.cca.approve_called is False
    assert bot._builder_pending_approval is False
