"""Tests for Hyperliquid builder code support.

Follows the same minimal-stubbing pattern as test_hyperliquid_balance_cache.py:
only stub passivbot_rust (handled by conftest), ccxt, and procedures.
Real modules import normally via the src/ path set by conftest.py.
"""

import importlib
import sys
import types

import pytest


@pytest.fixture
def stubbed_modules(monkeypatch):
    """Stub ccxt and procedures so exchanges.hyperliquid can be imported."""
    # ccxt stubs (prevent real exchange connections)
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

    # Stub procedures to bypass ccxt version assertion during import
    proc_module = types.ModuleType("procedures")
    proc_module.assert_correct_ccxt_version = lambda *args, **kwargs: None
    proc_module.print_async_exception = lambda *args, **kwargs: None
    proc_module.load_broker_code = lambda *args, **kwargs: {}
    proc_module.load_user_info = lambda *args, **kwargs: {"exchange": "hyperliquid"}
    proc_module.get_first_timestamps_unified = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "procedures", proc_module)

    yield

    # Cleanup: remove cached hyperliquid module so each test gets a fresh import
    if "exchanges.hyperliquid" in sys.modules:
        sys.modules.pop("exchanges.hyperliquid", None)


def _make_bot(HLBot, broker_code=None):
    """Create a minimal HyperliquidBot instance without full __init__."""
    bot = HLBot.__new__(HLBot)
    bot.broker_code = broker_code or {}
    bot.user_info = {"wallet_address": "0xabc123", "is_vault": False}
    bot._builder_approval_last_check_ms = 0
    bot._builder_initialized = False
    bot._builder_pending_approval = False
    bot._builder_disabled_ts = None
    bot.cca = None
    bot.ccp = None
    return bot


class DummyClient:
    """Mock CCXT client for testing."""

    def __init__(self, fetch_response=None, approve_raises=None):
        self.options = {}
        self._fetch_response = fetch_response
        self._approve_raises = approve_raises
        self._approve_called = False

    async def fetch(self, *args, **kwargs):
        return self._fetch_response

    async def approve_builder_fee(self, builder, fee_rate):
        self._approve_called = True
        if self._approve_raises:
            raise self._approve_raises
        return {"status": "ok"}


# ─── Static / pure method tests ───


def test_is_positive_fee_value(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    assert HLBot._is_positive_fee_value("0.01%") is True
    assert HLBot._is_positive_fee_value("0.02%") is True
    assert HLBot._is_positive_fee_value("0%") is False
    assert HLBot._is_positive_fee_value("0") is False
    assert HLBot._is_positive_fee_value(10) is True
    assert HLBot._is_positive_fee_value(0) is False
    assert HLBot._is_positive_fee_value(0.0) is False
    assert HLBot._is_positive_fee_value(None) is False
    assert HLBot._is_positive_fee_value("garbage") is False


def test_extract_max_builder_fee_various_shapes(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    # Direct value
    assert HLBot._extract_max_builder_fee("0.01%") == "0.01%"
    assert HLBot._extract_max_builder_fee(20) == 20

    # Nested in dict
    assert HLBot._extract_max_builder_fee({"maxBuilderFee": "0.02%"}) == "0.02%"
    assert HLBot._extract_max_builder_fee({"data": {"maxBuilderFee": 10}}) == 10

    # Deeply nested
    result = HLBot._extract_max_builder_fee({"response": {"data": {"maxBuilderFee": "0.01%"}}})
    assert result == "0.01%"

    # In list
    assert HLBot._extract_max_builder_fee([None, {"maxBuilderFee": "0.05%"}]) == "0.05%"

    # Empty / missing
    assert HLBot._extract_max_builder_fee({}) is None
    assert HLBot._extract_max_builder_fee(None) is None
    assert HLBot._extract_max_builder_fee([]) is None


def test_is_builder_fee_error(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot)
    assert bot._is_builder_fee_error(
        Exception("Builder fee has not been approved for this user")
    ) is True
    assert bot._is_builder_fee_error(
        Exception("BUILDER FEE has NOT BEEN APPROVED")
    ) is True
    assert bot._is_builder_fee_error(
        Exception("Order must have minimum value of $10")
    ) is False
    assert bot._is_builder_fee_error(Exception("random error")) is False


# ─── Config normalization / option application tests ───


def test_apply_builder_code_options_dict(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "ref": "PASSIVBOT",
        "builder": "0x123",
        "fee_rate": "0.02%",
        "fee_int": 20,
    })
    bot.cca = DummyClient()
    bot.ccp = DummyClient()
    bot._apply_builder_code_options()

    assert bot.cca.options["ref"] == "PASSIVBOT"
    assert bot.cca.options["builder"] == "0x123"
    assert bot.cca.options["feeRate"] == "0.02%"
    assert bot.cca.options["feeInt"] == 20
    assert bot.cca.options["builderFee"] is False  # suppressed for our control
    assert bot.cca.options["approvedBuilderFee"] is False
    assert bot.ccp.options["builder"] == "0x123"


def test_apply_builder_code_options_string(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code="PASSIVBOT")
    bot.cca = DummyClient()
    bot._apply_builder_code_options()

    assert bot.cca.options["ref"] == "PASSIVBOT"
    assert "builder" not in bot.cca.options


def test_apply_builder_code_options_invalid_fee_int(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "fee_int": "not_a_number",
    })
    bot.cca = DummyClient()
    bot._apply_builder_code_options()
    assert bot.cca.options["feeInt"] == 20  # falls back to default


def test_apply_builder_code_options_disabled(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "builder_fee": False,
    })
    bot.cca = DummyClient()
    bot._apply_builder_code_options()
    assert "builder" not in bot.cca.options  # nothing applied


def test_apply_builder_code_options_disabled_string(stubbed_modules):
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "builder_fee": "false",
    })
    bot.cca = DummyClient()
    bot._apply_builder_code_options()
    assert "builder" not in bot.cca.options


# ─── Async path tests ───


@pytest.mark.asyncio
async def test_path_a_already_approved(stubbed_modules):
    """Path A: maxBuilderFee returns positive → thank-you, no approval call."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "fee_rate": "0.02%",
        "fee_int": 20,
    })
    bot.cca = DummyClient(fetch_response="0.02%")
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot.cca._approve_called is False
    assert bot._builder_pending_approval is False


@pytest.mark.asyncio
async def test_path_b_main_wallet_auto_approve(stubbed_modules):
    """Path B: maxBuilderFee returns 0, approve succeeds → approved."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "fee_rate": "0.02%",
        "fee_int": 20,
    })
    bot.cca = DummyClient(fetch_response="0")
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot.cca._approve_called is True
    assert bot._builder_pending_approval is False


@pytest.mark.asyncio
async def test_path_c_agent_wallet_pending(stubbed_modules):
    """Path C: maxBuilderFee returns 0, approve fails → force-enable + pending."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "fee_rate": "0.02%",
        "fee_int": 20,
    })
    bot.cca = DummyClient(
        fetch_response="0",
        approve_raises=Exception("only main wallet can approve"),
    )
    await bot._init_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True  # force-enabled
    assert bot.cca._approve_called is True
    assert bot._builder_pending_approval is True


@pytest.mark.asyncio
async def test_check_builder_fee_approved_caching(stubbed_modules):
    """Status check interval cache prevents rapid re-queries."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={"builder": "0x123", "fee_int": 20})
    bot.cca = DummyClient(fetch_response="0")

    # First check queries the endpoint
    result1 = await bot._check_builder_fee_approved()
    assert result1 is False

    # Second check within cache window returns cached result
    bot.cca._fetch_response = "0.05%"
    result2 = await bot._check_builder_fee_approved()
    assert result2 is False  # still cached


@pytest.mark.asyncio
async def test_no_builder_config_skips_init(stubbed_modules):
    """No builder config → _init_builder_codes does nothing."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code="PASSIVBOT")  # string, no builder key
    bot.cca = DummyClient()
    await bot._init_builder_codes()
    assert bot.cca._approve_called is False
    assert bot._builder_pending_approval is False


# ─── camelCase alias tests ───


def test_apply_builder_code_options_camel_case_keys(stubbed_modules):
    """broker_codes.hjson can use camelCase keys (feeRate, feeInt, builderFee)."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "feeRate": "0.03%",
        "feeInt": 30,
        "builderFee": True,
    })
    bot.cca = DummyClient()
    bot._apply_builder_code_options()
    assert bot.cca.options["feeRate"] == "0.03%"
    assert bot.cca.options["feeInt"] == 30


def test_has_builder_config_respects_disabled_toggle(stubbed_modules):
    """_has_builder_config returns False when builder_fee is explicitly disabled."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "builder_fee": False,
    })
    assert bot._has_builder_config() is False

    bot2 = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "builderFee": "false",
    })
    assert bot2._has_builder_config() is False

    bot3 = _make_bot(HLBot, broker_code={
        "builder": "0x123",
        "builder_fee": True,
    })
    assert bot3._has_builder_config() is True


# ─── Re-enable cycle tests ───


@pytest.mark.asyncio
async def test_maybe_reenable_still_not_approved(stubbed_modules):
    """Re-enable cycle: still not approved → re-enable builder for next order attempt."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={"builder": "0x123", "fee_int": 20})
    bot.cca = DummyClient(fetch_response="0")
    bot._builder_pending_approval = True
    bot._builder_disabled_ts = 0  # long ago
    bot.BUILDER_NAG_INTERVAL_MS = 0  # force immediate re-check
    await bot._maybe_reenable_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True  # re-enabled
    assert bot._builder_pending_approval is True  # still pending
    assert bot._builder_disabled_ts is None  # cleared for next cycle


@pytest.mark.asyncio
async def test_maybe_reenable_now_approved(stubbed_modules):
    """Re-enable cycle: user approved externally → resolve pending state."""
    HLBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_bot(HLBot, broker_code={"builder": "0x123", "fee_int": 20})
    bot.cca = DummyClient(fetch_response="0.02%")
    bot._builder_pending_approval = True
    bot._builder_disabled_ts = 0
    bot.BUILDER_NAG_INTERVAL_MS = 0
    await bot._maybe_reenable_builder_codes()

    assert bot.cca.options.get("approvedBuilderFee") is True
    assert bot._builder_pending_approval is False  # resolved!
    assert bot._builder_disabled_ts is None
