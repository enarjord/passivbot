from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ccxt.async_support as ccxt
import pytest

from ccxt_contracts import build_contract_bot, get_bot_class
from exchanges.lighter import (
    SIGNER_SHA256,
    AsyncLighter,
    LighterBot,
    client_config,
    encode_client_id,
    decode_client_id,
)
from exchanges.lighter_fill_events import LighterFetcher
from fill_events_manager import _build_fetcher_for_bot
from utils import get_quote, filter_markets

SYMBOL = "ETH/USDC:USDC"


def market():
    return {
        "id": "0",
        "symbol": SYMBOL,
        "base": "ETH",
        "quote": "USDC",
        "settle": "USDC",
        "type": "swap",
        "spot": False,
        "swap": True,
        "future": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "active": True,
        "contractSize": 1000000,
        "precision": {"amount": 0.0001, "price": 0.01},
        "limits": {
            "amount": {"min": 0.002, "max": None},
            "price": {"min": None, "max": None},
            "cost": {"min": 10, "max": None},
            "leverage": {"min": None, "max": None},
        },
        "info": {
            "size_decimals": 4,
            "price_decimals": 2,
            "min_initial_margin_fraction": 200,
        },
    }


def exchange():
    x = AsyncLighter(
        {"options": {"accountIndex": 123, "apiKeyIndex": 4, "builderFee": False}}
    )
    x.set_markets([market()])
    return x


def bot():
    b = build_contract_bot("lighter", quote="USDC")
    b.hedge_mode = False
    b._lighter_write_lock = asyncio.Lock()
    return b


def raw_order(**changes):
    row = {
        "order_id": "281474976710660",
        "client_order_index": encode_client_id("0x000100000001"),
        "market_index": 0,
        "timestamp": 1700000000,
        "updated_at": 1700000001,
        "is_ask": False,
        "type": "limit",
        "reduce_only": False,
        "status": "open",
        "time_in_force": "post-only",
        "price": "2000",
        "initial_base_amount": ".01",
        "remaining_base_amount": ".006",
        "filled_base_amount": ".004",
        "filled_quote_amount": "8",
    }
    return dict(row, **changes)


def trade(**changes):
    row = {
        "trade_id": 10,
        "market_id": 0,
        "timestamp": 1700000000000,
        "ask_account_id": 456,
        "bid_account_id": 123,
        "ask_id": 100,
        "bid_id": 101,
        "is_maker_ask": True,
        "ask_client_id": 0,
        "bid_client_id": encode_client_id("0x000100000001"),
        "size": ".01",
        "price": "2000",
        "usd_amount": "20",
        "taker_position_size_before": "0",
        "taker_entry_quote_before": "0",
        "taker_fee": 280,
        "integrator_taker_fee": 0,
    }
    return dict(row, **changes)


def fetcher(pages=None):
    api = SimpleNamespace(
        options={"accountIndex": 123},
        markets_by_id={"0": [market()]},
        load_markets=AsyncMock(),
        prepare_api_key=AsyncMock(),
        privateGetTrades=AsyncMock(side_effect=pages),
    )
    return LighterFetcher(api), api


def test_registered_quote_and_fill_factory():
    assert get_bot_class("lighter") is LighterBot
    assert get_quote("lighter") == "USDC"
    assert isinstance(_build_fetcher_for_bot(bot(), []), LighterFetcher)
    assert SYMBOL in filter_markets({SYMBOL: market()}, "lighter")[0]


def test_sizing_is_base_quantity_and_leverage_from_imf():
    b = bot()
    b.markets_dict = {SYMBOL: market()}
    b.set_market_specific_settings()
    assert b.c_mults[SYMBOL] == 1
    assert b.max_leverage[SYMBOL] == 50
    assert b.min_costs[SYMBOL] == 10


@pytest.mark.parametrize("type_id", [0, 1, 255, 4095])
def test_client_id_roundtrip_and_bounds(type_id):
    custom_id = bot().format_custom_id_single(type_id)
    wire = encode_client_id(custom_id)
    assert 0 < wire < 2**48
    assert decode_client_id(wire) == custom_id
    assert decode_client_id("1234") == "1234"


def test_credentials_cannot_trigger_l1_key_rotation_or_builder_approval(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(SIGNER_SHA256, "test", hashlib.sha256(b"test").hexdigest())
    signer = tmp_path / "signer"
    signer.write_bytes(b"test")
    user = {
        "account_index": 123,
        "api_key_index": 4,
        "private_key": "a" * 80,
        "signer_path": str(signer),
        "options": {"builderFee": True, "accountIndex": 456},
    }
    config = client_config(user)
    assert "privateKey" not in config
    assert config["options"]["accountIndex"] == 123
    assert config["options"]["builderFee"] is False
    assert config["options"]["auths"]["123"]["4"]["lighterPrivateKey"] == "a" * 80
    from tools.fetch_balance import build_exchange

    balance_client = build_exchange({**user, "exchange": "lighter"})
    assert balance_client.options["accountIndex"] == 123
    assert not balance_client.privateKey
    assert balance_client.options["builderFee"] is False
    with pytest.raises(ValueError):
        client_config({**user, "api_key_index": 0})
    with pytest.raises(ValueError):
        client_config({**user, "private_key": "a" * 64})


@pytest.mark.asyncio
async def test_public_data_client_and_backward_listing_discovery():
    from utils import load_ccxt_instance

    x = load_ccxt_instance("lighter")
    assert isinstance(x, AsyncLighter)
    day = 86_400_000
    x.milliseconds = lambda: 1200 * day
    first = [550 * day, 1, 2, 1, 2, 10]
    x.fetch_ohlcv = AsyncMock(side_effect=[[[700 * day, 1, 2, 1, 2, 10]], [first], []])
    try:
        assert await x.fetch_first_candle(SYMBOL) == first
        assert [c.kwargs["params"]["until"] for c in x.fetch_ohlcv.call_args_list] == [
            1200 * day,
            700 * day,
            550 * day,
        ]
        x.fetch_ohlcv = AsyncMock(return_value=[[1200 * day, 1, 2, 1, 2, 10]])
        with pytest.raises(ValueError, match="progress"):
            await x.fetch_first_candle(SYMBOL)
        x.fetch_ohlcv = AsyncMock(side_effect=RuntimeError("unavailable"))
        with pytest.raises(RuntimeError):
            await x.fetch_first_candle(SYMBOL)
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_historical_manager_uses_lighter_listing_and_base_units(tmp_path):
    from hlcv_preparation import HLCVManager

    x = exchange()
    x.fetch_first_candle = AsyncMock(return_value=[1700000000000, 1, 2, 1, 2, 10])
    manager = HLCVManager("lighter", cc=x)
    manager.markets = {SYMBOL: market()}
    manager.cache_filepaths["first_timestamps"] = str(tmp_path / "first.json")
    try:
        assert await manager.get_first_timestamp("ETH") == 1700000000000
        x.fetch_first_candle.assert_awaited_once_with(SYMBOL)
        settings = manager.get_market_specific_settings("ETH")
        assert settings["c_mult"] == 1.0
        assert settings["hedge_mode"] is False
    finally:
        await manager.aclose()
        await x.close()


@pytest.mark.asyncio
async def test_live_market_age_discovery_uses_lighter_daily_history(
    tmp_path, monkeypatch
):
    import procedures

    monkeypatch.chdir(tmp_path)
    x = exchange()
    first = 1700000000000
    x.fetch_first_candle = AsyncMock(return_value=[first, 1, 2, 1, 2, 10])
    monkeypatch.setattr(procedures, "load_ccxt_instance", lambda _: x)
    monkeypatch.setattr(procedures, "coin_to_symbol", lambda *_: SYMBOL)
    monkeypatch.setattr(
        procedures, "load_markets", AsyncMock(return_value={SYMBOL: market()})
    )
    result = await procedures.get_first_timestamps_unified([SYMBOL], exchange="lighter")
    assert result == {SYMBOL: first}
    x.fetch_first_candle.assert_awaited_once_with(SYMBOL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side,pside,reduce_only",
    [
        ("buy", "long", False),
        ("sell", "short", False),
        ("sell", "long", True),
        ("buy", "short", True),
    ],
)
@pytest.mark.parametrize("tif", ["gtc", "post_only"])
async def test_actual_ccxt_signing_request(side, pside, reduce_only, tif):
    b = bot()
    b.config["live"]["time_in_force"] = tif
    order = {
        "side": side,
        "position_side": pside,
        "type": "limit",
        "custom_id": "0x000100000001",
    }
    x = exchange()
    try:
        request = x.create_order_request(
            SYMBOL, "limit", side, 0.01, 2000, b._build_order_params(order)
        )[0]
        assert request["reduce_only"] == int(reduce_only)
        assert request["is_ask"] == int(side == "sell")
        assert request["base_amount"] == 100
        assert request["avg_execution_price"] == 200000
        assert request["time_in_force"] == (2 if tif == "post_only" else 1)
        assert request["order_expiry"] == -1
        assert request["integrator_account_index"] == 0
        assert request["integrator_taker_fee"] == request["integrator_maker_fee"] == 0
        assert request["client_order_index"] == encode_client_id(order["custom_id"])
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_all_market_orders_are_unfiltered_and_remaining_quantity_preserved():
    x = exchange()
    try:
        x.prepare_api_key = AsyncMock()
        x.privateGetAccountActiveOrders = AsyncMock(
            return_value={"orders": [raw_order()]}
        )
        orders = await x.fetch_open_orders()
        params = x.privateGetAccountActiveOrders.call_args.args[0]
        assert "market_id" not in params
        assert params["market_type"] == "perp"
        assert orders[0]["remaining"] == 0.006
        assert orders[0]["clientOrderId"] == "0x000100000001"
        assert bot()._get_position_side_for_order(orders[0]) == "long"
        x.privateGetAccountActiveOrders.return_value = {}
        with pytest.raises(KeyError):
            await x.fetch_open_orders()
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_raw_side_and_close_only_required():
    x = exchange()
    try:
        row = raw_order(is_ask=True, reduce_only=True)
        parsed = x.parse_order(row)
        assert bot()._get_position_side_for_order(parsed) == "long"
        del parsed["info"]["reduce_only"]
        with pytest.raises(ValueError):
            bot()._get_position_side_for_order(parsed)
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_create_requires_exact_rest_confirmation_without_resubmission():
    x = exchange()
    try:
        x.fetch_open_orders = AsyncMock(
            return_value=[{"id": "123", "clientOrderId": "0x000100000001"}]
        )
        with patch.object(
            ccxt.lighter,
            "create_order",
            new=AsyncMock(return_value={"info": {"tx_hash": "tx"}}),
        ) as submit, patch("exchanges.lighter.asyncio.sleep", new=AsyncMock()):
            result = await x.create_order(
                SYMBOL,
                "limit",
                "buy",
                0.01,
                2000,
                {"clientOrderId": encode_client_id("0x000100000001")},
            )
            assert result["id"] == "123"
            assert submit.await_count == 1
            x.fetch_open_orders.return_value = []
            x.fetch_closed_orders = AsyncMock(return_value=[])
            with pytest.raises(ccxt.RequestTimeout):
                await x.create_order(
                    SYMBOL,
                    "limit",
                    "buy",
                    0.01,
                    2000,
                    {"clientOrderId": encode_client_id("0x000100000001")},
                )
            assert submit.await_count == 2
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_cancel_waits_for_authoritative_absence():
    x = exchange()
    try:
        x.fetch_open_orders = AsyncMock(side_effect=[[{"id": "123"}], []])
        with patch.object(
            ccxt.lighter, "cancel_order", new=AsyncMock()
        ) as submit, patch("exchanges.lighter.asyncio.sleep", new=AsyncMock()):
            result = await x.cancel_order("123", SYMBOL)
            assert result["id"] == "123"
            assert (
                result["_passivbot_cancel_requires_full_authoritative_confirmation"]
                is True
            )
            assert submit.await_count == 1
            assert x.fetch_open_orders.await_count == 2
    finally:
        await x.close()


def test_fill_pnl_fee_and_one_way_flip_split():
    f, _ = fetcher()
    opened = f.normalize_trade(trade())[0]
    assert opened["position_side"] == "long"
    assert opened["pnl"] == 0
    assert opened["fees"]["cost"] == pytest.approx(0.0056)
    flipped = f.normalize_trade(
        trade(
            taker_position_size_before="-.006",
            taker_entry_quote_before="12.6",
            bid_account_pnl=".6",
        )
    )
    assert [x["position_side"] for x in flipped] == ["short", "long"]
    assert [x["qty"] for x in flipped] == [0.006, 0.004]
    assert [x["pnl"] for x in flipped] == [0.6, 0]
    assert sum(x["fees"]["cost"] for x in flipped) == pytest.approx(0.0056)
    assert len({x["id"] for x in flipped}) == 2


def test_missing_pnl_and_nonfinite_trade_fail_closed():
    f, _ = fetcher()
    with pytest.raises(ValueError):
        f.normalize_trade(trade(taker_position_size_before="-.01"))
    with pytest.raises(ValueError):
        f.normalize_trade(trade(size="nan"))
    with pytest.raises(ValueError):
        f.normalize_trade(trade(ask_account_id=123))
    row = trade()
    del row["taker_fee"]
    assert f.normalize_trade(row)[0]["fees"] is None


def test_omitted_zero_pnl_requires_original_decimal_evidence():
    f, _ = fetcher()
    row = trade(
        taker_position_size_before="-1000000",
        taker_entry_quote_before="100000000000.000002",
        size="1000000",
        price="100000",
    )
    with pytest.raises(ValueError, match="missing realized PnL"):
        f.normalize_trade(row)
    row["taker_entry_quote_before"] = "100000000000"
    assert f.normalize_trade(row)[0]["pnl"] == 0.0


@pytest.mark.asyncio
async def test_trade_cursor_deduplication_and_range():
    newest = trade(trade_id=12, timestamp=3000)
    older = trade(trade_id=11, timestamp=2000)
    oldest = trade(trade_id=10, timestamp=1000)
    f, api = fetcher(
        [
            {"trades": [newest, older], "next_cursor": "next"},
            {"trades": [older, oldest]},
        ]
    )
    events = await f.fetch(1500, 3500, {})
    assert [x["id"] for x in events] == ["11", "12"]
    assert api.privateGetTrades.await_count == 2
    assert api.privateGetTrades.call_args.args[0]["cursor"] == "next"


@pytest.mark.asyncio
async def test_stuck_cursor_does_not_publish_partial_history():
    f, _ = fetcher(
        [
            {"trades": [trade()], "next_cursor": "same"},
            {"trades": [trade()], "next_cursor": "same"},
        ]
    )
    callback = []
    with pytest.raises(ValueError, match="progress"):
        await f.fetch(None, None, {}, callback.extend)
    assert not callback


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [None, "", "missing"])
@pytest.mark.parametrize("later_page", [False, True])
async def test_full_trade_page_without_cursor_does_not_publish_partial_history(
    cursor, later_page
):
    terminal = {
        "trades": [
            trade(trade_id=12, timestamp=3000),
            trade(trade_id=11, timestamp=2000),
        ]
    }
    if cursor != "missing":
        terminal["next_cursor"] = cursor
    pages = []
    if later_page:
        pages.append(
            {
                "trades": [
                    trade(trade_id=14, timestamp=5000),
                    trade(trade_id=13, timestamp=4000),
                ],
                "next_cursor": "page-2",
            }
        )
    pages.append(terminal)
    f, _ = fetcher(pages)
    f.trade_limit = 2
    callback = []
    with pytest.raises(ValueError, match="missing a continuation cursor"):
        await f.fetch(1000, 6000, {}, callback.extend)
    assert not callback


@pytest.mark.asyncio
async def test_cursorless_full_page_keeps_fill_coverage_unproven(tmp_path):
    from fill_events_manager import FillEventsManager

    f, _ = fetcher(
        [
            {
                "trades": [
                    trade(trade_id=12, timestamp=3000),
                    trade(trade_id=11, timestamp=2000),
                ]
            }
        ]
    )
    f.trade_limit = 2
    manager = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    with pytest.raises(ValueError, match="missing a continuation cursor"):
        await manager.refresh(start_ms=1000, end_ms=4000)
    assert not manager._events
    assert manager.cache.get_known_gaps()
    assert not manager.get_coverage_status(start_ms=1000, end_ms=4000)["ready"]
    restarted = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert not restarted._events
    assert not restarted.get_coverage_status(start_ms=1000, end_ms=4000)["ready"]


@pytest.mark.asyncio
@pytest.mark.parametrize("crossed_start", [False, True])
async def test_trade_history_requires_short_terminal_page_or_crossed_start(
    crossed_start,
):
    rows = [trade(trade_id=12, timestamp=3000), trade(trade_id=11, timestamp=2000)]
    if crossed_start:
        pages, since, expected = [{"trades": rows}], 2500, ["12"]
    else:
        pages = [
            {"trades": rows, "next_cursor": "page-2"},
            {"trades": [trade(trade_id=10, timestamp=1000)]},
        ]
        since, expected = 1000, ["10", "11", "12"]
    f, _ = fetcher(pages)
    f.trade_limit = 2
    assert [event["id"] for event in await f.fetch(since, 4000, {})] == expected


@pytest.mark.asyncio
async def test_fill_manager_restart_and_truncated_position_evidence(tmp_path):
    from fill_events_manager import FillEventsManager

    close = trade(
        taker_position_size_before="-.03",
        taker_entry_quote_before="63",
        bid_account_pnl="1",
    )
    f, api = fetcher()
    api.privateGetTrades.return_value = {"trades": [close]}
    api.privateGetTrades.side_effect = None
    manager = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    await manager.refresh(start_ms=close["timestamp"])
    assert len(manager._events) == 1
    event = manager._events[0]
    assert event.psize == pytest.approx(0.02)
    assert event.pprice == pytest.approx(2100)
    assert event.pnl == 1
    assert event.qty == 0.01
    restarted = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert restarted._events[0].psize == pytest.approx(0.02)
    assert restarted._events[0].pprice == pytest.approx(2100)


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["buy", "sell"])
@pytest.mark.parametrize("legacy_cache", [False, True])
async def test_flip_legs_survive_refresh_deduplication_and_restart(
    tmp_path, side, legacy_cache
):
    from fill_events_manager import FillEvent, FillEventsManager

    buying = side == "buy"
    pnl = 0.6 if buying else -0.6
    row = trade(
        ask_account_id=456 if buying else 123,
        bid_account_id=123 if buying else 456,
        is_maker_ask=buying,
        taker_position_size_before="-.006" if buying else ".006",
        taker_entry_quote_before="12.6",
        **{"bid_account_pnl" if buying else "ask_account_pnl": str(pnl)},
    )
    f, api = fetcher()
    api.privateGetTrades.side_effect = None
    api.privateGetTrades.return_value = {"trades": [row]}
    manager = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    if legacy_cache:
        # Older normalization retained only the opening leg with the raw trade
        # source identity. A history refresh must restore the lost closing leg.
        opening = f.normalize_trade(row)[1]
        opening["source_ids"] = [str(row["trade_id"])]
        manager.cache.save([FillEvent.from_dict(opening)])

    def assert_complete(current):
        events = {event.id: event for event in current._events}
        assert len(current._events) == 2
        assert set(events) == {"10:close", "10:open"}
        close, opened = events["10:close"], events["10:open"]
        assert close.source_ids == ["10:close"]
        assert opened.source_ids == ["10:open"]
        assert close.position_side == ("short" if buying else "long")
        assert opened.position_side == ("long" if buying else "short")
        assert close.pnl == pytest.approx(pnl)
        assert opened.pnl == 0
        assert sum(event.fee_paid for event in current._events) == pytest.approx(
            -0.0056
        )
        assert close.psize == 0
        assert close.pprice == 0
        assert abs(opened.psize) == pytest.approx(0.004)
        assert opened.pprice == 2000

    await manager.refresh(start_ms=row["timestamp"])
    assert_complete(manager)
    await manager.refresh(start_ms=row["timestamp"])
    assert_complete(manager)
    restarted = FillEventsManager(
        exchange="lighter", user="fixture", fetcher=f, cache_path=tmp_path
    )
    await restarted.ensure_loaded()
    assert_complete(restarted)
    await restarted.refresh(start_ms=row["timestamp"])
    assert_complete(restarted)


def test_balance_tool_direct_script_resolves_lighter_imports(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    keys = tmp_path / "keys.json"
    keys.write_text(
        json.dumps(
            {
                "fixture": {
                    "exchange": "lighter",
                    "account_index": True,
                    "api_key_index": 4,
                }
            }
        )
    )
    script = Path(__file__).resolve().parents[2] / "src/tools/fetch_balance.py"
    result = subprocess.run(
        [sys.executable, str(script), "--user", "fixture", "--api-keys", str(keys)],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Invalid local credentials stop before any network request, after imports.
    assert result.returncode == 1
    assert "account_index must be a nonnegative integer" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not result.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    "accounts",
    [
        [],
        [{"account_index": 123, "collateral": "12"}] * 2,
        [{"account_index": 456, "collateral": "12"}],
        [{"account_index": 123.5, "collateral": "12"}],
        [{"account_index": 123}],
        [{"account_index": 123, "collateral": "nan"}],
        [{"account_index": 123, "collateral": "inf"}],
        [{"account_index": 123, "collateral": True}],
        [{"account_index": 123, "collateral": "0"}],
        [{"account_index": 123, "collateral": "12.345678"}],
    ],
)
async def test_sync_and_async_balance_validate_account_and_collateral(sync, accounts):
    from unittest.mock import Mock
    from exchanges.lighter_balance import SyncLighterBalance

    x = SyncLighterBalance({"options": {"accountIndex": 123}}) if sync else exchange()
    x.set_markets([market()])
    x.publicGetAccount = (Mock if sync else AsyncMock)(
        return_value={"accounts": accounts}
    )
    valid = len(accounts) == 1 and accounts[0].get("collateral") in ("0", "12.345678")
    try:
        if valid:
            result = x.fetch_balance() if sync else await x.fetch_balance()
            assert result["total"]["USDC"] == float(accounts[0]["collateral"])
            assert result["USDC"]["total"] == result["total"]["USDC"]
        else:
            with pytest.raises((ValueError, KeyError, TypeError)):
                if sync:
                    x.fetch_balance()
                else:
                    await x.fetch_balance()
    finally:
        if not sync:
            await x.close()


@pytest.mark.asyncio
async def test_candle_requests_cannot_tail_skip_the_oldest_page():
    x = exchange()
    try:
        with patch.object(
            ccxt.lighter, "fetch_ohlcv", new=AsyncMock(return_value=[])
        ) as fetch:
            await x.fetch_ohlcv(SYMBOL, "1m", 60000, 1000)
            args = fetch.call_args.args
            assert args[2:4] == (60000, 500)
            assert args[4]["until"] == 60000 + 500 * 60000
    finally:
        await x.close()


def test_incompatible_signer_is_rejected_before_native_load(tmp_path):
    signer = tmp_path / "signer"
    signer.write_bytes(b"incompatible")
    with pytest.raises(ValueError, match="official revision"):
        client_config(
            {
                "account_index": 123,
                "api_key_index": 4,
                "private_key": "a" * 80,
                "signer_path": str(signer),
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,wire", [("cross", 0), ("isolated", 1)])
async def test_actual_ccxt_leverage_signing_boundary(mode, wire):
    x = exchange()
    try:
        x.load_account = AsyncMock(return_value=object())
        x.fetch_nonce = AsyncMock(return_value=17)
        x.publicPostSendTx = AsyncMock(return_value={"code": 200})
        with patch.object(
            x, "lighter_sign_update_leverage", return_value=(20, "signed")
        ) as sign:
            await x.set_leverage(5, SYMBOL, {"marginMode": mode})
            request = sign.call_args.args[1]
            assert request["margin_mode"] == wire
            assert request["initial_margin_fraction"] == 2000
            assert request["nonce"] == 17
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_malformed_account_responses_never_become_flat_or_zero():
    x = exchange()
    try:
        x.publicGetAccount = AsyncMock(return_value={"accounts": []})
        with pytest.raises(ValueError):
            await x.fetch_balance()
        with pytest.raises(ValueError):
            await x.fetch_positions()
        x.publicGetAccount.return_value = {
            "accounts": [
                {
                    "account_index": 123,
                    "positions": [{"position": "1", "sign": 1, "margin_mode": 0}],
                }
            ]
        }
        with pytest.raises(KeyError):
            await x.fetch_positions()
        x.publicGetAccount.side_effect = ccxt.NetworkError("unavailable")
        with pytest.raises(ccxt.NetworkError):
            await x.fetch_positions()
    finally:
        await x.close()


@pytest.mark.asyncio
async def test_websocket_quote_failure_propagates_and_never_uses_last_as_bid():
    b = bot()
    b.cca = SimpleNamespace(
        fetch_tickers=AsyncMock(return_value={SYMBOL: {"last": 2000}})
    )
    b.ccp = SimpleNamespace(
        watch_order_book=AsyncMock(side_effect=ccxt.NetworkError("offline"))
    )
    with pytest.raises(ccxt.NetworkError):
        await b.fetch_tickers_for_symbols([SYMBOL])
    b.ccp.watch_order_book = AsyncMock(
        return_value={"bids": [[1999, 1]], "asks": [[2001, 1]]}
    )
    quotes = await b.fetch_tickers_for_symbols([SYMBOL])
    assert quotes[SYMBOL]["bid"] == 1999
    assert quotes[SYMBOL]["ask"] == 2001
    assert quotes[SYMBOL]["last"] == 2000


@pytest.mark.asyncio
async def test_signed_order_writes_are_serialized():
    from exchanges.ccxt_bot import CCXTBot

    b = bot()
    in_flight = 0
    peak = 0

    async def write(self, order):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        return order

    with patch.object(CCXTBot, "execute_order", write), patch.object(
        CCXTBot, "execute_cancellation", write
    ):
        await asyncio.gather(
            b.execute_order({"id": "1"}),
            b.execute_cancellation({"id": "2"}),
            b.execute_order({"id": "3"}),
        )
    assert peak == 1
