from __future__ import annotations

import math

import pytest

from exchanges.binance import BinanceBot
from live.balance_composition import (
    ASSET_BALANCE_MAX_ROWS,
    balance_composition_signature,
    format_balance_composition_sample,
    malformed_balance_composition,
    normalize_ccxt_balance_composition,
    normalize_hyperliquid_unified_balance_composition,
    normalize_okx_balance_composition,
    public_balance_composition,
    unavailable_balance_composition,
)
from live.event_bus import (
    EventTypes,
    ListEventSink,
    LiveEvent,
    LiveEventPipeline,
    format_console_event,
)
from live import state_refresh


def _okx_payload(details):
    return {"info": {"data": [{"details": details}]}}


def _hyperliquid_unified_payload(balances):
    return {"info": {"balances": balances}}


def test_okx_balance_composition_keeps_only_proven_detail_fields():
    snapshot = normalize_okx_balance_composition(
        _okx_payload(
            [
                {
                    "ccy": "USDT",
                    "cashBal": "12.5",
                    "eqUsd": "12.4",
                    "upl": "-0.1",
                    "collateralEnabled": "true",
                    "liab": "2",
                    "secret": "must-not-leak",
                },
                {
                    "ccy": "btc",
                    "cashBal": "0.02",
                    "eqUsd": "1300",
                    "upl": "5",
                    "collateralEnabled": False,
                },
            ]
        )
    )

    assert snapshot["status"] == "available"
    assert snapshot["count"] == 2
    assert [row["asset"] for row in snapshot["asset_balances"]] == ["BTC", "USDT"]
    usdt = snapshot["asset_balances"][1]
    assert usdt == {
        "asset": "USDT",
        "field_provenance": {
            "asset": "ccy",
            "amount": "cashBal",
            "usd_value": "eqUsd",
            "unrealized_pnl": "upl",
            "liability": "liab",
            "collateral_enabled": "collateralEnabled",
        },
        "amount": 12.5,
        "usd_value": 12.4,
        "unrealized_pnl": -0.1,
        "liability": 2.0,
        "collateral_enabled": True,
    }
    assert "secret" not in str(snapshot)
    assert "_signature" in snapshot


def test_okx_balance_composition_marks_malformed_and_unsupported_states_explicitly():
    malformed = normalize_okx_balance_composition({"info": {"data": [{}]}})
    unsupported = unavailable_balance_composition()

    assert malformed["status"] == "malformed"
    assert malformed["reason"] == "missing_details"
    assert malformed["asset_balances"] == []
    assert unsupported["status"] == "unavailable"
    assert unsupported["reason"] == "unsupported_connector"
    assert public_balance_composition(malformed_balance_composition(
        source="normalizer", reason="normalizer_error"
    ))["status"] == "malformed"


def test_okx_balance_composition_omits_missing_and_nonfinite_but_keeps_proven_zeroes():
    snapshot = normalize_okx_balance_composition(
        _okx_payload(
            [
                {
                    "ccy": "USDC",
                    "cashBal": "0",
                    "eqUsd": "nan",
                    "upl": "inf",
                    "liab": None,
                    "collateralEnabled": "unknown",
                },
                {"ccy": "\x1b[31mBAD", "cashBal": "1"},
            ]
        )
    )

    assert snapshot["count"] == 1
    assert snapshot["invalid_row_count"] == 1
    assert snapshot["asset_balances"] == [
        {
            "asset": "USDC",
            "field_provenance": {"asset": "ccy", "amount": "cashBal"},
            "amount": 0.0,
        }
    ]


def test_ccxt_balance_composition_keeps_only_unified_balance_maps():
    snapshot = normalize_ccxt_balance_composition(
        {
            "total": {"USDT": 120.0, "BTC": "0.25"},
            "free": {"USDT": 100.0, "BTC": "0.2"},
            "used": {"USDT": 20.0, "BTC": "0.05"},
            "debt": {"USDT": "3.5"},
            "info": {"apiKey": "must-not-leak", "assets": [{"secret": "value"}]},
            "timestamp": 123456,
        }
    )

    assert snapshot["status"] == "available"
    assert snapshot["source"] == "ccxt.unified_balance"
    assert snapshot["count"] == 2
    assert snapshot["asset_balances"] == [
        {
            "asset": "BTC",
            "field_provenance": {
                "asset": "currency_map_key",
                "amount": "total",
                "free_amount": "free",
                "used_amount": "used",
            },
            "amount": 0.25,
            "free_amount": 0.2,
            "used_amount": 0.05,
        },
        {
            "asset": "USDT",
            "field_provenance": {
                "asset": "currency_map_key",
                "amount": "total",
                "free_amount": "free",
                "used_amount": "used",
                "liability": "debt",
            },
            "amount": 120.0,
            "free_amount": 100.0,
            "used_amount": 20.0,
            "liability": 3.5,
        },
    ]
    assert "secret" not in str(snapshot)
    assert "apiKey" not in str(snapshot)


def test_ccxt_balance_composition_is_explicit_for_missing_or_partial_fields():
    malformed = normalize_ccxt_balance_composition({"free": {"USDT": 1.0}})
    snapshot = normalize_ccxt_balance_composition(
        {
            "total": {"USDT": 0, "BTC": "nan", "\x1b[31mBAD": 1},
            "free": {"BTC": 0.1},
            "used": None,
            "debt": {"BTC": "inf"},
        }
    )

    assert malformed == malformed_balance_composition(
        source="ccxt.unified_balance", reason="missing_total"
    )
    assert snapshot["invalid_row_count"] == 1
    assert snapshot["asset_balances"] == [
        {
            "asset": "BTC",
            "field_provenance": {
                "asset": "currency_map_key",
                "free_amount": "free",
            },
            "free_amount": 0.1,
        },
        {
            "asset": "USDT",
            "field_provenance": {
                "asset": "currency_map_key",
                "amount": "total",
            },
            "amount": 0.0,
        },
    ]


def test_ccxt_balance_composition_public_contract_keeps_bounded_provenance():
    public = public_balance_composition(
        normalize_ccxt_balance_composition(
            {
                "total": {"USDT": 10.0},
                "free": {"USDT": 7.0},
                "used": {"USDT": 3.0},
                "debt": {"USDT": 2.0},
            }
        )
    )

    assert public["asset_balances"] == [
        {
            "asset": "USDT",
            "amount": 10.0,
            "free_amount": 7.0,
            "used_amount": 3.0,
            "liability": 2.0,
            "field_provenance": {
                "asset": "currency_map_key",
                "amount": "total",
                "free_amount": "free",
                "used_amount": "used",
                "liability": "debt",
            },
        }
    ]


def test_ccxt_balance_composition_case_collision_is_order_independent():
    first = normalize_ccxt_balance_composition(
        {"total": {"BTC": 1.0, "btc": 2.0}, "free": {"BTC": 0.5}}
    )
    second = normalize_ccxt_balance_composition(
        {"total": {"btc": 2.0, "BTC": 1.0}, "free": {"BTC": 0.5}}
    )

    assert first["invalid_row_count"] == 1
    assert first["asset_balances"] == [
        {
            "asset": "BTC",
            "field_provenance": {
                "asset": "currency_map_key",
                "free_amount": "free",
            },
            "free_amount": 0.5,
        }
    ]
    assert first["asset_balances"] == second["asset_balances"]
    assert balance_composition_signature(first) == balance_composition_signature(second)


def test_hyperliquid_unified_balance_composition_keeps_only_coin_and_signed_total():
    snapshot = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload(
            [
                {"coin": "USDC", "total": "-12.5", "hold": "1", "address": "secret"},
                {"coin": "btc", "total": "0", "apiKey": "must-not-leak"},
            ]
        )
    )

    assert snapshot["status"] == "available"
    assert snapshot["source"] == "hyperliquid.info.balances"
    assert snapshot["asset_balances"] == [
        {
            "asset": "BTC",
            "field_provenance": {"asset": "coin", "amount": "total"},
            "amount": 0.0,
        },
        {
            "asset": "USDC",
            "field_provenance": {"asset": "coin", "amount": "total"},
            "amount": -12.5,
        },
    ]
    assert public_balance_composition(snapshot)["asset_balances"] == snapshot["asset_balances"]
    assert "secret" not in str(snapshot)
    assert "apiKey" not in str(snapshot)


def test_hyperliquid_balance_composition_is_explicit_for_non_unified_and_malformed_rows():
    non_unified = normalize_hyperliquid_unified_balance_composition(
        {"info": {"marginSummary": {"accountValue": "10"}}}
    )
    malformed_balances = normalize_hyperliquid_unified_balance_composition(
        {"info": {"balances": {}}}
    )
    empty_balances = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload([])
    )
    malformed_rows = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload(
            [{"coin": "\x1b[31mBAD", "total": "1"}, {"coin": "USDC", "total": "nan"}]
        )
    )

    assert non_unified["status"] == "unavailable"
    assert non_unified["reason"] == "non_unified_payload"
    assert malformed_balances == malformed_balance_composition(
        source="hyperliquid.info.balances", reason="invalid_balances"
    )
    assert empty_balances == malformed_balance_composition(
        source="hyperliquid.info.balances", reason="empty_balances"
    )
    assert malformed_rows == malformed_balance_composition(
        source="hyperliquid.info.balances", reason="no_valid_rows"
    )


def test_hyperliquid_balance_composition_is_deterministic_bounded_and_drops_collisions():
    balances = [
        {"coin": f"ASSET{i:02d}", "total": str(i)}
        for i in range(ASSET_BALANCE_MAX_ROWS + 2, 0, -1)
    ]
    first = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload(balances)
    )
    second = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload(list(reversed(balances)))
    )
    collided = normalize_hyperliquid_unified_balance_composition(
        _hyperliquid_unified_payload(
            [
                {"coin": "BTC", "total": "1"},
                {"coin": "btc", "total": "2"},
                {"coin": "USDC", "total": "3"},
            ]
        )
    )

    assert first["asset_balances"] == second["asset_balances"]
    assert balance_composition_signature(first) == balance_composition_signature(second)
    assert first["count"] == ASSET_BALANCE_MAX_ROWS + 2
    assert first["retained"] == ASSET_BALANCE_MAX_ROWS
    assert first["truncated"] == 2
    assert collided["invalid_row_count"] == 1
    assert collided["asset_balances"] == [
        {
            "asset": "USDC",
            "field_provenance": {"asset": "coin", "amount": "total"},
            "amount": 3.0,
        }
    ]


def test_binance_balance_diagnostic_hook_uses_ccxt_unified_contract():
    snapshot = BinanceBot._normalize_balance_diagnostics(
        object(),
        {
            "total": {"USDT": 10.0},
            "free": {"USDT": 8.0},
            "used": {"USDT": 2.0},
            "info": {"private": "must-not-leak"},
        },
    )

    assert snapshot["source"] == "ccxt.unified_balance"
    assert snapshot["asset_balances"][0]["asset"] == "USDT"
    assert "private" not in str(snapshot)


def test_composition_signature_covers_omitted_rows_and_order_is_deterministic():
    details = [
        {"ccy": f"ASSET{i:02d}", "cashBal": str(i), "eqUsd": str(i * 10)}
        for i in range(ASSET_BALANCE_MAX_ROWS + 2, 0, -1)
    ]
    first = normalize_okx_balance_composition(_okx_payload(details))
    changed = normalize_okx_balance_composition(
        _okx_payload(
            [
                {**detail, "cashBal": "999"}
                if detail["ccy"] == "ASSET10"
                else detail
                for detail in details
            ]
        )
    )

    assert [row["asset"] for row in first["asset_balances"]] == sorted(
        row["asset"] for row in first["asset_balances"]
    )
    assert first["count"] == ASSET_BALANCE_MAX_ROWS + 2
    assert first["retained"] == ASSET_BALANCE_MAX_ROWS
    assert first["truncated"] == 2
    assert first["asset_balances"] == changed["asset_balances"]
    assert balance_composition_signature(first) != balance_composition_signature(changed)


def test_public_composition_drops_unknown_raw_fields_and_console_sample_is_bounded():
    snapshot = normalize_okx_balance_composition(
        _okx_payload(
            [
                {"ccy": "BTC", "cashBal": "1", "eqUsd": "100", "apiKey": "secret"},
                {"ccy": "USDT", "cashBal": "2", "eqUsd": "2"},
                {"ccy": "ETH", "cashBal": "3", "eqUsd": "9"},
            ]
        )
    )
    public = public_balance_composition(snapshot)
    public["asset_balances"][0]["raw_payload"] = {"password": "secret"}

    event = LiveEvent(
        EventTypes.BALANCE_CHANGED,
        data={
            "previous_balance_raw": 10.0,
            "balance_raw": 11.0,
            "balance_raw_delta": 1.0,
            "previous_balance_snapped": 10.0,
            "balance_snapped": 11.0,
            "balance_snapped_delta": 1.0,
            "equity": 11.0,
            "source": "REST",
            "balance_composition": public,
        },
    )
    rendered = format_console_event(event)

    assert "secret" not in str(public_balance_composition(public))
    assert "raw_payload" not in str(public_balance_composition(public))
    assert "assets=BTC:" in rendered
    assert "USDT:" not in rendered
    assert "+1 more" in rendered
    assert len(rendered) < 240
    assert not any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in rendered)
    assert format_balance_composition_sample(snapshot).count(";") <= 1
    assert math.isfinite(snapshot["asset_balances"][0]["amount"])


@pytest.mark.asyncio
async def test_generic_staged_capture_carries_normalized_diagnostics_not_raw_payload():
    raw_payload = {"info": {"data": [{"details": [{"ccy": "USDT", "cashBal": "1"}]}]}}

    class Bot:
        capture_calls = 0

        async def capture_balance_snapshot(self):
            self.capture_calls += 1
            return raw_payload, 10.0

        def _normalize_balance_diagnostics(self, payload):
            assert payload is raw_payload
            return normalize_okx_balance_composition(payload)

    bot = Bot()
    composition, balance = await state_refresh.capture_balance_staged_snapshot(bot)

    assert bot.capture_calls == 1
    assert balance == 10.0
    assert composition["asset_balances"][0]["asset"] == "USDT"
    assert raw_payload is not composition
    assert "info" not in composition


@pytest.mark.asyncio
async def test_diagnostic_normalizer_failure_is_explicit_without_losing_scalar_balance():
    class Bot:
        async def capture_balance_snapshot(self):
            return {"api_key": "secret"}, 10.0

        def _normalize_balance_diagnostics(self, _payload):
            raise RuntimeError("secret parser failure")

    composition, balance = await state_refresh.capture_balance_staged_snapshot(Bot())

    assert balance == 10.0
    assert composition == malformed_balance_composition(
        source="normalizer", reason="normalizer_error"
    )
    assert "secret" not in str(composition)


def test_composition_only_balance_event_remains_durable_but_is_not_console_visible():
    structured = ListEventSink()
    console = ListEventSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[structured], monitor_sinks=[], console_sink=console
    )
    event = LiveEvent(
        EventTypes.BALANCE_CHANGED,
        data={
            "previous_balance_raw": 100.0,
            "balance_raw": 100.0,
            "balance_raw_delta": 0.0,
            "previous_balance_snapped": 100.0,
            "balance_snapped": 100.0,
            "balance_snapped_delta": 0.0,
            "equity": 100.0,
            "source": "REST",
            "balance_composition": normalize_okx_balance_composition(
                _okx_payload([{"ccy": "USDT", "cashBal": "2"}])
            ),
        },
    )

    pipeline.emit(event)

    assert pipeline.flush(timeout=2.0) is True
    assert structured.events == [event]
    assert console.events == []
    assert pipeline.close(timeout=2.0) is True
