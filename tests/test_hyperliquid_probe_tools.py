import importlib
import sys

import pytest


class _DummySession:
    def __init__(self):
        self.cancelled = []
        self.created = []
        self.margin_mode_calls = []
        self.closed = False

    async def load_markets(self):
        return {
            "BTC/USDC:USDC": {
                "precision": {"amount": 0.0001, "price": 0.1},
                "limits": {"cost": {"min": 10.0}},
            }
        }

    async def fetch_ticker(self, symbol):
        assert symbol == "BTC/USDC:USDC"
        return {"last": 50000.0}

    async def fetch_balance(self):
        return {
            "free": {"USDC": 100.0},
            "used": {"USDC": 0.0},
            "total": {"USDC": 100.0},
            "info": {
                "marginSummary": {"accountValue": "100.0", "totalMarginUsed": "0.0", "totalNtlPos": "0.0"},
                "crossMarginSummary": {
                    "accountValue": "100.0",
                    "totalMarginUsed": "0.0",
                    "totalNtlPos": "0.0",
                },
                "withdrawable": "100.0",
                "assetPositions": [],
            },
        }

    async def fetch_positions(self, symbols=None):
        assert symbols == ["BTC/USDC:USDC"]
        return []

    async def create_order(self, symbol, order_type, side, amount, price, params=None):
        assert symbol == "BTC/USDC:USDC"
        assert order_type == "limit"
        assert side == "buy"
        assert amount > 0.0
        assert price > 0.0
        assert params == {"timeInForce": "Alo"}
        self.created.append(
            {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        return {"id": "probe-order"}

    async def fetch_open_orders(self, symbol=None):
        if symbol == "BTC/USDC:USDC":
            return [
                {
                    "id": "probe-order",
                    "symbol": symbol,
                    "side": "buy",
                    "amount": 0.0003,
                    "price": 37500.0,
                    "status": "open",
                }
            ]
        return []

    async def cancel_order(self, order_id, symbol=None, params=None):
        self.cancelled.append((order_id, symbol, params))
        return {"id": order_id}

    async def set_margin_mode(self, margin_mode, symbol=None, params=None):
        self.margin_mode_calls.append((margin_mode, symbol, params))
        return {"status": "ok"}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_hyperliquid_order_margin_probe_smoke(monkeypatch, capsys):
    mod = importlib.import_module("tools.probe_hyperliquid_order_margin")
    session = _DummySession()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_hyperliquid_order_margin.py",
            "--user",
            "hyperliquid_01",
            "--symbol",
            "BTC/USDC:USDC",
            "--yes",
        ],
    )
    monkeypatch.setattr(
        mod,
        "load_hyperliquid_wallet",
        lambda user, api_keys_path: ({}, "wallet", "key"),
    )
    monkeypatch.setattr(mod, "create_hyperliquid_probe_session", lambda wallet, key: session)

    assert await mod._main() == 0

    out = capsys.readouterr().out
    assert '"symbol": "BTC/USDC:USDC"' in out
    assert '"order"' in out
    assert session.cancelled
    assert session.closed is True


class _VaultDummySession(_DummySession):
    async def load_markets(self):
        return {
            "BTC/USDC:USDC": {
                "precision": {"amount": 0.0001, "price": 0.1},
                "limits": {"cost": {"min": 10.0}},
            }
        }

    async def fetch_ticker(self, symbol):
        assert symbol == "BTC/USDC:USDC"
        return {"last": 50000.0}

    async def fetch_positions(self, symbols=None):
        assert symbols == ["BTC/USDC:USDC"]
        return []

    async def create_order(self, symbol, order_type, side, amount, price, params=None):
        self.created.append(
            {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": params,
            }
        )
        if order_type == "market":
            return {"id": "entry-order"}
        return {"id": "resting-order"}

    async def fetch_open_orders(self, symbol=None):
        if symbol == "BTC/USDC:USDC":
            return [
                {
                    "id": "resting-order",
                    "symbol": symbol,
                    "side": "buy",
                    "amount": 0.0003,
                    "price": 37500.0,
                    "status": "open",
                }
            ]
        return []


@pytest.mark.asyncio
async def test_hyperliquid_position_probe_threads_vault_address(monkeypatch, capsys):
    mod = importlib.import_module("tools.probe_hyperliquid_position_balance")
    session = _VaultDummySession()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_hyperliquid_position_balance.py",
            "--user",
            "hyperliquid_vault",
            "--symbol",
            "BTC/USDC:USDC",
            "--yes",
            "--set-margin-mode",
            "cross",
            "--leave-open-after-entry",
            "--place-resting-entry-order",
        ],
    )
    monkeypatch.setattr(
        mod,
        "load_hyperliquid_wallet",
        lambda user, api_keys_path: (
            {"is_vault": True, "wallet_address": "0xvault"},
            "wallet",
            "key",
        ),
    )
    monkeypatch.setattr(mod, "create_hyperliquid_probe_session", lambda wallet, key: session)

    assert await mod._main() == 0

    out = capsys.readouterr().out
    assert '"symbol": "BTC/USDC:USDC"' in out
    assert session.margin_mode_calls == [("cross", "BTC/USDC:USDC", {"vaultAddress": "0xvault", "leverage": 5})]
    assert session.created
    assert session.created[0]["params"] == {"vaultAddress": "0xvault"}
    assert session.created[1]["params"] == {"vaultAddress": "0xvault", "timeInForce": "Alo"}
    assert session.closed is True
