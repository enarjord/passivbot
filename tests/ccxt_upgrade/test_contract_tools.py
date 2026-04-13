import json
from types import SimpleNamespace

import pytest

import ccxt_contracts
from ccxt_contracts import capture_contract_snapshot, diff_snapshots, dump_snapshot, sanitize_for_json


def test_sanitize_for_json_handles_non_json_types(tmp_path):
    dumped = sanitize_for_json(
        {
            "path": tmp_path / "a.json",
            "payload": {1, 2},
            "raw": b"abc",
            "nan": float("nan"),
        }
    )

    assert dumped["path"].endswith("a.json")
    assert sorted(dumped["payload"]) == [1, 2]
    assert dumped["raw"] == "abc"
    assert dumped["nan"] == "nan"


def test_diff_snapshots_reports_added_removed_and_changed_fields():
    old = {
        "meta": {"captured_at": "old", "exchange": "binance"},
        "markets": {"summary": {"contracts": {"BTC/USDT:USDT": {"min_qty": 0.001}}}},
    }
    new = {
        "meta": {"captured_at": "new", "exchange": "binance"},
        "markets": {
            "summary": {
                "contracts": {
                    "BTC/USDT:USDT": {"min_qty": 0.01},
                    "ETH/USDT:USDT": {"min_qty": 0.1},
                }
            }
        },
    }

    diff = diff_snapshots(old, new)

    assert diff["summary"] == {"added": 1, "removed": 0, "changed": 1}
    assert diff["added"][0]["path"].endswith("ETH/USDT:USDT.min_qty")
    assert diff["changed"][0]["path"].endswith("BTC/USDT:USDT.min_qty")
    assert diff["changed"][0]["old"] == 0.001
    assert diff["changed"][0]["new"] == 0.01


def test_dump_snapshot_writes_json_file(tmp_path):
    path = dump_snapshot({"hello": "world"}, tmp_path / "snapshot.json")

    assert json.loads(path.read_text()) == {"hello": "world"}


@pytest.mark.asyncio
async def test_capture_contract_snapshot_normalizes_from_single_fetch(monkeypatch):
    counts = {"balance": 0, "positions": 0, "open_orders": []}

    class FakeBot:
        exchange = "fake"
        quote = "USDT"

        def __init__(self):
            self.cca = SimpleNamespace(
                has={"fetchBalance": True},
                load_markets=self._load_markets,
            )
            self.markets_dict = {}

        async def _load_markets(self, reload=False):
            return {
                "BTC/USDT:USDT": {
                    "id": "BTCUSDT",
                    "active": True,
                    "swap": True,
                    "linear": True,
                    "limits": {"cost": {"min": 5.0}, "amount": {"min": 0.001}},
                    "precision": {"amount": 0.001, "price": 0.1},
                    "base": "BTC",
                }
            }

        def set_market_specific_settings(self):
            return None

        async def capture_balance_snapshot(self):
            counts["balance"] += 1
            return {"total": {"USDT": 123.0}, "info": {"seq": 1}}, 123.0

        async def capture_positions_snapshot(self):
            counts["positions"] += 1
            raw = [{"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 1.0, "entryPrice": 100.0}]
            normalized = [
                {
                    "symbol": "BTC/USDT:USDT",
                    "position_side": "long",
                    "size": 1.0,
                    "price": 100.0,
                }
            ]
            return raw, normalized

        async def capture_open_orders_snapshot(self, symbol=None):
            counts["open_orders"].append(symbol)
            raw = [{"id": f"{symbol or 'all'}-1", "symbol": symbol or "BTC/USDT:USDT", "amount": 0.5}]
            normalized = [{"id": f"{symbol or 'all'}-1", "symbol": symbol or "BTC/USDT:USDT", "qty": 0.5}]
            return raw, normalized

        async def fetch_balance(self):
            raise AssertionError("capture tool should not re-fetch balance for normalization")

        async def fetch_positions(self):
            raise AssertionError("capture tool should not re-fetch positions for normalization")

        async def fetch_open_orders(self, symbol=None):
            raise AssertionError("capture tool should not re-fetch open orders for normalization")

        async def close(self):
            return None

    fake_bot = FakeBot()
    monkeypatch.setattr(ccxt_contracts, "prepare_live_config_for_user", lambda user: {"live": {"user": user}})
    monkeypatch.setattr(ccxt_contracts, "setup_bot", lambda config: fake_bot)

    snapshot = await capture_contract_snapshot(
        user="fake_user",
        sections=("markets", "capabilities", "balance", "positions", "open_orders"),
        symbols=("BTC/USDT:USDT",),
    )

    assert counts["balance"] == 1
    assert counts["positions"] == 1
    assert counts["open_orders"] == [None, "BTC/USDT:USDT"]
    assert snapshot["balance"]["raw"]["info"]["seq"] == 1
    assert snapshot["balance"]["normalized"] == 123.0
    assert snapshot["positions"]["raw"][0]["contracts"] == 1.0
    assert snapshot["positions"]["normalized"][0]["size"] == 1.0
    assert snapshot["open_orders"]["by_symbol"]["BTC/USDT:USDT"]["raw"][0]["id"] == "BTC/USDT:USDT-1"
    assert (
        snapshot["open_orders"]["by_symbol"]["BTC/USDT:USDT"]["normalized"][0]["qty"] == 0.5
    )
