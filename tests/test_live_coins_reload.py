import asyncio
import json
from pathlib import Path

import hjson

import pytest

import procedures
from config_utils import load_config
import utils
from utils import format_approved_ignored_coins, normalize_coins_source


@pytest.mark.asyncio
async def test_external_approved_coins_reload(tmp_path: Path):
    """Approved coins file changes should be reflected without restarting the bot."""

    approved_file = tmp_path / "approved.hjson"
    approved_file.write_text('["BTC","ETH"]')

    config = {
        "live": {
            "approved_coins": str(approved_file),
            "ignored_coins": {"long": [], "short": []},
        }
    }

    # Initial formatting should read the file once.
    await format_approved_ignored_coins(config, ["binanceusdm"])
    assert config["live"]["approved_coins"]["long"] == ["BTC", "ETH"]

    # Update the on-disk file and expect a refresh to pick up the change.
    approved_file.write_text('["BTC","XRP"]')
    refreshed = normalize_coins_source(config["_coins_sources"]["approved_coins"])

    assert set(refreshed["long"]) == {"BTC", "XRP"}, "approved coins did not reload from file"


@pytest.mark.asyncio
async def test_external_ignored_coins_reload(tmp_path: Path):
    """Ignored coins file changes should be reflected without restarting the bot."""

    ignored_file = tmp_path / "ignored.hjson"
    ignored_file.write_text('["DOGE"]')

    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": ["BTC"]},
            "ignored_coins": str(ignored_file),
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])
    assert config["live"]["ignored_coins"]["long"] == ["DOGE"]

    ignored_file.write_text('["DOGE","SHIB"]')
    refreshed = normalize_coins_source(config["_coins_sources"]["ignored_coins"])

    assert set(refreshed["long"]) == {"DOGE", "SHIB"}, "ignored coins did not reload from file"


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_records_transform():
    config = {
        "live": {
            "approved_coins": "BTC,ETH",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])

    entry = config["_transform_log"][-1]
    assert entry["step"] == "format_approved_ignored_coins"
    approved_diff = entry["details"]["approved_coins"]
    assert sorted(approved_diff["new"]["long"]) == ["BTC", "ETH"]
    assert sorted(approved_diff["new"]["short"]) == ["BTC", "ETH"]


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_supports_explicit_per_side_all(monkeypatch):
    async def fake_load_markets(exchange, verbose=False, quote=None):
        return {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}, "SOL/USDT:USDT": {}}

    def fake_filter_markets(markets, exchange, quote=None):
        return markets, None

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(utils, "filter_markets", fake_filter_markets)

    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": "all"},
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])

    assert config["live"]["approved_coins"]["long"] == ["BTC"]
    assert config["live"]["approved_coins"]["short"] == ["BTC", "ETH", "SOL"]


def test_load_config_migrates_legacy_global_empty_means_all_approved(tmp_path: Path, caplog):
    raw = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {
            "approved_coins": [],
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        },
        "optimize": {"bounds": {}},
    }
    cfg_path = tmp_path / "legacy_empty_means_all.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level("WARNING"):
        cfg = load_config(str(cfg_path), live_only=False, verbose=False)

    assert "empty_means_all_approved" not in cfg["live"]
    assert cfg["live"]["approved_coins"] == {"long": ["all"], "short": ["all"]}
    assert cfg["_coins_sources"]["approved_coins"] == "all"
    assert cfg["_raw"]["live"]["empty_means_all_approved"] is True
    changes = cfg["_transform_log"][-1]["details"]["changes"]
    assert any(
        event["action"] == "remove" and event["path"] == "live.empty_means_all_approved"
        for event in changes
    )
    assert any(
        event["action"] == "update"
        and event["path"] == "live.approved_coins"
        and event["new"] == "all"
        for event in changes
    )
    messages = [rec.message for rec in caplog.records]
    assert any("live.empty_means_all_approved is deprecated" in msg for msg in messages)


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_supports_migrated_all(monkeypatch):
    async def fake_load_markets(exchange, verbose=False, quote=None):
        return {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}

    def fake_filter_markets(markets, exchange, quote=None):
        return markets, None

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(utils, "filter_markets", fake_filter_markets)

    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])

    assert config["live"]["approved_coins"] == {
        "long": ["BTC", "ETH"],
        "short": ["BTC", "ETH"],
    }


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_explicit_empty_side_stays_disabled(monkeypatch):
    async def fake_load_markets(exchange, verbose=False, quote=None):
        return {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}

    def fake_filter_markets(markets, exchange, quote=None):
        return markets, None

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(utils, "filter_markets", fake_filter_markets)

    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])

    assert config["live"]["approved_coins"] == {
        "long": ["BTC"],
        "short": [],
    }


def test_load_config_preserves_external_coin_sources(tmp_path: Path):
    approved_file = tmp_path / "approved.hjson"
    approved_file.write_text('["BTC","ETH"]')
    ignored_file = tmp_path / "ignored.hjson"
    ignored_file.write_text('["DOGE"]')

    template_path = Path("configs/examples/default_trailing_grid_long_npos7.json")
    data = hjson.loads(template_path.read_text())
    data["live"]["approved_coins"] = str(approved_file)
    data["live"]["ignored_coins"] = str(ignored_file)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))

    cfg = load_config(str(cfg_path), live_only=False, verbose=False)
    assert cfg["_coins_sources"]["approved_coins"] == str(approved_file)
    assert cfg["_coins_sources"]["ignored_coins"] == str(ignored_file)
    assert cfg["live"]["approved_coins"]["long"] == ["BTC", "ETH"]
    assert cfg["live"]["ignored_coins"]["long"] == ["DOGE"]


def test_load_config_preserves_partial_per_side_coin_sources(tmp_path: Path):
    template_path = Path("configs/examples/default_trailing_grid_long_npos7.json")
    data = hjson.loads(template_path.read_text())
    data["live"]["approved_coins"] = {"long": ["BTC"]}
    data["live"]["ignored_coins"] = {"long": [], "short": []}

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))

    cfg = load_config(str(cfg_path), live_only=False, verbose=False)

    assert cfg["_coins_sources"]["approved_coins"] == {"long": ["BTC"]}
    assert cfg["live"]["approved_coins"]["long"] == ["BTC"]
    assert cfg["live"]["approved_coins"]["short"] == []


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_supports_fake_exchange_all(tmp_path: Path):
    scenario_path = tmp_path / "fake_scenario.hjson"
    scenario_path.write_text(
        hjson.dumps(
            {
                "name": "approved_coins_fake_all",
                "exchange": "fake",
                "start_time": "2026-01-01T00:00:00Z",
                "tick_interval_seconds": 60,
                "boot_index": 1,
                "account": {"balance": 1000.0},
                "symbols": {
                    "BTC/USDT:USDT": {
                        "price_step": 0.1,
                        "qty_step": 0.001,
                        "min_qty": 0.001,
                        "min_cost": 5.0,
                    },
                    "ETH/USDT:USDT": {
                        "price_step": 0.01,
                        "qty_step": 0.001,
                        "min_qty": 0.001,
                        "min_cost": 5.0,
                    },
                    "SOL/USDT:USDT": {
                        "price_step": 0.01,
                        "qty_step": 0.01,
                        "min_qty": 0.01,
                        "min_cost": 5.0,
                    },
                },
                "timeline": [
                    {
                        "t": 0,
                        "prices": {
                            "BTC/USDT:USDT": 100000.0,
                            "ETH/USDT:USDT": 2500.0,
                            "SOL/USDT:USDT": 180.0,
                        },
                    },
                    {
                        "t": 1,
                        "prices": {
                            "BTC/USDT:USDT": 100100.0,
                            "ETH/USDT:USDT": 2510.0,
                            "SOL/USDT:USDT": 181.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": "all"},
            "ignored_coins": {"long": [], "short": ["SOL"]},
            "fake_scenario_path": str(scenario_path),
        }
    }

    await format_approved_ignored_coins(config, ["fake"], quote="USDT")

    assert config["live"]["approved_coins"] == {
        "long": ["BTC"],
        "short": ["BTC", "ETH", "SOL"],
    }
    assert config["live"]["ignored_coins"] == {"long": [], "short": ["SOL"]}


@pytest.mark.asyncio
async def test_format_approved_ignored_coins_supports_fake_exchange_all_via_api_keys(
    tmp_path: Path, monkeypatch
):
    scenario_path = tmp_path / "fake_scenario_via_api_keys.hjson"
    scenario_path.write_text(
        hjson.dumps(
            {
                "name": "approved_coins_fake_all_api_keys",
                "exchange": "fake",
                "start_time": "2026-01-01T00:00:00Z",
                "tick_interval_seconds": 60,
                "boot_index": 1,
                "account": {"balance": 1000.0},
                "symbols": {
                    "BTC/USDT:USDT": {
                        "price_step": 0.1,
                        "qty_step": 0.001,
                        "min_qty": 0.001,
                        "min_cost": 5.0,
                    },
                    "ETH/USDT:USDT": {
                        "price_step": 0.01,
                        "qty_step": 0.001,
                        "min_qty": 0.001,
                        "min_cost": 5.0,
                    },
                },
                "timeline": [
                    {
                        "t": 0,
                        "prices": {
                            "BTC/USDT:USDT": 100000.0,
                            "ETH/USDT:USDT": 2500.0,
                        },
                    },
                    {
                        "t": 1,
                        "prices": {
                            "BTC/USDT:USDT": 100100.0,
                            "ETH/USDT:USDT": 2510.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        procedures,
        "load_user_info",
        lambda user: {
            "exchange": "fake",
            "quote": "USDT",
            "fake_scenario_path": str(scenario_path),
        },
    )

    config = {
        "live": {
            "user": "fake_api_keys_user",
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["fake"], quote="USDT")

    assert config["live"]["approved_coins"] == {
        "long": ["BTC", "ETH"],
        "short": ["BTC", "ETH"],
    }
