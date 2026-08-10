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
async def test_unqualified_native_id_collision_across_exchanges_requires_scope(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "id": "12345",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
        "bybit": {
            "OTHER/USDT:USDT": {
                "id": "12345",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "OTHER",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["12345"], "short": ["12345"]},
            "ignored_coins": {"long": [], "short": []},
        }
    }

    with pytest.raises(
        utils.AmbiguousMarketIdentifier,
        match="different contracts across configured exchanges",
    ):
        await format_approved_ignored_coins(config, ["bitget", "bybit"])


@pytest.mark.asyncio
async def test_cross_exchange_native_id_collision_preserves_multiplier_identity(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "id": "SAME-ID",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
        "bybit": {
            "1000ABC/USDT:USDT": {
                "id": "SAME-ID",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000ABC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)

    with pytest.raises(
        utils.AmbiguousMarketIdentifier,
        match="different contracts across configured exchanges",
    ):
        await utils.reject_cross_exchange_market_identifier_collisions(
            ["SAME-ID"], ["bitget", "bybit"]
        )


@pytest.mark.asyncio
async def test_cross_exchange_lowercase_native_id_preserves_multiplier_identity(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "id": "same-id",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
        "bybit": {
            "1000ABC/USDT:USDT": {
                "id": "same-id",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000ABC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)

    with pytest.raises(
        utils.AmbiguousMarketIdentifier,
        match="different contracts across configured exchanges",
    ):
        await utils.reject_cross_exchange_market_identifier_collisions(
            ["same-id"], ["bitget", "bybit"]
        )


@pytest.mark.asyncio
async def test_cross_exchange_plain_alias_resolves_each_venue_denomination(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "id": "ABCUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
        "bybit": {
            "1000ABC/USDT:USDT": {
                "id": "1000ABCUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000ABC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)

    await utils.reject_cross_exchange_market_identifier_collisions(
        ["ABC"], ["bitget", "bybit"]
    )

    assert utils.coin_to_symbol("ABC", "bitget") == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("ABC", "bybit") == "1000ABC/USDT:USDT"


@pytest.mark.asyncio
async def test_user_plain_alias_remains_plain_when_matching_native_ids(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "id": "ABC",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
        "bybit": {
            "1000ABC/USDT:USDT": {
                "id": "ABC",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000ABC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["ABC"], "short": ["ABC"]},
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget", "bybit"])

    assert utils.coin_to_symbol("ABC", "bitget") == "ABC/USDT:USDT"
    assert utils.coin_to_symbol("ABC", "bybit") == "1000ABC/USDT:USDT"


@pytest.mark.asyncio
async def test_cross_exchange_plain_alias_unifies_prefix_and_suffix_denominations(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "binance": {
            "1000SHIB/USDT:USDT": {
                "id": "1000SHIBUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000SHIB",
            }
        },
        "bybit": {
            "SHIB1000/USDT:USDT": {
                "id": "SHIB1000USDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "SHIB1000",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)

    await utils.reject_cross_exchange_market_identifier_collisions(
        ["SHIB"], ["binance", "bybit"]
    )

    assert utils.coin_to_symbol("SHIB", "binance") == "1000SHIB/USDT:USDT"
    assert utils.coin_to_symbol("SHIB", "bybit") == "SHIB1000/USDT:USDT"


@pytest.mark.asyncio
async def test_approved_all_plain_alias_is_not_upgraded_by_coincident_native_id(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "binance": {
            "1000ABC/USDT:USDT": {
                "id": "1000ABCUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000ABC",
            }
        },
        "hyperliquid": {
            "ABC/USDC:USDC": {
                "id": "ABC",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "ABC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["binance", "hyperliquid"])

    assert config["live"]["approved_coins"] == {
        "long": ["ABC"],
        "short": ["ABC"],
    }


@pytest.mark.asyncio
async def test_cross_exchange_native_id_allows_quote_variants_of_same_underlying(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bybit": {
            "BTC/USDT:USDT": {
                "id": "SAME-ID",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "BTC",
            }
        },
        "hyperliquid": {
            "BTC/USDC:USDC": {
                "id": "SAME-ID",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "BTC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)

    await utils.reject_cross_exchange_market_identifier_collisions(
        ["SAME-ID"], ["bybit", "hyperliquid"]
    )


@pytest.mark.asyncio
async def test_approved_aliases_for_one_market_are_coalesced(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "BTC",
        }
    }

    assert utils.create_coin_symbol_map_cache("bitget", markets, verbose=False)
    config = {
        "live": {
            "approved_coins": {
                "long": ["BTC", "BTCUSDT"],
                "short": ["BTCUSDT", "BTC"],
            },
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"] == {
        "long": ["BTC"],
        "short": ["BTC"],
    }


@pytest.mark.asyncio
async def test_backtest_approved_aliases_use_source_key_from_override_only_venue(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bybit": {
            "EDGE/USDT:USDT": {
                "id": "EDGEUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "EDGE",
            }
        },
        "kucoin": {
            "EDGE/USDT:USDT": {
                "id": "EDGEUSDTM",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "EDGE",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "backtest": {"coin_sources": {"EDGEUSDTM": "kucoin"}},
        "live": {
            "approved_coins": {
                "long": ["EDGE"],
                "short": ["EDGEUSDTM"],
            },
            "ignored_coins": {"long": [], "short": []},
        },
    }

    await format_approved_ignored_coins(
        config, ["bybit"], prefer_backtest_coin_source_keys=True
    )

    assert config["live"]["approved_coins"] == {
        "long": ["EDGEUSDTM"],
        "short": ["EDGEUSDTM"],
    }


@pytest.mark.asyncio
async def test_live_approved_alias_ignores_backtest_coin_source(monkeypatch):
    async def unexpected_load_markets(exchange, verbose=False, quote=None):
        pytest.fail(f"live formatting must not load backtest-only source venue {exchange}")

    monkeypatch.setattr(utils, "load_markets", unexpected_load_markets)
    config = {
        "backtest": {"coin_sources": {"EDGEUSDTM": "kucoin"}},
        "live": {
            "approved_coins": {"long": ["EDGE"], "short": ["EDGE"]},
            "ignored_coins": {"long": [], "short": []},
        },
    }

    await format_approved_ignored_coins(config, ["bybit"])

    assert config["live"]["approved_coins"] == {
        "long": ["EDGE"],
        "short": ["EDGE"],
    }


@pytest.mark.asyncio
async def test_backtest_ignored_alias_resolves_on_source_only_venue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bybit": {},
        "kucoin": {
            "EDGE/USDT:USDT": {
                "id": "EDGEUSDTM",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "EDGE",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "backtest": {"coin_sources": {"EDGEUSDTM": "kucoin"}},
        "live": {
            "approved_coins": {
                "long": ["EDGEUSDTM"],
                "short": ["EDGEUSDTM"],
            },
            "ignored_coins": {"long": ["EDGE"], "short": ["EDGE"]},
        },
    }

    await format_approved_ignored_coins(
        config, ["bybit"], prefer_backtest_coin_source_keys=True
    )

    assert config["live"]["approved_coins"] == {"long": [], "short": []}


@pytest.mark.asyncio
async def test_exact_ignored_identifier_removes_matching_approved_alias(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "BTC",
        }
    }
    utils.create_coin_symbol_map_cache("bitget", markets, verbose=False)

    async def fake_load_markets(_exchange, verbose=False, quote=None):
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": ["BTC"]},
            "ignored_coins": {"long": ["BTCUSDT"], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"] == {"long": [], "short": ["BTC"]}
    assert config["live"]["ignored_coins"] == {"long": ["BTCUSDT"], "short": []}


@pytest.mark.asyncio
async def test_ignored_alias_removes_matching_exact_approval(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "BTC",
        }
    }
    utils.create_coin_symbol_map_cache("bitget", markets, verbose=False)

    async def fake_load_markets(_exchange, verbose=False, quote=None):
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["BTCUSDT"], "short": []},
            "ignored_coins": {"long": ["BTC"], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"]["long"] == []


@pytest.mark.asyncio
async def test_ignored_multiplier_alias_removes_matching_canonical_approval(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "1000SHIB/USDT:USDT": {
            "id": "1000SHIBUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "1000SHIB",
        }
    }
    utils.create_coin_symbol_map_cache("bitget", markets, verbose=False)

    async def fake_load_markets(_exchange, verbose=False, quote=None):
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["SHIB"], "short": []},
            "ignored_coins": {"long": ["1000SHIB"], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"]["long"] == []


@pytest.mark.asyncio
async def test_exact_ignored_identifier_does_not_remove_distinct_multiplier_market(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "ABC/USDT:USDT": {
            "id": "ABCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "ABC",
        },
        "1000ABC/USDT:USDT": {
            "id": "1000ABCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "1000ABC",
        },
    }
    utils.create_coin_symbol_map_cache("bitget", markets, verbose=False)

    async def fake_load_markets(_exchange, verbose=False, quote=None):
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    config = {
        "live": {
            "approved_coins": {"long": ["ABC/USDT:USDT"], "short": []},
            "ignored_coins": {"long": ["1000ABCUSDT"], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"]["long"] == ["ABC/USDT:USDT"]


@pytest.mark.asyncio
async def test_scoped_ignore_preserves_other_venues_from_approved_all(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "binance": {
            "1000SHIB/USDT:USDT": {
                "id": "1000SHIBUSDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "1000SHIB",
            }
        },
        "bybit": {
            "SHIB1000/USDT:USDT": {
                "id": "SHIB1000USDT",
                "swap": True,
                "linear": True,
                "active": True,
                "base": "SHIB1000",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {
                "long": ["binance::1000SHIBUSDT"],
                "short": [],
            },
        }
    }

    await format_approved_ignored_coins(config, ["binance", "bybit"])

    assert config["live"]["approved_coins"] == {
        "long": ["bybit::SHIB1000USDT"],
        "short": ["SHIB"],
    }


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
async def test_format_approved_all_uses_plain_id_for_denomination_variants(monkeypatch):
    markets = {
        "ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC",
            "id": "ABCUSDT",
        },
        "1000ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000ABC",
            "id": "1000ABCUSDT",
        },
        "BTC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "BTC",
            "id": "BTCUSDT",
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        return markets

    def fake_filter_markets(loaded_markets, exchange, quote=None):
        return loaded_markets, None

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(utils, "filter_markets", fake_filter_markets)

    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    expected = ["ABC", "BTC"]
    assert config["live"]["approved_coins"] == {
        "long": expected,
        "short": expected,
    }


@pytest.mark.asyncio
async def test_format_approved_all_preserves_distinct_numeric_suffix_ticker(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "1000ABC/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "1000ABC",
            "id": "1000ABCUSDT",
        },
        "ABC1000/USDT:USDT": {
            "swap": True,
            "linear": True,
            "base": "ABC1000",
            "id": "ABC1000USDT",
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda loaded, exchange, quote=None: (loaded, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    expected = ["ABC", "ABC1000"]
    assert config["live"]["approved_coins"] == {
        "long": expected,
        "short": expected,
    }


@pytest.mark.asyncio
async def test_format_coin_lists_preserves_distinct_exact_identifiers_and_coalesces_aliases(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets = {
        "ABC/USDT:USDT": {
            "id": "ABCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "ABC",
        },
        "1000ABC/USDT:USDT": {
            "id": "1000ABCUSDT",
            "swap": True,
            "linear": True,
            "active": True,
            "base": "1000ABC",
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        assert exchange == "bitget"
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    identifiers = ["bitget::ABCUSDT", "1000ABC/USDT:USDT", "ABCUSDT"]
    config = {
        "live": {
            "approved_coins": {"long": identifiers, "short": []},
            "ignored_coins": {"long": [], "short": identifiers},
        }
    }

    await format_approved_ignored_coins(config, ["bitget"])

    assert config["live"]["approved_coins"]["long"] == identifiers[:2]
    assert config["live"]["ignored_coins"]["short"] == identifiers


@pytest.mark.asyncio
async def test_format_approved_all_selects_one_denomination_per_underlying(monkeypatch):
    markets_by_exchange = {
        "bitget": {
            "ABC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "ABC",
                "id": "ABCUSDT",
            },
            "1000ABC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "1000ABC",
                "id": "1000ABCUSDT",
            },
            "BTC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "BTC",
                "id": "BTCUSDT",
            },
        },
        "bybit": {
            "ABC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "ABC",
                "id": "ABCUSDT",
            },
            "BTC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "BTC",
                "id": "BTCUSDT",
            },
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        return markets_by_exchange[exchange]

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bitget", "bybit"])

    expected = ["ABC", "BTC"]
    assert config["live"]["approved_coins"] == {
        "long": expected,
        "short": expected,
    }


@pytest.mark.asyncio
async def test_format_approved_all_unifies_cross_exchange_scaled_contracts(monkeypatch):
    markets_by_exchange = {
        "bybit": {
            "ABC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "ABC",
                "id": "ABCUSDT",
            }
        },
        "bitget": {
            "1000ABC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "1000ABC",
                "id": "1000ABCUSDT",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        return markets_by_exchange[exchange]

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bybit", "bitget"])

    expected = ["ABC"]
    assert config["live"]["approved_coins"] == {
        "long": expected,
        "short": expected,
    }


@pytest.mark.asyncio
async def test_format_approved_all_unifies_cross_exchange_quote_variants(monkeypatch):
    markets_by_exchange = {
        "bybit": {
            "BTC/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "BTC",
                "id": "BTCUSDT",
            }
        },
        "hyperliquid": {
            "BTC/USDC:USDC": {
                "swap": True,
                "linear": True,
                "base": "BTC",
                "id": "BTC",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        return markets_by_exchange[exchange]

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(config, ["bybit", "hyperliquid"])

    assert config["live"]["approved_coins"] == {
        "long": ["BTC"],
        "short": ["BTC"],
    }


@pytest.mark.asyncio
async def test_format_approved_all_unifies_exchange_denomination_conventions(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    markets_by_exchange = {
        "bybit": {
            "SHIB1000/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "SHIB1000",
                "id": "SHIB1000USDT",
            }
        },
        "binance": {
            "1000SHIB/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "1000SHIB",
                "id": "1000SHIBUSDT",
            }
        },
        "bitget": {
            "SHIB/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "SHIB",
                "id": "SHIBUSDT",
            }
        },
        "hyperliquid": {
            "KSHIB/USDC:USDC": {
                "swap": True,
                "linear": True,
                "base": "KSHIB",
                "baseName": "kSHIB",
                "id": "38",
            }
        },
        "mexc": {
            "SHIB100000/USDT:USDT": {
                "swap": True,
                "linear": True,
                "base": "SHIB100000",
                "baseName": "SHIB",
                "id": "SHIB100000USDT",
            }
        },
    }

    async def fake_load_markets(exchange, verbose=False, quote=None):
        markets = markets_by_exchange[exchange]
        assert utils.create_coin_symbol_map_cache(
            exchange, markets, quote=quote, verbose=False
        )
        return markets

    monkeypatch.setattr(utils, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        utils, "filter_markets", lambda markets, exchange, quote=None: (markets, None)
    )
    config = {
        "live": {
            "approved_coins": "all",
            "ignored_coins": {"long": [], "short": []},
        }
    }

    await format_approved_ignored_coins(
        config, ["binance", "bybit", "bitget", "hyperliquid", "mexc"]
    )

    assert config["live"]["approved_coins"] == {
        "long": ["SHIB"],
        "short": ["SHIB"],
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

    template_path = Path("configs/examples/default_trailing_martingale_long.json")
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
    template_path = Path("configs/examples/default_trailing_martingale_long.json")
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
