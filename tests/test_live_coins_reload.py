import asyncio
import json
from pathlib import Path

import pytest

from config_utils import load_config
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
            "empty_means_all_approved": False,
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
            "empty_means_all_approved": False,
        }
    }

    await format_approved_ignored_coins(config, ["binanceusdm"])
    assert config["live"]["ignored_coins"]["long"] == ["DOGE"]

    ignored_file.write_text('["DOGE","SHIB"]')
    refreshed = normalize_coins_source(config["_coins_sources"]["ignored_coins"])

    assert set(refreshed["long"]) == {"DOGE", "SHIB"}, "ignored coins did not reload from file"


def test_load_config_preserves_external_coin_sources(tmp_path: Path):
    approved_file = tmp_path / "approved.hjson"
    approved_file.write_text('["BTC","ETH"]')
    ignored_file = tmp_path / "ignored.hjson"
    ignored_file.write_text('["DOGE"]')

    template_path = Path("configs/template.json")
    data = json.loads(template_path.read_text())
    data["live"]["approved_coins"] = str(approved_file)
    data["live"]["ignored_coins"] = str(ignored_file)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))

    cfg = load_config(str(cfg_path), live_only=False, verbose=False)
    assert cfg["_coins_sources"]["approved_coins"] == str(approved_file)
    assert cfg["_coins_sources"]["ignored_coins"] == str(ignored_file)
    assert cfg["live"]["approved_coins"]["long"] == ["BTC", "ETH"]
    assert cfg["live"]["ignored_coins"]["long"] == ["DOGE"]
