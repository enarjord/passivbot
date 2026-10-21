"""Canonical operational ownership and explicit released-version boundaries."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.access import require_live_value, require_config_value
from config.field_registry import (
    LIVE_OPERATIONAL_PATHS,
    get_field_registry,
    live_field_path,
)
from config.migrations.v9 import migrate_v8_to_v9
from config.schema import get_template_config
from released_v8_fixtures import released_v8_policy


def test_all_operational_fields_have_one_grouped_path_and_registry_descriptor():
    template = get_template_config()
    assert set(template["live"]) == {
        "account",
        "api",
        "candles",
        "startup",
        "fills",
        "execution",
        "recovery",
        "modes",
    }
    assert len(LIVE_OPERATIONAL_PATHS) == 46
    registry = get_field_registry()
    for field_id, path in LIVE_OPERATIONAL_PATHS.items():
        assert len(path) == 3
        assert registry.find(path) is not None
        assert require_live_value(template, field_id) == require_config_value(
            template, live_field_path(field_id)
        )
        assert field_id not in template["live"]
    assert (
        prepare_config(template, verbose=False)["live"]["account"]
        == template["live"]["account"]
    )


@pytest.mark.parametrize("field", sorted(LIVE_OPERATIONAL_PATHS))
@pytest.mark.parametrize("location", ["root", "fixed", "scenario", "selector", "coin"])
def test_ungrouped_development_paths_are_rejected_without_loading_aliases(
    field, location
):
    config = get_template_config()
    if location == "root":
        config["live"][field] = None
    elif location == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {f"live.{field}": None}
    elif location == "scenario":
        config["backtest"]["scenarios"] = [
            {"label": "ungrouped", "overrides": {f"live.{field}": None}}
        ]
    elif location == "selector":
        config["optimize"]["fixed_params"] = [f"live.{field}"]
    else:
        config["coin_overrides"] = {"BTC": {"live": {field: None}}}
    with pytest.raises(ValueError, match="ungrouped live field"):
        prepare_config(config, verbose=False)


def test_released_operational_fields_move_with_exact_values_and_reports():
    source = released_v8_policy()
    original = deepcopy(source)
    result = migrate_v8_to_v9(source, memory_span_minutes=100.0)
    assert source == original
    converted_sentinels = {
        "warmup_concurrency": "auto",
        "forced_mode_long": None,
        "forced_mode_short": None,
    }
    for old, path in LIVE_OPERATIONAL_PATHS.items():
        if old in source["live"]:
            assert (
                require_config_value(result.config, ".".join(path))
                == converted_sentinels.get(old, source["live"][old])
            )
            assert any(
                change["path"] == f"live.{old}"
                and change["reason"] == "grouped released operational live field"
                for change in result.changes
            )
            if old in converted_sentinels:
                assert any(
                    change["path"] == ".".join(path)
                    and change["old"] == source["live"][old]
                    and change["new"] == converted_sentinels[old]
                    and change["reason"] == "explicit released-v8 control sentinel conversion"
                    for change in result.changes
                )
    assert result.config["live"]["modes"]["disapproved_position"] == "graceful_stop"
    assert (
        prepare_config(result.config, verbose=False)["live"]["modes"]
        == result.config["live"]["modes"]
    )


def test_coin_file_operational_values_keep_inline_precedence_after_grouping(tmp_path):
    from tools.migrate_config_v9 import migrate

    (tmp_path / "coin.json").write_text(
        json.dumps({"live": {"forced_mode_long": "normal", "leverage": 3}})
    )
    source = released_v8_policy()
    source["coin_overrides"] = {
        "BTC": {
            "override_config_path": "coin.json",
            "live": {"forced_mode_long": "manual", "leverage": 7},
        }
    }
    result = migrate(
        source,
        memory_span_minutes=100.0,
        base_config_path=str(tmp_path / "config.json"),
    )
    assert result.config["coin_overrides"]["BTC"]["live"] == {
        "modes": {"long": "manual"},
        "account": {"leverage": 7},
    }
    assert result.config["bot"]["long"]["forager"]["always_include"] == []


def test_cli_projection_keeps_nested_simulation_fee_settings():
    from config_utils import project_template_config_for_cli

    for command in ("backtest", "optimize"):
        projected = project_template_config_for_cli(get_template_config(), command)
        assert set(projected["live"]) == {"fills"}
        assert set(projected["live"]["fills"]) == {
            "fee_pct_fallback",
            "fee_pct_sanity_abs_max",
        }


@pytest.mark.parametrize(
    "exchange,seconds",
    [
        ("bybit", 0.2),
        ("hyperliquid", 0.2),
        ("kucoin", 0.2),
        ("binance", 0.0),
        ("fake", 0.0),
    ],
)
def test_exchange_candle_fetch_pacing_keeps_its_existing_default(exchange, seconds):
    from types import SimpleNamespace
    from passivbot import Passivbot

    assert (
        Passivbot._get_fetch_delay_seconds(SimpleNamespace(exchange=exchange))
        == seconds
    )


def test_internal_live_value_calls_use_declared_field_ids():
    import ast
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1] / "src"
    missing = []
    for path in source_root.rglob("*.py"):
        if "migrations" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            index = (
                1
                if name in {"require_live_value", "get_optional_live_value"}
                else 0 if name == "live_value" else None
            )
            if (
                index is None
                or len(node.args) <= index
                or not isinstance(node.args[index], ast.Constant)
            ):
                continue
            field_id = node.args[index].value
            if (
                field_id not in LIVE_OPERATIONAL_PATHS
                and field_id != "base_config_path"
            ):
                missing.append(
                    f"{path.relative_to(source_root)}:{node.lineno}: {field_id}"
                )
    assert missing == []
