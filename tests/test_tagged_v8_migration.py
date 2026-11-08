"""Migration behavior for defaults and stop policies from the actual v8 tags."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import hjson
import pytest

from config import prepare_config
from config.migrations.released_v8 import get_released_v8_defaults
from config.migrations.v9 import migrate_v8_to_v9
from tools.migrate_config_v9 import main, migrate


def sparse(version="v8.1.0"):
    return {
        "config_version": version,
        "bot": {"long": {}, "short": {}},
        "live": {"hsl_signal_mode": "coin"},
        "backtest": {},
        "optimize": {"scoring": [], "limits": []},
    }


@pytest.mark.parametrize("version,positions,repair,unstuck,initial,span,count_bounds", [
    ("v8.0.0", 5, True, 0.105, 0.0727, 120.0, [4, 12, 1]),
    ("v8.1.0", 7, False, 0.041, 0.0081, 790.0, [7, 7, 1]),
    ("v8.6.0", 7, False, 0.041, 0.0081, 790.0, [7, 7, 1]),
])
def test_sparse_config_uses_its_source_trading_defaults_and_domains(
    version, positions, repair, unstuck, initial, span, count_bounds,
):
    source = sparse(version)
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    prepared = prepare_config(result.config, verbose=False)
    assert source == before
    long = prepared["bot"]["long"]
    assert long["risk"]["max_positions"] == positions
    assert long["risk"]["total_exposure_enforcer"]["enabled"] is repair
    assert long["unstuck"]["close_fraction"] == unstuck
    assert long["unstuck"]["ema_span_0_minutes"] == span
    entry = long["strategy"]["trailing_martingale"]["entry"]
    assert entry["initial_qty_fraction"] == initial
    assert entry["ema_span_0_minutes"] == span
    assert prepared["optimize"]["bounds"]["long"]["risk"]["max_positions"] == count_bounds
    short_spans = prepared["optimize"]["bounds"]["short"]["strategy"]["trailing_martingale"]["entry"]
    assert short_spans["ema_span_0_minutes"] == [100, 2880 if version == "v8.0.0" else 1440, 10]
    if version != "v8.6.0":
        frozen = get_released_v8_defaults(schema_version=version)
        assert frozen["config_version"] == version
        assert frozen["bot"]["long"]["hsl"]["restart_after_red_policy"] == "threshold"


@pytest.mark.parametrize("version", ["v8.2.0", "v8.3.0", "v8.4.0", "v8.5.0", "v8.99.0"])
def test_unverified_schema_fails_before_opening_overrides(version):
    source = sparse(version)
    source["coin_overrides"] = {"AAA": {"override_config_path": "missing.json"}}
    with pytest.raises(ValueError, match="not a verified versioned v8 schema.*original runtime"):
        migrate(source, best_effort=True)


@pytest.mark.parametrize("location", ["root", "legacy_flags", "core_legacy_flags", "scenario", "fixed"])
@pytest.mark.parametrize("best_effort", [False, True])
def test_v8_0_coin_files_require_materialization_before_opening(location, best_effort, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("v8.0.0 file opened before rejecting its hydrated semantics")

    monkeypatch.setattr("config.overrides.load_override_config", unexpected_load)
    source = sparse("v8.0.0")
    patch = {"AAA": {"override_config_path": "coin.json"}}
    if location == "root":
        source["coin_overrides"] = patch
    elif location in {"legacy_flags", "core_legacy_flags"}:
        source["live"]["coin_flags"] = {"AAA": "-lc coin.json"}
    elif location == "scenario":
        source["backtest"]["scenarios"] = [
            {"label": "coin files", "overrides": {"coin_overrides": patch}}
        ]
    else:
        source["optimize"]["fixed_runtime_overrides"] = {
            "coin_overrides.AAA.override_config_path": "coin.json"
        }
    converter = migrate_v8_to_v9 if location == "core_legacy_flags" else migrate
    with pytest.raises(ValueError, match="hydrated coin-file semantics.*Materialize the effective inline"):
        converter(source, memory_span_minutes=123.5, best_effort=best_effort)


@pytest.mark.parametrize("settings", [
    {"hsl": {"enabled": True}},
    {"hsl_enabled": True},
    {"risk_entry_cooldown_minutes": 5.0},
    {"unstuck_ema_gating_enabled": False},
    {"unstuck": {"ema_span_0": 2.0}},
])
@pytest.mark.parametrize("best_effort", [False, True])
def test_v8_0_ignored_inline_coin_policy_cannot_become_active(settings, best_effort):
    source = sparse("v8.0.0")
    source["coin_overrides"] = {"AAA": {"bot": {"long": settings}}}
    with pytest.raises(ValueError, match="not an effective coin override in v8.0.0"):
        migrate(source, memory_span_minutes=123.5, best_effort=best_effort)


def test_v8_0_materialized_inline_coin_override_migrates():
    source = sparse("v8.0.0")
    source["coin_overrides"] = {"AAA": {"bot": {"long": {"wallet_exposure_limit": 0.43}}}}
    result = migrate_v8_to_v9(source, memory_span_minutes=123.5)
    assert result.config["coin_overrides"]["AAA"]["bot"]["long"]["wallet_exposure_limit_ratio"] == 0.43


@pytest.mark.parametrize("location", ["root", "scenario", "fixed", "file"])
@pytest.mark.parametrize("best_effort", [False, True])
@pytest.mark.parametrize("authored_null", [False, True])
def test_tagged_threshold_halt_requires_explicit_acknowledgment(
    location, best_effort, authored_null, tmp_path,
):
    source = sparse()
    source["bot"]["long"]["hsl"] = {"enabled": False, "no_restart_drawdown_threshold": 0.3}
    source["bot"]["short"]["hsl"] = {"enabled": False, "restart_after_red_policy": "always"}
    if authored_null and location != "file":
        source["bot"]["long"]["hsl"]["restart_after_red_policy"] = None
    base = str(tmp_path / "source.json")
    if location == "root":
        source["bot"]["long"]["hsl"]["enabled"] = True
    elif location == "scenario":
        source["backtest"]["scenarios"] = [
            {"label": "protected", "overrides": {"bot.long.hsl.enabled": True}}
        ]
    elif location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    else:
        path = tmp_path / "coin.json"
        policy = {"enabled": True}
        if authored_null:
            policy["restart_after_red_policy"] = None
        path.write_text(json.dumps({"bot": {"long": {"hsl": policy}}}))
        source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    kwargs = dict(memory_span_minutes=123.5, base_config_path=base, best_effort=best_effort)
    before = deepcopy(source)
    with pytest.raises(ValueError, match="threshold requires --acknowledge-permanent-stop-removal"):
        migrate(source, **kwargs)
    result = migrate(source, acknowledge_never_removal=True, **kwargs)
    assert source == before
    assert "restart_after_red_policy" not in json.dumps(result.config)
    assert any(
        item["old"] in ("threshold", None) and "explicitly acknowledged" in item["reason"]
        for item in result.changes
    )


def test_tagged_unified_does_not_select_one_side_controller_or_its_bounds():
    source = sparse()
    source["live"]["hsl_signal_mode"] = "unified"
    source["optimize"]["bounds"] = {"long": {"hsl": {"red_threshold": [0.05, 0.20]}}}
    with pytest.raises(ValueError, match="unified HSL requires an explicit bot.hsl portfolio policy"):
        migrate(source, best_effort=True)

    source["bot"]["hsl"] = {
        "enabled": False, "red_threshold": 0.2, "ema_span_minutes": 60.0,
        "cooldown_minutes_after_red": 20.0, "panic_close_order_type": "limit",
    }
    with pytest.raises(ValueError, match="inactive in portfolio mode"):
        migrate(source, best_effort=True)
    source["optimize"]["bounds"] = {"hsl": {"red_threshold": [0.05, 0.20]}}
    source["optimize"]["fixed_params"] = ["bot.long.hsl.red_threshold"]
    with pytest.raises(ValueError, match="inactive in portfolio mode"):
        migrate(source, best_effort=True)
    del source["optimize"]["fixed_params"]
    result = migrate(source, best_effort=True)
    assert result.config["bot"]["portfolio"]["hsl"]["enabled"] is False
    assert result.config["optimize"]["bounds"]["portfolio"]["hsl"]["drawdown_threshold_fraction"] == [0.05, 0.20]


def test_tagged_btc_example_stale_limit_requires_explicit_metric_decision():
    source = sparse()
    source["optimize"]["limits"] = [
        {"metric": "flat_btc_balance_hours", "penalize_if": "greater_than", "value": 672},
    ]
    with pytest.raises(ValueError, match="flat_btc_balance_hours.*requires --metric-replacement"):
        migrate(source, best_effort=True)


def test_cli_acknowledges_inherited_tagged_threshold_in_best_effort(tmp_path):
    source = sparse()
    source["bot"]["long"]["hsl"] = {"enabled": True, "no_restart_drawdown_threshold": 0.3}
    original = tmp_path / "source.json"
    original.write_text(json.dumps(source))
    output = tmp_path / "v9.json"
    assert main([str(original), str(output), "--best-effort", "--acknowledge-permanent-stop-removal"]) == 0
    assert json.loads(original.read_text()) == source
    assert json.loads(output.read_text())["bot"]["long"]["hsl"]["enabled"] is True


@pytest.fixture(scope="module", params=[
    pytest.param("a0897f83932db5e6888c1c96f8f1c668d452013f", id="v8.0.0"),
    pytest.param("7af64f3e930f4cf0cbb5fb88f31e9ce9594b0eb9", id="v8.1.0"),
])
def tagged_checkout(request):
    # Read public tag examples directly from Git without duplicating them.
    explicit = os.environ.get("PASSIVBOT_V8_FIXTURE_GIT_REPO")
    repo = Path(explicit) if explicit else Path(__file__).resolve().parents[1]
    commit = request.param
    try:
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        if explicit:
            pytest.fail("PASSIVBOT_V8_FIXTURE_GIT_REPO is missing the tagged v8 source")
        pytest.skip("fetch the public v8.0.0 and v8.1.0 tags to run the original examples")
    return repo, commit


@pytest.mark.parametrize("name,decision", [
    ("default_trailing_martingale_long", None),
    ("ema_anchor", None),
    ("BTC_ETH_XRP_SOL_ADA_long", "unified HSL requires an explicit bot.hsl"),
    ("suite_example", "unified HSL requires an explicit bot.hsl"),
    ("xmr_long_short", "unified HSL requires an explicit bot.hsl"),
    ("hsl_npos1", "threshold requires --acknowledge-permanent-stop-removal"),
    ("btc_long", "flat_btc_balance_hours.*requires --metric-replacement"),
])
def test_original_tagged_example_disposition(tagged_checkout, name, decision):
    repo, commit = tagged_checkout
    source = hjson.loads(subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:configs/examples/{name}.json"],
        text=True,
    ))
    original = deepcopy(source)
    if decision is None:
        converted = migrate(source, best_effort=True)
        assert converted.config["bot"]["long"]["hsl"]["enabled"] is False
        assert source == original
        return

    with pytest.raises(ValueError, match=decision):
        migrate(source, best_effort=True)
    assert source == original
    # Simulate a maintainer's explicit portfolio choice for this test. Each
    # tagged example has identical side values for the selected fields.
    fields = (
        "enabled", "red_threshold", "ema_span_minutes",
        "cooldown_minutes_after_red", "panic_close_order_type",
    )
    chosen = {key: source["bot"]["long"]["hsl"][key] for key in fields}
    assert chosen == {key: source["bot"]["short"]["hsl"][key] for key in fields}
    source["bot"]["hsl"] = chosen
    for side in ("long", "short"):
        source["optimize"]["bounds"][side].pop("hsl")
    if name == "btc_long":
        # A new search requires freshly authored objectives and thresholds.
        source["optimize"].update(scoring=[], limits=[])
    converted = migrate(
        source, best_effort=True, acknowledge_never_removal=name == "hsl_npos1",
    )
    assert converted.config["bot"]["portfolio"]["hsl"]["enabled"] is chosen["enabled"]
    assert converted.config["bot"]["portfolio"]["hsl"]["scope"] == "portfolio"
