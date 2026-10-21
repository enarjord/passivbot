"""Released example artifacts must not replace effective v8 policy."""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from config import prepare_config
from config.json_schema import generate_schema
from config.migrations.released_v8 import get_released_v8_defaults
from config.schema import get_template_config
from tools.migrate_config_v9 import main, migrate


def released(kind="trailing_martingale"):
    source = get_released_v8_defaults(strategy_kind=kind)
    source["backtest"]["visible_metrics"] = [
        name for name in source["backtest"]["visible_metrics"] if not name.endswith("_w")
    ]
    source["optimize"]["scoring"] = [
        item for item in source["optimize"]["scoring"] if not item["metric"].endswith("_w")
    ]
    return source


def discarded(result, path, value):
    changes = [item for item in result.changes if item["path"] == path
               and item["reason"].startswith("removed unused released-v8 backtest warmup cap")]
    assert len(changes) == 1
    assert changes[0]["old"] == value
    assert changes[0]["new"] is None
    assert not changes[0].get("best_effort", False)


@pytest.mark.parametrize("best_effort", [False, True])
@pytest.mark.parametrize("live_cap", [0, 12.5, None])
@pytest.mark.parametrize("unused_cap", [0, 720])
def test_unused_backtest_cap_never_overrides_or_supplies_the_shared_live_cap(
    best_effort, live_cap, unused_cap,
):
    source = released("ema_anchor")
    source["backtest"]["max_warmup_minutes"] = unused_cap
    if live_cap is None:
        source["live"].pop("max_warmup_minutes")
    else:
        source["live"]["max_warmup_minutes"] = live_cap
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5, best_effort=best_effort)
    assert result.config["bot"]["global"]["warmup"]["max_minutes"] == (
        live_cap if live_cap else None
    )
    assert "max_warmup_minutes" not in result.config["backtest"]
    discarded(result, "backtest.max_warmup_minutes", unused_cap)
    assert source == original


@pytest.mark.parametrize("best_effort", [False, True])
@pytest.mark.parametrize("layer", ["fixed", "dotted_scenario", "nested_scenario"])
def test_ignored_override_caps_disappear_without_creating_policy(layer, best_effort):
    source = released()
    source["live"]["max_warmup_minutes"] = 12.5
    patch = {"backtest.max_warmup_minutes": 99.5}
    if layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
        path = "optimize.fixed_runtime_overrides.backtest.max_warmup_minutes"
    else:
        if layer == "nested_scenario":
            patch = {"backtest": {"max_warmup_minutes": 99.5}}
        source["backtest"]["scenarios"] = [{"label": "ignored", "overrides": patch}]
        path = "backtest.scenarios[0].overrides.backtest.max_warmup_minutes"
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5, best_effort=best_effort)
    assert result.config["bot"]["global"]["warmup"]["max_minutes"] == 12.5
    discarded(result, path, 99.5)
    converted = (result.config["optimize"]["fixed_runtime_overrides"] if layer == "fixed"
                 else result.config["backtest"]["scenarios"][0]["overrides"])
    assert converted == {}
    assert source == original


@pytest.mark.parametrize("layer", ["root", "fixed", "scenario"])
def test_current_v9_still_rejects_the_unused_released_backtest_field(layer):
    source = get_template_config()
    if layer == "root":
        source["backtest"]["max_warmup_minutes"] = 0
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"backtest.max_warmup_minutes": 0}
    else:
        source["backtest"]["scenarios"] = [{"label": "old", "overrides": {
            "backtest.max_warmup_minutes": 0,
        }}]
    original = deepcopy(source)
    with pytest.raises((KeyError, ValueError), match="max_warmup_minutes"):
        migrate(source, best_effort=True)
    assert source == original


def test_unrelated_unknown_backtest_fields_are_not_discarded():
    source = released()
    source["backtest"]["max_warmup_minute"] = 0
    with pytest.raises(ValueError, match="max_warmup_minute.*unknown"):
        migrate(source, memory_span_minutes=123.5)


def test_cli_reports_ignored_cap_and_preserves_the_input_file(tmp_path):
    source = released("ema_anchor")
    source["backtest"]["max_warmup_minutes"] = 0
    source["live"]["max_warmup_minutes"] = 17.5
    source["backtest"]["maker_fee_override"] = "0.000375"
    src, dst, report = (tmp_path / name for name in ("v8.json", "v9.json", "report.json"))
    src.write_text(json.dumps(source))
    original = src.read_bytes()
    assert main([str(src), str(dst), "--lookback-ema-span-minutes", "123.5",
                 "--report", str(report)]) == 0
    exported = json.loads(dst.read_text())
    assert type(exported["backtest"]["maker_fee_override"]) is float
    assert exported["backtest"]["maker_fee_override"] == 0.000375
    assert prepare_config(exported, verbose=False)["bot"]["global"]["warmup"]["max_minutes"] == 17.5
    assert any(item["path"] == "backtest.max_warmup_minutes"
               and item["reason"].startswith("removed unused released-v8")
               for item in json.loads(report.read_text())["changes"])
    assert src.read_bytes() == original


@pytest.mark.parametrize("visible", [None, []])
def test_visible_metric_null_and_empty_list_keep_distinct_existing_meanings(visible):
    from analysis_visibility import filter_analysis_for_visibility

    source = get_template_config()
    source["backtest"]["visible_metrics"] = visible
    source["optimize"]["scoring"] = [{"goal": "max", "metric": "adg_strategy_eq"}]
    source["optimize"]["limits"] = []
    Draft202012Validator(generate_schema()).validate(source)
    prepared = prepare_config(source, verbose=False)
    analysis = {"adg_strategy_eq": 0.1, "volume_pct_per_day_avg": 0.2}
    result = filter_analysis_for_visibility(analysis, prepared)
    assert result.analysis == (analysis if visible == [] else {"adg_strategy_eq": 0.1})
    assert source["backtest"]["visible_metrics"] is visible


def test_migration_keeps_authored_null_visible_metrics_and_emits_valid_schema():
    source = released()
    source["backtest"]["visible_metrics"] = None
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["backtest"]["visible_metrics"] is None
    Draft202012Validator(generate_schema()).validate(result.config)
    assert source == original


def test_referenced_full_config_discards_only_its_unused_backtest_cap(tmp_path):
    source = released()
    source["live"]["max_warmup_minutes"] = 12.5
    policy = tmp_path / "coin.json"
    policy.write_text(json.dumps({"backtest": {"max_warmup_minutes": 777},
                                  "live": {"forced_mode_long": "manual"}}))
    source["coin_overrides"] = {"TEST": {"override_config_path": policy.name}}
    original, original_file = deepcopy(source), policy.read_bytes()
    result = migrate(source, memory_span_minutes=123.5,
                     base_config_path=str(tmp_path / "input.json"))
    assert result.config["bot"]["global"]["warmup"]["max_minutes"] == 12.5
    assert result.config["coin_overrides"]["TEST"] == {"live": {"modes": {"long": "manual"}}}
    discarded(result, "coin_overrides.TEST.override_config_path::backtest.max_warmup_minutes", 777)
    assert source == original and policy.read_bytes() == original_file


@pytest.mark.parametrize("fee_kind", ["maker", "taker"])
@pytest.mark.parametrize("value", ["0.000375", "-0.0001", " 0 ", None])
@pytest.mark.parametrize("layer", ["root", "fixed", "scenario"])
def test_released_fee_strings_keep_actual_fee_values_and_null_inheritance(fee_kind, value, layer):
    from backtest import _resolve_backtest_fees
    from optimization.warmup import _apply_config_overrides
    from suite_runner import apply_scenario_overrides

    source = released()
    name = f"{fee_kind}_fee_override"
    selector = f"backtest.{name}"
    if layer == "root":
        source["backtest"][name] = value
        path = selector
    elif layer == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {selector: value}
        path = f"optimize.fixed_runtime_overrides.{selector}"
    else:
        source["backtest"]["scenarios"] = [{"label": "fee", "overrides": {selector: value}}]
        path = f"backtest.scenarios[0].overrides.{selector}"
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    Draft202012Validator(generate_schema()).validate(result.config)
    effective = deepcopy(result.config)
    if layer == "fixed":
        _apply_config_overrides(effective, effective["optimize"]["fixed_runtime_overrides"])
    elif layer == "scenario":
        apply_scenario_overrides(effective, effective["backtest"]["scenarios"][0]["overrides"])
    final = effective["backtest"][name]
    if value is None:
        assert final is None
    else:
        assert type(final) is float and final == float(value)
    settings = dict(coins=["TEST"], mss={"TEST": {"maker": 0.0002, "taker": 0.0005}}, fee_kind=fee_kind)
    assert _resolve_backtest_fees(**settings, override_value=final) == _resolve_backtest_fees(
        **settings, override_value=value,
    )
    if isinstance(value, str):
        change = next(item for item in result.changes if item["path"] == path and item["old"] == value)
        numeric = change["new"].get(selector) if isinstance(change["new"], dict) else change["new"]
        assert type(numeric) is float and numeric == float(value)
    assert source == original


@pytest.mark.parametrize("best_effort", [False, True])
@pytest.mark.parametrize("value", [True, False, "nan", "inf", "-inf", "not-a-fee"])
def test_invalid_released_fee_inputs_fail_without_guessing_or_mutating(value, best_effort):
    source = released()
    source["backtest"]["maker_fee_override"] = value
    original = deepcopy(source)
    with pytest.raises(ValueError, match="backtest.maker_fee_override.*(?:boolean|finite)"):
        migrate(source, memory_span_minutes=123.5, best_effort=best_effort)
    assert source == original


@pytest.mark.parametrize("metric", ["volume_pct_per_day_avg_w", "mdg_strategy_eq_w", "adg_pnl_w"])
def test_best_effort_still_requires_choices_without_supported_exponential_metrics(metric):
    source = released()
    source["optimize"]["scoring"] = [{"goal": "max", "metric": metric}]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="no supported _ew replacement.*metric-replacement"):
        migrate(source, best_effort=True)
    assert source == original
