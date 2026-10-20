"""Descriptor regressions grounded in canonical export and record consumers."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from config.field_registry import get_field_registry
from config.field_spec import FieldType
from config.load import prepare_config
from config.scoring import extract_objective_specs, resolve_objective_basis
from config.schema import get_template_config
from suite_runner import build_scenarios, extract_suite_config
from tools.clean_config import render_source


def _roundtrip(source):
    original = deepcopy(source)
    rendered = render_source(json.dumps(source), source=Path("input.json"), mode="full")
    assert source == original
    assert render_source(rendered, source=Path("output.json"), mode="full") == rendered
    return json.loads(rendered)


def test_startup_budget_records_keep_integer_millisecond_leaves():
    source = get_template_config()
    source["live"]["startup_phase_budgets"] = {
        "account": {"elapsed_ms": 120_000, "since_previous_ms": 60_000},
        "full-warmup": {"elapsed_ms": 900_000},
    }
    registry = get_field_registry()
    for phase, budgets in source["live"]["startup_phase_budgets"].items():
        record = registry.find(("live", "startup_phase_budgets", phase))
        assert record.types == (FieldType.OBJECT,)
        assert record.numeric_type is None
        for key in budgets:
            field = registry.find(("live", "startup_phase_budgets", phase, key))
            assert field.numeric_type is int
            assert field.unit == "milliseconds"
            assert field.domain.minimum == 0
    exported = _roundtrip(source)
    assert exported["live"]["startup_phase_budgets"] == source["live"]["startup_phase_budgets"]
    assert all(type(value) is int
               for budget in exported["live"]["startup_phase_budgets"].values()
               for value in budget.values())
    assert prepare_config(exported, verbose=False)["live"]["startup_phase_budgets"] == (
        source["live"]["startup_phase_budgets"]
    )


@pytest.mark.parametrize("field_name", ["coins", "ignored_coins"])
@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("BTC", ["BTC"]),
    (["BTC", "ETH"], ["BTC", "ETH"]),
    ({"long": ["BTC"], "short": ["ETH"]}, ["BTC", "ETH"]),
])
def test_scenario_coin_forms_preserve_authored_form_and_suite_meaning(field_name, value, expected):
    field = get_field_registry().find(("backtest", "scenarios", 0, field_name))
    assert set(field.types) == {FieldType.ARRAY, FieldType.STRING, FieldType.OBJECT}
    assert field.nullable
    source = get_template_config()
    source["backtest"]["scenarios"] = [{"label": "base", field_name: value}]
    exported = _roundtrip(source)
    assert exported["backtest"]["scenarios"] == source["backtest"]["scenarios"]
    scenarios, _ = build_scenarios(extract_suite_config(exported, None))
    assert getattr(scenarios[0], field_name) == expected


@pytest.mark.parametrize("value,expected", [
    (None, ["binance", "bybit"]),
    ("bybit", ["bybit"]),
    (["bybit"], ["bybit"]),
])
def test_scenario_exchange_forms_preserve_authored_form_and_inheritance(value, expected):
    field = get_field_registry().find(("backtest", "scenarios", 0, "exchanges"))
    assert set(field.types) == {FieldType.ARRAY, FieldType.STRING}
    assert field.nullable
    source = get_template_config()
    source["backtest"]["scenarios"] = [{"label": "base", "exchanges": value}]
    exported = _roundtrip(source)
    assert exported["backtest"]["scenarios"] == source["backtest"]["scenarios"]
    scenarios, _ = build_scenarios(extract_suite_config(exported, None))
    assert scenarios[0].exchanges == expected


def test_nullable_scenario_record_fields_roundtrip_to_suite_inheritance():
    source = get_template_config()
    nullable_fields = {"label": None, "start_date": None, "end_date": None, "coin_sources": None}
    source["backtest"]["scenarios"] = [nullable_fields]
    registry = get_field_registry()
    for name in nullable_fields:
        assert registry.find(("backtest", "scenarios", 0, name)).nullable
    exported = _roundtrip(source)
    assert exported["backtest"]["scenarios"] == [nullable_fields]
    scenarios, _ = build_scenarios(extract_suite_config(exported, None))
    assert scenarios[0].label == "scenario_01"
    assert scenarios[0].start_date is None
    assert scenarios[0].end_date is None
    assert scenarios[0].coin_sources is None


def test_scoring_and_limit_scenario_null_remains_distinct_from_omission():
    source = get_template_config()
    source["optimize"]["objective_scenario"] = "base"
    source["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max", "scenario": None},
        {"metric": "mdg_strategy_eq", "goal": "max"},
    ]
    source["optimize"]["limits"] = [
        {"metric": "drawdown_worst_strategy_eq", "penalize_if": "greater_than", "value": 0.8,
         "scenario": None},
        {"metric": "backtest_completion_ratio", "penalize_if": "less_than", "value": 0.99},
    ]
    registry = get_field_registry()
    for section in ("scoring", "limits"):
        field = registry.find(("optimize", section, 0, "scenario"))
        assert field.types == (FieldType.STRING,)
        assert field.nullable
    exported = _roundtrip(source)
    for section in ("scoring", "limits"):
        assert exported["optimize"][section][0]["scenario"] is None
        assert "scenario" not in exported["optimize"][section][1]
    specs = extract_objective_specs(exported)
    bases = [resolve_objective_basis(spec, default_scenario="base", reducer_cfg={"default": "mean"})
             for spec in specs]
    assert bases[0].scenario is None
    assert bases[0].reducer == "mean"
    assert bases[1].scenario == "base"
    assert bases[1].reducer is None


def test_adjacent_reducer_nulls_are_omitted_and_scenario_override_null_is_rejected():
    source = get_template_config()
    source["optimize"]["scoring"][0]["reducer"] = None
    source["optimize"]["limits"][0]["reducer"] = None
    registry = get_field_registry()
    for section in ("scoring", "limits"):
        assert not registry.find(("optimize", section, 0, "reducer")).nullable
    exported = _roundtrip(source)
    for section in ("scoring", "limits"):
        assert "reducer" not in exported["optimize"][section][0]

    assert not registry.find(("backtest", "scenarios", 0, "overrides")).nullable
    source["backtest"]["scenarios"] = [{"label": "base", "overrides": None}]
    with pytest.raises(TypeError, match="overrides.*dict"):
        render_source(json.dumps(source), source=Path("input.json"), mode="full")
