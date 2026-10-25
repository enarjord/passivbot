"""Prefilter quantities preserve native measurements under current field names."""

from copy import deepcopy
import json

import passivbot_rust as pbr
import pytest

from config import get_template_config, prepare_config
from config.authoring import compact_config, resolved_config
from config.prefilter import PREFILTER_METRICS, resolve_prefilter
from config.field_registry import get_field_registry
from optimization.prefilter import inspect_config_prefilter
from released_v8_fixtures import released_v8_policy
from tools.clean_config import cleanup_config
from tools.gpu_parity import build_parser, fixture_inputs
from tools.migrate_config_v9 import migrate


KNOBS = [
    ("exposure_ratios", "exposure_probe_fractions", [0.25, 0.875]),
    ("volatility_ema_1m", "log_range_ema_1m", 0.007),
    ("volatility_ema_1h", "log_range_ema_1h", 0.017),
    ("adverse_directionality", "adverse_directionality_score", 0.375),
]
# Authored rejection spellings above are frozen separately from the current
# native probe contract; the log-range fields now use their canonical names.
NATIVE_KNOBS = {
    "exposure_probe_fractions": "exposure_ratios",
    "log_range_ema_1m": "log_range_ema_1m",
    "log_range_ema_1h": "log_range_ema_1h",
    "adverse_directionality_score": "adverse_directionality",
}
METRICS = [
    ("entry_close_span", "entry_close_span_fraction"),
    ("entry_retracement", "entry_retracement_fraction"),
    ("close_retracement", "close_retracement_fraction"),
    ("grid_span", "grid_span_fraction"),
]


def source():
    config = get_template_config()
    config["optimize"]["bounds"] = {}
    return config


@pytest.mark.parametrize("old,new,value", KNOBS)
@pytest.mark.parametrize("owner", ["resolver", "loader", "current_migration"])
def test_earlier_v9_prefilter_knobs_are_not_aliases(old, new, value, owner):
    config = source()
    policy = {old: deepcopy(value)}
    config["optimize"]["prefilter"] = policy
    before = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old):
        if owner == "resolver":
            resolve_prefilter(policy)
        elif owner == "loader":
            prepare_config(config, verbose=False)
        else:
            migrate(config)
    assert config == before


@pytest.mark.parametrize("old,new,value", KNOBS)
def test_new_prefilter_quantities_roundtrip_without_rescaling(old, new, value):
    config = source()
    config["optimize"]["prefilter"] = {new: deepcopy(value)}
    prepared = prepare_config(config, verbose=False)
    assert prepared["optimize"]["prefilter"][new] == value
    for export in (cleanup_config, compact_config, resolved_config):
        loaded = prepare_config(export(prepared), verbose=False)
        assert loaded["optimize"]["prefilter"][new] == value
    assert get_field_registry().find(("optimize", "prefilter", new)) is not None
    assert get_field_registry().find(("optimize", "prefilter", old)) is None


@pytest.mark.parametrize("old,new", METRICS)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("owner", ["resolver", "loader", "current_migration"])
def test_earlier_v9_prefilter_check_ids_are_rejected_even_when_disabled(old, new, enabled, owner):
    config = source()
    policy = {"checks": [{"metric": old, "enabled": enabled, "max": 0.25}]}
    config["optimize"]["prefilter"] = policy
    before = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match="metric|" + old):
        if owner == "resolver":
            resolve_prefilter(policy)
        elif owner == "loader":
            prepare_config(config, verbose=False)
        else:
            migrate(config)
    assert config == before


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("native_metric,public_metric", [*METRICS, ("entry_cooldown_minutes", "entry_cooldown_minutes")])
def test_new_probe_and_metric_names_preserve_actual_native_measurements(monkeypatch, side, native_metric, public_metric):
    config, *_ = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", side, "--coins", "1", "--bars", "64",
    ]))
    policy = resolve_prefilter({
        **{new: deepcopy(value) for old, new, value in KNOBS},
        "checks": [{"metric": public_metric, "enabled": True, "sides": [side], "min": 1000000.0}],
    })
    captured = []
    measurements = []
    native = pbr.config_prefilter_metrics_json
    def measure(text):
        captured.append(json.loads(text))
        output = native(text)
        measurements.append(json.loads(output))
        return output
    monkeypatch.setattr(pbr, "config_prefilter_metrics_json", measure)
    before = deepcopy(config)
    _penalty, findings = inspect_config_prefilter(config, policy)
    assert config == before and captured and findings
    for probe in captured:
        for old, new, value in KNOBS:
            native_key = NATIVE_KNOBS[new]
            assert probe[native_key] == value
            if new != native_key:
                assert new not in probe
            if old != native_key:
                assert old not in probe
                for keep_current in (False, True):
                    retired = deepcopy(probe)
                    retired[old] = deepcopy(value)
                    if not keep_current:
                        retired.pop(native_key)
                    with pytest.raises(ValueError, match=old):
                        native(json.dumps(retired))
    assert {row["metric"] for row in findings} == {public_metric}
    expected = (min(value["grid_span"] for value in measurements) if native_metric == "grid_span" else
                min(row[native_metric] for value in measurements for row in value["observations"]))
    assert findings[0]["value"] == expected
    assert public_metric in PREFILTER_METRICS


def test_released_v8_without_prefilter_gains_only_current_defaults():
    result = migrate(released_v8_policy(), memory_span_minutes=123.5)
    assert result.config["optimize"]["prefilter"] == resolve_prefilter()


@pytest.mark.parametrize("old,new,value", KNOBS)
def test_unpublished_prefilter_knobs_do_not_gain_a_released_v8_alias(old, new, value):
    config = released_v8_policy()
    config["optimize"]["prefilter"] = {old: deepcopy(value)}
    before = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old):
        migrate(config, memory_span_minutes=123.5)
    assert config == before


def test_evaluation_identity_versions_the_renamed_policy_explicitly():
    from optimization.evaluation_contract import build_evaluation_contract

    config = source()
    policy = {"checks": [{"metric": "entry_close_span_fraction", "enabled": True, "max": 0.25}]}
    config["optimize"]["prefilter"] = policy
    prepared = prepare_config(config, verbose=False)
    contract = build_evaluation_contract(prepared)
    assert contract["prefilter"] == {"version": 2, "policy": resolve_prefilter(policy)}


@pytest.fixture(scope="module")
def parser():
    from cli_utils import build_command_parser
    from config_utils import add_config_arguments

    result = build_command_parser(prog="optimize", description="", usage="", epilog="")
    add_config_arguments(result, source(), command="optimize", help_all=True)
    return result


@pytest.mark.parametrize("old,new,value", KNOBS)
def test_prefilter_cli_exposes_only_current_quantity_names(parser, old, new, value):
    text = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
    parsed = vars(parser.parse_args([f"--optimize.prefilter.{new}", text]))
    result = parsed[f"optimize.prefilter.{new}"]
    if isinstance(value, list):
        assert [float(item) for item in result] == value
    else:
        assert result == value
    with pytest.raises(SystemExit) as error:
        parser.parse_args([f"--optimize.prefilter.{old}", text])
    assert error.value.code == 2
