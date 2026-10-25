"""Quantity metadata covers owned fields without taking over runtime policy."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import re

import pytest

from config.metric_definitions import METRIC_DEFINITIONS, describe_metric, get_metric_definition
from config.metrics import ANALYSIS_SHARED_KEYS, CURRENCY_METRICS, metric_with_curve
from tools.generate_metric_reference import DEFAULT_OUTPUT, render_reference


def test_every_native_analysis_field_has_an_explicit_quantity_definition():
    source = (Path(__file__).resolve().parents[1] / "passivbot-rust/src/types.rs").read_text()
    body = source.split("pub struct Analysis {", 1)[1].split("\n}", 1)[0]
    names = set(re.findall(r"pub (\w+):", body))
    assert len(names) > 150
    assert not names - METRIC_DEFINITIONS.keys()


def test_selector_and_currency_variant_metadata_are_complete_without_fallbacks():
    assert not ANALYSIS_SHARED_KEYS - METRIC_DEFINITIONS.keys()
    for base in CURRENCY_METRICS:
        assert get_metric_definition(base).basis
        for curve in ("usd", "btc"):
            field = get_metric_definition(metric_with_curve(base, curve))
            assert field.curve == curve
            assert field.unit == get_metric_definition(base).unit
    for name, definition in METRIC_DEFINITIONS.items():
        assert definition.name == name
        assert all((definition.unit, definition.basis, definition.sampling,
                    definition.aggregation, definition.sign, definition.domain))


def test_definition_and_registry_are_immutable_and_reports_are_detached():
    field = get_metric_definition("adg_fraction")
    with pytest.raises(FrozenInstanceError):
        field.unit = "percent"
    with pytest.raises(TypeError):
        METRIC_DEFINITIONS["invented"] = field
    report = describe_metric("adg_fraction")
    report["unit"] = "percent"
    assert get_metric_definition("adg_fraction").unit == "fraction"
    assert not {"goal", "backend", "supported_backends"}.intersection(report)


@pytest.mark.parametrize("name", ["invented_metric", "adg_fraction_usd_mean", "strategy_eq_underwater_pct_mean", "drawdown_worst_hsl"])
def test_quantity_lookup_neither_guesses_nor_translates_names(name):
    with pytest.raises(KeyError):
        get_metric_definition(name)


def test_daily_growth_and_conventional_ratios_disclose_their_actual_estimators():
    assert get_metric_definition("gain_multiplier").unit == "multiplier"
    assert get_metric_definition("adg_fraction").aggregation == "geometric daily growth"
    assert "daily minimum" in get_metric_definition("sharpe_ratio").sampling
    assert "geometric" in get_metric_definition("sharpe_ratio").basis.lower()
    ew = get_metric_definition("sharpe_ratio_usd_ew")
    assert "arithmetic" in ew.basis
    assert "complete UTC daily end" in ew.sampling
    assert "exponential" in ew.weighting


def test_fraction_ratio_score_and_time_populations_are_distinct():
    assert get_metric_definition("long_pnl_to_total_pnl_ratio").domain == "real"
    assert "0.5" in get_metric_definition("long_pnl_to_total_pnl_ratio").notes
    assert get_metric_definition("positive_log_return_participation_score_strategy_equity").unit == "score"
    assert "authored" in get_metric_definition("gain_multiplier_per_twel_ratio_usd_long").denominator
    assert "None" in get_metric_definition("gain_multiplier_per_twel_ratio_usd_long").notes
    assert "strictly higher" in get_metric_definition("recovery_mean_days_strategy_equity").basis
    assert "daily" not in get_metric_definition("recovery_mean_days_strategy_equity").sampling
    assert "plus signed fees" in get_metric_definition("daily_realized_pnl_mean_fraction").basis
    assert "without fees" in get_metric_definition("trade_win_fraction").notes


def test_hsl_signal_and_panic_loss_metadata_do_not_imply_equity_measurements():
    ema = get_metric_definition("hsl_signal_ema_max_fraction")
    assert ema.unit == "fraction" and ema.curve == "hsl_signal"
    assert "remembered" in ema.basis
    assert get_metric_definition("hsl_panic_fill_loss_max_usd").unit == "usd"
    assert "individual fill" in get_metric_definition("hsl_panic_fill_loss_max_usd").notes
    assert "first panic fill" in get_metric_definition("hsl_panic_episode_loss_mean_fraction").denominator
    assert "starting balance" in get_metric_definition("hsl_panic_loss_starting_balance_fraction").denominator


def test_generated_metric_reference_is_current():
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == render_reference()
