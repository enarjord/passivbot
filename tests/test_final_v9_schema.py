"""Final-v9 loading is strict; released-v8 conversion is explicit and reported."""

from copy import deepcopy
import json

import pytest
from config import prepare_config
from config.overrides import parse_overrides
from config.parse import load_raw_config
from config.risk_memory import risk_bootstrap_horizon_minutes, risk_memory_spans
from config.schema import get_template_config
from config.migrations.v9 import migrate_v8_to_v9
from tools.migrate_config_v9 import main, migrate
from released_v8_fixtures import released_v8_policy


def v8_config():
    config = released_v8_policy()
    for side in ("long", "short"):
        config["bot"][side]["forager"]["volume_drop_pct"] = 0.25
        config["optimize"]["bounds"][side]["forager"]["volume_drop_pct"] = [0.1, 0.5]
    return config


def test_canonical_defaults_and_sparse_defaults_load():
    complete = prepare_config(get_template_config(), verbose=False)
    sparse = prepare_config({"config_version": "v9.0.0", "bot": {}, "live": {}}, verbose=False)
    assert complete["bot"]["portfolio"]["hsl"] == {"scope": "coin"}
    assert sparse["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
    assert sparse["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert complete["live"]["modes"]["disapproved_position"] == "graceful_stop"


@pytest.mark.parametrize("path", [
    ("typo",), ("live", "unknown"), ("bot", "long", "risk", "maximum_positions"),
    ("bot", "long", "strategy", "trailing_martingale", "entry", "wrong_span"),
    ("optimize", "bounds", "long", "risk", "wrong_limit"),
])
def test_unknown_authored_fields_fail_before_pruning(path):
    config = get_template_config()
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = 1.0
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("section,key,value", [
    ("live", "hsl_signal_mode", "coin"), ("live", "lookback_ema_span_minutes", 60.0),
    ("live", "auto_gs", True), ("backtest", "liquidation_threshold", 0.05),
])
def test_retired_names_cannot_be_smuggled_into_final_schema(section, key, value):
    config = get_template_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="final canonical"):
        prepare_config(config, verbose=False)


def test_conflicting_json_members_are_rejected_before_mapping_overwrite(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"config_version":"v9.0.0","live":{"leverage":1,"leverage":2}}')
    with pytest.raises(ValueError, match="defined more than once"):
        load_raw_config(str(path), log_errors=False)


def test_nested_and_dotted_patch_conflict_is_rejected():
    config = get_template_config()
    config["optimize"]["fixed_runtime_overrides"] = {
        "live": {"account": {"leverage": 1}},
        "live.account.leverage": 2,
    }
    with pytest.raises(ValueError, match="defined more than once"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("owner", ["scenario", "fixed_runtime"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("same_value", [False, True])
def test_authored_object_and_descendant_authority_rejects_in_both_orders(owner, reverse, same_value):
    config = get_template_config()
    risk = deepcopy(config["bot"]["long"]["risk"])
    risk["max_positions"] = 2
    entries = [("bot.long.risk", risk),
               ("bot.long.risk.max_positions", 2 if same_value else 3)]
    patch = dict(reversed(entries) if reverse else entries)
    if owner == "scenario":
        config["backtest"]["scenarios"] = [{"label": "overlap", "overrides": patch}]
    else:
        config["optimize"]["fixed_runtime_overrides"] = patch
    original = deepcopy(config)
    with pytest.raises(ValueError, match="overlapping configuration paths"):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("structured", [False, True])
def test_authored_patch_accepts_one_object_or_distinct_canonical_siblings(structured):
    config = get_template_config()
    risk = deepcopy(config["bot"]["long"]["risk"])
    risk.update(max_positions=2, total_wallet_exposure_limit=1.25)
    patch = ({"bot.long.risk": risk} if structured else {
        "bot.long.risk.max_positions": 2,
        "bot.long.risk.total_wallet_exposure_limit": 1.25,
    })
    config["backtest"]["scenarios"] = [{"label": "canonical", "overrides": patch}]
    prepared = prepare_config(config, verbose=False)
    assert prepared["backtest"]["scenarios"][0]["overrides"] == patch


def test_coin_hsl_span_and_budget_scaling_override_are_local():
    config = get_template_config()
    config["coin_overrides"] = {"BTC": {"bot": {"long": {"hsl": {
        "memory_ema_span_minutes": 19.5, "scale_budget_with_excess_allowance": True,
    }}}}}
    result = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    policy = result["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]
    assert policy["memory_ema_span_minutes"] == 19.5
    assert policy["scale_budget_with_excess_allowance"] is True
    assert result["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == 43200.0


@pytest.mark.parametrize("patch", [
    {"bot": {"long": {"risk": {"max_positions": 2}}}},
    {"bot": {"long": {"risk": {"total_wallet_exposure_limit": 2.0}}}},
    {"bot": {"portfolio": {'unstuck': {'memory_ema_span_minutes': 1.0}}}},
    {"bot": {"portfolio": {'loss_gate': {'max_drawdown_ratio': 0.2}}}},
])
def test_aggregate_controls_cannot_be_coin_overridden(patch):
    config = get_template_config()
    config["coin_overrides"] = {"BTC": patch}
    with pytest.raises(ValueError, match="not overridable"):
        prepare_config(config, verbose=False)


def test_new_memory_dimensions_default_fixed_at_effective_config_values():
    config = get_template_config()
    config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] = 71.5
    config["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] = 83.5
    config["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = 19.5
    result = prepare_config(config, verbose=False)
    bounds = result["optimize"]["bounds"]
    assert bounds["portfolio"]["unstuck"]["memory_ema_span_minutes"] == [71.5, 71.5]
    assert bounds["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == [83.5, 83.5]
    assert bounds["long"]["hsl"]["memory_ema_span_minutes"] == [19.5, 19.5]


def test_independent_gate_only_memory_contributes_to_bootstrap():
    config = get_template_config()
    for side in ("long", "short"):
        config["bot"][side]["unstuck"]["enabled"] = False
    config["bot"]["portfolio"]["loss_gate"].update(max_drawdown_ratio=0.1, memory_ema_span_minutes=17.5)
    config = prepare_config(config, verbose=False)
    assert risk_memory_spans(config) == (17.5,)
    assert risk_bootstrap_horizon_minutes(config) == (87.5, False)


def test_explicit_v8_conversion_reports_renames_and_independent_memories():
    source = v8_config()
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == before
    bot = result.config["bot"]
    assert bot["long"]["risk"]["max_positions"] == 7
    assert bot["long"]["hsl"]["memory_ema_span_minutes"] == 123.5
    assert bot["short"]["hsl"]["memory_ema_span_minutes"] == 123.5
    assert bot["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 123.5
    assert bot["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == 123.5
    assert "volume_drop_pct" not in json.dumps(result.config)
    assert any("volume_drop_pct" in change["path"] for change in result.changes)
    assert migrate(result.config).changes == []


@pytest.mark.parametrize("window", [-1.0, 0.0, 90.0])
def test_legacy_gate_windows_require_ack_and_are_never_converted_as_spans(window):
    source = v8_config()
    source["live"].update(max_realized_loss_pct=0.1, realized_loss_gate_lookback_minutes=window)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, memory_span_minutes=123.5)
    result = migrate(source, memory_span_minutes=123.5,
                     acknowledge_realized_loss_gate_memory_change=True)
    assert result.config["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == 123.5
    assert any(change["old"] == window and "retired rolling gate window" in change["reason"]
               for change in result.changes)


def test_explicit_converter_rejects_conflicting_renamed_values():
    source = v8_config()
    source["bot"]["long"]["risk"]["max_positions"] = 99
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)


def test_converter_explicit_independent_span_choices_do_not_widen_search():
    result = migrate(v8_config(), memory_span_minutes=123.5, memory_span_choices={
        "bot.portfolio.unstuck.memory_ema_span_minutes": 71.5,
        "bot.portfolio.loss_gate.memory_ema_span_minutes": 83.5,
        "bot.long.hsl.memory_ema_span_minutes": 19.5,
    })
    assert result.config["bot"]["short"]["hsl"]["memory_ema_span_minutes"] == 123.5
    assert result.config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 71.5
    assert result.config["optimize"]["bounds"]["long"]["hsl"]["memory_ema_span_minutes"] == [19.5, 19.5]


def test_portfolio_memory_and_smoothing_are_separate_from_scope():
    config = get_template_config()
    policy = deepcopy(config["bot"]["long"]["hsl"])
    policy.update(scope="portfolio", memory_ema_span_minutes=17.5,
                  drawdown_ratio_ema_span_minutes=9.5, enabled=True)
    config["bot"]["portfolio"]["hsl"] = policy
    for side in ("long", "short"):
        config["optimize"]["bounds"][side].pop("hsl")
    result = prepare_config(config, verbose=False)
    assert result["bot"]["portfolio"]["hsl"] == policy
    assert result["optimize"]["bounds"]["portfolio"]["hsl"]["memory_ema_span_minutes"] == [17.5, 17.5]


def test_referenced_coin_patch_unknowns_fail_during_prepare(tmp_path):
    path = tmp_path / "coin.json"
    path.write_text(json.dumps({"bot": {"long": {"hsl": {"wrong_memory": 4.0}}}}))
    config = get_template_config()
    config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    with pytest.raises(
        ValueError,
        match=r"config\.bot\.long\.hsl\.wrong_memory is an unknown configuration field",
    ):
        prepare_config(config, verbose=False)


def test_scenario_retired_gate_control_is_converted_and_acknowledged():
    source = v8_config()
    source["backtest"]["scenarios"] = [{"label": "gate", "overrides": {
        "live.max_realized_loss_pct": 0.1, "live.realized_loss_gate_lookback_minutes": -1,
    }}]
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, memory_span_minutes=123.5)
    result = migrate(source, memory_span_minutes=123.5,
                     acknowledge_realized_loss_gate_memory_change=True)
    assert result.config["backtest"]["scenarios"][0]["overrides"] == {
        "bot.portfolio.loss_gate.max_drawdown_ratio": 0.1,
    }


def test_migration_cli_writes_private_report_without_modifying_source(tmp_path):
    source, output, report = (tmp_path / name for name in ("v8.json", "v9.json", "report.json"))
    config = v8_config()
    config["backtest"].update(start_date="2020-01-02", end_date="2020-03-04",
                              maker_fee_override=0.00017, taker_fee_override=0.00029)
    config["backtest"]["scenarios"] = [{"label": "chosen", "coins": ["BTC", "ETH"]}]
    source.write_text(json.dumps(config))
    original = source.read_bytes()
    assert main([str(source), str(output), "--lookback-ema-span-minutes", "123.5",
                 "--report", str(report)]) == 0
    assert source.read_bytes() == original
    converted = json.loads(output.read_text())
    for key in ("start_date", "end_date", "maker_fee_override", "taker_fee_override", "scenarios"):
        assert converted["backtest"][key] == config["backtest"][key]
    assert report.stat().st_mode & 0o777 == 0o600
    assert json.loads(report.read_text())["target_version"] == "v9.0.0"


def test_invalid_intermediate_schema_never_creates_migration_output(tmp_path):
    source, output = tmp_path / "intermediate.json", tmp_path / "output.json"
    source.write_text(json.dumps({**get_template_config(), "config_version": "v9.1.0"}))
    with pytest.raises(SystemExit):
        main([str(source), str(output), "--lookback-ema-span-minutes", "123.5"])
    assert not output.exists()


def test_coin_only_unstuck_enablement_requires_global_memory_history():
    config = get_template_config()
    for side in ("long", "short"):
        config["bot"][side]["unstuck"]["enabled"] = False
    config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] = 71.5
    config["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {"enabled": True}}}}}
    config = prepare_config(config, verbose=False)
    assert risk_memory_spans(config) == (71.5,)
    assert risk_bootstrap_horizon_minutes(config) == (357.5, False)


def test_local_wallet_limit_is_validated_without_inventing_a_global_field(tmp_path):
    config = get_template_config()
    path = tmp_path / "coin.json"
    path.write_text(json.dumps({"bot": {"long": {"wallet_exposure_limit": 0.12}}}))
    config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    parsed = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["wallet_exposure_limit"] == 0.12
    assert "wallet_exposure_limit" not in parsed["bot"]["long"]


def test_explicit_file_conversion_preserves_sparse_override_inheritance(tmp_path):
    config = v8_config()
    path = tmp_path / "coin.json"
    path.write_text(json.dumps({"bot": {"long": {"unstuck": {"close_fraction": 0.02}}}}))
    config["coin_overrides"] = {"BTC": {"override_config_path": str(path)}}
    result = migrate(config, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert result.config["coin_overrides"]["BTC"] == {"bot": {"long": {"unstuck": {"close_fraction": 0.02}}}}


def test_released_pymoo_fallback_is_explicitly_materialized():
    config = v8_config()
    config["optimize"]["pymoo"]["shared"].pop("crossover_eta")
    config["optimize"]["crossover_eta"] = 31.5
    result = migrate(config, memory_span_minutes=123.5)
    assert result.config["optimize"]["pymoo"]["shared"]["crossover_eta"] == 31.5
    assert "crossover_eta" not in result.config["optimize"]
    assert any("effective released pymoo fallback" in item["reason"] for item in result.changes)


def test_deap_retirement_requires_separate_search_acknowledgment():
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_probability=0.3)
    with pytest.raises(ValueError, match="acknowledge-search-backend-change"):
        migrate(config, memory_span_minutes=123.5)
    result = migrate(config, memory_span_minutes=123.5, acknowledge_search_backend_change=True)
    assert result.config["optimize"]["backend"] == "pymoo"
    assert any(item["path"] == "optimize.mutation_probability" and item["old"] == 0.3
               for item in result.changes)


def test_weighted_metric_retirement_requires_choice_and_distinct_formula_ack():
    config = v8_config()
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq_w", "goal": "max"}]
    config["backtest"]["visible_metrics"] = ["adg_strategy_eq_w"]
    with pytest.raises(ValueError, match="metric-replacement"):
        migrate(config, memory_span_minutes=123.5)
    choices = {"adg_strategy_eq_w": "adg_strategy_eq"}
    with pytest.raises(ValueError, match="acknowledge-metric-formula-change"):
        migrate(config, memory_span_minutes=123.5, metric_replacements=choices)
    result = migrate(config, memory_span_minutes=123.5, metric_replacements=choices,
                     acknowledge_metric_formula_change=True)
    assert result.config["optimize"]["scoring"] == [{"metric": "adg_strategy_eq", "goal": "max"}]
    assert result.config["backtest"]["visible_metrics"] == ["adg_strategy_eq"]


@pytest.mark.parametrize("key", ["crossover_eta", "crossover_probability", "mutation_eta",
                                 "mutation_indpb", "mutation_probability", "offspring_multiplier"])
def test_retired_search_controls_are_not_normal_runtime_fields(key):
    config = get_template_config()
    config["optimize"][key] = 1.0
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(config, verbose=False)


def test_invalid_authored_constraint_is_not_replaced_by_default_limits():
    config = get_template_config()
    config["optimize"]["limits"] = [{"metric": "drawdown_worst_strategy_eq",
        "penalize_if": "greater_than", "value": "invalid"}]
    with pytest.raises(ValueError, match="numeric"):
        prepare_config(config, verbose=False)


def test_v8_legacy_limits_are_explicitly_converted_and_reported():
    config = v8_config()
    config["optimize"]["limits"] = {"penalize_if_greater_than_drawdown_worst_strategy_eq": 0.7}
    result = migrate(config, memory_span_minutes=123.5)
    assert result.config["optimize"]["limits"] == [{"metric": "drawdown_worst_strategy_eq",
        "penalize_if": "greater_than", "value": 0.7}]
    assert any(item["path"] == "optimize.limits" for item in result.changes)


def test_inactive_retired_strategy_subtree_is_reported_without_substitution():
    config = v8_config()
    config["bot"]["long"]["strategy"]["trailing_grid_v7"] = {"ema_span_0": 17.5}
    result = migrate(config, memory_span_minutes=123.5)
    assert "trailing_grid_v7" not in result.config["bot"]["long"]["strategy"]
    assert any("inactive retired strategy" in item["reason"] for item in result.changes)


def test_active_retired_strategy_requires_explicit_replacement():
    config = v8_config()
    config["live"]["strategy_kind"] = "trailing_grid_v7"
    config["bot"]["long"]["strategy"]["trailing_grid_v7"] = {"ema_span_0": 17.5}
    with pytest.raises(ValueError, match="choose a retained strategy explicitly"):
        migrate(config, memory_span_minutes=123.5)


@pytest.mark.parametrize("metric", ["adg_strategy_eq_w_per_exposure_long",
                                   "adg_w_per_exposure_short", "adg_w_usd_max"])
def test_dependent_weighted_metric_variants_require_explicit_formula_choice(metric):
    config = v8_config()
    config["optimize"]["scoring"] = [{"metric": metric, "goal": "max"}]
    with pytest.raises(ValueError, match="metric-replacement"):
        migrate(config, memory_span_minutes=123.5)
    result = migrate(config, memory_span_minutes=123.5,
                     metric_replacements={metric: "adg_strategy_eq"},
                     acknowledge_metric_formula_change=True)
    assert result.config["optimize"]["scoring"] == [{"metric": "adg_strategy_eq", "goal": "max"}]


def test_clean_export_preserves_explicit_global_portfolio_policy_and_fixed_span():
    from config.hsl import generated_template
    from config_utils import sanitize_prepared_config_for_dump

    source = generated_template(get_template_config(), "portfolio")
    source["bot"]["portfolio"]["hsl"].update(enabled=True, memory_ema_span_minutes=123.5,
                                        drawdown_ratio_ema_span_minutes=7.5)
    prepared = prepare_config(source, verbose=False)
    exported = sanitize_prepared_config_for_dump(prepared)
    assert exported["bot"]["portfolio"]["hsl"] == prepared["bot"]["portfolio"]["hsl"]
    assert exported["optimize"]["bounds"]["portfolio"]["hsl"]["memory_ema_span_minutes"] == [123.5, 123.5]
    loaded = prepare_config(exported, verbose=False)
    assert loaded["bot"]["portfolio"]["hsl"] == prepared["bot"]["portfolio"]["hsl"]


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_cli_independent_global_risk_controls_use_canonical_paths(command):
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli, update_config_with_args

    config = get_template_config()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_config_arguments(parser, project_template_config_for_cli(config, command), command=command)
    args = parser.parse_args(["--unstuck-memory-ema-span-minutes", "17.5",
                              "--loss-gate-memory-ema-span-minutes", "23.5",
                              "--loss-gate-max-drawdown-ratio", "0.12", "--hsl-scope", "coin"])
    update_config_with_args(config, args)
    assert config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 17.5
    assert config["bot"]["portfolio"]["loss_gate"] == {"memory_ema_span_minutes": 23.5,
                                                 "max_drawdown_ratio": 0.12}
    assert config["bot"]["portfolio"]["hsl"]["scope"] == "coin"
    for retired in ("--lookback-ema-span-minutes", "--hsl-signal-mode", "--max-realized-loss-pct"):
        with pytest.raises(SystemExit):
            parser.parse_args([retired, "1"])


@pytest.mark.parametrize("changed", [{"goal": "min"}, {"scenario": "other"}, {"reducer": "min"}])
def test_conflicting_repeated_objectives_are_never_silently_discarded(changed):
    source = get_template_config()
    objective = {"metric": "adg_strategy_eq", "goal": "max"}
    source["optimize"]["scoring"] = [objective, {**objective, **changed}]
    with pytest.raises(ValueError, match="conflicts with another objective"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("scoring", [
    ["adg_strategy_eq", "drawdown_worst_strategy_eq"],
    ["adg_strategy_eq", {"metric": "drawdown_worst_strategy_eq", "goal": "min"}],
])
def test_final_v9_scoring_shorthand_reaches_canonical_normalizer(scoring):
    config = get_template_config()
    config["optimize"]["scoring"] = deepcopy(scoring)
    before = deepcopy(config)
    prepared = prepare_config(config, verbose=False)
    assert prepared["optimize"]["scoring"] == [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
    ]
    assert config == before


def test_public_scoring_cli_shorthand_loads_through_prepare_config():
    import argparse
    from config_utils import add_config_arguments, project_template_config_for_cli, update_config_with_args

    config = get_template_config()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    allowed = add_config_arguments(
        parser, project_template_config_for_cli(config, "optimize"), command="optimize",
    )
    args = parser.parse_args(["--scoring", "adg_strategy_eq,drawdown_worst_strategy_eq"])
    update_config_with_args(config, args, allowed_keys=allowed)
    assert prepare_config(config, verbose=False)["optimize"]["scoring"] == [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
    ]


@pytest.mark.parametrize("metric,match", [
    ("unknown_objective", "no default optimization goal"),
    ("adg_strategy_eq_w", "retired"),
])
def test_scoring_shorthand_preserves_unknown_and_retired_metric_rejection(metric, match):
    config = get_template_config()
    config["optimize"]["scoring"] = [metric]
    with pytest.raises(ValueError, match=match):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("record", [None, True, 1, 1.0, ["adg_strategy_eq"]])
def test_scoring_shorthand_does_not_admit_other_record_types(record):
    config = get_template_config()
    config["optimize"]["scoring"] = [record]
    with pytest.raises(TypeError, match="must be a dict"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("section,key", [("optimize", "limits"), ("backtest", "scenarios")])
def test_scoring_shorthand_does_not_relax_other_record_lists(section, key):
    config = get_template_config()
    config[section][key] = ["adg_strategy_eq"]
    with pytest.raises(TypeError, match="must be a dict"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("days,span", [(2.125, 3060.0), (0, 1.0), ("all", 43200.0)])
def test_best_effort_infers_memory_and_reports_each_guess(days, span, caplog):
    config = v8_config()
    config["live"]["pnls_max_lookback_days"] = days
    original = deepcopy(config)
    result = migrate(config, best_effort=True)
    assert config == original
    assert result.config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == span
    assert result.config["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == span
    assert result.config["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == span
    guesses = [item for item in result.changes if item.get("best_effort")]
    assert guesses == [dict(path="live.pnls_max_lookback_days", old=days, new=span,
                            reason=guesses[0]["reason"], best_effort=True)]
    assert "[best-effort] live.pnls_max_lookback_days" in caplog.text


def test_best_effort_missing_window_uses_reported_canonical_default(caplog):
    config = v8_config()
    del config["live"]["pnls_max_lookback_days"]
    result = migrate(config, best_effort=True)
    assert result.config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
    assert "missing rolling window" in caplog.text


@pytest.mark.parametrize("days", [True, -1, float("nan"), float("inf"), "invalid", None])
def test_best_effort_rejects_invalid_authored_windows(days):
    config = v8_config()
    config["live"]["pnls_max_lookback_days"] = days
    with pytest.raises(ValueError, match="legacy duration|must be >= 0"):
        migrate(config, best_effort=True)


@pytest.mark.parametrize("location", ["visible", "scoring", "limits", "reducer"])
@pytest.mark.parametrize("old,new", [("adg_strategy_eq_w", "adg_strategy_eq_ew"),
                                   ("sharpe_ratio_w_usd", "sharpe_ratio_ew_usd"),
                                   ("sortino_ratio_w_btc", "sortino_ratio_ew_btc")])
def test_best_effort_metric_locations_enable_ew_and_log_choices(location, old, new, caplog):
    config = v8_config()
    if location == "visible":
        config["backtest"]["visible_metrics"] = [old]
    elif location == "scoring":
        config["optimize"]["scoring"] = [{"metric": old, "goal": "max"}]
    elif location == "limits":
        config["optimize"]["limits"] = {"penalize_if_lower_than_" + old: 0.1}
    else:
        config["backtest"]["reducer"] = {old: "mean"}
    result = migrate(config, best_effort=True)
    assert result.config["backtest"]["ew_half_life_days"] == 180.0
    assert any(item["old"] == old and item["new"] == new and item.get("best_effort")
               for item in result.changes)
    assert old in caplog.text and new in caplog.text
    assert "backtest.ew_half_life_days" in caplog.text


def test_best_effort_explicit_choices_take_precedence():
    config = v8_config()
    config["backtest"]["visible_metrics"] = ["adg_strategy_eq_w"]
    config["backtest"]["ew_half_life_days"] = 17.0
    result = migrate(config, best_effort=True, memory_span_minutes=99.5,
                     metric_replacements={"adg_strategy_eq_w": "adg_strategy_eq"})
    assert result.config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 99.5
    assert result.config["backtest"]["visible_metrics"] == ["adg_strategy_eq"]
    assert result.config["backtest"]["ew_half_life_days"] == 17.0
    assert not any(item["path"] == "live.pnls_max_lookback_days" and item.get("best_effort")
                   for item in result.changes)


def test_best_effort_retired_metric_without_ew_counterpart_requires_choice():
    config = v8_config()
    config["backtest"]["visible_metrics"] = ["adg_strategy_eq_w_per_exposure_long"]
    with pytest.raises(ValueError, match="no supported _ew replacement"):
        migrate(config, best_effort=True)


def test_best_effort_scenario_and_fixed_windows_keep_distinct_spans():
    config = v8_config()
    config["backtest"]["scenarios"] = [{"label": "window", "overrides": {
        "live.pnls_max_lookback_days": 2.5}}]
    config["optimize"]["fixed_runtime_overrides"] = {"live.pnls_max_lookback_days": 4.5}
    result = migrate(config, best_effort=True, memory_span_choices={
        "backtest.scenarios[0].overrides.live.pnls_max_lookback_days": 61.5})
    fixed = result.config["optimize"]["fixed_runtime_overrides"]
    scenario = result.config["backtest"]["scenarios"][0]["overrides"]
    assert fixed["bot.portfolio.unstuck.memory_ema_span_minutes"] == 6480.0
    assert scenario["bot.portfolio.unstuck.memory_ema_span_minutes"] == 61.5
    assert any(item["path"] == "optimize.fixed_runtime_overrides.live.pnls_max_lookback_days"
               and item.get("best_effort") for item in result.changes)


def test_best_effort_accepts_changed_never_and_loss_gate_policy(caplog):
    config = v8_config()
    config["bot"]["long"]["hsl"]["restart_after_red_policy"] = "never"
    config["live"]["max_realized_loss_pct"] = 0.05
    result = migrate(config, best_effort=True)
    assert result.config["bot"]["portfolio"]["loss_gate"]["max_drawdown_ratio"] == 0.05
    assert "restart_after_red_policy" not in result.config["bot"]["long"]["hsl"]
    assert "remove permanent stop" in caplog.text
    assert "decaying realized-loss memory" in caplog.text


def test_best_effort_deap_builds_reported_pymoo_policy(caplog):
    config = v8_config()
    config["optimize"].update(backend="deap", crossover_eta=31.5, mutation_indpb=0.25)
    config["optimize"].pop("pymoo")
    result = migrate(config, best_effort=True)
    assert result.config["optimize"]["backend"] == "pymoo"
    shared = result.config["optimize"]["pymoo"]["shared"]
    assert shared["crossover_eta"] == 31.5
    assert shared["mutation_prob_per_variable"] == 0.25
    assert "approximate DEAP policy" in caplog.text
    assert "start a fresh optimization" in caplog.text


def test_strict_migration_preserves_metric_choices_when_span_is_missing():
    config = v8_config()
    config["backtest"]["visible_metrics"] = ["adg_strategy_eq_w"]
    with pytest.raises(ValueError, match="explicitly choose --lookback-ema-span-minutes"):
        migrate(config, metric_replacements={"adg_strategy_eq_w": "adg_strategy_eq"},
                acknowledge_metric_formula_change=True)


def test_best_effort_cli_dry_run_and_report_are_observable(tmp_path, caplog):
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    report = tmp_path / "report.json"
    config = v8_config()
    config["backtest"]["visible_metrics"] = ["adg_strategy_eq_w"]
    source.write_text(json.dumps(config))
    before = source.read_bytes()
    assert main([str(source), str(target), "--best-effort", "--dry-run", "--report", str(report)]) == 0
    assert not target.exists()
    assert source.read_bytes() == before
    changes = json.loads(report.read_text())["changes"]
    assert any(item.get("best_effort") and item["new"] == "adg_strategy_eq_ew" for item in changes)
    for item in changes:
        if item.get("best_effort"):
            assert item["path"] in caplog.text
    assert main([str(source), str(target), "--best-effort"]) == 0
    loaded = prepare_config(load_raw_config(str(target)), verbose=False)
    assert loaded["backtest"]["visible_metrics"] == ["adg_strategy_eq_ew"]
    published = target.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), str(target), "--best-effort"])
    assert target.read_bytes() == published


def test_best_effort_stat_suffix_is_retained_in_objectives():
    config = v8_config()
    config["optimize"]["scoring"] = [{"metric": "adg_w_usd_max", "goal": "max"}]
    result = migrate(config, best_effort=True)
    assert result.config["optimize"]["scoring"][0]["metric"] == "adg_ew_usd_max"


def test_best_effort_coin_file_guesses_are_reported_and_file_is_unchanged(tmp_path, caplog):
    config = v8_config()
    coin = tmp_path / "coin.json"
    coin.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": False, "restart_after_red_policy": "never"}}}}))
    before = coin.read_bytes()
    config["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    result = migrate(config, best_effort=True, base_config_path=str(tmp_path / "root.json"))
    assert coin.read_bytes() == before
    assert "override_config_path" not in result.config["coin_overrides"]["AAA"]
    assert any(item.get("best_effort") and "override_config_path::" in item["path"]
               and "restart_after_red_policy" in item["path"] for item in result.changes)
    assert "coin_overrides.AAA.override_config_path::bot.long.hsl.restart_after_red_policy" in caplog.text


@pytest.mark.parametrize("kind", ["bounds", "unknown", "conflict", "strategy", "null_policy"])
def test_best_effort_does_not_hide_unsupported_or_conflicting_authority(kind):
    config = v8_config()
    if kind == "bounds":
        config["optimize"]["bounds"]["live"] = {"pnls_max_lookback_days": [1.0, 30.0]}
    elif kind == "unknown":
        config["live"]["unknown_option"] = True
    elif kind == "conflict":
        config["bot"]["long"]["n_positions"] = 3
        config["bot"]["long"]["risk"]["n_positions"] = 7
    elif kind == "strategy":
        config["live"]["strategy_kind"] = "trailing_grid_v7"
    else:
        config["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy=None)
    original = deepcopy(config)
    with pytest.raises((ValueError, TypeError)):
        migrate(config, best_effort=True)
    assert config == original


def test_best_effort_already_v9_is_a_noop_without_guesses():
    config = migrate(v8_config(), best_effort=True).config
    result = migrate(config, best_effort=True)
    assert result.config == config
    assert result.changes == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_best_effort_saved_result_metadata_is_reported_without_reusing_fitness(wrapped, caplog):
    config = v8_config()
    config["backtest"].update(coins={"bybit": ["AAA"]}, cache_dir={"bybit": "old-cache"})
    config["optimizer_evaluation_contract"] = {"live": {"pnls_max_lookback_days": 99.0}}
    config["metrics"] = {"adg_strategy_eq_w": 0.01}
    config["suite_metrics"] = {"aggregate": {"old": 0.01}}
    source = {"config": config} if wrapped else config
    original = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert source == original
    assert "optimizer_evaluation_contract" not in result.config
    assert "metrics" not in result.config and "suite_metrics" not in result.config
    assert "coins" not in result.config["backtest"] and "cache_dir" not in result.config["backtest"]
    for key in ("optimizer_evaluation_contract", "metrics", "suite_metrics", "backtest.coins", "backtest.cache_dir"):
        path = "config." + key if wrapped else key
        assert any(item["path"] == path and item["new"] is None and item.get("best_effort")
                   for item in result.changes)
        assert path in caplog.text
    assert "optimizer_evaluation_contract.live.pnls_max_lookback_days" not in caplog.text


def test_best_effort_does_not_discard_authored_backtest_coins():
    config = v8_config()
    config["backtest"]["coins"] = {"bybit": ["AAA"]}
    with pytest.raises(ValueError, match="backtest.coins is an unknown"):
        migrate(config, best_effort=True)


@pytest.mark.parametrize("mode", ["bounded", "legacy_raw", "invalid"])
def test_best_effort_retired_excess_mode_preserves_bounded_contract(mode):
    config = v8_config()
    config["bot"]["long"]["risk"]["we_excess_allowance_mode"] = mode
    if mode != "bounded":
        with pytest.raises(ValueError, match="only retired bounded allowance"):
            migrate(config, best_effort=True)
        return
    result = migrate(config, best_effort=True)
    assert "we_excess_allowance_mode" not in result.config["bot"]["long"]["risk"]
    assert any(item.get("best_effort") and item["path"] == "bot.long.risk.we_excess_allowance_mode"
               for item in result.changes)


@pytest.mark.parametrize("converter", [migrate, migrate_v8_to_v9])
def test_best_effort_reports_outer_and_inner_metadata_distinctly(converter, caplog):
    config = v8_config()
    config["analysis"] = {"inner_fitness": 0.1}
    source = {"config": config, "analysis": {"outer_fitness": 0.2},
              "metrics": {"old_fitness": 0.3}, "optimizer_evaluation_contract": {"version": 2}}
    original = deepcopy(source)
    result = converter(source, best_effort=True)
    assert source == original
    expected = {"analysis": source["analysis"], "config.analysis": config["analysis"],
                "metrics": source["metrics"],
                "optimizer_evaluation_contract": source["optimizer_evaluation_contract"]}
    for path, old in expected.items():
        entries = [item for item in result.changes if item["path"] == path]
        assert len(entries) == 1
        assert entries[0]["old"] == old
        assert entries[0]["new"] is None
        assert entries[0]["best_effort"] is True
        assert f"[best-effort] {path}:" in caplog.text
    assert "analysis" not in result.config
    assert "optimizer_evaluation_contract" not in result.config


@pytest.mark.parametrize("best_effort", [False, True])
def test_deap_transition_has_exactly_one_unambiguous_report_entry(best_effort):
    config = v8_config()
    config["optimize"]["backend"] = "deap"
    result = migrate(config, best_effort=best_effort, memory_span_minutes=43200.0,
                     acknowledge_search_backend_change=not best_effort)
    transitions = [item for item in result.changes if item["path"] == "optimize.backend"]
    assert len(transitions) == 1
    assert transitions[0]["old"] == "deap" and transitions[0]["new"] == "pymoo"
    assert transitions[0].get("best_effort", False) is best_effort
    assert ("acknowledged" in transitions[0]["reason"]) is not best_effort


@pytest.mark.parametrize("acknowledge", [False, True])
def test_best_effort_never_removal_has_exactly_one_report_entry(acknowledge):
    config = v8_config()
    config["bot"]["long"]["hsl"]["restart_after_red_policy"] = "never"
    result = migrate(config, best_effort=True, acknowledge_never_removal=acknowledge)
    entries = [item for item in result.changes
               if item["path"] == "bot.long.hsl.restart_after_red_policy"]
    assert len(entries) == 1
    assert entries[0]["old"] == "never" and entries[0]["new"] is None
    assert entries[0].get("best_effort", False) is not acknowledge


@pytest.mark.parametrize("source_policy", ["file", "inline", "both"])
@pytest.mark.parametrize("context", ["root", "scenario_audit", "fixed_audit", "scenario_patch", "legacy_flags"])
def test_best_effort_coin_warnings_match_retained_report_decisions(tmp_path, caplog, source_policy, context):
    config = v8_config()
    coin = tmp_path / "coin.json"
    file_policy = "never" if source_policy in {"file", "both"} else "always"
    coin.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": False, "restart_after_red_policy": file_policy}}}}))
    patch = {"AAA": {"override_config_path": "coin.json"}}
    if source_policy in {"inline", "both"}:
        patch["AAA"]["bot"] = {"long": {"hsl": {"restart_after_red_policy": "never"}}}
    config["coin_overrides"] = deepcopy(patch)
    if context == "scenario_audit":
        config["backtest"]["scenarios"] = [{"label": "audit", "overrides": {"live.leverage": 7}}]
    elif context == "fixed_audit":
        config["optimize"]["fixed_runtime_overrides"] = {"live.leverage": 7}
    elif context == "scenario_patch":
        config["backtest"]["scenarios"] = [{"label": "patch", "overrides": {"coin_overrides": deepcopy(patch)}}]
    elif context == "legacy_flags":
        config["live"]["coin_flags"] = {"AAA": "-lc coin.json"}
    result = migrate(config, best_effort=True, base_config_path=str(tmp_path / "root.json"))
    guesses = [item for item in result.changes if item.get("best_effort")]
    messages = [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("[best-effort]")]
    assert len(messages) == len(guesses)
    assert len({item["path"] for item in guesses}) == len(guesses)
    for item in guesses:
        assert sum(message.startswith(f"[best-effort] {item['path']}:") for message in messages) == 1


@pytest.mark.parametrize("candidate,variable,expected_variable", [(0.34, 0.01, 0.01), (0.0, 0.0, "auto")])
@pytest.mark.parametrize("authored", [False, True])
def test_best_effort_deap_preserves_candidate_and_variable_mutation_controls(candidate, variable, expected_variable, authored):
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_probability=candidate, mutation_indpb=variable)
    config["optimize"]["pymoo"]["shared"] = ({"mutation_prob": 0.12, "mutation_prob_per_variable": 0.21}
                                              if authored else {})
    result = migrate(config, best_effort=True)
    shared = result.config["optimize"]["pymoo"]["shared"]
    assert shared["mutation_prob"] == (0.12 if authored else candidate)
    assert shared["mutation_prob_per_variable"] == (0.21 if authored else expected_variable)
    guesses = [item for item in result.changes if item.get("best_effort")]
    assert any(item["path"] == "optimize.pymoo.shared.mutation_prob" for item in guesses) is not authored
    assert any(item["path"] == "optimize.pymoo.shared.mutation_prob_per_variable" for item in guesses) is not authored


@pytest.mark.parametrize("converter", [migrate, migrate_v8_to_v9])
@pytest.mark.parametrize("explicit", [False, True])
def test_memory_decision_provenance_is_unique_for_root_scenario_and_fixed(converter, explicit):
    config = v8_config()
    config["backtest"]["scenarios"] = [{"label": "span", "overrides": {
        "live.pnls_max_lookback_days": 2.5}}]
    config["optimize"]["fixed_runtime_overrides"] = {"live.pnls_max_lookback_days": 4.5}
    paths = ["live.pnls_max_lookback_days",
             "backtest.scenarios[0].overrides.live.pnls_max_lookback_days",
             "optimize.fixed_runtime_overrides.live.pnls_max_lookback_days"]
    options = {"memory_span_minutes": 123.0,
               "memory_span_choices": {path: 321.0 for path in paths[1:]}} if explicit else {}
    result = converter(config, best_effort=True, **options)
    for path in paths:
        decisions = [item for item in result.changes if item["path"] == path]
        assert len(decisions) == 1
        assert decisions[0].get("best_effort", False) is not explicit
        assert ("explicit" in decisions[0]["reason"]) is explicit


@pytest.mark.parametrize("acknowledge", [False, True])
def test_gate_report_does_not_claim_explicit_consent_without_it(acknowledge):
    config = v8_config()
    config["live"]["max_realized_loss_pct"] = 0.05
    result = migrate(config, best_effort=True,
                     acknowledge_realized_loss_gate_memory_change=acknowledge)
    consent = [item for item in result.changes if "acknowledged shared realized-loss memory" in item["reason"]]
    assert len(consent) == (1 if acknowledge else 0)
    inferred = [item for item in result.changes if "adopt independent decaying realized-loss memory" in item["reason"]]
    assert bool(inferred) is not acknowledge
    assert all(item["best_effort"] for item in inferred)


@pytest.mark.parametrize("acknowledge", [False, True])
def test_best_effort_backend_acknowledgment_is_preserved(acknowledge, caplog):
    config = v8_config()
    config["optimize"]["backend"] = "deap"
    config["optimize"]["pymoo"]["shared"] = {}
    result = migrate(config, best_effort=True, acknowledge_search_backend_change=acknowledge)
    decisions = [item for item in result.changes if item["path"] == "optimize.backend"]
    assert len(decisions) == 1
    assert decisions[0].get("best_effort", False) is not acknowledge
    assert ("acknowledged" in decisions[0]["reason"]) is acknowledge
    assert ("[best-effort] optimize.backend:" in caplog.text) is not acknowledge
    assert any(item.get("best_effort") and item["path"].startswith("optimize.pymoo.shared.")
               for item in result.changes)


@pytest.mark.parametrize("converter", [migrate, migrate_v8_to_v9])
def test_saved_analysis_aliases_cannot_become_configuration_authority(converter):
    config = v8_config()
    config["analysis"] = {"bot": {"long": {"n_positions": 3, "risk": {"n_positions": 7}}}}
    from config.migrations.final_v9 import reject_v8_alias_conflicts
    with pytest.raises(ValueError, match="conflicts"):
        reject_v8_alias_conflicts(config)
    result = converter(config, best_effort=True)
    assert "analysis" not in result.config
    assert any(item["path"] == "analysis" and item.get("best_effort") for item in result.changes)


@pytest.mark.parametrize("key", ["crossover_eta", "crossover_probability", "mutation_eta",
                                 "mutation_probability", "mutation_indpb", "offspring_multiplier"])
@pytest.mark.parametrize("value", [False, True])
def test_best_effort_rejects_boolean_deap_numeric_controls(key, value):
    config = v8_config()
    config["optimize"].update(backend="deap", **{key: value})
    config["optimize"]["pymoo"]["shared"] = {}
    original = deepcopy(config)
    with pytest.raises(ValueError, match=rf"optimize\.{key} must be numeric, not a boolean"):
        migrate(config, best_effort=True)
    assert config == original


def test_best_effort_explicit_pymoo_policy_does_not_hide_malformed_deap_input():
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_indpb=False)
    with pytest.raises(ValueError, match="mutation_indpb must be numeric, not a boolean"):
        migrate(config, best_effort=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_best_effort_rejects_nonfinite_deap_probabilities(value):
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_probability=value)
    config["optimize"]["pymoo"]["shared"] = {}
    with pytest.raises(ValueError, match="mutation_probability must be a finite number"):
        migrate(config, best_effort=True)


@pytest.mark.parametrize("value", [False, True])
def test_best_effort_cli_rejects_boolean_sentinel_before_writing(tmp_path, value, capsys):
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_indpb=value)
    config["optimize"]["pymoo"]["shared"] = {}
    source, target, report = (tmp_path / name for name in ("source.json", "target.json", "report.json"))
    source.write_text(json.dumps(config))
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(target), "--best-effort", "--report", str(report)])
    assert exc.value.code == 2
    assert "must be numeric, not a boolean" in capsys.readouterr().err
    assert source.read_bytes() == original
    assert not target.exists() and not report.exists()


@pytest.mark.parametrize("converter", [migrate, migrate_v8_to_v9])
@pytest.mark.parametrize("key", ["_raw", "_raw_effective", "_transform_log", "_coins_sources"])
def test_best_effort_internal_metadata_does_not_drive_inference(converter, key):
    config = v8_config()
    config[key] = {"backtest": {"visible_metrics": ["adg_strategy_eq_w_per_exposure_long"]},
                   "live": {"pnls_max_lookback_days": 999.0}}
    result = converter(config, best_effort=True)
    assert key not in result.config
    assert result.config["backtest"].get("ew_half_life_days") is None
    assert result.config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
    guesses = [item for item in result.changes if item.get("best_effort")]
    assert {item["path"] for item in guesses} == {key, "live.pnls_max_lookback_days"}


def test_internal_metadata_alone_does_not_authorize_dropping_authored_coins():
    config = v8_config()
    config["_raw"] = {}
    config["backtest"]["coins"] = {"bybit": ["AAA"]}
    with pytest.raises(ValueError, match="backtest.coins is an unknown"):
        migrate(config, best_effort=True)


@pytest.mark.parametrize("variable,expected", [("0", "auto"), ("0.0", "auto"),
                                              ("-0.0", "auto"), ("0e0", "auto"), ("0.01", 0.01)])
@pytest.mark.parametrize("candidate", ["0", "0.34"])
def test_best_effort_normalizes_deap_numeric_strings_before_sentinel_selection(variable, expected, candidate):
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_indpb=variable,
                               mutation_probability=candidate, crossover_eta="31.5")
    config["optimize"]["pymoo"]["shared"] = {}
    result = migrate(config, best_effort=True)
    shared = result.config["optimize"]["pymoo"]["shared"]
    assert shared["mutation_prob_per_variable"] == expected
    assert shared["mutation_prob"] == float(candidate)
    assert shared["crossover_eta"] == 31.5
    choices = {item["path"]: item for item in result.changes if item.get("best_effort")}
    assert choices["optimize.pymoo.shared.mutation_prob_per_variable"]["new"] == expected
    assert choices["optimize.pymoo.shared.mutation_prob"]["new"] == float(candidate)
    assert isinstance(choices["optimize.pymoo.shared.mutation_prob"]["new"], float)
    assert choices["optimize.pymoo.shared.crossover_eta"]["new"] == 31.5


def test_best_effort_cli_preserves_numeric_string_zero_sentinel(tmp_path, capsys):
    config = v8_config()
    config["optimize"].update(backend="deap", mutation_indpb="0", mutation_probability="0")
    config["optimize"]["pymoo"]["shared"] = {}
    source, target, report = (tmp_path / name for name in ("source.json", "target.json", "report.json"))
    source.write_text(json.dumps(config))
    assert main([str(source), str(target), "--best-effort", "--report", str(report)]) == 0
    shared = json.loads(target.read_text())["optimize"]["pymoo"]["shared"]
    assert shared["mutation_prob_per_variable"] == "auto"
    assert shared["mutation_prob"] == 0.0
    choices = {item["path"]: item for item in json.loads(report.read_text())["changes"]
               if item.get("best_effort")}
    assert choices["optimize.pymoo.shared.mutation_prob_per_variable"]["new"] == "auto"
    capsys.readouterr()
