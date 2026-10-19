"""Synthetic account economics and discovery evidence; no search or replay jobs."""

from copy import deepcopy
import math

import pytest

from tools.optimizer_benchmark.account_metrics import DAY_MS, extract_account_metrics
from tools.optimizer_benchmark.discovery import (
    COLD_PHASES,
    DiscoveryCounter,
    canonical_configuration_identity,
    compound_candidate_identity,
    validate_protocol,
)


def account(**updates):
    inputs = dict(start_ms=0, expected_end_ms=4 * DAY_MS, initial_equity=100.0,
                  timestamps_ms=[i * DAY_MS for i in range(5)],
                  equity=[100, 120, 90, 120, 108], gross_exposure_ratio=[0, 2, 1, .5, 0],
                  fills=[dict(identity="open", timestamp_ms=0, net_realized_pnl=-2,
                              fee_cashflow=-2, notional=200, is_close=False),
                         dict(identity="close", timestamp_ms=2 * DAY_MS, net_realized_pnl=5,
                              fee_cashflow=.25, notional=210, is_close=True)],
                  completed=True, liquidated=False, sample_interval_ms=DAY_MS)
    inputs.update(updates)
    return extract_account_metrics(**inputs)


def test_account_uses_ordinary_equity_and_does_not_double_charge_fees():
    result = account()
    assert result["net_return"] == pytest.approx(.08)
    assert result["geometric_growth_per_day"] == pytest.approx(1.08 ** .25 - 1)
    assert result["log_growth_per_day"] == pytest.approx(math.log(1.08) / 4)
    assert result["account_max_drawdown"] == pytest.approx(.25)
    assert result["net_realized_pnl"] == 3
    assert (result["fees_paid"], result["rebates_received"], result["net_fees"]) == (2, .25, 1.75)
    assert (result["fill_count"], result["close_fill_count"], result["active_days"]) == (2, 1, 2)
    assert result["traded_notional"] == 410
    assert result["gross_exposure_ratio_mean"] == pytest.approx(.875)
    assert result["gross_exposure_ratio_max"] == 2
    assert result["exposed_fraction"] == .75


def test_underwater_recovery_and_terminal_right_censor_are_distinct():
    result = account()
    assert result["underwater_fraction"] == .25
    assert result["recovery_duration_max_ms_lower_bound"] == 2 * DAY_MS
    assert result["terminal_underwater_censored"]
    recovered, censored = result["recovery_episodes"]
    assert recovered == dict(peak_timestamp_ms=DAY_MS, peak_equity=120,
                            underwater_start_ms=2 * DAY_MS, end_ms=3 * DAY_MS,
                            duration_ms=2 * DAY_MS, censored=False)
    assert censored == dict(peak_timestamp_ms=3 * DAY_MS, peak_equity=120,
                           underwater_start_ms=4 * DAY_MS, end_ms=4 * DAY_MS,
                           duration_ms=DAY_MS, censored=True)


def test_first_sample_loss_uses_prefill_baseline_and_incomplete_tail_is_not_fabricated():
    result = account(timestamps_ms=[0, DAY_MS], equity=[90, 80], gross_exposure_ratio=[2, 0],
                     fills=[], completed=False)
    assert result["net_return"] == pytest.approx(-.2)
    assert result["account_max_drawdown"] == .2
    assert result["underwater_fraction"] == 1
    assert result["gross_exposure_ratio_mean"] == 2
    assert result["completion_ratio"] == .25
    assert result["observed_end_ms"] == DAY_MS
    assert result["recovery_episodes"][0]["duration_ms"] == DAY_MS
    assert result["recovery_episodes"][0]["censored"]


def test_liquidation_and_nonpositive_equity_remain_explicit_without_fake_growth():
    result = account(equity=[100, 120, 90, 0, -10], liquidated=True)
    assert result["liquidated"] and result["nonpositive_equity_observed"]
    assert result["geometric_growth_per_day"] is None
    assert result["log_growth_per_day"] is None
    assert result["growth_unavailable_reason"] == "nonpositive_terminal_equity"
    assert result["account_max_drawdown"] == pytest.approx(130 / 120)


def test_declared_final_partial_grid_interval_is_time_weighted():
    result = account(expected_end_ms=DAY_MS + 100, timestamps_ms=[0, DAY_MS, DAY_MS + 100],
                     equity=[100, 90, 95], gross_exposure_ratio=[1, 2, 0], fills=[])
    assert result["gross_exposure_ratio_mean"] == pytest.approx((DAY_MS + 200) / (DAY_MS + 100))
    assert result["underwater_fraction"] == pytest.approx(100 / (DAY_MS + 100))


def test_native_numpy_observations_are_supported_without_rounding_fractional_clocks():
    import numpy as np

    result = account(timestamps_ms=np.arange(5, dtype=np.int64) * DAY_MS,
                     equity=np.array([100, 120, 90, 120, 108], dtype=np.float32),
                     gross_exposure_ratio=np.array([0, 2, 1, .5, 0], dtype=np.float32))
    assert result["account_max_drawdown"] == .25
    assert result["net_return"] == pytest.approx(.08)
    with pytest.raises(ValueError, match="exact integer"):
        account(timestamps_ms=[0, DAY_MS + .5, 2 * DAY_MS, 3 * DAY_MS, 4 * DAY_MS])


@pytest.mark.parametrize("updates,match", [
    ({"timestamps_ms": [0, DAY_MS, 3 * DAY_MS, 4 * DAY_MS],
      "equity": [100] * 4, "gross_exposure_ratio": [0] * 4}, "grid"),
    ({"equity": [100, float("nan"), 90, 120, 108]}, "finite"),
    ({"initial_equity": 10 ** 1000}, "finite"),
    ({"gross_exposure_ratio": [0, -1, 0, 0, 0]}, "nonnegative"),
    ({"completed": None}, "booleans"),
    ({"timestamps_ms": [0, DAY_MS], "equity": [100, 101],
      "gross_exposure_ratio": [0, 0]}, "completed"),
    ({"fills": [dict(identity="x", timestamp_ms=0)]}, "missing"),
    ({"fills": [dict(identity="x", timestamp_ms=5 * DAY_MS, net_realized_pnl=0,
                     fee_cashflow=0, notional=0, is_close=False)]}, "coverage"),
])
def test_missing_invalid_or_partial_account_inputs_fail(updates, match):
    with pytest.raises(ValueError, match=match):
        account(**updates)


def effective(gene=1):
    return dict(version="v9.0.0", bot={"long": {"gene": gene}},
                live={"lookback_ema_span_minutes": 100, "user": "operator"}, coin_overrides={})


def protocol(**updates):
    # Synthetic test thresholds only; production protocols must be calibrated.
    result = dict(version=1, scenarios={"base": "a" * 64, "stress": "b" * 64},
                  heldout_scenarios={"heldout": "c" * 64}, runtime_identities={"v9": "d" * 64},
                  axes=[dict(metric="net_return", goal="max", reducer="min", novelty_tolerance=.01),
                        dict(metric="account_max_drawdown", goal="min", reducer="max", novelty_tolerance=.01)],
                  constraints=[dict(metric="account_max_drawdown", op="<=", value=.4),
                               dict(metric="net_return", op=">=", value=0)],
                  minimum_activity={"close_fill_count": 1}, target_k=2, time_budget_seconds=100,
                  target_measure="retained_frontier", timer_origin="before_preparation",
                  timing_phases=sorted(COLD_PHASES))
    result.update(updates)
    return result


def evidence(gene, gain, drawdown, *, at=10, export=None, scenarios=None, engine="cpu", **metrics):
    config = effective(gene)
    identity = canonical_configuration_identity(config, model="v9")
    p = protocol()
    fields = dict(metric_definition="ordinary_account_v1", net_return=gain,
                  account_max_drawdown=drawdown, close_fill_count=2, completed=True,
                  liquidated=False, nonpositive_equity_observed=False)
    fields.update(metrics)
    confirmations = {}
    for label in scenarios if scenarios is not None else ["base", "stress", "heldout"]:
        confirmations[label] = dict(candidate_sha256=identity, input_sha256={
            **p["scenarios"], **p["heldout_scenarios"]}[label], runtime_sha256="d" * 64,
            engine=engine, completed_elapsed_seconds=at, metrics=deepcopy(fields))
    return dict(model="v9", effective_config=config, observed_elapsed_seconds=0,
                confirmations=confirmations,
                export_completed_elapsed_seconds=at if export is None else export)


def counter(**settings):
    return DiscoveryCounter(protocol(**settings), starting_pool=[evidence(0, .1, .2, at=5)])


def compound_evidence(gene, gain, drawdown, **kwargs):
    record = evidence(gene, gain, drawdown, **kwargs)
    config = record.pop("effective_config")
    configs = {}
    for label, month in (("base", "01"), ("stress", "03"), ("heldout", "08")):
        configs[label] = deepcopy(config)
        configs[label]["backtest"] = dict(
            start_date=f"2026-{month}-01", end_date=f"2026-{month}-02",
            starting_balance=100000.0, exchanges=["synthetic"], maker_fee=0.0002,
        )
    hashes = {label: canonical_configuration_identity(config, model="v9")
              for label, config in configs.items()}
    identity = compound_candidate_identity(hashes, model="v9")
    record["scenario_effective_configs"] = configs
    for label, confirmation in record["confirmations"].items():
        confirmation.update(candidate_sha256=identity, scenario_configuration_sha256=hashes[label])
    return record


def test_compound_identity_binds_each_exact_policy_with_dates_and_stable_label_order():
    record = compound_evidence(1, .2, .3)
    hashes = {label: canonical_configuration_identity(config, model="v9")
              for label, config in record["scenario_effective_configs"].items()}
    assert len(set(hashes.values())) == 3  # Different windows remain distinct.
    identity = compound_candidate_identity(hashes, model="v9")
    assert compound_candidate_identity(dict(reversed(list(hashes.items()))), model="v9") == identity
    assert compound_candidate_identity(hashes, model="v8") != identity
    renamed = dict(hashes, other=hashes["heldout"])
    del renamed["heldout"]
    assert compound_candidate_identity(renamed, model="v9") != identity
    assert identity == record["confirmations"]["base"]["candidate_sha256"]


@pytest.mark.parametrize("section,key,value", [
    ("bot", "long", {"gene": 2}),
    ("coin_overrides", "BTC", {"bot": {"long": {"gene": 3}}}),
    ("backtest", "exchanges", ["different_venue"]),
    ("backtest", "starting_balance", 12000.0),
    ("backtest", "maker_fee", -0.0001),
    ("backtest", "start_date", "2026-07-31"),
    ("backtest", "end_date", "2026-08-03"),
    ("live", "lookback_ema_span_minutes", 101.0),
])
def test_changed_heldout_policy_cannot_reuse_compound_confirmation(section, key, value):
    record = compound_evidence(1, .2, .3)
    record["scenario_effective_configs"]["heldout"][section][key] = value
    with pytest.raises(ValueError, match="different effective configuration"):
        counter().add(record)


def test_compound_partial_confirmations_merge_once_and_retain_exact_scenario_proofs():
    records = DiscoveryCounter(protocol(target_k=1), starting_pool=[compound_evidence(0, .1, .2, at=5)])
    first = compound_evidence(1, .2, .3, scenarios=["base", "stress"], at=10)
    identity = records.add(first)
    assert len(records.report()["pending"]) == 1
    assert records.add(compound_evidence(1, .2, .3, scenarios=["heldout"], at=30)) == identity
    assert records.report()["time_to_target_seconds"] == 30
    records.add(compound_evidence(1, .2, .3, at=40))
    assert records.report()["unique_observed"] == 2  # Starting reference plus candidate.
    assert records.report()["time_to_target_seconds"] == 30


@pytest.mark.parametrize("wrong", [None, "f" * 64, "other_scenario"])
def test_compound_confirmation_requires_actual_selected_scenario_policy(wrong):
    record = compound_evidence(1, .2, .3)
    confirmation = record["confirmations"]["base"]
    if wrong == "other_scenario":
        wrong = record["confirmations"]["heldout"]["scenario_configuration_sha256"]
    if wrong is None:
        del confirmation["scenario_configuration_sha256"]
    else:
        confirmation["scenario_configuration_sha256"] = wrong
    with pytest.raises(ValueError, match="scenario configuration differs"):
        counter().add(record)


@pytest.mark.parametrize("labels", [[], ["base", "stress"], ["base", "stress", "heldout", "extra"]])
def test_compound_candidate_must_bind_exact_frozen_scenario_set(labels):
    record = compound_evidence(1, .2, .3)
    configs = record["scenario_effective_configs"]
    record["scenario_effective_configs"] = {label: configs.get(label, effective()) for label in labels}
    with pytest.raises(ValueError, match="exactly the frozen"):
        counter().add(record)


@pytest.mark.parametrize("mapping", [{}, {"": "a" * 64}, {"base": "not-a-sha"}, {1: "a" * 64}])
def test_compound_identity_requires_explicit_valid_label_hash_mapping(mapping):
    with pytest.raises(ValueError):
        compound_candidate_identity(mapping, model="v9")


def test_compound_candidate_cannot_have_ambiguous_or_unmaterialized_policy():
    record = compound_evidence(1, .2, .3)
    record["effective_config"] = effective()
    with pytest.raises(ValueError, match="exactly one"):
        counter().add(record)
    del record["effective_config"]
    record["scenario_effective_configs"]["heldout"]["coin_overrides"] = {
        "BTC": {"override_config_path": "unresolved.json"},
    }
    with pytest.raises(ValueError, match="materialize"):
        counter().add(record)


def test_canonical_identity_ignores_export_noise_but_preserves_every_policy_difference():
    source = effective(1)
    first = canonical_configuration_identity(source, model="v9")
    noisy = deepcopy(source)
    noisy["analysis"] = {"fitness": 123}
    noisy["optimize"] = {"seed": 2}
    noisy["version"] = "v8.0.0"
    noisy["live"].update(user="another", _audit="ignored", base_config_path="source.json")
    noisy["bot"]["long"]["gene"] = 1.0
    assert canonical_configuration_identity(noisy, model="v9") == first
    for changed in (effective(1.000001), effective("1"), effective(True), effective({"number": "1"})):
        assert canonical_configuration_identity(changed, model="v9") != first
    changed = deepcopy(source)
    changed["coin_overrides"]["BTC"] = {"bot": {"long": {"gene": 2}}}
    assert canonical_configuration_identity(changed, model="v9") != first
    changed = deepcopy(source)
    changed["bot"]["long"]["economic_path"] = "policy"
    assert canonical_configuration_identity(changed, model="v9") != first
    assert canonical_configuration_identity(source, model="v8") != first
    assert source == effective(1)


def test_canonical_numbers_are_lossless_and_distinct_from_tag_shaped_user_data():
    number = 123456789012345678901234567890123456789
    assert canonical_configuration_identity(effective(number), model="v9") != \
        canonical_configuration_identity(effective(number + 1), model="v9")
    assert canonical_configuration_identity(effective(-0.0), model="v9") == \
        canonical_configuration_identity(effective(0), model="v9")
    assert canonical_configuration_identity(effective(["number", "1e0"]), model="v9") != \
        canonical_configuration_identity(effective(1), model="v9")


def test_unmaterialized_policy_cannot_get_a_candidate_identity():
    config = effective()
    config["coin_overrides"]["BTC"] = {"override_config_path": "unresolved.json"}
    with pytest.raises(ValueError, match="materialize"):
        canonical_configuration_identity(config, model="v9")
    config["coin_overrides"] = {"BTC": {}}
    materialized = canonical_configuration_identity(config, model="v9")
    config["coin_overrides"]["BTC"]["override_config_path"] = ""
    assert canonical_configuration_identity(config, model="v9") == materialized


def test_duplicates_near_copies_dominated_and_infeasible_do_not_inflate_useful_count():
    records = counter()
    records.add(evidence(1, .1001, .2001, at=10))  # Near baseline.
    records.add(evidence(2, .15, .25, at=20))  # New return/risk tradeoff.
    records.add(evidence(2, .15, .25, at=40))  # Same effective policy, CPU confirmation reused.
    records.add(evidence(3, .12, .3, at=30))  # Dominated.
    records.add(evidence(4, .5, .45, at=30))  # Drawdown constraint.
    records.add(evidence(5, .5, .1, at=30, close_fill_count=0))  # Inactive.
    report = records.report()
    assert report["unique_observed"] == 6
    assert len(report["broad_feasible_discoveries"]) == 3
    assert len(report["useful_frontier_discoveries"]) == 1
    assert report["useful_frontier_discoveries"][0]["confirmed_exported_seconds"] == 20
    assert len(report["infeasible"]) == 2
    assert report["retained_useful_count"] == 1
    assert report["right_censored"] and report["censor_seconds"] == 100


def test_time_to_k_counts_cpu_all_scenarios_heldout_and_export_in_cold_clock():
    records = counter()
    first = evidence(1, .2, .3, at=20, export=25)
    first["confirmations"]["heldout"]["completed_elapsed_seconds"] = 40
    records.add(first)
    records.add(evidence(2, .08, .1, at=30, export=50))
    report = records.report()
    assert report["target_reached"] and report["time_to_target_seconds"] == 50
    assert [row["confirmed_exported_seconds"] for row in report["useful_frontier_discoveries"]] == [40, 50]
    assert report["retained_useful_count"] == 2
    assert not report["right_censored"] and report["censor_seconds"] is None
    assert len(report["protocol_sha256"]) == 64


def test_partial_confirmations_merge_but_gpu_never_counts_as_cpu():
    records = counter(target_k=1)
    records.add(evidence(1, .2, .3, scenarios=["base", "stress"], at=10))
    assert len(records.report()["pending"]) == 1
    records.add(evidence(1, .2, .3, scenarios=["heldout"], engine="gpu", at=15))
    assert len(records.report()["pending"]) == 1
    records.add(evidence(1, .2, .3, scenarios=["heldout"], engine="cpu", at=30))
    assert records.report()["time_to_target_seconds"] == 30
    records.add(evidence(1, .3, .1, scenarios=["heldout"], engine="gpu", at=40))
    assert records.report()["time_to_target_seconds"] == 30


def test_export_pending_and_confirmation_past_budget_stay_censored():
    records = counter(target_k=1)
    pending = evidence(1, .2, .3)
    pending["export_completed_elapsed_seconds"] = None
    records.add(pending)
    records.add(evidence(2, .3, .35, at=90, export=101))
    report = records.report()
    assert len(report["pending"]) == len(report["late_confirmed_feasible"]) == 1
    assert not report["useful_frontier_discoveries"]
    assert report["right_censored"] and report["time_to_target_seconds"] is None


def test_baseline_confirmation_is_in_denominator_and_cannot_arrive_after_budget():
    records = DiscoveryCounter(protocol(target_k=1), starting_pool=[evidence(0, .1, .2, at=80)])
    records.add(evidence(1, .2, .3, at=10))
    assert records.report()["time_to_target_seconds"] == 80
    late = DiscoveryCounter(protocol(target_k=1), starting_pool=[evidence(0, .1, .2, at=110)])
    late.add(evidence(1, .2, .3, at=10))
    report = late.report()
    assert report["right_censored"] and not report["baseline_ready_within_budget"]
    assert report["current_frontier"] == {}


def test_retained_frontier_and_ever_addition_targets_are_distinct():
    retained, ever = counter(), counter(target_measure="ever_frontier_additions")
    for records in (retained, ever):
        records.add(evidence(1, .2, .2, at=10))
        records.add(evidence(2, .3, .2, at=20))
    assert retained.report()["retained_useful_count"] == 1
    assert retained.report()["right_censored"]
    assert ever.report()["time_to_target_seconds"] == 20
    assert len(ever.report()["useful_frontier_discoveries"]) == 2


def test_same_time_batch_never_counts_its_dominated_member():
    records = counter(target_measure="ever_frontier_additions")
    records.add(evidence(1, .2, .3, at=10))
    records.add(evidence(2, .3, .2, at=10))
    assert len(records.report()["broad_feasible_discoveries"]) == 2
    assert len(records.report()["useful_frontier_discoveries"]) == 1


def test_heldout_only_tests_economics_and_never_improves_search_objective():
    records = counter(target_k=1)
    heldout_star = evidence(1, .09, .21)
    heldout_star["confirmations"]["heldout"]["metrics"].update(net_return=100, account_max_drawdown=0)
    records.add(heldout_star)
    failed = evidence(2, .5, .1)
    failed["confirmations"]["heldout"]["metrics"]["account_max_drawdown"] = .5
    records.add(failed)
    report = records.report()
    assert not report["useful_frontier_discoveries"]
    assert len(report["infeasible"]) == 1


@pytest.mark.parametrize("field", ["candidate_sha256", "input_sha256", "runtime_sha256"])
def test_confirmation_lineage_mismatch_is_an_error(field):
    record = evidence(1, .2, .3)
    record["confirmations"]["base"][field] = "f" * 64
    with pytest.raises(ValueError, match="different|differs"):
        counter().add(record)


def test_conflicting_authoritative_cpu_confirmation_is_never_silently_replaced():
    records = counter()
    records.add(evidence(1, .2, .3))
    with pytest.raises(ValueError, match="conflicting"):
        records.add(evidence(1, .3, .1))


@pytest.mark.parametrize("metric,value,match", [
    ("metric_definition", "strategy_equity", "ordinary-account"),
    ("completed", None, "boolean"),
    ("net_return", float("nan"), "finite"),
])
def test_missing_or_wrong_account_evidence_cannot_become_neutral_score(metric, value, match):
    record = evidence(1, .2, .3)
    record["confirmations"]["base"]["metrics"][metric] = value
    records = counter()
    records.add(record)
    with pytest.raises(ValueError, match=match):
        records.report()


@pytest.mark.parametrize("updates", [
    {"constraints": []}, {"minimum_activity": {}}, {"minimum_activity": {"fill_count": 0}},
    {"axes": [dict(metric="adg_strategy_eq", goal="max", reducer="min", novelty_tolerance=0)]},
    {"heldout_scenarios": {}}, {"timer_origin": "after_warmup"}, {"timing_phases": ["search"]},
    {"heldout_scenarios": {"renamed_base": "a" * 64}},
    {"target_measure": "unspecified"},
])
def test_protocol_requires_explicit_economics_activity_novelty_heldout_and_cold_scope(updates):
    with pytest.raises(ValueError):
        validate_protocol(protocol(**updates))


def test_starting_pool_requires_complete_cpu_evidence():
    records = DiscoveryCounter(protocol(), starting_pool=[evidence(0, .1, .2, scenarios=["base"])])
    with pytest.raises(ValueError, match="starting reference"):
        records.report()


def test_caller_mutation_cannot_rewrite_frozen_protocol_or_evidence():
    frozen = protocol(target_k=1)
    records = DiscoveryCounter(frozen, starting_pool=[evidence(0, .1, .2, at=5)])
    frozen["constraints"][0]["value"] = 100
    exposed = records.protocol
    exposed["constraints"][0]["value"] = 100
    row = evidence(1, .5, .5)
    records.add(row)
    row["confirmations"]["base"]["metrics"]["account_max_drawdown"] = 0
    assert len(records.report()["infeasible"]) == 1


def test_duplicate_fill_and_missing_objective_are_explicit_errors():
    fill = dict(identity="same", timestamp_ms=0, net_realized_pnl=0,
                fee_cashflow=0, notional=1, is_close=False)
    with pytest.raises(ValueError, match="unique"):
        account(fills=[fill, deepcopy(fill)])
    records = counter(axes=[dict(metric="geometric_growth_per_day", goal="max", reducer="min",
                                novelty_tolerance=0)])
    with pytest.raises(ValueError, match="missing required account objective"):
        records.report()


def test_observation_arrival_order_does_not_change_confirmation_time_discoveries():
    earlier, later = evidence(1, .2, .3, at=10), evidence(2, .3, .2, at=20)
    one, two = counter(), counter()
    for row in (earlier, later):
        one.add(row)
    for row in (later, earlier):
        two.add(row)
    assert one.report() == two.report()


@pytest.mark.parametrize("retired", ["hsl_never", "hsl_threshold", "hsl_unified"])
def test_retired_benchmark_feature_requests_fail_before_workload_materialization(retired):
    from tools.optimizer_benchmark.catalogue import Case
    from tools.optimizer_benchmark.workloads import apply_features

    with pytest.raises(ValueError, match="retired benchmark feature; freeze a current v9 workload"):
        apply_features({}, Case("trailing_martingale", features=(retired,)), [], {}, 128)


def test_v9_benchmark_missing_native_service_fails_before_device_import(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from tools.optimizer_benchmark import worker

    monkeypatch.setitem(sys.modules, "optimization.gpu.native", ModuleType("optimization.gpu.native"))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requires NativeBacktestService.*legacy direct replay is unsupported"):
        worker._gpu_runs({"configs": [{"config_version": "v9.0.0"}]}, (), "1", 1, 0, 0, 1)
