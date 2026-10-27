"""Full native account artifacts, synthetic only; neither version's runner runs."""

from copy import deepcopy

import numpy as np
import pytest

from tools.optimizer_benchmark.replay_account import (
    account_valuation_input_identity,
    cpu_confirmation_from_native,
    metrics_from_native_replay,
)


def native_fill(index, coin, qty, post_size, price, kind, *, pnl=0, fee=0, timestamp=None):
    # Native columns 5..8/12 and trailing exposure diagnostics are deliberately
    # irrelevant: neither balance division nor changed strategy-equity is used.
    return [index, index * 60000 if timestamp is None else timestamp, coin, pnl, fee,
            0, 0, 0, 0, qty, price, post_size, 0, kind, "maker", 9999, 9999, -9999, 0]


def prepared():
    close = np.array([[100, 10], [110, 12], [90, 15], [120, 20], [100, 20]], dtype=float)
    candles = np.stack([close * 1.01, close * .99, close, np.full_like(close, 1000)], axis=2)
    return dict(hlcvs=candles, candle_timestamps_ms=np.arange(5, dtype=np.int64) * 60000,
                coins=["A", "B"], market_settings={"A": {"c_mult": 2.0}, "B": {"c_mult": 3.0}},
                initial_capital=1000.0, first_bar_index=0, last_bar_index=3, candle_interval_ms=60000,
                bootstrap_bar_indices=[])


def artifacts():
    fills = [native_fill(0, "A", 2, 2, 100, "entry_initial_long", fee=-1),
             native_fill(0, "A", -1, -1, 100, "entry_initial_short", fee=-.5),
             native_fill(1, "A", -1, 1, 110, "close_grid_long", pnl=20, fee=.2),
             native_fill(1, "B", -2, -2, 12, "entry_initial_short", fee=-1),
             native_fill(2, "A", 1, 0, 90, "close_grid_short", pnl=20, fee=-.1),
             native_fill(3, "A", -1, 0, 120, "close_grid_long", pnl=40, fee=-.4),
             native_fill(3, "B", 2, 0, 20, "close_panic_short", pnl=-48, fee=-.2, timestamp=240000)]
    equities = np.column_stack([np.arange(4) * 60000, [998.5, 1017.7, 999.6, 1029],
                                np.full(4, np.nan), np.full(4, -1e99)])
    return fills, equities


def extract(*, inputs=None, fills=None, equities=None, **updates):
    inputs = prepared() if inputs is None else inputs
    default_fills, default_equities = artifacts()
    return metrics_from_native_replay(
        native_fills=default_fills if fills is None else fills,
        native_equities=default_equities if equities is None else equities,
        liquidated=False, execution_completed=True, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs, **updates)


def test_full_account_clock_fees_hedges_and_multipliers_are_independent_of_risk_curves():
    result = extract()
    assert result["initial_equity"] == 1000
    assert result["net_return_fraction"] == pytest.approx(.029)
    assert result["net_realized_pnl"] == pytest.approx(29)
    assert result["fees_paid"] == pytest.approx(3.2)
    assert result["rebates_received"] == .2
    assert result["net_fees"] == pytest.approx(3)
    assert result["traded_notional"] == 1432
    assert result["gross_exposure_max_ratio"] == .6  # Long+short gross, never net.
    assert result["gross_exposure_mean_ratio"] == pytest.approx((.6 + .512 + .27) / 4)
    assert result["exposed_time_fraction"] == .75
    assert result["gross_exposure_terminal_ratio"] == 0
    assert result["exposure_denominator"] == "initial_capital"
    assert result["start_ms"] == 0 and result["observed_end_ms"] == 240000
    assert result["duration_days"] == pytest.approx(4 / 1440)
    assert result["completed"] and not result["liquidated"]
    assert result["fill_count"] == 7 and result["close_fill_count"] == 4
    assert result["account_drawdown_max_fraction"] == pytest.approx((1017.7 - 999.6) / 1017.7)


def test_next_open_position_is_never_backfilled_to_preceding_same_clock_close():
    inputs = prepared()
    fills = [native_fill(1, "A", 1, 1, 110, "entry_initial_long", timestamp=60000)]
    equities = np.array([[0, 1000], [60000, 1000], [120000, 980], [180000, 1040]])
    result = extract(inputs=inputs, fills=fills, equities=equities)
    # At time60000 the close[0] observation is still flat, despite the next
    # opening fill sharing its timestamp. Its first close mark is close[1].
    assert result["gross_exposure_mean_ratio"] == pytest.approx((.22 + .18) / 4)
    assert result["exposed_time_fraction"] == .5
    assert result["gross_exposure_terminal_ratio"] == .24


def test_liquidated_zero_balance_does_not_hide_or_invent_exposure():
    inputs = prepared()
    result = metrics_from_native_replay(
        native_fills=[native_fill(0, "A", 1, 1, 100, "entry_initial_long")],
        native_equities=np.array([[0, 0]]), liquidated=True, execution_completed=True,
        initial_positions={}, valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert not result["completed"] and result["liquidated"]
    assert result["completion_fraction"] == .25
    assert result["gross_exposure_max_ratio"] == result["gross_exposure_terminal_ratio"] == .2
    assert result["gross_exposure_mean_ratio"] == 0  # No fabricated pre-close exposure.
    assert result["net_return_fraction"] == -1 and result["log_return_per_day"] is None


def test_failed_execution_uses_only_observed_prefix_and_keeps_completion_false():
    inputs = prepared()
    fills, equities = artifacts()
    result = metrics_from_native_replay(native_fills=fills[:4], native_equities=equities[:2],
        liquidated=False, execution_completed=False, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert not result["completed"] and result["completion_fraction"] == .5
    assert result["observed_end_ms"] == 120000


def test_explicit_bootstrap_row_and_warmup_gap_do_not_invent_evaluated_coverage():
    inputs = prepared()
    inputs.update(first_bar_index=2, bootstrap_bar_indices=[0])
    equities = np.array([[0, 1000], [120000, 1000], [180000, 1000]])
    result = extract(inputs=inputs, fills=[], equities=equities)
    assert result["completed"] and result["completion_fraction"] == 1
    assert result["start_ms"] == 120000 and result["observed_end_ms"] == 240000
    assert result["duration_days"] == pytest.approx(2 / 1440)
    assert result["sample_count"] == 3
    assert result["excluded_bootstrap_bar_indices"] == [0]
    # A pre-evaluation fill makes the known-flat initialization unproven, even
    # when an account-equity row happens to equal its initial capital.
    with pytest.raises(ValueError, match="observed account coverage"):
        extract(inputs=inputs, equities=equities,
                fills=[native_fill(1, "A", 1, 1, 110, "entry_initial_long")])


def test_bootstrap_row_requires_frozen_declaration_and_exact_capital():
    inputs = prepared()
    inputs["first_bar_index"] = 2
    equities = np.array([[0, 1000], [120000, 1000], [180000, 1000]])
    with pytest.raises(ValueError, match="contiguous prefix"):
        extract(inputs=inputs, fills=[], equities=equities)
    inputs["bootstrap_bar_indices"] = [0]
    equities[0, 1] = 999
    with pytest.raises(ValueError, match="known-flat initial capital"):
        extract(inputs=inputs, fills=[], equities=equities)


def test_missing_held_mark_fails_but_inactive_coin_gap_is_not_fabricated():
    inputs = prepared()
    inputs["hlcvs"][0, 1, 2] = np.nan  # B is still flat at close0.
    assert extract(inputs=inputs)["completed"]
    inputs["hlcvs"][2, 1, 2] = np.nan  # B short is now held.
    with pytest.raises(ValueError, match="held B close mark"):
        extract(inputs=inputs)


def test_frozen_inputs_are_verified_and_bridge_does_not_mutate_them():
    inputs = prepared()
    before = inputs["hlcvs"].copy()
    identity = account_valuation_input_identity(**inputs)
    extract(inputs=inputs)
    np.testing.assert_array_equal(before, inputs["hlcvs"])
    assert account_valuation_input_identity(**inputs) == identity
    inputs["hlcvs"][0, 0, 2] += 1
    fills, equities = artifacts()
    with pytest.raises(ValueError, match="frozen identity"):
        metrics_from_native_replay(native_fills=fills, native_equities=equities,
            liquidated=False, execution_completed=True, initial_positions={},
            valuation_input_sha256=identity, **inputs)


@pytest.mark.parametrize("change,match", [
    (lambda fills, equities: fills.pop(0), "position transitions"),
    (lambda fills, equities: fills[0].__setitem__(13, "unknown"), "attribution"),
    (lambda fills, equities: fills[0].__setitem__(9, -2), "sign contradicts"),
    (lambda fills, equities: fills[0].__setitem__(11, 3), "position transitions"),
    (lambda fills, equities: fills[0].__setitem__(1, 60001), "bar index"),
    (lambda fills, equities: equities.__setitem__((2, 0), 60000), "contiguous prefix"),
    (lambda fills, equities: equities.__setitem__((2, 1), np.nan), "ordinary account equity"),
])
def test_corrupt_or_partial_full_tapes_fail_explicitly(change, match):
    fills, equities = artifacts()
    change(fills, equities)
    with pytest.raises(ValueError, match=match):
        extract(fills=fills, equities=equities)


def test_successful_missing_tail_and_metrics_only_outputs_are_not_neutral():
    fills, equities = artifacts()
    with pytest.raises(ValueError, match="omitted required"):
        extract(fills=fills[:4], equities=equities[:2])
    inputs = prepared()
    with pytest.raises(ValueError, match="metrics-only"):
        metrics_from_native_replay(native_fills=None, native_equities=equities,
            liquidated=False, execution_completed=True, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)


def test_nonflat_start_and_missing_multipliers_require_another_supported_adapter():
    inputs = prepared()
    fills, equities = artifacts()
    with pytest.raises(ValueError, match="known-flat"):
        metrics_from_native_replay(native_fills=fills, native_equities=equities,
            liquidated=False, execution_completed=True, initial_positions={"A": 1},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    del inputs["market_settings"]["A"]["c_mult"]
    with pytest.raises(ValueError, match="multiplier"):
        extract(inputs=inputs)


def test_cpu_confirmation_binds_own_runtime_config_and_samples_clock_after_extraction():
    inputs = prepared()
    fills, equities = artifacts()
    config = dict(bot={"long": {"gene": 1}}, live={}, coin_overrides={})
    result = (fills, equities, {"liquidated": False, "adg_fraction_strategy_equity": -999}, {}, {})
    calls = []
    def clock():
        calls.append("after account extraction")
        return 42.5
    row = cpu_confirmation_from_native(
        result, model="v8", effective_config=config, input_sha256="a" * 64, runtime_sha256="b" * 64,
        elapsed_seconds=clock, execution_completed=True, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert calls == ["after account extraction"]
    assert row["engine"] == "cpu" and row["completed_elapsed_seconds"] == 42.5
    assert row["input_sha256"] == "a" * 64 and row["runtime_sha256"] == "b" * 64
    assert row["metrics"]["net_return_fraction"] == pytest.approx(.029)
    assert "adg_fraction_strategy_equity" not in row["metrics"]
    unknown = deepcopy(result)
    del unknown[2]["liquidated"]
    with pytest.raises(ValueError, match="liquidation status"):
        cpu_confirmation_from_native(unknown, model="v9", effective_config=config,
            input_sha256="a" * 64, runtime_sha256="b" * 64, elapsed_seconds=clock,
            execution_completed=True, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)


def core_metadata(*, end=240000, count=5):
    return {"definition_version": 1, "start_timestamp_ms": 0,
            "intended_end_timestamp_ms": 240000, "actual_end_timestamp_ms": end,
            "observation_interval_ms": 60000, "equity_observation_count": count}


def core_artifacts():
    fills, legacy = artifacts()
    equities = legacy.copy()
    equities[:, 0] += 60000
    baseline = np.array([[0, 1000, np.nan, -1e99]])
    return fills, np.concatenate([baseline, equities])


def test_core_clock_bridge_uses_included_baseline_and_matches_legacy_account_ruler():
    from tools.optimizer_benchmark.account_metrics import SCORABLE_METRICS
    fills, equities = core_artifacts()
    current = extract(fills=fills, equities=equities, metric_metadata=core_metadata())
    legacy = extract()
    for name in SCORABLE_METRICS:
        assert current[name] == pytest.approx(legacy[name]), name
    assert current["sample_count"] == len(equities) == legacy["sample_count"]
    assert current["observed_end_ms"] == 240000
    assert current["native_equity_definition_version"] == 1
    assert legacy["native_equity_definition_version"] is None
    # A current layout without its version metadata must not be guessed/rebased.
    with pytest.raises(ValueError, match="contiguous prefix"):
        extract(fills=fills, equities=equities)


@pytest.mark.parametrize("field,value", [
    ("definition_version", 2), ("definition_version", True),
    ("start_timestamp_ms", 60000), ("actual_end_timestamp_ms", 180000),
    ("intended_end_timestamp_ms", 300000), ("observation_interval_ms", 120000),
    ("equity_observation_count", 4),
])
def test_core_clock_bridge_rejects_metadata_contradicting_frozen_inputs(field, value):
    fills, equities = core_artifacts()
    metadata = core_metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match="clock metadata"):
        extract(fills=fills, equities=equities, metric_metadata=metadata)


def test_core_clock_bridge_requires_baseline_full_grid_and_metadata_fields():
    fills, equities = core_artifacts()
    metadata = core_metadata()
    del metadata["actual_end_timestamp_ms"]
    with pytest.raises(ValueError, match="metadata is missing"):
        extract(fills=fills, equities=equities, metric_metadata=metadata)
    changed = equities.copy()
    changed[0, 1] = 999
    with pytest.raises(ValueError, match="baseline.*rebasing is unsupported"):
        extract(fills=fills, equities=changed, metric_metadata=core_metadata())
    changed = equities[[0, 1, 3, 4]]
    with pytest.raises(ValueError, match="contiguous prefix"):
        extract(fills=fills, equities=changed, metric_metadata=core_metadata(count=4))
    with pytest.raises(ValueError, match="omitted required"):
        extract(fills=fills[:5], equities=equities[:-1], metric_metadata=core_metadata(end=180000, count=4))
    changed = equities.copy()
    changed[-1, 0] = 300000
    with pytest.raises(ValueError, match="contiguous prefix"):
        extract(fills=fills, equities=changed, metric_metadata=core_metadata(end=300000))


@pytest.mark.parametrize("future_close", [9999.0, float("nan")])
@pytest.mark.parametrize("side,sign", [("long", 1), ("short", -1)])
def test_core_opening_terminal_uses_preceding_close_with_post_fill_inventory(future_close, side, sign):
    inputs = prepared()
    inputs["hlcvs"][2, 0, 2] = future_close  # close[2] has not completed at open[2].
    fills = [native_fill(0, "A", sign, sign, 100, f"entry_initial_{side}"),
             native_fill(2, "A", sign, 2 * sign, 99, f"entry_grid_{side}")]
    result = metrics_from_native_replay(
        native_fills=fills, native_equities=np.array([[0, 1000], [60000, 1000], [120000, 0]]),
        metric_metadata=core_metadata(end=120000, count=3), liquidated=True,
        execution_completed=True, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert result["observed_end_ms"] == 120000 and result["completion_fraction"] == .5
    assert result["sample_count"] == 3 and result["terminal_open_bar_index"] == 2
    assert not result["completed"] and result["liquidated"]
    assert result["gross_exposure_terminal_ratio"] == result["gross_exposure_max_ratio"] == .44
    assert result["gross_exposure_mean_ratio"] == .1
    assert result["net_return_fraction"] == -1 and result["fill_count"] == 2


def test_core_terminal_without_positive_duration_or_causal_marks_fails_closed():
    inputs = prepared()
    identity = account_valuation_input_identity(**inputs)
    with pytest.raises(ValueError, match="positive observed duration"):
        metrics_from_native_replay(native_fills=[], native_equities=np.array([[0, 0]]),
            metric_metadata=core_metadata(end=0, count=1), liquidated=True,
            execution_completed=True, initial_positions={}, valuation_input_sha256=identity, **inputs)
    inputs["hlcvs"][1, 0, 2] = np.nan
    with pytest.raises(ValueError, match="held A close mark"):
        metrics_from_native_replay(native_fills=[native_fill(2, "A", 1, 1, 99, "entry_initial_long")],
            native_equities=np.array([[0, 1000], [60000, 1000], [120000, 0]]),
            metric_metadata=core_metadata(end=120000, count=3), liquidated=True,
            execution_completed=True, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)


def test_current_confirmation_reads_declared_clock_metadata_and_rejects_null_metadata():
    fills, equities = core_artifacts()
    inputs = prepared()
    options = dict(model="v9", effective_config=dict(bot={}, live={}, coin_overrides={}),
                   input_sha256="a" * 64, runtime_sha256="b" * 64, elapsed_seconds=lambda: 1,
                   execution_completed=True, initial_positions={},
                   valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    result = (fills, equities, {"liquidated": False, "metric_metadata": core_metadata()}, {})
    row = cpu_confirmation_from_native(result, **options)
    assert row["metrics"]["native_equity_definition_version"] == 1
    assert row["metrics"]["sample_count"] == len(equities)
    result[2]["metric_metadata"] = None
    with pytest.raises(ValueError, match="clock metadata"):
        cpu_confirmation_from_native(result, **options)


def test_core_mark_liquidation_retains_completed_close_phase_without_opening_fill():
    inputs = prepared()
    result = metrics_from_native_replay(
        native_fills=[native_fill(0, "A", 1, 1, 100, "entry_initial_long")],
        native_equities=np.array([[0, 1000], [60000, 1000], [120000, 0]]),
        metric_metadata=core_metadata(end=120000, count=3), liquidated=True,
        execution_completed=True, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert result["terminal_open_bar_index"] is None
    assert result["last_native_bar_index"] == 1
    assert result["gross_exposure_terminal_ratio"] == .22
    assert result["observed_end_ms"] == 120000 and result["liquidated"]


@pytest.mark.parametrize("liquidated,completed,final_balance", [
    (False, False, 0), (True, False, 0), (True, True, 1000),
])
def test_core_partial_close_cannot_admit_an_unproven_next_open_fill(liquidated, completed, final_balance):
    inputs = prepared()
    opening = native_fill(2, "A", 1, 2, 99, "entry_grid_long")
    opening[5] = final_balance
    with pytest.raises(ValueError, match="observed account coverage"):
        metrics_from_native_replay(
            native_fills=[native_fill(0, "A", 1, 1, 100, "entry_initial_long"), opening],
            native_equities=np.array([[0, 1000], [60000, 1000], [120000, 0]]),
            metric_metadata=core_metadata(end=120000, count=3), liquidated=liquidated,
            execution_completed=completed, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)


@pytest.mark.parametrize("timestamp,balance,match", [
    (120001, 0, "observed account coverage"),
    (120000, True, "post-fill USD balance"),
    (120000, float("nan"), "post-fill USD balance"),
    (120000, float("inf"), "post-fill USD balance"),
])
def test_core_opening_terminal_requires_exact_clock_and_finite_balance(timestamp, balance, match):
    inputs = prepared()
    opening = native_fill(2, "A", 1, 2, 99, "entry_grid_long", timestamp=timestamp)
    opening[5] = balance
    with pytest.raises(ValueError, match=match):
        metrics_from_native_replay(
            native_fills=[native_fill(0, "A", 1, 1, 100, "entry_initial_long"), opening],
            native_equities=np.array([[0, 1000], [60000, 1000], [120000, 0]]),
            metric_metadata=core_metadata(end=120000, count=3), liquidated=True,
            execution_completed=True, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
