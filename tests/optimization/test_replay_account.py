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
    assert result["net_return"] == pytest.approx(.029)
    assert result["net_realized_pnl"] == pytest.approx(29)
    assert result["fees_paid"] == pytest.approx(3.2)
    assert result["rebates_received"] == .2
    assert result["net_fees"] == pytest.approx(3)
    assert result["traded_notional"] == 1432
    assert result["gross_exposure_ratio_max"] == .6  # Long+short gross, never net.
    assert result["gross_exposure_ratio_mean"] == pytest.approx((.6 + .512 + .27) / 4)
    assert result["exposed_fraction"] == .75
    assert result["gross_exposure_ratio_terminal"] == 0
    assert result["exposure_denominator"] == "initial_capital"
    assert result["start_ms"] == 0 and result["observed_end_ms"] == 240000
    assert result["duration_days"] == pytest.approx(4 / 1440)
    assert result["completed"] and not result["liquidated"]
    assert result["fill_count"] == 7 and result["close_fill_count"] == 4
    assert result["account_max_drawdown"] == pytest.approx((1017.7 - 999.6) / 1017.7)


def test_next_open_position_is_never_backfilled_to_preceding_same_clock_close():
    inputs = prepared()
    fills = [native_fill(1, "A", 1, 1, 110, "entry_initial_long", timestamp=60000)]
    equities = np.array([[0, 1000], [60000, 1000], [120000, 980], [180000, 1040]])
    result = extract(inputs=inputs, fills=fills, equities=equities)
    # At time60000 the close[0] observation is still flat, despite the next
    # opening fill sharing its timestamp. Its first close mark is close[1].
    assert result["gross_exposure_ratio_mean"] == pytest.approx((.22 + .18) / 4)
    assert result["exposed_fraction"] == .5
    assert result["gross_exposure_ratio_terminal"] == .24


def test_liquidated_zero_balance_does_not_hide_or_invent_exposure():
    inputs = prepared()
    result = metrics_from_native_replay(
        native_fills=[native_fill(0, "A", 1, 1, 100, "entry_initial_long")],
        native_equities=np.array([[0, 0]]), liquidated=True, execution_completed=True,
        initial_positions={}, valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert not result["completed"] and result["liquidated"]
    assert result["completion_ratio"] == .25
    assert result["gross_exposure_ratio_max"] == result["gross_exposure_ratio_terminal"] == .2
    assert result["gross_exposure_ratio_mean"] == 0  # No fabricated pre-close exposure.
    assert result["net_return"] == -1 and result["log_growth_per_day"] is None


def test_failed_execution_uses_only_observed_prefix_and_keeps_completion_false():
    inputs = prepared()
    fills, equities = artifacts()
    result = metrics_from_native_replay(native_fills=fills[:4], native_equities=equities[:2],
        liquidated=False, execution_completed=False, initial_positions={},
        valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
    assert not result["completed"] and result["completion_ratio"] == .5
    assert result["observed_end_ms"] == 120000


def test_explicit_bootstrap_row_and_warmup_gap_do_not_invent_evaluated_coverage():
    inputs = prepared()
    inputs.update(first_bar_index=2, bootstrap_bar_indices=[0])
    equities = np.array([[0, 1000], [120000, 1000], [180000, 1000]])
    result = extract(inputs=inputs, fills=[], equities=equities)
    assert result["completed"] and result["completion_ratio"] == 1
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
    result = (fills, equities, {"liquidated": False, "adg_strategy_eq": -999}, {}, {})
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
    assert row["metrics"]["net_return"] == pytest.approx(.029)
    assert "adg_strategy_eq" not in row["metrics"]
    unknown = deepcopy(result)
    del unknown[2]["liquidated"]
    with pytest.raises(ValueError, match="liquidation status"):
        cpu_confirmation_from_native(unknown, model="v9", effective_config=config,
            input_sha256="a" * 64, runtime_sha256="b" * 64, elapsed_seconds=clock,
            execution_completed=True, initial_positions={},
            valuation_input_sha256=account_valuation_input_identity(**inputs), **inputs)
