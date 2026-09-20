"""Executable examples for the proposed HSL, independent of production HSL."""

from dataclasses import replace
from decimal import Decimal
from itertools import permutations

import pytest

from hsl_reference import (
    MINUTE as M, Candle, Fill, LifecycleEvidence, Observation, Position,
    aggregate, dec, minimal_signal, minute_prices, permission, pnl,
    reconstruct, scope_budget, signal,
)


def rows(values):
    return tuple(Observation(i * M, dec(p), dec(u)) for i, (p, u) in enumerate(values))


def test_hand_calculated_equity_peak_and_fractional_ema():
    result = signal(rows([(0, 0), (0, -100), (0, -200)]), 800, 3, ".12")
    assert result.equity == (1000, 900, 800)
    assert result.peaks == (1000, 1000, 1000)
    assert result.raw == tuple(map(dec, [0, ".1", ".2"]))
    assert result.ema == tuple(map(dec, [0, ".05", ".125"]))
    assert result.panic == (False, False, True)
    fractional = signal(rows([(0, 0), (0, -100)]), 900, "2.5", ".05")
    assert float(fractional.ema[-1]) == pytest.approx(2 / 3.5 * .1)


def test_recovery_uses_min_of_raw_and_ema_not_ema_alone():
    result = signal(rows([(0, 0), (0, -100), (0, -200), (0, -50)]), 950, 3, ".06")
    assert result.raw[-1] == dec(".05")
    assert result.ema[-1] == dec(".0875")
    assert not result.panic[-1]


def test_past_unrealized_peak_counts_without_any_realized_pnl():
    result = signal(rows([(0, 0), (0, 200), (0, -100)]), 1000, 1, ".2")
    assert result.equity == (1100, 1300, 1000)
    assert float(result.raw[-1]) == pytest.approx(300 / 1300)
    assert result.panic[-1]


@pytest.mark.parametrize("threshold,expected", [(".124999", True), (".125", False), (".125001", False)])
def test_strict_threshold_boundary(threshold, expected):
    result = signal(rows([(0, 0), (0, -100), (0, -200)]), 800, 3, threshold)
    assert result.panic[-1] is expected


def test_same_minute_boundary_is_checked_without_advancing_ema_twice():
    samples = [Observation(0, dec(0), dec(0)),
               Observation(M, dec(-100), dec(0)),
               Observation(M + 1, dec(-200), dec(0))]
    result = signal(samples, 800, 3, ".075")
    assert result.ema == (dec(0), dec(".05"), dec(".1"))
    assert result.panic == (False, False, True)
    assert signal(samples, 800, 3, ".075") == result  # repeated poll is pure


@pytest.mark.parametrize("span", [1, "2.5", 10_000_000])
@pytest.mark.parametrize("upnl,expected", [(-100, True), (0, False), (100, False)])
def test_minimal_history_has_one_actual_ema_sample(span, upnl, expected):
    result = minimal_signal(upnl, 1000, span, ".09")
    assert result.ema == result.raw
    assert result.panic[-1] is expected
    assert result.equity == (1000,)
    assert float(result.raw[-1]) == pytest.approx(100 / 1100 if upnl < 0 else 0)


def test_no_implicit_zero_ema_seed_and_transition_to_real_history():
    sparse = minimal_signal(-100, 1000, 100, ".05")
    restored = signal(rows([(0, 0), (0, -100)]), 1000, 100, ".05")
    assert sparse.panic[-1] and not restored.panic[-1]
    # This discontinuity is explicit, not a reason to throw away retained history.
    assert restored.ema[0] == restored.raw[0] == 0


def test_aggregate_currency_before_peaks_opposite_paths_cancel():
    a, b = rows([(0, 0), (0, 100), (0, 0)]), rows([(0, 100), (0, 0), (0, 100)])
    combined = signal(aggregate([a, b]), 1000, 1, ".01")
    assert combined.raw == (0, 0, 0)
    assert signal(a, 1000, 1, ".01").panic[-1]
    assert aggregate([a, b]) == aggregate([b, a])


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_one_slot_scope_equivalence(mode):
    budget = scope_budget(mode, 1000, 1)
    assert signal(rows([(0, 100), (0, -100)]), budget, 1, ".1").panic[-1]


def test_inactive_coin_scope_does_not_get_invented_divisor():
    assert scope_budget("coin", 1000, 0) is None
    assert scope_budget("unified", 1000, 0) == 1000
    assert scope_budget("coin", 1000, 4) == 250


def test_budget_change_is_not_claimed_signal_invariant():
    tape = rows([(0, 0), (0, -100)])
    assert signal(tape, 100, 1, ".2").panic[-1]
    assert not signal(tape, 1000, 1, ".2").panic[-1]


def test_nonpositive_old_peak_has_explicit_impairment_sample():
    result = signal(rows([(0, -200), (0, -150), (0, 0)]), 100, 3, ".1")
    assert result.equity == (-100, -50, 100)
    assert result.raw == (1, 1, 0)
    assert result.ema == (1, 1, dec(".5"))
    assert not result.panic[-1]


def test_extreme_finite_amounts_do_not_create_nan_or_hide_current_budget():
    result = signal(rows([(0, "1e600"), (0, "-1e600")]), "1e-300", 1, ".9")
    assert result.equity[-1] == dec("1e-300")
    assert all(v.is_finite() for v in (*result.raw, *result.ema))
    assert result.panic[-1]


@pytest.mark.parametrize("budget,span,threshold", [(0, 1, .1), (-1, 1, .1), ("NaN", 1, .1),
                                                     (100, 0, .1), (100, "Infinity", .1),
                                                     (100, 1, -1), (100, 1, 2)])
def test_invalid_signal_domain_is_not_a_healthy_false(budget, span, threshold):
    with pytest.raises(ValueError):
        signal(rows([(0, 0)]), budget, span, threshold)


def clean_tape():
    position = Position("1.5", 90, 60)
    fills = [Fill("open", M, 1, 100, 0, -1),
             Fill("add", 2 * M, 1, 80, 0, -1),
             Fill("reduce", 3 * M, "-.5", 70, -10, -1)]
    prices = {0: 100, M: 100, 2 * M: 80, 3 * M: 70, 4 * M: 60}
    return position, fills, prices


def test_hand_calculated_fill_history_and_fee_accounting():
    position, fills, prices = clean_tape()
    history = reconstruct(position, fills, prices, 0, 4 * M)
    assert history.sizes == tuple(map(dec, [0, 1, 2, "1.5", "1.5"]))
    assert history.bases == (0, 100, 90, 90, 90)
    assert [r.pnl for r in history.rows] == [0, -1, -2, -13, -13]
    assert [r.upnl for r in history.rows] == [0, 0, -20, -30, -45]
    assert not history.reasons
    result = signal(history.rows, 1000, 1, ".05")
    assert result.equity == (1058, 1057, 1036, 1015, 1000)
    assert result.panic[-1]


@pytest.mark.parametrize("missing,equity,panic", [
    (0, (1077, 1077, 1056, 1030, 1000), True),
    (1, (1057, 1056, 1016, 1000, 1000), True),
    (2, (1047, 1046, 1035, 1020, 1000), False),
])
def test_missing_open_middle_or_latest_fill_still_evaluates_and_repairs(missing, equity, panic):
    position, fills, prices = clean_tape()
    damaged = reconstruct(position, fills[:missing] + fills[missing + 1:], prices, 0, 4 * M)
    assert damaged.sizes[-1] == dec(position.size)
    assert damaged.bases[-1] == dec(position.basis)
    assert damaged.rows[-1].upnl == -45
    result = signal(damaged.rows, 1000, 1, ".05")
    assert tuple(map(float, result.equity)) == pytest.approx(equity)
    assert result.panic[-1] is panic
    # Missing the latest reduction hides realized loss: this fixture documents
    # a missed clean-data stop, not merely successful computation of a bool.
    assert damaged.reasons
    repaired = reconstruct(position, fills, prices, 0, 4 * M)
    assert signal(repaired.rows, 1000, 1, ".05").panic[-1]
    assert not repaired.reasons


def test_missing_opening_uses_earliest_retained_price_without_poisoning_clean_episode():
    position = Position(2, 100, 80)
    fills = [Fill("old-add", M, 5, 50, 0), Fill("old-close", 2 * M, -7, 40, -70),
             Fill("new-open", 3 * M, 2, 100, 0)]
    history = reconstruct(position, fills, {M: 50, 2 * M: 40, 3 * M: 100, 4 * M: 80}, 0, 4 * M)
    assert history.sizes == (7, 0, 2, 2)
    assert history.bases == (50, 0, 100, 100)
    assert history.rows[-1].upnl == -40
    assert "estimated_opening_basis" in history.reasons
    current_episode = tuple(r for r in history.rows if r.timestamp >= 3 * M)
    assert signal(current_episode, 100, 1, ".2").panic[-1]


def test_duplicate_identity_and_explicit_revision_correction():
    position, fills, prices = clean_tape()
    expected = reconstruct(position, fills, prices, 0, 4 * M)
    assert reconstruct(position, [*fills, *fills], prices, 0, 4 * M) == expected
    correction = replace(fills[-1], realized=-12, revision=1)
    corrected = reconstruct(position, [*fills, correction], prices, 0, 4 * M)
    assert corrected.rows[-1].pnl == -15
    for permutation in permutations([fills[-1], correction]):
        assert reconstruct(position, [*fills[:2], *permutation], prices, 0, 4 * M) == corrected


def test_conflicting_identity_is_disclosed_and_excluded_not_double_counted():
    position, fills, prices = clean_tape()
    history = reconstruct(position, [*fills, replace(fills[-1], realized=-99)], prices, 0, 4 * M)
    assert "conflicting_identity" in history.reasons
    assert history.rows[-1].pnl == -2
    assert history.rows[-1].upnl == -45


def test_same_time_mixed_actions_are_deterministic_without_flat_certificate():
    position = Position(1, 100, 90)
    tape = [Fill("add", M, 1, 100, 0), Fill("close", M, -1, 90, -10)]
    results = [reconstruct(position, list(p), {0: 100, M: 90}, 0, M) for p in permutations(tape)]
    assert results[0] == results[1]
    assert "estimated_fill_order" in results[0].reasons
    assert results[0].rows[-1].pnl == -10


def test_known_sequence_wins_over_tie_convention():
    fills = [Fill("add", M, 1, 80, 0, sequence=2),
             Fill("close", M, -1, 90, -10, sequence=1)]
    history = reconstruct(Position(1, 80, 80), fills, {0: 100, M: 80}, 0, M)
    assert "estimated_fill_order" not in history.reasons
    assert history.bases[-1] == 80


@pytest.mark.parametrize("field,value,reason", [("delta", "NaN", "invalid_quantity"),
    ("price", "NaN", "estimated_fill_price"), ("fee", "NaN", "unknown_fee"),
    ("realized", None, "estimated_realized_pnl")])
def test_bad_historical_fields_degrade_locally(field, value, reason):
    position, fills, prices = clean_tape()
    fills[-1] = replace(fills[-1], **{field: value})
    history = reconstruct(position, fills, prices, 0, 4 * M)
    assert reason in history.reasons
    assert history.rows[-1].upnl == -45
    assert all(d.is_finite() for d in signal(history.rows, 1000, 1, ".1").raw)


def test_unknown_realized_loss_is_estimated_from_basis_without_double_fee():
    position, fills, prices = clean_tape()
    fills[-1] = replace(fills[-1], realized=None)
    assert reconstruct(position, fills, prices, 0, 4 * M).rows[-1].pnl == -13


def test_invalid_quantity_retains_independent_known_pnl_and_fee():
    position, fills, prices = clean_tape()
    fills[-1] = replace(fills[-1], delta="NaN")
    history = reconstruct(position, fills, prices, 0, 4 * M)
    assert "invalid_quantity" in history.reasons
    assert history.rows[-1].pnl == -13
    assert history.rows[-1].upnl == -45
    assert signal(history.rows, 1000, 1, ".05").panic[-1]


def test_partial_sequence_preserves_the_known_close_then_reopen():
    tape = [Fill("known-close", M, -1, 90, -10, sequence=1),
            Fill("known-add", M, 1, 80, 0, sequence=2),
            Fill("unknown-add", M, 1, 100, 0)]
    for order in permutations(tape):
        history = reconstruct(Position(2, 90, 90), list(order), {0: 90, M: 90, 2 * M: 90}, 0, 2 * M)
        assert history.bases[1] == 90
        assert history.rows[-1].pnl == -10
        assert "estimated_fill_order" in history.reasons


def test_missing_whole_roundtrip_cannot_be_recovered_from_current_quantity():
    position = Position(1, 100, 100)
    full = [Fill("open", M, 1, 100, 0), Fill("close", 2 * M, -1, 50, -50),
            Fill("reopen", 3 * M, 1, 100, 0)]
    prices = {0: 100, M: 100, 2 * M: 50, 3 * M: 100}
    clean = reconstruct(position, full, prices, 0, 3 * M)
    damaged = reconstruct(position, full[-1:], prices, 0, 3 * M)
    assert clean.rows[-1].pnl == -50
    assert damaged.rows[-1].pnl == 0
    assert clean.sizes[-1] == damaged.sizes[-1]
    # Equality of endpoint sizes is not a certificate of complete realized PnL.


def test_window_clips_fills_before_deduplication_and_reconstruction():
    position = Position(1, 100, 90)
    retained = Fill("new", 2 * M, 1, 100, 0)
    prefix = Fill("old", 0, 99, 999, -999)
    grid = {M: 100, 2 * M: 90}
    assert reconstruct(position, [prefix, retained], grid, M, 2 * M) == reconstruct(
        position, [retained], grid, M, 2 * M)


@pytest.mark.parametrize("inverse", [False, True])
def test_signed_short_and_contract_multiplier(inverse):
    p = Position(-2, 100, 125, 10, inverse, "short")
    expected = dec("-.04") if inverse else dec(-500)
    assert pnl(p, -2, 100, 125) == expected
    assert pnl(p, 2, 100, 125) == -expected


def test_inverse_add_uses_harmonic_basis():
    position = Position(2, dec(400) / 3, 200, inverse=True)
    tape = [Fill("first", M, 1, 100, 0), Fill("second", 2 * M, 1, 200, 0)]
    history = reconstruct(position, tape, {M: 100, 2 * M: 200, 3 * M: 200}, 0, 3 * M)
    assert history.bases[1] == dec(400) / 3
    assert float(history.rows[-1].upnl) == pytest.approx(.005)


def test_flat_short_history_keeps_side_independent_of_fill_input_order():
    p = Position(0, 0, 110, pside="short")
    tape = [Fill("open", M, -2, 100, 0), Fill("close", 2 * M, 2, 110, None)]
    for order in permutations(tape):
        history = reconstruct(p, list(order), {M: 100, 2 * M: 110}, 0, 2 * M)
        assert history.sizes == (-2, 0)
        assert history.rows[-1].pnl == -20


@pytest.mark.parametrize("p", [Position(1, 0, 100), Position(1, 100, "NaN"),
                                 Position(1, 100, 100, 0), Position(-1, 100, 100)])
def test_invalid_current_inputs_remain_errors(p):
    with pytest.raises(ValueError):
        reconstruct(p, [], {}, 0, M)


def test_close_only_ignores_recovered_wick():
    prices = minute_prices([Candle(0, 1, 100, 150, 20, 100)], 0, M)
    assert set(prices.values()) == {100}


@pytest.mark.parametrize("field,value", [("open", "NaN"), ("low", None), ("high", -1)])
def test_real_minute_valid_close_survives_unusable_wick_fields(field, value):
    damaged = replace(Candle(M, 1, 100, 110, 70, 80), **{field: value})
    prices = minute_prices([Candle(0, 1, 100, 100, 100, 100), damaged], 0, 2 * M)
    assert prices[2 * M] == 80
    history = reconstruct(Position(1, 100, 80), [Fill("open", 0, 1, 100, 0)], prices, 0, 3 * M)
    assert history.rows[-2].upnl == -20


@pytest.mark.parametrize("intervention,expected", [("panic", "panic"), ("normal", "normal")])
@pytest.mark.parametrize("restart", ["always", "never"])
def test_proven_flat_and_current_exposure_needs_no_reopening_fill(intervention, expected, restart):
    assert permit(30, LifecycleEvidence(10, 20), intervention=intervention,
                  restart=restart, exposed=True) == expected


@pytest.mark.parametrize("minutes", [5, 15, 60])
@pytest.mark.parametrize("close,first_extreme,second_extreme", [(120, 80, 140), (90, 140, 80)])
def test_coarse_zigzag_extrema_and_endpoint(minutes, close, first_extreme, second_extreme):
    prices = minute_prices([Candle(0, minutes, 100, 140, 80, close)], 0, minutes * M)
    assert prices[M] == 100
    assert prices[(minutes // 3 + 1) * M] == first_extreme
    assert prices[(2 * minutes // 3 + 1) * M] == second_extreme
    assert prices[minutes * M] == close


def test_real_minute_overrides_coarse_and_source_order_does_not_matter():
    candles = [Candle(0, 5, 100, 140, 80, 120), Candle(M, 1, 95, 100, 90, 95)]
    expected = minute_prices(candles, 0, 5 * M)
    assert expected[2 * M] == 95
    assert minute_prices(candles[::-1], 0, 5 * M) == expected


def test_leading_backfill_and_all_internal_tail_gaps_forward_fill():
    prices = minute_prices([Candle(2 * M, 1, 80, 80, 80, 80),
                            Candle(5 * M, 1, 100, 100, 100, 100)], 0, 8 * M)
    assert [prices[i * M] for i in range(9)] == [80] * 6 + [100] * 3


def test_source_must_be_available_at_evaluation_and_inside_window():
    coarse = Candle(0, 15, 100, 150, 50, 100)
    assert minute_prices([coarse], 0, 14 * M) == {}
    assert minute_prices([coarse], M, 15 * M) == {}
    assert minute_prices([replace(coarse, available_at=16 * M)], 0, 15 * M) == {}
    assert minute_prices([coarse], 0, 15 * M)


def test_empty_candles_take_explicit_minimal_branch_with_current_mark():
    p = Position(1, 100, 80)
    assert minute_prices([], 0, M) == {}
    result = minimal_signal(pnl(p, p.size, p.basis, p.mark), 100, 1000, ".1")
    assert result.panic[-1]


def test_live_endpoint_replaces_candle_close_with_current_mark():
    history = reconstruct(Position(1, 100, 80), [Fill("open", 0, 1, 100, 0)],
                          {0: 100, M: 100}, 0, M)
    assert history.rows[-1].upnl == -20


def permit(now, evidence=LifecycleEvidence(10, 20), *, window=100, cooldown=30,
           restart="always", intervention="panic", exposed=False, red=False):
    return permission(now, window, cooldown, restart, intervention, evidence,
                      exposed=exposed, red_now=red)


def test_cooldown_deadline_and_inclusive_window_expiry():
    assert permit(49) == "halted"
    assert permit(50) == "normal"
    assert permit(110, restart="never") == "halted"
    assert permit(111, restart="never") == "normal"
    assert permit(111, cooldown=1000) == "halted"  # flatten anchor remains in scope
    assert permit(120, cooldown=1000) == "halted"
    assert permit(121, cooldown=1000) == "normal"


@pytest.mark.parametrize("restart,expected", [("always", "normal"), ("never", "halted")])
def test_zero_cooldown_only_controls_waiting(restart, expected):
    assert permit(20, restart=restart, cooldown=0) == expected


@pytest.mark.parametrize("restart", ["always", "never"])
def test_two_intervention_choices_override_scope_halt_but_not_fresh_red(restart):
    evidence = LifecycleEvidence(10, 20)
    assert permit(30, evidence, restart=restart, exposed=True) == "panic"
    assert permit(30, evidence, restart=restart, intervention="normal", exposed=True) == "normal"
    assert permit(30, evidence, restart=restart, intervention="normal", exposed=True, red=True) == "panic"
    assert permit(30, evidence, restart=restart, intervention="normal", exposed=False) == "halted"


def test_partial_close_is_not_intervention_or_new_cooldown_anchor():
    evidence = LifecycleEvidence(10, None)
    for now in (15, 30, 100):
        assert permit(now, evidence, intervention="normal", exposed=True) == "panic"
    assert permit(111, evidence, exposed=True) == "normal"
    assert permit(111, evidence, exposed=True, red=True) == "panic"


def test_unobservable_red_is_void_after_recovery_even_without_restart():
    evidence = LifecycleEvidence()
    assert permit(10, evidence, exposed=True, red=True) == "panic"
    assert permit(11, evidence, exposed=True, red=False) == "normal"
    # There is no previous-decision argument or file that can change this result.
    assert permit(11, replace(evidence), exposed=True, red=False) == "normal"


def test_repanic_timer_starts_at_new_actual_flat_not_poll_time():
    new_flat = LifecycleEvidence(25, 35)
    assert permit(64, new_flat) == "halted"
    assert permit(65, new_flat) == "normal"
    assert permit(65, new_flat) == "normal"


@pytest.mark.parametrize("removed", ["threshold", "manual", "tp_only", "graceful_stop"])
def test_removed_policies_are_not_silently_translated(removed):
    with pytest.raises(ValueError):
        permit(30, restart=removed if removed == "threshold" else "always",
               intervention="panic" if removed == "threshold" else removed)
