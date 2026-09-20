"""Derive lifecycle boundaries and exercise changing immutable snapshots offline."""

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from hsl_reference import Fill, LifecycleEvidence, MINUTE as M, Observation, Position, dec, permission, signal
from hsl_reference_replay import (
    Settings, capture, capture_pair, estimate_pair, evaluate_bounded, scope_boundaries,
)


def pair(symbol="A", size=0, basis=0, mark=80, fills=(), prices=None,
         now=4 * M, position_at=None, mark_at=None, pside="long"):
    return capture_pair(symbol, Position(size, basis, mark, pside=pside),
                        now if position_at is None else position_at,
                        now if mark_at is None else mark_at, fills,
                        {} if prices is None else prices)


def frame(*pairs, now=4 * M, start=0, balance=100, balance_at=None, **kwargs):
    return capture(now, start, balance, now if balance_at is None else balance_at, pairs, **kwargs)


def closed_tape(symbol="A", close_at=3 * M, fee=0):
    return pair(symbol, fills=[Fill("entry", M, 2, 100, 0, fee),
                               Fill("partial", 2 * M, -1, 90, -10, fee),
                               Fill("final", close_at, -1, 80, -20, fee)])


@pytest.mark.parametrize("mode,kwargs", [("coin", {"pside": "long", "symbol": "A"}),
                                        ("pside", {"pside": "long"}), ("unified", {})])
def test_partial_close_is_not_flat_and_flat_cashflow_includes_fees(mode, kwargs):
    trace = scope_boundaries(frame(closed_tape(fee=-1)), mode, **kwargs)
    assert len(trace.boundaries) == 1
    boundary = trace.boundaries[0]
    assert boundary.timestamp == 3 * M
    assert boundary.observation.pnl == -33
    assert boundary.observation.upnl == 0
    assert boundary.consumed == ((("A", "long"), ("entry", "partial", "final")),)


def test_flatten_signal_drives_real_cooldown_anchor_before_reopening():
    p = closed_tape(fee=-1)
    p = replace(p, position=Position(1, 80, 80),
                fills=(*p.fills, Fill("reopen", 3 * M + 1, 1, 80, 0, -1)))
    boundary, = scope_boundaries(frame(p), "coin", pside="long", symbol="A").boundaries
    # This historical final-risk row comes from the actual fill prefix, not an
    # injected flat timestamp or the later same-minute reopened position.
    risk_rows = (Observation(0, dec(0), dec(0)), boundary.observation)
    risk = signal(risk_rows, 100, 1, ".2")
    assert risk.panic[-1]
    evidence = LifecycleEvidence(risk_rows[-1].timestamp, boundary.timestamp)
    assert permission(4 * M, 10 * M, 2 * M, "always", "panic", evidence,
                      exposed=True, red_now=False) == "panic"
    assert permission(4 * M, 10 * M, 2 * M, "always", "normal", evidence,
                      exposed=True, red_now=False) == "normal"
    assert permission(5 * M, 10 * M, 2 * M, "always", "panic", evidence,
                      exposed=False, red_now=False) == "normal"


def test_sequenced_same_millisecond_flats_keep_distinct_cashflow_prefixes():
    tape = [Fill("entry", M, 1, 100, 0, -1),
            Fill("close1", 2 * M, -1, 80, -20, -1, sequence=10),
            Fill("entry2", 2 * M, 1, 80, 0, -1, sequence=11),
            Fill("close2", 2 * M, -1, 70, -10, -1, sequence=12)]
    for ordering in permutations(tape):
        trace = scope_boundaries(frame(pair(fills=ordering)), "unified")
        assert [b.timestamp for b in trace.boundaries] == [2 * M, 2 * M]
        assert [b.observation.pnl for b in trace.boundaries] == [-22, -34]
        assert trace.boundaries[0].consumed != trace.boundaries[1].consumed


@pytest.mark.parametrize("other_side", ["long", "short"])
def test_cross_pair_close_open_cohort_does_not_invent_portfolio_flat(other_side):
    a = pair("A", fills=[Fill("open-a", M, 1, 100, 0), Fill("close-a", 2 * M, -1, 80, -20)])
    sign = 1 if other_side == "long" else -1
    b = pair("B", pside=other_side, fills=[Fill("open-b", 2 * M, sign, 80, 0),
                                          Fill("close-b", 3 * M, -sign, 70, -10)])
    for pairs in permutations([a, b]):
        snapshot = frame(*pairs)
        assert [x.timestamp for x in scope_boundaries(snapshot, "unified").boundaries] == [3 * M]
        expected = 3 * M if other_side == "long" else 2 * M
        assert [x.timestamp for x in scope_boundaries(snapshot, "pside", pside="long").boundaries] == [expected]
        assert [x.timestamp for x in scope_boundaries(snapshot, "coin", pside="long", symbol="A").boundaries] == [2 * M]


def test_opposite_signed_exposures_are_not_a_flat_portfolio():
    snapshot = frame(pair("A", 1, 100), pair("A", -1, 100, pside="short"))
    assert not scope_boundaries(snapshot, "unified").boundaries


def test_roundtrip_wholly_within_unsequenced_cohort_has_final_flat_only():
    tape = [Fill("open", M, 1, 100, 0), Fill("close", M, -1, 80, -20)]
    for ordering in permutations(tape):
        trace = scope_boundaries(frame(pair(fills=ordering)), "unified")
        assert [b.observation.pnl for b in trace.boundaries] == [-20]
        assert "estimated_fill_order" in trace.reasons


def test_unsequenced_cohort_cannot_prove_an_internal_flat_then_reopen():
    tape = [Fill("initial", 0, 1, 100, 0), Fill("reduce", M, -1, 80, -20),
            Fill("new", M, 1, 80, 0)]
    p = pair(size=1, basis=80, mark=40, fills=tape, prices={0: 100, M: 80})
    snapshot = frame(p)
    trace = scope_boundaries(snapshot, "unified")
    assert not trace.boundaries
    assert "estimated_fill_order" in trace.reasons
    assert estimate_pair(snapshot, p.key).signal.panic[-1]  # ambiguity is not a risk veto


def test_old_missing_opening_does_not_poison_later_flat():
    tape = [Fill("old-add", M, 5, 50, 0), Fill("old-close", 2 * M, -7, 40, -70),
            Fill("new-open", 3 * M, 2, 100, 0), Fill("new-close", 4 * M, -2, 80, -40)]
    trace = scope_boundaries(frame(pair(fills=tape)), "unified")
    assert [b.timestamp for b in trace.boundaries] == [2 * M, 4 * M]
    assert [b.observation.pnl for b in trace.boundaries] == [-70, -110]
    assert "estimated_opening_basis" in trace.reasons


def test_clamp_cannot_create_boundary_but_does_not_poison_clean_suffix():
    # The old add cannot coexist with the following opening from zero; backward
    # reconstruction clamps it. The newer complete episode remains independent.
    tape = [Fill("bad-add", M, 9, 100, 0), Fill("new-open", 2 * M, 1, 100, 0),
            Fill("new-close", 3 * M, -1, 70, -30)]
    trace = scope_boundaries(frame(pair(fills=tape)), "unified")
    assert [b.timestamp for b in trace.boundaries] == [3 * M]
    assert "clamped_quantity" in trace.reasons


def test_unknown_quantity_after_candidate_does_not_certify_flat():
    tape = [Fill("open", M, 1, 100, 0), Fill("close", 2 * M, -1, 80, -20),
            Fill("unknown", 3 * M, "NaN", 70, -5)]
    trace = scope_boundaries(frame(pair(fills=tape)), "unified")
    assert not trace.boundaries
    assert "uncertain_flat" in trace.reasons
    p = pair(size=1, basis=100, mark=50, fills=tape, prices={0: 100, M: 100, 2 * M: 80})
    assert estimate_pair(frame(p), p.key).signal.panic[-1]


def test_conflicting_prefix_is_local_and_repair_rebuilds_boundaries():
    conflict = Fill("old", M, 1, 100, 0, revision=1)
    tape = [conflict, replace(conflict, delta=2), Fill("open", 2 * M, 1, 100, 0),
            Fill("close", 3 * M, -1, 80, -20)]
    trace = scope_boundaries(frame(pair(fills=tape)), "unified")
    assert [b.timestamp for b in trace.boundaries] == [3 * M]
    assert "conflicting_identity" in trace.reasons
    # A correction of the close to an opening invalidates the old boundary.
    corrected = [*tape, replace(tape[-1], delta=1, revision=1)]
    assert not scope_boundaries(frame(pair(size=2, basis=90, fills=corrected)), "unified").boundaries


def test_missing_flat_fill_does_not_invent_timestamp_from_current_flat():
    p = pair(fills=[Fill("entry", M, 1, 100, 0)])
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries
    assert "clamped_quantity" in trace.reasons
    delivered = replace(p, fills=(*p.fills, Fill("close", 2 * M, -1, 80, -20)))
    assert [b.timestamp for b in scope_boundaries(frame(delivered), "unified").boundaries] == [2 * M]


def test_missing_final_reduction_cannot_promote_an_earlier_partial_close_to_flat():
    p = pair(fills=[Fill("open", M, 2, 100, 0), Fill("partial", 2 * M, -1, 90, -10)])
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries
    assert "uncertain_episode_flat" in trace.reasons
    delivered = replace(p, fills=(*p.fills, Fill("final", 3 * M, -1, 80, -20)))
    assert [b.timestamp for b in scope_boundaries(frame(delivered), "unified").boundaries] == [3 * M]


def test_snapshot_experiment_does_not_silently_claim_a_complete_minimal_history_model():
    # The dedicated minimal-history oracle is covered separately. This experiment
    # must not emit a healthy zero-DD signal just because its grid is absent.
    with pytest.raises(ValueError, match="minimal-history oracle"):
        estimate_pair(frame(pair(size=1, basis=100, mark=10)), ("A", "long"))


def test_exact_window_edge_and_empty_restart_reproduce_same_boundary():
    p = closed_tape()
    snapshot = frame(p, start=3 * M)
    expected = scope_boundaries(snapshot, "unified")
    assert [b.timestamp for b in expected.boundaries] == [3 * M]
    assert expected == scope_boundaries(frame(replace(p, fills=tuple(p.fills)), start=3 * M), "unified")
    assert not scope_boundaries(frame(p, start=3 * M + 1), "unified").boundaries


def test_cross_pair_boundary_cannot_postdate_an_older_position_anchor():
    a = pair("A", position_at=2 * M)
    b = closed_tape("B")
    trace = scope_boundaries(frame(a, b), "unified")
    assert not trace.boundaries
    assert "boundary_after_position_anchor" in trace.reasons
    fresh = replace(a, position_at=4 * M)
    assert len(scope_boundaries(frame(fresh, b), "unified").boundaries) == 1


def test_close_after_position_anchor_is_disclosed_until_positions_catch_up():
    p = pair(size=1, basis=100, position_at=2 * M,
             fills=[Fill("open", M, 1, 100, 0), Fill("close", 3 * M, -1, 80, -20)])
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries
    assert "post_position_fill" in trace.reasons
    current = replace(p, position_at=4 * M, position=replace(p.position, size=dec(0), basis=dec(0)))
    assert [b.timestamp for b in scope_boundaries(frame(current), "unified").boundaries] == [3 * M]


def open_frame(mark=80, *, now=4 * M, position_at=None, balance=100, settings=Settings(), revisions=(0,) * 6):
    p = pair(size=1, basis=100, mark=mark, fills=[Fill("open", M, 1, 100, 0)],
             prices={0: 100, M: 100, 2 * M: 100}, now=now, position_at=position_at)
    return frame(p, now=now, balance=balance, settings=settings, revisions=revisions)


def compute(snapshot):
    return estimate_pair(snapshot, ("A", "long"))


def observe_sequence(*snapshots):
    values = iter(snapshots)
    calls = []

    def observe():
        value = next(values)
        calls.append(value)
        return value

    return observe, calls


def test_capture_is_immutable_and_detached_from_source_containers():
    fills = [Fill("entry", M, 1, 100, 0)]
    prices = {0: 100, M: 100}
    pairs = [pair(size=1, basis=100, fills=fills, prices=prices)]
    snapshot = frame(*pairs)
    expected = compute(snapshot)
    fills.append(Fill("late", 2 * M, -1, 80, -20))
    prices[M] = 1
    pairs.clear()
    assert compute(snapshot) == expected
    with pytest.raises(FrozenInstanceError):
        snapshot.balance = 0
    with pytest.raises(FrozenInstanceError):
        snapshot.pairs[0].position.size = 0


def test_stable_snapshot_is_validated_once_and_reproducible_after_restart():
    snapshot = open_frame()
    observe, calls = observe_sequence(snapshot, snapshot)
    result = evaluate_bounded(observe, compute)
    assert result.revalidated and result.evaluations == 1 and len(calls) == 2
    fresh_observe, _ = observe_sequence(open_frame(), open_frame())
    assert evaluate_bounded(fresh_observe, compute) == result


@pytest.mark.parametrize("changed", ["balance", "position", "mark", "fill", "price", "config", "time"])
def test_changed_surface_cannot_install_the_old_result(changed):
    original = open_frame()
    p = original.pairs[0]
    if changed == "balance":
        updated = replace(original, balance=dec(200))
    elif changed == "position":
        updated = replace(original, pairs=(replace(p, position=replace(p.position, size=dec(2))),))
    elif changed == "mark":
        updated = replace(original, pairs=(replace(p, position=replace(p.position, mark=dec(100))),))
    elif changed == "fill":
        corrected = replace(p.fills[0], fee=dec(-10), revision=1)
        updated = replace(original, pairs=(replace(p, fills=(*p.fills, corrected)),))
    elif changed == "price":
        updated = replace(original, pairs=(replace(p, prices=(*p.prices, (3 * M, dec(200)))),))
    elif changed == "config":
        updated = replace(original, settings=Settings(1000, ".9", 1))
    else:
        updated = replace(original, now=original.now + 1, start=M + 1)
    # Structural comparison also catches a producer that accidentally reuses a
    # revision number for changed content. The old value is not installed.
    observe, _ = observe_sequence(original, updated, updated)
    result = evaluate_bounded(observe, compute)
    assert result.revalidated and result.evaluations == 2
    assert result.snapshot == updated
    assert result.value == compute(updated)
    assert result.value != compute(original)


@pytest.mark.parametrize("surface", range(6))
def test_revision_change_alone_invalidates_an_old_capture(surface):
    original = open_frame()
    revisions = list(original.revisions)
    revisions[surface] += 1
    updated = replace(original, revisions=tuple(revisions))
    observe, _ = observe_sequence(original, updated, updated)
    result = evaluate_bounded(observe, compute)
    assert result.snapshot == updated and result.evaluations == 2


def test_continuous_churn_is_bounded_and_preserves_latest_history():
    a, b, c = open_frame(80), open_frame(70), open_frame(60)
    observe, calls = observe_sequence(a, b, c)
    result = evaluate_bounded(observe, compute, max_attempts=2)
    assert len(calls) == 3 and result.evaluations == 3
    assert not result.revalidated and result.reasons == {"revision_churn"}
    assert result.snapshot == c and result.value == compute(c)
    assert len(result.value.history.rows) == 4  # not silently replaced with a singleton
    assert result.value.signal.panic[-1]


def test_a_history_refresh_timeout_need_not_erase_retained_evidence():
    prior = open_frame(80, settings=Settings(1000, ".1"))
    # The I/O owner retains previously captured exchange rows on a failed refresh.
    # No new revision/content means these observations remain usable by the oracle.
    retained = replace(prior, pairs=tuple(prior.pairs))
    observe, _ = observe_sequence(retained, retained)
    result = evaluate_bounded(observe, compute)
    assert not result.value.signal.panic[-1]  # retained EMA smoothing matters
    assert len(result.value.history.rows) > 1


def test_required_current_refresh_failure_does_not_return_the_previous_value():
    calls = []

    def observe():
        calls.append(1)
        if len(calls) == 1:
            return open_frame()
        raise ValueError("current account state unavailable")

    with pytest.raises(ValueError, match="account state unavailable"):
        evaluate_bounded(observe, compute)
    assert len(calls) == 2


def test_post_position_fill_is_isolated_until_position_refresh_then_counted_once():
    original = open_frame(60, position_at=2 * M)
    p = original.pairs[0]
    add = Fill("add", 3 * M, 1, 80, 0, -1)
    ahead = replace(original, pairs=(replace(p, fills=(*p.fills, add)),))
    old = compute(ahead)
    assert "post_position_fill" in old.reasons
    assert "snapshot_skew" in old.reasons
    assert old.history.sizes[-1] == 1 and old.history.rows[-1].pnl == 0
    fresh_pair = replace(ahead.pairs[0], position_at=4 * M,
                         position=replace(p.position, size=dec(2), basis=dec(90)))
    refreshed = replace(ahead, pairs=(fresh_pair,))
    new = compute(refreshed)
    assert "post_position_fill" not in new.reasons
    assert new.history.sizes[-1] == 2 and new.history.rows[-1].pnl == -1
    observe, _ = observe_sequence(ahead, refreshed, refreshed)
    assert evaluate_bounded(observe, compute).value == new


@pytest.mark.parametrize("problem", ["stale_position", "future_mark", "bad_balance", "bad_current_basis"])
def test_invalid_minimum_inputs_are_not_replaced_by_a_previous_healthy_decision(problem):
    with pytest.raises(ValueError):
        if problem == "stale_position":
            frame(pair(position_at=0))
        elif problem == "future_mark":
            frame(pair(mark_at=5 * M))
        elif problem == "bad_balance":
            frame(pair(), balance="NaN")
        else:
            frame(pair(size=1, basis=0))
