"""Derive lifecycle boundaries and exercise changing immutable snapshots offline."""

from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from hsl_reference import Fill, LifecycleEvidence, MINUTE as M, Observation, Position, dec, permission, signal
from hsl_reference_replay import (
    Settings, capture, capture_pair, estimate_pair, evaluate_bounded, position_anchor, scope_boundaries,
)


def pair(symbol="A", size=0, basis=0, mark=80, fills=(), prices=None,
         now=4 * M, position_at=None, mark_at=None, pside="long"):
    return capture_pair(symbol, Position(size, basis, mark, pside=pside),
                        now if position_at is None else position_at,
                        now if mark_at is None else mark_at, fills,
                        {} if prices is None else prices,
                        fills_started_at=now, fills_at=now, prices_at=now,
                        fills_after_position=True)


def after_position(p):
    """Fixture explicitly observes this position before starting its tail fetch."""
    return replace(p, fills_position_anchor=position_anchor(p))


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
    assert boundary.consumed == ((("A", "long"), closed_tape(fee=-1).fills),)


def test_flatten_signal_drives_real_cooldown_anchor_before_reopening():
    p = closed_tape(fee=-1)
    p = replace(p, position=Position(1, 80, 80),
                fills=(*p.fills, Fill("reopen", 3 * M + 1, 1, 80, 0, -1)))
    p = after_position(p)
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
        assert "estimated_flat" in trace.reasons
        assert not trace.boundaries[0].lifecycle_eligible


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
    fresh = after_position(replace(a, position_at=4 * M, fills_at=4 * M))
    assert len(scope_boundaries(frame(fresh, b), "unified").boundaries) == 1


def test_close_after_position_anchor_is_disclosed_until_positions_catch_up():
    p = pair(size=1, basis=100, position_at=2 * M,
             fills=[Fill("open", M, 1, 100, 0), Fill("close", 3 * M, -1, 80, -20)])
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries
    assert "post_position_fill" in trace.reasons
    current = replace(p, position_at=4 * M, fills_at=4 * M,
                      position=replace(p.position, size=dec(0), basis=dec(0)))
    current = after_position(current)
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
        updated = replace(original, pairs=(after_position(replace(p, position=replace(p.position, size=dec(2)))),))
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
    assert len(result.value.history.rows) == 5  # not silently replaced with a singleton
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
    ahead = replace(original, pairs=(replace(p, fills=(*p.fills, add), fills_at=4 * M),))
    old = compute(ahead)
    assert "post_position_fill" in old.reasons
    assert "snapshot_skew" in old.reasons
    assert old.history.sizes[-1] == 1 and old.history.rows[-1].pnl == 0
    fresh_pair = replace(ahead.pairs[0], position_at=4 * M,
                         position=replace(p.position, size=dec(2), basis=dec(90)))
    refreshed = replace(ahead, pairs=(after_position(fresh_pair),))
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


@pytest.mark.parametrize("field,value", [("price", 101), ("fee", -1), ("revision", 1)])
def test_consumed_boundary_records_corrections_even_with_same_flat_and_realized_pnl(field, value):
    original = closed_tape()
    corrected = replace(original.fills[0], **{field: value})
    updated = replace(original, fills=(corrected, *original.fills[1:]))
    before = scope_boundaries(frame(original), "unified")
    after = scope_boundaries(frame(updated), "unified")
    assert before.boundaries[0].timestamp == after.boundaries[0].timestamp
    assert before != after


@pytest.mark.parametrize("source", range(6))
def test_regressed_revision_cannot_be_revalidated_by_repeating_stale_snapshot(source):
    original = open_frame(revisions=(5,) * 6)
    versions = list(original.revisions)
    versions[source] = 4
    stale = replace(original, revisions=tuple(versions))
    observe, _ = observe_sequence(original, stale, stale)
    result = evaluate_bounded(observe, compute)
    assert not result.revalidated
    assert result.reasons == {"revision_regression"}
    assert result.value.signal.panic[-1]  # diagnostic does not veto risk


@pytest.mark.parametrize("source", ["fills", "prices", "config"])
def test_capture_time_change_invalidates_same_revision_result(source):
    original = open_frame()
    if source == "config":
        original = replace(original, config_at=3 * M)
        updated = replace(original, config_at=4 * M)
    else:
        timing = {source + "_at": 3 * M}
        if source == "fills":
            timing["fills_started_at"] = 3 * M
        original = replace(original, pairs=(replace(original.pairs[0], **timing),))
        updated = replace(original, pairs=(replace(original.pairs[0], **{k: 4 * M for k in timing}),))
    observe, _ = observe_sequence(original, updated, updated)
    result = evaluate_bounded(observe, compute)
    assert result.revalidated and result.evaluations == 2
    assert result.snapshot == updated


def test_older_fill_capture_cannot_certify_flat_against_newer_position():
    p = replace(closed_tape(), fills_started_at=2 * M, fills_at=2 * M)
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries and "fills_before_position" in trace.reasons
    # A missing earlier execution need not have a post-position event timestamp.
    original = open_frame()
    stale = replace(original, pairs=(replace(original.pairs[0], fills_started_at=2 * M, fills_at=2 * M),))
    observe, _ = observe_sequence(stale, stale)
    result = evaluate_bounded(observe, compute)
    assert not result.revalidated and result.reasons == {"fills_before_position"}
    assert result.value.signal.panic[-1]


def test_sparse_grid_uses_ffill_bfill_instead_of_shortening_ema_time():
    original = open_frame(settings=Settings(3, ".05"))
    sparse = replace(original, pairs=(replace(original.pairs[0], prices=((2 * M, dec(80)),)),))
    dense = replace(sparse, pairs=(replace(sparse.pairs[0], prices=tuple((t, dec(80)) for t in range(0, 5 * M, M))),))
    a, b = compute(sparse), compute(dense)
    assert a.history == b.history and a.signal == b.signal
    assert "filled_price_grid" in a.reasons
    assert "filled_price_grid" not in b.reasons
    assert len(a.history.rows) == 5


@pytest.mark.parametrize("timing,reason", [
    ({}, "fill_capture_unknown"),
    ({"fills_at": 4 * M}, "fill_capture_unknown"),
    ({"fills_started_at": 3 * M, "fills_at": 4 * M}, "fills_before_position"),
    ({"fills_started_at": 4 * M, "fills_at": 4 * M}, "fills_before_position"),
])
def test_fill_fetch_must_be_known_to_start_after_position_observation(timing, reason):
    original = closed_tape()
    p = capture_pair("A", original.position, 4 * M, 4 * M, original.fills, {}, **timing)
    trace = scope_boundaries(frame(p), "unified")
    assert not trace.boundaries and reason in trace.reasons
    fresh = after_position(replace(p, fills_started_at=4 * M, fills_at=4 * M))
    assert scope_boundaries(frame(fresh), "unified").boundaries[0].lifecycle_eligible


@pytest.mark.parametrize("other_side", ["long", "short"])
def test_degraded_other_pair_does_not_poison_coin_snapshot_quality(other_side):
    original = open_frame()
    bad = replace(pair("B", pside=other_side), fills_started_at=3 * M, prices_at=3 * M)
    snapshot = replace(original, pairs=(*original.pairs, bad))
    assert compute(snapshot).reasons == compute(original).reasons
    observe, _ = observe_sequence(snapshot, snapshot)
    coin = evaluate_bounded(observe, compute, scope_keys={("A", "long")})
    assert coin.revalidated and not coin.reasons
    observe, _ = observe_sequence(snapshot, snapshot)
    unified = evaluate_bounded(observe, compute)
    assert not unified.revalidated
    assert unified.reasons == {"fills_before_position", "prices_before_mark"}
    long_keys = {p.key for p in snapshot.pairs if p.position.pside == "long"}
    observe, _ = observe_sequence(snapshot, snapshot)
    long = evaluate_bounded(observe, compute, scope_keys=long_keys)
    assert long.revalidated == (other_side == "short")


def test_conflicting_post_position_variants_cannot_certify_an_older_flat():
    initial = closed_tape(close_at=2 * M)
    # Use a single close to avoid an unrelated same-time ordering ambiguity.
    tape = [initial.fills[0], Fill("close", 2 * M, -2, 80, -40)]
    late = Fill("late", 3 * M, 1, 90, 0, revision=1)
    p = pair(fills=(*tape, late, replace(late, delta=2)), position_at=2 * M)
    trace = scope_boundaries(frame(p), "unified")
    assert "post_position_fill" in trace.reasons
    assert trace.boundaries and not any(b.lifecycle_eligible for b in trace.boundaries)
    corrected = replace(p, position_at=4 * M, position=Position(1, 90, 80),
                        fills=(*p.fills, replace(late, revision=2)))
    corrected = after_position(corrected)
    recovered = scope_boundaries(frame(corrected), "unified")
    assert [b.timestamp for b in recovered.boundaries if b.lifecycle_eligible] == [2 * M]


def test_unrelated_pair_churn_cannot_invalidate_stable_coin_projection():
    a = open_frame()
    b = pair("B", size=1, basis=100)
    initial = replace(a, pairs=(*a.pairs, b))
    changed_b = replace(b, position=replace(b.position, mark=dec(70)), revisions=(0, 1, 0, 0))
    # The aggregate mark token changes too; A consumes only its own mark token.
    changed = replace(initial, pairs=(*a.pairs, changed_b), revisions=(0, 0, 1, 0, 0, 0))
    observe, calls = observe_sequence(initial, changed)
    result = evaluate_bounded(observe, compute, scope_keys={("A", "long")})
    assert result.revalidated and result.evaluations == 1 and len(calls) == 2
    assert result.snapshot == a and result.value == compute(a)


@pytest.mark.parametrize("source", range(4))
def test_scoped_pair_revision_updates_and_regressions_are_not_ignored(source):
    a = open_frame()
    revisions = [0] * 4
    revisions[source] = 1
    changed = replace(a, pairs=(after_position(replace(a.pairs[0], revisions=tuple(revisions))),))
    observe, _ = observe_sequence(a, changed, changed)
    result = evaluate_bounded(observe, compute, scope_keys={("A", "long")})
    assert result.revalidated and result.evaluations == 2
    observe, _ = observe_sequence(changed, a, a)
    result = evaluate_bounded(observe, compute, scope_keys={("A", "long")})
    assert not result.revalidated and "revision_regression" in result.reasons


@pytest.mark.parametrize("global_source", ["balance", "config"])
def test_coin_projection_retains_shared_global_dependencies(global_source):
    a = open_frame()
    changed = replace(a, **({"balance": dec(1000)} if global_source == "balance"
                           else {"settings": Settings(1000, ".9")}))
    observe, _ = observe_sequence(a, changed, changed)
    result = evaluate_bounded(observe, compute, scope_keys={("A", "long")})
    assert result.revalidated and result.evaluations == 2
    assert result.value != compute(a)


@pytest.mark.parametrize("conflicting", [False, True])
def test_known_post_position_fill_stays_unvalidated_until_position_refresh(conflicting):
    a = open_frame(position_at=2 * M)
    p = a.pairs[0]
    add = Fill("later-add", 3 * M, 1, 80, 0, revision=1)
    fills = (*p.fills, add, replace(add, delta=2)) if conflicting else (*p.fills, add)
    ahead = replace(a, pairs=(replace(p, fills=fills),))
    observe, _ = observe_sequence(ahead, ahead)
    result = evaluate_bounded(observe, compute, scope_keys={p.key})
    assert not result.revalidated and "post_position_fill" in result.reasons
    assert result.value.signal.panic[-1]
    fresh = replace(ahead, pairs=(replace(p, position_at=4 * M,
                    position=Position(2, 90, 80),
                    fills=(*fills, replace(add, revision=2))),))
    fresh = replace(fresh, pairs=(after_position(fresh.pairs[0]),))
    observe, _ = observe_sequence(fresh, fresh)
    recovered = evaluate_bounded(observe, compute, scope_keys={p.key})
    assert recovered.revalidated and "post_position_fill" not in recovered.reasons


def test_same_timestamp_tail_execution_is_uncertain_until_later_position_observation():
    a = open_frame(position_at=2 * M)
    p = a.pairs[0]
    close = Fill("tied-close", 2 * M, -1, 80, -20)
    tied = replace(a, pairs=(replace(p, fills=(*p.fills, close)),))
    observe, _ = observe_sequence(tied, tied)
    result = evaluate_bounded(observe, compute, scope_keys={p.key})
    assert not result.revalidated and "position_fill_timestamp_tie" in result.reasons
    assert result.value.signal.panic[-1]
    # A later observed zero position resolves the ordering without a local latch.
    fresh = replace(tied, pairs=(replace(tied.pairs[0], position_at=4 * M,
                                        position=Position(0, 0, 80)),))
    fresh = replace(fresh, pairs=(after_position(fresh.pairs[0]),))
    observe, _ = observe_sequence(fresh, fresh)
    assert evaluate_bounded(observe, compute, scope_keys={p.key}).revalidated
    assert scope_boundaries(fresh, "unified").boundaries[0].lifecycle_eligible


def test_missing_requested_pair_is_not_a_silent_flat_scope_member():
    a = open_frame()
    b = pair("B", size=1, basis=100)
    full = replace(a, pairs=(*a.pairs, b))
    keys = {("A", "long"), ("B", "long")}
    observe, _ = observe_sequence(full, a, a)
    with pytest.raises(ValueError, match="absent is not flat"):
        evaluate_bounded(observe, lambda s: sum(abs(p.position.size) for p in s.pairs), scope_keys=keys)
    observed_flat = replace(full, pairs=(*a.pairs, after_position(replace(b, position=Position(0, 0, 80)))))
    observe, _ = observe_sequence(observed_flat, observed_flat)
    result = evaluate_bounded(observe, lambda s: sum(abs(p.position.size) for p in s.pairs), scope_keys=keys)
    assert result.revalidated and result.value == 1


def test_boundary_selector_requires_explicit_current_coin_observation():
    with pytest.raises(ValueError, match="missing current coin position"):
        scope_boundaries(frame(pair("A")), "coin", symbol="B", pside="long")
    explicit = scope_boundaries(frame(pair("B")), "coin", symbol="B", pside="long")
    assert not explicit.boundaries and not explicit.reasons


def test_equal_clock_causal_fill_proof_is_bound_to_the_observed_position():
    original = open_frame()
    p = original.pairs[0]
    assert p.fills_started_at == p.position_at
    observe, _ = observe_sequence(original, original)
    assert evaluate_bounded(observe, compute).revalidated
    changed = replace(original, pairs=(replace(p, position=replace(p.position, size=dec(2))),))
    observe, _ = observe_sequence(changed, changed)
    assert not evaluate_bounded(observe, compute).revalidated


@pytest.mark.parametrize("source", ["balance_at", "config_at", "position_at", "mark_at",
                                  "fills_started_at", "fills_at", "prices_at"])
def test_source_capture_regression_cannot_validate_repeated_older_observation(source):
    a = open_frame()
    p = replace(a.pairs[0], position_at=3 * M, mark_at=3 * M,
                fills_started_at=3 * M + 1000)
    a = replace(a, pairs=(p,))
    if source in ("balance_at", "config_at"):
        stale = replace(a, **{source: getattr(a, source) - 1})
    else:
        stale = replace(a, pairs=(replace(p, **{source: getattr(p, source) - 1}),))
    observe, _ = observe_sequence(a, stale, stale)
    result = evaluate_bounded(observe, compute, scope_keys={p.key})
    assert not result.revalidated and "source_capture_regression" in result.reasons
    assert result.value.signal.panic[-1]
