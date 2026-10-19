import json

import pytest

from live.hsl_continuity import FactualReplays


class Native:
    def __init__(self):
        self.inputs = []
        self.cleared = 0

    def evaluate_grids(self, metadata, grids):
        self.inputs.append((metadata, grids))
        value = json.loads(metadata)
        snapshot = value["snapshot"]
        if not value.get("inactive", False) and not hasattr(self, "origin"):
            self.origin, self.observed = snapshot["start"], snapshot["now"]
        initialized = hasattr(self, "origin")
        return json.dumps({"continuity": {
            "initialized": initialized,
            "bootstrap_origin_ms": self.origin if initialized else None,
            "bootstrap_observed_ms": self.observed if initialized else None,
            "retained_from_ms": self.origin if initialized else None,
            "boundary_at_ms": None, "retention_horizon_ms": 60_000,
            "reconstruction": "bounded_bootstrap" if initialized else "not_initialized",
        }, "unavailable_reason": value.get("unavailable")})

    def clear_derived_cache(self):
        self.cleared += 1

    def cache_info(self):
        return {"calls": len(self.inputs)}


def metadata(*, start, now, mark):
    return json.dumps({"snapshot": {"start": start, "now": now, "mark": mark}})


def test_scope_owns_one_immutable_factual_anchor_across_sliding_io_windows():
    owner = FactualReplays(native_factory=Native)
    first = metadata(start=0, now=100, mark=90)
    grid = object()
    replay = owner.scope(("coin", "A", "long"), first, (grid,))
    later = metadata(start=100, now=200, mark=100)
    assert owner.scope(("coin", "A", "long"), later, ()) is replay
    assert replay.anchor.origin_ms == 0
    assert replay._pending == (first, (grid,))
    assert replay.anchor.origin_ms == 0
    assert replay.anchor.observed_ms == 100
    assert json.loads(replay.evaluate_grids(later, ()))["continuity"]["initialized"]
    assert replay.cache_info() == {"calls": 2}
    assert replay._native.inputs[0] == (first, (grid,))


def test_disposable_fold_loss_keeps_facts_and_other_scopes_independent():
    owner = FactualReplays(native_factory=Native)
    first = metadata(start=10, now=20, mark=90)
    left = owner.scope(("coin", "A", "long"), first, ())
    right = owner.scope(("coin", "A", "short"), first, ())
    assert left is not right
    owner.clear_derived_caches()
    assert left._native.cleared == right._native.cleared == 1
    assert left._pending == right._pending == (first, ())


class FailingNative(Native):
    def __init__(self):
        super().__init__()
        self.fail_on_call = 1

    def evaluate_grids(self, metadata, grids):
        output = super().evaluate_grids(metadata, grids)
        if len(self.inputs) == self.fail_on_call:
            raise ValueError("invalid factual capture")
        return output


@pytest.mark.parametrize("initial_query_is_anchor", [True, False])
def test_failed_initial_ingest_retries_the_original_anchor_before_newer_facts(
    initial_query_is_anchor,
):
    first = metadata(start=0, now=100, mark=90)
    later = metadata(start=100, now=200, mark=100)
    grid = object()
    replay = FactualReplays(native_factory=FailingNative).scope(
        ("coin", "A", "long"), first, (grid,)
    )
    with pytest.raises(ValueError, match="invalid factual capture"):
        replay.evaluate_grids(
            first if initial_query_is_anchor else later,
            (grid,) if initial_query_is_anchor else (),
        )
    assert not replay._initialized
    assert json.loads(replay.evaluate_grids(later, ()))["continuity"]["initialized"]
    assert replay._native.inputs == [
        (first, (grid,)),
        (first, (grid,)),
        (later, ()),
    ]
    assert replay.anchor.origin_ms == 0
    assert replay._pending is None


def test_failed_later_capture_keeps_the_successfully_ingested_anchor():
    first = metadata(start=0, now=100, mark=90)
    later = metadata(start=100, now=200, mark=100)
    replay = FactualReplays(native_factory=FailingNative).scope(
        ("coin", "A", "long"), first, ()
    )
    replay._native.fail_on_call = 2
    with pytest.raises(ValueError, match="invalid factual capture"):
        replay.evaluate_grids(later, ())
    assert replay._initialized
    replay.clear_derived_cache()
    assert json.loads(replay.evaluate_grids(later, ()))["continuity"]["initialized"]
    assert replay._native.inputs == [(first, ()), (later, ()), (later, ())]
    assert replay.anchor.origin_ms == 0


def test_first_anchor_evaluation_is_consumed_once():
    first = metadata(start=0, now=100, mark=90)
    grid = object()
    replay = FactualReplays(native_factory=Native).scope(
        ("coin", "A", "long"), first, (grid,)
    )
    assert json.loads(replay.evaluate_grids(first, (grid,)))["continuity"]["initialized"]
    assert replay._native.inputs == [(first, (grid,))]
    assert replay._pending is None


def test_successful_inactive_without_native_ledger_does_not_pin_later_initializer():
    value = json.loads(metadata(start=0, now=100, mark=90))
    value["inactive"] = True
    first = json.dumps(value)
    replay = FactualReplays(native_factory=Native).scope("A", first, ())
    assert not json.loads(replay.evaluate_grids(first, ()))["continuity"]["initialized"]
    assert not replay._initialized
    later = metadata(start=200, now=300, mark=80)
    replay.evaluate_grids(later, ())
    assert replay.anchor.origin_ms == 200
    assert replay.anchor.observed_ms == 300
    assert replay._native.inputs == [(first, ()), (later, ())]


def test_successful_unavailable_with_native_ledger_keeps_original_initializer():
    value = json.loads(metadata(start=0, now=100, mark=90))
    value["unavailable"] = "causal_position_mark_unavailable"
    first = json.dumps(value)
    replay = FactualReplays(native_factory=Native).scope("A", first, ())
    output = json.loads(replay.evaluate_grids(first, ()))
    assert output["unavailable_reason"]
    assert replay._initialized
    assert replay._pending is None
    replay.evaluate_grids(metadata(start=100, now=200, mark=80), ())
    assert replay.anchor.origin_ms == 0


def test_first_ready_ingest_failure_after_inactive_retries_ready_initializer():
    value = json.loads(metadata(start=0, now=100, mark=90))
    value["inactive"] = True
    first = json.dumps(value)
    replay = FactualReplays(native_factory=FailingNative).scope("A", first, ())
    replay._native.fail_on_call = 2
    ready = metadata(start=200, now=300, mark=80)
    with pytest.raises(ValueError, match="invalid factual capture"):
        replay.evaluate_grids(ready, ())
    assert replay._pending == (ready, ())
    later = metadata(start=300, now=400, mark=100)
    replay.evaluate_grids(later, ())
    assert replay._native.inputs == [(first, ()), (ready, ()), (ready, ()), (later, ())]
    assert replay.anchor.origin_ms == 200


def test_malformed_native_initialization_status_is_fatal_and_keeps_pending_facts():
    class Malformed(Native):
        def evaluate_grids(self, metadata, grids):
            return '{"continuity":{"initialized":1}}'

    first = metadata(start=0, now=100, mark=90)
    replay = FactualReplays(native_factory=Malformed).scope("A", first, ())
    with pytest.raises(ValueError, match="initialization status"):
        replay.evaluate_grids(first, ())
    assert replay._pending == (first, ())
    assert not replay._initialized


def test_rotating_dormant_owners_and_budget_maps_level_off_without_tombstones():
    from live.hsl_budget import BudgetObservations

    owner = FactualReplays(native_factory=Native)
    budgets = BudgetObservations()
    counts = []
    for minute in range(1, 80):
        now = minute * 60_000
        first = metadata(start=max(0, now - 3 * 60_000), now=now, mark=90)
        owner.scope(str(minute), first, ())
        budgets.capture(str(minute), start=max(0, now - 3 * 60_000), now=now,
                        balance=1000, slots=1, scale=False, exposure=None)
        budgets.retain_scopes(owner.prune(now))
        counts.append(owner.cache_info()["retained_scopes"])
        assert set(budgets._scopes) == set(owner._scopes)
        assert set(budgets._policies) == set(owner._scopes)
        assert set(budgets._captured) == set(owner._scopes)
    assert max(counts) == 4
    assert counts[-30:] == [4] * 30


def test_full_cooldown_protects_dormant_owner_and_partial_absence_is_not_flatness():
    owner = FactualReplays(native_factory=Native)
    value = json.loads(metadata(start=0, now=100, mark=90))
    value["cooldown_ms"] = 300_000
    first = json.dumps(value)
    replay = owner.scope("A", first, ())
    replay.evaluate_grids(first, ())
    owner.clear_derived_caches()
    assert "A" in owner.prune(300_100)
    assert replay.last_seen_ms == 100
    assert replay._native.inputs == [(first, ())]
    assert "A" not in owner.prune(300_101)
    replacement = owner.scope("A", metadata(start=300_000, now=400_000, mark=90), ())
    assert replacement is not replay
