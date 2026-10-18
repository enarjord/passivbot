import json

from live.hsl_continuity import FactualReplays


class Native:
    def __init__(self):
        self.inputs = []
        self.cleared = 0

    def evaluate_grids(self, metadata, grids):
        self.inputs.append((metadata, grids))
        return "output"

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
    assert replay.anchor.metadata == first
    assert replay.anchor.price_grids == (grid,)
    assert replay.anchor.origin_ms == 0
    assert replay.anchor.observed_ms == 100
    assert replay.evaluate_grids(later, ()) == "output"
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
    assert left.anchor.metadata == right.anchor.metadata == first
