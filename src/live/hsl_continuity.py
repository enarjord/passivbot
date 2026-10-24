"""Bounded native HSL owners and transactional factual initialization.

Native checkpoints retain compact forward accounting/lifecycle and a bounded fact
suffix. Losing an owner declares a new approximate bootstrap; eviction is never a
flatness observation. Python releases initializer grids after native ingestion.
"""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class BootstrapAnchor:
    origin_ms: int
    observed_ms: int


def continuity_status(output, *, now):
    """Validate the required native support envelope before trusting its state."""
    status = json.loads(output)["continuity"]
    if not isinstance(status, dict) or type(status["initialized"]) is not bool:
        raise ValueError("invalid HSL continuity initialization status")
    if type(status["retention_horizon_ms"]) is not int or status["retention_horizon_ms"] < 60_000:
        raise ValueError("invalid HSL continuity retention horizon")
    if status["reconstruction"] not in {
        "not_initialized", "bounded_bootstrap", "policy_changed_beyond_retained_support",
        "request_before_retained_support", "position_correction_before_retained_support",
        "budget_correction_before_retained_support",
        "unavailable_prefix_reconstruction",
    }:
        raise ValueError("invalid HSL continuity reconstruction status")
    keys = ("bootstrap_origin_ms", "bootstrap_observed_ms", "retained_from_ms")
    if status["initialized"]:
        if any(type(status[key]) is not int or not 0 <= status[key] <= now for key in keys):
            raise ValueError("invalid HSL continuity source clocks")
        origin, observed, retained = (status[key] for key in keys)
        boundary = status["boundary_at_ms"]
        if not origin <= observed <= now or retained < origin or (
            boundary is None and retained != origin
        ) or (boundary is not None and (
            type(boundary) is not int or not observed <= boundary < now or retained != boundary + 1
        )):
            raise ValueError("incoherent HSL continuity boundary")
    elif any(status[key] is not None for key in (*keys, "boundary_at_ms")):
        raise ValueError("uninitialized HSL continuity has retained history")
    return status


class ScopeReplay:
    def __init__(self, metadata, grids, *, native_factory=None):
        if native_factory is None:
            import passivbot_rust as pbr

            native_factory = pbr.HslV9Replay
        snapshot = json.loads(metadata)["snapshot"]
        self.anchor = BootstrapAnchor(snapshot["start"], snapshot["now"])
        self._pending = (metadata, tuple(grids))
        self._native = native_factory()
        self._initialized = False
        self.last_seen_ms = snapshot["now"]
        self.horizon_ms = max(snapshot["now"] - snapshot["start"],
                              json.loads(metadata).get("cooldown_ms", 0), 60_000)

    def _evaluate(self, metadata, grids):
        output = self._native.evaluate_grids(metadata, grids)
        status = continuity_status(output, now=json.loads(metadata)["snapshot"]["now"])
        self._initialized = status["initialized"]
        if self._initialized:
            anchor = BootstrapAnchor(status["bootstrap_origin_ms"], status["bootstrap_observed_ms"])
            if anchor != self.anchor:
                self.anchor = anchor
        return output

    def evaluate_grids(self, metadata, grids):
        current = (metadata, tuple(grids))
        if not self._initialized:
            if self._pending is None:
                self._pending = current
            initializer = self._pending
            # Exceptions retain the exact pending initializer for a retry. A
            # successful unavailable result may own a ledger; inactive/no-ledger
            # results must let a later ready capture become the initializer.
            output = self._evaluate(*initializer)
            self._pending = None
            if current == initializer:
                return output
        if not self._initialized:
            self._pending = current
        output = self._evaluate(*current)
        self._pending = None
        return output

    def clear_derived_cache(self):
        """Rebuild from the retained native boundary and factual suffix."""
        self._native.clear_derived_cache()

    def cache_info(self):
        return self._native.cache_info()


class FactualReplays:
    def __init__(self, *, native_factory=None):
        self._scopes = {}
        self._native_factory = native_factory

    def prune(self, now):
        """Discard dormant owners without synthesizing position/lifecycle facts."""
        self._scopes = {key: replay for key, replay in self._scopes.items()
                        if now - replay.last_seen_ms <= replay.horizon_ms}
        return frozenset(self._scopes)

    def scope(self, key, metadata, grids):
        value = json.loads(metadata)
        snapshot = value["snapshot"]
        self.prune(snapshot["now"])
        if key not in self._scopes:
            self._scopes[key] = ScopeReplay(metadata, grids, native_factory=self._native_factory)
        replay = self._scopes[key]
        replay.last_seen_ms = snapshot["now"]
        replay.horizon_ms = max(snapshot["now"] - snapshot["start"],
                                value.get("cooldown_ms", 0), 60_000)
        return replay

    def cache_info(self):
        return {"retained_scopes": len(self._scopes),
                "pending_initializers": sum(replay._pending is not None for replay in self._scopes.values())}

    def clear_derived_caches(self):
        for replay in self._scopes.values():
            replay.clear_derived_cache()
