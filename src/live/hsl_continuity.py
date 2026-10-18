"""Factual initialization owners for Rust's disposable live HSL replay cache.

The anchor holds the immutable first observed account/price facts. It contains no
permission, drawdown, EMA or cooldown. Losing this factual owner means a new,
explicitly approximate bootstrap; clearing its derived Rust fold preserves it.
"""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class BootstrapAnchor:
    origin_ms: int
    observed_ms: int
    metadata: str
    price_grids: tuple


class ScopeReplay:
    def __init__(self, metadata, grids, *, native_factory=None):
        if native_factory is None:
            import passivbot_rust as pbr

            native_factory = pbr.HslV9Replay
        snapshot = json.loads(metadata)["snapshot"]
        self.anchor = BootstrapAnchor(
            snapshot["start"], snapshot["now"], metadata, tuple(grids)
        )
        self._native = native_factory()
        self._initialized = False

    def evaluate_grids(self, metadata, grids):
        if not self._initialized:
            # A capture can be discarded before evaluation. The first observed
            # raw initializer must still own a later request's initialization.
            if metadata != self.anchor.metadata or tuple(grids) != self.anchor.price_grids:
                self._native.evaluate_grids(self.anchor.metadata, self.anchor.price_grids)
            self._initialized = True
        return self._native.evaluate_grids(metadata, grids)

    def clear_derived_cache(self):
        """Discard the fold, retaining the same canonical facts and initializer."""
        self._native.clear_derived_cache()

    def cache_info(self):
        return self._native.cache_info()


class FactualReplays:
    def __init__(self, *, native_factory=None):
        self._scopes = {}
        self._native_factory = native_factory

    def scope(self, key, metadata, grids):
        if key not in self._scopes:
            self._scopes[key] = ScopeReplay(
                metadata, grids, native_factory=self._native_factory
            )
        return self._scopes[key]

    def clear_derived_caches(self):
        for replay in self._scopes.values():
            replay.clear_derived_cache()
