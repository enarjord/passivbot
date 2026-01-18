"""Sync NSGA-II/III samplers for DEAP-style generation-based optimization.

These samplers compute generation from trial number, matching the synchronous
optimization model where trials are dispatched in batches of population_size.
"""
from __future__ import annotations

from optuna.samplers import NSGAIISampler, NSGAIIISampler
from optuna.trial import FrozenTrial, TrialState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optuna.study import Study


class SyncGASamplerMixin:
    """Mixin for DEAP-style sync optimization with batch dispatch.

    Generation is computed from trial number: gen = trial.number // population_size.
    This works correctly with parallel workers since trial numbers are globally unique.
    """

    def get_trial_generation(self, study: Study, trial: FrozenTrial) -> int:
        """Compute generation from trial number."""
        gen_key = self._get_generation_key()

        # Already assigned?
        if gen := trial.system_attrs.get(gen_key):
            return gen

        # Compute from trial number (assumes batch dispatch of population_size)
        generation = trial.number // self._population_size
        study._storage.set_trial_system_attr(trial._trial_id, gen_key, generation)
        return generation

    def get_population(self, study: Study, generation: int) -> list[FrozenTrial]:
        """Return complete trials in the given generation."""
        gen_key = self._get_generation_key()
        trials = study._get_trials(
            deepcopy=False, states=[TrialState.COMPLETE], use_cache=True
        )
        return [t for t in trials if t.system_attrs.get(gen_key) == generation]


class SyncNSGAIISampler(SyncGASamplerMixin, NSGAIISampler):
    """NSGA-II for synchronous, DEAP-style optimization."""
    pass


class SyncNSGAIIISampler(SyncGASamplerMixin, NSGAIIISampler):
    """NSGA-III for synchronous, DEAP-style optimization."""
    pass
