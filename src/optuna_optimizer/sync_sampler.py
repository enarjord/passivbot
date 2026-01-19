"""Sync NSGA-II/III samplers for DEAP-style generation-based optimization.

These samplers compute generation from trial number, matching the synchronous
optimization model where trials are dispatched in batches of population_size.

Uses a sliding window of 2 generations - O(population_size) instead of O(n_trials):
- Current generation (being evaluated)
- Previous generation (parent population for selection)

Storage still keeps all trials for persistence and final Pareto extraction.

Key optimization: Overrides infer_relative_search_space to cache the search space
after first trial, avoiding O(n) IntersectionSearchSpace.calculate() on every ask().
"""
from __future__ import annotations

from optuna.distributions import BaseDistribution
from optuna.samplers import NSGAIISampler, NSGAIIISampler
from optuna.trial import FrozenTrial, TrialState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optuna.study import Study


class SyncGASamplerMixin:
    """Mixin for DEAP-style sync optimization with batch dispatch.

    Generation is computed from trial number: gen = trial.number // population_size.

    Uses sliding window of 2 generations for O(population_size) memory:
    - _current_gen: Current generation number
    - _generations: {gen: [trials]} for current and previous gen only
    - _trial_by_id: {trial_id: trial} for current and previous gen only

    Caches search space after first trial to avoid O(n) iteration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sliding window: only keep 2 generations
        self._generations: dict[int, list[FrozenTrial]] = {}
        self._trial_by_id: dict[int, FrozenTrial] = {}
        self._current_gen: int = -1
        # Cached search space to avoid O(n) IntersectionSearchSpace.calculate()
        self._cached_search_space: dict[str, BaseDistribution] | None = None

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        """Return cached search space after first computation.

        The base NSGA sampler calls IntersectionSearchSpace.calculate() which
        iterates ALL trials - O(n). Since our search space is fixed (defined
        by bounds upfront), we cache it after the first trial.
        """
        if self._cached_search_space is not None:
            return self._cached_search_space

        # First call: compute via parent, then cache
        search_space = super().infer_relative_search_space(study, trial)
        self._cached_search_space = search_space
        return search_space

    def _prune_old_generations(self, current_gen: int) -> None:
        """Remove generations older than current_gen - 1."""
        if current_gen <= self._current_gen:
            return  # No change

        self._current_gen = current_gen
        min_gen_to_keep = max(0, current_gen - 1)

        # Find generations to remove
        gens_to_remove = [g for g in self._generations if g < min_gen_to_keep]

        for gen in gens_to_remove:
            # Remove trials from ID index
            for trial in self._generations[gen]:
                self._trial_by_id.pop(trial._trial_id, None)
            # Remove generation
            del self._generations[gen]

    def get_trial_generation(self, study: Study, trial: FrozenTrial) -> int:
        """Compute generation from trial number. O(1)."""
        gen_key = self._get_generation_key()

        # Already assigned? (use 'is not None' to handle generation 0 correctly)
        gen = trial.system_attrs.get(gen_key)
        if gen is not None:
            return gen

        # Compute from trial number (assumes batch dispatch of population_size)
        generation = trial.number // self._population_size
        study._storage.set_trial_system_attr(trial._trial_id, gen_key, generation)
        return generation

    def _index_trial(self, trial: FrozenTrial, generation: int) -> None:
        """Add a trial to the sliding window indices."""
        # Prune if we're moving to a new generation
        self._prune_old_generations(generation)

        # Add to indices
        if generation not in self._generations:
            self._generations[generation] = []

        # Avoid duplicates
        if trial._trial_id not in self._trial_by_id:
            self._generations[generation].append(trial)
            self._trial_by_id[trial._trial_id] = trial

    def get_population(self, study: Study, generation: int) -> list[FrozenTrial]:
        """Return complete trials in the given generation. O(population_size)."""
        # Check sliding window first
        if generation in self._generations:
            return [t for t in self._generations[generation]
                    if t.state == TrialState.COMPLETE]

        # Not in window - need to fetch from storage (rare, only on resume)
        gen_key = self._get_generation_key()
        trials = study._get_trials(
            deepcopy=False, states=[TrialState.COMPLETE], use_cache=True
        )

        # Find trials for this generation and index them
        result = []
        for t in trials:
            t_gen = t.system_attrs.get(gen_key)
            if t_gen is None:
                t_gen = t.number // self._population_size
            if t_gen == generation:
                self._index_trial(t, t_gen)
                result.append(t)

        return result

    def get_parent_population(self, study: Study, generation: int) -> list[FrozenTrial]:
        """Get parent population. O(population_size) via sliding window."""
        if generation == 0:
            return []

        # Check cache for parent population IDs
        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
        cache_key = self._get_parent_cache_key_prefix() + str(generation)
        cached_ids = study_system_attrs.get(cache_key, None)

        if cached_ids is not None:
            # Use our trial_by_id index
            return [self._trial_by_id[tid] for tid in cached_ids
                    if tid in self._trial_by_id]

        # Not cached - compute and cache
        parent_population = self.select_parent(study, generation)
        study._storage.set_study_system_attr(
            study._study_id,
            cache_key,
            [t._trial_id for t in parent_population],
        )
        return parent_population

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: list[float] | None,
    ) -> None:
        """Called after each trial completes. Index in sliding window."""
        super().after_trial(study, trial, state, values)

        if state == TrialState.COMPLETE:
            generation = trial.number // self._population_size
            self._index_trial(trial, generation)


class SyncNSGAIISampler(SyncGASamplerMixin, NSGAIISampler):
    """NSGA-II for synchronous, DEAP-style optimization."""
    pass


class SyncNSGAIIISampler(SyncGASamplerMixin, NSGAIIISampler):
    """NSGA-III for synchronous, DEAP-style optimization."""
    pass
