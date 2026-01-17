"""Windowed NSGA-II/III samplers that don't slow down at high trial counts.

Standard Optuna NSGA-II fetches ALL trials on every sample. At 300K trials, this
is slow. This module keeps only recent generations in memory.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from optuna.samplers import NSGAIISampler, NSGAIIISampler
from optuna.trial import FrozenTrial, TrialState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optuna.study import Study


class WindowedGASamplerMixin:
    """Mixin that caches recent generations in memory."""

    def _init_cache(self, window_generations: int = 50) -> None:
        self._window = window_generations
        self._trials: dict[int, FrozenTrial] = {}  # trial_id -> trial
        self._generations: dict[int, list[int]] = defaultdict(list)  # gen -> [trial_ids]
        self._max_gen = 0
        self._gen_count = 0  # trials in current generation
        self._ready = False

    def _bootstrap(self, study: "Study") -> None:
        """Load recent generations from DB on first use."""
        if self._ready:
            return

        gen_key = self._get_generation_key()
        all_trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE], use_cache=True)

        # Find current generation state
        for t in all_trials:
            gen = t.system_attrs.get(gen_key, 0)
            if gen > self._max_gen:
                self._max_gen = gen
                self._gen_count = 1
            elif gen == self._max_gen:
                self._gen_count += 1

        # Cache only recent generations
        min_gen = max(0, self._max_gen - self._window)
        for t in all_trials:
            gen = t.system_attrs.get(gen_key, 0)
            if gen >= min_gen:
                self._trials[t._trial_id] = t
                self._generations[gen].append(t._trial_id)

        logging.debug(f"Windowed sampler: cached {len(self._trials)} trials (gen {min_gen}-{self._max_gen})")
        self._ready = True

    def get_trial_generation(self, study: "Study", trial: FrozenTrial) -> int:
        """Return generation from cache, not DB."""
        self._bootstrap(study)

        gen_key = self._get_generation_key()

        # Already assigned?
        if gen := trial.system_attrs.get(gen_key):
            return gen

        # Assign to current or next generation
        if self._gen_count >= self._population_size:
            self._max_gen += 1
            self._gen_count = 0

        gen = self._max_gen
        study._storage.set_trial_system_attr(trial._trial_id, gen_key, gen)
        return gen

    def get_population(self, study: "Study", generation: int) -> list[FrozenTrial]:
        """Return population from cache, not DB."""
        self._bootstrap(study)
        return [self._trials[tid] for tid in self._generations.get(generation, []) if tid in self._trials]

    def get_parent_population(self, study: "Study", generation: int) -> list[FrozenTrial]:
        """Return parent population from cache."""
        self._bootstrap(study)

        if generation == 0:
            return []

        # Check Optuna's parent cache
        cache_key = self._get_parent_cache_key_prefix() + str(generation)
        cached_ids = study._storage.get_study_system_attrs(study._study_id).get(cache_key)

        if cached_ids is not None:
            return [self._trials[tid] for tid in cached_ids if tid in self._trials]

        # Compute and cache
        parents = self.select_parent(study, generation)
        study._storage.set_study_system_attr(study._study_id, cache_key, [t._trial_id for t in parents])
        return parents

    def after_trial(self, study: "Study", trial: FrozenTrial, state: TrialState, values) -> None:
        """Add completed trial to cache, prune old generations."""
        if state == TrialState.COMPLETE:
            self._bootstrap(study)

            gen = self.get_trial_generation(study, trial)
            self._trials[trial._trial_id] = trial

            if trial._trial_id not in self._generations[gen]:
                self._generations[gen].append(trial._trial_id)
                self._gen_count += 1

            # Prune old generations
            min_gen = max(0, self._max_gen - self._window)
            for old_gen in [g for g in self._generations if g < min_gen]:
                for tid in self._generations[old_gen]:
                    self._trials.pop(tid, None)
                del self._generations[old_gen]

        super().after_trial(study, trial, state, values)


class WindowedNSGAIISampler(WindowedGASamplerMixin, NSGAIISampler):
    """NSGA-II that stays fast at high trial counts."""

    def __init__(self, *, window_generations: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._init_cache(window_generations)


class WindowedNSGAIIISampler(WindowedGASamplerMixin, NSGAIIISampler):
    """NSGA-III that stays fast at high trial counts."""

    def __init__(self, *, window_generations: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._init_cache(window_generations)
