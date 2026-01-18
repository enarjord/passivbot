"""Sampler factory for Optuna optimizer."""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from .models import (
    GPSamplerConfig, NSGAIISamplerConfig, NSGAIIISamplerConfig,
    RandomSamplerConfig, SamplerConfig, TPESamplerConfig,
)
from .sync_sampler import SyncNSGAIISampler, SyncNSGAIIISampler

if TYPE_CHECKING:
    import optuna


def make_constraints_func() -> Callable[["optuna.trial.FrozenTrial"], list[float]]:
    """Create Optuna-compatible constraints_func for hard constraint mode.

    Returns:
        Function that extracts constraint violations from trial user attrs.
    """
    def constraints_func(trial: "optuna.trial.FrozenTrial") -> list[float]:
        return trial.user_attrs.get("constraint_violations", [])
    return constraints_func


def create_sampler(
    config: SamplerConfig,
    constraints_func: Callable[["optuna.trial.FrozenTrial"], list[float]] | None = None,
) -> "optuna.samplers.BaseSampler":
    """Create an Optuna sampler from config.

    Args:
        config: Sampler configuration
        constraints_func: Optional constraint function for hard constraint mode

    Returns:
        Configured Optuna sampler
    """
    import optuna  # Lazy import to avoid requiring optuna at module load
    if isinstance(config, TPESamplerConfig):
        return optuna.samplers.TPESampler(
            n_startup_trials=config.n_startup_trials,
            multivariate=config.multivariate,
            constant_liar=config.constant_liar,
            seed=config.seed,
            constraints_func=constraints_func,
        )
    if isinstance(config, NSGAIISamplerConfig):
        return SyncNSGAIISampler(
            population_size=config.population_size,
            mutation_prob=config.mutation_prob,
            crossover_prob=config.crossover_prob,
            seed=config.seed,
            constraints_func=constraints_func,
        )
    if isinstance(config, NSGAIIISamplerConfig):
        return SyncNSGAIIISampler(
            population_size=config.population_size,
            mutation_prob=config.mutation_prob,
            crossover_prob=config.crossover_prob,
            seed=config.seed,
            constraints_func=constraints_func,
        )
    if isinstance(config, GPSamplerConfig):
        return optuna.samplers.GPSampler(
            n_startup_trials=config.n_startup_trials,
            deterministic_objective=config.deterministic_objective,
            seed=config.seed,
            constraints_func=constraints_func,
        )
    if isinstance(config, RandomSamplerConfig):
        # RandomSampler doesn't support constraints_func
        return optuna.samplers.RandomSampler(seed=config.seed)
    raise ValueError(f"Unknown sampler: {type(config)}")


def get_sampler_config_by_name(name: str) -> SamplerConfig:
    """Get a sampler config by name.

    Args:
        name: One of 'tpe', 'nsgaii', 'nsgaiii', 'gp', 'random'

    Returns:
        Default config for the named sampler.

    Raises:
        KeyError: If name is not recognized.
    """
    configs = {
        "tpe": TPESamplerConfig,
        "nsgaii": NSGAIISamplerConfig,
        "nsgaiii": NSGAIIISamplerConfig,
        "gp": GPSamplerConfig,
        "random": RandomSamplerConfig,
    }
    return configs[name]()
