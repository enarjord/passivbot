"""Optuna-based optimizer for Passivbot."""
from .bot_params import BotParamsTemplate
from .config import (
    apply_params_to_config,
    extract_bounds,
    extract_constraints,
    extract_objectives,
    extract_optuna_config,
    extract_params_from_config,
    load_seed_configs,
)
from .models import (
    Bound,
    Constraint,
    NSGAIISamplerConfig,
    NSGAIIISamplerConfig,
    Objective,
    OptunaConfig,
    SamplerConfig,
)
from .pareto import extract_pareto
from .storage import InMemoryJournalBackend, dump_to_sqlite, load_from_sqlite
from .samplers import create_sampler, get_sampler_config_by_name, make_constraints_func
from .trial import build_distributions, check_constraints, compute_scores, resolve_metric, sample_params
from .sync_sampler import SyncNSGAIISampler, SyncNSGAIIISampler

__all__ = [
    # Bot params
    "BotParamsTemplate",
    # Models
    "Bound",
    "Constraint",
    "Objective",
    "OptunaConfig",
    "SamplerConfig",
    "NSGAIISamplerConfig",
    "NSGAIIISamplerConfig",
    # Config
    "extract_bounds",
    "extract_constraints",
    "extract_objectives",
    "extract_optuna_config",
    "extract_params_from_config",
    "apply_params_to_config",
    "load_seed_configs",
    # Storage
    "InMemoryJournalBackend",
    "dump_to_sqlite",
    "load_from_sqlite",
    # Samplers
    "create_sampler",
    "get_sampler_config_by_name",
    "make_constraints_func",
    "SyncNSGAIISampler",
    "SyncNSGAIIISampler",
    # Pareto
    "extract_pareto",
    # Trial
    "build_distributions",
    "sample_params",
    "check_constraints",
    "compute_scores",
    "resolve_metric",
]
