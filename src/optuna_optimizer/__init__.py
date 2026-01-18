"""Optuna-based optimizer for Passivbot."""
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
    GPSamplerConfig,
    NSGAIISamplerConfig,
    NSGAIIISamplerConfig,
    Objective,
    OptunaConfig,
    RandomSamplerConfig,
    SamplerConfig,
    TPESamplerConfig,
)
from .pareto import extract_pareto
from .storage import SharedMemoryJournalBackend, dump_to_sqlite, load_from_sqlite
from .samplers import create_sampler, get_sampler_config_by_name, make_constraints_func
from .shared_arrays import SharedArrayAttachment, SharedArrayManager, SharedArraySpec, attach_shared_array
from .trial import check_constraints, compute_penalty, compute_scores, resolve_metric, sample_params
from .worker import WorkerContext, WorkerInitData, get_context, init_worker
from .sync_sampler import SyncNSGAIISampler, SyncNSGAIIISampler

__all__ = [
    # Models
    "Bound",
    "Constraint",
    "Objective",
    "OptunaConfig",
    "SamplerConfig",
    "TPESamplerConfig",
    "NSGAIISamplerConfig",
    "NSGAIIISamplerConfig",
    "GPSamplerConfig",
    "RandomSamplerConfig",
    # Config
    "extract_bounds",
    "extract_constraints",
    "extract_objectives",
    "extract_optuna_config",
    "extract_params_from_config",
    "apply_params_to_config",
    "load_seed_configs",
    # Storage
    "SharedMemoryJournalBackend",
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
    "sample_params",
    "check_constraints",
    "compute_penalty",
    "compute_scores",
    "resolve_metric",
    # Shared arrays
    "SharedArrayManager",
    "SharedArraySpec",
    "SharedArrayAttachment",
    "attach_shared_array",
    # Worker
    "WorkerInitData",
    "WorkerContext",
    "init_worker",
    "get_context",
]
