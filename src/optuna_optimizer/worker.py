"""Worker process state and initialization.

This module encapsulates all worker-side state and initialization logic,
providing a clean typed interface for the orchestrator.
"""
from __future__ import annotations

import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import Bound, Constraint, Objective, SamplerConfig
from .samplers import create_sampler, make_constraints_func
from .shared_arrays import SharedArraySpec, SharedArrayAttachment, attach_shared_array
from .storage import create_journal_storage

if TYPE_CHECKING:
    import optuna


def _create_worker_storage(study_dir: Path):
    """Create JournalStorage for worker processes.

    Args:
        study_dir: Study directory containing study.log

    Returns:
        JournalStorage instance
    """
    return create_journal_storage(study_dir / "study.log")


@dataclass
class WorkerInitData:
    """Serializable bundle passed from parent to worker.

    All fields must be picklable for multiprocessing. Note that study_dir
    is a str (not Path) for pickle compatibility.
    """

    hlcvs_specs: dict[str, SharedArraySpec]
    btc_specs: dict[str, SharedArraySpec]
    ts_specs: dict[str, SharedArraySpec]
    mss: dict[str, dict]
    config: dict
    study_dir: str
    bounds: dict[str, Bound]
    constraints: list[Constraint]
    objectives: list[Objective]
    sampler_config: SamplerConfig
    fixed_params: dict[str, float] | None
    penalty_weight: float = 1000
    debug_level: int = 1
    logging_module: str | None = None  # e.g. "logging_setup" - if None, uses basic logging


@dataclass
class WorkerContext:
    """Typed worker state with attached shared memory.

    Created once per worker process via from_init_data(). Provides typed
    access to all state needed by the objective function.
    """

    config: dict
    study_dir: Path
    bounds: dict[str, Bound]
    constraints: list[Constraint]
    objectives: list[Objective]
    fixed_params: dict[str, float] | None
    penalty_weight: float
    mss: dict[str, dict]
    exchanges: list[str]
    hlcvs: dict[str, np.ndarray]
    btc: dict[str, np.ndarray]
    timestamps: dict[str, np.ndarray]
    study: "optuna.Study | None" = None
    _attachments: list[SharedArrayAttachment] = field(default_factory=list, repr=False)

    @classmethod
    def from_init_data(cls, data: WorkerInitData) -> WorkerContext:
        """Attach to shared memory and build context."""
        import optuna

        exchanges = list(data.hlcvs_specs.keys())
        hlcvs: dict[str, np.ndarray] = {}
        btc: dict[str, np.ndarray] = {}
        timestamps: dict[str, np.ndarray] = {}
        attachments: list[SharedArrayAttachment] = []

        for exchange in exchanges:
            hlcvs_att = attach_shared_array(data.hlcvs_specs[exchange])
            btc_att = attach_shared_array(data.btc_specs[exchange])
            ts_att = attach_shared_array(data.ts_specs[exchange])

            attachments.extend([hlcvs_att, btc_att, ts_att])
            hlcvs[exchange] = hlcvs_att.array
            btc[exchange] = btc_att.array
            timestamps[exchange] = ts_att.array

        # Load study once per worker for efficient single-trial dispatch
        study_dir = Path(data.study_dir)
        storage = _create_worker_storage(study_dir)

        # Recreate sampler in worker process (samplers aren't picklable)
        constraints_func = make_constraints_func() if data.penalty_weight == -1 else None
        sampler = create_sampler(data.sampler_config, constraints_func)
        study = optuna.load_study(study_name=study_dir.name, storage=storage, sampler=sampler)

        return cls(
            config=data.config,
            study_dir=study_dir,
            bounds=data.bounds,
            constraints=data.constraints,
            objectives=data.objectives,
            fixed_params=data.fixed_params,
            penalty_weight=data.penalty_weight,
            mss=data.mss,
            exchanges=exchanges,
            hlcvs=hlcvs,
            btc=btc,
            timestamps=timestamps,
            study=study,
            _attachments=attachments,
        )

    def close(self) -> None:
        """Close shared memory attachments."""
        for att in self._attachments:
            att.close()


# Worker-local context, initialized once per worker process.
# Each worker process has its own isolated copy due to multiprocessing fork/spawn.
# This is NOT shared state - it's process-private.
_ctx: WorkerContext | None = None


def init_worker(data: WorkerInitData) -> None:
    """Initialize worker process. Called once by Pool.

    Sets up logging, ignores SIGINT (parent handles interrupts),
    and attaches to shared memory.
    """
    global _ctx
    import logging

    # Ignore SIGINT in workers - let parent handle interrupts
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure logging - use injected module or fall back to basic config
    log_level = logging.INFO if data.debug_level >= 1 else logging.WARNING
    if data.logging_module:
        import importlib
        try:
            log_mod = importlib.import_module(data.logging_module)
            log_mod.configure_logging(debug=data.debug_level)
        except (ImportError, AttributeError) as e:
            logging.basicConfig(level=log_level)
            logging.warning(f"Could not load logging module '{data.logging_module}': {e}")
    else:
        logging.basicConfig(level=log_level)

    _ctx = WorkerContext.from_init_data(data)


def get_context() -> WorkerContext:
    """Get the initialized worker context.

    Raises RuntimeError if called before init_worker().
    """
    if _ctx is None:
        raise RuntimeError("Worker not initialized - call init_worker first")
    return _ctx


