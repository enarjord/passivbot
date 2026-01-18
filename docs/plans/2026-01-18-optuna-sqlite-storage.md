# Optuna SQLite Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace dual journal_mode with memory-only mode that dumps to SQLite at the end.

**Architecture:** SharedMemoryJournalBackend handles all trial writes during optimization (no disk I/O). At completion, `optuna.copy_study()` copies the in-memory study to a file-based SQLite database. Resume reverses this: copy SQLite to memory, continue, overwrite SQLite.

**Tech Stack:** Optuna's `copy_study()`, SQLite storage URL (`sqlite:///path/to/study.db`)

---

### Task 1: Add SQLite Storage Functions

**Files:**
- Modify: `src/optuna_optimizer/storage.py`

**Step 1: Add dump_to_sqlite function**

Add after the `SharedMemoryJournalBackend` class (around line 140):

```python
def dump_to_sqlite(
    backend: SharedMemoryJournalBackend,
    study_name: str,
    sqlite_path: Path,
) -> None:
    """Copy study from in-memory backend to SQLite file.

    Uses optuna.copy_study() to handle all trial data, user attrs, etc.
    Overwrites existing file if present.

    Args:
        backend: In-memory backend containing the study data
        study_name: Name of the study to copy
        sqlite_path: Path to write the SQLite database
    """
    import optuna

    # Delete existing file to ensure clean overwrite
    if sqlite_path.exists():
        sqlite_path.unlink()

    from_storage = JournalStorage(backend)
    to_storage = f"sqlite:///{sqlite_path}"

    optuna.copy_study(
        from_study_name=study_name,
        from_storage=from_storage,
        to_storage=to_storage,
    )
```

**Step 2: Add load_from_sqlite function**

Add after `dump_to_sqlite`:

```python
def load_from_sqlite(
    sqlite_path: Path,
    study_name: str,
    sampler: "optuna.samplers.BaseSampler | None" = None,
) -> tuple[SharedMemoryJournalBackend, "optuna.Study"]:
    """Load existing SQLite study into memory for resume.

    Creates fresh SharedMemoryJournalBackend, copies study into it,
    returns backend and loaded study ready for optimization.

    Args:
        sqlite_path: Path to the SQLite database
        study_name: Name of the study to load
        sampler: Optional sampler to use (recreated in caller)

    Returns:
        Tuple of (backend, study) ready for continued optimization
    """
    import optuna

    # Create fresh in-memory backend
    backend = SharedMemoryJournalBackend.create_parent()
    to_storage = JournalStorage(backend)

    from_storage = f"sqlite:///{sqlite_path}"

    optuna.copy_study(
        from_study_name=study_name,
        from_storage=from_storage,
        to_storage=to_storage,
    )

    # Load study from the in-memory backend
    study = optuna.load_study(
        study_name=study_name,
        storage=to_storage,
        sampler=sampler,
    )

    return backend, study
```

**Step 3: Remove create_journal_storage function**

Delete the `create_journal_storage` function (lines 142-158).

**Step 4: Remove dump_to_file method from SharedMemoryJournalBackend**

Delete the `dump_to_file` method (lines 124-135).

**Step 5: Add TYPE_CHECKING import for type hints**

Update imports at top of file:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna
```

**Step 6: Commit**

```bash
git add src/optuna_optimizer/storage.py
git commit -m "feat(optuna): add SQLite dump/load functions, remove journal helpers"
```

---

### Task 2: Update Package Exports

**Files:**
- Modify: `src/optuna_optimizer/__init__.py`

**Step 1: Update storage imports**

Change line 24 from:
```python
from .storage import create_journal_storage, SharedMemoryJournalBackend
```
to:
```python
from .storage import SharedMemoryJournalBackend, dump_to_sqlite, load_from_sqlite
```

**Step 2: Update __all__ list**

In the `__all__` list, replace:
```python
    # Storage
    "create_journal_storage",
    "SharedMemoryJournalBackend",
```
with:
```python
    # Storage
    "SharedMemoryJournalBackend",
    "dump_to_sqlite",
    "load_from_sqlite",
```

**Step 3: Commit**

```bash
git add src/optuna_optimizer/__init__.py
git commit -m "refactor(optuna): update storage exports for SQLite"
```

---

### Task 3: Update Worker Module

**Files:**
- Modify: `src/optuna_optimizer/worker.py`

**Step 1: Update storage import**

Change line 18 from:
```python
from .storage import create_journal_storage, SharedMemoryJournalBackend
```
to:
```python
from .storage import SharedMemoryJournalBackend
```

**Step 2: Simplify _create_worker_storage function**

Replace the entire `_create_worker_storage` function (lines 24-38) with:

```python
def _create_worker_storage(shared_state: tuple):
    """Create JournalStorage for worker processes.

    Args:
        shared_state: Shared memory state from parent process.

    Returns:
        JournalStorage instance backed by shared memory
    """
    from optuna.storages import JournalStorage

    backend = SharedMemoryJournalBackend.from_shared_state(shared_state)
    return JournalStorage(backend)
```

**Step 3: Update WorkerContext.from_init_data**

In the `from_init_data` method (around line 112), change:
```python
        storage = _create_worker_storage(study_dir, data.shared_storage_state)
```
to:
```python
        storage = _create_worker_storage(data.shared_storage_state)
```

**Step 4: Commit**

```bash
git add src/optuna_optimizer/worker.py
git commit -m "refactor(optuna): simplify worker storage to memory-only"
```

---

### Task 4: Remove journal_mode from Models

**Files:**
- Modify: `src/optuna_optimizer/models.py`

**Step 1: Remove journal_mode field**

Delete line 115:
```python
    journal_mode: Literal["file", "memory"] = "file"  # "memory" disables disk I/O
```

**Step 2: Clean up Literal import if unused**

Check if `Literal` is still used. It's used for sampler `name` fields, so keep it.

**Step 3: Commit**

```bash
git add src/optuna_optimizer/models.py
git commit -m "refactor(optuna): remove journal_mode config option"
```

---

### Task 5: Update Orchestrator - run_optimization

**Files:**
- Modify: `src/optuna_optimize.py`

**Step 1: Update imports**

Change the import block (lines 26-46) to remove `create_journal_storage`:

```python
from optuna_optimizer import (
    apply_params_to_config,
    check_constraints,
    compute_scores,
    create_sampler,
    dump_to_sqlite,
    extract_bounds,
    extract_constraints,
    extract_objectives,
    extract_optuna_config,
    extract_params_from_config,
    extract_pareto,
    get_sampler_config_by_name,
    load_from_sqlite,
    load_seed_configs,
    make_constraints_func,
    resolve_metric,
    sample_params,
    SharedArrayManager,
    SharedArraySpec,
    SharedMemoryJournalBackend,
)
```

**Step 2: Add JournalStorage import**

Add after the optuna import (around line 16):
```python
from optuna.storages import JournalStorage
```

**Step 3: Simplify run_optimization storage setup**

In `run_optimization` function, replace lines 388-398:
```python
    journal_path = study_dir / "journal.log"

    # Create storage backend based on journal_mode
    shared_backend = None
    if optuna_cfg.journal_mode == "memory":
        logging.info("Using shared memory journal (no disk I/O during optimization)")
        shared_backend = SharedMemoryJournalBackend.create_parent()
        storage = create_journal_storage(journal_path, shared_backend=shared_backend)
    else:
        storage = create_journal_storage(journal_path)
```

with:
```python
    # Always use shared memory during optimization (no disk I/O)
    shared_backend = SharedMemoryJournalBackend.create_parent()
    storage = JournalStorage(shared_backend)
```

**Step 4: Commit**

```bash
git add src/optuna_optimize.py
git commit -m "refactor(optuna): simplify run_optimization to memory-only storage"
```

---

### Task 6: Update Orchestrator - _run_optimization_core

**Files:**
- Modify: `src/optuna_optimize.py`

**Step 1: Update function signature**

Change the `_run_optimization_core` function signature (lines 215-230) to remove `shared_backend: SharedMemoryJournalBackend | None = None` and make it required:

```python
async def _run_optimization_core(
    config: dict,
    study_dir: Path,
    study: optuna.Study,
    bounds: dict,
    constraints: list,
    objectives: list[Objective],
    sampler_config,
    n_trials: int,
    n_cpus: int,
    fixed_params: dict | None,
    penalty_weight: float,
    max_best_trials: int,
    debug_level: int,
    shared_backend: SharedMemoryJournalBackend,
) -> None:
```

**Step 2: Update docstring**

Update the docstring to remove the conditional wording about shared_backend (around line 248):

```python
        shared_backend: Shared memory backend for worker communication
```

**Step 3: Remove lock file cleanup**

Delete the lock file cleanup block in the finally clause (lines 302-310):
```python
        # Clean up lock file on interruption (avoids 30s grace period wait on resume)
        journal_path = study_dir / "journal.log"
        lock_path = study_dir / "journal.log.lock"
        if lock_path.exists() or lock_path.is_symlink():
            try:
                lock_path.unlink()
                logging.debug("Cleaned up journal lock file")
            except OSError as e:
                logging.debug(f"Could not remove lock file: {e}")
```

**Step 4: Replace Pareto extraction and journal dump with SQLite dump**

Replace the block from line 312 to 337:
```python
        # Extract Pareto and log summary
        try:
            if shared_backend is not None:
                # Extract directly from shared memory (no file round-trip)
                storage = create_journal_storage(journal_path, shared_backend=shared_backend)
            elif journal_path.exists():
                storage = create_journal_storage(journal_path)
            else:
                storage = None
                logging.warning("No journal found, skipping Pareto extraction")

            if storage is not None:
                final_study = optuna.load_study(study_name=study_dir.name, storage=storage)
                if final_study.trials:
                    _log_optimization_summary(final_study, objectives, n_trials, start_time)
                    logging.info("Extracting Pareto front...")
                    extract_pareto(final_study, study_dir, objectives, config, max_best_trials)
                else:
                    logging.warning("No trials completed, skipping Pareto extraction")
        except Exception as e:
            logging.warning(f"Could not extract Pareto front: {e}")

        # Save journal to disk for resume capability (after Pareto extraction)
        if shared_backend is not None:
            logging.info("Saving journal to disk for resume...")
            shared_backend.dump_to_file(journal_path)
```

with:
```python
        # Extract Pareto and log summary (use study object directly from memory)
        try:
            if study.trials:
                _log_optimization_summary(study, objectives, n_trials, start_time)
                logging.info("Extracting Pareto front...")
                extract_pareto(study, study_dir, objectives, config, max_best_trials)
            else:
                logging.warning("No trials completed, skipping Pareto extraction")
        except Exception as e:
            logging.warning(f"Could not extract Pareto front: {e}")

        # Dump to SQLite for resume and optuna-dashboard
        sqlite_path = study_dir / "study.db"
        logging.info(f"Saving study to {sqlite_path}...")
        dump_to_sqlite(shared_backend, study.study_name, sqlite_path)
```

**Step 5: Commit**

```bash
git add src/optuna_optimize.py
git commit -m "refactor(optuna): update core loop to dump SQLite at end"
```

---

### Task 7: Update Orchestrator - resume_optimization

**Files:**
- Modify: `src/optuna_optimize.py`

**Step 1: Update resume_optimization function**

Replace the storage loading section (lines 547-575):
```python
    config_path = study_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in {study_dir}")

    journal_path = study_dir / "journal.log"
    if not journal_path.exists():
        raise ValueError(f"No journal.log found in {study_dir}")
    storage = create_journal_storage(journal_path)

    config = load_config(str(config_path), live_only=False, verbose=False)
    optuna_cfg = extract_optuna_config(config)
    bounds = extract_bounds(config)
    constraints = extract_constraints(config)
    objectives = extract_objectives(config)

    if not objectives:
        raise ValueError("No objectives defined in config['optimize']['objectives']")

    n_trials = n_trials or optuna_cfg.n_trials
    n_cpus = n_cpus or optuna_cfg.n_cpus
    penalty_weight = optuna_cfg.penalty_weight

    # Recreate sampler from config (load_study uses default otherwise)
    sampler_cfg = optuna_cfg.sampler
    constraints_func = make_constraints_func() if penalty_weight == -1 else None
    sampler = create_sampler(sampler_cfg, constraints_func)

    # Load existing study with correct sampler
    study = optuna.load_study(study_name=study_dir.name, storage=storage, sampler=sampler)
```

with:
```python
    config_path = study_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"No config.json found in {study_dir}")

    sqlite_path = study_dir / "study.db"
    if not sqlite_path.exists():
        raise ValueError(f"No study.db found in {study_dir}")

    config = load_config(str(config_path), live_only=False, verbose=False)
    optuna_cfg = extract_optuna_config(config)
    bounds = extract_bounds(config)
    constraints = extract_constraints(config)
    objectives = extract_objectives(config)

    if not objectives:
        raise ValueError("No objectives defined in config['optimize']['objectives']")

    n_trials = n_trials or optuna_cfg.n_trials
    n_cpus = n_cpus or optuna_cfg.n_cpus
    penalty_weight = optuna_cfg.penalty_weight

    # Recreate sampler from config
    sampler_cfg = optuna_cfg.sampler
    constraints_func = make_constraints_func() if penalty_weight == -1 else None
    sampler = create_sampler(sampler_cfg, constraints_func)

    # Load existing study from SQLite into memory
    shared_backend, study = load_from_sqlite(sqlite_path, study_dir.name, sampler)
```

**Step 2: Update the call to _run_optimization_core**

Replace lines 590-595:
```python
    # Resume always uses file mode (shared_backend=None) since we load from existing journal
    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params, penalty_weight,
        optuna_cfg.max_best_trials, debug_level, shared_backend=None
    )
```

with:
```python
    await _run_optimization_core(
        config, study_dir, study, bounds, constraints, objectives,
        sampler_cfg, n_trials, n_cpus, fixed_params, penalty_weight,
        optuna_cfg.max_best_trials, debug_level, shared_backend
    )
```

**Step 3: Commit**

```bash
git add src/optuna_optimize.py
git commit -m "refactor(optuna): update resume to load from SQLite"
```

---

### Task 8: Remove journal_mode from Config Defaults

**Files:**
- Modify: `src/config_utils.py`

**Step 1: Remove journal_mode from defaults**

Find line 2127 and delete:
```python
                "journal_mode": "file",
```

**Step 2: Commit**

```bash
git add src/config_utils.py
git commit -m "refactor(optuna): remove journal_mode from config defaults"
```

---

### Task 9: Remove journal_mode from Template Config

**Files:**
- Modify: `configs/template.json`

**Step 1: Remove journal_mode line**

Find line 389 and delete:
```json
            "journal_mode": "file",
```

Also remove the blank line after it if present.

**Step 2: Commit**

```bash
git add configs/template.json
git commit -m "refactor(optuna): remove journal_mode from template config"
```

---

### Task 10: Manual Verification

**Step 1: Verify imports work**

```bash
cd /home/fredb/passivbot && python3 -c "from optuna_optimizer import dump_to_sqlite, load_from_sqlite, SharedMemoryJournalBackend; print('Imports OK')"
```

Expected: `Imports OK`

**Step 2: Verify CLI help still works**

```bash
cd /home/fredb/passivbot && python3 src/optuna_optimize.py --help
```

Expected: Help output without errors

**Step 3: Commit all changes as final squash (optional)**

If all verification passes, the implementation is complete.
