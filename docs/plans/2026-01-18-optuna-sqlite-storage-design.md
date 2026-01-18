# Optuna SQLite Storage Design

## Overview

Replace the dual journal_mode (file/memory) with a single memory-only mode that dumps to SQLite format at the end. This provides:
- No disk I/O during optimization (performance)
- SQLite output for fast resume and optuna-dashboard compatibility

## Storage Flow

```
NEW STUDY:
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────┐
│ create_study()  │────▶│ SharedMemoryJournalBackend│────▶│ copy_study()│────▶ study.db
│ (in-memory)     │     │ (no disk I/O)            │     │ (at end)    │
└─────────────────┘     └──────────────────────────┘     └─────────────┘

RESUME:
┌───────────┐     ┌──────────────────────────┐     ┌─────────────┐
│ study.db  │────▶│ copy_study() to memory   │────▶│ optimization│────▶ overwrite study.db
│ (load)    │     │ SharedMemoryJournalBackend│     │ continues   │
└───────────┘     └──────────────────────────┘     └─────────────┘
```

## File Changes

| File | Changes |
|------|---------|
| `src/optuna_optimizer/storage.py` | Remove `create_journal_storage()`. Add `dump_to_sqlite()` and `load_from_sqlite()` helpers. Keep `SharedMemoryJournalBackend`. Remove `dump_to_file()` method. |
| `src/optuna_optimizer/models.py` | Remove `journal_mode` field from `OptunaConfig` |
| `src/optuna_optimize.py` | Remove all `journal_mode` branching. Always use `SharedMemoryJournalBackend`. Replace `dump_to_file()` with `dump_to_sqlite()`. Update resume to use `load_from_sqlite()`. Artifact: `study.db` not `journal.log`. |
| `src/optuna_optimizer/worker.py` | No changes—already receives shared backend state |
| `src/config_utils.py` | Remove `journal_mode` from defaults/validation |
| `configs/template.json` | Remove `journal_mode` setting |

## New Storage Functions

### `dump_to_sqlite()`

```python
def dump_to_sqlite(
    backend: SharedMemoryJournalBackend,
    study_name: str,
    sqlite_path: Path,
) -> None:
    """Copy study from in-memory backend to SQLite file.

    Uses optuna.copy_study() to handle all trial data, user attrs, etc.
    Overwrites existing file if present.
    """
```

### `load_from_sqlite()`

```python
def load_from_sqlite(
    sqlite_path: Path,
    study_name: str,
) -> tuple[SharedMemoryJournalBackend, optuna.Study]:
    """Load existing SQLite study into memory for resume.

    Creates fresh SharedMemoryJournalBackend, copies study into it,
    returns backend and loaded study ready for optimization.
    """
```

## Orchestrator Changes

### `run_optimization()`

```python
# Simplified - always memory mode:
shared_backend = SharedMemoryJournalBackend.create_parent()
storage = JournalStorage(shared_backend)
```

### `_run_optimization_core()` finally block

```python
# Dump to SQLite instead of journal:
sqlite_path = study_dir / "study.db"
dump_to_sqlite(shared_backend, study.study_name, sqlite_path)
```

### `resume_optimization()`

```python
# Load from SQLite into memory:
sqlite_path = study_dir / "study.db"
shared_backend, study = load_from_sqlite(sqlite_path, study_dir.name)
```

## Error Handling

- **Resume validation:** Check for `study.db` instead of `journal.log`
- **Interrupted optimization:** Still dump to SQLite in finally block to save completed trials
- **Pareto extraction:** Extract from in-memory study before dumping (no reload needed)
- **Old studies:** Fail with clear message if `journal.log` exists but no `study.db` (no migration)

## Removed

- `journal_mode` config option
- `create_journal_storage()` function
- `dump_to_file()` method on SharedMemoryJournalBackend
- Lock file cleanup logic (no lock files with SQLite)
