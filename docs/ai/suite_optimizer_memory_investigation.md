# Suite Optimizer Memory Notes (Condensed)

## Incident

Date: 2026-01 to 2026-02

Symptom: suite optimizer runs could crash with bus error/OOM on constrained hosts.

## Primary Cause

Peak memory from large shared-memory datasets and temporary copies exceeded practical limits for some VPS setups.

## Effective Mitigations

1. Reduce duplicate copies during dataset preparation.
2. Use lazy slicing strategy for suite datasets.
3. Free intermediate arrays earlier.
4. Emit memory estimate/warnings before heavy allocation.

## Operational Guidance

1. Treat `n_cpus`, scenario count, and date range as coupled memory multipliers.
2. If failures recur, reduce CPU count and scenario breadth first.
3. Validate shared-memory capacity (`/dev/shm`) on Linux hosts.

## Code Areas

- `src/optimize.py`
- `src/optimize_suite.py`
- `src/suite_runner.py`
- `src/shared_arrays.py`
