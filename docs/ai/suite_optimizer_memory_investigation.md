# Suite Optimizer Memory Investigation

**Date:** 2026-01-31
**Issue:** Bus error / OOM when running optimizer with suite mode on VPS (32GB RAM)
**Status:** UNRESOLVED - requires further investigation

## Problem Summary

The optimizer crashes with "Bus error (core dumped)" when running suite mode with many CPUs (16) on a 32GB VPS, even though it previously worked. The issue appeared after recent changes to the data strategy redesign.

## Key Observations

1. **Memory estimate showed incorrect scaling:**
   - Initial lazy slicing approach showed: `peak = master_size × (1 + n_cpus)`
   - With 2GB master and 16 CPUs: 2GB × 17 = 34GB (exceeds 32GB)

2. **Crash occurs BEFORE suite context preparation:**
   - Log shows "Successfully loaded hlcvs data from cache"
   - Then immediately "Bus error (core dumped)"
   - This suggests the issue is in `prepare_suite_contexts` or SharedMemory creation

3. **21 leaked shared_memory objects:**
   - Indicates SharedMemory segments were created but not properly cleaned up
   - Crash happens during SharedMemory allocation for scenarios

## Attempted Fixes

### Fix 1: Explicit array cleanup after SharedMemory creation
- Added `del hlcvs, btc_usd_prices` after `_build_dataset()` in `prepare_master_datasets()`
- **Result:** Did not fix the issue

### Fix 2: Memory estimation and warning
- Added memory estimation based on `master_size × (1 + n_cpus)`
- **Result:** Warning displayed but didn't prevent crash

### Fix 3: Revert lazy slicing to per-scenario SharedMemory
- Reverted from lazy slicing (workers copy) to pre-created per-scenario SharedMemory
- **Result:** Still crashes with Bus error

## Technical Analysis

### Memory Flow in Suite Mode

1. **Cache Loading:**
   - `load_coins_hlcvs_from_cache()` loads ~2GB compressed → ~7GB uncompressed for 1791 days

2. **Master Dataset Creation (`prepare_master_datasets`):**
   - Calls `prepare_hlcvs_mss()` which loads data
   - `_build_dataset()` creates SharedMemory via `shared_array_manager.create_from()`
   - Original arrays should be freed after copy

3. **Per-Scenario SharedMemory Creation:**
   - For each of 12 scenarios, creates sliced SharedMemory
   - Each scenario may be similar size to master if using full date range

### Potential Root Causes

1. **Total SharedMemory exceeds system limits:**
   - 12 scenarios × ~2GB each = ~24GB SharedMemory
   - Plus master dataset = ~2GB
   - Plus Python overhead = ~26-28GB total
   - May hit system SharedMemory limits (`/dev/shm` typically 50% of RAM = 16GB)

2. **Temporary memory spike during creation:**
   - Original array + contiguous copy + SharedMemory all exist simultaneously
   - Could spike to 3× scenario size during creation

3. **SharedMemory fragmentation:**
   - Creating many large SharedMemory segments may fail due to fragmentation

## Code Locations

### Key Files
- `src/optimize.py`: Main optimizer, creates `SuiteEvaluator`
- `src/optimize_suite.py`: `prepare_suite_contexts()`, `ScenarioEvalContext`
- `src/suite_runner.py`: `prepare_master_datasets()`, `_build_dataset()`, `_prepare_dataset_subset()`
- `src/shared_arrays.py`: `SharedArrayManager`, `create_from()`, `attach_shared_array()`

### Critical Functions
- `prepare_master_datasets()` at `suite_runner.py:436`
- `_build_dataset()` at `suite_runner.py:447`
- `prepare_suite_contexts()` at `optimize_suite.py:60`

## Data Sizes (Estimated for 1791 days, 22 coins)

- Minutes: ~2.58M (1791 × 24 × 60)
- HLCVS shape: (2.58M, 22, 5)
- Size: 2.58M × 22 × 5 × 8 bytes = ~2.3GB
- With 12 scenarios (many full-range): ~24GB+ SharedMemory total

## Debugging Suggestions

1. **Check /dev/shm limits:**
   ```bash
   df -h /dev/shm
   mount | grep shm
   ```

2. **Monitor SharedMemory during startup:**
   ```bash
   watch -n 1 'ls -la /dev/shm/'
   ```

3. **Add logging to SharedMemory creation:**
   - Log before/after each `create_from()` call
   - Log total SharedMemory allocated

4. **Test with fewer scenarios:**
   - Modify template.json to use only 2-3 scenarios
   - See if it completes without crash

5. **Test SharedMemory limits:**
   ```python
   from multiprocessing import shared_memory
   import numpy as np
   # Try creating a 2GB SharedMemory segment
   shm = shared_memory.SharedMemory(create=True, size=2*1024**3)
   ```

## Possible Solutions to Explore

1. **Reduce per-scenario memory:**
   - Don't create full SharedMemory for scenarios that use same date range as master
   - Use indices/views instead of copies for same-range scenarios

2. **Sequential scenario evaluation:**
   - Instead of pre-creating all scenario SharedMemory, create/destroy per evaluation
   - Trade-off: slower but uses less peak memory

3. **Memory-mapped files instead of SharedMemory:**
   - Use `np.memmap` instead of SharedMemory
   - Allows larger datasets with OS-managed paging

4. **Streaming data to Rust:**
   - Modify Rust backtest to accept data in chunks
   - Major architectural change

5. **Deduplication:**
   - If scenarios share same date range, share the same SharedMemory
   - Add scenario grouping by (start_date, end_date, coins)

## Configuration That Worked vs Failed

**Worked:** 6 CPUs, 2025-12 to 2026-01 (short range), local MacBook
**Failed:** 16 CPUs, 2021-03 to present (1791 days), VPS 32GB RAM

## Next Steps

1. Verify `/dev/shm` size on VPS
2. Add detailed logging to SharedMemory creation
3. Test with reduced scenario count
4. Consider scenario deduplication for same-range scenarios
