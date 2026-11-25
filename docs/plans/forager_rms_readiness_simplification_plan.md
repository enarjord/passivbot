# Forager RMS Directionality Readiness Simplification Plan

## Objective

Eliminate the failure class in which a single forager candidate with an incomplete one-minute
candle history permanently blocks all new entries on a trading side, while simplifying live RMS
directionality acquisition, gap handling, and forager readiness into the canonical `1m` EMA and
candidate-eligibility pipelines.

## Problem Statement

On `v9`, three independent design choices compound into a brittle live failure mode:

1. **Excessive unbroken 1m history requirement (`20 * span`):**
   - `passivbot-rust/src/rms_directionality.rs::warmup_returns()`,
     `src/live/rms_directionality.py::history_candle_count()`,
     `src/warmup_utils.py::_rms_warmup_minutes()`,
     `src/optimization/gpu/mps_kernel.py`, and
     `src/optimization/gpu/service.py` all hardcode `ceil(20.0 * span)` returns (`+ 1` closes).
   - For the default `forager.rms_directionality_ema_span_minutes = 2179.8856`, every candidate
     must maintain **43,599 consecutive 1-minute bars (30.28 days)** with zero breaks.
   - Because `alpha = 2 / (span + 1)`, the exponential tail weight remaining after `20 * span`
     bars is approximately `e^-40 ≈ 4.2e-18` (below IEEE-754 `f64` machine epsilon). Requiring
     `20 * span` buys sub-`1e-12` numerical parity at a steep operational cost in disk/RAM I/O,
     backfill time, and gap exposure.

2. **Bespoke RMS data and repair pipeline with split-brain gap semantics:**
   - Live RMS bypasses `CandlestickManager`'s canonical EMA metric path and loads raw candles via
     `src/live/rms_directionality.py::load()` with `standardize=False`, then splits contiguous
     segments on `np.diff(ts, append=-1) != 60_000`.
   - PR #2298 added a separate `"rms"` pseudo-timeframe repair surface in
     `src/passivbot.py` (`refresh_forager_candles`), but two structural defects remain:
     - **Verified `no_trades` deadlock:** `CandlestickManager.get_completed_candle_health()`
       subtracts `verified_no_trade_spans` (`GAP_REASON_NO_TRADES`) from
       `refreshable_missing_spans`, so the repair loop considers those minutes resolved.
       Meanwhile, `rms_directionality.load()` uses `standardize=False`, which skips
       `standardize_gaps()` and omits verified zero-trade candles. A single verified zero-trade
       minute anywhere inside a 30.28-day window permanently breaks RMS continuity for that
       symbol until the minute ages out 30 days later.
     - **Slow 100-bar repair pagination:** Each background `"rms"` repair attempt is capped at
       `99 * ONE_MIN_MS` (100 bars) while sharing `max_ohlcv_fetches_per_minute` with all other
       surfaces. Backfilling ~30k missing bars takes ~300 background turns even when the exchange
       supports up to 1,000 1m bars per request.

3. **Whole-side kill switch on any single missing candidate:**
   - In `passivbot-rust/src/orchestrator.rs::build_forager_candidates_into()`, missing
     `volume_ema`, `log_range_ema`, or `ema_bands` marks only the affected candidate
     `enabled: false` (`unavailable_forager_candidate`).
   - By contrast, when `cfg.require_forager && normalized_weights.rms_directionality > 0.0`,
     a missing RMS score on **any single candidate** sets `ranking_unavailable = true` and runs
     `out.clear()`, halting all new entries on the entire side.
   - Any newly listed coin with `< 30.28` days of history, any unrepairable exchange outage gap,
     or any verified zero-trade minute on one approved coin freezes new entries across all open
     slots for every healthy coin on that side.

## Design Goals

1. **Eliminate the entire failure class by removing special cases**, rather than adding more
   repair heuristics on top of a fragile 30-day unbroken-1m requirement.
2. **Preserve Rust as the single source of truth** for RMS math, warmup window sizing, and
   forager selection semantics, keeping live, CPU backtest, and GPU screening aligned.
3. **Preserve trading-safety invariants** from `docs/ai/error_contract.md`:
   - Never fabricate neutral RMS scores (`0.0`) for missing symbols.
   - Never weaken fatal validation of malformed or non-finite Rust outputs.
   - Keep protective closes, risk reducers, and panic actions completely independent of forager
     ranking readiness.
4. **Keep live state stateless and restart-reproducible**: no persisted synthetic candles for
   unverified gaps and no decision-changing local state that cannot be rederived on restart.

## Non-Goals

- Changing the mathematical definition of signed RMS directionality
  (`ema(r) / sqrt(ema(r^2))` over 1m log returns with zero-seeded exponential weights).
- Introducing multi-timeframe resampling for large RMS spans (keeping 1m returns avoids a
  separate multi-TF semantic contract across CPU, GPU, and live).
- Relaxing `current` (entry-cooldown) RMS strictness to allow stale cached ranking values.

## Target Architecture

The simplification rests on three orthogonal pillars.

### Pillar 1: Right-Size the RMS Finite Window (`20 * span` -> `6 * span`) With a Single Source of Truth

With `alpha = 2 / (span + 1)`, the residual exponential weight after `k * span` returns is
`(1 - alpha)^(k * span) ≈ e^(-2k)`:

| Multiplier `k` | Residual tail weight | Bars for default `span = 2179.8856` | Calendar days |
|---|---:|---:|---:|
| `20` (current) | `~4.2e-18` | `43,599` | `30.28 d` |
| `6` (proposed) | `~6.1e-6` | `13,081` | `9.08 d` |
| `5` (alternative) | `~4.5e-5` | `10,901` | `7.57 d` |

- Change the canonical warmup multiplier from `20.0` to **`6.0`** (configurable only as a
  compile-time constant in Rust, not a user-facing knob).
  - Truncation error `< 1e-5` is negligible for both forager ranking and entry cooldown, while
    cutting required 1m history, disk/memory reads, backfill requests, and gap surface area by
    **3.3x**.
- Define one canonical constant and helper in `passivbot-rust/src/rms_directionality.rs`:
  - `pub const RMS_WARMUP_SPAN_MULTIPLIER: f64 = 6.0;`
  - `pub fn warmup_returns(span: f64) -> Result<usize, String>`
  - Expose `rms_directionality_warmup_returns(span: f64) -> PyResult<usize>` via PyO3 in
    `passivbot-rust/src/python.rs`.
- Replace all duplicated `20.0 * span` formulas with the canonical Rust helper/constant:
  - `src/warmup_utils.py::_rms_warmup_minutes()`
  - `src/optimization/gpu/mps_kernel.py`
  - `src/optimization/gpu/service.py`
  - Live `CandlestickManager` / RMS loader
- Bump the GPU checkpoint/layout compatibility revision if optimizer checkpoints cache
  pre-computed RMS-windowed fitness/replay state, so `20 * span` and `6 * span` checkpoints
  cannot be mixed silently.

### Pillar 2: Unify Live RMS Acquisition Into `CandlestickManager`'s Canonical 1m Metric Pipeline

Remove the bespoke RMS data-loading and background-repair subsystem and make RMS a first-class
1m metric in `CandlestickManager`:

1. **Canonical gap semantics (fixing the `GAP_REASON_NO_TRADES` deadlock):**
   - Compute RMS from `CandlestickManager`'s standardized 1m series with
     `fill_leading_gaps=False, fill_trailing_gaps=False`:
     - **Verified internal `GAP_REASON_NO_TRADES` spans** are synthesized from the prior close,
       producing an exact `r = 0.0` log return (consistent with backtest zero-candle replay).
     - **Forager ranking (`forager` scope, current remote-enabled candidates)** may additionally
       bridge later-bounded internal gaps within `live.max_active_candle_tail_gap_minutes` via
       the same non-persisted provisional zero-volume continuity policy already documented for
       `volume` and `log_range` in `docs/ai/error_contract.md`. Missing rows remain unresolved
       and retryable and are never written to disk or runtime candle caches.
     - **Unverified gaps exceeding the tolerance, leading gaps (pre-inception), and open tails**
       remain unbridged so RMS reports unavailable rather than inventing returns.
     - **Cooldown (`current` scope)** remains strict on the current completed-candle cutoff and
       never consumes a stale cached ranking value.

2. **Delete the separate `"rms"` pseudo-timeframe surface in `src/passivbot.py`:**
   - Fold the `6 * span + 1` 1m candle requirement (when RMS scoring or adverse cooldown is
     enabled for a symbol) directly into the canonical `1m` warmup window computed by
     `compute_live_warmup_windows()`, subject to normal `1m` health tracking.
   - When `1m` background refresh repairs a historical gap, fetch up to the exchange's native
     page size (e.g. up to 1,000 1m bars per call, bounded by the existing per-surface time
     slice) instead of the 100-bar cap introduced in PR #2298.
   - Reuse `CandlestickManager._ema_cache` and its existing candle-correction / gap-evidence
     invalidation directly inside `get_latest_ema_metrics()` / `get_latest_ema_metric_spans()`,
     retiring `src/live/rms_directionality.py::load()` (keeping only lightweight config-gating
     helpers `scoring_enabled` / `adverse_enabled` if still useful).

### Pillar 3: Unified Forager Readiness & Single-Candidate Isolation

Replace the split between `unavailable_forager_candidate` (used for `volume`, `log_range`,
`ema_bands`) and whole-side `out.clear()` (used only for `rms_directionality`) with one uniform
readiness contract across **all** forager ranking features:

1. **Distinguish transient unavailability from persistent unavailability:**
   - **Transient unavailability** (e.g., startup warmup before initial candidate refresh sweep
     completes, or a bounded transient refresh/network lag within a short grace period):
     defer ranking for the side so faster-refreshing symbols do not win via selection bias.
   - **Persistent unavailability** (e.g., pre-inception / newly listed symbol lacking required
     history on the exchange, unrepairable exchange gap after retry exhaustion, or a symbol
     remaining unavailable past the transient deferral grace window):
     exclude **only that candidate** (`enabled: false` via `unavailable_forager_candidate`)
     and emit a structured warning/diagnostic so the remaining healthy universe continues
     ranking and opening positions.

2. **Uniform Rust handling in `passivbot-rust/src/orchestrator.rs`:**
   - Remove the RMS-specific `out.clear()` branch in `build_forager_candidates_into()`.
   - Model candidate readiness explicitly in the `SymbolInput` / `OrchestratorInput` contract:
     - A candidate marked transiently unready defers side ranking (bounded by Python's
       startup/refresh grace policy, or a side-level `forager_ranking_deferred` / per-candidate
       unavailability classification).
     - A candidate marked persistently unavailable for any required ranking feature
       (`volume`, `log_range`, `ema_bands`, `rms_directionality`) is excluded individually
       (`enabled: false`) and recorded in `OrchestratorDiagnostics` (with reason/feature).
   - Malformed or non-finite supplied values remain **fatal** before reconciliation.

3. **Backtest parity consideration:**
   - In multi-coin backtests where a newly listed coin enters the universe mid-run and is still
     in `rms_directionality_warmup_spans`, excluding only that warming coin (treating it as not
     yet eligible for ranking until its warmup completes) rather than freezing entries for all
     already-listed coins matches the persistent-unavailability live rule and prevents one new
     listing from stalling a multi-coin portfolio in backtests as well.

## Phased Implementation Plan

### Phase 1: Right-Size RMS Warmup Window & Centralize Constant

- **Rust (`passivbot-rust/`):**
  - Update `passivbot-rust/src/rms_directionality.rs`: set `RMS_WARMUP_SPAN_MULTIPLIER = 6.0`
    and update `warmup_returns()` and unit tests.
  - Export `rms_directionality_warmup_returns` in `passivbot-rust/src/python.rs`.
  - Update any hardcoded `20 * span` expectations in Rust backtest/orchestrator tests.
- **Python & GPU (`src/`):**
  - Replace hardcoded `20.0 * span` in `src/warmup_utils.py`,
    `src/optimization/gpu/mps_kernel.py`, and `src/optimization/gpu/service.py` with the
    exported Rust helper (or shared constant verified against Rust in tests).
  - Verify Metal/CUDA/ROCm RMS kernels read `rms_directionality_window` dynamically from the
    parameter matrix (no hardcoded `20` in shader code); bump GPU checkpoint layout revision if
    applicable.
- **Validation:**
  - `cargo test` in `passivbot-rust/`.
  - Rebuild PyO3 extension (`maturin develop --release`).
  - Run CPU/GPU parity tests (`tests/test_rms_directionality*`, GPU parity tests,
    `tests/test_warmup_utils.py`).

### Phase 2: Unify Live RMS Into `CandlestickManager`

- **`src/candlestick_manager.py`:**
  - Add RMS directionality support to the canonical 1m metric reader
    (`get_latest_ema_metrics` / `get_latest_ema_metric_spans` or dedicated method sharing the
    same standardized 1m fetch + `_ema_cache` invalidation path).
  - Ensure verified `GAP_REASON_NO_TRADES` internal spans are standardized (`r = 0.0`) while
    unverified leading/trailing gaps remain unbridged.
  - Apply the existing bounded provisional internal-gap continuity policy
    (`live.max_active_candle_tail_gap_minutes`) to remote-enabled forager ranking reads, with
    identical transition diagnostics as `volume` and `log_range`.
- **`src/passivbot.py` & `src/live/rms_directionality.py`:**
  - Remove the `"rms"` pseudo-timeframe surface and its 100-bar repair branch from
    `refresh_forager_candles()`.
  - Fold RMS 1m candle count into canonical `1m` warmup/health calculation and ensure
    historical 1m gap repair uses full exchange page size (up to 1,000 bars/request, bounded by
    the cycle time slice).
  - Route orchestrator RMS bundle loading through `CandlestickManager` and retire
    `src/live/rms_directionality.py::load()`.
- **Validation:**
  - Regression test: a symbol with a verified `GAP_REASON_NO_TRADES` minute inside its RMS
    window still produces a valid RMS score matching backtest zero-candle replay.
  - Regression test: a symbol with an unverified internal gap within
    `live.max_active_candle_tail_gap_minutes` uses bounded continuity for ranking without
    persisting synthetic rows, and resumes authoritative candles once repaired.
  - Regression test: a large historical 1m gap backfills in exchange-page-sized chunks rather
    than 100-bar increments.

### Phase 3: Unified Forager Readiness & Single-Candidate Isolation

- **Rust (`passivbot-rust/src/orchestrator.rs`, `coin_selection.rs`):**
  - Unify missing-feature handling across `volume_ema`, `log_range_ema`, `ema_bands`, and
    `rms_directionality_score` in `build_forager_candidates_into()`.
  - Replace the RMS-only `out.clear()` with explicit distinction between transient side-level
    deferral and persistent single-candidate exclusion (`unavailable_forager_candidate`), plus
    structured `OrchestratorDiagnostics` reporting of excluded candidates and reasons.
  - Align `rms_directionality_warmup_spans` in backtests with single-candidate exclusion (or
    document the exact chosen semantics if kept separate).
- **Python (`src/passivbot.py`, `src/passivbot_monitor.py`, `src/live/event_emitters.py`):**
  - Classify candidate unavailability as transient vs. persistent based on `CandlestickManager`
    health (leading/pre-inception gap, retry-exhausted gap, or grace window elapsed).
  - Surface persistent single-candidate exclusions clearly in logs, live events, and monitor
    state so operators immediately see which symbol is excluded and why, while healthy symbols
    continue trading.
- **Canonical Contracts & Docs:**
  - Update `docs/ai/error_contract.md` (Forager And Eligibility Inputs).
  - Update `docs/ai/features/strategy_runtime.md` (Adaptive entry timing and RMS
    directionality: `6 * span`, standardized `no_trades` continuity, unified candidate
    isolation).
  - Update `docs/ai/features/candlestick_manager.md` (remove `"rms"` pseudo-timeframe surface,
    document canonical 1m warmup + full-page repair).
  - Add net user-facing summary to `CHANGELOG.md` under `Unreleased` when implementing.
- **Validation:**
  - Rust unit/integration tests for single-candidate exclusion vs. transient deferral and fatal
    malformed-score rejection.
  - Python live/fake-harness regression test: 10-slot forager with 5 held positions and 1
    candidate missing RMS (e.g., pre-inception or unrepairable gap) continues ranking and
    opening entries on the remaining healthy candidates instead of producing zero orders.
  - Documentation checks:
    `PYTHONPATH=src python src/tools/check_ai_docs.py`,
    `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check`,
    `pytest tests/test_ai_docs.py tests/test_live_event_registry_docs.py`.

## Open Decisions to Confirm Before Implementation

1. **Warmup multiplier (`6 * span` vs. `5 * span`):**
   - `6 * span` (`~6.1e-6` tail weight, ~9.1 days for default span) is the recommended default;
     `5 * span` (`~4.5e-5` tail weight, ~7.6 days) is a viable tighter alternative.
2. **Transient deferral grace window:**
   - Proposed default: defer side ranking only during initial startup sweep or while an active
     repair is in progress up to a bounded grace period (e.g., `max_forager_candle_staleness_minutes`
     or 5 minutes); once a candidate is proven pre-inception or exceeds the grace window,
     exclude only that candidate.
3. **Backtest listing-warmup parity:**
   - Confirm whether mid-backtest newly listed coins in `rms_directionality_warmup_spans` should
     also be excluded individually (recommended, for live/backtest parity) rather than deferring
     ranking for the entire side.
