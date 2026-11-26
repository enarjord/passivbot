# Forager RMS Directionality Readiness Simplification Plan

## Objective

Remove the failure class where one forager candidate with incomplete one-minute history
blocks all new entries on a trading side. Do it by making RMS directionality follow the same
present-or-excluded rule as the other forager ranking features. Separately, fix the RMS data
path so a complete window is easier to obtain and keep.

## Problem Statement

This plan records the baseline before [PR #2343](https://github.com/enarjord/passivbot/pull/2343),
which shipped per-candidate exclusion and the RMS data-path changes on `v9`. The five
mechanisms below describe that earlier baseline.

1. **Whole-side kill switch (the failure class).**
   In `passivbot-rust/src/orchestrator.rs::build_forager_candidates_into()`, a missing
   `volume_ema`, `log_range_ema`, or `ema_bands` marks only that candidate `enabled: false`
   through `unavailable_forager_candidate`. With
   `cfg.require_forager && normalized_weights.rms_directionality > 0.0`, a missing RMS score on
   any one enabled candidate sets `ranking_unavailable` and runs `out.clear()`. No new entries
   open on that side. `docs/ai/features/strategy_runtime.md` documents this as "never select from
   a history-ready subset". Backtests share the code path: a coin listed in
   `rms_directionality_warmup_spans` raises `RmsDirectionalityUnavailable` and defers the side
   in the same way.

2. **Long unbroken history requirement.**
   `warmup_returns(span) = ceil(20 * span)`, and live history needs that many returns plus one
   close. For the default `forager.rms_directionality_ema_span_minutes = 2179.8856`, every
   candidate needs 43,599 consecutive one-minute bars (about 30.3 days). Live RMS reads with
   `standardize=False` and splits segments on any timestamp gap, so one missing minute in that
   window makes the candidate unavailable. The `20` multiplier is duplicated in
   `src/live/rms_directionality.py`, `src/warmup_utils.py`, `src/optimization/gpu/mps_kernel.py`,
   `src/optimization/gpu/service.py`, and `src/optimization/gpu/model.py`.

3. **Verified-no-trade deadlock.**
   `CandlestickManager.get_completed_candle_health()` subtracts verified
   `GAP_REASON_NO_TRADES` spans from `refreshable_missing_spans`, so repair treats them as
   resolved. The RMS reader does not synthesize those minutes, so they stay holes. One verified
   zero-trade minute inside the window keeps the symbol unavailable until the minute ages out.

4. **Slow repair.**
   The `"rms"` pseudo-timeframe repair in `src/passivbot.py::_refresh_forager_candidate_candles()` pages
   100 bars at a time (`page_end - 99 * ONE_MIN_MS`) under the shared
   `max_ohlcv_fetches_per_minute` budget. The exchange-side page limit is often 1,000 one-minute
   bars.

5. **Invisible stalls.**
   The forager selection event sets `candidate_count=len(top_scores)` and
   `eligible_count=len(top_scores)` in `src/passivbot.py`. An empty score list looks the same
   whether the universe was empty or ranking was blocked, so a stalled side raises no signal.

Mechanisms 2 to 4 only change how often mechanism 1 fires. Mechanism 1 is the failure class.
Any hole inside any finite window, whether 20 or 6 spans long, still triggers it.

## Design Goals

1. Remove the special case rather than add repair heuristics around it.
2. Keep Rust as the single source of truth for RMS math, window sizing, and forager selection,
   so live, CPU backtest, and GPU screening stay aligned.
3. Keep the trading-safety invariants in `docs/ai/error_contract.md`:
   - never fabricate a neutral RMS score for a missing symbol;
   - malformed or non-finite producer output stays fatal;
   - protective closes, risk reducers, and panic actions stay independent of forager ranking.
4. Keep decisions reproducible. Eligibility at a decision cutoff depends only on the canonical
   candle store and the cutoff. No elapsed-time grace windows, retry counters, or other
   restart-sensitive state change which candidates are eligible.

## Non-Goals

- Changing the definition of signed RMS directionality
  (`ema(r) / sqrt(ema(r^2))` over one-minute log returns with zero-seeded weights).
- Changing the RMS window length in this work (see PR-C).
- Relaxing `current`-scope (entry-cooldown) RMS strictness.
- Changing held-position behaviour. PR-A restores entries on new candidates only. A held
  position above its effective per-position exposure limit still gets no adds, and while its
  trailing close is not activated and the unstuck allowance is zero, it still gets no reduce
  orders. Held positions are already marked active before candidate collection
  (`actives_long` / `actives_short`), so candidate exclusion cannot affect them.
- Bridging unverified internal gaps for RMS. Once PR-A lands, an unverified hole excludes only
  that coin until it is repaired. A provisional flat bridge would approximate the squared-return
  moment, and liveness no longer needs it.

## Target Rule

For every required forager ranking feature (`volume`, `log_range`, `ema_bands`,
`rms_directionality`):

> The feature is present at the decision cutoff, or that candidate is excluded.

There is exactly one side-wide deferral: **until the first startup candidate refresh pass has
attempted every candidate**. This is a readiness bit, not a clock. It is set once that pass has
attempted each symbol or confirmed its local cache coverage, whether or not every symbol
succeeded. A hung or unrepairable coin therefore cannot hold it. The bit guards against startup
selection bias, where symbols that happen to refresh first would win ranking before others have
been read at all. After it is set, eligibility is a pure function of the stored candles and the
cutoff.

An excluded candidate is re-admitted on the first cycle in which its feature is present. No
state is kept between cycles.

## Implementation Plan

### PR-A: Per-candidate RMS exclusion and ranking observability (ships first, standalone)

**Rust (`passivbot-rust/src/orchestrator.rs`)**

- In `build_forager_candidates_into()`, replace `ranking_unavailable = true` / `out.clear()` with
  `out[pos] = unavailable_forager_candidate(out[pos].index)` only for
  `RmsDirectionalityUnavailable` after the existing `handle_strategy_input_error(...)`
  classification. Propagate every other error from the RMS lookup, including an unmarked
  `MissingEma` even if `allow_missing_strategy_inputs` is true. Non-finite and out-of-range
  scores stay fatal.
- Backtests share this path. A coin inside `rms_directionality_warmup_spans` is then excluded
  individually instead of deferring the side. With the default `minimum_coin_age_days = 60`,
  which is longer than the default 30.3-day window, age admission already removes such coins.
  The change matters only for configs with a minimum age shorter than the RMS window.
- Add a side-level deferral input for the startup readiness bit, for example
  `forager_ranking_ready: bool` per side, which defaults to `true`. Backtests always pass `true`.
- Extend `OrchestratorDiagnostics` with, per side: eligible count, whether ranking was required
  (`require_forager`), and each excluded candidate with its feature and reason
  (`missing_rms`, `rms_warmup`, `missing_volume`, and so on), or the deferral reason.

**Python (`src/passivbot.py`, `src/live/event_emitters.py`, `src/passivbot_monitor.py`)**

- Own the startup readiness bit: set it after the first candidate refresh pass has attempted
  every candidate.
- Fill the forager selection event from the Rust diagnostics instead of `len(top_scores)`.
- Warn when the same symbol stays excluded across many consecutive cycles, and include the
  feature and reason.
- Warn, and surface in the monitor, when a side with free slots and at least one eligible
  candidate opens no new entries for a configurable number of minutes. This catches stalls
  from any cause, not only RMS.

**Docs**

- Replace the historical whole-ranking deferral clause in
  `docs/ai/features/strategy_runtime.md` with the per-candidate rule. Document startup
  fairness as a distinct rule. In `docs/ai/error_contract.md`, record both rules and preserve
  the existing behavior when eligible candidates fit the free slots. In that case, unused
  ranking inputs may be absent without a ranking warning.
- Add a `CHANGELOG.md` `Unreleased` entry.

**Tests**

- Rust: one candidate without RMS among several healthy candidates is excluded, and the others
  rank; a malformed or non-finite score stays fatal; an unmarked missing score on a symbol
  with another scoped input marked unavailable still raises `MissingEma`. The side defers
  while `forager_ranking_ready` is false. When the eligible candidates fit the free slots,
  selection follows the case where ranking is unnecessary.
- Update the historical whole-side readiness assertions in `tests/test_adaptive_entry_cooldown.py`
  for per-candidate exclusion. The baseline test names were
  `test_mixed_rms_readiness_defers_entire_ranking`,
  `test_cpu_rms_warmup_defer_is_span_specific_and_never_masks_invalid_scores`, and
  `test_low_volume_candidates_still_require_complete_rms_scoring`. Add native coverage for the
  changed behavior in the shared Rust candidate builder.
- Reproducibility: an uninterrupted run and a restarted run with the same candle store and
  cutoff produce the same eligible set and selection once the readiness bit is set.
- Multi-symbol outage: when several symbols are missing RMS at once, only those are excluded.
- Live fake harness: a forager side with free slots and one candidate missing RMS opens entries
  on healthy candidates instead of producing zero orders.

### PR-B: RMS data fixes (independent of PR-A)

- **Verified no-trade minutes.** In the RMS read, synthesize only verified
  `GAP_REASON_NO_TRADES` spans that have a known prior close, as flat closes giving `r = 0`. Use
  an explicit allow-list or `excluded_synthetic_ranges`. Do not call unrestricted
  `standardize_gaps(fill_leading_gaps=False, fill_trailing_gaps=False)`, because it synthesizes
  every internal hole with real candles on both sides, including multi-week unverified gaps.
  Unverified holes stay holes.
- **Repair pages.** Size RMS history repair pages to the venue's supported one-minute page
  limit instead of 100 bars. Keep the shared request budget, and keep scheduling for a fresh
  tail independent from scheduling for an incomplete historical window, so a deep repair makes
  progress across restarts while ordinary and protective work continues.
- **Cache isolation.** Test that a ranking-only (forager-scope) cached RMS value cannot satisfy
  `current`-scope cooldown, including at the same cutoff.
- **Parity statement.** Backtest preparation accepts gaps up to `gap_tolerance_ohlcvs_minutes`
  (120 by default), while live RMS accepts only verified no-trade gaps. Document this as an
  intentional difference.

**Tests:** a verified no-trade minute inside the window yields the same score as backtest flat
replay; an unverified internal gap remains unavailable; a large historical gap backfills in
venue-sized pages; and repair resumes after a restart.

### PR-C: Centralize the window multiplier, keeping 20

- Define `pub const RMS_WARMUP_SPAN_MULTIPLIER: f64 = 20.0` and `warmup_returns()` in
  `passivbot-rust/src/rms_directionality.rs`, and expose the helper through PyO3.
- Replace the duplicated `20.0 * span` in `src/live/rms_directionality.py`,
  `src/warmup_utils.py`, `src/optimization/gpu/mps_kernel.py`,
  `src/optimization/gpu/service.py`, and `src/optimization/gpu/model.py` with the Rust helper,
  or with a constant that a test checks against Rust.
- No behaviour change.

**A shorter window is a separate experiment, not part of this plan.** The residual EMA weight
after `k` spans (about `e^(-2k)`) does not bound the error in the normalized score. The
numerator's residual scales like `e^(-2k)`, while `sqrt(ema(r^2))` keeps a residual of about
`e^(-k)`, so the ratio can move by roughly `e^(-k)`. For span 100, 1,400 log returns of 0.0001
followed by 600 zero returns, the existing recurrence gives about `0.00247826` over 20 spans and
exactly zero over 6. Before any change, measure signed-RMS deltas, ranking and slot-selection
changes, cooldown durations, backtest and optimizer outcomes, and CPU/GPU parity. Bump the GPU
checkpoint layout revision if one is adopted.

### PR-D: Consolidate RMS acquisition into `CandlestickManager` (cleanup)

- Remove the `"rms"` pseudo-timeframe surface from `_refresh_forager_candidate_candles()`, and retire
  `src/live/rms_directionality.py::load()`. Keep `scoring_enabled` and `adverse_enabled` if they
  are still useful.
- Read RMS windows through the canonical one-minute metric path, so they share the gap evidence
  and `_ema_cache` invalidation used by the other metrics. Keep the PR-B synthesis restrictions,
  and keep the window requirement separate from other 1m readiness requirements.
- Update `docs/ai/features/candlestick_manager.md`.

This is a simplification of code paths, not a liveness fix. PR-A and PR-B do not depend on it.

## Validation Commands

- `cargo test` in `passivbot-rust/`, then rebuild the extension with
  `maturin develop --release`.
- Targeted pytest for forager selection, RMS directionality, adaptive entry cooldown, warmup
  utilities, and GPU parity, as each PR touches them.
- `PYTHONPATH=src python src/tools/check_ai_docs.py`
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check` when PR-A changes
  event fields.

## Decisions Recorded

- Eligibility is evidence-based at the cutoff. The plan uses no grace timers and no
  retry-exhaustion classification.
- The startup readiness bit is the only side-wide deferral. It opens once the first pass has
  attempted every candidate. After a cold or budget-limited pass, candidates whose repair
  finished first may enter the first ranking while others are still ineligible. Track eligible
  and excluded counts by side through startup to see how long that imbalance lasts.
- The window stays at 20 spans in this work.
- Unverified internal gaps are never synthesized for RMS.
- Backtest per-coin exclusion comes from the shared Rust path. There is no separate backtest
  rule.
