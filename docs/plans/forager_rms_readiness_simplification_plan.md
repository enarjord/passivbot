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
- Bridging unverified internal gaps for RMS. An unverified hole excludes only that coin until
  it is repaired when RMS is required for ranking. A provisional flat bridge would approximate the squared-return
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

## Implementation Record

[PR #2343](https://github.com/enarjord/passivbot/pull/2343) shipped phases A through D on
`v9`. The labels below preserve the original plan's work breakdown.

### PR-A: Per-candidate RMS exclusion and ranking observability

In `passivbot-rust/src/orchestrator.rs`, `build_forager_candidates_into()` matches
`OrchestratorError::RmsDirectionalityUnavailable` before it calls
`handle_strategy_input_error(...)` and records a candidate exclusion. Every other RMS lookup
error is returned, including unmarked `MissingEma` even when
`allow_missing_strategy_inputs` is true. Non-finite and out-of-range scores remain fatal.
The initial `out.clear()` resets the output buffer for reuse.

Backtests share the candidate builder. An exact span in `rms_directionality_warmup_spans`
excludes that candidate when ranking is required. The default `minimum_coin_age_days = 60`
is longer than the default 30.3-day window, so the age filter already removes new listings
whose age is below that window.

Live orchestration supplies the startup readiness bit. The forager selection event reports
Rust's candidate and eligible counts, whether ranking was required, candidate exclusions, and
the deferral reason. `src/live/forager_health.py` tracks recurring exclusions and sustained
stalls when a side has candidates and free slots. The stall check accounts for working
entries and selected candidates that are all in cooldown.

`docs/ai/features/strategy_runtime.md` records the per-candidate ranking contract and the
separate startup rule. `docs/ai/error_contract.md` preserves the fits-in-slots rule, where
unused ranking inputs can be absent without a ranking warning.

The whole-side readiness expectations in `tests/test_adaptive_entry_cooldown.py` belonged to
the original baseline. They covered `test_mixed_rms_readiness_defers_entire_ranking`,
`test_cpu_rms_warmup_defer_is_span_specific_and_never_masks_invalid_scores`, and
`test_low_volume_candidates_still_require_complete_rms_scoring`. The landed behavior is
covered by `test_mixed_rms_readiness_excludes_only_the_incomplete_candidate` and by the
native forager tests. The validation cases for this design include unmarked missing RMS on a
symbol with another scoped input marked unavailable, fits-in-slots selection, and eligibility
after restart with the same candles and cutoff.

### PR-B: RMS data fixes

- **Verified no-trade minutes.** The RMS reader synthesizes flat closes only for spans
  with `GAP_REASON_NO_TRADES` evidence and a known preceding close. An unverified internal
  gap still makes a continuous RMS window unavailable.
- **Repair pages.** `CandlestickManager.plan_rms_history_repair()` sizes repair pages
  to the venue's supported capacity and respects its age limits. Tail and history jobs
  use the shared request budget.
- **Cache isolation.** Forager-scope RMS and current-scope cooldown use separate cache
  entries. `tests/test_forager_background_progress.py` covers cache isolation at the same
  cutoff and repair progress across restart.
- **Parity.** Backtest preparation can accept gaps up to `gap_tolerance_ohlcvs_minutes`
  (120 by default). Live RMS requires a continuous completed-minute window, with flat
  synthesis restricted to verified no-trade minutes.

### PR-C: Centralize the window multiplier, keeping 20

`passivbot-rust/src/rms_directionality.rs` owns the 20-span multiplier and
`warmup_returns()`. The Python warmup planner uses the PyO3 helper, and GPU screening
uses the exported Rust multiplier. The history window length is unchanged.

**A shorter window is a separate experiment, not part of this plan.** The residual EMA weight
after `k` spans (about `e^(-2k)`) does not bound the error in the normalized score. The
numerator's residual scales like `e^(-2k)`, while `sqrt(ema(r^2))` keeps a residual of about
`e^(-k)`, so the ratio can move by roughly `e^(-k)`. For span 100, 1,400 log returns of 0.0001
followed by 600 zero returns, the existing recurrence gives about `0.00247826` over 20 spans and
exactly zero over 6. Before any change, measure signed-RMS deltas, ranking and slot-selection
changes, cooldown durations, backtest and optimizer outcomes, and CPU/GPU parity. Bump the GPU
checkpoint layout revision if one is adopted.

### PR-D: Consolidate RMS acquisition into `CandlestickManager`

`_refresh_forager_candidate_candles()` schedules RMS repair as a `1m_history` job sized by
`CandlestickManager.rms_history_candle_count()`. The manager reads RMS from the canonical
one-minute store, applies the no-trade proof rules, and maintains scope-specific EMA cache
entries. The historical `"rms"` pseudo-timeframe and `rms_directionality.load()` were retired.
`src/live/rms_directionality.py` still owns the `scoring_enabled` and `adverse_enabled`
consumer predicates. The current data path is documented in
`docs/ai/features/candlestick_manager.md`.

## Validation Commands

- `cargo test` in `passivbot-rust/`, then rebuild the extension with
  `maturin develop --release`.
- Targeted pytest for forager selection, RMS directionality, adaptive entry cooldown, warmup
  utilities, and GPU parity when changing these paths.
- `PYTHONPATH=src python src/tools/check_ai_docs.py`
- `PYTHONPATH=src python src/tools/generate_live_event_registry.py --check` when event fields change.

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
