# Risk, Unstuck, and HSL Fable Audit Action Plan - 2026-07-02

Follow-up plan after Claude's fable audit, Codex verification in
`docs/plans/risk_unstuck_hsl_fable_audit_verification_20260702.md`, and user
comments on the proposed contracts.

This is a plan only. It does not record implemented behavior unless a later
work log says so.

## Source Inputs

- Claude risk/unstuck/HSL fable audit pasted on 2026-07-02.
- Codex read-only verification report:
  `docs/plans/risk_unstuck_hsl_fable_audit_verification_20260702.md`.
- User policy comments on unstuck loss allowance, HSL mode semantics,
  statelessness, entry cooldown, inherited account history, and Python/Rust
  authority boundaries.

## Scope

This plan covers risk, HSL, unstuck, reducer coordination, entry cooldown,
config validation, and live/backtest parity items surfaced by the fable audit.
It is not a logging-overhaul implementation plan, though several items should
emit structured diagnostics once the live event stream is mature enough.

The immediate goal is to decide contracts and PR order before touching code.
Behavior-affecting work should land as reviewed PRs into `v8`, with targeted
tests and explicit docs for any intentionally surprising behavior.

Status: planning document only. No code or user-facing behavior should be
inferred from this file until an implementation PR lands and is merged.

## User Decisions Folded Into This Plan

These points intentionally override or refine the first-pass audit
recommendations:

- Unstuck allowance is a pacing budget, not an absolute per-order realized-loss
  cap after exchange min qty/cost snapping. Keep the current min-qty overshoot
  behavior and document it as intentional. The separate
  `max_realized_loss_pct` gate still blocks non-panic lossy closes.
- HSL must remain stateless. Local cache/checkpoint files may speed replay and
  diagnostics, but trading state must be reconstructable from exchange
  data plus config on a fresh VPS.
- HSL panic eligibility should be anchored to the current trading episode.
  Flattening by any means resets the episode drawdown tracker; a separate
  broader tracker may still be needed for no-restart semantics.
- HSL restart-after-RED behavior should be explicit, not overloaded solely onto
  `hsl_no_restart_drawdown_threshold`. Add a scoped
  `restart_after_red_policy` enum: `always`, `threshold`, `never`.
  `threshold` preserves the current threshold-based behavior, `always`
  restarts after cooldown regardless of no-restart drawdown, and `never`
  permanently halts the affected scope after any episode in which RED was
  active (`red_seen_in_episode`).
- HSL restart/no-restart scope is mode-local. Unified mode applies the restart
  policy to the whole account. Pside mode applies it only to the affected side.
  Coin mode applies it only to the affected coin+pside.
- ORANGE `tp_only_with_active_entry_cancellation` should block initial entries
  in the affected HSL scope, not only entries for positions already open.
- Rust should own trading logic wherever feasible. Python should gather inputs,
  call Rust, and reconcile ideal orders against exchange state rather than
  re-deciding risk/unstuck/HSL policy.
- Entry ladder throttling and time-based entry cooldown should become separate
  config concepts.
- `pnls_max_lookback_days` remains one shared knob for now, but the coupling
  must be documented and surfaced in diagnostics.
- `reduce_overweight` should use dynamic currently-tradable slot count.
- Snapped/raw balance handling should not add a secondary raw-exposure cap for
  now. Prefer tests/docs plus a user-facing warning when hysteresis/snap pct is
  high enough to make unexpected boundary behavior likely.
- HSL coin/pside drawdown should keep slot-budget-style normalization rather
  than scaling red thresholds with effective WEL or excess allowance. Raising
  TWEL should not silently raise the drawdown percentage required to trigger
  RED.
- Live coin-HSL slot budgets use configured `bot.{pside}.n_positions` only.
  Backtests may use dynamic effective n-positions for historical coin
  availability only when explicitly enabled by backtest config.
- HSL cooldown reconstruction should use the canonical HSL equity/drawdown time
  series as the primary source of RED evidence (`red_seen_in_episode`), with
  the scoped episode-end fill timestamp as the cooldown anchor. It should not
  depend on passivbot order-type markers in fill events where that can be
  avoided.

## Priority Changes From The Initial Audit

Claude's first suggested priority list treated several items as mechanical
bugs. After Codex verification and user policy review, the immediate priority
order changes as follows:

- Do not "fix" unstuck min-qty loss-allowance overshoot by skipping the order
  after min-qty snapping. The intended contract is that unstuck allowance paces
  total activity over time, while exchange min qty/cost may force a larger
  single lossy close. Document this clearly and rely on
  `max_realized_loss_pct` as the hard non-panic realized-loss backstop.
- Treat HSL boot surprises as a contract/design problem, not as an isolated
  live bug. The preferred direction is stateless episode anchoring derived from
  exchange fills, plus separate broader accounting for no-restart semantics.
- Replace the implicit restart/no-restart contract with a scoped
  `restart_after_red_policy` enum. This gives users an explicit way to always
  resume after cooldown, threshold-halt as today, or permanently halt after any
  episode in which RED was active.
- Treat local checkpoints/cache as performance aids only. They may speed HSL
  replay and improve diagnostics, but they must not become authoritative panic,
  cooldown, or config-epoch state.
- Move live-only trading discretion toward Rust where feasible. Python should
  become thinner orchestration/reconciliation around Rust ideal orders, not a
  parallel risk-policy engine.
- Keep `pnls_max_lookback_days` as one coupled knob for now. The coupling is
  intentional enough to document and expose, but not yet worth schema
  expansion.

## Guiding Contracts

- Rust should be the source of truth for entries, closes, risk, HSL, and
  unstuck. Python should gather exchange data, build Rust payloads, unpack Rust
  ideal orders, and reconcile ideal vs actual exchange orders.
- Any HSL/risk/unstuck contract shared by live and backtest must be
  implemented in Rust and consumed by both paths, unless the implementing PR
  explicitly documents why centralization is infeasible or materially worse.
  Python-owned live HSL code is limited to exchange data gathering,
  cache/replay matrix construction, diagnostics, and reconciliation.
- Statelessness is required. Trading behavior must be reconstructable from
  exchange state plus config. Local caches/checkpoints may improve speed or
  diagnostics, but must not be authoritative trading state.
- Panic orders remain exempt from `max_realized_loss_pct`. Other loss-realizing
  closes should respect that gate.
- HSL, unstuck, max-realized-loss, and inherited fill-history behavior should be
  documented clearly enough that future audits do not treat intentional
  contracts as bugs.
- Local cache, checkpoint, or metadata files may speed replay or improve
  diagnostics, but must not become authoritative state. A fresh VPS with only
  exchange state plus config must reconstruct the same trading decisions.
- When current position health and reconstructed history disagree, the correct
  contract must be explicit per mechanism. Avoid hidden Python-side overrides
  that contradict Rust.

## Item Plan

### A1.1 - Unstuck Min-Qty Loss-Allowance Overshoot

Plan: retain current behavior and document it as intentional.

Contract: unstuck allowance is a pacing budget, not a hard per-order loss cap
after exchange min qty/cost snapping. If remaining unstuck allowance is small
but exchange min qty/cost forces a larger unstuck close, allow the close. The
self-healing mechanism is that further unstucking is blocked until realized
profits rebuild positive allowance.

Exception: `max_realized_loss_pct` still blocks non-panic lossy closes if the
order would exceed the configured realized-loss limit.

### A1.2 - Risk Gates Fail Open On Invalid Inputs

Plan: code fix.

Required risk inputs should be validated at the Rust/orchestrator boundary and
fail loudly. Do not rely on `log::error!` inside individual calculators as the
safety mechanism.

Implemented: Rust orchestrator core and PyO3 JSON entrypoints now reject
invalid account/risk globals before realized-loss or unstuck gates can
silently skip. Coverage includes non-positive raw balance and realized-PnL
peak/current inconsistencies.

### A1.3 - Pside HSL Side-Local Equity <= 0

Plan: code/design plus docs.

Recommendation: do not abort replay. Convert side-local equity `<= 0` into RED
semantics for the affected pside, then apply the explicit restart policy.

New config contract:

- `restart_after_red_policy=always`: ignore
  `hsl_no_restart_drawdown_threshold`; after panic/cooldown, restart the
  affected HSL scope even if reconstructed equity reached `<= 0`.
- `restart_after_red_policy=threshold`: current behavior. Restart after
  cooldown unless the no-restart threshold is breached.
- `restart_after_red_policy=never`: permanently halt the affected HSL scope
  after any episode in which RED was active (`red_seen_in_episode`).

Scope follows `live.hsl_signal_mode`:

- `unified`: policy applies to the whole account.
- `pside`: policy applies only to the affected pside.
- `coin`: policy applies only to the affected coin+pside.

This answers the previous open question: users who intentionally want continued
trading after side-local equity insolvency under pside HSL can choose
`restart_after_red_policy=always`.

### A1.4 - `hsl_ema_span_minutes` In `(0, 1)`

Plan: config formatting/validation fix.

Clamp `hsl_ema_span_minutes = max(1.0, hsl_ema_span_minutes)` during config
formatting, with visible warning/docs. Span `1.0` is equivalent to no EMA
smoothing. Apply the same principle to EMA spans generally where a sub-1 span
has no useful meaning.

### A1.5 - HSL Panic Order Type Per Pside

Plan: code fix.

Honor `config.bot.{pside}.hsl` panic order type independently in live and
backtest. Remove behavior where one side configured as market makes the other
side's HSL panic close market too.

### A1.6 - HSL Metrics

Plan: tests/investigation first.

Add metric-level regression tests for pside, unified, and coin HSL before
changing metric semantics.

Partial: `tests/test_hsl_metric_regression.py` now pins hand-computed
drawdown/tier ladders for all three signal modes, the pside scoped-signal and
baseline-minus-total-realized contracts, unified-mode `unrealized_pnl_total`
fail-loud requirement, the coin slot-budget denominator (exposure-knob
invariance, `n_positions` sensitivity, non-positive input rejection), red
latching across recovery, and the runtime's first-sample peak anchoring.

### A2.1 - Coin-Mode `hsl_no_restart_drawdown_threshold`

Plan: code/design plus docs.

Contract: replace the implicit no-restart-only behavior with
`restart_after_red_policy`, scoped by HSL signal mode.

- `always`: always restart the affected HSL scope after cooldown, regardless of
  `hsl_no_restart_drawdown_threshold`.
- `threshold`: close the affected scope on RED, then permanently halt that scope
  if `hsl_no_restart_drawdown_threshold` is breached. This preserves current
  threshold semantics, but makes the scope explicit.
- `never`: close the affected scope on RED and permanently halt that scope after
  any episode with `red_seen_in_episode` true.

Under the B2.1 contract, closing on RED is authorized by `red_active_now`
only; halt/restart decisions are evaluated per episode using
`red_seen_in_episode` once the scope fully flattens, and the no-restart
threshold uses `max(drawdown_raw, drawdown_ema)`.

Scopes:

- `unified`: close/halt/restart all trading.
- `pside`: close/halt/restart only the affected pside; continue the other pside
  if it is not halted.
- `coin`: close/halt/restart only the affected coin+pside; continue all other
  non-halted coin+psides.

### A2.2 - ORANGE `tp_only_with_active_entry_cancellation`

Plan: code/design plus docs.

Contract:

- `unified`: ORANGE blocks all entries, including initial entries.
- `pside`: ORANGE blocks all entries for that pside, including initial entries.
- `coin`: ORANGE blocks entries for that coin+pside, including initial entries.

Implemented: live ORANGE `tp_only_with_active_entry_cancellation` now blocks
flat initial entries in the affected scope instead of only symbols with an open
position.

### A2.3 - Bounded `we_excess` Invalid Base/TWEL

Plan: code fix.

Bounded mode should not degrade to raw semantics when base/TWEL is invalid.
Return zero for explicitly disabled/no-budget contexts and fail validation for
active configs with invalid limits.

Implemented: Rust bounded `we_excess` now returns zero allowed exposure for
non-positive/non-finite base WEL and zero excess headroom for
non-positive/non-finite TWEL instead of falling back to the raw excess
percentage. `legacy_raw` remains intentionally raw.

### A3.1 - Reducer Stacking

Plan: Rust-side code fix.

Only one panic/auto-reduce/auto-unstuck/other-close reducer should be emitted
per coin+pside per ideal-order batch.

Priority:

1. HSL panic. Panic orders may be market orders and must take absolute
   priority.
2. WEL/TWEL enforcer reducer, using existing deterministic priority if both
   could apply. These are expected to be emitted at bid/ask as limit orders.
3. Auto-unstuck. Despite the EMA gate, emitted unstuck orders should be at
   bid/ask; if the EMA gate prevents a bid/ask-safe candidate, no unstuck order
   should be scheduled.
4. Any other close order, lossy or profitable, trailing/grid/other. If a grid
   wants multiple closes, the existing close-order ordering should choose the
   one more likely to fill first.

Auto-unstuck remains one order per pside, but it should not stack on the same
coin+pside as a WEL/TWEL reducer or panic order in the same batch.

Reasoning: the "only one lossy order" policy is safe only when the competing
orders are all bid/ask-reachable. If one close is farther from market than
another, prioritizing it ahead of a bid/ask unstuck reducer can block the more
useful reduction.

### A3.2 - Snapped Vs Raw Balance

Plan: design/tests before code.

Recommendation: keep both raw and snapped balance, but define the balance basis
per mechanism and add near-boundary parity/oscillation tests. Avoid ad hoc
special cases. Snapped balance is appropriate where hysteresis is intentionally
part of sizing/gating; raw balance is appropriate where exact portfolio
exposure repair is intended.

Do not add a secondary raw-exposure cap for now. Add docs/tests around the
current snapped/raw separation, and emit a warning/preflight note when the
configured hysteresis/snap pct is high, for example above roughly 5%, because
unexpected boundary behavior becomes more likely.

### A3.3 - `reduce_overweight` Dynamic WEL

Plan: docs/test first.

Contract: `reduce_overweight` should use the dynamic currently-tradable slot
count, not only configured `n_positions`. Add tests that cover shrinking and
expanding currently-tradable universes.

### A3.4 - Coin-HSL Denominator

Plan: align with B1.2 design.

Contract direction: coin-HSL should keep slot-budget-style normalization and
should not use effective WEL (`eff_WEL`) or excess allowance to scale the RED
drawdown threshold. A 10% or 20% HSL RED threshold should remain that threshold
even if TWEL or excess allowance is raised.

Live formula: `slot_budget = current_raw_balance / configured_n_positions`.
Live should not use dynamic currently-tradable slots for HSL. If the live coin
universe changes, the operator is responsible for changing configured
`n_positions`.

Backtest exception: backtests may use
`effective_n_positions(t) = min(len(eligible_coins_at_t), configured_n_positions)`
when an explicit backtest/config option enables dynamic historical availability.
This is only for cases where early historical periods had fewer tradable coins
than the later configured universe.

Reasoning: many sizing/risk knobs reasonably scale with TWEL and excess
allowance, but HSL is a drawdown stop. Scaling the denominator with boosted
exposure can make the same configured RED threshold tolerate much deeper
account damage. The tradeoff is that high-TWEL configs will hit coin HSL sooner
than a pure "TWEL scales everything" mental model would imply. That is
acceptable because the alternative can make HSL less protective exactly when
gross exposure is higher.

Implementation should make this explicit in docs and tests. If the larger B1.2
canonical equity-history redesign changes the exact formula, the test contract
must still prove that excess allowance does not make HSL RED require a larger
percentage drawdown.

### A3.5 - `pnls_max_lookback_days`

Plan: retain one knob for now and document coupling.

`pnls_max_lookback_days` currently affects HSL memory, realized-loss gate
memory, unstuck allowance, and backtest PnL lookback. This is simple but
inflexible. Splitting knobs can be revisited later, but is not high priority.

### A4 - Entry Cooldown Semantics

Plan: simplify config surface.

Use `entry_cooldown_minutes` as the single entry-ladder/cooldown control.

Contract:

- If `entry_cooldown_minutes == 0.0` and
  `bot.{pside}.entry.retracement_base_pct <= 0.0`, cooldown is disabled and
  full simultaneous entry ladders may be emitted.
- If `entry_cooldown_minutes == 0.0` and
  `bot.{pside}.entry.retracement_base_pct > 0.0`, allow only one entry order on
  the book at a time because the next entry price depends on threshold and
  retracement state.
- If `entry_cooldown_minutes > 0.0`, stage at most one position-adding entry
  order and block any entry for coin+pside whose previous entry fill happened
  less than the configured duration ago, including fractional sub-minute
  durations.
- Backtests evaluate on one-minute steps, so any positive sub-minute cooldown
  prevents same-minute replacement/add and effectively waits until the next
  backtest decision minute. Live trading checks intra-minute and enforces the
  actual millisecond duration.

Keep float logic internally to support possible future sub-minute backtests.

### B1.1 - Live Omits TWEL Enforcer Policy

Plan: code fix.

Add `risk_twel_enforcer_policy` to the live Python -> Rust payload and add
BotParams coverage tests.

### B1.2 - Unified/Pside HSL Signal Construction Diverges

Plan: design and parity tests.

Define one canonical HSL signal construction for all modes, preferably in Rust
or with a Rust-owned parity contract first, then make live replay and backtest
match it. Add sample-parity fixtures for coin, pside, and unified.

Preferred design direction:

1. For every coin with fills inside the lookback window, split fills by pside
   and reconstruct psize/pprice per fill. A backward pass may be useful to
   recover position sizes, followed by a forward pass to recover pprice.
2. For the same coins, use candle closes to compute minute-by-minute UPnL.
3. Build per-coin+pside matrices with:
   `ts`, `price`, `psize`, `pprice`, `pnl`, and optional derived `upnl`.
   `pnl` is raw realized PnL inside the minute `(ts - 60_000, ts]`, not a
   cumulative value. `upnl` is deterministic from `price`, `psize`, `pprice`,
   contract multiplier, and side; it may be cached but can be recomputed.
4. Aggregate those matrices into:
   - `df_hsl_unified`: `ts`, `pnl`, `upnl`
   - `df_hsl_long_agg`: `ts`, `pnl`, `upnl`
   - `df_hsl_short_agg`: `ts`, `pnl`, `upnl`
   - `df_hsl_long_single[coin]`: `ts`, `pnl`, `upnl`
   - `df_hsl_short_single[coin]`: `ts`, `pnl`, `upnl`
5. Derive one account-level historical balance series from unified realized
   PnL after slicing the desired lookback window:
   `balance = pnl_cumsum - pnl_cumsum.iloc[-1] + current_balance`.
6. Compute HSL equity histories by mode:
   - unified: use reconstructed account equity:
     `equity = balance + unified_upnl`.
   - pside: use scoped realized PnL plus scoped UPnL anchored to the configured
     pside budget. The exact budget formula needs tests, but it should not
     scale RED sensitivity with excess allowance.
   - coin: use coin+pside realized PnL plus coin+pside UPnL anchored to the
     configured base slot budget: `current_raw_balance / configured_n_positions`
     in live. Backtest may substitute explicitly enabled dynamic historical
     effective n-positions.
7. Compute drawdown and smoothed drawdown from those equity histories using the
   industry-standard peak-equity denominator:
   `drawdown = (cummax(equity) - equity) / cummax(equity)`.

This would make all three HSL signal modes use equity history rather than
having coin mode use realized PnL only. It also provides a clearer persistent
data-store boundary for replay, doctor tools, and non-authoritative cache
resume.

Implementation notes / pushback:

- Equity `<= 0` still needs explicit RED/terminal handling before normal
  percentage math is trusted.
- Avoid formulas that accidentally apply `.cumsum()` twice to an already
  cumulative PnL series. The canonical persisted matrix should contain raw
  minute `pnl`; `pnl_cumsum` should be recomputed cheaply after slicing the
  desired lookback window.
- Do not use account-level balance plus scoped UPnL for coin/pside unless the
  contract intentionally changes to account-equity-relative HSL. Current
  direction is to preserve slot-budget-style sensitivity for scoped modes.
- Persisted DataFrames/checkpoints are acceptable only as performance caches.
  On a fresh VPS, exchange state plus config must still reconstruct the same
  trading state, even if slower.
- HSL replay may also feed dashboard/monitor views of performance over time,
  but dashboard convenience must not change the trading source-of-truth
  contract.

### B1.3 - Live-Only Unstuck Admission Logic

Plan: move authority to Rust and simplify Python.

Rust should emit ideal unstuck orders. Python should reconcile them through the
normal create/cancel pipeline and replacement tolerances. Remove bespoke Python
unstuck-specific order juggling where stronger general duplicate-order
guardrails make it unnecessary.

### B1.4 - Balance Hysteresis Parity

Plan: same direction as B1.3.

Move shared hysteresis policy to Rust where feasible. Document any intentional
live/backtest divergence that remains.

### B2.1 - Coin-HSL Boot Panic From Inherited Account History

Plan: redesign HSL around trading episodes.

New contract (clarified 2026-07-06):

- A trading episode starts when the relevant HSL scope goes from flat to
  non-flat.
- A trading episode ends at the fill timestamp that makes the full scoped HSL
  position set flat, by any means, not only by a bot-emitted panic order:
  - `unified`: all positions, all coins, both psides flat.
  - `pside`: all positions on that pside flat.
  - `coin`: that coin+pside flat.
  Episode end is explicitly not the RED breach timestamp, panic submit
  timestamp, panic order create timestamp, or panic initiation timestamp.
- Current panic eligibility is computed from the current episode.
- A separate broader drawdown tracker remains for
  `hsl_no_restart_drawdown_threshold`.

Split RED concepts:

- `red_active_now`: whether the current metric sample is in RED. Only this may
  authorize new panic orders.
- `red_seen_in_episode`: whether `red_active_now` was true at any point in the
  current episode. This affects post-episode cooldown/restart handling after
  the scope fully closes.
- If RED was active earlier in the episode but the current sample is no longer
  RED, the bot must not submit new panic orders solely because RED was seen
  earlier.

Drawdown trigger formulas:

- RED / panic-now: `min(drawdown_raw, drawdown_ema) >= red_threshold`.
  Rationale: EMA smoothing exists to avoid panicking on sharp V-shaped flash
  crashes. Users who want immediate raw sensitivity can set
  `hsl_ema_span_minutes = 1`.
- No-restart: `max(drawdown_raw, drawdown_ema) >=
  hsl_no_restart_drawdown_threshold`.
  Rationale: the permanent halt should be conservative and catch either
  catastrophic instantaneous damage or sustained smoothed damage.

Cooldown:

- Cooldown starts at the scoped episode end (the scope-flattening fill
  timestamp), not at RED breach and not at panic order submission.
- Cooldown is activated for an episode if `red_seen_in_episode` was true for
  that episode.

Incomplete panic / partial flattening:

- While the scoped positions remain non-flat, the episode remains active and
  keeps its current-episode drawdown tracker.
- If a panic partially flattens the scope and `red_active_now` later becomes
  false before the scope is fully flat, the bot must not keep submitting panic
  orders just because RED happened earlier in the episode.
- Remaining reduce-only orders should still be canceled when the relevant
  position/scope is flat, consistent with normal flat-position cleanup.

Incomplete history policy:

- For `restart_after_red_policy=always`, missing history before the current
  episode may warn/degrade but should not necessarily block startup if
  current-episode reconstruction is proven.
- For `threshold` or `never`, full configured lookback coverage is required for
  the affected scope, because no-restart depends on historical evidence.
- Missing current-episode fills/candles must block startup unless the user
  explicitly opts into a dangerous CLI override.
- Any such override must be per-run, loud, and documented as "HSL evidence
  incomplete; panic/cooldown/no-restart may be wrong."

Scopes:

- `coin`: coin+pside episode.
- `pside`: pside episode.
- `unified`: account-wide episode, reset when all positions are flat.

This should reduce surprising panics after config changes or inherited history
while preserving stateless reconstruction from exchange fills.

### B2.1b - HSL Replay Performance And Readiness Latency

Plan: performance redesign without changing the stateless trading contract.

HSL replay must become fast enough that startup does not leave the account
under-protected for many minutes. A 20-30 minute HSL reconstruction delay is
not acceptable for live safety. Correctness is still required, but the current
path likely has avoidable serialization, duplicated work, and cache underuse.

Recommended directions:

- Persist the canonical HSL time series described in B1.2:
  `df_hsl_unified`, `df_hsl_long_agg`, `df_hsl_short_agg`,
  `df_hsl_long_single`, and `df_hsl_short_single`, with rich metadata and
  resumable checkpoints.
- Persist raw per-minute source-derived facts, not cumulative HSL state:
  `ts`, `price`, `psize`, `pprice`, `pnl`, plus optional cached derived `upnl`.
  Do not persist `pnl_cumsum`, equity, drawdown, RED state, cooldown state, or
  no-restart state as authoritative values.
- Preferred first persisted format: versioned `.npz` arrays plus a JSON
  manifest/metadata file. This keeps dependencies limited to numpy and the
  standard library for the live-only install, avoids introducing Arrow/Parquet
  as a runtime dependency, and is fast enough to load on small VPSs. Pandas may
  still be used in memory or by doctor/export tools where already available.
- Add doctor tools that can recompute, verify integrity, and explain why a
  persisted HSL time series is reusable or invalid. Users should be able to run
  the doctor before booting a bot.
- On boot, validate whether the bot can resume from persisted HSL data. Validate
  at least:
  - schema/cache format version
  - exchange, market type, user/account, and config digest over HSL-relevant
    fields
  - HSL signal mode, pside settings, base slot-budget parameters, and
    lookback/cooldown parameters
  - symbol universe and pside coverage
  - fill coverage window, fill source/watermark or digest, and whether coverage
    proves the required lookback
  - candle coverage window, interval, repair/synthetic-gap metadata, and source
    freshness
  - matrix time bounds, row counts, and monotonic timestamp checks
  - current exchange position compatibility after replaying fills from the
    checkpoint watermark to now
  - current balance anchoring policy and timestamp
  Cache reuse should be allowed when the checkpoint can be extended safely from
  exchange state. It should be invalidated when a mismatch could change RED,
  cooldown, or no-restart decisions.
- Treat the initial costly HSL replay on a truly fresh account or fresh VPS as
  possibly unavoidable, but make all short-downtime restarts reuse proven local
  data where safe.
- Move the heavy computation toward vectorized numpy/pandas/numba or Rust where
  practical. Avoid Python primitive loops for dense minute-by-minute replay.
- Use local caches and rich checkpoints only as performance accelerators and
  diagnostics. They must never become authoritative trading state. A fresh VPS
  with only exchange state plus config must produce the same decisions, even if
  slower.
- For coin mode, prioritize held coin+psides first so currently exposed
  positions get HSL readiness before flat candidates. Flat scopes still matter
  for cooldown reconstruction, but they should not delay protection of held
  positions.
- Add event-stream and smoke-report timings for each replay phase so future
  latency regressions are visible.
- Parallelization is lower priority than cache reuse and vectorized/native
  computation because many live bots run on single-vCPU VPSs.
- First implementation placement: Python may own exchange data gathering,
  repair, and canonical matrix construction because live reconstruction is
  exchange-I/O-heavy and ambiguous. Rust should own threshold/tier/order
  decisions, and parity fixtures should prove that live matrices and backtest
  data produce equivalent HSL states. Move more computation into Rust later only
  where it reduces complexity or materially improves speed.

Pushback / implementation caution: do not skip timing instrumentation entirely.
It does not need to be a standalone "profiling-only" PR, but persisted replay
must emit enough timing/source/coverage evidence to prove that the checkpoint
actually speeds boot and does not hide correctness failures.

### B2.2 - Realized-Loss Gate Inherits Account History

Plan: retain current contract and document clearly.

`max_realized_loss_pct` considers all fill events inside the lookback window,
with no exception except panic orders. Users can reduce lookback or increase the
loss allowance if desired.

### B2.3 - Unstuck Allowance Inheritance

Plan: retain current contract and improve visibility.

Unstuck allowance considers all PnL from the configured lookback window. The
event stream should show the lookback window and allowance basis.

### B2.4 - Config Changes Retroactively Rescale HSL

Plan: retain stateless deterministic behavior and document the caveat.

Add a dedicated HSL/config-change risk document explaining how changing HSL
thresholds, TWEL, `n_positions`, signal mode, or enabling HSL on an account with
history can reinterpret exchange history.

### B2.5 - Signal-Mode Switches

Plan: retain statelessness.

Do not add authoritative local checkpoints, flags, or markers. All trading
behavior must remain reconstructable from exchange data and config. Episode
anchoring by flattening should reduce many signal-mode switch surprises.

### B2.6 - Deposits/Withdrawals Not Modeled In HSL Replay

Plan: docs and startup warning first.

Do not make the bot guess transfer intent. Add a must-read HSL risks/pitfalls
doc and print a startup warning when HSL is enabled.

### B2.7 - Flat Coins With Latched Drawdown Get Current-Time Cooldown

Plan: change with coin-HSL redesign.

Cooldown evidence should be reconstructed primarily from the canonical
HSL equity/drawdown time series plus fills, not from passivbot order-type
markers in fill events. If `min(drawdown_raw, drawdown_ema)` crossed
`red_threshold` inside a trading episode, that sets `red_seen_in_episode`;
when the scope later fully flattens, the scope-flattening fill timestamp (the
episode end) is the cooldown anchor, per the B2.1 contract. The RED breach
timestamp itself is evidence, not the anchor. This avoids dependence on
client-order-id/order-type history that some exchanges may omit or truncate.

This interacts with `config.live.hsl_position_during_cooldown_policy`:

- If the position is still open and `red_active_now` is true, Rust should
  emit the appropriate panic/protective action.
- If the scope is flat, `red_seen_in_episode` was true for the last episode,
  and the episode-end timestamp is inside `cooldown_minutes_after_red`, the
  affected HSL scope remains in cooldown and entries follow the configured
  cooldown-position policy.
- If reconstructed history cannot prove a RED crossing, do not invent cooldown
  from missing order markers. Warn and surface degraded cooldown confidence.

### B2.8 - Exchange-Dependent Replay Fragility For Panic Markers

Plan: reduce dependence on markers; keep diagnostics.

Do not add authoritative local panic markers/checkpoints. Prefer RED evidence
(`red_seen_in_episode`) and scoped episode-end timestamps derived from the
canonical equity/drawdown replay plus fills. Keep diagnostics for
exchanges where client-order-id history is incomplete, but those diagnostics
should explain marker confidence rather than decide cooldown by themselves.

### B2.9 - Trailing Anchor Fallback

Plan: derive from fill events.

The last fill timestamp for coin+pside should be derived from fill events. If
no fill event exists inside available runtime fill history, log a warning and
use the oldest available candle timestamp as trailing extrema anchor. Local
metadata/cache may speed lookup but must not be authoritative.

### B2.10 - Entry Cooldown At Boot

Plan: fill-history based, with explicit fallback.

If no entry fill event is found for coin+pside inside available history, use the
lookback/covered-start timestamp as the cooldown anchor and log it clearly.

Do not add a dedicated entry-cooldown history window for now. The bot may keep
more fill/candle data in memory or local caches than the configured trading
lookback, for example up to seven days, but trading decisions must only use the
allowed lookback contract.

### B3.1 - RED Supervisor Freezes Main Loop

Plan: move toward Rust-owned RED and normal reconciliation.

Rust should own the trading logic that blocks entries and emits panic/protective
ideal orders. Python should avoid a separate RED supervisor that freezes the
main loop if Rust can express the same behavior through ideal orders.

### B3.2 - Dead Legacy HSL In `passivbot.py`

Plan: cleanup after tests.

Delete the dead legacy HSL block after parity tests cover the active
`src/passivbot_hsl.py` path.

### B3.3 - HSL ORANGE Overrides Manual Forced Mode

Plan: design/docs.

Recommendation: forced/manual mode should override ORANGE, but RED should
override manual. If current behavior is retained, document it loudly.

### B3.4 - Residual Neutral Defaults On Risk Inputs

Plan: code hygiene.

Replace trading-risk neutral defaults with explicit errors once tests and stubs
provide required managers. Test-only no-manager paths should be isolated.

### C1 - `live.hsl_signal_mode` Defaults Differ

Status: implementation aligns live/backtest runtime paths and raw-config
diagnostics.

Plan: code fix.

Prefer fail-loud missing-key behavior in live/backtest after schema
normalization. The schema may hydrate default `coin`, but primary runtime paths
should not silently default raw missing config to a different mode.

### C2 - Coin Overrides Docs Vs Allowlist

Plan: docs and diagnostics.

Document exactly which risk/unstuck fields are overrideable and state that HSL
and forager are not generally per-coin overrideable. Unsupported override keys
should be visible to operators instead of silently dropped.

### C3 - HSL Panic Close Order Type Default Drift And String Validation

Plan: code fix.

Add Rust enum/string validation for HSL order type and orange mode. Align Rust,
schema, template, live, and backtest defaults.

### C4 - Template Short-Side HSL Defaults

Plan: docs/preflight.

Warn when enabling HSL from template-like aggressive short-side values. Decide
later whether dormant defaults should become safer or symmetric.

### C5 - HSL And Risk Numeric Validation

Plan: code/docs.

Clamp EMA spans to at least `1.0`. For other risk numerics, prefer fail-loud or
explicit documented normalization. Correct misleading CLI help for
`max_realized_loss_pct`.

### C6 - Entry Cooldown Docs

Plan: fold into A4 docs/redesign.

## Resolved Decisions And Remaining Implementation Details

Resolved enough to plan implementation:

- Scoped HSL should keep slot-budget-style normalization, not
  account-equity-relative normalization and not `eff_WEL` scaling.
- Live coin-HSL slot budget is `current_raw_balance / configured_n_positions`;
  backtest may use dynamic effective n-positions only behind an explicit
  historical-availability option.
- First persisted HSL cache format should be versioned `.npz` arrays plus JSON
  metadata.
- Persisted HSL matrices should store raw per-minute facts (`ts`, `price`,
  `psize`, `pprice`, `pnl`, optional derived `upnl`), not cumulative PnL,
  equity, drawdown, RED, cooldown, or no-restart state.
- Python may build canonical live replay matrices first; Rust should own
  threshold/tier/order decisions with parity tests.
- Hysteresis warning should stay simple: warn above roughly 5%.
- Cooldown activation is derived from `red_seen_in_episode` in the
  reconstructed HSL drawdown series rather than passivbot order-type markers
  where possible, and the cooldown anchor is the scoped episode-end fill
  timestamp, not the RED breach or panic submission time (B2.1/B2.7).
- RED / panic-now triggers on `min(drawdown_raw, drawdown_ema)`; no-restart
  triggers on `max(drawdown_raw, drawdown_ema)` (B2.1).

Remaining implementation details:

1. HSL checkpoint metadata versioning: the trust-boundary fields are listed in
   B2.1b, but the first implementation still needs exact manifest names,
   digests, and invalidation reason codes.
2. HSL RED edge cases, mostly resolved by the B2.1 contract: equity `<= 0`
   keeps `red_active_now` true while it persists; multiple RED crossings
   inside one episode collapse into a single `red_seen_in_episode`; candle or
   fill coverage starting after the configured lookback start follows the
   B2.1 incomplete-history policy per `restart_after_red_policy`. Still open:
   the exact flat-detection tolerance for the episode-end fill (qty-step
   epsilon versus exchange dust) and the CLI surface for the dangerous
   incomplete-evidence override.

## Recommended PR Order / Checklist

- [x] Live `risk_twel_enforcer_policy` plumbing and BotParams coverage tests.
      This is the cleanest confirmed live/backtest parity breach.
- [x] HSL per-pside panic order type plus Rust enum/default validation.
      Keep market/limit intent side-local unless an explicit emergency global
      override is added.
- [x] Numeric validation/clamping for HSL EMA spans and risk/unstuck config.
      Clamp EMA spans to at least `1.0`; fail loudly or normalize visibly for
      other risk numerics.
- [x] Risk-input fail-loud validation boundary.
      Invalid balance, position, exchange params, PnL stats, or active gate
      inputs must not disable risk gates permissively.
      Implemented: Rust core and JSON API validation reject invalid account/risk
      globals before realized-loss or unstuck gates can silently skip, including
      non-positive raw balance and realized-PnL peak/current inconsistencies.
- [x] Rust reducer priority: one reducer per coin+pside per ideal-order batch.
      Priority is HSL panic, WEL/TWEL reducer, auto-unstuck, then any other
      close order, with bid/ask-reachable reducers prioritized before farther
      passive closes.
      Partial: TWEL auto-reduce now removes same-position WEL and auto-unstuck
      reducers before insertion, with long/short JSON API regression coverage.
      Partial: auto-unstuck admission now skips same-position panic, TWEL, or
      WEL reducers already queued, with long/short WEL regression coverage.
      Partial: protective reducers now prune lower-priority same-position
      ordinary closes before size trimming; ordinary strategy-only close
      multiplicity is intentionally unchanged.
      Implemented: same-priority reducer pruning now has long/short regression
      coverage proving bid/ask-reachable reducers win before farther passive
      reducers.
- [x] Contract docs batch: unstuck min-qty overshoot, inherited lookbacks,
      HSL/config-change risks, statelessness, `pnls_max_lookback_days`, and
      HSL-enabled startup warning.
- [x] HSL restart policy enum.
      Add `restart_after_red_policy = always | threshold | never`, scoped by
      HSL signal mode, and migrate no-restart semantics onto that explicit
      surface.
      Implemented: Python live HSL replay/finalization and Rust backtest HSL
      finalization now use the explicit policy, with `threshold` as the
      behavior-preserving default and config/JSON validation for invalid
      values.
      Implemented (B2.1 contract, 2026-07-06 clarification): the no-restart
      trigger is now the shared Rust predicate
      `ehsl::no_restart_triggered(policy, drawdown_raw, drawdown_ema,
      threshold)` using `max(raw, ema)`, consumed by both the Rust backtest
      finalization (stop snapshots now carry the smoothed drawdown) and
      Python live finalize paths via PyO3, per the centralization guiding
      contract. The RED/panic-now tier score was verified to already use
      `min(raw, ema)` in the shared runtime and is pinned by regression
      tests rather than changed.
      Partial (B2.1 red split, groundwork): the shared Rust runtime now
      tracks `red_seen_in_episode` separately from the tier latch and every
      step reports `red_active_now` (whether the CURRENT sample crosses RED),
      exposed through the PyO3 runtime/stateless step and threaded into the
      live metrics dicts and hermetic test stubs. Behavior is unchanged in
      this slice: the tier latch still pins display red; the follow-up slice
      rewires panic authorization to consume `red_active_now` only and moves
      the cooldown anchor to the scoped episode-end fill timestamp.
      Implemented (panic authorization): live coin and unified/pside RED
      supervisors now refresh a sample each cycle and emit panic modes only
      while `red_active_now`; a recovered sample downgrades the scope to
      `tp_only_with_active_entry_cancellation` for the remainder of the
      episode and re-engages panic if RED re-activates. Backtest parity: the
      pside/coin mode overrides gate `TradingMode::Panic` on the runtime's
      latest `red_active_now` and fall back to tp-only. Live stop anchors
      already use the latest panic-fill timestamp (the flattening fill for
      bot-flattened episodes); the manual-flatten anchor refinement and the
      incomplete-history policy remain open.
      Implemented (episode-end anchor): all five live cooldown anchor sites
      (both supervisors, the check-path finalization, and both
      repanic-refresh paths) now anchor at the scope's latest fill of any
      type via `_equity_hard_stop_latest_flatten_fill_timestamp_ms`, falling
      back to the previous panic-fill anchor and then the caller fallback
      when no fill evidence exists in the window. Only the incomplete-history
      policy remains from the #1122 contract.
      Implemented (incomplete-history policy): HSL coverage assertions now
      accept an explicit allow_incomplete waiver that converts
      coverage-category failures into critical logs (corrupt pending/degraded
      PnL still hard-fails). Waivers are granted only by the per-run
      `live.hsl_accept_incomplete_history` CLI override (critical startup
      banner + per-use critical logs) or by `restart_after_red_policy=always`
      when the coin scope's current-episode start is provable from covered
      fills (reversed running-size reconstruction from the live position;
      ambiguity is unprovable). pside/unified scopes stay strict, which is
      conservative-compliant with the contract's "not necessarily block"
      wording. With this, every #1122 contract item is implemented.
      Implemented (A2.2 backtest parity): the backtest ORANGE `TpOnly`
      override now forces flat symbols too (`apply_orange_override` has-pos
      gate removed), matching the live overlay since #1098; the orchestrator's
      `TpOnly` generates no entries and no flat-symbol closes, so the forced
      mode is safe for flat scopes.
- [x] Dynamic WEL and snapped/raw balance docs/tests.
      Make `reduce_overweight` use dynamic currently-tradable slot count, keep
      snapped/raw balance separation, and add high-hysteresis warning/preflight
      visibility.
      Partial: Rust TWEL `reduce_overweight` now uses the dynamic effective
      slot count when selecting overweight positions, with Rust and
      orchestrator-JSON regression coverage. Existing raw-balance TWEL reducer
      and snapped-balance TWEL entry-gate tests remain in place. Preflight now
      reports `balance_hysteresis_snap_pct` and warns when it is invalid or
      above `0.05`; docs now spell out snapped-balance sizing/gating versus
      raw-balance exposure-repair surfaces.
      Implemented: JSON API coverage now pins both shrinking and expanding
      currently-tradable universes for `reduce_overweight`.
- [x] Entry cooldown design and migration plan.
      `entry_cooldown_minutes` is the sole ladder/cooldown control. Zero
      cooldown with entry retracement disabled may stage multiple
      position-adding orders. Any positive cooldown or enabled entry
      retracement limits staged adds to one order, and positive cooldowns
      enforce the exact post-fill cooldown window.
- [x] `live.hsl_signal_mode` runtime/default alignment.
      Runtime paths should require normalized `live.hsl_signal_mode`, while
      raw-config diagnostics should report the schema default `coin`.
- [ ] Canonical HSL equity-history signal design.
      Align coin, pside, and unified around one raw per-minute `pnl + upnl`
      data-store model, keep scoped modes on base slot-budget-style
      normalization, and add sample-parity tests before changing semantics.
      Partial: live HSL now has pure replay-matrix helpers that build
      non-authoritative raw rows (`ts`, `price`, `psize`, `pprice`, `pnl`,
      `upnl`) from authoritative candle/fill inputs, derive UPnL through the
      Rust PnL helpers, require contiguous one-minute rows, and recompute
      `pnl_cumsum`/equity from raw minute `pnl` instead of persisting cumulative
      state.
- [x] Coin-HSL episode anchoring redesign, including no-restart persistence and
      cooldown rules.
      Current panic eligibility should be based on the current trading episode;
      terminal no-restart accounting remains broader.
      Partial: live coin-HSL replay now resets current-episode realized-PnL
      and runtime drawdown state when fill replay proves a coin+pside flattened
      by an ordinary, non-panic close before a later re-entry. Panic-marker
      cooldown/no-restart handling is unchanged; broader terminal no-restart
      accounting from canonical RED timestamps remains future work.
      Partial: historical panic markers now require reconstructed confirmed RED
      tier/score, not raw-only drawdown. Raw RED pending remains diagnostic and
      does not reconstruct a cooldown/no-restart event by itself.
      Partial: replay cooldown/no-restart evidence is now canonical from
      reconstructed episodes. An episode with `red_seen_in_episode` that is
      flattened by an ordinary (non-panic) close fill latches cooldown at the
      flatten fill timestamp and evaluates `restart_after_red_policy` /
      no-restart via the shared Rust predicate at that stop, mirroring the
      confirmed-panic-marker path (source=red_episode_flatten in the
      reconstruction logs). Backtest parity is pre-existing, provided by the
      backtest's per-episode RED tier latch (step latch_red=true pins the
      stop path armed after the sample recovers until the episode resets),
      and is now pinned by Rust regression tests for both the pside and coin
      scopes (RED seen while open, sample recovered, ordinary flatten ->
      cooldown). Broader cross-episode no-restart peak accounting needs no
      further live work, verified 2026-07-08: the pside/unified live replay
      and forward finalize already maintain the persistent tracker via
      `_equity_hard_stop_record_no_restart_stop` (max of stop peaks across
      episodes), and for the coin scope the backtest's tracker is
      mathematically degenerate - the synthetic per-episode equity is
      normalized to peak 1.0 and never exceeds it, so
      `persistent_drawdown_raw` equals the at-stop drawdown ratio, which is
      exactly what the live coin replay and coin forward finalize evaluate.
      With that, every sub-item of this tracker entry is implemented.
- [ ] HSL replay performance/readiness slice.
      Persist verified non-authoritative HSL time series/checkpoints, add doctor
      tools, prioritize held scopes, keep timing/source evidence, and move dense
      computation away from Python primitive loops. First persisted format
      should be `.npz` arrays plus JSON metadata unless implementation evidence
      shows a better dependency-free option.
      Partial: pure live-HSL cache helpers now write the raw replay matrix as
      `hsl_replay_matrix.npz` plus `hsl_replay_manifest.json`, including schema
      version, fixed one-minute interval, row/time bounds, trust-boundary
      metadata, per-array dtype/shape/SHA-256, and explicit validation reason
      codes for reuse rejection. The cache remains unwired and
      non-authoritative.
      Partial: `cache-integrity-doctor` now discovers HSL replay manifests,
      runs the replay-cache validator, reports valid/invalid cache counts and
      validation reason counts under risk metadata, and emits warning issues
      for caches that must be rebuilt rather than reused.
      Partial: pure live-HSL cache helpers now include a fail-closed loader
      that validates manifest metadata and array hashes before returning copied
      manifest/array data for future replay reuse. The loader remains unwired
      and non-authoritative.
      Partial: raw replay arrays can now be converted to derived
      `pnl_cumsum`/`upnl`/equity arrays through a vectorized helper with the
      same continuity/value checks, reducing the need for Python row loops when
      cache reuse is later wired.
      Partial: coin-HSL replay lifecycle events now split startup timing into
      history-fetch, pre-replay, replay-loop, and total blocking elapsed fields
      so future cache-reuse work can prove which phase improved.
      Partial: live coin-HSL replay now persists write-only raw replay matrices
      for currently held coin+psides (`.npz` plus manifest, linear markets
      only) after a successful replay, keyed by the tested cache-dir/config
      digest helpers with fill/candle coverage metadata. Per-pair write
      failures warn and emit `hsl_replay_cache_write_failed` without touching
      replay results; successful writes emit `hsl_replay_cache_written`. The
      cache is still never read for trading decisions.
      Partial: replay cache manifests (schema v2) now record the pnls-manager
      fill-coverage proof at write time (`fill_history_scope`,
      `fill_coverage_proven`) via the canonical coverage-status check, with
      caller-provided fill events explicitly marked unproven. The future
      read/reuse slice must gate on a proven manifest plus a fresh coverage
      proof at load time.
      Partial: cache format (schema v3) now carries a manifest `series_kind`
      (`pair_matrix` | `account_pnl`), and live coin replay additionally
      persists the write-only account-level realized-PnL series (`ts`, raw
      minute `pnl`) alongside held-pair matrices, because per-minute slot
      budgets need account balance that pair rows cannot provide. Kind/field
      tampering is rejected (`fields_mismatch`/`series_kind_invalid`). The
      account series is only written together with at least one pair matrix
      and remains never read for trading decisions.
      Partial: a pure, unwired `_hsl_replay_timeline_rows_from_cache` now
      synthesizes coin-replay timeline rows from persisted pair+account
      arrays with fail-loud span/continuity/balance checks, and parity tests
      prove row equality against the authoritative
      `get_balance_equity_history` timeline plus identical per-sample coin
      metric sequences (across an orange tier transition) when the real coin
      initializer replays synthesized versus authoritative rows. This is the
      trust boundary the read/reuse slice must build on.
      Partial: pure, unwired watermark-extension helpers
      (`_hsl_replay_extend_pair_rows`/`_hsl_replay_extend_account_rows`) now
      extend cached pair/account arrays from the cache watermark to now using
      post-watermark extracted fills and candle closes, mirroring the
      authoritative position bookkeeping exactly. Slice-and-extend parity
      tests prove extension equals a full rebuild to 1e-12; fills inside the
      cached window, beyond the extension end, or from another pair are
      rejected fail-loud so double-counting rejects the cache instead of
      corrupting it.
      Partial: account-series manifests (schema v4) now persist the replay's
      panic flatten markers (raw exchange-derived facts: timestamp, minute,
      pside, symbol) so a future cache-fed replay does not lose in-window
      cooldown/no-restart evidence. Markers are validated fail-loud (grid
      alignment, series-span bounds, ascending order, account-kind only) and
      tamper-checked (missing/invalid/wrong-kind reasons).
      Partial: pside/unified startup replay now persists the same write-only
      cache as coin mode after a successful replay (held-pair matrices plus
      the account series; matrix collection in get_balance_equity_history is
      enabled for all three signal modes). The signal mode is part of the
      cache config digest, so caches never cross modes. The pside/unified
      reuse gate (synthesis + parity trust boundary for aggregate rows)
      remains future work.
      Implemented: live coin-mode startup now attempts cache reuse before the
      full replay. Gates: write-time proven coverage recorded in the manifest
      plus a fresh load-time pnls-manager coverage proof, config-digest
      identity, account/pair watermark agreement, gap extension from
      exchange fills/candles (panic fills inside the gap force full replay),
      and current-position reconciliation within qty-step tolerance. Any
      rejection or unexpected error falls back to the authoritative full
      replay; the completed event now reports `cache_reused` alongside phase
      timings. End-to-end test proves the cache-fed boot reaches state
      identical to the full replay with the full fetch provably skipped.
- [ ] Python simplification after Rust owns ideal protective orders and unstuck
      orders.
      Python should reconcile ideal vs actual orders, not re-decide trading
      policy where Rust can own it.
- [x] Dead legacy HSL cleanup after parity tests cover active
      `src/passivbot_hsl.py`.
      Implemented: removed shadowed legacy HSL method definitions from
      `src/passivbot.py`; active behavior is bound from `src/passivbot_hsl.py`
      with an AST regression guard.

## Non-Goals For First PRs

- Do not introduce authoritative local HSL state, panic markers, or config
  epochs.
- Do not mix the HSL episode redesign with unrelated logging-overhaul changes.
- Do not change panic exemption from `max_realized_loss_pct`.
- Do not broaden Python trading authority to paper over Rust contract gaps.
