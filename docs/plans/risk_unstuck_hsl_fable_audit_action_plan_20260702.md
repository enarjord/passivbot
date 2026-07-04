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
  permanently halts the affected scope after any RED breach.
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
  series as the primary source of RED timestamps. It should not depend on
  passivbot order-type markers in fill events where that can be avoided.

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
  RED breach.
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
  after any RED breach.

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
  any RED breach.

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

### A2.3 - Bounded `we_excess` Invalid Base/TWEL

Plan: code fix.

Bounded mode should not degrade to raw semantics when base/TWEL is invalid.
Return zero for explicitly disabled/no-budget contexts and fail validation for
active configs with invalid limits.

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

Plan: redesign config surface.

Introduce two concepts:

- `allow_simultaneous_grid_entries: bool`
- `entry_cooldown_minutes: float`

Contract:

- If `allow_simultaneous_grid_entries=true` and
  `bot.{pside}.entry.retracement_base_pct <= 0.0`, allow a ladder of entries on
  the book simultaneously.
- If `allow_simultaneous_grid_entries=false` or
  `bot.{pside}.entry.retracement_base_pct > 0.0`, allow only one entry order on
  the book at a time.
- If `entry_cooldown_minutes < 1.0`, no cooldown is enforced.
- If `entry_cooldown_minutes == 1.0`, do not allow the next entry inside the
  same minute as the previous filled entry.
- If `entry_cooldown_minutes > 1.0`, block any entry for coin+pside whose
  previous entry fill happened less than the configured duration ago.

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

New contract proposal:

- A trading episode starts when the relevant HSL scope goes from flat to
  non-flat.
- A trading episode resets when the scope fully flattens by any means, not only
  by a bot-emitted panic order.
- Current panic eligibility is computed from the current episode.
- A separate broader drawdown tracker remains for
  `hsl_no_restart_drawdown_threshold`.

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

Cooldown after RED should be reconstructed primarily from the canonical
HSL equity/drawdown time series, not from passivbot order-type markers in fill
events. If smoothed drawdown crossed RED inside the current trading episode,
the RED timestamp is the cooldown anchor. This avoids dependence on
client-order-id/order-type history that some exchanges may omit or truncate.

This interacts with `config.live.hsl_position_during_cooldown_policy`:

- If the position is still open and the current HSL state is RED, Rust should
  emit the appropriate panic/protective action.
- If the position is flat and the latest RED timestamp is inside
  `cooldown_minutes_after_red`, the affected HSL scope remains in cooldown and
  entries follow the configured cooldown-position policy.
- If reconstructed history cannot prove a RED crossing, do not invent cooldown
  from missing order markers. Warn and surface degraded cooldown confidence.

### B2.8 - Exchange-Dependent Replay Fragility For Panic Markers

Plan: reduce dependence on markers; keep diagnostics.

Do not add authoritative local panic markers/checkpoints. Prefer RED timestamps
derived from the canonical equity/drawdown replay. Keep diagnostics for
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
- Cooldown should be derived from canonical RED timestamps in the reconstructed
  HSL drawdown series rather than passivbot order-type markers where possible.

Remaining implementation details:

1. HSL checkpoint metadata versioning: the trust-boundary fields are listed in
   B2.1b, but the first implementation still needs exact manifest names,
   digests, and invalidation reason codes.
2. HSL RED timestamp edge cases: define how RED crossings are detected when
   equity is `<= 0`, when drawdown moves above and below RED multiple times
   inside one episode, and when historical candle/fill coverage starts after
   the configured lookback start.

## Recommended PR Order / Checklist

- [x] Live `risk_twel_enforcer_policy` plumbing and BotParams coverage tests.
      This is the cleanest confirmed live/backtest parity breach.
- [x] HSL per-pside panic order type plus Rust enum/default validation.
      Keep market/limit intent side-local unless an explicit emergency global
      override is added.
- [x] Numeric validation/clamping for HSL EMA spans and risk/unstuck config.
      Clamp EMA spans to at least `1.0`; fail loudly or normalize visibly for
      other risk numerics.
- [ ] Risk-input fail-loud validation boundary.
      Invalid balance, position, exchange params, PnL stats, or active gate
      inputs must not disable risk gates permissively.
      Partial: Rust orchestrator JSON now rejects invalid account/risk globals
      before realized-loss or unstuck gates can silently skip.
- [ ] Rust reducer priority: one reducer per coin+pside per ideal-order batch.
      Priority is HSL panic, WEL/TWEL reducer, auto-unstuck, then any other
      close order, with bid/ask-reachable reducers prioritized before farther
      passive closes.
- [ ] Contract docs batch: unstuck min-qty overshoot, inherited lookbacks,
      HSL/config-change risks, statelessness, `pnls_max_lookback_days`, and
      HSL-enabled startup warning.
- [ ] HSL restart policy enum.
      Add `restart_after_red_policy = always | threshold | never`, scoped by
      HSL signal mode, and migrate no-restart semantics onto that explicit
      surface.
- [ ] Dynamic WEL and snapped/raw balance docs/tests.
      Make `reduce_overweight` use dynamic currently-tradable slot count, keep
      snapped/raw balance separation, and add high-hysteresis warning/preflight
      visibility.
- [ ] Entry cooldown config split design and migration plan.
      Separate simultaneous ladder permission from time-based cooldown.
- [ ] Canonical HSL equity-history signal design.
      Align coin, pside, and unified around one raw per-minute `pnl + upnl`
      data-store model, keep scoped modes on base slot-budget-style
      normalization, and add sample-parity tests before changing semantics.
- [ ] Coin-HSL episode anchoring redesign, including no-restart persistence and
      cooldown rules.
      Current panic eligibility should be based on the current trading episode;
      terminal no-restart accounting remains broader.
- [ ] HSL replay performance/readiness slice.
      Persist verified non-authoritative HSL time series/checkpoints, add doctor
      tools, prioritize held scopes, keep timing/source evidence, and move dense
      computation away from Python primitive loops. First persisted format
      should be `.npz` arrays plus JSON metadata unless implementation evidence
      shows a better dependency-free option.
- [ ] Python simplification after Rust owns ideal protective orders and unstuck
      orders.
      Python should reconcile ideal vs actual orders, not re-decide trading
      policy where Rust can own it.
- [ ] Dead legacy HSL cleanup after parity tests cover active
      `src/passivbot_hsl.py`.

## Non-Goals For First PRs

- Do not introduce authoritative local HSL state, panic markers, or config
  epochs.
- Do not mix the HSL episode redesign with unrelated logging-overhaul changes.
- Do not change panic exemption from `max_realized_loss_pct`.
- Do not broaden Python trading authority to paper over Rust contract gaps.
