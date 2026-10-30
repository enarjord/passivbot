# Equity Hard Stop Loss contract

Rust owns one v9 forward risk core across live, CPU backtests and supported GPU paths.
Python acquires immutable facts and enforces execution admission. The
[numerical contract](../../v9_risk_model.md) freezes arithmetic, sampling and bootstrap;
the [user guide](../../equity_hard_stop_loss.md) covers configuration.

## Signal authority

- `coin` is one symbol/position-side scope; `position_side` is every position on one side;
  `portfolio` has one explicit `bot.portfolio.hsl` policy across both sides.
- Slow decaying currency drawdown uses scoped UPNL deltas and ordered net realized
  cashflows, including fees and signed rebates. Balance/slot changes are budget facts,
  never scoped trading PnL. The independent fast EMA smooths the contemporaneous ratio,
  rather than currency drawdown normalized against a later budget.
- Current RED is strict `min(raw_ratio, ratio_ema) > drawdown_threshold_fraction`. Finite nonpositive
  equity or denominator is immediate RED with diagnostic ratio 1. Nonfinite or
  overflowing required inputs are errors. GREEN retires unfinished panic intent.
- Completed minute marks, genuine fill boundaries, observed budget changes and factual
  account reconciliations commit economic samples.
  Completed marks precede next-bar executions at equal timestamps. A current quote
  previews cloned state and cannot become a later D/EMA baseline. Repeated queries
  never reapply fees or decay. Terminal completed marks remain observations even
  when their equity triggers liquidation; a depleted fill does not consume a future close.
- The new risk arithmetic follows the explicit binary32/FMA graph. Inventory and
  fee/PnL accounting retain their existing precision and conservation contracts.

## Episodes and finite cooldown

A proven flat-to-exposed transition initializes drawdown and fast EMA to zero before
entry economics. Partial closes and aggregate constituent changes retain the episode.
Flatness means zero gross inventory, including a hedged portfolio.

A final close qualifies when its latest causal pre-fill assessment OR its post-fill
accounting assessment is RED. The integer deadline is flatten time plus the configured
finite duration. Manual closes qualify identically. Later GREEN, balance changes and
flat queries cannot cancel, renew or recreate that event. Corrected facts require
reconstruction. Zero duration permits immediate restart. Actual new exposure clears
an old terminal record. No `never` policy or local permission journal exists.

Rust's reconciler owns missing or ambiguous historical facts and reports its estimates.
With an unknown exposed prefix, observed current size/basis/mark/budget initialize an
explicit approximate seed with M=R, avoiding a new fast-EMA warmup that conceals current
loss. This is not proof of an infinite history. Missing current facts remain unavailable.
A delayed or missing final fill may use the documented causal timestamp estimate;
without retained causal fill evidence it cannot invent a historical cooldown.

## Budgets, history and admission

Current HSL observations expose `signal_raw_fraction = D / (E + D)` and its fast
`signal_ema_fraction`; `trigger_signal_fraction` is their minimum. `D` is the USD drawdown
memory and `E` is the applicable scope budget plus unrealized PnL. The explicit insolvency
diagnostic remains `1.0`. The native risk observation names its currency amount
`drawdown_memory_usd`; the private realized-loss memory is a separate accounting surface.
`drawdown_threshold_fraction` is the existing configured threshold.
Detailed report summaries use `signal_raw_max_fraction`, `signal_ema_max_fraction` and
`panic_fill_loss_sum_usd`, the loss-only sum of net panic-fill PnL.
Current smoke reports expose `red_threshold_gap_fraction = threshold - trigger_signal`:
positive is below RED and negative is beyond the threshold. `red_threshold_proximity_percent`
is `100 * trigger_signal / threshold`; `ema_red_threshold_gap_percent` is
`100 * (threshold - signal_ema) / threshold`. These are scale-100 observations with unchanged
rounding, not scale-one fractions. Historical top-level event-tape keys remain isolated readers.

Coin budgets divide raw balance by configured applicable slots. Flat zero-slot scopes
are inactive; an enabled coin scope holding inventory with zero slots fails explicitly.
Aggregate budgets use raw balance. Coin-only excess-allowance scaling uses
configured WEL/TWEL headroom through the same bounded entry-sizing policy. Actual
exposure never sets that multiplier. Reject aggregate scaling; coin policies may override the scaling switch. Historical ratios retain observed contemporaneous budgets;
missing historical budgets require an explicit estimate diagnostic.

Each effective HSL policy uses its own `memory_ema_span_minutes` and independent
`drawdown_fraction_ema_span_minutes`. Bootstrap targets five memory spans, capped by `live.startup.risk_bootstrap_max_lookback_minutes` (at most 90 days), and retains
the full finite HSL cooldown horizon. Live policies whose cooldown exceeds the supported
cap fail explicitly. Acquisition coverage is distinct from exponential memory.
Historical candle gaps/coarse projections follow the numerical contract and disclose
approximations. An empty flat pair needs no invented mark or contract multiplier.

Fresh account balance, complete positions, orders and acted-on quotes remain mandatory
before writes. Historical uncertainty does not create a second fallback controller or
preserve old permission. Keep protection-first scheduling, shutdown checks, plan receipts,
bounded position/fill settling and connector admission. Routine account-change reporting
runs after protection in one owned background task with one coalesced pending receipt;
slow or failed diagnostics cannot become a protective prerequisite. Ordinary strategy, auto-unstuck
and the realized-loss gate retain their own readiness requirements. Unstuck and
the gate own independent account-wide net-realized drawdown memories and spans under
`bot.portfolio.unstuck` and `bot.portfolio.loss_gate`; the gate's allowance uses its fraction directly,
without TWEL. Both read one canonical tape, each enabled memory consumes each actual net
fill once even when HSL is disabled, and the ordinary ledger debits it once. A bounded authoritative
suffix starts from an explicitly unknown-prefix zero initializer unless a known
origin is factually proven. Attenuation is a disclosed estimate, not exact-history
or safety proof; losing the initializer declares a new bounded bootstrap.
Freeze each candle acquisition's requested symbol cohort when scheduling it. Later
position or fill refreshes cannot remove a source from that in-flight request.
Ordinary preparation retains the actual immutable HSL candle-source and fill receipts,
including for empty plans. A replacement receipt invalidates preparation and completed-plan
acceptance after protection, even when its payload is equal. Quotes refreshed by preparation
do not invalidate that receipt. These references belong only to the outstanding plan.
The offline fake runner drains unmet account confirmations, owner-required fill readiness and
ordinary execution changes before advancing a bar. After a typed create/cancel connector deferral
or an ordinary account-confirmation request, a fresh accepted ordinary plan must complete without
another write or deferral, with its real planning account generation still current. Protective
plans cannot clear that receipt. An accepted empty or policy-filtered nonwriting plan can settle
when its generation is current. Each pass uses the normal owner and fresh planning. The eight-pass
readiness budget permits one additional eight-pass drain after actual ordinary execution progress;
generation changes alone do not extend it. Exhaustion is explicit and cannot certify a completed
bar. Optional fill debt does not create a new readiness requirement.

## Configuration and validation

Normal backtest reporting retains native typed plot observations while exposed or in
cooldown, plus the first inactive boundary. This is independent of verbose diagnostic
dictionary retention and never supplies trading inputs or metric support. Preserve
native ordering at equal timestamps, immediate insolvency flags, and all lifecycle
events. Report-only gaps indicate omitted idle periods, not fabricated observations.
Executed-episode records use the same native owner, loss-only net cashflows and causal
account-balance anchor as the core metric accumulator. Metrics-only optimizer runs
retain neither plot observations nor verbose rows.

Released-v8 configs require explicit migration to config schema v9.0.0, including CLI mutations, overrides,
scenarios, seeds and optimizer bounds. Mode `unified`, rolling-day memory and restart
selectors and the separate realized-loss gate window are migration inputs only.
Active gate contexts require explicit acknowledgment of decaying memory with independent spans.
`portfolio` policy is never inferred from a side.
Removed `never` requires acknowledgement. The offline converter's explicit `--best-effort`
mode may supply these semantic choices and acknowledgements with per-choice warnings
and report entries; ordinary loading remains strict. See [migration policy](../../v8_to_v9_migration.md).
New memory span/allowance semantics require
retuning; migrated optimizer fitness and old checkpoints are invalid evidence.

CPU/GPU replay retains constant risk state per scope and never scans historical candles
after a fill or query. Live startup/correction may replay factual history. Disposable
derived-cache rebuild reproduces intent from the same retained boundary and ordered facts.
Bound factual history using the acquisition horizon, full supported cooldown and at least
one minute. Once history exceeds twice this horizon, checkpoint just before the completed
minute at `now - horizon`; retain compact D/M, clocks, lifecycle/deadline, causal inventory
and marks, the later factual suffix and one predecessor budget. Discard older raw grids,
steps, initializer duplicates and diagnostic event vectors. Routine pruning never resets
remembered drawdown or renews/cancels a known cooldown. A time horizon bounds age, not bytes
under arbitrary event rates.

Expire dormant owners and their budget maps after their horizon. Partial capture absence
is not proof of flatness. Restart, owner loss, incompatible policy or a correction requiring
discarded support uses explicit bounded approximate reconstruction, which may shift timing
in either direction. Exact unknown historical RED qualification cannot be promised after a
cold restart. Supported factual corrections and intentional policy changes may change a
deadline; routine pruning and cache rebuilding may not. Keep no persisted panic latch.
Corrections inside support rebuild exactly from the compact boundary. Native diagnostics
report initializer status/clocks, retained support, boundary, horizon and reconstruction
reason; cache counts cover facts, steps, event vectors and owners. Do not mark a no-ledger
unavailable/inactive result as initialized. Account reconciliation cannot use a later mark
as an earlier causal valuation.

Backtest stop counts describe observed current permissions and same-capture lifecycle
transitions. A newly reconstructed historical breach cannot add a stop or restart when
the current permission is GREEN. Same-time execution sequences remain observable even
when their elapsed RED duration is zero.

Require independent arithmetic fixtures, Rust/property tests, source-verified native
callers, real fake-live planning/reconciliation, history corrections, restart cuts,
pre-RED/post-GREEN flattening, cooldown boundaries and HSL-disabled controls. Supported
GPU numerical acceptance requires actual Metal/CUDA execution; absent hardware is missing
evidence. V8 is an economic/performance control, not the v9 correctness oracle. Account
return/drawdown and recovery metrics remain independent of the HSL signal. Useful-search
acceptance includes required CPU confirmation, held-out scenarios and durable export.
Live trials and merges need separate authorization.

## Code map

- `passivbot-rust/src/hsl_decay.rs`: shared forward arithmetic and lifecycle.
- `passivbot-rust/src/hsl_v9_live.rs`: factual reconstruction and live native API.
- `passivbot-rust/src/hsl_v9_continuity.rs`: retained facts, forward fold and correction rebuild.
- `passivbot-rust/src/hsl_{history,snapshot,prices,support,intent}.rs`: factual support and envelopes.
- `passivbot-rust/src/backtest_hsl_runtime.rs`: forward CPU integration.
- `passivbot-rust/src/backtest_hsl_report.rs`: detailed observational reporting and core episode facts.
- `passivbot-rust/src/backtest_core_metrics.rs`: causal selected performance observations.
- `src/live/hsl_*`: capture, acquisition, admission and diagnostics.
- `src/live/hsl_continuity.py`: bounded factual ownership and transactional initialization.
- `src/config/hsl.py`, `src/config/risk_memory.py`: canonical validation.
