# Equity Hard Stop Loss contract

Rust owns one v9 forward risk core across live, CPU backtests and supported GPU paths.
Python acquires immutable facts and enforces execution admission. The
[numerical contract](../../v9_risk_model.md) freezes arithmetic, sampling and bootstrap;
the [user guide](../../equity_hard_stop_loss.md) covers configuration.

## Signal authority

- `coin` is one symbol/position-side scope; `pside` is every position on one side;
  `portfolio` has one explicit `bot.hsl` policy across both sides.
- Slow decaying currency drawdown uses scoped UPNL deltas and ordered net realized
  cashflows, including fees and signed rebates. Balance/slot changes are budget facts,
  never scoped trading PnL. The independent fast EMA smooths the contemporaneous ratio,
  rather than currency drawdown normalized against a later budget.
- Current RED is strict `min(raw_ratio, ratio_ema) > red_threshold`. Finite nonpositive
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

Coin budgets divide raw balance by configured applicable slots. Flat zero-slot scopes
are inactive; an enabled coin scope holding inventory with zero slots fails explicitly.
Aggregate budgets use raw balance. Coin-only excess-allowance scaling uses
configured WEL/TWEL headroom through the same bounded entry-sizing policy. Actual
exposure never sets that multiplier. Reject aggregate scaling and per-coin overrides
of its side-global switch. Historical ratios retain observed contemporaneous budgets;
missing historical budgets require an explicit estimate diagnostic.

The shared slow span is `live.lookback_ema_span_minutes`. Bootstrap targets five slow
spans, capped by `live.risk_bootstrap_max_lookback_minutes` (at most 90 days), and retains
the full finite HSL cooldown horizon. Live policies whose cooldown exceeds the supported
cap fail explicitly. Acquisition coverage is distinct from exponential memory.
Historical candle gaps/coarse projections follow the numerical contract and disclose
approximations. An empty flat pair needs no invented mark or contract multiplier.

Fresh account balance, complete positions, orders and acted-on quotes remain mandatory
before writes. Historical uncertainty does not create a second fallback controller or
preserve old permission. Keep protection-first scheduling, shutdown checks, plan receipts,
bounded position/fill settling and connector admission. Ordinary strategy, auto-unstuck
and the independent realized-loss gate retain their own readiness requirements.
Freeze each candle acquisition's requested symbol cohort when scheduling it. Later
position or fill refreshes cannot remove a source from that in-flight request.

## Configuration and validation

V8 configs require explicit versioned migration, including CLI mutations, overrides,
scenarios, seeds and optimizer bounds. Mode `unified`, rolling-day memory and restart
selectors are migration inputs only. `portfolio` policy is never inferred from a side.
Removed `never` requires acknowledgement. New memory span/allowance semantics require
retuning; migrated optimizer fitness and old checkpoints are invalid evidence.

CPU/GPU replay retains constant risk state per scope and never scans historical candles
after a fill or query. Live startup/correction may replay factual history. Disposable
cache loss/rebuild cannot change intent for identical facts and initialization.
The live factual owner retains its immutable initial observation and later canonical
facts independently of bounded acquisition windows. Rust appends new economic events;
corrected prefixes or policy changes rebuild the derived fold from those same facts.
Losing the factual initializer is an explicit new unknown-prefix bootstrap. Account
reconciliation timestamps cannot precede any mark consumed by that reconciliation.

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
- `passivbot-rust/src/backtest_hsl_{analysis,report}.rs`: observational reporting.
- `src/live/hsl_*`: capture, acquisition, admission and diagnostics.
- `src/live/hsl_continuity.py`: immutable factual initialization ownership.
- `src/config/hsl.py`, `src/config/risk_memory.py`: canonical validation.
