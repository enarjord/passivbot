# Equity Hard Stop Loss (EMA-Smoothed) - Consolidated Spec

**Date:** 2026-03-03  
**Status:** Draft (implementation-ready)  
**Scope:** Live bot + backtest

## 1. Objective

Add an account-level hard stop based on equity drawdown from peak (realized + unrealized), with staged responses and a terminal RED latch requiring explicit human acknowledgement before restart.

## 2. Config

Use nested config under `config.live.equity_hard_stop_loss`:

```json
"equity_hard_stop_loss": {
  "enabled": false,
  "threshold": 0.25,
  "ema_span_minutes": 60.0,
  "tier_ratios": {
    "yellow": 0.5,
    "orange": 0.75
  },
  "orange_tier_mode": "tp_only_with_active_entry_cancellation",
  "panic_close_order_type": "market"
}
```

- `enabled` (`bool`): master switch.
- `threshold` (`float`): RED trigger drawdown. Must be `> 0`.
- `ema_span_minutes` (`float`): smoothing span in minutes (not samples).
- `tier_ratios.yellow` (`float`): default `0.5`.
- `tier_ratios.orange` (`float`): default `0.75`.
- `orange_tier_mode` (`enum`):
  - `graceful_stop`
  - `tp_only_with_active_entry_cancellation`
- `panic_close_order_type` (`enum`): `market` (default), `limit_panic`.

Validation:
- `0 < yellow < orange < 1`.
- `threshold > 0`, `ema_span_minutes > 0`.

Backtest receives these values via `prep_backtest_args()` passthrough from `config.live`.

## 3. Metric and State Machine

Per sample:
- `equity = balance_raw + upnl_sum_strict`
- `equity_peak = max(equity_peak_prev, equity)` (rolling-window constrained, see section 7)
- `drawdown_raw = max(0, 1 - equity / max(equity_peak, eps))`
- `span_samples = ema_span_minutes / sample_minutes` (float, no rounding)
- `alpha = 2 / (span_samples + 1)`
- `drawdown_ema = alpha * drawdown_raw + (1 - alpha) * drawdown_ema_prev`
- `drawdown_score = min(drawdown_raw, drawdown_ema)`

Tier classification:
- GREEN: `< yellow * threshold`
- YELLOW: `>= yellow * threshold`
- ORANGE: `>= orange * threshold`
- RED: `>= threshold`

`drawdown_score = min(raw, ema)` is retained to avoid stale-EMA RED triggers after recovery while filtering crash spikes.

## 4. Trading Mode Semantics

Current modes:
- `normal`: actively manage entries and closes; allows initial entries.
- `graceful_stop`: same as normal, but blocks initial entries.
- `manual`: neither cancel nor create.
- `tp_only`: actively manage closes; entries are manual (neither cancel nor create).
- `panic`: cancel all open orders; submit one full close.

New mode:
- `tp_only_with_active_entry_cancellation`:
  - actively manage close orders;
  - do not create entry orders;
  - actively cancel all existing entry orders.

Implementation detail:
- Keep existing `tp_only` behavior unchanged.
- Implement the new mode explicitly in live mode-filter/cancel policy and in backtest mode mapping.

## 5. Tier Actions

GREEN/YELLOW/ORANGE can de-escalate.  
RED is terminal and latched.

### ORANGE

Behavior selected by `orange_tier_mode`:
- `graceful_stop`: force `normal -> graceful_stop` globally; preserve stricter modes (`manual`, `tp_only`, `panic`, new mode).
- `tp_only_with_active_entry_cancellation`: force tradable sides with positions into this mode (or stricter). This is a stronger risk-off profile.

### RED

Strict sequence:
1. Persist latch file atomically.
2. Enter RED supervisory loop (do not exit main loop immediately).
3. Force panic mode for all non-zero positions.
4. Cancel all non-panic open orders.
5. Submit panic close orders (`market` by default; if rejected, retry with supported fallback and continue loop).
6. Re-fetch state and repeat until confirmed flat.

Flat confirmation rule ("beyond doubt"):
- Require **two consecutive successful refreshes** where:
  - all position sizes are zero;
  - no open entry orders remain;
  - no open non-panic close orders remain.

After confirmation:
- set `stop_signal_received = True`;
- exit trading loop without triggering auto-restart path.

## 6. Architecture

### 6.1 Rust (source of truth)

Add shared risk module used by live and backtest:
- `EquityHardStopConfig`
- `EquityHardStopState`
- `step(state, equity, timestamp_ms)`
- rolling-peak window support (section 7)
- tier evaluation using configurable ratios

### 6.2 Live (`src/passivbot.py`)

Startup:
- if disabled: no-op.
- if enabled and latch exists: log critical, set `stop_signal_received=True`, return from `start_bot()` (no generic exception).
- initialize hard-stop state from reconstructed history (section 7); fail loud on required-data failure.

Per cycle:
- evaluate after successful `update_pos_oos_pnls_ohlcvs()`, before normal order planning.
- use strict risk-input path (no silent `0.0` substitutions).
- apply ORANGE/RED overlays.

RED:
- run dedicated RED supervisory loop until flat confirmation.
- avoid generic exception path in `run_execution_loop()` (would trigger restart logic).

### 6.3 Backtest (`passivbot-rust/src/backtest.rs`)

- add hard-stop config/state.
- evaluate each step after `update_equities(k)`.
- apply ORANGE mode semantics.
- RED latches; force panic flatten logic; halt further trading after flat confirmation.

## 7. Equity Reconstruction, Rolling Lookback, and Cache

`pnls_max_lookback_days` must be applied consistently in live and backtest.

Policy:
- Peak reference is rolling over `pnls_max_lookback_days` (not all-time).
- backtest currently behaves as effectively infinite lookback; update to rolling-window behavior matching live.

Reconstruction sources:
1. Primary: canonical fill-event replay (authoritative for balance and position transitions).
2. Supplementary: minute-level mark-to-market reconstruction:
   - derive historical position size/price from fills;
   - apply candle closes to reconstruct minute uPnL;
   - build minute equity series between fills.
3. Local equity sample cache (non-authoritative, for precision/debug/QA):
   - write periodic live samples (e.g. 1m) to `caches/equity_samples/{exchange}/{user}.jsonl` (or parquet);
   - include `timestamp`, `balance_raw`, `upnl`, `equity`, position summary.

Conflict rule:
- replay-derived data is authoritative;
- cache is additive for interpolation/debug and restart diagnostics.

## 8. Halt Latch

Path:
- `caches/equity_hard_stop/{exchange}/{user}.json`

Required payload:
- `triggered_at`, `exchange`, `user`
- `threshold`, `ema_span_minutes`, `tier_ratios`, `orange_tier_mode`
- `equity`, `equity_peak`, `drawdown_raw`, `drawdown_ema`, `drawdown_score`
- `panic_close_order_type`, `tier=red`

Behavior:
- if latch exists and feature enabled: block startup until manual clear.

## 9. Interaction with Existing Risk Controls

- `max_realized_loss_pct` stays independent.
- panic close orders must bypass realized-loss gate.
- hard stop is account-level supervisory risk layer.

## 10. Logging

Use `[risk]` tag.

Log on tier transition and RED actions:
- `tier_prev`, `tier_new`, `equity`, `equity_peak`, `drawdown_raw`, `drawdown_ema`, `drawdown_score`
- `threshold`, `tier_ratios`, `orange_tier_mode`
- RED loop progress: canceled count, panic submitted count, remaining non-flat symbols, flat-confirmation streak.

No per-cycle spam outside transitions/RED progress checkpoints.

## 11. Test Plan

Rust unit:
- minute->samples conversion correctness (float span, no rounding).
- tier-ratio boundary tests with custom ratios.
- V-crash/recovery (`min(raw, ema)` guard).
- rolling lookback peak logic.

Live:
- nested config parsing and validation errors.
- ORANGE mode switch: both `graceful_stop` and `tp_only_with_active_entry_cancellation`.
- RED loop persists until two consecutive flat confirmations.
- market panic path + fallback behavior when market close rejected.
- latch blocks startup without entering restart loop.

Backtest:
- parity with live for tier transitions on same equity path.
- `pnls_max_lookback_days` rolling behavior parity.
- RED flatten + halt behavior parity.

## 12. Non-goals

- cooldown/auto-resume.
- per-side thresholds.
- derivative/velocity confirmation filter.
