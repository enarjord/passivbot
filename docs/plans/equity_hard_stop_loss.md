# Equity Hard Stop Loss — Specification

**Date:** 2026-03-05
**Status:** Partially implemented (backtest + live runtime); metric revision planned

## 1. Objective

Add a per-side hard stop based on strategy P&L drawdown, with staged responses (GREEN/YELLOW/ORANGE/RED), cooldown restart support, and a no-restart latch when drawdown is too deep. The metric must be immune to BTC collateral FX noise.

## 2. Implementation Status

### Implemented (Rust — backtest)

- **Dedicated Rust module:** `passivbot-rust/src/equity_hard_stop_loss.rs`
  - `HardStopConfig`, `HardStopState`, `HardStopStep`, `HardStopTier` types
  - `step()` / `step_with_equity_peak()` — core state machine
  - `RollingPeakTracker` — monotone-deque rolling max
  - `span_samples()` — minutes-to-samples conversion (float, no rounding)
  - Configurable tier ratios with strict validation (`0 < yellow < orange < 1`)
  - RED latch (once triggered, stays RED until explicit reset)
  - Comprehensive unit tests (tier boundaries, latch, rolling peak, score formula)

- **Backtest integration:** `passivbot-rust/src/backtest.rs`
  - HSL state evaluated each step after `update_equities(k)`
  - ORANGE tier: mode override to `tp_only_with_active_entry_cancellation` or `graceful_stop`
  - RED tier: force panic, verify flat (2 consecutive confirmations), halt
  - Cooldown restart: reset state after configured minutes, resume trading
  - No-restart latch: if stop-time raw drawdown > `no_restart_threshold`, halt is permanent
  - Rolling equity peak bounded by `pnls_max_lookback_days`

- **Analysis metrics:** `passivbot-rust/src/types.rs`
  - `hard_stop_triggers` (u32), `hard_stop_total_loss_pct` (f64), `hard_stop_restarts` (u32)
  - Exposed to Python via `run_backtest()` in `python.rs`

- **Rolling PnL cumsum:** `passivbot-rust/src/backtest.rs`
  - `VecDeque<(step, pnl)>` windowed by `pnl_lookback_bars`
  - `effective_pnl_cumsum()` returns rolling or all-time values
  - All unstuck allowance calculations and orchestrator input use this

- **PyO3 runtime class:** `passivbot-rust/src/python.rs`
  - `EquityHardStopRuntime` — wraps state + rolling peak for live use
  - `EquityHardStopRollingPeak` — standalone rolling peak tracker
  - `apply_sample()` method returns full metrics dict per cycle
  - `equity_hard_stop_step_py()` — stateless single-step function

- **Config types:** `passivbot-rust/src/types.rs`
  - `EquityHardStopLossConfig` with all fields, defaults, validation
  - Parsed from Python dict in `python.rs`

- **Template config:** `configs/template.json`
  - `live.equity_hard_stop_loss` section with all fields and defaults

### Implemented (Python — live bot)

- **Live runtime:** `src/passivbot.py`
  - Config parsing and validation (`_parse_equity_hard_stop_loss_config`)
  - Startup latch check (`_equity_hard_stop_handle_startup_latch`)
  - EMA warm-start from reconstructed equity history (`_equity_hard_stop_initialize_from_history`)
  - Per-cycle evaluation (`_equity_hard_stop_check`)
  - ORANGE mode overlay (`_apply_equity_hard_stop_orange_overlay`)
  - RED supervisor loop with flat verification (`_equity_hard_stop_run_red_supervisor`)
  - Halt latch persistence (`_equity_hard_stop_write_latch`)
  - Cooldown-based auto-restart (`_equity_hard_stop_wait_for_cooldown`)
  - State reset on restart (`_equity_hard_stop_reset_state`)

- **Risk module change:** `passivbot-rust/src/risk.rs`
  - Per-position TWEL share cap on floor_exposure (non-HSL, included in Codex base)

### Current metric (equity-based — to be revised)

The current implementation uses raw USD equity as the HSL input:

```
equity = balance_raw + upnl_sum
drawdown_raw = max(0, 1 - equity / equity_peak)
drawdown_ema = ema(drawdown_raw)
drawdown_score = min(drawdown_raw, drawdown_ema)
```

**Problem:** When BTC is collateral, a BTC/USD price drop reduces USD-denominated equity even when trading performance is fine. A 25% BTC drop on a BTC-collateral account triggers HSL at threshold=0.25 with zero strategy losses. Same issue for mixed (e.g. 50/50 BTC+USD) collateral.

## 3. Planned: FX-Robust Strategy P&L Metric

Replace raw equity with a strategy P&L-based metric that isolates trading performance from collateral valuation changes.

### Core pipeline

```
delta_t = pnl_cumsum_t + upnl_t              # total P&L (realized + unrealized) at time t
peak_delta = rolling_max(delta_t)             # over lookback window
dd_abs = peak_delta - delta_t                 # absolute strategy drawdown
capital_base = max(balance_raw + peak_delta, eps)  # synthetic peak NAV
dd_raw = dd_abs / capital_base                # percentage drawdown
dd_ema = ema(dd_raw)                          # smoothed drawdown
dd_score = min(dd_raw, dd_ema)                # robust score (anti-flash-crash + anti-stale-EMA)
```

### Why this works

- `delta_t` captures only trading activity (realized closes + mark-to-market unrealized). BTC collateral price changes don't appear here.
- `balance_raw` as denominator scales drawdown sensitivity with current account size — a $5k loss matters more when the account is $20k (BTC dropped) than when it's $40k.
- The `min(raw, ema)` guard prevents both failure modes:
  - **Stale EMA after recovery:** raw drawdown drops to 0, min(0, stale_ema) = 0.
  - **Flash crash bottom exit:** raw spikes, EMA lags, min(spike, low_ema) stays low.

### Why not EMA on P&L level

Considered smoothing the P&L cumsum directly and computing drawdown from the smoothed series. This would eliminate the need for `min(raw, ema)`. However:

- Smoothed peak lags on recovery — after full P&L recovery, the smoothed series still shows residual drawdown ("stale recovery").
- The `min(raw, ema)` approach on drawdown handles recovery correctly — raw drawdown goes to 0 immediately.
- EMA on drawdown is better understood and already proven in the current implementation.

### Why not hysteresis-snapped balance

Considered using `balance_hysteresis` instead of `balance_raw`. Rejected:

- Hysteresis snap is designed for order sizing stability — deliberately stale within the 2% band.
- For a risk metric, the step-jumps when the snap threshold is crossed create artificial drawdown discontinuities.
- `balance_raw` provides continuous sensitivity appropriate for risk gating.

### Data sources

- **Live:** `delta_t` from fill-event reconstruction via `get_balance_equity_history()` (minute resolution). `pnl_cumsum` from fill events, `upnl` from candle closes × reconstructed positions. `balance_raw` from latest exchange fetch.
- **Backtest:** `delta_t` from `pnl_cumsum_running + upnl` computed each step. `balance_raw` from `self.balance.usd_total_balance`. Already available — the change is what gets fed to the state machine.

## 4. Planned: Per-Side HSL

### Motivation

Most passivbot configuration is already per-side (`config.bot.long.*`, `config.bot.short.*`). HSL should follow this pattern because:

- Users often run asymmetric configs (aggressive longs, conservative shorts).
- A long-side blowup during a crash shouldn't force-close profitable shorts that are *benefiting* from the crash.
- Per-side params integrate naturally into the optimizer (same location as other optimizable params).
- Account-level behavior is recovered trivially by setting identical params on both sides.

### Config location

Move from `config.live.equity_hard_stop_loss` to `config.bot.{pside}.equity_hard_stop_loss`:

```json
"bot": {
    "long": {
        "equity_hard_stop_loss": {
            "enabled": true,
            "threshold": 0.25,
            "ema_span_minutes": 60.0,
            "cooldown_minutes_after_red": 0.0,
            "no_restart_threshold": 1.0,
            "tier_ratios": { "yellow": 0.5, "orange": 0.75 },
            "orange_tier_mode": "tp_only_with_active_entry_cancellation"
        },
        ...
    },
    "short": {
        "equity_hard_stop_loss": {
            "enabled": false
        },
        ...
    }
}
```

### Per-side metric

Same pipeline as section 3, but with side-filtered inputs:

```
delta_long_t = pnl_cumsum_long_t + upnl_long_t
delta_short_t = pnl_cumsum_short_t + upnl_short_t
```

Each side runs its own independent state machine (peak, EMA, tier, latch).

**Denominator:** Always total account `balance_raw` (+ side's `peak_delta`), not "side-allocated capital." Capital allocation between sides is fluid in cross-margin — a side-local denominator would be misleading.

### Per-side RED action

When long-side RED triggers:
- Panic-close long positions only
- Block long entries
- Short side continues unaffected (unless its own HSL also triggers)

This is straightforward since passivbot already manages modes per-side.

### Account-level as special case

If user sets identical HSL params on both sides, each side monitors its own P&L independently. The side that deteriorates faster hits RED first. This is slightly different from true account-level (which would sum both sides), but in practice:

- If both sides lose simultaneously, both will trigger
- If one side is profitable and masks the other's losses, that's correct behavior — the profitable side *should* continue

### Why not per-symbol

Considered per-symbol breakers. Rejected for now:

- **Attribution is ambiguous.** Realized PnL from closing coin A may fund entries on coin B. Isolating "coin A's P&L" requires capital allocation tracking that passivbot doesn't do.
- **Overlaps with unstuck.** Existing unstuck mechanism already identifies and gradually closes the worst-performing position per side.
- **Signal-to-noise is poor.** Single-coin P&L is much more volatile than aggregate. Would need very different EMA spans/thresholds and would produce noisy triggers.
- **Config complexity explodes.** Per-coin overrides, default thresholds, exclusion lists.

Can be revisited if concrete use cases demand it.

## 5. Current Config (implemented)

Located at `config.live.equity_hard_stop_loss`:

```json
"equity_hard_stop_loss": {
    "enabled": false,
    "threshold": 0.25,
    "ema_span_minutes": 60.0,
    "cooldown_minutes_after_red": 0.0,
    "no_restart_threshold": 1.0,
    "tier_ratios": { "yellow": 0.5, "orange": 0.75 },
    "orange_tier_mode": "tp_only_with_active_entry_cancellation",
    "panic_close_order_type": "market"
}
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `enabled` | bool | `false` | Master switch |
| `threshold` | float | `0.25` | Drawdown level triggering RED (25% from peak) |
| `ema_span_minutes` | float | `60.0` | EMA span in minutes. Higher = more flash-crash resistant, slower reaction |
| `cooldown_minutes_after_red` | float | `0.0` | `0` = permanent halt. `>0` = auto-restart after N minutes |
| `no_restart_threshold` | float | `1.0` | If stop-time drawdown exceeds this, halt is permanent even with cooldown. Must be > `threshold` |
| `tier_ratios.yellow` | float | `0.5` | Fraction of threshold where YELLOW begins |
| `tier_ratios.orange` | float | `0.75` | Fraction of threshold where ORANGE begins |
| `orange_tier_mode` | string | `"tp_only..."` | Mode applied at ORANGE: `"tp_only_with_active_entry_cancellation"` or `"graceful_stop"` |
| `panic_close_order_type` | string | `"market"` | Order type for RED panic closes |

Validation: `0 < yellow < orange < 1`, `threshold > 0`, `ema_span_minutes > 0`, `threshold < no_restart_threshold <= 1.0`.

## 6. Tier Actions

GREEN/YELLOW/ORANGE can de-escalate. RED latches until explicit reset (cooldown restart or manual clear).

| Tier | Condition | Action |
|------|-----------|--------|
| GREEN | `score < yellow * threshold` | Normal operation |
| YELLOW | `score >= yellow * threshold` | Warning logs only |
| ORANGE | `score >= orange * threshold` | Override mode to `orange_tier_mode` for affected side |
| RED | `score >= threshold` | Panic close affected side, verify flat, halt |

### ORANGE semantics

- `tp_only_with_active_entry_cancellation` (default): cancel all entry orders, manage closes only. Most defensive.
- `graceful_stop`: block new initial entries, allow DCA for existing positions. Less defensive, may allow recovery through averaging.

### RED semantics

1. Set affected positions to panic mode
2. Cancel all non-panic open orders for affected side
3. Submit panic close orders (market by default)
4. Verify flat: require 2 consecutive refreshes confirming zero positions and no blocking orders
5. After flat confirmation, evaluate no-restart threshold:
   - If `stop_drawdown_raw > no_restart_threshold`: permanent halt
   - Else if `cooldown_minutes > 0`: schedule restart after cooldown
   - Else: permanent halt
6. Write halt latch file

## 7. Trading Mode: `tp_only_with_active_entry_cancellation`

| Aspect | Behavior |
|--------|----------|
| Entry generation | None |
| Entry cancellation | Active — cancel all existing entry orders |
| Close generation | Normal |
| Close cancellation | Normal |

Distinct from `tp_only` (which does not cancel existing entry orders) and `graceful_stop` (which allows DCA entries for existing positions).

## 8. Rolling Lookback

`pnls_max_lookback_days` bounds the lookback window for both equity peak tracking and PnL cumsum.

- **Equity peak:** Rolling max via monotone deque, expiring entries older than lookback window.
- **PnL cumsum:** Rolling window via `VecDeque<(step, pnl)>`, `effective_pnl_cumsum()` returns rolling or all-time values.
- Applied consistently in live (fill event history) and backtest (step-based window).

## 9. Halt Latch (live)

Path: `caches/equity_hard_stop/{exchange}/{user}.json`

Payload includes: trigger timestamp, equity, equity_peak, drawdown metrics, threshold, config snapshot.

Startup behavior:
- Latch exists + `cooldown == 0`: refuse to start, log message to delete file
- Latch exists + `cooldown > 0` + cooldown expired: delete file, start normally, reset state
- Latch exists + `cooldown > 0` + cooldown not expired: refuse to start, log remaining wait

## 10. Interaction with Existing Systems

| System | Interaction |
|--------|-------------|
| `max_realized_loss_pct` | Orthogonal. HSL monitors strategy drawdown; loss gate blocks individual closes. Panic closes bypass loss gate. |
| `auto_gs` | ORANGE overrides modes, taking priority. On de-escalation, normal mode assignment resumes. |
| `forced_mode_{long,short}` | ORANGE preserves `panic`. RED overrides everything for affected side. |
| Unstuck | Independent. Unstuck handles per-coin gradual loss-taking. HSL is side-level circuit breaker. |
| Optimizer | HSL params optimizable when located in `config.bot.{pside}.*`. `hard_stop_triggers` usable as penalty metric. |

## 11. Analysis Metrics (backtest)

| Metric | Type | Meaning |
|--------|------|---------|
| `hard_stop_triggers` | u32 | Times RED triggered |
| `hard_stop_total_loss_pct` | f64 | Cumulative panic-close loss as % of starting balance |
| `hard_stop_restarts` | u32 | Successful restarts after RED |

## 12. Logging

Use `[risk]` tag. Log on tier transitions only (no per-cycle spam).

- Tier change: `tier_prev`, `tier_new`, `dd_raw`, `dd_ema`, `dd_score`, `delta_t`, `peak_delta`, `balance_raw`
- RED progress: canceled count, panic submitted count, remaining positions, flat-confirmation streak

## 13. Test Plan

### Rust unit (existing)

- EMA span minutes-to-samples conversion (float precision)
- Tier boundary tests with configurable ratios
- `drawdown_score = min(raw, ema)` — flash crash dampening, stale EMA recovery
- RED latch persistence and reset
- Rolling peak tracker correctness
- Pre-existing RED latch preserved on first sample

### To add

- FX-robust metric: BTC collateral price drop produces zero strategy drawdown
- Per-side delta isolation: long losses don't affect short HSL score
- Rolling PnL cumsum window correctness (expiry, max tracking)
- Per-side RED: only affected side's positions are panic-closed

### Integration (to add)

- Backtest with HSL enabled produces correct triggers/restarts/loss metrics
- ORANGE mode override applies to correct side only
- Optimizer can include HSL params in bounds and penalize triggers

## 14. Open Questions

1. **Config migration path.** Current config is at `config.live.equity_hard_stop_loss`. Moving to `config.bot.{pside}.equity_hard_stop_loss` is a breaking change. Support both locations with deprecation warning? Or clean break since the feature hasn't shipped to users yet?

2. **Backtest metric input.** In backtest, should `balance_raw` in the denominator be the step-by-step balance (mirrors live, includes BTC collateral effects) or `starting_balance` (fixed constant, purely strategy-focused)? Step-by-step provides better live parity. Fixed constant removes all collateral influence from the metric.

3. **Per-side PnL tracking in backtest.** The backtest currently tracks `pnl_cumsum_running` as a single aggregate. Per-side HSL needs `pnl_cumsum_long` and `pnl_cumsum_short` separately, plus per-side `upnl`. The per-side upnl is already available from positions. Need to split the rolling PnL cumsum by side.

4. **Live equity reconstruction per-side.** `get_balance_equity_history()` currently reconstructs total equity. Needs to be extended to produce per-side P&L series (filter fills by side, compute per-side upnl from per-side positions). Fills already have side information.

5. **`panic_close_order_type` field.** Currently in config but the `panic_close_order_type` field on `EquityHardStopLossConfig` is never read in Rust (compiler warning). Needs to be either wired up or removed. In per-side config, this may not belong per-side — panic order type is arguably account-level.

6. **Separate margin/liquidation guard.** With the FX-robust metric, HSL no longer protects against pure collateral-driven margin stress (BTC drops 50%, no strategy losses, but liquidation distance shrinks). Should there be a separate, simpler margin guard (e.g. `if liq_distance_pct < threshold: force graceful_stop`)? This is independent of HSL but related.

7. **EMA warm-start with new metric.** The live bot currently warm-starts EMA by replaying equity history through the state machine. With the new metric, it needs to replay the *strategy delta* series instead. The reconstruction data is the same (fills + candles), but the computation changes.

## 15. Non-Goals

- Per-symbol HSL (overlaps unstuck, attribution noise, config complexity)
- Condition-based auto-resume predicates beyond fixed-minute cooldown
- Velocity/derivative confirmation filters

## 16. Future Extensions

1. **Condition-based restart:** Restart only when `dd_score < restart_ratio * threshold` for N consecutive bars, optionally combined with volatility normalization.
2. **Catastrophic bypass:** Raw (unsmoothed) check: `if dd_raw > threshold * 2: immediate RED`. Handles extreme single-candle events.
3. **Per-side independent cooldowns:** Different cooldown durations for long vs short.
