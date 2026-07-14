# VPS3 ebybitsub03 Wrong HSL Panic Incident

Status: investigation report for HSL replay safety fixes
Incident account: `ebybitsub03` on `vps3`
Exchange: Bybit
Observed command: `passivbot live -u ebybitsub03 configs/xmr_migrated.json -bo 1000`
Observed VPS3 head: `v8` at `b972f0221fb5121330a7ee3a41b99c761317544c`
Current fixed `origin/v8` head inspected: `b207bb42a2bd4b8b0754775278bc50587be869ac`

## Summary

The XMR panic close was emitted because HSL `signal_mode=unified` reconstructed
a strategy-equity peak around `1213.127` USDT while current strategy equity was
around `1000.5` USDT. That made the bot see about `17.5%` raw drawdown. With
the configured EMA span of `563` minutes, drawdown EMA climbed until it crossed
the red thresholds:

- long red threshold: `0.067`
- short red threshold: `0.072`
- long RED triggered: `2026-06-28T23:23:02Z`
- short RED triggered: `2026-06-28T23:47:54Z`
- XMR long `close_panic_long` fill: `2026-06-28T23:47:55Z`, `0.8` XMR at `311.44`

The executor followed the HSL forced panic mode. The wrong decision happened
earlier: HSL reconstructed account-level drawdown from current balance override
plus historical realized PnL, then combined inconsistent lookback anchors for
the replay peak and the current sample.

Follow-up VPS3 inspection on `2026-06-29` confirmed the operator assessment:
the panic was wrong. The account was not near a real account-level drawdown
that should have triggered RED. The bot was still running the vulnerable
`b972f022` build, remained flat after the close, and kept logging RED cooldown
from the synthetic peak. Current `v8` contains a guard which blocks this exact
startup contract before HSL replay can trade.

Additional inspection on `2026-06-29T06:17Z` found VPS3 still on the stale
`b972f022` build while GitHub `origin/v8` was already at `b207bb42`. The
current `v8` runtime guard raises before account-level HSL replay starts when
`balance_override` is active with `hsl_signal_mode=unified` or `pside`; targeted
tests for the guard pass. Therefore the immediate operational fix is to update
VPS3 to current `v8` before running this account again, or run only `coin` HSL,
disabled HSL, or no balance override until an explicit baseline/checkpoint
contract exists.

## Key Evidence

The current HSL latch/cache after the incident recorded:

```text
balance                  1000.000000
realized_pnl_total        201.801644
baseline_balance          798.198356
peak_strategy_pnl         414.928620
peak_strategy_equity     1213.126976
strategy_equity          1000.499700
drawdown_raw                0.175272
drawdown_ema                0.075951
signal_mode              unified
```

These values are internally consistent with the current formula:

```text
baseline_balance = balance - realized_pnl_total
peak_strategy_equity = baseline_balance + peak_strategy_pnl
drawdown_raw ~= 1 - strategy_equity / peak_strategy_equity
```

The arithmetic explains the emitted panic, but the premise is wrong. First,
`balance=1000` came from `-bo 1000`, not necessarily from a real exchange
balance continuous with the replayed fill-history window. Second, follow-up
inspection showed the replay peak used cumulative realized PnL outside the
configured 30-day lookback while the current runtime sample used the 30-day
realized-PnL function. That mixed-window state manufactured a peak/current
relationship that did not represent the account's current drawdown.

## Timeline

Startup already showed the false raw drawdown:

```text
2026-06-28T21:10:47Z Using balance override: 1000.000000
2026-06-28T21:11:03Z HSL[long] status tier=green drawdown_raw=0.176071 drawdown_ema=0.002097 peak_strategy_equity=1213.260375
2026-06-28T21:11:03Z HSL[short] status tier=green drawdown_raw=0.176071 drawdown_ema=0.002097 peak_strategy_equity=1213.260375
```

The bot did not panic immediately because drawdown EMA was still low. It later
crossed red:

```text
2026-06-28T23:23:02Z HSL[long] RED triggered strategy_equity=1000.035700 peak_strategy_equity=1213.126976 drawdown_score=0.067112 red_threshold=0.067000
2026-06-28T23:47:54Z HSL[short] RED triggered strategy_equity=1000.499700 peak_strategy_equity=1213.126976 drawdown_score=0.075951 red_threshold=0.072000
2026-06-28T23:47:55Z post XMR sell long 0.8@311.44 close_panic_long
```

## Root Causes

1. Unified/pside HSL assumed historical realized PnL could be mapped onto
   current balance by subtracting cumulative realized PnL from current balance.
   That is only safe if current balance is a real exchange balance continuous
   with the replayed fill window.

With a balance override, that assumption can synthesize a historical peak
unrelated to the current operator allocation. In this incident:

- current overridden allocation: `1000`
- reconstructed baseline: `798.198`
- reconstructed peak: `1213.127`
- reconstructed current raw drawdown: about `17.5%`

That is unsafe for live panic decisions.

2. Balance/equity replay used pre-lookback events to reconstruct the balance
   path, but the HSL-facing realized-PnL timeline fields were not zero-anchored
   at the configured lookback boundary. That let old realized PnL contribute to
   the replayed peak while the current HSL sample used lookback-scoped realized
   PnL. This violated the active PnL lookback contract.

## Immediate Contract

Until an explicit HSL baseline/checkpoint mechanism exists, live HSL must not
run account-level equity replay (`signal_mode=unified` or `signal_mode=pside`)
with `balance_override` active. The safe short-term behavior is to fail loudly
before replay starts.

`signal_mode=coin` is not blocked by this guard because it uses per-coin realized
PnL cumsum semantics and does not reconstruct account-level historical equity
from `balance - realized_pnl_total`.

Separately, balance/equity replay may still use pre-lookback fills to reconstruct
open positions and absolute balance continuity, but all HSL-facing realized-PnL
fields in recorded timeline rows must be zero-anchored at the configured
lookback boundary.

This aligns replay with the live runtime sample path:
`_equity_hard_stop_realized_pnl_now()` consumes the same active
`live.pnls_max_lookback_days` window via `_pnls_lookback_start_ms()`, so replayed
peaks and current samples must use the same realized-PnL scope.

The merged short-term mitigation is intentionally fail-loud: if HSL is enabled
and `balance_override` is active, `signal_mode=unified` and `signal_mode=pside`
must not proceed into history replay. Operators must remove the balance
override, switch to `hsl_signal_mode=coin`, disable HSL, or wait for an explicit
baseline/checkpoint mechanism.

This fix prevents the same false-panic calculation on new starts. Follow-up
inspection showed a second unsafe replay behavior: after restart without
`-bo`, the latest historical `close_panic_long` marker reconstructed to only
`drawdown_raw=0.001123`, but replay still treated the marker as an active RED
cooldown. The recovery contract is now explicit: a historical `close_panic_*`
marker may preserve cooldown only when reconstructed HSL metrics at that marker
confirm RED by tier, score, or raw drawdown. If reconstructed metrics are
nowhere near the RED threshold, replay must ignore the marker, must not write a
latch, and must not rebuild cooldown from the same marker through the generic
latest-panic-fill fallback.

## Follow-Up Work

- Add explicit HSL baseline/checkpoint support for overridden strategy
  allocations.
- Include baseline source in HSL metrics/latch payloads.
- Add startup diagnostics when raw drawdown is above red but EMA has not caught
  up yet.
- Add richer operator-facing diagnostics for ignored historical panic markers,
  including marker timestamp, reconstructed drawdown, threshold, and whether the
  marker came from an older unsafe balance-override run.
- Add preflight/reporting that flags `balance_override` plus account-level HSL
  before an operator starts live, so this class of configuration cannot be
  missed in a manual restart.
