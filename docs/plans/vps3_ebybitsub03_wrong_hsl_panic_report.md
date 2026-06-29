# VPS3 ebybitsub03 Wrong HSL Panic Incident

Status: investigation report for the HSL balance-override safety guard
Incident account: `ebybitsub03` on `vps3`
Exchange: Bybit
Observed command: `passivbot live -u ebybitsub03 configs/xmr_migrated.json -bo 1000`
Observed VPS3 head: `v8` at `b972f0221fb5121330a7ee3a41b99c761317544c`

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

The executor followed the HSL forced panic mode. The questionable decision was
earlier: HSL reconstructed account-level drawdown from current balance override
plus historical realized PnL.

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

The problem is not arithmetic. The problem is that `balance=1000` came from
`-bo 1000`, not necessarily from a real exchange balance continuous with the
entire replayed fill-history window.

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

## Root Cause

Unified/pside HSL assumes historical realized PnL can be mapped onto current
balance by subtracting cumulative realized PnL from current balance. That is
only safe if current balance is a real exchange balance continuous with the
replayed fill window.

With a balance override, that assumption can synthesize a historical peak
unrelated to the current operator allocation. In this incident:

- current overridden allocation: `1000`
- reconstructed baseline: `798.198`
- reconstructed peak: `1213.127`
- reconstructed current raw drawdown: about `17.5%`

That is unsafe for live panic decisions.

## Immediate Contract

Until an explicit HSL baseline/checkpoint mechanism exists, live HSL must not
run account-level equity replay (`signal_mode=unified` or `signal_mode=pside`)
with `balance_override` active. The safe short-term behavior is to fail loudly
before replay starts.

`signal_mode=coin` is not blocked by this guard because it uses per-coin realized
PnL cumsum semantics and does not reconstruct account-level historical equity
from `balance - realized_pnl_total`.

## Follow-Up Work

- Add explicit HSL baseline/checkpoint support for overridden strategy
  allocations.
- Include baseline source in HSL metrics/latch payloads.
- Add startup diagnostics when raw drawdown is above red but EMA has not caught
  up yet.
- Decide how to treat existing latch files created by the unsafe overridden
  account-level replay model.
