# HSL risks and operational limits

HSL reconstructs its current decision from exchange facts and configuration. Local caches
improve performance; they do not preserve a panic decision or cooldown independently.
See the [HSL guide](equity_hard_stop_loss.md) for the signal and lifecycle.

## Incomplete historical evidence

Missing or delayed fills, ambiguous execution order and candle gaps can change the estimated
equity peak. The Rust reconciler preserves usable evidence and reports approximations.
It does not require complete historical proof before evaluating current protection.
An absent realized loss cannot be recovered from current positions alone, so an estimate
can stop earlier or later than a calculation with complete history.

There is no incomplete-history waiver or separate timed emergency fallback. With no useful
history, the same evaluator measures current loss from position size, entry basis, mark
and balance budget. Fresh usable current account and market facts are still required.
A GREEN label from an old observation is not proof that those facts are fresh now.

## Changes to the reconstructed signal

Deposits, withdrawals, balance overrides, configured slot counts, signal modes and policy
changes can reinterpret the retained equity curve and the latest episode's terminal RED.
This can remove or restore cooldown. The remaining wait is always measured from that
episode's flatten timestamp, not from the time the evidence changed.

HSL does not infer transfer intent or maintain an authoritative transfer ledger. Re-backtest
policy changes, inspect current diagnostics and treat changes to live risk settings as
operational decisions. TWEL does not scale HSL's balance budget. Aggregate modes do not
support a live balance override; live startup and capture reject that combination. Use
the actual account balance, select coin mode, or disable HSL. A local baseline or checkpoint
does not make the aggregate override supported.

## Execution and coverage

RED is current intent, not a guarantee of a fill. Limit orders may remain unfilled, and
network failures, missing current facts or exchange restrictions can delay execution.
Price recovery cancels panic intent; a partial close continues only while the current
signal remains RED. Restarts reconstruct intent from current exchange evidence.

A fixed, bounded settling gate lets exchange fills catch up after an observed position
change. Approximate historical evidence must not turn this gate into an indefinite lock.
Other ordinary strategy and unstucking inputs retain their own readiness requirements.

The lookback is intentionally finite. Expired fills and cooldown evidence are forgotten;
`restart_after_red_policy=never` is therefore lookback-bounded, not a permanent halt.
Neither HSL nor its tests remove the need to monitor running bots. Offline unit, native,
fake-live and GPU tests do not establish actual exchange execution correctness.
