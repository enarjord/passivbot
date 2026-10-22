# HSL risks and operational limits

V9 HSL folds exchange facts forward through its decaying risk model. Derived caches
do not preserve a panic decision or cooldown independently of their factual inputs.
See the [HSL guide](equity_hard_stop_loss.md) for the signal and lifecycle.

## Incomplete historical evidence

Missing or delayed fills, ambiguous execution order and candle gaps can change the estimated
loss-memory path. The Rust reconciler preserves usable evidence and reports approximations.
It does not require complete historical proof before evaluating current protection.
An absent realized loss cannot be recovered from current positions alone, so an estimate
can stop earlier or later than a calculation with complete history.

There is no incomplete-history waiver or separate timed emergency fallback. With no useful
history, observed position size, entry basis, mark and budget provide an explicitly
approximate unknown-prefix seed with the ratio EMA initialized to the observed ratio.
Fresh usable current account and market facts are still required.
A GREEN label from an old observation is not proof that those facts are fresh now.

## Budgets, corrections and policy changes

Supported historical budget facts preserve the ratios sampled at those times.
Deposits, withdrawals and later slot/budget observations are not trading cashflows and
do not retrospectively renormalize those ratios. After a qualifying flatten, later
balance changes and flat queries cannot cancel, renew or extend its fixed deadline.
Corrected facts or changed risk policy require a new fold from the retained factual
initializer; changing policy is an operational decision requiring fresh calibration.

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

Startup acquisition is bounded and may have an unknown prefix. Its cap does not reset
continuous same-run memory as acquisition windows move. The factual initializer and
retained observations are inputs: losing them requires a disclosed new bootstrap,
while clearing only derived caches preserves decisions for identical facts and policy.
There is no permanent shutdown selector; an external operator or supervisor owns that policy.
Neither HSL nor its tests remove the need to monitor running bots. Offline unit, native,
fake-live and GPU tests do not establish actual exchange execution correctness.
