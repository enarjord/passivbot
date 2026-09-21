# Revised HSL live validation and rollback

The revised engine is an opt-in replacement for HSL behavior within one process; legacy remains
the default. Simulation and offline fake-exchange validation do not authorize a live deployment.
Start this checklist only after the operator has explicitly approved the account, configuration,
launch command and intervention scope.

## Before the first live run

1. Record the reviewed commit, source-verified Rust extension, selected engine, signal mode and
   complete effective HSL policy. Preserve the original launch command and a legacy-compatible
   configuration for rollback. Keep account identifiers, credentials and operational artifacts
   outside the public repository.
2. Validate configuration through the canonical loader. Coin/pside policies belong under
   `bot.long.hsl` and `bot.short.hsl`; unified requires an explicitly supplied `bot.hsl` block.
   Confirm the chosen restart and manual-intervention policies, the 1–90 day lookback, the raw
   balance/coin-slot budget and fractional EMA span. See [configuration migration](configuration.md#experimental-revised-hsl-configuration).
3. Run a representative offline backtest and the fake runner using the intended mode, order type
   and scope count. Include both a panic and an ordinary-entry path. Verify the rebuilt extension,
   effective config, native scope diagnostics and resulting orders, not only process exit status.
4. Measure complete capture and connector admission with the intended lookback and scope count.
   The reproducible synthetic probe is
   `PYTHONPATH=src:tests python tests/hsl_revised_live_benchmark.py --mode coin --lookback-days 90 --coins 30 --admission`.
   Select the relevant mode and size. Admission must finish while actual input observations remain
   fresh; do not raise freshness limits to hide excessive computation. The advancing-clock probe
   includes the two actual evaluations, so inspect permissions as well as latency. It is one
   no-fill workload and cannot establish a universal performance guarantee.

## Observe during the approved run

- Confirm `live.hsl_engine=revised` and the expected scope topology: one controller per coin-side,
  one per active position side, or one portfolio controller. An inactive scope is distinct from a
  current GREEN result. Check thresholds, raw drawdown, drawdown EMA, selected signal and reported
  approximation reasons against the intended policy.
- Verify account and held-mark freshness. Historical damage should remain visible while Rust
  continues best-effort evaluation from valid minimum inputs. Missing current balance, positions,
  orders or required marks is a separate inability to authorize an exchange action; it must not
  appear as healthy protection.
- Observe ordinary entries and closes alongside protection. A historical HSL gap alone must not
  prohibit otherwise-valid position additions. Independent strategy, fill/PnL and account-input
  requirements still apply to the actions that consume them.
- For an operator-approved close test, confirm the native RED decision, cancellation/create
  admission, reduce-only full-position intent, actual exchange acknowledgements and the next
  authoritative position/order confirmation. A submission log does not prove a fill. Watch that
  unfilled limit orders, a missing peer quote and history repair do not monopolize independent
  coin scopes. A unified signal still requires the whole portfolio's current inputs.
- If partial execution or restart is deliberately tested, compare the reconstructed decision
  using the same exchange facts and configuration. The revised path has no durable local RED
  commitment: a decision with no reconstructible exchange evidence can disappear. Historical
  evidence, cooldown and `never` halts expire at the configured lookback boundary by design.
- If fresh positions are flat but the closing fill is delayed, verify that cooldown uses the
  latest retained scoped fill timestamp and does not renew on repeated reads. A delivered final
  fill may correct that timestamp and restore remaining cooldown. No retained history means no
  historical cooldown; this is an intentional best-effort policy.
- Confirm that diagnostics stay scoped and bounded, stale observations are labelled, and recovery
  is visible. Monitoring is evidence only; it must not authorize an order or delay admission.

Do not create extra market exposure merely to force a stop. Any authenticated read, order action,
restart or remote operation needs the explicit operational scope approved for that test.

## Rollback and acceptance

If observed behavior disagrees with the effective policy, preserve private diagnostic evidence and
return control to the operator. Inspect actual positions and resting orders before deciding the
next action. Changing a config file or stopping a process does not close a position or cancel an
order.

Rollback requires restarting with `live.hsl_engine=legacy` and a legacy-compatible configuration.
Do not silently translate a unified portfolio policy into side policies, or reuse revised optimizer
fitness as legacy fitness. Retain the exact original launch command unless the operator approves a
change. Confirm the resulting engine, exposure, orders and HSL readiness after restart.

The operator can accept the live trial after the applicable cases show the intended exchange
behavior, scope scheduling, source freshness, restart reconstruction and diagnostics. A successful
trial does not switch other bots or remove legacy code. Default replacement and legacy cleanup
remain a separate reviewed change after live validation.
