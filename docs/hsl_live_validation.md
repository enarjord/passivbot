# HSL live validation and rollback

HSL uses one implementation across live and simulation. Simulation and offline fake-exchange validation do not authorize a live deployment.
Start this checklist only after the operator has explicitly approved the account, configuration,
launch command and intervention scope.

## Before the first live run

1. Record the reviewed commit, source-verified Rust extension, signal mode and
   complete effective HSL policy. Preserve the previous reviewed release, its source-verified native extension, launch command
   and compatible configuration separately for rollback. Keep account identifiers, credentials and operational artifacts
   outside the public repository.
2. Validate configuration through the canonical loader. Coin/pside policies belong under
   `bot.long.hsl` and `bot.short.hsl`; portfolio requires an explicitly supplied `bot.hsl` block.
   Confirm the chosen slow/fast EMA spans, the bounded bootstrap horizon, the independent
   rolling loss-gate window and raw balance/coin-slot budget. There is no Never policy.
   A pre-final-fill RED or final-accounting RED qualifies a fixed finite cooldown.
   See [configuration migration](v8_to_v9_migration.md).
3. Run a representative offline backtest and the fake runner using the intended mode, order type
   and scope count. Include both a panic and an ordinary-entry path. Verify the rebuilt extension,
   effective config, native scope diagnostics and resulting orders, not only process exit status.
4. Measure complete factual capture, Rust reconstruction and connector admission with the
   intended spans, scope count and bounded bootstrap history. Admission must finish while
   current account/held-mark observations remain fresh. Inspect reconstructed permissions
   and unknown-prefix/truncation diagnostics alongside latency. Synthetic no-fill probes
   cannot establish a universal performance guarantee.

## Observe during the approved run

- Confirm the expected scope topology: one controller per coin-side,
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
  coin scopes. A portfolio signal still requires the whole portfolio's current inputs.
- If partial execution or restart is deliberately tested, compare the reconstructed decision
  using the same exchange facts and configuration. HSL has no durable local RED
  commitment: qualification comes from factual flatten events. Known-origin replay must
  reproduce decisions; unknown prefixes retain explicit approximation diagnostics. Later
  GREEN and budget changes cannot cancel or renew a qualified finite deadline.
- If fresh positions are flat but the final fill is delayed, verify bounded fact repair and
  the reconstruction diagnostics. The final event's pre/post RED assessments and factual
  flatten time determine cooldown. Repeated reads never renew it. Missing evidence cannot
  be replaced by an old local permission flag or order tag.
- Confirm that diagnostics stay scoped and bounded, stale observations are labelled, and recovery
  is visible. Monitoring is evidence only; it must not authorize an order or delay admission.

Do not create extra market exposure merely to force a stop. Any authenticated read, order action,
restart or remote operation needs the explicit operational scope approved for that test.

## Rollback and acceptance

If observed behavior disagrees with the effective policy, preserve private diagnostic evidence and
return control to the operator. Inspect actual positions and resting orders before deciding the
next action. Changing a config file or stopping a process does not close a position or cancel an
order.

Rollback requires installing a previous reviewed release and its matching native extension,
then restarting with that release's compatible configuration and saved launch arguments.
Validate the complete effective configuration with that release before an operator-approved
rollback. Do not reuse fitness calculated under different trading semantics. Verify exposure,
resting orders and protection readiness afterwards.

The operator can accept a live trial after the applicable cases show the intended exchange
behavior, scope scheduling, source freshness, restart reconstruction and diagnostics. A successful
trial does not authorize changes to other bots. See the [validation matrix](hsl_testing.md)
for the offline acceptance cases.
