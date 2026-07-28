# Live Logging Overhaul Current Status

Updated: 2026-07-28.

This is the compact resume point for the live logging overhaul. Historical PR,
deployment, and smoke evidence remains in
`docs/plans/live_logging_overhaul_progress.md`; do not copy that ledger back into
this file.

## Current State

The original architecture is substantially established:

- live events use a shared envelope, registry, routing policy, and bounded
  asynchronous pipeline;
- console and text output are projections rather than the forensic source;
- monitor-backed structured events are durable, rotated, queryable, and
  correlated across cycles, snapshots, remote calls, order waves, and actions;
- event and sink failures remain isolated from trading decisions;
- lifecycle, readiness, planning, execution, fills, positions, balance, risk,
  HSL, trailing, unstuck, candles, EMA, forager, and remote-call producers have
  structured coverage;
- the speculative per-snapshot planning-availability Cartesian product is not
  part of the event contract; actual snapshot provenance, readiness failures,
  Rust results, reconciliation, and entry outcomes retain separate canonical
  evidence;
- redaction and bounded-payload rules cover the primary live-event and
  diagnostic paths.

The overhaul is paused at a consolidation boundary, not at the beginning of
another open-ended producer or tooling expansion. Restart orchestration,
repository preparation, process inspection, performance analysis, incident
bundling, cache/HSL optimization, configuration preflight, and trading-policy
changes are useful adjacent capabilities, but they are not measures of logging
completion.

## Scope When Work Resumes

Resume logging work only for one of these reasons:

1. A concrete incident cannot be reconstructed from existing structured
   evidence.
2. A remaining text producer has duplicate ownership, violates the console
   budget, or retains unsafe/unbounded data.
3. Event loss, corruption, routing, retention, redaction, or sink isolation is
   incorrect.
4. Two or more existing readers or reports can be simplified around one shared
   bounded scan/model implementation with a net reduction in code.
5. A required migration exit criterion below is demonstrably unmet.

Do not resume the logging loop merely because another producer, report field,
query selector, smoke verdict, restart check, or operational edge case could be
added.

## Explicitly Separate Work

Route these through their own plans, issues, or focused PRs:

- trading behavior, readiness enforcement, reconciliation, and risk policy;
- HSL, candle, fill, startup, or shutdown performance;
- restart execution, tmux/process control, deployment, and repository
  preparation;
- general-purpose process, incident, performance, or configuration tooling;
- optional dashboard, relay, export, or query convenience features.

Those streams may consume the event pipeline. They do not extend the logging
overhaul unless they expose a specific defect in the event contract or an
unmet exit criterion.

## Completion Criteria

The logging overhaul may be declared complete when all of the following hold:

- one canonical structured path covers the high-value live decision and action
  boundaries;
- sink failure and queue pressure cannot change trading behavior;
- every sink and profile omits raw exchange/account payloads; retained payloads
  are bounded and omit credentials, unsafe exception values, and unbounded
  symbol/order collections;
- console INFO satisfies `docs/ai/logging_policy.md` in representative
  overnight operation, excluding its documented action/fill/safety
  exemptions;
- remaining direct `logging.*` producers are intentionally retained, have one
  human-output owner, and do not need structured migration;
- monitor rotation and recovery enforce configured per-bot byte/retention caps,
  and a representative overnight run has no queue drops or sink degradation
  attributable to normal logging load;
- the minimal event query can reconstruct lifecycle, degraded planning,
  exchange calls, order waves, fills, and safety transitions across rotation;
- there is no active must-have logging slice backed by an observed gap.

Completion does not require every live function to emit an event, every
diagnostic to have a report projection, or every item in the live-operations
backlog to be finished.

## Resume Procedure

1. Resolve the current default branch and any PR target from live
   repository/PR metadata, fetch that exact remote ref, and reconcile open
   logging PRs with their exact heads and review/check state. Ignore stale
   branch names and historical “next slice” text.
2. Inspect representative overnight event and INFO output already available in
   local artifacts before proposing code. If the evidence exists only on a
   remote host, obtain explicit approval in the current task before SSH or any
   other remote access.
3. Produce a finite list of observed gaps mapped to the completion criteria,
   excluding work already covered by an open PR.
4. Rank deletion, consolidation, demotion, or aggregation before adding a new
   event type, emitter, tool, or report field.
5. Implement one independently reviewable slice. Prefer net-negative or
   net-neutral complexity.
6. Validate routing, boundedness, redaction, sink isolation, event-registry
   parity, and the affected operator/forensic workflow.
7. Stop when the finite gap list is closed. Newly noticed adjacent work goes to
   the live-operations backlog and does not automatically reopen this plan.

Before any runtime pull or restart, independently reconcile each target host's
exact deployed SHA and local config against the intended target delta. Historical
deployment notes are evidence, not authorization or a current-state substitute;
an unresolved config-sensitive delta blocks deployment until an explicit rollout
decision is made.

## Next Slice

No implementation slice is preselected. The next logging PR, if any, should be
chosen only after the resume procedure identifies an unmet completion
criterion. A progress-only PR or another speculative producer/profile/report
slice is not a valid next step.

## References

- Architecture and boundaries: `live_logging_overhaul_plan.md`
- Historical evidence: `live_logging_overhaul_progress.md`
- Migration inventory: `live_logging_migration_audit.md`
- Console policy and budgets: `../ai/logging_policy.md`
- Operational backlog: `live_ops_improvement_backlog.md`
- Validation contract: `../ai/validation.md`
