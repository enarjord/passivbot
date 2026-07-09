# Live Logging Overhaul PR Loop Workflow

This note defines the resumed logging-overhaul implementation loop. It
complements:

- `docs/plans/live_logging_overhaul_plan.md`
- `docs/plans/live_logging_overhaul_progress.md`
- `docs/plans/live_performance_readiness_goals.md`
- `docs/plans/live_ops_improvement_backlog.md`

## Roles

- Codex owns implementation, local validation, PR creation, review iteration,
  merge to `v8` after the gate is satisfied, VPS5 deploy/smoke when appropriate,
  and progress evidence updates.
- Hermes, Claude Opus, Grok, and any additional reviewer agents review the
  current PR delta. Reviewers do not implement fixes or merge. Reusable
  reviewer-loop instructions live in `docs/ai/pr_auto_review_loop.md`.
- Maintainer-created PRs may also enter the same review loop. It is OK for
  reviewers to review non-logging-overhaul PRs when they target `v8`.

## PR Scope

- Prefer review-worthy slices over progress-only PRs.
- Keep each PR tied to one behavior/tooling/docs purpose and one validation
  story.
- Split PRs for behavior, restart/deploy behavior, exchange contact, HSL/risk,
  Rust extension behavior, and read-only report/tooling changes.
- Start dependent work only after its dependency is merged to `v8`; parallel PRs
  should be orthogonal.

## PR Body Contract

Every implementation PR should state:

- scope category: `read-only tooling`, `observability producer`, `runtime behavior`,
  `restart/deploy behavior`, or `trading-path`
- expected behavior impact
- expected VPS action: no restart, restart required, or observe only
- validation commands and results
- reviewer-specific notes for subtle contracts

## Review Gate

- Merge only after current-head green reviews from required reviewers and green
  CI.
- If the PR head changes after a review, require delta re-review for the new
  head before merging.
- For low-risk read-only tooling/docs PRs, use a degraded gate only when a
  reviewer is unavailable and the PR explicitly justifies that choice.

## Reviewer Output Contract

Ask reviewers to post one top-level PR review/comment with:

- findings first, ordered by severity
- exact file/line references for actionable findings
- commands run and evidence observed
- explicit approval/green-light when clean
- residual risk or untested surface

Reviewer focus areas:

- architecture and intent alignment
- trading-critical safety and error-contract compliance
- Rust/Python ownership boundaries
- statelessness and restart reproducibility
- tests that prove the claimed behavior
- operator usefulness in smoke, incident-bundle, debug-profile, and VPS workflows
- VPS deploy implications and whether restart/smoke evidence is sufficient

## Implementation Loop

1. Fetch current `origin/v8` and inspect open PRs/branches.
2. Reconcile `docs/plans/live_logging_overhaul_progress.md` against GitHub and
   VPS state before choosing the next slice.
3. Create a clean branch/worktree for the chosen slice when the main checkout is
   dirty.
4. Implement the narrow slice, update tests, and update progress docs only when
   the PR includes a real review-worthy code/tool/docs change.
5. Run targeted validation, `git diff --check`, and broader/fake-live/VPS tests
   when relevant.
6. Open the PR with the body contract above.
7. Wait for reviewer and CI feedback.
8. Apply narrow fixes for findings, push, and wait for delta re-review.
9. Merge only after the gate is satisfied.
10. Pull `v8` locally. Deploy to VPS5 according to the PR's declared deploy
    impact, restart bots only when the change requires running processes to load
    new code, and run bounded smoke/incident checks.
11. Record compact evidence in the progress ledger: PR, scope, validation,
    review gate, merge SHA, VPS action, smoke result, and remaining gap.

## VPS Policy

- Read-only tooling/report changes usually require pull plus bounded smoke, not
  bot restart.
- Producer/event payload changes require restart and observation.
- Runtime behavior, exchange contact, order/risk, and Rust changes require
  restart and observation.
- For Rust-touching deploys, rebuild and verify the extension before restarting.
- VPS actions must be explicit and evidence-driven; do not perform broad process
  kills or live-bot actions outside the declared deploy plan.
