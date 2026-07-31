# Live Logging Overhaul PR Loop Workflow

This workflow applies only when the resume procedure in
`live_logging_overhaul_current_status.md` has produced a finite, evidence-backed
logging gap list. Use `live_logging_overhaul_progress.md` only for historical
evidence.

Architecture and backlog references:

- `live_logging_overhaul_plan.md`
- `live_performance_readiness_goals.md`
- `live_ops_improvement_backlog.md`
- `../ai/runbooks/pr_review.md`

## Goal

Deliver bounded, correlated, operator-useful live observability through
reviewable PRs, current-head review gates, and evidence-based validation on an
approved representative runtime, without moving trading decisions out of Rust
or making observability a trading control plane.

## Tiered Ownership

Use the cheapest capable execution tier and reserve the strongest reasoning
tier for consequential decisions.

### Sol: Lead And Safety Owner

- Select architecture and PR boundaries.
- Own trading-critical, exchange-contract, Rust, risk/HSL/unstuck, security,
  concurrency, and live-runtime decisions.
- Adjudicate reviewer findings and conflicting evidence.
- Inspect delegated diffs and validation before publication.
- Decide merge readiness and perform or authorize VPS signals/restarts.
- Interpret incidents and choose the next dependent slice.

### Terra: Scoped Implementation Worker

- Implement straightforward docs, tests, report projections, query/tooling
  changes, and bounded observability producers with an explicit file scope.
- Run local validation and return a clean commit/diff plus evidence.
- Work only in an isolated branch/worktree.
- Do not merge, deploy, SSH, signal processes, or broaden scope.

### Luna: Polling And Read-Only Triage

- Poll low-token GitHub metadata and notify Sol only when state changes.
- Summarize CI/reviewer findings, current head/base, and merge state.
- Run read-only branch checks and parse test, smoke, incident, and performance
  outputs.
- Do not edit, approve on Sol's behalf, merge, deploy, SSH, or signal processes.

If these tiers are unavailable, preserve the role boundaries with deterministic
commands and direct Sol work. Do not spend strong-model context on unchanged
polls.

## Delegation Contract

Every delegated task must state:

- objective and non-goals
- base SHA and branch/worktree
- allowed files and ownership boundary
- forbidden actions, especially merge/SSH/live exchange/process signals
- required tests and evidence
- expected return: findings, diff/commit, validation, and residual risk

Parallel work is allowed only for orthogonal PRs. Work that depends on another
PR must wait until that PR is merged to `master`. Sol reviews every delegated diff
before push or merge.

## Current State And History

- `live_logging_overhaul_current_status.md` is the compact scope, completion,
  and resume source. It does not track every active PR, deployed SHA, or VPS
  action.
- `live_logging_overhaul_progress.md` is historical evidence. Do not load or
  rewrite the entire ledger during routine work.
- Update compact status only when scope, completion criteria, or the finite
  resume gap list changes.
- Add historical progress only for a material logging milestone or design
  decision, not every PR, deploy, or smoke.

## PR Scope

- Prefer one coherent behavior/tooling/docs purpose and one validation story.
- Combine adjacent low-risk report or query fields when they share the same
  producer, consumer, and test surface.
- Avoid progress-only PRs and excessively small projection PRs.
- Split changes at safety boundaries: trading behavior, exchange contact,
  Rust/PyO3, HSL/risk, sink/backpressure, raw retention, restart behavior, and
  read-only tooling.
- Keep dependent work serial. Use parallel PRs only when they are truly
  orthogonal and merge-order independent.

Regular ready-for-review PRs are the default and preferred publication state.
Finish implementation and author validation before publishing: the branch must
be clean, the PR body accurate, and required author tests passing. Do not
publish known-premature work merely to create a PR. Use a draft only when the
user or maintainer explicitly requests early collaboration on incomplete work.

## PR Body Contract

State:

- scope category: `docs`, `read-only tooling`, `observability producer`,
  `runtime behavior`, `restart/deploy behavior`, or `trading-path`
- touched surfaces and explicit non-goals
- expected behavior impact
- expected VPS action: none, pull/observe, or restart/smoke
- validation commands and results
- reviewer focus requests and known residual risk

## Review Gate

- Query the live GitHub ruleset/branch protection for the target branch and
  record its required checks and reviews. Add any semantic reviewers explicitly
  required by `AGENTS.md` or the current task. Those current sources define the
  gate; historical progress entries do not.
- Merge only after that required reviewer/check set is green on the exact
  current head SHA, or after prior semantic approval is carried by a documented
  current-head mechanical-delta adjudication under the canonical review runbook.
- Prefer formal GitHub reviews or commit-bound checks. If GitHub forbids
  self-approval, accept a structured comment only when it names the reviewer,
  verdict, and exact head SHA.
- A semantic head change invalidates prior approval unless the reviewer carries
  it to the new head after delta/integrated review. A conflict-only merge,
  changelog/ledger preservation, formatting, or equivalent mechanical no-op may
  retain prior semantic approval under the proportional mechanical-delta
  contract in `docs/ai/runbooks/pr_review.md`; record the exact delta and keep
  CI and repository protection green.
- Findings from optional reviewers still require verification and resolution.
- Use a degraded gate only with explicit maintainer authorization and record it
  in the PR and historical evidence.

## Cost-Aware Waiting

- Use deterministic metadata polling and a compact state digest as defined in
  `docs/ai/runbooks/pr_review.md`.
- Poll every minute only when the scheduler computes the digest without a model
  and wakes an agent only after a change. If every heartbeat consumes model
  context, use at least ten minutes; pending CI/reviews do not justify faster
  model wakeups.
- Wake Sol only for a new PR/head, a reviewer finding, a completed gate, a
  persistent blocker, or a deploy decision.
- Transient GitHub, network, rate-limit, or tool errors back off and retry; they
  do not end the goal.

## Implementation Loop

1. Read the compact current-status file, resolve the current default branch and
   PR target from live repository/PR metadata, fetch that exact remote ref, and
   reconcile open logging PRs with their exact heads and review/check state.
2. Confirm that an observed gap maps to an unmet logging completion criterion
   and is not already covered by an open PR. Read historical progress only when
   evidence is needed.
3. Select one review-worthy slice, preferring deletion, consolidation,
   aggregation, or demotion before a new producer or tool. Route routine scoped
   work to Terra when available; keep high-risk work with Sol.
4. Create a clean branch/worktree. Preserve unrelated tracked and untracked
   artifacts in the main checkout and on any approved runtime host.
5. Implement and validate proportionally. Use real fake-live, backtest, or
   optimize smokes when touched behavior warrants them.
6. Finish the selected slice and author validation, then open a regular
   ready-for-review PR with the PR body contract. Use a draft only after an
   explicit request for early collaboration on incomplete work.
7. Let Luna/deterministic polling monitor GitHub. Sol does not wait in a
   high-cost semantic loop.
8. Verify every finding against the current branch. Apply the narrow fix,
   rerun affected tests, push, and require current-head delta review.
9. Re-resolve and fetch the live PR target ref, then verify mergeability, gate
   SHAs, and CI immediately before merge.
10. Merge only after the gate is satisfied. Update local state.
11. Obtain explicit approval in the current task before any SSH, remote
    preflight, pull, restart, smoke, or other host action. After approval,
    compare each target host's exact deployed SHA and local config with the
    intended target delta. Stop on unresolved config-sensitive behavior
    changes. Then deploy according to the declared impact. Sol retains control
    of actual VPS signals/restarts; approved read-only preflight and output
    parsing may be delegated.
12. Run immediate and settled bounded smoke checks and leave expected bots
    running.
13. Record bounded post-deployment evidence: exact deployed SHA, target scope,
    commands/actions, outcomes, and limitations. A public PR may contain only a
    sanitized public-safe summary. Hostnames, account/config identifiers, local
    paths, process details, private commands, and VPS-derived telemetry belong
    only in an operator-approved private run record, never the public
    repository. Update compact status only if scope/completion changed. Append
    historical progress only for a material milestone.

## VPS Policy

- Docs-only changes require no VPS action.
- Read-only tooling/report changes usually require pull plus bounded smoke, not
  bot restart.
- Producer/event payload, console/text projection, startup/shutdown, live-loop,
  Rust, exchange-call, and trading-path changes normally require restart and
  observation.
- For Rust changes, rebuild and verify the extension before restart.
- Use exact verified process/session targets. Do not use broad process-pattern
  signals.
- Preserve known tracked edits and all local configs, logs, monitor data,
  reports, and temporary artifacts.

## Goal Health

- Keep the active goal objective short and point it to this workflow plus the
  compact status file.
- Do not mark the goal blocked for a single approval, network, quota, or tool
  failure. Back off and continue with available read-only work.
- Mark blocked only after the same persistent condition repeats and no useful
  progress remains possible under the goal policy.
