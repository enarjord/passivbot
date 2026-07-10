# PR Auto-Review Loop

Use this workflow when acting as an autonomous reviewer for Passivbot pull
requests targeting `v8`.

## Authority And Scope

- Review every eligible open PR targeting `v8`, including maintainer-created
  and non-logging-overhaul PRs.
- Review the current PR head and current merge base. An approval is valid only
  for the exact reviewed head SHA.
- Review code, tests, documentation, PR claims, and repository-policy
  alignment. Run real backtest, optimize, and fake-live smoke checks when the
  touched paths make them relevant.
- Do not implement fixes, push commits, merge, deploy, SSH, signal processes,
  or contact exchanges unless the maintainer explicitly changes the role.

## Durable Loop

Use goal mode, a scheduled heartbeat, cron, an automation, or the strongest
durable self-prompting mechanism available. The loop must survive an idle
terminal, one failed poll, a transient disconnect, and an agent restart.
If no durable mechanism exists, say so in the task, perform one complete
poll/review pass, and do not claim that continuous monitoring is active.

Persist only compact review state outside the repository, for example under
`$TMPDIR/passivbot-pr-review-<reviewer>.json`. Store:

- PR number
- last observed base and head SHAs
- last fully reviewed head SHA
- CI-state digest
- latest review/comment metadata digest
- retry count and next retry time
- last short error classification

Do not store credentials, full comments, diffs, test output, or exchange data in
the state file. GitHub and the fetched repository remain the sources of truth.

## Cost-Aware Polling

Polling should be deterministic and should not require semantic model work when
nothing changed.

1. Use one metadata request for open PRs, for example:

   ```bash
   gh pr list --base v8 --state open --limit 1000 \
     --json number,title,isDraft,headRefOid,updatedAt,statusCheckRollup
   ```

2. Compare PR number, head SHA, draft state, update time, and CI summary with the
   saved state.
3. Fetch detailed reviews/comments only for a new or changed PR.
4. Do not fetch the diff, create a worktree, rerun tests, or invoke a strong
   semantic model for an unchanged head.

Use a one-minute interval only when the scheduler runs the metadata command and
digest comparison without invoking a model, and wakes an agent only after the
digest changes. If every heartbeat invokes an agent/model, use at least ten
minutes even while CI or reviews are pending; pending state alone is not a
reason for repeated model wakeups. Add small jitter when several reviewers poll
the same repository.

Where model routing is available:

- use deterministic tooling or the cheapest capable tier for metadata polling
- use a lower-cost reasoning tier for changed-state triage and test-log summary
- reserve the strongest tier for full semantic review, trading-critical code,
  architecture, security, conflicting findings, and final judgment

Do not weaken a high-risk review because a strong model is rate-limited. Keep
the PR pending and retry later.

## Retry And Recovery

- Treat network, GitHub, checkout, authentication-refresh, subprocess, and
  temporary rate-limit failures as retryable unless proven persistent.
- Back off after consecutive failures, for example 1, 2, 5, then 10 minutes,
  capped at 10 minutes. Reset after a successful poll.
- A rate-limit response should delay the next metadata poll; it should not end
  the loop or produce repeated model narration.
- After reconnect or agent restart, reload compact state, fetch GitHub metadata,
  and reconcile by PR number and head SHA before doing semantic work.
- Do not mark the loop blocked after one failure. Escalate only after the same
  persistent blocker has repeated and no read-only progress remains possible.
- Stop only when explicitly told to stop, the assigned goal is complete, or the
  agent's blocked-goal policy requires a final blocked state.

## Review Trigger Rules

Perform a semantic review when:

- a ready-for-review PR is new
- the head SHA changed
- the PR was rebased or its effective merge base changed materially
- the maintainer explicitly requests a re-review
- new evidence invalidates an earlier conclusion

Do not duplicate a review merely because CI changed from pending to green. Do
not approve a draft unless explicitly asked. If the head changes after review,
review the delta from the last reviewed SHA and re-check the integrated full PR
before posting a new verdict.

## Isolated Checkout

- Work from the reviewer's own clone or clean read-only review worktree, never
  from the implementer's dirty checkout.
- Fetch `origin/v8` and the PR head immediately before review.
- Record the base SHA, head SHA, and merge state used for the review.
- Never modify the PR branch. Test artifacts belong in temporary directories.
- Re-check the remote head immediately before posting; discard or clearly mark
  results if the head changed during review.

## Review Method

1. Read `AGENTS.md`, `docs/ai/principles.yaml`,
   `docs/ai/error_contract.md`, and task-specific docs from
   `docs/ai/README.md`.
2. Read the PR body and classify the scope: docs/tooling, observability
   producer, runtime behavior, exchange integration, Rust/trading, backtest, or
   optimizer.
3. Review the complete diff against the current merge base. Verify that the PR
   body and docs do not overclaim the actual code surface.
4. Trace changed behavior through callers and consumers. Do not review an
   isolated function when safety depends on the full runtime path.
5. Run validation proportional to risk:
   - docs/tooling: claim verification, focused tests, compile/lint/diff checks
   - live observability: event routing, boundedness, redaction, hot-path cost,
     sink-failure isolation, fake-live where applicable
   - exchange code: focused connector tests and documented exchange contracts;
     no real exchange contact without permission
   - Rust/order/risk/HSL/unstuck: Rust tests, extension rebuild when needed,
     Python parity tests, and a real bounded backtest/fake-live smoke
   - backtest/optimize: real short backtest and/or optimize integration smoke
6. Distinguish PR regressions from failures reproduced on the current base.
7. Check mergeability and current CI, but do not treat CI as a substitute for
   semantic review.

## Review Focus

- Architecture and stated intent.
- Bugs, edge cases, concurrency, and behavioral regressions.
- Trading-critical safety and `docs/ai/error_contract.md` compliance.
- Rust/Python ownership boundaries.
- Stateless restart reproducibility.
- Signed quantity, side/position-side, EMA-span, and exchange-contract rules.
- Test quality: tests must prove the claim rather than only execute the path.
- Documentation accuracy and user/operator impact.
- Secret safety, bounded payloads, event volume, retention, and hot-path cost.
- Incident reconstruction, correlation IDs, reason codes, smoke reports, and
  VPS deploy/restart implications.

Avoid style-only findings unless they affect correctness, operations, or
maintainability.

## GitHub Output

Prefer a formal GitHub review:

- `APPROVE` when clean
- `REQUEST_CHANGES` for actionable findings
- `COMMENT` only for clarification or when the reviewer shares the author
  account and GitHub forbids self-approval

When formal review is unavailable, post one structured top-level comment. Every
verdict must include:

- reviewer identity
- PR number
- reviewed base and head SHAs
- scope: full review or exact delta
- decision: `approve`, `request changes`, or `needs clarification`
- findings first, ordered by severity, with exact file/line references
- commands run and observed results
- base-only failures or environmental limitations
- residual risk and untested surfaces

An approval comment must say explicitly that it is green for the named head
SHA. Never state that a PR is merge-ready unless all configured reviewers and
CI are green on that same head.

## State After Review

After posting, save the reviewed head SHA and return to metadata-only polling.
If the author pushes a fix, review the delta, rerun affected validation, inspect
the integrated result, and supersede the earlier verdict. Do not keep posting
unchanged approvals or status comments.
