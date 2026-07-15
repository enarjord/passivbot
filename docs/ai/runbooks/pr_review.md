# Pull Request Review Runbook

Use this for repository-owned semantic review. Automation platforms own their scheduler and
credential configuration; this runbook owns the durable polling, recovery, trigger, and review
contract the scheduler must preserve.

## Authority

Review is read-only unless the user explicitly asks for implementation. Do not push, merge, deploy,
SSH, signal processes, contact authenticated exchange endpoints, or run a live bot merely because
those actions might strengthen a review.

An approval applies only to the reviewed head SHA and merge base unless the proportional mechanical
delta exception below is documented and satisfied.

## Durable Autonomous Loop

Continuous review requires a scheduler, heartbeat, or equivalent mechanism that survives idle
terminals, transient disconnects, and agent restarts. If no durable mechanism exists, perform one
complete pass and do not claim continuous monitoring.

Persist only compact state outside the repository, such as:

- PR number and draft state
- last observed target/head SHAs and merge-base identity
- last fully reviewed head SHA
- CI and review/comment metadata digests
- retry count, next retry time, and bounded error classification

Do not persist credentials, full comments, diffs, test output, or exchange data. GitHub and fetched
repository refs remain authoritative.

Scope review state by reviewer identity and head SHA. One reviewer's approval does not supersede
another's request for changes, and old-head verdicts do not apply to a semantically changed head.
The proportional mechanical-delta contract below is the only carry-forward exception. Preserve
reviewer, head, decision, and submission time instead of collapsing review history. Ambiguity wakes
semantic review rather than being resolved by the polling tier.

### Target Selection And Branch Cutovers

- Treat each PR's current base ref and base SHA as its authoritative review target. Do not infer the
  target from the feature-branch name, an old prompt, or cached scheduler state.
- Repository-default review loops resolve the current default branch from live GitHub metadata.
  A loop intentionally scoped to a non-default branch records that exception explicitly rather than
  hardcoding a historical default.
- Persist the observed base ref and default branch in compact state. A PR base-ref change or a
  default-branch change for a default-target loop materially changes review scope: invalidate the
  cached merge base and gate digest, refetch exact refs, and perform a fresh integrated review.
- After a default-branch cutover, update scheduler filters, protected-check configuration, and
  compact cache. Before retiring the old target or compatibility routes, require a changed-head wake
  to discover a PR against the new default and load this canonical runbook successfully.

### Cost-Aware Polling

Unchanged polls are deterministic metadata work, not semantic-review work:

1. Fetch open-PR metadata including number, draft state, base ref/SHA, head SHA, update time, and CI
   summary. Default-target loops also fetch the repository's current default branch.
2. Compare it with compact saved state.
3. Fetch reviews/comments and wake semantic review only for a new or materially changed PR.
4. Do not fetch diffs, create worktrees, rerun tests, or invoke a reasoning model for an unchanged
   head merely because CI remains pending.

One-minute polling is appropriate only when the scheduler computes the digest without invoking a
model. If every heartbeat wakes an agent, use at least ten minutes and add small jitter.

Where model routing exists, use deterministic tooling for metadata, a lower-cost capable tier for
changed-state triage, and the strongest available tier for trading-critical, architectural,
security, conflicting-finding, and final-verdict work. Rate limits delay a high-risk review; they do
not justify weakening it.

### Retry And Recovery

- Retry transient network, GitHub, checkout, authentication-refresh, subprocess, and rate-limit
  failures with bounded backoff such as 1, 2, 5, then 10 minutes.
- Reset backoff after a successful poll. Rate limits delay metadata polling without generating
  repeated narration.
- After reconnect or restart, reload compact state and reconcile live metadata by PR number and
  exact SHAs before semantic work.
- One failure is not a blocker. Escalate only after the same persistent condition repeats and no
  read-only progress remains.

### Semantic Review Triggers

Review when a ready PR is new, its head or base ref changes, its effective merge base changes
materially, a default-target loop observes a new repository default, a maintainer requests
re-review, or new evidence invalidates the prior conclusion. A CI-only state change does not
duplicate semantic review. Do not approve drafts unless explicitly requested.

After a changed head, review the delta and then the integrated full PR. Post one current verdict;
do not repeat unchanged approvals or status narration.

### Proportional Mechanical Delta Review

Do not require a redundant full semantic review when a changed head only integrates the current
target branch or resolves a mechanical conflict without changing the already-reviewed behavior.
The final adjudicator may carry the prior semantic verdict forward when all of these are true:

1. The target-relative production, test, configuration, and contract diff is unchanged from the
   semantically reviewed scope.
2. The new commit delta is inspected directly and contains only mechanical integration work, such
   as preserving both sides of a changelog/ledger conflict, formatting, or an equivalent no-op.
3. The integrated branch is mergeable and required CI is green.
4. The PR records the old reviewed head, new head, target SHA, exact mechanical delta, validation,
   and the reason a full re-review is unnecessary.

This exception does not apply to code, test-contract, configuration, dependency, generated
contract, runtime, or substantive documentation changes, nor when the effective merge-base change
alters integrated behavior. Those changes require a current-head semantic delta and integrated
review. If an obsolete request-for-changes names only the resolved mechanical blocker, a maintainer
may dismiss or supersede it with the recorded mechanical-delta evidence; that is not a degraded
gate. Repository-enforced CI and protection still apply.

### Enforceable Review Gates

When repository policy makes semantic review mandatory, publish the result as a status or check
bound to the exact head SHA and enforce that check with repository merge protection. Comments,
polling cadence, and reviewer narration alone cannot prevent a new or unreviewed head from merging.

- A semantically changed head makes the semantic-review check pending until that head is reviewed.
- A stale check, comment, or approval alone cannot satisfy the current-head gate. A current-head
  mechanical-delta check may explicitly carry the prior semantic verdict under the contract above.
- The posting path re-fetches the head immediately before publishing the verdict.
- Merge readiness still requires every configured semantic review, carried mechanical-delta
  adjudication where applicable, and CI gate to be green for the integrated current head.
- A `COMMENT` review remains advisory even when its prose says `APPROVE`; it cannot satisfy a
  required GitHub approval or substitute for a protected status/check.

If the automation cannot publish or enforce a SHA-bound check, describe it as advisory monitoring,
not a mandatory merge gate.

## Fresh Isolated Checkout

1. Fetch the current target branch and PR head immediately before review.
2. Use a clean review worktree, not an implementer's dirty checkout.
3. Record target SHA, head SHA, merge base, and mergeability.
4. Re-check the remote head before posting. If it moved, discard or clearly scope stale results.

## Review Method

1. Read `AGENTS.md`, `docs/ai/principles.yaml`, `docs/ai/validation.md`, and routed subsystem
   contracts.
2. Read the PR body and classify affected contracts: documentation/tooling, observability, live
   orchestration, exchange, Rust/trading, backtest, optimizer, config, or data preparation.
3. Review the complete diff against the current merge base. Verify that PR claims match the actual
   code surface.
4. Trace behavior through callers and consumers, including restart and failure paths.
5. Validate according to `../validation.md`:
   - observability: routing, boundedness, redaction, hot-path cost, sink isolation
   - exchange: offline connector/request tests and documented payload contracts
   - Rust/order/risk: Rust tests, verified extension, parity tests, bounded backtest/fake-live
   - backtest/optimizer: bounded real CLI integration smoke
6. Distinguish change regressions from failures reproduced on the current target branch.
7. Check CI and mergeability without treating CI as a substitute for semantic review.

Never contact an exchange without the authority required by `AGENTS.md`. The local fake-live
harness is offline; a public-network probe is not.

## Review Focus

- Architecture and intended contract.
- Trading safety, error handling, and degraded behavior.
- Rust/Python ownership and live/backtest parity.
- Stateless restart reproducibility.
- Signed quantities, side semantics, EMA spans, and exchange contracts.
- Concurrency, ambiguous exchange writes, retry/idempotency, and shutdown behavior.
- Test quality: tests prove the claim rather than merely execute the path.
- Documentation accuracy and operator impact.
- Secrets, payload bounds, retention, event volume, and hot-path cost.

Avoid style-only findings unless they affect correctness or maintainability.

## Verdict

Use a formal review when available:

- `APPROVE` when no actionable findings remain
- `REQUEST_CHANGES` for actionable defects
- `COMMENT` for clarification or when self-approval is unavailable

When self-approval is unavailable, a `COMMENT` may record the semantic verdict and exact reviewed
SHAs, but it is not a formal approval. Describe the result as advisory unless a separate protected
SHA-bound review check records it.

Every verdict records:

- reviewer identity and PR number
- base, head, and merge-base SHAs
- full-review or exact-delta scope
- findings first, ordered by severity, with exact locations
- commands and observed results
- base-only failures and environmental limitations
- residual risk and untested surfaces

Never call a PR merge-ready unless required current-head review, or a documented proportional
mechanical carry-forward, and CI are green for the integrated head SHA.

After a new push, review the delta and re-check the integrated result before superseding the prior
verdict. Do not post repeated unchanged approvals or polling narration.
