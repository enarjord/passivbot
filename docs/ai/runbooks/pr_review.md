# Pull Request Review Runbook

Use this for repository-owned semantic review. Read `AGENTS.md`, `../principles.md`,
`../validation.md`, and the subsystem contracts routed by `../README.md`.

Review is read-only unless the user explicitly asks for implementation. Follow the authority and
network boundaries in `AGENTS.md` while gathering evidence.

## Review Loop

For a one-time review, make one complete pass. For continuous review, use a durable scheduler that
can resume after restarts and retain only enough compact state to identify the PR and completed
reviews. GitHub and fetched repository refs remain authoritative.

The compact change detector includes PR and draft state, exact base, head, and effective merge-base
identities, and digests of CI and review/comment metadata. Scope completed-review records by
reviewer and those identities, preserving the decision. Do not persist credentials, full comments,
diffs, test output, or exchange data.

- Discover the current base and head from live PR metadata. Default-branch loops discover the
  repository's current default branch rather than hardcoding a historical target.
- Review ready PRs when they are new, their head, base, or effective merge base changes, re-review is
  requested, or new evidence invalidates an earlier conclusion. Do not review drafts unless
  requested.
- A review applies to its exact base, head, and effective merge base. Do not duplicate reviews when
  those identities are unchanged, and do not treat CI-only updates as a reason to repeat semantic
  review.
- After interruption or a transient failure, reconcile live PR metadata and continue with bounded
  retry rather than relying on stale state.

## Review

1. Fetch the current base and PR head, compute their effective merge base, and review from a clean
   checkout or worktree.
2. Review the complete target-relative change, its relevant callers and consumers, and any material
   restart or failure behavior.
3. Apply the routed contracts and choose proportionate validation from `../validation.md`. Check CI
   and mergeability, but do not treat CI as a substitute for review.
4. Distinguish defects introduced by the PR from failures already present on the target branch or
   caused by the environment.
5. Report actionable findings first with exact locations, evidence, impact, and a credible fix.
   Avoid style-only findings unless they affect correctness or maintainability.

## Review Feedback Adjudication

Do not silently treat disputed reviewer feedback as addressed. When an agent disagrees with a
finding and the task authorizes review-comment writes:

1. Reply in the original review thread before moving on. State the decision, cite the exact code
   and governing contract, explain the regression or authority violation the recommendation would
   introduce, and identify the narrowest contract-preserving alternative when one exists.
2. Explicitly ask the reviewer to reconsider or provide counter-evidence. A private agent note or
   local classification is not a substitute for this visible exchange.
3. Leave the thread unresolved while the reviewer has an opportunity to respond. Do not resolve it
   in the same action as the disagreement reply.
4. Re-evaluate new evidence rather than repeating the prior conclusion. Resolve only after the
   reviewer acknowledges the rationale, a new exact-head review explicitly clears the dispute, or
   the user makes the final adjudication after the reviewer was asked to reconsider.

If the task does not authorize GitHub comment writes, draft the same evidence-backed reply and ask
the user for authorization instead of silently rejecting the finding. Continuous-review state may
record that adjudication is pending, but it must not record the thread as addressed.

## Architectural Proposal Check

Before recommending a trading-critical fix or fallback:

1. Name the current authority for the decision: Rust intent, exchange truth, canonical input
   readiness, reconciliation, or execution policy.
2. Distinguish a valid empty decision from malformed producer output and unavailable input.
3. Reject proposals that preserve, synthesize, or reinterpret strategy intent outside Rust.
4. Check restart reproducibility and live/backtest parity under realistic live data delay, including
   whether selection would favor whichever symbols refreshed first.
5. Fix a hypothetical producer defect at its producer boundary. Do not add consumer-side trading
   policy without evidence of the failure and an explicit contract authorizing that policy.
6. Distinguish bounded schema/cross-field validation from replaying the trading engine. Do not
   recommend duplicating Rust strategy, sizing, realized-loss, or exposure calculations in Python
   merely to detect a hypothetical internally consistent Rust decision.

When a new head only incorporates the target branch or resolves a mechanical conflict, prior
semantic approval may be carried forward after a focused delta review only when:

1. The target-relative production, test, configuration, and contract diff is unchanged.
2. Direct inspection confirms that the new commit delta is mechanical and does not change the
   reviewed behavior.
3. The integrated branch is mergeable and required CI is green.
4. The review records the old and new heads, target SHA, inspected delta, validation, and reason the
   prior approval still applies.

Any substantive change to code, tests, configuration, dependencies, contracts, runtime behavior,
or documentation requires a current-head review of the affected result.

Re-fetch the exact base and head and recompute the effective merge base immediately before posting.
If any identity changed, review the changed integrated result instead of publishing a stale verdict.

## Sign-Off

Every completed review records the reviewer identity, exact base, head, and effective merge-base
SHAs, and decision, and ends with:

```text
reviewed by <model/harness name>
```

This marker records completion by that reviewer, not approval. When actionable findings remain,
post them through the appropriate review mechanism, record a changes-requested decision, and do not
approve. When no actionable findings remain, approve when available. A requested draft review
remains advisory and uses `COMMENT` unless formal approval of the draft was explicitly requested.
If self-approval is unavailable, a completed-review comment is not a formal GitHub approval.

If semantic review is intended to be an enforced merge gate, use repository protection or a
head-bound check; comments alone are advisory.

Call a PR merge-ready only when the current head has the required review sign-off and required CI is
green. A draft is not merge-ready. After a new push, review the changed result and issue a new
sign-off for that integrated state.
