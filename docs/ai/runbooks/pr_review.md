# Pull Request Review Runbook

Use this for repository-owned semantic review. Read `AGENTS.md`, `../principles.md`,
`../validation.md`, and the subsystem contracts routed by `../README.md`.

Review is read-only unless the user explicitly asks for implementation. Follow the authority and
network boundaries in `AGENTS.md` while gathering evidence.

## Review Loop

For a one-time review, make one complete pass. For continuous review, use a durable scheduler that
can resume after restarts and retain only enough compact state to identify the PR and its last
reviewed head. GitHub and fetched repository refs remain authoritative.

The compact change detector includes PR and draft state, base and head identity, and digests of CI
and review/comment metadata. Scope completed-review records by reviewer and head, preserving the
decision. Do not persist credentials, full comments, diffs, test output, or exchange data.

- Discover the current base and head from live PR metadata. Default-branch loops discover the
  repository's current default branch rather than hardcoding a historical target.
- Review ready PRs when they are new, their head or base changes, re-review is requested, or new
  evidence invalidates an earlier conclusion. Do not review drafts unless requested.
- A review applies to the reviewed head. Do not duplicate reviews for an unchanged head, and do not
  treat CI-only updates as a reason to repeat semantic review.
- After interruption or a transient failure, reconcile live PR metadata and continue with bounded
  retry rather than relying on stale state.

## Review

1. Fetch the current base and PR head and review from a clean checkout or worktree.
2. Review the complete target-relative change, its relevant callers and consumers, and any material
   restart or failure behavior.
3. Apply the routed contracts and choose proportionate validation from `../validation.md`. Check CI
   and mergeability, but do not treat CI as a substitute for review.
4. Distinguish defects introduced by the PR from failures already present on the target branch or
   caused by the environment.
5. Report actionable findings first with exact locations, evidence, impact, and a credible fix.
   Avoid style-only findings unless they affect correctness or maintainability.

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

Re-fetch the PR head immediately before posting. If it moved, review the new head instead of
publishing a stale verdict.

## Sign-Off

Every completed review records the reviewer identity, exact head SHA, and decision, and ends with:

```text
reviewed by <model/harness name>
```

This marker records completion by that reviewer, not approval. When actionable findings remain,
post them through the appropriate review mechanism, record a changes-requested decision, and do not
approve. When no actionable findings remain, approve when available. If self-approval is
unavailable, a completed-review comment is not a formal GitHub approval.

If semantic review is intended to be an enforced merge gate, use repository protection or a
head-bound check; comments alone are advisory.

Call a PR merge-ready only when the current head has the required review sign-off and required CI is
green. After a new push, review the changed result and issue a new sign-off for that head.
