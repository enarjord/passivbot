# Pull Request Review Runbook

Use this for repository-owned semantic review. Read `AGENTS.md`, `../principles.md`,
`../validation.md`, and the subsystem contracts routed by `../README.md`.

Review is read-only unless the user explicitly asks for implementation. Follow the authority and
network boundaries in `AGENTS.md` while gathering evidence.

## Review Loop

For a one-time review, make one complete pass. For continuous review, use a durable scheduler that
can resume after restarts and retain only enough compact state to identify the PR and its last
reviewed head. GitHub and fetched repository refs remain authoritative.

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

When a new head only incorporates the target branch or resolves a mechanical conflict without
changing the reviewed behavior, a focused delta review is sufficient. Any substantive change to
code, tests, configuration, dependencies, contracts, runtime behavior, or documentation requires a
current-head review of the affected result.

Re-fetch the PR head immediately before posting. If it moved, review the new head instead of
publishing a stale verdict.

## Sign-Off

When actionable findings remain, post them through the appropriate review mechanism and do not sign
off. When no actionable findings remain, approve when available and leave a concise review record
ending with:

```text
reviewed by <model/harness name>
```

Associate the sign-off with the exact reviewed head. If self-approval is unavailable, a signed
comment may record the completed semantic review but is not a formal GitHub approval. If semantic
review is intended to be an enforced merge gate, use repository protection or a head-bound check;
comments alone are advisory.

Call a PR merge-ready only when the current head has the required review sign-off and required CI is
green. After a new push, review the changed result and issue a new sign-off for that head.
