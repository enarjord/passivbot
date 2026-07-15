# Autonomous PR Review Compatibility Route

This path is temporarily retained for active external review automations.

The canonical durable polling, recovery, trigger, validation, and verdict contract is
`runbooks/pr_review.md`. Read and follow that document in full. Do not fall back to an older cached
copy when the canonical path is available.

Use each PR's current base ref as its target. Repository-default loops must resolve the live default
branch from GitHub metadata and must not continue polling a cached historical branch after a branch
cutover.

Remove this compatibility route only after every scheduled reviewer has been migrated and a
changed-head wake has successfully loaded the canonical runbook.
