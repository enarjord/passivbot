# Branch And Porting Runbook

Use this when choosing a base branch, porting a change between release lines, integrating
`master` into `v9`, or preparing the v9 cutover.

## Release Lines

| Branch | Role |
|---|---|
| `master` | Repository default branch and the released v8 line (latest tag per [release status](../../releases.md)). Receives v8 fixes and maintenance. |
| `v9` | Development line for v9.0.0 (package `9.0.0.dev0`, config schema `v9.0.0`). Receives new features and v9 fixes. |

The lines differ in config schema, risk model, and contracts. Each branch's `AGENTS.md` and
`docs/ai/` describe that branch only. When working on `master`, read the documents checked out on
`master`; do not apply v9 contracts, config fields, or commands to v8 code, or the reverse.

## Choosing A Base

- Use the base the task names. A branch name, issue, or PR that names a line selects it.
- Otherwise, new features, refactors, and documentation target `v9`.
- A fix for behavior that runs in released v8 deployments targets `master`. When the defect also
  exists on `v9`, deliver a separate `v9` change unless the task says the v9 line is out of scope.
- Resolve the base from the PR metadata of the current task, not from the repository default.

## Porting Between Lines

- Port the intent, not the patch. Re-derive the change against the target line's code, contracts,
  and tests; a cherry-pick that applies cleanly can still violate a contract that differs between
  lines.
- Give each line its own branch, PR, regression test, and changelog entry. Name branches so
  the line is visible, for example `fix/v8-<topic>` and `fix/v9-<topic>`.
- Cross-link the paired PRs and state in each description what differs between the two versions.
- If a defect exists on only one line, say so in the PR description so reviewers do not wait for
  a counterpart.

## Integrating `master` Into `v9`

`master` is merged into `v9`; `v9` is not merged into `master` before the cutover.

- Merge in a dedicated branch and PR against `v9`.
- Resolve each conflict toward the v9 contract. A v8 fix whose v9 counterpart already landed keeps
  the v9 version; otherwise port its intent under the rules above.
- Changelog conflicts keep both lines' entries unless they describe the same change, in which
  case keep the v9 wording.
- Verify `git merge-base --is-ancestor origin/master <merge-head>` before requesting review.

## Cutover

The v9 cutover (making v9 the released line) follows the rollback-reference and qualification
checklist in [release status](../../releases.md#v8-rollback-reference) and the
[release runbook](release.md). After the cutover, update this runbook, the release runbook, and
any scheduled automation that filters by base branch.
