# Release Runbook

Use this runbook to decide when to propose a release and to prepare an explicitly authorized
release. Release prompts are advisory; version edits, tags, and publication require current-task
maintainer approval.

## Advisory Trigger

Count top-level user-facing entries under `../../../CHANGELOG.md` `Unreleased` and inspect the
latest stable `vMAJOR.MINOR.PATCH` tag. Review whether a release is appropriate when either
condition is true:

- at least 50 Unreleased entries; or
- at least 14 days since the stable tag and at least 10 Unreleased entries.

Prompt only when no version-bump or release PR is already active and the changes make a coherent,
supportable release. Consider user-visible features, exchange support, config changes, trading or
risk corrections, operator tooling, migration needs, and whether current `master` has a validated
head suitable for a release cut. Defer the prompt during incident response or active live-safety
work and mention it at handoff.

The prompt should state the tag date, entry count, major themes, proposed semantic version, and why
that level fits. Ask the maintainer for permission; do not edit, branch, tag, publish, deploy, or
control live processes merely because the threshold was crossed.

## Choose The Version

- Patch: compatible corrections without a meaningful new documented capability.
- Minor: compatible features, exchange support, config additions, or substantial operator tooling.
- Major: an intentional incompatible public config, CLI, strategy, or runtime contract.

Package and config-schema versions are separate decisions. Bump the schema when serialized config
compatibility changes or new configs must identify capabilities older runtimes do not understand.
Older supported schemas need explicit migration and regression coverage. Do not rewrite historical
release notes or migration examples to the new version.

## Prepare The Candidate

1. Refresh the default branch and record the exact release base SHA.
2. Work in a clean branch or worktree, preserving private configs, logs, credentials, and local
   artifacts outside the release diff.
3. Update the package version, current-schema constant when applicable, maintained public examples,
   current-version documentation, `../../../CHANGELOG.md`, and a curated release-notes file.
4. Leave a new empty `Unreleased` section. Summarize major themes, compatibility notes, required
   operator actions, and known limitations instead of copying the entire change ledger.
5. Add regression coverage for version surfaces and every supported schema transition.

## Validate And Publish

Run validation for every changed contract. A normal package-plus-schema release includes config
loader/default/round-trip tests, public-example parity, CLI version and package-metadata checks, AI
documentation checks, the full Python suite, Rust tests, a rebuilt and fingerprint-verified Python
extension, and the offline fake-live release smoke. Do not use authenticated exchange or live-bot
actions as release validation without separate explicit approval.

Publish a completed candidate as a regular ready-for-review PR. Before merge, recheck the exact
base/head identities, current-head review state, and CI. After merge, create an annotated
`vMAJOR.MINOR.PATCH` tag on the exact merge commit, push the tag, and publish GitHub release notes
from the curated document. Finally verify the remote tag, released CLI version, clean release diff,
and the fresh empty `Unreleased` section.
