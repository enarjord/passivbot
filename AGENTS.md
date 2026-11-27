# AGENTS.md

## Authority

Stay within the requested scope. Current-task approval is required to start a live or
authenticated paper/testnet bot, make any authenticated exchange request, use credentials,
change exchange orders, use SSH, deploy, or control a remote or live process. Demo, testnet,
sandbox, and paper modes need the same approval when they use credentials. The local `fake`
harness is deterministic and offline. Announce public unauthenticated network access, including
candle downloads, before running it. Tiers are defined in `docs/ai/validation.md`.

## Public Repository Boundary

Publish only material useful to readers with public sources. Private material can inform
conclusions, but release of the exact material requires explicit authorization. Keep private
configs, local paths, private hosts, accounts, logs, telemetry, dumps, credentials, and
operator-only evidence off public surfaces. The intentional templates under
`configs/examples/` are public. Keep private inputs outside the tracked tree.

## Branches

`master` is the default branch and released v8 line; `v9` is the v9 development line. These
documents describe the branch they are checked out on. Choose bases and port changes with
`docs/ai/runbooks/branches.md`.

## Contract Routing

Read this file, `docs/ai/principles.md`, and `docs/ai/README.md`. Use the router to load relevant
contracts. Trading-critical work requires `docs/ai/error_contract.md`.

Precedence runs from platform and system instructions through this file, current user
authority, canonical contracts, feature contracts, implementation evidence, and historical
notes. If a contract and runtime disagree, establish whether to restore the contract or
document the implementation. Surface material ambiguity. A change to documented behavior updates
its contract, generated references, and user guides in the same change.

## Working And Validation

Keep changes narrow and preserve unrelated user files. Before broad edits, inspect the branch,
recent commits, worktree, callers, and tests. For reviews against moving branches, refresh the
target and record the reviewed SHAs.

Run proportionate checks under `docs/ai/validation.md`. Bug fixes require regressions. Rust
changes require Rust tests, a rebuilt and verified Python extension where applicable, and
affected Python parity or integration checks. When auditing error handling, classify catches
and defaults in the touched diff and direct consumers before changing any code.

When a task includes delivering a change, publish validated work as a ready-for-review PR with a
clean branch, accurate description, and required author checks; this sets the format, not scope.
Drafts require an explicit request. Merge requires current-head review and CI.

Evidence-backed replies in assigned PR review threads (findings, fixes, validation, disagreement)
are authorized unless the user restricts public writes; private disclosures, unrelated messages,
and merging are not. Follow `docs/ai/runbooks/pr_review.md` for adjudication.

Describe net user-facing changes from the target branch in `CHANGELOG.md`: add new entries at the
top of `Unreleased`, and extend an existing entry instead of adding a second one for the same
behavior.
