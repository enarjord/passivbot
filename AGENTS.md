# AGENTS.md

## Authority

Stay within the requested scope. Current-task approval is required to start a live or
authenticated paper/testnet bot, make any authenticated exchange request, use credentials,
change exchange orders, use SSH, deploy, or control a remote or live process. The local
`fake` harness is deterministic and offline. Announce public unauthenticated market-data
probes before running them. Demo, testnet, sandbox, and paper modes can use credentials.

## Public Repository Boundary

Publish only material useful to readers with public sources. Private material can inform
conclusions, but release of the exact material requires explicit authorization. Keep private
configs, local paths, private hosts, accounts, logs, telemetry, dumps, credentials, and
operator-only evidence off public surfaces. The intentional templates under
`configs/examples/` are public. Keep private inputs outside the tracked tree.

## Contract Routing

Read this file, `docs/ai/principles.md`, and `docs/ai/README.md`. Use the router to load relevant
contracts. Trading-critical work requires `docs/ai/error_contract.md`.

Precedence runs from platform and system instructions through this file, current user
authority, canonical contracts, feature contracts, implementation evidence, and historical
notes. If a contract and runtime disagree, establish whether to restore the contract or
document the implementation. Surface material ambiguity.

## Working And Validation

Keep changes narrow and preserve unrelated user files. Before broad edits, inspect the branch,
recent commits, worktree, callers, and tests. For reviews against moving branches, refresh the
target and record the reviewed SHAs.

Run proportionate checks under `docs/ai/validation.md`. Bug fixes require regressions. Rust
changes require Rust tests, a rebuilt and verified Python extension where applicable, and
affected Python parity or integration checks. When auditing error handling, classify catches
and defaults in the touched diff and direct consumers before changing any code.

Publish completed, validated work as a ready-for-review PR with a clean branch, accurate
description, and required author checks. Drafts require an explicit request for early
collaboration. Merge requires current-head review and CI.

Routine evidence-backed replies in assigned PR review threads are authorized unless the user
restricts public writes. This covers in-scope findings, fixes, validation, and disagreement.
It does not authorize private disclosures, unrelated messages, or merging. When disagreeing,
explain the rationale in the original thread, leave it unresolved, and request reconsideration.
Follow `docs/ai/runbooks/pr_review.md` for adjudication.

Describe net user-facing changes from the target branch in `CHANGELOG.md` under `Unreleased`.
