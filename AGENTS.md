# AGENTS.md

Instructions for AI coding assistants working on Passivbot.

## Authority And Operational Safety

Only perform actions within the user's requested scope. Explicit approval in the current task is
required before any of the following:

1. starting a live bot or authenticated paper/testnet bot
2. making an authenticated exchange request, including read-only account probes
3. creating, cancelling, or modifying exchange orders
4. using account credentials or private API keys
5. using SSH, deploying, restarting, stopping, or signalling a remote or live process

The local `fake` exchange harness is deterministic and offline. Public unauthenticated market-data
probes use real networks but no account credentials; state that clearly before running them.
Testnet, sandbox, demo, and paper-trading modes are not assumed safe or unauthenticated.

## Instruction Precedence

When instructions conflict, use this order:

1. active platform and system instructions
2. this `AGENTS.md`
3. the user's current request and explicitly granted authority
4. canonical contracts in `docs/ai/`
5. subsystem contracts in `docs/ai/features/`
6. user-facing documentation and current code/tests as implementation evidence
7. plans, handoffs, case studies, and historical notes

If a normative contract disagrees with runtime behavior, do not silently choose one. Establish
whether the task is to restore the contract or document the implementation, and surface material
ambiguity to the user.

## Always Read

1. `AGENTS.md`
2. `docs/ai/principles.md`
3. `docs/ai/README.md`

Use the router to load only task-relevant contracts and runbooks. Read
`docs/ai/error_contract.md` for trading-critical, exchange, live-data, indicator, risk, fill/PnL,
or order-construction work; it is not mandatory for unrelated documentation or tooling tasks.

## Core Rules

1. Rust owns order, strategy, risk, unstuck, and backtesting behavior. Python owns orchestration,
   exchange I/O, configuration, and data plumbing.
2. Trading behavior must be reproducible after restart from exchange state and config.
   A reviewed RAM-only economy gate may reset toward Rust intent without preserving orders or
   weakening safety.
3. Never fabricate a required trading input. Follow the explicit failure and degradation contract.
4. Keep `position_side`/`pside` (long/short) separate from `side`/`order_side` (buy/sell).
   Quantities and position sizes are signed internally.
5. Preserve EMA spans as floats, including derived spans.
6. Keep changes narrow. Preserve unrelated user files and pre-existing worktree changes.

## Working And Validation

Before broad edits, inspect the branch, recent commits, worktree status, and relevant callers/tests.
For reviews against a moving branch, refresh the target ref and record the reviewed SHAs.

Publish completed, validated work as a regular ready-for-review pull request by default. Before
publication, make the branch clean, make the PR body accurate, and run the required author checks.
Do not use a draft PR as a holding area for work the agent knows is incomplete; continue locally or
report the blocker instead. Use a draft only when the user explicitly requests early collaboration
on incomplete work. A regular PR still requires current-head review and CI before merge.

Run validation proportional to the changed contract. Bug fixes require regression coverage. Rust
changes require Rust tests, a rebuilt and verified Python extension where applicable, and parity or
integration checks for affected Python callers. See `docs/ai/validation.md` and
`docs/ai/runbooks/commands.md`.

When auditing error handling, inspect the touched diff and its direct consumers first. Classify
each catch/default against `docs/ai/error_contract.md`; do not rewrite unrelated repository-wide
matches merely because a broad search finds them.

Add user-facing behavior changes to `CHANGELOG.md` under `Unreleased`. Describe the net change from
the target branch, not intermediate iterations within a development branch.
