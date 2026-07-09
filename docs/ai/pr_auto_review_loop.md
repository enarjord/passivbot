# PR Auto-Review Loop

Use this when acting as an autonomous reviewer for Passivbot PRs targeting
`v8`.

## Scope

- Review open PRs against `v8`, including logging-overhaul PRs and
  maintainer-created PRs that are not part of the logging overhaul.
- Review the current PR head. If the head changes after a review, re-check the
  delta before approving.
- Do not implement fixes, push commits, merge PRs, deploy, SSH, or contact
  exchanges unless explicitly instructed.

## Loop Contract

- Use the strongest available self-prompting mechanism in your agent surface:
  goal mode, scheduled heartbeat, cron, automation, or equivalent.
- If no durable self-prompting mechanism is available, state that limitation in
  the thread and perform one complete poll/review pass.
- Poll cheaply when there is no open work. A one-minute poll interval is fine
  when each poll is limited to low-token GitHub metadata, such as open PR
  numbers, head SHAs, review state, and CI state.
- If each poll requires expensive diff/test context, use a five- to ten-minute
  interval until a PR head or CI/review state changes.
- Treat transient network, GitHub, checkout, or tool errors as retryable unless
  evidence proves otherwise. Record the error briefly, back off, and keep the
  loop alive.
- Do not stop the loop after one unexpected disconnect or command failure. On
  resume, fetch current remote state and continue from GitHub as the source of
  truth.
- Stop only when explicitly told to stop, when the goal is complete, or when the
  same blocker has repeated enough times that your agent's blocked-goal policy
  requires reporting it.

## Poll Pass

1. Fetch current `origin/v8`.
2. List open PRs targeting `v8`.
3. For each PR, compare the current head SHA, CI status, and latest reviewer
   state with the last observed state.
4. If nothing changed, sleep until the next poll.
5. If a PR is new or changed, create or refresh a clean review worktree and
   review the current PR head.

## Review Focus

- Architecture and intent alignment.
- Bugs, edge cases, and behavioral regressions.
- Trading-critical safety and `docs/ai/error_contract.md` compliance.
- Rust/Python ownership boundaries, especially order, risk, HSL, and unstuck.
- Stateless restart reproducibility.
- Test coverage and whether tests prove the intended behavior.
- Docs and operator-facing clarity.
- VPS deploy implications and whether restart/smoke evidence is sufficient.
- Backtest, optimize, fake-live, or real smoke validation when the PR touches
  relevant paths.

## Review Output

Post one top-level GitHub PR review or comment:

- Findings first, ordered by severity.
- Exact file/line references for actionable findings.
- Commands run and evidence observed.
- Explicit approval/green-light when clean.
- Residual risk or untested surface.

Do not recommend merge until the current PR head has green required reviews and
green CI.
