# Passivbot Engineering Principles

Canonical repository-wide invariants; task documents link here instead of restating them.

## Architecture

- Rust is the source of truth for order logic, strategies, risk, unstuck, backtesting, and derived
  analysis metrics.
- A Rust ideal-order result is atomic current intent. Validate the complete result before
  reconciliation. Malformed output is fatal; Python must not substitute prior ideals, actual
  orders, or a usable-looking subset.
- Producer-boundary validation may enforce schema and compact deterministic invariants, but it must
  not reimplement Rust's strategy, sizing, or risk engines in Python to second-guess otherwise
  valid intent. Suspected semantic producer defects belong in Rust with Rust-side regressions.
- Python owns orchestration, exchange I/O, configuration, data plumbing, caching, reconciliation,
  and execution gating.
- Live and backtest behavior must implement equivalent trading and risk contracts, not identical
  raw-data availability. Runtime-specific readiness and bounded fallbacks require parity tests
  against a simple shared reference contract and must not bias selection toward fresher symbols.

## Statelessness

- Trading decisions must be reproducible after restart from exchange state and config.
- Do not add decision-changing local state that cannot be rederived.
- Performance caches are allowed only when cache loss, rejection, or rebuild does not change the
  intended decision.
- A reviewed RAM-only economy gate may reset only toward current Rust intent. It must not preserve
  or synthesize orders, weaken readiness/risk/mode checks, or hide resets. Test reset behavior.

## Terminology And Numeric Conventions

- `position_side`, `pos_side`, and `pside` mean `long`/`short`; `side` and `order_side` mean
  `buy`/`sell`.
- `qty` and `pos_size` are signed internally. Use `abs(qty)` only at exchange boundaries requiring
  unsigned quantities.
- EMA spans are floats. Do not round derived spans.
- Entries must observe effective minimum quantity.
- Closes should observe effective minimum cost. If a position is below effective minimum quantity,
  close quantity may equal the remaining position size.

## Broker Attribution

- Broker codes and broker-agreement attribution are exchange-critical behavior.
- Do not remove, bypass, rename, or weaken broker-code handling without explicit user approval.
- Broker-code loading must fail visibly on missing registries, invalid data, or unknown exchanges.
- Every relevant order request must carry the exchange-required broker field, header, or tag.
- Attribution changes require tests at the actual CCXT or raw signed-request boundary, not merely
  tests of local configuration values.

## Configuration Ownership

Place parameters by actual consumers:

- `config.live`: consumed by live and shared with backtest/optimizer
- `config.backtest`: simulation-only behavior
- `config.optimize`: optimizer-only behavior

Choose the narrowest consumer-owned surface; uncertainty does not justify `config.live`. Defaults
belong in canonical loading/formatting; runtime consumers must not reapply them.

## Failure Handling

- Required trading inputs must never be replaced with fabricated neutral values.
- Exchange fetch methods propagate failures to caller policy rather than silently downgrading them.
- Any allowed fallback must be bounded, observable, and tested.
- Use `error_contract.md` for the precise meanings of propagate, unavailable, defer, fail closed,
  degraded, and fatal.

## Scope, Testing, And Compatibility

- Keep changes aligned with the requested task; avoid speculative abstractions and unrelated cleanup.
- Add targeted tests for changed behavior, edge cases, and regressions.
- Compatibility code is for supported released-version boundaries, not intermediate development
  iterations, unless the user explicitly requests it or a documented compatibility contract exists.
- Add user-facing behavior changes to `../../CHANGELOG.md` under `Unreleased`.

## Release Hygiene

- Treat either 50 top-level user-facing entries under `../../CHANGELOG.md` `Unreleased`, or 14 days
  since the latest stable tag with at least 10 such entries, as an advisory release-review trigger.
- When a trigger is reached, no release is already in progress, and the accumulated changes form a
  coherent release, tell the maintainer and recommend the appropriate semantic version. Ask for
  explicit permission before editing versions, cutting a branch, tagging, or publishing a release.
- Do not interrupt incident response or live-safety work with a release prompt. Raise it at the
  handoff instead. Follow `runbooks/release.md` for version selection, validation, and publication.
