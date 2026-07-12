# Passivbot Engineering Principles

This is the canonical home for durable repository-wide engineering invariants. Task-specific
documents should link here rather than restating these rules.

## Architecture

- Rust is the source of truth for order logic, strategies, risk, unstuck, backtesting behavior,
  and analysis metrics derived from those behaviors.
- Python owns orchestration, exchange I/O, configuration, data collection, caching, reconciliation,
  and execution gating.
- Live and backtest behavior must implement equivalent trading and risk contracts. Separate
  implementations are acceptable when runtime contexts differ, but require parity tests against a
  simple shared reference contract.

## Statelessness

- Trading decisions must be reproducible after restart from exchange state and configuration.
- Do not add decision-changing local state that cannot be rederived.
- Performance caches are allowed only when cache loss, rejection, or rebuild does not change the
  intended trading decision.

## Terminology And Numeric Conventions

- `position_side`, `pos_side`, and `pside` mean `long` or `short`.
- `side` and `order_side` mean `buy` or `sell`.
- `qty` and `pos_size` are signed internally. Use `abs(qty)` only at an exchange boundary that
  requires unsigned payload quantities.
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

Place a parameter according to its actual consumers:

- `config.live`: consumed by live and shared with backtest/optimizer
- `config.backtest`: simulation-only behavior
- `config.optimize`: optimizer-only behavior

Do not place a parameter in `config.live` merely because ownership is uncertain. Trace the
consumers and choose the narrowest correct surface. Defaults belong in the canonical config loading
and formatting path; runtime consumers should not silently reapply them.

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
