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

- Trading decisions must be reproducible from the same exchange facts, configuration and
  declared initialization. Bounded live HSL reconstruction is an explicit exception to exact
  restart equivalence when the discarded historical prefix is unknown; see the HSL contract.
- HSL may retain a compact native accounting/lifecycle boundary plus bounded factual history.
  Routine pruning and derived-cache rebuilding preserve D/M and known cooldown deadlines.
  Restart, owner loss or unsupported correction declares approximate reconstruction; no local
  journal or persisted panic latch preserves a prior permission.
- Ordinary market EMA startup/cache loss may use the available completed bounded bootstrap;
  it need not reproduce an unknown infinite prefix. See `features/candlestick_manager.md`.
- Other performance caches may not change intent when lost, rejected or rebuilt. These explicit
  approximation exceptions do not change the independent realized-memory origin contract.
- A reviewed RAM-only economy gate may reset only toward current Rust intent. It must not preserve
  or synthesize orders, weaken readiness/risk/mode checks, or hide resets. Test reset behavior.

## Terminology And Numeric Conventions

- New or renamed numeric interfaces follow [numeric names and units](numeric_conventions.md);
  distinguish fractions, ratios, multipliers, probabilities, scores and display percentages.
- `position_side`, `pos_side`, and `pside` mean `long`/`short`; `side` and `order_side` mean
  `buy`/`sell`.
- `qty` and `pos_size` are signed internally. Use `abs(qty)` only at exchange boundaries requiring
  unsigned quantities.
- EMA spans are floats. Do not round derived spans.
- Entries must observe effective minimum quantity.
- Closes should observe effective minimum cost. If a position is below effective minimum quantity,
  close quantity may equal the remaining position size.

## Attribution And Configuration

Preserve broker attribution at the exchange request boundary. Changes require explicit approval
and signed-request or CCXT boundary tests. See [exchange integrations](features/exchange_integrations.md).

Put shared trading policy in `bot` and operational controls in the consuming `live`, `backtest`,
or `optimize` section. Resolve defaults in canonical config loading. See
[configuration ownership](architecture.md#configuration-ownership).

## Failure Handling

- Required trading inputs must never be replaced with fabricated neutral values.
- Exchange fetch methods propagate failures to caller policy rather than silently downgrading them.
- Any allowed fallback must be bounded, observable, and tested.
- Use `error_contract.md` for the precise meanings of propagate, unavailable, defer, fail closed,
  degraded, and fatal.

## Readable Output

Use bounded, actionable logs with repetitive noise suppressed. Follow [logging policy](logging_policy.md).
Use streamlined JSON for human-authored or human-edited configs and normal JSON for tool outputs.

## Scope, Testing, And Compatibility

- Keep changes aligned with the requested task; avoid speculative abstractions and unrelated cleanup.
- Add targeted tests for changed behavior, edge cases, and regressions.
- Compatibility code is for supported released-version boundaries, not intermediate development
  iterations, unless the user explicitly requests it or a documented compatibility contract exists.
- Add user-facing behavior changes to `../../CHANGELOG.md` under `Unreleased`.

## Release Hygiene

Use the advisory [release review triggers](runbooks/release.md). Release actions require explicit
approval. Raise them at handoff during active incident response or live-safety work.
