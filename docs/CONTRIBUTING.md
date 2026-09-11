# Contributing to Passivbot

Thanks for contributing! Please read and follow:

- [Repository instructions](https://github.com/enarjord/passivbot/blob/master/AGENTS.md)
- [Engineering principles](ai/principles.md)
- [Trading-critical error contract](ai/error_contract.md)
- The quick guidelines below, distilled from maintainer discussions.

## Expectations

1. **Rust is the source of truth.** Implement new logic in Rust whenever possible so the live bot and backtester share the same behaviour. Python should primarily orchestrate bindings, configuration, and experiments.
2. **Stateless by default.** Never rely on “what happened earlier” unless that information can be rederived from the exchange/state snapshot on startup. Avoid ad-hoc local caches that would break after a restart.
3. **Minimal time-based heuristics.** Outside of natural candle boundaries (e.g. 1m closing), avoid timers/countdowns that are not reproducible from exchange data.
4. **Isolate big features.** If a change touches many areas, put the core logic in a dedicated module that both the live bot and backtester invoke. Keep modules focused and readable.
5. **Prefer pure functions.** The heart of Passivbot is answering: _Given the current state (balances, positions, fills, candles, config), what orders should exist right now?_ Every new component should aim to be deterministic given its inputs (including auto-unstuck, risk enforcement, etc.).

## Documentation and changelog

User guides describe the code on their branch. Record user-facing changes under
[Unreleased](https://github.com/enarjord/passivbot/blob/master/CHANGELOG.md#unreleased), even after they merge to `master`; the section covers
changes since the latest tag. Consolidate superseded entries into the net behavior and retain
upgrade actions. Do not add post-tag changes to a released section. See [Releases](releases.md)
and the [release runbook](ai/runbooks/release.md).

Validate doc changes with the [documentation checks](ai/README.md) and check local links and
technical claims against current code. With the `[dev]` profile installed, preview the user site
with `mkdocs serve` or build it with `mkdocs build`.
