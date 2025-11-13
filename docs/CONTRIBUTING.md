# Contributing to Passivbot

Thanks for contributing! Please read and follow:

- `passivbot_agent_principles.yaml`
- The quick guidelines below, distilled from maintainer discussions.

## Expectations

1. **Rust is the source of truth.** Implement new logic in Rust whenever possible so the live bot and backtester share the same behaviour. Python should primarily orchestrate bindings, configuration, and experiments.
2. **Stateless by default.** Never rely on “what happened earlier” unless that information can be rederived from the exchange/state snapshot on startup. Avoid ad-hoc local caches that would break after a restart.
3. **Minimal time-based heuristics.** Outside of natural candle boundaries (e.g. 1m closing), avoid timers/countdowns that are not reproducible from exchange data.
4. **Isolate big features.** If a change touches many areas, put the core logic in a dedicated module that both the live bot and backtester invoke. Keep modules focused and readable.
5. **Prefer pure functions.** The heart of Passivbot is answering: _Given the current state (balances, positions, fills, candles, config), what orders should exist right now?_ Every new component should aim to be deterministic given its inputs (including auto-unstuck, risk enforcement, etc.).
