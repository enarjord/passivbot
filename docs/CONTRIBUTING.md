# Contributing to Passivbot

Thanks for contributing! Please read and follow:

- `passivbot_agent_principles.yaml` — the canonical reference for architecture and coding ethos.
- The quick guidelines below, distilled from maintainer discussions.

## Core Expectations

1. **Rust is the source of truth.** Implement shared logic in Rust so live bot and backtester stay
   in sync. Python is orchestration/glue, not business logic.
2. **Be stateless.** Don’t rely on “what happened earlier” unless it can be reconstructed from
   exchange data/state on startup.
3. **Avoid timers.** Aside from candle boundaries, avoid time-based heuristics that can’t be
   replayed from exchange data.
4. **Isolate big features.** Put substantial functionality in its own module and have both live and
   backtest paths call it.
5. **Prefer pure functions.** Passivbot’s core question is: _Given this state, what orders should
   exist right now?_ Logic should be deterministic for a given state/config.

Before opening a pull request:

- Ensure `passivbot_agent_principles.yaml` is respected (link to the relevant section in your PR if
  needed).
- Add or update tests for any behaviour change.
- Ensure both toolchains pass. Typical flow:
  ```bash
  cd passivbot-rust
  cargo fmt
  maturin develop --release
  cd ..
  source venv/bin/activate && pytest
  ```

Happy hacking! Feel free to open an issue if you need clarification on any of the above.
