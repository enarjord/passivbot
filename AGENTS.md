# AGENTS.md

Instructions for AI coding assistants working on Passivbot.

## Overview

Passivbot is a cryptocurrency trading bot for perpetual futures markets. It uses a contrarian market-making strategy, implemented in Python (orchestration) and Rust (order calculations, backtesting).

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd passivbot-rust && maturin develop --release && cd ..

# Test
pytest

# Run live bot
python3 src/main.py -u {account_name}

# Backtest
python3 src/backtest.py path/to/config.json

# Optimize
python3 src/optimize.py path/to/config.json
```

See [docs/ai/commands.md](docs/ai/commands.md) for full command reference.

## Critical Principles

### Rust is Source of Truth

All order calculation logic lives in `passivbot-rust/src/`. Both live bot and backtester use the same Rust code via PyO3 bindings.

- **Behavior changes** (order logic, risk, unstuck) → modify Rust
- **Python patches** to order logic → not acceptable
- After Rust changes: `cd passivbot-rust && maturin develop --release && cd ..`

### Stateless Design

Bot must behave identically after restart. Never rely on "what happened earlier" unless it can be rederived from exchange state.

- No local caches that change behavior (performance-only caches OK)
- Minimal time-based heuristics outside natural candle boundaries

### Fail Loudly

Prefer clear exceptions over silent error handling.

- Exchange fetch methods must NOT catch exceptions
- Let errors propagate to caller who handles via `restart_bot_on_too_many_errors()`
- Include actionable error messages

### Avoid Over-Engineering

Only make changes that are directly requested or clearly necessary.

- Don't add features, refactor, or "improve" beyond what was asked
- Don't add docstrings, comments, or type hints to unchanged code
- Three similar lines beats a premature abstraction

## Key Conventions

### Terminology

| Concept | Names Used |
|---------|------------|
| Position side (long/short) | `position_side`, `pos_side`, `pside` |
| Order side (buy/sell) | `side`, `order_side` |

**Don't confuse them.** A long position has both buy entries and sell closes.

### Signed Quantities

- `qty` and `pos_size` are signed: positive = long/buy, negative = short/sell
- Exception: final exchange payload may need `abs(qty)`

### EMA Spans

EMA spans are floats. Don't round intermediate calculations.

```python
# WRONG
span2 = int(sqrt(span0 * span1))

# CORRECT
span2 = sqrt(span0 * span1)
```

See [docs/ai/principles.yaml](docs/ai/principles.yaml) for full conventions.

## Before You Code

1. **Check [docs/ai/pitfalls.md](docs/ai/pitfalls.md)** for common mistakes
2. **Check [docs/ai/exchange_api_quirks.md](docs/ai/exchange_api_quirks.md)** if touching exchange code
3. **Check [docs/ai/features/](docs/ai/features/)** if a feature doc exists for your area
4. **Read [docs/ai/principles.yaml](docs/ai/principles.yaml)** for conventions

## Architecture at a Glance

```
Python (src/)                          Rust (passivbot-rust/src/)
├── passivbot.py (live loop)    ────►  ├── orchestrator.rs (order calc)
├── backtest.py (coordinator)   ────►  ├── backtest.rs (simulation)
├── optimize.py (genetic algo)  ────►  └── analysis.rs (metrics)
├── candlestick_manager.py (OHLCV)
├── fill_events_manager.py (PnL)
└── exchanges/ (API wrappers)
```

See [docs/ai/architecture.md](docs/ai/architecture.md) for detailed component descriptions.

## Configuration Hierarchy

| Section | Used By | Purpose |
|---------|---------|---------|
| `config.live` | Live, Backtest, Optimizer | Runtime behavior (order logic, risk) |
| `config.backtest` | Backtest, Optimizer | Simulation settings (dates, balance) |
| `config.optimize` | Optimizer | Optimization settings (bounds, population) |

**Rule**: When in doubt, prefer `config.live`.

## Logging

| Level | Audience | Content |
|-------|----------|---------|
| INFO | Operators | Essential events (orders, fills, positions) |
| DEBUG | Developers | Decision context, API timing |
| TRACE | Deep debugging | Full payloads, per-item iterations |

Use `[tag]` format: `[order]`, `[fill]`, `[pos]`, `[health]`

See [docs/ai/logging_guide.md](docs/ai/logging_guide.md) for detailed guidelines.

## Documentation Index

| File | When to Read |
|------|--------------|
| [docs/ai/principles.yaml](docs/ai/principles.yaml) | Always (core conventions) |
| [docs/ai/architecture.md](docs/ai/architecture.md) | New to codebase |
| [docs/ai/commands.md](docs/ai/commands.md) | Need to run/test something |
| [docs/ai/pitfalls.md](docs/ai/pitfalls.md) | Before implementing |
| [docs/ai/exchange_api_quirks.md](docs/ai/exchange_api_quirks.md) | Working on exchange code |
| [docs/ai/logging_guide.md](docs/ai/logging_guide.md) | Working on logging |
| [docs/ai/decisions.md](docs/ai/decisions.md) | Understanding "why" |
| [docs/ai/features/](docs/ai/features/) | Working on specific features |

Full index: [docs/ai/README.md](docs/ai/README.md)

## Testing

```bash
pytest                                    # All tests
pytest tests/test_specific.py             # Specific file
pytest tests/test_specific.py::test_name  # Specific test
```

Write tests for both normal and edge cases. Include property-based tests where applicable.

## Branch Context

Current branch: `feature/stock-perps-hyperliquid`

Recent work:
- Stock perpetuals (HIP-3) support for Hyperliquid
- Combined mode: crypto + stock perps with automatic margin mode
- Symbol normalization and routing

## Common Gotchas

- **Rust changes not reflected**: Run `cd passivbot-rust && maturin develop --release && cd ..`
- **Cache issues**: Delete `caches/ohlcv/{exchange}/{symbol}.parquet`
- **Lock contention**: Check for orphan `.lock` files in `caches/`
- **Stock perps on wrong exchange**: Only Hyperliquid supports HIP-3 markets

## Changelog

Maintain `CHANGELOG.md` for user-facing changes. Add entries under "Unreleased" as changes land.
