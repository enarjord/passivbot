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
- Silent error handling is forbidden in critical paths (exchange data, EMA computation, risk, order construction)

#### Forbidden Patterns (Critical Paths)

- `except Exception: pass`, `except Exception: continue`, or returning neutral defaults after catching errors
- `asyncio.gather(..., return_exceptions=True)` when exceptions are then ignored/dropped
- `dict.get(required_key, safe_default)` for required trading inputs
- `value = required_value or safe_default` for required trading inputs
- Rust `unwrap_or(...)` / `unwrap_or_default()` for required config or required trading inputs
- Converting required-input failures into `0.0`, `None`, `{}`, `[]`, or `False` without an explicit, documented fallback policy

#### Config Defaults Ownership

- Defaults are defined once in centralized config loading/formatting.
- Do not re-apply defaults in runtime paths (live loop, backtest prep, Rust parsers, orchestrator inputs).
- For required config fields, enforce presence and type at point-of-use (`require_config_value` / `extract_value`) and raise on missing/invalid.
- Only optional fields may use optional access patterns, and optionality must be explicit in schema/docs.

#### Required Error Contract

- Required inputs must be complete before handing data to Rust orchestrator/backtester
- If a required input fetch fails, either:
  - raise immediately with full context; or
  - use an explicitly allowed fallback with warning logs that include reason and context
- If fallback is unavailable, raise immediately (do not degrade silently)

#### Fallback Policy Matrix

| Path/Input | Default Policy | Allowed Fallback | If Fallback Used |
|------------|----------------|------------------|------------------|
| Exchange fetch methods (`fetch_balance`, `fetch_positions`, `fetch_open_orders`, etc.) | Raise | None | N/A |
| Required EMA inputs for orchestrator | Raise | Reuse previous EMA for same `symbol/span` only when explicitly implemented for that path | Log `[ema]` warning with symbol/span/reason/age/count and test fallback behavior |
| Risk-gating inputs (loss caps, exposure limits, position state) | Raise | None unless explicitly approved by user in task | Log explicit warning + add regression test |
| Any other trading-critical required field | Raise | None unless documented in feature docs + approved in task | Log explicit warning + add regression test |

#### Critical Path Rule

- In exchange data, EMA, risk, and order paths: hard-fail is the default.
- Any exception to hard-fail must be explicitly requested/approved in the current task and covered by tests.

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
5. **Run a silent-handling self-audit** for touched files:

```bash
rg -n "except Exception|return_exceptions=True|\\.get\\([^\\n]*,\\s*(0|0\\.0|None|False|\\{\\}|\\[\\])\\)" src tests
rg -n "unwrap_or\\(|unwrap_or_default\\(" passivbot-rust/src
```

6. **If matches are present in changed code**, either:
   - remove them; or
   - justify them explicitly in PR notes and add targeted tests.

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

Fallbacks in trading-critical paths must log a warning with a stable tag and context fields:
- Recommended tag: `[ema]` for EMA fallback, `[risk]` for risk fallback
- Required fields: `symbol`, `span`/parameter, `reason`, fallback value source, `age_ms` (if applicable), consecutive fallback count

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

For any new fallback behavior, tests must cover:
- fallback is actually used when primary input fails
- warning/visibility behavior (or explicit failure message content)
- hard failure when fallback input is unavailable
- explicit guard against unsafe substitution (e.g., last-price substitution for EMA)

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
