# Passivbot Architecture

Passivbot is a cryptocurrency trading bot for perpetual futures markets. It uses a contrarian market-making strategy inspired by Martingale betting.

## Python-Rust Division

**Critical principle: Rust is the source of truth.**

### Rust (`passivbot-rust/src/`)

All order calculation logic, backtesting engine, and analysis metrics:

| Module | Purpose |
|--------|---------|
| `orchestrator.rs` | Core order calculation engine |
| `backtest.rs` | Backtesting engine (replays historical candles) |
| `analysis.rs` | Performance metrics computation |
| `entries.rs` | Grid and trailing entry logic |
| `closes.rs` | Grid and trailing close logic |
| `risk.rs` | Wallet exposure enforcement, unstuck calculations |
| `coin_selection.rs` | Forager mode coin filtering by volatility |
| `python.rs` | PyO3 bindings exposing Rust to Python |

### Python (`src/`)

Orchestration, exchange communication, configuration, data management:

| Component | Purpose |
|-----------|---------|
| `passivbot.py` | Main live trading loop |
| `backtest.py` | Prepares data, calls Rust engine, plots results |
| `optimize.py` | Genetic algorithm, evaluates candidates via Rust |
| `candlestick_manager.py` | OHLCV fetching and caching |
| `fill_events_manager.py` | Fill tracking and PnL computation |
| `exchanges/` | Exchange-specific implementations |

### Implementation Rule

When implementing changes:
- **Behavior changes** (order logic, unstuck, risk) belong in Rust
- **Bug fixes** in order calculation must be ported to Rust
- Python patches to order logic are **not acceptable**
- Both live bot and backtester must use identical Rust code

## Core Components

### CandlestickManager

Fetches and caches 1m OHLCV data from exchanges.

- On-disk cache in `caches/ohlcv/` with Parquet storage
- Handles multiple exchange data sources
- Shared between live bot, backtester, and optimizer
- Per-symbol file locks for multi-process safety

**Code**: `src/candlestick_manager.py`

### Passivbot Live

Main live trading loop.

1. Fetch positions, balances, open orders from exchange
2. Update candle cache, compute EMAs
3. Call Rust orchestrator → get ideal orders
4. Reconcile ideal vs actual orders
5. Create/cancel orders as needed
6. Sleep, repeat

**Code**: `src/passivbot.py`

### Backtester

Replays historical candles through Rust engine.

- Single-coin and multi-coin backtests
- Suite mode: evaluate multiple scenarios in one run
- Outputs to `backtests/`

**Code**: `src/backtest.py`

### Optimizer

Uses NSGA-II genetic algorithm for multi-objective optimization.

- Evaluates thousands of candidates via Rust backtester
- Maintains Pareto front of non-dominated solutions
- Outputs to `optimize_results/`
- Shared-memory datasets for reduced RAM

**Code**: `src/optimize.py`

### Configuration

Loads and validates JSON/HJSON configs.

- Merges CLI arguments with config files
- Handles coin overrides (per-symbol tweaks)
- Normalizes approved/ignored coin lists

**Code**: `src/config_utils.py`

## Key Data Flows

### Live Trading Loop

```
Exchange State       →  CandlestickManager  →  EMA Computation
       ↓                                              ↓
Positions, Balances  →  Rust Orchestrator   ←  Config + EMAs
       ↓                       ↓
Open Orders          →  Order Reconciliation
       ↓                       ↓
                     Create/Cancel Orders
```

### Backtesting

```
Config + Date Range  →  CandlestickManager  →  Candle Data
       ↓                                            ↓
                     Rust Backtest Engine    ←  Config
                            ↓
                     Simulated Fills
                            ↓
                     Metrics + Plots
```

### Optimization

```
Base Config + Bounds  →  Genetic Algorithm
       ↓                       ↓
                     Generate Candidates
                            ↓
                     Rust Backtest (each)
                            ↓
                     Fitness Scores
                            ↓
                     NSGA-II Selection
                            ↓
                     Next Generation
```

## Configuration Hierarchy

Configuration sections form an inheritance hierarchy:

| Section | Used By | Purpose |
|---------|---------|---------|
| `config.live` | Live bot, Backtester, Optimizer | Runtime behavior (order logic, risk, hedge_mode) |
| `config.backtest` | Backtester, Optimizer | Simulation-specific (date ranges, starting_balance) |
| `config.optimize` | Optimizer only | Optimization process (population_size, bounds) |

**Rule**: When in doubt, prefer `config.live`. A parameter that works in both live and backtest belongs in `config.live`, not `config.backtest`.

### Config Files

| File | Purpose |
|------|---------|
| `configs/template.json` | Default config with all parameters |
| `api-keys.json` | API credentials |

Configs support HJSON (JSON with comments).

## Output Directories

| Directory | Contents |
|-----------|----------|
| `caches/ohlcv/` | Cached OHLCV data (Parquet) |
| `caches/{exchange}/` | Exchange-specific caches |
| `backtests/` | Backtest results and plots |
| `optimize_results/` | Optimization results and Pareto members |

## Stateless Design

- **Never rely on "what happened earlier"** unless rederivable from exchange state
- **No local ad-hoc caches** that change behavior (performance-only caches OK)
- **Bot must behave identically after restart**
