# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT:** Also read and follow the documentation in `docs/ai/`:
- `passivbot_agent_principles.yaml` - Conventions on terminology, error handling, testing, design principles
- `exchange_api_quirks.md` - Known exchange API limitations and workarounds (check before implementing exchange code)
- `debugging_case_studies.md` - Detailed debugging sessions as reference for complex investigations
- `log_analysis_prompt.md` - Logging level definitions and examples

## Known LLM Pitfalls — Avoid These

Common failure modes when coding with LLMs (ref: Karpathy, Dec 2025). Actively counteract:

- **Unchecked assumptions** — Don't silently assume intent; verify ambiguous requirements or ask
- **Hiding confusion** — Surface uncertainty explicitly; seek clarification rather than guessing
- **Ignoring inconsistencies** — Flag contradictions in specs/code; don't silently pick one interpretation
- **Concealing tradeoffs** — Present options with pros/cons when multiple valid approaches exist
- **Failure to push back** — Disagree when a request seems wrong, suboptimal, or underspecified
- **Sycophancy** — No cheerleading ("Great question!"); be direct, factual, and professionally objective
- **Overengineering** — Prefer the simplest solution; don't add abstraction layers, config options, or generality until justified by actual need
- **Abstraction bloat** — Resist premature DRY; three similar lines beats a premature helper
- **Dead code accumulation** — Clean up anything you obsolete; don't leave commented-out code or unused imports
- **Scope creep in edits** — Don't "improve" code orthogonal to the task; don't add docstrings, type hints, or refactors unless requested
- **Runaway implementation** — Stop and reconsider if a solution is growing large; ask "couldn't this be simpler?" before writing 500+ lines

## Overview

Passivbot is a cryptocurrency trading bot for perpetual futures markets. It uses a contrarian market-making strategy inspired by Martingale betting, written in Python for orchestration and Rust for performance-critical components (backtesting, order calculations, analysis).

## Development Commands

### Setup
```bash
# Install Rust (required)
# Visit https://www.rust-lang.org/tools/install

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build Rust extensions (usually automatic, but can be done manually)
cd passivbot-rust
maturin develop --release
cd ..
```

### Running
```bash
# Start live bot
python3 src/main.py -u {account_name_from_api-keys.json}
# or with custom config
python3 src/main.py path/to/config.json

# Adjust logging verbosity
python3 src/main.py path/to/config.json --debug-level {0-3}
# 0=warnings, 1=info, 2=debug, 3=trace
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_candlestick_manager.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_candlestick_manager.py::test_name
```

### Backtesting
```bash
# Run backtest with default config
python3 src/backtest.py

# Run with specific config
python3 src/backtest.py path/to/config.json

# Run suite scenarios
python3 src/backtest.py --suite

# Disable individual coin plotting
python3 src/backtest.py -dp

# Adjust logging
python3 src/backtest.py --debug-level 2
```

### Optimization
```bash
# Run optimizer with default config
python3 src/optimize.py

# Run with specific config
python3 src/optimize.py configs/template.json

# Start from existing configs
python3 src/optimize.py --start configs/starting_pool/

# Fine-tune specific parameters only
python3 src/optimize.py -ft long_entry_grid_spacing_pct,long_entry_initial_qty_pct

# Run with suite scenarios
python3 src/optimize.py --suite
```

### Analysis Tools
```bash
# Interactive Pareto dashboard (recommended)
python3 src/tools/pareto_dash.py --data-root optimize_results

# Generate market-cap based coin list
python3 src/tools/generate_mcap_list.py -n 80 -m 200 -e binance,bybit -o configs/approved_coins.json

# Verify cached OHLCV data
python3 src/tools/verify_hlcvs_data.py

# Normalize JSON configs
python3 src/tools/streamline_json.py configs/template.json
```

### Jupyter Lab
```bash
# Launch from passivbot root with venv activated
python3 -m jupyter lab
```

## Architecture

### Python-Rust Division

**Critical principle: Rust is the source of truth.**

- **Rust (`passivbot-rust/src/`)**: All order calculation logic, backtesting engine, and analysis metrics live here
  - `orchestrator.rs`: Core order calculation engine (replaces legacy Python paths)
  - `backtest.rs`: Backtesting engine that replays historical candles
  - `analysis.rs`: Performance metrics computation
  - `entries.rs`, `closes.rs`: Grid and trailing entry/close logic
  - `risk.rs`: Wallet exposure enforcement, unstuck calculations
  - `coin_selection.rs`: Forager mode coin filtering by volatility
  - `python.rs`: PyO3 bindings exposing Rust functions to Python

- **Python (`src/`)**: Orchestration, exchange communication, configuration, data management
  - Live bot talks to exchanges via CCXT, then calls Rust for order decisions
  - Backtester prepares data, calls Rust backtest engine, plots results
  - Optimizer manages genetic algorithm, evaluates candidates via Rust backtester

**When implementing changes:**
- Behavior changes (order logic, unstuck, risk) belong in Rust
- If fixing a bug in Python order calculation, port the fix to Rust instead
- Python patches to order logic are not acceptable; both live bot and backtester must use identical Rust code

### Core Components

#### CandlestickManager (`src/candlestick_manager.py`)
- Fetches and caches 1m OHLCV data from exchanges
- On-disk cache in `caches/ohlcv/` with Parquet storage
- Handles multiple exchange data sources (Binance, Bybit, Bitget, KuCoin archives + CCXT fallback)
- Shared between live bot, backtester, and optimizer
- Uses per-symbol file locks with automatic stale cleanup for multi-process safety
- EMA warm-up calculation for indicators

#### Passivbot Live (`src/passivbot.py`)
- Main live trading loop
- Fetches positions, balances, open orders from exchange
- Calls Rust orchestrator via `pbr.calc_ideal_orders_orchestrator()` to get desired orders
- Reconciles ideal vs actual orders, creates/cancels as needed
- Manages coin approval, forced modes, graceful stops
- Runs asynchronously with exchange websocket feeds

#### Backtester (`src/backtest.py`)
- Replays historical 1m candles through Rust backtest engine
- Supports single-coin and multi-coin backtests
- Suite mode: evaluate multiple scenarios (date ranges, coin sets, exchanges) in one run
- Outputs metrics, plots, fills CSVs to `backtests/`
- Can combine OHLCV from multiple exchanges ("best feed per coin")

#### Optimizer (`src/optimize.py`)
- Uses NSGA-II genetic algorithm for multi-objective optimization
- Evaluates thousands of config candidates via Rust backtester
- Maintains Pareto front of non-dominated solutions
- Supports suite scenarios (optimize across multiple market conditions)
- Outputs to `optimize_results/` with binary results log and Pareto JSON members
- Shared-memory datasets for reduced RAM usage across worker processes

#### Configuration (`src/config_utils.py`)
- Loads and validates JSON/HJSON configs
- Merges CLI arguments with config files
- Handles coin overrides (per-symbol parameter tweaks)
- Normalizes approved/ignored coin lists (can reference external files)
- Template config: `configs/template.json`

### Key Data Flows

1. **Live Trading Loop:**
   - Fetch exchange state (positions, balances, orders, candles)
   - Update CandlestickManager cache
   - Compute EMAs and volatility indicators (Python)
   - Call Rust orchestrator with full state → get ideal orders
   - Diff ideal vs open orders, execute changes
   - Sleep, repeat

2. **Backtesting:**
   - CandlestickManager fetches/caches OHLCV for requested coins + date range
   - If suite mode: prepare unified dataset across all scenarios
   - Call Rust `backtest_single_coin` or `backtest_multi_coin` with candle data + config
   - Rust engine simulates fills, updates balances, computes metrics
   - Python receives results, plots equity curves, writes analysis JSON

3. **Optimization:**
   - Load base config and bounds
   - Genetic algorithm generates candidate configs
   - Each candidate → full backtest via Rust engine
   - Fitness = multi-objective score from metrics (e.g., `mdg`, `sharpe_ratio`)
   - Apply penalties for violated limits
   - NSGA-II selection keeps Pareto front, mutates/crosses for next generation

### Multiprocessing & Shared Memory (Optimizer)

The optimizer uses `multiprocessing.Pool` with SharedMemory for parallel backtest evaluation.

**SharedMemory Allocation**
- Create SharedMemory ONCE for master datasets, not per-scenario/per-worker
- SharedMemory is shared across all workers (good), but each allocation consumes physical RAM
- N scenarios × dataset_size = disaster; master_dataset + indices = efficient

**Numpy Views vs Copies**
- **Basic slicing creates VIEWS** (O(1) memory): `array[start:end]`, `array[::2]`
- **Fancy indexing creates COPIES**: `array[:, [0, 3, 5], :]`, `array[indices]`
- When subsetting coins by index, you WILL get a copy - plan for it

**Memory-Efficient Data Flow**
1. Keep master data in SharedMemory (workers attach, not copy)
2. Use time slicing for views (basic slicing = no copy)
3. Pass coin indices downstream; let the lowest level create ONE copy
4. Don't pre-slice then pass to functions that will copy again (double-copy trap)

```python
# BAD: Double copy
slice = master[time_start:time_end, coin_indices, :]  # Copy #1 (fancy indexing)
payload = build_payload(slice)  # Copy #2 inside (ascontiguousarray)

# GOOD: Single copy
time_view = master[time_start:time_end]  # View (O(1))
payload = build_payload(time_view, coin_indices=coin_indices)  # Copy #1 inside
```

**Don't Rely on Garbage Collection**
- `del` doesn't guarantee immediate memory release
- `gc.collect()` is a bandaid, not a solution
- Design data flow so copies are created and freed within tight scopes
- Rust-owned data (via PyO3) is freed deterministically when dropped

**Worker Memory Model**
- Each worker is a separate process (fork or spawn)
- Python objects passed to workers are pickled → unpickled (copied)
- SharedMemory segments are attached by name (no copy, shared physical RAM)
- Contexts/configs are small (OK to copy); large arrays should use SharedMemory

### Important Conventions

**See `docs/ai/passivbot_agent_principles.yaml` for the full list.** Key points:

- **Position side** (long/short) → `[position_side, pos_side, pside, PositionSide, PosSide, Pside]`
- **Order side** (buy/sell) → `[side, order_side, Side, OrderSide]`
- **Signed quantities:** `buy qty` and `long pos_size` are positive; `sell qty` and `short pos_size` are negative (exception: final exchange payload may need `abs(qty)`)
- **EMA spans are floats:** Do not round derived spans (e.g., `span2 = sqrt(span0 * span1)`)
- **Effective min qty/cost:** Entries must observe effective min qty; closes observe effective min cost (unless pos size < min qty, then close qty = pos size)
- **OHLCV volume is base volume:** CandlestickManager normalizes CCXT OHLCV volume to base units. Some exchanges (e.g., GateIO swap) return quote volume from CCXT; this is converted to base on ingest.
- **CLI tools import path:** Tools under `src/tools/` should prepend the repo `src/` to `sys.path` (e.g., `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\")))`) so they can import project modules when run as scripts.

### Stateless Design

- **Never rely on "what happened earlier"** unless it can be rederived from exchange/state snapshot on startup
- **No local ad-hoc caches** that would break on restart (exception: performance-only caches that don't change behavior)
- **Minimal time-based heuristics** outside natural candle boundaries

### Error Handling

- **Fail loudly:** Prefer exceptions/panics over silent error handling
- **Assume upstream is correct:** Don't silently compensate for bad data
- **Clear, actionable error messages** for expected failures
- **Exchange fetch methods** (`fetch_balance`, `fetch_positions`, `fetch_open_orders`, etc.) must NOT catch exceptions. Let exceptions propagate to the caller who handles via `restart_bot_on_too_many_errors()`. This ensures clean return types (no `Union[list, bool]`), preserves exception context, and makes errors impossible to ignore.

### Logging

Use structured, leveled logging with clear separation between levels:

| Level | Audience | Content | Golden Rule |
|-------|----------|---------|-------------|
| **INFO** | Operators | Essential events: orders, fills, positions, balance, mode changes, health summaries | Must be sustainable to tail indefinitely in production |
| **DEBUG** | Developers | Internal state: API timing, cache updates, decision points, fetch summaries | Tolerable for short debugging sessions |
| **TRACE** | Deep debugging | Full firehose: API payloads, per-item iterations, raw data | Expect GB of logs; enable briefly for specific issues |

**Guidelines:**
- INFO should answer "what is the bot doing?" without overwhelming
- DEBUG should answer "why did it make that decision?"
- TRACE should answer "what exact data did it see?"
- Use `[tag]` format consistently: `[order]`, `[pos]`, `[fill]`, `[health]`, `[boot]`
- Prefer `logging.info("msg %s", var)` over f-strings for log aggregation compatibility
- Every remote API call should have a DEBUG-level log entry (endpoint, timing)
- See `docs/ai/log_analysis_prompt.md` for detailed level definitions and examples

## Testing

- Tests in `tests/` cover optimization, backtesting, config handling, candlestick management
- Write comprehensive tests for both normal and edge cases
- Include property-based or randomized tests where applicable
- Before committing changes, ensure tests pass

## Branch Context

Current branch: `refactor/merge-downloader-into-cm`
- Merged downloader functionality into CandlestickManager
- Removed legacy order calculation paths (orchestrator-only)
- Removed hedging overlay

## Configuration Files

- `configs/template.json`: Default config with all parameters documented
- `api-keys.json`: API credentials (copy from `api-keys.json.example`)
- Configs support HJSON (JSON with comments)
- Coin overrides allow per-symbol parameter tweaks
- Suite scenarios enable multi-condition evaluation

### Config Section Hierarchy

Configuration sections form an inheritance hierarchy. When adding new parameters, place them in the section that matches which components need to read them:

| Section | Used By | Purpose |
|---------|---------|---------|
| `config.live` | Live bot, Backtester, Optimizer | Runtime behavior params shared across all modes (order logic, risk, hedge_mode) |
| `config.backtest` | Backtester, Optimizer | Simulation-specific params (date ranges, starting_balance, data sources) |
| `config.optimize` | Optimizer only | Optimization process params (population_size, bounds, fitness settings) |

**Rule:** When in doubt, prefer `config.live`. A parameter that works in both live and backtest belongs in `config.live`, not `config.backtest`.

## Common Gotchas

- If Rust changes are detected, recompile with `cd passivbot-rust && maturin develop --release && cd ..`
- Multiple bot instances can share OHLCV cache; CandlestickManager uses self-healing locks
- Backtest results go to `backtests/{exchange}/timestamp/` (standalone) or `backtests/suite_runs/` (suite mode)
- Optimizer results go to `optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{n_days}days_{coin_label}_{hash}/`
- When adding dependencies, explain necessity and impact
- Before committing, simulate/dry-run changes

## Documentation Structure

This file (`CLAUDE.md`) serves as the entry point for AI agents. Detailed topic-specific documentation lives in `docs/ai/`:

| File | Purpose |
|------|---------|
| `passivbot_agent_principles.yaml` | Core conventions: terminology, error handling, testing |
| `exchange_api_quirks.md` | Exchange-specific API limitations and workarounds |
| `debugging_case_studies.md` | Detailed debugging sessions as learning references |
| `log_analysis_prompt.md` | Logging level definitions and analysis guidance |

**When to add new docs:**
- **Exchange quirks** → Add to `exchange_api_quirks.md` (or create `{exchange}_quirks.md` if extensive)
- **Complex debugging** → Add case study to `debugging_case_studies.md`
- **New subsystem** → Consider a dedicated `{subsystem}.md` if >50 lines of guidance

**Modularization guidelines:**
- Keep CLAUDE.md as a high-level overview (<300 lines ideal)
- Move detailed reference material to `docs/ai/` subdirectory
- Use consistent naming: `{topic}.md` or `{topic}_{subtopic}.md`
- Always reference new docs from CLAUDE.md's header list

## Changelog

Maintain `CHANGELOG.md` as single source of truth for user-facing changes. Add entries under "Unreleased" as changes land; move to dated version heading when tagging releases.
