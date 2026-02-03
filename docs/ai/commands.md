# Development Commands

Quick reference for common development tasks.

## Setup

```bash
# Install Rust (required for order calculation engine)
# Visit https://www.rust-lang.org/tools/install

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build Rust extensions (usually automatic)
cd passivbot-rust
maturin develop --release
cd ..
```

## Running the Live Bot

```bash
# Start with account name from api-keys.json
python3 src/main.py -u {account_name}

# Start with custom config
python3 src/main.py path/to/config.json

# Adjust logging verbosity
python3 src/main.py path/to/config.json --debug-level {0-3}
# 0=warnings, 1=info, 2=debug, 3=trace
```

## Testing

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

## Backtesting

```bash
# Run with default config
python3 src/backtest.py

# Run with specific config
python3 src/backtest.py path/to/config.json

# Run suite scenarios (multiple date ranges, coin sets)
python3 src/backtest.py --suite

# Disable individual coin plotting
python3 src/backtest.py -dp

# Adjust logging
python3 src/backtest.py --debug-level 2
```

Output goes to `backtests/{exchange}/timestamp/` (standalone) or `backtests/suite_runs/` (suite mode).

## Optimization

```bash
# Run with default config
python3 src/optimize.py

# Run with specific config
python3 src/optimize.py configs/template.json

# Start from existing configs (warm start)
python3 src/optimize.py --start configs/starting_pool/

# Fine-tune specific parameters only
python3 src/optimize.py -ft long_entry_grid_spacing_pct,long_entry_initial_qty_pct

# Run with suite scenarios
python3 src/optimize.py --suite
```

Output goes to `optimize_results/YYYY-MM-DDTHH_MM_SS_{exchanges}_{n_days}days_{coin_label}_{hash}/`

## Analysis Tools

```bash
# Interactive Pareto dashboard (recommended for reviewing optimization results)
python3 src/tools/pareto_dash.py --data-root optimize_results

# Generate market-cap based coin list
python3 src/tools/generate_mcap_list.py -n 80 -m 200 -e binance,bybit -o configs/approved_coins.json

# Verify cached OHLCV data integrity
python3 src/tools/verify_hlcvs_data.py

# Normalize JSON configs (consistent formatting)
python3 src/tools/streamline_json.py configs/template.json
```

## Jupyter Lab

```bash
# Launch from passivbot root with venv activated
python3 -m jupyter lab
```

## Rust Development

```bash
# Rebuild Rust extensions after changes
cd passivbot-rust
maturin develop --release
cd ..

# Run Rust tests
cd passivbot-rust
cargo test
cd ..

# Check Rust without building
cd passivbot-rust
cargo check
cd ..
```

## Common Gotchas

### Rust changes not taking effect
```bash
# Recompile Rust extensions
cd passivbot-rust && maturin develop --release && cd ..
```

### Cache issues
```bash
# Clear OHLCV cache for a symbol
rm caches/ohlcv/{exchange}/{symbol}.parquet

# Clear all caches (use with caution)
rm -rf caches/
```

### Orphan lock files
```bash
# Find stale lock files
find caches -name "*.lock"

# Remove if stale (normally auto-cleaned after 10 min)
rm caches/**/*.lock
```
