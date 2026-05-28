# Development Commands

Use from repo root unless noted.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .              # live-only
# or: python3 -m pip install -e ".[full]"
# or: python3 -m pip install -e ".[dev]"
```

## Tests

```bash
pytest
pytest tests/test_specific.py
pytest tests/test_specific.py::test_name
```

## Live Bot

```bash
passivbot live -u {account_name}
passivbot live path/to/config.json --debug-level {0-3}
```

## Backtest

```bash
passivbot backtest path/to/config.json
passivbot backtest --suite
```

## Optimize

```bash
passivbot optimize path/to/config.json
passivbot optimize --suite
```

## Rust

```bash
cd passivbot-rust && maturin develop --release && cd ..
cd passivbot-rust && cargo test --no-default-features && cd ..
cd passivbot-rust && cargo check --tests && cd ..
```

## Useful Tools

```bash
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto-compress optimize_results/.../pareto 8 --output-dir selected_pareto_8
passivbot tool pareto-dash --data-root optimize_results
passivbot tool verify-hlcvs-data
passivbot tool streamline-json configs/examples/default_trailing_martingale_long_npos4.json
```

## High-Signal Gotcha

If Rust changes seem ignored by tests/runtime, rebuild extension again before debugging behavior.
