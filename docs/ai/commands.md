# Development Commands

Use from repo root unless noted.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd passivbot-rust && maturin develop --release && cd ..
```

## Tests

```bash
pytest
pytest tests/test_specific.py
pytest tests/test_specific.py::test_name
```

## Live Bot

```bash
python3 src/main.py -u {account_name}
python3 src/main.py path/to/config.json --debug-level {0-3}
```

## Backtest

```bash
python3 src/backtest.py path/to/config.json
python3 src/backtest.py --suite
```

## Optimize

```bash
python3 src/optimize.py path/to/config.json
python3 src/optimize.py --suite
```

## Rust

```bash
cd passivbot-rust && maturin develop --release && cd ..
cd passivbot-rust && cargo test && cd ..
cd passivbot-rust && cargo check --tests && cd ..
```

## Useful Tools

```bash
python3 src/tools/pareto_dash.py --data-root optimize_results
python3 src/tools/verify_hlcvs_data.py
python3 src/tools/streamline_json.py configs/template.json
```

## High-Signal Gotcha

If Rust changes seem ignored by tests/runtime, rebuild extension again before debugging behavior.
