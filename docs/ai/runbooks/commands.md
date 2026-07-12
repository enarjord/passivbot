# Development And Validation Commands

Use from repo root unless noted.

Commands are grouped by authority tier. Local tests, local fake-live scenarios, and backtests with
already cached data are offline. Downloads and public probes contact unauthenticated network
endpoints. Live commands and account probes may use credentials and require explicit approval under
`AGENTS.md`.

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

## Live Bot (Explicit Approval Required)

```bash
passivbot live -u {account_name}
passivbot live path/to/config.json --log-level {warning|info|debug|trace|0-3}
```

Do not use these as smoke-test commands. They may load private keys, configure an account, and
create or cancel orders.

## Offline Fake-Live Harness

```bash
PYTHONPATH=src python src/tools/run_fake_live.py \
  configs/fake_live_hsl_btc.hjson \
  scenarios/fake_live/hsl_long_red_restart.hjson \
  --user fake_hsl_restart_test
pytest tests/test_run_fake_live.py -m fake_live
```

The `fake` exchange uses only local scenario state. Do not describe authenticated testnet, demo,
sandbox, or paper-trading sessions as this offline fake-live tier.

## Backtest

```bash
passivbot backtest path/to/config.json
passivbot backtest --suite
```

## Optimize

```bash
passivbot optimize path/to/config.json
passivbot optimize --suite
passivbot optimize path/to/config.json -t path/to/anchor_configs -ft long.forager,long.risk
```

When `-ft/--fine-tune-params` and `-t/--start` are used together, the start configs are
fine-tune anchors: non-tuned optimizer-bound bot params are fixed from the selected anchor and
only the fine-tune selectors are optimized. Runtime policy from the base config still wins, and
seed/anchor values outside `optimize.bounds` are clamped with aggregated source/key logging.

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
passivbot tool ohlcvs-doctor --repair-catalog
passivbot tool streamline-json configs/examples/default_trailing_martingale_long.json
```

## High-Signal Gotcha

If Rust changes seem ignored, follow `rust_extension.md` to verify the loaded artifact and source
stamp before debugging behavior.
