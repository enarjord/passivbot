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
  tests/fixtures/fake_live_hsl.json \
  scenarios/fake_live/hsl_long_red_restart.hjson \
  --user fake_hsl_restart_test
pytest tests/test_run_fake_live.py -m fake_live
```

The `fake` exchange uses only local scenario state. Do not describe authenticated testnet, demo,
sandbox, or paper-trading sessions as this offline fake-live tier.

## Backtest

```bash
passivbot backtest path/to/config.json
passivbot backtest configs/examples/default_trailing_martingale_long.json -s BTC --suite n
```

## Optimize

```bash
passivbot optimize path/to/config.json
passivbot optimize configs/examples/default_trailing_martingale_long.json --suite
passivbot optimize path/to/config.json -t path/to/anchor_configs -ft long.forager,long.risk
```

Prepare the exact optimizer data request before starting search:

```bash
passivbot download path/to/config.json --for-optimize --plan
passivbot download path/to/config.json --for-optimize --manifest prepared-inputs.json
```

The plan reads existing metadata only and leaves candle coverage and unresolved
market identities explicitly unknown. Preparation uses the optimizer's suite and
maximum searched warmup without evaluating candidates. The manifest records inputs;
it is not saved fitness or an alternative to evaluator compatibility checks. See
[`../../backtesting.md`](../../backtesting.md#pre-download-ohlcv) for offline preparation.

When `-ft/--fine-tune-params` and `-t/--start` are used together, the start configs are
fine-tune anchors: non-tuned optimizer-bound bot params are fixed from the selected anchor and
only the fine-tune selectors are optimized. Runtime policy from the base config still wins, and
seed/anchor values outside `optimize.bounds` are clamped with aggregated source/key logging.

CPU optimizers evaluate every deduplicated `--start` config with Rust CPU backtests.
The native GPU optimizer fully evaluates every deduplicated seed through its GPU service
before population trimming, without CPU validation. Seed evaluations are additional to
`optimize.iters`; suite scenario screening applies only to later offspring.
Explicitly enabled config prefilters may complete seeds or offspring with penalty-only evaluations
before replay. See `../../config_prefilter.md`; all checks are disabled by default.

## Rust

Follow the [Rust extension runbook](rust_extension.md) for the locked build,
source-fingerprint check, and both Cargo test modes.

## Useful Tools

```bash
passivbot tool pareto optimize_results/.../pareto
passivbot tool pareto optimize_results/.../pareto -s configs/promoted_candidate.local.json
passivbot tool pareto optimize_results/.../pareto -l 'cagr_fraction_strategy_equity>0.0' -f optimize_results/filtered_pareto
passivbot tool pareto-compress optimize_results/.../pareto 8 --output-dir selected_pareto_8
passivbot tool pareto-dash --data-root optimize_results
passivbot tool verify-hlcvs-data
passivbot tool ohlcvs-doctor --repair-catalog
passivbot tool streamline-json configs/examples/default_trailing_martingale_long.json
passivbot tool clean-config configs/private/input.json configs/private/clean.json
passivbot tool clean-config configs/private/input.json configs/private/live.json --mode live
passivbot tool clean-config input.json compact.json --mode compact
passivbot tool clean-config input.json resolved.json --mode resolved
passivbot tool config-explain input.json bot.long.unstuck.ema_span_0_minutes --json
passivbot tool clean-config configs/private/raw --dry-run --max-depth 2
passivbot tool compose-coin-overrides path/to/single_coins path/to/composed.json
passivbot tool migrate-config-v9 config_v8.json --best-effort --dry-run --diff
passivbot tool optimizer-bounds input.json search.json --preset recommended
passivbot tool compare-backtests path/to/v8/result path/to/v9/result
```

`compare-backtests --json` reports scale-one fractions: `0.05` means 5%, while the text summary
formats percentages. Metric `relative_delta_fraction` and final equity changes use the absolute
base value as denominator; `equity_rmse_initial_equity_fraction` uses absolute initial base
strategy equity. `max_abs_relative_equity_delta_fraction` uses each common sample's absolute
base equity, excluding zero denominators. Fill multiset match fractions divide matched row
multiplicity by the larger input row count (two empty inputs report zero).

## High-Signal Gotcha

If Rust changes seem ignored, follow `rust_extension.md` to verify the loaded artifact and source
stamp before debugging behavior.
