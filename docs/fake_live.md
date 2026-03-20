# Fake Live Harness

Passivbot includes a deterministic `fake` exchange for replaying live-bot behavior against a scripted local scenario.

This is useful when you want to:

- reproduce a live-bot bug without touching a real exchange
- step through HSL, mode changes, fills, and order placement with fixed candles
- regression-test live behavior after changing risk, orchestration, or exchange-facing code

## What It Runs

The harness runs the normal live bot against a fake exchange adapter.

- Python still does normal live orchestration.
- Rust still owns order behavior.
- The exchange state, prices, candles, fills, and time all come from a local scenario file.

The entry point is:

```shell
python3 src/tools/run_fake_live.py <config> <scenario>
```

## Quick Start

Use the included sample config and scenarios:

```shell
PYTHONPATH=src python3 src/tools/run_fake_live.py \
  configs/fake_live_hsl_btc.hjson \
  scenarios/fake_live/hsl_long_red_restart.hjson \
  --user fake_hsl_restart_test \
  --snapshot-each-step
```

That writes an artifact directory under `artifacts/fake_live/` by default.

For the terminal HSL case:

```shell
PYTHONPATH=src python3 src/tools/run_fake_live.py \
  configs/fake_live_hsl_btc.hjson \
  scenarios/fake_live/hsl_long_terminal_no_restart.hjson \
  --user fake_hsl_terminal_test \
  --snapshot-each-step
```

## Outputs

Each run writes a timestamped directory containing:

- `fake_live.log`
- `step_summaries.json`
- `fake_exchange_state.json`
- `fills.json`
- `positions.json`
- `hsl_trace.json`
- `snapshots/step_*.json` if `--snapshot-each-step` is enabled

This makes it easy to inspect the exact replay state at each step.

Regression philosophy:

- scenario files under `scenarios/fake_live/` are the durable in-repo behavioral regression specs
- generated artifact directories are debug evidence and local inspection output, not committed golden files
- the authoritative behavior contract lives in the scenario assertions plus the passing test run

## Scenario Structure

A scenario may define either:

- `timeline`: explicit step-by-step prices
- `replay.symbols.<symbol>.candles`: embedded OHLCV candles
- `replay.symbols.<symbol>.file`: path to OHLCV data on disk

Scenarios also define:

- `account.balance`
- boot positions
- boot fills
- boot open orders
- symbol metadata such as `price_step`, `qty_step`, `min_qty`, `min_cost`
- optional assertions for end-to-end validation

Minimal top-level fields:

- `name`
- `exchange: fake`
- `symbols`
- one of `timeline` or `replay`

Common optional fields:

- `start_time`
- `tick_interval_seconds`
- `boot_index`
- `run_initial_cycle`
- `account`
- `assertions`

Minimal shape example:

```hjson
{
  name: my_fake_case
  exchange: fake
  boot_index: 0
  account: {
    balance: 1000.0
  }
  symbols: {
    "BTC/USDT:USDT": {
      price_step: 0.1
      qty_step: 0.001
      min_qty: 0.001
      min_cost: 5.0
    }
  }
  timeline: [
    {t: 0, prices: {"BTC/USDT:USDT": 100.0}}
    {t: 1, prices: {"BTC/USDT:USDT": 99.0}}
    {t: 2, prices: {"BTC/USDT:USDT": 101.0}}
  ]
}
```

See:

- [hsl_long_red_restart.hjson](/Users/eiriknarjord/passivbot/scenarios/fake_live/hsl_long_red_restart.hjson)
- [hsl_long_terminal_no_restart.hjson](/Users/eiriknarjord/passivbot/scenarios/fake_live/hsl_long_terminal_no_restart.hjson)
- [minimal.hjson](/Users/eiriknarjord/passivbot/scenarios/fake_live/minimal.hjson)

## HSL Replay Notes

The fake harness supports HSL RED replay cases, including:

- RED trigger
- panic close placement/fill
- flat confirmation
- RED finalization
- cooldown-and-restart flows
- terminal no-restart flows
- manual-entry-during-cooldown flows for all supported live policies:
  `repanic_reset_cooldown`, `repanic_keep_original_cooldown`,
  `resume_normal_reset_drawdown`, `graceful_stop_keep_cooldown`, and
  `manual_quarantine`

The harness uses fake exchange time from the scenario, not wall-clock time.

## Speeding Up Scenarios Safely

Fake-live scenarios should be fast, but they should not change the logic being tested.

Good accelerations:

- choose short replay windows and only the candles needed to hit the target state transition
- use small test-only config values such as short cooldowns
- script direct timeline actions when you need a specific live-state event, such as a manual fill during cooldown
- use 30s or 60s step sizes when the behavior is minute-sensitive

Avoid these shortcuts:

- skipping execution cycles that would exist in the real live loop
- changing minute-quantized HSL semantics just to make a scenario finish faster
- collapsing multiple meaningful state transitions into one artificial instant when the state machine depends on ordering

Rule of thumb:

- shorten scenario duration
- do not distort the live state-machine contract

## Fake User Config

The fake exchange can be selected through `api-keys.json` style user info:

```json
{
  "exchange": "fake",
  "quote": "USDT",
  "fake_scenario_path": "scenarios/fake_live/minimal.hjson"
}
```

An example entry is included in [api-keys.json.example](/Users/eiriknarjord/passivbot/api-keys.json.example).

## Troubleshooting

Common issues:

- `Config user resolved to exchange '...' expected 'fake'`
  Use `--user <fake_user>` or point `live.user` at a fake entry from [api-keys.json.example](/Users/eiriknarjord/passivbot/api-keys.json.example).
- `Fake exchange requires live.fake_scenario_path or api-keys fake_scenario_path`
  Pass a scenario on the command line and make sure the fake user resolves to `exchange: "fake"`.
- `Fake scenario must define timeline rows or replay candles`
  Add either `timeline` rows or `replay.symbols.<symbol>.candles` / `.file`.
- Unexpected fills or no fills
  Check symbol metadata such as `price_step`, `qty_step`, `min_qty`, and `min_cost`, and inspect `snapshots/step_*.json`.
- HSL replay does not do what you expect
  Inspect `hsl_trace.json` and `fake_live.log` first. The harness uses scenario time, not wall-clock time.

## When To Use It

Use the fake harness when you need live-loop realism with deterministic local inputs.

Use the Rust backtester when you need broad historical evaluation, optimizer runs, or performance analysis over long date ranges.
