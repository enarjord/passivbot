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

The harness uses fake exchange time from the scenario, not wall-clock time.

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

## When To Use It

Use the fake harness when you need live-loop realism with deterministic local inputs.

Use the Rust backtester when you need broad historical evaluation, optimizer runs, or performance analysis over long date ranges.
