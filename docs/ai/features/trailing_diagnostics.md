# Trailing Diagnostics

## Contract

1. The diagnostics module/tool is read-only and must not affect trading behavior.
2. Whether the next order is trailing still comes from the Rust helper calls, not a Python reimplementation.
3. The tool may derive observability-only threshold/retracement fields in Python for explanation and parameter tuning.
4. Snapshot bootstrapping must fail clearly when required inputs are missing; wizard/manual mode is the fallback.

## Current Implementation Scope

Implemented now:

1. `src/trailing_diagnostics.py` reusable helpers for:
   - building trailing diagnostics from flat numeric inputs
   - bootstrapping those inputs from `config + state.latest.json`
   - computing both next-entry and next-close trailing diagnostics
2. `src/tools/trailing_diagnostics.py` interactive command-driven terminal explorer
3. bootstrap from:
   - `--config` + monitor snapshot
   - manual `--wizard` input flow
4. runtime commands for:
   - `set <key> <value>`
   - `edit <key> <value>`
   - `symbol <coin|symbol>`
   - `side <long|short>`
   - `reset`
   - `wizard`
   - `dump`
   - `list`
   - `help`
   - `quit`

## Non-Obvious Details

1. Exact recomputation needs more than the current trailing payload; it also needs market metadata such as `qty_step`, `price_step`, `min_qty`, `min_cost`, `c_mult`, EMA bands, and the active `h1` log-range EMA.
2. The monitor snapshot now includes `market[*].c_mult` and `market[*].entry_volatility_logrange_ema` so the tool can seed exact diagnostics from live monitor output.
3. The tool uses formatted config values from `load_config()`. It does not instantiate a live `Passivbot`.
4. Manual mode exists because historical snapshots or offline experiments may not have all required fields.
5. The wizard asks for the core trailing inputs first and only prompts for the extra sizing/grid parameters when the user opts into advanced mode.

## Test Focus

1. snapshot bootstrap should extract the required inputs cleanly
2. entry/close diagnostics should match the existing monitor trailing slice
3. command handling should mutate inputs deterministically and support dump/reset/help flows
4. the tool wrapper should start with `--help` without import errors

## Key Code

- `src/trailing_diagnostics.py`
- `src/trailing_diagnostics_tool.py`
- `src/tools/trailing_diagnostics.py`
- `src/passivbot_monitor.py`
- `tests/test_trailing_diagnostics.py`
