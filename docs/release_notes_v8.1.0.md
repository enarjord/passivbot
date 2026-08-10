# Passivbot v8.1.0

Released 2026-08-10.

V8.1 consolidates the first month of production hardening after the v8 mainline cutover. It keeps
the v8 strategy and configuration family while adding exchanges, configuration capabilities,
data-path improvements, and stricter live execution contracts.

## Highlights

- Added live USDT perpetual-futures support for Bitunix through a native connector and for WEEX
  through CCXT, including market metadata, account state, orders, fills, PnL, live candles, and
  exchange-specific safety handling.
- Made Rust orchestrator output an atomic live-trading contract. Malformed order or diagnostic
  output now fails before reconciliation, while quantity, price-step, minimum-cost, execution-type,
  mode, risk-gate, and protective-order invariants receive stricter boundary validation.
- Scoped candle, EMA, trailing, and fill-history readiness to the Rust actions that consume each
  input, preserving independent protective paths without fabricating missing values.
- Reworked coin overrides as explicit typed patches with deterministic file/inline precedence,
  canonical validation, selected per-coin risk and unstuck controls, and conditional per-coin HSL
  groups when the global signal mode is `coin`.
- Unified fill-history coverage and realized-PnL readiness, hardened restart reconstruction, and
  added multi-resolution candle ladders for HSL and trailing episodes older than an exchange's 1m
  retention window.
- Added verified public 1m WebSocket candle persistence for flat forager candidates and hardened
  combined-exchange selection, coverage, provenance, and robust volume normalization.
- Reduced backtest and optimizer-suite preparation time and peak memory through chunked
  materialization, direct shared-memory copies, and earlier release of candidate datasets.
- Expanded structured live events, runtime attribution, incident bundles, event queries,
  performance reports, restart-smoke evidence, and offline diagnostic tooling.

## Configuration And Compatibility

- The canonical config schema is now `v8.1.0`. Existing `v8.0.0` configs are accepted and
  normalized to the current v8 schema; source files are not rewritten automatically. Unreleased
  or unrecognized same-major schema versions are rejected instead of guessed compatible.
- New configs should start from a maintained file under `configs/examples/`, all of which now carry
  `config_version: "v8.1.0"`.
- The v7 migration boundary is unchanged. `trailing_grid_v7` remains the explicit compatibility
  strategy, while new strategy work should use `trailing_martingale` or `ema_anchor`.
- Coin overrides now reject malformed, ambiguous, strategy-mismatched, or unsupported patches
  instead of silently hydrating them into a usable-looking config.
- Canonical defaults have changed since v8.0.0. Operators whose configs omit fields should inspect
  the normalized configuration and review exposure, HSL, unstuck, forager, and approved-coin
  settings before restarting.

## Upgrade Checklist

1. Back up the running config and retain the exact prior revision needed for reproduction.
2. Pull v8.1.0, reinstall the Python package, and rebuild the Rust extension.
3. Normalize and inspect each config, paying particular attention to coin overrides and fields that
   previously relied on defaults.
4. Run representative backtests and the normal offline/preflight checks for the deployment.
5. Use the established operational rollout process for live bots; this release does not authorize
   or perform exchange requests, bot restarts, or order actions by itself.

The complete detailed change ledger is in [CHANGELOG.md](../CHANGELOG.md). The original v7-to-v8
migration boundary remains documented in [the v8.0.0 release notes](release_notes_v8.0.0.md).
