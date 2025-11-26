# Coin Overrides Guide

Per-coin overrides let you tweak bot parameters (and a few live flags) for specific coins without
forking the entire config. This guide explains what *is* and *is not* overrideable, how paths are
resolved, and shows examples for both inline and file-based overrides.

## What can be overridden

Allowed fields are intentionally limited:

- **Bot params** (per side): grid spacing, double-down factors, EMA spans, initial qty/EMA dist,
  trailing/unstuck settings, wallet exposure limits, selected risk knobs (see the allowlist in
  `config_utils.get_allowed_modifications` for the full set).
- **Live flags**: `forced_mode_long`, `forced_mode_short`, `leverage`.

Not overrideable: approved/ignored coins, exchange settings, arbitrary new keys—anything outside the
allowlist is ignored.

## How overrides are loaded

1) `coin_overrides` is read from your main config. Keys should be coin tickers (e.g., `"XRP"`).
2) If `override_config_path` is provided, the file is loaded. Relative paths are resolved against
   `live.base_config_path` (if set) or the current working directory.
3) The override content is filtered to the allowed fields and diffed against the base config; only
   differing allowed fields are kept.
4) During live startup, override keys are remapped to exchange symbols via `coin_to_symbol`; config
   lookups prefer these per-symbol values. In backtests, `prep_backtest_args` merges the override
   bot diffs directly per coin.

If the file cannot be found or yields no allowed diffs, the override is effectively empty and the
base values are used.

## Inline override example

```json
{
  "live": {
    "approved_coins": ["BTC", "XRP"],
    "base_config_path": "configs/running_config.json"
  },
  "coin_overrides": {
    "XRP": {
      "bot": {
        "long": {
          "entry_grid_spacing_pct": 0.05,
          "wallet_exposure_limit": 0.18
        },
        "short": {
          "entry_grid_spacing_pct": 0.055
        }
      },
      "live": {
        "forced_mode_long": "normal"
      }
    }
  }
}
```

## File-based override example

Main config:
```json
{
  "live": {
    "approved_coins": ["BTC", "BCH", "DOGE"],
    "base_config_path": "configs/running_config.json"
  },
  "coin_overrides": {
    "BCH": { "override_config_path": "configs/overrides/bch.json" },
    "DOGE": { "override_config_path": "configs/overrides/doge.json" }
  }
}
```

`configs/overrides/bch.json`:
```json
{
  "bot": {
    "long": {
      "entry_grid_spacing_pct": 0.021,
      "entry_initial_ema_dist": 0.001,
      "wallet_exposure_limit": 0.12
    },
    "short": {
      "entry_grid_spacing_pct": 0.019,
      "close_grid_markup_start": 0.004
    }
  },
  "live": {
    "forced_mode_short": "graceful_stop",
    "leverage": 4
  }
}
```

## How to validate overrides

- Run with `--log-level debug` to see which overrides were initialized and when a per-symbol override
  value is used.
- Ensure `live.base_config_path` is set so relative `override_config_path` values resolve.
- Verify the override file changes *allowed* fields versus the base config; disallowed keys are
  dropped.
- Don’t expect per-override approved coin lists to take effect; keep the master coin list in the
  main config.

## Common pitfalls

- Bad paths: `override_config_path` not found → override silently empty.
- Disallowed keys: fields outside the allowlist are ignored.
- No diff: if the override matches the base on allowed fields, nothing is applied.
- Mis-keyed coins: if a coin name cannot be mapped to an exchange symbol, the override is discarded.
