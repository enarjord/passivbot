# Coin Overrides Guide

Per-coin overrides let you tweak bot parameters (and a few live flags) for specific coins without
forking the entire config. This guide explains what *is* and *is not* overrideable, how paths are
resolved, and shows examples for both inline and file-based overrides.

## What can be overridden

Allowed fields are intentionally limited:

- **Bot params** (per side): per-coin wallet exposure limits; selected risk fields
  (`entry_cooldown_minutes`, position-exposure enforcer settings, and
  `we_excess_allowance_pct`); selected unstuck fields (`close_pct`, `ema_dist`,
  `ema_gating_enabled`, `enabled`, `loss_allowance_pct`, and `threshold`); and
  nested active strategy parameters under `bot.<side>.strategy.<strategy_kind>.*` (see the
  allowlist in `src/config/overrides.py:get_allowed_modifications()` for the full set).
- **Live flags**: `forced_mode_long`, `forced_mode_short`, `leverage`.

Not overrideable: approved/ignored coins, exchange settings, and arbitrary new keys. A disallowed or
unknown inline override is rejected with its full path. A full override config file may contain
ordinary non-override config fields; those fields are validated as part of that config and then
filtered out. Flat v7-style strategy keys such as `entry_grid_spacing_pct` are rejected; use the
nested v8 strategy path instead.

`bot.<side>.risk.we_excess_allowance_mode` is global policy, not a per-coin knob. Inline coin
patches that contain it fail with a migration message. A complete file used through
`override_config_path` may contain the global field, but it is warned about and ignored for the
coin patch; set the value in the main config instead.

## How overrides are loaded

1) `coin_overrides` is read from your main config. Keys should be coin tickers (e.g., `"XRP"`).
2) If `override_config_path` is provided, the file is loaded. Relative paths are resolved against
   `live.base_config_path` (if set) or the current working directory.
3) Explicit allowed values are extracted without hydrating omitted fields or diffing against the
   base config. This preserves intentional resets to a global/default value, `false`, or zero.
4) File values are applied first and inline values are applied second. Inline values therefore win
   at the individual leaf they specify.
5) The patch is type-checked, merged with the global config, and the resulting complete per-coin
   config is validated before startup continues.
6) During live startup, override keys are remapped to exchange symbols via `coin_to_symbol`; config
   lookups prefer these per-symbol values. In backtests, `prep_backtest_args` merges the override
   bot patch directly per coin.

Configured override files are required inputs. A missing, unreadable, malformed, or invalid file
stops configuration loading with the coin and path identified. The file's `live.strategy_kind`, if
present, must match the global strategy kind; per-coin strategy-kind changes are not supported.
Coin keys that normalize to the same ticker are also rejected instead of overwriting each other.

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
          "strategy": {
            "trailing_martingale": {
              "entry": {
                "threshold_base_pct": 0.05
              }
            }
          },
          "unstuck": {
            "ema_gating_enabled": false,
            "loss_allowance_pct": 0.005
          },
          "risk": {
            "entry_cooldown_minutes": 0.05
          },
          "wallet_exposure_limit": 0.18
        },
        "short": {
          "strategy": {
            "trailing_martingale": {
              "entry": {
                "threshold_base_pct": 0.055
              }
            }
          }
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
      "strategy": {
        "trailing_martingale": {
          "entry": {
            "threshold_base_pct": 0.021,
            "initial_ema_dist": 0.001
          }
        }
      },
      "wallet_exposure_limit": 0.12
    },
    "short": {
      "strategy": {
        "trailing_martingale": {
          "entry": {
            "threshold_base_pct": 0.019
          },
          "close": {
            "threshold_base_pct": 0.004
          }
        }
      }
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
- Verify that inline patches contain only allowed fields. Non-override fields in a complete file are
  filtered after the file is validated.
- Don’t expect per-override approved coin lists to take effect; keep the master coin list in the
  main config.
- A per-coin `unstuck.loss_allowance_pct` overrides only the selected coin+side's loss allowance
  percentage. It still uses the account-wide unstuck budget formula with `total_wallet_exposure_limit`;
  it does not create a separate per-coin realized-PnL tracker.
- A per-coin `risk.entry_cooldown_minutes` gates only position-increasing entries for the selected
  coin+side. A per-coin `unstuck.ema_gating_enabled=false` disables only that coin+side's unstuck
  EMA trigger/readiness gate; the other unstuck eligibility checks still apply.

## Common pitfalls

- Bad paths: a missing or unreadable `override_config_path` is fatal.
- Disallowed inline keys: fields outside the allowlist are rejected; flat strategy keys are also
  rejected so stale v7-style overrides cannot disappear silently.
- Explicit reset: an allowed value equal to the global/default value is still retained and may
  override a different value from `override_config_path`.
- Mis-keyed coins: invalid coin names and normalized-name collisions are rejected.
- Wrong types: strings such as `"false"`, nulls, and non-finite numbers are rejected rather than
  coerced into trading parameters.
