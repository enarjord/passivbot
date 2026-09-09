# Coin Overrides Guide

Per-coin overrides let you tweak bot parameters (and a few live flags) for specific coins without
forking the entire config. This guide explains what *is* and *is not* overrideable, how paths are
resolved, and shows examples for both inline and file-based overrides.

## What can be overridden

Allowed fields are intentionally limited:

- **Bot params** (per side): per-coin wallet exposure limits; selected risk fields
  (`entry_cooldown_minutes`, position-exposure enforcer settings, and
  `we_excess_allowance_pct`); selected unstuck fields (`close_pct`, `ema_dist`,
  `ema_gating_enabled`, `ema_span_0`, `ema_span_1`, `enabled`, `loss_allowance_pct`, and `threshold`); and
  every HSL field when the global `live.hsl_signal_mode` is `"coin"`; and
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

The complete conditional HSL override group is:

- `hsl.enabled`
- `hsl.red_threshold`
- `hsl.ema_span_minutes`
- `hsl.cooldown_minutes_after_red`
- `hsl.no_restart_drawdown_threshold`
- `hsl.restart_after_red_policy`
- `hsl.tier_ratios.yellow`
- `hsl.tier_ratios.orange`
- `hsl.orange_tier_mode`
- `hsl.panic_close_order_type`

These fields are per `coin+side` only when the main config selects
`live.hsl_signal_mode = "coin"`. The signal mode itself remains global and is not overridable.
An inline HSL patch in `pside` or `unified` mode is rejected. HSL values in a complete override
file are warned about and ignored in those modes, while other eligible fields in that file still
apply. A `live.hsl_signal_mode` value inside an override file or inline patch cannot authorize the
HSL patch; the main config's effective global mode always decides.

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
          "hsl": {
            "enabled": true,
            "red_threshold": 0.08,
            "ema_span_minutes": 10.0,
            "cooldown_minutes_after_red": 60.0,
            "no_restart_drawdown_threshold": 0.25,
            "restart_after_red_policy": "threshold",
            "tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "panic_close_order_type": "market"
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
- In global `coin` signal mode, per-coin HSL values drive the live supervisor and Rust backtest for
  only the selected `coin+side`, including enablement, tier thresholds, cooldown/restart policy,
  orange behavior, and panic execution type. Other coins inherit the main config.

## Composing single-coin configs

Use the offline composition tool to turn a directory of single-coin JSON/HJSON configs into one
config with inline patches:

```bash
passivbot tool compose-coin-overrides path/to/single_coins path/to/composed.json
```

Each input must validate as a current config, approve exactly one coin across its long/short lists,
reject the `all` sentinel, and contain no existing `coin_overrides`, including in nested-current
input. Files and coins are processed deterministically, with the alphabetically first filename
supplying the master config by default. Pass `--master-config FILE` to select another input or an
external JSON/HJSON config as the source of master/global values. A relative path is checked in the
input directory first, then relative to the working directory. An external master may be a
multi-coin config; only the directory inputs contribute approved coins and generated overrides.
The master must use the same strategy kind and HSL signal mode as the inputs and contain no
existing `coin_overrides`, so composition does not silently discard or merge an existing override
set. For example:

```bash
passivbot tool compose-coin-overrides path/to/single_coins path/to/composed.json \
  --master-config path/to/master.json
```

The tool rejects input aliases that resolve to the same configured
venue market, combines the per-side approved coin lists, removes approved market aliases from the
master's ignored lists, and expands `n_positions` to the approved-coin count on active sides.
Exchange-qualified identifiers contribute their explicit venue to alias resolution. Legal per-coin
values are written according to `--override-mode`; differing account-wide or
otherwise non-overridable values retain the master value and are listed in the command output.
Exact market identifiers must resolve unambiguously through cached market metadata; unresolved
exact approved or ignored identifiers, and identifiers resolving to different contracts across
configured venues, fail validation instead of falling back to a lossy ticker guess.

Choose how per-coin values are preserved:

- `--override-mode lean` (default) writes only values that differ from the master. Omitted values
  inherit future master edits, including for the input chosen as master.
- `--override-mode verbose` writes every allowed per-coin value from each normalized input,
  including values equal to the master, zero/false values, and inactive-feature settings. Every
  input gets an override, including the master when it is an input. Later changes to those master
  fields do not change the pinned coin values.

Use `--override-params` to pin only selected groups or leaves. It takes precedence over either
`--override-mode`: selected values are retained even when equal to the master, and every unselected
field inherits the master. Custom selection skips inactive-feature canonicalization, preserving
both the master's global settings and the selected source values.

```bash
passivbot tool compose-coin-overrides path/to/single_coins path/to/composed.json \
  --master-config path/to/master.json --include-backtest-optimize \
  --override-params long.strategy,long.risk.entry_cooldown_minutes
```

This pins each coin's long strategy and entry cooldown while inheriting the master's unstuck, HSL,
short-side, and live settings. You may omit `--override-mode`; specifying `lean` or `verbose` with
this selection produces the same patches.

Selectors use the fine-tune dotted-path matcher: an optional `bot.` prefix, groups or individual
leaves, full-segment prefix/suffix matching, and `*` as a one-segment wildcard. For example,
`bot.long.strategy`, `long.strategy.entry.initial_qty_pct` (Trailing Martingale),
`*.risk.entry_cooldown_minutes`, and `live.leverage` are valid selections. A bare leaf such as
`entry_cooldown_minutes` selects both sides. Overlapping selectors are deduplicated. Group selectors
include only fields allowed by the coin-override policy; `long.risk` does not override global
exposure or position-count settings. Empty selectors and selectors matching no allowed input
fields are errors, including typos, inactive-strategy paths, and HSL selectors outside coin mode.

For example, append `--override-mode verbose` to preserve the per-coin parameters when editing the
master afterward. This does not make each coin a standalone configuration: global values such as
`total_wallet_exposure_limit`, `n_positions`, Forager settings, and the HSL signal mode remain
shared. Runtime-derived wallet exposure is not automatically frozen. HSL fields can be pinned
only in `coin` signal mode. Verbose mode also pins the disabled side, so its values survive a later
global enablement change.

In lean mode without custom selectors, when HSL, auto unstuck, or a position/total-exposure enforcer
is disabled in the master and every single-coin input, parameters
used only by that disabled feature are normalized before diffing. Optimized numeric fields use the
lower bound from the master input and fixed fields use schema defaults, so inactive optimized values
do not create noise in `coin_overrides`. The total-exposure threshold is shared by the TWEL entry
gate and enforcer, so it is normalized only when both are disabled.

By default the output omits `backtest` and `optimize`, producing a lean live config. Add
`--include-backtest-optimize` to copy both sections from the master input, which makes the result
usable for backtesting or fine-tuning inherited master parameters while explicit coin overrides
remain fixed. In verbose and custom selection modes, pinned strategy fields also override optimizer
candidate values; only inherited fields can change through master-parameter optimization.
This also preserves `optimize.backend: gpu`: the GPU optimizer supports multi-coin
EMA Anchor and Trailing Martingale configs with static coin overrides. GPU-specific scope checks
remain the optimizer's responsibility; see [GPU support and limitations](optimizing.md).
Verbose output may exceed that scope: the GPU backend currently rejects bot overrides on a
disabled side and EMA Anchor position-exposure enforcer overrides, even when those values are
inert. Composition preserves the requested values; it does not silently remove them to satisfy
the GPU allowlist.

Existing output files are protected unless `--overwrite` is supplied. The selected master and
recognized single-coin inputs cannot be used as the output path. Output is published atomically,
so a failed write does not truncate an
existing config. New files use normal umask-controlled creation permissions. Replacements preserve
the destination's permission bits and POSIX owner/group; if those cannot be restored, publication
fails and the original file remains intact. Keep output outside the input directory where practical: the specified output is
excluded from discovery, but other JSON/HJSON files in that directory are treated as inputs.

## Common pitfalls

- Bad paths: a missing or unreadable `override_config_path` is fatal.
- Disallowed inline keys: fields outside the allowlist are rejected; flat strategy keys are also
  rejected so stale v7-style overrides cannot disappear silently.
- Explicit reset: an allowed value equal to the global/default value is still retained and may
  override a different value from `override_config_path`.
- Mis-keyed coins: invalid coin names and normalized-name collisions are rejected.
- Wrong types: strings such as `"false"`, nulls, and non-finite numbers are rejected rather than
  coerced into trading parameters.
- Conditional HSL: setting HSL fields in a coin patch while the main config uses `pside` or
  `unified` mode is invalid; changing `live.hsl_signal_mode` inside the patch does not change that.
