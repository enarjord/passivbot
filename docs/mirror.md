# Mirror Mode (Market-Neutral Hedging)

Mirror mode is a market-neutral overlay that automatically hedges your base positions by opening opposite-side positions on different symbols. This reduces net directional exposure while maintaining the contrarian entry logic of the base strategy.

## Overview

Mirror mode works by:

1. Running a **base strategy** (long-only or short-only) using Passivbot's standard grid/trailing entry logic
2. Automatically opening **mirror positions** on the opposite side to hedge the base exposure
3. Dynamically rebalancing mirror positions to maintain a target hedge ratio

### Example: Long-Only with Short Mirrors

If your base strategy is long-only on BTC, ETH, and SOL with total exposure of 1.0 (100% of balance):

- Mirror mode opens short positions on other approved coins (e.g., DOGE, XRP, AVAX)
- With `threshold: 1.0`, it targets equal short exposure (1.0) to offset the longs
- Net market exposure approaches zero, while you still capture the spread from entries/exits

## When to Use Mirror Mode

Mirror mode is useful when you want to:

- **Reduce directional risk**: Hedge against broad market moves (crashes or pumps)
- **Capture relative value**: Profit from your base coins outperforming the mirror coins
- **Trade in sideways markets**: Market-neutral strategies can profit even when overall market is flat
- **Manage drawdowns**: Hedging limits exposure to single-direction moves

## Configuration

Mirror mode is configured under `config.live.mirror`:

```json
{
  "live": {
    "mirror": {
      "mode": "mirror_shorts_for_longs",
      "threshold": 1.0,
      "tolerance_pct": 0.05,
      "max_n_positions": 0,
      "mirror_excess_allowance_pct": 0.2,
      "allocation_min_fraction": 0.1,
      "ema_dist_entry": 0.0
    }
  }
}
```

### Parameter Reference

#### `mode`

**Type:** `string`
**Default:** `"mirror_shorts_for_longs"`
**Options:** `"mirror_shorts_for_longs"`, `"mirror_longs_for_shorts"`

Determines the hedge direction:

| Mode | Base Side | Mirror Side | Use Case |
|------|-----------|-------------|----------|
| `mirror_shorts_for_longs` | Long-only | Shorts | Most common; hedge long exposure with shorts |
| `mirror_longs_for_shorts` | Short-only | Longs | Hedge short exposure with longs |

**Important:** The opposite base side must be disabled:
- `mirror_shorts_for_longs` requires `bot.short.n_positions = 0`
- `mirror_longs_for_shorts` requires `bot.long.n_positions = 0`

#### `threshold`

**Type:** `float`
**Default:** `0.0` (disabled)
**Range:** `0.0` to `1.0+`

Target mirror exposure as a multiple of base exposure:

| Value | Meaning |
|-------|---------|
| `0.0` | Mirror mode disabled |
| `0.5` | Target 50% hedge (half of base exposure) |
| `1.0` | Target 100% hedge (equal to base exposure) |
| `1.5` | Target 150% hedge (net short bias) |

**Example:** With base long exposure of 0.8 and `threshold: 1.0`, the bot targets 0.8 short exposure via mirror positions.

#### `tolerance_pct`

**Type:** `float`
**Default:** `0.05`
**Range:** `0.0` to `1.0`

Tolerance band (as percentage of target) to avoid rebalancing churn. The bot only rebalances when mirror exposure deviates from target by more than:

```
tolerance_band = target_hedge_exposure * tolerance_pct
```

**Example:** With target hedge of 0.8 and `tolerance_pct: 0.05`:
- Tolerance band = 0.8 * 0.05 = 0.04
- Rebalance only when mirror exposure < 0.76 or > 0.84

Lower values = tighter tracking but more frequent rebalancing.
Higher values = less churn but looser tracking.

#### `max_n_positions`

**Type:** `integer`
**Default:** `0`
**Range:** `0` to any positive integer

Maximum number of mirror positions. Set to `0` to use the base side's `n_positions`.

**Example:** If base long has `n_positions: 10` and `max_n_positions: 5`, the bot opens at most 5 mirror shorts, each with higher average exposure.

#### `mirror_excess_allowance_pct`

**Type:** `float`
**Default:** `0.2`
**Range:** `0.0` to `1.0+`

Per-mirror-position cap looseness. Each mirror position is capped at:

```
cap_per_position = (base_twel * threshold / max_positions) * (1 + mirror_excess_allowance_pct)
```

**Example:** With base TWEL of 1.5, threshold 1.0, max_positions 5, and allowance 0.2:
- Base cap per position = 1.5 * 1.0 / 5 = 0.3
- Actual cap = 0.3 * 1.2 = 0.36

This allows individual mirror positions to temporarily exceed their fair share, providing flexibility during volatile periods.

#### `allocation_min_fraction`

**Type:** `float`
**Default:** `0.1`
**Range:** `(0.0, 1.0]`

Minimum fraction of remaining hedge budget to allocate per step. Controls the granularity of mirror position sizing:

- Lower values (e.g., 0.05) = finer allocations, more orders, potentially more churn
- Higher values (e.g., 0.25) = coarser allocations, fewer orders, less precise tracking

The bot allocates at least this fraction of the remaining budget when adding to mirror positions.

#### `ema_dist_entry`

**Type:** `float`
**Default:** `0.0`
**Range:** `-1.0` to `0.1+`

EMA-based gating for opening new mirror positions:

| Mode | Condition to Open Mirror |
|------|-------------------------|
| `mirror_shorts_for_longs` | Price >= EMA_upper * (1 + ema_dist_entry) |
| `mirror_longs_for_shorts` | Price <= EMA_lower * (1 - ema_dist_entry) |

| Value | Behavior |
|-------|----------|
| `0.0` | Gate exactly at the EMA band |
| Positive (e.g., `0.01`) | Require price extended beyond EMA band (e.g., 1% above for shorts) |
| Negative (e.g., `-0.01`) | Allow entries slightly inside the EMA band |
| Large negative (e.g., `-1.0`) | Effectively disable EMA gating |

**Example:** With `ema_dist_entry: 0.01` in `mirror_shorts_for_longs` mode:
- Only open new short mirrors when price is at least 1% above the upper EMA band
- This avoids opening shorts during dips, waiting for relative strength

## How Mirror Positions Are Selected

### Opening New Mirrors

When the bot needs to add mirror exposure, it:

1. **Filters eligible symbols** from `approved_coins` for the mirror side (e.g., `approved_coins.short` for `mirror_shorts_for_longs`)
2. **Excludes symbols** where:
   - A base position already exists (one-way constraint)
   - A base entry order is pending (collision avoidance)
   - EMA gating condition is not met
3. **Ranks remaining symbols** using Borda ranking (low volatility + high volume preferred)
4. **Opens minimum-size positions** on the best-ranked symbols up to `max_n_positions`

### Adding to Existing Mirrors

When mirror exposure is below target but max positions reached:

1. **Identifies the most underwater mirror** (furthest from break-even in the losing direction)
2. **Allocates budget** to that position up to its per-position cap
3. **Repeats** until target exposure is reached or all mirrors are at cap

This "add to worst" strategy helps underwater mirrors recover while maintaining diversification.

### Closing Mirrors (Overhedged)

When mirror exposure exceeds target:

1. **Sorts mirrors by underwaterness** (least underwater first)
2. **Closes the least underwater mirror** (best P&L) fully
3. **Repeats** until exposure is within tolerance

Closing winners first preserves the underwater mirrors for potential recovery.

### Collision Handling

When the base side wants to enter a symbol that has a mirror position:

1. **Close the mirror position** on that symbol immediately
2. **Gate the base entry** for one cycle to avoid simultaneous long+short
3. **Next cycle**: base entry proceeds normally

This maintains the one-way constraint (no simultaneous long+short on the same symbol).

## Example Configurations

### Conservative Market-Neutral

```json
{
  "live": {
    "mirror": {
      "mode": "mirror_shorts_for_longs",
      "threshold": 1.0,
      "tolerance_pct": 0.10,
      "max_n_positions": 0,
      "mirror_excess_allowance_pct": 0.1,
      "allocation_min_fraction": 0.2,
      "ema_dist_entry": 0.01
    }
  },
  "bot": {
    "long": {
      "n_positions": 5,
      "total_wallet_exposure_limit": 1.0
    },
    "short": {
      "n_positions": 0
    }
  }
}
```

- Full hedge (threshold 1.0) with 10% tolerance
- EMA gating requires price to be 1% above EMA before opening shorts
- Matches long positions with equal number of short mirrors
- Tight per-position caps (10% excess allowance)

### Partial Hedge with More Mirrors

```json
{
  "live": {
    "mirror": {
      "mode": "mirror_shorts_for_longs",
      "threshold": 0.5,
      "tolerance_pct": 0.05,
      "max_n_positions": 10,
      "mirror_excess_allowance_pct": 0.3,
      "allocation_min_fraction": 0.1,
      "ema_dist_entry": -1.0
    }
  },
  "bot": {
    "long": {
      "n_positions": 5,
      "total_wallet_exposure_limit": 1.5
    },
    "short": {
      "n_positions": 0
    }
  }
}
```

- 50% hedge (retain some long bias)
- More mirror positions (10) than base (5) for diversification
- EMA gating disabled (`-1.0` = open mirrors at any price)
- Tighter tolerance (5%) for more precise tracking

### Short-Only with Long Mirrors

```json
{
  "live": {
    "mirror": {
      "mode": "mirror_longs_for_shorts",
      "threshold": 0.75,
      "tolerance_pct": 0.08,
      "max_n_positions": 0,
      "mirror_excess_allowance_pct": 0.2,
      "allocation_min_fraction": 0.15,
      "ema_dist_entry": 0.005
    }
  },
  "bot": {
    "long": {
      "n_positions": 0
    },
    "short": {
      "n_positions": 8,
      "total_wallet_exposure_limit": 1.2
    }
  }
}
```

- Base is short-only, mirrors are longs
- 75% hedge (retain some short bias)
- EMA gating at 0.5% for opening long mirrors

## Approved Coins for Mirrors

Mirror positions can only be opened on coins listed in the approved coins for the mirror side:

| Mode | Mirror Approved Coins |
|------|----------------------|
| `mirror_shorts_for_longs` | `live.approved_coins.short` (or `live.approved_coins` if not split) |
| `mirror_longs_for_shorts` | `live.approved_coins.long` (or `live.approved_coins` if not split) |

**Tip:** Use split approved coins to control which symbols can be base vs mirror:

```json
{
  "live": {
    "approved_coins": {
      "long": ["BTC", "ETH", "SOL"],
      "short": ["DOGE", "XRP", "AVAX", "ADA", "MATIC"]
    }
  }
}
```

In `mirror_shorts_for_longs` mode:
- Base longs open on BTC, ETH, SOL
- Mirror shorts open on DOGE, XRP, AVAX, ADA, MATIC

## Backtesting with Mirror Mode

Mirror mode is fully supported in backtests. Enable it the same way as live:

```bash
python3 src/backtest.py configs/my_mirror_config.json
```

The backtest engine:
- Runs the Rust mirror overlay on each tick
- Tracks mirror positions separately from base positions
- Reports combined P&L and metrics

## Metrics and Monitoring

When mirror mode is active, the bot logs:

- Base exposure vs mirror exposure each cycle
- Rebalancing decisions (open/close/add)
- Collision events and gating
- EMA gating blocks (if enabled)

Use `--debug-level 2` or higher to see detailed mirror cycle logs.

## Constraints and Limitations

### Position Mode: One-Way vs Two-Way (Hedge Mode)

Mirror mode supports both one-way and two-way position modes, controlled by `config.live.hedge_mode`:

| `hedge_mode` | Mode | Behavior |
|--------------|------|----------|
| `true` (default) | **Two-way (hedge mode)** | Simultaneous long+short positions allowed on the same symbol |
| `false` | **One-way** | Only one side can be open per symbol; collision handling applies |

**Two-way mode (`hedge_mode: true`):**
- Mirror positions can be opened on the same symbols as base positions
- No collision handling needed
- Useful when exchange supports hedge mode and you want maximum flexibility

**One-way mode (`hedge_mode: false`):**
- Mirror positions are only opened on symbols where no base position exists
- If base wants to enter a symbol with an existing mirror, the mirror is closed first and base entry is gated for one cycle
- Avoids simultaneous long+short on same symbol (required by some exchanges or strategies)

### Base Side Must Be Single-Direction

You must disable the opposite base side:
- `mirror_shorts_for_longs`: set `bot.short.n_positions = 0`
- `mirror_longs_for_shorts`: set `bot.long.n_positions = 0`

The bot will error if both sides are enabled with mirror mode active.

### Mark-to-Market Exposure

Mirror uses mark-to-market pricing for exposure calculations, not entry price. This means:
- Exposure reflects current market risk, not historical cost basis
- Rebalancing responds to price moves even without fills

### No Direct Mirror Parameter Optimization

Currently, mirror parameters cannot be optimized directly. Use backtesting to manually tune threshold, tolerance, and other settings.

## Troubleshooting

### Mirror Not Opening Positions

Check:
1. `threshold > 0.0` (0.0 disables mirror mode)
2. Approved coins exist for the mirror side
3. EMA gating isn't blocking all entries (try `ema_dist_entry: -1.0` to disable)
4. Base side has open positions (no base = no mirror needed)

### Too Much Rebalancing Churn

Increase `tolerance_pct` (e.g., from 0.05 to 0.10) to widen the tolerance band.

### Mirror Positions Too Small

- Increase `max_n_positions` to spread exposure across more symbols
- Decrease `mirror_excess_allowance_pct` to tighten per-position caps
- Ensure base exposure is large enough to warrant mirrors

### Collisions Blocking Base Entries

This is expected behavior. When a mirror exists on a symbol where base wants to enter:
1. Mirror closes
2. Base entry is gated for one cycle
3. Next cycle base enters normally

If this happens frequently, consider using different approved coins for base vs mirror.
