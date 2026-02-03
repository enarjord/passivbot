# Stock Perpetuals Trading Guide

This guide covers trading stock perpetuals (equity perps) on Hyperliquid using Passivbot.

## Overview

Stock perpetuals are perpetual futures contracts that track traditional equity prices (TSLA, NVDA, AAPL, etc.). On Hyperliquid, these are deployed via **HIP-3** (Hyperliquid Improvement Proposal 3), which enables permissionless perpetual market creation.

The primary stock perps provider on Hyperliquid is **TradeXYZ** (trade.xyz), which deploys markets with the `xyz:` prefix (e.g., `xyz:TSLA`, `xyz:NVDA`).

### Key Characteristics

| Aspect | Stock Perps | Crypto Perps |
|--------|-------------|--------------|
| Symbol format | `xyz:TSLA/USDC:USDC` | `BTC/USDC:USDC` |
| Margin mode | **Isolated only** | Cross or Isolated |
| Max leverage | 10x | Up to 50x |
| Trading hours | 24/7 | 24/7 |
| Fees | 2x standard Hyperliquid fees | Standard fees |
| Collateral | USDC | USDC |

### Available Stock Perps

As of early 2026, TradeXYZ offers perpetuals on:
- **Tech giants**: TSLA, NVDA, AAPL, MSFT, META, AMZN, GOOGL, NFLX, AMD
- **Fintech/Crypto-adjacent**: COIN, HOOD, PLTR, MSTR
- **Commodities**: GOLD, SILVER, COPPER, NATGAS, URANIUM
- **Currencies**: EUR, JPY
- **Index**: XYZ100 (Nasdaq-like index)

Other HIP-3 builders (FLX, KM, CASH, VNTL, HYNA, ABCD) also offer various perps. Passivbot supports any isolated-margin market automatically.

## Understanding Balance and Margin

### How Isolated Margin Works

HIP-3 stock perps use **isolated margin**, which is fundamentally different from cross margin used by most crypto perps.

| Cross Margin (HL crypto perps) | Isolated Margin (HIP-3/XYZ perps) |
|-------------------------------|-----------------------------------|
| Entire balance backs all positions | Each position has dedicated margin |
| One liquidation can cascade to others | Liquidations are contained per-position |
| More capital efficient | Safer but requires more upfront margin |
| Default for BTC, ETH, etc. | **Required** for all stock perps |

### Balance Display Differences

You'll see different balance breakdowns depending on which interface you use:

**Hyperliquid Web UI (app.hyperliquid.xyz)**:
- Shows your **total account value** (e.g., 105 USDC)
- This includes all free balance + margin locked in positions

**TradeXYZ Web UI (trade.xyz)**:
- Shows a **breakdown by builder**:
  - `USDC (HL perps)`: Free balance available for HL native perps or withdrawal
  - `USDC (XYZ perps)`: Margin currently locked in your XYZ stock perp positions

**Example**:
```
Hyperliquid UI:     105 USDC total
TradeXYZ UI:        11.91 USDC (HL perps) + 93.27 USDC (XYZ perps) = 105.18 USDC
```

The ~$93 locked in "XYZ perps" is your margin for open TSLA positions. If those positions get liquidated, only that $93 is at risk - your remaining $12 stays safe.

### Margin Calculation

When you open a position on an isolated margin market:

```
margin_required = position_notional / leverage
                = (quantity × price) / leverage
```

For example, opening 0.24 TSLA at $423 with 2x leverage:
```
margin = (0.24 × $423) / 2 = $50.76
```

This margin is **locked** for that position until you close it. Your free balance decreases accordingly.

### Passivbot Balance Display

Passivbot shows your **total account value** from the Hyperliquid API, which includes:
- Free balance
- All locked margin
- Unrealized PnL

The balance changes you see in logs reflect margin being allocated/released as positions open/close:
```
[balance] 105.21 -> 84.27   # Margin locked for new position
[balance] 84.27 -> 67.35    # More margin locked for second entry
```

## Requirements

### 1. Hyperliquid Account

You need a Hyperliquid account with USDC deposited. No KYC required - your wallet is your identity.

### 2. One-Time TradeXYZ Registration

**Important**: Before you can trade XYZ stock perps via API, you must complete a one-time wallet registration on the TradeXYZ platform:

1. Go to [trade.xyz](https://trade.xyz)
2. Click "Connect Wallet" and connect your Hyperliquid wallet
3. Sign the verification transaction to prove wallet ownership
4. Accept the terms of service and privacy policy
5. Click **"Enable Trading"** when prompted and sign the confirmation

This registration links your wallet to the TradeXYZ builder, enabling API access to their HIP-3 markets. Without this step, orders will fail with "Insufficient margin" errors even if you have sufficient balance.

### 3. Regional Restrictions

TradeXYZ prohibits access from:
- United States
- OFAC-sanctioned countries

Ensure you're not connecting from a restricted region.

## Configuration

### Symbol Selection

You can specify stock perps in `approved_coins` using any of these formats:

```json
{
  "live": {
    "approved_coins": ["TSLA"]
  }
}
```

or explicitly with the prefix:

```json
{
  "live": {
    "approved_coins": ["xyz:TSLA", "xyz:NVDA"]
  }
}
```

Passivbot automatically maps `TSLA` to `XYZ-TSLA/USDC:USDC` on Hyperliquid.

### Mixing Crypto and Stock Perps

You can run both crypto perps (cross margin) and stock perps (isolated margin) in the same bot:

```json
{
  "live": {
    "approved_coins": ["BTC", "ETH", "TSLA", "NVDA"]
  }
}
```

Passivbot automatically sets the correct margin mode for each symbol:
- **Crypto perps** (BTC, ETH, SOL, etc.) → cross margin
- **Stock perps** (TSLA, NVDA, AAPL, etc.) → isolated margin

```
BTC/USDC:USDC: margin=ok (cross)
XYZ-TSLA/USDC:USDC: margin=ok (isolated)
```

**How balance works in mixed mode:**

| Component | Backs What |
|-----------|------------|
| Free balance | Shared by all cross-margin positions (BTC, ETH, etc.) |
| Locked margin (per stock) | Only that specific stock perp position |

**Risk isolation**: If TSLA gets liquidated, only its locked margin is lost. Your BTC/ETH positions and their shared margin pool are unaffected. Stock perp positions are "firewalled" from each other and from crypto positions.

**Practical consideration**: Isolated margin positions lock up capital, reducing what's available for cross-margin positions. Plan your `n_positions` and `total_wallet_exposure_limit` accordingly.

### Automatic Detection

Passivbot automatically detects stock perps by:
1. The `xyz:` symbol prefix (or `XYZ-` in CCXT format)
2. The `onlyIsolated: true` market flag
3. Membership in the known stock ticker list (TSLA, NVDA, AAPL, etc.)

### Leverage and Margin

Stock perps require **isolated margin mode**. Passivbot handles this automatically:

1. Detects isolated-only markets
2. Sets isolated margin mode (not cross)
3. Calculates minimum leverage based on your TWEL setting
4. Caps leverage at 10x (HIP-3 maximum)

**Leverage calculation for isolated margin:**

For isolated margin, your margin requirement is: `margin = exposure / leverage`

To ensure you never exceed your balance, Passivbot uses:
```
min_leverage = ceil(max(long_TWEL, short_TWEL))
```

For example, with TWEL = 1.25:
- Minimum leverage = ceil(1.25) = 2x
- Maximum exposure at $100k balance = $125k
- Margin required = $125k / 2 = $62.5k (within balance)

### Minimum Order Size

Stock perps on Hyperliquid have a **$10 minimum order value**. With small balances, this limits how many grid entries you can place.

For a $100 balance trading TSLA at $400:
- Minimum qty = $10 / $400 = 0.025 TSLA
- At TWEL 1.0 with 2x leverage: max exposure = $200, max qty = 0.5 TSLA
- Practical grid depth: ~5-8 entries before hitting minimums

Consider setting `filter_by_min_effective_cost: false` if you want to trade with smaller balances and accept that some grid levels may be skipped.

### Example Config

Minimal test config for stock perps:

```json
{
  "live": {
    "user": "hyperliquid_01",
    "approved_coins": ["TSLA"],
    "empty_means_all_approved": false,
    "leverage": 2,
    "filter_by_min_effective_cost": false,
    "hedge_mode": false,
    "minimum_coin_age_days": 0
  },
  "bot": {
    "long": {
      "n_positions": 2,
      "total_wallet_exposure_limit": 1.0,
      "entry_initial_qty_pct": 0.4
    },
    "short": {
      "n_positions": 0,
      "total_wallet_exposure_limit": 0
    }
  },
  "logging": {
    "level": 2
  }
}
```

Key settings explained:
- `leverage: 2` - Safe starting point, meets isolated margin requirements
- `filter_by_min_effective_cost: false` - Allow trading even if balance is low
- `minimum_coin_age_days: 0` - Stock perps are new, don't filter by age
- `entry_initial_qty_pct: 0.4` - Larger initial entries (fewer grid levels with small balance)

## Oracle Pricing Behavior

Stock perps use oracle pricing from RedStone's HyperStone oracle. During market hours, prices track live stock prices. During market closure (weekends, holidays):

- Oracle maintains "stickiness" to the closing price
- Price boundaries are set based on the final open price
- Large trades can move the price within these boundaries

**Risk Warning**: Weekend trading carries additional risk. A December 2025 incident saw a whale trigger a 3.5% selloff on a Sunday, causing liquidations. Consider reducing position sizes or pausing bots during market closures.

## Data Sources for Backtesting and Live Trading

Passivbot uses multiple data sources for stock perps, automatically falling back to alternatives when primary sources are unavailable:

### 1. Hyperliquid API (Primary for Live Trading)
- **Coverage**: Last ~3.5 days (5000 1m candles)
- **Format**: Native perp data with oracle pricing
- **Usage**: Live trading, recent backtesting
- **No setup required** - Works automatically

### 2. Yahoo Finance (Free, Default for Historical)
- **Coverage**: Last 7 days of 1m data
- **Cost**: FREE, no API key required
- **Setup**: Installed automatically with yfinance package
- **Limitations**: Only market hours data, no weekends

### 3. Finnhub / Alpha Vantage (Optional, for Extended History)
- **Coverage**: Varies by provider
- **Cost**: Requires API key (free tiers available)
- **Setup**: Add to api-keys.json

```json
{
  "tradfi": {
    "provider": "finnhub",
    "api_key": "your_api_key"
  }
}
```

### Data Source Priority

CandlestickManager automatically selects the best source:
1. Local cache (if available)
2. Hyperliquid API (last 3.5 days)
3. Yahoo Finance (last 7 days, free)
4. Configured TradFi provider (older data)

### Important Notes About TradFi Data

TradFi data represents actual stock prices and **does not include**:
- Perpetual funding rates
- Oracle-driven pricing during market closure
- Weekend/after-hours price movements

This data is suitable for:
- Strategy development and initial backtesting
- Understanding general price behavior
- Warm-up periods for EMAs

For accurate backtesting of actual perp behavior, use native Hyperliquid data where available.

## Limitations

### Current Limitations

1. **Isolated margin only** - Cross margin support planned for future HIP-3 upgrade
2. **10x max leverage** - Lower than crypto perps
3. **Higher fees** - 2x standard Hyperliquid fees
4. **Builder registration required** - One-time setup per builder (TradeXYZ, FLX, etc.)
5. **No hedge mode** - Same as regular Hyperliquid
6. **$10 minimum order** - Limits grid depth on small accounts

## Troubleshooting

### "Insufficient margin to place order"

**Cause**: Wallet not registered with TradeXYZ

**Solution**: Complete the one-time registration at [trade.xyz](https://trade.xyz) (see Requirements section)

### "Too many DEXes found" when loading markets

**Cause**: CCXT needs HIP-3 DEX specification

**Solution**: Passivbot handles this automatically. If using CCXT directly:
```python
exchange.options["fetchMarkets"] = {
    "types": ["swap", "hip3"],
    "hip3": {"dex": ["xyz"]}
}
```

### "No long symbols are approved due to min effective cost too high"

**Cause**: Balance too low for $10 minimum order size

**Solution**: Either:
1. Increase account balance
2. Set `filter_by_min_effective_cost: false` in config
3. Reduce `n_positions` to concentrate capital

### Orders rejected during market closure

**Cause**: Oracle price boundaries or liquidity issues

**Solution**: Consider pausing trading during extended market closures (weekends, holidays)

### Symbol not found / "Skipping unsupported markets"

**Cause**: Symbol mapping issue or market not loaded

**Solution**: Ensure you're using a valid ticker (TSLA, NVDA, etc.) and that the Hyperliquid markets cache is fresh. Delete `caches/hyperliquid/markets.json` to force refresh.

## Live Test Results (January 2026)

Successfully tested with:
- **Account**: ~$105 USDC
- **Symbol**: TSLA (xyz:TSLA)
- **Leverage**: 2x
- **Results**:
  - Bot correctly detected isolated margin requirement
  - Margin mode set to isolated automatically
  - Multiple fills executed at ~$423
  - Position tracking and order management working correctly

## Resources

- [TradeXYZ Documentation](https://docs.trade.xyz)
- [Hyperliquid HIP-3 Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals)
- [HyperStone Oracle (RedStone)](https://blog.redstone.finance/2025/11/13/felix-launches-its-first-hyperliquid-hip-3-market-with-tsla-powered-by-hyperstone/)

---

*Last updated: January 2026*
*Status: Live tested and working*
