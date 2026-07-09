# Stock Perpetuals (HIP-3)

## Contract

1. Stock perps are Hyperliquid-only in Passivbot.
2. Stock perps use isolated margin behavior.
3. Symbol normalization must preserve HIP-3 routing semantics.

## Non-Obvious Details

1. User symbol forms (`TSLA`, `xyz:TSLA`, `XYZ-TSLA`) map to HIP-3 market forms.
2. Builder registration is required before API trading on supported builders.
3. Stock perps have a practical $10 minimum order constraint.
4. Historical data may blend exchange/tradfi sources depending on age and availability.
5. Backtests fill closed-market minutes with synthetic flat candles
   (h=l=c=prev_close, v=0) that are TRADABLE by design (`synthetic_gaps_tradable`,
   set only for hyperliquid + `ohlcv_source_dir` + `xyz:` coins): the live venue
   trades 24/7, and strict fill inequalities (`low < bid` / `high > ask`) prevent
   fills for orders resting exactly at the flat price. Do not "fix" this by
   capping or un-marking synthetic candles — maintainer decision 2026-07-09; see
   docs/plans/backtest_data_integrity_audit_20260709.md (I-3) and the
   "Synthetic Candles During Market Closure" section in docs/stock_perps.md for
   the accepted modeling caveats (flat overnight drift, zero-volume dilution of
   forager volume metrics, regime seam with native data).

## High-Impact Operational Gotchas

1. Missing builder registration can surface as misleading margin errors.
2. Small balances can hit minimum order constraints quickly due to the $10 floor.
3. Symbol routing must remain Hyperliquid-only for stock perps.

## Test Focus

1. Symbol normalization and routing.
2. Margin mode assignment by symbol type.
3. Mixed crypto + stock configs.
4. Minimum-order behavior on small balances.

## Key Code

- `src/exchanges/hyperliquid.py`
- `src/passivbot.py`
- `src/config/access.py`
- `src/candlestick_manager.py`
