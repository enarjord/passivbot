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
- `src/config_utils.py`
- `src/candlestick_manager.py`
