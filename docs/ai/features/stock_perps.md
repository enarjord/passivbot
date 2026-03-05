# Stock Perpetuals (HIP-3)

## Contract

1. Stock perps are Hyperliquid-only in Passivbot.
2. Stock perps use isolated margin behavior.
3. Symbol normalization must preserve HIP-3 routing semantics.

## Non-Obvious Details

1. User symbol forms (`TSLA`, `xyz:TSLA`, `XYZ-TSLA`) map to HIP-3 market forms.
2. Margin behavior differs from typical crypto-perp cross usage.
3. Historical data may blend exchange/tradfi sources depending on age and availability.

## Failure Modes To Watch

1. Symbol routed to non-Hyperliquid exchange.
2. Margin mode mismatch for stock perp symbols.
3. Missing builder/market registration producing misleading margin errors.

## Test Focus

1. Symbol normalization and routing.
2. Margin mode assignment by symbol type.
3. Mixed crypto + stock configs.

## Key Code

- `src/exchanges/hyperliquid.py`
- `src/passivbot.py`
- `src/config_utils.py`
- `src/candlestick_manager.py`
