# Hyperliquid Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).


## Delayed sparse-tail candles

Hyperliquid may initially omit a recent no-trade minute and later publish an
authoritative flat candle with zero volume for that same timestamp. Passivbot
must not synthesize or permanently classify the initially absent row. Recent
tail gaps use time-spaced retries, remain unavailable meanwhile, and are removed
from `known_gaps` as soon as the authoritative row is persisted.

## Candle retention and held trailing positions

Hyperliquid's `candleSnapshot` endpoint exposes only the most recent 5,000
candles. A trailing position whose latest fill predates that 1m window therefore
cannot reconstruct its extrema from a fresh host. Preserve and transfer the
existing 1m candle cache when migrating such a live position. Passivbot must
keep trailing state unavailable if the exact range from the first whole minute
after the fill is not locally present; wider-timeframe candles or an
earliest-available reset are not parity-safe substitutes. Once that range is
dense, only the bounded non-persistent open-tail projection described in the
candlestick-manager contract may bridge delayed current candles.
