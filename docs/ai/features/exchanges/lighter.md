# Lighter USDC Perpetuals Exchange Contract

Venue-specific contract. Shared exchange rules, validation, and broker attribution live in
[exchange integrations](../exchange_integrations.md); trading-critical failure semantics live
in [the error contract](../../error_contract.md).

## Contract

Use the dedicated CCXT adapter and existing L2 API key, with the official signer revision
specified in [the setup guide](../../../exchanges/lighter.md). Never pass that key as CCXT's L1
`privateKey`: the default L1 setup rotates API keys and may approve builder fees. Force
`builderFee=false` and provide the official zero-valued integrator signing fields.

Lighter is one-way. Normalize authoritative buy/sell plus explicit `reduce_only` into position
side; every close must be reduce-only. Encode Passivbot client IDs into the exchange's 48-bit
integer range with a namespace and reversible order-type marker. Pass order and client
indexes to the signer as exact integers; binary64 cannot represent every index above
2^53, and a rounded index cancels or modifies a different order. Do not infer ownership from
side, price, or quantity. Quantities are base units, not `quote_multiplier` contracts.

Serialize signed writes and propagate failures. A `sendTx` receipt is pending transaction evidence,
not an active-order acknowledgement. Confirm the exact client ID through active/inactive orders;
confirm cancellation by authoritative active-order removal. Never blindly resubmit an uncertain
write. Fetch all active perpetual orders in one request, rather than only configured markets.
Validate ownership in active, closed, and streamed order responses, and reject duplicate position markets
instead of allowing a later row to overwrite exposure or invent hedged state.
Stream ownership checks belong in per-row normalization so malformed updates request an account
refresh while valid rows in the same batch remain usable, without consuming the reconnect budget.

Cap candle pages at 500 and bound the end of each request to prevent the exchange from silently
tail-anchoring an over-wide warmup window. Public data clients and historical preparation must use
the same adapter. Discover market age by walking bounded daily pages backward, not by requesting
one pre-listing day from the epoch. Historical sizing uses base units and one-way positions.
Verify the pinned signer hash before native loading.

Use actual order-book quotes because ticker responses lack bid/ask. Preserve the earliest local
receipt of the required last-price and top-of-book observations when certifying their freshness.
Keep cross/isolated margin
and leverage together in their per-market configuration transaction. Balance is realized USDC
collateral and must validate the exact requested account, not CCXT's missing-field zero defaults.
The synchronous balance CLI and asynchronous trading adapter share this validation.

Cursor-paginate all perpetual trades and reject missing pages, stalled cursors, or conflicting
identities. Validate descending timestamps within and across pages while allowing identical
overlap and equal-timestamp fills. A full page without a continuation cursor cannot prove complete
history unless the validated descending page has already crossed the requested start. Retain raw
pre-fill position evidence for restart reconstruction, and split one-way
position flips into close/open components with distinct event and source identities so cache
deduplication retains both legs. Reductions require authoritative account-side PnL,
except omitted zero values independently proven by the exchange's before-state. Preserve missing
fee evidence for the canonical best-effort fee policy rather than fabricating zero fees.
A nonzero pre-fill position requires a strictly positive entry quote; a flat position requires zero.
