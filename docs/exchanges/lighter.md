# Lighter

Passivbot connects to Lighter's USDC perpetuals through the pinned CCXT release.
Lighter uses one-way positions: a market can hold a long or a short, but cannot hold both
at once. Passivbot's one-way arbitration remains responsible for choosing the active side.
Spot, deposits, withdrawals, transfers, account creation, and API-key rotation are outside
Passivbot's trading interface.

## Credentials and signer

Create a dedicated Lighter API key with an index from 4 to 254. Use its **L2 API private key**,
not the Ethereum wallet's private key. Give each concurrent trading process its own API key.
Do not share a key with another order writer.

CCXT requires the official native signer. Download the binary for your operating system from
[lighter-python revision 8bac9f56](https://github.com/elliottech/lighter-python/tree/8bac9f56b9d0dd0eedaeb53a00ccb4fc9d77082e/lighter/signers).
Use that exact revision with CCXT 4.5.66: newer signer releases have an incompatible function
signature. The adapter verifies the binary SHA-256 before loading it. Keep the binary outside Git, for example under `caches/lighter/`.

| System | Signer filename |
|---|---|
| Linux x86-64 | `lighter-signer-linux-amd64.so` |
| Linux ARM64 | `lighter-signer-linux-arm64.so` |
| macOS Apple Silicon | `lighter-signer-darwin-arm64.dylib` |
| Windows x86-64 | `lighter-signer-windows-amd64.dll` |

Add an entry to your private `api-keys.json`:

```json
{
  "lighter_01": {
    "exchange": "lighter",
    "account_index": 123,
    "api_key_index": 4,
    "private_key": "YOUR_80_HEX_CHARACTER_L2_API_PRIVATE_KEY",
    "signer_path": "/absolute/path/to/lighter-signer-linux-amd64.so"
  }
}
```

The EVM address and API public key are not required when the account index is supplied.
Passivbot uses the existing key without rotating it, approving an integrator, or adding a
builder fee. No L1 private key is needed.

Use `live.user = "lighter_01"` in your reviewed private configuration. Begin with a small
approved coin universe and exposure appropriate to your account. The quote currency is USDC;
for example, CCXT's ETH market is `ETH/USDC:USDC`.

## Trading and market data

- Limit orders use Lighter's good-till-time lifetime (28 days); `time_in_force=post_only`
  sends maker-only orders. Market orders use immediate-or-cancel with Passivbot's supplied
  execution price as the price bound.
- Every close carries `reduceOnly`. Amounts are base-asset quantities, independent of the
  exchange's integer wire scaling. Minimum notional, quantity precision, and leverage limits
  come from market metadata.
- Cross and isolated margin are configured per market together with leverage. An exchange
  account hedge-mode setting is not available.
- Order creation and cancellation are serialized. A transaction receipt is not treated as an
  order acknowledgement: the adapter confirms the exact client order identity or order removal
  from authoritative order queries. Order removal still requests full account confirmation because
  an intervening fill may have changed the position. An unconfirmed write remains an error and is not resubmitted
  inside the connector.
- Numeric client order IDs retain Passivbot's order-type marker across restarts. Private CCXT Pro
  order updates supplement authoritative REST reconciliation.
- Lighter's bulk ticker lacks bid/ask. Market snapshots combine its last-trade price with actual
  order-book quotes, using CCXT Pro when enabled and REST when WebSockets are disabled.
- Candles use bounded 500-row `since`/`limit` requests, preserving the requested oldest candle.
  Live warmup, EMA/trailing inputs, HSL, and historical downloads use the shared candlestick manager.
  Market-age discovery walks daily candles backward to the first available page. Backtesting and
  optimization can use `backtest.exchanges = ["lighter"]`, subject to available exchange history.

Standard accounts allow 60 REST requests per minute. The connector retains CCXT's rate limiter;
large coin universes and cold candle caches therefore take longer to initialize. WebSocket
subscriptions have their own limits. Keep those limits in mind when running multiple processes
behind the same IP. See [Lighter rate limits](https://apidocs.lighter.xyz/docs/rate-limits).

## Fills, fees, and restart

The fill manager cursor-paginates the account-wide perpetual-trade endpoint, retaining external
fills as well as Passivbot fills. Repeated cursors, conflicting duplicates, malformed responses,
and unproven reductions fail rather than presenting incomplete history as complete.

Realized PnL comes from the queried account's ask/bid PnL field. Opening fills have zero price PnL.
For an omitted zero reduction PnL, the exchange's before-position and entry-quote fields must prove
zero at six-decimal USDC precision. A fill that crosses through zero is split into a close and an
entry so each strategy side retains the correct quantity, fees, and realized PnL. Exchange
before-state evidence also restores the resulting position when cached history starts mid-position.

Reported fee rates use Lighter's millionths scale. Missing fee evidence follows Passivbot's
observable best-effort fee policy (`live.fee_pct_fallback`), so such fees are estimates rather than
exact reported amounts. Do not interpret an omitted fee as an authoritative zero.

## Validation

Offline adapter, signing-request, malformed-input, pagination, and fill-cache restart tests:

```sh
PYTHONPATH=src python -m pytest tests/exchanges/test_lighter.py
```

These tests do not load real credentials, submit orders, or require a signer binary.
Authenticated validation should use a dedicated funded account and explicit risk limits;
account settings and even post-only order tests can change account state.

Primary references: [Lighter API keys](https://apidocs.lighter.xyz/docs/api-keys),
[active orders](https://apidocs.lighter.xyz/reference/accountactiveorders),
[trades](https://apidocs.lighter.xyz/reference/trades), and
[official transaction constants](https://github.com/elliottech/lighter-go/blob/main/types/txtypes/constants.go).
